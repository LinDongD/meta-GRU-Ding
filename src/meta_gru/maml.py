from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

try:
    from torch.func import functional_call
except ImportError:  # PyTorch 2.0 compatibility
    from torch.nn.utils.stateless import functional_call


RUL_SCALE = 100.0


@dataclass(frozen=True)
class Episode:
    """一个元学习 episode：支持集用于适配，查询集用于元更新或评价。"""
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    support_indices: np.ndarray
    query_indices: np.ndarray


def sample_episode(
    x: np.ndarray,
    y: np.ndarray,
    support_size: int,
    query_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> Episode:
    """从一个轴承任务中无放回抽取互不重叠的支持集和查询集。"""
    required = support_size + query_size
    if len(x) < required:
        raise ValueError(f"Task has {len(x)} samples, but an episode needs {required}")
    indices = rng.choice(len(x), size=required, replace=False)
    support = indices[:support_size]
    query = indices[support_size:]
    to_tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    return Episode(
        to_tensor(x[support]),
        to_tensor(y[support]),
        to_tensor(x[query]),
        to_tensor(y[query]),
        support.copy(),
        query.copy(),
    )


def adapt_parameters(
    model: nn.Module,
    episode: Episode,
    inner_lr: float,
    inner_steps: int,
    first_order: bool,
    optimizer_name: str = "sgd",
) -> OrderedDict[str, torch.Tensor]:
    """执行 MAML 内循环，支持论文所用的可微 Adam 或标准 SGD。"""
    if optimizer_name not in {"sgd", "adam"}:
        raise ValueError("optimizer_name must be 'sgd' or 'adam'")
    parameters = OrderedDict(model.named_parameters())
    loss_fn = nn.MSELoss()
    first_moment = OrderedDict((name, torch.zeros_like(value)) for name, value in parameters.items())
    second_moment = OrderedDict((name, torch.zeros_like(value)) for name, value in parameters.items())
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for step in range(1, inner_steps + 1):
        prediction = functional_call(model, parameters, (episode.support_x,))
        loss = loss_fn(prediction, episode.support_y)
        # 二阶模式保留梯度图；一阶 MAML 则忽略 Hessian 项以节省显存。
        gradients = torch.autograd.grad(
            loss,
            tuple(parameters.values()),
            create_graph=not first_order,
        )
        if optimizer_name == "sgd":
            parameters = OrderedDict(
                (name, parameter - inner_lr * gradient)
                for (name, parameter), gradient in zip(parameters.items(), gradients)
            )
        else:
            updated = OrderedDict()
            for (name, parameter), gradient in zip(parameters.items(), gradients):
                first_moment[name] = beta1 * first_moment[name] + (1.0 - beta1) * gradient
                second_moment[name] = beta2 * second_moment[name] + (1.0 - beta2) * gradient.square()
                corrected_first = first_moment[name] / (1.0 - beta1**step)
                corrected_second = second_moment[name] / (1.0 - beta2**step)
                updated[name] = parameter - inner_lr * corrected_first / (
                    corrected_second.sqrt() + epsilon
                )
            parameters = updated
    return parameters


def meta_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    episodes: Iterable[Episode],
    inner_lr: float,
    inner_steps: int,
    first_order: bool,
    gradient_clip: float,
    inner_optimizer: str = "sgd",
) -> dict[str, float]:
    """汇总多个任务的 query MSE，完成一次跨任务外循环更新。"""
    loss_fn = nn.MSELoss()
    query_losses = []
    for episode in episodes:
        adapted = adapt_parameters(
            model, episode, inner_lr, inner_steps, first_order, optimizer_name=inner_optimizer
        )
        query_prediction = functional_call(model, adapted, (episode.query_x,))
        query_losses.append(loss_fn(query_prediction, episode.query_y))
    outer_loss = torch.stack(query_losses).mean()
    optimizer.zero_grad(set_to_none=True)
    outer_loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    loss_value = float(outer_loss.detach().cpu())
    return {
        "loss": loss_value,
        "rmse": float(np.sqrt(max(loss_value, 0.0)) * RUL_SCALE),
        "gradient_norm": float(gradient_norm.detach().cpu()),
    }


def evaluate_episode(
    model: nn.Module,
    episode: Episode,
    inner_lr: float,
    inner_steps: int,
    inner_optimizer: str = "sgd",
) -> dict[str, float]:
    """少样本适配后，仅在 query 集上计算 MAE/RMSE。"""
    adapted = adapt_parameters(
        model, episode, inner_lr, inner_steps, first_order=True, optimizer_name=inner_optimizer
    )
    with torch.no_grad():
        prediction = functional_call(model, adapted, (episode.query_x,))
        error = prediction - episode.query_y
        return {
            "mae": float(error.abs().mean().cpu() * RUL_SCALE),
            "rmse": float(error.square().mean().sqrt().cpu() * RUL_SCALE),
        }


def evaluate_episode_detailed(
    model: nn.Module,
    episode: Episode,
    all_x: np.ndarray,
    inner_lr: float,
    inner_steps: int,
    inner_optimizer: str = "sgd",
    batch_size: int = 1024,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """返回 query 指标、query 预测和完整寿命曲线预测，供保存与作图。"""
    adapted = adapt_parameters(
        model, episode, inner_lr, inner_steps, first_order=True, optimizer_name=inner_optimizer
    )

    def predict(tensor: torch.Tensor) -> np.ndarray:
        chunks = []
        with torch.no_grad():
            for start in range(0, len(tensor), batch_size):
                value = functional_call(model, adapted, (tensor[start : start + batch_size],))
                chunks.append(value.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0).reshape(-1)

    query_prediction = predict(episode.query_x) * RUL_SCALE
    query_truth = episode.query_y.detach().cpu().numpy().reshape(-1) * RUL_SCALE
    error = query_prediction - query_truth
    metrics = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }
    all_tensor = torch.as_tensor(all_x, dtype=torch.float32, device=episode.support_x.device)
    return metrics, query_prediction, predict(all_tensor) * RUL_SCALE
