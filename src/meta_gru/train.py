from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from .features import BearingSeries, load_or_extract_dataset
from .maml import evaluate_episode, evaluate_episode_detailed, meta_step, sample_episode
from .model import GRURegressor
from .preprocessing import SourcePreprocessor, make_windows
from .visualization import (
    plot_epoch_metrics,
    plot_search_trials,
    plot_target_predictions,
    plot_target_trial_metrics,
)


class EpochMetricLogger:
    """逐 epoch 追加写入 CSV；即使训练中断，已完成 epoch 的指标也不会丢失。"""

    fieldnames = [
        "phase",
        "trial",
        "epoch",
        "total_epochs",
        "train_loss",
        "train_rmse",
        "gradient_norm",
        "outer_lr",
        "epoch_seconds",
        "elapsed_seconds",
        "gpu_peak_memory_mb",
    ]

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("w", newline="", encoding="utf-8-sig")
        self.writer = csv.DictWriter(self.stream, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def log(self, record: dict[str, Any]) -> None:
        self.writer.writerow(record)
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机源，增强实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(value: str) -> torch.device:
    """auto 模式优先选择 CUDA，否则回退到 CPU。"""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def log_uniform(rng: np.random.Generator, bounds: list[float]) -> float:
    return float(math.exp(rng.uniform(math.log(bounds[0]), math.log(bounds[1]))))


def sample_hparams(search: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """从 YAML 定义的离散集合和对数均匀区间中抽取一组超参数。"""
    categorical = [
        "hidden_size",
        "num_layers",
        "dropout",
        "window_size",
        "inner_steps",
        "weight_decay",
        "meta_batch_size",
        "support_size",
        "query_size",
        "first_order",
    ]
    result = {name: rng.choice(search[name]).item() for name in categorical}
    result["inner_lr"] = log_uniform(rng, search["inner_lr"])
    result["outer_lr"] = log_uniform(rng, search["outer_lr"])
    return result


def prepare_tasks(
    source: list[BearingSeries],
    targets: list[BearingSeries],
    pca_variance: float,
    alignment: str,
    window_size: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray]], int]:
    """拟合无泄漏预处理器，并把每个轴承转换成一个元学习任务。"""
    preprocessor = SourcePreprocessor(pca_variance, alignment).fit(source)
    source_tasks = {item.name: make_windows(preprocessor.transform_source(item), window_size) for item in source}
    target_tasks = {item.name: make_windows(preprocessor.transform_target(item), window_size) for item in targets}
    input_size = next(iter(source_tasks.values()))[0].shape[-1]
    return source_tasks, target_tasks, input_size


def fit_model(
    tasks: dict[str, tuple[np.ndarray, np.ndarray]],
    input_size: int,
    hparams: dict[str, Any],
    epochs: int,
    gradient_clip: float,
    seed: int,
    device: torch.device,
    phase: str,
    trial: int,
    metric_logger: EpochMetricLogger,
) -> tuple[GRURegressor, list[dict[str, Any]]]:
    """训练一个 Meta-GRU，并实时显示/保存每个 epoch 的 MSE 与 RMSE。"""
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model = GRURegressor(input_size, int(hparams["hidden_size"]), int(hparams["num_layers"]), float(hparams["dropout"])).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(hparams["outer_lr"]), weight_decay=float(hparams["weight_decay"])
    )
    names = list(tasks)
    batch_size = min(int(hparams["meta_batch_size"]), len(names))
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    progress = tqdm(range(1, epochs + 1), desc=f"{phase}", unit="epoch", dynamic_ncols=True)
    for epoch in progress:
        epoch_started = time.perf_counter()
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        selected = rng.choice(names, size=batch_size, replace=False)
        episodes = [
            sample_episode(
                *tasks[name],
                int(hparams["support_size"]),
                int(hparams["query_size"]),
                rng,
                device,
            )
            for name in selected
        ]
        step_metrics = meta_step(
            model,
            optimizer,
            episodes,
            float(hparams["inner_lr"]),
            int(hparams["inner_steps"]),
            bool(hparams["first_order"]),
            gradient_clip,
        )
        train_loss = step_metrics["loss"]
        train_rmse = step_metrics["rmse"]
        record = {
            "phase": phase,
            "trial": trial,
            "epoch": epoch,
            "total_epochs": epochs,
            "train_loss": train_loss,
            "train_rmse": train_rmse,
            "gradient_norm": step_metrics["gradient_norm"],
            "outer_lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_peak_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
            ),
        }
        history.append(record)
        metric_logger.log(record)
        progress.set_postfix(loss=f"{train_loss:.5f}", rmse=f"{train_rmse:.4f}")
    return model, history


def validation_score(
    model: GRURegressor,
    tasks: dict[str, tuple[np.ndarray, np.ndarray]],
    hparams: dict[str, Any],
    repeats: int,
    seed: int,
    device: torch.device,
) -> float:
    """在源域留出轴承上执行 few-shot 适配，以 MAE 选择超参数。"""
    model.eval()
    rng = np.random.default_rng(seed)
    scores = []
    for x, y in tasks.values():
        for _ in range(repeats):
            episode = sample_episode(x, y, int(hparams["support_size"]), int(hparams["query_size"]), rng, device)
            scores.append(evaluate_episode(model, episode, float(hparams["inner_lr"]), int(hparams["inner_steps"]))["mae"])
    return float(np.mean(scores))


def run(config_path: Path, output_dir: Path) -> None:
    """运行超参数搜索、最终训练、目标域评价、结果保存与绘图。"""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    device = choose_device(config["train"]["device"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    device_description = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"运行设备: {device} ({device_description}), PyTorch={torch.__version__}, CUDA={torch.version.cuda}")

    # 特征只需提取一次；后续搜索试验直接复用压缩缓存。
    dataset = load_or_extract_dataset(
        Path(config["data_root"]), Path("artifacts/cache"), channels=config["channels"]
    )
    source = [item for item in dataset if item.condition == int(config["source_condition"])]
    target = [item for item in dataset if item.condition == int(config["target_condition"])]
    if len(source) < 3 or not target:
        raise ValueError("The selected conditions do not provide enough source/target bearings")

    shuffled = source.copy()
    rng.shuffle(shuffled)
    validation_count = int(config["search"]["validation_bearings"])
    meta_train_raw = shuffled[:-validation_count]
    meta_validation_raw = shuffled[-validation_count:]
    trials: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []
    best_score = float("inf")
    best_hparams: dict[str, Any] | None = None

    metric_logger = EpochMetricLogger(output_dir / "metrics" / "epoch_metrics.csv")
    try:
        for trial_index in range(int(config["search"]["trials"])):
            hparams = sample_hparams(config["search"], rng)
            train_tasks, validation_tasks, input_size = prepare_tasks(
                meta_train_raw,
                meta_validation_raw,
                float(config["pca_variance"]),
                config["alignment"],
                int(hparams["window_size"]),
            )
            model, history = fit_model(
                train_tasks,
                input_size,
                hparams,
                int(config["search"]["epochs_per_trial"]),
                float(config["train"]["gradient_clip"]),
                seed + trial_index,
                device,
                phase=f"search_trial_{trial_index}",
                trial=trial_index,
                metric_logger=metric_logger,
            )
            epoch_records.extend(history)
            score = validation_score(
                model,
                validation_tasks,
                hparams,
                int(config["search"]["episodes_per_validation_bearing"]),
                seed + 10_000 + trial_index,
                device,
            )
            trial = {"trial": trial_index, "validation_mae": score, **hparams}
            trials.append(trial)
            # 每个 trial 结束立即落盘，长时间搜索中断时仍可恢复分析。
            pd.DataFrame(trials).to_csv(output_dir / "metrics" / "search_trials.csv", index=False, encoding="utf-8-sig")
            (output_dir / "search_trials.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")
            tqdm.write(json.dumps(trial, ensure_ascii=False))
            if score < best_score:
                best_score, best_hparams = score, copy.deepcopy(hparams)

        assert best_hparams is not None
        all_source_tasks, target_tasks, input_size = prepare_tasks(
            source,
            target,
            float(config["pca_variance"]),
            config["alignment"],
            int(best_hparams["window_size"]),
        )
        final_model, final_history = fit_model(
            all_source_tasks,
            input_size,
            best_hparams,
            int(config["train"]["epochs"]),
            float(config["train"]["gradient_clip"]),
            seed + 20_000,
            device,
            phase="final",
            trial=-1,
            metric_logger=metric_logger,
        )
        epoch_records.extend(final_history)
    finally:
        metric_logger.close()

    final_model.eval()
    target_rng = np.random.default_rng(seed + 30_000)
    target_results: list[dict[str, Any]] = []
    query_predictions: list[dict[str, Any]] = []
    curve_predictions: list[dict[str, Any]] = []
    for name, (x, y) in target_tasks.items():
        for repeat in range(int(config["train"]["target_trials"])):
            episode = sample_episode(
                x,
                y,
                int(best_hparams["support_size"]),
                int(best_hparams["query_size"]),
                target_rng,
                device,
            )
            metrics, query_prediction, full_prediction = evaluate_episode_detailed(
                final_model,
                episode,
                x,
                float(best_hparams["inner_lr"]),
                int(best_hparams["inner_steps"]),
            )
            target_results.append({"bearing": name, "trial": repeat, **metrics})
            query_truth = episode.query_y.detach().cpu().numpy().reshape(-1)
            for sample_index, truth, prediction in zip(episode.query_indices, query_truth, query_prediction):
                error = float(prediction - truth)
                query_predictions.append(
                    {
                        "bearing": name,
                        "trial": repeat,
                        "sample_index": int(sample_index),
                        "acquisition_index": int(sample_index + best_hparams["window_size"] - 1),
                        "true_rul": float(truth),
                        "predicted_rul": float(prediction),
                        "error": error,
                        "absolute_error": abs(error),
                    }
                )

            # 第一次 episode 保存该轴承完整寿命曲线；support 点单独标记。
            if repeat == 0:
                support_indices = set(int(value) for value in episode.support_indices)
                for sample_index, (truth, prediction) in enumerate(zip(y.reshape(-1), full_prediction)):
                    curve_predictions.append(
                        {
                            "bearing": name,
                            "trial": repeat,
                            "sample_index": sample_index,
                            "acquisition_index": int(sample_index + best_hparams["window_size"] - 1),
                            "true_rul": float(truth),
                            "predicted_rul": float(prediction),
                            "error": float(prediction - truth),
                            "absolute_error": float(abs(prediction - truth)),
                            "is_support": sample_index in support_indices,
                        }
                    )

    summary = {
        "paper": "Ding et al., Applied Soft Computing 104 (2021) 107211",
        "source_condition": config["source_condition"],
        "target_condition": config["target_condition"],
        "device": str(device),
        "device_name": device_description,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "best_validation_mae": best_score,
        "best_hparams": best_hparams,
        "target_mae_mean": float(np.mean([item["mae"] for item in target_results])),
        "target_mae_std": float(np.std([item["mae"] for item in target_results])),
        "target_rmse_mean": float(np.mean([item["rmse"] for item in target_results])),
    }
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "target_trials.json").write_text(json.dumps(target_results, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(target_results).to_csv(metrics_dir / "target_trial_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(query_predictions).to_csv(metrics_dir / "target_query_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curve_predictions).to_csv(metrics_dir / "target_full_curves.csv", index=False, encoding="utf-8-sig")
    plot_epoch_metrics(epoch_records, plots_dir)
    plot_search_trials(trials, plots_dir)
    plot_target_predictions(query_predictions, curve_predictions, plots_dir)
    plot_target_trial_metrics(target_results, plots_dir)
    torch.save({"model_state": final_model.state_dict(), "input_size": input_size, "hparams": best_hparams}, output_dir / "model.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-GRU hyperparameter search and cross-condition evaluation")
    parser.add_argument("--config", type=Path, default=Path("configs/search.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/default"))
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
