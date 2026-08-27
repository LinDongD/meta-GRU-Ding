from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # 训练服务器没有显示器时也能保存图片。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _finish_figure(path: Path) -> None:
    """统一设置布局、分辨率并关闭画布，避免长时间训练积累显存/内存。"""
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_epoch_metrics(records: list[dict[str, Any]], plot_dir: Path) -> None:
    """绘制所有搜索试验及最终训练的逐 epoch Loss/RMSE。"""
    frame = pd.DataFrame(records)
    if frame.empty:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for phase, group in frame.groupby("phase", sort=False):
        is_final = phase == "final"
        style = {"linewidth": 2.2 if is_final else 0.8, "alpha": 1.0 if is_final else 0.35}
        axes[0].plot(group["epoch"], group["train_loss"], label=phase, **style)
        axes[1].plot(group["epoch"], group["train_rmse"], label=phase, **style)
    axes[0].set(title="Meta-training query loss", xlabel="Epoch", ylabel="MSE loss")
    axes[1].set(title="Meta-training RMSE", xlabel="Epoch", ylabel="RMSE (health/RUL points)")
    for axis in axes:
        axis.grid(alpha=0.25)
    if frame["phase"].nunique() <= 15:
        axes[1].legend(fontsize=8)
    _finish_figure(plot_dir / "epoch_loss_rmse.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    final = frame[frame["phase"] == "final"]
    diagnostics = final if not final.empty else frame
    axes[0].plot(diagnostics["epoch"], diagnostics.get("gradient_norm", 0.0))
    axes[0].set(title="Gradient norm", xlabel="Epoch", ylabel="L2 norm before clipping")
    axes[1].plot(diagnostics["epoch"], diagnostics.get("epoch_seconds", 0.0))
    axes[1].set(title="Epoch duration", xlabel="Epoch", ylabel="Seconds")
    axes[2].plot(diagnostics["epoch"], diagnostics.get("gpu_peak_memory_mb", 0.0))
    axes[2].set(title="CUDA peak allocated memory", xlabel="Epoch", ylabel="MiB")
    for axis in axes:
        axis.grid(alpha=0.25)
    _finish_figure(plot_dir / "training_diagnostics.png")


def plot_search_trials(trials: list[dict[str, Any]], plot_dir: Path) -> None:
    """绘制每组超参数在源域留出任务上的验证 MAE。"""
    if not trials:
        return
    frame = pd.DataFrame(trials)
    plt.figure(figsize=(9, 4.5))
    colors = np.where(frame["validation_mae"] == frame["validation_mae"].min(), "tab:red", "tab:blue")
    plt.bar(frame["trial"].astype(str), frame["validation_mae"], color=colors)
    plt.xlabel("Hyperparameter trial")
    plt.ylabel("Validation MAE")
    plt.title("Source-domain hyperparameter search")
    plt.grid(axis="y", alpha=0.25)
    _finish_figure(plot_dir / "hyperparameter_search_mae.png")


def plot_target_predictions(
    query_records: list[dict[str, Any]],
    curve_records: list[dict[str, Any]],
    plot_dir: Path,
) -> None:
    """绘制目标域 RUL 曲线、真实-预测散点、残差和误差随寿命变化。"""
    query = pd.DataFrame(query_records)
    curves = pd.DataFrame(curve_records)
    if query.empty or curves.empty:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)

    bearings = list(curves["bearing"].drop_duplicates())
    columns = min(2, len(bearings))
    rows = ceil(len(bearings) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 3.8 * rows), squeeze=False)
    for axis, bearing in zip(axes.flat, bearings):
        group = curves[curves["bearing"] == bearing].sort_values("sample_index")
        axis.plot(group["sample_index"], group["true_rul"], label="True RUL", linewidth=2)
        axis.plot(group["sample_index"], group["predicted_rul"], label="Estimated RUL", linewidth=1.4)
        axis.scatter(group.loc[group["is_support"], "sample_index"], group.loc[group["is_support"], "true_rul"], s=28, label="Support", zorder=3)
        axis.set(title=bearing, xlabel="Window index", ylabel="Health/RUL (%)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes.flat[len(bearings) :]:
        axis.set_visible(False)
    _finish_figure(plot_dir / "target_rul_curves.png")

    lower = min(query["true_rul"].min(), query["predicted_rul"].min())
    upper = max(query["true_rul"].max(), query["predicted_rul"].max())
    plt.figure(figsize=(6, 6))
    for bearing, group in query.groupby("bearing"):
        plt.scatter(group["true_rul"], group["predicted_rul"], s=14, alpha=0.45, label=bearing)
    plt.plot([lower, upper], [lower, upper], "k--", label="Ideal")
    plt.xlabel("True RUL (%)")
    plt.ylabel("Estimated RUL (%)")
    plt.title("Estimated versus true RUL")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    _finish_figure(plot_dir / "rul_true_vs_estimated.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(query["error"], bins=40, alpha=0.8)
    axes[0].axvline(0.0, color="black", linestyle="--")
    axes[0].set(title="RUL residual distribution", xlabel="Estimated - true", ylabel="Count")
    axes[1].scatter(query["true_rul"], query["absolute_error"], s=12, alpha=0.4)
    axes[1].set(title="Absolute error over lifetime", xlabel="True RUL (%)", ylabel="Absolute error")
    for axis in axes:
        axis.grid(alpha=0.25)
    _finish_figure(plot_dir / "rul_residuals_and_absolute_error.png")


def plot_target_trial_metrics(records: list[dict[str, Any]], plot_dir: Path) -> None:
    """对比每个目标轴承多次 few-shot 试验的 MAE 与 RMSE 分布。"""
    frame = pd.DataFrame(records)
    if frame.empty:
        return
    bearings = list(frame["bearing"].drop_duplicates())
    positions = np.arange(len(bearings))
    mae_values = [frame.loc[frame["bearing"] == name, "mae"].to_numpy() for name in bearings]
    rmse_values = [frame.loc[frame["bearing"] == name, "rmse"].to_numpy() for name in bearings]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].boxplot(mae_values, tick_labels=bearings, showmeans=True)
    axes[1].boxplot(rmse_values, tick_labels=bearings, showmeans=True)
    axes[0].set(title="Few-shot MAE by target bearing", ylabel="MAE")
    axes[1].set(title="Few-shot RMSE by target bearing", ylabel="RMSE")
    for axis in axes:
        axis.set_xticks(positions + 1, bearings, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    _finish_figure(plot_dir / "target_trial_mae_rmse.png")
