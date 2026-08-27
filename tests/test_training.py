from pathlib import Path

import numpy as np
import torch

from meta_gru.train import EpochMetricLogger, fit_model


def test_fit_model_writes_epoch_metrics(workspace_tmp_path: Path) -> None:
    """检查训练入口会为每个 epoch 写入完整指标。"""
    rng = np.random.default_rng(12)
    tasks = {}
    for index in range(2):
        x = rng.normal(size=(24, 4, 3)).astype(np.float32)
        y = x.mean(axis=(1, 2)).astype(np.float32)[:, None]
        tasks[f"task_{index}"] = (x, y)
    hparams = {
        "hidden_size": 4,
        "num_layers": 1,
        "dropout": 0.0,
        "outer_lr": 1e-3,
        "weight_decay": 0.0,
        "meta_batch_size": 2,
        "subtasks_per_epoch": 2,
        "support_size": 4,
        "query_size": 4,
        "inner_lr": 1e-2,
        "inner_steps": 1,
        "first_order": True,
        "inner_optimizer": "adam",
        "output_activation": "sigmoid",
    }
    path = workspace_tmp_path / "epoch_metrics.csv"
    logger = EpochMetricLogger(path)
    try:
        _, history = fit_model(
            tasks,
            3,
            hparams,
            epochs=2,
            gradient_clip=5.0,
            seed=3,
            device=torch.device("cpu"),
            phase="test",
            trial=0,
            metric_logger=logger,
        )
    finally:
        logger.close()
    assert len(history) == 2
    text = path.read_text(encoding="utf-8-sig")
    assert "train_loss" in text
    assert "gradient_norm" in text
    assert text.count("\n") == 3
