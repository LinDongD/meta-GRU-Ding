from pathlib import Path

from meta_gru.visualization import (
    plot_epoch_metrics,
    plot_search_trials,
    plot_target_predictions,
    plot_target_trial_metrics,
)


def test_all_plots_are_created(workspace_tmp_path: Path) -> None:
    """用极小模拟结果检查训练曲线和 RUL 图能够无界面生成。"""
    epochs = [
        {"phase": "final", "epoch": 1, "train_loss": 4.0, "train_rmse": 2.0, "gradient_norm": 3.0, "epoch_seconds": 0.2, "gpu_peak_memory_mb": 10.0},
        {"phase": "final", "epoch": 2, "train_loss": 1.0, "train_rmse": 1.0, "gradient_norm": 1.0, "epoch_seconds": 0.1, "gpu_peak_memory_mb": 10.0},
    ]
    trials = [{"trial": 0, "validation_mae": 2.0}, {"trial": 1, "validation_mae": 1.0}]
    query = [
        {"bearing": "Bearing2_1", "true_rul": 100.0, "predicted_rul": 95.0, "error": -5.0, "absolute_error": 5.0},
        {"bearing": "Bearing2_1", "true_rul": 0.0, "predicted_rul": 4.0, "error": 4.0, "absolute_error": 4.0},
    ]
    curves = [
        {"bearing": "Bearing2_1", "sample_index": 0, "true_rul": 100.0, "predicted_rul": 95.0, "is_support": True},
        {"bearing": "Bearing2_1", "sample_index": 1, "true_rul": 0.0, "predicted_rul": 4.0, "is_support": False},
    ]
    plot_epoch_metrics(epochs, workspace_tmp_path)
    plot_search_trials(trials, workspace_tmp_path)
    plot_target_predictions(query, curves, workspace_tmp_path)
    plot_target_trial_metrics(
        [
            {"bearing": "Bearing2_1", "mae": 5.0, "rmse": 6.0},
            {"bearing": "Bearing2_1", "mae": 4.0, "rmse": 5.0},
        ],
        workspace_tmp_path,
    )
    expected = {
        "epoch_loss_rmse.png",
        "training_diagnostics.png",
        "hyperparameter_search_mae.png",
        "target_rul_curves.png",
        "rul_true_vs_estimated.png",
        "rul_residuals_and_absolute_error.png",
        "target_trial_mae_rmse.png",
    }
    assert expected == {path.name for path in workspace_tmp_path.glob("*.png")}
