import numpy as np
import pytest

torch = pytest.importorskip("torch")

from meta_gru.maml import evaluate_episode, meta_step, sample_episode
from meta_gru.model import GRURegressor


def test_maml_step_and_adaptation() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(48, 6, 3)).astype(np.float32)
    y = x.mean(axis=(1, 2), keepdims=False).astype(np.float32)[:, None]
    device = torch.device("cpu")
    episode = sample_episode(x, y, 12, 12, rng, device)
    model = GRURegressor(3, 8, 1, 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    step_metrics = meta_step(model, optimizer, [episode], 1e-2, 1, True, 5.0)
    metrics = evaluate_episode(model, episode, 1e-2, 1)
    assert np.isfinite(step_metrics["loss"])
    assert step_metrics["gradient_norm"] >= 0.0
    assert metrics["mae"] >= 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_second_order_maml_on_cuda() -> None:
    """验证显式 GRU 门能够在 CUDA 上执行二阶 MAML 反向传播。"""
    rng = np.random.default_rng(8)
    x = rng.normal(size=(24, 4, 2)).astype(np.float32)
    y = x.mean(axis=(1, 2)).astype(np.float32)[:, None]
    device = torch.device("cuda")
    episode = sample_episode(x, y, 6, 6, rng, device)
    model = GRURegressor(2, 4, 1, 0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    step_metrics = meta_step(model, optimizer, [episode], 1e-2, 1, False, 5.0)
    assert np.isfinite(step_metrics["loss"])
