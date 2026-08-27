from pathlib import Path

import numpy as np

from meta_gru.features import extract_bearing, statistical_features
from meta_gru.preprocessing import make_windows


def test_statistical_features_are_finite() -> None:
    rng = np.random.default_rng(7)
    values = statistical_features(rng.normal(size=2560))
    assert values.shape == (30,)
    assert np.isfinite(values).all()


def test_real_bearing_smoke() -> None:
    root = Path("Data/Bearing3_3")
    if not root.exists():
        return
    series = extract_bearing(root, channels="both")
    assert series.features.shape == (352, 60)
    assert series.health[0] == 100.0
    assert series.health[-1] == 0.0
    x, y = make_windows(series, 8)
    assert x.shape == (345, 8, 60)
    assert y.shape == (345, 1)
