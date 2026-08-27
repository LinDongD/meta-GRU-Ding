from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


EPS = 1e-12


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), EPS))


def statistical_features(signal: np.ndarray, sample_rate: float = 25_600.0) -> np.ndarray:
    """提取论文 Tables A.1-A.2 中的 17 个时域和 13 个频域特征。

    输入是一份 PRONOSTIA 采样文件中的单个振动通道。所有除法都加入数值保护，
    避免恒定信号或极小方差导致 NaN/Inf。
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size < 4:
        raise ValueError("A vibration record must contain at least four samples")

    mean = float(x.mean())
    centered = x - mean
    std = float(x.std(ddof=1))
    abs_mean = float(np.mean(np.abs(x)))
    sqrt_amp = float(np.mean(np.sqrt(np.abs(x))) ** 2)
    variance = float(np.mean(centered**2))
    rms = float(np.sqrt(np.mean(x**2)))
    peak_abs = float(np.max(np.abs(x)))
    maximum = float(np.max(x))
    minimum = float(np.min(x))
    third_moment = float(np.mean(x**3))
    fourth_moment = float(np.mean(x**4))
    mean_square = float(np.mean(x**2))
    skewness_index = _safe_div(third_moment, mean_square ** 1.5)
    kurtosis_index = _safe_div(fourth_moment, mean_square**2)

    time_features = np.asarray(
        [
            mean,
            std,
            sqrt_amp,
            abs_mean,
            third_moment,
            fourth_moment,
            mean_square,
            maximum,
            minimum,
            maximum - minimum,
            rms,
            _safe_div(rms, abs_mean),
            _safe_div(peak_abs, rms),
            _safe_div(peak_abs, abs_mean),
            _safe_div(peak_abs, sqrt_amp),
            skewness_index,
            kurtosis_index,
        ],
        dtype=np.float64,
    )

    # 去除直流分量后执行实数 FFT；第 0 个频点不参与频域退化特征。
    spectrum = np.abs(np.fft.rfft(centered))[1:]
    frequencies = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)[1:]
    spectral_sum = float(spectrum.sum())
    spectral_mean = float(spectrum.mean())
    spectral_centered = spectrum - spectral_mean
    spectral_var = float(np.var(spectrum, ddof=1))
    spectral_std = float(np.sqrt(max(spectral_var, EPS)))
    spectral_skew = float(np.mean(spectral_centered**3) / spectral_std**3)
    spectral_kurt = float(np.mean(spectral_centered**4) / spectral_std**4)

    centroid = _safe_div(float(np.sum(frequencies * spectrum)), spectral_sum)
    freq_delta = frequencies - centroid
    freq_var = _safe_div(float(np.sum(freq_delta**2 * spectrum)), spectral_sum)
    freq_std = float(np.sqrt(max(freq_var, EPS)))
    f2 = float(np.sum(frequencies**2 * spectrum))
    f4 = float(np.sum(frequencies**4 * spectrum))
    rms_frequency = float(np.sqrt(_safe_div(f2, spectral_sum)))
    root_variance_frequency = float(np.sqrt(_safe_div(f4, f2)))
    frequency_ratio = _safe_div(f2, np.sqrt(max(f4 * spectral_sum, EPS)))
    freq_skew = _safe_div(float(np.sum(freq_delta**3 * spectrum)), spectral_sum * freq_std**3)
    freq_kurt = _safe_div(float(np.sum(freq_delta**4 * spectrum)), spectral_sum * freq_std**4)
    square_root_frequency = _safe_div(
        float(np.sum(np.sqrt(np.abs(freq_delta)) * spectrum)),
        spectral_sum * np.sqrt(max(freq_std, EPS)),
    )

    frequency_features = np.asarray(
        [
            spectral_mean,
            spectral_var,
            spectral_skew,
            spectral_kurt,
            centroid,
            freq_std,
            rms_frequency,
            root_variance_frequency,
            frequency_ratio,
            _safe_div(freq_std, centroid),
            freq_skew,
            freq_kurt,
            square_root_frequency,
        ],
        dtype=np.float64,
    )
    result = np.concatenate([time_features, frequency_features]).astype(np.float32)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(frozen=True)
class BearingSeries:
    name: str
    condition: int
    features: np.ndarray
    health: np.ndarray


def _read_vibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, delimiter=",", usecols=(4, 5), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Unexpected PRONOSTIA CSV layout: {path}")
    return values[:, 0], values[:, 1]


def extract_bearing(
    bearing_dir: Path,
    channels: str = "both",
    sample_rate: float = 25_600.0,
) -> BearingSeries:
    """读取一个轴承的完整寿命数据，并生成从 100 到 0 的健康度标签。"""
    files = sorted(bearing_dir.glob("acc_*.csv"))
    if not files:
        raise FileNotFoundError(f"No acc_*.csv files found in {bearing_dir}")
    rows: list[np.ndarray] = []
    for path in files:
        horizontal, vertical = _read_vibration(path)
        # 论文同时使用水平和垂直信号；默认分别提取 30 维后拼接为 60 维。
        if channels == "horizontal":
            row = statistical_features(horizontal, sample_rate)
        elif channels == "vertical":
            row = statistical_features(vertical, sample_rate)
        elif channels == "both":
            row = np.concatenate(
                [statistical_features(horizontal, sample_rate), statistical_features(vertical, sample_rate)]
            )
        else:
            raise ValueError("channels must be one of: horizontal, vertical, both")
        rows.append(row)
    count = len(rows)
    # 论文把初始完全健康定义为 100，把最后一次采样的失效状态定义为 0。
    health = np.linspace(100.0, 0.0, count, dtype=np.float32)
    condition = int(bearing_dir.name.split("_")[0].replace("Bearing", ""))
    return BearingSeries(bearing_dir.name, condition, np.stack(rows), health)


def load_or_extract_dataset(
    data_root: Path,
    cache_dir: Path,
    channels: str = "both",
    sample_rate: float = 25_600.0,
) -> list[BearingSeries]:
    """读取全部轴承；若已有特征缓存则避免重复解析上万个 CSV 文件。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    result: list[BearingSeries] = []
    for bearing_dir in sorted(data_root.glob("Bearing*_*")):
        cache_path = cache_dir / f"{bearing_dir.name}_{channels}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            result.append(
                BearingSeries(
                    bearing_dir.name,
                    int(cached["condition"]),
                    cached["features"],
                    cached["health"],
                )
            )
            continue
        series = extract_bearing(bearing_dir, channels, sample_rate)
        np.savez_compressed(
            cache_path,
            features=series.features,
            health=series.health,
            condition=np.asarray(series.condition),
        )
        result.append(series)
    if not result:
        raise FileNotFoundError(f"No Bearing*_* directories found under {data_root}")
    return result


def feature_names(prefix: str = "") -> list[str]:
    base = [f"DF{i}" for i in range(1, 31)]
    return [f"{prefix}{name}" for name in base]


def concatenate_feature_names(channels: str) -> Iterable[str]:
    if channels == "both":
        return feature_names("horizontal_") + feature_names("vertical_")
    return feature_names(f"{channels}_")
