from __future__ import annotations

from dataclasses import replace

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

from .features import BearingSeries


def _symmetric_matrix_power(matrix: np.ndarray, power: float, eps: float = 1e-6) -> np.ndarray:
    """通过特征值分解计算对称正定矩阵幂，并截断过小特征值。"""
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, eps)
    return (vectors * (values**power)) @ vectors.T


class SourcePreprocessor:
    """仅在源域拟合标准化/PCA，再把目标域对齐到源域。

    均值匹配对应线性核 MMD 的一阶统计量，CORAL 对齐二阶协方差。
    这是论文 TSUDA 目标的稳定工程近似，并非其未充分披露的黎曼优化器逐行复刻。
    """

    def __init__(self, variance: float = 0.99, alignment: str = "coral_mmd") -> None:
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=variance, svd_solver="full")
        self.alignment = alignment
        self.source_mean: np.ndarray | None = None
        self.source_cov: np.ndarray | None = None

    def fit(self, source: list[BearingSeries]) -> "SourcePreprocessor":
        # 严格只使用源域，防止目标域信息泄漏进标准化和 PCA 参数。
        matrix = np.concatenate([item.features for item in source], axis=0)
        reduced = self.pca.fit_transform(self.scaler.fit_transform(matrix))
        self.source_mean = reduced.mean(axis=0)
        self.source_cov = np.atleast_2d(np.cov(reduced, rowvar=False))
        return self

    def _project(self, values: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(values))

    def transform_source(self, series: BearingSeries) -> BearingSeries:
        return replace(series, features=self._project(series.features).astype(np.float32))

    def transform_target(self, series: BearingSeries) -> BearingSeries:
        projected = self._project(series.features)
        if self.alignment == "none":
            aligned = projected
        elif self.alignment == "coral_mmd":
            if self.source_mean is None or self.source_cov is None:
                raise RuntimeError("fit must be called before transform_target")
            target_mean = projected.mean(axis=0)
            target_cov = np.atleast_2d(np.cov(projected, rowvar=False))
            # 先白化目标域，再用源域协方差着色，最后匹配源域均值。
            whiten = _symmetric_matrix_power(target_cov, -0.5)
            color = _symmetric_matrix_power(self.source_cov, 0.5)
            aligned = (projected - target_mean) @ whiten @ color + self.source_mean
        else:
            raise ValueError(f"Unknown alignment: {self.alignment}")
        return replace(series, features=aligned.astype(np.float32))


def make_windows(series: BearingSeries, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """把退化特征转换为 GRU 序列，并把健康度从 0–100 归一化到 0–1。"""
    if window_size < 1 or len(series.features) < window_size:
        raise ValueError(f"Invalid window_size={window_size} for {series.name}")
    x = np.stack([series.features[i - window_size + 1 : i + 1] for i in range(window_size - 1, len(series.features))])
    # 归一化可避免 MSE 和内循环梯度随 0–100 标签尺度被放大约四个数量级。
    y = series.health[window_size - 1 :, None] / 100.0
    return x.astype(np.float32), y.astype(np.float32)


def domain_alignment_diagnostics(
    source: list[BearingSeries],
    targets: list[BearingSeries],
    variance: float = 0.99,
    sample_size: int = 500,
    seed: int = 42,
) -> tuple[list[dict[str, float | str]], int, float]:
    """量化目标域对齐前后的均值、协方差和非线性 RBF-MMD 距离。"""
    preprocessor = SourcePreprocessor(variance, "coral_mmd").fit(source)
    source_values = np.concatenate(
        [preprocessor.transform_source(item).features for item in source], axis=0
    )
    source_mean = source_values.mean(axis=0)
    source_covariance = np.atleast_2d(np.cov(source_values, rowvar=False))
    covariance_scale = max(float(np.linalg.norm(source_covariance)), 1e-12)
    rng = np.random.default_rng(seed)

    def rbf_mmd(first: np.ndarray, second: np.ndarray) -> float:
        count = min(sample_size, len(first), len(second))
        a = first[rng.choice(len(first), count, replace=False)]
        b = second[rng.choice(len(second), count, replace=False)]
        combined = np.concatenate([a, b], axis=0)
        probe = combined[: min(300, len(combined))]
        distances = np.sum((probe[:, None, :] - probe[None, :, :]) ** 2, axis=-1)
        positive = distances[distances > 0]
        median_distance = float(np.median(positive)) if positive.size else 1.0
        gamma = 1.0 / max(2.0 * median_distance, 1e-12)
        return float(
            rbf_kernel(a, a, gamma=gamma).mean()
            + rbf_kernel(b, b, gamma=gamma).mean()
            - 2.0 * rbf_kernel(a, b, gamma=gamma).mean()
        )

    records: list[dict[str, float | str]] = []
    for item in targets:
        before = preprocessor._project(item.features)
        after = preprocessor.transform_target(item).features
        records.append(
            {
                "bearing": item.name,
                "mean_distance_before": float(np.linalg.norm(before.mean(axis=0) - source_mean)),
                "mean_distance_after": float(np.linalg.norm(after.mean(axis=0) - source_mean)),
                "covariance_distance_before": float(
                    np.linalg.norm(np.cov(before, rowvar=False) - source_covariance) / covariance_scale
                ),
                "covariance_distance_after": float(
                    np.linalg.norm(np.cov(after, rowvar=False) - source_covariance) / covariance_scale
                ),
                "rbf_mmd_before": rbf_mmd(source_values, before),
                "rbf_mmd_after": rbf_mmd(source_values, after),
            }
        )
    return records, int(preprocessor.pca.n_components_), float(preprocessor.pca.explained_variance_ratio_.sum())
