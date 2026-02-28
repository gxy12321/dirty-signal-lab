from __future__ import annotations

import numpy as np
import pandas as pd


def set_seed(seed: int | None) -> None:
    if seed is not None:
        np.random.seed(seed)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / (std.replace(0, np.nan))


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def standardize(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(array, axis=0)
    std = np.nanstd(array, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (array - mean) / std, mean, std


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    n_features = x.shape[1]
    xtx = x.T @ x
    reg = alpha * np.eye(n_features)
    return np.linalg.solve(xtx + reg, x.T @ y)
