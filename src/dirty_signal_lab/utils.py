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


