from __future__ import annotations

import pandas as pd
import numpy as np
from .utils import ema, rolling_zscore


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # microprice
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["microprice"] = (df["bid"] * df["ask"].rolling(50).mean() + df["ask"] * df["bid"].rolling(50).mean()) / (
        df["ask"].rolling(50).mean() + df["bid"].rolling(50).mean()
    )

    # order flow imbalance proxy
    df["signed_size"] = np.where(df["side"] == "B", df["size"], -df["size"])
    df["ofi"] = df["signed_size"].rolling(200).sum()

    # volatility regime proxy
    df["ret"] = df["mid"].pct_change().fillna(0)
    df["vol"] = df["ret"].rolling(500).std(ddof=0)

    # z-scored features
    df["ofi_z"] = rolling_zscore(df["ofi"], 500)
    df["mp_z"] = rolling_zscore(df["microprice"], 500)

    # decay features
    df["ofi_ema"] = ema(df["ofi"], 200)

    # predictive signal (toy)
    df["signal"] = (df["ofi_z"] + df["mp_z"]) / 2
    df["signal"] = df["signal"].clip(-3, 3)

    return df
