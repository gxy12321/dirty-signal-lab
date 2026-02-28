from __future__ import annotations

import pandas as pd
import numpy as np
from .utils import ema, rolling_zscore, ridge_fit, standardize


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # microprice (cache rolling means to avoid recomputation)
    bid = df["bid"]
    ask = df["ask"]
    df["mid"] = (bid + ask) / 2
    bid_roll = bid.rolling(50).mean()
    ask_roll = ask.rolling(50).mean()
    df["microprice"] = (bid * ask_roll + ask * bid_roll) / (ask_roll + bid_roll)

    # order flow imbalance proxy (vectorized)
    signed_size = np.where(df["side"] == "B", df["size"], -df["size"])
    df["signed_size"] = signed_size
    df["ofi"] = pd.Series(signed_size, index=df.index).rolling(200).sum()

    # volatility regime proxy
    df["ret"] = df["mid"].pct_change().fillna(0)
    df["vol"] = df["ret"].rolling(500).std(ddof=0)

    # z-scored features
    df["ofi_z"] = rolling_zscore(df["ofi"], 500)
    df["mp_z"] = rolling_zscore(df["microprice"], 500)
    df["vol_z"] = rolling_zscore(df["vol"], 500)

    # decay features
    df["ofi_ema"] = ema(df["ofi"], 200)
    df["ofi_ema_z"] = rolling_zscore(df["ofi_ema"], 500)

    # target for model: next return
    df["target"] = df["mid"].pct_change().shift(-1)

    feature_cols = ["ofi_z", "mp_z", "vol_z", "ofi_ema_z"]
    x = df[feature_cols].to_numpy()
    y = df["target"].to_numpy()

    # train/test split for model (walk-forward style)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    n = len(df)
    split = int(n * 0.7)
    train_mask = valid & (np.arange(n) < split)

    if train_mask.sum() > 20:
        x_train = x[train_mask]
        y_train = y[train_mask]
        x_train_std, mean, std = standardize(x_train)
        weights = ridge_fit(x_train_std, y_train, alpha=1.0)
        x_all_std = (x - mean) / std
        model_score = x_all_std @ weights
    else:
        model_score = np.zeros(n)

    df["model_score"] = model_score
    df["signal"] = np.tanh(df["model_score"]).clip(-1, 1)

    return df
