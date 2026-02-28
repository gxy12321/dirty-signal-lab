from __future__ import annotations

import pandas as pd
import numpy as np
from .utils import ema, rolling_zscore
from .model import fit_predict, fit_predict_all


def compute_features(df: pd.DataFrame, model: str = "ridge", ensemble: bool = False) -> pd.DataFrame:
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

    if ensemble:
        preds = fit_predict_all(x, y, ["ridge", "rf", "extratrees", "xgb", "lgbm", "mlp"])
        for name, score in preds.items():
            df[f"model_score_{name}"] = score
        model_score = np.nanmean(np.column_stack(list(preds.values())), axis=1)
    else:
        model_score = fit_predict(x, y, model)

    df["model_score"] = model_score
    df["signal"] = np.tanh(df["model_score"]).clip(-1, 1)

    return df
