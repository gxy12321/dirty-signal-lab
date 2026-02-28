from __future__ import annotations

import pandas as pd
import numpy as np
from .utils import ema, rolling_zscore
from .model import fit_predict, fit_predict_all


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    bid = df["bid"].to_numpy(dtype=float)
    ask = df["ask"].to_numpy(dtype=float)
    mid = (bid + ask) / 2
    df["mid"] = mid

    bid_roll = pd.Series(bid, index=df.index).rolling(50).mean()
    ask_roll = pd.Series(ask, index=df.index).rolling(50).mean()
    df["microprice"] = (bid * ask_roll + ask * bid_roll) / (ask_roll + bid_roll)

    signed_size = np.where(df["side"].to_numpy() == "B", df["size"].to_numpy(dtype=float), -df["size"].to_numpy(dtype=float))
    df["signed_size"] = signed_size
    df["ofi"] = pd.Series(signed_size, index=df.index).rolling(200).sum()

    ret = pd.Series(mid, index=df.index).pct_change().fillna(0)
    df["ret"] = ret
    df["vol"] = ret.rolling(500).std(ddof=0)

    df["ofi_z"] = rolling_zscore(df["ofi"], 500)
    df["mp_z"] = rolling_zscore(df["microprice"], 500)
    df["vol_z"] = rolling_zscore(df["vol"], 500)

    df["ofi_ema"] = ema(df["ofi"], 200)
    df["ofi_ema_z"] = rolling_zscore(df["ofi_ema"], 500)

    df["target"] = pd.Series(mid, index=df.index).pct_change().shift(-1)

    feature_cols = ["ofi_z", "mp_z", "vol_z", "ofi_ema_z"]
    return df, feature_cols


def compute_features(df: pd.DataFrame, model: str = "ridge", ensemble: bool = False) -> pd.DataFrame:
    df, feature_cols = build_feature_frame(df)

    x = df[feature_cols].to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=float)

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
