from __future__ import annotations

import pandas as pd
import numpy as np


def backtest(df: pd.DataFrame, cost_bps: float = 1.0) -> dict:
    df = df.copy()
    df["ret"] = df["mid"].pct_change().fillna(0)

    # simple signal-based position (lagged)
    df["pos"] = df["signal"].shift(1).fillna(0)
    df["pos"] = df["pos"].clip(-1, 1)

    # turnover
    df["turnover"] = df["pos"].diff().abs().fillna(0)

    # cost
    cost = cost_bps / 1e4
    df["net_ret"] = df["pos"] * df["ret"] - cost * df["turnover"]

    perf = {
        "mean_return": float(df["net_ret"].mean()),
        "volatility": float(df["net_ret"].std(ddof=0)),
        "sharpe": float(df["net_ret"].mean() / (df["net_ret"].std(ddof=0) + 1e-9) * np.sqrt(252 * 6.5 * 60 * 60)),
        "turnover": float(df["turnover"].mean()),
        "n_obs": int(len(df)),
    }

    return perf
