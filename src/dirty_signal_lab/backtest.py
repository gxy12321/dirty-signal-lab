from __future__ import annotations

import pandas as pd
import numpy as np


def _compute_returns(mid: np.ndarray) -> np.ndarray:
    ret = np.empty_like(mid, dtype=float)
    ret[0] = 0.0
    ret[1:] = mid[1:] / mid[:-1] - 1.0
    return ret


def backtest(df: pd.DataFrame, cost_bps: float = 1.0) -> dict:
    mid = df["mid"].to_numpy(dtype=float)
    signal = df["signal"].to_numpy(dtype=float)

    ret = _compute_returns(mid)
    pos = np.roll(signal, 1)
    pos[0] = 0.0
    pos = np.clip(pos, -1, 1)

    turnover = np.abs(np.diff(pos, prepend=0.0))
    cost = cost_bps / 1e4
    net_ret = pos * ret - cost * turnover

    vol = float(net_ret.std(ddof=0))
    mean_ret = float(net_ret.mean())

    perf = {
        "mean_return": mean_ret,
        "volatility": vol,
        "sharpe": float(mean_ret / (vol + 1e-9) * np.sqrt(252 * 6.5 * 60 * 60)),
        "turnover": float(turnover.mean()),
        "n_obs": int(len(df)),
    }

    return perf


def backtest_detailed(
    df: pd.DataFrame,
    signal: np.ndarray | None = None,
    cost_bps: float = 1.0,
) -> tuple[dict, pd.Series]:
    mid = df["mid"].to_numpy(dtype=float)
    if signal is None:
        signal = df["signal"].to_numpy(dtype=float)
    else:
        signal = np.asarray(signal, dtype=float)

    ret = _compute_returns(mid)
    pos = np.roll(signal, 1)
    pos[0] = 0.0
    pos = np.clip(pos, -1, 1)

    turnover = np.abs(np.diff(pos, prepend=0.0))
    cost = cost_bps / 1e4
    net_ret = pos * ret - cost * turnover

    vol = float(net_ret.std(ddof=0))
    mean_ret = float(net_ret.mean())

    perf = {
        "mean_return": mean_ret,
        "volatility": vol,
        "sharpe": float(mean_ret / (vol + 1e-9) * np.sqrt(252 * 6.5 * 60 * 60)),
        "turnover": float(turnover.mean()),
        "n_obs": int(len(df)),
    }

    equity = pd.Series((1.0 + net_ret).cumprod())
    return perf, equity
