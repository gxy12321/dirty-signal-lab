from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from .utils import set_seed


def generate_dirty_ticks(symbol: str, n: int = 50000, seed: int | None = None) -> pd.DataFrame:
    set_seed(seed)

    # base time index
    ts = pd.date_range("2026-01-01", periods=n, freq="ms")

    # random walk mid price
    mid = 100 + np.cumsum(np.random.normal(0, 0.01, size=n))
    spread = np.abs(np.random.normal(0.01, 0.005, size=n))

    bid = mid - spread / 2
    ask = mid + spread / 2

    # inject dirty anomalies
    glitch_idx = np.random.choice(n, size=int(0.01 * n), replace=False)
    ask[glitch_idx] = bid[glitch_idx] - np.abs(np.random.normal(0.01, 0.005, size=len(glitch_idx)))  # inverted

    sizes = np.random.randint(1, 200, size=n)
    side = np.random.choice(["B", "S"], size=n)

    df = pd.DataFrame(
        {
            "ts": ts,
            "sym": symbol,
            "bid": bid,
            "ask": ask,
            "size": sizes,
            "side": side,
        }
    )

    # introduce duplicates and out-of-order rows
    dup = df.sample(frac=0.005, replace=False)
    df = pd.concat([df, dup], ignore_index=True)
    df = df.sample(frac=1.0).reset_index(drop=True)

    # remove some timestamps
    drop_idx = np.random.choice(len(df), size=int(0.005 * len(df)), replace=False)
    df.loc[drop_idx, "ts"] = pd.NaT

    return df


def write_raw(df: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dirty_ticks.csv"
    df.to_csv(path, index=False)
    return path


def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
    df = pd.read_csv(url)
    if df.empty:
        raise ValueError(f"No data returned from Stooq for symbol '{symbol}'")
    df = df.rename(columns={"Date": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    return df


def load_stooq_ticks(symbol: str) -> pd.DataFrame:
    daily = fetch_stooq_daily(symbol)
    # Use close as mid, synthesize bid/ask around it
    close = daily["Close"].to_numpy(dtype=float)
    spread = np.maximum(close * 0.0002, 0.01)
    bid = close - spread / 2
    ask = close + spread / 2

    volume = daily.get("Volume")
    if volume is None:
        size = np.full_like(close, 100.0)
    else:
        size = np.maximum(volume.to_numpy(dtype=float) / 1000.0, 1.0)

    side = np.where(daily["Close"] >= daily["Open"], "B", "S")

    df = pd.DataFrame(
        {
            "ts": daily["ts"],
            "sym": symbol.upper(),
            "bid": bid,
            "ask": ask,
            "size": size,
            "side": side,
        }
    )
    return df
