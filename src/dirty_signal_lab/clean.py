from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def clean_ticks(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {}

    # missing timestamps
    missing_ts = df["ts"].isna().sum()
    report["missing_ts"] = int(missing_ts)
    df = df.dropna(subset=["ts"]).copy()

    # duplicates
    dup_count = df.duplicated(subset=["ts", "sym"]).sum()
    report["duplicates"] = int(dup_count)
    df = df.drop_duplicates(subset=["ts", "sym"])

    # inverted spread
    inverted = (df["ask"] < df["bid"]).sum()
    report["inverted_spread"] = int(inverted)
    df.loc[df["ask"] < df["bid"], ["bid", "ask"]] = np.nan
    df = df.dropna(subset=["bid", "ask"])

    # ensure sort
    df = df.sort_values("ts").reset_index(drop=True)

    report["rows_after"] = int(len(df))
    return df, report


def write_clean(df: pd.DataFrame, report: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "clean_ticks.csv"
    rep_path = out_dir / "data_quality.json"
    df.to_csv(data_path, index=False)
    pd.Series(report).to_json(rep_path)
    return data_path, rep_path
