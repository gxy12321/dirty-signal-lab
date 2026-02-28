from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .ingest import generate_dirty_ticks, load_stooq_ticks
from .clean import clean_ticks, write_clean
from .features import build_feature_frame
from .model import fit_predict_all
from .backtest import backtest_detailed
from .plots import plot_model_metrics, plot_equity_curves, plot_runtime


@dataclass
class BenchmarkResult:
    perf_table: pd.DataFrame
    equity_curves: dict[str, pd.Series]
    data_quality_path: Path


def run_benchmark(
    source: str = "synthetic",
    symbol: str = "DEMO",
    n: int = 50000,
    seed: int | None = 7,
    cost_bps: float = 1.0,
    out_dir: str | Path = "reports/benchmark",
    top_k: int = 3,
) -> BenchmarkResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if source == "synthetic":
        raw_df = generate_dirty_ticks(symbol, n=n, seed=seed)
    elif source == "stooq":
        raw_df = load_stooq_ticks(symbol)
    else:
        raise ValueError("source must be one of: synthetic|stooq")

    clean_df, dq = clean_ticks(raw_df)
    _, dq_path = write_clean(clean_df, dq, Path("data/processed"))

    feat_df, feature_cols = build_feature_frame(clean_df)
    feat_df = feat_df.dropna().reset_index(drop=True)

    x = feat_df[feature_cols].to_numpy(dtype=float)
    y = feat_df["target"].to_numpy(dtype=float)

    model_names = ["ridge", "rf", "extratrees", "xgb", "lgbm", "mlp"]
    t0 = time.perf_counter()
    preds = fit_predict_all(x, y, model_names)
    fit_total = time.perf_counter() - t0

    rows = []
    curves: dict[str, pd.Series] = {}
    runtimes: dict[str, float] = {}

    for name in model_names:
        start = time.perf_counter()
        score = preds[name]
        signal = np.tanh(score).clip(-1, 1)
        perf, curve = backtest_detailed(feat_df, signal=signal, cost_bps=cost_bps)
        runtime = time.perf_counter() - start

        rows.append({"model": name, **perf, "runtime_s": runtime})
        curves[name] = curve
        runtimes[name] = runtime

    perf_df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

    # figures
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_model_metrics(perf_df, fig_dir)
    plot_runtime(perf_df, fig_dir, total_fit_time=fit_total)

    top_models = perf_df.head(top_k)["model"].tolist()
    top_curves = {m: curves[m] for m in top_models}
    plot_equity_curves(top_curves, fig_dir)

    # save table
    perf_path = out_dir / "model_benchmark.csv"
    perf_df.to_csv(perf_path, index=False)

    return BenchmarkResult(perf_table=perf_df, equity_curves=curves, data_quality_path=dq_path)
