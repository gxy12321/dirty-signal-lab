from __future__ import annotations

import argparse
from pathlib import Path
from .ingest import generate_dirty_ticks, write_raw
from .clean import clean_ticks, write_clean
from .features import compute_features
from .backtest import backtest
from .report import write_report


def run_pipeline(args: argparse.Namespace) -> None:
    raw_dir = Path("data/raw")
    proc_dir = Path("data/processed")
    report_dir = Path("reports")

    # 1) generate dirty data
    df = generate_dirty_ticks(args.symbol, n=args.n, seed=args.seed)
    write_raw(df, raw_dir)

    # 2) clean
    clean_df, dq = clean_ticks(df)
    clean_path, dq_path = write_clean(clean_df, dq, proc_dir)

    # 3) features
    feat_df = compute_features(clean_df, model=args.model, ensemble=args.ensemble).dropna()
    feat_path = proc_dir / "features.csv"
    feat_df.to_csv(feat_path, index=False)

    # 4) backtest
    perf = backtest(feat_df, cost_bps=args.cost_bps)

    # 5) report
    write_report(perf, dq_path, report_dir)

    print(f"✅ Pipeline complete. Clean: {clean_path} | Features: {feat_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dirty Signal Lab pipeline")
    sub = parser.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="Run full pipeline")
    run.add_argument("--symbol", default="DEMO")
    run.add_argument("--n", type=int, default=50000)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--cost-bps", type=float, default=1.0)
    run.add_argument("--model", default="ridge", help="Model: ridge|rf|extratrees|xgb|lgbm|mlp")
    run.add_argument("--ensemble", action="store_true", help="Average all models")
    run.set_defaults(func=run_pipeline)

    args = parser.parse_args()
    if not getattr(args, "cmd", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
