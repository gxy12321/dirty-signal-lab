from __future__ import annotations

from pathlib import Path
import json


def write_report(perf: dict, dq_report_path: str | Path, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dq = json.loads(Path(dq_report_path).read_text())

    lines = [
        "# Dirty Signal Lab Report",
        "",
        "## Data Quality",
        f"- Missing timestamps: {dq.get('missing_ts', 0)}",
        f"- Duplicates removed: {dq.get('duplicates', 0)}",
        f"- Inverted spreads fixed: {dq.get('inverted_spread', 0)}",
        "",
        "## Backtest Metrics",
        f"- Mean return: {perf['mean_return']:.6f}",
        f"- Volatility: {perf['volatility']:.6f}",
        f"- Sharpe (annualized): {perf['sharpe']:.2f}",
        f"- Turnover: {perf['turnover']:.4f}",
        f"- N obs: {perf['n_obs']}",
    ]

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path
