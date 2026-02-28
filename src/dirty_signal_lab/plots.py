from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_model_metrics(perf_df: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["sharpe", "mean_return", "volatility", "turnover"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics, strict=False):
        ax.bar(perf_df["model"], perf_df[metric], color="#4c78a8")
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "model_metrics.png", dpi=150)
    plt.close(fig)


def plot_equity_curves(curves: dict[str, pd.Series], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, curve in curves.items():
        ax.plot(curve.values, label=name)
    ax.set_title("Equity Curves (Top Models)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "equity_curves.png", dpi=150)
    plt.close(fig)


def plot_runtime(perf_df: pd.DataFrame, out_dir: str | Path, total_fit_time: float) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(perf_df["model"], perf_df["runtime_s"], color="#f58518")
    ax.set_title(f"Per-Model Backtest Runtime (Total fit: {total_fit_time:.2f}s)")
    ax.set_ylabel("Seconds")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "runtime.png", dpi=150)
    plt.close(fig)
