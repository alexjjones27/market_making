"""Equity curve, inventory time series/histogram, and fill scatter plots.
Saved to disk as PNGs (no interactive display assumed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtest.engine import BacktestResult


def plot_equity_curve(result: BacktestResult, out_path: Path) -> None:
    df = pd.DataFrame(result.global_equity_series, columns=["timestamp", "equity"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp"], df["equity"])
    ax.set_title("Portfolio equity (capital + total PnL)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity (USD)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_inventory(result: BacktestResult, out_dir: Path) -> None:
    for coin, ms in result.market_states.items():
        if not ms.inventory_series:
            continue
        df = pd.DataFrame(ms.inventory_series, columns=["timestamp", "position_base", "position_usd"])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(df["timestamp"], df["position_usd"])
        axes[0].set_title(f"{coin}: inventory over time")
        axes[0].set_ylabel("Position (USD)")
        axes[0].tick_params(axis="x", rotation=30)

        axes[1].hist(df["position_usd"], bins=40)
        axes[1].set_title(f"{coin}: inventory distribution")
        axes[1].set_xlabel("Position (USD)")

        fig.tight_layout()
        fig.savefig(out_dir / f"inventory_{coin}.png", dpi=120)
        plt.close(fig)


def plot_fills(result: BacktestResult, out_dir: Path) -> None:
    for coin, ms in result.market_states.items():
        if not ms.fills:
            continue
        df = pd.DataFrame(
            [(f.timestamp, f.price, f.side) for f in ms.fills],
            columns=["timestamp", "price", "side"],
        )
        candles = ms.df

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(candles["timestamp"], candles["close"], color="lightgray", linewidth=0.7, label="close")
        for side, marker, color in [("buy", "^", "green"), ("sell", "v", "red")]:
            sub = df[df["side"].isin([side, f"flatten_{side}"])]
            if not sub.empty:
                ax.scatter(sub["timestamp"], sub["price"], marker=marker, color=color, s=10, label=side)
        ax.set_title(f"{coin}: simulated fills")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / f"fills_{coin}.png", dpi=120)
        plt.close(fig)


def generate_all_plots(result: BacktestResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_equity_curve(result, out_dir / "equity_curve.png")
    plot_inventory(result, out_dir)
    plot_fills(result, out_dir)
