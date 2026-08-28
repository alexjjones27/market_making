"""Metrics for the directional carry backtests: total return, Sharpe, max
drawdown, trade count/win rate, and the directional-vs-funding PnL split.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from backtest.directional_engine import DirectionalResult

MIN_DAYS_FOR_SHARPE = 20


@dataclasses.dataclass
class DirectionalMetrics:
    label: str  # e.g. coin name, or "PORTFOLIO"
    total_pnl: float
    total_return_pct: float
    directional_pnl: float
    funding_pnl: float
    fees_pnl: float
    sharpe_annualized: float | None
    sharpe_note: str
    max_drawdown_usd: float
    max_drawdown_pct: float
    n_trades: int
    n_winning_trades: int
    win_rate: float | None
    avg_trade_pnl: float | None
    n_days: float


def _sharpe_and_dd(pnl_series: list, capital_usd: float) -> tuple:
    if not pnl_series:
        return None, "No bars in this run.", 0.0, 0.0, 0.0

    eq_df = pd.DataFrame(pnl_series, columns=["timestamp", "directional", "funding", "fees", "total_pnl"])
    eq_df = eq_df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    eq_df["equity"] = capital_usd + eq_df["total_pnl"]

    running_max = eq_df["equity"].cummax()
    drawdown = eq_df["equity"] - running_max
    max_dd_usd = float(drawdown.min()) if not drawdown.empty else 0.0
    max_dd_pct = (
        float(max_dd_usd / running_max[drawdown.idxmin()])
        if not drawdown.empty and running_max[drawdown.idxmin()] != 0 else 0.0
    )

    daily_equity = eq_df.set_index("timestamp")["equity"].resample("1D").last().dropna()
    daily_returns = daily_equity.diff().dropna()
    n_days = len(daily_returns)

    sharpe = None
    if n_days >= 3:
        std = daily_returns.std()
        sharpe = float(daily_returns.mean() / std * np.sqrt(365)) if std and std != 0 else None
    if n_days < MIN_DAYS_FOR_SHARPE:
        note = (
            f"Only {n_days} daily observations -- too short for a meaningful Sharpe "
            f"(want {MIN_DAYS_FOR_SHARPE}+). Treat any figure shown as noise, not signal."
        )
    else:
        note = f"Computed over {n_days} daily observations."

    return sharpe, note, max_dd_usd, max_dd_pct, float(n_days)


def compute_metrics(label: str, trades: list, pnl_series: list, capital_usd: float) -> DirectionalMetrics:
    sharpe, sharpe_note, max_dd_usd, max_dd_pct, n_days = _sharpe_and_dd(pnl_series, capital_usd)

    total_pnl = pnl_series[-1][4] if pnl_series else 0.0
    directional = sum(t.directional_pnl for t in trades)
    funding = sum(t.funding_pnl for t in trades)
    fees = sum(t.fees_pnl for t in trades)

    n_trades = len(trades)
    n_winning = sum(1 for t in trades if t.total_pnl > 0)
    win_rate = (n_winning / n_trades) if n_trades else None
    avg_trade = (sum(t.total_pnl for t in trades) / n_trades) if n_trades else None

    return DirectionalMetrics(
        label=label,
        total_pnl=total_pnl,
        total_return_pct=total_pnl / capital_usd * 100.0 if capital_usd else 0.0,
        directional_pnl=directional,
        funding_pnl=funding,
        fees_pnl=fees,
        sharpe_annualized=sharpe,
        sharpe_note=sharpe_note,
        max_drawdown_usd=max_dd_usd,
        max_drawdown_pct=max_dd_pct,
        n_trades=n_trades,
        n_winning_trades=n_winning,
        win_rate=win_rate,
        avg_trade_pnl=avg_trade,
        n_days=n_days,
    )


def combine_portfolio(results: list, capital_usd: float) -> DirectionalMetrics:
    """Combine several markets' DirectionalResult into one portfolio-level
    equity curve (summed total_pnl aligned by timestamp) and metrics."""
    all_trades = [t for r in results for t in r.trades]

    frames = []
    for r in results:
        if not r.pnl_series:
            continue
        df = pd.DataFrame(r.pnl_series, columns=["timestamp", "directional", "funding", "fees", "total_pnl"])
        df = df.drop_duplicates(subset="timestamp").set_index("timestamp")
        frames.append(df["total_pnl"].rename(r.coin))

    if not frames:
        combined_pnl_series = []
    else:
        wide = pd.concat(frames, axis=1).sort_index().ffill().fillna(0.0)
        summed = wide.sum(axis=1)
        combined_pnl_series = [(ts, 0.0, 0.0, 0.0, val) for ts, val in summed.items()]

    return compute_metrics("PORTFOLIO", all_trades, combined_pnl_series, capital_usd)
