"""Metrics computed from a BacktestResult: PnL decomposition, Sharpe, max
drawdown, fill rate, inventory distribution.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult


@dataclasses.dataclass
class MarketMetrics:
    coin: str
    n_fills: int
    n_buy_fills: int
    n_sell_fills: int
    n_quote_bars: int
    fill_rate_buy: float
    fill_rate_sell: float
    trading_pnl: float
    fees_pnl: float
    funding_pnl: float
    adverse_selection_pnl: float
    total_pnl: float
    max_abs_position_usd: float
    mean_abs_position_usd: float
    was_flattened: bool
    largest_fill_pnl_share: float  # flag: fraction of |trading_pnl| driven by one fill
    n_toxic_fills: int
    n_evaluable_fills: int


@dataclasses.dataclass
class PortfolioMetrics:
    total_pnl: float
    total_trading_pnl: float
    total_fees_pnl: float
    total_funding_pnl: float
    total_adverse_selection_pnl: float
    adverse_selection_enabled: bool
    sharpe_annualized: float | None
    sharpe_note: str
    max_drawdown_usd: float
    max_drawdown_pct: float
    n_trading_days: float
    daily_halts: int
    market_metrics: list


def _fill_pnl_shares(fills, fair_value_at_fill=None) -> float:
    """Rough per-fill spread-capture magnitude used only to flag whether a
    handful of large fills dominate the trading PnL -- not an exact
    per-fill attribution (that would require the fair value at fill time,
    which we don't retain per-fill to keep memory bounded)."""
    if not fills:
        return 0.0
    notionals = [f.notional_usd for f in fills]
    total = sum(notionals)
    if total == 0:
        return 0.0
    return max(notionals) / total


def compute_market_metrics(ms, adverse: dict | None = None) -> MarketMetrics:
    fills = ms.fills
    buy_fills = [f for f in fills if f.side in ("buy", "flatten_buy")]
    sell_fills = [f for f in fills if f.side in ("sell", "flatten_sell")]

    inv_usd = np.array([x[2] for x in ms.inventory_series]) if ms.inventory_series else np.array([0.0])
    trading_pnl = ms.pnl_series[-1][1] if ms.pnl_series else 0.0

    adverse_pnl = 0.0
    n_toxic = 0
    n_evaluable = 0
    if adverse is not None and ms.coin in adverse:
        a = adverse[ms.coin]
        adverse_pnl = a.penalty_cash
        n_toxic = a.n_toxic_fills
        n_evaluable = a.n_evaluable_fills

    total_pnl = (ms.pnl_series[-1][4] if ms.pnl_series else 0.0) + adverse_pnl

    return MarketMetrics(
        coin=ms.coin,
        n_fills=len(fills),
        n_buy_fills=len(buy_fills),
        n_sell_fills=len(sell_fills),
        n_quote_bars=0,
        fill_rate_buy=0.0,
        fill_rate_sell=0.0,
        trading_pnl=trading_pnl,
        fees_pnl=ms.fees_cash,
        funding_pnl=ms.funding_cash,
        adverse_selection_pnl=adverse_pnl,
        total_pnl=total_pnl,
        max_abs_position_usd=float(np.max(np.abs(inv_usd))) if len(inv_usd) else 0.0,
        mean_abs_position_usd=float(np.mean(np.abs(inv_usd))) if len(inv_usd) else 0.0,
        was_flattened=False,
        largest_fill_pnl_share=_fill_pnl_shares(fills),
        n_toxic_fills=n_toxic,
        n_evaluable_fills=n_evaluable,
    )


def compute_portfolio_metrics(result: BacktestResult, adverse: dict | None = None) -> PortfolioMetrics:
    market_metrics = []
    total_trading = total_fees = total_funding = total_adverse = 0.0

    for coin, ms in result.market_states.items():
        mm = compute_market_metrics(ms, adverse)
        mm.n_quote_bars = result.quote_bar_count.get(coin, 0)
        mm.fill_rate_buy = mm.n_buy_fills / mm.n_quote_bars if mm.n_quote_bars else 0.0
        mm.fill_rate_sell = mm.n_sell_fills / mm.n_quote_bars if mm.n_quote_bars else 0.0
        mm.was_flattened = coin in result.risk_state.flattened_markets
        market_metrics.append(mm)
        total_trading += mm.trading_pnl
        total_fees += mm.fees_pnl
        total_funding += mm.funding_pnl
        total_adverse += mm.adverse_selection_pnl

    total_pnl = total_trading + total_fees + total_funding + total_adverse

    eq_df = pd.DataFrame(result.global_equity_series, columns=["timestamp", "equity"])
    eq_df = eq_df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    if adverse:
        events = [ev for a in adverse.values() for ev in a.events]
        if events:
            pen = pd.DataFrame(events, columns=["timestamp", "penalty"])
            pen = pen.groupby("timestamp")["penalty"].sum().sort_index().cumsum()
            aligned = pen.reindex(eq_df["timestamp"], method="ffill").fillna(0.0)
            eq_df["equity"] = eq_df["equity"].to_numpy() + aligned.to_numpy()

    running_max = eq_df["equity"].cummax()
    drawdown = eq_df["equity"] - running_max
    max_dd_usd = float(drawdown.min()) if not drawdown.empty else 0.0
    max_dd_pct = float(max_dd_usd / running_max[drawdown.idxmin()]) if not drawdown.empty and running_max[drawdown.idxmin()] != 0 else 0.0

    # Daily returns for Sharpe: resample equity to daily last value, diff.
    eq_df_indexed = eq_df.set_index("timestamp")
    daily_equity = eq_df_indexed["equity"].resample("1D").last().dropna()
    daily_returns = daily_equity.diff().dropna()
    n_days = len(daily_returns)

    sharpe = None
    sharpe_note = ""
    MIN_DAYS_FOR_SHARPE = 20
    if n_days < MIN_DAYS_FOR_SHARPE:
        sharpe_note = (
            f"Only {n_days} daily return observations -- too short a sample for a "
            f"meaningful Sharpe ratio (want {MIN_DAYS_FOR_SHARPE}+ trading days). "
            "Treat any Sharpe figure below as noise, not a signal."
        )
        if n_days >= 3:
            std = daily_returns.std()
            sharpe = float(daily_returns.mean() / std * np.sqrt(365)) if std and std != 0 else None
    else:
        std = daily_returns.std()
        sharpe = float(daily_returns.mean() / std * np.sqrt(365)) if std and std != 0 else None
        sharpe_note = f"Computed over {n_days} daily observations."

    return PortfolioMetrics(
        total_pnl=total_pnl,
        total_trading_pnl=total_trading,
        total_fees_pnl=total_fees,
        total_funding_pnl=total_funding,
        total_adverse_selection_pnl=total_adverse,
        adverse_selection_enabled=adverse is not None,
        sharpe_annualized=sharpe,
        sharpe_note=sharpe_note,
        max_drawdown_usd=max_dd_usd,
        max_drawdown_pct=max_dd_pct,
        n_trading_days=n_days,
        daily_halts=len(result.risk_state.halt_log),
        market_metrics=market_metrics,
    )
