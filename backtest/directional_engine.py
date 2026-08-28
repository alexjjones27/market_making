"""Position-based backtest engine for the directional carry strategies.
Separate from backtest/engine.py (the market-making quote/fill engine) --
that engine's mechanics (bid/ask, top-of-book fills) don't map onto a
long/short/flat position with entries, exits, and stops, so this is a
new, purpose-built loop rather than a strained reuse of the MM engine.

Lookahead-safety: a decision made using bar[idx]'s close is only ever
executed at bar[idx+1]'s open. The funding rate used at any decision point
is the most recently SETTLED funding record as of that bar's timestamp
(merge_asof, backward) -- never a rate for a still-accruing interval.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd


@dataclasses.dataclass
class TradeRecord:
    coin: str
    side: str  # "long" or "short"
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size_usd: float
    size_multiplier: float
    directional_pnl: float
    funding_pnl: float
    fees_pnl: float
    total_pnl: float


@dataclasses.dataclass
class DirectionalResult:
    coin: str
    trades: list
    pnl_series: list  # (timestamp, directional_pnl, funding_cash, fees_cash, total_pnl)


def _align_funding(df: pd.DataFrame, funding_df: pd.DataFrame) -> np.ndarray:
    """Backward-looking funding rate known as of each bar's timestamp --
    i.e. the last funding record that had already settled. NaN where no
    funding has settled yet at the start of the series."""
    if funding_df.empty:
        return np.full(len(df), np.nan)
    merged = pd.merge_asof(
        df[["timestamp"]], funding_df[["timestamp", "fundingRate"]],
        on="timestamp", direction="backward",
    )
    return merged["fundingRate"].to_numpy()


def run_directional_backtest(
    coin: str, df: pd.DataFrame, funding_df: pd.DataFrame, signal_model,
    order_notional_usd: float, taker_rate: float,
) -> DirectionalResult:
    df = df.reset_index(drop=True)
    signal_model.prepare(df)
    funding_series = _align_funding(df, funding_df)

    n = len(df)
    position_sign = 0
    position_base = 0.0
    entry_price = 0.0
    entry_time = None
    entry_multiplier = 1.0
    entry_fee_paid = 0.0

    trade_cash = 0.0
    fees_cash = 0.0
    funding_cash = 0.0
    funding_cash_since_entry = 0.0

    trades: list[TradeRecord] = []
    pnl_series: list = []
    pending = None  # (target_sign, target_mult) decided at previous bar's close

    def close_position(exit_price: float, exit_time) -> None:
        nonlocal position_sign, position_base, trade_cash, fees_cash, entry_fee_paid, funding_cash_since_entry
        size_usd = abs(position_base) * exit_price
        if position_sign > 0:
            trade_cash += size_usd
        else:
            trade_cash -= size_usd
        exit_fee = size_usd * taker_rate
        fees_cash -= exit_fee

        directional_pnl = position_base * (exit_price - entry_price)
        funding_pnl = funding_cash_since_entry
        fees_pnl = -entry_fee_paid - exit_fee
        total = directional_pnl + funding_pnl + fees_pnl

        trades.append(TradeRecord(
            coin=coin, side="long" if position_sign > 0 else "short",
            entry_time=entry_time, exit_time=exit_time,
            entry_price=entry_price, exit_price=exit_price,
            size_usd=abs(position_base) * entry_price, size_multiplier=entry_multiplier,
            directional_pnl=directional_pnl, funding_pnl=funding_pnl,
            fees_pnl=fees_pnl, total_pnl=total,
        ))
        position_sign = 0
        position_base = 0.0

    def open_position(target_sign: int, target_mult: float, price: float, ts) -> None:
        nonlocal position_sign, position_base, entry_price, entry_time, entry_multiplier
        nonlocal trade_cash, fees_cash, entry_fee_paid, funding_cash_since_entry
        size_usd = order_notional_usd * target_mult
        position_base = target_sign * size_usd / price
        if target_sign > 0:
            trade_cash -= size_usd
        else:
            trade_cash += size_usd
        entry_fee_paid = size_usd * taker_rate
        fees_cash -= entry_fee_paid
        entry_price = price
        entry_time = ts
        entry_multiplier = target_mult
        position_sign = target_sign
        funding_cash_since_entry = 0.0

    for idx in range(n):
        bar = df.iloc[idx]
        funding_rate = funding_series[idx]
        funding_known = not np.isnan(funding_rate)

        if funding_known and position_sign != 0:
            payment = -position_base * float(bar["open"]) * float(funding_rate)
            funding_cash += payment
            funding_cash_since_entry += payment

        if pending is not None:
            target_sign, target_mult = pending
            if target_sign != position_sign:
                price = float(bar["open"])
                if position_sign != 0:
                    close_position(price, bar["timestamp"])
                if target_sign != 0:
                    open_position(target_sign, target_mult, price, bar["timestamp"])
            pending = None

        mark = float(bar["close"])
        directional_component = trade_cash + position_base * mark
        total_pnl = directional_component + funding_cash + fees_cash
        pnl_series.append((bar["timestamp"], directional_component, funding_cash, fees_cash, total_pnl))

        if idx < n - 1:
            decision_funding_rate = float(funding_rate) if funding_known else 0.0
            target_sign, target_mult = signal_model.decide(
                df, idx, position_sign, entry_price if position_sign != 0 else 0.0, decision_funding_rate
            )
            pending = (target_sign, target_mult)

    if position_sign != 0:
        last_bar = df.iloc[-1]
        close_position(float(last_bar["close"]), last_bar["timestamp"])
        directional_component = trade_cash + position_base * float(last_bar["close"])
        total_pnl = directional_component + funding_cash + fees_cash
        pnl_series[-1] = (last_bar["timestamp"], directional_component, funding_cash, fees_cash, total_pnl)

    return DirectionalResult(coin=coin, trades=trades, pnl_series=pnl_series)
