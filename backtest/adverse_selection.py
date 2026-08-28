"""Adverse-selection stress-test overlay (backtest-analysis-only).

The top-of-book fill model (backtest/fill_model.py) cannot distinguish a
"good" fill from a "toxic" one -- it only knows price touched your quote,
not whether that touch was the start of a move that keeps going against
you. This module estimates that blind spot's cost after the fact: for
each simulated fill, look `lookahead_bars` bars ahead; if price has since
moved more than `threshold_bps` against the fill, charge the full realized
adverse move on that fill's notional as an extra cost.

This requires knowing what happened AFTER the fill, so it is inherently a
look-ahead calculation. It can only ever be a backtest stress test to
gauge how much of the strategy's apparent edge might evaporate once
adverse selection is priced in -- it is not something a live strategy
could compute in real time, and must never be wired into engine.py's
quoting decisions.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class AdverseSelectionResult:
    coin: str
    penalty_cash: float  # negative = net cost
    n_evaluable_fills: int  # fills with enough future bars to evaluate
    n_toxic_fills: int  # fills where the penalty triggered
    events: list  # list of (timestamp, penalty_cash) for equity-curve adjustment


def compute_adverse_selection(market_states: dict, threshold_bps: float, lookahead_bars: int) -> dict:
    """market_states: coin -> MarketRunState (from backtest.engine).
    Returns coin -> AdverseSelectionResult.
    """
    results = {}
    for coin, ms in market_states.items():
        df = ms.df
        ts_to_idx = {ts: i for i, ts in enumerate(df["timestamp"])}
        closes = df["close"].to_numpy()
        n_bars = len(df)

        penalty_cash = 0.0
        n_evaluable = 0
        n_toxic = 0
        events = []

        for f in ms.fills:
            idx = ts_to_idx.get(f.timestamp)
            if idx is None:
                continue
            future_idx = idx + lookahead_bars
            if future_idx >= n_bars:
                continue  # not enough future data to evaluate this fill
            n_evaluable += 1

            future_price = float(closes[future_idx])
            is_buy = f.side in ("buy", "flatten_buy")
            if is_buy:
                adverse_frac = (f.price - future_price) / f.price
            else:
                adverse_frac = (future_price - f.price) / f.price

            if adverse_frac * 10_000 > threshold_bps:
                n_toxic += 1
                cost = -f.notional_usd * adverse_frac  # negative cash flow
                penalty_cash += cost
                events.append((f.timestamp, cost))

        results[coin] = AdverseSelectionResult(
            coin=coin,
            penalty_cash=penalty_cash,
            n_evaluable_fills=n_evaluable,
            n_toxic_fills=n_toxic,
            events=events,
        )
    return results
