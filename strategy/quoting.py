"""Quote generation: fair value -> (bid, ask), with inventory skew and
volatility-adjusted spread widening.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from config.loader import StrategyConfig


@dataclasses.dataclass
class Quote:
    bid: float
    ask: float
    fair_value: float
    spread_bps_used: float
    skew_bps_applied: float
    vol_widened: bool


def realized_vol_bps(
    history: pd.DataFrame, idx: int, window_secs: int, bar_interval_secs: int = 60
) -> float:
    """Annualized realized vol (in bps) of log returns of bar closes over
    a trailing window. Used purely as a spread-widening trigger, not a
    calibrated risk measure.

    Bounds the lookback slice instead of rescanning all history each call
    -- see strategy/fair_value.py's make_vwap docstring for why.
    """
    max_bars = max(3, int(window_secs / bar_interval_secs) + 2)
    now = history.iloc[idx]["timestamp"]
    window_start = now - pd.Timedelta(seconds=window_secs)
    lo = max(0, idx - max_bars)
    window = history.iloc[lo : idx + 1]
    window = window[window["timestamp"] >= window_start]
    if len(window) < 3:
        return 0.0
    closes = window["close"].to_numpy()
    log_ret = np.diff(np.log(closes))
    if log_ret.size == 0 or log_ret.std() == 0:
        return 0.0
    # Annualize assuming ~1 bar per `bar_interval`; we approximate using
    # the observed average spacing in the window rather than assuming 1m.
    dt_secs = window["timestamp"].diff().dt.total_seconds().dropna()
    avg_dt = dt_secs.mean() if not dt_secs.empty else 60.0
    bars_per_year = (365.25 * 24 * 3600) / avg_dt
    ann_vol = log_ret.std() * np.sqrt(bars_per_year)
    return float(ann_vol * 10_000)


def generate_quote(
    fair_value: float,
    position_usd: float,
    strategy: StrategyConfig,
    realized_vol_bps_value: float,
) -> Quote:
    spread_bps = strategy.quoting.spread_bps
    vol_widened = False
    if realized_vol_bps_value > strategy.volatility.widen_threshold_bps:
        spread_bps *= strategy.volatility.widen_multiplier
        vol_widened = True

    max_pos = strategy.inventory.max_position_usd
    inventory_frac = 0.0 if max_pos == 0 else max(-1.0, min(1.0, position_usd / max_pos))
    # Positive inventory (long) -> shift quotes down to encourage selling.
    skew_bps = -inventory_frac * strategy.quoting.max_skew_bps

    half_spread = fair_value * (spread_bps / 10_000) / 2.0
    skew_shift = fair_value * (skew_bps / 10_000)

    bid = fair_value - half_spread + skew_shift
    ask = fair_value + half_spread + skew_shift

    return Quote(
        bid=bid,
        ask=ask,
        fair_value=fair_value,
        spread_bps_used=spread_bps,
        skew_bps_applied=skew_bps,
        vol_widened=vol_widened,
    )
