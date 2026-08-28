"""Pluggable fair-value estimators. Each takes the bar history up to and
including the current bar (a pandas DataFrame with at least
timestamp/open/high/low/close/volume columns) and returns a single float
fair value for the current bar.

Both estimators are bar-close approximations: Phase 1 only has OHLCV bars,
not a real-time tick stream, so "VWAP over the last 45s" really means
"volume-weighted average of recent bar closes," which is coarser than a
true tick VWAP. This is noted in reporting/summary.py as a limitation.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class FairValueModel(Protocol):
    def __call__(self, history: pd.DataFrame, idx: int) -> float: ...


def mid_price(history: pd.DataFrame, idx: int, bar_interval_secs: int = 60) -> float:
    """Mid of the current bar's high/low as a proxy for top-of-book mid."""
    row = history.iloc[idx]
    return (row["high"] + row["low"]) / 2.0


def make_vwap(window_secs: int, bar_interval_secs: int = 60) -> FairValueModel:
    """Volume-weighted average of bar closes over a trailing time window.
    Bar-granularity approximation of a short-window trade VWAP -- see
    module docstring.

    Bounds the lookback slice to a small number of bars (derived from
    window_secs / bar_interval_secs) instead of re-scanning all history on
    every call -- with ~1 bar/minute over weeks of data, an unbounded scan
    is O(n^2) over the run and becomes the dominant cost.
    """
    max_bars = max(2, int(window_secs / bar_interval_secs) + 2)

    def _vwap(history: pd.DataFrame, idx: int) -> float:
        now = history.iloc[idx]["timestamp"]
        window_start = now - pd.Timedelta(seconds=window_secs)
        lo = max(0, idx - max_bars)
        window = history.iloc[lo : idx + 1]
        window = window[window["timestamp"] >= window_start]
        if window.empty or window["volume"].sum() == 0:
            return mid_price(history, idx)
        typical_price = (window["high"] + window["low"] + window["close"]) / 3.0
        return float((typical_price * window["volume"]).sum() / window["volume"].sum())

    return _vwap


def build_fair_value_model(method: str, vwap_window_secs: int, bar_interval_secs: int = 60) -> FairValueModel:
    if method == "mid":
        return mid_price
    if method == "vwap":
        return make_vwap(vwap_window_secs, bar_interval_secs)
    raise ValueError(f"Unknown fair_value method {method!r}; use 'mid' or 'vwap'")
