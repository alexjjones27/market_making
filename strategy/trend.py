"""Fallback trend/momentum baseline signal (no existing trend backtester
found in this repo). Kept intentionally simple and stateless: long while
close > N-bar MA and the MA is rising, short while close < N-bar MA and
falling, flat otherwise -- recomputed fresh every bar. Entries and exits
both fall out of this one rule, which is what lets carry-gating stay a
pure wrapper that only ever filters entries (see strategy/carry_wrappers.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class TrendSignal:
    def __init__(self, ma_window_bars: int):
        self.ma_window_bars = ma_window_bars
        self._ma: np.ndarray | None = None
        self._close: np.ndarray | None = None

    def prepare(self, df: pd.DataFrame) -> None:
        self._close = df["close"].to_numpy()
        self._ma = df["close"].rolling(self.ma_window_bars).mean().to_numpy()

    def decide(self, df: pd.DataFrame, idx: int, current_sign: int, entry_price: float, funding_rate: float) -> tuple[int, float]:
        if idx < self.ma_window_bars:
            return 0, 1.0
        ma_now = self._ma[idx]
        ma_prev = self._ma[idx - 1]
        close_now = self._close[idx]
        if np.isnan(ma_now) or np.isnan(ma_prev):
            return 0, 1.0

        if close_now > ma_now and ma_now > ma_prev:
            return 1, 1.0
        if close_now < ma_now and ma_now < ma_prev:
            return -1, 1.0
        return 0, 1.0
