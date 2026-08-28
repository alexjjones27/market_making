"""Fallback mean-reversion baseline signal (no existing MR backtester
found). Z-score of price vs. a rolling mean/std; enter on a large enough
dislocation, exit on reversion toward the mean or a stop-loss. Unlike
trend, this needs to know the current position and entry price (to check
reversion/stop), which the engine passes in -- the signal itself stays a
pure function of (df, idx, current_sign, entry_price).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class MeanReversionSignal:
    def __init__(self, lookback_bars: int, entry_z: float, exit_z: float, stop_loss_pct: float):
        self.lookback_bars = lookback_bars
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss_pct = stop_loss_pct
        self._z: np.ndarray | None = None
        self._close: np.ndarray | None = None

    def prepare(self, df: pd.DataFrame) -> None:
        self._close = df["close"].to_numpy()
        roll_mean = df["close"].rolling(self.lookback_bars).mean()
        roll_std = df["close"].rolling(self.lookback_bars).std()
        self._z = ((df["close"] - roll_mean) / roll_std).to_numpy()

    def decide(self, df: pd.DataFrame, idx: int, current_sign: int, entry_price: float, funding_rate: float) -> tuple[int, float]:
        if idx < self.lookback_bars:
            return 0, 1.0
        z = self._z[idx]
        if np.isnan(z):
            return 0, 1.0
        price = self._close[idx]

        if current_sign == 0:
            if z <= -self.entry_z:
                return 1, 1.0
            if z >= self.entry_z:
                return -1, 1.0
            return 0, 1.0

        if current_sign == 1:
            if z >= -self.exit_z:
                return 0, 1.0  # reverted back toward/through the mean
            if entry_price > 0 and (entry_price - price) / entry_price >= self.stop_loss_pct:
                return 0, 1.0  # stop-loss
            return 1, 1.0  # hold

        if current_sign == -1:
            if z <= self.exit_z:
                return 0, 1.0
            if entry_price > 0 and (price - entry_price) / entry_price >= self.stop_loss_pct:
                return 0, 1.0
            return -1, 1.0

        return 0, 1.0
