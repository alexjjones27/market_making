"""Carry-gating/bias wrappers around a base signal model. These NEVER
touch the base model's signal logic, sizing, or exit rules -- they only
intercept the moment a *new* entry is proposed (a transition from flat, or
a direct flip) and either allow it, block it, or size it, per the two
strategies' specs. Exits proposed by the base model always pass through
unchanged, since carry gating/bias is about which trades you take, not an
extra exit condition layered on top of the base strategy.

CarryOnlySignal is not a wrapper (there is no base directional signal to
wrap) -- it's the third comparison arm: hold whichever side funding
currently favours, full stop.
"""
from __future__ import annotations

import pandas as pd

from strategy import carry


class CarryGatedSignal:
    """Strategy 1: carry-aligned trend. Hard gate on funding SIGN only (no
    magnitude threshold -- that's Strategy 2's "strongly" concept, not
    used here). Disagreement skips the trade entirely; never flips it,
    never takes a reduced size."""

    def __init__(self, base_model):
        self.base = base_model

    def prepare(self, df: pd.DataFrame) -> None:
        self.base.prepare(df)

    def decide(self, df: pd.DataFrame, idx: int, current_sign: int, entry_price: float, funding_rate: float) -> tuple[int, float]:
        base_target, _ = self.base.decide(df, idx, current_sign, entry_price, funding_rate)

        if base_target == current_sign:
            return current_sign, 1.0  # no change proposed -- nothing to gate
        if base_target == 0:
            return 0, 1.0  # exit -- always allowed, base model's own logic

        # A new entry (from flat, or a direct flip) is being proposed.
        favoured_side = carry.side_funding_favours(funding_rate)
        if base_target == favoured_side:
            return base_target, 1.0
        return 0, 1.0  # disagreement -- skip, do not take, do not flip, do not reduce size


class CarryBiasedSignal:
    """Strategy 2: carry + mean-reversion, soft bias. Funding direction and
    magnitude bias the SIZE of a new entry (via multiplier), never blocks
    it outright unless the configured opposed_multiplier is 0."""

    def __init__(self, base_model, threshold_annualized_pct: float, aligned_multiplier: float, opposed_multiplier: float, neutral_multiplier: float):
        self.base = base_model
        self.threshold_annualized_pct = threshold_annualized_pct
        self.aligned_multiplier = aligned_multiplier
        self.opposed_multiplier = opposed_multiplier
        self.neutral_multiplier = neutral_multiplier

    def prepare(self, df: pd.DataFrame) -> None:
        self.base.prepare(df)

    def decide(self, df: pd.DataFrame, idx: int, current_sign: int, entry_price: float, funding_rate: float) -> tuple[int, float]:
        base_target, _ = self.base.decide(df, idx, current_sign, entry_price, funding_rate)

        if base_target == current_sign:
            return current_sign, 1.0
        if base_target == 0:
            return 0, 1.0

        favoured_side = carry.side_funding_favours(funding_rate)
        strong = carry.is_strong(funding_rate, self.threshold_annualized_pct)

        if not strong or favoured_side == 0:
            multiplier = self.neutral_multiplier
        elif base_target == favoured_side:
            multiplier = self.aligned_multiplier
        else:
            multiplier = self.opposed_multiplier

        if multiplier <= 0:
            return 0, 1.0  # configured as a hard skip when opposed
        return base_target, multiplier


class CarryOnlySignal:
    """Reference baseline: no directional signal at all. Just hold
    whichever side funding currently favours, with fixed (unbiased) size."""

    def prepare(self, df: pd.DataFrame) -> None:
        pass

    def decide(self, df: pd.DataFrame, idx: int, current_sign: int, entry_price: float, funding_rate: float) -> tuple[int, float]:
        favoured_side = carry.side_funding_favours(funding_rate)
        return favoured_side, 1.0
