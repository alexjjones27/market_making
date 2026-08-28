"""Funding-direction/bias logic shared by both carry strategies and the
carry-only baseline.

Sign convention (Hyperliquid, and standard perp convention generally):
  funding_rate > 0  ->  longs pay shorts   -> funding is "paying shorts"
  funding_rate < 0  ->  shorts pay longs   -> funding is "paying longs"

Hyperliquid funding settles hourly (confirmed empirically from cached
funding-history pulls: ~24 records/day). `annualized_pct` assumes an
hourly rate; if you point this at a different interval's funding data,
adjust the annualization factor.
"""
from __future__ import annotations

HOURS_PER_YEAR = 24 * 365


def funding_direction(rate: float) -> str:
    """'pays_longs', 'pays_shorts', or 'neutral' (exact zero)."""
    if rate < 0:
        return "pays_longs"
    if rate > 0:
        return "pays_shorts"
    return "neutral"


def annualized_pct(hourly_rate: float) -> float:
    return hourly_rate * HOURS_PER_YEAR * 100.0


def is_strong(hourly_rate: float, threshold_annualized_pct: float) -> bool:
    return abs(annualized_pct(hourly_rate)) >= threshold_annualized_pct


def side_funding_favours(rate: float) -> int:
    """+1 if funding favours being long (longs receive), -1 if it favours
    being short, 0 if exactly neutral."""
    direction = funding_direction(rate)
    if direction == "pays_longs":
        return 1
    if direction == "pays_shorts":
        return -1
    return 0
