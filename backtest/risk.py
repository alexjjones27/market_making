"""Risk controls enforced by the backtest engine, not just documented as
intent. All three of the spec'd controls live here:
  - max inventory per market (engine caps order size against this)
  - daily loss limit -> halts quoting for the rest of that UTC day
  - single-market stop loss -> auto-flatten that market's position
"""
from __future__ import annotations

import dataclasses
import datetime as dt

from config.loader import RiskConfig


@dataclasses.dataclass
class RiskState:
    capital_usd: float
    current_utc_day: dt.date | None = None
    day_start_equity: float = 0.0
    halted_today: bool = False
    flattened_markets: set = dataclasses.field(default_factory=set)
    halt_log: list = dataclasses.field(default_factory=list)
    flatten_log: list = dataclasses.field(default_factory=list)


def new_risk_state(config: RiskConfig) -> RiskState:
    return RiskState(capital_usd=config.capital_usd)


def max_position_usd(config: RiskConfig, strategy_max_position_usd: float) -> float:
    """The engine enforces the tighter of the strategy's inventory config
    and the risk config's hard cap."""
    return min(config.max_inventory_usd, strategy_max_position_usd)


def roll_day_if_needed(state: RiskState, timestamp, current_equity: float) -> None:
    day = timestamp.date()
    if state.current_utc_day != day:
        state.current_utc_day = day
        state.day_start_equity = current_equity
        state.halted_today = False


def check_daily_loss_limit(state: RiskState, timestamp, current_equity: float, config: RiskConfig) -> bool:
    """Returns True if quoting should be halted right now. Updates
    state.halted_today and logs the breach the first time it happens."""
    if state.halted_today:
        return True
    daily_pnl = current_equity - state.day_start_equity
    limit = -config.daily_loss_limit_pct * state.capital_usd
    if daily_pnl <= limit:
        state.halted_today = True
        state.halt_log.append(
            {"timestamp": timestamp, "daily_pnl": daily_pnl, "limit": limit}
        )
        return True
    return False


def check_single_market_stop(
    state: RiskState, timestamp, coin: str, market_cum_pnl: float, config: RiskConfig
) -> bool:
    """Returns True if `coin` should be flattened right now (first time the
    stop is breached only; once flattened it stays flattened for the rest
    of the run, matching 'auto-flatten' semantics rather than a resettable
    halt)."""
    if coin in state.flattened_markets:
        return True
    limit = -config.single_market_stop_loss_pct * state.capital_usd
    if market_cum_pnl <= limit:
        state.flattened_markets.add(coin)
        state.flatten_log.append(
            {"timestamp": timestamp, "coin": coin, "market_cum_pnl": market_cum_pnl, "limit": limit}
        )
        return True
    return False
