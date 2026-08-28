"""Bar-by-bar backtest engine. Runs one or more markets over the same
timeline, sharing a single risk.RiskState (capital, daily loss halt) while
tracking each market's inventory, fills, fees, and funding independently.

PnL is decomposed per market into three additive cash-flow streams that
always sum to that market's total PnL:
  - trading_pnl: mark-to-market value of trades (this is "spread capture"
    when the strategy is behaving as a market maker rather than taking
    directional risk -- it isn't separated from adverse selection, see
    reporting/summary.py)
  - fees_cash: cumulative maker fee/rebate cash flow (positive = net
    rebate received)
  - funding_cash: cumulative funding cash flow (positive = net received)
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from backtest import risk as risk_mod
from backtest.fill_model import FillModel, SimulatedFill
from config.loader import Config
from strategy import fair_value as fv
from strategy import quoting as qt

_INTERVAL_SECS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "8h": 28800, "12h": 43200,
    "1d": 86400, "3d": 259200, "1w": 604800, "1M": 2592000,
}


@dataclasses.dataclass
class MarketRunState:
    coin: str
    df: pd.DataFrame
    funding_df: pd.DataFrame
    bar_interval_secs: int
    funding_ptr: int = 0
    position_base: float = 0.0
    trade_cash: float = 0.0
    fees_cash: float = 0.0
    funding_cash: float = 0.0
    fills: list = dataclasses.field(default_factory=list)
    inventory_series: list = dataclasses.field(default_factory=list)  # (ts, position_base, position_usd)
    pnl_series: list = dataclasses.field(default_factory=list)  # (ts, trading_pnl, fees_cash, funding_cash, total_pnl)
    last_total_pnl: float = 0.0

    def total_pnl(self, mark_price: float) -> float:
        trading_pnl = self.trade_cash + self.position_base * mark_price
        return trading_pnl + self.fees_cash + self.funding_cash


@dataclasses.dataclass
class BacktestResult:
    market_states: dict  # coin -> MarketRunState
    risk_state: risk_mod.RiskState
    global_equity_series: list  # (ts, equity)
    quote_bar_count: dict  # coin -> number of bars where quoting was active


class BacktestEngine:
    def __init__(self, config: Config, fill_model: FillModel):
        self.config = config
        self.fill_model = fill_model
        self.taker_rate, self.maker_rate = config.fees.effective_rates()

    def run(self, data: dict) -> BacktestResult:
        """data: coin -> (candles_df, funding_df)"""
        risk_state = risk_mod.new_risk_state(self.config.risk)
        allowed_max_position = risk_mod.max_position_usd(
            self.config.risk, self.config.strategy.inventory.max_position_usd
        )

        market_states: dict[str, MarketRunState] = {}
        fair_value_fns: dict[str, fv.FairValueModel] = {}
        for coin, (candles_df, funding_df) in data.items():
            spec = next(m for m in self.config.markets.markets if m.coin == coin)
            interval_secs = _INTERVAL_SECS[spec.candle_interval]
            candles_df = candles_df.reset_index(drop=True)
            market_states[coin] = MarketRunState(
                coin=coin, df=candles_df, funding_df=funding_df.reset_index(drop=True),
                bar_interval_secs=interval_secs,
            )
            fair_value_fns[coin] = fv.build_fair_value_model(
                self.config.strategy.fair_value.method,
                self.config.strategy.fair_value.vwap_window_secs,
                interval_secs,
            )

        # Merge timelines: sorted union of all markets' bar timestamps.
        all_ts = np.unique(np.concatenate([ms.df["t"].to_numpy() for ms in market_states.values()]))
        ptrs = {coin: 0 for coin in market_states}
        quote_bar_count = {coin: 0 for coin in market_states}
        global_equity_series = []

        for t in all_ts:
            active_this_tick = []
            for coin, ms in market_states.items():
                p = ptrs[coin]
                if p < len(ms.df) and ms.df["t"].iat[p] == t:
                    active_this_tick.append(coin)

            for coin in active_this_tick:
                ms = market_states[coin]
                idx = ptrs[coin]
                bar = ms.df.iloc[idx]

                self._apply_funding(ms, bar)

                already_flattened = coin in risk_state.flattened_markets
                quoting_allowed = (not already_flattened) and (not risk_state.halted_today)

                if quoting_allowed:
                    quote_bar_count[coin] += 1
                    fair_value = fair_value_fns[coin](ms.df, idx)
                    vol_bps = qt.realized_vol_bps(
                        ms.df, idx, self.config.strategy.volatility.window_secs, ms.bar_interval_secs
                    )
                    ref_price = float(bar["open"])
                    current_inventory_usd = ms.position_base * ref_price
                    quote = qt.generate_quote(
                        fair_value, current_inventory_usd, self.config.strategy, vol_bps
                    )
                    order_size = self.config.strategy.quoting.order_size_usd
                    bid_size_usd = min(order_size, max(0.0, allowed_max_position - current_inventory_usd))
                    ask_size_usd = min(order_size, max(0.0, allowed_max_position + current_inventory_usd))

                    fills = self.fill_model.simulate_bar(
                        coin, bar, quote.bid, quote.ask, bid_size_usd, ask_size_usd, current_inventory_usd
                    )
                    for f in fills:
                        self._apply_fill(ms, f, taker=False)

                mark_price = float(bar["close"])
                total_pnl = ms.total_pnl(mark_price)
                ms.last_total_pnl = total_pnl
                ms.inventory_series.append((bar["timestamp"], ms.position_base, ms.position_base * mark_price))
                ms.pnl_series.append(
                    (bar["timestamp"], ms.trade_cash + ms.position_base * mark_price, ms.fees_cash, ms.funding_cash, total_pnl)
                )

                if not already_flattened:
                    should_flatten = risk_mod.check_single_market_stop(
                        risk_state, bar["timestamp"], coin, total_pnl, self.config.risk
                    )
                    if should_flatten:
                        self._flatten_position(ms, bar)
                        mark_price = float(bar["close"])
                        total_pnl = ms.total_pnl(mark_price)
                        ms.last_total_pnl = total_pnl
                        ms.inventory_series[-1] = (bar["timestamp"], ms.position_base, ms.position_base * mark_price)
                        ms.pnl_series[-1] = (
                            bar["timestamp"], ms.trade_cash + ms.position_base * mark_price,
                            ms.fees_cash, ms.funding_cash, total_pnl,
                        )

                ptrs[coin] = idx + 1

            global_equity = self.config.risk.capital_usd + sum(ms.last_total_pnl for ms in market_states.values())
            ts_dt = market_states[active_this_tick[0]].df.iloc[ptrs[active_this_tick[0]] - 1]["timestamp"]
            risk_mod.roll_day_if_needed(risk_state, ts_dt, global_equity)
            risk_mod.check_daily_loss_limit(risk_state, ts_dt, global_equity, self.config.risk)
            global_equity_series.append((ts_dt, global_equity))

        return BacktestResult(
            market_states=market_states,
            risk_state=risk_state,
            global_equity_series=global_equity_series,
            quote_bar_count=quote_bar_count,
        )

    def _apply_funding(self, ms: MarketRunState, bar: pd.Series) -> None:
        while ms.funding_ptr < len(ms.funding_df) and ms.funding_df["time"].iat[ms.funding_ptr] <= bar["t"]:
            rate = float(ms.funding_df["fundingRate"].iat[ms.funding_ptr])
            mark = float(bar["open"])
            ms.funding_cash -= ms.position_base * mark * rate
            ms.funding_ptr += 1

    def _apply_fill(self, ms: MarketRunState, f: SimulatedFill, taker: bool) -> None:
        rate = self.taker_rate if taker else self.maker_rate
        if f.side in ("buy", "flatten_buy"):
            ms.trade_cash -= f.notional_usd
            ms.position_base += f.size
        else:
            ms.trade_cash += f.notional_usd
            ms.position_base -= f.size
        ms.fees_cash -= f.notional_usd * rate
        ms.fills.append(f)

    def _flatten_position(self, ms: MarketRunState, bar: pd.Series) -> None:
        if ms.position_base == 0:
            return
        price = float(bar["close"])
        size = abs(ms.position_base)
        side = "flatten_sell" if ms.position_base > 0 else "flatten_buy"
        fill = SimulatedFill(
            timestamp=bar["timestamp"], coin=ms.coin, side=side, price=price,
            size=size, notional_usd=size * price, resulting_inventory_usd=0.0,
        )
        self._apply_fill(ms, fill, taker=True)
