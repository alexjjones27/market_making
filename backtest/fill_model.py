"""Fill simulation. `FillModel` is an abstract interface so the engine
never depends on which fill model is plugged in -- Phase 2's queue/depth
-aware model (once real L2 data exists) drops in here without touching
strategy/ or engine.py's control flow.

TopOfBookFillModel is the ONLY implemented model in Phase 1. It is a
crude approximation:
  - A fill is assumed whenever bar low <= bid or bar high >= ask, i.e. it
    assumes your resting order would have been reached by *some* trade in
    the bar.
  - It does NOT model queue position: on a real order book your resting
    order sits behind whatever depth was already there at that price, and
    may not fill even when price trades through it, if the volume that
    traded there was smaller than the queue ahead of you.
  - It approximates this missing queue effect with a single knob,
    `fill_fraction_of_bar_volume`, capping simulated fill size at that
    fraction of the bar's total volume. This is a blunt instrument, not a
    model of the real queue -- see reporting/summary.py for the caveat
    text shown with every backtest run.
"""
from __future__ import annotations

import abc
import dataclasses

import pandas as pd


@dataclasses.dataclass
class SimulatedFill:
    timestamp: pd.Timestamp
    coin: str
    side: str  # "buy" (hit bid) or "sell" (hit ask)
    price: float
    size: float  # base-asset units
    notional_usd: float
    resulting_inventory_usd: float


class FillModel(abc.ABC):
    @abc.abstractmethod
    def simulate_bar(
        self,
        coin: str,
        bar: pd.Series,
        bid: float,
        ask: float,
        bid_size_usd: float,
        ask_size_usd: float,
        current_inventory_usd: float,
    ) -> list[SimulatedFill]:
        """Given one OHLCV bar and the quotes resting during it, return
        zero or more simulated fills."""
        raise NotImplementedError


class TopOfBookFillModel(FillModel):
    def __init__(self, fill_fraction_of_bar_volume: float):
        self.fill_fraction_of_bar_volume = fill_fraction_of_bar_volume

    def simulate_bar(
        self,
        coin: str,
        bar: pd.Series,
        bid: float,
        ask: float,
        bid_size_usd: float,
        ask_size_usd: float,
        current_inventory_usd: float,
    ) -> list[SimulatedFill]:
        fills: list[SimulatedFill] = []
        bar_volume_base = float(bar["volume"])
        fillable_base = bar_volume_base * self.fill_fraction_of_bar_volume
        inventory_usd = current_inventory_usd

        if bar["low"] <= bid and bid_size_usd > 0:
            requested_base = bid_size_usd / bid
            fill_base = min(requested_base, fillable_base)
            if fill_base > 0:
                notional = fill_base * bid
                inventory_usd += notional
                fills.append(
                    SimulatedFill(
                        timestamp=bar["timestamp"],
                        coin=coin,
                        side="buy",
                        price=bid,
                        size=fill_base,
                        notional_usd=notional,
                        resulting_inventory_usd=inventory_usd,
                    )
                )

        if bar["high"] >= ask and ask_size_usd > 0:
            requested_base = ask_size_usd / ask
            fill_base = min(requested_base, fillable_base)
            if fill_base > 0:
                notional = fill_base * ask
                inventory_usd -= notional
                fills.append(
                    SimulatedFill(
                        timestamp=bar["timestamp"],
                        coin=coin,
                        side="sell",
                        price=ask,
                        size=fill_base,
                        notional_usd=notional,
                        resulting_inventory_usd=inventory_usd,
                    )
                )

        return fills


class QueueAwareFillModel(FillModel):
    """PHASE 2 EXTENSION POINT -- not implemented.

    Once data/collector_ws.py has accumulated real L2 snapshots + trades,
    this class should replay actual queue position: track where a resting
    order would sit in the book at its price level, decrement queue ahead
    of it as trades print, and only fill once the order reaches the front.
    Swapping this in for TopOfBookFillModel requires no changes to
    strategy/ or backtest/engine.py -- the engine only calls
    `simulate_bar` (or its future streaming equivalent) through the
    FillModel interface.
    """

    def simulate_bar(self, *args, **kwargs) -> list[SimulatedFill]:
        raise NotImplementedError("Phase 2: requires collected L2 book/trade data.")
