"""PHASE 2 SCAFFOLD -- NOT USED BY THE PHASE 1 BACKTESTER.

Intended to become a WebSocket collector that subscribes to Hyperliquid's
`l2Book` and `trades` channels for configured markets and appends snapshots
to disk (e.g. one newline-delimited JSON or parquet file per UTC day per
market), so that after 1-2 weeks of collection a queue/depth-aware,
event-driven backtester (Phase 2) has real order-book history to replay.

This module is intentionally unfinished. Do not build it out further until
Phase 1 backtest results (on the simplified top-of-book fill model) show a
strategy worth testing more rigorously -- collecting weeks of L2 data for a
strategy that doesn't survive Phase 1 is wasted effort.

Design notes for when this gets built out:
  - Subscribe via the SDK's `Info(..., skip_ws=False)` WS interface, or a
    raw websockets connection to wss://api.hyperliquid.xyz/ws.
  - Channels needed: `l2Book` (snapshot every 100-500ms is a client-side
    polling/throttling choice -- the feed itself pushes on book changes,
    so throttle on write, not on subscribe) and `trades` (pushed per fill).
  - Write append-only, partitioned by (coin, utc_date), so a crash loses at
    most one partial file and backtest replay can walk files in order.
  - Record wall-clock receipt time alongside exchange timestamps, since
    the replay engine will need to reason about feed latency, not just
    the exchange's own event ordering.
  - Output schema should be a strict superset of what backtest/fill_model.py
    needs so a future QueueAwareFillModel (see that file's stub) can be
    written against it without another data-format migration.

This file contains no live-trading or order-signing code, and never will --
it is a read-only market-data logger.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class CollectorConfig:
    coins: list[str]
    out_dir: Path
    l2_throttle_ms: int = 250


class L2TradeCollector:
    """Not implemented. Placeholder so the Phase 2 extension point exists
    in the package layout and its intended interface is documented before
    any of it is built."""

    def __init__(self, config: CollectorConfig):
        self.config = config

    def run(self) -> None:
        raise NotImplementedError(
            "Phase 2 data collector is scaffolded but not built. "
            "See module docstring for design notes."
        )
