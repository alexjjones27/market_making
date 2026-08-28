"""Thin wrapper over the official hyperliquid-python-sdk Info class.

Read-only market data only. No wallet, no signing, no order placement --
this module cannot place trades even if you tried; it never imports
Exchange or any signing utilities from the SDK.
"""
from __future__ import annotations

import time
from typing import Optional

from hyperliquid.info import Info
from hyperliquid.utils import constants

# Hyperliquid's stated cap on candleSnapshot: only the most recent 5000
# candles are returned per call, regardless of the requested range.
MAX_CANDLES_PER_CALL = 5000

_INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
    "1M": 30 * 24 * 60 * 60_000,
}


class HyperliquidClient:
    def __init__(self, base_url: str = constants.MAINNET_API_URL):
        self.info = Info(base_url, skip_ws=True)

    def get_candles(
        self,
        coin: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        request_delay_s: float = 0.15,
        max_retries: int = 5,
    ) -> list[dict]:
        """Paginate candleSnapshot forward from start_ms to end_ms, working
        around the 5000-candle-per-call cap. Returns raw candle dicts,
        deduplicated and sorted by open time.
        """
        if interval not in _INTERVAL_MS:
            raise ValueError(f"Unsupported interval {interval!r}; known: {sorted(_INTERVAL_MS)}")

        interval_ms = _INTERVAL_MS[interval]
        window_ms = MAX_CANDLES_PER_CALL * interval_ms

        all_candles: dict[int, dict] = {}
        cursor = start_ms
        while cursor < end_ms:
            window_end = min(cursor + window_ms, end_ms)
            candles = self._fetch_candles_with_retry(
                coin, interval, cursor, window_end, max_retries
            )
            if not candles:
                # No data in this window -- advance past it rather than
                # spinning forever (e.g. requesting before listing date).
                cursor = window_end
                time.sleep(request_delay_s)
                continue

            for c in candles:
                all_candles[c["t"]] = c

            last_open = max(c["t"] for c in candles)
            next_cursor = last_open + interval_ms
            if next_cursor <= cursor:
                # Safety valve against a pathological non-advancing response.
                break
            cursor = next_cursor
            time.sleep(request_delay_s)

        return [all_candles[k] for k in sorted(all_candles)]

    def _fetch_candles_with_retry(
        self, coin: str, interval: str, start_ms: int, end_ms: int, max_retries: int
    ) -> list[dict]:
        backoff = 1.0
        for attempt in range(max_retries):
            try:
                return self.info.candles_snapshot(coin, interval, start_ms, end_ms) or []
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff *= 2
        return []

    def get_funding_history(
        self,
        coin: str,
        start_ms: int,
        end_ms: int,
        request_delay_s: float = 0.15,
        max_retries: int = 5,
    ) -> list[dict]:
        """Paginate fundingHistory forward across [start_ms, end_ms).
        The endpoint's per-call record cap isn't documented, so this loop
        advances the cursor to just past the last returned timestamp and
        keeps going until it stops making progress -- safe whether the API
        caps at 500 records or returns everything in one shot.
        """
        all_records: dict[int, dict] = {}
        cursor = start_ms
        while cursor < end_ms:
            records = self._fetch_funding_with_retry(coin, cursor, end_ms, max_retries)
            if not records:
                break
            for r in records:
                all_records[r["time"]] = r
            last_time = max(r["time"] for r in records)
            next_cursor = last_time + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            time.sleep(request_delay_s)

        return [all_records[k] for k in sorted(all_records)]

    def _fetch_funding_with_retry(
        self, coin: str, start_ms: int, end_ms: int, max_retries: int
    ) -> list[dict]:
        backoff = 1.0
        for attempt in range(max_retries):
            try:
                return self.info.funding_history(coin, start_ms, end_ms) or []
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff *= 2
        return []
