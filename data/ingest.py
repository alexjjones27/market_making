"""Data ingestion: pull OHLCV candles + funding rate history for configured
markets, with disk caching so repeated runs don't re-hit the API.

Historical public trade-level data is deliberately NOT ingested here: the
Hyperliquid Info API has no endpoint for a market-wide trade tape (only a
user's own fills, which requires that user to have actually traded). The
fill simulator instead works off candle high/low/volume, which is what it
needs anyway -- see backtest/fill_model.py.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from config.loader import Config, MarketSpec
from data import cache
from data.hyperliquid_client import HyperliquidClient


def _date_to_ms(date_str: str) -> int:
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


def candles_to_df(raw_candles: list[dict]) -> pd.DataFrame:
    if not raw_candles:
        return pd.DataFrame(
            columns=["t", "T", "open", "high", "low", "close", "volume", "n_trades"]
        )
    df = pd.DataFrame(raw_candles)
    df = df.rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "n": "n_trades"}
    )
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df[["t", "T", "timestamp", "open", "high", "low", "close", "volume", "n_trades"]].sort_values("t").reset_index(drop=True)


def funding_to_df(raw_funding: list[dict]) -> pd.DataFrame:
    if not raw_funding:
        return pd.DataFrame(columns=["time", "timestamp", "fundingRate", "premium"])
    df = pd.DataFrame(raw_funding)
    df["fundingRate"] = df["fundingRate"].astype(float)
    if "premium" in df.columns:
        df["premium"] = df["premium"].astype(float)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.sort_values("time").reset_index(drop=True)


def ingest_market(client: HyperliquidClient, cache_dir: Path, spec: MarketSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (candles_df, funding_df) for one market, using cache when
    available and hitting the API only for missing data."""
    candles_df = cache.load_candles_from_cache(cache_dir, spec.coin, spec.candle_interval, spec.start_date, spec.end_date)
    if candles_df is None:
        start_ms = _date_to_ms(spec.start_date)
        end_ms = _date_to_ms(spec.end_date)
        raw = client.get_candles(spec.coin, spec.candle_interval, start_ms, end_ms)
        candles_df = candles_to_df(raw)
        cache.save_candles_to_cache(cache_dir, spec.coin, spec.candle_interval, spec.start_date, spec.end_date, candles_df)
        print(f"[ingest] {spec.coin}: fetched {len(candles_df)} candles from API")
    else:
        print(f"[ingest] {spec.coin}: loaded {len(candles_df)} candles from cache")

    funding_df = cache.load_funding_from_cache(cache_dir, spec.coin, spec.start_date, spec.end_date)
    if funding_df is None:
        start_ms = _date_to_ms(spec.start_date)
        end_ms = _date_to_ms(spec.end_date)
        raw = client.get_funding_history(spec.coin, start_ms, end_ms)
        funding_df = funding_to_df(raw)
        cache.save_funding_to_cache(cache_dir, spec.coin, spec.start_date, spec.end_date, funding_df)
        print(f"[ingest] {spec.coin}: fetched {len(funding_df)} funding records from API")
    else:
        print(f"[ingest] {spec.coin}: loaded {len(funding_df)} funding records from cache")

    return candles_df, funding_df


def ingest_all(config: Config) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    client = HyperliquidClient()
    cache_dir = Path(config.markets.cache_dir)
    result = {}
    for spec in config.markets.markets:
        result[spec.coin] = ingest_market(client, cache_dir, spec)
    return result
