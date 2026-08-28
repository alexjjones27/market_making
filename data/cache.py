"""Disk cache for raw API pulls, as parquet. Keyed by coin/interval/date
range so repeated backtest runs don't re-hit the API.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _candles_cache_path(cache_dir: Path, coin: str, interval: str, start_date: str, end_date: str) -> Path:
    return cache_dir / "candles" / f"{coin}_{interval}_{start_date}_{end_date}.parquet"


def _funding_cache_path(cache_dir: Path, coin: str, start_date: str, end_date: str) -> Path:
    return cache_dir / "funding" / f"{coin}_{start_date}_{end_date}.parquet"


def load_candles_from_cache(cache_dir: Path, coin: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    path = _candles_cache_path(cache_dir, coin, interval, start_date, end_date)
    if path.exists():
        return pd.read_parquet(path)
    return None


def save_candles_to_cache(cache_dir: Path, coin: str, interval: str, start_date: str, end_date: str, df: pd.DataFrame) -> None:
    path = _candles_cache_path(cache_dir, coin, interval, start_date, end_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_funding_from_cache(cache_dir: Path, coin: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    path = _funding_cache_path(cache_dir, coin, start_date, end_date)
    if path.exists():
        return pd.read_parquet(path)
    return None


def save_funding_to_cache(cache_dir: Path, coin: str, start_date: str, end_date: str, df: pd.DataFrame) -> None:
    path = _funding_cache_path(cache_dir, coin, start_date, end_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
