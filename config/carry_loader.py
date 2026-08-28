"""Config loading for the directional carry backtests. Deliberately
separate from config/loader.py (the market-making backtester's config),
since the two problems have different shaped configs -- kept isolated so
neither has to be contorted to fit the other.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclasses.dataclass
class MarketSpec:
    coin: str
    start_date: str
    end_date: str
    candle_interval: str


@dataclasses.dataclass
class MarketsConfig:
    markets: list[MarketSpec]
    cache_dir: str


@dataclasses.dataclass
class FeeTier:
    tier: int
    min_14d_volume_usd: float
    taker_rate: float
    maker_rate: float


@dataclasses.dataclass
class FeesConfig:
    volume_tiers: list[FeeTier]
    assumed_volume_tier: int

    def taker_rate(self) -> float:
        return next(t for t in self.volume_tiers if t.tier == self.assumed_volume_tier).taker_rate


@dataclasses.dataclass
class RiskConfig:
    capital_usd: float
    order_notional_usd: float


@dataclasses.dataclass
class TrendConfig:
    ma_window_bars: int


@dataclasses.dataclass
class MeanReversionConfig:
    lookback_bars: int
    entry_z: float
    exit_z: float
    stop_loss_pct: float


@dataclasses.dataclass
class CarryConfig:
    strong_threshold_annualized_pct: float
    aligned_multiplier: float
    opposed_multiplier: float
    neutral_multiplier: float


@dataclasses.dataclass
class CarryBacktestConfig:
    risk: RiskConfig
    fees: FeesConfig
    trend: TrendConfig
    mean_reversion: MeanReversionConfig
    carry: CarryConfig


def load_carry_config(config_dir: Path) -> CarryBacktestConfig:
    risk = RiskConfig(**_load_yaml(config_dir / "risk.yaml"))

    f = _load_yaml(config_dir / "fees.yaml")
    fees = FeesConfig(
        volume_tiers=[FeeTier(**t) for t in f["volume_tiers"]],
        assumed_volume_tier=f["assumed_volume_tier"],
    )

    trend = TrendConfig(**_load_yaml(config_dir / "trend.yaml"))
    mean_reversion = MeanReversionConfig(**_load_yaml(config_dir / "mean_reversion.yaml"))
    carry = CarryConfig(**_load_yaml(config_dir / "carry.yaml"))

    return CarryBacktestConfig(risk=risk, fees=fees, trend=trend, mean_reversion=mean_reversion, carry=carry)


def load_markets_config(path: Path) -> MarketsConfig:
    m = _load_yaml(path)
    return MarketsConfig(markets=[MarketSpec(**mm) for mm in m["markets"]], cache_dir=m["cache_dir"])
