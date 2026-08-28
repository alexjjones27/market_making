"""Typed config loading. Reads the YAML files in this directory (or a
caller-supplied directory) into plain dataclasses so the rest of the
codebase never touches raw dicts or hardcoded parameters.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import yaml

CONFIG_DIR = Path(__file__).parent


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
class MakerRebateTier:
    tier: int
    min_maker_volume_share_pct: float
    extra_rebate: float


@dataclasses.dataclass
class FeesConfig:
    volume_tiers: list[FeeTier]
    maker_volume_rebate_tiers: list[MakerRebateTier]
    assumed_volume_tier: int
    assumed_maker_rebate_tier: int

    def effective_rates(self) -> tuple[float, float]:
        """Returns (taker_rate, maker_rate) after applying the assumed
        volume tier and maker-volume rebate tier."""
        vol_tier = next(t for t in self.volume_tiers if t.tier == self.assumed_volume_tier)
        rebate_tier = next(
            t for t in self.maker_volume_rebate_tiers if t.tier == self.assumed_maker_rebate_tier
        )
        return vol_tier.taker_rate, vol_tier.maker_rate + rebate_tier.extra_rebate


@dataclasses.dataclass
class FairValueConfig:
    method: str
    vwap_window_secs: int


@dataclasses.dataclass
class QuotingConfig:
    spread_bps: float
    max_skew_bps: float
    order_size_usd: float


@dataclasses.dataclass
class VolatilityConfig:
    window_secs: int
    widen_threshold_bps: float
    widen_multiplier: float


@dataclasses.dataclass
class InventoryConfig:
    max_position_usd: float


@dataclasses.dataclass
class StrategyConfig:
    fair_value: FairValueConfig
    quoting: QuotingConfig
    volatility: VolatilityConfig
    inventory: InventoryConfig


@dataclasses.dataclass
class RiskConfig:
    capital_usd: float
    fill_fraction_of_bar_volume: float
    max_inventory_usd: float
    daily_loss_limit_pct: float
    single_market_stop_loss_pct: float


@dataclasses.dataclass
class SweepConfig:
    spread_bps: list[float]
    max_position_usd: list[float]
    skew_multiplier: list[float]
    min_fills_for_ranking: int
    max_single_fill_pnl_share: float


@dataclasses.dataclass
class Config:
    markets: MarketsConfig
    fees: FeesConfig
    strategy: StrategyConfig
    risk: RiskConfig
    sweep: SweepConfig


def load_config(config_dir: Optional[Path] = None) -> Config:
    config_dir = config_dir or CONFIG_DIR

    m = _load_yaml(config_dir / "markets.yaml")
    markets = MarketsConfig(
        markets=[MarketSpec(**mm) for mm in m["markets"]],
        cache_dir=m["cache_dir"],
    )

    f = _load_yaml(config_dir / "fees.yaml")
    fees = FeesConfig(
        volume_tiers=[FeeTier(**t) for t in f["volume_tiers"]],
        maker_volume_rebate_tiers=[MakerRebateTier(**t) for t in f["maker_volume_rebate_tiers"]],
        assumed_volume_tier=f["assumed_volume_tier"],
        assumed_maker_rebate_tier=f["assumed_maker_rebate_tier"],
    )

    s = _load_yaml(config_dir / "strategy.yaml")
    strategy = StrategyConfig(
        fair_value=FairValueConfig(**s["fair_value"]),
        quoting=QuotingConfig(**s["quoting"]),
        volatility=VolatilityConfig(**s["volatility"]),
        inventory=InventoryConfig(**s["inventory"]),
    )

    r = _load_yaml(config_dir / "risk.yaml")
    risk = RiskConfig(**r)

    sw = _load_yaml(config_dir / "sweep.yaml")
    sweep = SweepConfig(**sw)

    return Config(markets=markets, fees=fees, strategy=strategy, risk=risk, sweep=sweep)
