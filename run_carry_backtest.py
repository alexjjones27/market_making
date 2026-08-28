#!/usr/bin/env python3
"""Carry-overlay backtests: Strategy 1 (carry-aligned trend, hard gate),
Strategy 2 (carry + mean-reversion, soft bias), each compared against its
base strategy and a funding-only carry baseline, across two historical
regime windows.

Reuses the existing data ingestion layer (data/ingest.py, data/cache.py,
data/hyperliquid_client.py) as-is -- no changes to that code. No existing
trend or mean-reversion backtester was found in this repo, so both base
strategies are the simple fallbacks specified in the brief
(strategy/trend.py, strategy/mean_reversion.py); the carry gating/bias
logic (strategy/carry.py, strategy/carry_wrappers.py) is layered on top of
those without modifying their signal logic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.directional_engine import run_directional_backtest
from backtest.directional_metrics import combine_portfolio, compute_metrics
from config.carry_loader import load_carry_config, load_markets_config
from data.hyperliquid_client import HyperliquidClient
from data.ingest import ingest_market
from strategy.carry_wrappers import CarryBiasedSignal, CarryGatedSignal, CarryOnlySignal
from strategy.mean_reversion import MeanReversionSignal
from strategy.trend import TrendSignal

BANNER = "=" * 100


def compute_regime_stats(coin_data: dict) -> dict:
    """Realized regime description for a window, computed AFTER the fact
    from the actual data -- windows were not hand-picked to represent a
    particular regime."""
    btc_candles, btc_funding = coin_data["BTC"]
    closes = btc_candles["close"].to_numpy()
    price_return_pct = (closes[-1] / closes[0] - 1.0) * 100.0
    log_ret = np.diff(np.log(closes))
    realized_vol_ann_pct = float(np.std(log_ret) * np.sqrt(24 * 365) * 100.0) if len(log_ret) > 1 else 0.0

    rates = btc_funding["fundingRate"].to_numpy()
    signs = np.sign(rates)
    persistence = float(np.mean(signs[1:] == signs[:-1])) if len(signs) > 1 else float("nan")
    pct_paying_shorts = float(np.mean(rates > 0) * 100.0) if len(rates) else float("nan")

    return {
        "btc_price_return_pct": price_return_pct,
        "btc_realized_vol_annualized_pct": realized_vol_ann_pct,
        "btc_funding_sign_persistence": persistence,
        "btc_pct_hours_funding_pays_shorts": pct_paying_shorts,
    }


def print_regime(window_name: str, stats: dict) -> None:
    print(f"\n{window_name} realized regime (BTC, computed after the fact -- not hand-picked):")
    print(f"  Price return over window: {stats['btc_price_return_pct']:+.1f}%")
    print(f"  Realized vol (annualized): {stats['btc_realized_vol_annualized_pct']:.1f}%")
    print(f"  Funding sign persistence hour-to-hour: {stats['btc_funding_sign_persistence']:.1%}"
          " (higher = funding skew persists rather than flipping constantly)")
    print(f"  Hours funding paid shorts (rate > 0): {stats['btc_pct_hours_funding_pays_shorts']:.0f}%")


def run_variant(coin_data: dict, model_factory, order_notional_usd: float, taker_rate: float) -> tuple:
    results = []
    for coin, (candles, funding) in coin_data.items():
        model = model_factory()
        result = run_directional_backtest(coin, candles, funding, model, order_notional_usd, taker_rate)
        results.append(result)
    return results


def print_comparison_table(strategy_name: str, rows: list) -> None:
    print(f"\n{strategy_name}")
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(df.to_string(index=False))


def metrics_row(variant_name: str, pm) -> dict:
    return {
        "variant": variant_name,
        "total_return_pct": pm.total_return_pct,
        "sharpe": pm.sharpe_annualized if pm.sharpe_annualized is not None else float("nan"),
        "max_dd_usd": pm.max_drawdown_usd,
        "n_trades": pm.n_trades,
        "win_rate": pm.win_rate if pm.win_rate is not None else float("nan"),
        "directional_pnl": pm.directional_pnl,
        "funding_pnl": pm.funding_pnl,
        "fees_pnl": pm.fees_pnl,
    }


def run_window(window_name: str, markets_path: Path, cfg, client: HyperliquidClient, cache_dir: Path) -> None:
    print(f"\n{BANNER}\n{window_name}\n{BANNER}")
    markets_cfg = load_markets_config(markets_path)
    coin_data = {}
    for spec in markets_cfg.markets:
        coin_data[spec.coin] = ingest_market(client, cache_dir, spec)

    regime = compute_regime_stats(coin_data)
    print_regime(window_name, regime)

    taker_rate = cfg.fees.taker_rate()
    order_notional = cfg.risk.order_notional_usd
    capital = cfg.risk.capital_usd

    # ---- Strategy 1: carry-aligned trend ----
    base_trend = lambda: TrendSignal(cfg.trend.ma_window_bars)
    gated_trend = lambda: CarryGatedSignal(TrendSignal(cfg.trend.ma_window_bars))
    carry_only = lambda: CarryOnlySignal()

    trend_rows = []
    for label, factory in [
        ("1. Trend alone (base)", base_trend),
        ("2. Carry-gated trend", gated_trend),
        ("3. Carry-only (funding, no signal)", carry_only),
    ]:
        results = run_variant(coin_data, factory, order_notional, taker_rate)
        pm = combine_portfolio(results, capital)
        trend_rows.append(metrics_row(label, pm))
        if pm.n_trades < 30:
            print(f"  ** {label}: only {pm.n_trades} trades -- too few to trust Sharpe/win-rate as a real signal. **")

    print_comparison_table(f"STRATEGY 1 -- Carry-aligned trend ({window_name})", trend_rows)

    # ---- Strategy 2: carry + mean-reversion ----
    base_mr = lambda: MeanReversionSignal(
        cfg.mean_reversion.lookback_bars, cfg.mean_reversion.entry_z,
        cfg.mean_reversion.exit_z, cfg.mean_reversion.stop_loss_pct,
    )
    biased_mr = lambda: CarryBiasedSignal(
        MeanReversionSignal(
            cfg.mean_reversion.lookback_bars, cfg.mean_reversion.entry_z,
            cfg.mean_reversion.exit_z, cfg.mean_reversion.stop_loss_pct,
        ),
        cfg.carry.strong_threshold_annualized_pct, cfg.carry.aligned_multiplier,
        cfg.carry.opposed_multiplier, cfg.carry.neutral_multiplier,
    )

    mr_rows = []
    for label, factory in [
        ("1. Mean-reversion alone (base)", base_mr),
        ("2. Carry-biased mean-reversion", biased_mr),
        ("3. Carry-only (funding, no signal)", carry_only),
    ]:
        results = run_variant(coin_data, factory, order_notional, taker_rate)
        pm = combine_portfolio(results, capital)
        mr_rows.append(metrics_row(label, pm))
        if pm.n_trades < 30:
            print(f"  ** {label}: only {pm.n_trades} trades -- too few to trust Sharpe/win-rate as a real signal. **")

    print_comparison_table(f"STRATEGY 2 -- Carry-biased mean-reversion ({window_name})", mr_rows)


def main():
    parser = argparse.ArgumentParser(description="Carry-overlay backtests (trend + mean-reversion)")
    parser.add_argument("--config-dir", type=Path, default=Path("config_carry"))
    args = parser.parse_args()

    cfg = load_carry_config(args.config_dir)
    client = HyperliquidClient()
    cache_dir = Path("data_cache")

    print(BANNER)
    print("CARRY OVERLAY BACKTEST -- built on the fallback trend/mean-reversion baselines")
    print("(no existing directional strategy code was found in this repo)")
    print(BANNER)
    print(
        "Assumptions, stated up front (not tuned to these results):\n"
        f"  Trend baseline: {cfg.trend.ma_window_bars}-bar MA momentum (long/short/flat, recomputed each bar)\n"
        f"  Mean-reversion baseline: {cfg.mean_reversion.lookback_bars}-bar z-score, "
        f"entry |z|>={cfg.mean_reversion.entry_z}, exit |z|<={cfg.mean_reversion.exit_z}, "
        f"stop {cfg.mean_reversion.stop_loss_pct:.0%}\n"
        f"  Carry 'strong' threshold: {cfg.carry.strong_threshold_annualized_pct:.0f}% annualized funding "
        "(a prior, not fit to these results)\n"
        f"  Strategy 2 size multipliers: aligned={cfg.carry.aligned_multiplier}x, "
        f"opposed={cfg.carry.opposed_multiplier}x, neutral={cfg.carry.neutral_multiplier}x\n"
        f"  Position size: ${cfg.risk.order_notional_usd} notional per entry, taker fee "
        f"{cfg.fees.taker_rate()*10000:.2f}bps each way, capital base ${cfg.risk.capital_usd:,.0f}\n"
        "  Funding used at each decision is the last SETTLED hourly rate as of that bar -- never a "
        "still-accruing rate (lookahead guard)."
    )

    run_window("WINDOW A (earlier)", args.config_dir / "markets_window_a.yaml", cfg, client, cache_dir)
    run_window("WINDOW B (later)", args.config_dir / "markets_window_b.yaml", cfg, client, cache_dir)


if __name__ == "__main__":
    main()
