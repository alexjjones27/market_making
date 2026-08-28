"""Parameter sweep harness. Reuses already-ingested market data across all
combinations (no re-hitting the API) and re-runs the bar-by-bar engine per
combination.

Ranks results by a risk-adjusted return proxy (PnL / max drawdown) rather
than raw PnL, and flags combinations where a small number of fills account
for most of the trading PnL -- a sign the "good" result may be an artifact
of the coarse fill model rather than a repeatable edge.
"""
from __future__ import annotations

import dataclasses
import itertools

import pandas as pd

from backtest.engine import BacktestEngine
from backtest.fill_model import TopOfBookFillModel
from backtest.metrics import compute_portfolio_metrics
from config.loader import Config


def _build_combo_config(base: Config, spread_bps: float, max_position_usd: float, skew_multiplier: float) -> Config:
    quoting = dataclasses.replace(
        base.strategy.quoting,
        spread_bps=spread_bps,
        max_skew_bps=base.strategy.quoting.max_skew_bps * skew_multiplier,
    )
    inventory = dataclasses.replace(base.strategy.inventory, max_position_usd=max_position_usd)
    strategy = dataclasses.replace(base.strategy, quoting=quoting, inventory=inventory)
    risk = dataclasses.replace(base.risk, max_inventory_usd=max_position_usd)
    return dataclasses.replace(base, strategy=strategy, risk=risk)


def run_sweep(base_config: Config, data: dict) -> pd.DataFrame:
    sweep_cfg = base_config.sweep
    combos = list(
        itertools.product(sweep_cfg.spread_bps, sweep_cfg.max_position_usd, sweep_cfg.skew_multiplier)
    )
    rows = []
    for spread_bps, max_position_usd, skew_multiplier in combos:
        combo_config = _build_combo_config(base_config, spread_bps, max_position_usd, skew_multiplier)
        engine = BacktestEngine(combo_config, TopOfBookFillModel(combo_config.risk.fill_fraction_of_bar_volume))
        result = engine.run(data)
        pm = compute_portfolio_metrics(result)

        n_fills = sum(mm.n_fills for mm in pm.market_metrics)
        max_fill_share = max((mm.largest_fill_pnl_share for mm in pm.market_metrics), default=0.0)
        max_dd_abs = abs(pm.max_drawdown_usd) if pm.max_drawdown_usd else 0.0
        risk_adj_return = pm.total_pnl / max_dd_abs if max_dd_abs > 1e-9 else (
            pm.total_pnl if pm.total_pnl <= 0 else float("inf")
        )

        eligible = n_fills >= sweep_cfg.min_fills_for_ranking
        overfit_flag = max_fill_share > sweep_cfg.max_single_fill_pnl_share

        rows.append(
            {
                "spread_bps": spread_bps,
                "max_position_usd": max_position_usd,
                "skew_multiplier": skew_multiplier,
                "total_pnl_usd": pm.total_pnl,
                "trading_pnl_usd": pm.total_trading_pnl,
                "fees_pnl_usd": pm.total_fees_pnl,
                "funding_pnl_usd": pm.total_funding_pnl,
                "sharpe_annualized": pm.sharpe_annualized,
                "max_drawdown_usd": pm.max_drawdown_usd,
                "risk_adjusted_return": risk_adj_return,
                "n_fills": n_fills,
                "eligible_min_fills": eligible,
                "largest_fill_pnl_share": max_fill_share,
                "overfit_risk_flag": overfit_flag,
                "daily_halts": pm.daily_halts,
            }
        )

    df = pd.DataFrame(rows)
    df["ranked"] = df["eligible_min_fills"] & ~df["overfit_risk_flag"]
    df = df.sort_values(["ranked", "risk_adjusted_return"], ascending=[False, False]).reset_index(drop=True)
    return df
