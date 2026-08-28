#!/usr/bin/env python3
"""CLI entrypoint: ingest data once, then sweep spread_bps x max_position_usd
x skew_multiplier per config/sweep.yaml, and print a ranked results table.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest.sweep import run_sweep
from config.loader import load_config
from data.ingest import ingest_all


def main():
    parser = argparse.ArgumentParser(description="Run the Phase 1 parameter sweep")
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=Path("results/sweep_results.csv"))
    args = parser.parse_args()

    config = load_config(args.config_dir)
    print(f"Ingesting data for markets: {[m.coin for m in config.markets.markets]}")
    data = ingest_all(config)

    print(
        f"\nSweeping {len(config.sweep.spread_bps)} x {len(config.sweep.max_position_usd)} x "
        f"{len(config.sweep.skew_multiplier)} = "
        f"{len(config.sweep.spread_bps) * len(config.sweep.max_position_usd) * len(config.sweep.skew_multiplier)} "
        "combinations...\n"
    )
    results_df = run_sweep(config, data)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

    print("=" * 100)
    print("SWEEP RESULTS -- ranked by risk-adjusted return (PnL / max drawdown)")
    print("Rows with overfit_risk_flag=True or eligible_min_fills=False are pushed to the")
    print("bottom: they either have too few fills to trust, or a single fill dominates PnL.")
    print("=" * 100)
    cols = [
        "spread_bps", "max_position_usd", "skew_multiplier", "total_pnl_usd",
        "risk_adjusted_return", "sharpe_annualized", "max_drawdown_usd", "n_fills",
        "largest_fill_pnl_share", "eligible_min_fills", "overfit_risk_flag",
    ]
    print(results_df[cols].to_string(index=False))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.out_csv, index=False)
    print(f"\nFull results saved to {args.out_csv}")


if __name__ == "__main__":
    main()
