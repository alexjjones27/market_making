#!/usr/bin/env python3
"""CLI entrypoint: ingest data (or load from cache), run the Phase 1
top-of-book backtest, print the honest-caveats summary, and save plots.

Backtest-only. No live trading, no order signing, no wallet/private-key
handling anywhere in this codebase.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from backtest.engine import BacktestEngine
from backtest.fill_model import TopOfBookFillModel
from backtest.metrics import compute_portfolio_metrics
from config.loader import load_config
from data.ingest import ingest_all
from reporting.plots import generate_all_plots
from reporting.summary import print_summary


def main():
    parser = argparse.ArgumentParser(description="Run the Phase 1 Hyperliquid MM backtest")
    parser.add_argument("--config-dir", type=Path, default=None, help="Override config directory")
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="Where to save plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    config = load_config(args.config_dir)

    print(f"Ingesting data for markets: {[m.coin for m in config.markets.markets]}")
    data = ingest_all(config)

    fill_model = TopOfBookFillModel(config.risk.fill_fraction_of_bar_volume)
    engine = BacktestEngine(config, fill_model)
    result = engine.run(data)

    pm = compute_portfolio_metrics(result)
    print_summary(config, pm)

    if not args.no_plots:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        generate_all_plots(result, args.out_dir)
        print(f"\nPlots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
