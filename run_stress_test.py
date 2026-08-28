#!/usr/bin/env python3
"""Fill-model stress test: rerun the backtest across a grid of
`fill_fraction_of_bar_volume` values, with and without the adverse-selection
penalty overlay, and report how sensitive PnL/Sharpe/max-drawdown are to
those assumptions.

Purpose: the Phase 1 top-of-book fill model has two big, acknowledged
sources of optimism -- (1) it assumes a generous, constant fraction of bar
volume fills your resting order regardless of real queue position, and
(2) it can't see whether a fill was "toxic" (immediately followed by an
adverse price move). This script quantifies how much of the apparent edge
survives once both assumptions are tightened. If PnL/Sharpe collapse or
flip negative as fill_fraction shrinks and the adverse-selection penalty
is applied, that's the honest answer: no convincing edge yet, and Phase 2
(real L2 data, real queue position) is not yet worth building. If edge
survives even the tightest settings, that's a much stronger signal.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from backtest.adverse_selection import compute_adverse_selection
from backtest.engine import BacktestEngine
from backtest.fill_model import TopOfBookFillModel
from backtest.metrics import compute_portfolio_metrics
from config.loader import load_config
from data.ingest import ingest_all

DEFAULT_FILL_FRACTIONS = [0.07, 0.02, 0.01, 0.005]


def run_one(base_config, data, fill_fraction: float, adverse_enabled: bool):
    risk = dataclasses.replace(base_config.risk, fill_fraction_of_bar_volume=fill_fraction)
    config = dataclasses.replace(base_config, risk=risk)

    engine = BacktestEngine(config, TopOfBookFillModel(fill_fraction))
    result = engine.run(data)

    adverse = None
    if adverse_enabled:
        adverse = compute_adverse_selection(
            result.market_states,
            config.risk.adverse_selection.threshold_bps,
            config.risk.adverse_selection.lookahead_bars,
        )

    pm = compute_portfolio_metrics(result, adverse)
    n_fills = sum(mm.n_fills for mm in pm.market_metrics)
    n_toxic = sum(mm.n_toxic_fills for mm in pm.market_metrics)
    n_evaluable = sum(mm.n_evaluable_fills for mm in pm.market_metrics)

    return {
        "fill_fraction_of_bar_volume": fill_fraction,
        "adverse_selection": adverse_enabled,
        "total_pnl_usd": pm.total_pnl,
        "trading_pnl_usd": pm.total_trading_pnl,
        "fees_pnl_usd": pm.total_fees_pnl,
        "adverse_selection_pnl_usd": pm.total_adverse_selection_pnl,
        "sharpe_annualized": pm.sharpe_annualized,
        "max_drawdown_usd": pm.max_drawdown_usd,
        "max_drawdown_pct": pm.max_drawdown_pct,
        "n_fills": n_fills,
        "n_toxic_fills": n_toxic,
        "n_evaluable_fills": n_evaluable,
        "toxic_fill_rate": (n_toxic / n_evaluable) if n_evaluable else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Stress-test the Phase 1 fill model assumptions")
    parser.add_argument(
        "--config-dir", type=Path, default=Path("config_longrun"),
        help="Config dir to use (default: config_longrun, the 49-day/15m dataset -- "
             "needs enough sample length for the sensitivity read to mean anything)",
    )
    parser.add_argument(
        "--fill-fractions", type=float, nargs="+", default=DEFAULT_FILL_FRACTIONS,
        help=f"fill_fraction_of_bar_volume values to test (default: {DEFAULT_FILL_FRACTIONS})",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("results/stress_test.csv"))
    args = parser.parse_args()

    base_config = load_config(args.config_dir)
    print(f"Ingesting data for markets: {[m.coin for m in base_config.markets.markets]}")
    data = ingest_all(base_config)

    print(
        f"\nRunning {len(args.fill_fractions)} fill-fraction values x 2 adverse-selection "
        f"settings = {len(args.fill_fractions) * 2} combinations...\n"
        f"Adverse-selection overlay: >{base_config.risk.adverse_selection.threshold_bps}bps move "
        f"against the fill within {base_config.risk.adverse_selection.lookahead_bars} bars "
        "charges the full realized adverse move as an extra cost on that fill.\n"
    )

    rows = []
    for ff in args.fill_fractions:
        for adverse_enabled in (False, True):
            print(f"  fill_fraction={ff:.3%}  adverse_selection={adverse_enabled} ...")
            rows.append(run_one(base_config, data, ff, adverse_enabled))

    df = pd.DataFrame(rows)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    print("\n" + "=" * 100)
    print("STRESS TEST RESULTS")
    print("=" * 100)
    cols = [
        "fill_fraction_of_bar_volume", "adverse_selection", "total_pnl_usd", "sharpe_annualized",
        "max_drawdown_usd", "n_fills", "toxic_fill_rate",
    ]
    print(df[cols].to_string(index=False))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nFull results saved to {args.out_csv}")

    print("\n" + "=" * 100)
    print("SENSITIVITY READ")
    print("=" * 100)
    baseline = df[(df["fill_fraction_of_bar_volume"] == args.fill_fractions[0]) & (~df["adverse_selection"])]
    tightest = df[(df["fill_fraction_of_bar_volume"] == args.fill_fractions[-1]) & (df["adverse_selection"])]
    if not baseline.empty and not tightest.empty:
        base_pnl = baseline["total_pnl_usd"].iloc[0]
        tight_pnl = tightest["total_pnl_usd"].iloc[0]
        print(f"Baseline (fill_fraction={args.fill_fractions[0]:.1%}, no adverse-selection penalty): ${base_pnl:,.2f}")
        print(f"Tightest (fill_fraction={args.fill_fractions[-1]:.1%}, adverse-selection ON):         ${tight_pnl:,.2f}")
        if tight_pnl <= 0:
            print(
                "\n=> PnL flips to zero or negative under the tightest, most realistic assumptions.\n"
                "   Read this as: no convincing edge yet. The apparent profitability at looser\n"
                "   settings is very likely a fill-model artifact, not a repeatable strategy edge.\n"
                "   Do not invest in Phase 2 (L2 data collection) on the strength of this backtest."
            )
        elif tight_pnl < 0.25 * base_pnl:
            print(
                "\n=> PnL survives but collapses to a small fraction of the optimistic baseline.\n"
                "   Treat this as weak, unconfirmed evidence at best -- most of the apparent edge\n"
                "   was fill-model optimism, not strategy logic. Worth another look with even\n"
                "   tighter assumptions before considering Phase 2."
            )
        else:
            print(
                "\n=> PnL holds up reasonably well even under tighter fill/adverse-selection\n"
                "   assumptions. This is a meaningfully stronger signal than the original sweep --\n"
                "   worth considering Phase 2 (L2 collector + event-driven backtest), followed by\n"
                "   paper trading, before risking real capital."
            )


if __name__ == "__main__":
    main()
