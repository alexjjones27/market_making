"""Plain-language output. Every backtest run prints this -- the caveats
are not optional footnotes, they're part of the report.
"""
from __future__ import annotations

from backtest.metrics import PortfolioMetrics
from config.loader import Config

BANNER = "=" * 78


def print_summary(config: Config, pm: PortfolioMetrics) -> None:
    print(BANNER)
    print("PHASE 1 SIMPLIFIED BACKTEST -- TOP-OF-BOOK APPROXIMATION")
    print(BANNER)
    print(
        "This is NOT an order-book-level backtest. It assumes a fill whenever\n"
        "price trades through your quote, capped at "
        f"{config.risk.fill_fraction_of_bar_volume:.0%} of that bar's volume, with\n"
        "no model of queue position. On a real order book you queue behind\n"
        "whatever resting size was already at your price, and you very often\n"
        "do NOT fill even when price trades through your level, if the print\n"
        "that crossed it was smaller than the queue ahead of you.\n"
    )
    print(
        "Rough rule of thumb: expect real fills at somewhere around 20-50% of\n"
        "what this backtest shows, and expect adverse selection (fills that\n"
        "happen right before price moves against you) to be worse in reality\n"
        "than this bar-level model can see, since it has no visibility into\n"
        "what happened *within* a bar before or after your fill. Treat every\n"
        "number below as an optimistic upper bound, not a forecast.\n"
    )

    print(BANNER)
    print("CAPITAL & RUN CONFIG")
    print(BANNER)
    print(f"Starting capital (assumed USD): ${config.risk.capital_usd:,.2f}")
    print(f"Markets: {', '.join(m.coin for m in config.markets.markets)}")
    print(f"Fair value method: {config.strategy.fair_value.method}")
    print(f"Spread: {config.strategy.quoting.spread_bps} bps | Max skew: {config.strategy.quoting.max_skew_bps} bps")
    print(f"Max position per market: ${min(config.risk.max_inventory_usd, config.strategy.inventory.max_position_usd):,.2f}")
    print(f"Fee tier assumed: tier {config.fees.assumed_volume_tier} (taker/maker rates below)")
    taker_rate, maker_rate = config.fees.effective_rates()
    print(f"  taker={taker_rate*10000:.3f} bps, maker={maker_rate*10000:.3f} bps"
          f"{' (net rebate)' if maker_rate < 0 else ''}")

    print()
    print(BANNER)
    print("PNL SUMMARY")
    print(BANNER)
    print(f"Total PnL:        ${pm.total_pnl:,.2f}  ({pm.total_pnl / config.risk.capital_usd:+.2%} of capital)")
    print(f"  Trading/spread: ${pm.total_trading_pnl:,.2f}")
    print(f"  Fees:           ${pm.total_fees_pnl:,.2f}")
    print(f"  Funding:        ${pm.total_funding_pnl:,.2f}")
    print(f"Max drawdown:     ${pm.max_drawdown_usd:,.2f} ({pm.max_drawdown_pct:.2%})")
    if pm.sharpe_annualized is not None:
        print(f"Sharpe (annualized, daily returns): {pm.sharpe_annualized:.2f}  -- {pm.sharpe_note}")
    else:
        print(f"Sharpe: not computed -- {pm.sharpe_note}")
    print(f"Daily loss limit halts triggered: {pm.daily_halts}")

    print()
    print(BANNER)
    print("PER-MARKET DETAIL")
    print(BANNER)
    for mm in pm.market_metrics:
        print(f"\n{mm.coin}:")
        print(f"  Fills: {mm.n_fills} total ({mm.n_buy_fills} buy, {mm.n_sell_fills} sell)")
        print(f"  Fill rate: buy={mm.fill_rate_buy:.4f}/bar, sell={mm.fill_rate_sell:.4f}/bar (of {mm.n_quote_bars} quoted bars)")
        print(f"  PnL: total=${mm.total_pnl:,.2f}  trading=${mm.trading_pnl:,.2f}  fees=${mm.fees_pnl:,.2f}  funding=${mm.funding_pnl:,.2f}")
        print(f"  Position: max |pos|=${mm.max_abs_position_usd:,.2f}  mean |pos|=${mm.mean_abs_position_usd:,.2f}")
        if mm.was_flattened:
            print("  ** AUTO-FLATTENED: single-market stop loss was breached during this run **")
        if mm.n_fills > 0 and mm.largest_fill_pnl_share > 0.2:
            print(
                f"  ** CAUTION: largest single fill was {mm.largest_fill_pnl_share:.0%} of this market's "
                "gross fill notional -- PnL may be dominated by a handful of large fills, not a repeatable edge. **"
            )

    print()
    print(BANNER)
    print("HONEST READ")
    print(BANNER)
    if pm.n_trading_days < 20:
        print(
            f"Only {pm.n_trading_days:.0f} days of data were backtested. That is too short to draw\n"
            "conclusions about a market-making strategy's edge -- crypto realized vol and\n"
            "funding regimes shift over weeks, and a strategy can look profitable for 10-20\n"
            "days purely by chance. Treat this as a smoke test of the code, not a signal."
        )
    total_fills = sum(mm.n_fills for mm in pm.market_metrics)
    if total_fills < 100:
        print(
            f"Only {total_fills} simulated fills across all markets. Metrics computed from this\n"
            "many fills (Sharpe, fill rate, PnL split) carry very wide error bars."
        )
    print(BANNER)
