# Hyperliquid Perp Market-Making Backtester -- Phase 1

A simplified, top-of-book backtester for a market-making strategy on
Hyperliquid perpetuals, built for a £1,000-£5,000 (~$1,250-$6,250) bankroll.
**This is a backtest-only codebase.** There is no order-signing, wallet, or
live-trading code anywhere in this repository, and none will be added
without a separate, explicit request.

## What's real vs. simulated

| Component | Status |
|---|---|
| OHLCV candles | **Real**, pulled from Hyperliquid's public Info API (`candleSnapshot`) |
| Funding rate history | **Real**, pulled from Hyperliquid's public Info API (`fundingHistory`) |
| Fee schedule | **Real rates**, hand-maintained in `config/fees.yaml` from published docs (no API serves a fee-schedule table -- see below) |
| Historical trade tape | **Not available.** Hyperliquid's Info API has no market-wide historical trade endpoint (only a user's own past fills, which requires that user to have actually traded). The fill model uses candle high/low/volume instead. |
| Order fills | **Simulated**, top-of-book approximation (see Fill Model Limitations below) |
| Order book depth / queue position | **Not modeled at all in Phase 1.** This is the single biggest gap between this backtest and reality. |

## Data availability constraint you should know about

Hyperliquid's `candleSnapshot` endpoint only retains **the most recent 5000
candles per interval** -- this is a hard historical retention limit, not
just a per-request page size. That means:

- `1m` candles: only ~3.5 days of history available, ever.
- `5m` candles: ~17 days.
- `15m` candles: ~52 days.
- `1h` candles: ~208 days.
- `1d` candles: ~13.7 years.

Confirmed empirically: a `1m` request for a date 3 months ago returns `[]`,
while a `1h` request for the same date returns real data.

**Practical effect:** if you want a multi-week backtest, you need to use a
coarser candle interval, which coarsens the fill model's intrabar
resolution too (fills are decided per-bar from that bar's low/high/volume).
This repo ships both:
- `config/markets.yaml` set to `1m` -- gives the finest-grained fill
  simulation and best-matched vol/VWAP windows, but only ~3.5 days of data.
- A documented pattern (see "Running a longer backtest" below) for
  switching to `15m` or `1h` to get weeks/months of history at the cost of
  coarser intrabar fill resolution.

There is no way around this with the Info API as it exists today. Phase 2's
WebSocket collector (scaffolded, not built) is the actual fix: once it has
been logging L2 book + trade data for a while, you get real historical
granularity that doesn't depend on Hyperliquid's candle retention window.

## Fill model limitations (read this before trusting any PnL number)

`backtest/fill_model.py`'s `TopOfBookFillModel` is intentionally crude:

1. **A fill is assumed whenever price trades through your quote** (bar low
   <= bid, or bar high >= ask) -- with no check on whether your order would
   actually have been reached given the resting depth ahead of it. On a
   real order book, being "at the money" for a moment does not mean you
   filled; you queue behind whatever size was already resting there.
2. **Fill size is capped at a configurable fraction of that bar's total
   volume** (`risk.fill_fraction_of_bar_volume`, default 7%) as a blunt
   proxy for "you don't get 100% of every print." This is not a queue
   model -- it's a single global knob, not sensitive to your actual
   quoted size relative to real depth.
3. **No adverse-selection visibility within a bar.** The model can't see
   whether a fill happened right before an unfavorable price move within
   that same bar -- only the bar's OHLC is known.
4. **Fees and funding are otherwise realistic**: maker/taker rates come
   from `config/fees.yaml`, applied per simulated fill; funding is applied
   at each real funding-rate observation based on the position held at
   that time.

**Every backtest run prints an explicit caveat block with a rough
overstatement estimate (~20-50% of what the backtest shows is a reasonable
planning assumption for real fills) -- this is not just a code comment,
it's part of the tool's output**, per the design brief this was built to.

## Package layout

```
config/       YAML configs (markets, fees, strategy, risk, sweep) + typed loader
data/         Info API client, ingestion + parquet caching, Phase 2 WS collector scaffold
strategy/     Pluggable fair-value models, quote generation (skew + vol widening)
backtest/     Fill model (ABC + Phase 1 impl + Phase 2 stub), engine, risk controls, metrics, sweep
reporting/    Plain-language summary printer, plots (equity/inventory/fills)
run_backtest.py   CLI: ingest -> backtest -> summary + plots
run_sweep.py      CLI: ingest -> grid sweep -> ranked results table
```

`backtest/fill_model.py`'s `FillModel` is an abstract base class specifically
so Phase 2's queue-aware model can be swapped in later without touching
`strategy/` or `engine.py`'s control flow -- see `QueueAwareFillModel` in
that file for the documented (unimplemented) extension point.

## Setup

```bash
pip install -r requirements.txt
```

## Running a backtest

```bash
python run_backtest.py                    # uses config/*.yaml, saves plots to results/
python run_backtest.py --out-dir myrun     # custom output dir
python run_backtest.py --no-plots          # skip plot generation (faster)
```

First run hits the Hyperliquid Info API and caches raw pulls to
`data_cache/` as parquet; subsequent runs with the same market/date-range
config reuse the cache.

### Running a longer backtest (trading off resolution for history)

Edit `config/markets.yaml`: change `candle_interval` to `15m` (or `1h`) and
widen `start_date`/`end_date` to fit within that interval's retention
window (see table above). You should also widen
`strategy.yaml`'s `fair_value.vwap_window_secs` and
`volatility.window_secs` to be a multiple of the new bar size --
at `1m` bars a 45s VWAP window is meaningful; at `15m` bars it collapses
to just the current bar (still works, just degenerates to close-to-mid
behavior).

## Running a parameter sweep

```bash
python run_sweep.py
```

Sweeps `spread_bps x max_position_usd x skew_multiplier` per
`config/sweep.yaml`, re-running the engine over the same cached data for
each combination. Output is ranked by **risk-adjusted return** (PnL / max
drawdown), not raw PnL, and two flags are shown per row:

- `eligible_min_fills` -- False if the run had fewer than
  `sweep.min_fills_for_ranking` fills; too few fills to trust any metric
  computed from them.
- `overfit_risk_flag` -- True if a single fill accounted for more than
  `sweep.max_single_fill_pnl_share` of that market's gross fill notional --
  a sign the result may be a large-fill artifact, not a repeatable edge.

Both flagged rows are pushed to the bottom of the ranking rather than
excluded outright, so you can still see them.

## Config

Everything that should be tunable is in `config/*.yaml`, not hardcoded:

- `markets.yaml` -- which coins, date range, candle interval
- `fees.yaml` -- maker/taker rates by volume tier + maker-volume rebate
  tiers. **Hand-maintained**, since Hyperliquid's Info API has no endpoint
  serving a fee schedule table (only `userFees`, which returns an
  *authenticated user's current* tier based on their own trailing volume --
  not usable for a generic backtest config). Update this file by hand when
  Hyperliquid changes fees; it records an `as_of_date` for that reason.
- `strategy.yaml` -- fair value method, spread/skew/sizing, vol-widening
  thresholds, max position
- `risk.yaml` -- starting capital, fill-fraction assumption, max inventory,
  daily loss limit, single-market stop-loss
- `sweep.yaml` -- the parameter grid + overfitting-flag thresholds

## Phase 2 (scaffolded, not built out)

`data/collector_ws.py` documents the intended design for a WebSocket-based
L2 book + trade collector (subscribe to `l2Book` and `trades`, throttle
writes to every 100-500ms, partition output by coin/day) and
`backtest/fill_model.py`'s `QueueAwareFillModel` is the documented
extension point for an event-driven, queue-position-aware fill model once
that data exists. Neither is implemented yet -- per the original brief,
Phase 2 shouldn't be built out until Phase 1 backtest results show a
strategy worth testing more rigorously, since collecting weeks of L2 data
for a strategy that doesn't survive Phase 1 is wasted effort.

## What this codebase will never contain (Phase 1 scope)

No wallet or private-key handling, no order signing, no `Exchange` class
usage from the SDK, no live trading of any kind. `data/hyperliquid_client.py`
only wraps the SDK's read-only `Info` class.
