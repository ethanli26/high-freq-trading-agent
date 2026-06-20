# Swing Trading Agent

<!-- Replace OWNER/REPO with your GitHub path once pushed. -->
![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)
![coverage](https://img.shields.io/badge/coverage-pytest--cov-blue)

A personal swing trading agent that screens the market top-down (sectors first, then names), generates signals, gates every trade through an adjustable autonomy control, and executes on Interactive Brokers. Built paper-first.

See `PROJECT_PLAN.md` for the full stack, roadmap, and design decisions.

## Prerequisites

- Python 3.11 or higher
- An Interactive Brokers paper trading account
- TWS or IB Gateway installed and running locally, with the API enabled (Global Configuration > API > Settings)
- A free Finnhub API key from https://finnhub.io

## Setup

1. Clone the repo and move into it:
   ```
   git clone <your-repo-url>
   cd trading-agent
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate      # Mac/Linux
   venv\Scripts\activate         # Windows
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```
   cp .env.example .env
   ```
   Then open `.env` and fill in your real keys. Never commit `.env`.

## Running

`main.py` is the single entry point with four subcommands:

```
python main.py research   # full research pipeline -> consolidated report + artifacts
python main.py backtest   # canonical best-config backtest + benchmark comparison
python main.py factors    # factor IC evaluation scorecard
python main.py live       # paper decision loop (needs TWS/IB Gateway on the paper port)
```

Make sure TWS or IB Gateway is running and connected to your paper account before running `live` (or anything that touches the broker).

## Testing

The core math and the safety guards are covered by an offline, deterministic pytest
suite (no network). Install the dev deps and run it:

```
pip install -r requirements-dev.txt
pytest                              # full suite (the regression test needs a local price cache)
pytest -m "not slow and not network"   # the fast, offline suite that CI runs
```

What's covered:

- **risk/** — position sizing (1% risk, per-name and portfolio caps, min-size floor), ATR/stop math, conviction multiplier bounds.
- **signals/** — breakout (including the exclude-today window), the pullback rule, and the earnings-drift guard (never enters on/before the report date).
- **factors/** — NCSKEW/DUVOL formula fidelity and the look-ahead-safety property (factor value at *t* is unchanged when future bars are dropped).
- **backtest engine** — a synthetic scenario proving next-day-open entry fills and stop exits.
- **intraday engine** — event ordering, incremental rolling-state correctness, out-of-order/duplicate-bar handling, and deterministic signals/fills on a synthetic minute series.
- **safety guards** — the DU-account (paper-only) guard and `DRY_RUN` both block order placement and cannot be bypassed (broker mocked; no live connection).
- **regression** (marked `slow`/`network`, skipped in CI) — the canonical backtest reproduces 1675 trades / $1,967,830.

CI (`.github/workflows/ci.yml`) runs the offline suite with coverage on every push and PR and fails the build on any test failure.

## Project structure

```
trading-agent/
  config.py          # settings: autonomy_mode, risk_per_trade, ports, keys from env
  data/              # data fetching (yfinance, finnhub)
  screener/          # sector ranking + stock selection
  signals/           # entry rules + signal generation
  risk/              # position sizing + stop logic
  decision/          # compute_decision + autonomy gate
  execution/         # ib_async wrapper
  backtest/          # backtesting harness
  storage/           # sqlite layer
  logs/
  main.py
```

## Safety

- This runs against an IBKR paper account. Port 7497 is paper, 7496 is live. Stay on paper until a strategy has a real out-of-sample track record.
- Never commit API keys or your `.env` file.

## Status

In development. See the roadmap in `PROJECT_PLAN.md` for current phase.
