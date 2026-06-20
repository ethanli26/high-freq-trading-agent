# Swing Trading Agent — Project Plan

A personal trading agent that screens the market top-down, generates swing signals, gates every trade through an adjustable autonomy control, and executes on Interactive Brokers. Built paper-first.

## Decisions locked in

- Style: swing trades held days to weeks, positive-skew (chase the occasional big winner, keep losses small).
- Universe: top-down. Rank sectors by relative strength first, then pick the strongest names inside the leading sectors. No full-market scan.
- Autonomy: one setting with four modes. Start in Approve.
- Risk: built in from the start, not bolted on later.
- ML: added only after the rule-based system works and has been backtested honestly.
- Data budget: free tiers to start, scale up as the tool proves out.

## The autonomy modes

The execution layer reads one `autonomy_mode` setting. Same code path for all four.

1. `signal_only` — agent alerts you, places nothing.
2. `approve` — agent queues a trade and waits for your yes or no. (start here)
3. `semi_auto` — agent trades on its own, but only inside limits you set (max position size, min signal confidence, allowed symbols).
4. `full_auto` — agent places any approved signal immediately.

## Why the risk rules are non-negotiable

The big-winner approach only works under three conditions, all of them risk rules:

- Hard stop on every trade, so a loss is capped and stays small.
- Equal small risk per trade (default: risk 1% of account equity per position), so no single trade can hurt you.
- Enough trades for the edge to show. Expect a 35 to 45 percent win rate and real drawdowns. That is normal for this style.

Optimize the payoff ratio (avg win / avg loss), not the hit rate. You can chase one or the other, not both at once.

## Tech stack

Free or near-free to start. Every piece has a clear upgrade path.

- Language: Python 3.11+
- Broker / execution: `ib_async` (the maintained successor to ib_insync) against an IBKR paper account. TWS or IB Gateway running locally, API enabled, paper port 7497.
- Price and historical data: `yfinance` for daily bars (free, fine for backtesting and a swing horizon).
- Quotes, company news, sentiment: Finnhub free tier (60 calls/min, includes news and basic sentiment).
- Optional later: Alpha Vantage (50+ prebuilt technical indicators, has an official MCP server) and Tiingo (clean end-of-day history for backtests).
- Storage: SQLite to start (zero setup). Migrate to Supabase / Postgres later when you want a dashboard and remote access.
- Backtesting: pandas plus `vectorbt` for fast vectorized backtests. `backtrader` is an alternative if you prefer event-driven.
- Dashboard: Streamlit first (fast to build). React later if you want a richer UI.
- Scheduling: a once-daily run is enough for swing. Plain cron or a small Python scheduler.

## Sectors for the screener

Rank these 11 SPDR sector ETFs by momentum, take the top few, then pull constituents and rank names inside them:

`XLK XLF XLE XLV XLY XLP XLI XLB XLU XLRE XLC`

## Roadmap

### Phase 0 — Repo and paper connection
Goal: prove you can connect to IBKR paper and place one simulated order. Scaffold the repo, a config file, and a `connect()` that logs account summary. Place a tiny market order on paper and confirm it fills.

### Phase 1 — Data layer and sector screener
Goal: pull daily bars, rank the 11 sectors by relative strength (e.g. 3 and 6 month momentum), select the top sectors, then rank candidate stocks inside them. Output a ranked watchlist. No trading yet.

### Phase 2 — Signal, risk, decision, and the autonomy gate
Goal: turn the watchlist into trade proposals. One simple entry rule (a breakout or moving-average / momentum rule), an ATR-based stop, position sizing from the 1% risk rule, and a `compute_decision()` that returns propose / skip. Wire in the autonomy gate in `approve` mode so you confirm each trade.

### Phase 3 — Backtesting harness
Goal: measure the rule-based system honestly before trusting it. Walk-forward testing, strict train/test separation, and guards against look-ahead bias (never let a bar use data it could not have known in real time). Report win rate, payoff ratio, max drawdown, and expectancy.

### Phase 4 — ML signal layer
Goal: only after Phase 3 has a baseline. Frame it as "is this signal real or noise" — a classifier on top of the rule-based candidates, not a from-scratch price predictor. Strict train / validation / test split, walk-forward evaluation, and no leakage. Compare it head to head against the Phase 3 baseline. If it does not beat the simple rule out of sample, it does not ship.

### Phase 5 — Dashboard, monitoring, and graduating autonomy
Goal: Streamlit dashboard for the watchlist, open positions, and the trade log. Alerts. Once a strategy has a real paper track record, graduate it from `approve` to `semi_auto` within tight limits.

## Project layout

The system is a Python 3.11+ package organized by responsibility, with secrets loaded
from a `.env` file (never hardcoded) and a `.gitignore` that excludes `.env` and `logs`:

```
trading-agent/
  config.py          # settings: autonomy_mode, risk_per_trade, paper port, API keys from env
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
  requirements.txt
  .env.example
  README.md
```

Core dependencies: `ib_async`, `yfinance`, `finnhub-python`, `pandas`, `numpy`,
`vectorbt`, `python-dotenv`.

## Core components

### Broker connection (`execution/`)
An `ib_async` wrapper connects to IBKR paper trading on `127.0.0.1:7497` and exposes
`connect()`, `get_account_summary()`, `place_market_order(symbol, qty, action)`, and
`disconnect()`. A dropped connection retries with backoff. The connection is verified
with a read-only script that connects, prints the account summary, and exits without
placing any orders.

### Sector screener (`screener/`)
A cross-sectional sector momentum ranker. It pulls daily bars for the 11 SPDR sector
ETFs (XLK XLF XLE XLV XLY XLP XLI XLB XLU XLRE XLC) via yfinance, computes 3-month and
6-month total return for each, ranks them, and selects the top 3 sectors. For each
selected sector it ranks a list of large constituents by the same momentum measure and
outputs a ranked watchlist as a pandas DataFrame, persisted to the sqlite store. The
screener contains no trading logic.

## Guardrails to keep in mind

- Stay on paper until a strategy has a real out-of-sample track record.
- Look-ahead bias and overfitting are the two silent killers. The backtest harness in Phase 3 exists to catch them.
- Log every decision the agent makes, including the ones it skips. The log is how you debug and how the ML phase learns.
