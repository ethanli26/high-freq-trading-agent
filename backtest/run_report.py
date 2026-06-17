"""Run the best config four ways and report against passive baselines.

Builds four daily equity curves over the same window, same starting capital:
  (a) Strategy            — breakout + pullback, large-cap, regime + trend-exit,
                            honest costs (active only; idle cash earns nothing).
  (b) Strategy + overlay  — same, but idle cash is invested in SPY (index overlay).
  (c) S&P 500 buy & hold  — passive market.
  (d) SPY/cash blend       — constant mix scaled to the strategy's volatility (the
                            "risk-matched" passive: same risk level, trivially built).

Prints/saves the comparison table, equity + drawdown charts (all four lines), and an
honest summary. Read-only research. No IBKR, no orders.

Run from the repository root:

    python backtest/run_report.py
"""

import logging
import sys
from pathlib import Path

# Allow running this file directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import report  # noqa: E402
from backtest.benchmark import buy_and_hold, risk_matched_blend, vol_matched_weight  # noqa: E402
from backtest.data import BENCHMARK, build_sector_map, load_universe  # noqa: E402
from backtest.engine import STARTING_EQUITY, run_engine  # noqa: E402
from backtest.regime import compute_regime  # noqa: E402
from backtest.risk_metrics import DEFAULT_RISK_FREE  # noqa: E402
from config import OVERLAY_INSTRUMENT  # noqa: E402
from screener.sectors import SECTOR_ETFS  # noqa: E402
from signals.breakout import BreakoutStrategy  # noqa: E402
from signals.pullback import PullbackStrategy  # noqa: E402

log = logging.getLogger("run_report")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("risk.portfolio").setLevel(logging.WARNING)


def main() -> int:
    """Build the four curves and produce the report."""
    configure_logging()

    log.info("Loading large-cap universe...")
    bars, _ = load_universe()
    if BENCHMARK not in bars or OVERLAY_INSTRUMENT not in bars:
        log.error("Benchmark/overlay instrument missing; cannot build the report.")
        return 1

    regime = compute_regime(bars[BENCHMARK]["Close"])
    sector_map = build_sector_map()
    common = dict(strategies=[BreakoutStrategy(), PullbackStrategy()],
                  regime_filter=True, trend_exit=True, conviction_sizing=False)

    log.info("Running active-only strategy...")
    active_equity, _ = run_engine(bars, regime, sector_map, list(SECTOR_ETFS), **common)
    if active_equity.empty:
        log.error("No strategy equity produced; aborting.")
        return 1

    # Index overlay: idle cash earns SPY's daily return (aligned to the engine calendar).
    overlay_returns = bars[OVERLAY_INSTRUMENT]["Close"].reindex(regime.index).pct_change().to_numpy()
    log.info("Running strategy + index overlay...")
    overlay_equity, _ = run_engine(bars, regime, sector_map, list(SECTOR_ETFS),
                                   overlay=True, overlay_returns=overlay_returns, **common)

    # Passive baselines over the same window/capital.
    dates = active_equity.index
    spy_close = bars[BENCHMARK]["Close"]
    buyhold = buy_and_hold(spy_close, active_equity, STARTING_EQUITY)
    weight = vol_matched_weight(active_equity, buyhold)
    blend = risk_matched_blend(spy_close, dates, STARTING_EQUITY, weight, DEFAULT_RISK_FREE)
    log.info("Risk-matched blend: %.0f%% SPY / %.0f%% cash (vol-matched to the strategy).",
             weight * 100, (1 - weight) * 100)

    labels = {
        "Strategy": active_equity,
        "Strategy+overlay": overlay_equity.reindex(dates),
        "S&P 500 B&H": buyhold,
        f"SPY/cash blend ({weight * 100:.0f}% SPY)": blend,
    }
    blend_label = f"SPY/cash blend ({weight * 100:.0f}% SPY)"
    report.generate_report(labels, market_key="S&P 500 B&H", active="Strategy",
                           overlay="Strategy+overlay", buy_hold="S&P 500 B&H", blend=blend_label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
