"""Wire ReplayFeed -> EventEngine -> handlers, run on minute bars, and report.

Demonstrates the event-driven engine end to end on a deterministic synthetic minute
series (or a user-supplied CSV/parquet). Prints the full event chain and an
end-of-session summary. This is an ARCHITECTURE DEMO on replayed data — not a
validated or profitable strategy; intraday validation needs paid intraday data.

    python intraday/run_intraday.py
    python intraday/run_intraday.py --bars path/to/minute_bars.csv
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intraday.engine import (  # noqa: E402
    COMMISSION_PER_SHARE, INTRADAY_SLIPPAGE_PCT, EventEngine, IntradayRiskGate,
    Portfolio, SimulatedExecution,
)
from intraday.feed import ReplayFeed, make_synthetic_bars  # noqa: E402
from intraday.strategy import EmaCrossStrategy  # noqa: E402

# Demo defaults (minute-bar resolution).
EMA_FAST, EMA_SLOW = 5, 20
MAX_POSITIONS, STOP_PCT = 3, 0.005
STARTING_EQUITY = 100_000.0


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")


def _describe(event) -> str:
    """One-line rendering of an event for the chain log."""
    name, e = type(event).__name__, event
    ts = e.timestamp.strftime("%H:%M") if getattr(e, "timestamp", None) is not None else "  -  "
    if name == "BarEvent":
        return f"{ts}  BAR    {e.symbol}  close={e.close:.2f}"
    if name == "SignalEvent":
        return f"{ts}  SIGNAL {e.symbol}  {e.side}  ({e.reason})"
    if name == "OrderEvent":
        return f"{ts}  ORDER  {e.symbol}  {e.side} {e.quantity}"
    if name == "FillEvent":
        return f"{ts}  FILL   {e.symbol}  {e.side} {e.quantity} @ {e.fill_price:.4f}"
    return f"{ts}  {name}"


def build_feed(bars_path: str | None) -> ReplayFeed:
    """ReplayFeed from a user file, or a deterministic synthetic series."""
    if bars_path:
        path = Path(bars_path)
        return ReplayFeed.from_parquet(str(path)) if path.suffix == ".parquet" else ReplayFeed.from_csv(str(path))
    return ReplayFeed(make_synthetic_bars(periods=390))


def main(argv=None) -> int:
    """Run the engine and print the event chain + end-of-session summary."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Intraday event-engine demo (replayed minute bars).")
    parser.add_argument("--bars", help="CSV/parquet of minute bars (symbol,timestamp,OHLCV); default synthetic")
    args = parser.parse_args(argv)

    engine = EventEngine(
        feed=build_feed(args.bars),
        strategy=EmaCrossStrategy(),
        risk_gate=IntradayRiskGate(max_positions=MAX_POSITIONS, stop_pct=STOP_PCT),
        execution=SimulatedExecution(INTRADAY_SLIPPAGE_PCT, COMMISSION_PER_SHARE),
        portfolio=Portfolio(STARTING_EQUITY, COMMISSION_PER_SHARE),
        fast_period=EMA_FAST, slow_period=EMA_SLOW,
    )
    summary = engine.run()

    # Event chain (signals/orders/fills only — bars are too many to list).
    print("\n=== Event chain (signals, orders, fills) ===")
    shown = [ev for kind, ev in engine.event_log if kind != "BarEvent"]
    for event in shown:
        print("  " + _describe(event))
    if not shown:
        print("  (no signals fired on this series)")

    print("\n=== End-of-session summary ===")
    print(f"  bars seen/processed : {summary['bars_seen']} / {summary['bars_processed']}")
    print(f"  dropped (ooo/dup)   : {summary['dropped']}   gaps: {summary['gaps']}")
    print(f"  signals / orders / fills : {summary['signals']} / {summary['orders']} / {summary['fills']}")
    print(f"  realized P&L        : ${summary['realized_pnl']:,.2f}")
    print(f"  final equity        : ${summary['final_equity']:,.2f}  ({summary['return_pct'] * 100:+.2f}%)")
    print("\n  NOTE: architecture demo on replayed data — NOT a validated or profitable "
          "strategy. Intraday validation requires paid intraday data and microstructure "
          "modeling. Execution is simulated; no real orders are placed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
