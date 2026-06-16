"""Entry point: turn the latest watchlist into trade decisions.

Flow:
  1. Connect to the broker, read net liquidation as equity, run the DU paper guard.
  2. Load the most recent watchlist run from SQLite.
  3. For each symbol, fetch recent daily bars and compute a decision.
  4. Print a readable table of proposals and skips (with reasons).
  5. If DRY_RUN is True, stop after printing — place nothing.
     Otherwise hand the proposals to the autonomy gate (approve mode).

Run from the repository root:

    python decision/run_decision.py
"""

import logging
import sys
from pathlib import Path

# Allow running this file directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from data.prices import fetch_daily_bars  # noqa: E402
from decision.autonomy import assert_paper_account, run_gate  # noqa: E402
from decision.decision import compute_decision  # noqa: E402
from execution.broker import IBBroker  # noqa: E402
from risk.portfolio import apply_portfolio_limits, seed_exposure  # noqa: E402
from screener.stocks import SECTOR_CONSTITUENTS  # noqa: E402
from storage.database import load_latest_watchlist  # noqa: E402

log = logging.getLogger("run_decision")


def configure_logging() -> None:
    """Timestamped logs so every decision is time-stamped; quiet noisy libs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("ib_async").setLevel(logging.WARNING)


def log_decision(decision: dict) -> None:
    """Log a single decision (timestamp comes from the logging format)."""
    if decision["status"] == "propose":
        log.info(
            "DECISION propose %s: BUY %d @ %.2f, stop %.2f, risk $%.2f.",
            decision["symbol"],
            decision["shares"],
            decision["entry_ref"],
            decision["stop"],
            decision["risk_dollars"],
        )
    else:
        log.info("DECISION skip %s: %s.", decision["symbol"], decision["reason"])


def evaluate_watchlist(symbols: list[str], equity: float) -> list[dict]:
    """Fetch bars and compute a decision for every symbol."""
    decisions = []
    for symbol in symbols:
        bars = fetch_daily_bars(symbol)
        if bars is None:
            decision = {"status": "skip", "symbol": symbol, "reason": "no price data"}
        else:
            decision = compute_decision(symbol, bars, equity)
        decisions.append(decision)
        log_decision(decision)
    return decisions


def print_results(proposals: list[dict], skips: list[dict]) -> None:
    """Print proposals and skips as two readable tables."""
    print("\n=== Proposals ===")
    if proposals:
        columns = ["symbol", "action", "entry_ref", "stop", "atr", "shares", "risk_dollars", "est_value"]
        print(pd.DataFrame(proposals)[columns].to_string(index=False))
    else:
        print("(none)")

    print("\n=== Skips ===")
    if skips:
        print(pd.DataFrame(skips)[["symbol", "reason"]].to_string(index=False))
    else:
        print("(none)")
    print()


def build_sector_lookup(watchlist: pd.DataFrame) -> dict[str, str]:
    """Map symbol -> sector for portfolio limit checks.

    Seeds from the full constituent map (so currently held names are recognized
    too), then lets this run's watchlist sector column take precedence.
    """
    lookup: dict[str, str] = {}
    for sector, members in SECTOR_CONSTITUENTS.items():
        for symbol in members:
            lookup[symbol] = sector
    for _, row in watchlist.iterrows():
        lookup[row["symbol"]] = row["sector"]
    return lookup


def print_portfolio_results(
    proposals: list[dict],
    portfolio_skips: list[dict],
    equity: float,
    current_positions: list[dict],
    sector_of: dict[str, str],
) -> None:
    """Print final proposals with running sector/total exposure, plus skips."""
    print("\n=== Final proposals (after portfolio limits) ===")
    if proposals:
        # Replay the accepted sizes to show exposure building up after each name.
        total, by_sector = seed_exposure(current_positions, sector_of)
        rows = []
        for proposal in proposals:
            symbol = proposal["symbol"]
            sector = sector_of.get(symbol)
            total += proposal["est_value"]
            if sector is not None:
                by_sector[sector] = by_sector.get(sector, 0.0) + proposal["est_value"]
            rows.append(
                {
                    "symbol": symbol,
                    "sector": sector,
                    "shares": proposal["shares"],
                    "est_value": proposal["est_value"],
                    "risk_dollars": proposal["risk_dollars"],
                    "sector_exp_pct": round(100 * by_sector.get(sector, 0.0) / equity, 1) if sector else None,
                    "total_exp_pct": round(100 * total / equity, 1),
                }
            )
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("(none)")

    print("\n=== Portfolio skips ===")
    if portfolio_skips:
        print(pd.DataFrame(portfolio_skips)[["symbol", "reason"]].to_string(index=False))
    else:
        print("(none)")
    print()


def main() -> int:
    """Run the decision pipeline end to end."""
    configure_logging()

    broker = IBBroker()
    try:
        broker.connect()

        summary = broker.get_account_summary()
        equity = summary.get("net_liquidation")
        if equity is None:
            log.error("Could not read net liquidation; aborting.")
            return 1
        log.info("Account equity (net liquidation): $%s", f"{equity:,.2f}")

        # Hard safety guard: must be a paper (DU) account.
        assert_paper_account(broker)

        watchlist = load_latest_watchlist()
        if watchlist.empty:
            log.error("No watchlist found in storage; run the screener first.")
            return 1

        symbols = watchlist["symbol"].tolist()
        run_stamp = watchlist["run_timestamp"].iloc[0]
        log.info("Evaluating %d symbols from watchlist run %s.", len(symbols), run_stamp)

        decisions = evaluate_watchlist(symbols, equity)
        proposals = [d for d in decisions if d["status"] == "propose"]
        skips = [d for d in decisions if d["status"] == "skip"]

        print_results(proposals, skips)
        log.info("Summary: %d raw proposal(s), %d skip(s).", len(proposals), len(skips))

        # Portfolio-level risk limits on top of per-trade sizing.
        current_positions = broker.get_positions()
        log.info("Current open positions: %d.", len(current_positions))
        sector_of = build_sector_lookup(watchlist)
        final_proposals, portfolio_skips = apply_portfolio_limits(
            proposals, equity, current_positions, sector_of
        )
        print_portfolio_results(
            final_proposals, portfolio_skips, equity, current_positions, sector_of
        )
        log.info(
            "After portfolio limits: %d final proposal(s), %d portfolio skip(s).",
            len(final_proposals),
            len(portfolio_skips),
        )

        if config.DRY_RUN:
            log.info("DRY_RUN is True: printing only, placing nothing.")
        else:
            run_gate(final_proposals, broker)

    except RuntimeError as error:
        # Raised by the safety guard on a non-paper account.
        log.error("Aborting: %s", error)
        return 1
    finally:
        broker.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
