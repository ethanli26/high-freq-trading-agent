"""Cap-spectrum comparison: large vs mid vs small, with HONEST frictions.

Runs breakout + pullback on three size-tier universes (min-size floor + regime
filter + trend-riding exit + liquidity filter ON), with per-side slippage that scales
with the tier (large 5bps, mid 15bps, small 40bps). The liquidity filter requires
real dollar volume and caps each position to a fraction of ADV, so we don't model
fills we couldn't get.

It prints the side-by-side metrics table (incl. illiquid-skip counts), the per-regime
breakdown for the small-cap run, and a prominent Limitations section. The headline
question: does moving down the size spectrum produce materially higher returns AFTER
honest liquidity and slippage costs, and how much does drawdown rise with it?

Read-only research. No IBKR, no orders. Parameters are NOT tuned.

Run from the repository root:

    python backtest/run_backtest.py
"""

import logging
import sqlite3
import sys
from pathlib import Path

# Allow running this file directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from backtest import metrics, universe  # noqa: E402
from backtest.engine import run_engine  # noqa: E402
from backtest.regime import compute_regime  # noqa: E402
from config import SLIPPAGE_BPS_LARGE, SLIPPAGE_BPS_MID, SLIPPAGE_BPS_SMALL  # noqa: E402
from screener.sectors import SECTOR_ETFS  # noqa: E402
from signals.breakout import BreakoutStrategy  # noqa: E402
from signals.pullback import PullbackStrategy  # noqa: E402

log = logging.getLogger("run_backtest")

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Per-side slippage (fraction) by tier.
SLIPPAGE_BY_TIER = {
    "large": SLIPPAGE_BPS_LARGE / 10_000.0,
    "mid": SLIPPAGE_BPS_MID / 10_000.0,
    "small": SLIPPAGE_BPS_SMALL / 10_000.0,
}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("risk.portfolio").setLevel(logging.WARNING)


def _pct(v):
    return f"{v * 100:+.2f}%" if v is not None else "n/a"


def _neg_pct(v):
    return f"{-v * 100:.2f}%" if v is not None else "n/a"


def _ratio(v):
    return f"{v:.2f}" if v is not None else "n/a"


def _money(v):
    return f"${v:,.0f}" if v is not None else "n/a"


def print_comparison(overalls: dict[str, dict], illiquid: dict[str, int]) -> None:
    """Print key metrics side by side, one column per universe."""
    labels = list(overalls)
    spec = [
        ("Total return", lambda lab: _pct(overalls[lab]["total_return"])),
        ("CAGR", lambda lab: _pct(overalls[lab]["cagr"])),
        ("Max drawdown", lambda lab: _neg_pct(overalls[lab]["max_drawdown"])),
        ("Win rate", lambda lab: _pct(overalls[lab]["win_rate"])),
        ("Payoff ratio", lambda lab: _ratio(overalls[lab]["payoff_ratio"])),
        ("Profit factor", lambda lab: _ratio(overalls[lab]["profit_factor"])),
        ("Number of trades", lambda lab: str(overalls[lab]["num_trades"])),
        ("Illiquid skips", lambda lab: str(illiquid[lab])),
    ]
    rows = [[name] + [fmt(lab) for lab in labels] for name, fmt in spec]
    print("\n=== Cap-spectrum comparison (breakout + pullback; liquidity + slippage by tier) ===")
    print(pd.DataFrame(rows, columns=["metric"] + labels).to_string(index=False))
    print("  Per-side slippage: " + ", ".join(f"{lab} {SLIPPAGE_BY_TIER[lab] * 10000:.0f}bps"
                                               for lab in labels))


def print_by_regime(by_regime: dict[str, dict]) -> None:
    """Print the per-regime metric breakdown for the small-cap run."""
    rows = [{
        "regime": label, "days": by_regime[label]["days"], "trades": by_regime[label]["num_trades"],
        "win_rate": _pct(by_regime[label]["win_rate"]), "payoff": _ratio(by_regime[label]["payoff_ratio"]),
        "regime_return": _pct(by_regime[label]["total_return"]),
        "regime_max_dd": _neg_pct(by_regime[label]["max_drawdown"]),
        "total_pnl": _money(by_regime[label]["total_pnl"]),
    } for label in ("bull", "bear", "crash")]
    print("\n=== By regime (small-cap run) ===")
    print(pd.DataFrame(rows).to_string(index=False))


def print_limitations(data_skipped: dict[str, list]) -> None:
    """Print limitations specific to the cap-spectrum test."""
    print("\n=== Limitations (this test especially) ===")
    notes = [
        "SURVIVORSHIP BIAS (worst for small-caps): universes are TODAY's surviving "
        "tickers. Delisted, merged, and failed names are absent — small-cap failure is "
        "common, so small-cap returns here are biased optimistic, likely materially.",
        "No point-in-time constituents: membership is fixed to today; a name that was "
        "tiny 10y ago is treated as in-universe throughout. Strong small-cap results "
        "CANNOT be trusted with real money without point-in-time data.",
        "Liquidity modeling is approximate: a $5M ADV floor, $5 price floor, and a 1%-of-"
        "ADV position cap are coarse proxies; real fills depend on spread, depth, and "
        "intraday timing not modeled here.",
        "Slippage is an ESTIMATE: flat per-side bps by tier (large 5 / mid 15 / small 40) "
        "with a flat per-share commission; real small-cap slippage is variable and often "
        "worse, especially on exits and in stress.",
        "Adjusted prices + daily bars; long-only; one position per name; no borrow/taxes. "
        "Parameters are NOT tuned. Live out-of-sample confirmation required.",
    ]
    for i, note in enumerate(notes, 1):
        print(f"  {i}. {note}")
    for tier, skipped in data_skipped.items():
        if skipped:
            preview = ", ".join(skipped[:10]) + (" ..." if len(skipped) > 10 else "")
            print(f"  {tier}: {len(skipped)} symbols skipped for short/no history: {preview}")
    print()


def save_outputs(equity: pd.Series, trades: pd.DataFrame, label: str) -> None:
    """Save one run's equity curve and trade log to CSV and SQLite."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trades.to_csv(OUTPUT_DIR / f"trade_log_{label}.csv", index=False)
    equity_frame = equity.rename("equity").rename_axis("date").reset_index()
    equity_frame.to_csv(OUTPUT_DIR / f"equity_curve_{label}.csv", index=False)
    connection = sqlite3.connect(OUTPUT_DIR / "backtest.db")
    try:
        trades.to_sql(f"trades_{label}", connection, if_exists="replace", index=False)
        equity_frame.to_sql(f"equity_{label}", connection, if_exists="replace", index=False)
        connection.commit()
    finally:
        connection.close()


def main() -> int:
    """Run the large/mid/small comparison and report it honestly."""
    configure_logging()

    results: dict[str, tuple[pd.Series, pd.DataFrame]] = {}
    overalls: dict[str, dict] = {}
    illiquid: dict[str, int] = {}
    data_skipped: dict[str, list] = {}
    small_regime = None

    for tier in ("large", "mid", "small"):
        log.info("Loading and running universe '%s'...", tier)
        bars, skipped = universe.load_bars(tier)
        if universe.BENCHMARK not in bars:
            log.error("Benchmark missing for %s; skipping.", tier)
            continue
        regime = compute_regime(bars[universe.BENCHMARK]["Close"])
        sector_map = universe.sector_map(tier)
        stats: dict = {}
        equity, trades = run_engine(
            bars, regime, sector_map, list(SECTOR_ETFS),
            strategies=[BreakoutStrategy(), PullbackStrategy()],
            regime_filter=True, trend_exit=True, conviction_sizing=False,
            slippage_pct=SLIPPAGE_BY_TIER[tier], liquidity_filter=True, stats=stats,
        )
        results[tier] = (equity, trades)
        overalls[tier] = metrics.compute_overall(trades, equity)
        illiquid[tier] = stats.get("illiquid_skips", 0)
        data_skipped[tier] = skipped
        if tier == "small":
            small_regime = regime

    if not results:
        log.error("No universes produced results.")
        return 1

    any_equity = next(iter(results.values()))[0]
    print(f"\nWindow: {any_equity.index[0].date()} -> {any_equity.index[-1].date()}")
    print_comparison(overalls, illiquid)

    if "small" in results and small_regime is not None:
        small_equity, small_trades = results["small"]
        print_by_regime(metrics.compute_by_regime(small_trades, small_equity, small_regime))

    for tier, (eq, tr) in results.items():
        save_outputs(eq, tr, tier)
    print_limitations(data_skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
