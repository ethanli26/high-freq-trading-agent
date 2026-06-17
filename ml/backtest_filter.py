"""Does ML filtering actually improve the backtest on UNSEEN data?

Re-runs the strategy backtest on the held-out TEST period only, two ways:
  (a) take all signals (current behavior)
  (b) take only signals whose model win-probability >= a threshold (0.50/0.55/0.60)

CRITICAL leakage guard (asserted in code): the model is trained ONLY on the earlier
~70% of signals and only ever scores test-period signals — it never saw the test
window. The backtest config is held fixed (both strategies, regime filter ON,
trend-riding exit ON, conviction sizing OFF) so the ONLY thing that varies between
(a) and (b) is the ML filter. Read-only research: no IBKR, no orders.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running this file directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import metrics  # noqa: E402
from backtest.data import BENCHMARK, build_sector_map, load_universe  # noqa: E402
from backtest.engine import run_engine  # noqa: E402
from backtest.regime import compute_regime  # noqa: E402
from ml.dataset import FEATURE_COLUMNS  # noqa: E402
from ml.train import fit_model, load_dataset, temporal_split  # noqa: E402
from screener.sectors import SECTOR_ETFS  # noqa: E402
from signals.breakout import BreakoutStrategy  # noqa: E402
from signals.pullback import PullbackStrategy  # noqa: E402

log = logging.getLogger(__name__)

THRESHOLDS = [0.50, 0.55, 0.60]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("risk.portfolio").setLevel(logging.WARNING)


def build_probability_lookup(model, scaler, test: pd.DataFrame) -> dict:
    """Map ``(symbol, date, strategy) -> win probability`` for test-period signals."""
    probs = model.predict_proba(scaler.transform(test[FEATURE_COLUMNS].to_numpy(dtype=float)))[:, 1]
    return {
        (row.symbol, pd.Timestamp(row.date), row.strategy): float(prob)
        for row, prob in zip(test.itertuples(index=False), probs)
    }


def make_entry_filter(prob_lookup: dict, threshold: float):
    """Return an ``entry_filter(symbol, date, strategy)`` that keeps prob >= threshold.

    Signals the model never scored (missing from the lookup) are skipped — we only
    take signals the model has actually rated as strong enough.
    """
    def entry_filter(symbol: str, date, strategy: str) -> bool:
        prob = prob_lookup.get((symbol, pd.Timestamp(date), strategy))
        return prob is not None and prob >= threshold

    return entry_filter


def _pct(v):
    return f"{v * 100:+.2f}%" if v is not None else "n/a"


def _neg_pct(v):
    return f"{-v * 100:.2f}%" if v is not None else "n/a"


def _ratio(v):
    return f"{v:.2f}" if v is not None else "n/a"


def print_comparison(overalls: dict[str, dict], base_rate_test: float) -> None:
    """Print unfiltered vs filtered metrics side by side."""
    labels = list(overalls)
    spec = [
        ("Total return", lambda m: _pct(m["total_return"])),
        ("CAGR", lambda m: _pct(m["cagr"])),
        ("Max drawdown", lambda m: _neg_pct(m["max_drawdown"])),
        ("Win rate", lambda m: _pct(m["win_rate"])),
        ("Payoff ratio", lambda m: _ratio(m["payoff_ratio"])),
        ("Profit factor", lambda m: _ratio(m["profit_factor"])),
        ("Number of trades", lambda m: str(m["num_trades"])),
    ]
    rows = [[name] + [fmt(overalls[label]) for label in labels] for name, fmt in spec]
    print("\n=== Filtered vs unfiltered (held-out TEST period only) ===")
    print(pd.DataFrame(rows, columns=["metric"] + labels).to_string(index=False))
    print(f"\n  Dataset base rate (isolated-signal win rate, test) : {base_rate_test:.3f}")
    unfiltered_wr = overalls["unfiltered"]["win_rate"]
    print(f"  Engine win rate, unfiltered (take every signal)    : {unfiltered_wr:.3f}")
    print("  -> ML earns its place only if a filtered column's win rate clears the "
          "unfiltered win rate by a real margin.")


def main() -> int:
    """Train on early data, then compare filtered vs unfiltered on the test period."""
    configure_logging()

    dataset = load_dataset()
    train, test, split_date = temporal_split(dataset)
    scaler, model = fit_model(train)

    # CRITICAL LEAKAGE GUARDS — assert the model never saw the test period.
    assert train["date"].max() < split_date, "train contains test-period dates!"
    assert test["date"].min() >= split_date, "test starts before the split!"
    prob_lookup = build_probability_lookup(model, scaler, test)
    assert min(d for _, d, _ in prob_lookup) >= split_date, "filter scores a pre-test date!"
    log.info("Leakage guard OK: model trained on dates < %s; filter applies only to dates >= %s.",
             split_date.date(), split_date.date())

    bars, _ = load_universe()
    regime = compute_regime(bars[BENCHMARK]["Close"])
    sector_map = build_sector_map()
    etfs = list(SECTOR_ETFS)
    strategies = [BreakoutStrategy(), PullbackStrategy()]

    # Fixed config; only the ML entry filter varies. Trade only the test window.
    common = dict(strategies=strategies, regime_filter=True, trend_exit=True,
                  conviction_sizing=False, start_date=split_date)

    log.info("Running UNFILTERED (all signals) on the test period...")
    results = {"unfiltered": run_engine(bars, regime, sector_map, etfs, **common)}
    for threshold in THRESHOLDS:
        log.info("Running FILTERED p>=%.2f ...", threshold)
        results[f"p>={threshold:.2f}"] = run_engine(
            bars, regime, sector_map, etfs, entry_filter=make_entry_filter(prob_lookup, threshold), **common
        )

    overalls = {label: metrics.compute_overall(tr, eq) for label, (eq, tr) in results.items()}
    base_rate_test = test["label"].mean()

    print(f"\nTest period: {split_date.date()} -> {test['date'].max().date()}")
    print_comparison(overalls, base_rate_test)

    print("\n=== ML Limitations ===")
    for note in _LIMITATIONS:
        print(f"  - {note}")
    return 0


_LIMITATIONS = [
    "Single held-out split, not walk-forward: one test window can mislead.",
    "Survivorship-biased universe; in-sample feature/strategy choices.",
    "Filtering changes trade count, so fewer trades also means noisier metrics.",
    "Even a clean improvement here needs LIVE out-of-sample confirmation before use.",
]


if __name__ == "__main__":
    sys.exit(main())
