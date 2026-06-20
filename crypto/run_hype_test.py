"""Honest pilot: crypto hype-momentum vs buy-and-hold BTC, after brutal costs.

The question: does a hype-momentum signal (breakout + volume surge) on a broad
small-coin universe beat simply buying and holding BTC, AFTER realistic, illiquidity-
scaled trading costs — on a SURVIVOR-ONLY universe?

It runs the strategy on the SAME walk-forward engine, risk sizing, and benchmark code
used for equities, with crypto's per-coin cost model and liquidity caps wired into the
engine's hooks. It then prints after-cost metrics, a cost-sensitivity sweep (mirroring
the intraday cost test), an explicit verdict vs buy-and-hold BTC, and a prominent
Limitations section.

DATA HONESTY: by default this uses DETERMINISTIC SYNTHETIC data (no network), which has
no designed edge — it exercises the mechanics, the per-coin cost scaling, and the
honest-accounting pipeline, and PROVES NOTHING about real edge. Pass ``--real`` to pull
cached/live coins through ``CryptoDataProvider`` and run the identical pipeline on them.

Read-only research. No exchange orders.

Run from the repo root:  python crypto/run_hype_test.py
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from backtest import metrics, risk_metrics  # noqa: E402
from backtest.benchmark import buy_and_hold  # noqa: E402
from backtest.engine import run_engine  # noqa: E402
from backtest.regime import compute_regime  # noqa: E402
from crypto import universe as crypto_universe  # noqa: E402
from crypto.costs import CryptoCostModel, build_slippage_fn, representative_adv  # noqa: E402
from data.crypto_provider import CryptoDataProvider, make_synthetic_crypto  # noqa: E402
from strategies.crypto_hype import CryptoHypeStrategy  # noqa: E402

log = logging.getLogger("run_hype_test")

STARTING_EQUITY = 1_000_000.0
MIN_DOLLAR_VOLUME = 1_000_000.0   # crypto liquidity floor (no price floor for coins)
LIQUIDITY_LOOKBACK = 30
COMMISSION_PER_SHARE = 0.0        # crypto cost is bps-of-notional (in slippage), not per-share
STRESS_LEVELS = (0.0, 0.5, 1.0, 2.0, 4.0)  # cost-sensitivity sweep (x the spread term)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("backtest.engine", "crypto.universe", "data.crypto_provider",
                  "backtest.regime", "risk.portfolio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_bars(use_real: bool, n_coins: int) -> dict[str, pd.DataFrame]:
    """Load crypto bars: cached/live via the provider (``--real``) or synthetic."""
    if use_real:
        provider = CryptoDataProvider(backend="yfinance")
        symbols = ["BTC", "ETH", "SOL", "ADA", "AVAX", "LINK", "DOT", "MATIC", "ATOM",
                   "ALGO", "FIL", "XTZ", "EGLD", "FLOW", "MANA", "SAND", "AXS", "CHZ",
                   "ENJ", "ZIL", "ONE", "KAVA", "BAND", "OCEAN", "RLC"]
        bars = provider.get_price_bars(symbols)
        if "BTC" not in bars:
            log.error("No BTC data from provider; falling back to synthetic.")
            return load_bars(False, n_coins)
        return bars
    # Synthetic: BTC benchmark + n small coins (incl. one wash-like coin).
    symbols = ["BTC"] + [f"SC{i:02d}" for i in range(1, n_coins)] + ["SCWASH"]
    return make_synthetic_crypto(symbols, days=900, seed=11)


def run_pilot(bars, cost_model, tradable, sector_map, btc_close, regime, stats=None):
    """Run the hype strategy through the engine with the given cost model. Returns
    ``(equity, trades)``."""
    engine_bars = {**tradable, crypto_universe.BENCHMARK: bars[crypto_universe.BENCHMARK]}
    slippage_fn = build_slippage_fn(engine_bars, cost_model, LIQUIDITY_LOOKBACK)
    return run_engine(
        engine_bars, regime, sector_map, [crypto_universe.BENCHMARK],
        strategies=[CryptoHypeStrategy()],
        starting_equity=STARTING_EQUITY, commission_per_share=COMMISSION_PER_SHARE,
        regime_filter=False, trend_exit=True, conviction_sizing=False,
        liquidity_filter=True, slippage_fn=slippage_fn,
        min_dollar_volume=MIN_DOLLAR_VOLUME, min_price=0.0,
        max_adv_participation=cost_model.adv_participation_cap,
        liquidity_lookback=LIQUIDITY_LOOKBACK, stats=stats)


def _pct(v):
    return f"{v * 100:+.2f}%" if v is not None else "n/a"


def _ratio(v):
    return f"{v:.2f}" if v is not None else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description="Crypto hype-momentum pilot vs buy-and-hold BTC.")
    parser.add_argument("--real", action="store_true",
                        help="Use CryptoDataProvider (cached/live) instead of synthetic data.")
    parser.add_argument("--coins", type=int, default=25, help="Synthetic small-coin count.")
    args = parser.parse_args()
    configure_logging()

    bars = load_bars(args.real, args.coins)
    synthetic = not args.real
    if crypto_universe.BENCHMARK not in bars:
        log.error("Benchmark %s missing; cannot run.", crypto_universe.BENCHMARK)
        return 1

    # Build the tradable small-coin universe (drops mega-caps + illiquid/short/wash).
    tradable, skips = crypto_universe.build_universe(
        bars, min_dollar_volume=MIN_DOLLAR_VOLUME, min_history_days=200)
    if not tradable:
        log.error("No tradable coins after filtering.")
        return 1
    sector_map = crypto_universe.sector_map(tradable.keys())
    btc_close = bars[crypto_universe.BENCHMARK]["Close"]
    regime = compute_regime(btc_close)

    # --- Baseline run (stress = 1.0) ---------------------------------------------
    base_model = CryptoCostModel(stress_multiplier=1.0)
    stats: dict = {}
    equity, trades = run_pilot(bars, base_model, tradable, sector_map, btc_close, regime, stats)
    overall = metrics.compute_overall(trades, equity)
    rm = risk_metrics.compute_metrics(equity)

    bench = buy_and_hold(btc_close, equity, STARTING_EQUITY)
    bench_overall = metrics.equity_metrics(bench)
    bench_rm = risk_metrics.compute_metrics(bench)

    label = "SYNTHETIC (deterministic; no designed edge — mechanics/cost test only)" if synthetic \
        else "REAL coins via CryptoDataProvider (survivor-only)"

    print("=" * 78)
    print("CRYPTO HYPE-MOMENTUM PILOT  vs  BUY-AND-HOLD BTC")
    print("=" * 78)
    print(f"Data:            {label}")
    print(f"Window:          {equity.index[0].date()} -> {equity.index[-1].date()}")
    print(f"Tradable coins:  {len(tradable)}  (mega excluded: {len(skips['mega'])}, "
          f"illiquid: {len(skips['illiquid'])}, short-history: {len(skips['short'])}, "
          f"wash-flagged: {len(skips['wash'])})")
    print(f"In-engine illiquid skips (ADV/price gate at entry): {stats.get('illiquid_skips', 0)}")
    print(f"Cost model:      taker {base_model.taker_fee_bps:.0f}bps + spread "
          f"{base_model.base_spread_bps:.0f}bps x illiquidity, per side; "
          f"position cap {base_model.adv_participation_cap*100:.0f}% of ADV")
    print()
    print(f"{'metric':<22}{'hype strategy':>18}{'buy & hold BTC':>18}")
    print("-" * 58)
    rows = [
        ("Total return", _pct(overall["total_return"]), _pct(bench_overall["total_return"])),
        ("CAGR", _pct(overall["cagr"]), _pct(bench_overall["cagr"])),
        ("Sharpe", _ratio(rm["sharpe"]), _ratio(bench_rm["sharpe"])),
        ("Sortino", _ratio(rm["sortino"]), _ratio(bench_rm["sortino"])),
        ("Max drawdown", _pct(-overall["max_drawdown"] if overall["max_drawdown"] is not None else None),
         _pct(-bench_overall["max_drawdown"] if bench_overall["max_drawdown"] is not None else None)),
        ("Win rate", _pct(overall["win_rate"]), "—"),
        ("Payoff ratio", _ratio(overall["payoff_ratio"]), "—"),
        ("Trades", str(overall["num_trades"]), "1"),
    ]
    for name, a, b in rows:
        print(f"{name:<22}{a:>18}{b:>18}")
    print()

    # --- COST SENSITIVITY ---------------------------------------------------------
    median_adv = pd.Series(
        [representative_adv(f, LIQUIDITY_LOOKBACK) for f in tradable.values()]).median()
    print("-" * 78)
    print("COST-SENSITIVITY  (turning up the spread/slippage knob until the edge dies)")
    print("-" * 78)
    header = (f"{'stress x':>9} | {'rt cost (median coin)':>22} | {'strat total':>12} | "
              f"{'vs BTC':>10} | {'trades':>6}")
    print(header)
    print("-" * len(header))
    btc_total = bench_overall["total_return"]
    for stress in STRESS_LEVELS:
        model = CryptoCostModel(stress_multiplier=stress)
        st: dict = {}
        eq, tr = run_pilot(bars, model, tradable, sector_map, btc_close, regime, st)
        ov = metrics.equity_metrics(eq)
        rt_bps = model.round_trip_bps(median_adv)
        beats = ov["total_return"] is not None and btc_total is not None and ov["total_return"] > btc_total
        print(f"{stress:>9.1f} | {rt_bps:>18.0f} bps | {_pct(ov['total_return']):>12} | "
              f"{('BEATS' if beats else 'loses'):>10} | {len(tr):>6}")
    print("-" * len(header))
    print(f"(round-trip cost shown for the median-liquidity coin, ADV ~ ${median_adv:,.0f}; "
          f"thinner coins pay multiples more.)")
    print()

    # --- VERDICT ------------------------------------------------------------------
    print("=" * 78)
    print("VERDICT vs BUY-AND-HOLD BTC")
    print("=" * 78)
    strat_total, strat_sharpe = overall["total_return"], rm["sharpe"]
    if strat_total is not None and btc_total is not None:
        verdict = "BEATS" if strat_total > btc_total else "DOES NOT BEAT"
        print(f"At realistic (stress=1.0) costs, the hype strategy {verdict} buy-and-hold BTC "
              f"on total return ({_pct(strat_total)} vs {_pct(btc_total)}),")
        print(f"with Sharpe {_ratio(strat_sharpe)} vs BTC {_ratio(bench_rm['sharpe'])}.")
    print("The cost-sensitivity sweep shows how quickly the result decays as the spread/")
    print("slippage assumption rises — the level where 'BEATS' flips to 'loses' is the most")
    print("you can be wrong about costs before the edge is gone.")
    if synthetic:
        print()
        print("HONEST NOTE: this run is on SYNTHETIC data with NO designed edge, so the")
        print("comparison above reflects the cost/mechanics pipeline, NOT real alpha. It cannot")
        print("tell you the strategy works. What it CAN show: the per-coin cost toll a real")
        print("small-coin hype strategy must overcome, and that the toll explodes for thin coins.")
    print_limitations(skips)
    return 0


def print_limitations(skips: dict) -> None:
    print()
    print("=" * 78)
    print("LIMITATIONS (read before trusting any number above)")
    print("=" * 78)
    notes = [
        "SURVIVORSHIP BIAS (severe here): the universe is TODAY's surviving, currently-"
        "liquid coins. The thousands of small coins that rugged, died, or delisted are "
        "absent. In small-coin crypto the survivors are a lucky minority, so returns are "
        "biased materially OPTIMISTIC. Free data cannot fix this — only point-in-time "
        "historical listings could.",
        "LIQUIDITY & SLIPPAGE ARE ESTIMATES: the per-coin cost scales spread with ADV via "
        "a coarse formula, not order-book depth. Real small-coin spreads are wider and "
        "more variable, and blow out in exactly the stressed, hype-driven moments this "
        "strategy trades. The cost-sensitivity sweep exists because the true level is unknown.",
        "WASH / FAKE VOLUME: reported crypto volume is heavily inflated by wash trading. "
        "The wash screen is a coarse OHLCV-only heuristic (a partial mitigation, not a "
        "fix); inflated volume also makes the ADV-based liquidity filter and position cap "
        "too generous, understating true cost.",
        "SINGLE WINDOW, NO TUNING SHOWN HONESTLY: one historical window, daily bars, long "
        "only, one position per coin, no borrow/funding/taxes. Results are not robust "
        "evidence; out-of-sample, multi-window confirmation on point-in-time data is required.",
    ]
    for i, note in enumerate(notes, 1):
        print(f"  {i}. {note}")
    if skips.get("wash"):
        print(f"  Wash-flagged & excluded: {', '.join(skips['wash'][:12])}"
              + (" ..." if len(skips["wash"]) > 12 else ""))
    print()


if __name__ == "__main__":
    sys.exit(main())
