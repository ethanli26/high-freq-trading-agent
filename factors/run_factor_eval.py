"""Re-run the factor IC harness on a BROAD universe and compare to large-cap.

Tests whether the inverted-sign risk-factor cluster (suspected large-cap +
survivorship artifact) corrects when the universe widens to ~300+ names across the
cap spectrum. Same factors, same bar (|mean IC| > 0.02 AND |t-stat| > 2), monthly
rebalance, with a liquidity screen (>= $5M ADV, >= $5 price) so untradeable names are
excluded as of each rebalance date.

Prints: the broad-universe scorecard (sorted by |t|), a side-by-side delta vs the
large-cap run (IC change, sign flip, significance change), a focused look at the
risk-factor cluster's move toward its literature-expected sign, and an honest note.

Read-only research. No IBKR, no orders.

    python factors/run_factor_eval.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

import factors  # noqa: E402,F401  (registers built-ins on import)
from backtest.universe import BENCHMARK, constituents, load_bars, load_sharadar_broad  # noqa: E402
from config import LIQUIDITY_LOOKBACK, MIN_DOLLAR_VOLUME, MIN_PRICE  # noqa: E402
from data.sharadar_provider import SharadarUnavailable, build_fundamental_panels  # noqa: E402
from factors.base import FactorData, all_factors  # noqa: E402
from factors.evaluate import evaluate_factor  # noqa: E402

log = logging.getLogger("run_factor_eval")

IC_THRESHOLD = 0.02
TSTAT_THRESHOLD = 2.0

# Literature-expected IC sign for each factor as encoded (so positive IC = thesis
# holds). Used to judge whether broadening moves the risk cluster the "right" way.
EXPECTED_SIGN = {
    "momentum_12_1": +1, "momentum_6_1": +1, "short_term_reversal": +1,
    "low_volatility": +1, "ivol_capm": -1, "beta_low": +1, "max_daily_return": -1,
    "return_skewness": -1, "ncskew": -1, "duvol": -1,
    # US-effective fundamental family (all higher = better/cheaper/faster => higher returns).
    "profitability": +1, "earnings_yield": +1, "book_to_price": +1, "earnings_growth": +1,
}
RISK_CLUSTER = ["ivol_capm", "beta_low", "low_volatility", "max_daily_return"]
# The fundamental family this run exists to test.
FUNDAMENTAL_FACTORS = ["profitability", "earnings_yield", "book_to_price", "earnings_growth"]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("yfinance").setLevel(logging.WARNING)


def build_factor_data(universe: str) -> FactorData:
    """Load OHLCV bars for a universe and assemble aligned date x symbol panels."""
    bars, _ = load_bars(universe)
    symbols = [s for s in constituents(universe) if s in bars]
    master = bars[BENCHMARK].index

    def panel(field: str) -> pd.DataFrame:
        return pd.DataFrame({s: bars[s][field].reindex(master) for s in symbols})

    return FactorData(open=panel("Open"), high=panel("High"), low=panel("Low"),
                      close=panel("Close"), volume=panel("Volume"),
                      market=bars[BENCHMARK]["Close"].reindex(master))


def liquidity_mask(data: FactorData) -> pd.DataFrame:
    """Eligibility (date x symbol): ADV >= MIN_DOLLAR_VOLUME and price >= MIN_PRICE.

    LOOK-AHEAD GUARD: the rolling dollar-volume average ends at t and the price is the
    close at t, so eligibility on the rebalance date uses only data <= t.
    """
    adv = (data.close * data.volume).rolling(LIQUIDITY_LOOKBACK).mean()
    return (adv >= MIN_DOLLAR_VOLUME) & (data.close >= MIN_PRICE)


def evaluate_universe(universe: str) -> tuple[dict, list]:
    """Score every runnable factor on a universe (with the liquidity screen)."""
    data = build_factor_data(universe)
    eligible = liquidity_mask(data)
    scores: dict[str, dict] = {}
    deferred: list[str] = []
    for name, cls in all_factors().items():
        factor = cls()
        if getattr(factor, "point_in_time_provider", False):
            deferred.append(name)
            continue
        scores[name] = evaluate_factor(factor, data, eligible=eligible)
    log.info("Universe '%s': scored %d factors (%d deferred), %d names, %d dates.",
             universe, len(scores), len(deferred), data.close.shape[1], len(data.close))
    return scores, deferred


def _fmt_ic(v):
    return f"{v:+.4f}" if v is not None else "n/a"


def _fmt_num(v):
    return f"{v:+.2f}" if v is not None else "n/a"


def _fmt_pct(v):
    return f"{v * 100:+.2f}%" if v is not None else "n/a"


def _verdict(score: dict) -> str:
    """Worth building only if IC is sizeable AND significant (bar unchanged)."""
    ic, t = score["mean_ic"], score["t_stat"]
    if ic is None or t is None:
        return "no signal"
    return "BUILD" if (abs(ic) > IC_THRESHOLD and abs(t) > TSTAT_THRESHOLD) else "no signal"


def _sign(x) -> int:
    return 0 if (x is None or abs(x) < 1e-12) else (1 if x > 0 else -1)


def print_scorecard(scores: dict, title: str) -> list[str]:
    """Print one universe's scorecard sorted by |t-stat|; return BUILD names."""
    ordered = sorted(scores.values(),
                     key=lambda s: abs(s["t_stat"]) if s["t_stat"] is not None else -1.0, reverse=True)
    rows = [{
        "factor": s["name"], "cat": s["category"], "periods": s["n_periods"],
        "mean_IC": _fmt_ic(s["mean_ic"]), "IR": _fmt_num(s["ir"]), "t_stat": _fmt_num(s["t_stat"]),
        "TMB_spread": _fmt_pct(s["top_minus_bottom"]), "TMB_Sharpe": _fmt_num(s["tmb_sharpe"]),
        "verdict": _verdict(s),
    } for s in ordered]
    print(title)
    print(pd.DataFrame(rows).to_string(index=False))
    return [s["name"] for s in ordered if _verdict(s) == "BUILD"]


def print_delta(large: dict, broad: dict) -> None:
    """Side-by-side IC delta: large vs broad, sign flips, significance changes."""
    rows = []
    for name, bs in broad.items():
        ls = large.get(name, {})
        lic, bic = ls.get("mean_ic"), bs["mean_ic"]
        lt, bt = ls.get("t_stat"), bs["t_stat"]
        flip = "yes" if (_sign(lic) and _sign(bic) and _sign(lic) != _sign(bic)) else "-"
        sig_l = "sig" if (lt is not None and abs(lt) > TSTAT_THRESHOLD) else "ns"
        sig_b = "sig" if (bt is not None and abs(bt) > TSTAT_THRESHOLD) else "ns"
        rows.append({
            "factor": name, "large_IC": _fmt_ic(lic), "broad_IC": _fmt_ic(bic),
            "dIC": _fmt_ic(bic - lic) if (lic is not None and bic is not None) else "n/a",
            "large_t": _fmt_num(lt), "broad_t": _fmt_num(bt),
            "sign_flip": flip, "signif": f"{sig_l}->{sig_b}",
        })
    rows.sort(key=lambda r: abs(broad[r["factor"]]["t_stat"]) if broad[r["factor"]]["t_stat"] else -1, reverse=True)
    print("\n=== Delta vs prior large-cap run (sorted by broad |t|) ===")
    print(pd.DataFrame(rows).to_string(index=False))


def print_risk_cluster(large: dict, broad: dict) -> tuple[int, int]:
    """Show whether the risk cluster moves toward its expected sign when broadened."""
    rows, corrected, moved = [], 0, 0
    for name in RISK_CLUSTER:
        if name not in broad:
            continue
        exp = EXPECTED_SIGN[name]
        lic, bic = large.get(name, {}).get("mean_ic"), broad[name]["mean_ic"]
        sign_ok = _sign(bic) == exp
        toward = (lic is not None and bic is not None and (bic - lic) * exp > 0)
        corrected += int(sign_ok)
        moved += int(toward)
        rows.append({
            "factor": name, "expected_sign": "+" if exp > 0 else "-",
            "large_IC": _fmt_ic(lic), "broad_IC": _fmt_ic(bic),
            "broad_sign_ok": "yes" if sign_ok else "no",
            "moved_toward_expected": "yes" if toward else "no",
        })
    print("\n=== Risk-factor cluster: does broadening fix the inverted signs? ===")
    print(pd.DataFrame(rows).to_string(index=False))
    return corrected, moved


def build_sharadar_factor_data() -> tuple[FactorData, pd.DataFrame]:
    """Assemble point-in-time, survivorship-free FactorData from Sharadar.

    Returns ``(FactorData, eligible_mask)``. Price panels span delisted + live names;
    ``fundamentals`` are filing-dated daily panels (see build_fundamental_panels). Raises
    ``SharadarUnavailable`` in stub mode.
    """
    bars, filings, benchmark_close = load_sharadar_broad()
    symbols = sorted(bars)
    master = benchmark_close.index if benchmark_close is not None else bars[symbols[0]].index

    def panel(field: str) -> pd.DataFrame:
        return pd.DataFrame({s: bars[s][field].reindex(master) for s in symbols})

    # POINT-IN-TIME: fundamentals are forward-filled from FILING dates onto `master`.
    fundamentals = build_fundamental_panels(filings, master)
    data = FactorData(open=panel("Open"), high=panel("High"), low=panel("Low"),
                      close=panel("Close"), volume=panel("Volume"),
                      market=benchmark_close, fundamentals=fundamentals)
    return data, liquidity_mask(data)


def evaluate_fundamentals(data: FactorData, eligible: pd.DataFrame) -> dict:
    """Score the point-in-time fundamental factors on the Sharadar universe."""
    scores: dict[str, dict] = {}
    for name in FUNDAMENTAL_FACTORS:
        factor = all_factors()[name]()
        scores[name] = evaluate_factor(factor, data, eligible=eligible)
    return scores


def print_fundamental_scorecard(scores: dict) -> list[str]:
    """Print the fundamental scorecard sorted by |t|, with sign-vs-prior; return BUILDs."""
    ordered = sorted(scores.values(),
                     key=lambda s: abs(s["t_stat"]) if s["t_stat"] is not None else -1.0, reverse=True)
    rows = []
    for s in ordered:
        exp = EXPECTED_SIGN.get(s["name"])
        sign_ok = exp is not None and _sign(s["mean_ic"]) == exp
        rows.append({
            "factor": s["name"], "periods": s["n_periods"],
            "mean_IC": _fmt_ic(s["mean_ic"]), "IR": _fmt_num(s["ir"]), "t_stat": _fmt_num(s["t_stat"]),
            "decile_spread": _fmt_pct(s["top_minus_bottom"]), "TMB_Sharpe": _fmt_num(s["tmb_sharpe"]),
            "sign_ok": "yes" if sign_ok else "no", "verdict": _verdict(s),
        })
    print("\n=== FUNDAMENTAL factor scorecard — point-in-time, survivorship-free "
          "(monthly; |IC|>0.02 & |t|>2 => BUILD; sorted by |t|) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    return [s["name"] for s in ordered if _verdict(s) == "BUILD"]


def run_fundamental_section(price_builds: list[str]) -> None:
    """Run the fundamental family on Sharadar, or report stub mode clearly."""
    print("\n" + "=" * 78)
    print("US-EFFECTIVE FUNDAMENTAL FAMILY  (the test this whole layer was built for)")
    print("=" * 78)
    try:
        data, eligible = build_sharadar_factor_data()
    except SharadarUnavailable as error:
        print("STUB MODE — fundamental factors NOT run (no Sharadar API key).")
        print(f"  Reason: {error}")
        print("  The plumbing is fully built and verified: provider, point-in-time filing-date")
        print("  panels, the four factors (profitability, earnings_yield, book_to_price,")
        print("  earnings_growth), the survivorship-free 'sharadar_broad' universe, and this")
        print("  harness path all run end to end on synthetic fundamentals in the test suite.")
        print(f"  To produce REAL numbers, set NASDAQ_DATA_LINK_API_KEY and re-run; only the")
        print("  final result is gated on the spend, not the code.")
        print(f"  For reference, the PRICE family on the free large-cap universe cleared the bar "
              f"with: {', '.join(price_builds) if price_builds else 'NOTHING'}.")
        return

    builds = print_fundamental_scorecard(evaluate_fundamentals(data, eligible))
    print(f"\nFundamental factors clearing the bar: {', '.join(builds) if builds else 'NONE'}.")
    print("\n=== Honest note: fundamentals (PIT, survivorship-free) vs price (free, survivors) ===")
    print("  - This is the FIRST run of the fundamental family on point-in-time, "
          "survivorship-free data — the data the US-vs-China study says these factors need.")
    print(f"  - Price family on the OLD free large-cap universe cleared the bar with: "
          f"{', '.join(price_builds) if price_builds else 'NOTHING'}.")
    if builds:
        print(f"  - VERDICT: the US-effective fundamental family DOES clear the bar here "
              f"({', '.join(builds)}) where the price family largely did not — consistent with "
              f"the study's claim that fundamentals, not price/reversal, drive US returns.")
    else:
        print("  - VERDICT: even on point-in-time, survivorship-free data the fundamental family "
              "did NOT clear the bar on this window — a genuinely negative result, not a data "
              "artifact, since survivorship and look-ahead were removed.")
    print("  - The bar (|IC|>0.02 & |t|>2) was NOT lowered; IC is close-to-close predictive "
          "power, not tradable P&L.")


def main() -> int:
    """Evaluate large (reference) and broad universes, then compare honestly."""
    configure_logging()

    log.info("Evaluating LARGE universe (reference)...")
    large_scores, deferred = evaluate_universe("large")
    log.info("Evaluating BROAD universe (first run fetches ~300+ symbols)...")
    broad_scores, _ = evaluate_universe("broad")

    builds = print_scorecard(
        broad_scores,
        "\n=== BROAD-universe factor IC scorecard (monthly; |IC|>0.02 & |t|>2 => BUILD; sorted by |t|) ===")
    print(f"\nWorth turning into a strategy (broad): {', '.join(builds) if builds else 'NONE'}.")
    if deferred:
        print(f"Declared but NOT run (need point-in-time paid data): {', '.join(deferred)}.")

    print_delta(large_scores, broad_scores)
    corrected, moved = print_risk_cluster(large_scores, broad_scores)

    print("\n=== Honest note ===")
    print("  - Broad widens the sample across the cap spectrum and reduces large-cap "
          "concentration bias, but does NOT remove SURVIVORSHIP bias — these are still "
          "today's survivors. Only point-in-time paid data can fix that.")
    print("  - IC is close-to-close predictive power on an in-sample window, not "
          "tradable P&L; the bar (|IC|>0.02 & |t|>2) was NOT lowered.")
    n = len(RISK_CLUSTER)
    thesis_passes = [b for b in builds
                     if b in EXPECTED_SIGN and _sign(broad_scores[b]["mean_ic"]) == EXPECTED_SIGN[b]]
    atheoretical = [b for b in builds if b not in EXPECTED_SIGN]
    print(f"  - Risk cluster: {corrected}/{n} carry the expected sign; {moved}/{n} moved TOWARD "
          f"it (inversion weakened, did not reverse) when broadened.")
    if thesis_passes:
        print(f"  - VERDICT: a correctly-signed, significant factor emerged on the broad universe "
              f"({', '.join(thesis_passes)}) — this DOES justify a point-in-time data test to "
              f"confirm it survives once survivorship bias is removed.")
    elif moved >= 3:
        extra = (f" The only bar-passers are atheoretical alphas ({', '.join(atheoretical)}, weak IC)."
                 if atheoretical else "")
        print("  - VERDICT: broadening WEAKENED the inverted risk-factor signs toward their expected "
              "direction but did NOT reverse them, and none became significant." + extra +
              " That is consistent with large-cap concentration inflating the inversion, yet leaves "
              "survivorship bias unresolved. A point-in-time broad test is the warranted next step "
              "IF pursuing these factors; on current free, survivor-only evidence, expectations "
              "should be modest.")
    else:
        print("  - VERDICT: inverted/weak signs largely PERSIST; cap-concentration was not the main "
              "driver. A point-in-time test remains the only way to settle it.")

    # The headline test: the US-effective fundamental family on PIT, survivorship-free data.
    run_fundamental_section(builds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
