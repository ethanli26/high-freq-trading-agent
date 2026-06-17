"""Multi-curve performance report: side-by-side table, charts, and honest summary.

Takes an ordered set of named daily equity curves (e.g. strategy, strategy+overlay,
S&P 500 buy-and-hold, risk-matched blend), computes the full risk-adjusted metric
set for each, and the relative metrics (correlation/beta/CAPM alpha) of each versus a
designated market curve. Prints + saves a comparison table (CSV), saves equity and
drawdown charts (all curves), optionally a quantstats tearsheet, and a plain-spoken
summary. The hand-computed metrics in risk_metrics.py are the source of truth.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files, no display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from backtest import risk_metrics  # noqa: E402

log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
_COLORS = ["#1f77b4", "#2ca02c", "#888888", "#ff7f0e", "#9467bd"]


def _pct(v):
    return f"{v * 100:+.2f}%" if v is not None else "n/a"


def _ratio(v):
    return f"{v:.2f}" if v is not None else "n/a"


def _money(v):
    return f"${v:,.0f}" if v is not None else "n/a"


def build_table(metrics_by: dict[str, dict], relative_by: dict[str, dict],
                labels: list[str]) -> pd.DataFrame:
    """Assemble the comparison table: one row per metric, one column per curve."""
    def srow(name, key, fmt):
        return [name] + [fmt(metrics_by[lab][key]) for lab in labels]

    def rrow(name, key, fmt):
        return [name] + [fmt(relative_by[lab][key]) for lab in labels]

    rows = [
        srow("Final equity", "final_equity", _money),
        srow("Total return", "total_return", _pct),
        srow("CAGR", "cagr", _pct),
        srow("Annualized volatility", "ann_volatility", _pct),
        srow("Sharpe (rf=4%)", "sharpe", _ratio),
        srow("Sortino", "sortino", _ratio),
        srow("Calmar", "calmar", _ratio),
        srow("Max drawdown", "max_drawdown", lambda v: _pct(-v) if v is not None else "n/a"),
        rrow("Correlation to S&P", "correlation", _ratio),
        rrow("Beta to S&P", "beta", _ratio),
        rrow("Annualized alpha (CAPM)", "alpha", _pct),
    ]
    return pd.DataFrame(rows, columns=["metric"] + labels)


def plot_equity(curves: dict[str, pd.Series], path: Path, log_scale: bool = True) -> None:
    """Save an equity-curve chart with all curves on the same axes."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for (label, equity), color in zip(curves.items(), _COLORS):
        ax.plot(equity.index, equity.values, label=label, color=color, linewidth=1.3)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("Equity curves (same capital, same window)")
    ax.set_ylabel("Equity ($, log scale)" if log_scale else "Equity ($)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_drawdown(curves: dict[str, pd.Series], path: Path) -> None:
    """Save a drawdown-over-time chart with all curves on the same axes."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for (label, equity), color in zip(curves.items(), _COLORS):
        drawdown = (equity / equity.cummax() - 1.0) * 100
        ax.plot(drawdown.index, drawdown.values, label=label, color=color, linewidth=1.1)
    ax.set_title("Drawdown over time")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def honest_summary(metrics_by: dict[str, dict], active: str, overlay: str,
                   buy_hold: str, blend: str) -> str:
    """Plain summary: overlay effect, and whether either variant beats the passives."""
    m = metrics_by

    overlay_line = (
        f"Overlay effect: deploying idle cash into SPY moved CAGR "
        f"{_pct(m[active]['cagr'])} -> {_pct(m[overlay]['cagr'])} and max drawdown "
        f"{_pct(-m[active]['max_drawdown'])} -> {_pct(-m[overlay]['max_drawdown'])}.")

    # Does the better active variant beat the passives on any metric?
    best = overlay if (m[overlay]["cagr"] or -1) > (m[active]["cagr"] or -1) else active
    beats = []
    for passive in (buy_hold, blend):
        wins = [name for name, key, hb in [("CAGR", "cagr", True), ("Sharpe", "sharpe", True),
                                           ("Sortino", "sortino", True), ("Calmar", "calmar", True),
                                           ("max drawdown", "max_drawdown", False)]
                if (m[best][key] is not None and m[passive][key] is not None
                    and ((m[best][key] > m[passive][key]) == hb))]
        verdict = ", ".join(wins) if wins else "nothing"
        beats.append(f"vs {passive}, the best active variant ({best}) wins on: {verdict}.")

    return overlay_line + " " + " ".join(beats)


def generate_report(curves: dict[str, pd.Series], market_key: str, *, active: str,
                    overlay: str, buy_hold: str, blend: str,
                    rf: float = risk_metrics.DEFAULT_RISK_FREE) -> dict:
    """Compute metrics for every curve, print/save the table, charts, and summary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = list(curves)
    metrics_by = {lab: risk_metrics.compute_metrics(eq, rf) for lab, eq in curves.items()}
    relative_by = {lab: risk_metrics.relative_metrics(eq, curves[market_key], rf)
                   for lab, eq in curves.items()}

    table = build_table(metrics_by, relative_by, labels)
    first = curves[active]
    print(f"\n=== Four-way comparison ({first.index[0].date()} -> {first.index[-1].date()}) ===")
    print(table.to_string(index=False))

    table.to_csv(OUTPUT_DIR / "report_comparison.csv", index=False)
    plot_equity(curves, OUTPUT_DIR / "report_equity.png")
    plot_drawdown(curves, OUTPUT_DIR / "report_drawdown.png")
    tearsheet = _maybe_quantstats(curves[active], curves[market_key], OUTPUT_DIR / "report_tearsheet.html")

    print("\n=== Honest summary ===")
    print("  " + honest_summary(metrics_by, active, overlay, buy_hold, blend))

    artifacts = ["report_comparison.csv", "report_equity.png", "report_drawdown.png"]
    if tearsheet:
        artifacts.append("report_tearsheet.html")
    print(f"\nSaved to {OUTPUT_DIR}: " + ", ".join(artifacts))
    return {"metrics": metrics_by, "relative": relative_by}


def _maybe_quantstats(strategy: pd.Series, benchmark: pd.Series, path: Path) -> bool:
    """Generate a quantstats HTML tearsheet if available; never the source of truth."""
    try:
        import quantstats as qs
    except Exception:
        log.info("quantstats not installed; skipping HTML tearsheet (hand metrics stand).")
        return False
    try:
        qs.reports.html(strategy.pct_change().dropna(), benchmark=benchmark.pct_change().dropna(),
                        output=str(path), title="Strategy vs S&P 500")
        return True
    except Exception as error:  # noqa: BLE001 - optional artifact only
        log.warning("quantstats tearsheet failed (non-fatal): %s", error)
        return False
