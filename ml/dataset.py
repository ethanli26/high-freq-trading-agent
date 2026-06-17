"""Build the ML training set: one row per historical signal fire.

Each row is a breakout or pullback signal that fired and passed the sector gate,
labeled by whether the resulting trade (under the trend-riding exit rules) won.

THE LEAKAGE RULE, stated once and enforced everywhere below:
  * FEATURES are computed from completed bars on or before the signal day only.
    The signal day ``d`` is the bar whose close produced the signal; entry would be
    at ``d+1``'s open. Every feature reads the symbol's own causal indicator series
    indexed at ``d`` — never ``d+1`` or later.
  * The LABEL is allowed to use the future: it simulates the trade forward from
    ``d+1`` under the exit rules and records win (1) / loss (0). Labels may look
    ahead; features may not. ``main`` runs an explicit causality check.

Read-only research: no IBKR, no orders, no live-trading changes.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this file directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.data import BENCHMARK, build_sector_map, load_universe
from backtest.engine import COMMISSION_PER_SHARE, SLIPPAGE_PCT, atr_series, momentum_series
from backtest.regime import compute_regime
from config import ATR_MULTIPLE, ATR_PERIOD, CHANDELIER_ATR_MULT, TREND_EXIT_MA
from screener.momentum import LOOKBACK_3M, LOOKBACK_6M
from screener.sectors import SECTOR_ETFS
from signals.breakout import BreakoutStrategy
from signals.pullback import PullbackStrategy

log = logging.getLogger(__name__)

FEATURE_MA_FAST = 50
FEATURE_MA_SLOW = 200
RSI_PERIOD = 14
TOP_SECTORS = 3  # sector gate: only signals in the top-N sectors are kept

# The model features, in a fixed order. Everything else (date/symbol/strategy/label)
# is metadata, not fed to the model.
FEATURE_COLUMNS = [
    "dist_ma50",     # close / 50d MA - 1   (trend extension, fast)
    "dist_ma200",    # close / 200d MA - 1  (trend extension, slow)
    "atr_pct",       # ATR / close          (recent volatility)
    "mom_3m",        # 3-month total return
    "mom_6m",        # 6-month total return
    "sector_rank",   # 1..TOP_SECTORS, 1 = strongest sector that day
    "rsi_14",        # Wilder RSI (overbought/oversold)
    "is_pullback",   # 1 if pullback fired, 0 if breakout
    "regime_bear",   # 1 if the signal day's regime is bear (bull = baseline)
    "regime_crash",  # 1 if the signal day's regime is crash
]

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATASET_PATH = OUTPUT_DIR / "dataset.parquet"


def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI as a causal series (each value uses past bars only)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _sector_rank_frame(bars: dict[str, pd.DataFrame], etfs: list[str],
                       master: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-date sector momentum rank (1 = strongest) across the sector ETFs.

    Uses the same momentum measure as the live gate; computed causally per bar.
    """
    momentum = {etf: momentum_series(bars[etf]["Close"]).reindex(master) for etf in etfs}
    momentum_df = pd.DataFrame(momentum)
    return momentum_df.rank(axis=1, ascending=False, method="min")  # NaN momentum -> NaN rank


def _symbol_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Precompute causal indicator arrays for one symbol (aligned to its own bars)."""
    close, high, low = frame["Close"], frame["High"], frame["Low"]
    return {
        "open": frame["Open"].to_numpy(dtype=float),
        "high": high.to_numpy(dtype=float),
        "low": low.to_numpy(dtype=float),
        "close": close.to_numpy(dtype=float),
        # atr_series drops its first row; reindex to restore alignment (NaN at bar 0)
        # so arr["atr"][d] is the ATR AT bar d, never bar d+1 (no look-ahead).
        "atr": atr_series(high, low, close, ATR_PERIOD).reindex(frame.index).to_numpy(dtype=float),
        "ma50": close.rolling(FEATURE_MA_FAST).mean().to_numpy(dtype=float),
        "ma200": close.rolling(FEATURE_MA_SLOW).mean().to_numpy(dtype=float),
        "ma_trend": close.rolling(TREND_EXIT_MA).mean().to_numpy(dtype=float),  # exit MA
        "mom3": (close / close.shift(LOOKBACK_3M) - 1.0).to_numpy(dtype=float),
        "mom6": (close / close.shift(LOOKBACK_6M) - 1.0).to_numpy(dtype=float),
        "rsi": _rsi(close).to_numpy(dtype=float),
    }


def simulate_trade(arr: dict[str, np.ndarray], d: int) -> dict | None:
    """Simulate the trade forward from the signal at ``d`` under the exit rules.

    LABEL ONLY — this is the one place that reads bars after ``d``. It reproduces the
    engine's trend-riding exit exactly (initial ATR stop, 2*ATR trail pre-profit,
    chandelier + trend-MA-break once up >= 1 ATR). Win/loss is independent of share
    count, so per-share P&L net of round-trip slippage and commission decides it.

    Returns ``{label, exit_i, exit_reason, entry_fill, exit_fill}`` or ``None``.
    """
    n = len(arr["close"])
    entry_i = d + 1
    if entry_i >= n:
        return None
    entry_fill = arr["open"][entry_i] * (1.0 + SLIPPAGE_PCT)
    atr_entry = arr["atr"][d]
    if not np.isfinite(entry_fill) or not np.isfinite(atr_entry) or atr_entry <= 0:
        return None

    initial_stop = entry_fill - ATR_MULTIPLE * atr_entry
    current_stop = initial_stop
    highest_close = highest_high = -np.inf
    trend_mode = False
    exit_fill = exit_i = exit_reason = None

    for j in range(entry_i + 1, n):  # exits begin the day after entry, as in the engine
        pj = j - 1
        close_pj, high_pj = arr["close"][pj], arr["high"][pj]
        if np.isfinite(close_pj):
            highest_close = max(highest_close, close_pj)
        if np.isfinite(high_pj):
            highest_high = max(highest_high, high_pj)

        if not trend_mode and np.isfinite(close_pj) and close_pj - entry_fill >= atr_entry:
            trend_mode = True

        if trend_mode:
            current_stop = max(current_stop, highest_high - CHANDELIER_ATR_MULT * atr_entry)
        elif np.isfinite(close_pj):
            current_stop = max(current_stop, highest_close - ATR_MULTIPLE * atr_entry)

        open_j, low_j = arr["open"][j], arr["low"][j]
        if trend_mode:  # trend-MA-break exit at the open
            ma_pj = arr["ma_trend"][pj]
            if np.isfinite(ma_pj) and np.isfinite(close_pj) and close_pj < ma_pj and np.isfinite(open_j):
                exit_fill, exit_i, exit_reason = open_j * (1.0 - SLIPPAGE_PCT), j, "ma_break"
                break
        if not np.isfinite(low_j):
            continue
        if low_j <= current_stop:
            raw = open_j if (np.isfinite(open_j) and open_j < current_stop) else current_stop
            exit_fill, exit_i = raw * (1.0 - SLIPPAGE_PCT), j
            exit_reason = "chandelier_stop" if trend_mode else (
                "trailing_stop" if current_stop > initial_stop + 1e-9 else "stop")
            break

    if exit_fill is None:  # never exited: liquidate at the final close
        exit_fill, exit_i, exit_reason = arr["close"][-1] * (1.0 - SLIPPAGE_PCT), n - 1, "end_of_backtest"

    per_share_pnl = exit_fill - entry_fill - 2.0 * COMMISSION_PER_SHARE
    return {"label": 1 if per_share_pnl > 0 else 0, "exit_i": exit_i,
            "exit_reason": exit_reason, "entry_fill": entry_fill, "exit_fill": exit_fill}


def _features_at(arr: dict[str, np.ndarray], d: int, sector_rank: float,
                 is_pullback: float, regime_label: str) -> dict | None:
    """Build the feature dict for a signal at ``d`` from causal arrays only.

    Returns ``None`` if any continuous feature is NaN (insufficient history).
    """
    close_d, ma50_d, ma200_d = arr["close"][d], arr["ma50"][d], arr["ma200"][d]
    atr_d, mom3_d, mom6_d, rsi_d = arr["atr"][d], arr["mom3"][d], arr["mom6"][d], arr["rsi"][d]

    continuous = [close_d, ma50_d, ma200_d, atr_d, mom3_d, mom6_d, rsi_d]
    if any(not np.isfinite(v) for v in continuous) or close_d <= 0 or ma50_d <= 0 or ma200_d <= 0:
        return None

    return {
        "dist_ma50": close_d / ma50_d - 1.0,
        "dist_ma200": close_d / ma200_d - 1.0,
        "atr_pct": atr_d / close_d,
        "mom_3m": mom3_d,
        "mom_6m": mom6_d,
        "sector_rank": sector_rank,
        "rsi_14": rsi_d,
        "is_pullback": is_pullback,
        "regime_bear": 1.0 if regime_label == "bear" else 0.0,
        "regime_crash": 1.0 if regime_label == "crash" else 0.0,
    }


def build_dataset(years: int = 15, refresh: bool = False) -> pd.DataFrame:
    """Build the labeled signal dataset across the universe.

    Returns a DataFrame: ``date, symbol, strategy, <features>, label``.
    """
    bars, skipped = load_universe(years=years, refresh=refresh)
    if BENCHMARK not in bars:
        raise RuntimeError(f"Benchmark {BENCHMARK} missing; cannot tag regime.")

    master = bars[BENCHMARK].index
    regime = compute_regime(bars[BENCHMARK]["Close"])
    sector_map = build_sector_map()
    etfs = [e for e in SECTOR_ETFS if e in bars]
    sector_ranks = _sector_rank_frame(bars, etfs, master)
    strategies = [BreakoutStrategy(), PullbackStrategy()]

    rows: list[dict] = []
    for symbol, frame in bars.items():
        sector = sector_map.get(symbol)
        if sector is None:  # ETFs / benchmark are not tradable names
            continue
        arr = _symbol_arrays(frame)
        dates = frame.index
        for strat in strategies:
            fired = strat.signal_series(frame).fillna(False).to_numpy(dtype=bool)
            for d in np.nonzero(fired)[0]:
                d = int(d)
                if d + 1 >= len(frame):  # need a next-day open to enter
                    continue
                date = dates[d]
                # Sector gate, evaluated at d (rank uses momentum through d only).
                rank = sector_ranks.at[date, sector] if (date in sector_ranks.index
                                                         and sector in sector_ranks.columns) else np.nan
                if not np.isfinite(rank) or rank > TOP_SECTORS:
                    continue
                regime_label = regime.get(date)
                features = _features_at(arr, d, float(rank),
                                        1.0 if strat.name == "pullback" else 0.0, regime_label)
                if features is None:
                    continue
                sim = simulate_trade(arr, d)  # LABEL uses future bars (allowed)
                if sim is None:
                    continue
                rows.append({"date": date, "symbol": symbol, "strategy": strat.name,
                             **features, "label": sim["label"]})

    dataset = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    log.info("Built dataset: %d signals, win rate %.3f, %s -> %s.",
             len(dataset), dataset["label"].mean() if len(dataset) else float("nan"),
             dataset["date"].min() if len(dataset) else "n/a",
             dataset["date"].max() if len(dataset) else "n/a")
    return dataset


def _verify_causality(bars: dict[str, pd.DataFrame], dataset: pd.DataFrame, n_samples: int = 200) -> None:
    """Assert features at ``d`` do not change when bars after ``d`` are removed.

    Recomputes the indicator arrays on the truncated history ``frame.iloc[:d+1]`` and
    checks the feature values match — a direct, empirical no-look-ahead guarantee.
    """
    sample = dataset.sample(min(n_samples, len(dataset)), random_state=0)
    checked = 0
    for _, row in sample.iterrows():
        frame = bars[row["symbol"]]
        d = frame.index.get_loc(row["date"])
        truncated = _symbol_arrays(frame.iloc[: d + 1])  # nothing after the signal day
        td = d  # last index of the truncated frame
        for name, col in (("dist_ma50", truncated["close"][td] / truncated["ma50"][td] - 1.0),
                          ("atr_pct", truncated["atr"][td] / truncated["close"][td]),
                          ("rsi_14", truncated["rsi"][td]),
                          ("mom_6m", truncated["mom6"][td])):
            if abs(col - row[name]) > 1e-9:
                raise AssertionError(f"LEAKAGE: {name} for {row['symbol']} @ {row['date']} "
                                     f"changed when future bars were removed.")
        checked += 1
    log.info("Causality check passed on %d sampled signals (features use no future data).", checked)


def main() -> int:
    """Build the dataset, run the causality check, and cache it."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    bars, _ = load_universe()
    dataset = build_dataset()
    if dataset.empty:
        log.error("No signals found; nothing to save.")
        return 1

    _verify_causality(bars, dataset)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_PATH)
    print(f"\nDataset: {len(dataset)} signals | win rate {dataset['label'].mean():.3f} | "
          f"{dataset['date'].min().date()} -> {dataset['date'].max().date()}")
    print("By strategy:")
    print(dataset.groupby("strategy")["label"].agg(["count", "mean"]).to_string())
    print(f"Saved to {DATASET_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
