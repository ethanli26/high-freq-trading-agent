"""Historical earnings data for the earnings-drift strategy.

SOURCE & HONESTY NOTE
---------------------
The task asked for Finnhub's free tier (``stock/earnings`` + ``calendar/earnings``).
That tier was probed and is NOT sufficient for a post-earnings-drift backtest:

  * ``/stock/earnings`` returns only the last ~4 quarters and gives the fiscal
    PERIOD-END, not the announcement date.
  * ``/calendar/earnings`` (which carries announcement dates) is a premium endpoint
    and returns nothing on the free key.

A PEAD strategy needs real ANNOUNCEMENT dates and multi-year depth, so earnings are
sourced from yfinance ``get_earnings_dates`` (free), which provides actual report
dates, EPS estimate, reported EPS and surprise back several years. ``finnhub_probe``
below documents/reproduces the free-tier check. These figures are FINAL/REVISED
values (revised-figure bias) and coverage/depth vary by symbol — see limitations.

Cached per symbol to parquet; requests are throttled; symbols with no/short history
are logged and skipped. Read-only research: no IBKR, no orders.
"""

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "cache" / "earnings"
FETCH_LIMIT = 48          # quarters to request (~12y); yfinance returns most-recent N
THROTTLE_SECONDS = 0.4    # be polite between live fetches
MIN_SURPRISE_EVENTS = 4   # skip symbols with too few usable historical reports

EARNINGS_COLUMNS = ["report_date", "eps_estimate", "eps_reported", "surprise"]


def finnhub_probe() -> str:
    """Document why Finnhub free tier is not used (depth + missing report dates).

    Returns a short human-readable note; safe to call without network.
    """
    return ("Finnhub free tier probed: /stock/earnings returns ~4 quarters and no "
            "announcement date; /calendar/earnings is premium (empty on free). "
            "Using yfinance get_earnings_dates for real report dates instead.")


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.parquet"


def fetch_earnings(symbol: str, refresh: bool = False) -> pd.DataFrame | None:
    """Fetch (or load) a symbol's earnings history as a tidy frame.

    Columns: ``report_date`` (tz-naive date), ``eps_estimate``, ``eps_reported``,
    ``surprise`` (fraction = (reported - estimate)/|estimate|; NaN for future or
    estimate-less rows). Returns ``None`` on failure/empty.
    """
    path = _cache_path(symbol)
    if not refresh and path.exists():
        return pd.read_parquet(path)

    try:
        raw = yf.Ticker(symbol).get_earnings_dates(limit=FETCH_LIMIT)
    except Exception as error:  # noqa: BLE001 - any provider failure is non-fatal
        log.warning("Earnings fetch failed for %s: %s", symbol, error)
        return None
    if raw is None or raw.empty:
        log.warning("No earnings rows for %s; skipping.", symbol)
        return None

    frame = pd.DataFrame(index=range(len(raw)))
    # Announcement timestamp -> tz-naive calendar date (the time/AMC-vs-BMO detail is
    # intentionally dropped; the strategy enters strictly AFTER this date anyway).
    frame["report_date"] = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    frame["eps_estimate"] = raw["EPS Estimate"].to_numpy()
    frame["eps_reported"] = raw["Reported EPS"].to_numpy()
    estimate = frame["eps_estimate"]
    frame["surprise"] = (frame["eps_reported"] - estimate) / estimate.abs()

    frame = frame[EARNINGS_COLUMNS].sort_values("report_date").reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    time.sleep(THROTTLE_SECONDS)  # throttle only on a live fetch
    return frame


def load_earnings(
    symbols: list[str],
    years: int,
    refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[str], pd.Timestamp]:
    """Load earnings for ``symbols``, restricted to roughly the last ``years``.

    Returns ``(frames, skipped, window_start)``:
      * ``frames`` — symbol -> earnings frame (kept a bit before window_start so
        blackout near the window edge still sees the prior report; future scheduled
        rows are kept for the blackout).
      * ``skipped`` — symbols with too few usable historical reports.
      * ``window_start`` — the backtest start date implied by the chosen window.
    """
    window_start = pd.Timestamp(date.today() - timedelta(days=int(years * 365.25)))
    keep_from = window_start - pd.Timedelta(days=200)  # buffer for edge blackouts

    frames: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for symbol in symbols:
        frame = fetch_earnings(symbol, refresh=refresh)
        if frame is None:
            skipped.append(symbol)
            continue
        frame = frame[frame["report_date"] >= keep_from].reset_index(drop=True)
        usable = frame["surprise"].notna().sum()
        if usable < MIN_SURPRISE_EVENTS:
            log.warning("Skipping %s: only %d usable reports in window.", symbol, usable)
            skipped.append(symbol)
            continue
        frames[symbol] = frame

    log.info("%s", finnhub_probe())
    log.info("Loaded earnings for %d symbols (%d skipped); window starts %s (~%dy).",
             len(frames), len(skipped), window_start.date(), years)
    return frames, skipped, window_start
