"""Daily OHLC bar fetching via yfinance, for signals and risk.

The screener only needs closing prices, but the entry rule and ATR stop need full
OHLC bars, so this helper returns Open/High/Low/Close. Incomplete bars (any NaN
field, e.g. a partial current-day bar before the close) are dropped so downstream
logic only ever sees completed sessions.
"""

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# Enough calendar history (~4 months) to cover the breakout lookback and ATR
# period with a comfortable buffer for weekends and holidays.
DEFAULT_LOOKBACK_DAYS = 120

OHLC_COLUMNS = ["Open", "High", "Low", "Close"]


def fetch_daily_bars(symbol: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame | None:
    """Download recent daily OHLC bars for one symbol.

    Returns a DataFrame indexed by date with Open/High/Low/Close columns and no
    NaN rows, or ``None`` if the download fails or comes back empty.
    """
    end = date.today() + timedelta(days=1)  # end is exclusive; include today
    start = end - timedelta(days=lookback_days)

    try:
        bars = yf.Ticker(symbol).history(
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=True,
        )
    except Exception as error:  # noqa: BLE001 - any yfinance failure is non-fatal
        log.warning("Download failed for %s: %s", symbol, error)
        return None

    if bars is None or bars.empty or not set(OHLC_COLUMNS).issubset(bars.columns):
        log.warning("No OHLC data returned for %s; skipping.", symbol)
        return None

    # Keep only completed sessions; a partial current-day bar shows up as NaN.
    bars = bars[OHLC_COLUMNS].dropna()
    if bars.empty:
        log.warning("Only incomplete bars for %s; skipping.", symbol)
        return None

    return bars
