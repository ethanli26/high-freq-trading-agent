"""Entry signals.

Phase 2 ships one simple, transparent rule: a Donchian-style breakout. The latest
completed close must exceed the highest close of the prior ``BREAKOUT_LOOKBACK``
sessions. Only completed sessions are used; any partial current-day bar is ignored.
"""

import logging

import pandas as pd

import config

log = logging.getLogger(__name__)


def latest_close(bars: pd.DataFrame) -> float | None:
    """Return the most recent completed (non-NaN) close, or ``None``."""
    if bars is None or "Close" not in getattr(bars, "columns", []):
        return None
    close = bars["Close"].dropna()
    if close.empty:
        return None
    return float(close.iloc[-1])


def breakout_signal(bars: pd.DataFrame) -> bool:
    """True if the latest close breaks above the prior ``BREAKOUT_LOOKBACK`` highs.

    Today's (latest) bar is excluded from the lookback window: the latest close is
    compared against the highest close of the ``BREAKOUT_LOOKBACK`` sessions before
    it. Returns ``False`` if there is not enough history.
    """
    if bars is None or "Close" not in getattr(bars, "columns", []):
        return False

    close = bars["Close"].dropna()
    lookback = config.BREAKOUT_LOOKBACK
    if len(close) < lookback + 1:
        return False

    latest = close.iloc[-1]
    prior_window = close.iloc[-(lookback + 1):-1]  # the N bars before the latest
    return bool(latest > prior_window.max())
