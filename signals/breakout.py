"""Breakout strategy (Donchian-style): close above the prior N-day high.

This wraps the existing signals/entry.breakout_signal as a pluggable Strategy so it
can run alongside other strategies in the backtest. Behavior is identical to
signals/entry.py.
"""

import pandas as pd

import config
from signals.base import Strategy
from signals.entry import breakout_signal

# How far above the breakout level counts as "full strength" (5% over the prior high).
BREAKOUT_STRENGTH_SCALE = 0.05


class BreakoutStrategy(Strategy):
    """Enter when the latest close breaks above the prior ``BREAKOUT_LOOKBACK`` highs."""

    name = "breakout"

    def generate_signal(self, bars: pd.DataFrame, symbol: str | None = None) -> bool:
        """Canonical rule — identical to signals/entry.breakout_signal."""
        return breakout_signal(bars)

    def signal_series(self, bars: pd.DataFrame, symbol: str | None = None) -> pd.Series:
        """Vectorized breakout flag, equal to generate_signal at each bar.

        ``prior_high[t]`` is the max close over ``[t-lookback, t-1]`` (today
        excluded), so the signal uses only completed bars — no look-ahead.
        """
        close = bars["Close"]
        prior_high = close.rolling(config.BREAKOUT_LOOKBACK).max().shift(1)
        return close > prior_high

    def strength_series(self, bars: pd.DataFrame) -> pd.Series:
        """Strength = how far the close exceeds the breakout level, scaled to 0..1.

        ``(close / prior_high - 1) / SCALE`` clipped to 0..1; uses completed bars
        only (prior_high excludes today), so no look-ahead.
        """
        close = bars["Close"]
        prior_high = close.rolling(config.BREAKOUT_LOOKBACK).max().shift(1)
        raw = (close / prior_high - 1.0) / BREAKOUT_STRENGTH_SCALE
        return raw.clip(lower=0.0, upper=1.0)
