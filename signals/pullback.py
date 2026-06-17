"""Moving-average pullback strategy.

Buy strength on a dip: a name in an uptrend that recently pulled back near its
moving average and is now bouncing. All three conditions use completed bars only.

The rule (on completed bars, with MA = ``PULLBACK_MA``-day simple moving average):
  1. Uptrend     — the latest completed close is above the MA.
  2. Pullback    — within the last ``PULLBACK_BOUNCE_LOOKBACK`` sessions, a low came
                   within ``PULLBACK_TOUCH_PCT`` of the MA (price dipped toward it).
  3. Bounce      — the latest close is up versus the prior close AND up versus the
                   close ``PULLBACK_BOUNCE_LOOKBACK`` sessions ago (net up move).
"""

import pandas as pd

import config
from signals.base import Strategy

# Net up-move over the bounce window that counts as "full strength" (3%).
PULLBACK_STRENGTH_SCALE = 0.03


class PullbackStrategy(Strategy):
    """Enter an uptrend after a pullback to the MA confirms a bounce."""

    name = "pullback"

    def generate_signal(self, bars: pd.DataFrame, symbol: str | None = None) -> bool:
        """Canonical rule on completed bars.

        LOOK-AHEAD SAFETY: only the passed-in bars are used, and the latest
        completed bar is ``bars.iloc[-1]``. No future bar is ever referenced; the
        touch and bounce windows look backward from the latest bar.
        """
        if bars is None or not {"Close", "Low"}.issubset(getattr(bars, "columns", [])):
            return False

        close = bars["Close"].dropna()
        low = bars["Low"].reindex(close.index)
        ma_period = config.PULLBACK_MA
        lookback = config.PULLBACK_BOUNCE_LOOKBACK
        if len(close) < max(ma_period, lookback + 1):
            return False

        moving_avg = close.rolling(ma_period).mean()
        if pd.isna(moving_avg.iloc[-1]):
            return False

        # 1) Uptrend: latest completed close above the MA.
        uptrend = close.iloc[-1] > moving_avg.iloc[-1]
        # 2) Recent pullback: a low within TOUCH_PCT of the MA over the last few bars.
        touch_window_low = low.iloc[-lookback:]
        touch_window_ma = moving_avg.iloc[-lookback:]
        touched = bool((touch_window_low <= touch_window_ma * (1 + config.PULLBACK_TOUCH_PCT)).any())
        # 3) Bounce: up close versus prior bar, and net up over the lookback span.
        bounce = close.iloc[-1] > close.iloc[-2] and close.iloc[-1] > close.iloc[-(lookback + 1)]

        return bool(uptrend and touched and bounce)

    def signal_series(self, bars: pd.DataFrame, symbol: str | None = None) -> pd.Series:
        """Vectorized pullback signal; equals generate_signal at each bar."""
        close = bars["Close"]
        low = bars["Low"]
        ma_period = config.PULLBACK_MA
        lookback = config.PULLBACK_BOUNCE_LOOKBACK

        moving_avg = close.rolling(ma_period).mean()
        uptrend = close > moving_avg
        # Any low within TOUCH_PCT of the MA over the last `lookback` bars (incl. now).
        touch_flag = low <= moving_avg * (1 + config.PULLBACK_TOUCH_PCT)
        touched = touch_flag.rolling(lookback).sum() >= 1
        # Up versus prior bar, and up versus `lookback` bars ago.
        bounce = (close > close.shift(1)) & (close > close.shift(lookback))

        return (uptrend & touched & bounce).fillna(False)

    def strength_series(self, bars: pd.DataFrame) -> pd.Series:
        """Strength = size of the bounce over the lookback, scaled to 0..1.

        ``(close / close.shift(lookback) - 1) / SCALE`` clipped to 0..1; uses
        completed bars only, so no look-ahead.
        """
        close = bars["Close"]
        raw = (close / close.shift(config.PULLBACK_BOUNCE_LOOKBACK) - 1.0) / PULLBACK_STRENGTH_SCALE
        return raw.clip(lower=0.0, upper=1.0)
