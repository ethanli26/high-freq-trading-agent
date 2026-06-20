"""Crypto hype-momentum strategy — breakout confirmed by a volume surge.

THESIS. In small-coin crypto, sustained moves are driven by attention ("hype"): a coin
breaks out on a burst of volume as new buyers pile in. We try to ride that, long only,
and lean on the engine's ATR trailing stop for a positive-skew exit (cut losers fast,
let winners run). This is a momentum/breakout cousin of the equity ``BreakoutStrategy``,
implemented on the SAME look-ahead-safe Strategy interface.

ENTRY (both conditions, on the latest COMPLETED bar):
  * price breaks above its prior ``lookback``-day high (Donchian breakout), AND
  * volume is well above its own trailing average (``volume_mult`` x the trailing
    ``vol_lookback``-day average) — the hype/attention confirmation.
Direction is UP only.

EXIT is handled by the engine's ATR trailing stop (run with ``trend_exit=True`` for the
chandelier/positive-skow trail). Because the stop is a multiple of ATR, it AUTO-WIDENS
for high-volatility coins — the crypto "retune" comes for free from ATR scaling rather
than from new constants.

LOOK-AHEAD SAFETY: both the prior high and the trailing volume average are computed
with ``.shift(1)``, so the signal on bar ``t`` uses only bars ``<= t-1`` for its
thresholds and the breakout compares them to bar ``t``'s own completed close/volume.
The backtest fills the resulting entry at the NEXT bar's open. ``signal_series`` equals
``generate_signal`` bar by bar (verified in tests).
"""

import pandas as pd

from strategies.base import Strategy
from strategies.registry import register

DEFAULT_LOOKBACK = 20        # Donchian breakout window (days)
DEFAULT_VOLUME_MULT = 2.0    # require volume > this x its trailing average
DEFAULT_VOL_LOOKBACK = 20    # trailing average-volume window (days)
# How far above the breakout level counts as "full strength" (10% over the prior high).
HYPE_STRENGTH_SCALE = 0.10


@register
class CryptoHypeStrategy(Strategy):
    """Long-only breakout gated by a volume surge (attention/hype confirmation)."""

    name = "crypto_hype"
    category = "price"
    requires = ("price_bars",)

    def __init__(self, lookback: int = DEFAULT_LOOKBACK,
                 volume_mult: float = DEFAULT_VOLUME_MULT,
                 vol_lookback: int = DEFAULT_VOL_LOOKBACK):
        self.lookback = lookback
        self.volume_mult = volume_mult
        self.vol_lookback = vol_lookback
        self.params = {"lookback": lookback, "volume_mult": volume_mult,
                       "vol_lookback": vol_lookback}

    def _components(self, bars: pd.DataFrame):
        """Prior high, breakout flag, and volume-surge flag — all from completed bars."""
        close, volume = bars["Close"], bars["Volume"]
        prior_high = close.rolling(self.lookback).max().shift(1)   # excludes today
        avg_volume = volume.rolling(self.vol_lookback).mean().shift(1)  # excludes today
        broke_out = close > prior_high
        volume_surge = volume > self.volume_mult * avg_volume
        return prior_high, broke_out, volume_surge

    def generate_signal(self, bars: pd.DataFrame, symbol: str | None = None) -> bool:
        """Canonical rule on the latest completed bar: breakout AND volume surge."""
        if bars is None or len(bars) < max(self.lookback, self.vol_lookback) + 1:
            return False
        if not {"Close", "Volume"}.issubset(bars.columns):
            return False
        _, broke_out, volume_surge = self._components(bars)
        return bool(broke_out.iloc[-1] and volume_surge.iloc[-1])

    def signal_series(self, bars: pd.DataFrame, symbol: str | None = None) -> pd.Series:
        """Vectorized breakout-and-surge flag, equal to generate_signal at each bar."""
        if not {"Close", "Volume"}.issubset(bars.columns):
            return pd.Series(False, index=bars.index)
        _, broke_out, volume_surge = self._components(bars)
        return (broke_out & volume_surge).fillna(False)

    def strength_series(self, bars: pd.DataFrame) -> pd.Series:
        """Strength = how far the close exceeds the breakout level, scaled to 0..1.

        ``(close / prior_high - 1) / SCALE`` clipped to 0..1; uses completed bars only
        (prior_high excludes today), so no look-ahead.
        """
        if not {"Close", "Volume"}.issubset(bars.columns):
            return pd.Series(0.0, index=bars.index)
        prior_high, _, _ = self._components(bars)
        raw = (bars["Close"] / prior_high - 1.0) / HYPE_STRENGTH_SCALE
        return raw.clip(lower=0.0, upper=1.0).fillna(0.0)
