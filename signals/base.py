"""Pluggable strategy interface for entry signals.

A strategy is a small object with a ``name`` and two equivalent ways to ask
"is there an entry signal on the latest completed bar?":

  * ``generate_signal(bars) -> bool`` — the canonical, look-ahead-safe rule
    evaluated on a frame of completed bars. This is the source of truth and is what
    the live decision path uses.
  * ``signal_series(bars) -> pd.Series`` — the same rule vectorized to a per-bar
    boolean series, which the backtest engine consumes for speed. It MUST equal
    ``generate_signal`` evaluated bar by bar (the backtest verifies this).

Both use only completed bars — no look-ahead. The optional ``symbol`` argument lets
strategies that need per-symbol side data (e.g. earnings dates) look it up; purely
price-based strategies ignore it. New strategies subclass this and implement the
two methods.
"""

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base class for a pluggable entry strategy."""

    name: str

    @abstractmethod
    def generate_signal(self, bars: pd.DataFrame, symbol: str | None = None) -> bool:
        """Return True if there is an entry signal on the latest completed bar."""

    @abstractmethod
    def signal_series(self, bars: pd.DataFrame, symbol: str | None = None) -> pd.Series:
        """Per-bar boolean signal; must equal ``generate_signal`` at each bar."""

    def strength_series(self, bars: pd.DataFrame) -> pd.Series:
        """Per-bar trigger strength in 0..1 (1 = strongest), for conviction sizing.

        Default is a neutral 0.5; strategies override to express how strong each
        setup is. Uses completed bars only — no look-ahead.
        """
        return pd.Series(0.5, index=bars.index)
