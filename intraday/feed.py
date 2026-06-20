"""Bar feeds: where minute bars come from.

``BarFeed`` is the seam. ``ReplayFeed`` replays historical/synthetic minute bars in
time order (deterministic — the basis for testing and for this architecture demo).
``LiveFeed`` is a documented stub showing exactly where an IBKR real-time
subscription drops in later, without implementing the live connection now.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
import pandas as pd

from intraday.events import BarEvent

log = logging.getLogger(__name__)

_BAR_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


class BarFeed(ABC):
    """Yields ``BarEvent``s in non-decreasing timestamp order."""

    @abstractmethod
    def __iter__(self) -> Iterator[BarEvent]:
        ...


class ReplayFeed(BarFeed):
    """Replay a fixed set of minute bars in time order (deterministic)."""

    def __init__(self, bars: pd.DataFrame):
        missing = set(_BAR_COLUMNS) - set(bars.columns)
        if missing:
            raise ValueError(f"ReplayFeed bars missing columns: {sorted(missing)}")
        # Emit in time order (ties broken by symbol) — simulates the passage of time.
        self._bars = bars.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    @classmethod
    def from_csv(cls, path: str) -> "ReplayFeed":
        return cls(pd.read_csv(path, parse_dates=["timestamp"]))

    @classmethod
    def from_parquet(cls, path: str) -> "ReplayFeed":
        return cls(pd.read_parquet(path))

    def __iter__(self) -> Iterator[BarEvent]:
        for row in self._bars.itertuples(index=False):
            yield BarEvent(symbol=row.symbol, timestamp=row.timestamp,
                           open=float(row.open), high=float(row.high), low=float(row.low),
                           close=float(row.close), volume=float(row.volume))


class LiveFeed(BarFeed):
    """STUB for a live IBKR minute-bar feed — not implemented (no live connection now).

    Drop-in plan (kept here so it's a one-class change later):
      * Subscribe via ``ib.reqRealTimeBars(contract, 5, "TRADES", useRTH=True)`` for
        each symbol (5-second bars are IBKR's finest real-time stream).
      * Aggregate consecutive 5s bars into 1-minute ``BarEvent``s (OHLC = first open /
        running max high / running min low / last close; volume summed).
      * Push each completed minute bar onto a thread-safe queue; ``__iter__`` blocks on
        the queue and yields ``BarEvent``s as they arrive — same interface as
        ``ReplayFeed``, so the engine is unchanged.
      * Live order placement would route through the EXISTING paper-safety guards
        (DU-account check + DRY_RUN in decision/autonomy.py); this engine only
        simulates fills and never places real orders.
    """

    def __init__(self, broker, symbols: list[str], resolution: str = "1min"):
        self.broker = broker
        self.symbols = symbols
        self.resolution = resolution

    def __iter__(self) -> Iterator[BarEvent]:
        raise NotImplementedError(
            "LiveFeed is a stub. Wire ib.reqRealTimeBars -> 1-min aggregation -> queue "
            "here; the engine consumes it exactly like ReplayFeed.")


def make_synthetic_bars(symbol: str = "DEMO", periods: int = 390,
                        start: str = "2024-01-02 09:30", amplitude: float = 8.0,
                        cycle_minutes: float = 130.0, base: float = 100.0) -> pd.DataFrame:
    """Deterministic minute-bar series with oscillation (to exercise EMA crosses).

    A clean sine path — NOT a market model. It exists only to drive the engine so the
    event chain produces signals and fills; it implies nothing about profitability.
    """
    timestamps = pd.date_range(start=start, periods=periods, freq="1min")
    t = np.arange(periods)
    close = base + amplitude * np.sin(2 * np.pi * t / cycle_minutes)
    open_ = np.empty(periods)
    open_[0] = close[0]
    open_[1:] = close[:-1]  # open = prior close
    return pd.DataFrame({
        "symbol": symbol, "timestamp": timestamps, "open": open_,
        "high": close + 0.05, "low": close - 0.05, "close": close, "volume": 1000.0,
    })
