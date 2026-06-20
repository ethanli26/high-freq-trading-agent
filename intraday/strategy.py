"""Intraday strategy handlers.

ONE minimal strategy — a fast/slow EMA cross on minute bars — exists ONLY to exercise
the event engine. It is deliberately simple and is NOT expected to be profitable; an
intraday EMA cross on free/synthetic bars says nothing about real edge (that needs
paid intraday data and microstructure modeling). It reads the engine's incremental
RollingState and emits a SignalEvent on a cross.
"""

from abc import ABC, abstractmethod

from intraday.engine import RollingState
from intraday.events import BarEvent, SignalEvent


class IntradayStrategy(ABC):
    """A handler that maps the current bar + rolling state to an optional signal.

    ``bar`` is the just-completed minute bar (open/high/low/close/timestamp); ``state``
    is the engine's incremental per-symbol state. Strategies use whichever they need —
    price-only strategies ignore ``bar``; session-aware ones (e.g. momentum) use it.
    """

    name: str

    @abstractmethod
    def on_bar(self, bar: BarEvent, state: RollingState) -> SignalEvent | None:
        """Return a SignalEvent for this bar, or None."""


class EmaCrossStrategy(IntradayStrategy):
    """BUY when the fast EMA crosses above the slow EMA; SELL (flatten) on cross down.

    Cross detection is precomputed incrementally in RollingState, so this handler is a
    trivial read of ``crossed_up`` / ``crossed_down``. Demo only — not validated.
    """

    name = "ema_cross"

    def on_bar(self, bar: BarEvent, state: RollingState) -> SignalEvent | None:
        if not state.ready:
            return None
        if state.crossed_up:
            return SignalEvent(state.symbol, state.last_timestamp, "BUY", "ema_cross_up", state.last_close)
        if state.crossed_down:
            return SignalEvent(state.symbol, state.last_timestamp, "SELL", "ema_cross_down", state.last_close)
        return None
