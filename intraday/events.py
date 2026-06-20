"""Event types for the intraday event-driven engine.

The engine is a clean event pipeline. Each minute bar drives one pass of:

    BarEvent ──(strategy.on_bar)──▶ SignalEvent
                                       │
                                (risk gate: size + limits)
                                       ▼
                                   OrderEvent ──(execution)──▶ FillEvent ─▶ portfolio

A market OrderEvent created on bar *t* fills at bar *t+1*'s open (a FillEvent), so the
chain never uses information from the bar that hasn't completed — the same
no-look-ahead discipline as the daily backtester, applied to minute bars.

Events are immutable (frozen dataclasses) so a handler can never mutate an event in
flight. Resolution is minute bars.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class BarEvent:
    """A completed minute bar (timestamp = the bar's close time)."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    resolution: str = "1min"


@dataclass(frozen=True)
class SignalEvent:
    """A strategy's intent on a bar: a side plus the reference (trigger) price."""

    symbol: str
    timestamp: datetime
    side: str          # "BUY" (enter long) | "SELL" (flatten long)
    reason: str        # human-readable trigger, e.g. "ema_cross_up"
    reference_price: float


@dataclass(frozen=True)
class OrderEvent:
    """A sized order produced by the risk gate; fills on the next bar's open."""

    symbol: str
    timestamp: datetime
    side: str
    quantity: int
    order_type: str = "MKT"


@dataclass(frozen=True)
class FillEvent:
    """A simulated execution of an order (price net of slippage; commission charged)."""

    symbol: str
    timestamp: datetime
    side: str
    quantity: int
    fill_price: float
    commission: float
    slippage_pct: float
