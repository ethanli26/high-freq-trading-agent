"""Event-driven intraday engine (minute bars) — the architecture showcase.

OVERVIEW
--------
A synchronous event loop pulls ``BarEvent``s from a feed and dispatches them through
a per-bar event queue to three handlers — strategy, risk gate, execution:

    for bar in feed:
        drop out-of-order / duplicate timestamps (log)         # real-time robustness
        note gaps (missing minutes)                            # real-time robustness
        queue = [pending fill (prev order @ this open), bar]
        dispatch queue:
            FillEvent   -> portfolio.apply_fill
            BarEvent    -> update rolling state (incremental) -> strategy.on_bar -> SignalEvent?
            SignalEvent -> risk gate (size via risk.size_position + intraday limits) -> OrderEvent?
            OrderEvent  -> execution.submit (fills on the NEXT bar's open)
    shutdown: cancel pending orders, flatten open positions, summarize

Per-symbol state is updated INCREMENTALLY each bar (EMAs and a bounded recent-close
window) — never recomputed from scratch. Execution is SIMULATED only; a live path
would route orders through the existing paper-safety guards (DU-account + DRY_RUN).
This is an architecture demo on replayed data; it is NOT a validated strategy, and
validating intraday alpha would require paid intraday data + microstructure modeling.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from intraday.events import BarEvent, FillEvent, OrderEvent, SignalEvent
from risk.position import size_position  # reuse the existing risk sizing

log = logging.getLogger(__name__)

# Intraday cost assumptions (estimates; kept local so the subsystem is self-contained).
INTRADAY_SLIPPAGE_PCT = 0.0005   # 5 bps per side
COMMISSION_PER_SHARE = 0.005     # $/share


@dataclass
class RollingState:
    """Per-symbol state updated incrementally as each bar arrives (no recompute)."""

    symbol: str
    fast_period: int
    slow_period: int
    fast_ema: float | None = None
    slow_ema: float | None = None
    bar_count: int = 0
    last_close: float | None = None
    last_timestamp: datetime | None = None
    crossed_up: bool = False
    crossed_down: bool = False
    _fast_above: bool | None = None
    recent_closes: deque = field(default_factory=lambda: deque(maxlen=64))

    @property
    def ready(self) -> bool:
        """Enough bars seen for the slow EMA to be meaningful."""
        return self.bar_count >= self.slow_period

    def update(self, bar: BarEvent) -> None:
        """Fold one bar into the rolling state incrementally; recompute nothing."""
        price = bar.close
        fast_alpha = 2.0 / (self.fast_period + 1)
        slow_alpha = 2.0 / (self.slow_period + 1)
        self.fast_ema = price if self.fast_ema is None else fast_alpha * price + (1 - fast_alpha) * self.fast_ema
        self.slow_ema = price if self.slow_ema is None else slow_alpha * price + (1 - slow_alpha) * self.slow_ema
        self.bar_count += 1

        # Incremental cross detection: compare the new sign to the stored prior sign.
        self.crossed_up = self.crossed_down = False
        if self.ready:
            fast_above = self.fast_ema > self.slow_ema
            if self._fast_above is not None and fast_above != self._fast_above:
                self.crossed_up = fast_above
                self.crossed_down = not fast_above
            self._fast_above = fast_above

        self.last_close = price
        self.last_timestamp = bar.timestamp
        self.recent_closes.append(price)


class Portfolio:
    """Cash, per-symbol long positions, and realized P&L, updated from fills."""

    def __init__(self, starting_equity: float, commission_per_share: float):
        self.starting_equity = starting_equity
        self.cash = starting_equity
        self.commission_per_share = commission_per_share
        self.positions: dict[str, dict] = {}   # symbol -> {shares, avg_price, cost}
        self.last_prices: dict[str, float] = {}
        self.realized_pnl = 0.0

    def mark(self, symbol: str, price: float) -> None:
        self.last_prices[symbol] = price

    def is_long(self, symbol: str) -> bool:
        return self.positions.get(symbol, {}).get("shares", 0) > 0

    def shares(self, symbol: str) -> int:
        return self.positions.get(symbol, {}).get("shares", 0)

    def num_positions(self) -> int:
        return sum(1 for p in self.positions.values() if p["shares"] > 0)

    def equity(self) -> float:
        """Cash plus open positions marked at their last seen price."""
        held = sum(p["shares"] * self.last_prices.get(s, p["avg_price"])
                   for s, p in self.positions.items())
        return self.cash + held

    def apply_fill(self, fill: FillEvent) -> None:
        """Update cash/positions/realized P&L from a fill (long-only demo)."""
        if fill.side == "BUY":
            cost = fill.quantity * fill.fill_price + fill.commission
            self.cash -= cost
            self.positions[fill.symbol] = {"shares": fill.quantity, "avg_price": fill.fill_price, "cost": cost}
        else:  # SELL closes the position
            position = self.positions.get(fill.symbol)
            if not position or position["shares"] <= 0:
                return
            proceeds = fill.quantity * fill.fill_price - fill.commission
            self.cash += proceeds
            self.realized_pnl += proceeds - position["cost"]
            self.positions[fill.symbol] = {"shares": 0, "avg_price": 0.0, "cost": 0.0}


class SimulatedExecution:
    """Simulated broker: an order placed on bar t fills at bar t+1's open + slippage.

    Filling on the NEXT bar's open (not the signal bar's close) keeps the engine
    free of look-ahead. Places NO real orders.
    """

    def __init__(self, slippage_pct: float = INTRADAY_SLIPPAGE_PCT,
                 commission_per_share: float = COMMISSION_PER_SHARE):
        self.slippage_pct = slippage_pct
        self.commission_per_share = commission_per_share
        self.pending: dict[str, OrderEvent] = {}  # one pending order per symbol (demo)

    def submit(self, order: OrderEvent) -> None:
        self.pending[order.symbol] = order

    def fill_pending(self, bar: BarEvent) -> FillEvent | None:
        """Fill a symbol's pending order at THIS bar's open (it was placed last bar)."""
        order = self.pending.pop(bar.symbol, None)
        if order is None:
            return None
        sign = 1.0 if order.side == "BUY" else -1.0
        price = bar.open * (1.0 + sign * self.slippage_pct)
        commission = order.quantity * self.commission_per_share
        return FillEvent(order.symbol, bar.timestamp, order.side, order.quantity,
                         round(price, 4), round(commission, 4), self.slippage_pct)

    def cancel_all(self) -> list[OrderEvent]:
        cancelled = list(self.pending.values())
        self.pending.clear()
        return cancelled


class IntradayRiskGate:
    """Size signals via the existing risk sizing, behind simple intraday limits."""

    def __init__(self, max_positions: int = 3, stop_pct: float = 0.005):
        self.max_positions = max_positions
        self.stop_pct = stop_pct

    def on_signal(self, signal: SignalEvent, portfolio: Portfolio, equity: float) -> OrderEvent | None:
        """Turn a signal into a sized order, or block it (returns None)."""
        symbol = signal.symbol
        if signal.side == "BUY":
            if portfolio.is_long(symbol) or portfolio.num_positions() >= self.max_positions:
                return None  # one position per name; cap concurrent positions
            stop = signal.reference_price * (1.0 - self.stop_pct)
            shares, _ = size_position(equity, signal.reference_price, stop)  # existing 1%-risk sizing
            if shares < 1:
                return None
            return OrderEvent(symbol, signal.timestamp, "BUY", shares)
        if signal.side == "SELL":
            if not portfolio.is_long(symbol):
                return None
            return OrderEvent(symbol, signal.timestamp, "SELL", portfolio.shares(symbol))
        return None


class EventEngine:
    """The event loop: feed -> (robustness checks) -> handler chain -> portfolio."""

    def __init__(self, feed, strategy, risk_gate: IntradayRiskGate, execution: SimulatedExecution,
                 portfolio: Portfolio, *, fast_period: int, slow_period: int,
                 resolution_minutes: int = 1):
        self.feed = feed
        self.strategy = strategy
        self.risk_gate = risk_gate
        self.execution = execution
        self.portfolio = portfolio
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.resolution_minutes = resolution_minutes
        self.states: dict[str, RollingState] = {}
        self._last_ts: dict[str, datetime] = {}
        self.event_log: list[tuple[str, object]] = []
        self.stats = {"bars_seen": 0, "bars_processed": 0, "dropped": 0, "gaps": 0,
                      "signals": 0, "orders": 0, "fills": 0}

    def _state(self, symbol: str) -> RollingState:
        if symbol not in self.states:
            self.states[symbol] = RollingState(symbol, self.fast_period, self.slow_period)
        return self.states[symbol]

    def _accept(self, bar: BarEvent) -> bool:
        """Reject out-of-order or duplicate timestamps (real-time robustness)."""
        last = self._last_ts.get(bar.symbol)
        if last is not None and bar.timestamp <= last:
            kind = "duplicate" if bar.timestamp == last else "out-of-order"
            log.warning("DROP %s bar for %s at %s (last was %s).", kind, bar.symbol, bar.timestamp, last)
            self.stats["dropped"] += 1
            return False
        return True

    def _check_gap(self, bar: BarEvent) -> None:
        """Log missing-minute gaps (real-time robustness); never fabricate bars."""
        last = self._last_ts.get(bar.symbol)
        if last is None:
            return
        minutes = (bar.timestamp - last).total_seconds() / 60.0
        if minutes > self.resolution_minutes + 1e-9:
            missing = int(round(minutes / self.resolution_minutes)) - 1
            log.warning("GAP for %s: %d missing bar(s) before %s.", bar.symbol, missing, bar.timestamp)
            self.stats["gaps"] += 1

    def _log(self, event) -> None:
        self.event_log.append((type(event).__name__, event))

    def _dispatch(self, event) -> list:
        """Route one event to its handler; return any events it produces."""
        self._log(event)
        if isinstance(event, FillEvent):
            self.stats["fills"] += 1
            self.portfolio.apply_fill(event)
            return []
        if isinstance(event, BarEvent):
            state = self._state(event.symbol)
            state.update(event)                       # incremental state
            self.portfolio.mark(event.symbol, event.close)
            # Handlers receive the raw bar (open/time) plus the rolling state, so a
            # strategy can use either; price-only strategies ignore the bar.
            signal = self.strategy.on_bar(event, state)
            return [signal] if signal else []
        if isinstance(event, SignalEvent):
            self.stats["signals"] += 1
            order = self.risk_gate.on_signal(event, self.portfolio, self.portfolio.equity())
            return [order] if order else []
        if isinstance(event, OrderEvent):
            self.stats["orders"] += 1
            self.execution.submit(event)              # fills on the next bar's open
            return []
        return []

    def run(self) -> dict:
        """Run the event loop to exhaustion, then shut down cleanly. Returns the summary."""
        try:
            for bar in self.feed:
                self.stats["bars_seen"] += 1
                if not self._accept(bar):
                    continue
                self._check_gap(bar)
                queue = deque()
                fill = self.execution.fill_pending(bar)   # prior order fills at this open
                if fill is not None:
                    queue.append(fill)
                queue.append(bar)
                while queue:
                    for produced in self._dispatch(queue.popleft()):
                        queue.append(produced)
                self._last_ts[bar.symbol] = bar.timestamp
                self.stats["bars_processed"] += 1
        except KeyboardInterrupt:  # clean shutdown on interrupt
            log.warning("Interrupted; shutting down cleanly.")
        return self._shutdown()

    def _shutdown(self) -> dict:
        """Cancel pending orders, flatten open positions at last price, summarize."""
        for order in self.execution.cancel_all():
            log.info("SHUTDOWN cancel pending %s %d %s.", order.side, order.quantity, order.symbol)
        for symbol, position in list(self.portfolio.positions.items()):
            if position["shares"] > 0:
                price = self.portfolio.last_prices.get(symbol, position["avg_price"])
                log.info("SHUTDOWN flatten %d %s @ %.4f (end-of-session).", position["shares"], symbol, price)
                self.portfolio.apply_fill(FillEvent(symbol, None, "SELL", position["shares"],
                                                    round(price, 4),
                                                    round(position["shares"] * self.portfolio.commission_per_share, 4),
                                                    0.0))
        return self.summary()

    def summary(self) -> dict:
        """End-of-session counters and simulated P&L."""
        equity = self.portfolio.equity()
        return {
            **self.stats,
            "realized_pnl": round(self.portfolio.realized_pnl, 2),
            "final_equity": round(equity, 2),
            "return_pct": equity / self.portfolio.starting_equity - 1.0,
        }
