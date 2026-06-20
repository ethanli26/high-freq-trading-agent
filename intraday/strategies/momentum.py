"""Intraday time-series momentum — a mechanically specified engine handler.

RATIONALE & CITATION
--------------------
Time-series ("trend") momentum (Moskowitz, Ooi & Pedersen, 2012, *Journal of
Financial Economics*, "Time Series Momentum") says an asset's own recent return
predicts its near-future return. The intraday instantiation tested here follows Gao,
Han, Li & Zhou (2018, *JFE*, "Market Intraday Momentum"): the morning return (open ->
midday) predicts the rest-of-day return.

RULE (no discretion)
--------------------
Each trading day, per symbol:
  * morning_return = close[cutoff] / open[first bar of day] - 1   (both completed bars)
  * if morning_return >  threshold  -> go LONG for the rest of the day
  * if morning_return < -threshold  -> go SHORT for the rest of the day
  * else                            -> stay flat
  * EXIT near the close; one position per symbol per day.

LOOK-AHEAD SAFETY: the decision is taken on the COMPLETED cutoff bar (open is the
day's first bar, close is the cutoff bar's close — both already in the past), and the
engine fills the resulting order at the NEXT bar's open. The EXIT is signalled one
bar before the session end so its fill (next open) still lands inside the session,
never overnight. The harness/tests assert the fill timestamp is strictly after the
decision bar.
"""

from dataclasses import dataclass
from datetime import time

from intraday.engine import RollingState
from intraday.events import BarEvent, OrderEvent, SignalEvent
from risk.position import size_position  # reuse the existing risk sizing


@dataclass
class _DayState:
    """Per-symbol intraday bookkeeping for one trading day."""

    day: object = None
    day_open: float | None = None
    decided: bool = False
    in_position: bool = False


class MomentumStrategy:
    """Open-to-midday momentum: long/short for the rest of the day, flat by the close."""

    name = "intraday_momentum"

    def __init__(self, threshold: float = 0.001,
                 cutoff: time = time(12, 45), close: time = time(15, 58)):
        self.threshold = threshold
        self.cutoff = cutoff
        self.close = close
        self._state: dict[str, _DayState] = {}

    def on_bar(self, bar: BarEvent, state: RollingState) -> SignalEvent | None:
        """Decide at the cutoff; exit near the close. Uses only completed bars."""
        day_state = self._state.setdefault(bar.symbol, _DayState())
        bar_date, bar_time = bar.timestamp.date(), bar.timestamp.time()

        if day_state.day != bar_date:  # first bar of a new day sets the session open
            self._state[bar.symbol] = day_state = _DayState(day=bar_date, day_open=bar.open)

        # Decision at the cutoff (morning_return uses only the day's completed bars).
        if not day_state.decided and bar_time >= self.cutoff and day_state.day_open:
            day_state.decided = True
            morning_return = bar.close / day_state.day_open - 1.0
            if morning_return > self.threshold:
                day_state.in_position = True
                return SignalEvent(bar.symbol, bar.timestamp, "LONG", "morning_up", bar.close)
            if morning_return < -self.threshold:
                day_state.in_position = True
                return SignalEvent(bar.symbol, bar.timestamp, "SHORT", "morning_down", bar.close)
            return None

        # Exit near the close (one bar early so the next-open fill stays in-session).
        if day_state.in_position and bar_time >= self.close:
            day_state.in_position = False
            return SignalEvent(bar.symbol, bar.timestamp, "EXIT", "session_close", bar.close)
        return None


class MomentumRiskGate:
    """Size LONG/SHORT entries via the existing risk sizing; flatten on EXIT."""

    def __init__(self, stop_pct: float = 0.005):
        self.stop_pct = stop_pct

    def on_signal(self, signal: SignalEvent, portfolio, equity: float) -> OrderEvent | None:
        symbol = signal.symbol
        held = portfolio.signed_shares(symbol)
        if signal.side in ("LONG", "SHORT"):
            if held != 0:  # one position per symbol per day
                return None
            stop = signal.reference_price * (1.0 - self.stop_pct)  # risk magnitude (symmetric)
            shares, _ = size_position(equity, signal.reference_price, stop)
            if shares < 1:
                return None
            order_side = "BUY" if signal.side == "LONG" else "SELL"
            return OrderEvent(symbol, signal.timestamp, order_side, shares)
        if signal.side == "EXIT":
            if held == 0:
                return None
            order_side = "SELL" if held > 0 else "BUY"  # close in the opposite direction
            return OrderEvent(symbol, signal.timestamp, order_side, abs(held))
        return None


class LongShortPortfolio:
    """Signed-position portfolio with per-round-trip, after-cost realized P&L.

    Supports the flat -> entry -> flat round trip the momentum strategy produces (long
    or short). Each closed round trip records its P&L NET of the spread/slippage baked
    into the fill prices and both commissions.
    """

    def __init__(self, starting_equity: float, commission_per_share: float = 0.005):
        self.starting_equity = starting_equity
        self.cash = starting_equity
        self.commission_per_share = commission_per_share
        self.positions: dict[str, dict] = {}   # symbol -> {shares (signed), avg, entry_commission}
        self.last_prices: dict[str, float] = {}
        self.realized_pnl = 0.0
        self.trades: list[dict] = []            # one entry per closed round trip

    def mark(self, symbol: str, price: float) -> None:
        self.last_prices[symbol] = price

    def signed_shares(self, symbol: str) -> int:
        return self.positions.get(symbol, {}).get("shares", 0)

    def equity(self) -> float:
        held = sum(p["shares"] * self.last_prices.get(s, p["avg"]) for s, p in self.positions.items())
        return self.cash + held

    def apply_fill(self, fill) -> None:
        """Open from flat, or close to flat, updating cash and realized P&L."""
        signed_qty = fill.quantity if fill.side == "BUY" else -fill.quantity
        self.cash -= signed_qty * fill.fill_price   # buy: cash down; sell: cash up
        self.cash -= fill.commission

        position = self.positions.get(fill.symbol)
        if not position or position["shares"] == 0:  # opening from flat
            self.positions[fill.symbol] = {"shares": signed_qty, "avg": fill.fill_price,
                                           "entry_commission": fill.commission}
            return

        starting_shares = position["shares"]
        if (starting_shares > 0) != (signed_qty > 0):  # closing (opposite direction)
            closed = min(abs(signed_qty), abs(starting_shares))
            direction = 1.0 if starting_shares > 0 else -1.0
            price_pnl = closed * (fill.fill_price - position["avg"]) * direction
            net = price_pnl - position.get("entry_commission", 0.0) - fill.commission
            self.realized_pnl += net
            self.trades.append({"symbol": fill.symbol, "timestamp": fill.timestamp, "pnl": net})
            self.positions[fill.symbol] = {"shares": starting_shares + signed_qty, "avg": 0.0,
                                           "entry_commission": 0.0}
