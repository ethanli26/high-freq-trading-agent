"""Realistic intraday transaction-cost model — the crux of the honest test.

Intraday P&L lives or dies on costs. Each trade pays, on EACH side (entry and exit):

    half-spread (bps)  +  slippage (bps)  +  per-share commission

The bps figures are ESTIMATES, not measured — they dominate intraday results, so the
test sweeps them. ~5 bps half-spread / side is a rough liquid-large-cap figure; real
spreads vary by name, time of day, and venue and can only be known from paid,
point-in-time intraday quote data. A round trip therefore costs roughly
``2 * (half_spread_bps + slippage_bps)`` in price plus two commissions.

``CostAwareExecution`` is a drop-in execution handler for the intraday engine that
applies this model (it shares the engine's interface: ``submit`` / ``fill_pending`` /
``cancel_all``). It places NO real orders.
"""

from dataclasses import dataclass

from intraday.events import BarEvent, FillEvent, OrderEvent


@dataclass(frozen=True)
class CostModel:
    """Per-side transaction costs (half-spread + slippage in bps, plus commission)."""

    half_spread_bps: float = 5.0      # half the quoted spread, per side
    slippage_bps: float = 2.0         # market-impact / timing slippage, per side
    commission_per_share: float = 0.005

    @property
    def per_side_bps(self) -> float:
        """Total price cost charged on each side, in basis points."""
        return self.half_spread_bps + self.slippage_bps

    @property
    def round_trip_bps(self) -> float:
        """Price cost of a full entry+exit round trip, in basis points (ex-commission)."""
        return 2.0 * self.per_side_bps

    def fill_price(self, reference_price: float, side: str) -> float:
        """Adverse fill: a BUY pays up, a SELL receives less, by ``per_side_bps``."""
        adjustment = self.per_side_bps / 10_000.0
        sign = 1.0 if side == "BUY" else -1.0
        return reference_price * (1.0 + sign * adjustment)

    def commission(self, quantity: int) -> float:
        """Per-share commission for an order."""
        return abs(quantity) * self.commission_per_share


class CostAwareExecution:
    """Engine execution handler that fills next-bar-open under a ``CostModel``.

    An order placed on bar *t* fills at bar *t+1*'s open, adjusted by the cost model
    (so there is no look-ahead). Simulation only — never places real orders.
    """

    def __init__(self, cost_model: CostModel):
        self.cost_model = cost_model
        self.pending: dict[str, OrderEvent] = {}
        self.total_cost = 0.0       # $ paid to costs (price impact + commission)
        self.total_commission = 0.0

    def submit(self, order: OrderEvent) -> None:
        self.pending[order.symbol] = order

    def fill_pending(self, bar: BarEvent) -> FillEvent | None:
        """Fill a symbol's pending order at THIS bar's open, net of costs."""
        order = self.pending.pop(bar.symbol, None)
        if order is None:
            return None
        price = self.cost_model.fill_price(bar.open, order.side)
        commission = self.cost_model.commission(order.quantity)
        # The cost toll = price impact vs a frictionless fill at the open, + commission.
        impact = abs(order.quantity) * bar.open * (self.cost_model.per_side_bps / 10_000.0)
        self.total_cost += impact + commission
        self.total_commission += commission
        return FillEvent(order.symbol, bar.timestamp, order.side, order.quantity,
                         round(price, 4), round(commission, 4), self.cost_model.per_side_bps / 10_000.0)

    def cancel_all(self) -> list[OrderEvent]:
        cancelled = list(self.pending.values())
        self.pending.clear()
        return cancelled
