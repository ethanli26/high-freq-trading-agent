"""A brutal, illiquidity-scaled cost model for small-coin crypto.

Small-coin crypto backtests lie unless costs are taken seriously. Three things make
crypto trading expensive, and ALL of them get worse the smaller/thinner the coin:

  * exchange taker fees — a flat per-side fee on notional (~10 bps is typical retail;
    worse on small venues),
  * spread — thin books quote wide,
  * slippage / market impact — your own order walks the book.

This model charges, PER SIDE (entry AND exit):

    taker_fee_bps  +  base_spread_bps * illiquidity_factor(ADV) * stress_multiplier

where ``illiquidity_factor`` rises as a coin's average daily dollar volume (ADV) falls
below a "liquid" reference, so a coin trading $0.5M/day pays far more than one trading
$50M/day. ``stress_multiplier`` is the knob the cost-sensitivity sweep turns to find
the level at which any edge dies. Every number here is an ESTIMATE — real per-coin
spreads need order-book data this model does not have, so treat outputs as a plausible
floor on costs, not a measurement.

The model also exposes ``adv_participation_cap`` — the maximum fraction of a coin's ADV
a single position may take — so the engine never simulates an unfillable trade. It
places NO real orders.
"""

from dataclasses import dataclass

MAX_PER_SIDE_BPS = 1_000.0  # clamp: even the worst modeled side costs <= 10% (sanity)


@dataclass(frozen=True)
class CryptoCostModel:
    """Per-side crypto trading cost that scales with a coin's illiquidity."""

    taker_fee_bps: float = 10.0           # exchange taker fee, per side, on notional
    base_spread_bps: float = 10.0         # half-spread+slippage for a LIQUID coin, per side
    ref_dollar_volume: float = 50_000_000.0  # ADV at/above which a coin is "liquid"
    illiquidity_exponent: float = 0.5     # how fast cost grows as ADV falls (0.5 = sqrt)
    max_illiquidity_factor: float = 25.0  # cap the multiplier so it stays finite
    stress_multiplier: float = 1.0        # sweep knob: scales the spread/slippage term
    adv_participation_cap: float = 0.01   # a position may take <= this fraction of ADV

    def illiquidity_factor(self, adv_dollar: float) -> float:
        """Cost multiplier vs a liquid coin: 1.0 at the reference, higher when thinner.

        ``(ref / adv) ** exponent``, clamped to ``max_illiquidity_factor``. A coin with
        1/100th the reference ADV pays ``100 ** 0.5 = 10x`` the base spread, for example.
        """
        adv = max(float(adv_dollar), 1.0)
        factor = (self.ref_dollar_volume / adv) ** self.illiquidity_exponent
        return float(min(max(factor, 1.0), self.max_illiquidity_factor))

    def per_side_bps(self, adv_dollar: float) -> float:
        """Total per-side cost in basis points for a coin of the given ADV."""
        spread = self.base_spread_bps * self.illiquidity_factor(adv_dollar) * self.stress_multiplier
        return min(self.taker_fee_bps + spread, MAX_PER_SIDE_BPS)

    def per_side_fraction(self, adv_dollar: float) -> float:
        """Per-side cost as a fraction (what the engine multiplies the fill price by)."""
        return self.per_side_bps(adv_dollar) / 10_000.0

    def round_trip_bps(self, adv_dollar: float) -> float:
        """Entry+exit cost in basis points (two sides) for a coin of the given ADV."""
        return 2.0 * self.per_side_bps(adv_dollar)


def representative_adv(bars, lookback: int = 30) -> float:
    """A single representative ADV (median trailing dollar volume) for a coin.

    The engine's ADV *cap* uses a time-varying trailing mean; the cost FUNCTION uses
    one representative figure per coin (the median of the trailing dollar-volume series)
    so a coin's modeled spread is stable. Median resists the volume spikes that wash
    trading produces. Returns 0.0 if volume is unavailable.
    """
    if "Volume" not in getattr(bars, "columns", []):
        return 0.0
    dollar_volume = (bars["Close"] * bars["Volume"]).rolling(lookback).mean().dropna()
    if dollar_volume.empty:
        return float((bars["Close"] * bars["Volume"]).median())
    return float(dollar_volume.median())


def build_slippage_fn(bars: dict, cost_model: CryptoCostModel, lookback: int = 30):
    """Return ``f(symbol) -> per-side fraction`` for the engine's ``slippage_fn`` hook.

    Precomputes each coin's representative ADV once, so the engine can price costs per
    name. Unknown symbols fall back to the reference (liquid) cost.
    """
    adv_by_symbol = {sym: representative_adv(frame, lookback) for sym, frame in bars.items()}

    def slippage_fn(symbol: str) -> float:
        adv = adv_by_symbol.get(symbol, cost_model.ref_dollar_volume)
        return cost_model.per_side_fraction(adv)

    return slippage_fn
