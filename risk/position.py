"""Position sizing and stop logic.

Stops are volatility-based (ATR), and size comes from the fixed-fractional risk
rule: risk a small, equal fraction of equity per trade, capped by a maximum
position size. This keeps any single loss small, which is what makes the
positive-skew style work.
"""

import logging
import math

import pandas as pd

import config

log = logging.getLogger(__name__)


def compute_atr(bars: pd.DataFrame, period: int) -> float | None:
    """Average True Range (Wilder's smoothing) over ``period``, latest value.

    Returns the most recent ATR as a float, or ``None`` if the bars are missing
    the needed columns or are too short.
    """
    needed = {"High", "Low", "Close"}
    if bars is None or not needed.issubset(getattr(bars, "columns", [])):
        return None

    data = bars[["High", "Low", "Close"]].dropna()
    if len(data) < period + 1:
        return None

    prev_close = data["Close"].shift(1)
    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - prev_close).abs(),
            (data["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    true_range = true_range.iloc[1:]  # drop first row (no previous close)

    # Wilder's smoothing is an EWM with alpha = 1 / period.
    atr = true_range.ewm(alpha=1.0 / period, adjust=False).mean()
    if atr.empty:
        return None
    return float(atr.iloc[-1])


def compute_stop(entry_price: float, atr: float) -> float:
    """Stop price below entry by ``ATR_MULTIPLE`` ATRs."""
    return entry_price - config.ATR_MULTIPLE * atr


def size_position(equity: float, entry_price: float, stop_price: float) -> tuple[int, float]:
    """Size a position from the per-trade risk budget and the max-position cap.

    Returns ``(shares, risk_dollars)``. Shares is the smaller of the
    risk-budget size and the max-position cap. If the per-share risk is
    non-positive or the result rounds to zero shares, returns ``(0, 0.0)``.
    """
    risk_dollars = equity * config.RISK_PER_TRADE
    per_share_risk = entry_price - stop_price

    if per_share_risk <= 0 or entry_price <= 0:
        return 0, 0.0

    raw_shares = math.floor(risk_dollars / per_share_risk)
    max_shares = math.floor((equity * config.MAX_POSITION_PCT) / entry_price)
    shares = min(raw_shares, max_shares)

    if shares <= 0:
        return 0, 0.0
    return int(shares), float(risk_dollars)
