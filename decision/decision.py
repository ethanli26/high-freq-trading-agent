"""Turn bars into a single trade decision: propose or skip.

``compute_decision`` ties the entry rule, the ATR stop, and position sizing
together into one dict. It never connects to a broker or places anything — it is
pure logic over the inputs, which keeps it easy to test and to reuse in the
backtester later.
"""

import logging

import pandas as pd

import config
from risk.position import compute_atr, compute_stop, size_position
from signals.entry import breakout_signal, latest_close

log = logging.getLogger(__name__)


def _skip(symbol: str, reason: str) -> dict:
    """Build a skip decision with a clear reason."""
    return {"status": "skip", "symbol": symbol, "reason": reason}


def compute_decision(symbol: str, bars: pd.DataFrame, equity: float) -> dict:
    """Decide whether to propose a trade for ``symbol``.

    Returns a skip dict if there is no breakout, no usable ATR, or the position
    sizes to zero shares. Otherwise returns a proposal dict with the entry
    reference price, stop, ATR, share count, dollar risk, and estimated value.
    """
    if not breakout_signal(bars):
        return _skip(symbol, "no breakout")

    atr = compute_atr(bars, config.ATR_PERIOD)
    if atr is None or atr <= 0:
        return _skip(symbol, "ATR unavailable")

    entry_ref = latest_close(bars)
    if entry_ref is None:
        return _skip(symbol, "no entry price")

    stop = compute_stop(entry_ref, atr)
    shares, risk_dollars = size_position(equity, entry_ref, stop)
    if shares <= 0:
        return _skip(symbol, "position sizes to zero shares")

    return {
        "status": "propose",
        "symbol": symbol,
        "action": "BUY",
        "entry_ref": round(entry_ref, 4),
        "stop": round(stop, 4),
        "atr": round(atr, 4),
        "shares": shares,
        "risk_dollars": round(risk_dollars, 2),
        "est_value": round(shares * entry_ref, 2),
    }
