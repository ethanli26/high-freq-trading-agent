"""Engine test: a tiny synthetic scenario proving next-open entry and stop exit.

No network — synthetic bars only. Confirms the two core engine guarantees:
  * an entry fills at the NEXT day's open (no look-ahead), and
  * a stop exit fills at the stop (here a gap-through fills at that day's open).
"""

import numpy as np
import pandas as pd

from backtest.engine import SLIPPAGE_PCT, run_engine
from signals.breakout import BreakoutStrategy


def _scenario():
    """A name that rises (triggering a breakout), is entered, then gaps down to stop out."""
    dates = pd.bdate_range("2015-01-01", periods=160)
    close = np.concatenate([np.linspace(50.0, 100.0, 131), np.full(29, 70.0)])  # rise, then gap-down flat
    close = pd.Series(close, index=dates)
    open_ = close.shift(1)
    open_.iloc[0] = close.iloc[0]
    open_.iloc[131] = 70.0           # day 131 OPENS at 70 (a gap down through the stop)
    high, low = close + 0.5, close - 0.5
    high.iloc[131], low.iloc[131] = 70.5, 69.5
    aaa = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})

    xlk_close = pd.Series(np.linspace(50.0, 90.0, 160), index=dates)  # ETF: positive momentum, sole sector
    xlk = pd.DataFrame({"Open": xlk_close, "High": xlk_close + 0.5, "Low": xlk_close - 0.5, "Close": xlk_close})

    regime = pd.Series("bull", index=dates, name="regime")
    return {"AAA": aaa, "XLK": xlk}, regime, dates


def test_entry_fills_next_open_and_stop_exit():
    bars, regime, dates = _scenario()
    equity, trades = run_engine(
        bars, regime, {"AAA": "XLK"}, ["XLK"], strategies=[BreakoutStrategy()],
        regime_filter=False, trend_exit=False, conviction_sizing=False, warmup_bars=130,
    )

    assert len(trades) == 1
    trade = trades.iloc[0]

    # Entry: the breakout completes at bar 129; the engine fills at the NEXT open (bar 130).
    assert trade["entry_date"] == dates[130]
    assert trade["entry_price"] == round(bars["AAA"]["Open"].iloc[130] * (1 + SLIPPAGE_PCT), 4)

    # Exit: day 131 gaps below the stop, so the fill is that day's open minus slippage.
    assert trade["exit_date"] == dates[131]
    assert trade["exit_price"] == round(70.0 * (1 - SLIPPAGE_PCT), 4)
    assert trade["exit_reason"] in ("stop", "trailing_stop")
    assert trade["bars_held"] == 1
    assert trade["pnl"] < 0  # bought ~100, stopped out ~70
