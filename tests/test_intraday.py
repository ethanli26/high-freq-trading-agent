"""Tests for the event-driven intraday engine (offline, deterministic).

Covers: feed time-ordering, incremental rolling-state correctness, out-of-order /
duplicate handling, and deterministic signals/fills on a known synthetic series.
"""

import numpy as np
import pandas as pd

from intraday.engine import (
    COMMISSION_PER_SHARE,
    INTRADAY_SLIPPAGE_PCT,
    EventEngine,
    IntradayRiskGate,
    Portfolio,
    RollingState,
    SimulatedExecution,
)
from intraday.events import BarEvent
from intraday.feed import ReplayFeed, make_synthetic_bars
from intraday.strategy import EmaCrossStrategy

FAST, SLOW = 5, 20


def _bar(symbol, timestamp, price):
    return BarEvent(symbol, timestamp, open=price, high=price + 0.05,
                    low=price - 0.05, close=price, volume=1000.0)


def _engine(feed):
    return EventEngine(feed, EmaCrossStrategy(),
                       IntradayRiskGate(max_positions=3, stop_pct=0.005),
                       SimulatedExecution(INTRADAY_SLIPPAGE_PCT, COMMISSION_PER_SHARE),
                       Portfolio(100_000.0, COMMISSION_PER_SHARE),
                       fast_period=FAST, slow_period=SLOW)


def test_replayfeed_yields_in_time_order():
    df = make_synthetic_bars(periods=20).sample(frac=1.0, random_state=1)  # shuffle input
    timestamps = [bar.timestamp for bar in ReplayFeed(df)]
    assert timestamps == sorted(timestamps)


def test_rolling_state_ema_is_incremental_and_correct():
    closes = list(100 + 5 * np.sin(np.arange(120) / 7.0))
    times = pd.date_range("2024-01-02 09:30", periods=len(closes), freq="1min")
    state = RollingState("X", FAST, SLOW)
    for ts, price in zip(times, closes):
        state.update(_bar("X", ts, price))
    # Incremental EMAs must match a from-scratch pandas EWM (adjust=False seeds at x0).
    expected_fast = pd.Series(closes).ewm(span=FAST, adjust=False).mean().iloc[-1]
    expected_slow = pd.Series(closes).ewm(span=SLOW, adjust=False).mean().iloc[-1]
    assert abs(state.fast_ema - expected_fast) < 1e-9
    assert abs(state.slow_ema - expected_slow) < 1e-9
    assert state.bar_count == len(closes)


def test_engine_drops_out_of_order_and_duplicate_bars():
    ts = pd.date_range("2024-01-02 09:30", periods=4, freq="1min")
    events = [
        _bar("X", ts[0], 100.0), _bar("X", ts[1], 101.0), _bar("X", ts[2], 102.0),
        _bar("X", ts[2], 102.5),   # duplicate timestamp -> dropped
        _bar("X", ts[1], 101.5),   # out-of-order timestamp -> dropped
        _bar("X", ts[3], 103.0),
    ]
    engine = _engine(events)  # the engine accepts any iterable of BarEvents
    engine.run()
    assert engine.stats["dropped"] == 2
    assert engine.stats["bars_processed"] == 4
    assert engine.states["X"].last_timestamp == ts[3]  # state stayed monotonic


def test_known_series_produces_deterministic_signals_and_fills():
    summary_1 = _engine(ReplayFeed(make_synthetic_bars(periods=260))).run()
    engine_2 = _engine(ReplayFeed(make_synthetic_bars(periods=260)))
    summary_2 = engine_2.run()
    assert summary_1 == summary_2  # fully deterministic replay

    signal_sides = [e.side for kind, e in engine_2.event_log if kind == "SignalEvent"]
    fill_sides = [e.side for kind, e in engine_2.event_log if kind == "FillEvent"]
    assert "BUY" in signal_sides and "SELL" in signal_sides   # crosses both ways
    assert "BUY" in fill_sides                                # at least one entry filled
    assert summary_2["fills"] >= 1
