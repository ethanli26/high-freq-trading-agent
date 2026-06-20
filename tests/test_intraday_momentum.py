"""Tests for the intraday-momentum strategy and its cost model (offline, deterministic).

Covers: signal correctness on a known morning move, look-ahead safety (orders fill
strictly after the decision bar), cost-model arithmetic, and run determinism.
"""

from datetime import time

import pandas as pd

from intraday.costs import CostAwareExecution, CostModel
from intraday.engine import EventEngine
from intraday.events import BarEvent
from intraday.feed import ReplayFeed
from intraday.strategies.momentum import (
    LongShortPortfolio,
    MomentumRiskGate,
    MomentumStrategy,
)
from intraday.run_momentum_test import make_random_walk_sessions, run_once


def _session(open_price, cutoff_price, *, symbol="TST", date="2024-01-02"):
    """One trading day: flat-ish morning that lands at ``cutoff_price`` by 12:45,
    then holds flat to the close. Lets us force a known morning return sign."""
    open_t = pd.Timestamp(f"{date} 09:30")
    cutoff_t = pd.Timestamp(f"{date} 12:45")
    rows = []
    # 09:30 .. 12:44 ramp linearly from open_price to just under cutoff, 12:45 = cutoff.
    times = pd.date_range(open_t, cutoff_t, freq="1min")
    n = len(times)
    for i, ts in enumerate(times):
        price = open_price + (cutoff_price - open_price) * (i / (n - 1))
        is_open = i == 0
        rows.append((symbol, ts, open_price if is_open else prev, price))
        prev = price
    # Hold flat from 12:46 to 16:00 so the EXIT (15:58) and its next-open fill exist.
    for ts in pd.date_range(pd.Timestamp(f"{date} 12:46"), pd.Timestamp(f"{date} 16:00"), freq="1min"):
        rows.append((symbol, ts, cutoff_price, cutoff_price))
        prev = cutoff_price
    df = pd.DataFrame(rows, columns=["symbol", "timestamp", "open", "close"])
    df["high"] = df[["open", "close"]].max(axis=1) + 0.01
    df["low"] = df[["open", "close"]].min(axis=1) - 0.01
    df["volume"] = 1000.0
    return df


def _run(df, cost_model=None):
    cost_model = cost_model or CostModel(half_spread_bps=5.0, slippage_bps=0.0)
    execution = CostAwareExecution(cost_model)
    portfolio = LongShortPortfolio(100_000.0, cost_model.commission_per_share)
    engine = EventEngine(ReplayFeed(df), MomentumStrategy(threshold=0.001),
                         MomentumRiskGate(stop_pct=0.005), execution, portfolio,
                         fast_period=9, slow_period=21)
    engine.run()
    return engine, portfolio, execution


# ---- signal correctness -------------------------------------------------------

def test_morning_up_goes_long():
    """+0.5% morning move (> 0.1% threshold) opens a LONG (a BUY entry then SELL exit)."""
    engine, portfolio, _ = _run(_session(100.0, 100.5))
    fills = [e for kind, e in engine.event_log if kind == "FillEvent"]
    assert fills[0].side == "BUY"          # entered long
    assert fills[-1].side == "SELL"        # flattened at the close
    assert portfolio.signed_shares("TST") == 0


def test_morning_down_goes_short():
    """-0.5% morning move opens a SHORT (a SELL entry then BUY exit)."""
    engine, portfolio, _ = _run(_session(100.0, 99.5))
    fills = [e for kind, e in engine.event_log if kind == "FillEvent"]
    assert fills[0].side == "SELL"         # entered short
    assert fills[-1].side == "BUY"         # bought to cover at the close
    assert portfolio.signed_shares("TST") == 0


def test_flat_morning_no_trade():
    """A morning move inside the threshold band produces no position."""
    engine, portfolio, _ = _run(_session(100.0, 100.02))  # +0.02% < 0.1% threshold
    fills = [e for kind, e in engine.event_log if kind == "FillEvent"]
    assert fills == []
    assert portfolio.signed_shares("TST") == 0


# ---- look-ahead safety --------------------------------------------------------

def test_fills_strictly_after_decision_bar():
    """Every order fills on a LATER bar than the signal that produced it."""
    engine, _, _ = _run(_session(100.0, 100.5))
    signals = [e for kind, e in engine.event_log if kind == "SignalEvent"]
    orders = [e for kind, e in engine.event_log if kind == "OrderEvent"]
    fills = [e for kind, e in engine.event_log if kind == "FillEvent"]
    assert signals and orders and fills
    for order, fill in zip(orders, fills):
        assert fill.timestamp > order.timestamp   # next-open fill, never same bar
    # The entry decision is taken at/after the cutoff, never before.
    assert signals[0].timestamp.time() >= time(12, 45)


# ---- cost-model arithmetic ----------------------------------------------------

def test_cost_model_arithmetic():
    cm = CostModel(half_spread_bps=5.0, slippage_bps=2.0, commission_per_share=0.005)
    assert cm.per_side_bps == 7.0
    assert cm.round_trip_bps == 14.0
    # BUY pays up by 7 bps; SELL receives 7 bps less.
    assert cm.fill_price(100.0, "BUY") == 100.0 * (1 + 7e-4)
    assert cm.fill_price(100.0, "SELL") == 100.0 * (1 - 7e-4)
    assert cm.commission(200) == 1.0


def test_cost_toll_accumulates_and_scales():
    """The reported toll is positive and grows with the spread assumption."""
    df = _session(100.0, 100.5)
    _, _, exec_cheap = _run(df, CostModel(half_spread_bps=2.0, slippage_bps=0.0))
    _, _, exec_dear = _run(df, CostModel(half_spread_bps=10.0, slippage_bps=0.0))
    assert exec_cheap.total_cost > 0
    assert exec_dear.total_cost > exec_cheap.total_cost
    # Zero spread leaves only commission as the toll.
    _, _, exec_free = _run(df, CostModel(half_spread_bps=0.0, slippage_bps=0.0))
    assert exec_free.total_cost == exec_free.total_commission


# ---- determinism --------------------------------------------------------------

def test_run_is_deterministic():
    bars = make_random_walk_sessions(["AAA", "BBB"], days=5)
    a = run_once(bars, CostModel(half_spread_bps=5.0, slippage_bps=0.0))
    b = run_once(bars, CostModel(half_spread_bps=5.0, slippage_bps=0.0))
    assert a["trades"] == b["trades"]
    assert a["net_pnl"] == b["net_pnl"]
    assert a["cost_toll"] == b["cost_toll"]
