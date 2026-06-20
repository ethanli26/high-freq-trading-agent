"""Honest harness for the intraday-momentum strategy under a realistic cost model.

The question this answers: does open-to-midday intraday momentum survive a realistic
cost model, or do costs kill it before any edge could matter?

DATA HONESTY
------------
With no paid intraday data on hand, this defaults to DETERMINISTIC SYNTHETIC random-walk
minute bars (fixed seed). A random walk has NO built-in edge, so the synthetic strategy
P&L is meaningless as a statement about real alpha — it only proves the mechanics work
and, crucially, lets us read the COST TOLL: how much money the round trips hand to the
market regardless of edge. Point ``--bars PATH`` at real point-in-time minute bars (CSV
with the BarEvent columns) to run the same harness on real data.

What we report and how to read it:
  * Strategy net P&L (synthetic — ignore its sign; it is noise around zero by construction).
  * The COST TOLL in dollars and as % of equity (THIS is the decision number).
  * A cost-sensitivity sweep at 0 / 2 / 5 / 10 bps per side, showing where the toll lands.
  * A verdict: whether the daily cost drag alone already answers "is paid data worth it?".

Read-only research. No live orders. All existing guards untouched.
"""

import argparse
import logging

import numpy as np
import pandas as pd

# Overnight boundaries between sessions look like "gaps" to the real-time robustness
# checks; that is expected for stitched daily sessions, so quiet those warnings here.
logging.getLogger("intraday.engine").setLevel(logging.ERROR)

from intraday.costs import CostAwareExecution, CostModel
from intraday.engine import EventEngine
from intraday.feed import ReplayFeed
from intraday.strategies.momentum import (LongShortPortfolio, MomentumRiskGate, MomentumStrategy)
from risk.position import size_position

STARTING_EQUITY = 100_000.0
STOP_PCT = 0.005
SESSION_OPEN = "09:30"
SESSION_MINUTES = 390          # 09:30 -> 15:59 inclusive
COMMISSION_PER_SHARE = 0.005


def make_random_walk_sessions(symbols, days=20, sigma=0.0008, seed=7,
                              base=100.0, start_date="2024-01-02"):
    """Deterministic random-walk minute bars across several trading days.

    A random walk: close[t] = close[t-1] * (1 + sigma * z). NO drift, NO intraday
    pattern, so momentum has nothing real to find here — by design, so the only thing
    the test can honestly measure is the cost toll. Returns a tidy BarEvent DataFrame.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start_date, periods=days)
    frames = []
    for s_i, symbol in enumerate(symbols):
        price = base * (1.0 + 0.1 * s_i)  # stagger symbols so they are not identical
        for day in sessions:
            start = pd.Timestamp(f"{day.date()} {SESSION_OPEN}")
            timestamps = pd.date_range(start=start, periods=SESSION_MINUTES, freq="1min")
            shocks = rng.normal(0.0, sigma, SESSION_MINUTES)
            closes = price * np.cumprod(1.0 + shocks)
            opens = np.empty(SESSION_MINUTES)
            opens[0] = price
            opens[1:] = closes[:-1]
            frames.append(pd.DataFrame({
                "symbol": symbol, "timestamp": timestamps, "open": opens,
                "high": np.maximum(opens, closes) + 0.01,
                "low": np.minimum(opens, closes) - 0.01,
                "close": closes, "volume": 1000.0}))
            price = closes[-1]  # carry overnight (level only; strategy never trades it)
    return pd.concat(frames, ignore_index=True)


def run_once(bars, cost_model):
    """Run momentum over the bars under one cost model. Returns metrics + the toll."""
    execution = CostAwareExecution(cost_model)
    portfolio = LongShortPortfolio(STARTING_EQUITY, cost_model.commission_per_share)
    engine = EventEngine(
        ReplayFeed(bars), MomentumStrategy(threshold=0.001),
        MomentumRiskGate(stop_pct=STOP_PCT), execution, portfolio,
        fast_period=9, slow_period=21)
    summary = engine.run()
    trades = portfolio.trades
    wins = sum(1 for t in trades if t["pnl"] > 0)
    n = len(trades)
    return {
        "trades": n,
        "net_pnl": portfolio.realized_pnl,
        "win_rate": (wins / n) if n else 0.0,
        "avg_pnl": (portfolio.realized_pnl / n) if n else 0.0,
        "cost_toll": execution.total_cost,
        "commission": execution.total_commission,
        "avg_cost_per_trade": (execution.total_cost / n) if n else 0.0,
        "summary": summary,
    }


def buy_and_hold_intraday(bars, cost_model):
    """Benchmark: buy each symbol at the day's first open, sell at the day's last close.

    Same 1%-risk sizing and same cost model as the strategy, so the comparison is fair.
    """
    total = 0.0
    trades = 0
    for (symbol, day), group in bars.groupby(["symbol", bars["timestamp"].dt.date]):
        group = group.sort_values("timestamp")
        entry_ref, exit_ref = group.iloc[0]["open"], group.iloc[-1]["close"]
        shares, _ = size_position(STARTING_EQUITY, entry_ref, entry_ref * (1.0 - STOP_PCT))
        if shares < 1:
            continue
        buy = cost_model.fill_price(entry_ref, "BUY")
        sell = cost_model.fill_price(exit_ref, "SELL")
        commission = cost_model.commission(shares) * 2
        total += shares * (sell - buy) - commission
        trades += 1
    return {"net_pnl": total, "trades": trades}


def fmt_money(x):
    return f"${x:,.2f}"


def main():
    parser = argparse.ArgumentParser(description="Honest intraday-momentum cost test.")
    parser.add_argument("--bars", help="CSV of real minute bars (BarEvent columns). "
                                        "Omit to use deterministic synthetic data.")
    parser.add_argument("--symbols", nargs="+", default=["AAA", "BBB", "CCC"])
    parser.add_argument("--days", type=int, default=20)
    args = parser.parse_args()

    if args.bars:
        bars = pd.read_csv(args.bars, parse_dates=["timestamp"])
        data_label = f"REAL minute bars from {args.bars}"
        synthetic = False
    else:
        bars = make_random_walk_sessions(args.symbols, days=args.days)
        data_label = (f"SYNTHETIC deterministic random walk "
                      f"({len(args.symbols)} symbols x {args.days} days, seed=7)")
        synthetic = True

    baseline = CostModel(half_spread_bps=5.0, slippage_bps=0.0,
                         commission_per_share=COMMISSION_PER_SHARE)
    base_result = run_once(bars, baseline)
    bh = buy_and_hold_intraday(bars, baseline)

    print("=" * 74)
    print("INTRADAY MOMENTUM — HONEST COST TEST")
    print("=" * 74)
    print(f"Data:            {data_label}")
    print(f"Starting equity: {fmt_money(STARTING_EQUITY)}")
    print(f"Cost model:      half-spread+slippage applied EACH side + "
          f"${COMMISSION_PER_SHARE}/share commission")
    print(f"Baseline:        5 bps/side  (round trip = {baseline.round_trip_bps:.0f} bps + commissions)")
    print()
    print(f"Round-trip trades:        {base_result['trades']}")
    print(f"Per-day win rate:         {base_result['win_rate']*100:.1f}%")
    print(f"Avg P&L / trade (net):    {fmt_money(base_result['avg_pnl'])}")
    print(f"Strategy net P&L:         {fmt_money(base_result['net_pnl'])}"
          + ("   <- SYNTHETIC: noise, ignore the sign" if synthetic else ""))
    print(f"Buy & hold intraday P&L:  {fmt_money(bh['net_pnl'])}  ({bh['trades']} day-trades)")
    print(f"Cash (do nothing):        {fmt_money(0.0)}")
    print()

    # ---- COST SENSITIVITY: the toll at several spread/slippage assumptions ----
    print("-" * 74)
    print("COST-SENSITIVITY TABLE  (the toll is what we actually decide on)")
    print("-" * 74)
    header = f"{'bps/side':>9} | {'trades':>6} | {'cost toll':>13} | {'% equity':>8} | {'avg/trade':>10} | {'net P&L':>13}"
    print(header)
    print("-" * len(header))
    rows = []
    for side_bps in (0.0, 2.0, 5.0, 10.0):
        cm = CostModel(half_spread_bps=side_bps, slippage_bps=0.0,
                       commission_per_share=COMMISSION_PER_SHARE)
        r = run_once(bars, cm)
        rows.append((side_bps, r))
        print(f"{side_bps:>9.0f} | {r['trades']:>6} | {fmt_money(r['cost_toll']):>13} | "
              f"{r['cost_toll']/STARTING_EQUITY*100:>7.2f}% | {fmt_money(r['avg_cost_per_trade']):>10} | "
              f"{fmt_money(r['net_pnl']):>13}")
    print("-" * len(header))
    print("(bps/side = half-spread+slippage charged on EACH entry and EACH exit; "
          f"commission ${COMMISSION_PER_SHARE}/share is included in the toll at every row.)")
    print()

    # ---- VERDICT ----
    toll5 = dict(rows)[5.0]["cost_toll"]
    toll5_pct = toll5 / STARTING_EQUITY * 100
    per_trade5 = dict(rows)[5.0]["avg_cost_per_trade"]
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"At a realistic 5 bps/side, {base_result['trades']} round trips hand the market "
          f"{fmt_money(toll5)}")
    print(f"  = {toll5_pct:.2f}% of starting equity, ~{fmt_money(per_trade5)} per trade, "
          f"BEFORE any edge.")
    print()
    print("This is the toll the strategy must out-earn every single day just to break even.")
    print("Even at 2 bps/side the round-trip drag is already material; at 10 bps/side it is")
    print("punishing. The toll grows linearly with trade count and spread — it is paid")
    print("whether or not the signal is right.")
    print()
    if synthetic:
        print("HONEST NOTE: the data is synthetic, so this proves NOTHING about whether the")
        print("momentum signal has real edge. It does, however, quantify the COST HURDLE that")
        print("any real edge has to clear. Decide on paid intraday data by asking: is the")
        print("expected gross edge per round trip plausibly larger than the per-trade toll")
        print(f"above (~{fmt_money(per_trade5)} at 5 bps/side)? If the daily cost drag already")
        print("rivals any edge a once-a-day signal could realistically produce, the answer is")
        print("'don't buy the data' — the cost model already told you. Confirming actual edge")
        print("would require paid point-in-time intraday bars WITH measured per-name spreads;")
        print("synthetic data cannot.")
    else:
        print("Compare the per-trade toll above against the strategy's measured gross edge per")
        print("round trip on this real data. If the toll is a large fraction of (or exceeds)")
        print("the gross edge, the strategy is not viable net of costs regardless of hit rate.")
    print("=" * 74)


if __name__ == "__main__":
    main()
