"""Buy-and-hold S&P 500 benchmark, apples-to-apples with the strategy.

The benchmark invests the SAME starting capital in SPY (dividend-adjusted close) on
the strategy's first equity date and holds to the end, producing a daily equity
series aligned to the strategy's equity-curve dates. Same date range, same starting
capital, and split/dividend-adjusted prices on both sides — so the only difference
being measured is the active strategy versus passive holding. Read-only research.
"""

import logging

import pandas as pd

log = logging.getLogger(__name__)


def buy_and_hold(spy_close: pd.Series, strategy_equity: pd.Series,
                 starting_equity: float) -> pd.Series:
    """Daily buy-and-hold SPY equity aligned to the strategy's equity dates.

    Holds a fixed (fractional) SPY position bought at the first strategy date; with
    dividend-adjusted close this is a total-return hold. Returns a Series named
    ``benchmark`` on exactly ``strategy_equity.index``.
    """
    dates = strategy_equity.index
    spy = spy_close.reindex(dates).ffill()  # align to strategy dates; fill rare gaps
    first_price = spy.iloc[0]
    if pd.isna(first_price) or first_price <= 0:
        raise ValueError("SPY adjusted close is missing at the strategy's first date.")

    benchmark = starting_equity * spy / first_price
    return benchmark.rename("benchmark")


def risk_matched_blend(spy_close: pd.Series, dates: pd.DatetimeIndex,
                       starting_equity: float, spy_weight: float, rf: float) -> pd.Series:
    """Constant-mix SPY/cash benchmark scaled to a target market exposure.

    A daily-rebalanced blend holding ``spy_weight`` in SPY and the rest in cash that
    earns the risk-free rate: ``daily_return = w*spy_return + (1-w)*rf/252``. With a
    weight chosen to match the strategy's volatility, this answers "did the active
    machinery beat a trivial passive mix at the SAME risk level?" Aligned to ``dates``.
    """
    spy = spy_close.reindex(dates).ffill()
    spy_return = spy.pct_change().fillna(0.0)
    blend_return = spy_weight * spy_return + (1.0 - spy_weight) * (rf / 252.0)
    equity = starting_equity * (1.0 + blend_return).cumprod()
    return equity.rename("blend")


def vol_matched_weight(strategy_equity: pd.Series, benchmark_equity: pd.Series) -> float:
    """SPY weight whose blend vol ≈ the strategy's (cash vol ≈ 0, so w ≈ vol ratio).

    Clipped to [0, 1].
    """
    strat_vol = strategy_equity.pct_change().dropna().std(ddof=1)
    bench_vol = benchmark_equity.pct_change().dropna().std(ddof=1)
    if bench_vol == 0:
        return 0.0
    return float(min(1.0, max(0.0, strat_vol / bench_vol)))
