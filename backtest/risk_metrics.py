"""Professional risk-adjusted performance metrics from a daily equity curve.

All metrics are computed from daily equity. Annualization uses 252 trading days.
Returns are simple daily returns; the risk-free rate is annual and converted to a
daily rate as ``rf / 252``. Each function documents its formula.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_RISK_FREE = 0.04  # ~4% annual


def daily_returns(equity: pd.Series) -> pd.Series:
    """Simple daily returns of an equity curve."""
    return equity.pct_change().dropna()


def cagr(equity: pd.Series) -> float | None:
    """Compound annual growth rate: (end/start)^(1/years) - 1, calendar years."""
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return None
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else None


def annualized_volatility(returns: pd.Series) -> float:
    """Std of daily returns scaled by sqrt(252)."""
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf: float = DEFAULT_RISK_FREE) -> float | None:
    """Annualized Sharpe: mean daily excess / std daily * sqrt(252)."""
    excess = returns - rf / TRADING_DAYS
    std = returns.std(ddof=1)
    if std == 0:
        return None
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf: float = DEFAULT_RISK_FREE) -> float | None:
    """Annualized Sortino: mean daily excess / downside deviation * sqrt(252).

    Downside deviation is sqrt(mean of squared negative excess returns) over all
    periods (target = the risk-free rate).
    """
    excess = returns - rf / TRADING_DAYS
    downside = excess.clip(upper=0.0)
    downside_dev = np.sqrt((downside ** 2).mean())
    if downside_dev == 0:
        return None
    return float(excess.mean() / downside_dev * np.sqrt(TRADING_DAYS))


def max_drawdown_detail(equity: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """Max peak-to-trough drawdown (positive fraction) with its peak/trough dates."""
    equity = equity.dropna()
    if equity.empty:
        return 0.0, None, None
    underwater = equity / equity.cummax() - 1.0
    trough = underwater.idxmin()
    peak = equity.loc[:trough].idxmax()
    return float(-underwater.min()), peak, trough


def calmar(equity: pd.Series) -> float | None:
    """Calmar ratio: CAGR / max drawdown."""
    growth = cagr(equity)
    mdd, _, _ = max_drawdown_detail(equity)
    if growth is None or mdd == 0:
        return None
    return growth / mdd


def pct_positive_months(equity: pd.Series) -> float | None:
    """Fraction of calendar months with a positive return."""
    monthly = equity.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return None
    return float((monthly > 0).mean())


def compute_metrics(equity: pd.Series, rf: float = DEFAULT_RISK_FREE) -> dict:
    """Full single-curve metric set as a dict."""
    equity = equity.dropna()
    returns = daily_returns(equity)
    mdd, peak, trough = max_drawdown_detail(equity)
    return {
        "final_equity": float(equity.iloc[-1]) if len(equity) else None,
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else None,
        "cagr": cagr(equity),
        "ann_volatility": annualized_volatility(returns),
        "sharpe": sharpe(returns, rf),
        "sortino": sortino(returns, rf),
        "calmar": calmar(equity),
        "max_drawdown": mdd,
        "dd_peak": peak,
        "dd_trough": trough,
        "best_day": float(returns.max()) if len(returns) else None,
        "worst_day": float(returns.min()) if len(returns) else None,
        "pct_positive_months": pct_positive_months(equity),
    }


def relative_metrics(strategy: pd.Series, benchmark: pd.Series,
                     rf: float = DEFAULT_RISK_FREE) -> dict:
    """Strategy-vs-benchmark stats: return correlation, beta, and annualized alpha.

    Beta = cov(strategy, benchmark) / var(benchmark). CAPM alpha (daily) =
    mean(strategy_excess) - beta * mean(benchmark_excess), annualized by * 252.
    """
    paired = pd.concat([daily_returns(strategy), daily_returns(benchmark)],
                       axis=1, keys=["s", "b"]).dropna()
    s, b = paired["s"], paired["b"]
    var_b = b.var(ddof=1)
    beta = float(s.cov(b) / var_b) if var_b > 0 else None
    rf_daily = rf / TRADING_DAYS
    alpha = None
    if beta is not None:
        alpha = float(((s - rf_daily).mean() - beta * (b - rf_daily).mean()) * TRADING_DAYS)
    return {"correlation": float(s.corr(b)), "beta": beta, "alpha": alpha}
