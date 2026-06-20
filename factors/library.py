"""A small library of price/volume factors testable on free data.

Every factor is computed as a backward-looking panel operation, so the value on
date ``t`` uses only data up to and including ``t`` (look-ahead safe — see comments).
The WorldQuant-style alphas are attributed to Kakushadze (2016), "101 Formulaic
Alphas", arXiv:1601.00991, and kept deliberately simple.

The fundamental factor family (profitability, value, earnings growth) lives in
``factors/fundamentals.py`` — it needs the point-in-time provider (Sharadar) and so is
kept out of this free-data price/volume library.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from factors.base import Factor, FactorData, register


def _rolling_columnwise(returns: pd.DataFrame, window: int, stat_fn) -> pd.DataFrame:
    """Apply a vectorized ``stat_fn((m, window)) -> (m,)`` over each trailing window.

    LOOK-AHEAD GUARD: window k = ``col[k:k+window]`` is placed at output row
    ``window-1+k`` (its LAST element's date), so every value uses only data <= t.
    """
    out = np.full(returns.shape, np.nan)
    arr = returns.to_numpy(dtype=float)
    if arr.shape[0] >= window:
        for j in range(arr.shape[1]):
            windows = sliding_window_view(arr[:, j], window)  # (T-window+1, window), trailing
            out[window - 1:, j] = stat_fn(windows)
    return pd.DataFrame(out, index=returns.index, columns=returns.columns)


def _rolling_capm(stock_returns: pd.DataFrame, market_returns: pd.Series, window: int):
    """Trailing CAPM beta and residual std of each stock vs the market.

    Vectorized via rolling sums (ddof=1). residual variance = var(r) - cov^2/var(m).
    LOOK-AHEAD GUARD: every rolling sum ends at t (uses only data <= t).
    """
    r, m, n = stock_returns, market_returns, window
    rm = r.mul(m, axis=0)
    sum_r, sum_m = r.rolling(n).sum(), m.rolling(n).sum()
    sum_rr, sum_mm = (r * r).rolling(n).sum(), (m * m).rolling(n).sum()
    sum_rm = rm.rolling(n).sum()

    cov = (sum_rm - sum_r.mul(sum_m, axis=0) / n) / (n - 1)
    var_m = (sum_mm - sum_m ** 2 / n) / (n - 1)
    var_r = (sum_rr - sum_r ** 2 / n) / (n - 1)
    beta = cov.div(var_m, axis=0)
    resid_std = np.sqrt((var_r - (cov ** 2).div(var_m, axis=0)).clip(lower=0))
    return beta, resid_std


@register
class Momentum12_1(Factor):
    """12-1 momentum: return from ~12 months ago to ~1 month ago (skip last month).

    Higher = recent winner (the momentum premium predicts higher forward returns).
    """

    name = "momentum_12_1"
    category = "price"
    requires = ("close",)

    def compute(self, data: FactorData) -> pd.DataFrame:
        # LOOK-AHEAD GUARD: shift(21)/shift(252) use only PAST closes (t-21, t-252).
        return data.close.shift(21) / data.close.shift(252) - 1.0


@register
class ShortTermReversal(Factor):
    """Short-term reversal: negative of the last week's (5-session) return.

    Higher = recent loser (expected to bounce); positive IC = reversal present.
    """

    name = "short_term_reversal"
    category = "price"
    requires = ("close",)

    def compute(self, data: FactorData) -> pd.DataFrame:
        # LOOK-AHEAD GUARD: close.shift(5) is the price 5 sessions ago (past only).
        return -(data.close / data.close.shift(5) - 1.0)


@register
class LowVolatility(Factor):
    """Trailing 60-day volatility, signed so HIGHER = LOWER vol ("low-vol anomaly").

    SIGN: by convention lower realized vol is "better", so the factor value is the
    NEGATIVE of 60-day return volatility. A positive IC therefore means the low-vol
    anomaly is present (calmer names earn higher forward returns).
    """

    name = "low_volatility"
    category = "price"
    requires = ("close",)

    def compute(self, data: FactorData) -> pd.DataFrame:
        daily_returns = data.close.pct_change(fill_method=None)  # don't forward-fill gaps
        # LOOK-AHEAD GUARD: rolling(60) ends at t, using returns through t only.
        return -daily_returns.rolling(60).std()


@register
class Alpha101(Factor):
    """WorldQuant Alpha#101: (close - open) / ((high - low) + 0.001).

    Intraday close strength relative to the day's range. Source: Kakushadze (2016),
    arXiv:1601.00991, Alpha#101. Uses day-t OHLC (known at t's close).
    """

    name = "wq_alpha_101"
    category = "price"
    requires = ("open", "high", "low", "close")

    def compute(self, data: FactorData) -> pd.DataFrame:
        # LOOK-AHEAD GUARD: only date-t OHLC, all known at t's close.
        return (data.close - data.open) / ((data.high - data.low) + 0.001)


@register
class Alpha12(Factor):
    """WorldQuant Alpha#12: sign(delta(volume,1)) * (-1 * delta(close,1)).

    A volume-confirmed 1-day reversal. Source: Kakushadze (2016), arXiv:1601.00991,
    Alpha#12. Uses t and t-1 only.
    """

    name = "wq_alpha_12"
    category = "volume"
    requires = ("close", "volume")

    def compute(self, data: FactorData) -> pd.DataFrame:
        import numpy as np

        # LOOK-AHEAD GUARD: diff(1) compares t to t-1 (past), never t+1.
        return pd.DataFrame(np.sign(data.volume.diff()), index=data.volume.index,
                            columns=data.volume.columns) * (-data.close.diff())


# --- A. Crash-risk factors (Chen-Hong-Stein; Orient Securities A-share report) ----

def _ncskew_stat(windows: np.ndarray) -> np.ndarray:
    """NCSKEW per trailing window (rows): negative coefficient of skewness."""
    n = windows.shape[1]
    dev = windows - windows.mean(axis=1, keepdims=True)
    m2 = (dev ** 2).sum(axis=1)
    m3 = (dev ** 3).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = -(n * (n - 1) ** 1.5 * m3) / ((n - 1) * (n - 2) * m2 ** 1.5)
    out[(m2 <= 0) | np.isnan(windows).any(axis=1)] = np.nan
    return out


def _duvol_stat(windows: np.ndarray) -> np.ndarray:
    """DUVOL per trailing window: log(down-day variance / up-day variance)."""
    rbar = windows.mean(axis=1, keepdims=True)
    dev = windows - rbar
    down, up = windows < rbar, windows > rbar
    n_down, n_up = down.sum(axis=1), up.sum(axis=1)
    ss_down = np.where(down, dev ** 2, 0.0).sum(axis=1)
    ss_up = np.where(up, dev ** 2, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.log(((n_up - 1) * ss_down) / ((n_down - 1) * ss_up))
    bad = (n_up < 2) | (n_down < 2) | (ss_up <= 0) | (ss_down <= 0) | np.isnan(windows).any(axis=1)
    out[bad] = np.nan
    return out


@register
class NCSKEW(Factor):
    """Negative coefficient of skewness of trailing 60-day daily returns.

    Source: Chen, Hong & Stein (2001), JFE; applied to A-shares in the Orient
    Securities crash-risk report (the internship factor set).
    Rationale: a crash-risk measure — higher NCSKEW = more left-tail (crash) prone.
    Expected IC sign: ambiguous/negative — crash-prone names tend to underperform
    (a negative crash-risk premium was reported in A-shares).
    Validation: US (CHS 2001) and Chinese A-shares (Orient); this run is US large-cap.
    """

    name = "ncskew"
    category = "price"
    requires = ("close",)

    def compute(self, data: FactorData) -> pd.DataFrame:
        returns = data.close.pct_change(fill_method=None)
        # LOOK-AHEAD GUARD: each 60-window ends at t (see _rolling_columnwise).
        return _rolling_columnwise(returns, 60, _ncskew_stat)


@register
class DUVOL(Factor):
    """Down-to-up volatility: log ratio of down-day to up-day return variance (60d).

    Source: Chen, Hong & Stein (2001), JFE; Orient Securities A-share report.
    Formula: DUVOL = log[ (n_up - 1) * sum_down(r-rbar)^2 / ((n_down - 1) * sum_up(r-rbar)^2) ],
    with up/down defined relative to the window mean rbar.
    Sign: higher DUVOL = down-day variance dominates = more crash-prone (like NCSKEW).
    Expected IC sign: ambiguous/negative (crash risk).
    Validation: US (CHS 2001) and Chinese A-shares (Orient); this run is US large-cap.
    """

    name = "duvol"
    category = "price"
    requires = ("close",)

    def compute(self, data: FactorData) -> pd.DataFrame:
        returns = data.close.pct_change(fill_method=None)
        # LOOK-AHEAD GUARD: each 60-window ends at t (see _rolling_columnwise).
        return _rolling_columnwise(returns, 60, _duvol_stat)


@register
class IVCAPM(Factor):
    """Idiosyncratic volatility: std of CAPM residuals over a trailing ~120 days.

    Source: Ang, Hodrick, Xing & Zhang (2006), JF ("idiosyncratic volatility puzzle");
    the IVmonthly proxy in the Orient Securities report. Regress each stock's daily
    returns on SPY's daily returns over 120 days and take the residual std.
    Rationale/sign: EXPECTED NEGATIVE IC — high idiosyncratic vol predicts LOWER
    forward returns (the IVOL puzzle).
    Validation: US (Ang et al. 2006) and internationally (Ang et al. 2009); A-shares.
    """

    name = "ivol_capm"
    category = "price"
    requires = ("close",)

    def compute(self, data: FactorData) -> pd.DataFrame:
        if data.market is None:
            raise ValueError("IVCAPM needs FactorData.market (benchmark series).")
        stock_returns = data.close.pct_change(fill_method=None)
        market_returns = data.market.pct_change(fill_method=None)
        # LOOK-AHEAD GUARD: rolling CAPM uses only sums ending at t.
        _, resid_std = _rolling_capm(stock_returns, market_returns, 120)
        return resid_std


# --- B. Academically supported public factors -------------------------------------

@register
class Momentum6_1(Factor):
    """6-1 momentum: return from ~6 months ago to ~1 month ago (skip last month).

    Source: Jegadeesh & Titman (1993), JF. Rationale: medium-horizon return
    continuation. Expected IC sign: POSITIVE (winners keep winning).
    Validation: strong in US and most international markets (notably weak in Japan).
    """

    name = "momentum_6_1"
    category = "price"
    requires = ("close",)

    def compute(self, data: FactorData) -> pd.DataFrame:
        # LOOK-AHEAD GUARD: shift(21)/shift(126) use only past closes.
        return data.close.shift(21) / data.close.shift(126) - 1.0


@register
class MaxDailyReturn(Factor):
    """MAX: the single highest daily return over the past ~21 sessions (lottery effect).

    Source: Bali, Cakici & Whitelaw (2011), JFE ("Maxing out"). Rationale: lottery-like
    stocks are overpriced. Expected IC sign: NEGATIVE (high-MAX names underperform).
    Validation: US (Bali et al. 2011) and replicated across many international markets.
    """

    name = "max_daily_return"
    category = "price"
    requires = ("close",)

    def compute(self, data: FactorData) -> pd.DataFrame:
        returns = data.close.pct_change(fill_method=None)
        # LOOK-AHEAD GUARD: rolling(21).max ends at t (past month only).
        return returns.rolling(21).max()


@register
class BetaLow(Factor):
    """Betting-against-beta: NEGATIVE of trailing ~250-day CAPM beta vs SPY.

    Source: Frazzini & Pedersen (2014), JFE. Rationale: leverage-constrained investors
    bid up high-beta names, so low beta earns higher risk-adjusted returns. SIGN: the
    factor is ``-beta`` so HIGHER = LOWER beta = attractive; expected IC sign POSITIVE.
    Validation: US and globally across asset classes (Frazzini-Pedersen 2014).
    """

    name = "beta_low"
    category = "price"
    requires = ("close",)

    def compute(self, data: FactorData) -> pd.DataFrame:
        if data.market is None:
            raise ValueError("BetaLow needs FactorData.market (benchmark series).")
        stock_returns = data.close.pct_change(fill_method=None)
        market_returns = data.market.pct_change(fill_method=None)
        # LOOK-AHEAD GUARD: rolling beta uses only sums ending at t.
        beta, _ = _rolling_capm(stock_returns, market_returns, 250)
        return -beta  # encode "low beta = attractive" so positive IC = BAB holds


@register
class ReturnSkewness(Factor):
    """Plain trailing skewness of daily returns over ~250 sessions.

    Source: Harvey & Siddique (2000), JF; idiosyncratic-skewness work (Boyer, Mitton &
    Vorkink 2010). Rationale: investors pay for lottery-like positive skew, so
    high-skew names are overpriced. Expected IC sign: NEGATIVE (positive skew -> lower
    forward returns). Validation: US; mixed internationally.
    """

    name = "return_skewness"
    category = "price"
    requires = ("close",)

    def compute(self, data: FactorData) -> pd.DataFrame:
        returns = data.close.pct_change(fill_method=None)
        # LOOK-AHEAD GUARD: rolling(250).skew ends at t (past only).
        return returns.rolling(250).skew()
