"""Factor tests: NCSKEW/DUVOL formula fidelity and the look-ahead-safety property."""

import numpy as np
import pandas as pd

import factors  # noqa: F401  (registers the factors on import)
from factors.base import FactorData, all_factors
from factors.library import _duvol_stat, _ncskew_stat

# A fixed return window with both up and down days for the formula checks.
FIXED = np.array([0.01, -0.02, 0.03, -0.01, 0.05, -0.04, 0.02, -0.03])


def test_ncskew_symmetric_window_is_zero():
    symmetric = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    assert abs(_ncskew_stat(symmetric)[0]) < 1e-12


def test_ncskew_matches_hand_formula():
    r = FIXED
    n = len(r)
    dev = r - r.mean()
    m2, m3 = (dev ** 2).sum(), (dev ** 3).sum()
    expected = -(n * (n - 1) ** 1.5 * m3) / ((n - 1) * (n - 2) * m2 ** 1.5)
    assert abs(_ncskew_stat(r.reshape(1, -1))[0] - expected) < 1e-12


def test_duvol_matches_hand_formula():
    r = FIXED
    rbar = r.mean()
    dev = r - rbar
    down, up = r < rbar, r > rbar
    expected = np.log(((up.sum() - 1) * (dev[down] ** 2).sum())
                      / ((down.sum() - 1) * (dev[up] ** 2).sum()))
    assert abs(_duvol_stat(r.reshape(1, -1))[0] - expected) < 1e-12


def test_every_factor_is_look_ahead_safe():
    """A factor's value at t must not change when bars AFTER t are removed."""
    rng = np.random.default_rng(1)
    n, syms = 600, ["A", "B", "C", "D", "E"]
    close = (pd.DataFrame({s: 50 + np.cumsum(rng.normal(0, 1, n)) for s in syms})).abs() + 10
    market = (50 + pd.Series(np.cumsum(rng.normal(0, 1, n)))).abs() + 10
    data = FactorData(open=close * 0.99, high=close * 1.02, low=close * 0.97, close=close,
                      volume=pd.DataFrame({s: rng.uniform(1e6, 5e6, n) for s in syms}), market=market)
    cutoff = 400
    truncated = FactorData(data.open.iloc[:cutoff + 1], data.high.iloc[:cutoff + 1],
                           data.low.iloc[:cutoff + 1], data.close.iloc[:cutoff + 1],
                           data.volume.iloc[:cutoff + 1], market=data.market.iloc[:cutoff + 1])
    for name, cls in all_factors().items():
        factor = cls()
        if getattr(factor, "point_in_time_provider", False):
            continue  # PIT stubs raise on purpose; not runnable on free data
        full = factor.compute(data).iloc[cutoff]
        trunc = factor.compute(truncated).iloc[cutoff]
        diff = (full - trunc).abs().max()
        assert pd.isna(diff) or diff < 1e-9, f"{name} leaks future data"
