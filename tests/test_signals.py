"""Unit tests for the entry signals: breakout window, pullback rule, earnings guard."""

import numpy as np
import pandas as pd

import config
from signals.breakout import BreakoutStrategy
from signals.earnings_drift import EarningsDriftStrategy
from signals.entry import breakout_signal
from signals.pullback import PullbackStrategy


def _ohlc(closes):
    return pd.DataFrame({"Open": closes, "High": [c + 1 for c in closes],
                         "Low": [c - 1 for c in closes], "Close": closes})


def test_breakout_fires_above_prior_high():
    closes = [100.0] * (config.BREAKOUT_LOOKBACK + 4) + [105.0]
    assert breakout_signal(_ohlc(closes)) is True


def test_breakout_requires_strictly_greater():
    closes = [100.0] * (config.BREAKOUT_LOOKBACK + 4) + [100.0]
    assert breakout_signal(_ohlc(closes)) is False


def test_breakout_excludes_today_but_counts_yesterday():
    # Yesterday (110) is inside the lookback window and above today (105) -> no breakout.
    closes = [100.0] * (config.BREAKOUT_LOOKBACK + 3) + [110.0, 105.0]
    assert breakout_signal(_ohlc(closes)) is False


def test_breakout_insufficient_history():
    assert breakout_signal(_ohlc([1.0, 2.0, 3.0])) is False


def test_breakout_strategy_matches_canonical_signal():
    closes = [100.0] * (config.BREAKOUT_LOOKBACK + 4) + [105.0]
    bars = _ohlc(closes)
    assert BreakoutStrategy().generate_signal(bars) == breakout_signal(bars)


def test_pullback_downtrend_never_fires():
    closes = list(np.linspace(200.0, 100.0, 120))  # strict downtrend: close always below its MA
    bars = pd.DataFrame({"Open": closes, "High": [c + 1 for c in closes],
                         "Low": [c - 1 for c in closes], "Close": closes})
    assert not PullbackStrategy().signal_series(bars).any()


def test_pullback_matches_documented_rule():
    # The strategy's signal must equal an independent re-implementation of its rule.
    rng = np.random.default_rng(0)
    n = 220
    close = pd.Series(100.0 + np.cumsum(rng.normal(0.05, 1.0, n)))
    low = close - rng.uniform(0.0, 1.5, n)
    bars = pd.DataFrame({"Open": close.shift(1).fillna(close.iloc[0]),
                         "High": close + 1.0, "Low": low, "Close": close})
    got = PullbackStrategy().signal_series(bars)

    ma = close.rolling(config.PULLBACK_MA).mean()
    k = config.PULLBACK_BOUNCE_LOOKBACK
    uptrend = close > ma
    touched = (low <= ma * (1 + config.PULLBACK_TOUCH_PCT)).rolling(k).sum() >= 1
    bounce = (close > close.shift(1)) & (close > close.shift(k))
    expected = (uptrend & touched & bounce).fillna(False)
    pd.testing.assert_series_equal(got.reset_index(drop=True), expected.reset_index(drop=True),
                                   check_names=False)


def _earnings_bars(dates, closes, opens):
    close = pd.Series(closes, index=dates, dtype=float)
    open_ = pd.Series(opens, index=dates, dtype=float)
    return pd.DataFrame({"Open": open_, "High": close + 1, "Low": close - 1, "Close": close})


def test_earnings_never_enters_on_or_before_report():
    dates = pd.bdate_range("2020-01-01", periods=12)
    # report on idx 3; the post-report session (idx 4) closes UP (108 > 104).
    closes = [100, 101, 102, 103, 108, 109, 110, 111, 112, 113, 114, 115]
    opens = [100, 101, 102, 103, 104, 109, 110, 111, 112, 113, 114, 115]
    bars = _earnings_bars(dates, closes, opens)
    report_date = dates[3]
    earnings = {"AAA": pd.DataFrame({"report_date": [report_date], "surprise": [0.10]})}

    signal = EarningsDriftStrategy(earnings).signal_series(bars, "AAA")
    fired = list(signal[signal].index)
    assert fired, "expected an earnings-drift signal"
    for signal_date in fired:
        s = bars.index.get_loc(signal_date)
        assert signal_date > report_date           # signal strictly after the report
        assert bars.index[s + 1] > report_date      # and the engine's fill is later still


def test_earnings_skips_small_surprise():
    dates = pd.bdate_range("2020-01-01", periods=12)
    closes = [100, 101, 102, 103, 108, 109, 110, 111, 112, 113, 114, 115]
    opens = [100, 101, 102, 103, 104, 109, 110, 111, 112, 113, 114, 115]
    bars = _earnings_bars(dates, closes, opens)
    earnings = {"AAA": pd.DataFrame({"report_date": [dates[3]], "surprise": [0.01]})}  # < 5%
    assert not EarningsDriftStrategy(earnings).signal_series(bars, "AAA").any()


def test_earnings_requires_confirm_up():
    dates = pd.bdate_range("2020-01-01", periods=12)
    closes = [100, 101, 102, 103, 100, 99, 98, 97, 96, 95, 94, 93]  # post-report session closes DOWN
    opens = [100, 101, 102, 103, 105, 100, 99, 98, 97, 96, 95, 94]   # idx4: close 100 < open 105
    bars = _earnings_bars(dates, closes, opens)
    earnings = {"AAA": pd.DataFrame({"report_date": [dates[3]], "surprise": [0.10]})}
    assert not EarningsDriftStrategy(earnings).signal_series(bars, "AAA").any()
