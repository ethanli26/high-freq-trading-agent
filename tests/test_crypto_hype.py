"""Tests for the crypto hype-momentum pilot (offline, deterministic).

Covers: hype-signal correctness (breakout AND volume surge), look-ahead safety,
the illiquidity-scaled cost model arithmetic, the liquidity + wash-volume screens,
and run determinism. Also asserts the engine's per-coin slippage hook is honored.
"""

import numpy as np
import pandas as pd

from backtest.regime import compute_regime
from crypto import universe as crypto_universe
from crypto.costs import CryptoCostModel, build_slippage_fn
from data.crypto_provider import (
    average_dollar_volume,
    liquidity_filter,
    make_synthetic_crypto,
    wash_volume_score,
)
from crypto.run_hype_test import run_pilot
from strategies.crypto_hype import CryptoHypeStrategy


def _frame(closes, volumes):
    n = len(closes)
    idx = pd.bdate_range("2022-01-03", periods=n)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "Open": closes, "High": closes + 0.01, "Low": closes - 0.01,
        "Close": closes, "Volume": np.asarray(volumes, dtype=float)}, index=idx)


# ---- signal correctness -------------------------------------------------------

def test_breakout_with_volume_surge_fires():
    """A new 20-day high on >2x average volume triggers the entry."""
    closes = [100.0] * 25 + [110.0]          # day 26 breaks the prior 100 high
    volumes = [1000.0] * 25 + [5000.0]       # ...on a 5x volume surge
    strat = CryptoHypeStrategy(lookback=20, volume_mult=2.0, vol_lookback=20)
    assert strat.generate_signal(_frame(closes, volumes)) is True


def test_breakout_without_volume_does_not_fire():
    """A breakout on ordinary volume is NOT a hype entry."""
    closes = [100.0] * 25 + [110.0]
    volumes = [1000.0] * 26                   # no surge
    strat = CryptoHypeStrategy(lookback=20, volume_mult=2.0, vol_lookback=20)
    assert strat.generate_signal(_frame(closes, volumes)) is False


def test_volume_surge_without_breakout_does_not_fire():
    """A volume spike with no new high is NOT an entry (direction is up-breakout only)."""
    closes = [100.0] * 26                      # never makes a new high
    volumes = [1000.0] * 25 + [9000.0]
    strat = CryptoHypeStrategy(lookback=20, volume_mult=2.0, vol_lookback=20)
    assert strat.generate_signal(_frame(closes, volumes)) is False


def test_signal_series_matches_generate_signal():
    """The vectorized series equals the canonical rule evaluated bar by bar."""
    rng = np.random.default_rng(0)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.03, 120))
    volumes = rng.uniform(800, 6000, 120)
    bars = _frame(closes, volumes)
    strat = CryptoHypeStrategy()
    series = strat.signal_series(bars)
    for i in range(len(bars)):
        expected = strat.generate_signal(bars.iloc[: i + 1])
        assert bool(series.iloc[i]) == expected


# ---- look-ahead safety --------------------------------------------------------

def test_signal_is_causal():
    """Mutating a FUTURE bar must not change an earlier bar's signal value."""
    rng = np.random.default_rng(1)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.03, 80))
    volumes = rng.uniform(800, 6000, 80)
    bars = _frame(closes, volumes)
    strat = CryptoHypeStrategy()
    base = strat.signal_series(bars).to_numpy()

    tampered = bars.copy()
    tampered.iloc[-1, tampered.columns.get_loc("Close")] *= 5.0   # huge future breakout
    tampered.iloc[-1, tampered.columns.get_loc("Volume")] *= 50.0
    after = strat.signal_series(tampered).to_numpy()
    assert np.array_equal(base[:-1], after[:-1])  # earlier signals unchanged


# ---- cost model arithmetic ----------------------------------------------------

def test_cost_scales_with_illiquidity():
    """A thinner coin pays strictly more per side than a liquid one."""
    model = CryptoCostModel(taker_fee_bps=10.0, base_spread_bps=10.0,
                            ref_dollar_volume=50_000_000.0)
    liquid = model.per_side_bps(50_000_000.0)       # at the reference
    thin = model.per_side_bps(500_000.0)            # 1/100th the ADV
    assert liquid == 20.0                            # taker 10 + spread 10*1
    assert thin > liquid
    assert model.round_trip_bps(500_000.0) == 2.0 * thin


def test_cost_factor_clamped_and_stress_knob():
    model = CryptoCostModel(max_illiquidity_factor=25.0)
    assert model.illiquidity_factor(1.0) == 25.0    # clamped for a near-zero-ADV coin
    # stress=0 removes the spread term, leaving only the taker fee.
    assert CryptoCostModel(stress_multiplier=0.0).per_side_bps(1_000_000.0) == 10.0


def test_slippage_fn_differentiates_coins():
    bars = make_synthetic_crypto(["BTC", "SC01", "SCWASH"], days=400, seed=3)
    model = CryptoCostModel()
    fn = build_slippage_fn(bars, model, lookback=30)
    # An unknown symbol falls back to the liquid reference cost.
    assert fn("UNKNOWN") == model.per_side_fraction(model.ref_dollar_volume)
    assert fn("SC01") > 0


# ---- liquidity & wash screens -------------------------------------------------

def test_wash_coin_scores_high_real_coin_low():
    bars = make_synthetic_crypto(["BTC", "SC01", "SCWASH"], days=500, seed=5)
    assert wash_volume_score(bars["SCWASH"]) >= 0.6   # flat price + huge volume
    assert wash_volume_score(bars["SC01"]) < 0.6      # real volume moves price


def test_liquidity_filter_drops_short_and_illiquid():
    bars = make_synthetic_crypto(["BTC", "SC01", "SC02"], days=500, seed=6)
    bars["SHORTY"] = bars["SC01"].iloc[:50].copy()     # too little history
    kept, skips = liquidity_filter(bars, min_dollar_volume=1_000_000.0, min_history_days=200)
    assert "SHORTY" in skips["short"]
    assert all(average_dollar_volume(f) >= 1_000_000.0 for f in kept.values())


# ---- determinism --------------------------------------------------------------

def test_pilot_run_is_deterministic():
    bars = make_synthetic_crypto(["BTC"] + [f"SC{i:02d}" for i in range(1, 10)],
                                 days=500, seed=7)
    tradable, _ = crypto_universe.build_universe(bars, min_dollar_volume=1_000_000.0,
                                                 min_history_days=200)
    smap = crypto_universe.sector_map(tradable.keys())
    btc_close = bars[crypto_universe.BENCHMARK]["Close"]
    regime = compute_regime(btc_close)
    model = CryptoCostModel()
    eq_a, tr_a = run_pilot(bars, model, tradable, smap, btc_close, regime)
    eq_b, tr_b = run_pilot(bars, model, tradable, smap, btc_close, regime)
    assert len(tr_a) == len(tr_b)
    assert float(eq_a.iloc[-1]) == float(eq_b.iloc[-1])
