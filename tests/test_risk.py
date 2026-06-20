"""Unit tests for the deterministic risk math: sizing, caps, ATR/stop, conviction."""

import pandas as pd

import config
from risk.conviction import conviction_multiplier
from risk.portfolio import apply_portfolio_limits, meets_min_size
from risk.position import compute_atr, compute_stop, size_position

EQUITY = 1_000_000.0


def test_compute_stop_is_atr_multiple_below_entry():
    assert compute_stop(100.0, 2.0) == 100.0 - config.ATR_MULTIPLE * 2.0


def test_size_position_per_name_cap_binds():
    # risk budget ($10k / $4 per share = 2500) exceeds the 10% per-name cap (1000).
    shares, risk = size_position(EQUITY, 100.0, 96.0)
    assert shares == 1000
    assert risk == config.RISK_PER_TRADE * EQUITY  # 10,000


def test_size_position_risk_budget_binds():
    # wider stop ($20/share) -> 500 shares, below the cap.
    shares, risk = size_position(EQUITY, 100.0, 80.0)
    assert shares == 500
    assert risk == 10_000.0


def test_size_position_conviction_cannot_exceed_per_name_cap():
    base, _ = size_position(EQUITY, 100.0, 96.0, risk_mult=1.0)
    doubled, _ = size_position(EQUITY, 100.0, 96.0, risk_mult=2.0)
    assert base == 1000 and doubled == 1000  # the 10% cap binds both


def test_size_position_rejects_non_positive_risk():
    assert size_position(EQUITY, 100.0, 100.0) == (0, 0.0)   # stop == entry
    assert size_position(EQUITY, 100.0, 110.0) == (0, 0.0)   # stop above entry
    assert size_position(EQUITY, 0.0, -5.0) == (0, 0.0)      # entry <= 0


def test_compute_atr_constant_true_range():
    n = 40
    bars = pd.DataFrame({"Open": [101.0] * n, "High": [102.0] * n,
                         "Low": [100.0] * n, "Close": [101.0] * n})
    atr = compute_atr(bars, 14)  # constant TR of 2 -> Wilder ATR converges to 2
    assert atr is not None and abs(atr - 2.0) < 1e-9


def test_compute_atr_too_short_returns_none():
    bars = pd.DataFrame({"High": [1.0, 2.0], "Low": [0.0, 1.0], "Close": [1.0, 1.5]})
    assert compute_atr(bars, 14) is None


def test_meets_min_size_floor():
    # value floor = 1% of equity = $10,000; share floor = 10.
    assert meets_min_size(15, 1.0, EQUITY) is True       # clears the share floor
    assert meets_min_size(1, 20_000.0, EQUITY) is True   # clears the value floor
    assert meets_min_size(5, 100.0, EQUITY) is False     # below BOTH floors
    assert meets_min_size(1, 100.0, EQUITY) is False


def test_conviction_multiplier_bounds_and_monotonic():
    lo, hi = config.CONVICTION_MIN_MULT, config.CONVICTION_MAX_MULT
    assert conviction_multiplier(0.0) == lo
    assert conviction_multiplier(1.0) == hi
    assert abs(conviction_multiplier(0.5) - (lo + (hi - lo) * 0.5)) < 1e-9
    assert conviction_multiplier(-5.0) == lo   # clipped below
    assert conviction_multiplier(5.0) == hi    # clipped above
    assert conviction_multiplier(0.2) < conviction_multiplier(0.8)


def _proposal(symbol, shares, entry=100.0):
    return {"status": "propose", "symbol": symbol, "action": "BUY", "entry_ref": entry,
            "stop": entry * 0.95, "shares": shares, "est_value": shares * entry}


def test_apply_portfolio_limits_sector_and_total_caps():
    # equity 1e6 -> sector cap 30% = $300k, total cap 60% = $600k.
    sector_of = {"A": "XLK", "B": "XLK", "C": "XLE"}
    proposals = [_proposal("A", 4000), _proposal("B", 4000), _proposal("C", 4000)]  # $400k each
    adjusted, _ = apply_portfolio_limits(proposals, EQUITY, [], sector_of, enforce_min_size=False)
    sized = {p["symbol"]: p["shares"] for p in adjusted}
    assert sized["A"] == 3000   # trimmed to the $300k sector cap
    assert "B" not in sized     # XLK sector now full
    assert sized["C"] == 3000   # trimmed to the remaining $300k total budget


def test_apply_portfolio_limits_min_size_skip():
    tiny = [_proposal("A", 3, 100.0)]  # 3 shares, $300 -> below both floors
    adjusted, skips = apply_portfolio_limits(tiny, EQUITY, [], {"A": "XLK"}, enforce_min_size=True)
    assert adjusted == []
    assert any(s["reason"] == "below minimum size" for s in skips)
    kept, _ = apply_portfolio_limits(tiny, EQUITY, [], {"A": "XLK"}, enforce_min_size=False)
    assert len(kept) == 1  # floor off -> kept
