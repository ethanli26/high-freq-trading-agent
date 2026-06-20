"""Regression test: the canonical backtest must reproduce 1675 trades / $1,967,830.

Marked ``slow`` and ``network`` so the offline CI suite skips it. It also skips
gracefully (with a clear message) when the local price cache is absent, so it never
forces a network fetch.
"""

import pytest

import backtest.data as data
from backtest.data import BENCHMARK, build_sector_map, load_universe
from backtest.engine import run_engine
from backtest.regime import compute_regime
from screener.sectors import SECTOR_ETFS
from signals.breakout import BreakoutStrategy
from signals.pullback import PullbackStrategy

pytestmark = [pytest.mark.slow, pytest.mark.network]

CANONICAL_TRADES = 1675
CANONICAL_FINAL = 1_967_830


def _cache_present() -> bool:
    """True if the large-cap price cache exists (so no network fetch is needed)."""
    return data._cache_path(BENCHMARK).exists()


@pytest.mark.skipif(not _cache_present(),
                    reason="price cache absent — run `python main.py backtest` once to populate it")
def test_canonical_backtest_reproduces():
    bars, _ = load_universe()
    regime = compute_regime(bars[BENCHMARK]["Close"])
    equity, trades = run_engine(
        bars, regime, build_sector_map(), list(SECTOR_ETFS),
        strategies=[BreakoutStrategy(), PullbackStrategy()],
        regime_filter=True, trend_exit=True, conviction_sizing=False,
    )
    assert len(trades) == CANONICAL_TRADES
    assert abs(equity.iloc[-1] - CANONICAL_FINAL) < 5000
