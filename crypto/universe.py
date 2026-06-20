"""A broad small-coin crypto universe — where hype lives — built honestly.

The pilot deliberately targets the SMALL/MID tier and excludes the mega-caps (BTC,
ETH, and the largest, most-efficient coins): hype-momentum is a small-coin phenomenon,
and including the mega-caps would both dilute the test and understate costs. BTC is
kept ONLY as the market benchmark / regime source, never as a tradable name.

PROMINENT SURVIVORSHIP CAVEAT
-----------------------------
This universe is whatever the data provider returns AFTER the liquidity and wash-volume
screens — i.e. TODAY'S surviving, currently-liquid coins. The thousands of small coins
that rugged, died, or delisted are absent. In small-coin crypto that survivor set is a
tiny, lucky minority, so ANY backtest here is biased materially OPTIMISTIC. Free data
cannot fix this; only point-in-time historical listings could. Read results as a
mechanism/cost study, not as evidence a hype strategy makes money.
"""

import logging

import pandas as pd

from data.crypto_provider import liquidity_filter

log = logging.getLogger(__name__)

# Mega-caps to EXCLUDE from the tradable universe (kept only as benchmark/regime).
MEGA_CAPS = {"BTC", "ETH", "BTC-USD", "ETH-USD", "BTC/USDT", "ETH/USDT"}

# The market benchmark / regime source (buy-and-hold comparison is vs this).
BENCHMARK = "BTC"


def build_universe(
    bars: dict[str, pd.DataFrame],
    *,
    min_dollar_volume: float = 1_000_000.0,
    min_history_days: int = 200,
    wash_threshold: float = 0.6,
    exclude: set[str] = MEGA_CAPS,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """Return ``(tradable_bars, skips)`` for the small-coin universe.

    Drops the mega-caps, then applies the liquidity + wash screens. ``skips`` reports
    dropped coins by reason (``mega``, ``illiquid``, ``short``, ``wash``). The
    benchmark coin is excluded from the tradable set here; the harness adds it back
    only as the regime/benchmark series.
    """
    candidates = {sym: frame for sym, frame in bars.items()
                  if sym not in exclude and sym != BENCHMARK}
    mega_skipped = sorted(set(bars) & exclude)
    kept, skips = liquidity_filter(
        candidates, min_dollar_volume=min_dollar_volume,
        min_history_days=min_history_days, wash_threshold=wash_threshold)
    skips = {"mega": mega_skipped, **skips}
    log.info("Crypto universe: %d tradable coins (mega excluded: %d).",
             len(kept), len(mega_skipped))
    return kept, skips


def sector_map(symbols) -> dict[str, str]:
    """Map every coin to a single synthetic 'sector' = the benchmark.

    The equities engine gates entries on cross-sectional SECTOR rank, a concept crypto
    lacks. Mapping all coins to one sector (proxied by the benchmark, which is always
    top-ranked) neutralizes that gate while still routing all crypto exposure through
    the engine's per-sector cap — a sensible single-bucket risk limit on total crypto.
    """
    return {sym: BENCHMARK for sym in symbols}
