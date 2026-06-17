"""Size-tier universes (large / mid / small) for the cap-spectrum comparison.

Each universe is a fixed, representative, sector-keyed constituent list. The large
universe is the existing screener set; mid and small are hardcoded S&P 400 / Russell
2000-style names.

SURVIVORSHIP CAVEAT (read this): these are TODAY'S surviving tickers. Without
point-in-time constituent data, delisted/merged/failed names are absent — a bias
that is worst for small-caps (where failures are common). This comparison is
SUGGESTIVE, NOT CONCLUSIVE; strong small-cap results would need point-in-time data
to trust with real money.

This module also loads OHLCV bars (with Volume, needed for the liquidity filter)
into its own cache, leaving the OHLC-only cache used elsewhere untouched.
"""

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from screener.sectors import SECTOR_ETFS
from screener.stocks import SECTOR_CONSTITUENTS

log = logging.getLogger(__name__)

BENCHMARK = "SPY"
HISTORY_YEARS = 15
MIN_HISTORY_BARS = 252
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
CACHE_DIR = Path(__file__).resolve().parent / "cache" / "ohlcv"
THROTTLE_SECONDS = 0.15

# Large-cap baseline: the existing screener constituents.
LARGE = SECTOR_CONSTITUENTS

# Mid-cap (S&P 400-style) representative names, sector-keyed. Today's survivors.
MID: dict[str, list[str]] = {
    "XLK": ["NTAP", "JBL", "ZBRA", "TYL", "MANH", "FFIV"],
    "XLF": ["RJF", "SF", "CBSH", "WAL", "PB", "EWBC"],
    "XLI": ["NDSN", "AOS", "GGG", "WWD", "CSL", "HUBB"],
    "XLV": ["MASI", "CHE", "HAE", "BRKR", "PEN", "NBIX"],
    "XLY": ["POOL", "WSM", "TPX", "CROX", "DECK", "BBY"],
    "XLP": ["CASY", "LANC", "JJSF", "INGR", "CHD"],
    "XLE": ["MUR", "SM", "RRC", "CHRD", "PR"],
    "XLB": ["RPM", "CE", "WLK", "SON", "AVY"],
    "XLU": ["IDA", "POR", "NWE", "OGE", "BKH"],
    "XLRE": ["EGP", "STAG", "CUZ", "HIW", "ELS"],
    "XLC": ["TTWO", "NYT", "IPG", "OMC"],
}

# Small-cap (Russell 2000-style) representative names, sector-keyed. Today's survivors.
SMALL: dict[str, list[str]] = {
    "XLK": ["EXTR", "DGII", "CEVA", "PLXS", "MXL"],
    "XLF": ["CATY", "HOPE", "TRMK", "BANF", "INDB", "FFBC"],
    "XLI": ["ALG", "MLI", "GVA", "AIN", "THRM"],
    "XLV": ["SUPN", "AMPH", "CORT", "LMAT", "TARO"],
    "XLY": ["SHOO", "BKE", "CATO", "CRMT"],
    "XLP": ["NATH", "BGS", "MGPI", "USNA"],
    "XLE": ["CLB", "NGS", "REX", "DMLP"],
    "XLB": ["KOP", "HWKN", "SCL", "IOSP"],
    "XLU": ["MGEE", "NWN", "YORW", "UTL"],
    "XLRE": ["GTY", "LTC", "UMH", "ALEX"],
    "XLC": ["SALM", "CCO"],
}

UNIVERSES: dict[str, dict[str, list[str]]] = {"large": LARGE, "mid": MID, "small": SMALL}


def sector_map(name: str) -> dict[str, str]:
    """Map each constituent symbol to its sector ETF for a universe."""
    return {sym: etf for etf, members in UNIVERSES[name].items() for sym in members}


def constituents(name: str) -> list[str]:
    """Sorted unique constituent tickers of a universe."""
    return sorted({sym for members in UNIVERSES[name].values() for sym in members})


def _symbols(name: str) -> list[str]:
    """All symbols needed: constituents + the 11 sector ETFs + the benchmark."""
    return sorted(set(constituents(name)) | set(SECTOR_ETFS) | {BENCHMARK})


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.parquet"


def fetch_ohlcv(symbol: str, years: int = HISTORY_YEARS, refresh: bool = False) -> pd.DataFrame | None:
    """Fetch (or load) ~``years`` of adjusted daily OHLCV for one symbol.

    Includes Volume (needed for the liquidity filter). Returns ``None`` on failure.
    """
    path = _cache_path(symbol)
    if not refresh and path.exists():
        return pd.read_parquet(path)

    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=int(years * 365.25) + 5)
    try:
        bars = yf.Ticker(symbol).history(
            start=start.isoformat(), end=end.isoformat(), interval="1d", auto_adjust=True
        )
    except Exception as error:  # noqa: BLE001 - any provider failure is non-fatal
        log.warning("Download failed for %s: %s", symbol, error)
        return None
    if bars is None or bars.empty or not set(OHLCV_COLUMNS).issubset(bars.columns):
        log.warning("No OHLCV data for %s.", symbol)
        return None

    bars = bars[OHLCV_COLUMNS].dropna()
    if bars.empty:
        return None
    bars.index = pd.DatetimeIndex(pd.to_datetime(bars.index).date)
    bars.index.name = "Date"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(path)
    time.sleep(THROTTLE_SECONDS)  # throttle only on a live fetch
    return bars


def load_bars(name: str, years: int = HISTORY_YEARS, refresh: bool = False) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load OHLCV bars for a universe (constituents + ETFs + benchmark).

    Skips symbols with fewer than ``MIN_HISTORY_BARS`` rows (logged).
    """
    bars: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for symbol in _symbols(name):
        frame = fetch_ohlcv(symbol, years=years, refresh=refresh)
        if frame is None or len(frame) < MIN_HISTORY_BARS:
            skipped.append(symbol)
            continue
        bars[symbol] = frame
    log.info("Universe '%s': loaded %d symbols (%d skipped).", name, len(bars), len(skipped))
    return bars, skipped
