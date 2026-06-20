"""Point-in-time fundamental & price data via Sharadar (Nasdaq Data Link).

This is the paid data the fundamental factor family has always needed. It targets three
Sharadar tables on Nasdaq Data Link:

  * ``SHARADAR/SF1``    — company fundamentals. We use dimension ``ARQ`` (As-Reported
    Quarterly) and key every value to its ``datekey`` (the SEC FILING date), NOT
    ``calendardate`` (the fiscal period end). This is the entire point of paying: a
    fundamental value is only KNOWABLE once it has been filed, so a factor on date ``t``
    may use a statement only if its ``datekey <= t``. Using ``calendardate`` would leak
    weeks-to-months of future information.
  * ``SHARADAR/SEP``    — daily prices (we use the dividend+split-adjusted close).
  * ``SHARADAR/TICKERS`` — the survivorship-free symbol master, INCLUDING delisted
    names, with first/last price dates so a universe can be reconstructed as-of any date.

STUB MODE. If no API key is present (env ``NASDAQ_DATA_LINK_API_KEY``), the provider
constructs fine but every REAL data call raises :class:`SharadarUnavailable` with a
descriptive message. That lets the whole pipeline — providers, factors, universe,
harness, tests — be built and verified without the subscription; only the final numbers
are gated on the key.

The point-in-time panel builders below (``build_fundamental_panels`` and helpers) are
PURE functions on filing-dated frames, so they are fully testable with synthetic
filings and never need the key.

Read-only research. No orders.
"""

import logging
import os
from pathlib import Path

import pandas as pd

from data.providers import DataProvider

log = logging.getLogger(__name__)

API_KEY_ENV = "NASDAQ_DATA_LINK_API_KEY"
CACHE_DIR = Path(__file__).resolve().parent / "cache" / "sharadar"

# Raw SF1 (ARQ) fields we pull; mapped to friendly names in the panels below.
SF1_FIELDS = ["gp", "assets", "netinc", "equity", "sharesbas"]
SF1_RENAME = {"gp": "gross_profit", "assets": "total_assets", "netinc": "net_income",
              "equity": "book_equity", "sharesbas": "shares"}
FILING_DATE_COLUMN = "datekey"   # the FILING date — the point-in-time key (never calendardate)


class SharadarUnavailable(RuntimeError):
    """Raised when a real Sharadar call is attempted without an API key (stub mode)."""


class SharadarProvider(DataProvider):
    """Point-in-time prices/fundamentals/universe from Sharadar, with a parquet cache."""

    name = "sharadar"

    def __init__(self, api_key: str | None = None, cache_dir: Path = CACHE_DIR):
        self.api_key = api_key or os.getenv(API_KEY_ENV)
        self.stub = not self.api_key
        self.cache_dir = Path(cache_dir)
        if self.stub:
            log.info("SharadarProvider in STUB mode (no %s); real data calls will raise.", API_KEY_ENV)

    # --- plumbing ---------------------------------------------------------------

    def _require_live(self, what: str) -> None:
        """Guard every real data call; in stub mode raise a descriptive error."""
        if self.stub:
            raise SharadarUnavailable(
                f"Cannot fetch {what}: no Sharadar API key. Set {API_KEY_ENV} to a Nasdaq "
                f"Data Link key with a Sharadar subscription. Until then the pipeline runs "
                f"in stub mode and fundamental factors are deferred.")

    def _client(self):
        """Return an authenticated Nasdaq Data Link client (lazy import)."""
        try:
            import nasdaqdatalink
        except ImportError as error:  # optional dependency
            raise SharadarUnavailable(
                "The 'nasdaqdatalink' package is not installed; cannot reach Sharadar.") from error
        nasdaqdatalink.ApiConfig.api_key = self.api_key
        return nasdaqdatalink

    def _cache_path(self, kind: str, symbol: str) -> Path:
        return self.cache_dir / kind / f"{symbol}.parquet"

    # --- prices (SEP) -----------------------------------------------------------

    def get_price_bars(self, symbols: list[str], start=None, end=None) -> dict[str, pd.DataFrame]:
        """Daily adjusted OHLCV per symbol from SHARADAR/SEP (cached to parquet)."""
        self._require_live("SEP daily prices")
        bars: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            frame = self._load_or_fetch_prices(symbol)
            if frame is None or frame.empty:
                continue
            if start is not None:
                frame = frame[frame.index >= pd.Timestamp(start)]
            if end is not None:
                frame = frame[frame.index <= pd.Timestamp(end)]
            bars[symbol] = frame
        return bars

    def _load_or_fetch_prices(self, symbol: str, refresh: bool = False) -> pd.DataFrame | None:
        path = self._cache_path("sep", symbol)
        if not refresh and path.exists():
            return pd.read_parquet(path)
        client = self._client()
        raw = client.get_table("SHARADAR/SEP", ticker=symbol, paginate=True)
        if raw is None or raw.empty:
            return None
        raw = raw.sort_values("date")
        # closeadj = dividend+split adjusted close (total-return basis for factor returns).
        frame = pd.DataFrame({
            "Open": raw["open"].to_numpy(), "High": raw["high"].to_numpy(),
            "Low": raw["low"].to_numpy(), "Close": raw["closeadj"].to_numpy(),
            "Volume": raw["volume"].to_numpy(),
        }, index=pd.DatetimeIndex(pd.to_datetime(raw["date"]).dt.date, name="Date"))
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        return frame

    # --- fundamentals (SF1, ARQ, filing-dated) ----------------------------------

    def get_fundamentals(self, symbols: list[str], fields: list[str] = SF1_FIELDS,
                         start=None, end=None) -> dict[str, pd.DataFrame]:
        """As-reported quarterly fundamentals per symbol, INDEXED BY FILING DATE.

        Returns ``{symbol: DataFrame}`` where the index is ``datekey`` (the SEC filing
        date) — the point-in-time availability date — and columns are the renamed
        ``fields``. We deliberately drop ``calendardate`` (period end) as an index to
        prevent look-ahead. Dimension is ARQ (as-reported quarterly).
        """
        self._require_live("SF1 fundamentals")
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            frame = self._load_or_fetch_fundamentals(symbol, fields)
            if frame is not None and not frame.empty:
                out[symbol] = frame
        return out

    def _load_or_fetch_fundamentals(self, symbol: str, fields: list[str],
                                    refresh: bool = False) -> pd.DataFrame | None:
        path = self._cache_path("sf1", symbol)
        if not refresh and path.exists():
            return pd.read_parquet(path)
        client = self._client()
        # POINT-IN-TIME GUARD: dimension ARQ = as-reported quarterly; we request datekey
        # (filing date) alongside the values and index by it, never by calendardate.
        columns = [FILING_DATE_COLUMN, *fields]
        raw = client.get_table("SHARADAR/SF1", ticker=symbol, dimension="ARQ",
                               qopts={"columns": columns}, paginate=True)
        if raw is None or raw.empty:
            return None
        frame = as_reported_frame(raw, fields)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        return frame

    # --- survivorship-free universe (TICKERS) -----------------------------------

    def get_universe(self, min_first_price_year: int | None = None) -> pd.DataFrame:
        """Survivorship-free symbol master from SHARADAR/TICKERS (INCLUDES delisted).

        Returns a DataFrame with at least ``ticker``, ``firstpricedate``,
        ``lastpricedate``, and ``isdelisted``. Delisted names are KEPT — that inclusion
        is the real fix for survivorship bias; a name in/out of the universe on date
        ``t`` is decided as-of ``t`` from its first/last price dates, not from whether it
        still trades today.
        """
        self._require_live("TICKERS universe")
        client = self._client()
        raw = client.get_table("SHARADAR/TICKERS", table="SF1", paginate=True)
        cols = ["ticker", "firstpricedate", "lastpricedate", "isdelisted", "category"]
        frame = raw[[c for c in cols if c in raw.columns]].copy()
        for date_col in ("firstpricedate", "lastpricedate"):
            if date_col in frame.columns:
                frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        if min_first_price_year is not None and "firstpricedate" in frame.columns:
            frame = frame[frame["firstpricedate"].dt.year >= min_first_price_year]
        return frame.reset_index(drop=True)


# --- Point-in-time panel construction (pure, key-free, fully testable) -------------

def as_reported_frame(raw: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Index a raw SF1 pull by FILING date and rename to friendly columns.

    POINT-IN-TIME GUARD: the index becomes ``datekey`` (filing date). One filing per
    datekey is kept (the last, if a restatement shares a datekey), sorted ascending.
    """
    if FILING_DATE_COLUMN not in raw.columns:
        raise ValueError(f"SF1 pull missing '{FILING_DATE_COLUMN}'; cannot guarantee point-in-time.")
    frame = raw.copy()
    frame[FILING_DATE_COLUMN] = pd.to_datetime(frame[FILING_DATE_COLUMN])
    frame = (frame.sort_values(FILING_DATE_COLUMN)
                  .drop_duplicates(subset=[FILING_DATE_COLUMN], keep="last")
                  .set_index(FILING_DATE_COLUMN))
    present = [f for f in fields if f in frame.columns]
    frame = frame[present].rename(columns={k: v for k, v in SF1_RENAME.items() if k in present})
    frame.index.name = FILING_DATE_COLUMN
    return frame


def _ttm(series: pd.Series, min_quarters: int = 4) -> pd.Series:
    """Trailing-twelve-month sum of a quarterly (filing-frequency) flow series.

    Uses the 4 most recent FILINGS (each with ``datekey <= t``), so it stays
    point-in-time. NaN until four quarters exist.
    """
    return series.rolling(4, min_periods=min_quarters).sum()


def _yoy(series: pd.Series) -> pd.Series:
    """Year-over-year growth of a quarterly series: (x_t - x_{t-4}) / |x_{t-4}|.

    Compares a quarter to the same quarter a year earlier (4 filings back), so both
    inputs are filed on or before ``t``. Absolute denominator keeps the sign meaningful
    when the year-ago value is negative.
    """
    prior = series.shift(4)
    return (series - prior) / prior.abs()


def _as_of_daily(quarterly: pd.Series, master: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill a filing-dated quarterly series onto the daily ``master`` calendar.

    LOOK-AHEAD GUARD: the value is placed on its FILING date and forward-filled ONLY
    (never back-filled), so ``daily.loc[t]`` is the most recent filing with
    ``datekey <= t``. Asserts nothing is visible before its first filing date.
    """
    quarterly = quarterly[~quarterly.index.duplicated(keep="last")].sort_index()
    union = master.union(pd.DatetimeIndex(quarterly.index))
    daily = quarterly.reindex(union).ffill().reindex(master)
    first_filing = quarterly.first_valid_index()
    if first_filing is not None:
        before = daily[daily.index < first_filing]
        assert before.isna().all(), "point-in-time violation: value visible before its filing date"
    return daily


def build_fundamental_panels(filings_by_symbol: dict[str, pd.DataFrame],
                             master: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Turn per-symbol filing-dated frames into PIT daily ``date x symbol`` panels.

    Derived quarterly quantities (TTM sums, YoY growth) are computed at FILING
    frequency first, then forward-filled to the daily calendar — so every derived value
    on date ``t`` is built only from filings with ``datekey <= t``. Returns the panels
    the fundamental factors consume.
    """
    fields = ["gross_profit_ttm", "total_assets", "net_income_ttm", "net_income_yoy",
              "book_equity", "shares"]
    columns: dict[str, dict[str, pd.Series]] = {f: {} for f in fields}
    for symbol, filings in filings_by_symbol.items():
        filings = filings.sort_index()
        derived = {
            "gross_profit_ttm": _ttm(filings["gross_profit"]) if "gross_profit" in filings else None,
            "total_assets": filings["total_assets"] if "total_assets" in filings else None,
            "net_income_ttm": _ttm(filings["net_income"]) if "net_income" in filings else None,
            "net_income_yoy": _yoy(filings["net_income"]) if "net_income" in filings else None,
            "book_equity": filings["book_equity"] if "book_equity" in filings else None,
            "shares": filings["shares"] if "shares" in filings else None,
        }
        for field, series in derived.items():
            if series is not None:
                columns[field][symbol] = _as_of_daily(series, master)
    return {field: pd.DataFrame(cols, index=master) for field, cols in columns.items()}
