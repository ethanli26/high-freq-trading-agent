"""Crypto daily OHLCV behind the existing DataProvider interface.

``CryptoDataProvider`` fetches daily OHLCV for a set of coins, caches to parquet, and
returns the same ``{symbol: DataFrame}`` shape the rest of the system consumes, so the
backtest engine treats crypto exactly like equities.

THREE HONESTY HAZARDS, handled here as well as free data allows:

  1. SURVIVORSHIP BIAS (unfixable with free data). Whatever source we hit returns
     TODAY'S surviving coins. Coins that rugged, died, or delisted are simply absent —
     and in small-coin crypto that is the majority of all coins ever launched. Every
     backtest on this data is therefore biased OPTIMISTIC; nothing in this file can
     repair that, only point-in-time historical constituent data could.

  2. FAKE / WASH VOLUME. Reported crypto volume is notoriously inflated by wash
     trading, especially on small coins and small venues. :func:`wash_volume_flags`
     applies a coarse OHLCV-only sanity check (real volume should move price); it is a
     PARTIAL mitigation, not a fix — true detection needs trade-level and on-chain data.

  3. LIQUIDITY. :func:`liquidity_filter` drops coins below a minimum average daily
     dollar volume and a minimum history length, so we do not model trades in names we
     could not realistically have traded. Thresholds are configurable.

DATA BACKENDS. Two are supported and both cache to parquet:
  * ``ccxt`` — daily OHLCV from any CCXT-supported exchange (broadest small-coin
    coverage; requires the optional ``ccxt`` package and network).
  * ``yfinance`` — ``BTC-USD`` style pairs (works with the package already used here,
    but only covers larger, listed coins).
When no backend / network is available, callers fall back to clearly-labeled synthetic
data (see ``crypto/run_hype_test.py``); synthetic results test mechanics and costs
only and prove nothing about real edge.

Read-only research. No exchange orders are ever placed.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.providers import DataProvider

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
CACHE_DIR = Path(__file__).resolve().parent / "cache" / "crypto"


# --- Liquidity & wash-volume screens (the heart of honest small-coin crypto) -------

def average_dollar_volume(bars: pd.DataFrame, lookback: int = 30) -> float:
    """Median trailing average daily dollar volume (close * volume). 0.0 if missing."""
    if not {"Close", "Volume"}.issubset(getattr(bars, "columns", [])):
        return 0.0
    adv = (bars["Close"] * bars["Volume"]).rolling(lookback).mean().dropna()
    return float(adv.median()) if not adv.empty else 0.0


def wash_volume_score(bars: pd.DataFrame) -> float:
    """A 0..1 wash-trading suspicion score from OHLCV alone (higher = more suspect).

    Premise: REAL volume moves price. Wash trades inflate volume while leaving price
    roughly unchanged. We look at the highest-volume days (top quartile by dollar
    volume) and measure the fraction whose absolute close-to-close return was
    negligible (< 0.5%). A coin where most of its heaviest-volume days barely moved is
    a wash-trading red flag.

    This is a COARSE heuristic on daily bars — a partial mitigation, not proof. Genuine
    detection needs trade-level prints and on-chain flow that free daily data lacks.
    """
    if not {"Close", "Volume"}.issubset(getattr(bars, "columns", [])) or len(bars) < 30:
        return 0.0
    dollar_volume = bars["Close"] * bars["Volume"]
    returns = bars["Close"].pct_change().abs()
    heavy = dollar_volume >= dollar_volume.quantile(0.75)  # the busiest 25% of days
    heavy_returns = returns[heavy].dropna()
    if heavy_returns.empty:
        return 0.0
    flat_heavy = float((heavy_returns < 0.005).mean())  # heavy days that barely moved
    return flat_heavy


def wash_volume_flags(bars: dict[str, pd.DataFrame], threshold: float = 0.6) -> dict[str, bool]:
    """Map each coin to True when its wash-volume score exceeds ``threshold``."""
    return {sym: wash_volume_score(frame) >= threshold for sym, frame in bars.items()}


def liquidity_filter(
    bars: dict[str, pd.DataFrame],
    *,
    min_dollar_volume: float = 1_000_000.0,
    min_history_days: int = 200,
    wash_threshold: float = 0.6,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """Keep only coins that clear the liquidity and wash-volume screens.

    Returns ``(kept, skips)`` where ``skips`` has keys ``"illiquid"``, ``"short"``,
    and ``"wash"`` listing the dropped coins by reason. Thresholds are configurable;
    crypto uses NO price floor (cheap unit price is normal for coins).
    """
    kept: dict[str, pd.DataFrame] = {}
    skips: dict[str, list[str]] = {"illiquid": [], "short": [], "wash": []}
    for symbol, frame in bars.items():
        if frame is None or len(frame) < min_history_days:
            skips["short"].append(symbol)
            continue
        if average_dollar_volume(frame) < min_dollar_volume:
            skips["illiquid"].append(symbol)
            continue
        if wash_volume_score(frame) >= wash_threshold:
            skips["wash"].append(symbol)
            continue
        kept[symbol] = frame
    log.info("Crypto liquidity filter: kept %d, skipped %d illiquid / %d short / %d wash.",
             len(kept), len(skips["illiquid"]), len(skips["short"]), len(skips["wash"]))
    return kept, skips


# --- The provider -------------------------------------------------------------------

class CryptoDataProvider(DataProvider):
    """Daily crypto OHLCV with a parquet cache, behind the DataProvider interface.

    ``backend`` selects the source: ``"ccxt"`` (broad, needs the ccxt package +
    network) or ``"yfinance"`` (``BTC-USD`` style pairs only). Prices are NOT
    point-in-time survivorship-corrected — see the module docstring.
    """

    name = "crypto"

    def __init__(self, backend: str = "ccxt", exchange: str = "binance",
                 quote: str = "USDT", cache_dir: Path = CACHE_DIR):
        self.backend = backend
        self.exchange = exchange
        self.quote = quote
        self.cache_dir = Path(cache_dir)

    def _cache_path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_")
        return self.cache_dir / f"{safe}.parquet"

    def get_price_bars(self, symbols: list[str], start=None, end=None) -> dict[str, pd.DataFrame]:
        """Fetch (or load) daily OHLCV for each coin, optionally clipped to a window."""
        bars: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            frame = self._load_or_fetch(symbol)
            if frame is None or frame.empty:
                continue
            if start is not None:
                frame = frame[frame.index >= pd.Timestamp(start)]
            if end is not None:
                frame = frame[frame.index <= pd.Timestamp(end)]
            bars[symbol] = frame
        return bars

    def get_fundamentals(self, symbols, fields, start=None, end=None):
        """Crypto has no equity-style fundamentals here."""
        raise NotImplementedError("CryptoDataProvider serves price bars only.")

    def _load_or_fetch(self, symbol: str, refresh: bool = False) -> pd.DataFrame | None:
        """Return cached parquet if present, else fetch from the chosen backend."""
        path = self._cache_path(symbol)
        if not refresh and path.exists():
            return pd.read_parquet(path)
        frame = self._fetch_ccxt(symbol) if self.backend == "ccxt" else self._fetch_yfinance(symbol)
        if frame is None or frame.empty:
            return None
        frame = frame[OHLCV_COLUMNS].dropna()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        return frame

    def _fetch_ccxt(self, symbol: str) -> pd.DataFrame | None:
        """Daily OHLCV via CCXT (optional dependency; documented, network-dependent)."""
        try:
            import ccxt  # optional; only needed for the ccxt backend
        except ImportError:
            log.warning("ccxt not installed; cannot fetch %s. Use cached or synthetic data.", symbol)
            return None
        try:
            client = getattr(ccxt, self.exchange)()
            market = symbol if "/" in symbol else f"{symbol}/{self.quote}"
            raw = client.fetch_ohlcv(market, timeframe="1d", limit=1500)
        except Exception as error:  # noqa: BLE001 - any provider failure is non-fatal
            log.warning("ccxt fetch failed for %s: %s", symbol, error)
            return None
        if not raw:
            return None
        frame = pd.DataFrame(raw, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame["ts"], unit="ms").dt.date)
        frame.index.name = "Date"
        return frame[OHLCV_COLUMNS]

    def _fetch_yfinance(self, symbol: str) -> pd.DataFrame | None:
        """Daily OHLCV via yfinance for a ``BTC-USD`` style pair."""
        try:
            import yfinance as yf
        except ImportError:
            log.warning("yfinance not installed; cannot fetch %s.", symbol)
            return None
        ticker = symbol if symbol.endswith("-USD") else f"{symbol}-USD"
        try:
            raw = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
        except Exception as error:  # noqa: BLE001
            log.warning("yfinance fetch failed for %s: %s", ticker, error)
            return None
        if raw is None or raw.empty or not set(OHLCV_COLUMNS).issubset(raw.columns):
            return None
        raw = raw[OHLCV_COLUMNS].dropna()
        raw.index = pd.DatetimeIndex(pd.to_datetime(raw.index).date)
        raw.index.name = "Date"
        return raw


def make_synthetic_crypto(symbols: list[str], days: int = 900, seed: int = 11,
                          start: str = "2021-01-01") -> dict[str, pd.DataFrame]:
    """Deterministic synthetic daily crypto OHLCV across a spread of liquidity tiers.

    Clearly labeled FAKE data for exercising the pipeline offline. Coins are given a
    range of volatilities and dollar-volume levels (some thin, some liquid) plus one
    intentionally wash-like coin, so the liquidity and wash screens, the per-coin cost
    scaling, and the engine all get exercised. It has NO designed edge, so any P&L is
    noise — it cannot speak to whether a hype signal really works.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=days)
    bars: dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(symbols):
        # Spread coins across volatility and liquidity tiers.
        is_wash = symbol.endswith("WASH")
        sigma = 0.002 if is_wash else 0.03 + 0.01 * (i % 5)    # wash coin: near-flat price
        base_price = 0.5 * (i + 1)
        target_dollar_volume = 10_000_000.0 / (1 + 3 * (i % 6))  # liquid -> thin across coins
        shocks = rng.normal(0.0, sigma, days)

        # "Hype episodes": on ~3% of days a coin pops on a volume burst, then keeps
        # random-walking. Timing and aftermath are RANDOM, so there is NO designed edge
        # (a pop is as likely to fade as to follow through) — it only makes the
        # breakout + volume-surge entry fire so the cost machinery is exercised.
        hype = (rng.random(days) < 0.03) & (not is_wash)
        shocks = shocks + hype * rng.uniform(0.05, 0.15, days)  # price pop on hype days
        close = base_price * np.cumprod(1.0 + shocks)
        open_ = np.empty(days)
        open_[0] = base_price
        open_[1:] = close[:-1]
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, sigma / 2, days)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, sigma / 2, days)))

        # Raw volume (coin units) tracks move size and bursts on hype days, so REAL
        # volume moves price (the wash screen passes). dollar volume = close * volume.
        base_units = target_dollar_volume / base_price
        volume = base_units * (1.0 + 4.0 * np.abs(shocks)) * np.where(hype, rng.uniform(3.0, 6.0, days), 1.0)
        if symbol.endswith("WASH"):  # one coin with huge, price-detached volume (wash-like)
            volume = base_units * 50.0 * (1.0 + rng.random(days))
        bars[symbol] = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=pd.DatetimeIndex(dates.date, name="Date"))
    return bars
