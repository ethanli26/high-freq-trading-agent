"""The US-effective fundamental factor family: profitability, value, earnings growth.

A US-vs-China factor study (Orient Securities report 22) found that in US equities the
FUNDAMENTAL factors — profitability, value, and earnings growth — are the most
effective, in contrast to the price/reversal factors that dominate Chinese A-shares.
Those factors could never be tested in this harness before because they require
point-in-time fundamentals; with the Sharadar provider they finally can.

Each factor reads ``FactorData.fundamentals`` — ``{field: date x symbol}`` panels that
are forward-filled from FILING dates (see data/sharadar_provider.build_fundamental_panels),
so the value on date ``t`` uses only statements filed on or before ``t``. Combined with
the day-``t`` close, each factor is point-in-time safe. ``point_in_time_provider = True``
marks them so the free-data harness defers them.

Expected IC sign is POSITIVE for all four (higher quality / cheaper / faster-growing =>
higher forward returns).

Read-only research: no orders.
"""

import pandas as pd

from factors.base import Factor, FactorData, register


def _require_fundamentals(data: FactorData, fields: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Return the needed fundamental panels, or raise if running without PIT data.

    Aligns each panel to the close calendar so an element-wise combine with price is
    index-matched (and therefore look-ahead safe: panel[t] is filing-dated <= t).
    """
    if data.fundamentals is None:
        raise NotImplementedError(
            "Fundamental factors need point-in-time data. Run on the 'sharadar_broad' "
            "universe with a Sharadar API key (NASDAQ_DATA_LINK_API_KEY); the free data "
            "path cannot honor filing-dated fundamentals.")
    missing = [f for f in fields if f not in data.fundamentals]
    if missing:
        raise KeyError(f"FactorData.fundamentals missing required panels: {missing}")
    return {f: data.fundamentals[f].reindex(data.close.index) for f in fields}


@register
class Profitability(Factor):
    """Gross profitability: trailing-12m gross profit / total assets.

    Source: Novy-Marx (2013), "The Other Side of Value: The Gross Profitability
    Premium", JFE 108(1). Rationale: more profitable firms earn higher returns; gross
    profit is the cleanest (least-manipulated) profitability line. Expected IC: POSITIVE.
    """

    name = "profitability"
    category = "fundamental"
    requires = ("gross_profit_ttm", "total_assets")
    point_in_time_provider = True

    def compute(self, data: FactorData) -> pd.DataFrame:
        f = _require_fundamentals(data, self.requires)
        # LOOK-AHEAD GUARD: both panels are filing-dated (datekey <= t); ratio is element-
        # wise, so value[t] uses only statements filed on or before t.
        assets = f["total_assets"].where(f["total_assets"] > 0)
        return f["gross_profit_ttm"] / assets


@register
class EarningsYield(Factor):
    """Earnings yield (E/P): trailing-12m net income / market cap.

    Source: Basu (1977), "Investment Performance of Common Stocks in Relation to Their
    Price-Earnings Ratios", JF 32(3). Rationale: cheap (high E/P) stocks outperform.
    Expected IC: POSITIVE. Market cap = filing-dated shares x day-t close.
    """

    name = "earnings_yield"
    category = "fundamental"
    requires = ("net_income_ttm", "shares")
    point_in_time_provider = True

    def compute(self, data: FactorData) -> pd.DataFrame:
        f = _require_fundamentals(data, self.requires)
        # LOOK-AHEAD GUARD: net income & shares are filing-dated (<= t); close is day-t.
        market_cap = (f["shares"] * data.close).where(lambda x: x > 0)
        return f["net_income_ttm"] / market_cap


@register
class BookToPrice(Factor):
    """Book-to-price (value): book equity / market cap.

    Source: the value factor of Fama & French (1992, 1993) (HML uses book-to-market).
    Rationale: high book-to-market ("value") stocks earn higher returns. Expected IC:
    POSITIVE. Market cap = filing-dated shares x day-t close.
    """

    name = "book_to_price"
    category = "fundamental"
    requires = ("book_equity", "shares")
    point_in_time_provider = True

    def compute(self, data: FactorData) -> pd.DataFrame:
        f = _require_fundamentals(data, self.requires)
        # LOOK-AHEAD GUARD: book equity & shares are filing-dated (<= t); close is day-t.
        market_cap = (f["shares"] * data.close).where(lambda x: x > 0)
        return f["book_equity"] / market_cap


@register
class EarningsGrowth(Factor):
    """Earnings growth: year-over-year growth in quarterly net income.

    Source: the growth factor flagged as US-effective in Orient Securities report 22;
    report 5 found net-profit YoY growth had a backtest Sharpe of ~1.82 in its set.
    Rationale: accelerating earnings predict higher forward returns. Expected IC:
    POSITIVE. Growth compares a quarter to the same quarter a year earlier (4 filings
    back), both filed on or before t.
    """

    name = "earnings_growth"
    category = "fundamental"
    requires = ("net_income_yoy",)
    point_in_time_provider = True

    def compute(self, data: FactorData) -> pd.DataFrame:
        f = _require_fundamentals(data, self.requires)
        # LOOK-AHEAD GUARD: the YoY series is computed at filing frequency then forward-
        # filled from datekey, so value[t] uses only filings with datekey <= t.
        return f["net_income_yoy"]
