"""Earnings-drift catalyst strategy (post-earnings-announcement drift).

Idea: after a company reports a positive EPS surprise and the market reacts up,
ride the well-documented drift — but ENTER ONLY AFTER the announcement, never
holding through it (earnings gaps are the single biggest single-name gap risk).

Entry rule (completed bars only; CRITICALLY no entry on or before the report day):
  1. The most recent report had EPS surprise >= EARNINGS_SURPRISE_MIN.
  2. The signal fires on the EARNINGS_ENTRY_DELAY_DAYS-th trading session STRICTLY
     AFTER the report date (so the announcement gap is already in the past), and the
     engine fills the entry at the NEXT open — strictly later still.
  3. If EARNINGS_CONFIRM_UP, that post-report session must close above its open.
  (The sector-momentum gate is applied by the engine, like the other strategies.)

GAP-RISK AVOIDANCE: because the signal session is strictly after the report date and
the fill is one session later again, the fill date is always strictly after the
report — the announcement is never traded through. This is asserted per signal.
"""

import pandas as pd

import config
from signals.base import Strategy


class EarningsDriftStrategy(Strategy):
    """Enter after a confirmed positive-surprise earnings report.

    Constructed with per-symbol earnings frames (``report_date``, ``surprise``);
    ``signal_series``/``generate_signal`` need the ``symbol`` to look them up.
    """

    name = "earnings_drift"

    def __init__(self, earnings: dict[str, pd.DataFrame]):
        self.earnings = earnings

    def signal_series(self, bars: pd.DataFrame, symbol: str | None = None) -> pd.Series:
        """Per-bar earnings-drift signal for ``symbol`` (False where no setup)."""
        signal = pd.Series(False, index=bars.index)
        events = self.earnings.get(symbol) if symbol is not None else None
        if events is None or events.empty:
            return signal

        idx = bars.index
        opens, closes = bars["Open"], bars["Close"]
        delay = config.EARNINGS_ENTRY_DELAY_DAYS

        for report_date, surprise in zip(events["report_date"], events["surprise"]):
            if pd.isna(surprise) or surprise < config.EARNINGS_SURPRISE_MIN:
                continue
            # The signal session is the delay-th trading session STRICTLY after the
            # report date — the announcement and its gap are already in the past.
            after = idx[idx > report_date]
            if len(after) < delay:
                continue
            s = idx.get_loc(after[delay - 1])
            if s + 1 >= len(idx):  # need a next session for the engine to fill at
                continue
            if config.EARNINGS_CONFIRM_UP and not (closes.iloc[s] > opens.iloc[s]):
                continue

            # HARD GUARD: the engine fills at idx[s+1]; that date must be strictly
            # after the report date, so we never trade through the announcement.
            assert idx[s + 1] > report_date, "earnings entry on/before the report date!"
            signal.iloc[s] = True

        return signal

    def generate_signal(self, bars: pd.DataFrame, symbol: str | None = None) -> bool:
        """Canonical rule: is the latest completed bar an earnings-drift signal?"""
        if bars is None or bars.empty:
            return False
        return bool(self.signal_series(bars, symbol).iloc[-1])
