"""Walk-forward backtest engine for the breakout + ATR-stop strategy.

The whole point of this module is an HONEST simulation with NO LOOK-AHEAD BIAS.
The guarantees, and where they are enforced, are:

  * Decisions on day ``i`` read indicators only at ``p = i - 1`` (the prior
    completed bar). Search for "LOOK-AHEAD GUARD" comments below.
  * Entries fill at day ``i``'s OPEN, never the signal day's close.
  * Sizing equity and portfolio caps are valued at the prior close.
  * The trailing stop active during day ``i`` uses closes only through ``p``.
  * Regime at entry is read at ``p``.

Strategies are pluggable (signals/base.Strategy): each provides a vectorized
signal_series that reproduces its canonical generate_signal bar by bar (verified by
tests). ATR, sizing, stops, portfolio caps, and the regime filter all apply the same
way regardless of which strategy fired. Read-only: no IBKR, no orders.
"""

import logging
import math

import numpy as np
import pandas as pd

from config import (
    ATR_MULTIPLE,
    ATR_PERIOD,
    BEAR_MAX_TOTAL_EXPOSURE,
    BEAR_SIZE_MULT,
    CHANDELIER_ATR_MULT,
    CONVICTION_SIZING_ENABLED,
    CRASH_BLOCK_NEW_ENTRIES,
    EARNINGS_BLACKOUT_DAYS,
    EARNINGS_BLACKOUT_ENABLED,
    LIQUIDITY_LOOKBACK,
    MAX_ADV_PARTICIPATION,
    MIN_DOLLAR_VOLUME,
    MIN_PRICE,
    OVERLAY_ENABLED,
    REGIME_FILTER_ENABLED,
    TREND_EXIT_ENABLED,
    TREND_EXIT_MA,
)
from risk.conviction import conviction_multiplier, score_from_factors
from risk.portfolio import apply_portfolio_limits, meets_min_size
from risk.position import compute_stop, size_position
from screener.momentum import LOOKBACK_3M, LOOKBACK_6M
from signals.base import Strategy
from signals.breakout import BreakoutStrategy

log = logging.getLogger(__name__)

# Simulated account and cost assumptions (all configurable via run_engine).
STARTING_EQUITY = 1_000_000.0
COMMISSION_PER_SHARE = 0.005   # $/share, IBKR-tiered-like
SLIPPAGE_PCT = 0.0005          # 5 bps, applied against us on entry and exit
WARMUP_BARS = 200              # need 200 bars for the regime's 200-day average

TRADE_COLUMNS = [
    "symbol", "strategy", "sector", "entry_date", "entry_price", "exit_date", "exit_price",
    "shares", "pnl", "pnl_pct", "regime_at_entry", "bars_held", "exit_reason",
]


# --- Vectorized indicators (reproduce the live per-bar functions) -------------

def atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Per-bar Wilder ATR, reproducing risk/position.compute_atr at every bar.

    Drops the first true-range value (no prior close) and applies Wilder smoothing
    (EWM with alpha = 1/period, adjust=False), which is causal — value at ``t``
    depends only on bars up to ``t``.
    """
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    true_range = true_range.iloc[1:]  # drop first row (no previous close)
    return true_range.ewm(alpha=1.0 / period, adjust=False).mean()


def momentum_series(close: pd.Series) -> pd.Series:
    """Per-bar momentum score: average of the 3- and 6-month total returns.

    Mirrors screener.momentum.compute_momentum, the same measure used live.
    """
    return_3m = close / close.shift(LOOKBACK_3M) - 1.0
    return_6m = close / close.shift(LOOKBACK_6M) - 1.0
    return 0.5 * return_3m + 0.5 * return_6m


# --- Engine internals ---------------------------------------------------------

def _build_panels(
    bars: dict[str, pd.DataFrame],
    master: pd.DatetimeIndex,
    strategies: list[Strategy],
    liquidity_lookback: int = LIQUIDITY_LOOKBACK,
) -> dict[str, dict]:
    """Precompute indicators and per-strategy signals per symbol, aligned to master.

    Computing on each symbol's continuous history first guarantees the indicators
    and signals equal the live per-bar functions; reindexing afterward leaves NaN on
    dates a symbol had no data (so it is simply untradeable then).
    """
    n_days = len(master)
    panels: dict[str, dict] = {}
    for symbol, frame in bars.items():
        close, high, low, open_ = frame["Close"], frame["High"], frame["Low"], frame["Open"]
        atr = atr_series(high, low, close, ATR_PERIOD)
        mom = momentum_series(close)
        trend_ma = close.rolling(TREND_EXIT_MA).mean()  # MA for the trend-riding exit
        # Average daily dollar volume for the liquidity filter (NaN if no Volume).
        # LOOK-AHEAD GUARD: a trailing mean of completed bars; read at p when used.
        if "Volume" in frame.columns:
            dollar_volume = close * frame["Volume"]
            adv = dollar_volume.rolling(liquidity_lookback).mean().reindex(master).to_numpy(dtype=float)
        else:
            adv = np.full(n_days, np.nan)
        signals = {
            strat.name: strat.signal_series(frame, symbol).reindex(master, fill_value=False).to_numpy(dtype=bool)
            for strat in strategies
        }
        # Per-strategy trigger strength (0..1) for conviction sizing.
        strength = {
            strat.name: strat.strength_series(frame).reindex(master).to_numpy(dtype=float)
            for strat in strategies
        }
        panels[symbol] = {
            "open": open_.reindex(master).to_numpy(dtype=float),
            "high": high.reindex(master).to_numpy(dtype=float),
            "low": low.reindex(master).to_numpy(dtype=float),
            "close": close.reindex(master).to_numpy(dtype=float),
            "atr": atr.reindex(master).to_numpy(dtype=float),
            "mom": mom.reindex(master).to_numpy(dtype=float),
            "trend_ma": trend_ma.reindex(master).to_numpy(dtype=float),
            "adv": adv,
            "signals": signals,
            "strength": strength,
        }
    return panels


def _top_sectors_by_day(panels: dict[str, dict], etf_symbols: list[str], n_days: int) -> list[dict]:
    """Precompute, per day, the top-3 sector ETFs by momentum as ``{etf: rank}``.

    Rank is 1-based (1 = strongest). Membership in the dict is the sector gate; the
    rank feeds conviction sizing.
    """
    top_ranks = []
    for t in range(n_days):
        scored = [(etf, panels[etf]["mom"][t]) for etf in etf_symbols
                  if not np.isnan(panels[etf]["mom"][t])]
        scored.sort(key=lambda item: item[1], reverse=True)
        top_ranks.append({etf: rank for rank, (etf, _) in enumerate(scored[:3], start=1)})
    return top_ranks


def _record_trade(position: dict, exit_i: int, exit_price: float, reason: str,
                  master: pd.DatetimeIndex) -> dict:
    """Build a closed-trade record from a position and its exit."""
    proceeds = position["shares"] * exit_price - position["shares"] * position["commission_ps"]
    pnl = proceeds - position["entry_cost"]
    return {
        "symbol": position["symbol"],
        "strategy": position["strategy"],
        "sector": position["sector"],
        "entry_date": position["entry_date"],
        "entry_price": round(position["entry_price"], 4),
        "exit_date": master[exit_i],
        "exit_price": round(exit_price, 4),
        "shares": position["shares"],
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl / position["entry_cost"], 4) if position["entry_cost"] else 0.0,
        "regime_at_entry": position["regime_at_entry"],
        "bars_held": exit_i - position["entry_i"],
        "exit_reason": reason,
    }


def run_engine(
    bars: dict[str, pd.DataFrame],
    regime: pd.Series,
    sector_map: dict[str, str],
    etf_symbols: list[str],
    *,
    strategies: list[Strategy] | None = None,
    starting_equity: float = STARTING_EQUITY,
    commission_per_share: float = COMMISSION_PER_SHARE,
    slippage_pct: float = SLIPPAGE_PCT,
    warmup_bars: int = WARMUP_BARS,
    regime_filter: bool = REGIME_FILTER_ENABLED,
    enforce_min_size: bool = True,
    trend_exit: bool = TREND_EXIT_ENABLED,
    conviction_sizing: bool = CONVICTION_SIZING_ENABLED,
    start_date: pd.Timestamp | None = None,
    entry_filter=None,
    earnings_dates: dict | None = None,
    earnings_blackout: bool = EARNINGS_BLACKOUT_ENABLED,
    blackout_days: int = EARNINGS_BLACKOUT_DAYS,
    liquidity_filter: bool = False,
    stats: dict | None = None,
    overlay: bool = OVERLAY_ENABLED,
    overlay_returns: np.ndarray | None = None,
    strategy_active=None,
    slippage_fn=None,
    min_dollar_volume: float | None = None,
    min_price: float | None = None,
    max_adv_participation: float | None = None,
    liquidity_lookback: int | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run the walk-forward backtest.

    Args:
        bars: symbol -> adjusted OHLC DataFrame.
        regime: daily regime labels; its index defines the trading calendar.
        sector_map: symbol -> sector ETF (only these symbols are tradable).
        etf_symbols: the sector ETFs available for the momentum gate.
        strategies: active strategies in PRIORITY order; the first to fire on a name
            tags the entry. Defaults to breakout-only.
        regime_filter: when True, the prior day's regime scales new entries
            (half-size + tighter total cap in bear, no new entries in crash). When
            False, behavior matches the unfiltered baseline exactly.
        enforce_min_size: when True, drop positions below the minimum size floor.
        trend_exit: when True, winners (up >= 1 ATR) switch to a wide chandelier
            trail plus a trend-MA-break exit. When False, the standard trail is used.
        conviction_sizing: when True, scale per-trade risk by a conviction multiplier
            (before caps). When False, sizing is unchanged.
        start_date: if given, only trade/record from this date on (indicators still
            use full history). Used to backtest a held-out test period only.
        entry_filter: optional ``f(symbol, signal_date, strategy) -> bool``; when it
            returns False the candidate is skipped (e.g. an ML signal-quality gate).
        earnings_dates: optional ``{symbol: array of report Timestamps}`` for the
            earnings blackout (next-report dates are public in advance).
        earnings_blackout: when True, refuse new entries within ``blackout_days``
            sessions before a symbol's next known report (avoid holding through it).
        blackout_days: blackout width in trading sessions.
        liquidity_filter: when True, require trailing avg dollar volume >=
            MIN_DOLLAR_VOLUME and price >= MIN_PRICE before entry, and cap a position
            at MAX_ADV_PARTICIPATION of ADV (skip reason "illiquid").
        stats: optional dict that, if given, receives run counters (e.g.
            ``illiquid_skips``).
        overlay: when True, idle cash is invested in the overlay instrument and earns
            its daily return (marked to market at the close) instead of earning 0.
        overlay_returns: the overlay instrument's daily returns aligned to ``master``
            (required when ``overlay`` is True).
        strategy_active: optional ``f(regime_label) -> set of active strategy names``;
            evaluated on the prior-day regime so only those strategies may fire that
            day (a regime-aware selector). None means all strategies are always active.
        slippage_fn: optional ``f(symbol) -> per-side slippage fraction`` applied at
            every fill instead of the flat ``slippage_pct``. Lets an asset class price
            costs PER NAME (e.g. crypto, where thin coins pay far more). None keeps the
            flat ``slippage_pct`` for every symbol (prior behavior).
        min_dollar_volume / min_price / max_adv_participation / liquidity_lookback:
            optional overrides for the liquidity filter's thresholds (default to the
            config constants, i.e. prior behavior). A different asset class can retune
            them — e.g. crypto needs no $5 price floor and a different ADV reference.

    Returns:
        ``(equity_curve, trades)`` — a daily equity Series and a trade-log DataFrame.
    """
    if strategies is None:
        strategies = [BreakoutStrategy()]  # default: breakout-only (prior behavior)
    # Liquidity thresholds default to the config constants (unchanged behavior); an
    # asset class may override them per run without touching global config.
    min_dollar_volume = MIN_DOLLAR_VOLUME if min_dollar_volume is None else min_dollar_volume
    min_price = MIN_PRICE if min_price is None else min_price
    max_adv_participation = MAX_ADV_PARTICIPATION if max_adv_participation is None else max_adv_participation
    liq_lookback = LIQUIDITY_LOOKBACK if liquidity_lookback is None else liquidity_lookback
    # Per-side slippage: per-name when slippage_fn is given, else the flat rate.
    def slip(symbol: str) -> float:
        return slippage_fn(symbol) if slippage_fn is not None else slippage_pct
    master = regime.index
    n_days = len(master)
    panels = _build_panels(bars, master, strategies, liq_lookback)
    regime_arr = regime.to_numpy()
    top_ranks = _top_sectors_by_day(panels, [e for e in etf_symbols if e in panels], n_days)

    # Only screener constituents we actually loaded are tradable candidates.
    tradable = [s for s in sector_map if s in panels]

    # Map each symbol's report dates to sorted master positions for the blackout.
    blackout_pos: dict[str, np.ndarray] = {}
    if earnings_blackout and earnings_dates:
        for symbol, dates in earnings_dates.items():
            if len(dates) == 0:
                continue
            positions = master.searchsorted(np.sort(pd.DatetimeIndex(dates).to_numpy()))
            blackout_pos[symbol] = np.asarray(positions)

    cash = starting_equity
    book: dict[str, dict] = {}
    trades: list[dict] = []
    equity_dates: list = []
    equity_values: list[float] = []
    illiquid_skips = 0

    start_index = max(warmup_bars, 1)
    if start_date is not None:
        # Trade/record only from start_date on; indicators above still use full
        # history. This isolates a held-out test period without losing warmup.
        start_index = max(start_index, int(master.searchsorted(pd.Timestamp(start_date))))
    for i in range(start_index, n_days):
        p = i - 1

        # --- Equity & open book valued at the PRIOR close ------------------------
        # LOOK-AHEAD GUARD: size and cap against close[p] (known at today's open),
        # never close[i]. last_close was set to close[p] at the end of day p.
        book_positions = [
            {"symbol": s, "shares": pos["shares"], "market_value": pos["shares"] * pos["last_close"]}
            for s, pos in book.items()
        ]
        equity_prev = cash + sum(bp["market_value"] for bp in book_positions)

        # --- Entries at TODAY's OPEN --------------------------------------------
        # LOOK-AHEAD GUARD: every signal input — including the regime — is read at p
        # (the prior completed bar). We never look at day i's regime or prices here.
        prior_regime = regime_arr[p]

        # Regime gate. Only active when the filter is on; otherwise these are no-ops
        # (size_mult=1.0, default total cap, entries not blocked) so the run matches
        # the unfiltered baseline exactly.
        block_entries = regime_filter and CRASH_BLOCK_NEW_ENTRIES and prior_regime == "crash"
        if regime_filter and prior_regime == "bear":
            size_mult = BEAR_SIZE_MULT          # bear: half-size each new position
            day_total_cap = BEAR_MAX_TOTAL_EXPOSURE  # bear: tighter total cap today
        else:
            size_mult = 1.0
            day_total_cap = None                # None -> normal MAX_TOTAL_EXPOSURE

        candidates = []
        if not block_entries:  # crash: take no new entries today (exits still run)
            sector_ranks = top_ranks[p]
            # Regime-aware selector: which strategies may fire today, from the PRIOR
            # day's regime. LOOK-AHEAD GUARD: regime read at p, never at i.
            active_names = strategy_active(prior_regime) if strategy_active is not None else None
            for symbol in tradable:
                if symbol in book:  # at most one open position per name
                    continue
                panel = panels[symbol]
                # Check strategies in priority order; the first to fire as of the
                # prior bar p tags the entry. LOOK-AHEAD GUARD: signals read at p.
                triggered = None
                for strat in strategies:
                    if active_names is not None and strat.name not in active_names:
                        continue
                    if panel["signals"][strat.name][p]:
                        triggered = strat.name
                        break
                if triggered is None:
                    continue
                atr_p, mom_p, open_i = panel["atr"][p], panel["mom"][p], panel["open"][i]
                if np.isnan(atr_p) or atr_p <= 0 or np.isnan(mom_p) or np.isnan(open_i):
                    continue
                sector_rank = sector_ranks.get(sector_map.get(symbol))  # sector gate as of p
                if sector_rank is None:
                    continue

                # Liquidity gate (all strategies). LOOK-AHEAD GUARD: ADV and price are
                # read at the prior bar p. Require enough dollar volume and a min price
                # so we don't model trades we couldn't realistically fill.
                adv_p = panel["adv"][p]
                if liquidity_filter:
                    if np.isnan(adv_p) or adv_p < min_dollar_volume or panel["close"][p] < min_price:
                        illiquid_skips += 1
                        continue

                # Earnings blackout (all strategies): refuse to ENTER within
                # blackout_days sessions before this symbol's next known report, so
                # we never hold a fresh position through an announcement. The entry
                # would fill at i; report positions are known in advance (public).
                if symbol in blackout_pos:
                    rp = blackout_pos[symbol]
                    k = int(np.searchsorted(rp, i, side="left"))  # first report at/after entry
                    if k < len(rp) and 0 <= rp[k] - i <= blackout_days:
                        continue

                # Optional external entry filter (e.g. an ML signal-quality gate),
                # evaluated on the signal day p. Returning False skips this entry.
                if entry_filter is not None and not entry_filter(symbol, master[p], triggered):
                    continue

                # Conviction sizing: scale per-trade risk by the setup's strength.
                # LOOK-AHEAD GUARD: rank, momentum, and signal strength all read at p.
                if conviction_sizing:
                    strength_p = panel["strength"][triggered][p]
                    strength_p = 0.0 if np.isnan(strength_p) else strength_p
                    score = score_from_factors(sector_rank, mom_p, strength_p)
                    risk_mult = conviction_multiplier(score)
                else:
                    risk_mult = 1.0

                # Entry fills at today's open plus slippage (never the signal close).
                entry_fill = open_i * (1.0 + slip(symbol))
                stop = compute_stop(entry_fill, atr_p)                          # reuse risk/position
                shares, _ = size_position(equity_prev, entry_fill, stop, risk_mult)  # 1% risk x conviction, 10% cap
                if size_mult != 1.0:  # bear: scale the share size down
                    shares = int(math.floor(shares * size_mult))
                # Liquidity cap: a position may not exceed max_adv_participation of the
                # name's average daily dollar volume (don't model unfillable size).
                if liquidity_filter and entry_fill > 0:
                    max_shares_adv = int(math.floor(max_adv_participation * adv_p / entry_fill))
                    shares = min(shares, max_shares_adv)
                if shares <= 0:
                    continue
                candidates.append({
                    "symbol": symbol, "strategy": triggered,
                    "entry_ref": entry_fill, "stop": stop,
                    "shares": shares, "est_value": shares * entry_fill,
                    "atr": atr_p, "momentum": mom_p,
                })

        # Strongest-first, then portfolio caps (30% sector, total cap by regime) vs book.
        candidates.sort(key=lambda c: c["momentum"], reverse=True)
        accepted, _ = apply_portfolio_limits(
            candidates, equity_prev, book_positions, sector_map,
            max_total_pct=day_total_cap, enforce_min_size=enforce_min_size,
        )

        for proposal in accepted:
            symbol, entry_fill = proposal["symbol"], proposal["entry_ref"]
            # Cash constraint: never spend cash we do not have.
            unit_cost = entry_fill + commission_per_share
            shares = min(proposal["shares"], int(cash // unit_cost)) if unit_cost > 0 else 0
            if shares < 1:
                continue
            # Re-apply the min-size floor after the cash trim (also a sizing finalize).
            if enforce_min_size and not meets_min_size(shares, entry_fill, equity_prev):
                continue
            cost = shares * entry_fill + shares * commission_per_share
            cash -= cost
            initial_stop = compute_stop(entry_fill, proposal["atr"])
            book[symbol] = {
                "symbol": symbol, "strategy": proposal["strategy"], "sector": sector_map.get(symbol),
                "shares": shares, "entry_price": entry_fill, "entry_cost": cost,
                "commission_ps": commission_per_share, "atr_entry": proposal["atr"],
                "initial_stop": initial_stop, "current_stop": initial_stop,
                "highest_close": -np.inf, "highest_high": -np.inf, "trend_mode": False,
                "entry_i": i, "entry_date": master[i],
                "regime_at_entry": regime_arr[p], "last_close": entry_fill,
            }

        # --- Exits during today (positions opened on a PRIOR day) ----------------
        # LOOK-AHEAD GUARD: highs/closes/MA are updated only through p, so every stop
        # level (and the MA-break signal) checked against today's bar was knowable
        # before today. When trend_exit is False this reduces to the standard trail.
        for symbol in list(book.keys()):
            position = book[symbol]
            if position["entry_i"] == i:  # entered today; not eligible to exit yet
                continue
            panel = panels[symbol]
            close_p, high_p = panel["close"][p], panel["high"][p]
            if not np.isnan(close_p):
                position["highest_close"] = max(position["highest_close"], close_p)
            if not np.isnan(high_p):
                position["highest_high"] = max(position["highest_high"], high_p)

            # Latch trend-riding mode once the position is up >= 1 ATR (on close).
            if trend_exit and not position["trend_mode"] and not np.isnan(close_p):
                if close_p - position["entry_price"] >= position["atr_entry"]:
                    position["trend_mode"] = True

            # Update the active trailing stop (ratchets up only).
            if trend_exit and position["trend_mode"]:
                # Winner: wide chandelier trail = highest high since entry - mult*ATR.
                chandelier = position["highest_high"] - CHANDELIER_ATR_MULT * position["atr_entry"]
                position["current_stop"] = max(position["current_stop"], chandelier)
            elif not np.isnan(close_p):
                # Pre-profit / feature off: standard 2*ATR trail on the highest close.
                trail = position["highest_close"] - ATR_MULTIPLE * position["atr_entry"]
                position["current_stop"] = max(position["current_stop"], trail)

            open_i, low_i = panel["open"][i], panel["low"][i]

            # Trend-MA-break exit (winners only): a completed close below the trend MA
            # exits at today's OPEN (the next bar after the signal) — no look-ahead.
            if trend_exit and position["trend_mode"]:
                ma_p = panel["trend_ma"][p]
                if (not np.isnan(ma_p) and not np.isnan(close_p)
                        and close_p < ma_p and not np.isnan(open_i)):
                    exit_fill = open_i * (1.0 - slip(symbol))
                    cash += position["shares"] * exit_fill - position["shares"] * commission_per_share
                    trades.append(_record_trade(position, i, exit_fill, "ma_break", master))
                    del book[symbol]
                    continue

            # Price-stop exit: today's low touches the trailing/initial stop.
            if np.isnan(low_i):
                continue
            if low_i <= position["current_stop"]:
                # Fill at the stop, or at the open if the day gapped through it.
                if not np.isnan(open_i) and open_i < position["current_stop"]:
                    raw = open_i
                else:
                    raw = position["current_stop"]
                exit_fill = raw * (1.0 - slip(symbol))
                cash += position["shares"] * exit_fill - position["shares"] * commission_per_share
                if position["trend_mode"]:
                    reason = "chandelier_stop"
                else:
                    reason = "trailing_stop" if position["current_stop"] > position["initial_stop"] + 1e-9 else "stop"
                trades.append(_record_trade(position, i, exit_fill, reason, master))
                del book[symbol]

        # --- End-of-day mark-to-market (reporting only) --------------------------
        # Marking the equity curve to close[i] is fine: it feeds reports, never a
        # decision. Sizing/caps above already used close[p].
        for symbol, position in book.items():
            close_i = panels[symbol]["close"][i]
            if not np.isnan(close_i):
                position["last_close"] = close_i

        # Index overlay: idle cash earns the overlay instrument's day-i return.
        # LOOK-AHEAD GUARD: overlay_returns[i] is day i's return, applied only to the
        # day-i equity mark (reporting). equity_prev used for sizing is as-of p, so no
        # future info enters any decision. (Intraday flow timing is approximated at
        # the daily close: today's entries are removed before growth, exits after.)
        if overlay and overlay_returns is not None and not np.isnan(overlay_returns[i]):
            cash *= 1.0 + overlay_returns[i]

        equity_values.append(cash + sum(pos["shares"] * pos["last_close"] for pos in book.values()))
        equity_dates.append(master[i])

    # --- Liquidate any open positions at the final close ------------------------
    final_i = n_days - 1
    for symbol in list(book.keys()):
        position = book[symbol]
        close_f = panels[symbol]["close"][final_i]
        price = position["last_close"] if np.isnan(close_f) else close_f
        exit_fill = price * (1.0 - slip(symbol))
        cash += position["shares"] * exit_fill - position["shares"] * commission_per_share
        trades.append(_record_trade(position, final_i, exit_fill, "end_of_backtest", master))
        del book[symbol]

    if stats is not None:
        stats["illiquid_skips"] = illiquid_skips
    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates), name="equity")
    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    log.info("Backtest complete: %d trades, final equity $%s.",
             len(trades_df), f"{equity_curve.iloc[-1]:,.0f}" if not equity_curve.empty else "n/a")
    return equity_curve, trades_df
