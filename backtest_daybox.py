# backtest_daybox.py
"""Numba backtest of the "Day Box Master 15M / 30M / 1H" Pine strategy.

Same contract as backtest_fast.run_backtest ({"trade_dicts": [...], "indicators": {}}).

The previous calendar day's high/low forms the box. Two entry modes:
  * breakout + retest: a close beyond the box, then a later bar that touches the
    broken level (within retestTolerancePct) and closes back outside it;
  * rejection / bounce: a bar touching the opposite edge (within
    rejectionTolerancePct) that closes back inside the box.
Stop sits slBufferPct beyond the signal bar (or box edge for rejections) and the
target is riskReward times the stop distance. Entries fill at the signal bar close;
a new entry may fire on the bar where the previous trade exited (Pine
calc_on_order_fills). One position at a time (pyramiding 0).

Optimized parameters: retestTolerancePct, rejectionTolerancePct, riskReward,
slBufferPct. The remaining parameters are fixed switches mirroring the Pine inputs.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from numba import njit

from backtest_fast import _REASONS, _record

DAYBOX_DEFAULTS: Dict[str, Any] = {
    "retestTolerancePct": 0.10,
    "rejectionTolerancePct": 0.10,
    "riskReward": 2.0,
    "slBufferPct": 0.10,
}

_SL, _TP, _EOD = 1, 2, 4
_PATHS = {"olhc": 0, "ohlc": 1, "tv": 2}


@njit(cache=True)
def _exit_check(is_long, stop, target, o, h, l, path):
    """Returns (reason, fill). Gaps through a level fill at the open; when a bar
    touches both levels, `path` decides which is hit first
    (0: low first, 1: high first, 2: TradingView's nearest-extreme-to-open rule)."""
    if is_long:
        if o <= stop:
            return _SL, o
        if o >= target:
            return _TP, o
        hit_stop = l <= stop
        hit_tp = h >= target
    else:
        if o >= stop:
            return _SL, o
        if o <= target:
            return _TP, o
        hit_stop = h >= stop
        hit_tp = l <= target
    if not hit_stop and not hit_tp:
        return 0, 0.0
    if hit_stop and hit_tp:
        if path == 2:
            high_first = abs(h - o) < abs(l - o)
        else:
            high_first = path == 1
        # a long's target is the high, a short's stop is the high
        if high_first == is_long:
            return _TP, target
        return _SL, stop
    if hit_stop:
        return _SL, stop
    return _TP, target


@njit(cache=True)
def _simulate(open_, high, low, close, day_id,
              retest_pct, reject_pct, rr, sl_buf_pct,
              en_breakouts, en_rejections, en_longs, en_shorts,
              req_bo_close, req_close_inside, mintick,
              slippage, commission_pct, position_size, path):
    n = len(close)
    tr_side = np.empty(n + 1, dtype=np.int8)
    tr_entry_bar = np.empty(n + 1, dtype=np.int64)
    tr_exit_bar = np.empty(n + 1, dtype=np.int64)
    tr_entry_px = np.empty(n + 1)
    tr_exit_px = np.empty(n + 1)
    tr_size = np.empty(n + 1)
    tr_pnl = np.empty(n + 1)
    tr_gross = np.empty(n + 1)
    tr_comm = np.empty(n + 1)
    tr_reason = np.empty(n + 1, dtype=np.int8)
    nt = 0

    day_high = high[0]
    day_low = low[0]
    prev_high = np.nan
    prev_low = np.nan

    bull_active = False
    bear_active = False
    bull_bar = -1
    bear_bar = -1

    pos = 0  # 1 long, -1 short, 0 flat
    entry_px = 0.0
    stop = 0.0
    target = 0.0
    entry_bar = 0

    for i in range(n):
        o = open_[i]
        h = high[i]
        l = low[i]
        c = close[i]

        # Broker emulator: exits resolve before the script runs on this bar.
        if pos != 0:
            reason, level = _exit_check(pos == 1, stop, target, o, h, l, path)
            if reason != 0:
                fill = level - slippage if pos == 1 else level + slippage
                _record(nt, tr_side, tr_entry_bar, tr_exit_bar, tr_entry_px, tr_exit_px, tr_size,
                        tr_pnl, tr_gross, tr_comm, tr_reason,
                        pos, entry_bar, entry_px, position_size, fill, reason, i, commission_pct)
                nt += 1
                pos = 0

        if i > 0 and day_id[i] != day_id[i - 1]:
            prev_high = day_high
            prev_low = day_low
            day_high = h
            day_low = l
            bull_active = False
            bear_active = False
            bull_bar = -1
            bear_bar = -1
        else:
            if h > day_high:
                day_high = h
            if l < day_low:
                day_low = l

        valid = (not np.isnan(prev_high)) and (not np.isnan(prev_low)) and prev_high - prev_low > 0
        if not valid:
            continue

        bull_breakout = c > prev_high if req_bo_close else h > prev_high
        bear_breakout = c < prev_low if req_bo_close else l < prev_low
        if bull_breakout and not bull_active:
            bull_active = True
            bear_active = False
            bull_bar = i
            bear_bar = -1
        if bear_breakout and not bear_active:
            bear_active = True
            bull_active = False
            bear_bar = i
            bull_bar = -1

        up_tol = prev_high * (retest_pct / 100.0)
        lo_tol = prev_low * (retest_pct / 100.0)
        bull_retest = (bull_active and bull_bar >= 0 and i > bull_bar
                       and l <= prev_high + up_tol and h >= prev_high - up_tol and c > prev_high)
        bear_retest = (bear_active and bear_bar >= 0 and i > bear_bar
                       and h >= prev_low - lo_tol and l <= prev_low + lo_tol and c < prev_low)

        long_touched = l <= prev_low + prev_low * (reject_pct / 100.0)
        short_touched = h >= prev_high - prev_high * (reject_pct / 100.0)
        long_closed = c > prev_low if req_close_inside else c > o
        short_closed = c < prev_high if req_close_inside else c < o

        breakout_long = en_breakouts and en_longs and bull_retest
        breakout_short = en_breakouts and en_shorts and bear_retest
        rejection_long = en_rejections and en_longs and long_touched and long_closed
        rejection_short = en_rejections and en_shorts and short_touched and short_closed

        if pos != 0:
            continue

        long_sig = breakout_long or rejection_long
        short_sig = (not long_sig) and (breakout_short or rejection_short)

        if long_sig:
            if breakout_long:
                sl = l * (1.0 - sl_buf_pct / 100.0)
                bull_active = False
                bull_bar = -1
            else:
                sl = min(l, prev_low) * (1.0 - sl_buf_pct / 100.0)
            risk = c - sl
            if risk > mintick:
                pos = 1
                stop = sl
                target = c + risk * rr
                entry_px = c + slippage
                entry_bar = i
        elif short_sig:
            if breakout_short:
                sl = h * (1.0 + sl_buf_pct / 100.0)
                bear_active = False
                bear_bar = -1
            else:
                sl = max(h, prev_high) * (1.0 + sl_buf_pct / 100.0)
            risk = sl - c
            if risk > mintick:
                pos = -1
                stop = sl
                target = c - risk * rr
                entry_px = c - slippage
                entry_bar = i

    if pos != 0:
        final_close = close[n - 1]
        fill = final_close - slippage if pos == 1 else final_close + slippage
        _record(nt, tr_side, tr_entry_bar, tr_exit_bar, tr_entry_px, tr_exit_px, tr_size,
                tr_pnl, tr_gross, tr_comm, tr_reason,
                pos, entry_bar, entry_px, position_size, fill, _EOD, n - 1, commission_pct)
        nt += 1

    return (nt, tr_side, tr_entry_bar, tr_exit_bar, tr_entry_px, tr_exit_px, tr_size,
            tr_pnl, tr_gross, tr_comm, tr_reason)


class _DayBoxArrays:
    """Numpy view of a candle list plus memoized per-day ids."""

    def __init__(self, candles: List[Dict[str, Any]]):
        self.times = [c.get("time") for c in candles]
        self.open = np.array([float(c["open"]) for c in candles])
        self.high = np.array([float(c["high"]) for c in candles])
        self.low = np.array([float(c["low"]) for c in candles])
        self.close = np.array([float(c["close"]) for c in candles])
        self._days: Dict[Any, np.ndarray] = {}

    def day_ids(self, timezone: str, utc_offset: int) -> np.ndarray:
        key = (timezone, utc_offset)
        if key not in self._days:
            index = pd.DatetimeIndex(self.times)
            index = index.tz_localize("UTC") if index.tz is None else index
            local = index.tz_convert(timezone) + pd.Timedelta(hours=utc_offset)
            days = local.tz_localize(None).values.astype("datetime64[D]").astype(np.int64)
            self._days[key] = days
        return self._days[key]


_CACHES: List[Any] = []
_MAX_CACHED_CANDLE_SETS = 8


def _arrays_for(candles: List[Dict[str, Any]]) -> _DayBoxArrays:
    for i, (ref, length, data) in enumerate(_CACHES):
        if ref is candles and length == len(candles):
            if i:
                _CACHES.insert(0, _CACHES.pop(i))
            return data
    data = _DayBoxArrays(candles)
    _CACHES.insert(0, (candles, len(candles), data))
    del _CACHES[_MAX_CACHED_CANDLE_SETS:]
    return data


def run_backtest(candles: List[Dict[str, Any]], params: Dict[str, Any],
                 include_indicators: bool = True) -> Dict[str, Any]:
    """Run the Day Box strategy. `include_indicators` is accepted for interface
    parity with backtest_fast; this strategy has no indicator series."""
    if len(candles) == 0:
        return {"trade_dicts": [], "indicators": {}}

    def num(name: str) -> float:
        return float(params.get(name, DAYBOX_DEFAULTS[name]))

    data = _arrays_for(candles)
    day_id = data.day_ids(str(params.get("dayTimezone", "America/New_York")),
                          int(params.get("utcOffset", 0)))
    path = _PATHS.get(str(params.get("intrabar_path", "ohlc")).lower(), 1)

    (nt, side, entry_bar, exit_bar, entry_px, exit_px, size, pnl, gross, comm, reason) = _simulate(
        data.open, data.high, data.low, data.close, day_id,
        num("retestTolerancePct"), num("rejectionTolerancePct"), num("riskReward"),
        num("slBufferPct"),
        bool(params.get("enableBreakouts", True)), bool(params.get("enableRejections", True)),
        bool(params.get("enableLongs", True)), bool(params.get("enableShorts", True)),
        bool(params.get("requireBreakoutClose", True)),
        bool(params.get("requireCloseInside", True)),
        float(params.get("mintick", 0.01)),
        float(params.get("slippage", 0.0)), float(params.get("commission_pct", 0.0)),
        float(params.get("position_size", params.get("contracts", 1.0))), path,
    )

    times = data.times
    trade_dicts = []
    for k in range(nt):
        eb = int(entry_bar[k])
        xb = int(exit_bar[k])
        trade_dicts.append({
            "side": "long" if side[k] == 1 else "short",
            "entry_time": times[eb],
            "entry_price": float(entry_px[k]),
            "exit_time": times[xb],
            "exit_price": float(exit_px[k]),
            "size": float(size[k]),
            "pnl": float(pnl[k]),
            "pnl_gross": float(gross[k]),
            "commission": float(comm[k]),
            "exit_reason": _REASONS[reason[k]],
            "entry_bar_index": eb,
            "exit_bar_index": xb,
        })
    return {"trade_dicts": trade_dicts, "indicators": {}}


def run_backtest_lean(candles: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    return run_backtest(candles, params, include_indicators=False)
