# backtest_fast.py
"""Numba-accelerated drop-in for backtest_engine.run_backtest.

Same signature and result shape ({"trade_dicts": [...], "indicators": {...}});
backtest_engine.py remains the reference implementation and tests/test_fast_engine.py
checks the two produce identical trades.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from numba import njit

from presets import get_presets

_REASONS = ("", "sl", "tp", "reverse", "eod")
_SL, _TP, _REVERSE, _EOD = 1, 2, 3, 4


@njit(cache=True)
def _true_range(high, low, close):
    n = len(close)
    tr = np.empty(n)
    for i in range(n):
        prev_close = close[i - 1] if i > 0 else close[i]
        tr[i] = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
    return tr


@njit(cache=True)
def _sma(series, length):
    n = len(series)
    out = np.full(n, np.nan)
    if length <= 0:
        return out
    window_sum = 0.0
    for i in range(n):
        window_sum += series[i]
        if i >= length:
            window_sum -= series[i - length]
        if i >= length - 1:
            out[i] = window_sum / length
    return out


@njit(cache=True)
def _ema(close, length):
    n = len(close)
    out = np.empty(n)
    alpha = 2.0 / (length + 1.0)
    for i in range(n):
        if i == 0:
            out[i] = close[i]
        else:
            out[i] = alpha * close[i] + (1 - alpha) * out[i - 1]
    return out


@njit(cache=True)
def _supertrend(close, atr, mult):
    n = len(close)
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    trend = np.ones(n, dtype=np.int8)
    for i in range(n):
        if np.isnan(atr[i]):
            trend[i] = trend[i - 1] if i > 0 else 1
            continue
        base_up = close[i] - mult * atr[i]
        base_dn = close[i] + mult * atr[i]
        up1 = up[i - 1] if (i > 0 and not np.isnan(up[i - 1])) else base_up
        dn1 = dn[i - 1] if (i > 0 and not np.isnan(dn[i - 1])) else base_dn
        prev_close = close[i - 1] if i > 0 else close[i]
        up[i] = max(base_up, up1) if prev_close > up1 else base_up
        dn[i] = min(base_dn, dn1) if prev_close < dn1 else base_dn
        if i == 0:
            trend[i] = 1
        elif trend[i - 1] == -1 and close[i] > dn1:
            trend[i] = 1
        elif trend[i - 1] == 1 and close[i] < up1:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    return up, dn, trend


@njit(cache=True)
def _exit_check(is_long, stop, limit, o, h, l, c, ohlc):
    """Returns 0 (no exit), 1 (stop) or 2 (limit), following the reference engine's
    open -> high/low -> close stage order."""
    for stage in range(4):
        # stage ids: 0 open, 1 high, 2 low, 3 close; olhc swaps high and low
        s = stage
        if not ohlc and (stage == 1 or stage == 2):
            s = 3 - stage
        if s == 0:
            if is_long:
                if o <= stop:
                    return 1
                if o >= limit:
                    return 2
            else:
                if o >= stop:
                    return 1
                if o <= limit:
                    return 2
        elif s == 1:
            if is_long:
                if h >= limit:
                    return 2
                if h <= stop:
                    return 1
            else:
                if h >= stop:
                    return 1
                if h <= limit:
                    return 2
        elif s == 2:
            if is_long:
                if l <= stop:
                    return 1
                if l >= limit:
                    return 2
            else:
                if l <= limit:
                    return 2
                if l >= stop:
                    return 1
        else:
            if is_long:
                if c <= stop:
                    return 1
                if c >= limit:
                    return 2
            else:
                if c >= stop:
                    return 1
                if c <= limit:
                    return 2
    return 0


@njit(cache=True)
def _record(out_i, tr_side, tr_entry_bar, tr_exit_bar, tr_entry_px, tr_exit_px, tr_size,
            tr_pnl, tr_gross, tr_comm, tr_reason,
            side, entry_bar, entry_px, size, fill, reason, exit_bar, commission_pct):
    commission = abs(fill * size) * commission_pct
    if side == 1:
        gross = (fill - entry_px) * size
    else:
        gross = (entry_px - fill) * size
    tr_side[out_i] = side
    tr_entry_bar[out_i] = entry_bar
    tr_exit_bar[out_i] = exit_bar
    tr_entry_px[out_i] = entry_px
    tr_exit_px[out_i] = fill
    tr_size[out_i] = size
    tr_pnl[out_i] = gross - commission
    tr_gross[out_i] = gross
    tr_comm[out_i] = commission
    tr_reason[out_i] = reason


@njit(cache=True)
def _simulate(open_, high, low, close, atr, trend, ema,
              max_candle_atr, atr_sl_mult, atr_tp_mult, slippage, commission_pct, position_size,
              pyramiding, ohlc, same_bar):
    n = len(close)
    cap = max(pyramiding, 1) + 1
    p_side = np.zeros(cap, dtype=np.int8)  # 1 long, -1 short
    p_px = np.zeros(cap)
    p_bar = np.zeros(cap, dtype=np.int64)
    n_pos = 0
    long_count = 0
    short_count = 0

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

    for i in range(n):
        o = open_[i]
        h = high[i]
        l = low[i]
        c = close[i]
        a = atr[i]
        have_atr = not np.isnan(a)

        long_sig = False
        short_sig = False
        # Pine: |close-open| >= nz(atr[1], atr) * maxEntryCandleATR rejects the entry
        big_breakout = False
        if have_atr:
            atr_ref = atr[i - 1] if (i > 0 and not np.isnan(atr[i - 1])) else a
            big_breakout = abs(c - o) >= atr_ref * max_candle_atr

        if i > 0 and not big_breakout:
            if trend[i] == 1 and trend[i - 1] == -1 and ema[i] > ema[i - 1]:
                long_sig = True
            elif trend[i] == -1 and trend[i - 1] == 1 and ema[i] < ema[i - 1]:
                short_sig = True

        atr_sl = a * atr_sl_mult
        atr_tp = a * atr_tp_mult

        if have_atr:
            kept = 0
            for k in range(n_pos):
                is_long = p_side[k] == 1
                px = p_px[k]
                if is_long:
                    stop = px - atr_sl
                    limit = px + atr_tp
                else:
                    stop = px + atr_sl
                    limit = px - atr_tp
                res = _exit_check(is_long, stop, limit, o, h, l, c, ohlc)
                if res == 0:
                    p_side[kept] = p_side[k]
                    p_px[kept] = px
                    p_bar[kept] = p_bar[k]
                    kept += 1
                    continue
                level = stop if res == 1 else limit
                if is_long:
                    fill = level + slippage if res == 1 else level - slippage
                    long_count -= 1
                else:
                    fill = level - slippage if res == 1 else level + slippage
                    short_count -= 1
                _record(nt, tr_side, tr_entry_bar, tr_exit_bar, tr_entry_px, tr_exit_px, tr_size,
                        tr_pnl, tr_gross, tr_comm, tr_reason,
                        p_side[k], p_bar[k], px, position_size, fill, res, i, commission_pct)
                nt += 1
            n_pos = kept

        if long_sig or short_sig:
            kept = 0
            for k in range(n_pos):
                opposite = (long_sig and p_side[k] == -1) or (short_sig and p_side[k] == 1)
                if not opposite:
                    p_side[kept] = p_side[k]
                    p_px[kept] = p_px[k]
                    p_bar[kept] = p_bar[k]
                    kept += 1
                    continue
                if long_sig:
                    fill = c + slippage
                    short_count -= 1
                else:
                    fill = c - slippage
                    long_count -= 1
                _record(nt, tr_side, tr_entry_bar, tr_exit_bar, tr_entry_px, tr_exit_px, tr_size,
                        tr_pnl, tr_gross, tr_comm, tr_reason,
                        p_side[k], p_bar[k], p_px[k], position_size, fill, 3, i, commission_pct)
                nt += 1
            n_pos = kept

            if long_sig and long_count < pyramiding:
                p_side[n_pos] = 1
                p_px[n_pos] = c + slippage
                p_bar[n_pos] = i
                n_pos += 1
                long_count += 1
            elif short_sig and short_count < pyramiding:
                p_side[n_pos] = -1
                p_px[n_pos] = c - slippage
                p_bar[n_pos] = i
                n_pos += 1
                short_count += 1

        if same_bar and n_pos > 0 and have_atr:
            kept = 0
            for k in range(n_pos):
                if p_bar[k] != i:
                    p_side[kept] = p_side[k]
                    p_px[kept] = p_px[k]
                    p_bar[kept] = p_bar[k]
                    kept += 1
                    continue
                is_long = p_side[k] == 1
                px = p_px[k]
                if is_long:
                    stop = px - atr_sl
                    limit = px + atr_tp
                else:
                    stop = px + atr_sl
                    limit = px - atr_tp
                res = _exit_check(is_long, stop, limit, o, h, l, c, ohlc)
                if res == 0:
                    p_side[kept] = p_side[k]
                    p_px[kept] = px
                    p_bar[kept] = p_bar[k]
                    kept += 1
                    continue
                level = stop if res == 1 else limit
                if is_long:
                    fill = level + slippage if res == 1 else level - slippage
                    long_count -= 1
                else:
                    fill = level - slippage if res == 1 else level + slippage
                    short_count -= 1
                _record(nt, tr_side, tr_entry_bar, tr_exit_bar, tr_entry_px, tr_exit_px, tr_size,
                        tr_pnl, tr_gross, tr_comm, tr_reason,
                        p_side[k], p_bar[k], px, position_size, fill, res, i, commission_pct)
                nt += 1
            n_pos = kept

    final_close = close[n - 1]
    for k in range(n_pos):
        if p_side[k] == 1:
            fill = final_close - slippage
        else:
            fill = final_close + slippage
        _record(nt, tr_side, tr_entry_bar, tr_exit_bar, tr_entry_px, tr_exit_px, tr_size,
                tr_pnl, tr_gross, tr_comm, tr_reason,
                p_side[k], p_bar[k], p_px[k], position_size, fill, 4, n - 1, commission_pct)
        nt += 1

    return (nt, tr_side, tr_entry_bar, tr_exit_bar, tr_entry_px, tr_exit_px, tr_size,
            tr_pnl, tr_gross, tr_comm, tr_reason)


# Randomized sampling touches every (period, multiplier, emaLen) combination,
# so unbounded caches grow to gigabytes; keep them small.
_MAX_CACHED_SERIES = 48


class _CandleArrays:
    """Numpy view of a candle list plus memoized indicators, reused across a grid."""

    def __init__(self, candles: List[Dict[str, Any]]):
        self.times = [c.get("time") for c in candles]
        self.open = np.array([float(c["open"]) for c in candles])
        self.high = np.array([float(c["high"]) for c in candles])
        self.low = np.array([float(c["low"]) for c in candles])
        self.close = np.array([float(c["close"]) for c in candles])
        self.tr = _true_range(self.high, self.low, self.close)
        self._atr: Dict[int, np.ndarray] = {}
        self._st: Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._ema: Dict[int, np.ndarray] = {}

    def atr(self, period: int) -> np.ndarray:
        if period not in self._atr:
            self._atr[period] = _sma(self.tr, period)
        return self._atr[period]

    def supertrend(self, period: int, mult: float):
        key = (period, mult)
        if key not in self._st:
            if len(self._st) >= _MAX_CACHED_SERIES:
                self._st.clear()
            self._st[key] = _supertrend(self.close, self.atr(period), mult)
        return self._st[key]

    def ema(self, length: int) -> np.ndarray:
        if length not in self._ema:
            if len(self._ema) >= _MAX_CACHED_SERIES:
                self._ema.clear()
            self._ema[length] = _ema(self.close, length)
        return self._ema[length]


_CACHES: List[Tuple[List[Dict[str, Any]], int, _CandleArrays]] = []
_MAX_CACHED_CANDLE_SETS = 8


def _arrays_for(candles: List[Dict[str, Any]]) -> _CandleArrays:
    for i, (ref, length, data) in enumerate(_CACHES):
        if ref is candles and length == len(candles):
            if i:
                _CACHES.insert(0, _CACHES.pop(i))
            return data
    data = _CandleArrays(candles)
    _CACHES.insert(0, (candles, len(candles), data))
    del _CACHES[_MAX_CACHED_CANDLE_SETS:]
    return data


def _opt(arr: np.ndarray) -> List[Any]:
    return [None if v != v else float(v) for v in arr.tolist()]


def run_backtest(candles: List[Dict[str, Any]], params: Dict[str, Any],
                 include_indicators: bool = True) -> Dict[str, Any]:
    """Same contract as backtest_engine.run_backtest."""
    ticker = params.get("ticker", "NVDA")
    timeframe = params.get("timeframe") or "15M"
    p = get_presets(ticker, timeframe)

    st_mult = float(params.get("stMultiplier", p.get("stMultiplier", 3.0)))
    st_period = int(params.get("stPeriod", p.get("stPeriod", 12)))
    atr_sl = float(params.get("atrSLmult", p.get("atrSLmult", 1.4)))
    atr_tp = float(params.get("atrTPmult", p.get("atrTPmult", 2.4)))
    ema_len = int(params.get("emaLen", p.get("emaLen", 200)))
    max_candle_atr = float(params.get("maxEntryCandleATR", p.get("maxEntryCandleATR", 1.5)))

    slippage = float(params.get("slippage", 0.0))
    commission_pct = float(params.get("commission_pct", 0.0))
    position_size = float(params.get("position_size", params.get("contracts", 1.0)))
    pyramiding = int(params.get("pyramiding", 1))
    ohlc = params.get("intrabar_path", "ohlc") == "ohlc"
    same_bar = bool(params.get("allow_same_bar_exit_entry", False))

    if len(candles) == 0:
        return {"trade_dicts": [], "indicators": {}}

    data = _arrays_for(candles)
    atr = data.atr(st_period)
    up, dn, trend = data.supertrend(st_period, st_mult)
    ema = data.ema(ema_len)

    (nt, side, entry_bar, exit_bar, entry_px, exit_px, size, pnl, gross, comm, reason) = _simulate(
        data.open, data.high, data.low, data.close, atr, trend, ema,
                max_candle_atr, atr_sl, atr_tp, slippage, commission_pct, position_size,
        pyramiding, ohlc, same_bar,
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

    if not include_indicators:
        return {"trade_dicts": trade_dicts, "indicators": {}}

    ema_up = np.zeros(len(ema), dtype=bool)
    ema_down = np.zeros(len(ema), dtype=bool)
    ema_up[1:] = ema[1:] > ema[:-1]
    ema_down[1:] = ema[1:] < ema[:-1]
    indicators = {
        "tr": data.tr.tolist(),
        "atr": _opt(atr),
        "up": _opt(up),
        "dn": _opt(dn),
        "trend": trend.tolist(),
        "ema": ema.tolist(),
        "emaUp": ema_up.tolist(),
        "emaDown": ema_down.tolist(),
    }
    return {"trade_dicts": trade_dicts, "indicators": indicators}


def run_backtest_lean(candles: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    return run_backtest(candles, params, include_indicators=False)
