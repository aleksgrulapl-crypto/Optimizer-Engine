# backtest_engine.py
"""
Drop-in replacement for backtest_engine.py

This module provides a backtest engine compatible with the project's existing
API: run_backtest(candles, params) -> {"trade_dicts": [...], "indicators": {...}}

Key improvements and guarantees:
- Deterministic intrabar simulation supporting "ohlc" and "olhc" paths.
- ATR computed as SMA(TR, stPeriod) to match Pine script behavior.
- Supertrend implementation with nz semantics and exact recursion logic.
- Clear warm-up handling: signals are only generated once required indicators are available.
- Reversal handling, pyramiding, slippage and commission applied consistently.
- Returns indicators dictionary for parity checks.
- Keeps the same trade dict keys as the rest of the project.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from presets import get_presets
import math
import copy

@dataclass
class Position:
    side: str                     # "long" or "short"
    entry_time: Any
    entry_price: float
    size: float                   # number of contracts/shares
    avg_price: float              # average entry price (for partial fills)
    entry_bar_index: int

# -------------------------
# Indicator helpers
# -------------------------
def compute_tr(candles: List[Dict[str, Any]]) -> List[float]:
    """True range per bar matching Pine ta.tr(true)."""
    n = len(candles)
    tr = [0.0] * n
    for i in range(n):
        high = float(candles[i]["high"])
        low = float(candles[i]["low"])
        prev_close = float(candles[i-1]["close"]) if i > 0 else float(candles[i]["close"])
        tr[i] = max(high - low, abs(high - prev_close), abs(low - prev_close))
    return tr

def compute_sma(series: List[float], length: int) -> List[Optional[float]]:
    """Simple moving average with None for warmup bars (length-1 first bars)."""
    n = len(series)
    out: List[Optional[float]] = [None] * n
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

def compute_ema_from_close(candles: List[Dict[str, Any]], length: int) -> List[Optional[float]]:
    """EMA seeded with first close (matches many Pine behaviors). Returns None for empty input."""
    n = len(candles)
    if n == 0:
        return []
    ema: List[Optional[float]] = [None] * n
    alpha = 2.0 / (length + 1.0)
    for i in range(n):
        close = float(candles[i]["close"])
        if i == 0:
            ema[i] = close
        else:
            prev = ema[i-1] if ema[i-1] is not None else close
            ema[i] = alpha * close + (1 - alpha) * prev
    return ema

# -------------------------
# Supertrend implementation
# -------------------------
def compute_supertrend(candles: List[Dict[str, Any]], st_period: int, st_multiplier: float
                      ) -> Tuple[List[Optional[float]], List[Optional[float]], List[int]]:
    """
    Compute Supertrend bands and trend recursion following the Pine logic:
      atr = SMA(TR, st_period)
      up_base = close - st_multiplier * atr
      dn_base = close + st_multiplier * atr
      up := close[1] > up1 ? max(up_base, up1) : up_base
      dn := close[1] < dn1 ? min(dn_base, dn1) : dn_base
      trend recursion with nz semantics
    Returns (up, dn, trend) where trend is 1 or -1 per bar.
    """
    n = len(candles)
    tr = compute_tr(candles)
    atr = compute_sma(tr, st_period)

    up: List[Optional[float]] = [None] * n
    dn: List[Optional[float]] = [None] * n
    trend: List[int] = [1] * n

    for i in range(n):
        if atr[i] is None:
            up[i] = None
            dn[i] = None
            trend[i] = trend[i-1] if i > 0 else 1
            continue

        close = float(candles[i]["close"])
        base_up = close - st_multiplier * atr[i]
        base_dn = close + st_multiplier * atr[i]

        # nz semantics: up1 = up[i-1] if not None else base_up
        up1 = up[i-1] if (i > 0 and up[i-1] is not None) else base_up
        dn1 = dn[i-1] if (i > 0 and dn[i-1] is not None) else base_dn

        prev_close = float(candles[i-1]["close"]) if i > 0 else close

        # replicate Pine's assignment with previous close comparisons
        up_val = max(base_up, up1) if prev_close > up1 else base_up
        dn_val = min(base_dn, dn1) if prev_close < dn1 else base_dn

        up[i] = up_val
        dn[i] = dn_val

        # trend recursion
        if i == 0:
            trend[i] = 1
        else:
            # Use dn1/up1 (previous band values) for comparisons as in Pine
            if trend[i-1] == -1 and close > dn1:
                trend[i] = 1
            elif trend[i-1] == 1 and close < up1:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]

    return up, dn, trend

# -------------------------
# Intrabar simulation
# -------------------------
def _intrabar_sequence(open_p: float, high: float, low: float, close: float, path: str) -> List[Tuple[str, float]]:
    """
    Return a sequence of (stage, value) tuples representing the intrabar path.
    stage names: "open", "high", "low", "close" but order depends on path.
    """
    if path == "ohlc":
        return [("open", open_p), ("high", high), ("low", low), ("close", close)]
    else:
        return [("open", open_p), ("low", low), ("high", high), ("close", close)]

def simulate_intrabar_exit_long(entry_price: float, stop: float, limit: float,
                                open_p: float, high: float, low: float, close: float,
                                path: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Simulate intrabar exit for a long position.
    path: "ohlc" or "olhc" (open->high->low->close or open->low->high->close)
    Returns (exit_reason, exit_price) or (None, None) if not hit.
    Exit price uses the stop/limit price (not bar extreme) to match Pine's stop/limit fills.
    """
    seq = _intrabar_sequence(open_p, high, low, close, path)

    # For each stage, determine whether stop or limit was reached at that stage.
    # For long: stop < entry_price, limit > entry_price
    for stage, val in seq:
        if stage == "open":
            # open can immediately hit stop or limit
            if open_p <= stop:
                return "sl", stop
            if open_p >= limit:
                return "tp", limit
        elif stage == "high":
            # high stage: check limit first (price went up)
            if high >= limit:
                return "tp", limit
            # it's possible high is also <= stop in degenerate bars; check stop after
            if high <= stop:
                return "sl", stop
        elif stage == "low":
            # low stage: check stop first (price went down)
            if low <= stop:
                return "sl", stop
            if low >= limit:
                return "tp", limit
        else:  # close
            if close <= stop:
                return "sl", stop
            if close >= limit:
                return "tp", limit
    return None, None

def simulate_intrabar_exit_short(entry_price: float, stop: float, limit: float,
                                 open_p: float, high: float, low: float, close: float,
                                 path: str) -> Tuple[Optional[str], Optional[float]]:
    """Simulate intrabar exit for a short position. stop above entry, limit below entry."""
    seq = _intrabar_sequence(open_p, high, low, close, path)

    for stage, val in seq:
        if stage == "open":
            if open_p >= stop:
                return "sl", stop
            if open_p <= limit:
                return "tp", limit
        elif stage == "high":
            # high stage: check stop first (price went up)
            if high >= stop:
                return "sl", stop
            if high <= limit:
                return "tp", limit
        elif stage == "low":
            # low stage: check limit first (price went down)
            if low <= limit:
                return "tp", limit
            if low >= stop:
                return "sl", stop
        else:  # close
            if close >= stop:
                return "sl", stop
            if close <= limit:
                return "tp", limit
    return None, None

# -------------------------
# Backtest core
# -------------------------
def _apply_slippage_and_commission_on_exit(side: str, exit_price: float, slippage: float, commission_pct: float, size: float, exit_reason: str) -> Tuple[float, float]:
    """
    Apply slippage and compute commission for an exit fill.
    Returns (fill_price_after_slippage, commission_amount).
    Slippage is applied in the direction that is worse for the trader.
    """
    if side == "long":
        # For long exit: take worse price by slippage
        fill_price = exit_price + slippage if exit_reason == "sl" else exit_price - slippage
    else:
        # For short exit: take worse price by slippage
        fill_price = exit_price - slippage if exit_reason == "sl" else exit_price + slippage

    trade_value = abs(fill_price * size)
    commission = trade_value * commission_pct
    return fill_price, commission

def _apply_slippage_on_entry(side: str, entry_price: float, slippage: float) -> float:
    """
    Apply slippage for entry fills. Slippage is applied in the direction worse for the trader.
    """
    if side == "long":
        return entry_price + slippage
    else:
        return entry_price - slippage

def run_backtest(candles: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main backtest entry point.
    Expected candle dict keys: time, open, high, low, close, volume (volume optional).
    Params supports:
      ticker, stMultiplier, stPeriod, atrSLmult, atrTPmult, emaLen,
      slippage (price units), commission_pct (fraction), position_size (contracts),
      leverage (margin), pyramiding (int), intrabar_path ("ohlc" or "olhc"),
      allow_same_bar_exit_entry (bool) - whether entries can be exited on same bar intrabar.
    Returns dict with trade list and per-bar indicators for parity checks.
    """
    # Load presets if not provided explicitly
    ticker = params.get("ticker", "NVDA")
    p = get_presets(ticker)

    stMultiplier = float(params.get("stMultiplier", p.get("stMultiplier", 3.0)))
    stPeriod     = int(params.get("stPeriod", p.get("stPeriod", 12)))
    atrSLmult    = float(params.get("atrSLmult", p.get("atrSLmult", 1.4)))
    atrTPmult    = float(params.get("atrTPmult", p.get("atrTPmult", 2.4)))
    emaLen       = int(params.get("emaLen", p.get("emaLen", 200)))

    slippage = float(params.get("slippage", 0.0))            # price units per fill
    commission_pct = float(params.get("commission_pct", 0.0)) # fraction of trade value per side
    position_size = float(params.get("position_size", params.get("contracts", 1.0)))  # fixed contracts/shares
    leverage = float(params.get("leverage", 1.0))            # margin factor; not used for sizing here but available
    pyramiding = int(params.get("pyramiding", 1))          # max concurrent entries same direction
    intrabar_path = params.get("intrabar_path", "ohlc")  # "ohlc" or "olhc"
    allow_same_bar_exit_entry = bool(params.get("allow_same_bar_exit_entry", False))

    n = len(candles)
    if n == 0:
        return {"trade_dicts": [], "indicators": {}}

    # Ensure numeric fields are floats
    for c in candles:
        for k in ("open", "high", "low", "close"):
            c[k] = float(c[k])

    # Indicators
    tr = compute_tr(candles)
    atr = compute_sma(tr, stPeriod)
    up, dn, trend = compute_supertrend(candles, stPeriod, stMultiplier)
    emaTrend = compute_ema_from_close(candles, emaLen)

    emaUp = [False] * n
    emaDown = [False] * n
    for i in range(1, n):
        if emaTrend[i] is None or emaTrend[i-1] is None:
            emaUp[i] = False
            emaDown[i] = False
        else:
            emaUp[i] = emaTrend[i] > emaTrend[i-1]
            emaDown[i] = emaTrend[i] < emaTrend[i-1]

    trade_dicts: List[Dict[str, Any]] = []
    pos_list: List[Position] = []  # allow multiple positions for pyramiding
    long_count = 0
    short_count = 0

    # Helper to close a position at a given fill price and reason
    def _close_position(pos: Position, fill_price: float, reason: str, exit_bar_index: int, exit_time: Any):
        nonlocal trade_dicts, long_count, short_count
        trade_value = fill_price * pos.size
        commission = abs(trade_value) * commission_pct
        if pos.side == "long":
            pnl = (fill_price - pos.avg_price) * pos.size
        else:
            pnl = (pos.avg_price - fill_price) * pos.size
        pnl_net = pnl - commission
        trade_dicts.append({
            "side": pos.side,
            "entry_time": pos.entry_time,
            "entry_price": pos.entry_price,
            "exit_time": exit_time,
            "exit_price": fill_price,
            "size": pos.size,
            "pnl": pnl_net,
            "pnl_gross": pnl,
            "commission": commission,
            "exit_reason": reason,
            "entry_bar_index": pos.entry_bar_index,
            "exit_bar_index": exit_bar_index,
        })
        if pos.side == "long":
            long_count -= 1
        else:
            short_count -= 1

    # Main loop
    for i, c in enumerate(candles):
        time = c.get("time")
        open_p = c["open"]
        high = c["high"]
        low = c["low"]
        close = c["close"]

        # Determine buy/sell from Supertrend trend flips (only when trend values exist)
        buy = False
        sell = False
        if i > 0 and trend[i] is not None and trend[i-1] is not None:
            buy = (trend[i] == 1 and trend[i-1] == -1)
            sell = (trend[i] == -1 and trend[i-1] == 1)

        # Gate signals by indicator warm-up: require ATR and EMA available
        atr_i = atr[i] if i < len(atr) else None
        ema_ok = (emaTrend[i] is not None and emaTrend[i-1] is not None) if i > 0 else False

        longSignal = buy and ema_ok and emaUp[i]
        shortSignal = sell and ema_ok and emaDown[i]

        # Compute ATR-based SL/TP distances for this bar (use atr[i] if available)
        atrSL = (atr_i * atrSLmult) if atr_i is not None else None
        atrTP = (atr_i * atrTPmult) if atr_i is not None else None

        # --- Step A: simulate intrabar exits for existing positions using chosen path ---
        # We iterate a copy because we may remove positions
        remaining_positions: List[Position] = []
        for pos in list(pos_list):
            exit_reason = None
            exit_price = None

            # If ATR not available, we cannot compute SL/TP; keep position open
            if atr_i is None:
                remaining_positions.append(pos)
                continue

            if pos.side == "long":
                stop = pos.avg_price - atrSL
                limit = pos.avg_price + atrTP
                exit_reason, exit_price = simulate_intrabar_exit_long(
                    pos.avg_price, stop, limit, open_p, high, low, close, intrabar_path
                )
            else:
                stop = pos.avg_price + atrSL
                limit = pos.avg_price - atrTP
                exit_reason, exit_price = simulate_intrabar_exit_short(
                    pos.avg_price, stop, limit, open_p, high, low, close, intrabar_path
                )

            if exit_reason is not None:
                # apply slippage and commission
                fill_price, commission = _apply_slippage_and_commission_on_exit(pos.side, exit_price, slippage, commission_pct, pos.size, exit_reason)
                # record exit
                _close_position(pos, fill_price, exit_reason, i, time)
                # remove pos from pos_list
                try:
                    pos_list.remove(pos)
                except ValueError:
                    pass
            else:
                remaining_positions.append(pos)
        pos_list = remaining_positions

        # --- Step B: evaluate entry signals at bar close (matching Pine backtest behavior) ---
        # Only generate entries if indicators are ready
        if (longSignal or shortSignal):
            # If reversal is required, close opposite positions at close (apply slippage/commission)
            if longSignal:
                # close all shorts at close price (reverse)
                if short_count > 0:
                    shorts_to_close = [p for p in list(pos_list) if p.side == "short"]
                    for p in shorts_to_close:
                        fill_price = close + slippage  # short exit worse by slippage
                        _close_position(p, fill_price, "reverse", i, time)
                        try:
                            pos_list.remove(p)
                        except ValueError:
                            pass

                # open new long(s) if pyramiding allows
                while long_count < pyramiding:
                    entry_price = close
                    entry_price_with_slip = _apply_slippage_on_entry("long", entry_price, slippage)
                    pos = Position("long", time, entry_price_with_slip, position_size, entry_price_with_slip, i)
                    pos_list.append(pos)
                    long_count += 1
                    # If only one entry per signal desired, break; keep loop to allow pyramiding >1
                    break

            elif shortSignal:
                # close all longs at close price (reverse)
                if long_count > 0:
                    longs_to_close = [p for p in list(pos_list) if p.side == "long"]
                    for p in longs_to_close:
                        fill_price = close - slippage  # long exit worse by slippage
                        _close_position(p, fill_price, "reverse", i, time)
                        try:
                            pos_list.remove(p)
                        except ValueError:
                            pass

                # open new short(s) if pyramiding allows
                while short_count < pyramiding:
                    entry_price = close
                    entry_price_with_slip = _apply_slippage_on_entry("short", entry_price, slippage)
                    pos = Position("short", time, entry_price_with_slip, position_size, entry_price_with_slip, i)
                    pos_list.append(pos)
                    short_count += 1
                    break

        # Note: we intentionally do not attempt same-bar entry->exit intrabar checks here by default.
        # If allow_same_bar_exit_entry is True, we should simulate intrabar exits for positions opened this bar.
        if allow_same_bar_exit_entry and (len(pos_list) > 0):
            # simulate intrabar exits for positions that were opened at this bar (entry_bar_index == i)
            # We will iterate a copy to avoid modifying while iterating.
            for pos in list(pos_list):
                if pos.entry_bar_index != i:
                    continue
                # compute SL/TP using current atr_i (which is based on this bar)
                if atr_i is None:
                    continue
                if pos.side == "long":
                    stop = pos.avg_price - atrSL
                    limit = pos.avg_price + atrTP
                    exit_reason, exit_price = simulate_intrabar_exit_long(
                        pos.avg_price, stop, limit, open_p, high, low, close, intrabar_path
                    )
                else:
                    stop = pos.avg_price + atrSL
                    limit = pos.avg_price - atrTP
                    exit_reason, exit_price = simulate_intrabar_exit_short(
                        pos.avg_price, stop, limit, open_p, high, low, close, intrabar_path
                    )
                if exit_reason is not None:
                    fill_price, commission = _apply_slippage_and_commission_on_exit(pos.side, exit_price, slippage, commission_pct, pos.size, exit_reason)
                    _close_position(pos, fill_price, exit_reason, i, time)
                    try:
                        pos_list.remove(pos)
                    except ValueError:
                        pass

    # At the end of series, close remaining positions at final close (mark-to-market)
    final_close = candles[-1]["close"]
    final_time = candles[-1].get("time")
    for p in list(pos_list):
        fill_price = final_close - slippage if p.side == "long" else final_close + slippage
        _close_position(p, fill_price, "eod", n-1, final_time)
        try:
            pos_list.remove(p)
        except ValueError:
            pass

    indicators = {
        "tr": tr,
        "atr": atr,
        "up": up,
        "dn": dn,
        "trend": trend,
        "ema": emaTrend,
        "emaUp": emaUp,
        "emaDown": emaDown,
    }

    # Defensive: ensure numeric types are serializable (convert numpy types if any)
    def _sanitize_trade(t: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in t.items():
            if isinstance(v, (float, int, str, type(None))):
                out[k] = v
            else:
                try:
                    out[k] = float(v)
                except Exception:
                    out[k] = v
        return out

    trade_dicts = [_sanitize_trade(t) for t in trade_dicts]

    return {"trade_dicts": trade_dicts, "indicators": indicators}
