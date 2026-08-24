# strategy_indicators.py
"""
Drop-in replacement for strategy_indicator.py / strategy_indicators.py

Provides:
- true_range
- compute_atr_sma_tr
- compute_ema
- compute_supertrend_v4

Behavior guarantees:
- ATR is computed as SMA(TR, length) with None for warmup bars (length-1 first bars).
- EMA seeded with first close (common Pine behavior).
- Supertrend v4 implements the Pine-style logic with nz semantics and exact recursion:
    up_base = close - stMultiplier * atr
    dn_base = close + stMultiplier * atr
    up := close[1] > up1 ? max(up_base, up1) : up_base
    dn := close[1] < dn1 ? min(dn_base, dn1) : dn_base
  trend recursion uses previous trend and comparisons to dn1/up1.
- All functions accept a list of candle dicts with numeric 'open','high','low','close' keys.
"""

from typing import List, Dict, Any, Optional, Tuple


def true_range(candles: List[Dict[str, Any]], i: int) -> float:
    """
    True range matching Pine ta.tr(true):
      TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    For i==0 prev_close is taken as current close (matches many implementations).
    """
    c = candles[i]
    high = float(c["high"])
    low = float(c["low"])
    if i == 0:
        prev_close = float(c["close"])
    else:
        prev_close = float(candles[i - 1]["close"])
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def compute_atr_sma_tr(candles: List[Dict[str, Any]], length: int) -> List[Optional[float]]:
    """
    ATR computed as SMA of TR (Pine: ta.sma(ta.tr(true), length)).
    Returns a list with None for warmup bars (first length-1 entries).
    """
    n = len(candles)
    if n == 0:
        return []
    trs: List[float] = [0.0] * n
    for i in range(n):
        trs[i] = true_range(candles, i)

    atr: List[Optional[float]] = [None] * n
    if length <= 0:
        return atr

    window_sum = 0.0
    for i in range(n):
        window_sum += trs[i]
        if i >= length:
            window_sum -= trs[i - length]
        if i >= length - 1:
            atr[i] = window_sum / length
    return atr


def compute_ema(candles: List[Dict[str, Any]], length: int) -> List[Optional[float]]:
    """
    EMA seeded with the first close value (matches common Pine behavior).
    Returns a list of floats (None only if candles is empty).
    """
    n = len(candles)
    if n == 0:
        return []
    ema: List[Optional[float]] = [None] * n
    # handle degenerate length
    alpha = 2.0 / (length + 1.0) if length > 0 else 1.0
    for i, c in enumerate(candles):
        close = float(c["close"])
        if i == 0:
            ema[i] = close
        else:
            prev = ema[i - 1] if ema[i - 1] is not None else close
            ema[i] = alpha * close + (1 - alpha) * prev
    return ema


def compute_supertrend_v4(
    candles: List[Dict[str, Any]],
    stMultiplier: float,
    atr: List[Optional[float]],
) -> Dict[str, List]:
    """
    Supertrend v4 implementation following Pine logic.

    Inputs:
      - candles: list of OHLC dicts with 'close' key
      - stMultiplier: multiplier used in bands
      - atr: list produced by compute_atr_sma_tr (may contain None for warmup)

    Returns dict with:
      - up: list[Optional[float]]
      - dn: list[Optional[float]]
      - trend15: list[int] (1 or -1)
      - buy15: list[bool]
      - sell15: list[bool]

    Notes:
      - Uses nz semantics: when previous up/dn is None, fallback to current base value.
      - Matches Pine recursion:
          up_base = close - stMultiplier * atr
          dn_base = close + stMultiplier * atr
          up := close[1] > up1 ? max(up_base, up1) : up_base
          dn := close[1] < dn1 ? min(dn_base, dn1) : dn_base
          trend recursion uses previous trend and comparisons to dn1/up1
    """
    n = len(candles)
    up: List[Optional[float]] = [None] * n
    dn: List[Optional[float]] = [None] * n
    trend15: List[int] = [1] * n

    # Defensive: if atr length mismatches, treat missing as None
    for i in range(n):
        close = float(candles[i]["close"])
        atr_i = atr[i] if i < len(atr) else None

        if atr_i is None:
            # warmup: keep bands None and carry previous trend (or default 1)
            up[i] = None
            dn[i] = None
            trend15[i] = trend15[i - 1] if i > 0 else 1
            continue

        base_up = close - (stMultiplier * atr_i)
        base_dn = close + (stMultiplier * atr_i)

        # previous band values with nz semantics
        up1 = up[i - 1] if (i > 0 and up[i - 1] is not None) else base_up
        dn1 = dn[i - 1] if (i > 0 and dn[i - 1] is not None) else base_dn

        prev_close = float(candles[i - 1]["close"]) if i > 0 else close

        # Pine logic replication:
        # up := close[1] > up1 ? max(base_up, up1) : base_up
        if prev_close > up1:
            up_val = max(base_up, up1)
        else:
            up_val = base_up

        # dn := close[1] < dn1 ? min(base_dn, dn1) : base_dn
        if prev_close < dn1:
            dn_val = min(base_dn, dn1)
        else:
            dn_val = base_dn

        up[i] = up_val
        dn[i] = dn_val

        # trend recursion
        if i == 0:
            trend15[i] = 1
        else:
            t_prev = trend15[i - 1]
            t = t_prev
            # Use dn1/up1 (previous band values) for comparisons as in Pine
            if t_prev == -1 and close > dn1:
                t = 1
            elif t_prev == 1 and close < up1:
                t = -1
            trend15[i] = t

    buy15: List[bool] = [False] * n
    sell15: List[bool] = [False] * n
    for i in range(1, n):
        buy15[i] = (trend15[i] == 1 and trend15[i - 1] == -1)
        sell15[i] = (trend15[i] == -1 and trend15[i - 1] == 1)

    return {
        "up": up,
        "dn": dn,
        "trend15": trend15,
        "buy15": buy15,
        "sell15": sell15,
    }


# Optional convenience wrapper used by some code paths:
def compute_supertrend_from_params(
    candles: List[Dict[str, Any]],
    stPeriod: int,
    stMultiplier: float,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[int]]:
    """
    Convenience wrapper that computes ATR (SMA of TR) and then supertrend v4,
    returning (up, dn, trend15) to match other modules' expectations.
    """
    atr = compute_atr_sma_tr(candles, stPeriod)
    res = compute_supertrend_v4(candles, stMultiplier, atr)
    return res["up"], res["dn"], res["trend15"]
