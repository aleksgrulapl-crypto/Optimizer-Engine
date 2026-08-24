# data_loader.py
from typing import List, Dict, Any, Optional
import pandas as pd
import pytz
from datetime import datetime

def _to_float_safe(val: Any) -> Optional[float]:
    try:
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None

def _normalize_col_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

def _detect_epoch_unit(series: pd.Series) -> str:
    """
    Heuristic to detect whether epoch values are in seconds or milliseconds.
    Returns 's' or 'ms'.
    """
    if series.dropna().empty:
        return "s"
    # try numeric sample
    try:
        sample = int(series.dropna().iloc[0])
    except Exception:
        return "s"
    # If value is > 1e12 assume milliseconds, >1e9 assume seconds
    if sample > 1_000_000_000_000:
        return "ms"
    if sample > 1_000_000_000:
        return "s"
    return "s"

def load_candles_from_csv(path: str,
                          time_col: str = "time",
                          tz_target: Optional[str] = None,
                          sep: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load OHLC candles from a CSV/TSV file and return a list of candle dicts.

    - Detects TSV vs CSV automatically if sep is None.
    - Accepts epoch timestamps in seconds or milliseconds or ISO datetimes.
    - Converts timestamps to tz_target if provided (default: keep UTC).
    - Normalizes column names and tolerates missing optional columns.
    - Returns list of dicts with keys:
        time (timezone-aware datetime), open, high, low, close,
        mid (optional float), up_trend (optional float), down_trend (optional float)
    """
    if sep is None:
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()
        sep = "\t" if "\t" in header else ","

    df = pd.read_csv(path, sep=sep, dtype=str)
    df.columns = [_normalize_col_name(c) for c in df.columns]

    col_time = _normalize_col_name(time_col)
    if col_time not in df.columns:
        for alt in ("timestamp", "date", "datetime", "time_utc"):
            if alt in df.columns:
                col_time = alt
                break
        else:
            raise ValueError(f"Time column '{time_col}' not found. Available: {list(df.columns)}")

    # detect numeric epoch vs ISO strings
    series = df[col_time].dropna()
    is_numeric = True
    try:
        _ = series.astype(float)
    except Exception:
        is_numeric = False

    if is_numeric:
        epoch_unit = _detect_epoch_unit(df[col_time])
        if epoch_unit == "ms":
            df["_parsed_time"] = pd.to_datetime(df[col_time].astype(float), unit="ms", utc=True)
        else:
            df["_parsed_time"] = pd.to_datetime(df[col_time].astype(float), unit="s", utc=True)
    else:
        df["_parsed_time"] = pd.to_datetime(df[col_time], utc=True, errors="coerce")
        if df["_parsed_time"].isna().all():
            raise ValueError("Could not parse time column as epoch or ISO datetimes.")

    if tz_target:
        try:
            tz = pytz.timezone(tz_target)
            df["_parsed_time"] = df["_parsed_time"].dt.tz_convert(tz)
        except Exception:
            pass

    # sort and deduplicate by time
    df = df.sort_values("_parsed_time").drop_duplicates(subset=["_parsed_time"], keep="first").reset_index(drop=True)

    # column mapping
    mapping = {
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c"],
        "mid": ["mid", "mid_price", "midprice"],
        "up_trend": ["up_trend", "up", "uptrend"],
        "down_trend": ["down_trend", "dn", "downtrend"]
    }

    def find_col(possible):
        for name in possible:
            if name in df.columns:
                return name
        return None

    col_open = find_col(mapping["open"])
    col_high = find_col(mapping["high"])
    col_low = find_col(mapping["low"])
    col_close = find_col(mapping["close"])
    col_mid = find_col(mapping["mid"])
    col_up = find_col(mapping["up_trend"])
    col_dn = find_col(mapping["down_trend"])

    for req, name in (("open", col_open), ("high", col_high), ("low", col_low), ("close", col_close)):
        if name is None:
            raise ValueError(f"Required OHLC column '{req}' not found in CSV. Available columns: {list(df.columns)}")

    candles: List[Dict[str, Any]] = []
    skipped = 0
    for _, row in df.iterrows():
        time_val = row["_parsed_time"].to_pydatetime()

        open_v = _to_float_safe(row[col_open])
        high_v = _to_float_safe(row[col_high])
        low_v = _to_float_safe(row[col_low])
        close_v = _to_float_safe(row[col_close])
        mid_v = _to_float_safe(row[col_mid]) if col_mid is not None else None
        up_v = _to_float_safe(row[col_up]) if col_up is not None else None
        dn_v = _to_float_safe(row[col_dn]) if col_dn is not None else None

        if open_v is None or high_v is None or low_v is None or close_v is None:
            skipped += 1
            continue

        candles.append({
            "time": time_val,
            "open": open_v,
            "high": high_v,
            "low": low_v,
            "close": close_v,
            "mid": mid_v,
            "up_trend": up_v,
            "down_trend": dn_v,
        })

    if skipped:
        print(f"[data_loader] skipped {skipped} malformed rows from {path}")

    return candles
