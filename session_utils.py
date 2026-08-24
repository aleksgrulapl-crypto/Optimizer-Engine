# session_utils.py
from datetime import datetime, time
from typing import Union
import pytz

ET_TZ = pytz.timezone("America/New_York")

# TradingView extended-hours windows in ET
_PRE_MARKET_START = time(4, 0)    # 04:00 ET
_PRE_MARKET_END   = time(9, 30)   # 09:30 ET (exclusive of regular open)
_REGULAR_START    = time(9, 30)   # 09:30 ET (inclusive)
_REGULAR_END      = time(16, 0)   # 16:00 ET
_AFTER_HOURS_START= time(16, 0)   # 16:00 ET (inclusive)
_AFTER_HOURS_END  = time(20, 0)   # 20:00 ET

def _ensure_aware_utc(dt: datetime) -> datetime:
    """
    Ensure datetime is timezone-aware in UTC.
    Accepts naive datetimes (interpreted as UTC) or timezone-aware datetimes.
    """
    if dt.tzinfo is None:
        return pytz.utc.localize(dt)
    return dt.astimezone(pytz.utc)

def utc_to_et(dt_utc: Union[datetime, int, float]) -> datetime:
    """
    Convert a UTC datetime (or epoch seconds) to America/New_York timezone-aware datetime.
    - Accepts: timezone-aware UTC datetime, naive datetime (treated as UTC), or epoch seconds (int/float).
    - Returns: timezone-aware datetime in ET (with DST handled).
    """
    if isinstance(dt_utc, (int, float)):
        # treat as epoch seconds
        dt_utc = datetime.utcfromtimestamp(int(dt_utc))
    dt_utc = _ensure_aware_utc(dt_utc)
    return dt_utc.astimezone(ET_TZ)

def is_nvda_session(dt_utc: Union[datetime, int, float]) -> bool:
    """
    Return True if the provided UTC datetime (or epoch seconds) falls within TradingView's
    NVDA session windows including extended hours:
      - Pre-market: 04:00–09:30 ET
      - Regular:    09:30–16:00 ET
      - After-hours:16:00–20:00 ET
    Returns False for 20:00–04:00 ET (no trading).
    """
    dt_et = utc_to_et(dt_utc)
    t = dt_et.time()

    # Pre-market: 04:00 <= t < 09:30
    if (_PRE_MARKET_START <= t) and (t < _PRE_MARKET_END):
        return True

    # Regular: 09:30 <= t < 16:00
    if (_REGULAR_START <= t) and (t < _REGULAR_END):
        return True

    # After-hours: 16:00 <= t < 20:00
    if (_AFTER_HOURS_START <= t) and (t < _AFTER_HOURS_END):
        return True

    return False

def is_regular_session(dt_utc: Union[datetime, int, float]) -> bool:
    """
    Return True only for regular session (09:30–16:00 ET).
    """
    dt_et = utc_to_et(dt_utc)
    t = dt_et.time()
    return (_REGULAR_START <= t) and (t < _REGULAR_END)

def is_extended_hours(dt_utc: Union[datetime, int, float]) -> bool:
    """
    Return True for extended hours (pre-market or after-hours), i.e. 04:00–09:30 ET or 16:00–20:00 ET.
    """
    dt_et = utc_to_et(dt_utc)
    t = dt_et.time()
    return ((_PRE_MARKET_START <= t) and (t < _PRE_MARKET_END)) or ((_AFTER_HOURS_START <= t) and (t < _AFTER_HOURS_END))
