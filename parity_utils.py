# parity_utils.py
import pandas as pd
from typing import List, Dict, Any, Tuple
import math

def float_eq(a, b, eps=1e-6):
    try:
        return abs(float(a) - float(b)) <= eps
    except Exception:
        return str(a).strip() == str(b).strip()

def compare_indicators(engine_indicators: Dict[str, List[Any]],
                       tv_indicators: Dict[str, List[Any]],
                       keys: List[str] = ["atr", "up", "dn", "trend", "ema"]) -> Dict[str, Any]:
    """
    Compare per-bar indicators. Returns:
      { 'total_bars': N, 'matches': M, 'match_fraction': M/N, 'first_mismatch_index': idx or None }
    engine_indicators and tv_indicators are dicts mapping indicator name -> list
    """
    # find common length
    lengths = [len(engine_indicators.get(k, [])) for k in keys if k in engine_indicators]
    if not lengths:
        return {"total_bars": 0, "matches": 0, "match_fraction": 0.0, "first_mismatch_index": None}
    n = min(len(engine_indicators.get(keys[0], [])), len(tv_indicators.get(keys[0], [])))
    matches = 0
    first_mismatch = None
    for i in range(n):
        all_match = True
        for k in keys:
            e = engine_indicators.get(k, [])
            t = tv_indicators.get(k, [])
            if i >= len(e) or i >= len(t):
                continue
            ev = e[i]
            tv = t[i]
            if isinstance(ev, (float, int)) and isinstance(tv, (float, int)):
                if not float_eq(ev, tv, eps=1e-6):
                    all_match = False
                    break
            else:
                if ev != tv:
                    all_match = False
                    break
        if all_match:
            matches += 1
        elif first_mismatch is None:
            first_mismatch = i
    return {"total_bars": n, "matches": matches, "match_fraction": matches / n if n else 0.0, "first_mismatch_index": first_mismatch}

def build_tv_pairs_from_csv(tv_path: str) -> List[Tuple[Dict[str,Any], Dict[str,Any]]]:
    """
    Read TradingView CSV and return list of (entry_row, exit_row) pairs grouped by trade_number.
    This function is tolerant to common column names.
    """
    encs = ["utf-8-sig","utf-8","latin1","cp1252"]
    df = None
    for e in encs:
        try:
            df = pd.read_csv(tv_path, dtype=str, encoding=e)
            break
        except Exception:
            df = None
    if df is None:
        return []
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "trade_number" in df.columns:
        df["trade_number"] = pd.to_numeric(df["trade_number"], errors="coerce").astype("Int64")
    pairs = []
    if "trade_number" in df.columns:
        for tn, g in df.groupby("trade_number"):
            entry = None; exit_row = None
            for _, r in g.iterrows():
                typ = str(r.get("type","")).lower()
                sig = str(r.get("signal","")).lower()
                if "entry" in typ or "entry" in sig:
                    entry = r.to_dict()
                if "exit" in typ or "exit" in sig:
                    exit_row = r.to_dict()
            if entry is not None and exit_row is not None:
                pairs.append((entry, exit_row))
    else:
        # fallback adjacency
        for i in range(0, len(df)-1, 2):
            a = df.iloc[i].to_dict(); b = df.iloc[i+1].to_dict()
            pairs.append((a,b))
    return pairs

def compare_trades(engine_trades: List[Dict[str,Any]], tv_pairs: List[Tuple[Dict[str,Any],Dict[str,Any]]]) -> Dict[str,Any]:
    """
    Compare engine trades (list of dicts) to TV pairs.
    Returns summary: {tv_count, engine_count, matched_pairs, first_mismatch_index or None, trade_count_diff}
    Matching is done by index and by comparing side/entry_price/exit_price/pnl within tolerances.
    """
    tv_count = len(tv_pairs)
    eng_count = len(engine_trades)
    min_len = min(tv_count, eng_count)
    first_mismatch = None
    matched = 0
    for i in range(min_len):
        tv_entry, tv_exit = tv_pairs[i]
        eng = engine_trades[i]
        # canonical side
        tv_side = "long" if ("long" in str(tv_entry.get("type","")).lower() or "long" in str(tv_entry.get("signal","")).lower()) else ("short" if ("short" in str(tv_entry.get("type","")).lower() or "short" in str(tv_entry.get("signal","")).lower()) else "")
        eng_side = str(eng.get("side","")).lower()
        tv_entry_price = None
        tv_exit_price = None
        for k in ("price","price_usd","entry_price"):
            if k in tv_entry and pd.notna(tv_entry.get(k)):
                try:
                    tv_entry_price = float(tv_entry.get(k))
                    break
                except Exception:
                    pass
        for k in ("price","price_usd","exit_price"):
            if k in tv_exit and pd.notna(tv_exit.get(k)):
                try:
                    tv_exit_price = float(tv_exit.get(k))
                    break
                except Exception:
                    pass
        eng_entry_price = eng.get("entry_price")
        eng_exit_price = eng.get("exit_price")
        # compare
        ok = True
        if tv_side and eng_side and tv_side != eng_side:
            ok = False
        if tv_entry_price is not None and eng_entry_price is not None and not float_eq(tv_entry_price, eng_entry_price, eps=1e-3):
            ok = False
        if tv_exit_price is not None and eng_exit_price is not None and not float_eq(tv_exit_price, eng_exit_price, eps=1e-3):
            ok = False
        if ok:
            matched += 1
        elif first_mismatch is None:
            first_mismatch = i
    return {"tv_count": tv_count, "engine_count": eng_count, "matched_pairs": matched, "first_mismatch_index": first_mismatch, "trade_count_diff": eng_count - tv_count}
