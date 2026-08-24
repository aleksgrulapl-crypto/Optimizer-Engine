"""
Compare engine trades CSV to TradingView export CSV.
Usage:
  python compare_parity.py engine_trades.csv tradingview.csv
"""

import sys
import csv
from typing import List, Dict, Any, Tuple


def _to_float(v: Any) -> float:
    if v is None:
        raise ValueError("None")
    s = str(v).replace(",", "").strip()
    return float(s)


def _canonical_side(s: Any) -> str:
    text = str(s or "").strip().lower()
    if "long" in text or text == "buy":
        return "long"
    if "short" in text or text == "sell":
        return "short"
    return text


def _float_eq(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


def load_engine_csv(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                out.append({
                    "side": _canonical_side(r.get("side")),
                    "entry_price": _to_float(r.get("entry_price")),
                    "exit_price": _to_float(r.get("exit_price")),
                    "entry_bar_index": int(float(r.get("entry_bar_index", -1))),
                    "exit_bar_index": int(float(r.get("exit_bar_index", -1))),
                })
            except Exception:
                continue
    return out


def _build_tv_pairs(rows: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    by_trade: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        k = (r.get("Trade number") or r.get("trade_number") or "").strip()
        if not k:
            continue
        by_trade.setdefault(k, []).append(r)

    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for _, grp in by_trade.items():
        entry = None
        exit_row = None
        for r in grp:
            t = str(r.get("Type") or r.get("type") or "").lower()
            if "entry" in t:
                entry = r
            elif "exit" in t:
                exit_row = r
        if entry is not None and exit_row is not None:
            pairs.append((entry, exit_row))
    return pairs


def load_tv_trades(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    trades: List[Dict[str, Any]] = []
    for entry, exit_row in _build_tv_pairs(rows):
        try:
            side = _canonical_side(entry.get("Signal") or entry.get("signal") or entry.get("Type") or entry.get("type"))
            trades.append({
                "side": side,
                "entry_price": _to_float(entry.get("Price USD") or entry.get("price_usd") or entry.get("entry_price") or entry.get("price")),
                "exit_price": _to_float(exit_row.get("Price USD") or exit_row.get("price_usd") or exit_row.get("exit_price") or exit_row.get("price")),
                "entry_bar_index": -1,
                "exit_bar_index": -1,
            })
        except Exception:
            continue
    return trades


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python compare_parity.py engine_trades.csv tradingview.csv")
        return

    engine_path = sys.argv[1]
    tv_path = sys.argv[2]

    engine = load_engine_csv(engine_path)
    tv = load_tv_trades(tv_path)

    if not engine:
        print("Engine trade list is empty or unreadable.")
        return
    if not tv:
        print("TradingView trade list is empty or unreadable.")
        return

    n = min(len(engine), len(tv))
    for i in range(n):
        e = engine[i]
        t = tv[i]
        mismatch = (
            e["side"] != t["side"]
            or not _float_eq(e["entry_price"], t["entry_price"], eps=1e-4)
            or not _float_eq(e["exit_price"], t["exit_price"], eps=1e-4)
        )
        if mismatch:
            print("First mismatch at index", i)
            print("Engine:", e)
            print("TV    :", t)
            return

    if len(engine) != len(tv):
        print(f"No mismatch in first {n} trades, but counts differ: engine={len(engine)} tv={len(tv)}")
    else:
        print(f"Parity match for all {n} trades")


if __name__ == "__main__":
    main()
