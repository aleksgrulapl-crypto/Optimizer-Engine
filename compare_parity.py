# compare_parity.py
"""
Simple parity comparator: compare engine trades to TradingView CSV export.
Usage:
  python compare_parity.py engine_trades.json tradingview.csv
It prints the first mismatched trade and a short summary.
"""

import sys
import csv
import json
from typing import List, Dict, Any

def load_engine_trades(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_tv_csv(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def normalize_trade(t: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "side": t.get("side"),
        "entry_price": float(t.get("entry_price", 0.0)),
        "exit_price": float(t.get("exit_price", 0.0)),
        "entry_bar_index": int(t.get("entry_bar_index", -1)),
        "exit_bar_index": int(t.get("exit_bar_index", -1)),
    }

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_parity.py engine_trades.json tradingview.csv")
        return
    engine_path = sys.argv[1]
    tv_path = sys.argv[2]
    engine = load_engine_trades(engine_path)
    tv = load_tv_csv(tv_path)
    eng_norm = [normalize_trade(t) for t in engine]
    tv_norm = []
    for r in tv:
        try:
            tv_norm.append({
                "side": r.get("side") or r.get("Side") or r.get("direction"),
                "entry_price": float(r.get("entry_price") or r.get("Entry Price") or r.get("entry")),
                "exit_price": float(r.get("exit_price") or r.get("Exit Price") or r.get("exit")),
                "entry_bar_index": int(r.get("entry_bar_index") or r.get("entry_index") or -1),
                "exit_bar_index": int(r.get("exit_bar_index") or r.get("exit_index") or -1),
            })
        except Exception:
            continue

    # naive pairwise compare
    n = min(len(eng_norm), len(tv_norm))
    for i in range(n):
        e = eng_norm[i]
        t = tv_norm[i]
        if (e["side"] != t["side"] or abs(e["entry_price"] - t["entry_price"]) > 1e-6 or abs(e["exit_price"] - t["exit_price"]) > 1e-6):
            print("First mismatch at index", i)
            print("Engine:", e)
            print("TV    :", t)
            return
    print("No mismatch in first", n, "trades. Engine trades:", len(eng_norm), "TV trades:", len(tv_norm))

if __name__ == "__main__":
    main()
