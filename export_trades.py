# run_backtest_and_export.py
"""
Robust runner to load candles, run the backtest engine, and export trades to CSV.
- Uses data_loader.load_candles_from_csv (expects data_loader.py from previous step).
- Uses backtest_engine.run_backtest (expects backtest_engine.py from previous step).
- Writes a CSV with a stable set of columns and ISO timestamps.
- Safe parsing, logging, and configurable paths/params.
"""

import csv
import sys
import argparse
from typing import List, Dict, Any
from pathlib import Path

from backtest_engine import run_backtest
from data_loader import load_candles_from_csv


DEFAULT_FIELDNAMES = [
    "entry_time",
    "exit_time",
    "side",
    "entry_price",
    "exit_price",
    "size",
    "pnl",
    "pnl_gross",
    "commission",
    "exit_reason",
    "entry_bar_index",
    "exit_bar_index",
]


def _normalize_trade_row(trade: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure the trade dict contains the expected keys and format timestamps as ISO strings.
    Missing keys are filled with empty strings or zeros where appropriate.
    """
    out = {}
    # Time fields: accept datetime-like or numeric; convert to ISO string if possible
    for k in ("entry_time", "exit_time"):
        v = trade.get(k, "")
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = str(v) if v is not None else ""

    out["side"] = trade.get("side", "")
    out["entry_price"] = trade.get("entry_price", "")
    out["exit_price"] = trade.get("exit_price", "")
    out["size"] = trade.get("size", "")
    out["pnl"] = trade.get("pnl", "")
    out["pnl_gross"] = trade.get("pnl_gross", "")
    out["commission"] = trade.get("commission", "")
    out["exit_reason"] = trade.get("exit_reason", "")
    out["entry_bar_index"] = trade.get("entry_bar_index", "")
    out["exit_bar_index"] = trade.get("exit_bar_index", "")
    return out


def export_trades_csv(trades: List[Dict[str, Any]], out_path: str, fieldnames: List[str] = None) -> None:
    """
    Export trades to CSV. If trades contain additional keys, they are ignored.
    """
    if fieldnames is None:
        fieldnames = DEFAULT_FIELDNAMES

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            row = _normalize_trade_row(t)
            # Keep only requested columns
            filtered = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(filtered)


def main():
    parser = argparse.ArgumentParser(description="Run backtest and export trades to CSV.")
    parser.add_argument("--input", "-i", required=True, help="Path to candles CSV/TSV file.")
    parser.add_argument("--output", "-o", default="trades_export.csv", help="Output CSV path.")
    parser.add_argument("--ticker", "-t", default="NVDA", help="Ticker preset to use.")
    parser.add_argument("--intrabar-path", choices=["ohlc", "olhc"], default="ohlc",
                        help="Intrabar simulation path to use (ohlc or olhc).")
    parser.add_argument("--slippage", type=float, default=0.0, help="Slippage per fill (price units).")
    parser.add_argument("--commission-pct", type=float, default=0.0, help="Commission fraction per trade (e.g., 0.001).")
    parser.add_argument("--position-size", type=float, default=1.0, help="Fixed contract/share size per entry.")
    parser.add_argument("--pyramiding", type=int, default=1, help="Max concurrent entries per side.")
    args = parser.parse_args()

    try:
        candles = load_candles_from_csv(args.input)
    except Exception as e:
        print(f"Failed to load candles from {args.input}: {e}", file=sys.stderr)
        sys.exit(2)

    params = {
        "ticker": args.ticker,
        "intrabar_path": args.intrabar_path,
        "slippage": args.slippage,
        "commission_pct": args.commission_pct,
        "position_size": args.position_size,
        "pyramiding": args.pyramiding,
    }

    try:
        result = run_backtest(candles, params)
    except Exception as e:
        print(f"Backtest failed: {e}", file=sys.stderr)
        sys.exit(3)

    trades = result.get("trade_dicts", [])
    if not isinstance(trades, list):
        print("Backtest returned unexpected trade list format.", file=sys.stderr)
        sys.exit(4)

    try:
        export_trades_csv(trades, args.output)
    except Exception as e:
        print(f"Failed to export trades to CSV: {e}", file=sys.stderr)
        sys.exit(5)

    print(f"Export complete: {args.output} ({len(trades)} trades)")


if __name__ == "__main__":
    main()
