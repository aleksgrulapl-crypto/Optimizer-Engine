"""
Run backtest and export trades for TradingView parity checks.
"""

import csv
import sys
import argparse
import re
from typing import List, Dict, Any
from pathlib import Path

from backtest_engine import run_backtest
from data_loader import load_candles_from_csv
from presets import get_presets, normalize_timeframe


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
    out = {}
    for k in ("entry_time", "exit_time"):
        v = trade.get(k, "")
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = str(v) if v is not None else ""

    for k in DEFAULT_FIELDNAMES:
        if k not in out:
            out[k] = trade.get(k, "")
    return out


def export_trades_csv(trades: List[Dict[str, Any]], out_path: str, fieldnames: List[str] = None) -> None:
    if fieldnames is None:
        fieldnames = DEFAULT_FIELDNAMES

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            row = _normalize_trade_row(t)
            filtered = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(filtered)


def _infer_timeframe(path: str) -> str:
    match = re.search(r"_(\d+[mhd])\.", Path(path).name, re.IGNORECASE)
    if not match:
        return "15M"
    return normalize_timeframe(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest and export trades to CSV.")
    parser.add_argument("--input", "-i", required=True, help="Path to candles CSV/TSV file.")
    parser.add_argument("--output", "-o", default="trades_export.csv", help="Output CSV path.")
    parser.add_argument("--ticker", "-t", default="NVDA", help="Ticker preset to use.")
    parser.add_argument("--timeframe", default=None, help="Preset timeframe to use (15M or 30M). Defaults to input filename inference.")
    parser.add_argument("--intrabar-path", choices=["ohlc", "olhc"], default="ohlc")
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--commission-pct", type=float, default=0.0)
    parser.add_argument("--position-size", type=float, default=1.0)
    parser.add_argument("--pyramiding", type=int, default=1)

    # Optional explicit strategy params to exactly match TradingView setup.
    parser.add_argument("--st-multiplier", type=float, default=None)
    parser.add_argument("--st-period", type=int, default=None)
    parser.add_argument("--atr-sl-mult", type=float, default=None)
    parser.add_argument("--atr-tp-mult", type=float, default=None)
    parser.add_argument("--ema-len", type=int, default=None)

    args = parser.parse_args()

    try:
        candles = load_candles_from_csv(args.input)
    except Exception as e:
        print(f"Failed to load candles from {args.input}: {e}", file=sys.stderr)
        sys.exit(2)

    timeframe = normalize_timeframe(args.timeframe) if args.timeframe else _infer_timeframe(args.input)
    preset = get_presets(args.ticker, timeframe)
    params = {
        "ticker": args.ticker,
        "timeframe": timeframe,
        "intrabar_path": args.intrabar_path,
        "slippage": args.slippage,
        "commission_pct": args.commission_pct,
        "position_size": args.position_size,
        "pyramiding": args.pyramiding,
        "stMultiplier": args.st_multiplier if args.st_multiplier is not None else preset.get("stMultiplier"),
        "stPeriod": args.st_period if args.st_period is not None else preset.get("stPeriod"),
        "atrSLmult": args.atr_sl_mult if args.atr_sl_mult is not None else preset.get("atrSLmult"),
        "atrTPmult": args.atr_tp_mult if args.atr_tp_mult is not None else preset.get("atrTPmult"),
        "emaLen": args.ema_len if args.ema_len is not None else preset.get("emaLen"),
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
