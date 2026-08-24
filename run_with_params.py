# run_with_params.py
"""
Run the backtest with explicit TradingView parameters and export trades to CSV.

Usage:
    python run_with_params.py                # uses defaults (intrabar_path="ohlc", output="nvda_test_tvparams.csv")
    python run_with_params.py --intrabar olhc --output nvda_test_olhc.csv
"""

import argparse
import csv
from pathlib import Path

from data_loader import load_candles_from_csv
from backtest_engine import run_backtest

def write_trades_csv(trades, out_path: Path):
    if not trades:
        # write an empty CSV with a sensible header
        header = ["entry_time","exit_time","side","entry_price","exit_price","size","pnl","pnl_gross","commission","exit_reason","entry_bar_index","exit_bar_index"]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        return

    # Ensure consistent column order
    keys = list(trades[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for t in trades:
            # convert any non-serializable values to strings
            row = {k: ("" if v is None else v) for k, v in t.items()}
            writer.writerow(row)

def main():
    p = argparse.ArgumentParser(description="Run backtest with explicit params and export trades")
    p.add_argument("--candles", "-i", default="data/NVDA_ETH_15m.tsv", help="Input candles CSV/TSV")
    p.add_argument("--output", "-o", default="nvda_test_tvparams.csv", help="Output trades CSV")
    p.add_argument("--intrabar", choices=["ohlc", "olhc"], default="ohlc", help="Intrabar path to simulate")
    p.add_argument("--slippage", type=float, default=0.0, help="Slippage per fill")
    p.add_argument("--commission-pct", type=float, default=0.0, help="Commission percent")
    p.add_argument("--position-size", type=float, default=1.0, help="Fixed position size (contracts)")
    args = p.parse_args()

    candles_path = Path(args.candles)
    if not candles_path.exists():
        raise FileNotFoundError(f"Candles file not found: {candles_path}")

    candles = load_candles_from_csv(str(candles_path))

    # Exact TradingView params (override here if you want different values)
    params = {
        "ticker": "NVDA",
        "stMultiplier": 2.0,
        "stPeriod": 8,
        "atrSLmult": 1.2,
        "atrTPmult": 4.0,
        "emaLen": 50,
        # runtime settings
        "intrabar_path": args.intrabar,
        "slippage": args.slippage,
        "commission_pct": args.commission_pct,
        "position_size": args.position_size,
    }

    print("Running backtest with params:")
    for k in sorted(params.keys()):
        print(f"  {k}: {params[k]}")

    res = run_backtest(candles, params)

    # Expecting run_backtest to return a dict with "trade_dicts" or "trades"
    trades = res.get("trade_dicts") or res.get("trades") or []

    out_path = Path(args.output)
    write_trades_csv(trades, out_path)
    print(f"Exported {len(trades)} trades to {out_path}")

if __name__ == "__main__":
    main()
