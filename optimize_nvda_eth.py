# optimize_nvda_eth.py
import argparse
import csv
from typing import Dict, Any, Optional

from data_loader import load_candles_from_csv
from backtest_engine import run_backtest
from metrics import compute_metrics


def optimize_nvda_eth(path: str,
                      st_mult_range=None,
                      st_period_range=None,
                      atr_sl_range=None,
                      atr_tp_range=None,
                      ema_len_range=None,
                      verbose: bool = True) -> Dict[str, Any]:
    """
    Grid-search optimizer for the NVDA 15m strategy.
    Returns the best parameter set according to Profit Factor (pf) with net_profit tie-breaker.
    """

    # Default ranges if not provided
    if st_mult_range is None:
        st_mult_range = [1.4, 1.7, 2.0]
    if st_period_range is None:
        st_period_range = [6, 8, 10]
    if atr_sl_range is None:
        atr_sl_range = [1.2, 1.4, 1.6]
    if atr_tp_range is None:
        atr_tp_range = [3.0, 4.0, 5.0]
    if ema_len_range is None:
        ema_len_range = [30, 50, 80]

    candles = load_candles_from_csv(path)

    best: Optional[Dict[str, Any]] = None
    total_iterations = (len(st_mult_range) * len(st_period_range) *
                        len(atr_sl_range) * len(atr_tp_range) * len(ema_len_range))
    iter_count = 0

    for st_m in st_mult_range:
        for st_p in st_period_range:
            for sl_m in atr_sl_range:
                for tp_m in atr_tp_range:
                    for ema_len in ema_len_range:
                        iter_count += 1
                        params = {
                            # pass the grid params into the backtest engine so it uses them
                            "ticker": "NVDA",
                            "stMultiplier": st_m,
                            "stPeriod": st_p,
                            "atrSLmult": sl_m,
                            "atrTPmult": tp_m,
                            "emaLen": ema_len,
                            # keep other defaults; you can add slippage/commission here if desired
                        }

                        result = run_backtest(candles, params)
                        metrics = compute_metrics(result.get("trade_dicts", []))

                        pf = metrics.get("profit_factor")
                        # Treat None (infinite) as very large for comparison
                        pf_cmp = float("inf") if pf is None else float(pf)

                        net_profit = metrics.get("net_profit", 0.0)
                        # Decide if this is better: higher pf, tie-breaker higher net_profit
                        is_better = False
                        if best is None:
                            is_better = True
                        else:
                            best_pf = best["pf_cmp"]
                            if pf_cmp > best_pf:
                                is_better = True
                            elif pf_cmp == best_pf and net_profit > best["metrics"].get("net_profit", -1e18):
                                is_better = True

                        if is_better:
                            best = {
                                "params": params,
                                "pf": pf,
                                "pf_cmp": pf_cmp,
                                "metrics": metrics,
                            }

                        if verbose:
                            print(f"[{iter_count}/{total_iterations}] st_m={st_m} st_p={st_p} "
                                  f"sl_m={sl_m} tp_m={tp_m} ema={ema_len} -> PF={pf} Net={net_profit}")

    return best


def export_best_to_csv(best: Dict[str, Any], out_path: str = "best_params.csv") -> None:
    """Export best params and metrics to a small CSV for record keeping."""
    if not best:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k, v in best["params"].items():
            writer.writerow([k, v])
        # metrics flattened
        writer.writerow([])
        writer.writerow(["metric", "value"])
        for k, v in best["metrics"].items():
            writer.writerow([k, v])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize NVDA 15m Supertrend grid.")
    parser.add_argument("--input", "-i", required=True, help="Path to candles CSV/TSV file.")
    parser.add_argument("--export", "-e", default=None, help="Optional CSV path to export best params.")
    args = parser.parse_args()

    best = optimize_nvda_eth(args.input)
    if best is None:
        print("No result found.")
    else:
        print("\nBest params:")
        for k, v in best["params"].items():
            print(f"  {k}: {v}")
        print("\nMetrics:")
        for k, v in best["metrics"].items():
            print(f"  {k}: {v}")

        if args.export:
            export_best_to_csv(best, args.export)
            print(f"\nExported best params to {args.export}")
