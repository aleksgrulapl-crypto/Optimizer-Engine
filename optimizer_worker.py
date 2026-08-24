# optimizer_worker.py
"""
Worker with candidate tracking, resume, and indicator caching.

- Writes per-ticker completed_runs JSONL to completed_runs/{SYMBOL}.jsonl
- Skips already-evaluated parameter sets (deterministic key)
- Updates optimizer_results/best_{SYMBOL}.csv and report_{SYMBOL}.json as it finds better presets
- Updates optimizer_results/progress.csv (via optimize_all.py aggregator)
"""

import time
import csv
import json
import itertools
import random
from pathlib import Path
from typing import Dict, Any, List, Iterator, Tuple

from data_loader import load_candles_from_csv
from backtest_engine import run_backtest
from presets import get_presets

# -------------------------
# Helpers: grid expansion (same as before)
# -------------------------
def _linspace_float(start: float, stop: float, count: int) -> List[float]:
    if count <= 1:
        return [float(start)]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]

def _frange_step(start: float, stop: float, step: float) -> List[float]:
    vals = []
    v = float(start)
    eps = 1e-12
    while v <= stop + eps:
        vals.append(round(v, 12))
        v += float(step)
    return vals

def _expand_spec(value_spec: Any) -> List[Any]:
    if isinstance(value_spec, list):
        return value_spec
    if isinstance(value_spec, dict):
        if "start" in value_spec and "stop" in value_spec and "step" in value_spec:
            return _frange_step(value_spec["start"], value_spec["stop"], value_spec["step"])
        if "min" in value_spec and "max" in value_spec and "count" in value_spec:
            start = float(value_spec["min"])
            stop = float(value_spec["max"])
            count = int(value_spec["count"])
            if float(start).is_integer() and float(stop).is_integer():
                vals = _linspace_float(start, stop, count)
                return [int(round(x)) for x in vals]
            return _linspace_float(start, stop, count)
    return [value_spec]

def _count_combinations(grid_expanded: Dict[str, List[Any]]) -> int:
    total = 1
    for v in grid_expanded.values():
        total *= max(1, len(v))
        if total > 10_000_000:
            return total
    return total

def grid_search_params(grid: Dict[str, Any],
                       search_mode: str = "auto",
                       n_samples: int = 1000,
                       max_exhaustive: int = 200_000,
                       seed: int = 0) -> Iterator[Dict[str, Any]]:
    grid_expanded: Dict[str, List[Any]] = {}
    for k, v in grid.items():
        grid_expanded[k] = _expand_spec(v)

    total = _count_combinations(grid_expanded)
    mode = search_mode
    if search_mode == "auto":
        mode = "exhaustive" if total <= max_exhaustive else "sample"

    keys = list(grid_expanded.keys())
    if mode == "exhaustive":
        lists = [grid_expanded[k] for k in keys]
        for combo in itertools.product(*lists):
            yield dict(zip(keys, combo))
        return

    random.seed(seed)
    if total <= n_samples:
        for combo in itertools.product(*(grid_expanded[k] for k in keys)):
            yield dict(zip(keys, combo))
        return

    seen = set()
    attempts = 0
    max_attempts = n_samples * 10
    while len(seen) < n_samples and attempts < max_attempts:
        attempts += 1
        candidate = {}
        for k in keys:
            vals = grid_expanded[k]
            v = random.choice(vals)
            candidate[k] = v
        keyt = tuple((kk, candidate[kk]) for kk in keys)
        if keyt in seen:
            continue
        seen.add(keyt)
        yield candidate

    if len(seen) < n_samples:
        for combo in itertools.product(*(grid_expanded[k] for k in keys)):
            keyt = tuple((kk, combo[i]) for i, kk in enumerate(keys))
            if keyt in seen:
                continue
            seen.add(keyt)
            yield dict(zip(keys, combo))
            if len(seen) >= n_samples:
                break

# -------------------------
# Metrics and scoring (profit-first, modest drawdown penalty)
# -------------------------
def _build_equity_series_from_trades(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_bar = {}
    for t in trades:
        idx = t.get("exit_bar_index", None)
        pnl = float(t.get("pnl", 0.0))
        if idx is None:
            continue
        by_bar[idx] = by_bar.get(idx, 0.0) + pnl

    equity = []
    cum = 0.0
    for idx in sorted(by_bar.keys()):
        cum += by_bar[idx]
        equity.append({"bar_index": int(idx), "equity": float(cum)})
    return equity

def _compute_max_drawdown(equity_series: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not equity_series:
        return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0, "max_drawdown_duration": 0}
    peak_equity = equity_series[0]["equity"]
    peak_idx = equity_series[0]["bar_index"]
    max_dd = 0.0
    max_dd_pct = 0.0
    trough_idx = peak_idx
    recovery_idx = None
    for point in equity_series:
        idx = point["bar_index"]
        eq = point["equity"]
        if eq > peak_equity:
            peak_equity = eq
            peak_idx = idx
        dd = peak_equity - eq
        dd_pct = (dd / peak_equity) if peak_equity != 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
            trough_idx = idx
            recovery_idx = None
    if max_dd > 0:
        peak_val = None
        for p in equity_series:
            if p["bar_index"] <= trough_idx:
                if peak_val is None or p["equity"] > peak_val:
                    peak_val = p["equity"]
        rec_idx = None
        for p in equity_series:
            if p["bar_index"] > trough_idx and p["equity"] >= (peak_val or 0.0):
                rec_idx = p["bar_index"]
                break
        recovery_idx = rec_idx
    duration = 0
    if recovery_idx is not None and peak_idx is not None:
        duration = recovery_idx - peak_idx
    elif trough_idx is not None and peak_idx is not None:
        duration = trough_idx - peak_idx
    return {"max_drawdown": float(max_dd), "max_drawdown_pct": float(max_dd_pct), "max_drawdown_duration": int(duration)}

def compute_metrics_from_run(res: Dict[str, Any]) -> Dict[str, Any]:
    trades = res.get("trade_dicts", [])
    net = sum(float(t.get("pnl", 0.0)) for t in trades)
    wins = [t for t in trades if float(t.get("pnl_gross", 0.0)) > 0]
    losses = [t for t in trades if float(t.get("pnl_gross", 0.0)) <= 0]
    trade_count = len(trades)
    win_rate = len(wins) / trade_count if trade_count else 0.0
    profit = sum(float(t.get("pnl_gross", 0.0)) for t in wins)
    loss = -sum(float(t.get("pnl_gross", 0.0)) for t in losses) if losses else 0.0
    pf = (profit / loss) if loss > 0 else float("inf") if profit > 0 else 0.0
    equity_series = _build_equity_series_from_trades(trades)
    dd = _compute_max_drawdown(equity_series)
    return {
        "net_profit": float(net),
        "trade_count": int(trade_count),
        "win_rate": float(win_rate),
        "profit_factor": float(pf),
        "equity_series": equity_series,
        "max_drawdown": dd["max_drawdown"],
        "max_drawdown_pct": dd["max_drawdown_pct"],
        "max_drawdown_duration": dd["max_drawdown_duration"],
    }

def score_candidate(metrics: Dict[str, Any]) -> float:
    net_profit = metrics.get("net_profit", 0.0)
    dd_pct = metrics.get("max_drawdown_pct", 0.0)
    pf = metrics.get("profit_factor", 0.0)
    win_rate = metrics.get("win_rate", 0.0)
    trade_count = metrics.get("trade_count", 1)
    profit_component = net_profit * 1.0
    drawdown_penalty = dd_pct * 100.0
    pf_component = pf * 50.0
    win_component = win_rate * 50.0
    trade_penalty = max(0, trade_count - 400) * 0.1
    score = profit_component + pf_component + win_component - drawdown_penalty - trade_penalty
    return score

# -------------------------
# Tracking helpers
# -------------------------
def _param_key(params: Dict[str, Any]) -> str:
    # deterministic key: sorted items JSON
    return json.dumps({k: params[k] for k in sorted(params.keys())}, sort_keys=True, separators=(",", ":"))

def _append_completed_run(symbol: str, record: Dict[str, Any]):
    out_dir = Path("completed_runs")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{symbol}.jsonl"
    # atomic append
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def _load_completed_keys(symbol: str) -> set:
    path = Path("completed_runs") / f"{symbol}.jsonl"
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                k = rec.get("_param_key")
                if k:
                    keys.add(k)
            except Exception:
                continue
    return keys

def _write_best_csv(symbol: str, top: List[Dict[str, Any]]):
    out_dir = Path("optimizer_results")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"best_{symbol}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "score", "net_profit", "trade_count", "profit_factor", "max_drawdown", "max_drawdown_pct", "params"
        ])
        for rank, c in enumerate(top, start=1):
            m = c["metrics"]
            writer.writerow([
                rank,
                c["score"],
                m["net_profit"],
                m["trade_count"],
                m["profit_factor"],
                m.get("max_drawdown", 0.0),
                m.get("max_drawdown_pct", 0.0),
                json.dumps(c["params"])
            ])

def _write_report(symbol: str, top: List[Dict[str, Any]]):
    out_dir = Path("optimizer_results")
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / f"report_{symbol}.json"
    if top:
        best = top[0]
        r = {"symbol": symbol, "best_params": best["params"], "metrics": best["metrics"], "score": best["score"]}
        report_path.write_text(json.dumps(r, indent=2))
    else:
        report_path.write_text(json.dumps({"symbol": symbol, "error": "no profitable candidates"}, indent=2))

# -------------------------
# Worker
# -------------------------
def optimize_ticker(cfg: Dict[str, Any],
                    grid: Dict[str, Any],
                    intrabar_paths: List[str],
                    top_k: int = 3,
                    time_budget: int = 3600,
                    search_mode: str = "auto",
                    n_samples: int = 1000,
                    seed: int = 0,
                    max_exhaustive: int = 200000) -> Dict[str, Any]:
    symbol = cfg.get("symbol")
    tsv = cfg.get("tsv")
    start_time = time.time()

    # Load candles once (caching)
    candles = load_candles_from_csv(tsv)

    # Load completed keys to resume
    completed_keys = _load_completed_keys(symbol)

    best_candidates: List[Dict[str, Any]] = []
    evaluated = 0

    # Iterate candidates
    for params in grid_search_params(grid, search_mode=search_mode, n_samples=n_samples, max_exhaustive=max_exhaustive, seed=seed):
        if time.time() - start_time > time_budget:
            break

        for path in intrabar_paths:
            run_params = dict(params)
            run_params.update({
                "ticker": symbol,
                "intrabar_path": path,
                "position_size": 1.0,
                "slippage": 0.0,
                "commission_pct": 0.0
            })

            key = _param_key(run_params)
            if key in completed_keys:
                # skip already evaluated
                continue

            # Run backtest
            res = run_backtest(candles, run_params)
            metrics = compute_metrics_from_run(res)

            # Hard filters: require positive profit and PF>1
            if metrics.get("net_profit", 0.0) <= 0.0 or metrics.get("profit_factor", 0.0) <= 1.0:
                # record as completed but do not include in best_candidates
                rec = {
                    "timestamp": time.time(),
                    "_param_key": key,
                    "params": run_params,
                    "metrics": metrics,
                    "score": None,
                    "status": "rejected"
                }
                _append_completed_run(symbol, rec)
                completed_keys.add(key)
                evaluated += 1
                continue

            # compute score and record
            score = score_candidate(metrics)
            candidate = {"params": run_params, "metrics": metrics, "score": score, "res": res}
            best_candidates.append(candidate)

            rec = {
                "timestamp": time.time(),
                "_param_key": key,
                "params": run_params,
                "metrics": metrics,
                "score": score,
                "status": "accepted"
            }
            _append_completed_run(symbol, rec)
            completed_keys.add(key)
            evaluated += 1

            # keep only top_k in memory
            best_candidates.sort(key=lambda x: x["score"], reverse=True)
            best_candidates = best_candidates[:max(10, top_k)]

            # update best CSV/report incrementally
            top = best_candidates[:top_k]
            _write_best_csv(symbol, top)
            _write_report(symbol, top)

            # check time budget
            if time.time() - start_time > time_budget:
                break

    # Finalize
    best_candidates.sort(key=lambda x: x["score"], reverse=True)
    top = best_candidates[:top_k]

    # If no accepted candidates, write a report indicating none found
    if not top:
        _write_best_csv(symbol, [])
        _write_report(symbol, [])

    elapsed = time.time() - start_time
    return {"symbol": symbol, "top": top, "csv": str(Path("optimizer_results") / f"best_{symbol}.csv"), "report": str(Path("optimizer_results") / f"report_{symbol}.json"), "evaluated": evaluated, "elapsed_seconds": elapsed}
