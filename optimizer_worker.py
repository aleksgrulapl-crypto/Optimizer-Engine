"""
Worker with candidate tracking, resume, phased runs, robustness filtering, and
per-ticker staged search (initial random -> expanded -> refined grids).
"""

import time
import csv
import json
import itertools
import hashlib
import random
from pathlib import Path
from typing import Dict, Any, List, Iterator, Tuple

from data_loader import load_candles_from_csv
from backtest_engine import run_backtest

# Default acceptance filters for a candidate to be considered suitable:
# win rate >= 50%, profit factor >= 1.2, strictly positive net profit.
DEFAULT_FILTERS: Dict[str, Any] = {
    "min_win_rate": 0.50,
    "min_profit_factor": 1.2,
    "min_net_profit": 0.0,
    "min_trades": 10,
}

# Strong-candidate bar: a ticker/timeframe run is only considered complete
# once at least one candidate clears these thresholds — strictly positive net
# profit, profit factor >= 1.4, win rate >= 45% and a reasonably low max
# drawdown. The optimizer does not treat a ticker as finished until a strong
# candidate has been found (or its time budget is exhausted).
STRONG_FILTERS: Dict[str, Any] = {
    "min_win_rate": 0.45,
    "min_profit_factor": 1.4,
    "min_net_profit": 0.0,
    "min_trades": 10,
    "max_drawdown_pct": 0.25,
}

# Fallback step sizes per parameter when the base grid does not define one
# (single-valued specs): (float step, integer step).
NEIGHBORHOOD_STEPS: Dict[str, Tuple[float, int]] = {
    "stMultiplier": (0.1, 0),
    "stPeriod": (0.0, 1),
    "atrSLmult": (0.1, 0),
    "atrTPmult": (0.2, 0),
    "emaLen": (0.0, 5),
}


# -------------------------
# Grid helpers
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


# -------------------------
# Metrics and scoring
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
    duration = max(0, trough_idx - peak_idx)
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


def _safe_pf(pf: float) -> float:
    if pf == float("inf"):
        return 10.0
    return max(0.0, min(float(pf), 10.0))


def score_candidate(metrics: Dict[str, Any]) -> float:
    # Priority: PF and drawdown; secondary: net profit + win rate.
    net_profit = float(metrics.get("net_profit", 0.0))
    dd_pct = float(metrics.get("max_drawdown_pct", 0.0))
    pf = _safe_pf(float(metrics.get("profit_factor", 0.0)))
    win_rate = float(metrics.get("win_rate", 0.0))
    trade_count = int(metrics.get("trade_count", 0))

    pf_component = pf * 250.0
    drawdown_penalty = dd_pct * 180.0
    profit_component = net_profit * 0.6
    win_component = win_rate * 30.0
    low_sample_penalty = 25.0 if trade_count < 30 else 0.0

    return pf_component + profit_component + win_component - drawdown_penalty - low_sample_penalty


def passes_filters(metrics: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Suitability check: win rate, profit factor, net profit and trade count floors."""
    pf = float(metrics.get("profit_factor", 0.0))
    net = float(metrics.get("net_profit", 0.0))
    wr = float(metrics.get("win_rate", 0.0))
    tc = int(metrics.get("trade_count", 0))
    if net <= float(filters.get("min_net_profit", 0.0)):
        return False
    if pf < float(filters.get("min_profit_factor", 1.2)):
        return False
    if wr < float(filters.get("min_win_rate", 0.50)):
        return False
    if tc < int(filters.get("min_trades", 10)):
        return False
    return True


def is_strong_candidate(metrics: Dict[str, Any], strong_filters: Dict[str, Any] = None) -> bool:
    """Strong-candidate check: positive net profit, profit factor >= 1.4,
    win rate >= 45% and a reasonably low max drawdown. A ticker/timeframe run
    is only treated as complete once a candidate passes this bar."""
    if not passes_filters(metrics, {**STRONG_FILTERS, **(strong_filters or {})}):
        return False
    dd_pct = float(metrics.get("max_drawdown_pct", 0.0))
    cap = float((strong_filters or {}).get("max_drawdown_pct", STRONG_FILTERS["max_drawdown_pct"]))
    return dd_pct <= cap


def derive_seed(base_seed: int, symbol: Any, timeframe: Any) -> int:
    """Per-ticker random seed: stable hash of (base seed, symbol, timeframe)
    so parallel workers sample different points of the grid instead of all
    walking the same random sequence (spreads the load across the CPU)."""
    key = f"{base_seed}|{symbol}|{timeframe}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def _candidate_rank_key(candidate: Dict[str, Any]) -> Tuple[float, float]:
    """Rank by score first; on ties prefer the lower max drawdown."""
    m = candidate.get("metrics", {}) or {}
    return (float(candidate.get("score", 0.0)), -float(m.get("max_drawdown", float("inf"))))


def _sort_candidates(candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    candidates.sort(key=_candidate_rank_key, reverse=True)
    return candidates[:max(20, top_k)]


def _unique_sorted(values: List[Any]) -> List[Any]:
    out = []
    for v in values:
        if v not in out:
            out.append(v)
    out.sort()
    return out


def _dedup_keep_order(values: List[Any]) -> List[Any]:
    out = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


def build_neighborhood_grid(grid: Dict[str, Any], center: Dict[str, Any], radius: float) -> Dict[str, Any]:
    """Build a grid centered on ``center`` params around a promising candidate.

    Integer parameters (e.g. stPeriod, emaLen): ``radius`` is a count of grid
    steps taken in each direction. Float parameters: ``radius`` is the distance
    covered in each direction, keeping the base grid's step (never coarser than
    the base grid). Step sizes are inferred from the base grid, falling back to
    ``NEIGHBORHOOD_STEPS`` for single-valued specs.
    """
    neighborhood: Dict[str, Any] = {}
    for key, spec in grid.items():
        base_values = _expand_spec(spec)
        center_value = center.get(key, base_values[0] if base_values else None)
        try:
            f_center = float(center_value)
        except (TypeError, ValueError):
            neighborhood[key] = _dedup_keep_order([center_value] + list(base_values))
            continue

        uniq = _unique_sorted(base_values)
        is_int = bool(uniq) and all(isinstance(v, int) and not isinstance(v, bool) for v in uniq)
        if len(uniq) >= 2:
            base_step = round(min(float(uniq[i + 1]) - float(uniq[i]) for i in range(len(uniq) - 1)), 10)
        else:
            base_step = 0.0

        if is_int:
            step = max(1.0, base_step or float(NEIGHBORHOOD_STEPS.get(key, (0.1, 1))[1]))
            n_steps = max(0, int(round(float(radius))))
            offsets = [i * step for i in range(-n_steps, n_steps + 1)]
            values = _unique_sorted([int(round(f_center + o)) for o in offsets])
        else:
            r = abs(float(radius))
            if base_step > 0.0:
                n_steps = max(1, int(round(r / base_step))) if r > 0.0 else 0
                step = min(base_step, r / n_steps) if r > 0.0 else base_step
            else:
                step = float(NEIGHBORHOOD_STEPS.get(key, (0.1, 1))[0])
                n_steps = max(0, int(round(r / step))) if step > 0.0 else 0
            offsets = [i * step for i in range(-n_steps, n_steps + 1)]
            values = _unique_sorted([round(f_center + o, 10) for o in offsets])

        # Keep values within the base grid's range so neighborhoods stay valid.
        if len(uniq) >= 2:
            lo, hi = float(uniq[0]), float(uniq[-1])
            values = [v for v in values if lo - 1e-9 <= float(v) <= hi + 1e-9]
            if f_center < lo or f_center > hi:
                nearest = min(uniq, key=lambda v: abs(float(v) - f_center))
                values = list(values) + [nearest]
            values = _unique_sorted(values)

        neighborhood[key] = values if values else _dedup_keep_order([center_value] + list(base_values))
    return neighborhood


# -------------------------
# Tracking helpers
# -------------------------
def _param_key(params: Dict[str, Any]) -> str:
    return json.dumps({k: params[k] for k in sorted(params.keys())}, sort_keys=True, separators=(",", ":"))


def _run_label(symbol: str, timeframe: Any = None) -> str:
    """File/tracking label for a run; includes the timeframe when available so
    15m and 30m runs for the same symbol do not overwrite each other."""
    label = str(symbol)
    tf = str(timeframe or "").strip().lower()
    if tf:
        label = f"{label}_{tf}"
    return label


def _append_completed_run(label: str, phase: str, record: Dict[str, Any]) -> None:
    out_dir = Path("completed_runs")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{label}_{phase}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _load_completed_keys(label: str, phase: str) -> set:
    path = Path("completed_runs") / f"{label}_{phase}.jsonl"
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


def _write_best_csv(label: str, top: List[Dict[str, Any]], phase: str) -> str:
    out_dir = Path("optimizer_results")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"best_{label}_{phase}.csv"
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
    return str(csv_path)


def _write_report(label: str, top: List[Dict[str, Any]], phase: str, note: str, symbol: str = None, timeframe: Any = None) -> str:
    out_dir = Path("optimizer_results")
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / f"report_{label}_{phase}.json"
    if top:
        best = top[0]
        r = {
            "symbol": symbol if symbol is not None else label,
            "timeframe": timeframe,
            "phase": phase,
            "note": note,
            "best_params": best["params"],
            "metrics": best["metrics"],
            "score": best["score"],
            "robustness": best.get("robustness"),
        }
        report_path.write_text(json.dumps(r, indent=2))
    else:
        report_path.write_text(json.dumps({"symbol": symbol if symbol is not None else label, "timeframe": timeframe, "phase": phase, "note": note, "error": "no valid candidates"}, indent=2))
    return str(report_path)


def _segment_candles(candles: List[Dict[str, Any]], segments: int) -> List[List[Dict[str, Any]]]:
    if segments <= 1 or len(candles) < segments * 20:
        return [candles]
    n = len(candles)
    out = []
    for i in range(segments):
        s = int(i * n / segments)
        e = int((i + 1) * n / segments)
        part = candles[s:e]
        if len(part) >= 20:
            out.append(part)
    return out if out else [candles]


def _robustness_filter(
    candles: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    robustness_cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], str]:
    if not bool(robustness_cfg.get("enabled", True)):
        return candidates, "robustness disabled"

    segments = int(robustness_cfg.get("segments", 3))
    evaluate_top_n = int(robustness_cfg.get("evaluate_top_n", 8))
    pf_floor = float(robustness_cfg.get("reject_if_any_segment_pf_below", 1.0))
    net_floor = float(robustness_cfg.get("reject_if_any_segment_net_profit_below_or_equal", 0.0))

    segmented = _segment_candles(candles, segments)
    if len(segmented) <= 1:
        return candidates, "robustness skipped (not enough candles)"

    checked = candidates[:max(1, evaluate_top_n)]
    survivors: List[Dict[str, Any]] = []

    for c in checked:
        params = c["params"]
        seg_stats = []
        reject = False
        for seg in segmented:
            res = run_backtest(seg, params)
            m = compute_metrics_from_run(res)
            seg_stats.append({
                "net_profit": m.get("net_profit", 0.0),
                "profit_factor": m.get("profit_factor", 0.0),
                "max_drawdown_pct": m.get("max_drawdown_pct", 0.0),
                "trade_count": m.get("trade_count", 0),
            })
            if float(m.get("profit_factor", 0.0)) <= pf_floor or float(m.get("net_profit", 0.0)) <= net_floor:
                reject = True
                break
        if reject:
            continue
        c2 = dict(c)
        c2["robustness"] = {"segments": len(segmented), "segment_metrics": seg_stats}
        survivors.append(c2)

    if survivors:
        survivors.sort(key=_candidate_rank_key, reverse=True)
        return survivors + candidates[evaluate_top_n:], f"robustness kept {len(survivors)}/{len(checked)} checked candidates"

    return candidates, "robustness rejected all checked candidates; fallback to unfiltered ranking"


# -------------------------
# Worker
# -------------------------
def optimize_ticker(cfg: Dict[str, Any],
                    grid: Dict[str, Any],
                    intrabar_paths: List[str],
                    top_k: int = 5,
                    time_budget: int = 1800,
                    search_mode: str = "auto",
                    n_samples: int = 1000,
                    seed: int = 0,
                    max_exhaustive: int = 200000,
                    execution: Dict[str, Any] = None,
                    robustness: Dict[str, Any] = None,
                    phase: str = "constrained",
                    filters: Dict[str, Any] = None) -> Dict[str, Any]:
    symbol = cfg.get("symbol")
    tsv = cfg.get("tsv")
    timeframe = cfg.get("timeframe")
    label = _run_label(symbol, timeframe)
    start_time = time.time()

    execution = execution or {}
    robustness = robustness or {}
    filters = {**DEFAULT_FILTERS, **(filters or {})}

    candles = load_candles_from_csv(tsv)
    completed_keys = _load_completed_keys(label, phase)

    best_candidates: List[Dict[str, Any]] = []
    evaluated = 0
    best_score_so_far: float = float("-inf")

    slippage = float(execution.get("slippage", 0.0))
    commission_pct = float(execution.get("commission_pct", 0.0))
    position_size = float(execution.get("position_size", 1.0))
    pyramiding = int(execution.get("pyramiding", 1))

    # Pre-count total combinations for progress display
    grid_expanded: Dict[str, List[Any]] = {k: _expand_spec(v) for k, v in grid.items()}
    total_combos = _count_combinations(grid_expanded) * len(intrabar_paths)

    scan_counter = 0

    for params in grid_search_params(grid, search_mode=search_mode, n_samples=n_samples, max_exhaustive=max_exhaustive, seed=seed):
        if time.time() - start_time > time_budget:
            break

        for path in intrabar_paths:
            scan_counter += 1
            run_params = dict(params)
            run_params.update({
                "ticker": symbol,
                "timeframe": timeframe,
                "intrabar_path": path,
                "position_size": position_size,
                "slippage": slippage,
                "commission_pct": commission_pct,
                "pyramiding": pyramiding,
            })

            key = _param_key(run_params)

            print(f"[{label}][{phase}] Scanning {scan_counter}/{total_combos} | evaluated={evaluated} | best_score={best_score_so_far:.2f}" if best_score_so_far != float('-inf') else f"[{label}][{phase}] Scanning {scan_counter}/{total_combos} | evaluated={evaluated}", flush=True)

            if key in completed_keys:
                continue

            res = run_backtest(candles, run_params)
            metrics = compute_metrics_from_run(res)

            # Hard filters (defaults: net profit > 0, PF >= 1.2, win rate >= 50%, at least 10 trades)
            pf = float(metrics.get("profit_factor", 0.0))
            net = float(metrics.get("net_profit", 0.0))
            wr = float(metrics.get("win_rate", 0.0))
            tc = int(metrics.get("trade_count", 0))

            if not passes_filters(metrics, filters):
                rec = {
                    "timestamp": time.time(),
                    "_param_key": key,
                    "params": run_params,
                    "metrics": metrics,
                    "score": None,
                    "status": "rejected"
                }
                _append_completed_run(label, phase, rec)
                completed_keys.add(key)
                evaluated += 1
                continue

            score = score_candidate(metrics)
            candidate = {"params": run_params, "metrics": metrics, "score": score, "res": res}
            best_candidates.append(candidate)

            if score > best_score_so_far:
                best_score_so_far = score
                print(
                    f"\n*** [{label}][{phase}] NEW BEST FOUND ***\n"
                    f"    Score:         {score:.4f}\n"
                    f"    Profit Factor: {pf:.4f}\n"
                    f"    Net Profit:    {net:.4f}\n"
                    f"    Win Rate:      {wr*100:.1f}%\n"
                    f"    Max DD:        {metrics.get('max_drawdown_pct', 0.0)*100:.2f}%\n"
                    f"    Trade Count:   {tc}\n"
                    f"    Params:        {json.dumps(run_params)}\n",
                    flush=True,
                )

            rec = {
                "timestamp": time.time(),
                "_param_key": key,
                "params": run_params,
                "metrics": metrics,
                "score": score,
                "status": "accepted"
            }
            _append_completed_run(label, phase, rec)
            completed_keys.add(key)
            evaluated += 1

            best_candidates = _sort_candidates(best_candidates, top_k)

            if time.time() - start_time > time_budget:
                break

    _sort_candidates(best_candidates, top_k)

    robust_note = "robustness not run"
    if best_candidates:
        best_candidates, robust_note = _robustness_filter(candles, best_candidates, robustness)
        _sort_candidates(best_candidates, top_k)

    top = best_candidates[:top_k]

    csv_path = _write_best_csv(label, top, phase)
    report_path = _write_report(label, top, phase, robust_note, symbol=symbol, timeframe=timeframe)

    elapsed = time.time() - start_time
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "phase": phase,
        "top": top,
        "csv": csv_path,
        "report": report_path,
        "evaluated": evaluated,
        "elapsed_seconds": elapsed,
        "note": robust_note,
    }


# -------------------------
# Staged search
# -------------------------
def staged_search(ticker: Dict[str, Any],
                  grid: Dict[str, Any],
                  intrabar_paths: List[str],
                  top_k: int = 5,
                  time_budget: int = 1800,
                  n_samples: int = 1000,
                  seed: int = 0,
                  max_exhaustive: int = 200000,
                  execution: Dict[str, Any] = None,
                  robustness: Dict[str, Any] = None,
                  staged_cfg: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Run the staged search strategy for a single ticker/timeframe entry.

    Stage order:
      1) initial   - random sampling of the base grid (wide, loose ranges)
      2) expanded  - grid expanded around the best suitable initial candidate
      3) refined   - narrower grid around the best candidate from the expanded stage

    Candidates must satisfy the configured filters (win rate, profit factor,
    net profit, trade count); ranking prefers lower drawdown on score ties.
    Later stages are skipped when no suitable candidate is found, and the
    refined stage only runs once a strong candidate (positive net profit,
    PF >= 1.4, WR >= 45%, low drawdown) has been found. Each stage reports
    ``strong_candidate_found`` so the orchestrator only treats a ticker as
    finished once the strong bar is cleared (or its budget is exhausted).
    """
    staged_cfg = staged_cfg or {}
    filters = staged_cfg.get("filters", None) or None
    strong_filters = {**STRONG_FILTERS, **(staged_cfg.get("strong_filters", {}) or {})}
    expand_radius = float(staged_cfg.get("expand_radius", 2.0))
    refine_radius = float(staged_cfg.get("refine_radius", 1.0))
    budget_split = staged_cfg.get("time_budget_split", [0.5, 0.3, 0.2]) or [0.5, 0.3, 0.2]
    budget_split = [float(x) for x in budget_split]
    total_weight = sum(budget_split) or 1.0

    def _budget(idx: int) -> int:
        weight = budget_split[idx] if idx < len(budget_split) else budget_split[-1]
        return max(1, int(time_budget * weight / total_weight))

    def _mark_strong(res: Dict[str, Any]) -> Dict[str, Any]:
        top = res.get("top", []) or []
        res["strong_candidate_found"] = any(
            is_strong_candidate(c.get("metrics", {}) or {}, strong_filters) for c in top
        )
        return res

    def _run(phase: str, phase_grid: Dict[str, Any], mode: str, samples: int, stage_seed: int) -> Dict[str, Any]:
        return _mark_strong(optimize_ticker(
            ticker,
            phase_grid,
            intrabar_paths,
            top_k=top_k,
            time_budget=_budget(len(results)),
            search_mode=mode,
            n_samples=samples,
            seed=stage_seed,
            max_exhaustive=max_exhaustive,
            execution=execution,
            robustness=robustness,
            phase=phase,
            filters=filters,
        ))

    results: List[Dict[str, Any]] = []

    # Stage 1: initial random grid search (per-ticker seed spreads sampled
    # points across parallel workers, taking pressure off the CPU).
    stage_seed = derive_seed(seed, ticker.get("symbol"), ticker.get("timeframe"))
    results.append(_run("initial", grid, "sample", n_samples, stage_seed))
    top = results[-1].get("top", []) or []
    if not top:
        return results

    # Stage 2: expanded grid around the best suitable candidate
    expanded_grid = build_neighborhood_grid(grid, top[0].get("params", {}), expand_radius)
    expand_samples = max(n_samples, int(staged_cfg.get("expand_samples", 2 * n_samples)))
    results.append(_run("expanded", expanded_grid, "sample", expand_samples, stage_seed + 1))
    top = results[-1].get("top", []) or []
    if not top:
        return results

    # Stage 3: refined grid around the top expanded candidate. Only refine
    # once the strong bar has been cleared — without a strong candidate there
    # is nothing worth narrowing down further.
    if not results[-1].get("strong_candidate_found"):
        return results
    refined_grid = build_neighborhood_grid(expanded_grid, top[0].get("params", {}), refine_radius)
    results.append(_run("refined", refined_grid, "auto", n_samples, stage_seed + 2))
    return results
