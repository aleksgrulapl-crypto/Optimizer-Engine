"""
Orchestrator with parity gating and per-ticker staged optimization.

Flow:
1) Data sanity checks (timeframe file exists + non-empty candles)
2) Parity gate checks (tv_export exists and parity_ok=true when enabled)
3) Initial random grid search over a wide, loose base grid (per ticker and
   timeframe; per-ticker seeds spread sampled points across workers to take
   pressure off the CPU)
4) Expanded grid around the best suitable candidate
5) Refined grid around the top expanded candidate (only once a strong
   candidate exists)
6) Robustness-filtered top candidates returned by worker

A ticker/timeframe run is only treated as finished once a strong candidate
has been found — positive net profit, profit factor >= 1.4, win rate >= 45%
and a reasonably low drawdown — or its time budget (about an hour by
default) is exhausted; the optimizer then moves on to the next ticker from
the config, which covers both the 15m and 30m timeframes per ticker.
"""

import json
import csv
import re
from multiprocessing import Pool
from pathlib import Path
from time import time
from typing import Dict, Any, List, Tuple

from data_loader import load_candles_from_csv

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    yaml = None  # type: ignore
    _HAS_YAML = False

from optimizer_worker import DEFAULT_FILTERS, STRONG_FILTERS, optimize_ticker, staged_search

DEFAULT_CONFIG: Dict[str, Any] = {
    "tickers": [],
    "grid": {
        "stMultiplier": {"start": 1.0, "stop": 5.0, "step": 0.2},
        "stPeriod": {"min": 6, "max": 18, "count": 13},
        "atrSLmult": {"start": 1.0, "stop": 2.6, "step": 0.2},
        "atrTPmult": {"start": 1.6, "stop": 10.0, "step": 0.4},
        "emaLen": {"min": 20, "max": 300, "count": 15},
    },
    "grid_constrained": None,
    "grid_expand": None,
    "grid_fallback": None,
    "parallel_workers": 2,
    "intrabar_paths": ["ohlc"],
    "top_k_per_ticker": 5,
    "time_budget_seconds_per_ticker": 3600,
    "search_mode": "auto",
    "n_samples_per_ticker": 1000,
    "random_seed": 0,
    "max_exhaustive": 150000,
    "execution": {
        "intrabar_path": "ohlc",
        "slippage": 0.0,
        "commission_pct": 0.0,
        "position_size": 1.0,
        "pyramiding": 1,
    },
    "parity": {
        "require_tv_export": True,
        "require_parity_ok": True,
    },
    "robustness": {
        "enabled": True,
        "segments": 3,
        "evaluate_top_n": 8,
        "reject_if_any_segment_pf_below": 1.0,
        "reject_if_any_segment_net_profit_below_or_equal": 0.0,
    },
    # Staged search per ticker/timeframe: initial random grid -> expanded grid
    # around the best suitable candidate -> refined grid around the top expanded candidate.
    "staged_search": {
        "enabled": True,
        "filters": dict(DEFAULT_FILTERS),
        "strong_filters": dict(STRONG_FILTERS),
        "require_strong_candidate": True,
        "expand_radius": 2.0,
        "refine_radius": 1.0,
        "time_budget_split": [0.5, 0.3, 0.2],
    },
}


def load_yaml_config(path: str = "tickers.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    if not _HAS_YAML:
        raise RuntimeError("Found tickers.yaml but PyYAML is not installed. Install with: python -m pip install pyyaml")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def load_json_config(path: str = "tickers.json") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg if isinstance(cfg, dict) else {}


def merge_with_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            block = merged[k]
            block.update(v)
            merged[k] = block
        else:
            merged[k] = v
    if merged.get("grid_constrained") is None:
        merged["grid_constrained"] = merged.get("grid") or DEFAULT_CONFIG["grid"]
    if merged.get("grid_expand") is None:
        merged["grid_expand"] = merged.get("grid") or DEFAULT_CONFIG["grid"]
    # grid_fallback intentionally left as None when not specified — absence means disabled
    return merged


def load_config() -> Dict[str, Any]:
    yaml_cfg = load_yaml_config("tickers.yaml")
    if yaml_cfg:
        return merge_with_defaults(yaml_cfg)
    json_cfg = load_json_config("tickers.json")
    if json_cfg:
        return merge_with_defaults(json_cfg)
    return json.loads(json.dumps(DEFAULT_CONFIG))


def discover_tsvs_auto() -> List[Dict[str, Any]]:
    data_dir = Path("data")
    tickers = []
    if not data_dir.exists():
        return tickers
    tf_suffix = re.compile(r"_(\d+[mhd])$", re.IGNORECASE)
    for p in sorted(data_dir.glob("*.tsv")):
        symbol = p.stem.split("_")[0]
        m = tf_suffix.search(p.stem)
        timeframe = m.group(1).lower() if m else "15m"
        tickers.append({"symbol": symbol, "timeframe": timeframe, "tsv": str(p), "tv_export": None, "parity_ok": False})
    return tickers


def _sanity_check_ticker(t: Dict[str, Any]) -> Tuple[bool, str]:
    symbol = t.get("symbol", "")
    tsv = t.get("tsv")
    if not tsv:
        return False, f"{symbol}: missing tsv path"
    if not Path(tsv).exists():
        return False, f"{symbol}: missing tsv file {tsv}"
    tf = str(t.get("timeframe", "15m")).strip().lower()
    if not re.fullmatch(r"\d+[mhd]", tf):
        return False, f"{symbol}: invalid timeframe '{tf}' (expected values like 15m or 30m)"
    try:
        candles = load_candles_from_csv(tsv)
    except Exception as e:
        return False, f"{symbol}: failed to load candles ({e})"
    if len(candles) < 50:
        return False, f"{symbol}: insufficient candles ({len(candles)})"
    return True, f"{symbol}: sanity ok ({len(candles)} candles)"


def _parity_gate_pass(t: Dict[str, Any], parity_cfg: Dict[str, Any]) -> Tuple[bool, str]:
    symbol = t.get("symbol", "")
    if bool(parity_cfg.get("require_tv_export", True)):
        tv = t.get("tv_export")
        if not tv:
            return False, f"{symbol}: tv_export missing"
        if not Path(tv).exists():
            return False, f"{symbol}: tv_export file missing ({tv})"
    if bool(parity_cfg.get("require_parity_ok", True)) and not bool(t.get("parity_ok", False)):
        return False, f"{symbol}: parity_ok is false (run parity and set parity_ok=true)"
    return True, f"{symbol}: parity gate passed"


def _write_progress_rows(out_path: Path, rows: List[List[Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "phase", "status", "evaluated", "elapsed_seconds", "top_score", "note"])
        for r in rows:
            w.writerow(r)


def _run_phase(
    phase_name: str,
    phase_tickers: List[Dict[str, Any]],
    grid: Dict[str, Any],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    workers = int(cfg.get("parallel_workers", 2))
    top_k = int(cfg.get("top_k_per_ticker", 5))
    time_budget = int(cfg.get("time_budget_seconds_per_ticker", 1800))
    search_mode = cfg.get("search_mode", "auto")
    n_samples = int(cfg.get("n_samples_per_ticker", cfg.get("n_samples", 1000)))
    seed = int(cfg.get("random_seed", 0))
    max_exhaustive = int(cfg.get("max_exhaustive", 150000))
    execution = cfg.get("execution", {}) or {}
    intrabar_paths = [execution.get("intrabar_path", "ohlc")]
    robustness = cfg.get("robustness", {}) or {}
    filters = (cfg.get("staged_search", {}) or {}).get("filters", None) or None

    args = []
    for t in phase_tickers:
        args.append((t, grid, intrabar_paths, top_k, time_budget, search_mode, n_samples, seed, max_exhaustive, execution, robustness, phase_name, filters))

    with Pool(workers) as pool:
        return pool.starmap(optimize_ticker, args)


def _run_staged_search(
    tickers: List[Dict[str, Any]],
    grid: Dict[str, Any],
    cfg: Dict[str, Any],
) -> List[List[Dict[str, Any]]]:
    """Run the staged search (initial random -> expanded -> refined) per ticker/timeframe.

    When ``staged_search.require_strong_candidate`` is enabled, tickers whose
    strong-candidate bar (positive net profit, PF >= 1.4, WR >= 45%, low
    drawdown) was not cleared within their budget get a second pass with the
    remaining tickers before the optimizer moves on — a ticker is only
    considered finished once a strong candidate has been found or every
    budgeted pass over it is exhausted.
    """
    workers = int(cfg.get("parallel_workers", 2))
    top_k = int(cfg.get("top_k_per_ticker", 5))
    time_budget = int(cfg.get("time_budget_seconds_per_ticker", 3600))
    n_samples = int(cfg.get("n_samples_per_ticker", cfg.get("n_samples", 1000)))
    seed = int(cfg.get("random_seed", 0))
    max_exhaustive = int(cfg.get("max_exhaustive", 150000))
    execution = cfg.get("execution", {}) or {}
    intrabar_paths = [execution.get("intrabar_path", "ohlc")]
    robustness = cfg.get("robustness", {}) or {}
    staged_cfg = cfg.get("staged_search", {}) or {}
    require_strong = bool(staged_cfg.get("require_strong_candidate", True))
    max_retries = int(staged_cfg.get("max_retries_per_ticker", 1)) if require_strong else 0

    results: Dict[int, List[Dict[str, Any]]] = {}
    pending = list(range(len(tickers)))
    attempt = 0

    def _found_strong(stage_list: List[Dict[str, Any]]) -> bool:
        return any(bool(r.get("strong_candidate_found")) for r in stage_list)

    while pending:
        attempt += 1
        if attempt > 1:
            # Resume files skip already-evaluated combos, so each extra pass
            # spends the ticker's budget exploring fresh parameter space.
            print(f"Strong candidate not found for {len(pending)} ticker/timeframe entries; re-running with fresh budget (pass {attempt})")
        args = []
        for i in pending:
            t = tickers[i]
            args.append((t, grid, intrabar_paths, top_k, time_budget, n_samples, seed, max_exhaustive, execution, robustness, staged_cfg))
        with Pool(workers) as pool:
            pass_results = pool.starmap(staged_search, args)
        still_pending: List[int] = []
        for i, stage_list in zip(pending, pass_results):
            results[i] = stage_list
            if require_strong and not _found_strong(stage_list) and attempt <= max_retries:
                still_pending.append(i)
        if not require_strong or not still_pending:
            break
        pending = still_pending

    return [results.get(i, []) for i in range(len(tickers))]


def _run_legacy_phases(
    gated_tickers: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    progress_rows: List[List[Any]],
) -> List[Dict[str, Any]]:
    """Original phased flow (constrained -> fallback -> expanded)."""
    print(f"Starting constrained optimization for {len(gated_tickers)} tickers")

    constrained_grid = cfg.get("grid_constrained") or cfg.get("grid") or {}
    constrained_results = _run_phase("constrained", gated_tickers, constrained_grid, cfg)

    # Tickers with no constrained candidates get a fallback grid run (if configured)
    fallback_grid = cfg.get("grid_fallback")
    fallback_results: List[Dict[str, Any]] = []
    fallback_symbols: set = set()
    if fallback_grid:
        fallback_candidates: List[Dict[str, Any]] = []
        for r in constrained_results:
            if not (r.get("top") or []):
                t_match = next((t for t in gated_tickers if t.get("symbol") == r.get("symbol")), None)
                if t_match is not None:
                    fallback_candidates.append(t_match)
        if fallback_candidates:
            print(f"No constrained matches for {[t['symbol'] for t in fallback_candidates]}; running fallback grid")
            fallback_results = _run_phase("fallback", fallback_candidates, fallback_grid, cfg)
            for r in fallback_results:
                symbol = r.get("symbol", "")
                fallback_symbols.add(symbol)
                top = r.get("top", []) or []
                top_score = top[0].get("score", "") if top else ""
                progress_rows.append([symbol, "fallback", "done", r.get("evaluated", 0), round(float(r.get("elapsed_seconds", 0.0)), 2), top_score, r.get("note", "")])

    expand_candidates: List[Dict[str, Any]] = []
    for r in constrained_results:
        symbol = r.get("symbol", "")
        top = r.get("top", []) or []
        top_score = top[0].get("score", "") if top else ""
        progress_rows.append([symbol, "constrained", "done", r.get("evaluated", 0), round(float(r.get("elapsed_seconds", 0.0)), 2), top_score, r.get("note", "")])
        # Advance to expand if constrained found candidates; fallback tickers are handled below
        if top and symbol not in fallback_symbols:
            t_match = next((t for t in gated_tickers if t.get("symbol") == symbol), None)
            if t_match is not None:
                expand_candidates.append(t_match)

    # Tickers that found candidates via fallback also advance to expand
    for r in fallback_results:
        symbol = r.get("symbol", "")
        if r.get("top") or []:
            t_match = next((t for t in gated_tickers if t.get("symbol") == symbol), None)
            if t_match is not None:
                expand_candidates.append(t_match)

    expanded_results: List[Dict[str, Any]] = []
    if expand_candidates:
        print(f"Starting expanded optimization for {len(expand_candidates)} tickers")
        expand_grid = cfg.get("grid_expand") or cfg.get("grid") or {}
        expanded_results = _run_phase("expanded", expand_candidates, expand_grid, cfg)
        for r in expanded_results:
            symbol = r.get("symbol", "")
            top = r.get("top", []) or []
            top_score = top[0].get("score", "") if top else ""
            progress_rows.append([symbol, "expanded", "done", r.get("evaluated", 0), round(float(r.get("elapsed_seconds", 0.0)), 2), top_score, r.get("note", "")])

    # prefer expanded > fallback > constrained
    final: Dict[str, Dict[str, Any]] = {}
    for r in constrained_results:
        final[r.get("symbol", "")] = r
    for r in fallback_results:
        final[r.get("symbol", "")] = r
    for r in expanded_results:
        final[r.get("symbol", "")] = r
    return list(final.values())


def main() -> None:
    cfg = load_config()
    tickers = cfg.get("tickers", []) or []
    if not tickers:
        tickers = discover_tsvs_auto()
    if not tickers:
        print("No tickers found in config and no TSVs discovered in data/. Exiting.")
        return

    parity_cfg = cfg.get("parity", {}) or {}
    progress_rows: List[List[Any]] = []

    gated_tickers: List[Dict[str, Any]] = []
    for t in tickers:
        symbol = t.get("symbol", "")
        ok, note = _sanity_check_ticker(t)
        if not ok:
            progress_rows.append([symbol, "sanity", "skipped", 0, 0, "", note])
            continue
        p_ok, p_note = _parity_gate_pass(t, parity_cfg)
        if not p_ok:
            progress_rows.append([symbol, "parity", "skipped", 0, 0, "", p_note])
            continue
        progress_rows.append([symbol, "parity", "ready", 0, 0, "", p_note])
        gated_tickers.append(t)

    progress_path = Path("optimizer_results") / "progress.csv"
    _write_progress_rows(progress_path, progress_rows)

    if not gated_tickers:
        print("No tickers passed sanity + parity gates. Exiting.")
        return

    start = time()
    staged_cfg = cfg.get("staged_search", {}) or {}
    if bool(staged_cfg.get("enabled", True)):
        base_grid = cfg.get("grid_constrained") or cfg.get("grid") or {}
        print(f"Starting staged search for {len(gated_tickers)} ticker/timeframe entries")
        stage_results = _run_staged_search(gated_tickers, base_grid, cfg)
        final_results: List[Dict[str, Any]] = []
        for stage_list in stage_results:
            final_results.append(stage_list[-1] if stage_list else {})
            for r in stage_list:
                symbol = r.get("symbol", "")
                top = r.get("top", []) or []
                top_score = top[0].get("score", "") if top else ""
                note = r.get("note", "")
                if r.get("strong_candidate_found"):
                    note = (note + "; " if note else "") + "strong candidate found"
                progress_rows.append([symbol, r.get("phase", ""), "done", r.get("evaluated", 0), round(float(r.get("elapsed_seconds", 0.0)), 2), top_score, note])
    else:
        final_results = _run_legacy_phases(gated_tickers, cfg, progress_rows)

    # pick final per symbol/timeframe entry
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for r in final_results:
        key = f"{r.get('symbol', '')}_{str(r.get('timeframe') or '').strip().lower()}"
        by_symbol[key] = r

    out = Path("optimizer_results")
    out.mkdir(exist_ok=True)
    agg_path = out / "best_presets.csv"
    with agg_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "phase", "score", "net_profit", "trade_count", "profit_factor", "max_drawdown_pct", "params", "report"]) 
        for symbol in sorted(by_symbol.keys()):
            r = by_symbol[symbol]
            top_list = r.get("top", []) or []
            report = r.get("report", "")
            phase = r.get("phase", "")
            if top_list:
                best = top_list[0]
                params = best.get("params", {})
                metrics = best.get("metrics", {})
                try:
                    params_str = json.dumps(params)
                except Exception:
                    params_str = str(params)
                w.writerow([
                    symbol,
                    phase,
                    best.get("score", ""),
                    metrics.get("net_profit", ""),
                    metrics.get("trade_count", ""),
                    metrics.get("profit_factor", ""),
                    metrics.get("max_drawdown_pct", ""),
                    params_str,
                    report,
                ])
            else:
                w.writerow([symbol, phase, "", "", "", "", "", "", report])

    _write_progress_rows(progress_path, progress_rows)
    print("Optimization complete. Results in optimizer_results/")
    print(f"Elapsed: {time() - start:.1f}s")


if __name__ == "__main__":
    main()
