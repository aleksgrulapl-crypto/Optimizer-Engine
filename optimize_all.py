"""
Orchestrator with parity gating and per-ticker staged optimization.

Flow:
1) Data sanity checks (timeframe file exists + non-empty candles)
2) Parity gate checks (tv_export exists and parity_ok=true when enabled)
3) Per-symbol preset checks across the available 15m/30m entries
4) Initial random grid search over a wide, loose base grid
5) Expanded grid around the best suitable candidate
6) Manual gating before moving to the next ticker
7) Robustness-filtered top candidates returned by worker
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

from optimizer_worker import DEFAULT_FILTERS, STRONG_FILTERS, build_neighborhood_grid, is_strong_candidate, optimize_ticker, staged_search
from presets import get_presets, normalize_timeframe

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
    # Staged search per ticker/timeframe: preset check -> initial random grid
    # -> expanded grid around the best suitable candidate.
    "staged_search": {
        "enabled": True,
        "filters": dict(DEFAULT_FILTERS),
        "strong_filters": dict(STRONG_FILTERS),
        "expand_radius": 2.0,
        "time_budget_split": [0.6, 0.4],
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


def _timeframe_sort_key(timeframe: Any) -> Tuple[int, str]:
    tf = str(timeframe or "").strip().lower()
    order = {"15m": 0, "30m": 1}
    return order.get(tf, 99), tf


def _group_tickers_by_symbol(tickers: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ticker in tickers:
        symbol = str(ticker.get("symbol", "")).strip().upper()
        grouped.setdefault(symbol, []).append(ticker)
    return [
        (symbol, sorted(entries, key=lambda item: _timeframe_sort_key(item.get("timeframe"))))
        for symbol, entries in grouped.items()
    ]


def _mark_suitable(result: Dict[str, Any], strong_filters: Dict[str, Any]) -> Dict[str, Any]:
    top = result.get("top", []) or []
    result["strong_candidate_found"] = any(
        is_strong_candidate(candidate.get("metrics", {}) or {}, strong_filters) for candidate in top
    )
    return result


def _append_progress_row(progress_rows: List[List[Any]], result: Dict[str, Any]) -> None:
    symbol = result.get("symbol", "")
    top = result.get("top", []) or []
    top_score = top[0].get("score", "") if top else ""
    note = result.get("note", "")
    if result.get("strong_candidate_found"):
        note = (note + "; " if note else "") + "suitable candidate found"
    progress_rows.append([
        symbol,
        result.get("phase", ""),
        "done",
        result.get("evaluated", 0),
        round(float(result.get("elapsed_seconds", 0.0)), 2),
        top_score,
        note,
    ])


def _build_preset_grid(symbol: str, timeframe: Any) -> Dict[str, List[Any]]:
    preset = get_presets(symbol, normalize_timeframe(str(timeframe or "15m")))
    return {key: [value] for key, value in preset.items()}


def _prompt_yes_no(message: str) -> bool:
    while True:
        reply = input(f"{message} [y/n]: ").strip().lower()
        if reply in {"y", "yes"}:
            return True
        if reply in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _print_symbol_summary(symbol: str, results: List[Dict[str, Any]]) -> None:
    print(f"\n{symbol} summary:")
    for result in results:
        timeframe = str(result.get("timeframe", "")).strip().lower()
        phase = result.get("phase", "")
        top = result.get("top", []) or []
        if not top:
            print(f"  - {timeframe or 'n/a'} [{phase}]: no suitable candidates")
            continue
        best = top[0]
        metrics = best.get("metrics", {}) or {}
        print(
            "  - "
            f"{timeframe or 'n/a'} [{phase}]: "
            f"net={metrics.get('net_profit', 0.0):.2f}, "
            f"pf={metrics.get('profit_factor', 0.0):.2f}, "
            f"wr={float(metrics.get('win_rate', 0.0)) * 100:.1f}%, "
            f"dd={float(metrics.get('max_drawdown_pct', 0.0)) * 100:.2f}%"
        )


def _run_preset_phase(ticker: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(ticker.get("symbol", "")).strip().upper()
    timeframe = ticker.get("timeframe")
    strong_filters = {**STRONG_FILTERS, **((cfg.get("staged_search", {}) or {}).get("strong_filters", {}) or {})}
    try:
        preset_grid = _build_preset_grid(symbol, timeframe)
    except Exception as exc:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "phase": "preset",
            "top": [],
            "evaluated": 0,
            "elapsed_seconds": 0.0,
            "note": f"preset skipped ({exc})",
            "strong_candidate_found": False,
        }

    result = optimize_ticker(
        ticker,
        preset_grid,
        [str((cfg.get("execution", {}) or {}).get("intrabar_path", "ohlc"))],
        top_k=int(cfg.get("top_k_per_ticker", 5)),
        time_budget=int(cfg.get("time_budget_seconds_per_ticker", 3600)),
        search_mode="auto",
        n_samples=1,
        seed=int(cfg.get("random_seed", 0)),
        max_exhaustive=int(cfg.get("max_exhaustive", 150000)),
        execution=(cfg.get("execution", {}) or {}),
        robustness=(cfg.get("robustness", {}) or {}),
        phase="preset",
        filters=((cfg.get("staged_search", {}) or {}).get("filters", None) or None),
    )
    return _mark_suitable(result, strong_filters)


def _run_staged_cycle(ticker: Dict[str, Any], grid: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    n_samples = int(cfg.get("n_samples_per_ticker", cfg.get("n_samples", 1000)))
    staged_cfg = dict(cfg.get("staged_search", {}) or {})
    staged_cfg["expand_samples"] = max(n_samples, int(staged_cfg.get("expand_samples", 2 * n_samples)))
    return staged_search(
        ticker,
        grid,
        [str((cfg.get("execution", {}) or {}).get("intrabar_path", "ohlc"))],
        top_k=int(cfg.get("top_k_per_ticker", 5)),
        time_budget=int(cfg.get("time_budget_seconds_per_ticker", 3600)),
        n_samples=n_samples,
        seed=int(cfg.get("random_seed", 0)),
        max_exhaustive=int(cfg.get("max_exhaustive", 150000)),
        execution=(cfg.get("execution", {}) or {}),
        robustness=(cfg.get("robustness", {}) or {}),
        staged_cfg=staged_cfg,
    )


def _run_expanded_cycle(
    ticker: Dict[str, Any],
    grid: Dict[str, Any],
    cfg: Dict[str, Any],
    cycle_index: int,
    previous_result: Dict[str, Any],
) -> Dict[str, Any]:
    n_samples = int(cfg.get("n_samples_per_ticker", cfg.get("n_samples", 1000)))
    staged_cfg = dict(cfg.get("staged_search", {}) or {})
    strong_filters = {**STRONG_FILTERS, **(staged_cfg.get("strong_filters", {}) or {})}
    expand_radius = float(staged_cfg.get("expand_radius", 2.0)) * max(2, cycle_index + 1)
    expand_samples = max(n_samples, int(staged_cfg.get("expand_samples", 2 * n_samples))) * max(2, cycle_index + 1)
    center = {}
    top = (previous_result or {}).get("top", []) or []
    if top:
        center = top[0].get("params", {}) or {}
    if not center:
        try:
            center = get_presets(
                str(ticker.get("symbol", "")).strip().upper(),
                normalize_timeframe(str(ticker.get("timeframe", "15m"))),
            )
        except Exception:
            center = {}
    if not center:
        return {
            "symbol": ticker.get("symbol", ""),
            "timeframe": ticker.get("timeframe"),
            "phase": "expanded",
            "top": [],
            "evaluated": 0,
            "elapsed_seconds": 0.0,
            "note": "expanded search skipped (no center parameters available)",
            "strong_candidate_found": False,
        }

    result = optimize_ticker(
        ticker,
        build_neighborhood_grid(grid, center, expand_radius),
        [str((cfg.get("execution", {}) or {}).get("intrabar_path", "ohlc"))],
        top_k=int(cfg.get("top_k_per_ticker", 5)),
        time_budget=int(cfg.get("time_budget_seconds_per_ticker", 3600)),
        search_mode="sample",
        n_samples=expand_samples,
        seed=int(cfg.get("random_seed", 0)) + (cycle_index * 100003),
        max_exhaustive=int(cfg.get("max_exhaustive", 150000)),
        execution=(cfg.get("execution", {}) or {}),
        robustness=(cfg.get("robustness", {}) or {}),
        phase="expanded",
        filters=(staged_cfg.get("filters", None) or None),
    )
    return _mark_suitable(result, strong_filters)


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
    """Run the staged search (initial random -> expanded) per ticker/timeframe."""
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
    args = []
    for ticker in tickers:
        args.append((ticker, grid, intrabar_paths, top_k, time_budget, n_samples, seed, max_exhaustive, execution, robustness, staged_cfg))
    with Pool(workers) as pool:
        return pool.starmap(staged_search, args)


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
        final_results: List[Dict[str, Any]] = []
        symbol_groups = _group_tickers_by_symbol(gated_tickers)
        print(f"Starting staged search for {len(symbol_groups)} ticker(s)")
        for symbol, symbol_tickers in symbol_groups:
            latest_by_timeframe: Dict[str, Dict[str, Any]] = {}
            print(f"\nStarting {symbol} with {len(symbol_tickers)} timeframe preset(s)")
            for ticker in symbol_tickers:
                preset_result = _run_preset_phase(ticker, cfg)
                latest_by_timeframe[str(ticker.get('timeframe', '')).strip().lower()] = preset_result
                _append_progress_row(progress_rows, preset_result)

            cycle_index = 0
            while True:
                print(f"{symbol}: staged cycle {cycle_index + 1}")
                for ticker in symbol_tickers:
                    timeframe_key = str(ticker.get("timeframe", "")).strip().lower()
                    if cycle_index == 0:
                        stage_list = _run_staged_cycle(ticker, base_grid, cfg)
                        for result in stage_list:
                            _append_progress_row(progress_rows, result)
                        final_result = stage_list[-1] if stage_list else {}
                    else:
                        final_result = _run_expanded_cycle(
                            ticker,
                            base_grid,
                            cfg,
                            cycle_index,
                            latest_by_timeframe.get(timeframe_key, {}),
                        )
                        _append_progress_row(progress_rows, final_result)
                    if final_result:
                        latest_by_timeframe[timeframe_key] = final_result

                summarized_results = [latest_by_timeframe[key] for key in sorted(latest_by_timeframe.keys(), key=_timeframe_sort_key)]
                _print_symbol_summary(symbol, summarized_results)
                suitable_found = any(bool(result.get("strong_candidate_found")) for result in summarized_results)
                print(f"{symbol}: optimizer {'found' if suitable_found else 'did not find'} a suitable candidate under the current acceptance filters.")
                if _prompt_yes_no(f"{symbol}: are the current candidates suitable"):
                    if _prompt_yes_no(f"{symbol}: move to the next ticker"):
                        final_results.extend(summarized_results)
                        break
                    print(f"{symbol}: current candidates kept; widening the expanded search before asking again.")
                cycle_index += 1
                print(f"{symbol}: continuing with a wider expanded search.")
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
