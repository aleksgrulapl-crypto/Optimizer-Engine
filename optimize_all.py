"""
Orchestrator with parity gating and phased optimization.

Flow:
1) Data sanity checks (15m file exists + non-empty candles)
2) Parity gate checks (tv_export exists and parity_ok=true when enabled)
3) Constrained optimization
4) Expanded optimization
5) Robustness-filtered top candidates returned by worker
"""

import json
import csv
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

from optimizer_worker import optimize_ticker

DEFAULT_CONFIG: Dict[str, Any] = {
    "tickers": [],
    "grid": {
        "stMultiplier": [1.6, 2.0, 2.4],
        "stPeriod": [6, 8, 10],
        "atrSLmult": [1.0, 1.2],
        "atrTPmult": [2.0, 3.0, 4.0],
        "emaLen": [20, 50]
    },
    "grid_constrained": None,
    "grid_expand": None,
    "parallel_workers": 2,
    "intrabar_paths": ["ohlc"],
    "top_k_per_ticker": 5,
    "time_budget_seconds_per_ticker": 1800,
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
    for p in sorted(data_dir.glob("*_15m.tsv")):
        symbol = p.stem.split("_")[0]
        tickers.append({"symbol": symbol, "timeframe": "15m", "tsv": str(p), "tv_export": None, "parity_ok": False})
    if not tickers:
        for p in sorted(data_dir.glob("*.tsv")):
            symbol = p.stem.split("_")[0]
            tickers.append({"symbol": symbol, "timeframe": "15m", "tsv": str(p), "tv_export": None, "parity_ok": False})
    return tickers


def _sanity_check_ticker(t: Dict[str, Any]) -> Tuple[bool, str]:
    symbol = t.get("symbol", "")
    tsv = t.get("tsv")
    if not tsv:
        return False, f"{symbol}: missing tsv path"
    if not Path(tsv).exists():
        return False, f"{symbol}: missing tsv file {tsv}"
    tf = str(t.get("timeframe", "15m")).lower()
    if tf != "15m":
        return False, f"{symbol}: timeframe must be 15m"
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

    args = []
    for t in phase_tickers:
        args.append((t, grid, intrabar_paths, top_k, time_budget, search_mode, n_samples, seed, max_exhaustive, execution, robustness, phase_name))

    with Pool(workers) as pool:
        return pool.starmap(optimize_ticker, args)


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

    print(f"Starting constrained optimization for {len(gated_tickers)} tickers")
    start = time()

    constrained_grid = cfg.get("grid_constrained") or cfg.get("grid") or {}
    constrained_results = _run_phase("constrained", gated_tickers, constrained_grid, cfg)

    expand_candidates: List[Dict[str, Any]] = []
    for r in constrained_results:
        symbol = r.get("symbol", "")
        top = r.get("top", []) or []
        top_score = top[0].get("score", "") if top else ""
        progress_rows.append([symbol, "constrained", "done", r.get("evaluated", 0), round(float(r.get("elapsed_seconds", 0.0)), 2), top_score, r.get("note", "")])
        if top:
            # only expand symbols with at least one constrained candidate
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

    # pick final per symbol: prefer expanded if present, otherwise constrained
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for r in constrained_results:
        by_symbol[r.get("symbol", "")] = r
    for r in expanded_results:
        by_symbol[r.get("symbol", "")] = r

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
