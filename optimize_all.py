# optimize_all.py
"""
Orchestrator with run tracking and resume support.

- Reads tickers.yaml (PyYAML) or tickers.json.
- Passes search_mode / sampling settings into workers.
- Writes per-ticker completed_runs JSONL for resume.
- Writes optimizer_results/progress.csv with per-ticker progress.
"""

import json
import csv
from multiprocessing import Pool
from pathlib import Path
from time import time
from typing import Dict, Any

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
    "parallel_workers": 4,
    "intrabar_paths": ["ohlc", "olhc"],
    "top_k_per_ticker": 3,
    # default per-ticker budget: 1 hour
    "time_budget_seconds_per_ticker": 3600,
    # sampling/search defaults
    "search_mode": "auto",
    "n_samples_per_ticker": 1000,
    "random_seed": 0,
    "max_exhaustive": 200000
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
    merged = DEFAULT_CONFIG.copy()
    for k, v in cfg.items():
        if k == "grid" and isinstance(v, dict):
            grid = merged.get("grid", {}).copy()
            grid.update(v)
            merged["grid"] = grid
        else:
            merged[k] = v
    if "intrabar_paths" not in merged or not isinstance(merged["intrabar_paths"], list):
        merged["intrabar_paths"] = DEFAULT_CONFIG["intrabar_paths"]
    return merged


def discover_tsvs_auto() -> list:
    data_dir = Path("data")
    tickers = []
    if not data_dir.exists():
        return tickers
    for p in sorted(data_dir.glob("*_15m.tsv")):
        symbol = p.stem.split("_")[0]
        tickers.append({"symbol": symbol, "timeframe": "15m", "tsv": str(p), "tv_export": None})
    if not tickers:
        for p in sorted(data_dir.glob("*.tsv")):
            symbol = p.stem.split("_")[0]
            tickers.append({"symbol": symbol, "timeframe": None, "tsv": str(p), "tv_export": None})
    return tickers


def load_config() -> Dict[str, Any]:
    yaml_cfg = load_yaml_config("tickers.yaml")
    if yaml_cfg:
        return merge_with_defaults(yaml_cfg)
    json_cfg = load_json_config("tickers.json")
    if json_cfg:
        return merge_with_defaults(json_cfg)
    return DEFAULT_CONFIG.copy()


def _write_progress_row(out_path: Path, rows: list):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "status", "evaluated", "elapsed_seconds", "top_score"])
        for r in rows:
            w.writerow(r)


def main():
    cfg = load_config()
    tickers = cfg.get("tickers", []) or []
    grid = cfg.get("grid", {}) or {}
    workers = int(cfg.get("parallel_workers", 4))
    intrabar_paths = cfg.get("intrabar_paths", ["ohlc", "olhc"])
    top_k = int(cfg.get("top_k_per_ticker", 3))
    time_budget = int(cfg.get("time_budget_seconds_per_ticker", 3600))

    search_mode = cfg.get("search_mode", "auto")
    n_samples = int(cfg.get("n_samples_per_ticker", cfg.get("n_samples", 1000)))
    seed = int(cfg.get("random_seed", 0))
    max_exhaustive = int(cfg.get("max_exhaustive", 200000))

    if not tickers:
        tickers = discover_tsvs_auto()

    if not tickers:
        print("No tickers found in config and no TSVs discovered in data/. Exiting.")
        return

    print(f"Starting optimization for {len(tickers)} tickers with {workers} workers")
    start = time()

    args = []
    for t in tickers:
        args.append((t, grid, intrabar_paths, top_k, time_budget, search_mode, n_samples, seed, max_exhaustive))

    # Run workers in parallel
    with Pool(workers) as pool:
        results = pool.starmap(optimize_ticker, args)

    # Aggregate results into best_presets.csv
    out = Path("optimizer_results")
    out.mkdir(exist_ok=True)
    agg_path = out / "best_presets.csv"
    with agg_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "score", "net_profit", "trade_count", "profit_factor", "params", "report"])
        for r in results:
            symbol = r.get("symbol", "")
            top_list = r.get("top", []) or []
            report = r.get("report", "")
            if top_list:
                best = top_list[0]
                params = best.get("params", {})
                metrics = best.get("metrics", {})
                score = best.get("score", "")
                net_profit = metrics.get("net_profit", "")
                trade_count = metrics.get("trade_count", "")
                profit_factor = metrics.get("profit_factor", "")
                try:
                    params_str = json.dumps(params)
                except Exception:
                    params_str = str(params)
                w.writerow([symbol, score, net_profit, trade_count, profit_factor, params_str, report])
            else:
                w.writerow([symbol, "", "", "", "", "", report])

    print("Optimization complete. Results in optimizer_results/")
    print(f"Elapsed: {time() - start:.1f}s")


if __name__ == "__main__":
    main()
