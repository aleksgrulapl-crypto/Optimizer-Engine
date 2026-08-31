"""
Single-ticker refinement optimizer.

Use this after optimize_all.py to fine-tune one ticker around its best known params.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

from optimize_all import (
    discover_tsvs_auto,
    load_config,
    _parity_gate_pass,
    _sanity_check_ticker,
)
from optimize_all import _write_progress_rows as write_progress_rows
from optimizer_worker import optimize_ticker
from presets import get_presets


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return default


def _frange(start: float, stop: float, step: float) -> list:
    out = []
    v = float(start)
    eps = 1e-12
    while v <= stop + eps:
        out.append(round(v, 10))
        v += float(step)
    return out


def _load_best_params_from_aggregate(symbol: str, path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("symbol", "")).strip().upper() != symbol.upper():
                continue
            raw = row.get("params")
            if not raw:
                continue
            try:
                params = json.loads(raw)
                if isinstance(params, dict):
                    return params
            except Exception:
                continue
    return None


def _build_refinement_grid(base: Dict[str, Any]) -> Dict[str, Any]:
    st_mult = _safe_float(base.get("stMultiplier"), 2.0)
    st_period = _safe_int(base.get("stPeriod"), 10)
    atr_sl = _safe_float(base.get("atrSLmult"), 1.2)
    atr_tp = _safe_float(base.get("atrTPmult"), 3.0)
    ema_len = _safe_int(base.get("emaLen"), 50)

    return {
        "stMultiplier": _frange(max(0.2, st_mult - 0.4), st_mult + 0.4, 0.1),
        "stPeriod": list(range(max(2, st_period - 3), st_period + 4)),
        "atrSLmult": _frange(max(0.2, atr_sl - 0.3), atr_sl + 0.3, 0.1),
        "atrTPmult": _frange(max(0.5, atr_tp - 0.8), atr_tp + 0.8, 0.2),
        "emaLen": list(range(max(5, ema_len - 30), ema_len + 31, 10)),
    }


def _find_ticker(cfg: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
    tickers = cfg.get("tickers", []) or []
    if not tickers:
        tickers = discover_tsvs_auto()
    for t in tickers:
        if str(t.get("symbol", "")).strip().upper() == symbol.upper():
            return t
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine a single ticker around its best known params."
    )
    parser.add_argument("--symbol", required=True, help="Ticker symbol (e.g. NVDA)")
    parser.add_argument(
        "--phase",
        default="refine",
        help="Output phase label for report/csv filenames (default: refine)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top candidates to keep",
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=None,
        help="Override per-ticker time budget in seconds",
    )
    args = parser.parse_args()

    symbol = args.symbol.strip().upper()
    cfg = load_config()
    ticker_cfg = _find_ticker(cfg, symbol)
    if not ticker_cfg:
        print(f"{symbol}: not found in tickers config or auto-discovered TSVs.")
        return

    ok, note = _sanity_check_ticker(ticker_cfg)
    if not ok:
        print(note)
        return
    p_ok, p_note = _parity_gate_pass(ticker_cfg, cfg.get("parity", {}) or {})
    if not p_ok:
        print(p_note)
        return

    aggregate = Path("optimizer_results") / "best_presets.csv"
    best_params = _load_best_params_from_aggregate(symbol, aggregate)
    if not best_params:
        best_params = get_presets(symbol)
        print(f"{symbol}: best_presets.csv entry not found, using presets.py defaults.")
    else:
        print(f"{symbol}: using best params from optimizer_results/best_presets.csv")

    grid = _build_refinement_grid(best_params)
    top_k = int(args.top_k) if args.top_k is not None else int(cfg.get("top_k_per_ticker", 5))
    time_budget = int(args.time_budget) if args.time_budget is not None else int(cfg.get("time_budget_seconds_per_ticker", 1800))

    result = optimize_ticker(
        ticker_cfg,
        grid,
        [str((cfg.get("execution", {}) or {}).get("intrabar_path", "ohlc"))],
        top_k=top_k,
        time_budget=time_budget,
        search_mode=str(cfg.get("search_mode", "auto")),
        n_samples=int(cfg.get("n_samples_per_ticker", cfg.get("n_samples", 1000))),
        seed=int(cfg.get("random_seed", 0)),
        max_exhaustive=int(cfg.get("max_exhaustive", 150000)),
        execution=(cfg.get("execution", {}) or {}),
        robustness=(cfg.get("robustness", {}) or {}),
        phase=args.phase.strip() or "refine",
    )

    progress_path = Path("optimizer_results") / "progress_single.csv"
    top = result.get("top", []) or []
    top_score = top[0].get("score", "") if top else ""
    write_progress_rows(
        progress_path,
        [[
            symbol,
            result.get("phase", "refine"),
            "done",
            result.get("evaluated", 0),
            round(float(result.get("elapsed_seconds", 0.0)), 2),
            top_score,
            result.get("note", ""),
        ]],
    )

    print(f"Single-ticker refinement complete for {symbol}.")
    print(f"CSV: {result.get('csv')}")
    print(f"Report: {result.get('report')}")
    print(f"Progress: {progress_path}")


if __name__ == "__main__":
    main()
