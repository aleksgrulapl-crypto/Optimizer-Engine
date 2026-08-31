"""
Single-ticker refinement optimizer.

Use this after optimize_all.py to fine-tune one ticker around its presets.py params.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from optimize_all import (
    discover_tsvs_auto,
    load_config,
    _parity_gate_pass,
    _sanity_check_ticker,
)
from optimize_all import _write_progress_rows as write_progress_rows
from optimizer_worker import optimize_ticker
from presets import get_presets, normalize_timeframe


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


def _find_tickers(cfg: Dict[str, Any], symbol: str) -> List[Dict[str, Any]]:
    tickers = list(cfg.get("tickers", []) or [])
    discovered = discover_tsvs_auto()
    by_timeframe: Dict[str, Dict[str, Any]] = {}
    for t in tickers:
        if str(t.get("symbol", "")).strip().upper() == symbol.upper():
            tf = str(t.get("timeframe", "")).strip().lower()
            if tf:
                by_timeframe[tf] = t
    for t in discovered:
        if str(t.get("symbol", "")).strip().upper() != symbol.upper():
            continue
        tf = str(t.get("timeframe", "")).strip().lower()
        if tf not in by_timeframe:
            by_timeframe[tf] = t
    return [by_timeframe[tf] for tf in sorted(by_timeframe.keys())]


def _phase_with_timeframe(phase: str, timeframe: str) -> str:
    return f"{phase}_{timeframe.lower()}"


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
    base_phase = args.phase.strip() or "refine"
    cfg = load_config()
    ticker_cfgs = _find_tickers(cfg, symbol)
    if not ticker_cfgs:
        print(f"{symbol}: not found in tickers config or auto-discovered TSVs.")
        return

    parity_cfg = cfg.get("parity", {}) or {}
    top_k = int(args.top_k) if args.top_k is not None else int(cfg.get("top_k_per_ticker", 5))
    time_budget = int(args.time_budget) if args.time_budget is not None else int(cfg.get("time_budget_seconds_per_ticker", 1800))
    progress_path = Path("optimizer_results") / "progress_single.csv"
    progress_rows = []

    for ticker_cfg in ticker_cfgs:
        ok, note = _sanity_check_ticker(ticker_cfg)
        if not ok:
            print(note)
            progress_rows.append([symbol, _phase_with_timeframe(base_phase, str(ticker_cfg.get("timeframe", "15m"))), "skipped", 0, 0, "", note])
            continue
        p_ok, p_note = _parity_gate_pass(ticker_cfg, parity_cfg)
        if not p_ok:
            print(p_note)
            progress_rows.append([symbol, _phase_with_timeframe(base_phase, str(ticker_cfg.get("timeframe", "15m"))), "skipped", 0, 0, "", p_note])
            continue

        timeframe = normalize_timeframe(str(ticker_cfg.get("timeframe", "15m")))
        best_params = get_presets(symbol, timeframe)
        print(f"{symbol}: using presets.py defaults for {timeframe}")
        grid = _build_refinement_grid(best_params)
        phase = _phase_with_timeframe(base_phase, timeframe)

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
            phase=phase,
        )

        top = result.get("top", []) or []
        top_score = top[0].get("score", "") if top else ""
        progress_rows.append([
            symbol,
            result.get("phase", phase),
            "done",
            result.get("evaluated", 0),
            round(float(result.get("elapsed_seconds", 0.0)), 2),
            top_score,
            result.get("note", ""),
        ])
        print(f"Single-ticker refinement complete for {symbol} {timeframe}.")
        print(f"CSV: {result.get('csv')}")
        print(f"Report: {result.get('report')}")

    write_progress_rows(progress_path, progress_rows)
    print(f"Progress: {progress_path}")


if __name__ == "__main__":
    main()
