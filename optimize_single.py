"""
Single-ticker refinement optimizer.

Use this after optimize_all.py to fine-tune one ticker around its presets.py params.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from optimize_all import (
    STRATEGY_LABELS,
    apply_strategy,
    discover_tsvs_auto,
    load_config,
    resolve_strategy,
    _parity_gate_pass,
    _sanity_check_ticker,
)
from optimize_all import _write_progress_rows as write_progress_rows
from optimizer_worker import DEFAULT_STRATEGY, derive_seed, optimize_ticker, strategy_dir
from presets import get_presets, normalize_timeframe

REFINEMENT_FILTERS = {
    "min_win_rate": 0.60,
    "min_profit_factor": 1.8,
    "min_net_profit": 0.0,
    "min_trades": 100,
    "max_drawdown_pct": 0.20,
}


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
    max_candle = _safe_float(base.get("maxEntryCandleATR"), 1.5)

    return {
        "stMultiplier": _frange(max(0.2, st_mult - 1.0), st_mult + 1.0, 0.1),
        "stPeriod": list(range(max(2, st_period - 1), st_period + 2)),
        "atrSLmult": _frange(max(0.2, atr_sl - 0.5), atr_sl + 0.5, 0.1),
        "atrTPmult": _frange(max(0.2, atr_tp - 0.5), atr_tp + 0.5, 0.1),
        "emaLen": list(range(max(5, ema_len - 10), ema_len + 11)),
        "maxEntryCandleATR": _frange(max(0.1, max_candle - 2.0), max_candle + 2.0, 0.1),
    }


def _build_daybox_refinement_grid(base: Dict[str, Any]) -> Dict[str, Any]:
    from backtest_daybox import DAYBOX_DEFAULTS

    def value(name: str) -> float:
        return _safe_float(base.get(name), DAYBOX_DEFAULTS[name])

    retest = value("retestTolerancePct")
    rejection = value("rejectionTolerancePct")
    rr = value("riskReward")
    buffer = value("slBufferPct")
    return {
        "retestTolerancePct": _frange(max(0.01, retest - 0.10), retest + 0.10, 0.01),
        "rejectionTolerancePct": _frange(max(0.01, rejection - 0.10), rejection + 0.10, 0.01),
        "riskReward": _frange(max(0.5, rr - 0.5), rr + 0.5, 0.1),
        "slBufferPct": _frange(max(0.01, buffer - 0.10), buffer + 0.10, 0.01),
        "enableBreakouts": [bool(base.get("enableBreakouts", True))],
        "enableRejections": [bool(base.get("enableRejections", True))],
    }


def _daybox_best_params(symbol: str, timeframe: str) -> Dict[str, Any]:
    """Best Day Box params from a previous optimize_all run, else {} (defaults)."""
    path = strategy_dir("optimizer_results", "daybox") / "best_presets.csv"
    if not path.exists():
        return {}
    key = f"{symbol}_{timeframe}".lower()
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("symbol", "")).lower() == key:
                try:
                    return json.loads(row.get("params") or "{}")
                except ValueError:
                    return {}
    return {}


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
    parser.add_argument(
        "--strategy",
        choices=("supertrend", "daybox"),
        default=None,
        help="Strategy to refine (prompted when omitted in a terminal)",
    )
    args = parser.parse_args()

    strategy = resolve_strategy(args.strategy)
    symbol = args.symbol.strip().upper()
    base_phase = args.phase.strip() or "refine"
    cfg = apply_strategy(load_config(), strategy)
    print(f"Strategy: {STRATEGY_LABELS[strategy]}")
    ticker_cfgs = _find_tickers(cfg, symbol)
    if not ticker_cfgs:
        print(f"{symbol}: not found in tickers config or auto-discovered TSVs.")
        return

    parity_cfg = cfg.get("parity", {}) or {}
    top_k = int(args.top_k) if args.top_k is not None else int(cfg.get("top_k_per_ticker", 5))
    time_budget = int(args.time_budget) if args.time_budget is not None else int(cfg.get("time_budget_seconds_per_ticker", 1800))
    progress_path = strategy_dir("optimizer_results", strategy) / "progress_single.csv"
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
        if strategy == "daybox":
            best_params = _daybox_best_params(symbol, str(ticker_cfg.get("timeframe", "15m")).strip().lower())
            print(f"{symbol}: refining around "
                  f"{'previous Day Box best' if best_params else 'Day Box defaults'} for {timeframe}")
            grid = _build_daybox_refinement_grid(best_params)
        else:
            best_params = get_presets(symbol, timeframe)
            print(f"{symbol}: using presets.py defaults for {timeframe}")
            grid = _build_refinement_grid(best_params)
        # The worker namespaces outputs per timeframe, so the phase stays bare.
        phase = base_phase

        result = optimize_ticker(
            ticker_cfg,
            grid,
            [str((cfg.get("execution", {}) or {}).get("intrabar_path", "ohlc"))],
            top_k=top_k,
            time_budget=time_budget,
            # Checkpointed random-order walk over the whole refinement grid.
            search_mode="randomized",
            seed=derive_seed(int(cfg.get("random_seed", 0)), symbol, timeframe),
            execution=(cfg.get("execution", {}) or {}),
            robustness=(cfg.get("robustness", {}) or {}),
            phase=phase,
            filters=REFINEMENT_FILTERS,
            strategy=strategy,
            fixed_params=(cfg.get("strategy_params", None) or None),
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
