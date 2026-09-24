"""Run a resumable randomized optimization grid for every ticker and timeframe."""

import csv
import json
import os
import re
import sys
from pathlib import Path
from time import sleep, time
from typing import Any, Dict, List, Tuple

from data_loader import load_candles_from_csv
from optimizer_worker import (
    DEFAULT_FILTERS,
    STRONG_FILTERS,
    derive_seed,
    is_strong_candidate,
    optimize_ticker,
)

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    yaml = None  # type: ignore
    _HAS_YAML = False


DEFAULT_CONFIG: Dict[str, Any] = {
    "tickers": [],
    "grid": {
        "stMultiplier": {"start": 2.0, "stop": 10.0, "step": 0.1},
        "stPeriod": {"start": 5, "stop": 18, "step": 1},
        "atrSLmult": {"start": 1.0, "stop": 3.0, "step": 0.1},
        "atrTPmult": {"start": 1.2, "stop": 5.0, "step": 0.1},
        "emaLen": {"start": 40, "stop": 240, "step": 1},
    },
    "intrabar_paths": ["ohlc"],
    "top_k_per_ticker": 10,
    "time_budget_seconds_per_ticker": 3600,
    "random_seed": 0,
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
    "optimization": {
        "filters": dict(DEFAULT_FILTERS),
        "strong_filters": dict(STRONG_FILTERS),
        "confirm_continue_every_cycles": 3,
    },
    "robustness": {
        "enabled": True,
        "segments": 3,
        "evaluate_top_n": 10,
        "reject_if_any_segment_pf_below": 1.0,
        "reject_if_any_segment_net_profit_below_or_equal": 0.0,
    },
}


def load_yaml_config(path: str = "tickers.yaml") -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    if not _HAS_YAML:
        raise RuntimeError(
            "Found tickers.yaml but PyYAML is not installed. "
            "Install with: python -m pip install pyyaml"
        )
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return config if isinstance(config, dict) else {}


def load_json_config(path: str = "tickers.json") -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config if isinstance(config, dict) else {}


def merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def load_config() -> Dict[str, Any]:
    yaml_config = load_yaml_config("tickers.yaml")
    if yaml_config:
        return merge_with_defaults(yaml_config)
    json_config = load_json_config("tickers.json")
    if json_config:
        return merge_with_defaults(json_config)
    return json.loads(json.dumps(DEFAULT_CONFIG))


def discover_tsvs_auto() -> List[Dict[str, Any]]:
    data_dir = Path("data")
    tickers = []
    if not data_dir.exists():
        return tickers
    timeframe_suffix = re.compile(r"_(\d+[mhd])$", re.IGNORECASE)
    for path in sorted(data_dir.glob("*.tsv")):
        symbol = path.stem.split("_")[0]
        match = timeframe_suffix.search(path.stem)
        timeframe = match.group(1).lower() if match else "15m"
        tickers.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "tsv": str(path),
            "tv_export": None,
            "parity_ok": False,
            "Status": "Incomplete",
        })
    return tickers


def _sanity_check_ticker(ticker: Dict[str, Any]) -> Tuple[bool, str]:
    symbol = ticker.get("symbol", "")
    tsv = ticker.get("tsv")
    if not tsv:
        return False, f"{symbol}: missing tsv path"
    if not Path(tsv).exists():
        return False, f"{symbol}: missing tsv file {tsv}"
    timeframe = str(ticker.get("timeframe", "15m")).strip().lower()
    if not re.fullmatch(r"\d+[mhd]", timeframe):
        return False, f"{symbol}: invalid timeframe '{timeframe}' (expected values like 15m, 30m, or 60m)"
    try:
        candles = load_candles_from_csv(tsv)
    except Exception as exc:
        return False, f"{symbol}: failed to load candles ({exc})"
    if len(candles) < 50:
        return False, f"{symbol}: insufficient candles ({len(candles)})"
    return True, f"{symbol}: sanity ok ({len(candles)} candles)"


def _parity_gate_pass(
    ticker: Dict[str, Any],
    parity_config: Dict[str, Any],
) -> Tuple[bool, str]:
    symbol = ticker.get("symbol", "")
    if bool(parity_config.get("require_tv_export", True)):
        tv_export = ticker.get("tv_export")
        if not tv_export:
            return False, f"{symbol}: tv_export missing"
        if not Path(tv_export).exists():
            return False, f"{symbol}: tv_export file missing ({tv_export})"
    if bool(parity_config.get("require_parity_ok", True)) and not bool(
        ticker.get("parity_ok", False)
    ):
        return False, f"{symbol}: parity_ok is false (run parity and set parity_ok=true)"
    return True, f"{symbol}: parity gate passed"


def _write_progress_rows(out_path: Path, rows: List[List[Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as progress_file:
            writer = csv.writer(progress_file)
            writer.writerow([
                "symbol",
                "phase",
                "status",
                "evaluated",
                "elapsed_seconds",
                "top_score",
                "note",
            ])
            writer.writerows(rows)
        os.replace(temporary_path, out_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _timeframe_sort_key(timeframe: Any) -> Tuple[int, str]:
    normalized = str(timeframe or "").strip().lower()
    return {"15m": 0, "30m": 1, "60m": 2}.get(normalized, 99), normalized


def _group_tickers_by_symbol(
    tickers: List[Dict[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ticker in tickers:
        symbol = str(ticker.get("symbol", "")).strip().upper()
        grouped.setdefault(symbol, []).append(ticker)
    return [
        (
            symbol,
            sorted(entries, key=lambda item: _timeframe_sort_key(item.get("timeframe"))),
        )
        for symbol, entries in sorted(grouped.items())
    ]


def _ticker_status(ticker: Dict[str, Any]) -> str:
    raw_status = ticker.get("Status", ticker.get("status", "Incomplete"))
    status = str(raw_status or "Incomplete").strip().lower() or "incomplete"
    if status not in {"completed", "incomplete"}:
        symbol = str(ticker.get("symbol", "")).strip().upper() or "UNKNOWN"
        timeframe = str(ticker.get("timeframe", "")).strip().lower() or "n/a"
        raise ValueError(
            f"{symbol} {timeframe}: invalid Status '{raw_status}' "
            "(expected Completed or Incomplete)"
        )
    return status


def _mark_suitable(
    result: Dict[str, Any],
    strong_filters: Dict[str, Any],
) -> Dict[str, Any]:
    result["strong_candidate_found"] = any(
        is_strong_candidate(candidate.get("metrics", {}) or {}, strong_filters)
        for candidate in (result.get("top", []) or [])
    )
    return result


def _append_progress_row(
    progress_rows: List[List[Any]],
    result: Dict[str, Any],
) -> None:
    top = result.get("top", []) or []
    note = result.get("note", "")
    if result.get("strong_candidate_found"):
        note = (note + "; " if note else "") + "suitable candidate found"
    progress_rows.append([
        result.get("symbol", ""),
        result.get("phase", ""),
        "done",
        result.get("evaluated", 0),
        round(float(result.get("elapsed_seconds", 0.0)), 2),
        top[0].get("score", "") if top else "",
        note,
    ])


def _best_scores_by_timeframe(results: List[Dict[str, Any]]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for result in results:
        top = result.get("top", []) or []
        if not top:
            continue
        try:
            score = float(top[0].get("score"))
        except (TypeError, ValueError):
            continue
        timeframe = str(result.get("timeframe", "")).strip().lower()
        scores[timeframe] = score
    return scores


def _has_new_best_score(
    current_scores: Dict[str, float],
    previous_scores: Dict[str, float],
) -> bool:
    return any(
        score > previous_scores.get(timeframe, float("-inf"))
        for timeframe, score in current_scores.items()
    )


def _prompt_yes_no(message: str, default: Any = None) -> bool:
    if default is None:
        prompt = f"{message} [y/n]: "
    else:
        prompt = f"{message} [{'Y' if default else 'y'}/{'n' if default else 'N'}]: "
    while True:
        try:
            reply = input(prompt).strip().lower()
        except EOFError:
            if default is None:
                raise RuntimeError(f"No interactive input available for prompt: {message}")
            print(
                f"{message}: no interactive input available; "
                f"defaulting to {'yes' if default else 'no'}."
            )
            return bool(default)
        if not reply and default is not None:
            return bool(default)
        if reply in {"y", "yes"}:
            return True
        if reply in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _read_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _print_symbol_summary(symbol: str, results: List[Dict[str, Any]]) -> None:
    print(f"\n{symbol} summary:")
    for result in results:
        timeframe = str(result.get("timeframe", "")).strip().lower()
        top = result.get("top", []) or []
        if not top:
            print(f"  - {timeframe or 'n/a'}: no suitable candidates")
            continue
        best = top[0]
        metrics = best.get("metrics", {}) or {}
        print(
            "  - "
            f"{timeframe or 'n/a'}: "
            f"score={float(best.get('score', 0.0)):.2f}/100, "
            f"net={metrics.get('net_profit', 0.0):.2f}, "
            f"pf={metrics.get('profit_factor', 0.0):.2f}, "
            f"wr={float(metrics.get('win_rate', 0.0)) * 100:.1f}%, "
            f"dd={float(metrics.get('max_drawdown_pct', 0.0)) * 100:.2f}%"
        )


class _TickerLock:
    def __init__(self, symbol: str):
        safe_symbol = re.sub(r"[^A-Za-z0-9_.-]+", "_", symbol.upper())
        self.path = Path("optimizer_results") / ".locks" / f"{safe_symbol}.lock"
        self.handle = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        if self.handle.seek(0, os.SEEK_END) == 0:
            self.handle.write(b"\0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError):
            self.handle.close()
            self.handle = None
            return False
        return True

    def release(self) -> None:
        if self.handle is None:
            return
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


def _run_grid_cycle(
    ticker: Dict[str, Any],
    grid: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    optimization_config = config.get("optimization", {}) or {}
    strong_filters = {
        **STRONG_FILTERS,
        **(optimization_config.get("strong_filters", {}) or {}),
    }
    result = optimize_ticker(
        ticker,
        grid,
        [str((config.get("execution", {}) or {}).get("intrabar_path", "ohlc"))],
        top_k=int(config.get("top_k_per_ticker", 10)),
        time_budget=int(config.get("time_budget_seconds_per_ticker", 3600)),
        search_mode="randomized",
        seed=derive_seed(
            int(config.get("random_seed", 0)),
            ticker.get("symbol"),
            ticker.get("timeframe"),
        ),
        execution=(config.get("execution", {}) or {}),
        robustness=(config.get("robustness", {}) or {}),
        phase="grid",
        filters=(optimization_config.get("filters", None) or None),
    )
    return _mark_suitable(result, strong_filters)


def _run_symbol(
    symbol: str,
    symbol_tickers: List[Dict[str, Any]],
    grid: Dict[str, Any],
    config: Dict[str, Any],
    progress_rows: List[List[Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    optimization_config = config.get("optimization", {}) or {}
    prompt_cadence = _read_positive_int(
        optimization_config.get("confirm_continue_every_cycles", 3),
        3,
    )
    interactive_prompts = sys.stdin.isatty()
    latest_by_timeframe: Dict[str, Dict[str, Any]] = {}
    last_declined_suitable_scores: Dict[str, float] = {}
    cycles_without_better = 0
    cycle_index = 0

    print(f"\nStarting {symbol} with {len(symbol_tickers)} timeframe(s)")
    while True:
        current_cycle = cycle_index + 1
        print(f"{symbol}: grid cycle {current_cycle}")
        for ticker in symbol_tickers:
            result = _run_grid_cycle(ticker, grid, config)
            timeframe = str(ticker.get("timeframe", "")).strip().lower()
            latest_by_timeframe[timeframe] = result
            _append_progress_row(progress_rows, result)

        summarized_results = [
            latest_by_timeframe[key]
            for key in sorted(latest_by_timeframe, key=_timeframe_sort_key)
        ]
        _print_symbol_summary(symbol, summarized_results)
        suitable_found = any(
            bool(result.get("strong_candidate_found"))
            for result in summarized_results
        )
        print(
            f"{symbol}: optimizer "
            f"{'found' if suitable_found else 'did not find'} "
            "a candidate that meets the gating thresholds."
        )

        if not interactive_prompts:
            if suitable_found:
                print(
                    f"{symbol}: no interactive input available; "
                    "accepting the gated candidates and moving on."
                )
                return summarized_results, False
            evaluated_this_cycle = sum(
                int(result.get("evaluated", 0) or 0)
                for result in summarized_results
            )
            if evaluated_this_cycle <= 0:
                print(
                    f"{symbol}: no interactive input available, no gated candidate "
                    "found, and no new combinations evaluated; stopping."
                )
                return summarized_results, True
            cycle_index += 1
            print(
                f"{symbol}: no interactive input available and no gated candidate "
                "found; automatically continuing the grid."
            )
            continue

        if suitable_found:
            current_best_scores = _best_scores_by_timeframe(summarized_results)
            if not _has_new_best_score(
                current_best_scores,
                last_declined_suitable_scores,
            ):
                cycles_without_better += 1
                if (
                    cycles_without_better % prompt_cadence == 0
                    and not _prompt_yes_no(
                        f"{symbol}: Current Cycle {current_cycle} - no better "
                        f"candidate found in {cycles_without_better} cycle(s). "
                        "Continue grid search",
                        default=True,
                    )
                ):
                    print(
                        f"{symbol}: no better candidate accepted; "
                        "moving to the next ticker."
                    )
                    return summarized_results, False
                cycle_index += 1
                print(
                    f"{symbol}: no better suitable score since the last review; "
                    "automatically continuing the grid."
                )
                continue
            if _prompt_yes_no(
                f"{symbol}: Current Cycle {current_cycle} - "
                "are the current candidates suitable",
                default=True,
            ):
                if _prompt_yes_no(
                    f"{symbol}: Current Cycle {current_cycle} - move to the next ticker",
                    default=True,
                ):
                    return summarized_results, False
                print(f"{symbol}: current candidates kept; stopping before the next ticker.")
                return summarized_results, True
            last_declined_suitable_scores = current_best_scores
            cycles_without_better = 0
            cycle_index += 1
            print(
                f"{symbol}: suitable candidates declined; continuing the grid "
                "until a better score is found."
            )
            continue

        no_suitable_cycles = cycle_index + 1
        should_prompt = no_suitable_cycles % prompt_cadence == 0
        if should_prompt and not _prompt_yes_no(
            f"{symbol}: Current Cycle {current_cycle} - no suitable candidate after "
            f"{no_suitable_cycles} cycle(s). Continue grid search",
            default=True,
        ):
            print(f"{symbol}: no suitable candidate accepted; moving to the next ticker.")
            return summarized_results, False
        cycle_index += 1
        print(
            f"{symbol}: no suitable candidate found; "
            "automatically continuing the grid."
        )


def _write_aggregate_results(results: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "symbol",
        "phase",
        "score",
        "net_profit",
        "trade_count",
        "profit_factor",
        "max_drawdown_pct",
        "params",
        "report",
    ]
    new_rows: Dict[str, Dict[str, Any]] = {}
    for result in results:
        key = (
            f"{result.get('symbol', '')}_"
            f"{str(result.get('timeframe') or '').strip().lower()}"
        )
        top = result.get("top", []) or []
        row: Dict[str, Any] = {
            "symbol": key,
            "phase": result.get("phase", ""),
            "score": "",
            "net_profit": "",
            "trade_count": "",
            "profit_factor": "",
            "max_drawdown_pct": "",
            "params": "",
            "report": result.get("report", ""),
        }
        if top:
            best = top[0]
            metrics = best.get("metrics", {})
            row.update({
                "score": best.get("score", ""),
                "net_profit": metrics.get("net_profit", ""),
                "trade_count": metrics.get("trade_count", ""),
                "profit_factor": metrics.get("profit_factor", ""),
                "max_drawdown_pct": metrics.get("max_drawdown_pct", ""),
                "params": json.dumps(best.get("params", {})),
            })
        new_rows[key] = row

    aggregate_lock = _TickerLock("__aggregate_results__")
    for _ in range(100):
        if aggregate_lock.acquire():
            break
        sleep(0.05)
    else:
        raise RuntimeError("Could not acquire the aggregate results lock within 5 seconds")

    try:
        output_dir = Path("optimizer_results")
        output_dir.mkdir(exist_ok=True)
        aggregate_path = output_dir / "best_presets.csv"
        merged_rows: Dict[str, Dict[str, Any]] = {}
        if aggregate_path.exists():
            with aggregate_path.open("r", newline="", encoding="utf-8") as aggregate_file:
                for row in csv.DictReader(aggregate_file):
                    symbol = str(row.get("symbol", ""))
                    if symbol:
                        merged_rows[symbol] = row
        merged_rows.update(new_rows)

        temporary_path = output_dir / f".best_presets.{os.getpid()}.tmp"
        try:
            with temporary_path.open("w", newline="", encoding="utf-8") as aggregate_file:
                writer = csv.DictWriter(aggregate_file, fieldnames=fieldnames)
                writer.writeheader()
                for symbol in sorted(merged_rows):
                    writer.writerow(merged_rows[symbol])
            os.replace(temporary_path, aggregate_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    finally:
        aggregate_lock.release()


def main() -> None:
    config = load_config()
    tickers = config.get("tickers", []) or discover_tsvs_auto()
    if not tickers:
        print("No tickers found in config and no TSVs discovered in data/. Exiting.")
        return

    parity_config = config.get("parity", {}) or {}
    progress_rows: List[List[Any]] = []
    gated_tickers: List[Dict[str, Any]] = []
    for ticker in tickers:
        symbol = ticker.get("symbol", "")
        if _ticker_status(ticker) == "completed":
            progress_rows.append([
                symbol,
                "status",
                "skipped",
                0,
                0,
                "",
                f"{symbol}: skipped because Status is Completed",
            ])
            continue
        sane, sanity_note = _sanity_check_ticker(ticker)
        if not sane:
            progress_rows.append([
                symbol, "sanity", "skipped", 0, 0, "", sanity_note
            ])
            continue
        parity_ok, parity_note = _parity_gate_pass(ticker, parity_config)
        if not parity_ok:
            progress_rows.append([
                symbol, "parity", "skipped", 0, 0, "", parity_note
            ])
            continue
        progress_rows.append([
            symbol, "parity", "ready", 0, 0, "", parity_note
        ])
        gated_tickers.append(ticker)

    progress_path = Path("optimizer_results") / "progress.csv"
    _write_progress_rows(progress_path, progress_rows)
    if not gated_tickers:
        print("No tickers passed the status, sanity, and parity gates. Exiting.")
        return

    started_at = time()
    final_results: List[Dict[str, Any]] = []
    symbol_groups = _group_tickers_by_symbol(gated_tickers)
    grid = config.get("grid") or DEFAULT_CONFIG["grid"]
    print(f"Starting unified grid search for {len(symbol_groups)} ticker(s)")
    for symbol, symbol_tickers in symbol_groups:
        with _TickerLock(symbol) as lock_acquired:
            if not lock_acquired:
                note = (
                    f"{symbol}: skipped because another optimizer process "
                    "is already working on this ticker"
                )
                print(note)
                for ticker in symbol_tickers:
                    progress_rows.append([
                        symbol,
                        "lock",
                        "skipped",
                        0,
                        0,
                        "",
                        f"{note} ({ticker.get('timeframe', '')})",
                    ])
                continue
            symbol_results, stop_after_current = _run_symbol(
                symbol,
                symbol_tickers,
                grid,
                config,
                progress_rows,
            )
            final_results.extend(symbol_results)
            if stop_after_current:
                break

    _write_aggregate_results(final_results)
    _write_progress_rows(progress_path, progress_rows)
    print("Optimization complete. Results in optimizer_results/")
    print(f"Elapsed: {time() - started_at:.1f}s")


if __name__ == "__main__":
    main()
