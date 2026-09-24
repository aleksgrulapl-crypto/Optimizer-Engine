import csv
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optimize_all as oa
import optimizer_worker as ow


REPO_ROOT = Path(__file__).resolve().parent.parent


def make_candles(count=300):
    candles = []
    for index in range(count):
        base = 100.0 + index * 0.05
        close = base + 2.0 * math.sin(index / 6.0)
        open_price = base + 2.0 * math.sin((index - 1) / 6.0)
        candles.append({
            "time": 1_700_000_000 + index * 900,
            "open": open_price,
            "high": max(open_price, close) + 0.3,
            "low": min(open_price, close) - 0.3,
            "close": close,
            "volume": 1_000.0,
        })
    return candles


class UnifiedGridConfigTests(unittest.TestCase):
    def test_default_grid_has_requested_ranges_and_size(self):
        config = oa.merge_with_defaults({})
        grid = config["grid"]
        self.assertEqual(grid["stMultiplier"], {"start": 2.0, "stop": 10.0, "step": 0.1})
        self.assertEqual(grid["stPeriod"], {"start": 5, "stop": 18, "step": 1})
        self.assertEqual(grid["atrSLmult"], {"start": 1.0, "stop": 3.0, "step": 0.1})
        self.assertEqual(grid["atrTPmult"], {"start": 1.2, "stop": 5.0, "step": 0.1})
        self.assertEqual(grid["emaLen"], {"start": 40, "stop": 240, "step": 1})
        expanded = {key: ow._expand_spec(spec) for key, spec in grid.items()}
        self.assertEqual([len(values) for values in expanded.values()], [81, 14, 21, 39, 201])
        self.assertEqual(ow._count_combinations(expanded), 186_677_946)
        self.assertTrue(all(isinstance(value, int) for value in expanded["stPeriod"]))
        self.assertTrue(all(isinstance(value, int) for value in expanded["emaLen"]))

    def test_defaults_use_one_hour_cycles_top_ten_and_third_cycle_prompt(self):
        config = oa.merge_with_defaults({})
        self.assertEqual(config["time_budget_seconds_per_ticker"], 3600)
        self.assertEqual(config["top_k_per_ticker"], 10)
        self.assertEqual(config["optimization"]["confirm_continue_every_cycles"], 3)
        self.assertEqual(config["robustness"]["evaluate_top_n"], 10)
        self.assertNotIn("staged_search", config)
        self.assertNotIn("grid_constrained", config)
        self.assertNotIn("grid_expand", config)

    def test_default_candidate_requirements_match_requested_thresholds(self):
        config = oa.merge_with_defaults({})
        expected = {
            "min_win_rate": 0.45,
            "min_profit_factor": 1.4,
            "min_net_profit": 0.0,
            "min_trades": 100,
            "max_drawdown_pct": 0.25,
        }
        self.assertEqual(config["optimization"]["filters"], expected)
        self.assertEqual(config["optimization"]["strong_filters"], expected)

    def test_repo_yaml_matches_unified_defaults(self):
        old_cwd = Path.cwd()
        os.chdir(REPO_ROOT)
        try:
            config = oa.load_config()
        finally:
            os.chdir(old_cwd)
        self.assertEqual(config["grid"], oa.DEFAULT_CONFIG["grid"])
        self.assertEqual(config["top_k_per_ticker"], 10)
        self.assertEqual(config["optimization"]["confirm_continue_every_cycles"], 3)

    def test_repo_yaml_configures_all_three_timeframes_per_symbol(self):
        old_cwd = Path.cwd()
        os.chdir(REPO_ROOT)
        try:
            tickers = oa.load_config()["tickers"]
        finally:
            os.chdir(old_cwd)
        grouped = oa._group_tickers_by_symbol(tickers)
        self.assertTrue(grouped)
        for _, entries in grouped:
            self.assertEqual(
                [entry["timeframe"] for entry in entries],
                ["15m", "30m", "60m"],
            )

    def test_invalid_prompt_cadence_falls_back_to_three(self):
        self.assertEqual(oa._read_positive_int("bad", 3), 3)
        self.assertEqual(oa._read_positive_int(0, 3), 3)


class RandomizedGridTests(unittest.TestCase):
    GRID = {
        "a": [1, 2, 3],
        "b": [10, 20],
        "c": ["x", "y"],
    }

    def test_randomized_mode_visits_every_combination_once(self):
        params = list(ow.grid_search_params(self.GRID, search_mode="randomized", seed=17))
        keys = {ow._param_key(param) for param in params}
        self.assertEqual(len(params), 12)
        self.assertEqual(len(keys), 12)

    def test_randomized_order_is_stable_and_seeded(self):
        first = list(ow.grid_search_params(self.GRID, search_mode="randomized", seed=17))
        repeated = list(ow.grid_search_params(self.GRID, search_mode="randomized", seed=17))
        different = list(ow.grid_search_params(self.GRID, search_mode="randomized", seed=18))
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)
        self.assertNotEqual(first[0], {"a": 1, "b": 10, "c": "x"})

    def test_resume_position_continues_same_permutation(self):
        full = list(ow.grid_search_params(self.GRID, search_mode="randomized", seed=17))
        resumed = list(
            ow.grid_search_params(
                self.GRID,
                search_mode="randomized",
                seed=17,
                start_position=5,
            )
        )
        self.assertEqual(resumed, full[5:])

    def test_single_combination_grid_is_supported(self):
        self.assertEqual(
            list(ow.grid_search_params({"only": [1]}, search_mode="randomized", seed=1)),
            [{"only": 1}],
        )


class ScoringTests(unittest.TestCase):
    def test_score_is_bounded_and_rounded_to_reduce_noise(self):
        weak = ow.score_candidate({
            "net_profit": -100,
            "profit_factor": 0,
            "win_rate": 0,
            "trade_count": 0,
            "max_drawdown_pct": 1,
        })
        strong = ow.score_candidate({
            "net_profit": 1_000_000,
            "profit_factor": float("inf"),
            "win_rate": 1,
            "trade_count": 10_000,
            "max_drawdown_pct": 0,
        })
        typical = ow.score_candidate({
            "net_profit": 123.456,
            "profit_factor": 1.75,
            "win_rate": 0.53,
            "trade_count": 41,
            "max_drawdown_pct": 0.08,
        })
        self.assertEqual(weak, 0.0)
        self.assertEqual(strong, 100.0)
        self.assertGreaterEqual(typical, 0.0)
        self.assertLessEqual(typical, 100.0)
        self.assertEqual(typical, round(typical, 2))

    def test_score_and_lower_drawdown_control_ranking(self):
        high_score = {"score": 80.0, "metrics": {"max_drawdown": 50.0}}
        low_score = {"score": 70.0, "metrics": {"max_drawdown": 1.0}}
        low_drawdown = {"score": 80.0, "metrics": {"max_drawdown": 10.0}}
        ordered = sorted(
            [low_score, high_score, low_drawdown],
            key=ow._candidate_rank_key,
            reverse=True,
        )
        self.assertIs(ordered[0], low_drawdown)
        self.assertIs(ordered[1], high_score)


class ResumeTests(unittest.TestCase):
    def test_saved_candidates_are_rescored_to_current_scale(self):
        metrics = {
            "net_profit": 100.0,
            "trade_count": 40,
            "win_rate": 0.6,
            "profit_factor": 2.0,
            "max_drawdown": 10.0,
            "max_drawdown_pct": 0.1,
        }
        params = {"stMultiplier": 2.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            completed_dir = root / "completed_runs"
            completed_dir.mkdir()
            (completed_dir / "TEST_15m_grid.jsonl").write_text(json.dumps({
                "_param_key": ow._param_key(params),
                "params": params,
                "metrics": metrics,
                "score": 9999.0,
                "status": "accepted",
            }) + "\n", encoding="utf-8")
            old_cwd = Path.cwd()
            os.chdir(root)
            try:
                _, candidates, best_score, _ = ow._load_completed_state(
                    "TEST_15m",
                    "grid",
                )
            finally:
                os.chdir(old_cwd)
        expected = ow.score_candidate(metrics)
        self.assertEqual(best_score, expected)
        self.assertEqual(candidates[0]["score"], expected)
        self.assertLessEqual(best_score, 100.0)

    def test_randomized_worker_skips_persisted_positions_and_evaluates_next(self):
        ticker = {"symbol": "TEST", "timeframe": "15m", "tsv": "ignored.tsv"}
        grid = {"stMultiplier": [1.0, 2.0, 3.0]}
        seed = 7
        entries = list(
            ow._grid_search_entries(grid, search_mode="randomized", seed=seed)
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            completed_dir = root / "completed_runs"
            completed_dir.mkdir()
            completed_path = completed_dir / "TEST_15m_grid.jsonl"
            with completed_path.open("w", encoding="utf-8") as completed_file:
                for position, params in entries[:2]:
                    run_params = {
                        **params,
                        "ticker": "TEST",
                        "timeframe": "15m",
                        "intrabar_path": "ohlc",
                        "position_size": 1.0,
                        "slippage": 0.0,
                        "commission_pct": 0.0,
                        "pyramiding": 1,
                    }
                    completed_file.write(json.dumps({
                        "_search_position": position,
                        "_param_key": ow._param_key(run_params),
                        "params": run_params,
                        "metrics": {},
                        "score": None,
                        "status": "rejected",
                    }) + "\n")

            old_cwd = Path.cwd()
            os.chdir(root)
            try:
                with mock.patch("optimizer_worker.load_candles_from_csv", return_value=[]), \
                     mock.patch("optimizer_worker.run_backtest", return_value={"trade_dicts": []}) as run_backtest, \
                     mock.patch("optimizer_worker.compute_metrics_from_run", return_value={
                         "net_profit": -1.0,
                         "trade_count": 0,
                         "win_rate": 0.0,
                         "profit_factor": 0.0,
                         "max_drawdown": 0.0,
                         "max_drawdown_pct": 0.0,
                     }):
                    result = ow.optimize_ticker(
                        ticker,
                        grid,
                        ["ohlc"],
                        search_mode="randomized",
                        seed=seed,
                        phase="grid",
                        robustness={"enabled": False},
                    )
            finally:
                os.chdir(old_cwd)

        self.assertEqual(run_backtest.call_count, 1)
        self.assertEqual(result["evaluated"], 1)


class WorkerIntegrationTests(unittest.TestCase):
    def test_actual_backtest_runs_through_unified_worker(self):
        ticker = {"symbol": "NVDA", "timeframe": "15m", "tsv": "ignored.tsv"}
        grid = {
            "stMultiplier": [2.0, 2.1],
            "stPeriod": [5],
            "atrSLmult": [1.0],
            "atrTPmult": [1.2],
            "emaLen": [40],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            old_cwd = Path.cwd()
            os.chdir(temp_dir)
            try:
                with mock.patch(
                    "optimizer_worker.load_candles_from_csv",
                    return_value=make_candles(),
                ):
                    result = ow.optimize_ticker(
                        ticker,
                        grid,
                        ["ohlc"],
                        top_k=10,
                        time_budget=30,
                        search_mode="randomized",
                        seed=11,
                        phase="grid",
                        robustness={"enabled": False},
                    )
                self.assertTrue(Path(result["csv"]).exists())
                self.assertTrue(Path(result["report"]).exists())
            finally:
                os.chdir(old_cwd)
        self.assertEqual(result["phase"], "grid")
        self.assertEqual(result["evaluated"], 2)
        self.assertTrue(all(0.0 <= candidate["score"] <= 100.0 for candidate in result["top"]))


class GridCycleTests(unittest.TestCase):
    def test_grid_cycle_uses_full_hour_randomized_phase_and_top_ten(self):
        config = oa.merge_with_defaults({})
        ticker = {"symbol": "AAPL", "timeframe": "15m", "tsv": "ignored.tsv"}
        mocked_result = {
            "symbol": "AAPL",
            "timeframe": "15m",
            "phase": "grid",
            "top": [],
            "evaluated": 0,
            "elapsed_seconds": 0.0,
            "note": "",
        }
        with mock.patch("optimize_all.optimize_ticker", return_value=mocked_result) as optimize:
            result = oa._run_grid_cycle(ticker, config["grid"], config)
        _, kwargs = optimize.call_args
        self.assertEqual(kwargs["time_budget"], 3600)
        self.assertEqual(kwargs["top_k"], 10)
        self.assertEqual(kwargs["search_mode"], "randomized")
        self.assertEqual(kwargs["phase"], "grid")
        self.assertFalse(result["strong_candidate_found"])

    def test_no_candidate_prompt_occurs_on_third_cycle(self):
        config = oa.merge_with_defaults({
            "tickers": [{"symbol": "NVDA", "timeframe": "15m", "tsv": "ignored.tsv"}],
        })
        result = {
            "symbol": "NVDA",
            "timeframe": "15m",
            "phase": "grid",
            "top": [],
            "evaluated": 1,
            "elapsed_seconds": 0.0,
            "note": "",
            "strong_candidate_found": False,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            old_cwd = Path.cwd()
            os.chdir(temp_dir)
            try:
                with mock.patch("optimize_all.load_config", return_value=config), \
                     mock.patch("optimize_all._sanity_check_ticker", return_value=(True, "ok")), \
                     mock.patch("optimize_all._parity_gate_pass", return_value=(True, "ok")), \
                     mock.patch("optimize_all._run_grid_cycle", return_value=result) as run_cycle, \
                     mock.patch("optimize_all._prompt_yes_no", return_value=False) as prompt, \
                     mock.patch("optimize_all._write_progress_rows"), \
                     mock.patch("sys.stdin.isatty", return_value=True):
                    oa.main()
            finally:
                os.chdir(old_cwd)
        self.assertEqual(run_cycle.call_count, 3)
        prompt.assert_called_once()
        self.assertIn("Current Cycle 3", prompt.call_args.args[0])

    def test_no_better_candidate_prompt_occurs_every_third_cycle(self):
        config = oa.merge_with_defaults({})
        result = {
            "symbol": "NVDA",
            "timeframe": "15m",
            "phase": "grid",
            "top": [{"score": 75.0, "params": {}, "metrics": {}}],
            "evaluated": 1,
            "elapsed_seconds": 0.0,
            "note": "",
            "strong_candidate_found": True,
        }
        with mock.patch("optimize_all._run_grid_cycle", return_value=result) as run_cycle, \
             mock.patch("optimize_all._prompt_yes_no", side_effect=[False, False]) as prompt, \
             mock.patch("sys.stdin.isatty", return_value=True):
            results, stop = oa._run_symbol(
                "NVDA",
                [{"symbol": "NVDA", "timeframe": "15m"}],
                config["grid"],
                config,
                [],
            )
        self.assertFalse(stop)
        self.assertEqual(results[0]["top"][0]["score"], 75.0)
        self.assertEqual(run_cycle.call_count, 4)
        self.assertEqual(prompt.call_count, 2)
        self.assertIn("are the current candidates suitable", prompt.call_args_list[0].args[0])
        self.assertIn("no better candidate found in 3 cycle(s)", prompt.call_args_list[1].args[0])

    def test_completed_status_skips_ticker(self):
        config = oa.merge_with_defaults({
            "tickers": [{
                "symbol": "NVDA",
                "timeframe": "15m",
                "tsv": "ignored.tsv",
                "Status": "Completed",
            }],
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            old_cwd = Path.cwd()
            os.chdir(temp_dir)
            try:
                with mock.patch("optimize_all.load_config", return_value=config), \
                     mock.patch("optimize_all._sanity_check_ticker") as sanity, \
                     mock.patch("optimize_all._write_progress_rows"):
                    oa.main()
            finally:
                os.chdir(old_cwd)
        sanity.assert_not_called()


class TickerLockTests(unittest.TestCase):
    def test_second_lock_for_same_symbol_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            old_cwd = Path.cwd()
            os.chdir(temp_dir)
            first = oa._TickerLock("AAPL")
            second = oa._TickerLock("AAPL")
            try:
                self.assertTrue(first.acquire())
                self.assertFalse(second.acquire())
            finally:
                second.release()
                first.release()
                os.chdir(old_cwd)


class AggregateResultsTests(unittest.TestCase):
    def test_aggregate_write_preserves_results_from_other_processes(self):
        first = {
            "symbol": "AAPL",
            "timeframe": "15m",
            "phase": "grid",
            "top": [],
            "report": "aapl.json",
        }
        second = {
            "symbol": "NVDA",
            "timeframe": "30m",
            "phase": "grid",
            "top": [],
            "report": "nvda.json",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            old_cwd = Path.cwd()
            os.chdir(temp_dir)
            try:
                oa._write_aggregate_results([first])
                oa._write_aggregate_results([second])
                with Path("optimizer_results/best_presets.csv").open(
                    "r",
                    newline="",
                    encoding="utf-8",
                ) as aggregate_file:
                    rows = list(csv.DictReader(aggregate_file))
            finally:
                os.chdir(old_cwd)
        self.assertEqual(
            [row["symbol"] for row in rows],
            ["AAPL_15m", "NVDA_30m"],
        )


if __name__ == "__main__":
    unittest.main()
