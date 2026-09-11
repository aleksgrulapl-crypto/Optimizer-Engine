"""Tests for the staged search strategy (initial random -> expanded)."""

import csv
import math
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optimizer_worker as ow
from optimize_all import _build_preset_grid, _group_tickers_by_symbol, _run_preset_phase, merge_with_defaults
from optimizer_worker import (
    DEFAULT_FILTERS,
    STRONG_FILTERS,
    build_neighborhood_grid,
    derive_seed,
    is_strong_candidate,
    passes_filters,
    staged_search,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_candles(n: int = 600, tf_minutes: int = 15):
    """Deterministic synthetic candles (gentle uptrend + oscillation) that trade well."""
    candles = []
    t0 = 1_700_000_000
    for i in range(n):
        base = 100.0 + i * 0.05
        osc = 2.0 * math.sin(i / 6.0)
        close = base + osc
        open_ = base + 2.0 * math.sin((i - 1) / 6.0)
        high = max(open_, close) + 0.3
        low = min(open_, close) - 0.3
        candles.append({
            "time": t0 + i * tf_minutes * 60,
            "open": open_, "high": high, "low": low, "close": close,
            "volume": 1000.0,
        })
    return candles


def write_tsv(path: Path, candles) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([c["time"], c["open"], c["high"], c["low"], c["close"], c["volume"]])


class PassesFiltersTests(unittest.TestCase):
    def test_default_thresholds(self):
        self.assertEqual(DEFAULT_FILTERS["min_win_rate"], 0.40)
        self.assertEqual(DEFAULT_FILTERS["min_profit_factor"], 1.4)
        self.assertEqual(DEFAULT_FILTERS["min_net_profit"], 0.0)
        self.assertEqual(DEFAULT_FILTERS["max_drawdown_pct"], 0.25)

    def test_accepts_candidate_meeting_criteria(self):
        m = {"net_profit": 100.0, "profit_factor": 1.5, "win_rate": 0.40, "trade_count": 25, "max_drawdown_pct": 0.25}
        self.assertTrue(passes_filters(m, DEFAULT_FILTERS))

    def test_rejects_boundary_failures(self):
        base = {"net_profit": 100.0, "profit_factor": 1.5, "win_rate": 0.60, "trade_count": 25, "max_drawdown_pct": 0.20}
        self.assertFalse(passes_filters({**base, "win_rate": 0.39}, DEFAULT_FILTERS))
        self.assertFalse(passes_filters({**base, "profit_factor": 1.39}, DEFAULT_FILTERS))
        self.assertFalse(passes_filters({**base, "net_profit": 0.0}, DEFAULT_FILTERS))
        self.assertFalse(passes_filters({**base, "net_profit": -1.0}, DEFAULT_FILTERS))
        self.assertFalse(passes_filters({**base, "trade_count": 9}, DEFAULT_FILTERS))
        self.assertFalse(passes_filters({**base, "max_drawdown_pct": 0.26}, DEFAULT_FILTERS))

    def test_filters_are_configurable(self):
        m = {"net_profit": 100.0, "profit_factor": 1.1, "win_rate": 0.40, "trade_count": 5, "max_drawdown_pct": 0.40}
        self.assertFalse(passes_filters(m, DEFAULT_FILTERS))
        relaxed = {"min_win_rate": 0.30, "min_profit_factor": 1.0, "min_net_profit": 0.0, "min_trades": 1, "max_drawdown_pct": 0.50}
        self.assertTrue(passes_filters(m, relaxed))


class StrongCandidateTests(unittest.TestCase):
    def test_default_thresholds(self):
        self.assertEqual(STRONG_FILTERS["min_win_rate"], 0.40)
        self.assertEqual(STRONG_FILTERS["min_profit_factor"], 1.4)
        self.assertEqual(STRONG_FILTERS["min_net_profit"], 0.0)
        self.assertIn("max_drawdown_pct", STRONG_FILTERS)

    def test_accepts_strong_candidate(self):
        m = {"net_profit": 250.0, "profit_factor": 1.6, "win_rate": 0.52, "trade_count": 40, "max_drawdown_pct": 0.10}
        self.assertTrue(is_strong_candidate(m))

    def test_rejects_weak_candidates(self):
        base = {"net_profit": 250.0, "profit_factor": 1.6, "win_rate": 0.52, "trade_count": 40, "max_drawdown_pct": 0.10}
        self.assertFalse(is_strong_candidate({**base, "profit_factor": 1.39}))
        self.assertFalse(is_strong_candidate({**base, "win_rate": 0.39}))
        self.assertFalse(is_strong_candidate({**base, "net_profit": 0.0}))
        self.assertFalse(is_strong_candidate({**base, "max_drawdown_pct": 0.40}))

    def test_drawdown_cap_configurable(self):
        m = {"net_profit": 250.0, "profit_factor": 1.6, "win_rate": 0.52, "trade_count": 40, "max_drawdown_pct": 0.40}
        self.assertTrue(is_strong_candidate(m, {"max_drawdown_pct": 0.50}))


class PresetAndGroupingTests(unittest.TestCase):
    def test_build_preset_grid_uses_single_values(self):
        with mock.patch("optimize_all.get_presets", return_value={
            "stMultiplier": 9.9,
            "stPeriod": 21,
            "atrSLmult": 1.7,
            "atrTPmult": 6.3,
            "emaLen": 55,
        }):
            grid = _build_preset_grid("ANY", "15m")
        self.assertEqual(grid["stMultiplier"], [9.9])
        self.assertEqual(grid["stPeriod"], [21])
        self.assertEqual(grid["atrSLmult"], [1.7])
        self.assertEqual(grid["atrTPmult"], [6.3])
        self.assertEqual(grid["emaLen"], [55])

    def test_group_tickers_by_symbol_preserves_timeframe_order(self):
        grouped = _group_tickers_by_symbol([
            {"symbol": "NVDA", "timeframe": "30m"},
            {"symbol": "MU", "timeframe": "30m"},
            {"symbol": "NVDA", "timeframe": "15m"},
        ])
        self.assertEqual(grouped[0][0], "NVDA")
        self.assertEqual([item["timeframe"] for item in grouped[0][1]], ["30m", "15m"])
        self.assertEqual(grouped[1][0], "MU")

    def test_run_preset_phase_marks_unsupported_symbols_as_skipped(self):
        result = _run_preset_phase({"symbol": "UNKNOWN", "timeframe": "15m"}, {"staged_search": {}})
        self.assertEqual(result["phase"], "preset")
        self.assertFalse(result["top"])
        self.assertFalse(result["strong_candidate_found"])
        self.assertIn("preset skipped", result["note"])

    def test_run_preset_phase_marks_supported_candidates(self):
        ticker = {"symbol": "NVDA", "timeframe": "15m", "tsv": "ignored.tsv"}
        cfg = {
            "staged_search": {"filters": {"min_win_rate": 0.40}},
            "execution": {"intrabar_path": "hl"},
            "robustness": {"enabled": False},
            "top_k_per_ticker": 7,
            "time_budget_seconds_per_ticker": 321,
            "max_exhaustive": 1234,
            "random_seed": 99,
        }
        mocked_result = {
            "symbol": "NVDA",
            "timeframe": "15m",
            "phase": "preset",
            "top": [{
                "params": {"stMultiplier": 2.0},
                "metrics": {
                    "net_profit": 100.0,
                    "profit_factor": 1.5,
                    "win_rate": 0.45,
                    "trade_count": 12,
                    "max_drawdown_pct": 0.20,
                },
            }],
            "evaluated": 1,
            "elapsed_seconds": 0.01,
            "note": "ok",
        }
        with mock.patch("optimize_all.get_presets", return_value={"stMultiplier": 2.0}), mock.patch(
            "optimize_all.optimize_ticker",
            return_value=mocked_result,
        ) as optimize_ticker_mock:
            result = _run_preset_phase(ticker, cfg)
        self.assertTrue(result["strong_candidate_found"])
        self.assertEqual(result["top"][0]["metrics"]["profit_factor"], 1.5)
        _, kwargs = optimize_ticker_mock.call_args
        self.assertEqual(kwargs["top_k"], 7)
        self.assertEqual(kwargs["time_budget"], 321)
        self.assertEqual(kwargs["search_mode"], "auto")
        self.assertEqual(kwargs["n_samples"], 1)
        self.assertEqual(kwargs["seed"], 99)
        self.assertEqual(kwargs["max_exhaustive"], 1234)
        self.assertEqual(kwargs["execution"], {"intrabar_path": "hl"})
        self.assertEqual(kwargs["robustness"], {"enabled": False})
        self.assertEqual(kwargs["phase"], "preset")
        self.assertEqual(kwargs["filters"], {"min_win_rate": 0.40})


class DeriveSeedTests(unittest.TestCase):
    def test_distinct_per_ticker(self):
        self.assertNotEqual(derive_seed(42, "NVDA", "15m"), derive_seed(42, "NVDA", "30m"))
        self.assertNotEqual(derive_seed(42, "NVDA", "15m"), derive_seed(42, "MU", "15m"))

    def test_stable(self):
        self.assertEqual(derive_seed(42, "NVDA", "15m"), derive_seed(42, "NVDA", "15m"))

    def test_depends_on_base_seed(self):
        self.assertNotEqual(derive_seed(0, "NVDA", "15m"), derive_seed(1, "NVDA", "15m"))


class CandidateRankingTests(unittest.TestCase):
    def test_lower_drawdown_preferred_on_score_tie(self):
        high_dd = {"score": 100.0, "metrics": {"max_drawdown": 50.0}}
        low_dd = {"score": 100.0, "metrics": {"max_drawdown": 10.0}}
        ordered = sorted([high_dd, low_dd], key=ow._candidate_rank_key, reverse=True)
        self.assertIs(ordered[0], low_dd)

    def test_score_dominates_drawdown(self):
        better = {"score": 120.0, "metrics": {"max_drawdown": 50.0}}
        worse = {"score": 100.0, "metrics": {"max_drawdown": 1.0}}
        ordered = sorted([worse, better], key=ow._candidate_rank_key, reverse=True)
        self.assertIs(ordered[0], better)


class NeighborhoodGridTests(unittest.TestCase):
    GRID = {
        "stMultiplier": {"start": 1.0, "stop": 3.0, "step": 0.2},
        "stPeriod": {"min": 6, "max": 14, "count": 5},
        "atrTPmult": [2.0, 3.0, 4.0],
        "emaLen": {"min": 20, "max": 120, "count": 6},
        "mode": ["a", "b"],
    }
    CENTER = {"stMultiplier": 2.0, "stPeriod": 10, "atrTPmult": 3.0, "emaLen": 60, "mode": "a"}

    def test_center_always_included(self):
        ng = build_neighborhood_grid(self.GRID, self.CENTER, 2.0)
        for key, val in self.CENTER.items():
            self.assertIn(val, ng[key])

    def test_int_radius_counts_grid_steps(self):
        ng = build_neighborhood_grid(self.GRID, self.CENTER, 2.0)
        self.assertEqual(ng["stPeriod"], [6, 8, 10, 12, 14])
        self.assertEqual(ng["emaLen"], [20, 40, 60, 80, 100])

    def test_float_radius_is_distance(self):
        ng = build_neighborhood_grid(self.GRID, self.CENTER, 0.4)
        self.assertEqual(ng["stMultiplier"], [1.6, 1.8, 2.0, 2.2, 2.4])

    def test_refined_float_radius_uses_finer_step(self):
        ng = build_neighborhood_grid(self.GRID, self.CENTER, 0.4)
        self.assertEqual(ng["atrTPmult"], [2.6, 3.0, 3.4])

    def test_small_radius_keeps_base_step(self):
        ng = build_neighborhood_grid(self.GRID, self.CENTER, 0.2)
        self.assertEqual(ng["stMultiplier"], [1.8, 2.0, 2.2])

    def test_small_radius_refines_single_value_floats(self):
        ng = build_neighborhood_grid({"stMultiplier": [2.0]}, {"stMultiplier": 2.0}, 0.2)
        self.assertEqual(ng["stMultiplier"], [1.8, 1.9, 2.0, 2.1, 2.2])

    def test_non_numeric_values_pass_through(self):
        ng = build_neighborhood_grid(self.GRID, self.CENTER, 2.0)
        self.assertEqual(ng["mode"], ["a", "b"])

    def test_neighborhood_clamped_to_base_grid_range(self):
        center = {"stMultiplier": 1.0, "stPeriod": 6}
        ng = build_neighborhood_grid(self.GRID, center, 3.0)
        self.assertGreaterEqual(min(ng["stMultiplier"]), 1.0)
        self.assertLessEqual(max(ng["stMultiplier"]), 3.0)
        self.assertGreaterEqual(min(ng["stPeriod"]), 6)
        self.assertLessEqual(max(ng["stPeriod"]), 14)

    def test_out_of_range_center_snaps_to_nearest_grid_value(self):
        center = {"stMultiplier": 99.0, "stPeriod": 10}
        ng = build_neighborhood_grid(self.GRID, center, 1.0)
        self.assertIn(3.0, ng["stMultiplier"])
        self.assertTrue(all(1.0 <= v <= 3.0 for v in ng["stMultiplier"]))

    def test_single_value_fallback_steps(self):
        ng = build_neighborhood_grid({"stPeriod": [10]}, {"stPeriod": 10}, 2.0)
        self.assertEqual(ng["stPeriod"], [8, 9, 10, 11, 12])
        ng = build_neighborhood_grid({"stMultiplier": [2.0]}, {"stMultiplier": 2.0}, 0.2)
        self.assertEqual(ng["stMultiplier"], [1.8, 1.9, 2.0, 2.1, 2.2])


class StagedSearchEndToEndTests(unittest.TestCase):
    """Run the 2-stage pipeline against synthetic data in a temp CWD."""

    GRID = {
        "stMultiplier": [1.6, 2.0],
        "stPeriod": [8, 10],
        "atrSLmult": [1.0, 1.2],
        "atrTPmult": [2.0, 3.0],
        "emaLen": [20, 50],
    }

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="staged_test_"))
        self._old_cwd = Path.cwd()
        os.chdir(self._tmp)
        write_tsv(self._tmp / "NVDA_15m.tsv", make_candles())
        self.ticker = {"symbol": "NVDA", "timeframe": "15m", "tsv": str(self._tmp / "NVDA_15m.tsv")}

    def tearDown(self):
        os.chdir(self._old_cwd)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run(self, **overrides):
        kwargs = dict(
            ticker=self.ticker,
            grid=self.GRID,
            intrabar_paths=["ohlc"],
            top_k=3,
            time_budget=120,
            n_samples=64,
            seed=7,
            max_exhaustive=100000,
            execution={"intrabar_path": "ohlc"},
            robustness={"enabled": False},
            staged_cfg={"expand_radius": 1, "time_budget_split": [0.6, 0.4]},
        )
        kwargs.update(overrides)
        return staged_search(**kwargs)

    def test_runs_both_stages(self):
        results = self._run()
        self.assertEqual([r["phase"] for r in results], ["initial", "expanded"])

    def test_final_candidate_meets_criteria(self):
        results = self._run()
        final = results[-1]
        self.assertEqual(final["phase"], "expanded")
        self.assertTrue(final["top"])
        best = final["top"][0]
        self.assertTrue(passes_filters(best["metrics"], DEFAULT_FILTERS))

    def test_expanded_grid_centers_on_initial_winner(self):
        initial = self._run()[0]
        self.assertTrue(initial["top"])
        center = initial["top"][0]["params"]
        ng = build_neighborhood_grid(self.GRID, center, 1)
        for key, spec in self.GRID.items():
            self.assertIn(center[key], ng[key])

    def test_skips_later_stages_when_nothing_suitable(self):
        impossible = {"min_win_rate": 1.1, "min_profit_factor": 99.0, "min_net_profit": 0.0, "min_trades": 10}
        results = self._run(staged_cfg={"filters": impossible, "time_budget_split": [0.6, 0.4]})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["phase"], "initial")
        self.assertFalse(results[0]["top"])

    def test_marks_strong_candidate_found(self):
        results = self._run()
        self.assertTrue(all("strong_candidate_found" in r for r in results))
        self.assertTrue(any(r["strong_candidate_found"] for r in results))

    def test_stops_after_expanded_when_strong_bar_is_unreachable(self):
        unreachable_strong = {
            "strong_filters": {
                "min_win_rate": 0.40,
                "min_profit_factor": 1.4,
                "min_net_profit": 0.0,
                "min_trades": 10,
                "max_drawdown_pct": -1.0,
            },
            "expand_radius": 1,
            "time_budget_split": [0.6, 0.4],
        }
        results = self._run(staged_cfg=unreachable_strong)
        self.assertEqual([r["phase"] for r in results], ["initial", "expanded"])
        self.assertFalse(results[-1]["strong_candidate_found"])


class StagedConfigTests(unittest.TestCase):
    def test_defaults_present(self):
        cfg = merge_with_defaults({})
        staged = cfg["staged_search"]
        self.assertTrue(staged["enabled"])
        self.assertEqual(staged["filters"], DEFAULT_FILTERS)
        self.assertEqual(staged["strong_filters"], STRONG_FILTERS)
        self.assertIn("expand_radius", staged)
        self.assertIn("time_budget_split", staged)
        self.assertEqual(cfg["time_budget_seconds_per_ticker"], 3600)

    def test_repo_yaml_loads_staged_defaults(self):
        from optimize_all import load_config
        old_cwd = Path.cwd()
        os.chdir(REPO_ROOT)
        try:
            cfg = load_config()
        finally:
            os.chdir(old_cwd)
        staged = cfg["staged_search"]
        self.assertTrue(staged["enabled"])
        self.assertEqual(staged["filters"]["min_win_rate"], 0.40)
        self.assertEqual(staged["filters"]["min_profit_factor"], 1.4)
        self.assertEqual(staged["filters"]["min_net_profit"], 0.0)
        self.assertEqual(staged["filters"]["max_drawdown_pct"], 0.25)
        strong = staged["strong_filters"]
        self.assertEqual(strong["min_win_rate"], 0.40)
        self.assertEqual(strong["min_profit_factor"], 1.4)
        self.assertEqual(strong["min_net_profit"], 0.0)
        self.assertEqual(strong["max_drawdown_pct"], 0.25)

    def test_merge_preserves_user_overrides(self):
        cfg = merge_with_defaults({"staged_search": {"enabled": False, "expand_radius": 3.0}})
        self.assertFalse(cfg["staged_search"]["enabled"])
        self.assertEqual(cfg["staged_search"]["expand_radius"], 3.0)
        self.assertEqual(cfg["staged_search"]["filters"], DEFAULT_FILTERS)

    def test_repo_yaml_wide_initial_grid(self):
        from optimize_all import load_config
        old_cwd = Path.cwd()
        os.chdir(REPO_ROOT)
        try:
            cfg = load_config()
        finally:
            os.chdir(old_cwd)
        grid = cfg["grid_constrained"]
        self.assertEqual(grid["stMultiplier"]["start"], 1.0)
        self.assertEqual(grid["stMultiplier"]["stop"], 5.0)
        self.assertEqual(grid["stPeriod"]["min"], 6)
        self.assertEqual(grid["stPeriod"]["max"], 18)
        self.assertEqual(grid["atrSLmult"]["start"], 1.0)
        self.assertEqual(grid["atrSLmult"]["stop"], 2.6)
        self.assertEqual(grid["atrTPmult"]["start"], 1.6)
        self.assertEqual(grid["atrTPmult"]["stop"], 10.0)
        self.assertEqual(grid["emaLen"]["min"], 20)
        self.assertEqual(grid["emaLen"]["max"], 300)


if __name__ == "__main__":
    unittest.main()
