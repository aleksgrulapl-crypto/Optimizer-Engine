import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimize_single import REFINEMENT_FILTERS, _build_refinement_grid
from optimizer_worker import passes_filters


class RefinementGridTests(unittest.TestCase):
    def test_builds_tight_grid_around_presets(self):
        grid = _build_refinement_grid({
            "stMultiplier": 2.8,
            "stPeriod": 10,
            "atrSLmult": 1.4,
            "atrTPmult": 4.9,
            "emaLen": 140,
        })
        self.assertEqual(grid["stMultiplier"], [2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2])
        self.assertEqual(grid["stPeriod"], [9, 10, 11])
        self.assertEqual(grid["atrSLmult"], [1.2, 1.3, 1.4, 1.5, 1.6])
        self.assertEqual(grid["atrTPmult"], [4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4])
        self.assertEqual(grid["emaLen"], list(range(130, 151)))

    def test_refinement_grid_clamps_small_values(self):
        grid = _build_refinement_grid({
            "stMultiplier": 0.3,
            "stPeriod": 2,
            "atrSLmult": 0.3,
            "atrTPmult": 0.7,
            "emaLen": 8,
        })
        self.assertEqual(grid["stMultiplier"], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        self.assertEqual(grid["stPeriod"], [2, 3])
        self.assertEqual(grid["atrSLmult"], [0.2, 0.3, 0.4, 0.5])
        self.assertEqual(grid["atrTPmult"], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        self.assertEqual(grid["emaLen"], list(range(5, 19)))


class RefinementFilterTests(unittest.TestCase):
    def test_passes_when_all_refinement_thresholds_are_met(self):
        metrics = {
            "net_profit": 100.0,
            "profit_factor": 1.4,
            "win_rate": 0.50,
            "trade_count": 25,
            "max_drawdown_pct": 0.25,
        }
        self.assertTrue(passes_filters(metrics, REFINEMENT_FILTERS))

    def test_rejects_when_drawdown_exceeds_cap(self):
        metrics = {
            "net_profit": 100.0,
            "profit_factor": 1.6,
            "win_rate": 0.55,
            "trade_count": 25,
            "max_drawdown_pct": 0.2501,
        }
        self.assertFalse(passes_filters(metrics, REFINEMENT_FILTERS))


if __name__ == "__main__":
    unittest.main()
