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
            "maxEntryCandleATR": 1.5,
        })
        self.assertEqual(grid["maxEntryCandleATR"], [round(0.1 * k, 1) for k in range(1, 36)])
        self.assertEqual(len(grid["stMultiplier"]), 21)
        self.assertEqual((grid["stMultiplier"][0], grid["stMultiplier"][-1]), (1.8, 3.8))
        self.assertEqual(grid["stPeriod"], [9, 10, 11])
        self.assertEqual((grid["atrSLmult"][0], grid["atrSLmult"][-1], len(grid["atrSLmult"])), (0.9, 1.9, 11))
        self.assertEqual((grid["atrTPmult"][0], grid["atrTPmult"][-1], len(grid["atrTPmult"])), (4.4, 5.4, 11))
        self.assertEqual(grid["emaLen"], list(range(130, 151)))

    def test_refinement_grid_clamps_small_values(self):
        grid = _build_refinement_grid({
            "stMultiplier": 0.3,
            "stPeriod": 2,
            "atrSLmult": 0.3,
            "atrTPmult": 0.7,
            "emaLen": 8,
        })
        self.assertEqual(grid["stMultiplier"][:3], [0.2, 0.3, 0.4])
        self.assertEqual(grid["stPeriod"], [2, 3])
        self.assertEqual(grid["atrSLmult"], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        self.assertEqual(grid["atrTPmult"], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        self.assertEqual(grid["emaLen"], list(range(5, 19)))


class RefinementFilterTests(unittest.TestCase):
    def test_passes_when_all_refinement_thresholds_are_met(self):
        metrics = {
            "net_profit": 100.0,
            "profit_factor": 1.5,
            "win_rate": 0.60,
            "trade_count": 100,
            "max_drawdown_pct": 0.25,
        }
        self.assertTrue(passes_filters(metrics, REFINEMENT_FILTERS))

    def test_rejects_when_drawdown_exceeds_cap(self):
        metrics = {
            "net_profit": 100.0,
            "profit_factor": 1.6,
            "win_rate": 0.65,
            "trade_count": 100,
            "max_drawdown_pct": 0.2501,
        }
        self.assertFalse(passes_filters(metrics, REFINEMENT_FILTERS))


if __name__ == "__main__":
    unittest.main()
