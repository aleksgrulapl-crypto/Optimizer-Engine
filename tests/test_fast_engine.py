import copy
import itertools
import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest_engine as ref
import backtest_fast as fast


def make_candles(n=1500, seed=3):
    rng = random.Random(seed)
    price = 100.0
    candles = []
    for i in range(n):
        o = price
        drift = math.sin(i / 60.0) * 0.4
        c = max(1.0, o + drift + rng.gauss(0, 0.8))
        h = max(o, c) + abs(rng.gauss(0, 0.5))
        l = min(o, c) - abs(rng.gauss(0, 0.5))
        candles.append({"time": f"t{i}", "open": o, "high": h, "low": l, "close": c, "volume": 1})
        price = c
    return candles


class FastEngineParityTests(unittest.TestCase):
    def assert_same(self, candles, params):
        expected = ref.run_backtest(copy.deepcopy(candles), params)
        actual = fast.run_backtest(candles, params)
        self.assertEqual(len(expected["trade_dicts"]), len(actual["trade_dicts"]))
        for e, a in zip(expected["trade_dicts"], actual["trade_dicts"]):
            for key, value in e.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(value, a[key], places=9, msg=key)
                else:
                    self.assertEqual(value, a[key], key)
        for name in ("tr", "atr", "up", "dn", "ema"):
            for e, a in zip(expected["indicators"][name], actual["indicators"][name]):
                if e is None:
                    self.assertIsNone(a)
                else:
                    self.assertAlmostEqual(e, a, places=9)
        for name in ("trend", "emaUp", "emaDown"):
            self.assertEqual(expected["indicators"][name], actual["indicators"][name])
        return expected["trade_dicts"]

    def test_parity_across_parameters(self):
        candles = make_candles()
        total = 0
        for mult, period, sl, tp, ema, path in itertools.product(
            (1.5, 3.0), (5, 12), (1.0, 2.0), (1.2, 4.0), (20, 100), ("ohlc", "olhc")
        ):
            params = {"ticker": "X", "timeframe": "15m", "stMultiplier": mult, "stPeriod": period,
                      "atrSLmult": sl, "atrTPmult": tp, "emaLen": ema, "intrabar_path": path,
                      "slippage": 0.01, "commission_pct": 0.001}
            total += len(self.assert_same(candles, params))
        self.assertGreater(total, 100)

    def test_parity_with_pyramiding_and_same_bar_exit(self):
        candles = make_candles(seed=9)
        for pyramiding, same_bar in itertools.product((1, 3), (False, True)):
            params = {"ticker": "X", "timeframe": "15m", "stMultiplier": 1.5, "stPeriod": 6,
                      "atrSLmult": 1.2, "atrTPmult": 2.0, "emaLen": 30, "pyramiding": pyramiding,
                      "allow_same_bar_exit_entry": same_bar, "position_size": 2.0, "slippage": 0.02}
            self.assert_same(candles, params)

    def test_lean_skips_indicators(self):
        candles = make_candles(200)
        result = fast.run_backtest_lean(candles, {"ticker": "X", "timeframe": "15m"})
        self.assertEqual(result["indicators"], {})

    def test_empty_candles(self):
        self.assertEqual(fast.run_backtest([], {}), {"trade_dicts": [], "indicators": {}})


if __name__ == "__main__":
    unittest.main()
