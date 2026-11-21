import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest_daybox as db
import optimize_all as oa
import optimizer_worker as ow


def _candle(t, o, h, l, c):
    return {"time": t, "open": o, "high": h, "low": l, "close": c}


def _day(start_utc, bars):
    """Candles one hour apart starting 15:00 UTC (10-11am New York)."""
    return [_candle(start_utc + timedelta(hours=i), *bar) for i, bar in enumerate(bars)]


D1 = datetime(2026, 3, 2, 15, tzinfo=timezone.utc)
D2 = D1 + timedelta(days=1)


class DayBoxEngineTests(unittest.TestCase):
    def run_bt(self, candles, **params):
        return db.run_backtest(candles, {"ticker": "X", **params})["trade_dicts"]

    def test_no_trades_without_a_previous_day(self):
        candles = _day(D1, [(100, 101, 99, 100)] * 5)
        self.assertEqual(self.run_bt(candles), [])

    def test_rejection_long_hits_take_profit(self):
        # Day 1 box: high 110, low 100. Day 2 bar 0 dips to the low and closes inside.
        d1 = _day(D1, [(105, 110, 100, 105), (105, 108, 102, 106)])
        d2 = _day(D2, [(103, 104, 100, 103), (103, 109, 103, 108)])
        trades = self.run_bt(d1 + d2, enableBreakouts=False, slBufferPct=0.0, riskReward=2.0)
        self.assertEqual(len(trades), 1)
        t = trades[0]
        # entry at close 103, stop = min(low 100, box low 100) = 100, risk 3, target 109
        self.assertEqual((t["side"], t["entry_price"], t["exit_reason"]), ("long", 103.0, "tp"))
        self.assertAlmostEqual(t["exit_price"], 109.0)
        self.assertAlmostEqual(t["pnl"], 6.0)
        self.assertEqual((t["entry_bar_index"], t["exit_bar_index"]), (2, 3))

    def test_rejection_short_hits_stop_on_gap(self):
        d1 = _day(D1, [(105, 110, 100, 105), (105, 108, 102, 106)])
        d2 = _day(D2, [(107, 110, 106, 107), (112, 113, 111, 112)])
        trades = self.run_bt(d1 + d2, enableBreakouts=False, slBufferPct=0.0, riskReward=2.0,
                             enableLongs=False)
        self.assertEqual(len(trades), 1)
        t = trades[0]
        # stop = max(high 110, box high 110) = 110; gap open 112 fills at the open
        self.assertEqual((t["side"], t["exit_reason"]), ("short", "sl"))
        self.assertAlmostEqual(t["exit_price"], 112.0)
        self.assertLess(t["pnl"], 0)

    def test_breakout_retest_long_requires_later_bar(self):
        d1 = _day(D1, [(105, 110, 100, 105), (105, 108, 102, 106)])
        # bar 0 breaks out (close 112 > 110); bar 1 retests the high and closes above.
        d2 = _day(D2, [(108, 113, 108, 112), (112, 113, 110, 111), (111, 130, 111, 125)])
        trades = self.run_bt(d1 + d2, enableRejections=False, slBufferPct=0.0, riskReward=2.0)
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual((t["side"], t["entry_bar_index"]), ("long", 3))
        # stop = retest bar low 110, risk 1, target 113 hit on the next bar
        self.assertEqual(t["exit_reason"], "tp")
        self.assertAlmostEqual(t["exit_price"], 113.0)

    def test_day_ids_follow_exchange_timezone(self):
        # 23:30 and 00:30 UTC on the same New York date -> one day, not two.
        candles = [_candle(datetime(2026, 3, 2, 23, 30, tzinfo=timezone.utc), 1, 1, 1, 1),
                   _candle(datetime(2026, 3, 3, 0, 30, tzinfo=timezone.utc), 1, 1, 1, 1)]
        ids = db.run_backtest and db._DayBoxArrays(candles).day_ids("America/New_York", 0)
        self.assertEqual(ids[0], ids[1])
        utc_ids = db._DayBoxArrays(candles).day_ids("UTC", 0)
        self.assertNotEqual(utc_ids[0], utc_ids[1])

    def test_empty_candles(self):
        self.assertEqual(db.run_backtest([], {}), {"trade_dicts": [], "indicators": {}})


class StrategySeparationTests(unittest.TestCase):
    def test_daybox_paths_are_namespaced(self):
        self.assertEqual(ow.strategy_dir("optimizer_results"), Path("optimizer_results"))
        self.assertEqual(ow.strategy_dir("optimizer_results", "daybox"),
                         Path("optimizer_results") / "daybox")
        with self.assertRaises(ValueError):
            ow.normalize_strategy("nope")

    def test_apply_strategy_swaps_grid_and_merges_overrides(self):
        base = oa.merge_with_defaults({"daybox": {"execution": {"intrabar_path": "tv"},
                                                  "params": {"enableShorts": False}}})
        day = oa.apply_strategy(base, "daybox")
        self.assertEqual(set(day["grid"]), set(oa.DAYBOX_DEFAULT_GRID))
        self.assertEqual(day["execution"]["intrabar_path"], "tv")
        self.assertEqual(day["execution"]["pyramiding"], 1)
        self.assertEqual(day["strategy_params"], {"enableShorts": False})
        sup = oa.apply_strategy(base, "supertrend")
        self.assertIn("stMultiplier", sup["grid"])
        self.assertEqual(sup["execution"]["intrabar_path"], "ohlc")

    def test_daybox_worker_writes_only_to_daybox_folders(self):
        candles = _day(D1, [(105, 110, 100, 105)] * 2) + _day(D2, [(103, 104, 100, 103)] * 3)
        old = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                with mock.patch.object(ow, "load_candles_from_csv", return_value=candles):
                    ow.optimize_ticker(
                        {"symbol": "X", "timeframe": "15m", "tsv": "x.tsv"},
                        {"riskReward": [1.0, 2.0]}, ["ohlc"], search_mode="randomized",
                        phase="grid", strategy="daybox", filters={"min_trades": 0},
                        fixed_params={"enableBreakouts": False},
                    )
                found = {p.relative_to(tmp).parts[0] for p in Path(tmp).rglob("*") if p.is_file()}
                self.assertTrue({"optimizer_results", "checkpoints"} <= found)
                for p in Path(tmp).rglob("*"):
                    if p.is_file():
                        self.assertEqual(p.relative_to(tmp).parts[1], "daybox", p)
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
