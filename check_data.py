"""Check that every ticker in tickers.yaml has loadable data.

Usage: python check_data.py [SYMBOL ...]
"""
import sys

from data_loader import load_candles_from_csv
from optimize_all import _sanity_check_ticker, load_yaml_config

EXPECTED_STEP_MINUTES = {"15m": 15, "30m": 30, "60m": 60}


def main(symbols):
    tickers = load_yaml_config("tickers.yaml").get("tickers", []) or []
    wanted = {s.upper() for s in symbols}
    bad = 0
    print(f"{'symbol':<7}{'tf':<5}{'rows':>8}  {'first':<20}{'last':<20}{'gaps':>6}  status")
    for t in tickers:
        sym = str(t.get("symbol", "")).upper()
        if wanted and sym not in wanted:
            continue
        tf = str(t.get("timeframe", "")).lower()
        ok, note = _sanity_check_ticker(t)
        if not ok:
            bad += 1
            print(f"{sym:<7}{tf:<5}{'':>8}  {'':<20}{'':<20}{'':>6}  FAIL {note}")
            continue
        candles = load_candles_from_csv(t["tsv"])
        times = [c["time"] for c in candles]
        problems = []
        if times != sorted(times):
            problems.append("not sorted")
        if len(set(times)) != len(times):
            problems.append("duplicate timestamps")
        if any(c["high"] < c["low"] for c in candles):
            problems.append("high<low")
        if any(c["close"] <= 0 for c in candles):
            problems.append("non-positive price")
        step = EXPECTED_STEP_MINUTES.get(tf, 0)
        gaps = 0
        if step:
            for a, b in zip(times, times[1:]):
                if (b - a).total_seconds() > 4 * 24 * 3600:
                    gaps += 1
        status = "OK" if not problems else "WARN " + ", ".join(problems)
        bad += bool(problems)
        print(f"{sym:<7}{tf:<5}{len(candles):>8}  {str(times[0])[:19]:<20}{str(times[-1])[:19]:<20}{gaps:>6}  {status}")
    print(f"\n{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
