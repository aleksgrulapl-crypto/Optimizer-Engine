# check_candles_range.py
from data_loader import load_candles_from_csv
candles = load_candles_from_csv("data/NVDA_ETH_15m.tsv")
print("Candles:", len(candles))
print("First candle:", candles[0]["time"] if "time" in candles[0] else candles[0])
print("Last candle:", candles[-1]["time"] if "time" in candles[-1] else candles[-1])
