# show_trade_ranges.py
import pandas as pd
tv = pd.read_csv("AutoTrader_15M_NASDAQ_NVDA_2026-08-20.csv", encoding="utf-8-sig", dayfirst=True)
our = pd.read_csv("nvda_eth_trades.csv", parse_dates=["entry_time","exit_time"])
print("TV first row time:", tv.iloc[0].get("Date and time") or tv.iloc[0].get("date_and_time"))
print("TV last row time:", tv.iloc[-1].get("Date and time") or tv.iloc[-1].get("date_and_time"))
print("Our first trade entry_time:", our.iloc[0]["entry_time"])
print("Our last trade entry_time:", our.iloc[-1]["entry_time"])
