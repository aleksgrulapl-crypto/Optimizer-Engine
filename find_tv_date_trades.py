# find_tv_date_trades.py
import pandas as pd
our = pd.read_csv("nvda_eth_trades.csv", parse_dates=["entry_time","exit_time"])
mask = (our["entry_time"].dt.date == pd.to_datetime("2026-01-05").date()) | (our["exit_time"].dt.date == pd.to_datetime("2026-01-05").date())
print("Trades touching 2026-01-05:", our[mask].to_string(index=False))
