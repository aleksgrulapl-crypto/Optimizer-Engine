# show_first_trades.py
import pandas as pd
tv = pd.read_csv("AutoTrader_15M_NASDAQ_NVDA_2026-08-20.csv", encoding="utf-8-sig", dayfirst=True)
tv.columns = [c.strip().lower().replace(" ", "_") for c in tv.columns]
tv["trade_number"] = pd.to_numeric(tv["trade_number"], errors="coerce").astype("Int64")
pairs = []
for tn, g in tv.groupby("trade_number"):
    entry = g[g["type"].str.lower().str.contains("entry", na=False)].iloc[0]
    exit_ = g[g["type"].str.lower().str.contains("exit", na=False)].iloc[0]
    pairs.append((tn, entry["date_and_time"], entry.get("price_usd"), exit_["date_and_time"], exit_.get("price_usd")))
print("First 10 TV trade pairs:")
for p in pairs[:10]:
    print(p)

our = pd.read_csv("nvda_test_tvparams.csv", parse_dates=["entry_time","exit_time"], infer_datetime_format=True)
print("\nFirst 10 our trades:")
print(our.head(10).to_string(index=False))
