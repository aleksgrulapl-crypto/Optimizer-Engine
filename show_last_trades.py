import pandas as pd

df = pd.read_csv("nvda_eth_trades.csv")
print(df.tail(5).to_string())
