README - quick run and parity checklist

1) Install dependencies (if not already):
   python -m pip install pyyaml pandas pytz

2) Place your tickers.yaml in repo root (or tickers.json).
   Example tickers.yaml already discussed.

3) Run the optimizer:
   python optimize_all.py

4) Per-ticker outputs:
   optimizer_results/best_{SYMBOL}.csv
   optimizer_results/report_{SYMBOL}.json

5) Parity check:
   - Export TradingView trades for a symbol (CSV).
   - Save engine trades for that symbol: open optimizer_results/report_{SYMBOL}.json and extract 'res' if needed.
   - Run:
       python compare_parity.py engine_trades.json tradingview.csv
     (engine_trades.json should be a JSON array of trade dicts; you can create it from the 'res' object.)

6) If parity mismatches:
   - Run a single backtest for the preset and paste the first 10 engine trades and the first 10 TV CSV rows into the chat.
   - I will analyze and produce a targeted fix.

Notes:
- Scoring: profit-first, modest drawdown penalty, PF rewarded, WR secondary.
- If you want stronger drawdown penalty or parity-first optimization, tell me and I will regenerate optimizer_worker.py with new weights.
