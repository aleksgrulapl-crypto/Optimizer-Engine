Optimizer Engine - Runbook

Core files in this repository:
- /home/runner/work/Optimizer-Engine/Optimizer-Engine/backtest_engine.py
- /home/runner/work/Optimizer-Engine/Optimizer-Engine/data_loader.py
- /home/runner/work/Optimizer-Engine/Optimizer-Engine/optimizer_worker.py
- /home/runner/work/Optimizer-Engine/Optimizer-Engine/optimize_all.py
- /home/runner/work/Optimizer-Engine/Optimizer-Engine/presets.py
- /home/runner/work/Optimizer-Engine/Optimizer-Engine/tickers.yaml
- /home/runner/work/Optimizer-Engine/Optimizer-Engine/export_trades.py
- /home/runner/work/Optimizer-Engine/Optimizer-Engine/compare_parity.py

Workflow (required order):
1) Data sanity check
2) Parity check
3) Constrained optimize
4) Expand optimize
5) Robustness check
6) Finalize presets

1) Install dependencies:
   python -m pip install pyyaml pandas pytz

2) Parity export (single ticker, fixed params):
   python export_trades.py \
     --input data/NVDA_15m.tsv \
     --output optimizer_results/parity_nvda.csv \
     --ticker NVDA \
     --intrabar-path ohlc \
     --slippage 0.0 \
     --commission-pct 0.0 \
     --position-size 1.0

3) Compare with TradingView export:
   python compare_parity.py optimizer_results/parity_nvda.csv AutoTrader_15M_NASDAQ_NVDA_2026-08-20.csv

4) If parity is acceptable, set parity_ok: true in /home/runner/work/Optimizer-Engine/Optimizer-Engine/tickers.yaml

5) Run phased optimization:
   python optimize_all.py

Outputs:
- optimizer_results/best_{SYMBOL}_constrained.csv
- optimizer_results/best_{SYMBOL}_expanded.csv
- optimizer_results/report_{SYMBOL}_constrained.json
- optimizer_results/report_{SYMBOL}_expanded.json
- optimizer_results/best_presets.csv
- optimizer_results/progress.csv
