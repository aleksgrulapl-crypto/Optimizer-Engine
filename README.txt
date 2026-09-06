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
3) Staged search per ticker and timeframe (15m and 30m):
   a. Initial random grid search over the base grid (grid_constrained)
   b. Expanded grid around the best suitable candidate from the initial stage
   c. Refined grid around the top candidate from the expanded stage
4) Robustness check
5) Finalize presets

Candidate suitability filters (configurable via the staged_search block in
tickers.yaml): win rate >= 50%, profit factor >= 1.2, strictly positive net
profit, and a minimum trade count. Candidates are ranked by score, preferring
lower drawdown on ties. Set staged_search.enabled: false to fall back to the
legacy constrained/fallback/expanded phases.

1) Install dependencies:
   python -m pip install pyyaml pandas pytz

2) Parity export (single ticker, fixed params, use the same timeframe as tickers.yaml):
   python export_trades.py \
     --input data/NVDA_30m.tsv \
     --output optimizer_results/parity_nvda.csv \
     --ticker NVDA \
     --intrabar-path ohlc \
     --slippage 0.0 \
     --commission-pct 0.0 \
     --position-size 1.0

3) Compare with TradingView export:
   python compare_parity.py optimizer_results/parity_nvda.csv AutoTrader_30M_NASDAQ_NVDA_YYYY-MM-DD.csv

4) If parity is acceptable, set parity_ok: true in /home/runner/work/Optimizer-Engine/Optimizer-Engine/tickers.yaml

5) Run staged optimization (per ticker entry, i.e. each ticker at 15m and 30m):
   python optimize_all.py

6) Refine a single ticker after optimize_all:
   python optimize_single.py --symbol NVDA

   Optional overrides:
   - --phase refine_nvda
   - --top-k 10
   - --time-budget 2400

   This command runs a focused grid around the timeframe-specific values in
   presets.py. If both 15m and 30m TSVs are available for the symbol, refinement
   runs once per timeframe and writes timeframe-specific output files.

Outputs:
- optimizer_results/best_{SYMBOL}_initial.csv
- optimizer_results/best_{SYMBOL}_expanded.csv
- optimizer_results/best_{SYMBOL}_refined.csv
- optimizer_results/report_{SYMBOL}_initial.json
- optimizer_results/report_{SYMBOL}_expanded.json
- optimizer_results/report_{SYMBOL}_refined.json
- optimizer_results/best_presets.csv
- optimizer_results/progress.csv
- optimizer_results/best_{SYMBOL}_refine.csv        (optimize_single.py)
- optimizer_results/report_{SYMBOL}_refine.json     (optimize_single.py)
- optimizer_results/progress_single.csv             (optimize_single.py)

Tests:
   python -m unittest discover -s tests -v
