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
3) Per-symbol preset check for the available 15m and 30m entries
4) Staged search per ticker and timeframe:
   a. Initial random grid search over a wide, loose base grid
      (grid_constrained: stMultiplier 1-5, stPeriod 6-18, atrSLmult 1-2.6,
      atrTPmult 1.6-10, emaLen 20-300), seeded per ticker/timeframe so
      parallel workers explore different grid points and take pressure off
      the CPU
   b. Expanded grid narrowed around the best suitable candidate from the
      initial stage
   c. Interactive gate: review the current results and decide whether to move
      to the next ticker or keep widening the expanded search
5) Robustness check
6) Finalize presets

Each ticker/timeframe entry gets about an hour of budget (configurable via
time_budget_seconds_per_ticker, i.e. roughly 30m per 15M preset and 30m per
30M preset for a ticker). optimize_all.py now runs one symbol at a time,
checks both timeframe presets first, and then asks whether the current
candidates are suitable before it will move on. The acceptance bar is
positive net profit, profit factor >= 1.4, win rate >= 40%, and a reasonably
low max drawdown (<= 25% by default).

Candidate suitability filters and the gating summary thresholds are
configurable via the staged_search block in tickers.yaml (filters /
strong_filters). Candidates are ranked by score, preferring lower drawdown on
ties. Set
staged_search.enabled: false to fall back to the legacy
constrained/fallback/expanded phases.

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
   runs once per timeframe and writes timeframe-specific output files. The
   refinement grid narrows around each preset with these fixed windows:
   stMultiplier ±0.4 by 0.1, stPeriod ±1 by 1, atrSLmult ±0.2 by 0.1,
   atrTPmult ±0.5 by 0.1, and emaLen ±10 by 1. Single-ticker refinement keeps
   only candidates with positive net profit, win rate >= 50%, profit factor >=
   1.4, at least 10 trades, and max drawdown <= 25%.

Outputs:
- optimizer_results/best_{SYMBOL}_preset.csv
- optimizer_results/best_{SYMBOL}_initial.csv
- optimizer_results/best_{SYMBOL}_expanded.csv
- optimizer_results/report_{SYMBOL}_preset.json
- optimizer_results/report_{SYMBOL}_initial.json
- optimizer_results/report_{SYMBOL}_expanded.json
- optimizer_results/best_presets.csv
- optimizer_results/progress.csv
- optimizer_results/best_{SYMBOL}_refine.csv        (optimize_single.py)
- optimizer_results/report_{SYMBOL}_refine.json     (optimize_single.py)
- optimizer_results/progress_single.csv             (optimize_single.py)

Tests:
   python -m unittest discover -s tests -v
