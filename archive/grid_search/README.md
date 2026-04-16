# archive/grid_search/ — Phase 2 Grid Search Results

Historical hyperparameter sweep outputs from Phase 2 strategy discovery.
**Superseded by autoresearch/ which uses a Claude agent for continuous optimization.**

| File | Params Swept | Best Result |
|------|-------------|-------------|
| `grid_search_results.csv` | CombinedStrategy: ma_short, ma_long, mr_window, mr_band, atr_stop | Phase 2 baseline |
| `grid_search_guard_results.csv` | LongOnlyGuard: vix_exit, vix_reentry, ma_long, confirm_bars | LongOnlyGuard winner |
| `grid_search_v2_results.csv` | LongOnlyGuardV2: 4 drawdown mechanisms | V2 baseline config |
| `grid_search_v3_results.csv` | V2 expanded: wider params for >30% CAGR target | Final Phase 2 config |

Current production config lives in `config/strategy_config.py`.
Ongoing optimization: `autoresearch/` (score: 0.4964 → 0.6564).
