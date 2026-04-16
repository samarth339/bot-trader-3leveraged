# scripts/dev/ — Phase 1 & 2 Development Scripts

These scripts were used during strategy research and parameter discovery.
They are **not part of the production pipeline** — kept for reference only.

| Script | Phase | Purpose |
|--------|-------|---------|
| `strategy_runner.py` | Phase 2 | Compare all candidate strategies (MomentumROC, MeanReversion, Combined, etc.) |
| `grid_search.py` | Phase 2 | Grid search for CombinedStrategy parameters |
| `grid_search_guard.py` | Phase 2 | Grid search for LongOnlyGuardStrategy |
| `grid_search_v2.py` | Phase 2 | Grid search for LongOnlyGuardV2 (4 drawdown mechanisms) |
| `grid_search_v3.py` | Phase 2 | Expanded grid search targeting CAGR >30% |
| `fine_tune.py` | Phase 2 | Micro-optimization of VIX thresholds, allocations, MA window |

## Usage
Run from the project root:
```bash
python3 scripts/dev/strategy_runner.py --no-chart
python3 scripts/dev/grid_search_v3.py
```

## Results
Grid search output CSVs are archived in `archive/grid_search/`.
