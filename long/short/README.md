## Long/Short (bootstrap project)

This folder is a **new project area** for building a long/short version of the Macro Quadrant Strategy.

### Why these files look “copied” but still import the base
To keep this lightweight and runnable immediately, the scripts here **import the existing Macro Quadrant code** from the repo root (by adding the root to `sys.path`). This gives you:

- A clean place to start long/short work
- No duplication of large modules yet (e.g. `quad_portfolio_backtest.py`)

When you’re ready to diverge, we can replace the shims with full copies of the underlying modules.

### Run

```bash
python run_production_backtest.py --start 2025-01-01 --end 2025-11-30
```

