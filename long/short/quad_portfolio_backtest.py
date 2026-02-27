"""
Shim module for the long/short bootstrap project.

This file lets you run scripts from `long/short/` without copying the full
`quad_portfolio_backtest.py` module yet.

When you’re ready to diverge, we can replace this shim with a real fork.
"""

from _bootstrap_path import ensure_repo_root_on_path

ensure_repo_root_on_path()

# Re-export the production backtest implementation
from quad_portfolio_backtest import *  # type: ignore  # noqa: F401,F403,E402

