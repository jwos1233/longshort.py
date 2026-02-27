"""
Shim module for the long/short bootstrap project.

Imports the existing `live_trader_simple.py` from the Macro_Quadrant_Strategy root.
"""

from _bootstrap_path import ensure_repo_root_on_path

ensure_repo_root_on_path()

from live_trader_simple import *  # type: ignore  # noqa: F401,F403,E402

