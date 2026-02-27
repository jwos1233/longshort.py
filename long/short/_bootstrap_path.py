from pathlib import Path
import sys


def ensure_repo_root_on_path():
    """
    Make Macro_Quadrant_Strategy root importable when running from long/short/.
    This avoids duplicating the full codebase until we're ready to fork it.
    """
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

