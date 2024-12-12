from pathlib import Path
import sys


def find_project_root(marker: str = "pyproject.toml") -> Path:
    """Locate the root directory by looking for a marker file."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Marker file '{marker}' not found in any parent directory")


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))