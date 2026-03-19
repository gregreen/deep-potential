"""Project-root-aware path utilities.

No matter where Python is invoked (project root, ``notebooks/``, or any other
working directory), the helpers below resolve paths relative to the repository
root so that ``data/``, ``runs/``, ``configs/`` etc. are always reachable.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Locate project root
# ---------------------------------------------------------------------------
# Strategy: walk upward from *this* file until we find pyproject.toml
# (the canonical marker for the repo root).

def _find_project_root() -> Path:
    """Return the first ancestor directory that contains ``pyproject.toml``."""
    anchor = Path(__file__).resolve().parent  # dpjax/
    for parent in [anchor] + list(anchor.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: two levels up from dpjax/paths.py
    return anchor.parent


PROJECT_ROOT: Path = _find_project_root()

DATA_DIR: Path = PROJECT_ROOT / "data"
RUNS_DIR: Path = PROJECT_ROOT / "runs"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
PLOTS_DIR: Path = PROJECT_ROOT / "plots"


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    """Resolve *path* against *base* (default: ``PROJECT_ROOT``).

    If *path* is already absolute it is returned as-is.  Otherwise it is
    interpreted relative to *base*.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    base = base or PROJECT_ROOT
    return (base / p).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if it does not exist, then return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
