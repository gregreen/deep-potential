"""Configuration loading and merging utilities.

Provides a thin wrapper around YAML config files so that both CLI scripts
and Jupyter notebooks can load / override hyper-parameters in a uniform way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover – ruamel fallback
    from ruamel.yaml import YAML as _YAML
    import io as _io

    class _YAMLCompat:
        """Minimal shim so ``yaml.safe_load`` / ``yaml.safe_dump`` work."""

        @staticmethod
        def safe_load(stream):
            return _YAML(typ="safe").load(stream)

        @staticmethod
        def safe_dump(data, **kw):
            buf = _io.StringIO()
            _YAML(typ="safe").dump(data, buf)
            return buf.getvalue()

    yaml = _YAMLCompat()  # type: ignore[assignment]

from dpjax.paths import CONFIGS_DIR, resolve_path


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file and return it as a plain dict.

    *path* is resolved via :func:`dpjax.paths.resolve_path` so that
    relative paths like ``"configs/df_plummer.yaml"`` work from any CWD.
    """
    p = resolve_path(path)
    if not p.exists():
        # Also try looking inside CONFIGS_DIR
        alt = CONFIGS_DIR / Path(path).name
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"Config not found: {p}")
    return yaml.safe_load(p.read_text()) or {}


def merge_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *overrides* into a copy of *base*.

    Useful when a Notebook user wants to tweak a few hyper-parameters on top
    of a standard YAML config::

        cfg = load_config("configs/df_plummer.yaml")
        cfg = merge_config(cfg, {"train": {"epochs": 4, "batch_size": 256}})
    """
    merged = dict(base)
    for key, val in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = merge_config(merged[key], val)
        else:
            merged[key] = val
    return merged
