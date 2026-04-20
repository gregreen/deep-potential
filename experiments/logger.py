"""Unified experiment logger with W&B, TensorBoard, and CSV backends.

Usage
-----
>>> logger = ExperimentLogger(run_dir, project="dp-plummer", backend="wandb+tb", config=cfg)
>>> logger.log_scalars(step, {"loss": 0.1, "lr": 1e-3})
>>> logger.finish()

Backend strings: ``"wandb"``, ``"tensorboard"`` (or ``"tb"``), ``"wandb+tb"``, ``"csv"``.
If a requested backend package is not installed the logger silently falls back to CSV-only.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


# ---------------------------------------------------------------------------
# Thin backend wrappers
# ---------------------------------------------------------------------------

class _CSVBackend:
    """Append-only CSV log – always available."""

    def __init__(self, run_dir: Path, fieldnames: Sequence[str] = ()):
        self._path = run_dir / "metrics.csv"
        self._fieldnames = list(fieldnames)
        self._file = None
        self._writer = None

    # lazy open so header can be set on first log_scalars call
    def _ensure_open(self, keys: Sequence[str]) -> None:
        if self._writer is not None:
            return
        if not self._fieldnames:
            self._fieldnames = ["step"] + sorted(k for k in keys if k != "step")
        self._file = self._path.open("a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames, extrasaction="ignore")
        if self._path.stat().st_size == 0:
            self._writer.writeheader()

    def log_scalars(self, step: int, metrics: Dict[str, Any]) -> None:
        row = {"step": step, **metrics}
        self._ensure_open(list(row.keys()))
        self._writer.writerow(row)  # type: ignore[union-attr]
        self._file.flush()  # type: ignore[union-attr]

    def log_image(self, step: int, tag: str, fig_or_path: Any) -> None:
        pass  # CSV backend ignores images

    def finish(self) -> None:
        if self._file is not None:
            self._file.close()


class _WandbBackend:
    """Weights & Biases backend (optional dependency)."""

    def __init__(
        self,
        run_dir: Path,
        project: str,
        run_name: Optional[str],
        config: Optional[Dict[str, Any]],
    ):
        import wandb  # noqa: F811

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            name=run_name or run_dir.name,
            dir=str(run_dir),
            config=config or {},
            reinit=True,
        )

    def log_scalars(self, step: int, metrics: Dict[str, Any]) -> None:
        self._run.log(metrics, step=step)

    def log_image(self, step: int, tag: str, fig_or_path: Any) -> None:
        if isinstance(fig_or_path, (str, Path)):
            self._run.log({tag: self._wandb.Image(str(fig_or_path))}, step=step)
        else:
            # assume matplotlib figure
            self._run.log({tag: self._wandb.Image(fig_or_path)}, step=step)

    def finish(self) -> None:
        self._run.finish()


class _TensorBoardBackend:
    """TensorBoard backend via ``torch.utils.tensorboard`` or ``tensorboardX``."""

    def __init__(self, run_dir: Path):
        log_dir = run_dir / "tb_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from tensorboardX import SummaryWriter  # type: ignore[no-redef]
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalars(self, step: int, metrics: Dict[str, Any]) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(k, v, global_step=step)

    def log_image(self, step: int, tag: str, fig_or_path: Any) -> None:
        if isinstance(fig_or_path, (str, Path)):
            import numpy as np
            from PIL import Image

            img = np.array(Image.open(str(fig_or_path)))
            self._writer.add_image(tag, img, global_step=step, dataformats="HWC")
        else:
            self._writer.add_figure(tag, fig_or_path, global_step=step)

    def finish(self) -> None:
        self._writer.close()


# ---------------------------------------------------------------------------
# Public unified logger
# ---------------------------------------------------------------------------

_BACKEND_ALIASES = {
    "tb": "tensorboard",
    "wandb+tensorboard": "wandb+tb",
    "tensorboard+wandb": "wandb+tb",
    "tb+wandb": "wandb+tb",
}


class ExperimentLogger:
    """Multiplexing logger that fans out to one or more backends.

    Parameters
    ----------
    run_dir : Path
        Directory for this experiment run (checkpoints, logs, etc.).
    project : str
        W&B project name (ignored for non-wandb backends).
    run_name : str | None
        Optional human-readable run name for W&B.
    backend : str
        One of ``"wandb"``, ``"tensorboard"`` / ``"tb"``, ``"wandb+tb"``, ``"csv"``.
    config : dict | None
        Hyperparameter dict logged to W&B at init time.
    csv_fieldnames : list[str]
        Column names for the CSV backend.  If empty, inferred from the first
        ``log_scalars`` call.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        project: str = "dp-plummer",
        run_name: Optional[str] = None,
        backend: str = "csv",
        config: Optional[Dict[str, Any]] = None,
        csv_fieldnames: Sequence[str] = (),
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._backends: list = []

        backend = _BACKEND_ALIASES.get(backend.lower().strip(), backend.lower().strip())

        want_wandb = "wandb" in backend
        want_tb = "tensorboard" in backend or "tb" in backend
        want_csv = backend == "csv" or want_wandb or want_tb  # always keep csv

        if want_csv:
            self._backends.append(_CSVBackend(self.run_dir, csv_fieldnames))

        if want_wandb:
            try:
                self._backends.append(
                    _WandbBackend(self.run_dir, project, run_name, config)
                )
            except Exception as exc:
                warnings.warn(f"W&B init failed ({exc}); falling back to CSV-only.")

        if want_tb:
            try:
                self._backends.append(_TensorBoardBackend(self.run_dir))
            except Exception as exc:
                warnings.warn(f"TensorBoard init failed ({exc}); falling back to CSV-only.")

        # Save config snapshot
        if config is not None:
            cfg_path = self.run_dir / "config.json"
            cfg_path.write_text(json.dumps(config, indent=2, default=str))

    def log_scalars(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log scalar metrics to all active backends."""
        for b in self._backends:
            b.log_scalars(step, metrics)

    def log_image(self, step: int, tag: str, fig_or_path: Any) -> None:
        """Log an image (matplotlib figure or file path) to supporting backends."""
        for b in self._backends:
            b.log_image(step, tag, fig_or_path)

    def finish(self) -> None:
        """Flush and close all backends."""
        for b in self._backends:
            b.finish()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()
