from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any
import warnings

import orbax.checkpoint as ocp


def create_manager(ckpt_dir: str | Path, *, max_to_keep: int = 3) -> ocp.CheckpointManager:
    ckpt_dir = Path(ckpt_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # If a previous run crashed mid-save, Orbax can leave temporary directories
    # like "123.orbax-checkpoint-tmp" which are not restorable checkpoints.
    # Clean them up so a new run can proceed and `latest_step()` behaves.
    for child in ckpt_dir.iterdir():
        if child.is_dir() and child.name.endswith(".orbax-checkpoint-tmp"):
            shutil.rmtree(child, ignore_errors=True)

    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    # New API (Orbax >= 0.10): use item_handlers, do not pass `checkpointers`.
    return ocp.CheckpointManager(str(ckpt_dir), item_handlers=ocp.PyTreeCheckpointHandler(), options=options)


def save(manager: ocp.CheckpointManager, step: int, item: Any) -> None:
    manager.save(step, item)


def finalize(manager: ocp.CheckpointManager) -> None:
    """Flush/close background checkpoint workers (best-effort).

    Orbax may use async workers; if the process exits immediately after a save,
    you can see noisy shutdown errors like "cannot schedule new futures after shutdown".
    Calling this at the end of a script avoids that.
    """

    wait = getattr(manager, "wait_until_finished", None)
    if callable(wait):
        wait()

    close = getattr(manager, "close", None)
    if callable(close):
        close()


def restore_latest(manager: ocp.CheckpointManager) -> Any:
    step = manager.latest_step()
    if step is None:
        raise FileNotFoundError(f"No checkpoints found in {manager.directory!r}.")
    return restore_step(manager, step)


def restore_step(manager: ocp.CheckpointManager, step: int) -> Any:
    # Orbax may warn about missing sharding info on restore; it's harmless for
    # single-host evaluation/training and just adds noise to logs.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Sharding info not provided when restoring\..*",
            category=UserWarning,
        )
        try:
            return manager.restore(step)
        except ValueError as e:
            msg = str(e)
            if "sharding passed to deserialization" in msg and "Got None" in msg:
                raise RuntimeError(
                    "Orbax restore failed due to missing/invalid sharding. This often happens when JAX "
                    "cannot initialize the intended platform (e.g. GPU OOM at startup) or when running "
                    "with a different platform than the one available.\n"
                    "Fix: ensure GPU is available (or set env `XLA_PYTHON_CLIENT_PREALLOCATE=false`), "
                    "or force CPU via `JAX_PLATFORM_NAME=cpu` for a quick smoke run."
                ) from e
            raise
