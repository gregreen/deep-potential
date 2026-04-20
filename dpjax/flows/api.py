"""Unified DF flow API — backend-agnostic factory, log_prob, score, sample.

All experiment scripts should import from here instead of directly from
``dpjax.flows.realnvp`` (or future ``dpjax.flows.ffjord``).

Usage
-----
>>> from dpjax.flows.api import build_flow, init_flow, log_prob_apply, score_apply, sample_apply, load_df
"""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import yaml
from flax import linen as nn

from dpjax.data import Normalizer
from dpjax.utils.ckpt import create_manager, restore_latest


# ── Registry ──────────────────────────────────────────────────────────
# Maps flow.type string → (build_fn, init_fn, log_prob_fn, score_fn, sample_fn)

_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_flow(name: str, *, build, init, log_prob, score, sample, log_prob_reg=None):
    """Register a flow backend under *name*."""
    _REGISTRY[name] = {
        "build": build,
        "init": init,
        "log_prob": log_prob,
        "log_prob_reg": log_prob_reg,
        "score": score,
        "sample": sample,
    }


# ── RealNVP registration (always available) ───────────────────────────

def _realnvp_build(flow_cfg: dict) -> nn.Module:
    from dpjax.flows.realnvp import RealNVP, RealNVPConfig

    return RealNVP(
        RealNVPConfig(
            dim=int(flow_cfg.get("dim", 6)),
            n_coupling=int(flow_cfg.get("n_coupling", 10)),
            hidden_sizes=tuple(int(x) for x in flow_cfg.get("hidden_sizes", [128, 128])),
            s_max=float(flow_cfg.get("s_max", 2.0)),
        )
    )


def _realnvp_init(model: nn.Module, rng: jax.Array, dim: int) -> dict:
    from dpjax.flows.realnvp import RealNVP

    dummy = jnp.zeros((1, dim), dtype=jnp.float32)
    return model.init(rng, dummy, method=RealNVP.log_prob)["params"]


def _realnvp_log_prob(model: nn.Module, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    from dpjax.flows.realnvp import log_prob_apply as _lp
    return _lp(model, params, x)


def _realnvp_score(model: nn.Module, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    from dpjax.flows.realnvp import score_apply as _sc
    return _sc(model, params, x)


def _realnvp_sample(model: nn.Module, params: dict, rng: jax.Array, n: int) -> jnp.ndarray:
    from dpjax.flows.realnvp import RealNVP
    return model.apply({"params": params}, rng, int(n), method=RealNVP.sample)


def _realnvp_log_prob_reg(model: nn.Module, params: dict, x: jnp.ndarray):
    """RealNVP has no ODE regularisation — return zeros."""
    lp = _realnvp_log_prob(model, params, x)
    return lp, jnp.zeros_like(lp)


register_flow(
    "realnvp",
    build=_realnvp_build,
    init=_realnvp_init,
    log_prob=_realnvp_log_prob,
    log_prob_reg=_realnvp_log_prob_reg,
    score=_realnvp_score,
    sample=_realnvp_sample,
)


# ── FFJORD registration ──────────────────────────────────────────────

def _ffjord_build(flow_cfg: dict) -> nn.Module:
    from dpjax.flows.ffjord import FFJORD, FFJORDConfig

    ffjord_sub = flow_cfg.get("ffjord", {})
    return FFJORD(
        FFJORDConfig(
            dim=int(flow_cfg.get("dim", 6)),
            hidden_sizes=tuple(int(x) for x in ffjord_sub.get("hidden_sizes", flow_cfg.get("hidden_sizes", [128, 128, 128]))),
            n_blocks=int(ffjord_sub.get("n_blocks", 3)),
            solver=str(ffjord_sub.get("solver", "tsit5")),
            rtol=float(ffjord_sub.get("rtol", 1e-5)),
            atol=float(ffjord_sub.get("atol", 1e-5)),
            trace_type=str(ffjord_sub.get("trace_type", "exact")),
            dt0=float(ffjord_sub.get("dt0", 0.01)),
            max_steps=int(ffjord_sub.get("max_steps", 4096)),
            kin_reg=float(ffjord_sub.get("kin_reg", 0.0)),
            jac_reg=float(ffjord_sub.get("jac_reg", 0.0)),
        )
    )


def _ffjord_init(model: nn.Module, rng: jax.Array, dim: int) -> dict:
    from dpjax.flows.ffjord import FFJORD

    dummy = jnp.zeros((1, dim), dtype=jnp.float32)
    return model.init(rng, dummy, method=FFJORD.init_only)["params"]


def _ffjord_log_prob(model: nn.Module, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    from dpjax.flows.ffjord import log_prob_apply as _lp
    return _lp(model, params, x)


def _ffjord_score(model: nn.Module, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    from dpjax.flows.ffjord import score_apply as _sc
    return _sc(model, params, x)


def _ffjord_sample(model: nn.Module, params: dict, rng: jax.Array, n: int) -> jnp.ndarray:
    from dpjax.flows.ffjord import FFJORD
    return model.apply({"params": params}, rng, int(n), method=FFJORD.sample)


def _ffjord_log_prob_reg(model: nn.Module, params: dict, x: jnp.ndarray):
    from dpjax.flows.ffjord import log_prob_reg_apply as _lpr
    return _lpr(model, params, x)


register_flow(
    "ffjord",
    build=_ffjord_build,
    init=_ffjord_init,
    log_prob=_ffjord_log_prob,
    log_prob_reg=_ffjord_log_prob_reg,
    score=_ffjord_score,
    sample=_ffjord_sample,
)


# ── Public helpers ────────────────────────────────────────────────────

def _resolve_type(flow_cfg: dict) -> str:
    """Return the flow backend name from a config dict (default ``realnvp``)."""
    return str(flow_cfg.get("type", "realnvp")).lower()


def _backend(flow_cfg: dict) -> Dict[str, Any]:
    name = _resolve_type(flow_cfg)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown flow type {name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def _infer_type_from_model(model: nn.Module) -> str | None:
    """Best-effort backend inference from model class/module/cfg type."""
    cls_name = model.__class__.__name__.lower()
    mod_name = model.__class__.__module__.lower()

    if "ffjord" in cls_name or "ffjord" in mod_name:
        return "ffjord"
    if "realnvp" in cls_name or "realnvp" in mod_name:
        return "realnvp"

    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        cfg_name = cfg.__class__.__name__.lower()
        if "ffjord" in cfg_name:
            return "ffjord"
        if "realnvp" in cfg_name:
            return "realnvp"

    return None


def _dispatch_name(model: nn.Module, flow_cfg: dict | None = None) -> str:
    """Resolve backend name, preferring model-inferred type if available."""
    inferred = _infer_type_from_model(model)

    if flow_cfg is None:
        return inferred or "realnvp"

    cfg_name = _resolve_type(flow_cfg)
    if cfg_name not in _REGISTRY:
        raise ValueError(f"Unknown flow type {cfg_name!r}. Registered: {sorted(_REGISTRY)}")

    if inferred is not None and inferred in _REGISTRY and inferred != cfg_name:
        return inferred
    return cfg_name


def build_flow(flow_cfg: dict) -> nn.Module:
    """Instantiate a flow model from *flow_cfg* (dispatches on ``flow.type``)."""
    return _backend(flow_cfg)["build"](flow_cfg)


def init_flow(model: nn.Module, rng: jax.Array, flow_cfg: dict) -> dict:
    """Initialise flow parameters and return the ``params`` dict."""
    dim = int(flow_cfg.get("dim", 6))
    name = _dispatch_name(model, flow_cfg)
    return _REGISTRY[name]["init"](model, rng, dim)


def log_prob_apply(model: nn.Module, params: dict, x: jnp.ndarray, flow_cfg: dict | None = None) -> jnp.ndarray:
    """Compute log-probability for batch *x*.

    If *flow_cfg* is ``None``, falls back to ``realnvp`` (backward compat).
    """
    name = _dispatch_name(model, flow_cfg)
    return _REGISTRY[name]["log_prob"](model, params, x)


def score_apply(model: nn.Module, params: dict, x: jnp.ndarray, flow_cfg: dict | None = None) -> jnp.ndarray:
    """Compute score ∇_x log p(x) for batch *x*.

    If *flow_cfg* is ``None``, falls back to ``realnvp`` (backward compat).
    """
    name = _dispatch_name(model, flow_cfg)
    return _REGISTRY[name]["score"](model, params, x)


def log_prob_reg_apply(
    model: nn.Module, params: dict, x: jnp.ndarray, flow_cfg: dict | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(log_probs, reg_costs)`` for batch *x*.

    The regularisation cost is per-sample and backend-specific (zero for RealNVP).
    """
    name = _dispatch_name(model, flow_cfg)
    fn = _REGISTRY[name].get("log_prob_reg")
    if fn is None:
        lp = _REGISTRY[name]["log_prob"](model, params, x)
        return lp, jnp.zeros_like(lp)
    return fn(model, params, x)


def sample_apply(
    model: nn.Module,
    params: dict,
    rng: jax.Array | int,
    n: int | dict | None = None,
    flow_cfg: dict | None = None,
) -> jnp.ndarray:
    """Draw samples from the flow.

    Supports both call styles:
    - ``sample_apply(model, params, rng, n, flow_cfg=None)``
    - ``sample_apply(model, params, n, flow_cfg=None)`` (legacy; uses a default RNG key)

    If *flow_cfg* is ``None``, falls back to ``realnvp`` (backward compat).
    """
    legacy_call = isinstance(rng, Integral) and (n is None or isinstance(n, dict))
    if legacy_call:
        rng_key = jax.random.PRNGKey(0)
        n_samples = int(rng)
        resolved_flow_cfg = n if isinstance(n, dict) else flow_cfg
    else:
        if n is None:
            raise TypeError("sample_apply requires `n` when `rng` is provided")
        rng_key = rng
        n_samples = int(n)
        resolved_flow_cfg = flow_cfg

    name = _dispatch_name(model, resolved_flow_cfg)
    return _REGISTRY[name]["sample"](model, params, rng_key, n_samples)


# ── Unified loader ────────────────────────────────────────────────────

def load_df(df_run_dir: str | Path) -> Tuple[nn.Module, dict, Normalizer, dict]:
    """Load a trained DF from *df_run_dir* (backend-agnostic).

    Returns ``(model, params, normalizer, full_config_dict)``.
    """
    df_run_dir = Path(df_run_dir)
    cfg_path = df_run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text())
    flow_cfg = cfg.get("flow", {})

    model = build_flow(flow_cfg)

    norm_path = df_run_dir / "normalizer.npz"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing {norm_path}")
    norm = Normalizer.load_npz(norm_path)

    ckpt_mgr = create_manager(df_run_dir / "ckpt")
    restored = restore_latest(ckpt_mgr)
    params = restored["params"]

    return model, params, norm, cfg
