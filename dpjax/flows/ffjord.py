"""FFJORD (Free-Form Jacobian of Reversible Dynamics) — JAX/Flax implementation.

A continuous normalizing flow (CNF) that parameterises the velocity field
with an MLP and computes the log-density change via the instantaneous
change-of-variables formula.

Requires ``diffrax`` for ODE solving::

    pip install diffrax

Usage through the unified API
-----------------------------
>>> from dpjax.flows.api import build_flow, log_prob_apply, score_apply
>>> flow_cfg = {"type": "ffjord", "dim": 6, "hidden_sizes": [64, 64, 64]}
>>> model = build_flow(flow_cfg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


# ── Configuration ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class FFJORDConfig:
    dim: int = 6
    hidden_sizes: tuple[int, ...] = (128, 128, 128)
    n_blocks: int = 3
    solver: str = "tsit5"
    rtol: float = 1e-5
    atol: float = 1e-5
    trace_type: str = "exact"   # "exact" or "hutchinson"
    dt0: float = 0.01
    max_steps: int = 4096
    # Finlay et al. (2020) regularisation — 0 disables
    kin_reg: float = 0.0   # kinetic energy  ||f(t,x)||^2
    jac_reg: float = 0.0   # Jacobian Frobenius  ||∂f/∂x||_F^2


# ── Velocity field MLP ────────────────────────────────────────────────

class VelocityField(nn.Module):
    """MLP mapping ``(t, x) → dx/dt``."""
    hidden_sizes: Sequence[int]
    dim: int

    @nn.compact
    def __call__(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        # t: scalar, x: (dim,)
        t_feat = jnp.atleast_1d(jnp.asarray(t, dtype=x.dtype))
        inp = jnp.concatenate([t_feat, x])
        h = inp
        for i, width in enumerate(self.hidden_sizes):
            h = nn.Dense(width, name=f"dense_{i}")(h)
            h = nn.tanh(h)
        out = nn.Dense(
            self.dim,
            name="dense_out",
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(h)
        return out


# ── Trace estimators ──────────────────────────────────────────────────

def _exact_divergence(f_fn, x: jnp.ndarray) -> jnp.ndarray:
    """Compute divergence ``tr(∂f/∂x)`` exactly using dim forward-mode JVPs."""
    dim = x.shape[-1]
    eye = jnp.eye(dim, dtype=x.dtype)

    def _jvp_diag(ei):
        _, tangent = jax.jvp(f_fn, (x,), (ei,))
        return jnp.dot(ei, tangent)

    return jnp.sum(jax.vmap(_jvp_diag)(eye))


def _hutchinson_divergence(f_fn, x: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    """Estimate divergence via Hutchinson's trick: ``ε^T (∂f/∂x) ε``."""
    _, jvp_val = jax.jvp(f_fn, (x,), (eps,))
    return jnp.sum(eps * jvp_val)


# ── Combined div + Jacobian Frobenius (for regularisation) ───────────

def _exact_div_and_jac_norm(f_fn, x: jnp.ndarray):
    """Return ``(div, ||J||_F^2)`` reusing the same JVPs."""
    dim = x.shape[-1]
    eye = jnp.eye(dim, dtype=x.dtype)

    def _jvp_col(ei):
        _, tangent = jax.jvp(f_fn, (x,), (ei,))
        return tangent                       # full J @ e_i

    cols = jax.vmap(_jvp_col)(eye)           # (dim, dim)
    div = jnp.trace(cols)                    # tr(J)
    jac_frob_sq = jnp.sum(cols ** 2)         # ||J||_F^2
    return div, jac_frob_sq


def _hutchinson_div_and_jac_norm(f_fn, x: jnp.ndarray, eps: jnp.ndarray):
    """Return ``(div_est, ||J||_F^2_est)`` via Hutchinson."""
    _, jvp_val = jax.jvp(f_fn, (x,), (eps,))
    div_est = jnp.sum(eps * jvp_val)
    jac_frob_sq_est = jnp.sum(jvp_val ** 2)  # E[||J eps||^2] = ||J||_F^2
    return div_est, jac_frob_sq_est


# ── Standard-normal helpers ───────────────────────────────────────────

def _standard_normal_log_prob_single(z: jnp.ndarray) -> jnp.ndarray:
    """Log-prob of *z* under a standard multivariate normal (unbatched)."""
    dim = z.shape[-1]
    return -0.5 * (jnp.sum(z ** 2) + dim * jnp.log(2.0 * jnp.pi))


def _standard_normal_log_prob(z: jnp.ndarray) -> jnp.ndarray:
    """Batched version — *z* has shape ``(N, dim)``."""
    dim = z.shape[-1]
    return -0.5 * (jnp.sum(z ** 2, axis=-1) + dim * jnp.log(2.0 * jnp.pi))


# ── ODE-solver helpers ────────────────────────────────────────────────

def _get_solver(name: str):
    import diffrax
    _solvers = {
        "tsit5": diffrax.Tsit5,
        "dopri5": diffrax.Dopri5,
        "euler": diffrax.Euler,
        "heun": diffrax.Heun,
    }
    if name not in _solvers:
        raise ValueError(f"Unknown solver {name!r}. Available: {sorted(_solvers)}")
    return _solvers[name]()


# ── FFJORD Flax module ────────────────────────────────────────────────

class FFJORD(nn.Module):
    """Multi-block FFJORD continuous normalizing flow.

    Each block contains an independent :class:`VelocityField` and performs
    an ODE integration from ``t=0`` (base) to ``t=1`` (data).

    Public methods
    --------------
    * ``log_prob(x)`` — compute ``log p(x)`` for a batch.
    * ``sample(rng, n)`` — draw *n* samples from the learned distribution.
    """
    cfg: FFJORDConfig

    def setup(self) -> None:
        self.vfs = [
            VelocityField(
                hidden_sizes=self.cfg.hidden_sizes,
                dim=self.cfg.dim,
                name=f"vf_{i}",
            )
            for i in range(self.cfg.n_blocks)
        ]

    # ── internal helpers ──────────────────────────────────────────────

    def _solve_block(
        self,
        vf: VelocityField,
        x: jnp.ndarray,
        t0: float,
        t1: float,
        eps: jnp.ndarray,
        *,
        compute_trace: bool = True,
        compute_reg: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Solve one FFJORD block for a **single** point *x* of shape ``(dim,)``.

        Returns ``(x_final, delta_logp, reg_cost)`` where
        * delta_logp accumulates ``-∫ div(f) dt``
        * reg_cost accumulates ``∫ (kin_reg·||f||² + jac_reg·||J||_F²) dt``
          (zero when *compute_reg* is ``False``).
        """
        import diffrax

        trace_type = self.cfg.trace_type
        _kin_w = self.cfg.kin_reg if compute_reg else 0.0
        _jac_w = self.cfg.jac_reg if compute_reg else 0.0
        _need_reg = compute_reg and (_kin_w > 0 or _jac_w > 0)
        _need_jac = _need_reg and _jac_w > 0

        def dynamics(t, state, args):
            x_t, _, _ = state
            f_val = vf(t, x_t)

            if not compute_trace and not _need_reg:
                zero = jnp.zeros((), dtype=x_t.dtype)
                return (f_val, zero, zero)

            def f_of_x(x_in):
                return vf(t, x_in)

            # Choose path: combined (div + jac) or div-only
            if _need_jac:
                if trace_type == "exact":
                    div, jac_sq = _exact_div_and_jac_norm(f_of_x, x_t)
                else:
                    div, jac_sq = _hutchinson_div_and_jac_norm(f_of_x, x_t, args)
            else:
                if not compute_trace:
                    div = jnp.zeros((), dtype=x_t.dtype)
                elif trace_type == "exact":
                    div = _exact_divergence(f_of_x, x_t)
                else:
                    div = _hutchinson_divergence(f_of_x, x_t, args)
                jac_sq = jnp.zeros((), dtype=x_t.dtype)

            # Regularisation rate
            reg_rate = jnp.zeros((), dtype=x_t.dtype)
            if _need_reg:
                if _kin_w > 0:
                    reg_rate = reg_rate + _kin_w * jnp.sum(f_val ** 2)
                if _jac_w > 0:
                    reg_rate = reg_rate + _jac_w * jac_sq

            return (f_val, -div, reg_rate)

        term = diffrax.ODETerm(dynamics)
        solver = _get_solver(self.cfg.solver)
        sc = diffrax.PIDController(rtol=self.cfg.rtol, atol=self.cfg.atol)

        dt0_val = self.cfg.dt0 if t1 > t0 else -self.cfg.dt0

        zero = jnp.zeros((), dtype=x.dtype)
        y0 = (x, zero, zero)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0_val,
            y0=y0,
            args=eps,
            stepsize_controller=sc,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=self.cfg.max_steps,
        )

        x_final = sol.ys[0][0]       # (dim,)
        delta_logp = sol.ys[1][0]    # scalar
        reg_cost = sol.ys[2][0]      # scalar
        return x_final, delta_logp, reg_cost

    # ── public API ────────────────────────────────────────────────────

    def init_only(self, x: jnp.ndarray) -> jnp.ndarray:
        """Lightweight parameter initialization path (no ODE solve).

        Flax ``model.init`` traces this method to create all parameters of the
        velocity-field submodules while avoiding ``diffrax.diffeqsolve``.
        """

        xi = x[0]
        t0 = jnp.asarray(0.0, dtype=xi.dtype)
        for vf in self.vfs:
            xi = vf(t0, xi)
        return xi

    def _make_eps(self, x: jnp.ndarray):
        """Create Hutchinson noise or dummy eps for the batch."""
        n = x.shape[0]
        n_b = self.cfg.n_blocks
        dim = self.cfg.dim
        if self.cfg.trace_type == "hutchinson":
            rng = self.make_rng("hutchinson")
            return jax.random.normal(rng, (n, n_b, dim), dtype=x.dtype)
        return jnp.zeros((n, n_b, dim), dtype=x.dtype)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute ``log p(x)`` for a batch *x* of shape ``(N, dim)``.

        For *hutchinson* trace estimation the ``"hutchinson"`` RNG collection
        must be provided in the ``rngs`` dict when calling ``model.apply``.
        """
        n_b = self.cfg.n_blocks
        eps = self._make_eps(x)

        def _single_lp(xi, eps_i):
            z = xi
            total_delta = jnp.zeros((), dtype=xi.dtype)
            for i in range(n_b - 1, -1, -1):
                z, d, _ = self._solve_block(self.vfs[i], z, 1.0, 0.0, eps_i[i])
                total_delta = total_delta + d
            base_lp = _standard_normal_log_prob_single(z)
            return base_lp - total_delta

        return jax.vmap(_single_lp)(x, eps)

    def log_prob_with_reg(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute ``(log p(x), reg_cost)`` for a batch.

        ``reg_cost`` is per-sample ``\int (kin_reg ||f||^2 + jac_reg ||J||_F^2) dt``
        summed over blocks.  Zero when both ``kin_reg`` and ``jac_reg`` are 0.
        """
        n_b = self.cfg.n_blocks
        eps = self._make_eps(x)

        def _single(xi, eps_i):
            z = xi
            total_delta = jnp.zeros((), dtype=xi.dtype)
            total_reg = jnp.zeros((), dtype=xi.dtype)
            for i in range(n_b - 1, -1, -1):
                z, d, r = self._solve_block(
                    self.vfs[i], z, 1.0, 0.0, eps_i[i], compute_reg=True,
                )
                total_delta = total_delta + d
                total_reg = total_reg + r
            base_lp = _standard_normal_log_prob_single(z)
            return base_lp - total_delta, total_reg

        return jax.vmap(_single)(x, eps)

    def sample(self, rng: jax.Array, n: int) -> jnp.ndarray:
        """Draw *n* samples from the flow (no trace computation)."""
        z = jax.random.normal(rng, (n, self.cfg.dim), dtype=jnp.float32)
        n_b = self.cfg.n_blocks
        dim = self.cfg.dim
        eps_dummy = jnp.zeros((n, n_b, dim), dtype=jnp.float32)

        def _single_sample(zi, eps_i):
            x = zi
            for i in range(n_b):
                x, _, _ = self._solve_block(
                    self.vfs[i], x, 0.0, 1.0, eps_i[i], compute_trace=False,
                )
            return x

        return jax.vmap(_single_sample)(z, eps_dummy)


# ── Functional helpers (matching realnvp.py signatures) ───────────────

def log_prob_apply(
    flow: FFJORD, params: dict, x: jnp.ndarray, *, rng: jax.Array | None = None,
) -> jnp.ndarray:
    """Compute ``log p(x)`` using a bound *flow* and *params*."""
    rngs = {"hutchinson": rng} if rng is not None else {}
    return flow.apply({"params": params}, x, rngs=rngs, method=FFJORD.log_prob)


def log_prob_reg_apply(
    flow: FFJORD, params: dict, x: jnp.ndarray, *, rng: jax.Array | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(log_probs, reg_costs)`` — regularised training path."""
    rngs = {"hutchinson": rng} if rng is not None else {}
    return flow.apply(
        {"params": params}, x, rngs=rngs, method=FFJORD.log_prob_with_reg,
    )


def score_apply(flow: FFJORD, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Compute score ``∇_x log p(x)`` for batch *x* with shape ``(N, dim)``."""

    def lp_single(xi: jnp.ndarray) -> jnp.ndarray:
        return flow.apply({"params": params}, xi[None], method=FFJORD.log_prob)[0]

    return jax.vmap(jax.grad(lp_single))(x)
