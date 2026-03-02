from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dpjax.data import Normalizer
from dpjax.flows.realnvp import RealNVP, RealNVPConfig, log_prob_apply, score_apply
from dpjax.models.potential import PotentialConfig, PotentialMLP, grad_phi_apply
from dpjax.physics.cbe import residual_A


def main() -> int:
    key = jax.random.key(0)

    # Fake standardized batch
    x = jax.random.normal(key, shape=(128, 6), dtype=jnp.float32)

    flow = RealNVP(RealNVPConfig())
    params_flow = flow.init(key, x, method=RealNVP.log_prob)["params"]

    lp = log_prob_apply(flow, params_flow, x)
    score = score_apply(flow, params_flow, x[:16])

    phi = PotentialMLP(PotentialConfig())
    params_phi = phi.init(key, x[:, :3])["params"]
    grad_phi = grad_phi_apply(phi, params_phi, x[:16, :3])

    norm = Normalizer(mean=np.zeros(6, dtype=np.float32), std=np.ones(6, dtype=np.float32))
    r = residual_A(x[:16], score, grad_phi, norm)

    print("log_prob shape:", lp.shape)
    print("score shape:", score.shape)
    print("grad_phi shape:", grad_phi.shape)
    print("residual shape:", r.shape)
    print("finite?", bool(jnp.all(jnp.isfinite(lp))) and bool(jnp.all(jnp.isfinite(r))))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
