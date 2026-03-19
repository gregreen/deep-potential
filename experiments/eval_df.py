from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from dpjax.data import Normalizer, load_eta_h5
from dpjax.flows.realnvp import RealNVP, RealNVPConfig
from dpjax.paths import ensure_dir, resolve_path
from dpjax.utils.ckpt import create_manager, restore_latest
from dpjax.plotting.flow_projections import calc_coords, plot_1d_marginals, plot_2d_marginal


def _load_df(df_run_dir: Path) -> tuple[RealNVP, dict, Normalizer]:
    df_cfg_path = df_run_dir / "config.yaml"
    if not df_cfg_path.exists():
        raise FileNotFoundError(f"Missing {df_cfg_path}")

    df_cfg = yaml.safe_load(df_cfg_path.read_text())
    flow_cfg = df_cfg.get("flow", {})
    model = RealNVP(
        RealNVPConfig(
            dim=int(flow_cfg.get("dim", 6)),
            n_coupling=int(flow_cfg.get("n_coupling", 10)),
            hidden_sizes=tuple(int(x) for x in flow_cfg.get("hidden_sizes", [128, 128])),
            s_max=float(flow_cfg.get("s_max", 2.0)),
        )
    )

    norm = Normalizer.load_npz(df_run_dir / "normalizer.npz")

    ckpt_mgr = create_manager(df_run_dir / "ckpt")
    restored = restore_latest(ckpt_mgr)
    params = restored["params"]

    return model, params, norm


# ---------------------------------------------------------------------------
# Core evaluation function – callable from both CLI and Jupyter
# ---------------------------------------------------------------------------

def run_eval_df(
    data_path: str | Path,
    df_run_dir: str | Path,
    *,
    out_dir: Optional[str | Path] = None,
    coordsys: str = "cart",
    n_samples: int = 262144,
    seed: int = 0,
    dim1: str = "x",
    dim2: str = "y",
    logscale: bool = False,
) -> Dict[str, Any]:
    """Evaluate DF by comparing marginals: train data vs flow samples.

    Returns
    -------
    dict
        ``{"eta_coords": ..., "samp_coords": ..., "out_dir": Path}``
    """
    data_path = resolve_path(data_path)
    df_run_dir = resolve_path(df_run_dir)

    df_model, df_params, normalizer = _load_df(Path(df_run_dir))

    out_dir = ensure_dir(out_dir or (Path(df_run_dir) / "plots"))

    eta = load_eta_h5(data_path, dataset="eta")
    eta_coords = calc_coords(eta)

    # Sample from flow in standardized coordinates, then inverse-transform to physical
    rng = jax.random.key(int(seed))
    x_std = df_model.apply({"params": df_params}, rng, int(n_samples), method=RealNVP.sample)
    eta_samp = np.asarray(normalizer.inverse(np.asarray(x_std)))
    samp_coords = calc_coords(eta_samp)

    plot_1d_marginals(eta_coords, samp_coords, fig_dir=str(out_dir), coordsys=coordsys, fig_fmt=("png",))
    plot_2d_marginal(
        eta_coords,
        samp_coords,
        fig_dir=str(out_dir),
        dim1=dim1,
        dim2=dim2,
        fig_fmt=("png",),
        logscale=logscale,
    )

    print(f"Wrote DF plots to {out_dir}")
    return {"eta_coords": eta_coords, "samp_coords": samp_coords, "out_dir": out_dir}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DF by comparing marginals: train vs flow samples.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--df-run-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--coordsys", type=str, default="cart", choices=["cart", "cyl", "sph"])
    parser.add_argument("--n-samples", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim1", type=str, default="x")
    parser.add_argument("--dim2", type=str, default="y")
    parser.add_argument("--logscale", action="store_true")
    args = parser.parse_args()

    run_eval_df(
        args.data, args.df_run_dir,
        out_dir=args.out_dir, coordsys=args.coordsys,
        n_samples=args.n_samples, seed=args.seed,
        dim1=args.dim1, dim2=args.dim2, logscale=args.logscale,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
