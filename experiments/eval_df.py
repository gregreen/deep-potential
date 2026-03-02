from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from dpjax.data import Normalizer, load_eta_h5
from dpjax.flows.realnvp import RealNVP, RealNVPConfig
from dpjax.utils.ckpt import create_manager, restore_latest
from dpjax.plotting.flow_projections import calc_coords, plot_1d_marginals, plot_2d_marginal


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


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

    df_model, df_params, normalizer = _load_df(Path(args.df_run_dir))

    out_dir = _ensure_dir(args.out_dir or (Path(args.df_run_dir) / "plots"))

    eta = load_eta_h5(args.data, dataset="eta")
    eta_coords = calc_coords(eta)

    # Sample from flow in standardized coordinates, then inverse-transform to physical
    rng = jax.random.key(int(args.seed))
    x_std = df_model.apply({"params": df_params}, rng, int(args.n_samples), method=RealNVP.sample)
    eta_samp = np.asarray(normalizer.inverse(np.asarray(x_std)))
    samp_coords = calc_coords(eta_samp)

    plot_1d_marginals(eta_coords, samp_coords, fig_dir=str(out_dir), coordsys=args.coordsys, fig_fmt=("png",))
    plot_2d_marginal(
        eta_coords,
        samp_coords,
        fig_dir=str(out_dir),
        dim1=args.dim1,
        dim2=args.dim2,
        fig_fmt=("png",),
        logscale=bool(args.logscale),
    )

    print(f"Wrote DF plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
