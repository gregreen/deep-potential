from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from dpjax.data import load_eta_h5
from dpjax.flows.api import load_df, score_apply, sample_apply
from dpjax.paths import ensure_dir, resolve_path
from dpjax.physics.analytic import plummer_score_std_batch, plummer_rv_ideal_grid
from dpjax.plotting.flow_projections import (
    calc_coords,
    plot_1d_marginals,
    plot_2d_marginal,
    plot_2d_marginals_grid,
)


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
    plummer_diag: bool = False,
    n_diag_points: int = 16384,
) -> Dict[str, Any]:
    """Evaluate DF by comparing marginals: train data vs flow samples.

    Returns
    -------
    dict
        ``{"eta_coords": ..., "samp_coords": ..., "out_dir": Path}``
    """
    data_path = resolve_path(data_path)
    df_run_dir = resolve_path(df_run_dir)

    df_model, df_params, normalizer, df_cfg = load_df(df_run_dir)
    flow_cfg = df_cfg.get("flow", {})

    out_dir = ensure_dir(out_dir or (Path(df_run_dir) / "plots"))

    eta = load_eta_h5(data_path, dataset="eta")
    eta_coords = calc_coords(eta)

    # Sample from flow in standardized coordinates, then inverse-transform to physical
    rng = jax.random.key(int(seed))
    x_std = sample_apply(df_model, df_params, rng, n_samples, flow_cfg)
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

    # 2D marginal grid (train vs sample): x-y, x-z, vx-vy
    _plot_2d_grid(eta, eta_samp, out_dir)

    # Save sample data for offline notebook plotting
    np.savez(
        out_dir / "df_samples.npz",
        eta_train=eta,
        eta_sample=eta_samp,
    )
    print(f"  Saved df_samples.npz ({eta.shape[0]} train, {eta_samp.shape[0]} sample)")

    result: Dict[str, Any] = {
        "eta_coords": eta_coords, "samp_coords": samp_coords, "out_dir": out_dir,
    }

    # Plummer-specific diagnostics
    if plummer_diag:
        diag = _plummer_diagnostics(
            df_model, df_params, normalizer, flow_cfg,
            n_points=n_diag_points, n_rv_samples=n_samples, seed=seed, out_dir=out_dir,
        )
        result["plummer_diag"] = diag

    print(f"Wrote DF plots to {out_dir}")
    return result


# ---------------------------------------------------------------------------
# 2D marginal grid helper
# ---------------------------------------------------------------------------

def _plot_2d_grid(
    eta_train: np.ndarray,
    eta_sample: np.ndarray,
    out_dir: Path,
) -> None:
    coords_train = calc_coords(
        eta_train,
        spherical_origin=(0.0, 0.0, 0.0),
        cylindrical_origin=(8.3, 0.0, 0.0),
    )
    coords_sample = calc_coords(
        eta_sample,
        spherical_origin=(0.0, 0.0, 0.0),
        cylindrical_origin=(8.3, 0.0, 0.0),
    )
    fig = plot_2d_marginals_grid(
        coords_train, coords_sample,
        dims=[("x", "y"), ("x", "z"), ("vx", "vy")],
        fig_dir=str(out_dir),
        fig_fmt=["png"],
        bins=128,
        logscale=False,
        diff_vmax=5.0,
    )
    if fig is not None:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plummer-specific diagnostics
# ---------------------------------------------------------------------------

def _plummer_diagnostics(
    df_model,
    df_params,
    normalizer,
    flow_cfg: dict,
    *,
    n_points: int = 16384,
    n_rv_samples: int = 262144,
    seed: int = 0,
    out_dir: Path | None = None,
) -> Dict[str, Any]:
    """Generate Plummer-specific diagnostic plots.

    Returns dict with slope/R² per dimension and residual stats.
    """
    import sys
    _repo = Path(__file__).resolve().parents[1]
    if str(_repo / "scripts") not in sys.path:
        sys.path.insert(0, str(_repo / "scripts"))
    from plummer.plummer_gendata import sample_df

    out_dir = ensure_dir(out_dir or Path("."))

    # ---- 1. Gradient comparison: flow score vs analytic Plummer score ----
    eta_phys = np.asarray(sample_df(n_points, max_dist=10.0), dtype=np.float32)
    eta_std = np.asarray(normalizer.transform(eta_phys), dtype=np.float32)
    eta_std_j = jnp.asarray(eta_std)

    mean_j = jnp.asarray(normalizer.mean, dtype=jnp.float32)
    std_j = jnp.asarray(normalizer.std, dtype=jnp.float32)

    score_true = np.asarray(
        jax.jit(plummer_score_std_batch)(eta_std_j, mean_j, std_j), dtype=np.float32,
    )
    score_est = np.asarray(
        score_apply(df_model, df_params, eta_std_j, flow_cfg), dtype=np.float32,
    )

    dim_labels = ["x", "y", "z", "vx", "vy", "vz"]
    slopes, r2s = [], []

    fig, ax_arr = plt.subplots(2, 3, figsize=(16, 9))
    for i, ax in enumerate(ax_arr.flat):
        ax.set_aspect("equal")
        ax.scatter(score_true[:, i], score_est[:, i], alpha=0.1, s=2, edgecolors="none")

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lo = min(xlim[0], ylim[0])
        hi = max(xlim[1], ylim[1])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.plot([lo, hi], [lo, hi], c="k", alpha=0.25)

        mask = np.isfinite(score_true[:, i]) & np.isfinite(score_est[:, i])
        if mask.sum() > 10:
            st, se = score_true[mask, i], score_est[mask, i]
            slope = float(np.sum(st * se) / (np.sum(st ** 2) + 1e-30))
            ss_res = np.sum((se - slope * st) ** 2)
            ss_tot = np.sum((se - np.mean(se)) ** 2) + 1e-30
            r2 = 1.0 - ss_res / ss_tot
        else:
            slope, r2 = float("nan"), float("nan")
        slopes.append(slope)
        r2s.append(r2)
        ax.text(
            0.05, 0.95, f"slope={slope:.3f}\nR\u00b2={r2:.3f}",
            ha="left", va="top", transform=ax.transAxes,
            fontsize=9, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        ax.set_xlabel("true")
        ax.set_ylabel("normalizing flow")
        ax.set_title(f"dlogf/d{dim_labels[i]}")

    fig.subplots_adjust(hspace=0.25, wspace=0.3, top=0.91, bottom=0.06)
    fig.suptitle("Performance of normalizing flow score gradients", fontsize=20)
    fig.savefig(out_dir / "flow_gradients_comparison.png", dpi=100)
    plt.close(fig)
    print(f"  Wrote gradient comparison plot")

    # Save gradient data for offline notebook plotting
    np.savez(
        out_dir / "gradient_comparison.npz",
        score_true=score_true,
        score_est=score_est,
    )
    print(f"  Saved gradient_comparison.npz")

    # ---- 2. Score residual histograms ----
    score_resid = score_est - score_true
    fig, ax_arr = plt.subplots(2, 3, figsize=(16, 9))
    resid_stats = {}
    for i, ax in enumerate(ax_arr.flat):
        resid = score_resid[:, i]
        ax.hist(resid, bins=51, range=(-0.05, 0.05), log=True)
        ax.set_xlabel("(normalizing flow) - (true)")
        ax.set_title(f"dlogf/deta_{i}")

        sigma = float(np.std(resid))
        mu = float(np.mean(resid))
        centered = resid - mu
        m2 = float(np.mean(centered ** 2))
        m4 = float(np.mean(centered ** 4))
        kurt = m4 / (m2 ** 2 + 1e-12) - 3.0
        resid_stats[dim_labels[i]] = {"sigma": sigma, "kurtosis": kurt}

        ax.text(
            0.95, 0.95, f"sigma = {sigma:.4f}\nkurt = {kurt:.2f}",
            ha="right", va="top", transform=ax.transAxes,
        )

    fig.subplots_adjust(hspace=0.25, wspace=0.3, top=0.91, bottom=0.06)
    fig.suptitle("Score-gradient residual histograms", fontsize=20)
    fig.savefig(out_dir / "flow_gradients_comparison_hist.png", dpi=100)
    plt.close(fig)
    print(f"  Wrote gradient residual histogram plot")

    # ---- 3. r-v distribution comparison ----
    r_lim, v_lim, bins_rv = (0.0, 5.0), (0.0, 1.5), (50, 50)
    rv = plummer_rv_ideal_grid(r_lim=r_lim, v_lim=v_lim, bins=bins_rv)
    r_grid, v_grid, n_ideal = rv["r"], rv["v"], rv["n_ideal"]

    rng = jax.random.key(int(seed))
    x_std_rv = sample_apply(df_model, df_params, rng, n_rv_samples, flow_cfg)
    eta_flow = np.asarray(normalizer.inverse(np.asarray(x_std_rv)), dtype=np.float32)

    r_samp = np.sqrt(np.sum(eta_flow[:, :3] ** 2, axis=1))
    v_samp = np.sqrt(np.sum(eta_flow[:, 3:] ** 2, axis=1))

    fig, ax_arr = plt.subplots(3, 2, figsize=(11, 16))
    fig.subplots_adjust(left=0.1)

    # Row 0: Ideal (linear + log)
    ax_arr[0, 0].imshow(
        n_ideal, extent=r_lim + v_lim, origin="lower", aspect="auto", interpolation="nearest",
    )
    img_log = np.log(np.clip(n_ideal, 1e-30, np.inf))
    vmax_log = np.max(img_log)
    ax_arr[0, 1].imshow(
        img_log, extent=r_lim + v_lim, vmax=vmax_log, vmin=vmax_log - 10.0,
        origin="lower", aspect="auto", interpolation="nearest",
    )

    # Row 1: Flow samples (linear + log)
    n_samp, _, _, _ = ax_arr[1, 0].hist2d(r_samp, v_samp, bins=bins_rv, range=[r_lim, v_lim])
    n_samp = n_samp.T
    ax_arr[1, 1].hist2d(r_samp, v_samp, bins=bins_rv, range=[r_lim, v_lim], norm=mcolors.LogNorm())

    # Row 2: Residuals
    dr = r_grid[1] - r_grid[0]
    dv = v_grid[1] - v_grid[0]
    n0 = n_ideal * dr * dv * len(r_samp)
    denom = np.clip(n0, 1e-12, np.inf)
    rel_resid = (n_samp - n0) / denom

    ax_arr[2, 0].imshow(
        rel_resid, extent=r_lim + v_lim, vmax=0.1, vmin=-0.1,
        origin="lower", aspect="auto", cmap="coolwarm_r", interpolation="nearest",
    )
    log_resid = np.log(np.clip(n_samp, 1.0, np.inf)) - np.log(np.clip(n0, 1.0, np.inf))
    ax_arr[2, 1].imshow(
        log_resid, extent=r_lim + v_lim, vmax=1.0, vmin=-1.0,
        origin="lower", aspect="auto", cmap="coolwarm_r", interpolation="nearest",
    )

    # Zero-energy boundary overlay
    for a in ax_arr.flat:
        a.plot(r_grid, np.sqrt(2.0) * (1.0 + r_grid ** 2) ** (-0.25), c="r")
        a.set_xlabel(r"$r$")
        a.set_ylabel(r"$v$")
        a.text(0.95, 0.95, r"$E > 0$", ha="right", va="top", fontsize=14, c="r", transform=a.transAxes)

    # Row labels
    for idx, label in enumerate(["Ideal DF", "Samples from Flow DF", "Residuals (Flow - Ideal)"]):
        pos = ax_arr[idx, 0].get_position()
        y_txt = 0.5 * (pos.y0 + pos.y1)
        fig.text(0.02, y_txt, label, rotation=90.0, ha="left", va="center", fontsize=16)

    ax_arr[0, 0].set_title("Linear Scale", fontsize=16)
    ax_arr[0, 1].set_title("Log Scale", fontsize=16)

    fig.savefig(out_dir / "df_rv_comparison.png", dpi=100)
    plt.close(fig)
    print(f"  Wrote r-v comparison plot")

    # Save r-v data for offline notebook plotting
    np.savez(
        out_dir / "rv_comparison.npz",
        r=r_grid,
        v=v_grid,
        n_ideal=n_ideal,
        n_samp=n_samp,
        n_flow_total=len(r_samp),
    )
    print(f"  Saved rv_comparison.npz")

    return {
        "slopes": dict(zip(dim_labels, slopes)),
        "r2": dict(zip(dim_labels, r2s)),
        "resid_stats": resid_stats,
    }


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
    parser.add_argument("--plummer-diag", action="store_true", help="Enable Plummer-specific diagnostics (gradient, r-v, residuals).")
    parser.add_argument("--n-diag-points", type=int, default=16384, help="Number of points for gradient comparison.")
    args = parser.parse_args()

    run_eval_df(
        args.data, args.df_run_dir,
        out_dir=args.out_dir, coordsys=args.coordsys,
        n_samples=args.n_samples, seed=args.seed,
        dim1=args.dim1, dim2=args.dim2, logscale=args.logscale,
        plummer_diag=args.plummer_diag, n_diag_points=args.n_diag_points,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
