"""Diagnostic plotting functions for Deep Potential evaluation.

All functions are pure numpy + matplotlib — no JAX or model dependency.
They consume pre-computed `.npz` data saved by ``eval_df.py`` / ``eval_phi.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# ── 1. Phase-space overview grid ─────────────────────────────────────────

def plot_phase_space_grid(
    eta: np.ndarray,
    *,
    eta_sample: Optional[np.ndarray] = None,
    xlim: tuple[float, float] = (-3.0, 3.0),
    vlim: tuple[float, float] = (-1.5, 1.5),
    bins_2d: int = 31,
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
):
    """3×3 phase-space overview inspired by ``plummer_sphere_example`` cell 9.

    Row 0: spatial projections (x-y, x-z, y-z)
    Row 1: velocity projections (vx-vy, vx-vz, vy-vz)
    Row 2: r-v diagram, spatial isotropy, velocity isotropy

    If *eta_sample* is provided, a second (outline) histogram is overlaid.
    """
    import matplotlib.pyplot as plt

    eta = np.asarray(eta)

    def _vec2ang(x: np.ndarray):
        phi = np.arctan2(x[:, 1], x[:, 0])
        theta = np.arctan2(x[:, 2], np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2))
        return theta, phi

    fig, ax_arr = plt.subplots(3, 3, figsize=(13, 12), dpi=dpi)
    fig.subplots_adjust(wspace=0.30, hspace=0.25)

    for k, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        ax_arr[0, k].hist2d(eta[:, i], eta[:, j], bins=bins_2d, range=[xlim, xlim])
        ax_arr[1, k].hist2d(eta[:, i + 3], eta[:, j + 3], bins=bins_2d, range=[vlim, vlim])
        dim_labels = ["x", "y", "z"]
        ax_arr[0, k].set_xlabel(rf"${dim_labels[i]}$")
        ax_arr[0, k].set_ylabel(rf"${dim_labels[j]}$", labelpad=-5)
        ax_arr[1, k].set_xlabel(rf"$v_{dim_labels[i]}$")
        ax_arr[1, k].set_ylabel(rf"$v_{dim_labels[j]}$", labelpad=-5)
        ax_arr[0, k].set_aspect("equal")
        ax_arr[1, k].set_aspect("equal")

    r = np.sqrt(np.sum(eta[:, :3] ** 2, axis=1))
    v = np.sqrt(np.sum(eta[:, 3:] ** 2, axis=1))
    ax_arr[2, 0].hist2d(r, v, bins=bins_2d, range=[(0.0, 5.0), (0.0, 1.5)])
    ax_arr[2, 0].set_xlabel(r"$r$")
    ax_arr[2, 0].set_ylabel(r"$v$", labelpad=0)

    iso_bins = 11
    v0 = eta.shape[0] / iso_bins ** 2
    dv = 0.5 * v0

    theta, phi = _vec2ang(eta[:, :3])
    ax_arr[2, 1].hist2d(phi, np.sin(theta), bins=iso_bins, vmin=v0 - dv, vmax=v0 + dv, cmap="bwr_r")
    ax_arr[2, 1].set_xlabel(r"$\varphi_x$")
    ax_arr[2, 1].set_ylabel(r"$\sin \theta_x$", labelpad=-5)

    theta, phi = _vec2ang(eta[:, 3:])
    ax_arr[2, 2].hist2d(phi, np.sin(theta), bins=iso_bins, vmin=v0 - dv, vmax=v0 + dv, cmap="bwr_r")
    ax_arr[2, 2].set_xlabel(r"$\varphi_v$")
    ax_arr[2, 2].set_ylabel(r"$\sin \theta_v$", labelpad=-5)

    for a in ax_arr[2]:
        a.set_aspect("auto")

    if eta_sample is not None:
        eta_s = np.asarray(eta_sample)
        for k, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
            ax_arr[0, k].hist2d(eta_s[:, i], eta_s[:, j], bins=bins_2d, range=[xlim, xlim], alpha=0.0)
            counts_s, xedges, yedges = np.histogram2d(eta_s[:, i], eta_s[:, j], bins=bins_2d, range=[xlim, xlim])
            ax_arr[0, k].contour(
                counts_s.T,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                colors="r",
                linewidths=0.5,
                levels=5,
                alpha=0.6,
            )
        r_s = np.sqrt(np.sum(eta_s[:, :3] ** 2, axis=1))
        v_s = np.sqrt(np.sum(eta_s[:, 3:] ** 2, axis=1))
        counts_rv, xedges, yedges = np.histogram2d(r_s, v_s, bins=bins_2d, range=[(0.0, 5.0), (0.0, 1.5)])
        ax_arr[2, 0].contour(
            counts_rv.T,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            colors="r",
            linewidths=0.5,
            levels=5,
            alpha=0.6,
        )

    fig.suptitle("Phase-Space Overview", fontsize=16)

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"phase_space_grid.{fmt}", dpi=dpi)
        plt.close(fig)
        return None
    return fig


# ── 2. Score gradient scatter comparison ─────────────────────────────────

def plot_score_comparison(
    score_true: np.ndarray,
    score_est: np.ndarray,
    *,
    dim_labels: tuple[str, ...] = ("x", "y", "z", "vx", "vy", "vz"),
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
):
    """2×3 scatter: true vs learned score per dimension, with slope/R² annotations.

    Inspired by ``plummer_sphere_example`` cell 33 and ``eval_df._plummer_diagnostics``.
    """
    import matplotlib.pyplot as plt

    score_true = np.asarray(score_true)
    score_est = np.asarray(score_est)

    fig, ax_arr = plt.subplots(2, 3, figsize=(16, 9), dpi=dpi)

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

        ax.text(
            0.05, 0.95, f"slope={slope:.3f}\nR\u00b2={r2:.3f}",
            ha="left", va="top", transform=ax.transAxes,
            fontsize=9, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        ax.set_xlabel("true")
        ax.set_ylabel("learned")
        ax.set_title(rf"$\partial \log f / \partial {dim_labels[i]}$")

    fig.subplots_adjust(hspace=0.25, wspace=0.3, top=0.91, bottom=0.06)
    fig.suptitle("Score Gradient: True vs Learned", fontsize=20)

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"score_comparison.{fmt}", dpi=dpi)
        plt.close(fig)
        return None
    return fig


# ── 3. Score gradient residual histograms ────────────────────────────────

def plot_score_residual_hist(
    score_true: np.ndarray,
    score_est: np.ndarray,
    *,
    dim_labels: tuple[str, ...] = ("x", "y", "z", "vx", "vy", "vz"),
    resid_range: tuple[float, float] = (-0.05, 0.05),
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
):
    """2×3 histograms of score residuals per dimension with σ and kurtosis.

    Inspired by ``plummer_sphere_example`` cell 35.
    """
    import matplotlib.pyplot as plt

    score_true = np.asarray(score_true)
    score_est = np.asarray(score_est)
    resid = score_est - score_true

    fig, ax_arr = plt.subplots(2, 3, figsize=(16, 9), dpi=dpi)

    for i, ax in enumerate(ax_arr.flat):
        r_i = resid[:, i]
        ax.hist(r_i, bins=51, range=resid_range, log=True)
        ax.set_xlabel("(learned) - (true)")
        ax.set_title(rf"$\partial \log f / \partial {dim_labels[i]}$")

        sigma = float(np.std(r_i))
        mu = float(np.mean(r_i))
        centered = r_i - mu
        m2 = float(np.mean(centered ** 2))
        m4 = float(np.mean(centered ** 4))
        kurt = m4 / (m2 ** 2 + 1e-12) - 3.0

        ax.text(
            0.95, 0.95, f"$\\sigma = {sigma:.4f}$\n$\\kappa = {kurt:.2f}$",
            ha="right", va="top", transform=ax.transAxes, fontsize=10,
        )

    fig.subplots_adjust(hspace=0.25, wspace=0.3, top=0.91, bottom=0.06)
    fig.suptitle("Score Gradient Residual Histograms", fontsize=20)

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"score_residual_hist.{fmt}", dpi=dpi)
        plt.close(fig)
        return None
    return fig


# ── 4. r-v distribution comparison ──────────────────────────────────────

def plot_rv_comparison(
    n_ideal: np.ndarray,
    n_samp: np.ndarray,
    r: np.ndarray,
    v: np.ndarray,
    n_flow_total: int,
    *,
    r_lim: tuple[float, float] = (0.0, 5.0),
    v_lim: tuple[float, float] = (0.0, 1.5),
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
):
    """3×2 r-v comparison: ideal / flow samples / residuals × linear / log.

    Inspired by ``plummer_sphere_example`` cell 13.

    Parameters
    ----------
    n_ideal : (Nv, Nr) ideal density grid
    n_samp : (Nv, Nr) histogram of flow samples (already transposed to match n_ideal)
    r, v : 1-D bin-center arrays
    n_flow_total : total number of flow samples used to build n_samp
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n_ideal = np.asarray(n_ideal)
    n_samp = np.asarray(n_samp)

    extent = r_lim + v_lim

    fig, ax_arr = plt.subplots(3, 2, figsize=(11, 16), dpi=dpi)
    fig.subplots_adjust(left=0.1)

    # Row 0: ideal
    ax_arr[0, 0].imshow(n_ideal, extent=extent, origin="lower", aspect="auto", interpolation="nearest")
    img_log = np.log(np.clip(n_ideal, 1e-30, np.inf))
    vmax_log = np.max(img_log)
    ax_arr[0, 1].imshow(
        img_log, extent=extent, vmax=vmax_log, vmin=vmax_log - 10.0,
        origin="lower", aspect="auto", interpolation="nearest",
    )

    # Row 1: flow samples
    ax_arr[1, 0].imshow(n_samp, extent=extent, origin="lower", aspect="auto", interpolation="nearest")
    n_samp_log = np.log(np.clip(n_samp, 1.0, np.inf))
    vmax_s = np.max(n_samp_log)
    ax_arr[1, 1].imshow(
        n_samp_log, extent=extent, vmax=max(vmax_s, 1.0), vmin=max(vmax_s - 10.0, 0.0),
        origin="lower", aspect="auto", interpolation="nearest",
    )

    # Row 2: residuals
    dr = r[1] - r[0] if len(r) > 1 else 1.0
    dv = v[1] - v[0] if len(v) > 1 else 1.0
    n0 = n_ideal * dr * dv * n_flow_total
    denom = np.clip(n0, 1e-12, np.inf)
    rel_resid = (n_samp - n0) / denom

    ax_arr[2, 0].imshow(
        rel_resid, extent=extent, vmax=0.1, vmin=-0.1,
        origin="lower", aspect="auto", cmap="coolwarm_r", interpolation="nearest",
    )
    log_resid = np.log(np.clip(n_samp, 1.0, np.inf)) - np.log(np.clip(n0, 1.0, np.inf))
    ax_arr[2, 1].imshow(
        log_resid, extent=extent, vmax=1.0, vmin=-1.0,
        origin="lower", aspect="auto", cmap="coolwarm_r", interpolation="nearest",
    )

    # Zero-energy line
    r_line = np.linspace(r_lim[0] + 0.01, r_lim[1], 200)
    v_line = np.sqrt(2.0) * (1.0 + r_line ** 2) ** (-0.25)
    for a in ax_arr.flat:
        a.plot(r_line, v_line, c="r")
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

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"rv_comparison.{fmt}", dpi=dpi)
        plt.close(fig)
        return None
    return fig


# ── 5. Phi / ρ / |a| 2D slices ──────────────────────────────────────────

def plot_phi_rho_slice(
    x: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    rho: np.ndarray,
    acc_mag: np.ndarray,
    *,
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
):
    """Three-panel 2D slice plot: Φ(x,y), ρ(x,y), |a|(x,y).

    Inspired by ``harmonic_blob_example`` cells 25-26 and ``plot_phi_slice.py``.

    Parameters
    ----------
    x, y : 1-D grid arrays
    phi, rho, acc_mag : 2-D images with shape (len(y), len(x))
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    rmax_x = max(abs(x[0]), abs(x[-1]))
    rmax_y = max(abs(y[0]), abs(y[-1]))
    extent = [-rmax_x, rmax_x, -rmax_y, rmax_y]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=dpi)

    # Phi (mean-subtracted)
    phi0 = phi - np.nanmean(phi)
    vmin, vmax = np.nanpercentile(phi0, [1, 99])
    if vmin * vmax < 0:
        divnorm = colors.TwoSlopeNorm(vcenter=0.0, vmin=float(vmin), vmax=float(vmax))
        im0 = axes[0].imshow(phi0, extent=extent, origin="lower", cmap="seismic", norm=divnorm)
    else:
        im0 = axes[0].imshow(phi0, extent=extent, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title(r"$\Phi(x,y)$ (mean-sub.)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # ρ (log scale)
    rho_pos = np.clip(rho, 1e-12, np.inf)
    vmin_r, vmax_r = np.nanpercentile(rho_pos, [5, 99])
    im1 = axes[1].imshow(
        rho_pos, extent=extent, origin="lower", cmap="magma",
        norm=colors.LogNorm(vmin=max(vmin_r, 1e-12), vmax=max(vmax_r, 1e-11)),
    )
    axes[1].set_title(r"$\rho = \nabla^2\Phi / (4\pi)$")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # |a| (log scale)
    vmin_a, vmax_a = np.nanpercentile(acc_mag, [5, 99])
    im2 = axes[2].imshow(
        acc_mag, extent=extent, origin="lower", cmap="cubehelix",
        norm=colors.LogNorm(vmin=max(vmin_a, 1e-12), vmax=max(vmax_a, 1e-11)),
    )
    axes[2].set_title(r"$|\mathbf{a}(x,y)|$")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(r"Potential / Density / Acceleration (z = 0 slice)", fontsize=13)
    fig.tight_layout()

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"phi_rho_acc_slice.{fmt}", dpi=dpi)
        plt.close(fig)
        return None
    return fig


# ── 6. Radial density profile ρ(r) ──────────────────────────────────────

def plot_density_profile(
    r: np.ndarray,
    rho_learned: np.ndarray,
    rho_analytic: np.ndarray,
    *,
    phi_learned_shift: Optional[np.ndarray] = None,
    phi_true: Optional[np.ndarray] = None,
    ar_learned: Optional[np.ndarray] = None,
    ar_true: Optional[np.ndarray] = None,
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
):
    """Radial profiles: ρ(r) learned vs analytic, optionally Φ(r) and a_r(r).

    If ``phi_learned_shift`` / ``phi_true`` and ``ar_learned`` / ``ar_true``
    are provided, the figure has 3 columns; otherwise only the ρ(r) panel.
    """
    import matplotlib.pyplot as plt

    has_phi = phi_learned_shift is not None and phi_true is not None
    has_ar = ar_learned is not None and ar_true is not None
    n_cols = 1 + int(has_phi) + int(has_ar)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5), dpi=dpi)
    if n_cols == 1:
        axes = [axes]

    col = 0
    if has_phi:
        axes[col].plot(r, phi_true, "k-", lw=2, label="analytic")
        axes[col].plot(r, phi_learned_shift, "--", lw=1.5, label="learned")
        axes[col].set_xscale("log")
        axes[col].set_xlabel("r")
        axes[col].set_ylabel(r"$\Phi(r)$")
        axes[col].set_title("Gravitational Potential")
        axes[col].legend()
        axes[col].grid(True, alpha=0.2)
        col += 1

    if has_ar:
        axes[col].plot(r, ar_true, "k-", lw=2, label="analytic")
        axes[col].plot(r, ar_learned, "--", lw=1.5, label="learned")
        axes[col].set_xscale("log")
        axes[col].set_xlabel("r")
        axes[col].set_ylabel(r"$a_r(r)$")
        axes[col].set_title("Radial Acceleration")
        axes[col].legend()
        axes[col].grid(True, alpha=0.2)
        col += 1

    axes[col].plot(r, rho_analytic, "k-", lw=2, label="analytic")
    axes[col].plot(r, rho_learned, "--", lw=1.5, label="learned", color="tab:orange")
    axes[col].set_xscale("log")
    axes[col].set_yscale("log")
    axes[col].set_xlabel("r")
    axes[col].set_ylabel(r"$\rho(r)$")
    axes[col].set_title("Mass Density")
    axes[col].legend()
    axes[col].grid(True, alpha=0.2)

    fig.suptitle("Plummer Sphere: Learned vs Analytic", fontsize=13)
    fig.tight_layout()

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"density_profile.{fmt}", dpi=dpi)
        plt.close(fig)
        return None
    return fig


# ── 7. CBE residual spatial map ──────────────────────────────────────────

def plot_residual_spatial(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    residual: np.ndarray,
    *,
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
):
    """2D hexbin map of CBE residual magnitude in (x,y) and (x,z) projections.

    Parameters
    ----------
    x, y, z : (N,) physical spatial coordinates
    residual : (N,) CBE residual values
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    abs_r = np.abs(residual)
    vmax = float(np.percentile(abs_r, 99))
    vmax = max(vmax, 1e-6)

    fig, (ax1, ax2, cax) = plt.subplots(
        1, 3, figsize=(12, 4.5), dpi=dpi,
        gridspec_kw=dict(width_ratios=[1, 1, 0.04]),
    )

    hb1 = ax1.hexbin(
        x, y, C=abs_r, reduce_C_function=np.median,
        gridsize=50, cmap="inferno",
        norm=colors.LogNorm(vmin=max(vmax * 1e-3, 1e-8), vmax=vmax),
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("CBE residual |r|  (x-y)")
    ax1.set_aspect("equal")

    hb2 = ax2.hexbin(
        x, z, C=abs_r, reduce_C_function=np.median,
        gridsize=50, cmap="inferno",
        norm=colors.LogNorm(vmin=max(vmax * 1e-3, 1e-8), vmax=vmax),
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    ax2.set_title("CBE residual |r|  (x-z)")
    ax2.set_aspect("equal")

    cb = fig.colorbar(hb2, cax=cax)
    cb.set_label("median |residual|")

    fig.suptitle("CBE Residual Spatial Distribution", fontsize=13)
    fig.tight_layout()

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"residual_spatial.{fmt}", dpi=dpi)
        plt.close(fig)
        return None
    return fig
