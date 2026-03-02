from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def calc_coords(
    eta: np.ndarray,
    *,
    spherical_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cylindrical_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, np.ndarray]:
    """Coordinate transforms matching the legacy plotting utilities.

    Input eta is (N,6) with columns: x,y,z,vx,vy,vz.

    Returns a dict containing cart, cylindrical, spherical fields.
    """

    eta = np.asarray(eta)
    if eta.ndim != 2 or eta.shape[1] != 6:
        raise ValueError("eta must have shape (N,6)")

    sph_x0 = np.asarray(spherical_origin, dtype=np.float32)
    cyl_x0 = np.asarray(cylindrical_origin, dtype=np.float32)

    # Cylindrical
    dx = eta[:, 0] - cyl_x0[0]
    dy = eta[:, 1] - cyl_x0[1]
    dz = eta[:, 2] - cyl_x0[2]

    cyl_R = np.sqrt(dx * dx + dy * dy)
    cyl_z = dz
    cyl_phi = np.arctan2(dy, dx)

    eps = 1.0e-12
    inv_R = 1.0 / np.maximum(cyl_R, eps)
    cyl_cos_phi = dx * inv_R
    cyl_sin_phi = dy * inv_R

    vx, vy, vz = eta[:, 3], eta[:, 4], eta[:, 5]
    cyl_vR = vx * cyl_cos_phi + vy * cyl_sin_phi
    cyl_vT = -vx * cyl_sin_phi + vy * cyl_cos_phi

    cyl = {
        "cylR": cyl_R,
        "cylz": cyl_z,
        "cylphi": cyl_phi,
        "cylvR": cyl_vR,
        "cylvz": vz,
        "cylvT": cyl_vT,
    }

    # Cartesian (shift spherical origin)
    x = eta[:, 0] - sph_x0[0]
    y = eta[:, 1] - sph_x0[1]
    z = eta[:, 2] - sph_x0[2]
    cart = {"x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz}

    # Spherical
    rx = eta[:, 0] - sph_x0[0]
    ry = eta[:, 1] - sph_x0[1]
    rz = eta[:, 2] - sph_x0[2]

    r = np.sqrt(rx * rx + ry * ry + rz * rz)
    inv_r = 1.0 / np.maximum(r, eps)

    vr = (rx * vx + ry * vy + rz * vz) * inv_r
    costheta = rz * inv_r
    sph_R = np.sqrt(rx * rx + ry * ry)
    inv_sph_R = 1.0 / np.maximum(sph_R, eps)

    phi = np.arctan2(ry, rx)
    vth = (rz * vr - r * vz) * inv_sph_R

    # Use cylindrical phi basis for vT (same as legacy)
    cos_phi = dx * inv_R
    sin_phi = dy * inv_R
    vT = -vx * sin_phi + vy * cos_phi

    sph = {"r": r, "cth": costheta, "phi": phi, "vr": vr, "vth": vth, "vT": vT}

    return dict(**cart, **cyl, **sph)


@dataclass(frozen=True)
class MarginalSpec:
    coordsys: str = "cart"  # cart | cyl | sph


def _keys_and_labels(coordsys: str) -> tuple[list[str], list[str]]:
    if coordsys == "cart":
        return ["x", "y", "z", "vx", "vy", "vz"], [r"$x$", r"$y$", r"$z$", r"$v_x$", r"$v_y$", r"$v_z$"]
    if coordsys == "cyl":
        return ["cylR", "cylz", "cylphi", "cylvR", "cylvz", "cylvT"], [r"$R$", r"$z$", r"$\phi$", r"$v_R$", r"$v_z$", r"$v_T$"]
    if coordsys == "sph":
        return ["r", "cth", "phi", "vr", "vth", "vT"], [r"$r$", r"$\cos\theta$", r"$\phi$", r"$v_r$", r"$v_\theta$", r"$v_T$"]
    raise ValueError(f"Unknown coordsys: {coordsys}")


def plot_1d_marginals(
    coords_train: dict[str, np.ndarray],
    coords_sample: dict[str, np.ndarray],
    *,
    fig_dir: str | None = None,
    coordsys: str = "cart",
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
    loss: float | None = None,
):
    import matplotlib.pyplot as plt

    keys, labels = _keys_and_labels(coordsys)

    fig, ax_arr = plt.subplots(2, 3, figsize=(8, 5), dpi=dpi)

    for ax, key, lab in zip(ax_arr.flat, keys, labels):
        x_train = coords_train[key]
        x_samp = coords_sample[key]

        xlim = np.nanpercentile(x_train, [1.0, 99.0])
        w = xlim[1] - xlim[0]
        xlim = [xlim[0] - 0.2 * w, xlim[1] + 0.2 * w]
        if key in {"cylR", "r"}:
            xlim[0] = max(float(xlim[0]), 0.0)
        if key == "phi" or key == "cylphi":
            xlim = [-np.pi, np.pi]
        if key == "cth":
            xlim = [-1.0, 1.0]

        kw = dict(range=(float(xlim[0]), float(xlim[1])), bins=101, density=True)
        ax.hist(x_train, label="train", alpha=0.6, **kw)
        ax.hist(x_samp, histtype="step", label="sample", alpha=0.9, **kw)
        ax.set_xlim(xlim)
        ax.set_xlabel(lab)
        ax.set_yticklabels([])

    ax_arr.flat[0].legend()
    if loss is not None:
        ax_arr.flat[1].set_title(rf"$\langle -\ln p \rangle={loss:.3g}$")

    fig.tight_layout()

    if fig_dir is None:
        return fig

    from pathlib import Path

    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    for fmt in fig_fmt:
        fig.savefig(Path(fig_dir) / f"df_marginals_1d_{coordsys}.{fmt}")
    plt.close(fig)
    return None


def plot_2d_marginal(
    coords_train: dict[str, np.ndarray],
    coords_sample: dict[str, np.ndarray],
    _eta_train=None,
    _eta_sample=None,
    fig_dir: str | None = None,
    dim1: str = "x",
    dim2: str = "y",
    *,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
    logscale: bool = False,
    bins: int = 128,
):
    """2D marginal comparison (train vs sample) + signed significance-like difference.

    This is a lightweight, TF-free analogue of `scripts/plot_flow_projections.plot_2d_marginal`.
    Passing `fig_fmt=[]` returns the matplotlib figure.
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    x_train = coords_train[dim1]
    y_train = coords_train[dim2]
    x_samp = coords_sample[dim1]
    y_samp = coords_sample[dim2]

    lims = []
    for z, key in [(x_train, dim1), (y_train, dim2)]:
        zlim = np.nanpercentile(z, [1.0, 99.0])
        w = zlim[1] - zlim[0]
        zlim = [zlim[0] - 0.2 * w, zlim[1] + 0.2 * w]
        if key in {"cylR", "r"}:
            zlim[0] = max(float(zlim[0]), 0.0)
        if key == "phi" or key == "cylphi":
            zlim = [-np.pi, np.pi]
        if key == "cth":
            zlim = [-1.0, 1.0]
        lims.append(zlim)

    xlim, ylim = lims

    fig, (ax_t, ax_s, ax_d, cax_d) = plt.subplots(
        1, 4, figsize=(10, 3), dpi=dpi, gridspec_kw=dict(width_ratios=[1, 1, 1, 0.05])
    )

    norm_hist = LogNorm() if logscale else None

    nt, xedges, yedges, _ = ax_t.hist2d(
        x_train,
        y_train,
        bins=bins,
        range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]],
        norm=norm_hist,
    )
    ax_t.set_title("train")

    # Scale sample normalization to train peak (like legacy)
    ns, _, _, _ = ax_s.hist2d(
        x_samp,
        y_samp,
        bins=[xedges, yedges],
        norm=norm_hist,
    )
    ax_s.set_title("sample")

    n_train = len(x_train)
    n_samp = len(x_samp)

    # Difference in densities, scaled by poisson-like uncertainty of sample
    # dn ~ (ns/n_s - nt/n_t) / sqrt(ns*(n_t/n_s))/n_t
    dn = (ns / max(n_samp, 1) - nt / max(n_train, 1))
    denom = (np.sqrt(np.maximum(ns, 1.0) * (n_train / max(n_samp, 1))) / max(n_train, 1))
    dn = dn / np.maximum(denom, 1.0e-12)

    vmax = np.nanpercentile(np.abs(dn), 99.0)
    vmax = max(float(vmax), 1.0)
    im = ax_d.imshow(
        dn.T,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin="lower",
        aspect="auto",
        cmap="coolwarm_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax_d.set_title("diff")

    cb = fig.colorbar(im, cax=cax_d)
    cb.set_label("diff (a.u.)")

    for ax in (ax_t, ax_s, ax_d):
        ax.set_xlabel(dim1)
        ax.set_ylabel(dim2)

    fig.tight_layout()

    if fig_dir is None or len(list(fig_fmt)) == 0:
        return fig

    from pathlib import Path

    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    for fmt in fig_fmt:
        fig.savefig(Path(fig_dir) / f"df_marginals_2d_{dim1}_{dim2}.{fmt}")
    plt.close(fig)
    return None
