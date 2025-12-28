import jax.numpy as jnp
import jax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from typing import Optional

import equinox as eqx
import diffrax
import e3nn_jax as e3nn


class MLP(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key, input_dim: int, width: int, depth: int, cond_dim: int = 0):
        net_key, _ = jax.random.split(key)
        self.mlp = eqx.nn.MLP(
            in_size=input_dim + 1 + cond_dim, out_size=input_dim, width_size=width,
            depth=depth, key=net_key, activation=jax.nn.silu,
        )

    def __call__(self, t: Float, z: Array, condition: Optional[Array] = None) -> Array:
        t_vec = jnp.broadcast_to(t, z.shape[:-1] + (1,))
        net_input = [t_vec, z]
        if condition is not None:
            net_input.append(condition)
        net_input = jnp.concatenate(net_input, axis=-1)
        return self.mlp(net_input)


class FourierFeatures(eqx.Module):
    in_dim: int
    num_freqs: int
    max_freq: float
    include_self: bool
    # Non-trainable, static parameters of the module
    freqs: Float[Array, "num_freqs"]
    out_dim: int

    def __init__(
        self,
        in_dim: int,
        num_freqs: int,
        max_freq: float,
        include_self: bool = True,
    ):
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.max_freq = max_freq
        self.include_self = include_self
        # Generate frequencies on a log scale
        self.freqs = jnp.geomspace(1., max_freq, num=num_freqs)
        # Calculate the output dimension
        self.out_dim = in_dim * (2 * num_freqs)
        if self.include_self:
            self.out_dim += in_dim

    def __call__(self, x: Float | Array) -> Array:
        x = jnp.asarray(x)
        if x.ndim == 0:
            if self.in_dim != 1:
                raise ValueError("Scalar input requires in_dim=1.")
            x = x.reshape(1)

        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Input's last dim ({x.shape[-1]}) must equal in_dim ({self.in_dim}).")

        x_proj = x[..., None] * self.freqs
        encoding = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
        flat_encoding = encoding.reshape(*x.shape[:-1], -1)

        if self.include_self:
            return jnp.concatenate([x, flat_encoding], axis=-1)
        return flat_encoding


class FourierTimeMLP(eqx.Module):
    time_encoder: FourierFeatures
    mlp: eqx.nn.MLP

    def __init__(self, key, input_dim: int, width: int, depth: int, time_embedding_dim: int = 32, cond_dim: int = 0):
        self.time_encoder = FourierFeatures(
            in_dim=1, num_freqs=time_embedding_dim//2, max_freq=2**10, include_self=False
        )
        self.mlp = eqx.nn.MLP(
            in_size=input_dim + time_embedding_dim + cond_dim,
            out_size=input_dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.silu,
            key=key,
        )

    def __call__(self, t: Float, z: Array, condition: Optional[Array] = None) -> Array:
        t_emb = self.time_encoder(t)
        net_input = [t_emb, z]
        if condition is not None:
            net_input.append(condition)
        net_input = jnp.concatenate(net_input, axis=-1)
        return self.mlp(net_input)


class ResBlock(eqx.Module):
    """A standard residual block with two linear layers."""
    fn: eqx.Module

    def __init__(self, width_size: int, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key)
        self.fn = eqx.nn.Sequential([
            eqx.nn.Linear(width_size, width_size, key=key1),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(width_size, width_size, key=key2),
        ])

    def __call__(self, x: Array) -> Array:
        return x + self.fn(x)


class ResMLP(eqx.Module):
    """A ResNet-style MLP with an initial layer, residual blocks, and a final layer."""
    initial_layer: eqx.nn.Linear
    blocks: list[ResBlock]
    final_layer: eqx.nn.Linear

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int, # Depth is the number of residual blocks
        key: PRNGKeyArray,
    ):
        init_key, final_key, *block_keys = jax.random.split(key, 2 + depth)

        self.initial_layer = eqx.nn.Linear(in_size, width_size, key=init_key)
        self.blocks = [ResBlock(width_size, key=k) for k in block_keys]
        self.final_layer = eqx.nn.Linear(
            width_size, out_size, key=final_key,
        )

    def __call__(self, x: Array) -> Array:
        # Initial projection and activation
        h = jax.nn.silu(self.initial_layer(x))

        # Pass through all residual blocks
        for block in self.blocks:
            h = block(h)

        # Final projection to output size
        return self.final_layer(h)


class FourierTimeResMLP(eqx.Module):
    time_encoder: FourierFeatures
    mlp: ResMLP

    def __init__(
        self,
        key,
        input_dim: int,
        width_size: int,
        depth: int,
        time_embedding_dim: int = 32,
        cond_dim: int = 0,
    ):
        self.time_encoder = FourierFeatures(
            in_dim=1,
            num_freqs=time_embedding_dim // 2,
            max_freq=1024.0,
            include_self=False,
        )

        self.mlp = ResMLP(
            in_size=input_dim + time_embedding_dim + cond_dim,
            out_size=input_dim,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(self, t: Float, z: Array, condition: Optional[Array] = None) -> Array:
        t_emb = self.time_encoder(t)
        net_input = [t_emb, z]
        if condition is not None:
            net_input.append(condition)
        net_input = jnp.concatenate(net_input, axis=-1)
        return self.mlp(net_input)


class PosResMLP(eqx.Module):
    time_encoder: FourierFeatures
    pos_mlp: ResMLP
    mlp: ResMLP

    def __init__(
        self, key, input_dim: int, pos_width_size: int, pos_depth: int,
        width_size: int, depth: int,
        time_embedding_dim: int = 32,
        cond_dim: int = 0
    ):
        pos_mlp_key, mlp_key = jax.random.split(key)

        self.time_encoder = FourierFeatures(
            in_dim=1,
            num_freqs=time_embedding_dim // 2,
            max_freq=1024.0, # Using float for max_freq
            include_self=False,
        )

        self.pos_mlp = ResMLP(
            in_size=input_dim // 2,
            out_size=pos_width_size,
            width_size=pos_width_size,
            depth=pos_depth,
            key=pos_mlp_key,
        )

        self.mlp = ResMLP(
            in_size=input_dim + time_embedding_dim + pos_width_size + cond_dim,
            out_size=input_dim,
            width_size=width_size,
            depth=depth,
            key=mlp_key,
        )

    def __call__(self, t: Float, z: Array, condition: Optional[Array] = None) -> Array:
        t_emb = self.time_encoder(t)
        pos_emb = self.pos_mlp(z[:z.shape[-1] // 2])
        net_input = [t_emb, z, pos_emb]
        if condition is not None:
            net_input.append(condition)
        net_input = jnp.concatenate(net_input, axis=-1)
        return self.mlp(net_input)


class PositionalFourierTimeMLP(eqx.Module):
    time_encoder: FourierFeatures
    pos_encoder: FourierFeatures
    vel_encoder: FourierFeatures
    mlp: eqx.nn.MLP

    def __init__(self, key, input_dim: int, width: int, depth: int, time_embedding_dim: int = 32,
                 pos_embedding_dim: int = 12, vel_embedding_dim: int = 12, posvel_max_freq: float = 20.,
                 cond_dim: int = 0):
        self.time_encoder = FourierFeatures(
            in_dim=1, num_freqs=time_embedding_dim//2, max_freq=2**(time_embedding_dim//2 - 1), include_self=False
        )
        self.pos_encoder = FourierFeatures(
            in_dim=3, num_freqs=pos_embedding_dim//2, max_freq=posvel_max_freq, include_self=False
        )
        self.vel_encoder = FourierFeatures(
            in_dim=3, num_freqs=vel_embedding_dim//2, max_freq=posvel_max_freq, include_self=False
        )

        self.mlp = eqx.nn.MLP(
            in_size=input_dim + time_embedding_dim + 3*pos_embedding_dim + 3*vel_embedding_dim + cond_dim,
            out_size=input_dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.silu,
            key=key,
        )

    def __call__(self, t: Float, z: Array, condition: Optional[Array] = None) -> Array:
        t_emb = self.time_encoder(t)
        pos_emb = self.pos_encoder(z[:3])  # Assuming first 3 dims are position
        vel_emb = self.vel_encoder(z[3:6])  # Assuming next 3 dims are velocity
        net_input = [t_emb, z, pos_emb, vel_emb]
        if condition is not None:
            net_input.append(condition)
        net_input = jnp.concatenate(net_input, axis=-1)
        return self.mlp(net_input)


class AlternateFourierFeatures(eqx.Module):
    in_dim: int
    num_freqs: int
    sigma: float
    include_self: bool
    # Non-trainable, static parameters of the module
    freqs: Float[Array, "num_freqs"]
    out_dim: int

    def __init__(
        self,
        key: PRNGKeyArray,
        in_dim: int,
        num_freqs: int,
        sigma: float,
        include_self: bool = True,
    ):
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.sigma = sigma
        self.include_self = include_self
        # Generate frequencies on a log scale
        self.freqs = jax.random.normal(key, (num_freqs, in_dim)) * sigma
        # Calculate the output dimension
        self.out_dim = in_dim * (2 * num_freqs)
        if self.include_self:
            self.out_dim += in_dim

    def __call__(self, x: Float | Array) -> Array:
        x = jnp.asarray(x)
        if x.ndim == 0:
            if self.in_dim != 1:
                raise ValueError("Scalar input requires in_dim=1.")
            x = x.reshape(1)

        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Input's last dim ({x.shape[-1]}) must equal in_dim ({self.in_dim}).")

        x_proj = x[None, ...] * self.freqs
        encoding = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
        flat_encoding = encoding.reshape(*x.shape[:-1], -1)

        if self.include_self:
            return jnp.concatenate([x, flat_encoding], axis=-1)
        return flat_encoding


class AlternatePositionalFourierTimeMLP(eqx.Module):
    time_encoder: FourierFeatures
    pos_encoder: AlternateFourierFeatures
    vel_encoder: AlternateFourierFeatures
    mlp: eqx.nn.MLP

    def __init__(self, key, input_dim: int, width: int, depth: int, time_embedding_dim: int = 32,
                 pos_embedding_dim: int = 12, vel_embedding_dim: int = 12, sigma_pos: float = 20.,
                 sigma_vel: float = 20, cond_dim: int = 0):
        pos_key, vel_key, mlp_key = jax.random.split(key, 3)
        self.time_encoder = FourierFeatures(
            in_dim=1, num_freqs=time_embedding_dim//2, max_freq=2**(time_embedding_dim//2 - 1), include_self=False
        )
        self.pos_encoder = AlternateFourierFeatures(
            pos_key, in_dim=3, num_freqs=pos_embedding_dim//2, sigma=sigma_pos, include_self=False
        )
        self.vel_encoder = AlternateFourierFeatures(
            vel_key, in_dim=3, num_freqs=vel_embedding_dim//2, sigma=sigma_vel, include_self=False
        )

        self.mlp = eqx.nn.MLP(
            in_size=input_dim + time_embedding_dim + 3*pos_embedding_dim + 3*vel_embedding_dim + cond_dim,
            out_size=input_dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.silu,
            key=mlp_key,
        )

    def __call__(self, t: Float, z: Array, condition: Optional[Array] = None) -> Array:
        t_emb = self.time_encoder(t)
        pos_emb = self.pos_encoder(z[:3])  # Assuming first 3 dims are position
        vel_emb = self.vel_encoder(z[3:6])  # Assuming next 3 dims are velocity
        net_input = [t_emb, z, pos_emb, vel_emb]
        if condition is not None:
            net_input.append(condition)
        net_input = jnp.concatenate(net_input, axis=-1)
        return self.mlp(net_input)


class FactorizedResMLP(eqx.Module):
    """Architecture 1: Separate ResMLP encoders for pos and vel, then a main ResMLP."""
    time_encoder: FourierFeatures
    pos_encoder: ResMLP
    vel_encoder: ResMLP
    main_mlp: ResMLP

    pos_dim: int = eqx.field(static=True)
    vel_dim: int = eqx.field(static=True)

    def __init__(
        self, key, input_dim,
        pos_width, pos_depth,
        vel_width, vel_depth,
        main_width, main_depth,
        time_embedding_dim=32,
        cond_dim: int = 0
    ):
        self.pos_dim = input_dim // 2
        self.vel_dim = input_dim - self.pos_dim

        pos_key, vel_key, main_key = jax.random.split(key, 3)
        self.time_encoder = FourierFeatures(1, time_embedding_dim//2, 1024.0, False)

        self.pos_encoder = ResMLP(
            in_size=self.pos_dim,
            out_size=pos_width,
            width_size=pos_width,
            depth=pos_depth,
            key=pos_key
        )
        self.vel_encoder = ResMLP(
            in_size=self.vel_dim,
            out_size=vel_width,
            width_size=vel_width,
            depth=vel_depth,
            key=vel_key
        )

        main_in_size = pos_width + vel_width + time_embedding_dim + cond_dim
        self.main_mlp = ResMLP(
            in_size=main_in_size,
            out_size=input_dim,
            width_size=main_width,
            depth=main_depth,
            key=main_key
        )

    def __call__(self, t: Float, z: Array, condition: Optional[Array] = None) -> Array:
        pos, vel = z[:self.pos_dim], z[self.pos_dim:]
        t_emb = self.time_encoder(t)
        pos_emb = self.pos_encoder(pos)
        vel_emb = self.vel_encoder(vel)

        net_input = [pos_emb, vel_emb, t_emb]
        if condition is not None:
            net_input.append(condition)
        net_input = jnp.concatenate(net_input, axis=-1)
        return self.main_mlp(net_input)


class SHPosTimeResMLP(eqx.Module):
    """Architecture 2: ResMLP with e3nn-jax Spherical Harmonic embedding."""
    time_encoder: FourierFeatures = eqx.field(static=True)
    mlp: ResMLP
    sh_irreps: e3nn.Irreps = eqx.field(static=True)
    sh_max_l: int = eqx.field(static=True)
    pos_dim: int = eqx.field(static=True)
    vel_dim: int = eqx.field(static=True)
    pos_mean: Array
    pos_std: Array
    normalize_SH: bool = eqx.field(static=True)
    include_SH: bool = eqx.field(static=True)
    include_r2: bool = eqx.field(static=True)

    def __init__(
            self, key, input_dim, width_size, depth, sh_max_l=3, time_embedding_dim=32,
            pos_mean=jnp.zeros(3), pos_std=jnp.ones(3),
            normalize_SH=False, include_SH=True, include_r2=False,
            cond_dim: int = 0
    ):
        self.pos_dim = input_dim // 2
        self.vel_dim = input_dim // 2

        self.sh_max_l = sh_max_l
        self.sh_irreps = e3nn.Irreps.spherical_harmonics(lmax=sh_max_l) # This describes the shape of the sph features

        self.pos_mean = jnp.asarray(pos_mean)
        self.pos_std = jnp.asarray(pos_std)
        self.normalize_SH = normalize_SH
        self.include_SH = include_SH
        self.include_r2 = include_r2

        self.time_encoder = FourierFeatures(1, time_embedding_dim//2, 1024, False)

        num_sh_features = self.sh_irreps.dim
        in_size = input_dim + time_embedding_dim + cond_dim
        if include_SH:
            in_size += num_sh_features
        if include_r2:
            in_size += 1

        self.mlp = ResMLP(
            in_size=in_size,
            out_size=input_dim,
            width_size=width_size,
            depth=depth,
            key=key
        )
        # self.mlp = reinit_final_layer_small(self.mlp, time_key)

    def __call__(self, t: Float, z: Array, condition: Optional[Array] = None) -> Array:
        pos, vel = z[:self.pos_dim], z[self.pos_dim:]
        # e3nn.spherical_harmonics is robust and works directly on the position vector
        pos_emb = [pos]
        if self.include_SH:
            pos_lab_frame = pos * self.pos_std + self.pos_mean
            sh_features = e3nn.spherical_harmonics(
                self.sh_irreps, pos_lab_frame, normalize=self.normalize_SH, normalization='component'
            ).array
            pos_emb.append(sh_features)
        if self.include_r2:
            pos_lab_frame = pos * self.pos_std + self.pos_mean
            radius2 = pos_lab_frame[..., 0]**2 + pos_lab_frame[..., 1]**2 + pos_lab_frame[..., 2]**2
            pos_emb.append(radius2[..., None])  # Add an extra dimension for concatenation
        # supplement with original position, radius, radius squared
        pos_emb = jnp.concatenate(pos_emb, axis=-1)

        t_emb = self.time_encoder(t)
        net_input = [pos_emb, vel, t_emb]
        if condition is not None:
            net_input.append(condition)
        net_input = jnp.concatenate(net_input, axis=-1)

        return self.mlp(net_input)


class AngularFourierFeatures(eqx.Module):
    """Fourier features for longitude/latitude angular coordinates."""
    num_freqs: int = eqx.field(static=True)
    include_self: bool = eqx.field(static=True)

    def __init__(self, num_freqs: int, include_self: bool = False):
        self.num_freqs = num_freqs
        self.include_self = include_self

    def __call__(self, angles: Array) -> Array:
        """
        angles: shape (..., 2), [longitude, latitude] in radians
        """
        lon, lat = angles[..., 0], angles[..., 1]

        freqs = jnp.arange(1, self.num_freqs + 1)
        lon_proj = lon[..., None] * freqs
        lat_proj = lat[..., None] * freqs

        enc = [
            jnp.sin(lon_proj), jnp.cos(lon_proj),
            jnp.sin(lat_proj), jnp.cos(lat_proj),
        ]
        flat_enc = jnp.concatenate(enc, axis=-1)

        if self.include_self:
            flat_enc = jnp.concatenate([angles, flat_enc], axis=-1)

        return flat_enc


class RadialRBF(eqx.Module):
    """Radial basis expansion in r (log-spaced centers)."""
    centers: Array = eqx.field(static=True)
    gamma: float = eqx.field(static=True)

    def __init__(self, num_centers: int, r_min: float, r_max: float, gamma: float = 10.0):
        # centers spaced in log(r) to cover wide dynamic range
        self.centers = jnp.geomspace(r_min, r_max, num_centers)
        self.gamma = gamma

    def __call__(self, r: Array) -> Array:
        """
        r: shape (...,)
        returns (..., num_centers)
        """
        r = r[..., None]  # shape (..., 1)
        return jnp.exp(-self.gamma * (jnp.log(r + 1e-8) - jnp.log(self.centers))**2)


class SHRadialEmbeddingResMLP(eqx.Module):
    """Architecture 4: ResMLP with Spherical harmonics in l, b and gaussian radial basis in r."""
    time_encoder: FourierFeatures = eqx.field(static=True)
    sh_irreps: e3nn.Irreps = eqx.field(static=True)
    sh_max_l: int = eqx.field(static=True)
    rad_encoder: RadialRBF = eqx.field(static=True)
    mlp: ResMLP
    pos_dim: int = eqx.field(static=True)
    vel_dim: int = eqx.field(static=True)
    r_min: float = eqx.field(static=True)
    pos_mean: Array
    pos_std: Array

    def __init__(
            self, key, input_dim, width_size, depth,
            sh_max_l=180, num_radial=50, r_min=0.1, r_max=5.0, gamma=10.0,
            time_embedding_dim=32,
            pos_mean=jnp.zeros(3), pos_std=jnp.ones(3),
            cond_dim: int = 0
    ):
        self.pos_dim = input_dim // 2
        self.vel_dim = input_dim // 2

        self.pos_mean = jnp.asarray(pos_mean)
        self.pos_std = jnp.asarray(pos_std)

        self.time_encoder = FourierFeatures(1, time_embedding_dim//2, 1024, False)
        self.sh_max_l = sh_max_l
        self.sh_irreps = e3nn.Irreps.spherical_harmonics(lmax=sh_max_l) # This describes the shape of the sph features

        self.rad_encoder = RadialRBF(num_radial, r_min, r_max, gamma)
        self.r_min = r_min

        num_pos_features = self.sh_irreps.dim + num_radial

        in_size = input_dim + time_embedding_dim + num_pos_features + cond_dim

        self.mlp = ResMLP(
            in_size=in_size,
            out_size=input_dim,
            width_size=width_size,
            depth=depth,
            key=key
        )

    def __call__(self, t: Float, z: Array, condition: Optional[Array] = None) -> Array:
        pos, vel = z[:self.pos_dim], z[self.pos_dim:]
        pos_lab_frame = pos * self.pos_std + self.pos_mean

        r_lab_frame = jnp.linalg.norm(pos_lab_frame)
        sh_features = e3nn.spherical_harmonics(
            self.sh_irreps, pos_lab_frame, normalize=True, normalization='component'
        ).array
        # Suppress the singularity at the origin
        sh_features *= 1 - jnp.exp(-r_lab_frame**2/self.r_min**2)
        rad_features = self.rad_encoder(r_lab_frame)

        t_emb = self.time_encoder(t)
        net_input = [pos, sh_features, rad_features, vel, t_emb]
        if condition is not None:
            net_input.append(condition)
        net_input = jnp.concatenate(net_input, axis=-1)

        return self.mlp(net_input)


class Integrated3DFourierFeatures(eqx.Module):
    """
    A basis of integrated Fourier features for 3D inputs. Each Fourier feature
    is integrated along the line of sight from 0 to the input position. This is meant
    to imitate the line-of-sight effects from, for instance, integrated reddening.

    Fourier features transforms like
        \sin(\vec k \cdot \vec x + \phi_0) -->
        |\vec x|/\alpha * (\cos(\phi_0) - \cos(\alpha+\phi_0)), \alpha = \vec k \cdot \vec x

    The features are described using a wavector. The num_components wavevectors
    are uniformly spaced between k_min and k_max in log space and are isotropically
    distributed on the unit sphere. include_zero_freq adds a |vec k| = 0 component
    which is just |r|^2.
    """
    num_components: int
    k_min: float
    k_max: float
    include_zero_freq: bool
    k_basis: Array
    phases: Array
    in_dim: int
    out_dim: int

    def __init__(
        self,
        key,
        num_components: int,
        k_min: float,
        k_max: float,
        include_zero_freq: bool = True,
    ):
        self.num_components = num_components
        self.k_min = k_min
        self.k_max = k_max
        self.include_zero_freq = include_zero_freq
        self.in_dim = 3
        self.out_dim = num_components
        if include_zero_freq:
            self.out_dim += 1

        # Generate isotropic wavevectors
        k_amp = jnp.geomspace(k_min, k_max, num_components)

        # Generate unit vectors isotropically on the sphere
        def sample_unit_sphere(key, num_samples):
            u_key, v_key = jax.random.split(key)

            phi = jax.random.uniform(u_key, shape=(num_samples,), minval=0.0, maxval=2*jnp.pi)
            costheta = jax.random.uniform(v_key, shape=(num_samples,), minval=-1.0, maxval=1.0)
            sintheta = jnp.sqrt(1.0 - costheta**2)

            x = sintheta * jnp.cos(phi)
            y = sintheta * jnp.sin(phi)
            z = costheta

            return jnp.stack([x, y, z], axis=-1)

        key, subkey = jax.random.split(key)
        self.k_basis = sample_unit_sphere(key, num_components) * k_amp[:, None] # (num_components, 3)
        self.phases = jax.random.uniform(subkey, shape=(num_components,), minval=0.0, maxval=2*jnp.pi)

    def __call__(self, x: Array) -> Array:
        # Input shape is (3,), output dim is (out_dim,)
        x = jnp.asarray(x).flatten()
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Input's last dim ({x.shape[-1]}) must equal in_dim ({self.in_dim}).")

        alpha = jnp.sum(self.k_basis * x[None, :], axis=1) + 1e-6  # (num_components,)
        r = jnp.linalg.norm(x) # Scalar
        fourier_features = r / alpha * (jnp.cos(self.phases) - jnp.cos(alpha + self.phases))  # (num_components,)
        if self.include_zero_freq:
            r2_feature = jnp.array([r**2])
            fourier_features = jnp.concatenate([r2_feature, fourier_features], axis=0)  # (num_components + 1,)
        return fourier_features


class IntFourierResNet(eqx.Module):
    """ResMLP with integrated Fourier features."""
    mlp: ResMLP
    pos_dim: int
    pos_mean: Array
    pos_std: Array
    pos_encoder: Integrated3DFourierFeatures
    time_encoder: FourierFeatures = eqx.field(static=True)

    def __init__(
            self, key, input_dim, width_size, depth, time_embedding_dim=32,
            pos_mean=jnp.zeros(3), pos_std=jnp.ones(3),
            k_min=1.0, k_max=10.0, num_components=10, include_zero_freq=False
    ):
        self.pos_dim = input_dim

        key, subkey = jax.random.split(key)
        self.pos_encoder = Integrated3DFourierFeatures(
            subkey, num_components, k_min, k_max, include_zero_freq
        )

        self.pos_mean = jnp.asarray(pos_mean)
        self.pos_std = jnp.asarray(pos_std)

        self.time_encoder = FourierFeatures(1, time_embedding_dim//2, 1024, False)

        self.mlp = ResMLP(
            in_size=input_dim + self.pos_encoder.out_dim + time_embedding_dim,
            out_size=1,
            width_size=width_size,
            depth=depth,
            key=key
        )

    def __call__(self, t: Float, pos: Array, condition: Optional[Array] = None) -> Array:
        pos_emb = [pos]
        pos_lab_frame = pos * self.pos_std + self.pos_mean
        integrated_emb = self.pos_encoder(pos_lab_frame)
        pos_emb.append(integrated_emb)
        pos_emb.append(self.time_encoder(t))
        pos_emb = jnp.concatenate(pos_emb, axis=-1)
        return self.mlp(pos_emb)


def _augmented_dynamics_fn(t: Float, y: Array, args: PyTree) -> Array:
    z, _ = y
    net, condition = args
    dz_dt = net(t, z, condition)
    jac = jax.jacfwd(net, argnums=1)(t, z, condition)
    d_log_det_dt = jnp.trace(jac)
    return (dz_dt, d_log_det_dt)


class VectorField(eqx.Module):
    shape: tuple[int, ...] = eqx.field(static=True)
    cond_shape: Optional[tuple[int, ...]] = eqx.field(static=True)
    dynamics_net: MLP
    solver: diffrax.AbstractSolver = eqx.field(static=True)
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(static=True)

    def __init__(
        self, key, params,
        solver=None, stepsize_controller=None,
    ):
        input_dim = params["base_dist_dim"]
        cond_dim = params.get("cond_dim", 0)
        self.shape = (input_dim,)
        self.cond_shape = (cond_dim,) if cond_dim > 0 else None

        width = params["width"]
        depth = params["depth"]
        model_type = params["type"]
        time_emb_dim = params.get("time_embedding_dim", 32)

        if model_type == "FourierTimeMLP":
            self.dynamics_net = FourierTimeMLP(key, input_dim, width, depth, time_emb_dim, cond_dim)
        elif model_type == "MLP":
            self.dynamics_net = MLP(key, input_dim, width, depth, cond_dim)
        elif model_type == "PositionalFourierTimeMLP":
            pos_embedding_dim = params["pos_embedding_dim"]
            vel_embedding_dim = params["vel_embedding_dim"]
            self.dynamics_net = PositionalFourierTimeMLP(
                key, input_dim, width, depth, time_emb_dim,
                pos_embedding_dim, vel_embedding_dim,
                posvel_max_freq=params["posvel_max_freq"],
                cond_dim=cond_dim
            )
        elif model_type == "AlternatePositionalFourierTimeMLP":
            pos_embedding_dim = params["pos_embedding_dim"]
            vel_embedding_dim = params["vel_embedding_dim"]
            self.dynamics_net = AlternatePositionalFourierTimeMLP(
                key, input_dim, width, depth, time_emb_dim,
                pos_embedding_dim, vel_embedding_dim,
                sigma_pos=params["sigma_pos"],
                sigma_vel=params["sigma_vel"],
                cond_dim=cond_dim
            )
        elif model_type == "FourierTimeResMLP":
            self.dynamics_net = FourierTimeResMLP(
                key, input_dim, width, depth, time_emb_dim, cond_dim
            )
        elif model_type == "PosResMLP":
            pos_width = params["nn_pos_width"]
            pos_depth = params["nn_pos_depth"]
            self.dynamics_net = PosResMLP(
                key, input_dim, pos_width, pos_depth, width, depth, time_emb_dim, cond_dim
            )
        elif model_type == "FactorizedResMLP":
            self.dynamics_net = FactorizedResMLP(
                key, input_dim,
                pos_width=params["nn_pos_width"],
                pos_depth=params["nn_pos_depth"],
                vel_width=params["nn_vel_width"],
                vel_depth=params["nn_vel_depth"],
                main_width=width,
                main_depth=depth,
                time_embedding_dim=time_emb_dim,
                cond_dim=cond_dim
            )
        elif model_type == "SHPosTimeResMLP":
            self.dynamics_net = SHPosTimeResMLP(
                key, input_dim, width, depth,
                sh_max_l=params["sh_max_l"],
                time_embedding_dim=time_emb_dim,
                pos_mean=jnp.array(params["pos_mean"]),
                pos_std=jnp.array(params["pos_std"]),
                normalize_SH=params.get("normalize_SH", False),
                include_SH=params.get("include_SH", True),
                include_r2=params.get("include_r2", False),
                cond_dim=cond_dim
            )
        elif model_type == "SHRadialEmbeddingResMLP":
            self.dynamics_net = SHRadialEmbeddingResMLP(
                key, input_dim, width, depth,
                pos_mean=jnp.array(params["pos_mean"]),
                pos_std=jnp.array(params["pos_std"]),
                sh_max_l=params["sh_max_l"],
                time_embedding_dim=time_emb_dim,
                num_radial=params["pos_emb_num_radial"],
                r_min=params["pos_emb_r_min"],
                r_max=params["pos_emb_r_max"],
                gamma=params.get("pos_emb_gamma", 10),
                cond_dim=cond_dim
            )
        elif model_type == "IntFourierResNet":
            self.dynamics_net = IntFourierResNet(
                key, input_dim, width, depth,
                pos_mean=jnp.array(params["pos_mean"]),
                pos_std=jnp.array(params["pos_std"]),
                time_embedding_dim=time_emb_dim,
                k_min=params["k_min"],
                k_max=params["k_max"],
                num_components=params["num_components"],
                include_zero_freq=params["include_zero_freq"]
            )
        else:
            raise ValueError

        self.solver = solver if solver is not None else diffrax.Tsit5()
        self.stepsize_controller = (
            stepsize_controller if stepsize_controller is not None
            else diffrax.PIDController(rtol=1e-4, atol=1e-5)
        )

    def transform_and_log_det(self, x: Array, condition: Optional[Array] = None) -> tuple[Array, Array]:
        y0 = (x, jnp.zeros(()))
        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(_augmented_dynamics_fn),
            solver=self.solver, t0=0.0, t1=1.0, dt0=None, y0=y0,
            args=(self.dynamics_net, condition), saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=self.stepsize_controller, max_steps=100000
        )
        y, log_det = jax.tree_util.tree_map(lambda leaf: leaf[0], solution.ys)
        return y, log_det

    def inverse_and_log_det(self, y: Array, condition: Optional[Array] = None) -> tuple[Array, Array]:
        y0 = (y, jnp.zeros(()))
        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(_augmented_dynamics_fn),
            solver=self.solver, t0=1.0, t1=0.0, dt0=None, y0=y0,
            args=(self.dynamics_net, condition), saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=self.stepsize_controller, max_steps=100000
        )
        x, log_det_inv = jax.tree_util.tree_map(lambda leaf: leaf[0], solution.ys)
        return x, log_det_inv

    def transform(self, x: Array, condition: Optional[Array] = None) -> Array:
        """Computes the forward transformation from x to y."""
        y, _ = self.transform_and_log_det(x, condition)
        return y

    def inverse(self, y: Array, condition: Optional[Array] = None) -> Array:
        """Computes the inverse transformation from y to x."""
        x, _ = self.inverse_and_log_det(y, condition)
        return x