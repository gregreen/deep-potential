# deep-potential
Deep learning for gravitational potentials, based on a snapshot of well-mixed
tracer particles in phase space.

The basic idea of this approach is to first model the distribution function of
the tracers using a normalizing flow. One can then calculate gradients of the
distribution function at a large number of points in phase space. Then, we
find the potential that renders the distribution function stationary at these
points. We model the potential using a feed-forward neural network, which is
both extremely flexible and easily differentiable. This latter property is
critical, as the collisionless Boltzmann equation contains gradients of the
potential (and of the distribution function).

See `notebooks/06_full_pipeline.ipynb` for an end-to-end demonstration of the
method on the Plummer Sphere toy system, and `README_JAX.md` for detailed
setup and training instructions.

This version is implemented in JAX/Flax, with some legacy code in Tensorflow 2.x.
There is an out-of-date PyTorch implementation at
[tingyuansen/deep-potential](https://github.com/tingyuansen/deep-potential).
