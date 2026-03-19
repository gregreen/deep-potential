# Copilot Instructions

## Architecture & Big Picture

*   **Deep Potential (JAX/Flax)**: This project models gravitational potentials from phase-space snapshots based on the Collisionless Boltzmann Equation (CBE).
*   **Two-Stage + Joint Training**:
    1.  **Stage 1 (DF)**: Train a Normalizing Flow (RealNVP in `dpjax/flows/`) on tracer data to fit `log_prob(eta_std)`.
    2.  **Stage 2 ($\Phi$)**: Freeze the DF, and train the Potential network (`dpjax/models/`) to minimize the CBE residual constraint.
    3.  **Joint Fine-tuning**: Jointly fine-tune both networks to enforce physical consistency ($L=\lambda_{\mathrm{cbe}}\,L_{\mathrm{CBE}}+\lambda_{\mathrm{nll}}\,\mathrm{NLL}$).
*   **Legacy Code**: Ignore legacy TensorFlow `.py` scripts in `scripts/` containing `_tf` in their names and older notebook analysis unless explicitly asked. The active framework is **JAX** + **Flax** + **Optax** located in the `dpjax/` and `experiments/` directories.

## Developer Workflows & Commands

*   **Handling JAX GPU OOM**: If GPU initialization encounters OOM, temporarily disable preallocation: `export XLA_PYTHON_CLIENT_PREALLOCATE=false`.
*   **CPU Smoke Tests**: If you need to quickly verify the pipeline without GPU compilation overhead, use `export JAX_PLATFORM_NAME=cpu`.
*   **Generating Dummy Data**: Tracers data is processed from HDF5 (shape `(N, 6)` with order `[x,y,z,vx,vy,vz]`). Generate Plummer data using `PYTHONPATH=./scripts python scripts/plummer/plummer_gendata.py -n 131072 -o data/plummer_n131072.h5`.
*   **Training Scripts (`experiments/`)**:
    *   `train_df.py`, `train_phi.py`, `finetune_joint.py` handle phases of training.
    *   Configurations are driven by YAML files in `configs/` (e.g., `configs/joint_plummer.yaml`).

## Code Patterns & JAX Defaults

*   **Functional Purity & Randomness**: Follow JAX functional programming standards. Do not mutate state. Use explicit PRNGKey passing (`jax.random.PRNGKey`).
*   **Flax Modules**: Build neural network components as `flax.linen` Modules. Use `setup` or `compact` methods correctly.
*   **Optax**: Optimization loops must use `optax` for gradient transformations and state updates.
*   **Checkpoints**: The project relies on `orbax-checkpoint` (`dpjax/utils/ckpt.py`) for saving/loading model state, separated for `df` and `phi`. Note that joint fine-tuning exports updated separated checkpoints into subdirectories.

## Integration & Extensibility

*   **CBE Loss Interaction**: The core physics interaction happens between the Flow (DF) and Potential networks calculating gradients $\nabla_x \Phi$ and $\nabla_v f$ to compute the CBE loss. Any change to the physics model requires updating `dpjax/physics/cbe.py`.
*   **Metric Logging**: Evaluation scripts output metrics to `metrics.csv` and plots to `.png` inside the `runs/` output directories.
