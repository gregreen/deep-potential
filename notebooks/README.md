# Notebooks

Interactive Jupyter notebooks for the Deep Potential (JAX) project.

## Prerequisites

```bash
# From the project root, install dpjax as an editable package with notebook extras
pip install -e ".[notebook]"
```

This ensures `dpjax` is importable from any working directory, including `notebooks/`.

## Notebook Index

| Notebook | Description |
|----------|-------------|
| `01_data_generation.ipynb` | Generate Plummer mock data, load HDF5, visualize phase-space distributions |
| `02_train_df.ipynb` | Train the distribution function (RealNVP normalizing flow) |
| `03_train_phi.ipynb` | Train the gravitational potential network with frozen DF |
| `04_joint_finetuning.ipynb` | Joint fine-tuning of DF + Phi |
| `05_visualization.ipynb` | Visualize trained models: potential slices, radial curves, training metrics |
| `06_full_pipeline.ipynb` | End-to-end pipeline: data generation -> training -> evaluation |

## Tips

- **CPU debugging**: Set `JAX_PLATFORM_NAME=cpu` before starting Jupyter to avoid GPU compilation overhead during prototyping.
- **Shared GPU**: Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` if you share the GPU with other users.
- **JIT caching**: Once a cell with `@jax.jit` functions runs, the compiled code stays in memory. Re-running the cell is nearly instant.
- **Path handling**: All notebooks use `dpjax.paths` for path resolution, so they work regardless of the current working directory.
