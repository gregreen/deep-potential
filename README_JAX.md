# deep-potential (dpjax)

本仓库在保留 TF legacy 代码的同时，新增 `dpjax/`：JAX/Flax 版本最小闭环实现。

当前目标：先跑通 Plummer toy（N=2**17），两阶段训练：
1) 训练 DF（RealNVP）拟合 `log_prob(eta_std)`
2) 冻结 DF，训练势 `Φ(x)` 使 CBE residual A 最小

## 环境（GPU, CUDA 12）
建议单独建环境：

```bash
conda create -n dp-jax python=3.11 pip -y
conda activate dp-jax
pip install -U pip
pip install -U "jax[cuda12]"
pip install -e .
```

验证 GPU：

```bash
python - <<'PY'
import jax
print('Backend:', jax.default_backend())
print('Devices:', jax.devices())
PY
```

如果你在启动时遇到 GPU 初始化 OOM（共享/繁忙 GPU 常见），可以临时关闭 JAX 预分配：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

或用 CPU 兜底做 smoke（慢但便于验证流程）：

```bash
export JAX_PLATFORM_NAME=cpu
```

## 生成 Plummer 数据（HDF5）
生成的 HDF5 只要求有数据集 `eta`，shape `(N,6)`，顺序 `[x,y,z,vx,vy,vz]`。

说明：Plummer 数据生成不需要 TensorFlow（已做成可选依赖）；在 `dp-jax` 环境下可直接运行。

```bash
mkdir -p data
python scripts/plummer/plummer_gendata.py -n 131072 -o data/plummer_n131072.h5
```

如果你仍遇到 `ModuleNotFoundError: toy_systems`，也可以用下面这种方式显式指定模块搜索路径：

```bash
PYTHONPATH=./scripts python scripts/plummer/plummer_gendata.py -n 131072 -o data/plummer_n131072.h5
```

在 Jupyter Notebook 里也可以运行（建议在仓库根目录启动 notebook），例如：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts").resolve()))

from plummer import plummer_gendata

eta = plummer_gendata.sample_df(131072)
plummer_gendata.save_data(eta, "data/plummer_n131072.h5")
```

## 训练 DF（RealNVP）
```bash
python experiments/train_df.py --config configs/df_plummer.yaml --data data/plummer_n131072.h5 --run-dir runs/plummer/df
```

## 训练 Φ（冻结 DF）
```bash
python experiments/train_phi.py --config configs/phi_plummer.yaml --data data/plummer_n131072.h5 --df-run-dir runs/plummer/df --run-dir runs/plummer/phi
```

## 方案 B：DF+Φ 联合微调（Joint fine-tune）
当你发现学到的力曲线（例如 `|a|(r)`）趋势不对时，通常意味着 DF 的 `score=∇ log f` 还不够物理一致。可以在两阶段完成后，用较小学习率做一段联合微调：

损失：$L=\lambda_{\mathrm{cbe}}\,L_{\mathrm{CBE}}+\lambda_{\mathrm{nll}}\,\mathrm{NLL}$

```bash
python experiments/finetune_joint.py \
	--config configs/joint_plummer.yaml \
	--data data/plummer_n131072.h5 \
	--df-run-dir runs/plummer/df \
	--phi-run-dir runs/plummer/phi \
	--run-dir runs/plummer/joint
```

输出目录：
- `runs/plummer/joint/metrics.csv`：联合训练曲线（可用 `experiments/plot_training.py` 画图）
- `runs/plummer/joint/df/`：按 `train_df.py` 的格式导出 DF（含 `normalizer.npz` 与 `ckpt/`）
- `runs/plummer/joint/phi/`：按 `train_phi.py` 的格式导出 Φ（含 `ckpt/`）

用联合微调后的模型评估（注意把 `--df-run-dir/--phi-run-dir` 指到 joint 子目录）：

```bash
python experiments/eval_phi.py \
	--data data/plummer_n131072.h5 \
	--df-run-dir runs/plummer/joint/df \
	--phi-run-dir runs/plummer/joint/phi

python experiments/plot_phi_slice.py \
	--df-run-dir runs/plummer/joint/df \
	--phi-run-dir runs/plummer/joint/phi
```

建议起点（A100）：`batch_size=2048~4096`、`lr_df=lr_phi=1e-4`、`lambda_nll=0.1~1.0`，先用 `mode: alt` 跑稳；需要更强耦合时再尝试 `mode: both`。

## Smoke（不依赖数据）
```bash
python experiments/smoke_dpjax.py
```

输出（每次运行）：
- `run-dir/config.yaml`
- `run-dir/normalizer.npz`
- `run-dir/ckpt/`（Orbax）
- `run-dir/metrics.csv`

## Jupyter Notebook 交互式开发

项目支持通过 Jupyter Notebook 进行交互式调试和训练。所有训练/评估脚本均已重构为可导入的函数，可以在 Notebook 中直接调用。

### 安装 Notebook 依赖

```bash
pip install -e ".[notebook]"
```

### 启动 JupyterLab

```bash
# 本地 CPU 调试模式（秒级编译，适合原型验证）
JAX_PLATFORM_NAME=cpu jupyter lab

# GPU 模式
XLA_PYTHON_CLIENT_PREALLOCATE=false jupyter lab
```

### Notebook 索引

`notebooks/` 目录包含以下示例：

| Notebook | 说明 |
|----------|------|
| `01_data_generation.ipynb` | 生成 Plummer 模拟数据，加载 HDF5，可视化相空间分布 |
| `02_train_df.ipynb` | 训练分布函数（RealNVP 归一化流） |
| `03_train_phi.ipynb` | 训练势能网络（冻结 DF） |
| `04_joint_finetuning.ipynb` | DF + Φ 联合微调 |
| `05_visualization.ipynb` | 可视化：势能切片、径向曲线、训练指标 |
| `06_full_pipeline.ipynb` | 端到端完整流程 |

### 在 Notebook 中调用训练

```python
from dpjax.config import load_config, merge_config
from dpjax.paths import DATA_DIR, RUNS_DIR
from experiments.train_df import run_df_training

cfg = load_config("configs/df_plummer.yaml")
cfg = merge_config(cfg, {"train": {"epochs": 4, "batch_size": 256}})

result = run_df_training(cfg, DATA_DIR / "plummer_n131072.h5", RUNS_DIR / "plummer/df")
```

### 远程 GPU 服务器（Docker）

项目提供了一个 `docker/docker-compose.yml`，可在搭载 NVIDIA GPU 的 Linux 服务器上启动 JupyterLab：

```bash
cd docker
docker compose up -d
# 浏览器打开 http://<server-ip>:8888
```
