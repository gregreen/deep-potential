# deep-potential (dpjax)

本仓库在保留 TF legacy 代码的同时，新增 `dpjax/`：JAX/Flax 版本最小闭环实现。

当前目标：先跑通 Plummer toy（N=2**17），两阶段训练：
1) 训练 DF（RealNVP 或 FFJORD）拟合 `log_prob(eta_std)`
2) 冻结 DF，训练势 `Φ(x)` 使 CBE residual A 最小

### 双后端 DF 支持

项目同时支持两种归一化流后端：
- **RealNVP**（离散耦合层）— 默认，训练快
- **FFJORD**（连续归一化流 / Neural ODE）— 需 `diffrax`，理论表达力更强

通过 `flow.type` 配置项切换（默认 `realnvp`）。所有下游脚本（`train_phi`、`finetune_joint`、`eval_*`）自动适配。

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

## 训练 DF（RealNVP，默认）
```bash
python experiments/train_df.py --config configs/df_plummer.yaml --data data/plummer_n131072.h5 --run-dir runs/plummer/df
```

## 训练 DF（FFJORD）
```bash
pip install diffrax   # 首次使用 FFJORD 需安装
python experiments/train_df.py --config configs/df_plummer_ffjord.yaml --data data/plummer_n131072.h5 --run-dir runs/plummer/df_ffjord
```

FFJORD 关键超参（在 `configs/df_plummer_ffjord.yaml` 中调整）：
- `flow.ffjord.hidden_sizes` — 速度场 MLP 宽度（默认 `[128,128,128]`）
- `flow.ffjord.n_blocks` — ODE 块数（默认 `3`）
- `flow.ffjord.solver` — ODE 求解器（`tsit5` / `dopri5`）
- `flow.ffjord.trace_type` — 散度估计（`exact` 对 dim=6 足够，`hutchinson` 可选）

> **注意**：FFJORD 每步训练开销显著高于 RealNVP（ODE 积分 + 散度计算），建议先用较少 epochs 验证收敛趋势。

### FFJORD 正则化改动说明（Jacobian / Kinetic）

> 本节用于归档“FFJORD 梯度爆炸根因修复”相关改动，便于复现与后续调参对照。

#### 1) 背景与目标

- 现象：`score_p99` 与 `score_max_abs` 在训练后期持续增大，梯度对比图中 NF 预测存在系统性放大。
- 根因：FFJORD ODE 动力学缺少复杂度约束，导致速度场与 Jacobian 过大。
- 目标：在 DF 训练中加入 Finlay et al. (2020) 风格 Jacobian/Kinetic 正则，稳定 `∇ log f` 学习。

#### 2) 核心数学形式

在每个 ODE block 中累积正则项：

`reg_cost = ∫ (kin_reg * ||f(t, x)||^2 + jac_reg * ||∂f/∂x||_F^2) dt`

训练损失改为：

`loss = - mean(log_prob) + mean(reg_cost)`

#### 3) 代码改动范围

- `dpjax/flows/ffjord.py`
  - `FFJORDConfig` 新增 `kin_reg`、`jac_reg`。
  - `_solve_block(...)` 返回从二元组升级为三元组：`(x_final, delta_logp, reg_cost)`。
  - 新增联合计算 helper（divergence + Jacobian 范数）以复用 JVP。
  - 新增 `log_prob_with_reg(...)` 与 `log_prob_reg_apply(...)`。
- `dpjax/flows/api.py`
  - registry 增加 `log_prob_reg` 路径。
  - 新增统一接口 `log_prob_reg_apply(...)`。
  - FFJORD 构建路径可读取 `flow.ffjord.kin_reg/jac_reg`。
- `dpjax/flows/__init__.py`
  - 导出 `log_prob_reg_apply`。
- `experiments/train_df.py`
  - `loss_fn` 使用 `log_prob_reg_apply`。
  - 训练日志 `metrics.csv` 新增验证列：`val_loss`、`val_score_p99`。
  - 增加 `data.val_frac` 划分训练/验证集。
  - 兼容历史 `metrics.csv` 表头并修复重复 header 导致的解析问题。

#### 4) 当前配置（`configs/df_plummer_ffjord.yaml`）

- `flow.ffjord.kin_reg: 2.0e-4`
- `flow.ffjord.jac_reg: 2.0e-4`
- `data.val_frac: 0.1`
- `train.batch_size: 4096`
- `train.epochs: 96`
- `train.lr.max: 0.003`
- `train.lr.final: 0.0002`

#### 5) 预期效果与验收

- 训练过程：`score_p99` 不再单调失控，`score_max_abs` 峰值显著下降。
- 梯度对比图：各维度散点更贴近对角线。
- 定量指标建议：
  - 每维回归斜率 `slope` 接近 `1.0`
  - 每维 `R²` 接近 `1.0`

#### 6) 快速复现实验

```bash
conda activate dp-jax
python experiments/train_df.py \
  --config configs/df_plummer_ffjord.yaml \
  --data data/plummer_n131072.h5 \
  --run-dir runs/plummer/df_ffjord
```

训练后可在 `notebooks/06_full_pipeline.ipynb` 的 DF 梯度检查单元中复核散点图与 slope/R² 标注。

## 训练 Φ（冻结 DF）
```bash
python experiments/train_phi.py --config configs/phi_plummer.yaml --data data/plummer_n131072.h5 --df-run-dir runs/plummer/df --run-dir runs/plummer/phi
```

`phi_plummer.yaml` 默认采用论文风格设置：
- `potential.hidden_sizes: [512,512,512,512]`（4 层 tanh MLP）
- `train.loss_type: robust`（`asinh(|CBE|)` + 负密度惩罚）
- `train.l2_reg: 0.1`（Φ 网络权重 L2 正则）

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
| `06_full_pipeline.ipynb` | 端到端完整流程（数据生成、DF 训练、Φ 训练、联合微调、可视化） |

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
