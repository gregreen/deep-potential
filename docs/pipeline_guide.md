# Pipeline 操作指南

本文档说明 **dpjax** 项目重构后的脚本化训练流程：做了哪些改动、为什么这样做、以及如何一步步操作。

---

## 目录

1. [改动总览](#1-改动总览)
2. [核心概念](#2-核心概念)
3. [环境准备](#3-环境准备)
4. [完整操作流程](#4-完整操作流程)
5. [参数覆盖 (Override)](#5-参数覆盖-override)
6. [实验监控 (W&B / TensorBoard)](#6-实验监控-wb--tensorboard)
7. [后台运行与断线恢复](#7-后台运行与断线恢复)
8. [训练后分析 (Notebook)](#8-训练后分析-notebook)
9. [文件清单](#9-文件清单)

---

## 1. 改动总览

### 之前（Notebook 一体化）

```
06_full_pipeline.ipynb  ←  所有逻辑都在这里
  ├── 数据生成
  ├── DF 训练          ← GPU 显存在训练后无法释放
  ├── DF 评估/画图
  ├── Phi 训练         ← 前面的显存还占着，可能 OOM
  ├── Phi 评估/画图
  └── （断开 SSH → 训练中断）
```

**问题**：
- Jupyter 内核持续占用 GPU 显存，训练完一步后显存不释放，影响后续步骤
- SSH 断开 → Jupyter 内核死掉 → 训练中断
- `.ipynb` 文件是 JSON 格式，Git diff 几乎不可读

### 现在（脚本 + 工具监控 + 分析 Notebook）

```
终端运行脚本（每个脚本是独立进程）
  ├── experiments/gendata_plummer.py   ← 数据生成（CPU）
  ├── experiments/train_df.py          ← DF 训练（GPU，进程结束 → 显存释放）
  ├── experiments/eval_df.py           ← DF 评估（GPU，进程结束 → 显存释放）
  ├── experiments/train_phi.py         ← Phi 训练（GPU，进程结束 → 显存释放）
  └── experiments/eval_phi.py          ← Phi 评估（GPU，进程结束 → 显存释放）

W&B / TensorBoard ← 实时监控训练曲线（浏览器/手机随时看）

notebooks/07_analysis.ipynb ← 训练完成后加载结果画图分析（不占 GPU）
```

**好处**：
- 每个脚本跑完进程退出，GPU 显存**自动完全释放**
- 可以用 `nohup` / `tmux` 后台运行，SSH 断开不影响
- `.py` 文件 Git diff 清晰
- W&B 实时看训练曲线，不依赖 Notebook

---

## 2. 核心概念

### 2.1 为什么 Notebook 无法释放 GPU 显存？

JAX（和 PyTorch/TensorFlow 一样）使用 **GPU 显存池**：

1. 第一次使用 GPU 时，JAX 默认**预分配 ~90%** 的 GPU 显存（`XLA_PYTHON_CLIENT_PREALLOCATE=true`）
2. 即使 `del model` + `gc.collect()`，显存仍然在 JAX 的内存池中，**不会归还给操作系统**
3. 唯一可靠释放显存的方式是**结束进程**

所以：Notebook 内核（一个长驻进程）里跑完训练后，显存永远被占着。
而独立脚本跑完进程就退出了，操作系统会回收所有显存。

### 2.2 什么是 `--override`？

训练脚本接受一个 YAML 配置文件（如 `configs/phi_plummer.yaml`），定义了所有超参数。
`--override` 允许你在命令行上**临时修改**部分参数，无需创建新的配置文件：

```bash
# 基础配置: phi_plummer.yaml 里 epochs=1024, hidden_sizes=[512,512,512,512]
# 用 --override 临时改为 epochs=32, hidden_sizes=[128,128]
python experiments/train_phi.py \
  --config configs/phi_plummer.yaml \
  --override '{"train": {"epochs": 32}, "potential": {"hidden_sizes": [128, 128]}}'
```

Override 是一个 **JSON 字符串**，会**递归合并**到基础配置中（只替换指定的键，其余不变）。

### 2.3 什么是 W&B (Weights & Biases)？

W&B 是一个**实验追踪工具**，类似训练过程的"仪表盘"：

- 训练脚本每隔几步自动上传 loss、学习率等指标到 W&B 服务器
- 你可以在**浏览器**（甚至手机）上实时看训练曲线
- 训练脚本在后台静默运行，你不需要开着 Notebook 来看图
- 免费版足够个人使用，注册地址: https://wandb.ai

### 2.4 什么是 TensorBoard？

TensorBoard 是 Google 出的**本地实验可视化工具**：

- 训练时日志写入本地 `{run_dir}/tb_logs/` 目录
- 另开一个终端启动 `tensorboard --logdir runs/`，然后浏览器打开 `http://localhost:6006`
- 不需要注册账号，完全离线使用

### 2.5 ExperimentLogger — 统一日志层

我们创建了 `experiments/logger.py`，它是一个**适配器**，同时支持多种日志后端：

```
ExperimentLogger(backend="wandb+tb")
  ├── CSV 后端  ← 永远启用，写 metrics.csv（保底）
  ├── W&B 后端  ← 可选，实时上传到云端
  └── TB 后端   ← 可选，写入本地 TensorBoard 日志
```

通过 `--logger` 参数选择：
- `--logger csv` — 只写 CSV（默认，最简单）
- `--logger wandb` — CSV + W&B
- `--logger tensorboard` — CSV + TensorBoard
- `--logger wandb+tb` — CSV + W&B + TensorBoard（全开）

如果 W&B 或 TensorBoard 没安装，会**自动降级**到 CSV，不会报错。

---

## 3. 环境准备

```bash
# 激活环境
conda activate dp-jax

# 安装实验追踪依赖（可选，但推荐）
pip install wandb tensorboard

# 首次使用 W&B 需要登录（会给你一个 API key）
wandb login
```

---

## 4. 完整操作流程

以 Plummer 球为例，完整的 5 步 pipeline：

### Step 1: 生成数据

```bash
conda activate dp-jax

python experiments/gendata_plummer.py \
  --total-n 524288 \
  --test-frac 0.1 \
  --max-dist 10.0 \
  --train-out data/plummer_train.h5 \
  --test-out data/plummer_test.h5
```

**参数说明**：
| 参数 | 含义 |
|------|------|
| `--total-n` | 总样本数（会按 test-frac 切分） |
| `--test-frac` | 测试集比例（0.1 = 10%） |
| `--max-dist` | Plummer 球采样的最大半径 |
| `--train-out` | 训练集输出路径 |
| `--test-out` | 测试集输出路径 |

这一步是纯 CPU 操作，不需要 GPU。

### Step 2: 训练 DF（Density Field / 归一化流）

```bash
python experiments/train_df.py \
  --config configs/df_plummer_ffjord.yaml \
  --data data/plummer_train.h5 \
  --run-dir runs/plummer/df_full \
  --logger wandb+tb
```

**参数说明**：
| 参数 | 含义 |
|------|------|
| `--config` | 基础配置文件（YAML） |
| `--data` | 训练数据路径 |
| `--run-dir` | 实验输出目录（checkpoint、metrics.csv、日志） |
| `--logger` | 日志后端（见 2.5 节） |
| `--override` | 可选，JSON 覆盖参数（见 2.2 节） |
| `--resume` | 可选 flag，从上次 checkpoint 恢复训练 |

**输出**：
```
runs/plummer/df_full/
  ├── ckpt/              ← 模型 checkpoint（orbax 格式）
  ├── metrics.csv        ← 训练指标（step, loss, score stats, ...）
  ├── config.json        ← 本次运行的完整配置快照
  ├── normalizer.npz     ← 数据标准化参数
  ├── flow_config.yaml   ← 流模型配置
  └── tb_logs/           ← TensorBoard 日志（如果开了）
```

### Step 3: 评估 DF

```bash
python experiments/eval_df.py \
  --data data/plummer_train.h5 \
  --df-run-dir runs/plummer/df_full \
  --plummer-diag
```

`--plummer-diag` 启用 Plummer 球特有的诊断图：
- **梯度对比图** (`flow_gradients_comparison.png`)：学到的 score 梯度 vs 解析真值
- **残差直方图** (`flow_gradients_comparison_hist.png`)：每个维度的 score 误差分布
- **r-v 分布对比** (`df_rv_comparison.png`)：流模型采样 vs 理想分布

**输出**保存到 `{df-run-dir}/plots/`。

### Step 4: 训练 Phi（势能网络）

```bash
python experiments/train_phi.py \
  --config configs/phi_plummer.yaml \
  --data data/plummer_train.h5 \
  --df-run-dir runs/plummer/df_full \
  --run-dir runs/plummer/phi_full \
  --logger wandb+tb
```

注意 `--df-run-dir` 指向已训练好的 DF 目录。Phi 训练**冻结 DF**，只训练势能网络。

### Step 5: 评估 Phi

```bash
python experiments/eval_phi.py \
  --data data/plummer_test.h5 \
  --df-run-dir runs/plummer/df_full \
  --phi-run-dir runs/plummer/phi_full
```

**输出**：
```
runs/plummer/phi_full/
  ├── eval_stats.json              ← 残差统计（mean, std, p99, max）
  ├── radial_curves_plummer.npz    ← 径向曲线数据（可在 notebook 中加载画图）
  └── plots/
      ├── phi_r_plummer.png        ← 势能径向曲线
      └── ar_r_plummer.png         ← 径向加速度曲线
```

---

## 5. 参数覆盖 (Override)

### 快速验证用（小网络、少 epoch）

```bash
python experiments/train_phi.py \
  --config configs/phi_plummer.yaml \
  --data data/plummer_train.h5 \
  --df-run-dir runs/plummer/df_full \
  --run-dir runs/plummer/phi_smoke \
  --override '{
    "potential": {"hidden_sizes": [64, 64]},
    "train": {"epochs": 2, "batch_size": 256, "lr": 0.001, "log_every": 1, "ckpt_every": 0}
  }'
```

### 正式训练用（大网络、多 epoch）

```bash
python experiments/train_phi.py \
  --config configs/phi_plummer.yaml \
  --data data/plummer_train.h5 \
  --df-run-dir runs/plummer/df_full \
  --run-dir runs/plummer/phi_full \
  --override '{
    "potential": {"hidden_sizes": [512, 512, 512, 512]},
    "train": {"epochs": 1024, "batch_size": 2048}
  }' \
  --logger wandb+tb
```

---

## 6. 实验监控 (W&B / TensorBoard)

### 使用 W&B

```bash
# 训练时加 --logger wandb（或 wandb+tb）
python experiments/train_df.py --config ... --logger wandb --project dp-plummer --run-name df-full-v1

# 打开浏览器访问 https://wandb.ai/<你的用户名>/dp-plummer 即可实时看曲线
```

### 使用 TensorBoard

```bash
# 训练时加 --logger tensorboard
python experiments/train_df.py --config ... --logger tensorboard

# 另开终端启动 TensorBoard
conda activate dp-jax
tensorboard --logdir runs/ --port 6006

# 浏览器打开 http://localhost:6006
```

如果是通过 SSH 连接服务器，需要做端口转发：
```bash
# 在本地机器上执行
ssh -L 6006:localhost:6006 your-server
```

---

## 7. 后台运行与断线恢复

### 方法 A: nohup（最简单）

```bash
nohup python experiments/train_phi.py \
  --config configs/phi_plummer.yaml \
  --data data/plummer_train.h5 \
  --df-run-dir runs/plummer/df_full \
  --run-dir runs/plummer/phi_full \
  --logger wandb \
  > logs/train_phi.log 2>&1 &

# 查看日志
tail -f logs/train_phi.log

# 查看进程是否在跑
ps aux | grep train_phi
```

### 方法 B: tmux（推荐，可以随时回来看）

```bash
# 创建一个命名会话
tmux new -s train

# 在 tmux 里运行训练
conda activate dp-jax
python experiments/train_phi.py --config ... --logger wandb+tb

# 断开（不会终止训练）：按 Ctrl+B 然后按 D

# 下次 SSH 回来后，重新连接
tmux attach -t train
```

### 断线恢复训练

如果训练被中断了（无论什么原因），用 `--resume` 从最近的 checkpoint 继续：

```bash
python experiments/train_phi.py \
  --config configs/phi_plummer.yaml \
  --data data/plummer_train.h5 \
  --df-run-dir runs/plummer/df_full \
  --run-dir runs/plummer/phi_full \
  --resume \
  --logger wandb
```

---

## 8. 训练后分析 (Notebook)

训练和评估都在终端完成后，打开分析 Notebook：

```
notebooks/07_analysis.ipynb
```

这个 Notebook **不导入 JAX，不占用 GPU**。它只做：
- 从 `metrics.csv` 读取训练曲线画图
- 从 `radial_curves_plummer.npz` 加载径向曲线对比图
- 从 `eval_stats.json` 读取残差统计
- 显示 `plots/` 目录下的诊断图
- 支持多个实验的对比

使用方法：打开 notebook 后修改顶部的 `DF_RUN` 和 `PHI_RUN` 路径指向你的实验目录，然后从上到下运行所有 cell。

---

## 9. 文件清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `experiments/logger.py` | 统一日志层（W&B + TensorBoard + CSV） |
| `experiments/gendata_plummer.py` | Plummer 数据生成 CLI 脚本 |
| `notebooks/07_analysis.ipynb` | 训练后分析 Notebook（不占 GPU） |
| `docs/pipeline_guide.md` | 本文档 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `experiments/train_df.py` | 新增 `--override`、`--logger`、`--project`、`--run-name` 参数；训练循环集成 logger |
| `experiments/train_phi.py` | 同上 |
| `experiments/eval_df.py` | 新增 `--plummer-diag` 参数；迁移 Notebook 中的梯度对比、r-v 分布、残差直方图 |
| `pyproject.toml` | 新增 `tracking` 可选依赖组（wandb, tensorboard） |

### 删除文件

| 文件 | 原因 |
|------|------|
| `notebooks/06_full_pipeline.ipynb` | 被脚本 pipeline + `07_analysis.ipynb` 替代 |
