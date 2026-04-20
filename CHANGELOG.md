# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - 2026-04-20

### 🎯 Major Refactor: Notebook → Script-Based Pipeline

**Motivation**: 解决 Jupyter Notebook 在训练流程中的核心问题
- GPU 显存无法在训练步骤间释放（JAX 内存池机制）
- SSH 断开导致训练中途中断
- Notebook JSON 格式不利于 Git 版本控制

### ✨ New Features

#### 1. Unified Experiment Logger (`experiments/logger.py`)
- 支持多后端：W&B (Weights & Biases) + TensorBoard + CSV
- 自动降级：如果 W&B/TB 未安装，自动回退到 CSV
- 统一接口：`log_scalars()`, `finish()`, 上下文管理器支持

#### 2. Standalone CLI Scripts
| Script | Purpose |
|--------|---------|
| `experiments/gendata_plummer.py` | Plummer 球数据生成（支持 train/test 切分） |
| `experiments/train_df.py` | DF (归一化流) 训练，支持 `--override` 参数覆盖 |
| `experiments/train_phi.py` | Phi (势能网络) 训练，支持多 GPU 并行 |
| `experiments/eval_df.py` | DF 评估，新增 `--plummer-diag` 诊断模式 |
| `experiments/eval_phi.py` | Phi 评估，生成径向曲线对比图 |

#### 3. Analysis Notebook (`notebooks/07_analysis.ipynb`)
- **零 GPU 占用**：仅用于训练后可视化
- 支持加载 `metrics.csv`, `radial_curves_plummer.npz`, `eval_stats.json`
- 多实验对比功能

#### 4. Complete Documentation (`docs/pipeline_guide.md`)
- 详细说明架构变更原因（为什么 Notebook 无法释放 GPU 显存）
- 完整操作流程（5 步 pipeline）
- 后台运行指南（nohup/tmux）
- 实验监控设置（W&B / TensorBoard）

### 🔧 CLI 增强

所有训练脚本新增参数：
```bash
--override    # JSON 格式参数覆盖，如 '{"train": {"epochs": 32}}'
--logger      # 日志后端: csv | wandb | tensorboard | wandb+tb
--project     # W&B 项目名称
--run-name    # 实验运行名称
--resume      # 从 checkpoint 恢复训练
```

### 🗑️ Removed

删除过时的 Notebook 文件：
- `notebooks/01_data_generation.ipynb`
- `notebooks/02_train_df.ipynb`
- `notebooks/03_train_phi.ipynb`
- `notebooks/04_joint_finetuning.ipynb`
- `notebooks/05_visualization.ipynb`
- `notebooks/06_full_pipeline.ipynb` ⬅️ 被新流程完全替代

### 📦 Dependencies

`pyproject.toml` 新增可选依赖组：
```toml
[project.optional-dependencies]
tracking = ["wandb", "tensorboard"]
```

### 🚀 Usage Migration Guide

**Before (Notebook)**:
```python
# 在 06_full_pipeline.ipynb 中顺序执行所有 cell
# 问题：训练完 DF 后 GPU 显存不释放，Phi 训练可能 OOM
```

**After (Scripts)**:
```bash
# 1. 数据生成（CPU）
python experiments/gendata_plummer.py --total-n 524288 --test-frac 0.1 ...

# 2. DF 训练（GPU，进程结束后自动释放显存）
CUDA_VISIBLE_DEVICES=2,3 python experiments/train_df.py \
    --config configs/df_plummer_ffjord.yaml \
    --data data/plummer_train.h5 \
    --run-dir runs/plummer/df_full \
    --logger wandb+tb

# 3. DF 评估
python experiments/eval_df.py --data ... --df-run-dir ... --plummer-diag

# 4. Phi 训练（GPU，完全独立的进程）
CUDA_VISIBLE_DEVICES=2,3 python experiments/train_phi.py \
    --config configs/phi_plummer.yaml \
    --data data/plummer_train.h5 \
    --df-run-dir runs/plummer/df_full \
    --run-dir runs/plummer/phi_full \
    --logger wandb+tb

# 5. Phi 评估
python experiments/eval_phi.py --data ... --df-run-dir ... --phi-run-dir ...
```

**后台运行（SSH 安全）**:
```bash
# 方法 1: nohup
nohup python experiments/train_phi.py ... > logs/train.log 2>&1 &

# 方法 2: tmux（推荐）
tmux new -s train
# 在 tmux 会话中运行训练
# Ctrl+B 然后 D  detach，训练继续后台运行
```

### 📊 Monitoring

训练过程中实时监控（无需保持 Notebook 打开）：

- **W&B**: https://wandb.ai/<username>/dp-plummer
- **TensorBoard**: `tensorboard --logdir runs/ --port 6006`

### 🔬 Technical Details

**关键概念解释**（来自 `docs/pipeline_guide.md`）：

1. **为什么 Notebook 无法释放 GPU 显存？**
   - JAX 默认预分配 ~90% GPU 显存作为内存池
   - `del model` + `gc.collect()` 无法将内存归还给操作系统
   - 唯一可靠方式：结束进程

2. **ExperimentLogger 设计**
   - 适配器模式封装多后端差异
   - 训练脚本无感知切换日志后端
   - CSV 作为保底，确保数据不丢失

3. **Override 机制**
   - JSON 字符串递归合并到 YAML 配置
   - 快速调整参数无需创建新配置文件

### 📝 Commit Reference

```
ac5d55d refactor: migrate from notebook pipeline to script-based workflow
```

### 👥 Contributors

- Refactor designed and implemented by Cascade AI assistant
- Git commit: `refactor-jupyter` branch → `ac5d55d`

---

## [Legacy] Pre-2026-04-20

- Initial JAX/Flax implementation with RealNVP and CBE residual
- Jupyter Notebook based workflow (01-06)
- TensorFlow legacy code preserved in `legacy_tf/`
