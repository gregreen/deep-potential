# Changelog

All notable changes to this project will be documented in this file.

**Format**: Each entry records the session intent, what changed, and links to commits.
**Trigger**: Update this file by running `/log` in Windsurf, or manually after a `git push`.
**Types**: `[AI-assisted]` = AI did implementation | `[Manual]` = human only | `[AI-generated]` = AI drafted, human approved

---

## [2026-04-22] DF 训练 v2 + 数据生成修复 `[AI-assisted]`

> **Session intent**: 使用当前 FFJORD 配置从头训练 DF（解决 v1 因 resume 不匹配导致的 ODE max_steps 崩溃），修复数据生成脚本的过采样问题，并生成 2^19 / 2^20 规模的 train/test 数据集用于正式复现实验。

### Added
- `data/plummer_n524288_train.h5` — 2^19 (524,288) 训练数据 (seed=2024, max_dist=10)
- `data/plummer_n1048576_test.h5` — 2^20 (1,048,576) 测试数据 (同源划分，无重叠)

### Changed
- `experiments/gendata_plummer.py` — 加入过采样循环保证输出恰好等于 `--total-n`；新增 `--test-n` 精确指定测试集数量；`np.random.seed()` 使采样可复现

### Training runs
- `runs/plummer/df_full_ffjord_v2` — FFJORD DF 全新训练 (96 epochs, 9696 steps)
  - 最终 loss: -130.4 | val_loss: -130.5 | score_p99: 4.22 | score_max_abs: 9.93
  - W&B: `dp-plummer/df-ffjord-v2`

---

## [2026-04-21] Vibe Coding 日志规范 `[AI-assisted]`

> **Session intent**: 建立适合 AI 辅助开发的 CHANGELOG 规范和轻量自动化提醒机制，通过 `/log` workflow 减少手动维护成本。

### Added
- `~/.windsurf/workflows/log.md` — `/log` 斜线命令：自动读取 git log → 生成格式化 CHANGELOG 条目 → commit push
- `AGENTS.md` 提醒规则 — 仅在 `git push` 或 TODO 全完成时末尾一句话提醒，其余场景零额外 token

### Changed
- `CHANGELOG.md` — 重写为 Vibe Coding 格式：Session intent + `[AI-assisted]` 标签 + Added/Changed/Removed/Commits 分类

### Commits
```
c400258 docs: normalize CHANGELOG format + add vibe-coding log workflow rules
3989b61 docs: add CHANGELOG.md documenting notebook-to-script refactor
```

---

## [2026-04-20] Script Pipeline Refactor `[AI-assisted]`

> **Session intent**: 解决 Jupyter Notebook 中 GPU 显存无法释放、SSH 断开训练中断的问题，将整个训练流程迁移为独立 CLI 脚本，并集成 W&B/TensorBoard 实验监控。

### Added
- `experiments/logger.py` — 统一日志层，支持 W&B + TensorBoard + CSV，自动降级
- `experiments/gendata_plummer.py` — Plummer 球数据生成 CLI，支持 train/test 切分
- `notebooks/07_analysis.ipynb` — 训练后分析 Notebook（零 GPU 占用）
- `docs/pipeline_guide.md` — 完整操作手册，含概念解释和命令示例
- `configs/df_plummer_ffjord.yaml` — FFJORD 流模型配置
- `dpjax/flows/ffjord.py` — FFJORD 连续归一化流实现
- `dpjax/physics/analytic.py` — Plummer 球解析函数
- `dpjax/plotting/training_curves.py` — 训练曲线绘图工具

### Changed
- `experiments/train_df.py` — 新增 `--override`, `--logger`, `--project`, `--run-name` CLI 参数；集成 ExperimentLogger
- `experiments/train_phi.py` — 同上
- `experiments/eval_df.py` — 新增 `--plummer-diag` 标志，迁移 Notebook 中的梯度/r-v/残差诊断图
- `pyproject.toml` — 新增 `tracking` 可选依赖组（wandb, tensorboard）

### Removed
- `notebooks/01_data_generation.ipynb` ~ `notebooks/06_full_pipeline.ipynb` — 被脚本流程替代

### Commits
```
ac5d55d refactor: migrate from notebook pipeline to script-based workflow
3989b61 docs: add CHANGELOG.md documenting notebook-to-script refactor
```

---

## [Legacy] Pre-2026-04-19 `[Manual]`

> **Session intent**: 初始 JAX/Flax 实现，建立 RealNVP + CBE residual 训练闭环。

### Added
- `dpjax/` — JAX/Flax 版本核心实现（RealNVP, CBE residual, Phi MLP）
- `notebooks/01-06` — Jupyter Notebook 全流程（数据生成 → DF 训练 → Phi 训练 → 评估）
- `experiments/train_df.py`, `train_phi.py`, `eval_df.py`, `eval_phi.py` — 早期脚本版本

### Commits
```
e2cf32c Refactor dpjax: Extract core logic, add Jupyter notebooks, update configs
549700d Implement JAX/Flax core for DeepPotential with RealNVP and CBE residual
```
