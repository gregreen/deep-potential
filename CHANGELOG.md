# Changelog

All notable changes to this project will be documented in this file.

**Format**: Each entry records the session intent, what changed, and links to commits.
**Trigger**: Update this file by running `/log` in Windsurf, or manually after a `git push`.
**Types**: `[AI-assisted]` = AI did implementation | `[Manual]` = human only | `[AI-generated]` = AI drafted, human approved

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
