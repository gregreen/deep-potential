# 并行训练配置实验记录（2026-03-30）

## 1. 目标与背景

本次实验的主要目标是：

- 完成 `dpjax` 训练流程的多 GPU 并行配置；
- 验证并行训练在实际机器上确实生效（不仅是配置项打开）；
- 将过程记录成可复现、可解释的文档，便于后续复用。

补充说明：

- 本文面向的是 **JAX 多卡并行实现**，重点在 `experiments/train_df.py` 里的并行逻辑；
- 文档中同时记录了 notebook 输出清理（保证版本库整洁、降低无关 diff）。

---

## 2. 代码层并行实现说明（核心）

### 2.1 并行开关与触发条件

在 `experiments/train_df.py` 中，并行是否启用由两部分共同决定：

1. 配置开关：`train.use_pmap`（历史命名，当前实现是基于 `Mesh + NamedSharding` 的 sharding 路径）；
2. 设备数量：`jax.local_device_count() > 1`。

等价逻辑可概括为：

`use_sharding = (train.use_pmap == true) and (n_devices > 1)`

含义：

- 如果只有单卡，即使配置开关为真，也会自动退化为单设备路径；
- 只有在“配置允许 + 多设备存在”同时满足时才走并行分片。

### 2.2 Batch 与设备数对齐策略

并行训练对 batch 有硬约束：

- `batch_size >= n_devices`；
- `batch_size % n_devices == 0`。

若不能整除，代码会自动把 batch 调整为最接近的可整除值（向下取整到设备数倍数），并打印提示日志。

这样做的原因：

- 保证每张卡拿到同样大小的 `per_device_batch`；
- 避免 uneven shard 导致的 shape 不一致、编译失败或性能抖动。

### 2.3 设备网格与分片方式

并行路径里使用了：

- `Mesh(np.asarray(jax.local_devices()[:n_devices]), axis_names=("batch",))`
- 参数分片：`PartitionSpec()`（参数复制到设备）；
- 批数据分片：`PartitionSpec("batch", None)`（沿 batch 维切分）。

可理解为：

- 模型参数在各卡一致（复制）；
- 输入 batch 在 `batch` 轴上拆分到多卡；
- `jax.jit` 的训练步在这一 sharding 规则下执行，JAX 在后端完成并行调度。

### 2.4 数据放置与训练步执行

并行路径中关键步骤：

1. `params` 与 `opt_state` 放到 `replicated_sharding`；
2. 每个训练 batch 放到 `batch_sharding`；
3. 训练步 `train_step` 使用 `@jax.jit` 编译并执行；
4. 仅在需要日志或 checkpoint 时，把状态从设备取回 host（减少不必要的数据搬运）。

这个设计兼顾了：

- 并行吞吐（训练主循环大部分时间在设备端）；
- 记录与保存（周期性 host 化，便于 CSV/ckpt 持久化）。

---

## 3. 本次实验改动记录

### 3.1 Notebook 输出清理

文件：`notebooks/06_full_pipeline.ipynb`

针对“score-gradient residual histogram”绘图单元执行了输出清理：

- `execution_count` 设为 `null`；
- 清空 `outputs`；
- 保留源代码。

目的：

- 避免把大体积图片二进制输出写入 notebook diff；
- 保证后续任何人在同环境可重跑得到结果。

涉及绘图主题：

- `# Histogram of score-gradient residuals in each phase-space dimension`
- 输出文件路径：
  - `/localdisk/kosmos/my-deep-potential/runs/plummer/df_full_ffjord/plots/flow_gradients_comparison_hist.png`

### 3.2 并行状态验证（运行侧）

在项目根目录多次执行 `nvidia-smi`，观察到：

- 训练进程（`.../envs/dp-jax/bin/python`）稳定同时驻留 GPU `1/2/3`；
- 显存占用约 `30748~30768 MiB`/卡（约 `30.7 GiB`）；
- 重启后 PID 会变化，但“同一进程多卡驻留”模式保持一致；
- 抽样时刻 GPU-Util 可能接近 `0%`，属于瞬时采样现象，不影响“并行已生效”判断。

---

## 4. 如何判断“并行真的生效”

建议按下面顺序检查：

1. **看进程维度**：同一个训练 PID 是否出现在多张目标卡；
2. **看显存维度**：多卡是否都有稳定显存占用（而非偶发抖动）；
3. **看代码配置维度**：`train.use_pmap` 是否开启，`batch_size` 是否满足可整除；
4. **看日志维度**：启动日志是否出现 batch 自动调整提示（若初始 batch 不整除）。

只看单一指标（例如某个时刻 GPU 利用率）容易误判，建议至少联合前 2~3 项判断。

---

## 5. 复现命令（fish + conda）

```bash
conda activate dp-jax
nvidia-smi
```

若要进行训练复现（示例）：

```bash
conda activate dp-jax
python experiments/train_df.py \
  --config configs/df_plummer_ffjord.yaml \
  --data data/plummer_n131072.h5 \
  --run-dir runs/plummer/df_full_ffjord
```

---

## 6. 常见问题与排障建议

### 6.1 只跑到单卡

可能原因：

- `jax.local_device_count()` 只有 1（环境或可见设备受限）；
- 配置中 `train.use_pmap` 关闭；
- batch 太小或不满足分片约束，触发异常/退化路径。

建议：

- 先确认设备可见性；
- 再确认配置开关；
- 最后确认 batch 与设备数整除关系。

### 6.2 显存够但 GPU-Util 很低

常见于：

- `nvidia-smi` 抽样时刻正好在 host 日志/IO/同步阶段；
- 编译完成后训练间歇被采到；
- 作业进入数据准备或 checkpoint 保存阶段。

因此 GPU-Util 低不等于并行失效；应结合“同一 PID 多卡驻留 + 显存持续占用”判断。

### 6.3 共享集群卡位冲突

本次环境中可见多个长期驻留进程（`python3`、其他 conda 环境），会占用大量显存。

建议：

- 开跑前先看目标卡剩余显存；
- 训练期间定期采样 `nvidia-smi`；
- 必要时切换空闲卡或调整 batch。

---

## 7. 本次结论

- 并行训练路径已经配置完成；
- 从运行观测上已确认多卡并行生效；
- notebook 输出已清理，实验记录可复现、版本管理更干净；
- 当前文档可作为后续并行实验的基线模板（实现原理 + 验证标准 + 排障路径）。
