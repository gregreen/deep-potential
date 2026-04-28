# PHI Training Log — Plummer Model

本文档记录了 Plummer 模型 PHI（势函数网络）的训练迭代过程，包括每次参数调整的动机、变更内容和结果。

所有 PHI 训练均使用冻结的 **df_v4a**（FFJORD, 3 blocks, [256,256,256]）作为 DF 模型。

---

## 实验总览

| Run | loss_type | l2_reg | lr_max | lr_final | warmup | batch | epochs | 额外 | 结果 |
|-----|-----------|--------|--------|----------|--------|-------|--------|------|------|
| phi_v4a | robust | 0.1 | 0.05 | 1e-5 | 0.1 | 2048 | 128 | — | ❌ Φ=常数 |
| phi_v4a_mse | mse | 0.001 | 0.01 | 5e-6 | 0.1 | 2048 | 64 | — | ⚠️ 有趋势但振荡 |
| **phi_v4a_mse2** | **mse** | **0.001** | **0.003** | **1e-5** | **0.1** | **2048** | **128** | **—** | **✅ 最优基线** |
| phi_v4a_robust | robust | 0.001 | 0.001 | 1e-6 | 0.1 | 2048 | 128 | init from mse2 | ✅ a_r 更光滑 |
| phi_v5 | mse | 0.0001 | 0.003 | 1e-6 | 0.05 | 4096 | 256 | init from mse2 | ✅ 无进一步改善 |
| phi_bc32 | mse | 0.001 | 0.001 | 1e-6 | 0.1 | 2048 | 32 | BC loss λ=0.1 | ❌ CUDA OOM |
| phi_rw32 | mse | 0.001 | 0.001 | 1e-6 | 0.1 | 2048 | 32 | reweight γ=1 | ⚠️ 尾部未改善 |

---

## 详细记录

### 1. phi_v4a — 初始尝试（robust loss）

**动机**：首次 PHI 训练，使用默认 robust (arcsinh) loss。

**参数**：
- `loss_type: robust`, `l2_reg: 0.1`, `lr_max: 0.05`
- `hidden_sizes: [512, 512, 512, 512]`, `batch_size: 2048`, `epochs: 128`

**结果**：❌ **完全失败 — Φ 输出为常数**
- Φ(r) 是一条水平线，∇Φ ≈ 0
- loss 停在 ~0.51，residual_std = 0.84

**分析**：
- `l2_reg=0.1` 太强 → 权重被推向 0 → Φ 输出常数
- `arcsinh` loss 梯度在残差较大时衰减（`arcsinh'(5) ≈ 0.20`）→ 梯度信号弱
- 两者组合形成 **常数陷阱**：l2 把权重推零，CBE 梯度太弱拉不回来

---

### 2. phi_v4a_mse — 切换 MSE loss

**动机**：突破 arcsinh 常数陷阱。MSE 梯度 ∝ residual，不会衰减。

**变更**：
- `loss_type: robust → mse`
- `l2_reg: 0.1 → 0.001`
- `lr_max: 0.05 → 0.01`
- `epochs: 128 → 64`（先跑短的看效果）

**结果**：⚠️ **突破常数陷阱，但 lr 过高导致振荡**
- loss: 1.99 → 0.025 (epoch 3) → 0.36 (epoch 63) — 早期过拟合后反弹
- Φ(r) 有正确趋势但 r~0.3-1 区域剧烈振荡
- a_r 不光滑
- residual_std = 0.63, p99 = 2.25

**分析**：
- MSE 方向正确，成功突破常数陷阱
- `lr_max=0.01` 仍然过高，导致 epoch 3 后 loss 反弹

---

### 3. phi_v4a_mse2 — 降低学习率 ⭐ 最优

**动机**：降低 lr 避免 loss 反弹和振荡。

**变更**：
- `lr_max: 0.01 → 0.003`
- `epochs: 64 → 128`

**结果**：✅ **Φ(r) 和 a_r(r) 几乎完美匹配解析解**
- loss: 1.99 → 0.018 (best, epoch 89) → 0.023 (epoch 127) — 稳定收敛
- Φ(r) 两条曲线几乎重合
- a_r(r) 光滑且准确

**Eval 指标**：
| 指标 | 值 |
|------|-----|
| residual_std | 0.1482 |
| residual_p99 | 0.5121 |
| residual_p999 | 1.0076 |
| residual_max | 2.1407 |
| ar_mae | 5.26e-3 |
| phi_mae | 4.20e-3 |

**分段 phi_mae**：
| r 区间 | phi_mae |
|--------|---------|
| 0-0.1 | 0.0029 |
| 0.1-1 | 0.0013 |
| 1-3 | 0.0027 |
| 3-10 | 0.0161 |

---

### 4. phi_v4a_robust — Robust Fine-tune

**动机**：从 mse2 的已训好权重出发，用 robust loss 做 fine-tune，尝试压制尾部异常值。

**变更**：
- `loss_type: mse → robust`
- `lr_max: 0.003 → 0.001`（phase 1），后 resume `lr_max → 0.0003`（phase 2）
- 使用新增的 `--init-params` 功能从 mse2 加载权重

**结果**：✅ **a_r 更光滑，但残差未显著改善**
| 指标 | mse2 | robust | 变化 |
|------|------|--------|------|
| residual_std | 0.1482 | 0.1492 | +0.7% |
| residual_p99 | 0.5121 | 0.5271 | +2.9% |
| ar_mae | 5.26e-3 | 4.27e-3 | **-19%** |
| phi_mae | 4.20e-3 | 6.21e-3 | +48% |

**分析**：
- Robust loss 的 arcsinh 使 a_r 曲线更光滑（ar_mae 降 19%）
- 但 phi_mae 反而升高，残差分位数也略差
- arcsinh 对 Plummer 问题的尾部压制效果有限

---

### 5. phi_v5 — 大 batch + 低 l2 + 更多 epochs

**动机**：大 r 处数据稀疏，增大 batch_size 以获得更多尾部样本。

**变更**：
- `batch_size: 2048 → 4096`
- `epochs: 128 → 256`
- `l2_reg: 0.001 → 0.0001`
- `lr_final: 1e-5 → 1e-6`
- `warmup_frac: 0.1 → 0.05`

**结果**：✅ **与 mse2 持平，尾部未改善**
| 指标 | mse2 | v5 | 变化 |
|------|------|-----|------|
| residual_std | 0.1482 | 0.1484 | +0.1% |
| ar_mae | 5.26e-3 | 4.27e-3 | -19% |
| phi_mae (r>3) | 0.0161 | 0.0175 | +8% |

**分析**：
- 增大 batch 和降低 l2 没有改善尾部
- 尾部偏差的瓶颈不在超参，而是数据稀疏 + CBE 间接监督的固有限制

---

### 6. phi_bc32 — Boundary Condition Loss 尝试

**动机**：尾部误差 phi_mae(r>3)=0.016 无法通过调参改善，尝试从物理约束入手。对孤立自引力系统，势函数满足 Φ(r→∞)→0。添加一个 BC 正则项，在远离中心的球壳上采样点，惩罚 Φ 偏离零。

**思路**：
- 在 r ∈ [r_min, r_max] = [5, 10] 的球壳内均匀采样 n_points=64 个点
- 计算 Φ(x_bc)，添加正则项 λ_bc · mean(Φ²)
- 该约束适用于所有孤立系统（不仅是 Plummer），因为只要求 Φ 在远处趋于常数

**参数设置**：
- `loss_type: mse`, `l2_reg: 0.001`, `lr_max: 0.001`
- `bc.lambda_bc: 0.1`, `bc.r_min: 5.0`, `bc.r_max: 10.0`, `bc.n_points: 64`
- 从 mse2 权重初始化，32 epochs

**实现**（在 `train_phi.py` 中）：
```python
# 在 loss_fn 内部：
key_bc = jax.random.fold_in(jax.random.key(seed), step_idx)
# 球壳内均匀体积采样
direction = jax.random.normal(key_dir, (n_points, 3))
radius = ((r_max³ - r_min³) * u + r_min³)^(1/3)
x_bc_phys = direction * radius
x_bc_std = (x_bc_phys - mean_x) / std_x
phi_bc = phi_apply(model, params, x_bc_std)
bc_loss = lambda_bc * mean(phi_bc²)
```

**结果**：❌ **CUDA graph OOM，训练无法完成**

- 第一次尝试：`jax.random` 在 `@jax.jit` 内部用 `step_idx` 做 `fold_in`，导致每步生成不同的随机 key，XLA 为每个不同 key 缓存一个 CUDA graph → 在 ~3300 步时 `RESOURCE_EXHAUSTED: Underlying backend ran out of memory trying to instantiate command buffer`
- 第二次尝试：将随机采样移到 JIT 外部（用 numpy 在 host 侧生成 BC 点，再作为参数传入 `train_step`），但 `jnp.empty((0, 3))` 作为 `lambda_bc=0` 时的占位会导致 shape 不一致 → 仍然 OOM
- 尝试添加 `XLA_FLAGS='--xla_gpu_enable_command_buffer='` 禁用 CUDA graph 可以绕过，但这会影响整体性能

**分析**：
- BC loss 的物理思路正确，但实现上与 JAX/XLA 的 CUDA graph 缓存机制冲突
- 即使采样移到 JIT 外部，额外的 `phi_apply` 前向+反向传播仍增加了编译图的复杂度
- 在当前 A100-40GB 环境下不实用

---

### 7. phi_rw32 — Radial Reweighting 尝试

**动机**：BC loss 不可行后，尝试另一种零 JIT 开销的方法。核心想法：训练数据按 DF 分布采样（∝ ρ(r)），大 r 处样本稀疏，CBE loss 中这些样本的梯度贡献微弱。通过给每个样本一个与 r 成正比的权重，放大尾部梯度信号。

**思路**：
- 对 batch 中每个样本计算物理半径 r = |x_phys|
- 权重 w(r) = (r / r_ref)^γ，然后归一化使 mean(w)=1
- 用加权均值替代等权均值：`loss = mean(w * residual²)` 或 `mean(w * arcsinh(|residual|))`
- γ=0 退化为原始等权（向后兼容），γ>0 加权尾部

**参数设置**：
- `loss_type: mse`, `l2_reg: 0.001`, `lr_max: 0.001`
- `reweight.gamma: 1.0`, `reweight.r_ref: 1.0`
- 从 mse2 权重初始化，32 epochs

**实现**（在 `train_phi.py` 中）：
```python
# 在 loss_fn 内部，利用 batch 已有的 x_std：
x_phys = x_std * std_x + mean_x
r_phys = sqrt(sum(x_phys², axis=-1) + 1e-12)
raw_w = (r_phys / r_ref) ** gamma
weights = raw_w / mean(raw_w)  # 归一化
# 传入 loss_cbe_A 或 loss_cbe_robust 做加权均值
```

在 `dpjax/physics/cbe.py` 中，`loss_cbe_A` 和 `loss_cbe_robust` 新增了可选 `weights` 参数：
- `weights=None` → 等权 `mean(...)`，完全向后兼容
- `weights=w` → 加权 `mean(w * ...)`

**结果**：⚠️ **训练完成，但尾部未改善，中心区域反而变差**

| 指标 | mse2 (基线) | rw32 (γ=1) | 变化 |
|------|-------------|------------|------|
| residual_std | 0.1482 | 0.1491 | +0.6% |
| residual_p99 | 0.5121 | 0.5168 | +0.9% |
| ar_mae | 5.26e-3 | 5.66e-3 | +7.7% |
| phi_mae | 4.20e-3 | 4.97e-3 | +18.4% |

分段 phi_mae：
| r 区间 | mse2 | rw32 | 变化 |
|--------|------|------|------|
| 0-0.1 | 0.0029 | 0.0038 | +33% |
| 0.1-1 | 0.0013 | 0.0018 | +36% |
| 1-3 | 0.0027 | 0.0029 | +8% |
| 3-10 | 0.0161 | 0.0173 | +7% |

**分析**：
- γ=1 的 reweighting 在 32 epoch 内没有改善尾部，中心区域反而显著变差
- 可能原因：(1) γ=1 太弱，权重差异不足以产生显著效果；(2) 32 epoch 不够，reweight 改变了 loss landscape，网络需要更长时间重新收敛；(3) 从 mse2 权重出发，mse2 已是等权下的最优，reweight 短期内破坏了已有的拟合
- reweighting 代码保留在代码库中（`gamma=0` 默认关闭），可供后续更长训练或更激进参数实验

---

## 关键发现

### 超参调整阶段（v4a → mse2）

1. **`l2_reg` 是最关键的参数**：0.1 → 0.001 是突破常数陷阱的核心。`l2_reg=0.1` 将权重推向零，使 Φ 输出常数；降到 0.001 后网络才能自由学习
2. **MSE loss 优于 robust loss 用于初始训练**：arcsinh 梯度 ∝ 1/√(1+x²) 在残差大时衰减，从零训练时梯度信号不足。MSE 梯度 ∝ residual，信号更强
3. **`lr_max=0.003` 是 sweet spot**：0.01 导致 epoch 3 后 loss 反弹振荡，0.001 收敛太慢
4. **Robust loss 适合 fine-tune 而非初始训练**：从 mse2 出发做 robust fine-tune，a_r 更光滑（-19%），但 phi_mae 反而升高

### 尾部改进尝试（v5 → bc32 → rw32）

5. **纯调参无法突破尾部瓶颈**：phi_v5 尝试了大 batch（4096）、低 l2（0.0001）、长训练（256 epochs），尾部 phi_mae 仍为 0.016–0.018
6. **BC loss 物理正确但工程不可行**：Φ(r→∞)→0 是正确的物理约束，但额外的 `phi_apply` 计算增加了 XLA 编译图复杂度，在 A100-40GB 上导致 CUDA graph OOM
7. **Reweighting 短期内无效**：γ=1 的 radial reweighting 在 32 epoch fine-tune 中未改善尾部，反而损害了中心区域精度
8. **尾部误差 phi_mae(r>3) ≈ 0.016 是 CBE 间接监督的固有限制**：训练数据按 DF 分布采样，大 r 处样本天然稀疏；CBE residual 是一个微分方程约束，不直接监督 Φ 的绝对值

### 当前最优配置

**phi_v4a_mse2** 仍为最佳：`loss_type=mse, l2_reg=0.001, lr_max=0.003, batch_size=2048, epochs=128`

## 未验证的潜在改进方向

- **架构级渐近约束**：修改 PotentialMLP 输出为 Φ(x) = -g(x)/|x|，内置 1/r 衰减行为，网络只需学习 g(x)→GM。最彻底但改动大，且限制了表达能力
- **更激进的 Reweighting**：γ=2 或更高，配合更长训练（128+ epochs）和从随机初始化开始（而非 fine-tune）
- **Two-phase 训练**：先全局 MSE 训练，再冻结中心区域权重只训练尾部
- **更好的 DF**：提升 DF 在大 r 处的 score 估计质量（当前 FFJORD 在尾部可能不准确）
- **Direct supervision**：对已知解析解的模型，在少量点上直接监督 Φ 值（半监督）

## 代码改进

在调试过程中，对代码库做了以下改进：

### `experiments/train_phi.py`
- 新增 `--init-params <run_dir>`：仅加载 Φ 网络权重（weights-only init），用于 fine-tune
- `--resume` 和 `--init-params` 互斥校验
- `--resume` 增加 opt_state 兼容性检查：不兼容时自动重新初始化 optimizer 并打印 WARNING
- 新增 reweighting 支持：`train.reweight.gamma` 和 `train.reweight.r_ref` 配置项，`gamma=0` 时等价于原始等权（默认关闭）
- 启动时打印 debug 信息：backend、sharding、device 数、batch size、step 计数等

### `dpjax/physics/cbe.py`
- `loss_cbe_A` 和 `loss_cbe_robust` 新增可选 `weights` 参数，支持加权均值，`weights=None` 时完全向后兼容

### 工具脚本
- `experiments/check_phi.py`：单 run 训练/eval 指标快速摘要
- `experiments/compare_phi.py`：多 run eval 结果对比，含分段 phi_mae 和 delta 计算
