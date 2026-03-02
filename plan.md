下面是一份**完整、可执行、可迭代**的《DeepPotential JAX/Flax 实现与调试方案》（面向你当前目标：先在现有仓库里实现 `dpjax/`，用 **RealNVP + log-density 接口**，先验证 **Plummer** toy model；物理残差采用 **A：`v·∇x logf - ∇Φ·∇v logf`**，并且**暂不引入 frameshift**）。

---

# 0. 总体目标与原则

## 目标
1) 在 `Mynanase/deep-potential` 仓库中新增 **JAX/Flax 版本核心实现**（不改动原 TF legacy 代码）。  
2) DF 模型采用 **离散 normalizing flow（RealNVP）**，提供 `logf(eta)` 接口。  
3) 势能模型 `Φ(q)` 用 Flax MLP，实现 `∇Φ`（必要）和可选 `∇²Φ`（以后再加）。  
4) 先在两个 toy model 上跑通**规范化调试流程**：
   - Toy 1：Plummer sphere（球对称稳态）
   - Toy 2：Harmonic blob（更贴近后续模拟数据工作流）
5) 建立一套**可复现、可定位问题、可扩展**的研究代码结构与调试规范。

## 核心原则
- **先 DF 后 Φ**：先把 flow 训练到可用，再固定 flow 训练 Φ（避免混合错误源）。  
- **以 log-density 为中心**：所有物理 loss 优先使用 `∇ log f`，比用 `∇ f` 稳定得多。  
- **强制标准化**：`eta` 六维分别标准化，否则梯度尺度与数��稳定性会拖垮你。  
- **JIT/Vmap 从第一天就用**：step 函数纯函数 + `jax.jit`；批量梯度用 `vmap`。  
- **toy 上写单元测试**：解析 toy residual≈0 是你物理实现正确性的生命线。

---

# 1. 仓库组织（不新建仓库，新增 `dpjax/`）

在 `Mynanase/deep-potential` 根目录新增：

- `dpjax/`：JAX/Flax 实现的可复用包
  - `dpjax/data.py`
  - `dpjax/flows/realnvp.py`
  - `dpjax/models/potential.py`
  - `dpjax/physics/cbe.py`
  - `dpjax/utils/{logging.py, ckpt.py, metrics.py}`（可选）
- `configs/`
  - `configs/df_plummer.yaml`
  - `configs/phi_plummer.yaml`
  - `configs/df_harmonic.yaml`
  - `configs/phi_harmonic.yaml`
- `experiments/`
  - `experiments/train_df.py`
  - `experiments/train_phi.py`
  - `experiments/eval_df.py`
  - `experiments/eval_phi.py`
- `README_JAX.md`（或在 README 里加一节）

> 原 TF notebooks/scripts 保留不动，作为参考与对照��

---

# 2. 环境与安装（miniforge conda + GPU）

## 2.1 创建环境
建议 Python 3.11（JAX 生态普遍稳定）：

```bash
mamba create -n dp-jax python=3.11 pip -y
mamba activate dp-jax
pip install -U pip
```

## 2.2 安装 JAX GPU（CUDA 12）
在 conda 里**用 pip 装 JAX CUDA wheel**（最省事）：

```bash
pip install -U "jax[cuda12]"
```

然后安装核心依赖：

```bash
pip install flax optax distrax chex orbax-checkpoint
pip install numpy scipy h5py matplotlib tqdm pyyaml
```

## 2.3 验证 GPU
```bash
python - <<'PY'
import jax
print("JAX:", jax.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
PY
```

期望看到：`Backend: gpu` 且 devices 列出你的 GPU。

---

# 3. 数据接口与标准化（必须实现且全程一致）

## 3.1 统一数据表示
- `eta`: shape `(N, 6)`，顺序固定：`[x, y, z, vx, vy, vz]`
- dtype：`float32`

## 3.2 标准化（强烈���议）
在 `dpjax/data.py` 中实现：

- `eta_mean: (6,)`
- `eta_std: (6,)`
- `eta_std = (eta - mean) / std`

并保存 `(mean, std)` 到 checkpoint 或 `run_dir/normalizer.npz`，训练与评估一致使用。

### 梯度尺度换算（关键）
若你在标准化空间里定义 `logf(eta_std)`，则：

- `score_std = ∂ logf / ∂ eta_std`
- `score_phys = score_std / eta_std_scale`（逐维除以 std）

同理，如果 `Φ` 的输入也用 `q_std`（位置的标准化），则：

- `grad_phi_std = ∂Φ / ∂ q_std`
- `grad_phi_phys = grad_phi_std / q_std_scale`

> 这一步必须在 `dpjax/physics/cbe.py` 里统一处理，否则 CBE residual 尺度会错。

---

# 4. DF：RealNVP（log-density 接口）

## 4.1 RealNVP 结构（第一版默认）
- 输入维度：6
- coupling 层数：10（起步）
- conditioner：MLP，隐藏层 2 层，hidden=128
- scale 限幅：`s = s_max * tanh(raw_s)`，`s_max = 2.0`
- 每层 permutation：最简单先用 “reverse” 或固定随机 permutation（确保表达能力）

## 4.2 mask 设计（与你物理变量更贴合）
交替两种 coupling（推荐）：
- mask0：保持 `(x,y,z)`，变换 `(vx,vy,vz)`（速度由位置条件化）
- mask1：保持 `(vx,vy,vz)`，变换 `(x,y,z)`（位置由速度条件化）

这比随机切更物理、更稳。

## 4.3 DF 训练���标
最大似然：
- loss：`nll = -mean(log_prob(eta_std))`
- 优化器：Adam（optax）
- 建议：全局范数裁剪 `clip_by_global_norm(1.0)`
- batch size：8192（起步）；显存足够再 16384

## 4.4 DF 训练成功的最低验收标准
1) `nll` 明显下降并进入平台期  
2) `sample(flow)` 的 1D/2D 边际分布接近数据  
3) `log_prob` 与 `score` 不出现 NaN/inf  
4) `score` 的统计不过分极端（比如 abs(score) 大量 > 1e3 就需要处理）

---

# 5. 势模型 Φ（Flax MLP）

## 5.1 第一版势模型（简单可用）
- 输入：`q_std = eta_std[:, :3]`
- 输出：`phi(q_std) -> (N,)`
- 网络：MLP（例如 3~4 层、hidden 256/512）  
  第一版可以用 `hidden=256`，层数 3；后面看稳定性调整。

## 5.2 必要导数
- `grad_phi = ∇_q Φ`：必须
- `laplacian`（`∇²Φ`）可先不做；等你 CBE 流程稳定后再引入密度惩罚项。

---

# 6. 物理约束：CBE residual（选择 A，且不引 frameshift）

你已选定残差形式：

\[
r(\eta) = v\cdot \nabla_x \log f(\eta)\;-\;\nabla_x\Phi(x)\cdot \nabla_v \log f(\eta)
\]

实现步骤（在标准化/物理量之间要统一处理）：

1) 从 `eta_std` 得到：
   - `x_std = eta_std[:, :3]`
   - `v_std = eta_std[:, 3:]`

2) 得到 score（标准化空间）：
   - `score_std = ∇_{eta_std} logf(eta_std)`，shape `(N,6)`
   - `score_x_std = score_std[:, :3]`
   - `score_v_std = score_std[:, 3:]`

3) 将 score 转回物理尺度（链式法则）：
   - `score_x_phys = score_x_std / std_x`
   - `score_v_phys = score_v_std / std_v`

4) 势梯度：
   - `grad_phi_std = ∇_{x_std} phi(x_std)`
   - `grad_phi_phys = grad_phi_std / std_x`

5) 速度也要取物理尺度：
   - `v_phys = v_std * std_v + mean_v`（如果你标准化时减了均值）
   - 对 toy 来说 mean_v 通常 0；但规范上写全。

6) residual：
   - `r = sum(v_phys * score_x_phys, axis=1) - sum(grad_phi_phys * score_v_phys, axis=1)`

loss：
- `loss_cbe = mean(r^2)`

> 第一版只用 `loss_cbe` 就足够验证 pipeline。后续可加入 density penalty、势的平滑正则等。

---

# 7. 训练流程（严格两阶段）

## 阶段 I：训练 DF（flow）
- 输入：toy 数据 eta
- 输出：`params_df` + normalizer(mean/std)
- 评估：NLL、样本边际对比图、score sanity check

## 阶段 II：固定 DF，训练 Φ
- `params_df` 冻结
- 训练 `params_phi` 最小化 `loss_cbe`
- 评估：
  - `loss_cbe` 是否下降
  - `grad_phi` 是否发散（统计）
  - toy 上是否能恢复正确势的形状/趋势（定性优先）

> 不建议第一版就 joint training；等两个阶段都能稳定跑通后，再考虑交替训练或端到端。

---

# 8. Toy Model 验证方案（你希望的“两套 toy”）

## Toy 1：Plummer sphere
目标：验证 stationarity residual 与势学习在经典稳态系统上合理。

建议验证顺序：
1) **解析验证（单元测试）**：先写解析 `logf(E)` 与解析 `Φ(r)`（或先只做理想 case），检查 residual 近似 0  
2) **用 RealNVP 拟合 df**：训练 flow 拟合 plummer 样本  
3) **固定 flow 训练 Φ**：看 `Φ` 是否回到正确势的形状/幅度趋势

你需要重点观察的现象：
- flow 的梯度噪声会影响势拟合 bias（原 repo 中已有类似现象）
- 训练域（r 太大点）可能导致梯度爆炸，需要裁剪（你也应在 toy 上先形成策略）

## Toy 2：Harmonic rotating blob（无 frameshift 版本）
目标：更贴近你的模拟数据工作流（裁剪、HDF5 attrs、批量采样、稳定性）。

建议先做简化：
- 先不用 frameshift，只验证基本 CBE residual 的训练与稳定性
- 等基本稳定后，再决定要不要加旋转参考系项（那会引入更多参数与退化）

---

# 9. 调试规范流程（你要求的“有规范的流程”）

## 9.1 每次实验的固定输出
- `run_dir/config.yaml`
- `run_dir/normalizer.npz`（mean/std）
- `run_dir/df_ckpt/`，`run_dir/phi_ckpt/`
- `run_dir/metrics.csv`（或 wandb/MLflow 也行）
- `run_dir/plots/`（边际分布、loss 曲线、residual 分布）

## 9.2 分层调试 checklist（照这个走）
### A. 数据层
- 维度/顺序/dtype
- 标准化统计
- 1D/2D 分布可视化

### B. Flow 层
- NLL 下降
- 采样边际对比
- `log_prob` finite
- score 统计合理
- 小样本有限差分验证 score

### C. 物理残差层
- residual 在解析 toy 上接近 0（你最重要的物理单测）
- residual 数值尺度合理（不过分大）
- 训练 Φ 时 loss 能下降

### D. 性能层
- `jax.jit` 是否反复编译（batch shape 是否固定）
- first step 编译慢是正常的，后续 step 时间稳定

## 9.3 常见问题与定位（速查）
- NaN：优先查 `exp(s)`、学习率、grad clip、标准化
- residual 不降：优先查链式法则尺度、解析 toy 单测是否过
- Φ 抖动：限制训练域、加正则、降低 lr、提升 flow 的稳定性或 ensemble（后续）

---

# 10. 默认超参数（第一版建议）
- DF (RealNVP):
  - coupling_layers: 10
  - conditioner hidden: 128
  - conditioner layers: 2
  - s_max: 2.0
  - batch_size: 8192
  - lr: 1e-3
  - grad_clip: 1.0
- Φ:
  - MLP hidden: 256
  - layers: 3
  - batch_size: 4096（CBE更重）
  - lr: 1e-3（必要时降到 3e-4）
  - grad_clip: 1.0

---

# 11. 你接下来该做的 5 个具体 TODO（最小落地顺序）
1) `dpjax/data.py`：实现标准化 + batch loader（先 numpy 即可）
2) `dpjax/flows/realnvp.py`：实现 `log_prob` 与 `sample`
3) `experiments/train_df.py`：训练 DF，并输出 NLL/采样图/score统计
4) `dpjax/models/potential.py` + `dpjax/physics/cbe.py`：实现 residual 与 `loss_cbe`
5) `experiments/train_phi.py`：加载 df_ckpt（冻结），训练 Φ，并输出 residual 分布与 loss 曲线

---

# 12. 关于“是否做成可安装包”的建议
为了后续调试不乱，我建议你采用 **A：可安装包**（`pyproject.toml`）。  
好处是：
- 导入路径稳定（不会在 notebook 里各种 `sys.path.append`）
- 后续加 pytest/CI 很自然

如果你希望我继续把方案落到“工程文件级别”（`pyproject.toml`、目录树、脚本入口、config 模板），你只要回复一句：
- 你希望包名就叫 `dpjax` 吗？（默认我建议就叫这个）

我就能给你下一步的“最小工程骨架清单”（不涉及旧仓库代码引用，完全自洽）。

# 13. 关于版本管理

使用 Git 进行版本管理，我可以相对于原项目，建立新的分支。