# Deep Potential (JAX) 项目结构与核心算法说明文档

本项目（Deep Potential）旨在基于无碰撞玻尔兹曼方程 (CBE)，从相空间快照数据中恢复引力势能。当前活跃的核心代码库完全基于 **JAX + Flax + Optax** 构建。

以下是项目的整体架构、网络设计、算法实现细节、数据集及图表绘制逻辑的全面说明。

---

## 1. 整体架构与项目目录

项目采用“两阶段 + 联合微调”的训练管线：
*   **Stage 1 (DF)**: 训练归一化流以拟合示踪天体的相空间分布密度。
*   **Stage 2 ($\Phi$)**: 冻结 DF，利用 CBE 物理残差最小化来训练势能网络。
*   **Joint Fine-tuning**: 联合微调两个网络，同时最小化负对数似然 (NLL) 与 CBE 损失。

### 核心目录结构
*   **`dpjax/`**: JAX 核心实现库。
    *   `flows/`：归一化流 (Normalizing Flow) 模型定义。
    *   `models/`：引力势能 ($\Phi$) 神经网络模型。
    *   `physics/`：定义 CBE 相关的物理残差与损失函数。
    *   `data.py`：数据加载与标准化。
    *   `utils/ckpt.py`：基于 `orbax-checkpoint` 的检查点管理。
    *   `plotting/`：核心绘图与投影逻辑基础组件。
*   **`experiments/`**: 驱动整个生命周期的脚本集合。包含 `train_df.py`, `train_phi.py`, `finetune_joint.py`，以及各类独立绘图与评估脚本 (`eval_df.py`, `plot_phi_slice.py` 等)。
*   **`configs/`**: 各种训练作业的 YAML 配置文件设定（如 Batch Size、学习率调度、网络超参数等）。
*   **`scripts/`**: 包含旧版 TensorFlow 代码（带有 `_tf.py` 后缀的文件或旧笔记本，正常开发应忽略）。但该目录保留了重要的**测试数据生成脚本**（例如 `plummer/plummer_gendata.py`）。
*   **`runs/`**: 各项实验运行结果、模型权重 (Checkpoints)、日志 (`metrics.csv`) 以及可视化输出的默认保存目录。

---

## 2. 网络模型设置与配置

项目使用 `flax.linen` 进行面向对象的网络定义。

### 分布函数 (DF) 网络：统一后端（RealNVP / FFJORD）
*   **位置**: `dpjax/flows/api.py`（统一入口），后端实现位于 `dpjax/flows/realnvp.py` 和 `dpjax/flows/ffjord.py`。
*   **架构描述**:
    *   **RealNVP**：基于 Affine Coupling 的离散流，适合作为快速基线；
    *   **FFJORD**：连续归一化流（Neural ODE），当前默认参数与论文设定对齐为 **3 blocks × (3 hidden layers, 每层 128 神经元, tanh)**。
*   **输入维度**: 6D 相空间数据坐标 $(x, y, z, v_x, v_y, v_z)$。
*   **切换方式**: 通过配置项 `flow.type` 选择后端（`realnvp` / `ffjord`）。

### 引力势能 ($\Phi$) 网络
*   **位置**: `dpjax/models/potential.py`
*   **架构描述**: 标准的多层感知机 (MLP)。
*   **网络细节**:
    *   将 3D 的空间坐标 $(x, y, z)$ 映射到 1D 的标量势能 $\Phi$。
    *   默认结构与论文实现对齐：**4 层隐藏层，每层 512 神经元**。
    *   **激活函数特别要求**：各隐藏层使用 `tanh` 激活函数。避免使用 `relu`，因为 `relu` 的二阶偏导为 0，这会导致物理引力（梯度本身）在后续计算或推导时变得病态或不连续。

### CBE 损失（鲁棒形式）
*   **位置**: `dpjax/physics/cbe.py`
*   **默认训练目标**:
    *   使用 `asinh(|residual|)` 形式惩罚非平稳性，降低极端样本对训练的主导；
    *   额外加入负密度惩罚项 `asinh(max(-\nabla^2\Phi, 0))`，抑制非物理质量分布；
    *   在 `train_phi` / `finetune_joint` 中可通过 `train.loss_type` 切换为传统 `mse`。

---

## 3. 核心物理算法实现与 Python/JAX 用法

核心的物理约束逻辑集中在 `dpjax/physics/cbe.py` 中。

### 无碰撞玻尔兹曼方程 (CBE) 损失
*   **物理公式**: $v \cdot \nabla_x \log f(x,v) - \nabla_x \Phi(x) \cdot \nabla_v \log f(x,v) = 0$
*   **重缩放机制 (Rescaling)**: 这是此处的实现难点。深度学习倾向于使用归一化在 $[-1, 1]$ 附近的数据 `eta_std` 以保持梯度稳定。然而，CBE 是带有确切物理单位的方程：
    *   算法先利用封装的 `Normalizer` 获取物理数据的方差 (std) 和均值 (mean)。
    *   通过 `jax.grad` 对网络输出求导，得到在**标准化**参数空间的梯度：$\nabla_{\eta\_{std}} \log f$ (即 Score 函数) 和 $\nabla_{x\_{std}} \Phi$。
    *   方程在组合前，会将导数乘以相应坐标或速度的缩放因子将其转变回**物理单位梯度**，从而代入物理公式完成无偏计算。

### JAX 范式与语言用法
1.  **纯函数与随机化 (PRNGKey)**: 所有的网络调用（无论是前向推断 `flow.apply` 还是采样随机数）均不附带隐藏状态。遇到随机性（初始化参数、Dropout 或正常采样）严格传递伪随机数密钥 `jax.random.PRNGKey(seed)`。
2.  **批处理向量化 (vmap)**: 为了在一次矩阵运算中为数千个不同样本求解物理方程并计算损失，使用了 `jax.vmap` 对基础的损失评价函数进行并行扩展包装。
3.  **自动微分 (grad)**: 物理引力计算基于对网络的解析求导，在代码中表现为大量直接嵌入模型的偏导数计算封装函数（如 `jax.value_and_grad`）。
4.  **状态与梯度更新 (Optax)**: 没有类似 PyTorch 的 `optimizer.step()`。使用 `optax` 定义更新链（如梯度裁剪 `clip_by_global_norm` + Adam/RAdam），并通过 `optax.apply_updates(params, updates)` 返回一棵新的模型权重树。

---

## 4. 数据集处理与生成逻辑

*   **数据格式设计**: 
    模型接受的数据主要来源于仿真生成的 HDF5 文件，维度为 `(N, 6)`的数组，代表特定系统在某刻的全息相空间快照（$x, y, z, v_x, v_y, v_z$）。
*   **模拟数据生成**: 
    可以使用遗留目录下的独立 Python 脚本生成测试数据集。利用 Plummer 模型生成数据的命令样例：
    ```bash
    PYTHONPATH=./scripts python scripts/plummer/plummer_gendata.py -n 131072 -o data/plummer_n131072.h5
    ```
*   **数据流向 (dpjax/data.py)**:
    在 `train_df.py` 中，程序调用 `load_eta_h5` 将 HDF5 载入内存，并通过 `fit_normalizer` 得到标准化算子，将原始数据 $\eta$ 转换为 $\eta_{std}$，再分割成多个 Batch 交由模型消费。

---

## 5. 图表绘制与可视化

项目的图表绘制处理链路涵盖了从日志记录到切片渲染的全套逻辑。

1.  **打点与持久化追踪**: 
    运行 `train_*.py` 时，每一次 epoch 或者按设定步骤（`log_every`），代码不会产生冗余的终端文本流，而是统一将指标（如损失, 梯度的 p50/p99 值）作为结构化行写入运行目录下的 `runs/xxxx/metrics.csv`。
2.  **网络相空间投影映射**:
    `dpjax/plotting/flow_projections.py`：此模块专门针对高维分布进行边际投影，提供可复用的绘制函数。
3.  **渲染脚本**: 
    提供在 `experiments/` 目录中的功能脚本将权重和结果转为科学图表。
    *   `plot_training.py`：读取并解析 `metrics.csv`，输出训练动态曲线；
    *   `plot_phi_slice.py` / `render_phi_evolution.py`：加载独立保存の DF 和 $\Phi$ 模型参数 (通过 Orbax Checkpoint)，以三维空间进行网格切片，渲染预测引力势的等高线图或热力图并输出 `.png` 文件。

---

## 6. 开发者排错工作流与环境变量

为调试底层 JAX/XLA 问题，核心贡献者常用的环境变量调试策略：
*   **处理显存 OOM**: 默认情况 JAX 预分配 90% 显存。如果遇到多进程或显卡占用瓶颈，在终端禁掉预分配：
    ```bash
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    ```
*   **CPU Smoke 调试**: 要规避长达数分钟的 XLA 显卡编译惩罚以验证快速的语法或逻辑流变化，可强制指定使用 CPU 计算后端：
    ```bash
    export JAX_PLATFORM_NAME=cpu
    ```