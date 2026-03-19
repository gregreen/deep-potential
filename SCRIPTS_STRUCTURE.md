# Deep Potential (Legacy TensorFlow) 脚本结构说明文档

> ⚠️ **重要警示**：
> 本目录 (`scripts/`) 包含了项目早期基于 **TensorFlow** 和 **Sonnet** 框架构建的代码实现（通常文件名带有 `_tf.py`）。
> 这些代码目前已被 `dpjax/` 中的 **JAX/Flax** 重构版所取代。文档中所描述的 `tf.Variable`、`snt.Module` 等模型构建方法在最新的开发与主分支中**已被完全废除**，请勿在当前业务流中使用。
> 编写本文档的目的主要用于历史对照，以及保留其中依然可用的 **测试数据生成逻辑**（例如 `plummer_gendata.py`）。

本项目（Deep Potential）由于框架演进，遗留的 TensorFlow 脚本完整地记录了通过流模型还原引力势的算法雏形。以下是对该部分核心架构与实现的总结。

---

## 1. 整体架构与项目分类

旧版代码同样依赖“利用归一化流求分布函数（DF），再最小化 CBE 计算势能”的两阶段思想。但结构与封装不如当前版本严密，多以扁平脚本分散存在：

### 核心目录分类与结构
*   **归一化流实现**:
    *   `flow_tf.py`: 用于密度估计的离散常规标准化流。
    *   `flow_ffjord_tf.py`: 用于密度估计的连续标准化流 (CNF)。
*   **引力势模型**:
    *   `potential_tf.py`: 基于 TensorFlow 构建的参数化势能网络和物理 CBE 损失。
*   **动力学测试与数据生成子目录**:
    *   `plummer/`: 普卢默球测试环境与数据生成器（当前仍在使用）。
    *   `harmonic/`: 简谐子理论分布的模型。
    *   `miyamoto_nagai/`: Miyamoto-Nagai 盘模型。
*   **独立分析工具**:
    *   各种以 `plot_*` 开头的脚本，用以提取 TF 检查点数据渲染物理切片。

---

## 2. 网络模型设置与配置 (TensorFlow 时代的架构)

此部分网络采用 DeepMind 的 `sonnet` (`snt`) 库作为构建组件。

### 分布函数网络 1：离散流 (`flow_tf.py`)
*   **架构描述**: 离散标准化流。核心组件为 RealNVP 仿射耦合层。
*   **网络细节**:
    *   为了提高拟合非刚性分布的能力，通常与**有理二次样条** (Rational Quadratic Splines, RQS) 交替堆叠。
    *   使用多层感知机 (MLP) 来参数化箱体的宽度、高度和节点的斜率。基分布 (Base Distribution) 使用多元标准正态分布。
*   **特殊封装**: 提供了一个 `trainable_lu_factorization` 模块，使用 LU 分解使得 $1\times 1$ 的矩阵旋转化为可训练的一维卷积层。

### 分布函数网络 2：连续流 (`flow_ffjord_tf.py`)
*   **架构描述**: FFJORD（Free-form Continuous Dynamics），依赖于常微分方程求解。
*   **网络细节**:
    *   核心是一个名为 `ForceFieldModel` 的模块（基于 `snt.Module`）。它通过一个多层感知机来直接参数化相空间轨迹演化的导数 $dz/dt$。
    *   由于需要精确跟踪系统体积变化，依赖 `tensorflow_probability` (TFP) 提供的 Dormand-Prince 自适应时间步长 ODE 求解器。为提高求解稳定性，设计了精准闭包计算的 Jacobian 迹，并支持加入 $|v|^2$ 或 $|\nabla v|^2$ 的范数正则化进行截断惩罚。

### 引力势能网络 (`potential_tf.py`)
*   **架构描述**: 基于 `sonnet` 的前馈神经网络。
*   **网络特征**:
    *   网络也是利用 MLP 直接进行隐式表示，并且如同新版一样强制使用 `tanh` 这种高阶偏导连续且平滑的激活函数。
    *   提供了一个增强版本 `PhiNNGuided`，它不仅使用神经网络自由逼近，还会引入一定的**解析分布项指导**以加速特定物理背景下的收敛。

---

## 3. 核心物理算法机制与 Python 用法

这些旧版脚本在如何强加数学物理约束方面极具教学意义。

### 无碰撞波尔兹曼方程 (CBE) 求解法则
*   **核心函数**: `get_phi_loss_gradients` （位于 `potential_tf.py`）。
*   **算法约束细节**:
    除了和现在一致的 $(v \cdot \nabla_x \log f) - (\nabla_x \Phi \cdot \nabla_v \log f) \approx 0$ 残差计算，该脚本进一步引入了关于非惯性参考系的模型。
*   **FrameShift (非惯性系矫正层)**:
    在实际星系中，探测者可能不在稳态（例如存在自转中心偏移），代码实现了一个包含 5 个独立张量变量（例如代表旋转角速度 $w$、系统平动速度 $u$ 等）的可导网络外壳 `FrameShift`。CBE 方程在此模式下被改写为受离心力乃至科里奥利力偏置的修正模式：
    $$(v - u)\cdot \nabla_x f - (\nabla_x \Phi + w)\cdot \nabla_v f \approx 0$$
    这些非惯性特征可随梯度共同下降以还原出真实的观测系状态。

---

## 4. 模拟数据生成脚本及 Python 具体用法

测试用例库和生成器目前是被保留的重要资产。以下解析 `scripts/plummer/plummer_gendata.py`：

*   **物理对象**: Plummer 球 (一种用于模拟球状星团或矮星系的解析理论模型)。
*   **算法设计**: 
    1.  脚本首先从理论公式组装出精确的 Plummer 引力势方程与分布密度函数。
    2.  **拒绝采样 (Rejection Sampling)**: 由于从真实的六维相空间密度直接抽样极其困难，代码设计了一套蒙特卡洛随机数拒绝算法（在特定的能量空间利用包络面上均匀撒点，并判定概率保留有效坐标）。
*   **生成管线与数据格式**:
    *   利用原生的 `argparse` 获取终端要求生成的天体数量 (例如 `-n 131072`)。
    *   经过多线程随机计算拼接，得到一个形状为 `(N, 6)` 的 NumPy Array，数组包含位置 $(x, y, z)$ 和速度 $(v_x, v_y, v_z)$。
    *   **固化逻辑**：调用 Python 的 `h5py` 标准库模块：
        ```python
        with h5py.File(output_path, "w") as f:
            f.create_dataset("eta", data=sampled_array)
        ```
    最终生成的这种 `.h5` 文件成为了我们在新版 JAX 中不可或缺的基准测试源。