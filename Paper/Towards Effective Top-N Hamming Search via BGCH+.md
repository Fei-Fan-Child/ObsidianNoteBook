---
tags:
  - paper
  - graph-hashing
  - gnn
  - bipartite-graph
  - recommendation
  - contrastive-learning
source: "Paper/Towards_Effective_Top-N_Hamming_Search_via_Bipartite_Graph_Contrastive_Hashing.pdf"
venue: "IEEE TKDE Vol. 36, No. 12, Dec 2024"
authors: "Yankai Chen, Yixiang Fang, Yifei Zhang, Chenhao Ma, Yang Hong, Irwin King"
created: 2026-07-19
---

# Towards Effective Top-N Hamming Search via Bipartite Graph Contrastive Hashing (BGCH+)

> Chen et al., **IEEE TKDE 2024** · 二部图哈希检索 + 对比学习

---

## 🎯 问题背景

**二部图（Bipartite Graph）** 在现实世界极其常见：

- 用户-商品推荐（user-product）
- 查询-文档匹配（query-document）
- 在线问答系统

**Top-N 搜索任务**：给定一个 query 节点，找到最相关的 N 个目标节点。

传统做法：用 GCN 学习**连续向量嵌入**再做相似度匹配 —— 但有**两个核心问题**：

| 问题 | 描述 |
|------|------|
| 🔴 **计算成本** | 全精度 float32 嵌入比对慢、内存占用大 |
| 🔴 **信息损失** | Hash 码维度 d 只有 $2^d$ 种编码模式，表达力远低于连续向量 |
| 🔴 **优化困难** | sign(·) 函数在 0 处不可微、其余处处导数为 0，传统用 tanh(·) 替代会**前后向优化方向不一致** |

---

## 💡 核心贡献：BGCH+

BGCH+ 是 BGCH 的改进版，主要引入 **Dual Feature Contrastive Learning（双视图特征对比学习）**。

### 整体框架（Fig. 2）

```
                  ┌────────────────────────────────────────────┐
                  │        Bipartite Graph G = {V₁, V₂, E}     │
                  └────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌──────────────────────────────────────────────────────┐
        │   Adaptive Graph Convolutional Hashing (III-B)        │
        │   ─ 节点自适应 rescaling 因子 x(l) = 1/(d‖V‖₁)        │
        │   ─ L 层卷积后拼接 Qx = Qx⁽⁰⁾‖Qx⁽¹⁾‖…‖Qx⁽ᴸ⁾           │
        └──────────────────────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
    ┌──────────────────────────┐      ┌──────────────────────────┐
    │  Continuous Embedding Vₓ │      │  Hash Code Qₓ ∈ {-1,+1}^d │
    └──────────────────────────┘      └──────────────────────────┘
                  │                                   │
                  ▼                                   ▼
    ┌──────────────────────────┐      ┌──────────────────────────┐
    │  Augmentation Vₓ', Vₓ''  │      │  Augmentation Qₓ', Qₓ''  │
    │  (add random noise)      │      │  (add noise to scalar x)  │
    └──────────────────────────┘      └──────────────────────────┘
                  │                                   │
                  ▼                                   ▼
    ┌──────────────────────────┐      ┌──────────────────────────┐
    │  L₁ᵧˡ (continuous CL)    │      │  L₂ᵧˡ (binary CL)        │
    │  InfoNCE loss            │      │  Hamming-space inner prod │
    └──────────────────────────┘      └──────────────────────────┘
                  └─────────────────┬─────────────────┘
                                    ▼
                  ┌─────────────────────────────────────┐
                  │  L = L_BPR + λ₁L_CL + λ₂‖Θ‖²₂       │
                  └─────────────────────────────────────┘
                                    │
                                    ▼
                       Top-N Hamming Search
```

---

## 🔬 三大核心模块

### 1️⃣ Adaptive Graph Convolutional Hashing（自适应图卷积哈希）

**创新点**：每个节点都有一个**自适应的 rescaling 因子**：

$$
\alpha_x^{(l)} = \frac{1}{d\,\|V_x^{(l)}\|_1}, \quad V_x^{(l)} \in \mathbb{R}^d_+, \quad Q_x^{(l)} \leftarrow \alpha_x^{(l)} Q_x^{(l)}
$$

**作用**：
- 不引入额外可学习参数（确定性计算，节省参数空间）
- 自适应平滑损失景观（smooth loss landscape）
- 提高 hash 码的表达力

**多层拼接**（公式 6）：

$$
\alpha_x = \alpha_x^{(0)} \| \alpha_x^{(1)} \| \cdots \| \alpha_x^{(L)}, \quad Q_x = Q_x^{(0)} \| Q_x^{(1)} \| \cdots \| Q_x^{(L)}
$$

> 💡 **关键 insight**：拼接所有层的中间哈希码，既保留了中间知识，又维持了对原始连续嵌入的近似。

---

### 2️⃣ Dual Feature Contrastive Learning（双视图对比学习）

**核心思想**：对**特征**做增强（而非对**图结构**做 dropout），避免稀疏图上结构扰动带来的不可控影响。

#### 连续嵌入增强（公式 7-8）

$$
\tilde{V}_x^{(l)} = V_x^{(l)} + \varepsilon_x^{(l)}, \quad \hat{V}_x^{(l)} = V_x^{(l)} + \hat{\varepsilon}_x^{(l)}
$$

噪声约束（保证不破坏原信息方向）：

$$
\varepsilon_x^{(l)} \sim \mathcal{U}(0,1), \quad \varepsilon_x^{(l)} = \varepsilon_x^{(l)} \odot Q_x^{(l)}
$$

- 第一项控制噪声**幅度**（位于半径为 ρ 的超球面）
- 第二项保持与 $Q_x^{(l)}$ **方向一致**，避免显著偏离原向量

#### Hash 码增强（公式 9）

直接扰动二进制会破坏哈希的离散性，所以改为扰动 **scalar 因子 $\alpha_x^{(l)}$**：

$$
\tilde{\alpha}_x^{(l)} = \alpha_x^{(l)} + \varepsilon_x^{(l)}, \quad \hat{\alpha}_x^{(l)} = \alpha_x^{(l)} + \hat{\varepsilon}_x^{(l)}
$$

> 这样扰动通过 $\alpha \times Q$ 间接作用到 hash 码，但**保留了 binarization 性质**。

#### 双 InfoNCE Loss（公式 10、12）

$$
\mathcal{L}_{1}^{cl} = -\log \frac{\exp(\tilde{V}_x^\top \hat{V}_x)}{\sum_{y \in \mathcal{B}} \exp(\tilde{V}_x^\top \hat{V}_y)}
$$

$$
\mathcal{L}_{2}^{cl} = -\log \frac{\exp(\rho \cdot Q_x^\top \hat{Q}_x)}{\sum_{y \in \mathcal{B}} \exp(\rho \cdot Q_x^\top \hat{Q}_y)}
$$

> 其中 $\mathcal{L}_2^{cl}$ 中的 $Q_x^\top Q_y$ 可在 **Hamming 空间**用位运算高效计算。

---

### 3️⃣ Fourier Serialized Gradient Estimation（傅里叶级数梯度估计）

**问题**：$\text{sign}(x)$ 在 0 处不可微、其余处处导数为 0。

**传统做法**：用 $\tanh(x)$ 近似 → 但**前向（sign）和后向（tanh 梯度）方向不一致**，导致训练震荡。

**本文做法**：将 $\text{sign}(x)$ 视为**方波的特例**，用傅里叶级数展开（公式 13-15）：

$$
\text{sign}(x) = \frac{4}{\pi} \sum_{i=1,3,5,...}^{n} \frac{1}{i} \sin\left(\frac{ix}{H}\right), \quad \|x\| < H
$$

$$
\frac{d\,\text{sign}(x)}{dx} = \frac{4}{\pi} \sum_{i=1,3,5,...}^{n} \cos\left(\frac{ix}{H}\right)
$$

**优点**：因为是无穷级数在有限项截断下的**理论等价变换**，前后向**方向一致**，优化更稳定。

---

## 🎯 Top-N Hamming Search（公式 17、18）

**关键定理**：内积可等价转化为汉明距离计算：

$$
Q_x^\top Q_y = \sum_{i | (Q_x)_i = (Q_y)_i} 1 - \sum_{i | (Q_x)_i \neq (Q_y)_i} 1 = d - 2 D_H(Q_x, Q_y)
$$

因此评分函数（公式 17）：

$$
\hat{Y}_{x,y} = (\alpha_x Q_x)^\top (\alpha_y Q_y) = \alpha_x \alpha_y (d - 2 D_H(Q_x, Q_y))
$$

> ⚡ **效率**：位运算替代浮点乘加 → 实测 **8× 加速**。

---

## 📊 最终损失函数

$$
\mathcal{L} = \mathcal{L}_{BPR} + \lambda_1 \mathcal{L}_{CL} + \lambda_2 \|\Theta\|_2^2
$$

| 损失项 | 作用 |
|--------|------|
| $\mathcal{L}_{BPR}$ | Bayesian Personalized Ranking：让观测边的得分高于未观测边 |
| $\mathcal{L}_{CL} = \mathcal{L}_1^{cl} + \mathcal{L}_2^{cl}$ | 双视图对比学习：让同一节点的两个增强视图相似，与其他节点远离 |
| $\lambda_2\|\Theta\|_2^2$ | L2 正则防过拟合 |

---

## 🔬 复杂度分析（Table II / III）

| 指标 | 复杂度 |
|------|--------|
| 训练时间 | $O(snd\|E\|)$，对边数 $\|E\|$ **二次**（GCN 标配） |
| Hash 码空间 | $O((\|V_1\|+\|V_2\|)(d+32(L+1)))$ bits |
| 推理加速 | **8×** 相比 float32 GCN |

---

## 📌 实验结果（从摘要 / Fig. 1）

> 在 **6 个真实数据集**上验证，密度从 0.0419 到 0.00062（极稀疏）。

**关键结论**：
- BGCH+ **超越 BGCH 前作**
- 在保持 **8× 推理加速**的同时，**达到接近 float32 GCN** 的精度
- 图 1 显示 hash 码的**分布均匀性**显著提升（uniformity）

---

## 🧠 我的理解（与同类工作对比）

### 与传统 Learning to Hash 的区别

| 维度 | 传统 Hash (LSH / DSH) | GCN-based Hash (BGCH/BGCH+) |
|------|----------------------|----------------------------|
| 是否利用图结构 | ❌ | ✅ |
| 二值化时机 | 后处理（post-hoc） | **集成到 GCN 训练中** |
| 损失景观 | 平滑但语义丢失 | 震荡但语义完整 |
| 推荐系统适配 | 弱 | 强 |

### 与对比学习图神经网络（GRACE / GCA）的区别

| 维度 | 传统图 CL | BGCH+ |
|------|----------|-------|
| 增强对象 | 图结构（节点/边 dropout） | **特征**（连续嵌入 + hash 码） |
| 对离散 hash 的适用性 | 差（破坏离散性） | ✅ 专门设计了扰动 scalar 的方式 |
| 目标 | 通用节点分类 | **专攻 hash 检索** |

---

## 💡 可借鉴的工程技巧

1. **多层哈希码拼接**（公式 6）：把 GCN 各层 hash 码拼起来，相当于**深度监督 + 集成**
2. **Rescaling 因子** $\alpha = 1/(d\|V\|_1)$：比 learnable scale 更稳定
3. **对 $\alpha$ 加噪声再生成 hash 码扰动**：避开对离散码直接加噪的难题
4. **傅里叶级数近似 sign**：比 tanh 更理论严谨，前后向方向一致

---

## ❓ 待解决的问题 / 思考

1. 傅里叶级数展开的截断项 $n$ 多大合适？原文应有 ablation（Section V-D）
2. $\rho$（噪声超球半径）和 $\rho$（$\mathcal{L}_2^{cl}$ 中的温度参数）名字相同，是否冲突？
3. 是否可以扩展到**异构图**或**多模态推荐**？
4. 与近期大模型 embedding 检索（如 LLM-based dense retrieval）的对比？

---

## 📎 相关引用

- BGCH（前作）：Chen et al., 2023
- GCN 基线：Kipf & Welling, 2017（本文采用）
- InfoNCE / SimCLR 系列对比学习
- BPR 损失：Rendle et al., 2009

---

> 📌 **阅读建议**：先看 Fig. 2 框架图，再看公式 (6)(10)(12)(18) —— 这四个公式串联了**多层哈希 → 对比学习 → 检索**的完整链路。