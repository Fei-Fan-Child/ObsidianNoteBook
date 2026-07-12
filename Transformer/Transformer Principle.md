---
tags:
  - transformer
  - deep-learning
  - attention
  - nlp
created: 2026-06-26
source: "https://mathor.blog.csdn.net/article/details/107352167"
---

## 概述

Transformer 是 Google 在 2017 年论文 *Attention Is All You Need* 中提出的序列到序列（Seq2Seq）模型。

**核心思想**：完全基于**自注意力机制（Self-Attention）**，摒弃 RNN 的循环结构。与 LSTM 逐字串行处理不同，Transformer **所有字同时训练**，大幅提升并行效率。它通过位置编码理解语序，用自注意力机制和全连接层计算。

> 本文聚焦 **Encoder（编码器）部分，这也是 BERT 等模型所使用的核心结构。

### 整体架构

![[assets/Transformer Principle/file-20260626090428465.png]]

上图是 Transformer 的宏观结构，左侧为 Encoder，右侧为 Decoder：

- **Encoder（编码器）**：将输入序列编码为富含上下文的表示。由 $N=6$ 个相同的 Encoder Layer 堆叠而成。
- **Decoder（解码器）**：根据 Encoder 输出 + 已生成序列，自回归地生成目标。同样堆叠 $N=6$ 层。

---

## 1. Input Processing（输入处理）

### 1.1 Tokenization（分词）

Transformer 以**字/子词（subword）**为单位进行编码，而非以词为单位：

- 词编码：`I don't like this` → 3 个 token
- 字编码：`I do n't like this` → 5 个 token（更细粒度，减少 OOV 问题）

![[assets/Transformer Principle/file-20260626100751094.png]]

> 上图展示了词编码与字编码的对比。实际工程中常用 BPE、WordPiece 或 SentencePiece 等子词分词算法。

### 1.2 Input Embedding

![[assets/Transformer Principle/file-20260626091904500.png]]

输入 token 通过 Embedding 层映射为稠密向量（上图展示了 Embedding 矩阵的查找过程）：

$$
X = \text{EmbeddingLookup}(\text{token\_ids})
$$

- Input：$X$（token id 序列）
- Input Embedding：$X_{\text{embedding}} \in \mathbb{R}^{\text{seq\_len} \times d_{\text{model}}}$

> 论文中 $d_{\text{model}} = 512$，即每个字被表示为一个 512 维向量。

### 1.3 Positional Encoding（位置编码）

![[assets/Transformer Principle/file-20260626102036170.png]]

Transformer **没有任何循环或卷积结构**，因此需要显式注入位置信息，否则模型无法感知 token 的顺序。

#### 关键特点

| 特性 | Transformer（原始论文） | BERT / GPT 等变体 |
|------|------------------------|-------------------|
| 训练方式 | **固定（不训练）**，使用数学公式 | **可训练**，作为可学习参数 |
| 维度 | 与 $X_{\text{embedding}}$ 一致 | 同上 |
| 组合方式 | 直接相加：$X_{\text{embedding}} + PE$ | 同上 |

> 位置编码维度必须与 Embedding 维度一致，**因为两者需要直接相加**。

#### 公式推导

位置编码使用正弦和余弦函数，为每个位置生成唯一的编码模式：

$$
\begin{aligned}
PE_{(pos,\;2i)} &= \sin\left(\frac{pos}{10000^{\,2i / d_{\text{model}}}}\right) \\[8pt]
PE_{(pos,\;2i+1)} &= \cos\left(\frac{pos}{10000^{\,2i / d_{\text{model}}}}\right)
\end{aligned}
$$

参数含义：

| 符号 | 含义 | 范围 |
|------|------|------|
| $pos$ | token 在句子中的位置 | $[0,\; \text{max\_sequence\_length})$ |
| $i$ | 词向量维度的序号 | $[0,\; d_{\text{model}} / 2)$ |
| $d_{\text{model}}$ | Embedding 的总维度 | 论文中为 512 |

- **偶数维度**（$2i$）使用 $\sin$，**奇数维度**（$2i+1$）使用 $\cos$
- 每个 $i$ 生成一对（sin + cos）两个维度的值，所以 $i$ 只需迭代 $d_{\text{model}}/2$ 次

![[assets/Transformer Principle/file-20260626103651977.png]]

> 上图展示了位置编码的计算原理：对于句子中每个位置 $pos$，在 embedding 的每个维度对上分别计算 sin 和 cos 值。

![[assets/Transformer Principle/file-20260626103751408.png]]

> 上图为位置编码的矩阵视角：行代表不同位置（$pos=0,1,2,\dots$），列代表不同维度（$2i$ 为偶数列用 sin，$2i+1$ 为奇数列用 cos），满足 $0 < 2i < d_{\text{model}}$ 且 $0 < 2i+1 < d_{\text{model}}$。

![[assets/Transformer Principle/file-20260626103944460.png]]

> 上图进一步展示了奇数维度和偶数维度的编码值如何在位置维度上展开——每个 (pos, i) 组合对应唯一的编码值。

![[assets/Transformer Principle/file-20260626104118357.png]]

> 上图为完整的位置编码计算流程：从 pos 和 i 出发，经由 sin/cos 函数生成最终的 PE 矩阵，然后与 Input Embedding 逐元素相加，得到 Encoder 的真正输入。

#### 可视化理解

![[assets/Transformer Principle/file-20260626104310880.png]]

上图是位置编码的热力图，横轴为 embedding 维度（共 16 维），纵轴为序列位置：

- **低维度（$i$ 较小，图左半部分）**：颜色变化剧烈，周期短 → 捕获**细粒度**位置差异
- **高维度（$i$ 较大，图右半部分）**：颜色变化平缓，周期长 → 捕获**粗粒度**位置差异
- 多频率叠加使得每个位置都有**独一无二的编码"纹理"**

> 💡 选择正弦函数的直觉：$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数，使模型更容易学习**相对位置**关系。

---

## 2. Self-Attention（自注意力机制）

自注意力是 Transformer 最核心的组件，让序列中每个 token 都能**直接关注**所有其他 token，一步到位捕获全局依赖。

### 2.1 Q、K、V 的由来

![[assets/Transformer Principle/file-20260626110000864.png]]

如上图所示，每个输入 $x$ 通过乘以三个不同的可学习权重矩阵，转换为三个角色向量：

$$
\boxed{q = x \cdot W^Q} \qquad \boxed{k = x \cdot W^K} \qquad \boxed{v = x \cdot W^V}
$$

| 向量 | 全称 | 直觉含义 | 矩阵运算 |
|------|------|----------|----------|
| **Q**（Query） | 查询向量 | "我在找什么？" | $Q = X \cdot W^Q$ |
| **K**（Key） | 键向量 | "我是什么？" | $K = X \cdot W^K$ |
| **V**（Value） | 值向量 | "我包含什么信息？" | $V = X \cdot W^V$ |

> $W^Q, W^K, W^V$ 都是可学习的参数矩阵，$X$ 乘以它们相当于进行一次线性变换。

### 2.2 Scaled Dot-Product Attention（缩放点积注意力）

![[assets/Transformer Principle/file-20260626110538701.png]]

上图是 Self-Attention 的完整计算流程。核心公式为：

$$
\boxed{\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V}
$$

#### 计算步骤（以 $x_1$ 为例，共 3 个输入）

**Step 1 — 计算相似度分数**：

![[assets/Transformer Principle/file-20260626110938929.png]]

如上图，$x_1$ 的 Query $q_1$ 分别与每个 token 的 Key（$k_1, k_2, k_3$）做点积，得到 $x_1$ 对所有 token 的原始注意力分数。

写出矩阵形式，Score 矩阵的第 $i$ 行第 $j$ 列即为 $q_i$ 与 $k_j$ 的点积结果：

$$
\text{Scores} = QK^T = \begin{bmatrix}
q_1 \cdot k_1 & q_1 \cdot k_2 & q_1 \cdot k_3 \\
q_2 \cdot k_1 & q_2 \cdot k_2 & q_2 \cdot k_3 \\
q_3 \cdot k_1 & q_3 \cdot k_2 & q_3 \cdot k_3
\end{bmatrix} \in \mathbb{R}^{3 \times 3}
$$
![[assets/Transformer Principle/file-20260626112339634.gif]]
为了获得第一个字的注意力权重，我们需要用第一个字的查询向量 $q_1$ 乘以**键矩阵的转置 $K^T$**：

$$
q_1 \cdot K^T = \begin{bmatrix} 1 & 0 & 2 \end{bmatrix} \begin{bmatrix} 0 & 4 & 2 \\ 1 & 4 & 3 \\ 1 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 4 & 4 \end{bmatrix}
$$


**Step 2 — 缩放（Scaling）**：

将分数除以 $\sqrt{d_k}$：

$$
\text{ScaledScores} = \frac{QK^T}{\sqrt{d_k}}
$$

> ⚠️ **为什么要除以 $\sqrt{d_k}$？**
>
> 当 $d_k$ 较大时，点积结果的方差会随之增大。方差过大会导致 Softmax 后的分布趋近于 one-hot（梯度趋近于 0），模型难以训练。除以 $\sqrt{d_k}$ 将方差控制为 1，保持梯度稳定。

**Step 3 — Softmax 归一化**：
![[assets/Transformer Principle/file-20260626112013759.gif]]

对缩放后的每一行应用 Softmax，得到归一化的注意力权重矩阵：
$$
\text{softmax}([2, 4, 4]) = [0.0, 0.5, 0.5]
$$


其中每个元素的计算方式为（以第一行为例）：

$$
\alpha_{1j} = \frac{\exp\left(\frac{q_1 \cdot k_j}{\sqrt{d_k}}\right)}{\sum_{m=1}^{3} \exp\left(\frac{q_1 \cdot k_m}{\sqrt{d_k}}\right)}, \quad \sum_{j=1}^{3}\alpha_{1j} = 1
$$

> 每一行的权重之和为 1，表示第 $i$ 个 token 对所有 token 的注意力分布。

**Step 4 — 加权求和得到 Context Vector**：

如上图，用注意力权重矩阵 $A$ 对 Value 矩阵 $V$ 做加权求和：

$$
\text{Context} = A \cdot V
$$

对于 $x_1$，其上下文向量为：

$$
z_1 = \alpha_{11} \cdot v_1 + \alpha_{12} \cdot v_2 + \alpha_{13} \cdot v_3
$$

![[assets/Transformer Principle/file-20260626112749880.gif]]
![[assets/Transformer Principle/file-20260626112720297.png]]

![[assets/Transformer Principle/file-20260626112837870.gif]]
![[assets/Transformer Principle/file-20260626112828688.png]]

> 上面的动图和图片展示了：注意力权重 $\alpha_{1j}$ 分别乘以对应的 $v_j$，再全部相加，得到 $x_1$ 的上下文向量 $z_1$。

**最终结果**——所有 token 都完成了相同的操作：

![[assets/Transformer Principle/file-20260626112952395.gif]]

> 如上动图，$x_1, x_2, x_3$ 各自通过 Self-Attention 得到融合了全局上下文的新表示 $z_1, z_2, z_3$。

![[assets/Transformer Principle/file-20260626111130471.png]]
![[assets/Transformer Principle/file-20260626111233952.png]]
![[assets/Transformer Principle/file-20260626111443663.png]]
自注意力层**仅包含三个可学习参数矩阵**：$W^Q, W^K, W^V$，没有其他参数（不考虑偏置项时）。这也是它极其简洁高效的原因。

> 以上三张图分别展示了从单个 $x_1$ 到所有 $x$ 的上下文向量计算过程：每个 token 的 Query 与所有 Key 交互得到注意力权重，再与所有 Value 加权求和，最终每个 token 都得到融合了全局信息的 Context Vector。

---
## 3. Attention Mask（注意力掩码）

> ⚠️ **重要补充概念（来自 mathor 原文）**

实际训练使用 mini-batch 时，一个 batch 内句子长度不一致，短的句子需要用 `<pad>` 填充（padding）。但 Softmax 中 $e^0 = 1$，padding 位置会被赋予非零权重，干扰计算。

### 解决方案

给 padding 位置的分数加上一个**极大的负数偏置**，使 Softmax 后的权重趋近于 0：

$$
\text{Scores}_{\text{masked}} = \text{Scores} + \text{Mask}
$$

其中 Mask 矩阵在 padding 位置为 $-\infty$（实际代码中用 $-10^9$），其余位置为 $0$：

```python
scores = scores.masked_fill(mask, -1e9)  # mask 为 True 的位置填 -1e9
```

Softmax 之后再乘以 $V$ 时，padding 位置的贡献几乎为 0：

$$
\text{softmax}(-\infty) \approx 0
$$

### 两种 Mask 类型

| 类型 | 使用位置 | 作用 |
|------|---------|------|
| **Padding Mask** | Encoder 和 Decoder | 屏蔽 `<pad>` 填充位置 |
| **Sequence Mask**（Look-ahead Mask） | Decoder | 屏蔽未来 token，保证自回归生成 |

---

## 4. Multi-Head Attention（多头注意力）

> ⚠️ **补充概念**

单一注意力头只能学到一种关系模式。多头注意力并行运行多个独立的注意力头，每个头在不同的表示子空间中学习：

$$
\begin{aligned}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h) \cdot W^O \\[6pt]
\text{head}_i &= \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)
\end{aligned}
$$

- 每个头有独立的 $W^Q_i, W^K_i, W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$，其中 $d_k = d_{\text{model}} / h$
- 所有头的结果拼接后，通过 $W^O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$ 投影回原维度

### 直觉理解

| 头 | 可能捕获的关系 |
|----|--------------|
| Head 1 | 语法依赖（主谓关系、动宾关系） |
| Head 2 | 指代关系（代词 → 先行词） |
| Head 3 | 语义相似性（近义词、同类概念） |
| ... | 更复杂的隐式模式 |

> 论文中使用 $h = 8$ 个头，每个头 $d_k = d_v = d_{\text{model}} / h = 64$。

---

## 5. Add & Norm（残差连接 + 层归一化）

> ⚠️ **核心概念（mathor 原文有详细推导）**

每个子层（Self-Attention / FFN）之后都执行：

$$
\boxed{\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))}
$$

### 5.1 Add — 残差连接（Residual Connection）

$$
X_{\text{attention}} = X_{\text{embedding}} + \text{Attention}(Q, K, V)
$$

- **做法**：子层的输入 $x$ 与子层的输出直接相加
- **作用**：梯度可以通过残差路径直传回初始层，有效缓解深层网络的梯度消失问题，使模型可以堆叠更多层

> 每经过一个子层，运算前后的值相加：$X + \text{SubLayer}(X)$

### 5.2 Norm — 层归一化（Layer Normalization）

将隐藏层归一化为标准正态分布，加速训练收敛。**按行**计算均值和方差：

$$
\mu_j = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} x_{ij}, \qquad
\sigma^2_j = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} (x_{ij} - \mu_j)^2
$$

归一化公式：

$$
\boxed{\text{LayerNorm}(x) = \alpha \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta}
$$

| 符号         | 含义                    |
| ---------- | --------------------- |
| $\mu$      | 每一行的均值                |
| $\sigma^2$ | 每一行的方差                |
| $\epsilon$ | 防止除零的小常数（如 $10^{-8}$） |
| $\alpha$   | 可训练的缩放参数，初始化为全 1      |
| $\beta$    | 可训练的平移参数，初始化为全 0      |
| $\odot$    | 逐元素相乘                 |

### 5.3 Post-LN vs Pre-LN

| 方式          | 计算顺序                                                   | 使用场景                   |
| ----------- | ------------------------------------------------------ | ---------------------- |
| **Post-LN** | $x \to \text{Sublayer} \to \text{Add} \to \text{Norm}$ | 原始论文                   |
| **Pre-LN**  | $x \to \text{Norm} \to \text{Sublayer} \to \text{Add}$ | GPT、LLaMA 等现代变体（训练更稳定） |

---

## 6. Feed Forward Network（前馈网络）


每个 Encoder / Decoder Layer 中还包含一个**位置独立**的两层全连接网络：

$$
\boxed{\text{FFN}(x) = \text{Activate}(xW_1 + b_1)W_2 + b_2}
$$

> 原文使用 ReLU：$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$。现代变体常用 GELU、Swish 等激活函数。

- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$，$W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$
- 中间维度 $d_{\text{ff}} = 2048$（通常为 $4 \times d_{\text{model}}$）
- **对每个位置独立应用相同参数** → 可并行计算所有位置

---

## 7. Encoder 完整结构

> ⚠️ **核心总结（mathor 原文的完整 Encoder 流程）**

### 单个 Encoder Layer

```
Input → [Multi-Head Self-Attention] → Add & Norm → [FFN] → Add & Norm → Output
```

### 完整计算步骤

**Step 1 — 字向量 + 位置编码**：
$$
X = \text{EmbeddingLookup}(X) + \text{PositionalEncoding}(X)
$$

**Step 2 — 自注意力**：
$$
\begin{aligned}
Q &= X W^Q, \quad K = X W^K, \quad V = X W^V \\[4pt]
X_{\text{attention}} &= \text{SelfAttention}(Q, K, V)
\end{aligned}
$$

**Step 3 — 残差连接 + LayerNorm**：
$$
\begin{aligned}
X_{\text{attention}} &= X + X_{\text{attention}} \\[4pt]
X_{\text{attention}} &= \text{LayerNorm}(X_{\text{attention}})
\end{aligned}
$$

**Step 4 — FeedForward**：
$$
X_{\text{hidden}} = \text{Activate}(\text{Linear}(\text{Linear}(X_{\text{attention}})))
$$

即两层线性映射 + 激活函数。

**Step 5 — 再次残差 + LayerNorm**：
$$
\begin{aligned}
X_{\text{hidden}} &= X_{\text{attention}} + X_{\text{hidden}} \\[4pt]
X_{\text{hidden}} &= \text{LayerNorm}(X_{\text{hidden}})
\end{aligned}
$$

**最终输出维度**：
$$
X_{\text{hidden}} \in \mathbb{R}^{\text{batch\_size} \;\times\; \text{seq\_len} \;\times\; d_{\text{model}}}
$$

> 堆叠 $N=6$ 个相同的 Encoder Layer，上一层的输出作为下一层的输入。

### 矩阵计算

下面以具体的矩阵维度为例，演示 Self-Attention 中的矩阵运算流程。

![[assets/Transformer Principle/file-20260626113749066.png]]

**Step 1 — 线性变换得到 Q、K、V**：

输入矩阵 $X$ 通过三个权重矩阵分别投影：

$$
Q = X \cdot W^Q, \quad K = X \cdot W^K, \quad V = X \cdot W^V
$$

| 矩阵 | 维度 | 说明 |
|------|------|------|
| $X$ | $2 \times 4$ | 2 个 token，每个 4 维 |
| $W^Q, W^K, W^V$ | $4 \times 3$ | 将 4 维投影到 3 维 |
| $Q, K, V$ | $2 \times 3$ | 投影后的结果 |

**Step 2 — 计算注意力分数**：

将 $Q$ 与 $K^T$ 相乘，再除以 $\sqrt{d_k}$ 进行缩放：

$$
\text{ScaledScores} = \frac{QK^T}{\sqrt{d_k}}
$$

![[assets/Transformer Principle/file-20260626113753762.png]]

**Step 3 — Softmax 归一化 + 加权求和**：

$$
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right), \quad Z = A \cdot V
$$

---

### 3.1 Multi-Head Attention 的矩阵视角

前面的 Self-Attention 只用了一组 $Q, K, V$，让每个 token 关注上下文。Multi-Head Attention 则使用**多组**独立的 $Q, K, V$，每组在不同的表示子空间中学习不同的关系模式：

![[assets/Transformer Principle/file-20260626113801760.png]]

线性变换的权重矩阵从一组 $(W^Q, W^K, W^V)$ 变为多组：

$$
(W^Q_0, W^K_0, W^V_0), \quad (W^Q_1, W^K_1, W^V_1), \quad \dots
$$

每个头（head）独立计算 Self-Attention，各自得到一个输出矩阵 $Z_i$：

![[assets/Transformer Principle/file-20260626113806274.png]]

---

### 3.2 完整流程总览

以下图片展示了从输入到输出的完整矩阵流动：

![[assets/Transformer Principle/file-20260626113809803.png]]
![[assets/Transformer Principle/file-20260626113815471.png]]
### 3.3 残差连接与 Layer Normalization

经过 Self-Attention 后，将输出与输入做**残差连接（Add）**，再通过 **Layer Normalization** 归一化：

$$
X_{\text{out}} = \text{LayerNorm}\big(X_{\text{embedding}} + \text{Attention}(Q, K, V)\big)
$$

**残差连接**：将 Self-Attention 的输入 $X_{\text{embedding}}$ 与其输出直接相加，使梯度能直传回初始层，缓解深层网络的梯度消失。

**Layer Normalization**：将每个样本的隐藏层归一化为标准正态分布，加速训练收敛：

$$
\mu_j = \frac{1}{m} \sum_{i=1}^{m} x_{ij}, \qquad
\sigma^2_j = \frac{1}{m} \sum_{i=1}^{m} (x_{ij} - \mu_j)^2
$$

$$
\text{LayerNorm}(x) = \frac{x_{ij} - \mu_j}{\sqrt{\sigma^2_j + \epsilon}}
$$

> 公式中以矩阵的**列（column）**为单位求均值和方差。$\epsilon$ 防止分母为 0。

![[assets/Transformer Principle/file-20260626113818821.png]]


![[assets/Transformer Principle/file-20260626113818821.png]]​
 下图展示了更多细节：输入 $x_1, x_2$ 经 Self-Attention 后变为 $z_1, z_2$，与输入做残差连接，经 LayerNorm 后送入全连接层。FFN 同样有残差 + LayerNorm，最终输出进入下一个 Encoder Layer。进行残差连接，经过LayerNorm后输出给全连接层。全连接层也有一个残差连接和一个LayerNorm，最后再输出给下一个Encoder（每个Encoder Block中的FeedForward层权重都是共享的）

![[assets/Transformer Principle/file-20260626113828253.png|681]]
![[assets/Transformer Principle/file-20260626113844780.png]]
![[assets/Transformer Principle/file-20260626113848757.png]]
![[assets/Transformer Principle/file-20260626113853867.png]]

> 以上图片完整展示了 Encoder 内部数据从 Input Embedding → Positional Encoding → QKV 线性变换 → Self-Attention → Add & Norm → FFN → 再 Add & Norm 的逐层流动过程。

---

## 8. Decoder 结构（简要）

> ⚠️ **补充概念（本文主要聚焦 Encoder，Decoder 仅作简要介绍）**

每个 Decoder Layer 包含**三个**子层：

```
Input → [Masked Multi-Head Self-Attention] → Add & Norm
      → [Cross-Attention (Encoder-Decoder)] → Add & Norm
      → [FFN] → Add & Norm → Output
```

### 8.1 Masked Self-Attention

- 与 Encoder 自注意力类似，但使用**下三角 Mask（Sequence Mask）**屏蔽未来位置
- 确保位置 $i$ 只能关注位置 $1, 2, \dots, i$，不能看到后面的 token
- 用于**自回归生成**（预测下一个 token 时不能偷看答案）

### 8.2 Cross-Attention（Encoder-Decoder Attention）

- **$Q$**：来自 Decoder 上一层的输出
- **$K, V$**：来自 **Encoder 的最终输出**
- 使 Decoder 在生成每个 token 时都能关注输入序列的任意位置

---

## 9. 输出层

Decoder 的最终输出通过线性映射 + Softmax 得到预测概率：

$$
\text{Output} = \text{Softmax}(\text{Linear}(X_{\text{hidden}}))
$$

- **Linear**：将 $d_{\text{model}}$ 映射到词表大小
- **Softmax**：得到每个 token 的生成概率分布

---

## 10. 为什么 Self-Attention 优于 RNN/CNN？

| 对比维度 | RNN（LSTM/GRU） | CNN | Self-Attention |
|----------|----------------|-----|----------------|
| 长距离依赖 | 差（梯度消失/爆炸） | 需多层堆叠扩大感受野 | ✅ 恒定 $O(1)$ 路径长度 |
| 并行计算 | 差（序列依赖，必须逐字） | ✅ 好 | ✅ 好 |
| 每层计算复杂度 | $O(n \cdot d^2)$ | $O(k \cdot n \cdot d^2)$ | $O(n^2 \cdot d)$ |
| 可解释性 | 差 | 一般 | ✅ 注意力权重可直接可视化 |

> 注意：$O(n^2 \cdot d)$ 在长序列时是瓶颈，后续工作（Longformer、FlashAttention 等）对此进行了优化。

---

## 关键公式速查

| 模块 | 公式 |
|------|------|
| Positional Encoding | $PE_{(pos,2i)} = \sin(pos/10000^{2i/d}),\; PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d})$ |
| Self-Attention | $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$ |
| Multi-Head | $\text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$ |
| LayerNorm | $\alpha \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ |
| FFN | $\max(0, xW_1+b_1)W_2 + b_2$ |
| Add & Norm | $\text{LayerNorm}(x + \text{Sublayer}(x))$ |
| Encoder 输入 | $X = \text{EmbeddingLookup}(X) + \text{PositionalEncoding}(X)$ |

---

## 相关笔记

- [[Attention Mechanism]]（注意力机制基础）
- [[BERT]] / [[GPT]] / [[LLaMA]]（Transformer 变体应用）
