# ObsidianNoteBook 📓

> 个人 AI / 深度学习 / 计算机视觉 知识库，基于 Obsidian 双链笔记构建。

[![Obsidian](https://img.shields.io/badge/Obsidian-7C3AED?logo=obsidian&logoColor=white)](https://obsidian.md)
[![GitHub last commit](https://img.shields.io/github/last-commit/Fei-Fan-Child/ObsidianNoteBook)](https://github.com/Fei-Fan-Child/ObsidianNoteBook)

---

## 📖 简介

这是我在深度学习、计算机视觉和强化学习方向的学习笔记仓库。所有笔记使用 Obsidian 的 **Wiki 链接（`[[note]]`）** 互相关联，形成知识网络。同时在 GitHub 上托管作为云备份 + 公开分享。
---

## 🖼️ 笔记预览

### ✍️ 纯手绘 Transformer 架构

| 完整架构 | Encoder 细节 | Decoder 细节 |
|:---:|:---:|:---:|
| ![Transformer](Transformer/Hand-Drawn-Transformer/Transformer/Transformer-Transformer.png) | ![Encoder](Transformer/Hand-Drawn-Transformer/Transformer/Transformer-Encoder.png) | ![Decoder](Transformer/Hand-Drawn-Transformer/Transformer/Transformer-Decoder.png) |

> 纯手绘 Transformer 架构图，涵盖 Multi-Head Attention、Add & Norm、Feed Forward、Cross-Attention 等全部组件。

### 📐 Self-Attention 矩阵计算

![Self-Attention](Transformer/assets/%E8%BE%85%E5%8A%A9Transformer%E7%9A%84%E5%A4%8D%E7%8E%B0%E6%89%8B%E7%BB%98%E6%9E%B6%E6%9E%84/Pasted%20Image%2020260712084004_715.png)

> QKV 线性变换 → 缩放点积 → Softmax → 加权求和全过程手写推导。

### 🧠 ViT 架构分析

![ViT](ViTSeries/ViT/assets/Algorithm/Pasted%20Image%2020260708155435_119.png)

> Vision Transformer 的 Patch Embedding + Position Encoding + Transformer Encoder 完整流程。

---

## 🗂️ 知识地图

### 🔬 计算机视觉

| 笔记 | 简介 |
|------|------|
| [YOLOv1](YOLO/YOLOv1%20笔记.md) | YOLOv1 目标检测论文精读 |
| [YOLOv2](YOLO/YOLOV2/YOLOV2.md) | YOLOv2 改进点详解 |
| [YOLOv3](YOLO/YOLOv3%20笔记.md) | YOLOv3 多尺度检测 |
| [ViT 算法分析](ViTSeries/ViT/Algorithm.md) | Vision Transformer 架构详解 |
| [EViT](ViTSeries/EViT/Note.md) | Efficient Vision Transformer |
| [ViT Token 计算](Com-GPU/ViT中的Token计算方式.md) | ViT 中 Token 的计算与显存分析 |
| [推理加速](Com-GPU/Computer%20Vision%20Accelerate%20Inference.md) | CV 模型推理加速方法 |
| [模型剪枝](Com-GPU/模型剪枝核心原理.md) | 模型剪枝算法与流程 |

### 🤖 Transformer & 深度学习

| 笔记 | 简介 |
|------|------|
| [Transformer 原理详解](Transformer/Transformer%20Principle.md) | 含 Self-Attention / Multi-Head / Add & Norm / 位置编码完整推导 |
| [Transformer 手绘架构](Transformer/Hand-Drawn-Transformer/README.md) | 纯手绘 Transformer 架构图 |
| [Transformer 原理 (GPU版)](Com-GPU/Transformer%20原理详解.md) | 显存角度的 Transformer 分析 |
| [显卡/显存知识](Com-GPU/显卡（CPU）的知识.md) | GPU 硬件基础 |
| [显存使用计算](Com-GPU/显存使用计算.md) | 训练时显存占用量估算 |
| [PyTorch 笔记](Pytorch/note.md) | PyTorch 使用技巧与速查 |

### 🎮 强化学习（RL）

| 笔记 | 简介 |
|------|------|
| [RL 基础概念](RL/Concept/Reinforcement%20Learning%20Basic%20Concept.md) | MDP、贝尔曼方程等核心概念 |
| [DQN](RL/DQN/Deep%20Q%20Learning.md) | Deep Q-Network 算法详解 |
| [RL 课程 Slides](RL/PDF/) | 完整课程 PDF 讲义（24 篇） |

### ☁️ 工程 & 工具

| 笔记 | 简介 |
|------|------|
| [SSH 概念](SSH/SSH的概念.md) | SSH 原理与配置 |
| [AutoDL 云训练](SSH/如何在云服务器上训练模型-AutoDL为例.md) | 云服务器上训练模型的完整流程 |
| [Agent & MCP](Agent/AgentFunctionCallingMCP.md) | AI Agent 函数调用与 MCP 协议 |

### 📚 参考资料

| 资源 | 说明 |
|------|------|
| [一本不太简介的 LaTeX 介绍](ReferenceBooks/一本不太简介的LaTeX介绍.pdf) | LaTeX 入门 PDF |
| [动手学深度学习](Somebooks/动手学深度学习.pdf) | 李沐《动手学深度学习》 |
| [CV 技术指南](Somebooks/CV技术指南.pdf) | 计算机视觉技术综述 |
| [深度学习的数学](Somebooks/深度学习的数学.pdf) | 深度学习数学基础 |

---

## 🛠️ 使用说明

### 一、克隆到本地并打开

```bash
# 1. 克隆仓库
git clone https://github.com/Fei-Fan-Child/ObsidianNoteBook.git

# 2. 进入目录
cd ObsidianNoteBook
```

打开 Obsidian → 点击左下角 **"打开其他仓库"** → 选择 `ObsidianNoteBook` 文件夹 → 所有双链 `[[note]]` 和图片自动解析。

> ⚠️ 插件配置（Git、PDF++、Excalidraw 等）已内置在 `.obsidian/` 中，打开后**需要在设置里手动启用**这些第三方插件，否则 PDF 标注、自动备份等功能不会生效。

---

### 二、仓库目录结构

```
ObsidianNoteBook/
├── README.md                  ← 你正在看的文件
├── Transformer/               ← Transformer 原理、手绘架构、复现笔记
├── YOLO/                      ← YOLOv1 / v2 / v3 论文笔记
├── ViTSeries/                 ← ViT / EViT 算法分析
├── Com-GPU/                   ← GPU 显存、推理加速、模型剪枝
├── RL/                        ← 强化学习：概念、DQN、课程 PDF
│   └── PDF/                   ← RL 课程幻灯片（可直接在 Obsidian 中标注）
├── SSH/                       ← SSH 原理 + 云服务器训练部署
├── Pytorch/                   ← PyTorch 学习笔记
├── Agent/                     ← AI Agent / MCP 协议
├── Plan/                      ← 学习计划与进度
├── Templates/                 ← 笔记模板（Books、Capture、Callouts）
├── Somebooks/                 ← 参考电子书 PDF
└── .obsidian/                 ← Obsidian 配置（插件、主题、快捷键）
```

---

### 三、在 Obsidian 中浏览笔记

笔记之间通过 **`[[双向链接]]`** 互相关联。打开任意一篇笔记后：

| 操作 | 方法 |
|------|------|
| 点击 `[[链接]]` 跳转 | 直接点击笔记中的蓝色链接 |
| 查看反向链接 | 右侧边栏 → "反链" 面板 |
| 关系图谱 | 左侧 Ribbon → "图谱" 图标（可视化知识网络） |
| 全局搜索 | `Ctrl+Shift+F` 搜索所有笔记内容 |
| 快速跳转 | `Ctrl+O` 输入文件名 |

> 💡 **推荐入口**：从 [Transformer 原理详解](Transformer/Transformer%20Principle.md) 或 [RL 基础概念](RL/Concept/Reinforcement%20Learning%20Basic%20Concept.md) 开始，沿双链层层深入。

---

### 四、Git 自动备份（核心功能）

本仓库通过 **Obsidian Git** 插件实现自动备份，配置规则：

| 行为 | 说明 |
|------|------|
| 🔄 **自动 Pull** | 每次打开 Obsidian 时自动拉取远程更新 |
| 💾 **自动 Commit** | 每 1 分钟检查文件变化并自动提交 |
| 📤 **手动 Push** | `Ctrl+P` → 搜索 `Obsidian Git: Push` |

> ⚠️ 如果你也修改了笔记并想同步到自己的仓库，需要先 Fork 本仓库，然后将远程地址改为你自己的。

---

### 五、PDF 标注

本仓库已配置 **PDF++** 插件，打开任意 PDF 后：

1. 选中文字 → 颜色面板弹出
2. 左键点颜色 = **高亮**，右键点颜色 = 切换标注类型（高亮 / 下划线 / 波浪线 / 删除线）
3. 高亮后右键 → Copy → 选择格式 → 粘贴到笔记中（自动生成双向链接）

详见 [Transformer Principle](Transformer/Transformer%20Principle.md) 中 PDF++ 的矩阵计算示例。

---

### 六、纯 GitHub 端阅读（不用 Obsidian）

如果只是想在网页上看笔记，也可以：直接在 GitHub 上点击 `.md` 文件即可渲染预览。但以下功能仅 Obsidian 中可用：

| 功能 | GitHub | Obsidian |
|------|--------|----------|
| 基础 Markdown 渲染 | ✅ | ✅ |
| `[[双链]]` 跳转 | ❌ | ✅ |
| PDF 标注 | ❌ | ✅ |
| Excalidraw 手绘图 | ❌ | ✅ |
| Dataview 动态查询 | ❌ | ✅ |
| 关系图谱 | ❌ | ✅ |

---

### 七、核心插件列表

| 插件 | 用途 | 是否需要手动启用 |
|------|------|:--:|
| **Git** | 自动备份到 GitHub | ✅ |
| **PDF++** | PDF 高亮与批注 | ✅ |
| **Excalidraw** | 手绘架构图 / 流程图 | ✅ |
| **Dataview** | 动态查询笔记（如按标签聚合） | ✅ |
| **Templater** | 高级模板系统 | ✅ |
| **Day Planner** | 时间线任务管理 | ✅ |
| **Kanban** | 看板视图 | ✅ |
| **Hover Editor** | 悬浮预览编辑 | ✅ |
| **Style Settings** | 主题细节定制 | ✅ |

---

### 八、自定义适配（Fork 后）

如果你 Fork 了本仓库用于自己的笔记：

1. **删掉不需要的文件夹**，保留 `.obsidian/` 配置和 `Templates/` 模板
2. 在 `Plan/Plan.md` 中更新你的学习计划
3. 修改 `README.md`，替换为自己的知识地图
4. 将 Git remote 改为你自己的仓库地址：

```bash
git remote set-url origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

---

## 📬 联系

- **GitHub**: [@Fei-Fan-Child](https://github.com/Fei-Fan-Child)

---

> 💡 本仓库持续更新中。发现问题或想交流，欢迎提 Issue / PR。
