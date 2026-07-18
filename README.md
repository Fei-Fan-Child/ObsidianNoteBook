# ObsidianNoteBook 📓

> 个人 AI / 深度学习 / 视觉 / 强化学习 学习笔记库，基于 Obsidian 构建。

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Obsidian](https://img.shields.io/badge/Obsidian-7C3AED?logo=obsidian&logoColor=white)](https://obsidian.md)

---

## 📖 这是什么？

这是我个人在深度学习、计算机视觉和强化学习方向整理的学习笔记。内容包括：

- **论文精读**（YOLO、Transformer、ViT 等）
- **手写推导**（Self-Attention、Positional Encoding 等）
- **概念整理**（RL、CV、GPU、SSH 等）
- **工具实操**（云服务器训练、LaTeX、Agent 工具链）

笔记之间通过 Obsidian 的 **`[[双向链接]]`** 互相关联，配合 **Excalidraw** 手绘图、**Dataview** 动态查询、**PDF++** 标注，形成一张可查询的知识网络。

---

## 📂 目录速览

| 目录 | 内容 |
|------|------|
| `Transformer/` | Transformer 原理、手绘图、位置编码、Self-Attention 矩阵计算 |
| `YOLO/` | YOLOv1 / v2 / v3 论文笔记 |
| `ViTSeries/` | ViT / EViT 算法分析、前缀和求方差技巧 |
| `Com-GPU/` | GPU 显存、推理加速、模型剪枝 |
| `RL/` | 强化学习基础 + DQN + 24 节课程 PDF |
| `SSH/` | SSH 原理 + AutoDL 云训练流程 |
| `Agent/` | AI Agent / MCP 协议 |
| `Pytorch/` | PyTorch 使用技巧 |
| `Somebooks/` `/` `ReferenceBooks/` | 参考电子书 PDF |

> 📌 **完整 8 节使用说明 + 截图预览**见 vault 内的原版说明。

---

## ⚠️ 免责声明

1. **笔记内容来自网络资料整理**：本仓库中的笔记（包括手绘图、公式推导、概念总结）大量来源于公开的技术博客、知乎、B 站视频、GitHub 开源教程、PaperWithCode 等渠道。**仅供个人学习使用，不构成原创学术成果**。
2. **不保证准确性**：作者在整理过程中可能存在理解偏差或记录错误，请**勿直接用于作业、论文或商业用途**。
3. **第三方资料版权**：部分截图为引用内容，版权归原作者所有。如有版权疑虑请联系删除。
4. **PDF 文献**：仓库内的 PDF（`Somebooks/`、`ReferenceBooks/`、`RL/PDF/`）来自公开下载资源，仅供学习交流，请于 24 小时内删除。
5. **本项目无任何明示或暗示的保证**，使用者需自行承担使用风险。

---

## 🚀 如何下载并使用

### 前置：安装 Obsidian

1. 前往 [obsidian.md](https://obsidian.md/) 下载适合你系统的版本（Windows / macOS / Linux / iOS / Android）
2. 安装并打开（无需注册账号，"打开本地仓库" 即可完全离线使用）

### 步骤 1：克隆本仓库

```bash
git clone https://github.com/Fei-Fan-Child/ObsidianNoteBook.git
```

或直接前往仓库主页点击 **Code → Download ZIP** 解压。

### 步骤 2：在 Obsidian 中打开

1. 启动 Obsidian
2. 点击左下角 **"打开其他仓库"**（Open folder as vault）
3. 选择刚克隆 / 解压的 `ObsidianNoteBook` 文件夹
4. Obsidian 会自动加载所有 Markdown 笔记和 `[[双链]]`

### 步骤 3（推荐）：安装核心插件

打开仓库后，推荐安装以下插件以获得完整体验：

| 插件 | 用途 |
|------|------|
| **Git** | 自动备份到 GitHub |
| **Excalidraw** | 渲染手绘笔记（必须） |
| **PDF++** | PDF 高亮与批注 |
| **Dataview** | 动态查询笔记 |
| **Templater** | 模板系统 |

> 这些插件在 `community-plugins.json` 中已列出，但首次打开时 Obsidian 出于安全考虑会要求手动启用。

### 步骤 4：开始浏览

- 按 **Ctrl+O** 快速跳转任意笔记
- 点击笔记中的 `[[链接]]` 在知识网络间穿梭
- 右侧栏查看 **反向链接** + 大纲
- 左下角打开 **关系图谱**（Graph view）查看整体知识结构

---

## 📬 联系

- GitHub: [@Fei-Fan-Child](https://github.com/Fei-Fan-Child)

---

> 📖 本仓库遵循 MIT 许可证发布，详见 [LICENSE](LICENSE)。
