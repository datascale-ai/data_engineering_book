# 《大模型数据工程：架构、算法及项目实战》

[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://datascale-ai.github.io/data_engineering_book/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**[English](README_en.md) | 中文 | [日本語](README_ja.md)**

> **版本说明**：中文版是当前 2026 Springer 出版主线，结构冻结为 14 篇、48 章、15 个实战项目与 7 个附录（A–G）。英文版和日文版仍在跟进翻译，站点中会保留翻译状态说明页。

## 简介

> *"Data is the new oil, but only if you know how to refine it."*

在大模型时代，**数据质量决定模型上限**。然而，市面上关于 LLM 数据工程的系统性资料极为稀缺——大多数团队仍在"摸着石头过河"。

本书正是为解决这一痛点而生。我们系统性地梳理了从**预训练数据清洗**到**多模态对齐**、从 **RAG 检索增强**到**合成数据生成**，再到 **DataOps 平台建设**与**隐私合规治理**的完整技术体系，涵盖：

- 🧹 **预训练数据工程**：如何从 Common Crawl 等海量噪声数据中提炼出高质量语料
- 🖼️ **多模态数据处理**：图文对、视频、音频数据的采集、清洗与对齐
- 🎯 **对齐数据构造**：SFT 指令数据、RLHF 偏好数据、CoT 推理数据的自动化生成
- 🤖 **推理与 Agent 数据**：思维链、Tool-Use、多轮交互与记忆数据工程
- 🔍 **RAG 数据流水线**：企业级文档解析、语义切片与多模态检索
- ⚙️ **DataOps 与平台建设**：团队组织、数据版本管理、平台可观测性
- 🧾 **数据资产与数据产品**：资产目录、数据契约、价值评估与共享治理
- 🧠 **智能化数据工程**：Data Engineering Agent、自动化采集清洗、评测与 DataOps 自治
- 🔒 **隐私合规与安全**：数据治理框架、联邦学习与隐私保护技术
- 📊 **专项数据集与评测实践**：文档、表格、图表、医学 VQA、语音控制与推理数据工程案例

本书不仅有深入的理论讲解，更包含 **15 个端到端项目案例研究**，提供可运行代码、架构设计与工程复盘，让你能够把数据工程方法落到真实场景中。

**在线阅读**: [https://datascale-ai.github.io/data_engineering_book/](https://datascale-ai.github.io/data_engineering_book/)

## 全书架构

![大模型数据工程全书架构](images/structure_cn.png)

*从原始数据到端到端应用的完整数据工程流水线*

## 目录结构

```
📖 全书十四篇，48章 + 15个实战项目 + 7个附录（A–G）
│
├── 第一篇：总论与基础设施（第1-3章）
├── 第二篇：文本预训练数据工程（第4-7章）
├── 第三篇：多模态数据工程（第8-11章）
├── 第四篇：指令微调与偏好数据（第12-14章）
├── 第五篇：合成数据工程（第15-17章）
├── 第六篇：推理与 Agent 数据工程（第18-20章）
├── 第七篇：应用级数据工程（第21-23章）
├── 第八篇：数据运营与平台建设（第24-26章）
├── 第九篇：数据资产、数据产品与数据契约（第27-30章）
├── 第十篇：智能化数据工程与 Data Engineering Agent（第31-35章）
├── 第十一篇：隐私合规与数据安全（第36-37章）
├── 第十二篇：专项数据集与数据工程实践（第38-43章）
├── 第十三篇：开源大模型数据工程配方与范式（第44-48章）
└── 第十四篇：项目案例研究（P01-P15）
```

## 核心亮点

### 理论体系完整
- **Data-Centric AI** 理念贯穿全书
- 覆盖 LLM 数据全生命周期：预训练 → 微调 → RLHF → RAG → DataOps
- 深入讲解 Scaling Laws、数据质量评估、多模态对齐、隐私合规等前沿话题

### 技术栈现代化
| 领域 | 技术选型 |
|------|----------|
| 分布式计算 | Ray Data, Spark, Dask |
| 数据存储 | Parquet, WebDataset, 向量数据库 (Milvus/Qdrant) |
| 文本处理 | Trafilatura, KenLM, MinHash LSH, fastText 质量评分 |
| 多模态 | CLIP, ColPali, img2dataset |
| 数据版本 | DVC, LakeFS, MLflow |
| 平台可观测 | Great Expectations, Evidently AI, Apache Airflow |
| 隐私保护 | 联邦学习, 差分隐私, 安全多方计算 |

### 实战项目丰富

| 项目 | 核心技术 | 输出 |
|------|----------|------|
| Mini-C4 预训练集 | Trafilatura + Ray + MinHash | 高质量文本语料库 |
| 法律专家 SFT | Self-Instruct + CoT | 领域指令数据集 |
| LLaVA 多模态指令 | Bbox 对齐 + 多图交错 | 视觉指令数据集 |
| 合成数学教材 | Evol-Instruct + 沙箱验证 | PoT 推理数据集 |
| 财报 RAG | ColPali + Qwen-VL | 多模态问答系统 |
| CoT 推理 + PRM | 过程奖励模型 | 推理过程数据集 |
| Agent Tool-Use | 工具调用链 + 轨迹标注 | Agent 训练数据集 |
| DataOps 平台 | Airflow + DVC + 质量监控 | 企业级数据运营体系 |
| 隐私保护流水线 | 联邦学习 + 差分隐私 | 合规训练数据流水线 |
| LLM 数据飞轮 | 在线反馈 + 持续迭代 | 端到端闭环系统 |
| Mini-DeepSeek 复现 | 多源混合 + 跨源去重 + Tokenizer | 开源预训练数据配方复现 |
| R1 推理飞轮 | 多路采样 + verifier + 拒绝采样 | 推理数据闭环 |
| 多模态指令工厂 | VLM 生成 + Judge 过滤 + 多语言扩展 | 多模态 SFT 数据集 |
| 视频生成数据集 | 镜头切分 + 运动过滤 + 多帧 Caption | T2V 训练数据流水线 |
| DataAgent 语义问数助手 | 语义层 + Text-to-SQL + 权限审计 | 企业级问数 Agent 案例研究 |

## 本地运行

### 环境要求

- Python 3.8+
- MkDocs Material
- mkdocs-static-i18n（多语言支持）

### 安装与预览

```bash
# 克隆仓库
git clone https://github.com/datascale-ai/data_engineering_book.git
cd data_engineering_book

# 安装依赖
pip install mkdocs-material mkdocs-glightbox pymdown-extensions "mkdocs-static-i18n[material]"

# 本地预览
mkdocs serve
```

访问 http://127.0.0.1:8000 即可预览书籍（支持中/英/日切换）。

### 构建静态站点

```bash
mkdocs build
```

生成的静态文件位于 `site/` 目录。

### 导出 Springer 16K LaTeX 样稿

导出脚本需要本机可用 `tectonic`；使用 `--split --compile` 合并分篇 PDF 时还需要 `pdfunite`。图片完整性校验依赖 Pillow。

```bash
# 生成完整中文书稿 LaTeX，不立即编译 PDF
python scripts/export_zh_book_latex.py

# 按篇编译并合并 16K 审校 PDF（推荐）
python scripts/export_zh_book_latex.py --split --compile
```

输出文件位于 `output/pdf/`。若只需抽样验证，可使用 `--limit 3 --compile` 或 `--only part1/ --split --compile`。

### 验证发布与项目

```bash
# 严格构建站点，检查导航、断链和多语言配置
mkdocs build --strict --clean

# 检查站点图片体积预算
python scripts/check_image_sizes.py

# 运行 P01-P14 的统一 smoke test，并生成 smoke_reports/
# P15 当前为文稿案例研究章，暂无独立代码目录
python scripts/run_all_project_smoke_tests.py
```

## 项目结构

```
data_engineering_book/
├── docs/
│   ├── zh/                    # 中文内容
│   │   ├── index.md           # 中文首页
│   │   └── part1/ ~ part14/   # 各章节
│   ├── en/                    # 英文内容
│   ├── ja/                    # 日文内容
│   ├── images/                # 图片资源（中英共享）
│   ├── stylesheets/           # 自定义样式
│   └── javascripts/           # JavaScript (MathJax等)
├── .github/workflows/         # GitHub Actions 自动部署
├── images/                    # 项目图片资源
├── mkdocs.yml                 # MkDocs 配置文件
├── LICENSE                    # 开源协议
├── README.md                  # 中文说明（本文件）
├── README_en.md               # English README
└── README_ja.md               # 日本語 README
```

## 适合读者

- 大模型研发工程师
- 数据工程师 / MLOps / DataOps 工程师
- AI 产品经理（技术向）
- 对 LLM 数据流水线感兴趣的研究人员

## 主要作者

於俊教授团队

**实验室信息**：  
中国科学技术大学-语音及语言信息处理国家工程研究中心；中国科学技术大学-自动化系-多媒体计算及智能机器人研究中心；中国科学技术大学-自动化系-多模态智能体联合研究中心

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 联系我们

- GitHub Issues: [提交问题](https://github.com/datascale-ai/data_engineering_book/issues)
- 在线阅读: [https://datascale-ai.github.io/data_engineering_book/](https://datascale-ai.github.io/data_engineering_book/)

---

**如果这本书对你有帮助，欢迎 Star 支持！** ⭐
