# **《大模型数据工程：架构、算法及项目实战》**

(Large Model Data Engineering: A Practical Guide from Pre-training to Multimodal Alignment)*

------

# 全书目录概览

- **第一部分：基础设施与核心理念** (构建数据底座)
- **第二部分：文本预训练数据工程** (清洗与去噪)
- **第三部分：多模态数据工程** (图文、视频与音频)
- **第四部分：对齐与合成数据工程** (指令与质量)
- **第五部分：应用级数据工程** (RAG与Agent)
- **第六部分：实战项目集 (Capstone Projects)** (端到端代码实现)

------

# 详细大纲内容

## 第一部分：基础设施与核心理念

> **目标：** 建立 Data-Centric AI 的认知，完成高性能数据处理环境的搭建。

### 第1章：大模型时代的数据变革

- 1_1 **Scaling Laws 的启示：** 数据质量 > 数量，从“大数据”到“高质量数据”的范式转移。
- 1_2 **LLM 数据全生命周期：** 预训练(Pre-train) $\rightarrow$ 微调(SFT) $\rightarrow$ 强化学习(RLHF) $\rightarrow$ 检索增强(RAG)。
- 1_3 **挑战与机遇：** 异构多模态、版权合规与算力成本的博弈。

### 第2章：数据基础设施选型

- 2_1 **现代数据栈 (Modern Data Stack)：**
  - 存储：对象存储 (S3/MinIO) vs 数据湖 (Iceberg/Hudi)。
  - 计算：Spark (传统霸主) vs **Ray Data** (AI原生计算框架)。
- 2_2 **数据格式与I/O优化：**
  - Parquet vs JSONL vs WebDataset (多模态场景)。
  - 压缩算法与读取性能优化。
- 2_3 **数据版本控制 (DataOps)：** 使用 DVC 和 LakeFS 管理 PB 级数据集。

------

## 第二部分：文本预训练数据工程 (Text Pre-training)

> **目标：** 处理海量无结构文本，构建模型的语言认知基座。

### 第3章：数据获取与采集

- 3_1 **开源数据集解构：** Common Crawl, C4, RefinedWeb, The Pile 深度剖析。
- 3_2 **高性能爬虫系统：** `Trafilatura` 解析库应用与分布式爬虫架构设计。
- 3_3 **特种数据获取：** 代码 (GitHub)、论文 (ArXiv/S2ORC)、书籍数据的提取策略。

### 第4章：清洗与去噪 (Cleaning & Deduplication)

- 4_1 **启发式过滤规则：** 语言识别 (FastText)、困惑度 (Perplexity) 过滤、长度与标点分布。
- 4_2 **大规模去重技术：**
  - **模糊去重 (Fuzzy Deduplication)：** MinHash LSH 算法原理与分布式实现。
  - **文档内去重：** 消除重复段落与导航栏。
- 4_3 **隐私清洗 (PII Removal)：** 使用 Presidio 识别并掩盖 Email、IP、电话、地址。

### 第5章：分词与序列化 (Tokenization)

- 5_1 **分词器原理：** BPE, WordPiece, Unigram 及其对模型性能的影响。
- 5_2 **高效词表构建：** 如何为特定领域（如医疗、法律）扩充词表。
- 5_3 **数据混合 (Data Mixing)：** 动态采样策略与 Curriculum Learning (课程学习) 数据排布。

------

## 第三部分：多模态数据工程 (Multimodal Data Engineering)

> **目标：** 处理图像、视频与音频，支持 GPT-4V/Sora 类模型的训练。

### 第6章：图文对数据处理 (Image-Text Pairs)

- 6_1 **数据范式：** 图文对 (LAION-5B) vs 交错文档 (OBELICS/MMC4)。
- 6_2 **图像获取与预处理：**
  - `img2dataset` 高并发下载实战。
  - GPU 加速解码与变换 (NVIDIA DALI)。
- 6_3 **多模态清洗流水线：**
  - **美学评分 (Aesthetics)：** 使用 CLIP-Score 筛选高美感图片。
  - **图文对齐过滤：** 剔除描述与图片不符的样本。
  - **安全性检测：** NSFW 与水印识别。

### 第7章：数据重描述 (Recaptioning)

- 7_1 **Alt-text 的局限性：** 为什么原始网页描述不可用？
- 7_2 **合成描述工厂：**
  - 利用 BLIP-2 / LLaVA / CogVLM 重新生成详细 Caption。
  - **Prompt 策略：** 控制生成描述的颗粒度（简略 vs 详尽）。
- 7_3 **OCR 增强：** 提取图中文字并融合进文本描述（对文档理解至关重要）。

### 第8章：视频与音频数据

- 8_1 **视频处理流水线：** 场景切分 (Scene Detection) 与关键帧提取策略。
- 8_2 **视频 Tokenization：** 视频压缩与离散化表示。
- 8_3 **音频对齐：** 使用 Whisper 进行大规模 ASR 及 Force Alignment (时间戳对齐)。

------

## 第四部分：对齐与合成数据工程 (Alignment & Synthetic Data)

> **目标：** 让模型听懂指令，并突破人类数据的瓶颈。

### 第9章：指令微调数据 (SFT Data)

- 9_1 **Prompt Engineering 为数据生产服务：** 编写高鲁棒性的 System Prompt。
- 9_2 **自动化构造方法：**
  - **Self-Instruct：** 利用强模型生成指令。
  - **Evol-Instruct：** 指令复杂度的进化策略。
- 9_3 **思维链 (CoT) 数据：** 构造 Step-by-Step 的推理样本。

### 第10章：合成数据 (Synthetic Data)

- 10_1 **教科书级数据 (Textbooks Are All You Need)：** 合成高质量领域知识。
- 10_2 **代码与数学合成：**
  - **PoT (Program of Thought)：** 生成代码并执行，以执行结果验证数据正确性。
- 10_3 **多模态指令合成：** 利用 GPT-4o 构造基于图像的复杂推理问答。

### 第11章：人类偏好数据 (RLHF/DPO)

- 11_1 **偏好数据格式：** Chosen vs Rejected 样本对构建。
- 11_2 **标注平台与质检：** 众包管理与 IAA (标注一致性) 分析。
- 11_3 **RLAIF (AI Feedback)：** 使用 LLM 代替人类进行偏好打分。

------

## 第五部分：应用级数据工程 (RAG & Agent)

> **目标：** 面向企业落地，解决外部知识库的解析与检索。

### 第12章：RAG 数据流水线

- 12_1 **文档深度解析：**
  - 复杂 PDF 处理：表格还原、多栏识别 (`Unstructured`, `LlamaParse`).
- 12_2 **切片策略 (Chunking)：** 语义切片、递归切片与父子索引 (Parent-Child Indexing)。
- 12_3 **向量化与存储：** Embedding 模型微调与向量数据库优化。

### 第13章：多模态 RAG

- 13_1 **跨模态检索：** 使用 CLIP/SigLIP 实现“以文搜图”与“以图搜文”。
- 13_2 **ColPali 架构实战：** 基于视觉语言模型的文档检索（跳过 OCR，直接理解文档图像）。

------

## 第六部分：实战项目集 (Capstone Projects)

> **目标：** 通过 5 个端到端项目，串联全书技术点，提供可运行的代码库。

### 项目一：构建“Mini-C4”预训练集

- **场景：** 从 Common Crawl 原始数据 (WARC) 到高质量 Parquet 数据。
- **核心技术：** Trafilatura 解析、Spark/Ray 分布式 MinHash 去重、KenLM 质量过滤。
- **输出：** 清洗后的纯文本语料库与处理 Pipeline。

### 项目二：垂直领域专家 SFT (法律/医疗)

- **场景：** 基于非结构化 PDF 文档构建行业专家微调数据。
- **核心技术：** Self-Instruct 构造指令、CoT 推理增强、数据多样性平衡。
- **输出：** `domain_expert.jsonl` 指令微调集。

### 项目三：构建 LLaVA 多模态指令集

- **场景：** 训练一个能看懂图片的多模态模型。
- **核心技术：** 使用 GPT-4o API 生成多轮图文对话、Bounding Box 数据对齐、多图交错格式处理。
- **输出：** 包含视觉指令的图文数据集。

### 项目四：合成数学/代码教科书

- **场景：** 提升小模型的逻辑推理能力。
- **核心技术：** Evol-Instruct 进化策略、Python 代码执行沙箱 (Sandbox) 验证、PoT 数据格式化。
- **输出：** 经过验证的高质量合成推理数据集。

### 项目五：多模态 RAG 企业财报助手

- **场景：** 检索并回答包含复杂图表的年度财报问题。
- **核心技术：** PDF 表格与图表解析、多路召回 (混合检索)、ColPali 视觉检索应用。
- **输出：** 一个支持图表问答的 RAG 知识库系统。

------

## 附录

- **附录 A：** 常用工具速查 (Hugging Face Datasets, LangChain, Ray Data)。
- **附录 B：** 数据合规自查清单 (版权、GDPR、机器人协议)。
- **附录 C：** 算力成本估算表 (不同规模数据清洗的 GPU/CPU 消耗参考)。