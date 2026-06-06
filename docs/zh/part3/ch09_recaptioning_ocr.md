# 第9章 重标注与文档理解

## 摘要
本章深入探讨了多模态数据工程中的高阶策略：高质量重标注（Re-captioning）与 OCR 结构化文档理解。章节首先剖析了原生网络图文对存在的“弱描述”与“漏描述”现象及其对模型细粒度能力的制约，随后引入了分层重述流水线，涵盖开源小模型快刷、多模型交叉互审（MoE-Judge）及人工精调。此外，本章强调了细粒度对齐与包围盒（BBox）在培养模型空间感知能力中的重要性。针对长文档与复杂排版，探讨了外挂 OCR 增强与结构化版面拆解技术。最后，本章介绍了工业级数据质检框架，旨在通过自动化评分与专家抽检，确保合成数据的可靠性。

**关键词**：重标注 (Re-captioning)；OCR；文档理解；多模型互审 (MoE-Judge)；细粒度对齐 (Grounding)；数据质检

**学习目标**
- 解释网络原生图文对在细粒度特征上的缺陷及其对训练的影响。
- 掌握工业级金字塔分层重标注流水线的核心架构与权衡。
- 理解细粒度包围盒（BBox）注入技术对增强视觉模型空间感知能力的作用。
- 掌握复杂文档的版面拆解及外挂 OCR 增强的工程实施方法。
- 熟悉多模态数据质量评估与抽检归因框架的构建方法。

---

在上一章中，我们详细探讨了多模态数据清洗的前置流水线。通过剥离低分辨率图像、去除水印干扰、过滤敏感内容，并利用 CLIP Score 截断图文语义背离的劣质样本，我们在物理层面上构建了一个相对纯净的数据湖。

然而，即使使用这批高质量数据训练，模型在精细视觉任务上的表现往往依然存在局限。
当你问模型：“图中穿红衣服的女孩在干什么？”
模型会回答：“这里有一个穿红色衣服的女性。”
当处理复杂的长文档（如全英文财报扫描件）时，模型可能无法准确提取信息，甚至产生幻觉。

这就引出了多模态数据工程步入深水区的核心命题：**为什么干净的图文数据，依然不足以支撑一个能思考的高阶多模态模型？** 本章将解答这个问题，并带你深入多模态两项关键技术：**高质量重标注策略（Re-captioning）** 与 **OCR 结构化文档理解（Document Understanding）**。

---

## 9.1 为什么原始 Caption 远远不够用？

### 9.1.1 “弱描述”与“漏描述”带来的特征缺失现象

即使那些经过前置流水线（如基于 CLIP Score 的双塔计算，余弦相似度拉满的对齐操作）层层筛选出的多模态数据，其绝大部分源头依然是基于全网爬取的。互联网文本生态存在一个普遍问题：网页开发者为了追求页面加载效率，或受限于 HTML 文本填充组件（`Alt-text`）的设计局限，**原生网页对网络图片的伴生描述普遍较为简略，缺乏细节**。

以大规模开源数据集 LAION-5B (Schuhmann et al. 2022) 中的某个典型样本为例：一张拍摄到了金色阳光斜射的实木书桌，桌上放着一把樱桃青轴机械键盘、一杯喝了一大半的冰美式咖啡，且背景有着极具层次感的模糊书架（一张极高分辨率 of 1080p 摄影图）。但在数据工厂爬取的原生态配对中，其极其简短的文本标签（Caption）很可能只是：
> “*一个普通的办公桌桌面*” 或者仅仅是 “*IMG_2023_Office.jpg*”

**对模型底层表征的负面影响：**
如果将海量模糊且泛化的标签用于模型训练，模型将难以学到图像中的细节特征。模型在压缩高分辨率图像时，试图拟合过于简略的文本。
因此，在这场跨模态的训练中，模型无法建立“半杯咖啡的反光”、“机械键盘键帽的凹陷”以及“阳光照射的真实光影方向”在视觉编码器隐空间里的精确表征。这种由于文本特征缺失导致视觉模型自动屏蔽小面积物体的现象，在深度学习工程界被称为**“弱描述导致的密集物体失明（Dense Object Blindness / Entity Dropout）”**。

这不仅导致模型丧失了细节特征捕捉的精细颗粒度，更重要的是，**漏描述会引发逻辑对齐冲突与训练震荡**。设想图片里明明占据主位的是一只白猫，而爬取的短文本只字未提“猫”，只提了“白色背景”。那么当视觉编码器（Vision Encoder）抽取出猫的轮廓特征并要求与文本对齐时，视觉网络就会产生语义冲突。它会错误地将猫的轮廓与“背景”建立关联，从而损害最终多模态基础理解能力的稳定性（例如导致模型在 MME (Fu et al. 2023)、MMBench (Liu et al. 2023b) 的目标检测与归纳分数受限）。

### 9.1.2 简略描述与长文本细节描述的生态壁垒差异

在真实的研发与数据蒸馏管线中，数据架构师通常需要针对不同的训练阶段，将视觉数据划分为不同文本粒度的类别进行输入：

1. **简短核心感知描述（Brief / Short-form Caption）**：
   - 展现形式：例如“一只奔跑的金毛犬”或“两辆停泊的红色轿跑”。
   - 核心效用：此类数据成本低廉且容易获得（数十亿条级别）。它的主要价值，是能够在**多模态预训练早期的对齐阶段（Stage 1: Modality Alignment）**迅速建立基本的投影关系。
   - 工程意义：这有助于在早期阶段建立模型对基础物体外观与词汇的对应关系。

2. **长文本密集结构化描述（Dense Detailed Caption / Recounting）**：
   - 展现形式：如“在一片光线充足的午后草地上，天空飘着两朵稀薄的积雨云。一只金毛犬正张着嘴向画面右侧奔跑...”
   - 核心效用：此类数据资源稀缺，原始互联网上抓取量极少，通常依靠二次计算合成或人工标注。
   - 工程价值：它在高级阶段发挥重要作用。只有在**高阶 SFT 微调阶段（Stage 2 & Stage 3: Visual Instruction Tuning & Preference Alignment）**被精准地引入，才能训练模型对周围复杂环境进行精细的观察与叙述。

3. **混合配比（Data Mixing）的艺术**：
   - **不能多喂短描述**：预训练后期如果 90% 都是短描述，模型会丧失长句生成能力（Caption Degradation）。
   - **黄金比例**：在工业界标准的 SFT 阶段，通常采用 **30% 短描述 + 50% 密集描述 + 20% 多模态对话** 的混合比例，以兼顾概念认知与指令服从性。

因此，要让模型从单纯的“看图识字（感知）”走向深度的“图像理解与物理世界规律推理（认知）”，提升多模态认知与推理能力的关键途径之一是引入**大模型合成重标注策略（Synthetic Re-captioning）**。

---

## 9.2 工业级重描述策略 (Re-captioning) 的工程实现

重描述（Re-captioning）的核心底层思想其实非常直白：面对原生数据缺乏细节的问题，一种有效的策略是使用能力更强的视觉大模型或专家对图像进行重新标注，生成高质量的长文本描述。

在大规模多模态模型预训练中，对十亿量级的图像数据进行重标注面临极高的算力与时间成本。因此，需要构建金字塔式的分层调度流水线，对不同质量和复杂度的图像采取差异化的处理策略。

### 9.2.1 金字塔分层流水线：从小尺寸快刷到大模型互审 (MoE-Judge)

为了在算力成本与数据质量之间取得平衡，业界通常采用**分层过滤与多级调度策略**：

#### （1）基础图像重注：开源模型自动化批量重注（Fast Prompting）
对于那些构图简单、基础物体占比超过 70% 的常识自然图像（如一张风景照或商品图），考虑到庞大的数量，通常依赖部署于内部集群上的开源视觉模型（如 LLaVA-1.5、Qwen-VL-Chat 等）进行快速重标注。

**Prompt 模板约束示例**：
为了避免开源模型生成发散或冗余的文本，Prompt 设计必须高度收敛。
**代码清单 9-1：重标注提示词（Prompt）约束示例**
```text
[System Instruction]: You are a neutral, highly objective visually impaired helper. 
[Task]: Describe the main objects, actions, and physical background in this image concisely and accurately. 
[Constraint]: Do NOT use any generic filler words like 'This is an image of' or 'I can see'. Do NOT guess the location if no text is shown. Keep the entire response strictly under 50 words. Focus solely on visible facts.
```

#### （2）中坚层对冲：多模型交叉互审机制（Multi-Model-as-a-Judge）
当我们面对复杂的“交错场景”图像、密集的杂乱环境或是含有细微文化特征的图片时，单一的模型可能会产生幻觉。
为了降低单一模型的偏差，流水线会自动将此类复杂数据路由至**“多模型交叉互审制（MoE-Judge）”**流水线中：
1. **并行分诊（Parallel Inference）**：图像同时且独立地送入架构截然不同的视觉引擎 $V_1$ (如 LLaVA)、$V_2$ (如 InternVL) 以及 $V_3$ (如 Pix2Struct 或 Donut)。
2. **多样化描述生成（Diverse Description Generation）**：不同的视觉模型会基于其参数特性生成不同侧重点 of 描述 $C_1, C_2, C_3$。
3. **判官裁决（LLM Judgement & Fusion）**：裁判模型会分析三份描述中的共性实体与属性，通过交叉验证过滤掉单一模型可能产生的幻觉与孤立噪声，最终融合成一份兼顾细节且事实准确度高的无争议重述文本。

#### （3）顶层提纯：强人工精调与 Golden Truth 标尺确立
在流水线最顶层，自动化脚本的作用有限，通常需要依赖高质量的人工标注。
这些人工标注需具备极高的精确度，参与者需经过系统培训。这一小撮数据虽然少，但它构成了随后重描述打分系统或微调基座模型时至关重要的 **Golden Truth（金标真值库）**。

**表9-1：重描述自动化生产梯队对比与优劣表**

| 重述层级调度方式 | 每百万张评估成本 | 集群并发生产吞吐速度 | 复杂场景及图表解析能力 | 核心优越性与落地防范隐患 |
| :--- | :--- | :--- | :--- | :--- |
| **小参数 VLM 本地批刷** (参数 $< 15B$) | \~$100 极低 | > 14,000 张/节点/小时 | 较弱（面对复杂图表解析能力不足） | **优势**：计算成本低，能快速建立底层基础概念的跨模态对齐。<br>**防范隐患**：容易产生局部幻觉，不适用于高精度细粒度训练。 |
| **头部商业 API 提纯** (API 如 GPT-4o) | \~$15,000 | 严重受限 (并发限流 \~5K/小时) | 极强，逻辑满分 | **优势**：拥有高水平的语境常识感知，产出的长文本高密度、高质量。<br>**防范隐患**：调用成本高，且受限于 API 的安全对齐策略，可能会被频繁拒绝回答。 |
| **私有化混合框架多路互审** | \~$800 (纯折算内部消耗卡时) | ~ 2,000 张/多节点/小时 | 较弱 | **优势**：本地自建部署保护数据隐私，通过交叉验证有效过滤单一模型幻觉。<br>**防范隐患**：系统架构较为复杂，多模型推理会增加整体延迟。 |
| **高标准人工精标注** | \~$200,000 以上 (高昂成本) | < 50 张/专家/小时 | 高精度解析 | **优势**：作为高质量基准数据，用于模型微调与评估基准。<br>**防范隐患**：难以规模化量产，且人类标注员长时间工作可能因视觉疲劳导致标注偏差。 |

### 9.2.2 视觉定位与空间感知：细粒度对齐与包围盒双向注入

传统 Image Caption 技术仅将图像表示为词汇集合。为了使模型具备空间感知和方位理解能力，必须引入**细粒度属性定位标记 (Fine-grained Grounding)**。

在重写图像描述流水线上，系统会引入如 **GroundingDINO** (Liu et al. 2023c) 或 **SAM (Segment Anything Model)** (Kirillov et al. 2023) 等零样本目标检测器。这些检测器可以提取图像中所对应物体的精确像素坐标，并将其映射为归一化坐标序列（例如：一颗苹果定位于 `[x_min=320, y_min=550, x_max=450, y_max=690]` 这一包围盒中）。

随后，文本组装脚本会将这些坐标结构化并向自然语言中**注入闭合的 XML 定位标记**：

**代码清单 9-2：包含坐标信息的 XML 标注示例**
```xml
在画面深处的木制方桌的左下位置（阳光暗面），静静放置着一颗红润透亮的 <object name="apple" bbox="[[320, 550, 450, 690]]">苹果</object>，不仅如此，紧贴着它的左侧还有一摞极其陈旧的厚重 <object name="book" bbox="[[500, 520, 680, 750]]">相关医学书籍</object>。
```

由于 Transformer 本身不具备固有的空间感知，当海量包含离散坐标的数据输入 SFT 流水线后，模型在图像上的像素区域与文本之间建立了精确的映射。在推理阶段，模型不仅能够描述“图里有什么”，还能在被要求指出具体位置时，输出精确的坐标点阵并在用户界面上进行框选。这不仅能有效降低模型的幻觉率，也是开发智能视觉代理（Visual Agent）的核心数据基础。

**代码清单 9-3：工业级重描述 JSONL 样例（Re-captioning Schema）**
最终经过 VLM 合成的重描述数据，会被封装成带有严格元数据的 JSONL 文件：

```json
{
  "image_id": "laion_5b_recap_001923",
  "image_path": "s3://dataset/images/001923.jpg",
  "original_caption": "IMG_2023_Office.jpg",
  "recaption": {
    "dense_caption": "在画面深处的木制方桌的左下位置，静静放置着一颗红润透亮的苹果...",
    "source_model": "InternVL-1.2-MoE",
    "generation_prompt": "You are a neutral, highly objective visually impaired helper..."
  },
  "grounding_bboxes": [
    {"entity": "apple", "bbox": [320, 550, 450, 690]}
  ],
  "clip_score": 0.82,
  "quality_flag": "PASS"
}
```
**字段解释**：
- `original_caption`：原始网页抓取的简略标签。
- `recaption`：大模型合成的长文本描述与生成模型源记录。
- `grounding_bboxes`：通过 GroundingDINO 提取并强行映射 of 细粒度实体坐标，是训练基座具备空间感知能力的核心。
- `clip_score`与`quality_flag`：用于前置校验过滤的自动打分，低于 0.65 则设为 REJECT 丢弃。

![图9-1：重标注与 OCR 双流线增强图](../../images/part3/recaptioning_ocr_pipeline.png)

*图9-1：重标注与 OCR 增强联合的双轨高阶管道图（Dual-track Pipeline） —— 左侧展现了宽泛语义叙述流（Semantic Vision Track），而右侧展示了包含 DOM 排版分割与表格矩阵的结构流（Structural Text Track）。最终归一融合结为统一的混合监督格式。（来源：本书自绘）*

至此，针对自然图像与纯景物类图片的重标注强化管线已经建立完毕。然而，真正能够在商业落地中区分企业级视觉大模型竞争力的，是接下来的高密度字符深水区：面向长文档的强阅读推理与复杂商业报表的结构化解码解析。

---

## 9.3 突破视觉像素极限：OCR 增强与高密度文档理解

在自然景物图片中，普通的 VLM 似乎对分辨物体驾轻就熟。可一旦处理扫描版的增值税发票或包含复杂表格的 PDF 商业报告，即使将图像切分推高至 AnyRes 或 4K 分辨率输入，模型依然可能读错关键数字，或将不同列的财报数据错误关联，导致幻觉。

这背后的深度机理在于：无论架构多复杂，Vision Transformer (ViT) **提取的主要是大块连贯区域的光影与颜色纹理规律**。然而，文字是极端稀疏、充满高频断崖的离散突变点。对于字符而言，差一个偏旁部首所代表的语义可能完全不同。这对于纯视觉特征提取来说是巨大的挑战。

为了攻克这一难题，数据工程领域通常会引入辅助流水线：**外挂 OCR 与文档解析强制增强流水线 (Optical Character Recognition & Layout Boosting)**。

### 9.3.1 文档图像的结构化拆解与版面分析

处理长文档不仅需要获取连续的文本内容，更需要解析复杂的排版结构。商业长文本常包含**非线性排版（Non-linear Layouts）**，如双栏排版、嵌入式图表、侧边注释以及水印页眉等。若在输入模型前未进行结构解构与版面分析，直接暴力拼接文本序列将导致原本的阅读顺序被打乱，从而破坏文本的语义连贯性。

在工业级的数据预处理流水线中，文档解析流程通常包含以下几个层级：

1. **第一级 OCR：版面边界截断网（Layout Detection）**
   第一步是利用版面定位网络（如基于 YOLOv8 或 LayoutLMv3 (Huang et al. 2022)）。它的任务是**版面定位**，精准识别：标题组（Title）、正文池（Body Text）、脚注（Footnote）、柱状图容器及代码块。
2. **第二级 OCR：多维领域特化提取管线（Domain-specific Extraction Pipeline）**
   PDF 文档经版面拆解后，不同类型的区域模块（如文本块、公式块、表格块）会被分发给特定的提取算子进行解析：
   - **文档级文本提取**：对于纯文字段落，分发给 Tesseract 或 PaddleOCR 进行高精度拼写矫正提取。
   - **数学公式逆向编译**：遇到密集公式组，标准 OCR 错误率极高。路由给专门微调的开源引擎（如 Nougat (Blecher et al. 2023)）或商业服务（如 Mathpix），将图像直接还原为严格的 LaTeX 代码流（如：`\int_{0}^{\infty} e^{-x^2} dx`）。
   - **复杂表格拓扑重构**：对于包含合并单元格与跨页表头的复杂表格，常规 OCR 的识别效果有限。在工程上，通常采用专门的表格识别模型（如 TableMaster）解析表格的拓扑结构，并将其转换为 HTML 标签或 Markdown 格式的结构化数据。

在多级 OCR 提取后，核心工程难点在于**坐标全对准机制（Modality Absolute Geometric Alignment）**。提取出的这三类文字如果不与图片上的像素区域建立映射，模型依旧不知道眼睛往哪儿看。业界在每段文本后强行追加 `<box_coord>` 的文本映射串，迫使注意力机制遵循这些“空间参考信息”。

![图9-2：文档结构 Layout-to-Token 映射图](../../images/part3/document_structure_sample.png)

*图9-2：文档结构 Layout-to-Token 映射图 —— 展示了一份复杂双栏学术报告残页通过 Bounding Box 阵列进行拆卸提取，由外挂特化模型（如 Nougat+PaddleOCR）解析后，通过脚本归并生成 Markdown 文本与精确坐标的富文本数据流。这为模型提供了重要的版面认知知识。（来源：本书自绘）*

### 9.3.2 视觉与结构化文本的语义融合

这种建立在“视觉特征提取 -> Bounding Box -> 结构化离散字符串序列”基础上的数据预处理机制，一旦打通，将极大地释放底层训练集群的计算瓶颈。因为原本需要视觉模型在训练期耗费大量算力去识别的高难度字符集，已经在外挂预处理车间（CPU/GPU 混合 OCR Pipeline）中被解析为了长文本 Prompt，并作为上下文输入。此时，视觉模型只需在相对较低的分辨率下快速处理全图，**提取宏观的排版布局与物理空间特征即可**。

系统随后将把提取的文本序列交由模型内部的**长上下文序列分析器（Long-context LLM）**处理。
这种方式将多模态图文理解转化为长文本理解，发挥了语言模型处理长上下文的优势。这也解释了近期的一些架构（如 Qwen-VL 系列）为何能仅凭百亿级参数，在商业文件和财报解读榜单中展现出优异的表现。

---

## 9.4 质量评价框架、抽检漏斗与缺陷归因测试矩阵

在大规模的数据预处理流水线中，如果缺乏严格的质量监督，幻觉样本的混入将严重损害训练效果。因此，必须建立起一套严密的**工业级数据质检框架**。

### 9.4.1 基于规则与多维评估模型的数据过滤

我们首先必须部署全量自动化的大规模启发式与验证模型：

1. **定量一致性校验（Consistency Penalty Test）**：
   - **算法架构流水线**：重注中心通常会产出长达 500 字的详细描述（Dense Caption）。前置质检探针会先将这些详细描述通过轻量级分词与命名实体识别工具（如 SpaCy），强制抽提浓缩成数个最核心的物理实体名词（如“键盘、咖啡、桌子、显示器、阳光”）。
   - **相似度判定**：随后，计算这些核心名词的特征向量与图像特征的 CLIP Score 余弦相似度。如果发现其内积均值未达到设定阈值，或相比原始简短 Caption 出现异常跌落，系统会触发过滤机制并丢弃该样本。
2. **基础标点、正则流与格式校验过滤（Syntax & Glitch Sweeping）**：
   - 即便不看复杂的几何语义模型反馈，光看生成文本的字符排布也能抓住严重的合成质量问题。如果大批经过 OCR 模型（如 PaddleOCRv4）加持后生成的 PDF 富文本数据末尾，频频出现了孤立异常规模的未闭合 HTML 标签诸如 `</html>`、连串的 `[ERROR] [NO_RESPONSE]` 或乱码污染。
   - 此外，还需要过滤大模型长程推理中容易出现的连续的大段重复文本（通常由生成循环导致）。在工程实践中，一旦在生成数据中检测到超过设定阈值的重复率或格式异常，数据流水线应当自动熔断该样本的生成并予以剔除。

### 9.4.2 人工盲抽与归因框架

即便自动化检测能够过滤大部分低质数据，高质量的**人工抽样校验（Human-in-the-Loop Sampling & Verification）**仍然是不可或缺的环节。在质检流程中，通常会定期随机抽样一部分复杂的文档测试样本，分发给专业的人工审核团队进行细致校验。

这批专家不仅要评判优劣，更要从生成样本中提取多维度的特征日志，为模型训练与数据工程团队提供详尽的“缺陷归因分析”。在多模态模型表现未达预期或训练异常时，通过建立标准化的排查矩阵，定位是视觉编码端、数据预处理端还是语言解码端的缺陷：

**表9-2：跨模态及高级文档识别 OCR 核心错误归因与修复阵列矩阵**

| 缺陷特征在模型端的表现层 | 数据工程根源诊断 | 数据流水线与模型架构修复策略 |
| :--- | :--- | :--- |
| **数学公式或财务票据关键字符识别错误** | 外挂 OCR 或表格识别引擎在特定排版下的识别精度缺陷。 | 优化或更换专用公式识别引擎（如 Mathpix/Nougat），微调视觉端模型，并提高 AnyRes 切片的分辨率上限。 |
| **版面结构严重错乱（如分栏混淆、图表图例串区）** | 物理版面分析算法（Layout Analysis）对复杂背景或水印的边界框定失效。 | 优化版面分析算法（如采用全监督 LayoutLM 或 YOLO 架构的定位器），结合启发式版面合并算法重建阅读顺序。 |
| **无中生有的常识性幻觉（如将常见物品过度发散描述）** | 标注大模型（Re-captioning）在合成长文本时过度发散，产生了常识性或多模态 of 联想幻觉。 | 采用具备强对齐与拒绝机制的大尺寸模型进行重标注，并在流水线中引入多模型投票判决机制以剔除主观修饰性词汇。 |
| **输出完全无序、出现非相关乱码或输出全零矩阵** | 底层数据多线程序列化、字节打包或分词器（Tokenizer）映射转换过程发生越界或指针泄漏。 | 暂停训练任务，检查数据加载与读取算子层（C++/CUDA 接口），排查内存泄漏与分词字典越界问题。 |

---

## 9.5 真实工程案例与本章总结

### 9.5.1 金融文档解析重构案例

在构建行业级金融研报与财务文档解析引擎的初期阶段，某团队直接使用多模态大模型对含有大量复杂图表的招股书及财报 PDF 进行问答。然而，在初步测试中发现，模型在面对细节性财务指标（如跨年度的同比与环比数据对比）时，表现出显著的准确率不足，甚至会产生数值性幻觉；同时，在包含页眉、水印和多栏复杂排版的扫描件上，模型的语义提取能力也受到较大制约。

为了解决这一问题，该团队对数据工程流程进行了系统性重构：
1. **精准版面切割与去噪**：引入版面定位模型，将复杂的财报页面解构为正文块、表格块与图表块，并去除无关的水印与页眉干扰。
2. **结构化表格重建**：采用高精度表格识别算法，将财报中的财务报表转换为 HTML 与 Markdown 格式的结构化数据，并附带单元格在页面中的包围盒坐标。
3. **图表语义重描述**：利用大模型对饼状图、折线图等数据分析图表进行语义提炼，生成高密度的结构化文本描述。

在重构后的数据集支持下，仅经过数个周期的训练，该大基座模型在 ChartQA (Masry et al. 2022) 与 TabMWP (Lu et al. 2022) 等多模态图表理解 benchmark 上的表现得到了显著提升，充分验证了结构化文档数据工程对多模态模型推理能力的关键作用。

### 9.5.2 本章总结与后续技术关联

本章系统探讨了多模态数据工程中关于图像重标注与结构化文档理解的实践方案。通过分层重标注流水线与细粒度包围盒（BBox）的注入，我们可以显著提高模型对自然图像与空间几何排版的感知精度；而通过版面拆解与特化 OCR 的语义融合，我们成功将二维平面上的密集文档数据转换为模型可理解的高质量结构化序列。

然而，当前的讨论主要局限于静态二维图像与平面文档。真实物理世界通常具有时序上的连续性与多模态感官的交织性（如视频与音频的结合）。当处理大规模、高帧率的时序媒体数据时，静态 AnyRes 等切片策略将面临严重的算力与内存带宽瓶颈。下一章将探讨“第10章 视频与音频流多模态数据工程基建”，介绍时序动态压缩、多轨音频对齐以及工业级长视频管线的构建方法。

## 参考文献

Bai J, Bai S, Yang S, Wang S, Tan S, Wang P, Lin J, Zhou C, Zhou J (2023) Qwen-VL: A Versatile Vision-Language Model. arXiv preprint arXiv:2308.12966.

Blecher N, Cresci G, Ballas N, Bautista M (2023) Nougat: Neural Optical Understanding for Academic Documents. arXiv preprint arXiv:2308.13418.

Fu C, Chen P, Shen Y, Qin Y, Zhang M, Lin X, Qiu Z, Lin W, Yang J, Zheng X, Li K, Sun X, Wu E (2023) MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. arXiv preprint arXiv:2306.13394.

Huang Y, Lv T, Cui L, Lu Y, Wei F (2022) LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. In: Proceedings of the 30th ACM International Conference on Multimedia, pp 4083-4091.

Kim G, Moon S, Xu R, Yim J, Park J, Seo J, Baek J, Yoo M, Park S, Park S (2022) OCR-Free Document Understanding Transformer (Donut). In: European Conference on Computer Vision, pp 498-517.

Kirillov A, Mintun E, Ravi N, Mao H, Rolland C, Gustafson L, Xiao T, Whitehead S, Berg A C, Lo W Y, others (2023) Segment Anything (SAM). In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 4015-4026.

Lee J, Jia M, Sangkloy P, Krishnamurthy J, Han S, Chang S F, Hutchinson B (2023) Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding. In: Proceedings of the 40th International Conference on Machine Learning, pp 18893-18912.

Liu H, Li C, Wu Q, Lee Y J (2023b) MMBench: Is Your Multi-modal Model an All-around Player? arXiv preprint arXiv:2307.06281.

Liu S, Zeng Z, Ren T, Li F, Zhang H, Yang J, Li C, Yang J, Su H, Zhu J, Zhang L (2023c) Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. arXiv preprint arXiv:2303.05499.

Liu H, Li C, Li Y, Lee Y J (2024) Improved Baselines with Visual Instruction Tuning (LLaVA-1.5). In: CVPR 2024, pp 26296-26306.

Lu P, Qiu L, Chang K W, Zhu W, Rajpurohit T, Clark P, Kalyan A (2022) Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning (TabMWP). arXiv preprint arXiv:2209.14610.

Masry A, Long D, Tan J Q, Joty S, Hoque E (2022) ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. In: Findings of the Association for Computational Linguistics: ACL 2022, pp 2263-2279.

Radford A, Kim J W, Hallacy C, Ramesh A, Goh G, Agarwal S, Sastry G, Askell A, Mishkin P, Clark J, others (2021) Learning Transferable Visual models From Natural Language Supervision (CLIP). In: ICML 2021, pp 8748-8763.

Schuhmann C, Beaumont R, Vencu R, Gordon C, Wightman R, Cherti M, Coombes T, Katta A, Mullis C, Wortsman M, others (2022) LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models. Advances in Neural Information Processing Systems 35:25278-25294.
