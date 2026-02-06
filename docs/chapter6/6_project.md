
# 实战部分


## 项目一：构建“Mini-C4”预训练集

### 1. 项目背景 (Project Brief)

*   **任务定义：** 构建一个微缩版 C4 (Colossal Clean Crawled Corpus) 数据集流水线。我们的目标是将杂乱无章的原始网页数据（Common Crawl）转化为低噪、去重、高质量的纯文本数据，可直接用于大模型预训练。
*   **输入与输出：**
    *   **Input:** Common Crawl 的原始 WARC 压缩包（包含 HTTP 响应头、HTML 源码、乱码等）。
    *   **Output:** 分类分级后的 JSONL 文件（如 `data_en.jsonl`, `final_data.jsonl`），包含纯净文本及其质量评分。
*   **难点分析：**
    *   **信噪比极低：** 原始网页中 90% 以上是导航栏、广告、JavaScript 代码和无意义的占位符。
    *   **计算密集：** 在大规模语料中进行两两比对去重（Deduplication）极其消耗资源。
    *   **质量量化：** 如何让机器自动判断一句话是“人类高质量语言”还是“机器生成的垃圾”？

### 2. 架构设计 (Architecture Design)

为了处理非结构化的 Web 数据，我们设计了如下的漏斗型（Funnel）处理架构：

**数据流水线图：**

![图1：构建“Mini-C4”预训练集数据流水线图](../images/实战项目/图1_构建“Mini-C4”预训练集数据流水线图.png)
<!-- ![图1：构建“Mini-C4”预训练集数据流水线图](images/实战项目/图1_构建“Mini-C4”预训练集数据流水线图.png) -->

**技术栈清单：**

*   **解析层：Trafilatura**
    *   *决策理由：* 相比传统的 BeautifulSoup，Trafilatura 专为网页正文提取优化，能自动去除导航、页脚和样板文字，提取效率和准确率更高。
*   **计算层：Ray**
    *   *决策理由：* Python 原生多进程处理大数据较为吃力。Ray 提供了极其简单的分布式原语，能让我们用几行代码将 MinHash 计算并行化到多核 CPU 甚至集群上。
*   **质量层：KenLM**
    *   *决策理由：* 这是一个轻量级的 N-gram 语言模型库。在 GPT-3 和 CCNet 的论文中，均使用 KenLM 的困惑度（Perplexity）作为衡量文本自然度的核心指标。

### 3. Step-by-Step 实战 (Implementation)

#### 阶段一：从 HTML 泥潭中提取正文 (Extraction & Cleaning)

原始 WARC 文件包含大量非文本噪声。我们首先使用 `warcio` 流式读取压缩包，并利用 `trafilatura` 提取核心内容。随后，应用启发式规则进行初筛。

**核心代码：解析与启发式清洗**

```python
import trafilatura
from warcio.archiveiterator import ArchiveIterator

# 1. 提取逻辑 (来自 2_process_warc.py)
def extract_text(content_stream):
    # no_fallback=False 保证速度，include_tables=False 去除干扰
    text = trafilatura.extract(
        content_stream, 
        include_comments=False, 
        include_tables=False
    )
    return text

# 2. 启发式清洗规则 (来自 3_clean_data.py)
def is_high_quality(text):
    # 规则 A: 长度与平均词长过滤
    words = text.split()
    mean_word_len = sum(len(w) for w in words) / len(words)
    if mean_word_len > 15: # 词太长通常是乱码或代码
        return False
        
    # 规则 B: 符号密度 (Symbol Ratio)
    code_symbols = {'{', '}', '[', ']', '<', '>', '\\'}
    symbol_count = sum(1 for char in text if char in code_symbols)
    if symbol_count / len(text) > 0.1: # 代码符号过多
        return False
        
    # 规则 C: 黑名单关键词
    bad_phrases = ["lorem ipsum", "enable cookies", "403 forbidden"]
    if any(p in text.lower() for p in bad_phrases):
        return False
        
    return True
```

#### 阶段二：分布式 MinHash 去重 (Deduplication)

互联网上存在大量重复内容（转载、镜像）。我们使用 Ray 实现并行的 MinHash 计算，结合 LSH（局部敏感哈希）将 $O(N^2)$ 的复杂度降低到 $O(N)$。

**核心代码：Ray 并行计算签名**

```python
import ray
from datasketch import MinHash

# 初始化 Ray 利用所有 CPU 核心
ray.init()

@ray.remote
def process_batch(lines, num_perm=128):
    """Ray Worker: 并行计算一批数据的 MinHash 指纹"""
    results = []
    for line in lines:
        item = json.loads(line)
        m = MinHash(num_perm=num_perm)
        # Shingling: 按单词更新哈希
        for w in item['text'].split():
            m.update(w.encode('utf8'))
        results.append((item['url'], m, item['text']))
    return results

# 主流程：Map-Reduce 风格
# Map: 分发计算任务
futures = [process_batch.remote(batch) for batch in batches]
# Reduce: 收集结果并构建 LSH 索引
results = ray.get(futures)
# ...后续接 MinHashLSH 索引构建...
```

#### 阶段三：语言识别与困惑度过滤 (Quality Filtering)

清洗后的数据混合了多种语言且质量参差不齐。我们先用 FastText 分流语言，再用 KenLM 计算困惑度（Perplexity）。困惑度越低，代表句子越通顺、越像“人话”。

**核心代码：KenLM 评分**

```python
import kenlm
import fasttext

# 1. 语言分流 (来自 5_split_lang.py)
lid_model = fasttext.load_model('lid.176.ftz')
def predict_lang(text):
    # k=1 取概率最高的语言
    predictions = lid_model.predict(text, k=1)
    return predictions[0][0].replace('__label__', '')

# 2. 困惑度过滤 (来自 6_quality_filter.py)
kenlm_model = kenlm.Model('en.arpa.bin')
PERPLEXITY_THRESHOLD = -6.0  # 经验阈值：低于此值通常为低质量文本

def filter_by_perplexity(text):
    words = text.split()
    # 计算归一化得分 (Log Score / Length)
    log_score = kenlm_model.score(text)
    normalized_score = log_score / len(words)
    
    if normalized_score > PERPLEXITY_THRESHOLD:
        return True, normalized_score
    return False, normalized_score
```

### 4. 效果展示 (Showcase)

经过这一套 Pipeline 处理，数据的面貌发生了根本性变化：

**Case 1: 导航栏噪声 (已去除)**
> *Raw:* "Home | About Us | Contact | Enable Cookies | Copyright 2023..."
> *Result:* **[已丢弃]** (触发短文本和关键词黑名单规则)

**Case 2: 代码片段 (已去除)**
> *Raw:* "function(x) { return x > 0 ? true : false; } var a = [1,2,3];"
> *Result:* **[已丢弃]** (触发符号密度 > 10% 规则)

**Case 3: 高质量正文 (保留并评分)**
> *Raw:* "The James Webb Space Telescope has captured a new image of the Pillars of Creation..."
> *Result:* **[保留]**
> *KenLM Score:* -4.82 (优于阈值 -6.0)

**数据统计：**
在单次 Crawl 的采样测试中：
*   **原始记录：** 10,000 条
*   **提取有效文本：** 约 4,500 条 (HTML 解析损耗)
*   **清洗后剩余：** 约 2,800 条 (启发式过滤损耗)
*   **去重后剩余：** 约 2,100 条 (重复率约 25%)
*   **最终高质量集：** 约 1,800 条 (KenLM 过滤)

### 5. 成本与优化 (Cost & Optimization)

*   **资源消耗：**
    *   **计算：** 本项目代码在单机 16核 CPU、64G 内存环境下，处理 1GB WARC 数据耗时约 5-8 分钟。
    *   **瓶颈：** `MinHashLSH` 的索引构建目前是单线程的（在 `4_deduplicate.py` 中），且完全依赖内存。

*   **扩展性思考 (Scaling to TBs)：**
    如果数据量扩大到 PB 级别（如真实的 C4 数据集），当前架构需要升级：
    1.  **LSH 存储：** 不能再使用内存版 `MinHashLSH`，需改用 Redis 或 Cassandra 存储哈希桶。
    2.  **并行策略：** 将 Ray 任务从“单机多核”扩展到“多机集群”。
    3.  **IO 优化：** 数据读取需从本地文件系统迁移至 S3，并使用 PyArrow 进行流式列存处理。



---

## 项目二：垂直领域专家 SFT (法律)

> **场景**：基于非结构化 PDF 文档构建行业专家微调数据。
> **核心技术**：Self-Instruct 构造指令、CoT 推理增强、数据多样性平衡。
> **输出**：`domain_expert.jsonl` 指令微调集。

### 1. 项目背景 (Project Brief)

- **任务定义：** 从非结构化的法律法规 PDF 文档中提取知识，利用大模型 Self-Instruct 技术构建具备“思维链（CoT）”能力的垂直领域指令微调数据集。
- **输入与输出：**
  - **Input:** 原始 PDF 文档（如《民法典》、《刑法》等，包含页眉、页脚、水印干扰）。
  - **Output:** `domain_expert.jsonl`，包含 Instruction（用户指令）与 Output（包含思考过程的专家回复）。
- **难点分析：**
  1. **PDF 噪音清洗**：法律文档中常见的引用标号（如 `[1]`）、被换行符切断的中文词汇（如`法 律`）、以及嵌入正文的页码（如 `- 195 -`）极难清理。
  2. **数据单一性**：简单的“法条解释”不足以训练专家模型，需要构造复杂的案情分析、文书写作等多样化任务。
  3. **推理能力缺失**：普通 QA 对缺乏逻辑推导，需强制模型生成 CoT（Chain of Thought）。

### 2. 架构设计 (Architecture Design)

**数据流水线图：**

![图2：构建垂直领域专家 SFT](../images/实战项目/图2_构建垂直领域专家SFT数据流水线图.png)


- **技术栈清单：**
  - **PDF 解析 (pdfplumber)**：相比 PyPDF2，pdfplumber 提供更精准的 Bounding Box 控制，方便切除页眉页脚（代码中设定切除上下 5%）。
  - **清洗引擎 (Regex)**：针对中文断词和引用标号的“胶水代码”，是提升数据质量的关键。
  - **生成模型 (DeepSeek-V3)**：利用其强大的逻辑推理能力和低成本 API 进行 Self-Instruct 数据合成。
  - **编排逻辑 (Python)**：使用加权轮盘赌（Weighted Roulette Wheel）算法实现任务类型的多样性平衡。

### 3. Step-by-Step 实战 (Implementation)

#### 阶段一：数据获取与智能清洗 (The Dirty Work)

PDF 提取最大的痛点在于格式错乱。代码 `data_processing.py` 中的 `clean_text_smart` 函数是处理这一问题的核心。我们重点解决了“中文假性空格”和“嵌入式页码”问题。

**关键代码逻辑：**

```python
def clean_text_smart(text):
    """
    清洗核心逻辑：修复 PDF 解析带来的格式损伤
    """
    # 1. 去除参考文献引用 (如 [1], [1-3])
    text = re.sub(r'\[\s*\d+(?:[-–,]\d+)*\s*\]', '', text)

    # 2. 去除嵌在文本中间的页码 (如 "- 195 -")
    # 使用 Lookahead 断言防止误删正文中的编号
    text = re.sub(r'(?:^|\s|\\n)[-—–－]\s*\d+\s*[-—–－](?=\s|\\n|$)', ' ', text)

    # 3. 修复中文断词 (核心修复)
    # 场景：PDF中 "法 律 规 定" 会被识别为带空格，需合并
    pattern_broken_zh = r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])'
    # 执行两次以处理连续断词
    text = re.sub(pattern_broken_zh, r'\1\2', text)
    text = re.sub(pattern_broken_zh, r'\1\2', text) 
    
    return text.strip()
```

#### 阶段二：多样化指令合成 (Diversity & CoT)

为了避免模型变成只会背法条的书呆子，我们在 `generate_instructions.py` 中设计了**任务池（Task Pool）**和**概率采样**机制，强制模型生成三种不同类型的任务。

**多样性平衡策略：**

```python
# 任务权重配置 (实现数据分布控制)
TASK_POOL = [
    # 任务A: 复杂案情分析 (侧重推理) - 权重 60%
    ("case_analysis", PROMPT_CASE_ANALYSIS, 0.6),
    # 任务B: 法律文书起草 (侧重生成) - 权重 20%
    ("doc_drafting", PROMPT_DOCUMENT_DRAFTING, 0.2),
    # 任务C: 法律概念辨析 (侧重知识) - 权重 20%
    ("concept_explain", PROMPT_CONCEPT_EXPLAIN, 0.2)
]

# 轮盘赌选择逻辑
rand = random.random()
cumulative_prob = 0
for name, tpl, prob in TASK_POOL:
    cumulative_prob += prob
    if rand <= cumulative_prob:
        # 命中任务类型，使用对应 Prompt
        selected_prompt_tpl = tpl
        break
```

#### 阶段三：格式化与 CoT 增强

在 Prompt 中，我们明确要求模型返回 JSON，并强制包含“思考过程”。在后处理阶段，我们将隐式的思维链显性化，拼接成最终的训练目标格式。

**CoT 格式化逻辑：**

```python
# 解析模型返回的 JSON，强制构建思维链格式
if isinstance(raw_output, dict):
    thought = raw_output.get("思考过程") or raw_output.get("analysis")
    answer = raw_output.get("法律建议") or raw_output.get("conclusion")
    
    # 将思考过程显式写入 Output，训练模型学会"先想后说"
    formatted_output = f"#### 🧠 思考过程\n{thought}\n\n#### 📝 专家建议\n{answer}"
```

### 4. 效果展示 (Showcase)

经过处理，原始枯燥的法条被转化为具备深度推理的问答对。以下是 `domain_expert.jsonl` 中的单条数据样本：

**Source (原始法条片段):**
> ...第一百五十条 一方以欺诈手段，使对方在违背真实意思的情况下实施的民事法律行为，受欺诈方有权请求人民法院或者仲裁机构予以撤销...

**Generated SFT Data (JSONL Entry):**

| 字段 | 内容示例 |
| :--- | :--- |
| **Instruction** | 我刚才签合同的时候被对方骗了，他说那个古董是真品，结果是假的。我现在该怎么办？能直接不认账吗？ |
| **Output** | **#### 🧠 思考过程**<br>1. **分析法律关系**：用户在对方欺诈下签订了合同（实施民事法律行为）。<br>2. **适用法条**：根据《民法典》第一百五十条，这是“可撤销”的民事法律行为，而非直接“无效”或“不认账”。<br>3. **操作路径**：受欺诈方需行使撤销权，且必须通过法院或仲裁机构进行。<br><br>**#### 📝 专家建议**<br>您好，根据您描述的情况，您在违背真实意思的情况下签订了合同，这属于**可撤销的民事法律行为**。您不能直接“不认账”或单方面撕毁合同，而是建议采取以下措施：<br>1. **行使撤销权**：您可以向人民法院提起诉讼或向仲裁机构申请仲裁，请求撤销该合同。<br>2. ... |

**数据分布分析：**
- **案情分析 (60%)**：提升模型处理复杂逻辑的能力。
- **文书起草 (20%)**：提升模型的专业写作能力。
- **概念解释 (20%)**：巩固基础领域知识。

### 5. 成本与优化 (Cost & Optimization)

- **资源消耗：**
  - **API 成本**：使用 DeepSeek-V3，生成 1000 条高质量 CoT 数据约为 $0.5 - $1.0（输入输出 token 较长）。
  - **时间成本**：单线程处理约 2秒/条。
- **扩展性思考：**
  - **并发加速**：当前代码为单线程（`time.sleep`），生产环境应使用 `asyncio` + `Semaphore` 实现并发请求，效率可提升 10-20 倍。
  - **质量控制**：目前仅依赖 Prompt 约束，建议增加一步“Reward Model 打分”或“规则过滤器”，剔除生成过短或 JSON 解析失败的样本。

---
  

## 项目三：构建 LLaVA 多模态指令集

> **适用范围**：多模态大模型（LMM）开发、数据工程、视觉指令微调（Visual Instruction Tuning）

#### 1. 项目背景 (Project Brief)

- **任务定义：**
  构建一个高质量的视觉指令微调数据集，支持单图问答（Visual QA）、物体定位（Grounding）以及多图上下文推理（Interleaved Image-Text），用于训练像 LLaVA 或 Qwen-VL 这样的多模态模型。

- **输入与输出：**
  - **Input:** 
    - 原始图片库 (`.jpg` / `.png`)
    - 结构化标注数据（如 COCO 格式的 `instances.json`，包含 Bbox 坐标）
  - **Output:** 
    - 符合 LLaVA 训练标准的 JSON 文件（包含 `image`, `conversations` 字段）。
    - 经过坐标归一化和格式对齐的 Grounding 数据。

- **难点分析：**
  1.  **坐标系对齐（Coordinate Alignment）：** 原始检测数据的坐标通常是像素绝对值（x, y, w, h），而 LLaVA 模型要求归一化到 `[0-1000]` 区间且顺序为 `[ymin, xmin, ymax, xmax]`，一旦算错，模型将出现严重的“幻觉”。
  2.  **多图逻辑构建：** 传统的 Image-Caption 数据是一图一文，构建“多图交错”对话需要构造合理的对比性 Prompt，诱导模型理解图像间的关联。

#### 2. 架构设计 (Architecture Design)

- **数据流水线图：**
![图3：构建LLaVA多模态](../images/实战项目/图3_构建LLaVA多模态指令集数据流水线图.png)



- **技术栈清单：**
  - **OpenAI Compatible API (SiliconFlow/Qwen):** 用于生成高质量的图文描述和多图对比逻辑，利用大模型的 Reasoning 能力构造对话。
  - **Python & OpenCV:** 核心胶水语言。OpenCV 必不可少，用于读取图像真实尺寸（H, W）以进行坐标归一化，并用于可视化的“画框验证”。
  - **JSON:** LLaVA 标准数据交换格式。

#### 3. Step-by-Step 实战 (Implementation)

##### 阶段一：多图交错数据生成 (Interleaved Data Generation)
为了让模型学会“对比”两张图片，我们利用 API 动态输入多张图像并请求对比。

**关键逻辑：** 利用 VLM API 构造多图输入的 Prompt。

```python
# 摘自 interleaved.py
def generate_comparison(img1_path, img2_path):
    # 构造 Prompt：要求多图对比
    prompt = "Here are two images. Please briefly compare them..."
    
    # 构建多图 Payload
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"...{img1_path}..."}}, # 图1
                {"type": "image_url", "image_url": {"url": f"...{img2_path}..."}}  # 图2
            ]
        }
    ]
    # ... 发送请求并解析结果 ...
```

##### 阶段二：核心处理——Bounding Box 对齐 (Alignment)
这是本项目最核心的数学部分。COCO 数据集使用 `[x_topleft, y_topleft, width, height]`，而 LLaVA 需要 `[ymin, xmin, ymax, xmax]` 且数值需归一化为 0-1000 的整数。

**关键函数：** 坐标归一化转换

```python
# 摘自 alignment.py
def convert_bbox(bbox, width, height):
    # COCO 原始输入: x, y, w, h
    x, y, w, h = bbox
    
    # 转换为 LLaVA 格式: [ymin, xmin, ymax, xmax] 并归一化到 0-1000
    # 必须使用 max/min 截断，防止浮点误差导致越界
    xmin = int((x / width) * 1000)
    ymin = int((y / height) * 1000)
    xmax = int((x + w) / width * 1000)
    ymax = int((y + h) / height * 1000)
    
    return [
        max(0, min(1000, ymin)),
        max(0, min(1000, xmin)),
        max(0, min(1000, ymax)),
        max(0, min(1000, xmax))
    ]
```

##### 阶段三：格式化与验证 (Verification)
数据生成后，绝不能直接送入训练。必须通过**可视化反向验证**。如果我们在图片上画出的框是歪的，训练出来的模型一定是废的。

**验证逻辑：** 解析生成的 JSON，将 `[0-1000]` 坐标还原回像素坐标并绘图。

```python
# 摘自 visualize_bbox.py
def draw_bbox(image, bbox, label, color):
    h, w, _ = image.shape
    ymin, xmin, ymax, xmax = bbox # 读取 LLaVA 格式
    
    # 还原为像素坐标用于画图
    x1 = int(xmin / 1000 * w)
    y1 = int(ymin / 1000 * h)
    x2 = int(xmax / 1000 * w)
    y2 = int(ymax / 1000 * h)
    
    # OpenCV 画框
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    # ...
```

#### 4. 效果展示 (Showcase)

**1. 数据结构示例：**
最终生成的 `llava_instruct.json` 呈现如下标准结构，可以直接被 Training Pipeline 读取：

```json
{
  "id": "1296_laptop",
  "image": "000000001296.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "Where is the laptop in the image? <image>"
    },
    {
      "from": "qwen",
      "value": "The laptop is located at [350, 201, 680, 505]."
    }
  ]
}
```

**2. 可视化验证报告：**
运行 `visualize_bbox.py` 后，在 `viz_debug` 目录下生成的验证图。如果框精准地套住了物体（如下图所示），说明数据流水线逻辑正确。

 **效果图生成 **

![图4：效果图生成](../images/实战项目/图4_viz_000000001490.jpg)


#### 5. 成本与优化 (Cost & Optimization)

- **资源消耗：**
  - **API 成本：** `interleaved.py` 依赖外部 LLM API。生成 10,000 条多图对比数据，按照 $0.5/1M Tokens 计算，成本约为 $20-$30。
  - **计算耗时：** `alignment.py` 是纯 CPU 计算，处理 COCO 验证集（5k 张图）仅需数秒。

- **扩展性思考：**
  - **并发处理：** 当处理百万级图像（如 Objects365）时，单线程读取图片获取 `(h, w)` 会成为瓶颈。可以引入 `multiprocessing` 库，开启 16 个进程并行读取和转换。
  - **负样本挖掘：** 当前代码只生成了“物体在哪里”的正样本。为了增强模型鲁棒性，需要扩展代码生成“图片里有大象吗？-> No”这类负样本数据。


  ---
  

## 项目四：合成数学/代码教科书 


> **场景**：提升小模型的逻辑推理能力。
>
> **核心技术**：Evol-Instruct 进化策略、Python 代码执行沙箱 (Sandbox) 验证、PoT (Program of Thought) 数据格式化。
>
> **输出**：经过验证的高质量合成推理数据集。

### 1. 项目背景 (Project Brief)

*   **任务定义：** 构建一个高质量的“思维程序”（Program of Thought, PoT）数据集。我们将利用大模型（DeepSeek-V3）将简单的数学问题“进化”为复杂应用题，并生成相应的 Python 代码解法，最后通过代码执行沙箱验证答案的正确性。
*   **输入与输出：**
    *   **Input:** 基础数学数据集（如 GSM8K, MBPP）的原始 JSONL 文件。
    *   **Output:** 包含 `question`（进化后的问题）、`thought_process`（代码解题思路）、`execution_output`（执行结果）的清洗版 JSONL 数据集。
*   **难点分析：** 本项目最大的难点在于**“幻觉消除”**。大模型生成的代码经常看似正确但无法运行（语法错误或逻辑漏洞）。我们需要构建一个自动化的“沙箱（Sandbox）”来清洗掉无法执行的样本，确保“教科书”的严谨性。

### 2. 架构设计 (Architecture Design)

### 数据流水线图
![图5：合成数学/代码教科书](../images/实战项目/图5_合成数学代码教科书数据流水线图.png)

### 技术栈清单

*   **数据源 (Source):** `HuggingFace Datasets` (获取 GSM8K/MBPP)。
*   **生成引擎 (Generator):** `DeepSeek-V3` (via SiliconFlow API) —— 性价比极高的代码生成模型。
*   **编排逻辑 (Orchestration):** Python 脚本 (Evol-Instruct 策略)。
*   **验证环境 (Validator):** Python `subprocess` (本地沙箱) —— *生产环境建议使用 Docker 或 MicroVM。*

### 3. Step-by-Step 实战 (Implementation)

### 阶段一：种子数据获取 (Seed Preparation)

一切始于高质量的种子。我们不需要海量数据，只需要具有代表性的逻辑内核。

**关键动作：**
1.  下载 GSM8K（数学）和 MBPP（代码）数据。
2.  从中随机采样作为“进化”的基石。

**胶水代码 (Data Sampler):**
*代码引用自 `download_data.py` 与 `sampler.py`*

```python
# 核心逻辑：从海量数据中抽取种子，只保留 Question 字段
# 原始的 Answer 被丢弃，因为我们要让模型重新生成基于代码的解答
sampled = random.sample(data, SAMPLE_SIZE)
for entry in sampled:
    seed_entry = {
        "id": random.randint(1000, 9999), 
        "seed_question": entry['question'], # 仅保留问题
        "original_answer": entry['answer']  # 仅作参考
    }
```

### 阶段二：Evol-Instruct 与 PoT 生成 (Evolution & Generation)

这是本项目的核心。我们不能只做简单的“问答对”，我们需要让模型像人类专家一样思考。

**流程逻辑：**
1.  **Evol (进化):** 将简单问题（如“1+1=?”）重写为复杂场景（如“小明有1个苹果，受到通货膨胀影响...”），增加约束条件。
2.  **PoT (代码解题):** 强制模型写 Python 代码来解决问题，而不是直接输出文本答案。

**核心 Prompts (Prompt Engineering):**
*代码引用自 `evol.py`*

```python
def get_evol_prompt(seed_question):
    return f"""
    你是一个专业的数学竞赛命题专家。请将下面这个基础数学问题重写为一个更复杂、逻辑更严密的问题。
    【原题】: {seed_question}
    【重写要求】:
    1. 增加约束条件：引入更多变量或限制。
    2. 增加推理深度：不要直接给出数字，让数字之间存在逻辑关联。
    3. 场景化：将抽象的数字放入具体的物理或商业场景中。
    ...
    """

def get_pot_prompt(evolved_question):
    return f"""
    请编写一段 Python 代码来解决以下数学问题。
    ...
    1. 编写一个名为 `solve()` 的函数。
    2. 在代码注释中清晰地写出推理步骤。
    3. `solve()` 函数必须返回最终的数值答案。
    ...
    """
```

### 阶段三：沙箱验证 (Sandbox Verification)

生成的数据有大量“坏死”样本（Syntax Error, Timeout, Loop）。必须通过执行验证。

**沙箱逻辑：**
1.  使用正则提取 Markdown 中的代码块。
2.  开启子进程 (`subprocess`) 执行代码。
3.  **关键：** 设置 `timeout` 防止死循环卡死数据流水线。

**验证脚本:**
*代码引用自 `sandbox.py`*

```python
def execute_code(code, timeout=5):
    """
    执行 Python 代码并获取输出。
    """
    try:
        # 使用 subprocess 启动独立进程
        result = subprocess.run(
            ['python3', '-c', code],
            capture_output=True, # 捕获 stdout
            text=True,
            timeout=timeout # 必须设置超时！
        )
        
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, f"Error: {result.stderr.strip()}"
            
    except subprocess.TimeoutExpired:
        return False, "Error: Execution Timed Out"
```

### 4. 效果展示 (Showcase)

经过沙箱清洗后，我们得到了 `verified_textbook.jsonl`。这是一本教科书级的合成数据。

**数据样本对比：**

| 阶段 | 内容示例 |
| :--- | :--- |
| **原始种子** | 珍妮有5个苹果，吃了2个，还剩几个？ |
| **Evol 进化** | 珍妮经营一家水果店，库存5箱苹果（每箱12个）。周一她卖出了库存的40%，且由于存储不当损耗了2个单品。请计算剩余可售卖的苹果具体数量。 |
| **PoT 解法** | `def solve(): total = 5 * 12; sold = total * 0.4; ... return remaining` |
| **执行结果** | `34` (验证通过，存入数据集) |

**验证统计：**
通常，经过 Evol 后的代码一次通过率（Pass@1）在 **60%-80%** 之间。被沙箱剔除的 20% 错误数据正是污染模型训练的元凶，**剔除它们显著提升了SFT后的模型逻辑一致性。**

### 5. 成本与优化 (Cost & Optimization)

*   **资源消耗：**
    *   **API 成本:** 每条有效数据消耗约 2 次 LLM 调用（进化+解题）。使用 DeepSeek-V3 等高性价比模型，生成 1k 条高质量教科书数据的成本可控制在 $5 以内。
    *   **时间成本:** 本地 Python 单线程执行较慢，验证 1k 条代码约需 5-10 分钟。

*   **安全性警示 (Critical):**
    *   本项目使用了 `subprocess` 本地执行代码。在处理未知来源或不可信模型生成的代码时，**存在极高风险**（如 `os.system('rm -rf /')`）。
    *   **生产级改造方案：** 必须将 `sandbox.py` 的执行环境迁移至 **Docker 容器** 或 **AWS Firecracker** 微虚拟机中，并禁用网络访问权限。

*   **扩展性思考：**
    *   如果数据量扩大到百万级，单机脚本将无法支撑。需要引入 `RabbitMQ` 或 `Kafka` 进行任务分发，构建分布式的“生成-验证”集群。


---
## 项目五：多模态 RAG 企业财报助手

> **适用范围**：Capstone Projects - 解决复杂文档（图表、表格）检索难题

### 1. 项目背景 (Project Brief)

- **任务定义：** 构建一个能够“看懂”企业年报中复杂图表与数据表格的 RAG 系统，通过视觉检索（Visual Retrieval）和多模态大模型（VLM）实现对财报的深度问答。
- **输入与输出：**
  - **Input:** PDF 格式的企业年度财务报告（包含混合排版的文本、跨页表格、趋势折线图、饼图等）。
  - **Output:** 基于图表数据趋势和具体数值的自然语言分析回答。
- **难点分析：** 
  1. **结构丢失**：传统 RAG 使用 OCR 转文字，容易丢失表格的行列对应关系，且完全无法处理不带文字说明的趋势图。
  2. **语义断层**：财报中常出现“见下图”的指代，文本与图表分离导致检索截断。
  3. **检索噪音**：目录页（Table of Contents）常包含关键词，容易误召回，挤占上下文窗口。

### 2. 架构设计 (Architecture Design)

本项目的核心理念是 **"ViR (Vision in Retrieval) + VLM (Vision Language Model)"**。我们不再将 PDF 强行转为文本，而是利用 **ColPali** 将每一页 PDF 视为一张图片进行视觉编码，直接检索视觉特征，最后将命中的图片原图喂给多模态大模型进行解读。

### 数据流水线图

![图6：多模态RAG企业财报助手](../images/实战项目/图6_多模态RAG企业财报助手数据流水线图.png)


### 技术栈清单

| 组件 | 工具/模型 | 选择理由 |
| :--- | :--- | :--- |
| **视觉检索模型** | **ColPali (v1.2)** | 当前 SOTA 的文档检索模型，基于 PaliGemma，能理解页面布局、字体大小和图表视觉特征，无需 OCR。 |
| **索引框架** | **Byaldi** | ColPali 的轻量级封装库，简化了多模态模型的张量存储和检索流程。 |
| **多模态大模型** | **Qwen2.5-VL-72B** | 阿里通义千问最新视觉模型，在图表理解（ChartQA）和文档解析（DocVQA）任务上表现极佳。 |


### 3. Step-by-Step 实战 (Implementation)

### 阶段一：视觉索引构建 (Visual Indexing)

不同于传统 RAG 的 `Chunking -> Embedding`，这里我们进行的是 `Page -> Screenshot -> Visual Embedding`。

**关键代码逻辑 (`index.py`)：**

```python
from byaldi import RAGMultiModalModel

# 1. 加载本地 ColPali 模型 (解决 HuggingFace 连接问题)
MODEL_PATH = "/path/to/models/colpali-v1_2-merged"
INDEX_NAME = "finance_report_2024"

def build_index():
    # 2. 初始化模型 (支持 load_in_4bit 降低显存需求)
    RAG = RAGMultiModalModel.from_pretrained(MODEL_PATH, verbose=1)
    
    # 3. 建立索引
    # 原理：Byaldi 会将 PDF 转为图片，计算视觉向量并存储
    RAG.index(
        input_path="annual_report_2024.pdf",
        index_name=INDEX_NAME,
        store_collection_with_index=True, # 必须存储原图引用
        overwrite=True
    )
```

**实战复盘：**
*   **Debug:** 首次运行时遇到 OOM（显存溢出）。
*   **Solution:** ColPali 完整版需要约 10GB+ 显存。显存不足时，可在 `from_pretrained` 中添加 `load_in_4bit=True` 参数。

### 阶段二：多路视觉检索 (Multi-Page Retrieval)

财报问答的一个典型坑点是：**关键词“经营结果”在目录页也会出现**。如果只检索 Top-1，很可能只拿到目录，导致模型无法回答。因此，策略上需要检索 Top-K (建议 3-5 页) 并过滤。

**关键代码逻辑 (`rag_chat.py` - Retrieval Part)：**

```python
# 加载索引
RAG = RAGMultiModalModel.from_index(INDEX_NAME)

# 增加检索页数，防止只命中目录页
RETRIEVAL_K = 4 

results = RAG.search(user_query, k=RETRIEVAL_K)

# 结果包含：page_num (页码), base64 (图片数据), score (相关性)
```

### 阶段三：多图上下文生成 (Multi-Image Generation)

我们将检索到的 K 张图片全部作为上下文喂给 VLM，利用模型的长窗口和多图处理能力进行综合分析。

**关键代码逻辑 (`rag_chat.py` - Generation Part)：**

```python
# 构建多模态 Payload
content_payload = []

# 1. System Prompt: 明确角色与抗干扰指令
content_payload.append({
    "type": "text", 
    "text": f"你是一个专业的CFO助手...其中可能包含目录页，请忽略目录，直接根据包含具体数据的页面回答问题..."
})

# 2. 注入多张图片
for res in results:
    content_payload.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{res.base64}", 
            "detail": "high" # 开启高清模式以识别小字
        }
    })

# 3. 发送给 Qwen2.5-VL-72B
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-72B-Instruct",
    messages=[{"role": "user", "content": content_payload}],
    temperature=0.1 # 保持事实性
)
```

### 4. 效果展示 (Showcase)

以下是系统针对华为 2024 财报的真实运行记录。可以看到系统成功跳过了目录页，综合了第 49、91、130、8 页的信息，对研发投入、无形资产变化进行了量化分析。

```text
>>> 请提问: 经营结果如何？

🔍 正在检索 Top-4 个相关页面...
📄 命中页码: [49] [91] [130] [8]
🚀 正在发送给大模型进行综合分析...

🤖 财报助手回答:
----------------------------------------
从提供的财报截图来看，经营结果可以从以下几个方面进行分析：

### 1. **研发与创新投入**
- **研发费用**：2024年，研发费用支出为人民币1,797亿元，占全年收入的20.8%。近十年累计投入的研发费用超过人民币12,490亿元。这表明公司在研发和创新方面持续投入大量资源，致力于技术的突破和产品的升级。
- **研发人员**：截至2024年12月31日，研发员工约11.3万名，占总员工数量的54.1%，显示出公司对研发团队的重视和对技术创新的持续投入。

### 2. **无形资产**
- **商誉及无形资产**：截图显示了商誉、软件、专利权及特许权使用费、商标使用权及其他无形资产的详细数据。从2023年到2024年，商誉从4,424百万元增加到4,496百万元...（中略）...这表明公司在无形资产方面持续增长。

### 3. **社会责任与可持续发展**
- **科技小学堂项目**：华为正式发起“科技小学堂”项目...（中略）
- **移动数字课堂项目**：华为携手伙伴在肯尼亚发起DigiTruck项目...

### 4. **董事长致辞**
- **战略聚焦与持续创新**：董事长在致辞中提到，2024年是充满挑战的一年，华为全体员工坚定信心、积极奋进，实现业务发展目标，整体经营达到预期...

综上所述，华为在2024年的经营结果表现出色，公司在研发与创新、无形资产、社会责任与可持续发展等方面均取得了显著成就。
----------------------------------------
```

### 5. 成本与优化 (Cost & Optimization)

- **资源消耗：**
  - **索引成本**：ColPali 处理速度较慢（约 0.5s/页），一份 200 页的财报索引需 2-3 分钟。
  - **推理成本**：多模态 Token 消耗巨大。一张 1024x1024 的图片约为 1000-1500 tokens。每次 Top-4 检索意味着 Input Token 至少 5000+。使用 SiliconFlow API 调用 Qwen2.5-VL-72B，单次问答成本约 0.05-0.1 元人民币。

- **优化思路：**
  1. **精度优化**： 对于超大分辨率的财务大表，可以在索引前对 PDF 页面进行“切片（Cropping）”处理，将一张大图切成 4 张小图分别建立索引，提高局部检索的清晰度。
  2. **图片裁剪**：ColPali 能够定位相关区域（Patch-level retrieval），未来可只将页面中相关的“图表区域”裁剪出来喂给大模型，大幅降低 Token 消耗。
  3. **缓存机制**：对于“营收多少”、“净利润多少”等高频固定问题，将 VLM 的解析结果缓存，避免重复进行视觉推理。
```