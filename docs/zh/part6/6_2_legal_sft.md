# 项目二：垂直领域专家 SFT (法律)

> **场景**：基于非结构化 PDF 文档构建行业专家微调数据。
> **核心技术**：Self-Instruct 构造指令、CoT 推理增强、加权轮盘赌采样。
> **输出**：`domain_expert.jsonl` 指令微调集。

### 1. 项目背景 (Project Brief)

* **任务定义：** 本项目旨在从非结构化的法律法规 PDF 文档（如《民法典》、《刑法》）中提取知识，利用大模型 Self-Instruct 技术构建具备“思维链（Chain of Thought, CoT）”能力的垂直领域指令微调数据集。
* **输入与输出：**
    * **Input:** 原始 PDF 文档（包含页眉、页脚、水印干扰、以及被换行符切断的中文词汇）。
    * **Output:** `domain_expert.jsonl`，包含 `Instruction`（用户指令）与 `Output`（包含显式思考过程的专家回复）。
* **难点分析：**
    * **“脏”数据的清洗：**：法律文档极其严谨，但 PDF 解析生成的文本往往包含“假性空格”（如`法 律`）和嵌入正文的页码（如 `- 195 -`），普通正则容易误删正文中的编号。
    * **任务单一性陷阱：**简单的“法条解释”不足以训练专家模型，必须构造复杂的案情分析、文书写作等多样化任务。
    * **推理黑盒：**普通 QA 对缺乏逻辑推导，需强制模型生成 CoT，并将其格式化为训练数据。


### 2. 架构设计 (Architecture Design)

**数据流水线图：**

![图2：构建垂直领域专家 SFT](../../images/part6/图2_构建垂直领域专家SFT数据流水线图.png)

* **技术栈清单：**

| 组件 | 选型 | 决策理由 |
| :--- | :--- | :--- |
| **PDF 解析** | `pdfplumber` | 相比 `PyPDF2` 仅提取纯文本，`pdfplumber` 提供精准的像素级 Bounding Box 控制，可精细化切除页眉、页脚（如设定上下 5% 冗余区）以减少噪声。 |
| **清洗引擎** | `Regex` (正则表达式) | 针对中文特有的断词错误、长引用标号及特殊字符编写的“胶水代码”，是确保后续模型训练数据纯净度的核心关卡。 |
| **生成模型** | `DeepSeek-V3` | 利用其卓越的逻辑推理能力与极高的 API 性价比，通过 Self-Instruct 机制进行大规模高质量合成数据（Synthetic Data）的自动化生产。 |
| **编排逻辑** | `Python` (加权轮盘赌) | 基于权重随机算法（Weighted Roulette Wheel）实现任务分配，确保在合成数据时，各类任务（如代码、翻译、推理）的分布能够达到动态平衡。 |



### 3. Step-by-Step 实战 (Implementation)

#### 阶段一：数据获取与智能清洗 (The Dirty Work)

PDF 提取最大的痛点在于格式错乱。在 `data_processing.py` 中，我们不仅要提取文本，更要修复格式损伤。

**1. 裁剪页眉页脚**
为了防止每一页的页眉（如“中华人民共和国民法典”）重复出现在正文中，我们在解析时直接裁剪了页面上下 5% 的区域。

```python
# data_processing.py 核心片段
with pdfplumber.open(file_path) as pdf:
    for page in tqdm(pdf.pages):
        # 裁剪页眉页脚 (上下各切除 5%)
        width, height = page.width, page.height
        bbox = (0, height * 0.05, width, height * 0.95)
        page_crop = page.crop(bbox=bbox) # 仅提取中间区域
        text = page_crop.extract_text()

```

**2. 正则清洗的“至暗时刻”**
最棘手的问题是**嵌入式页码**。很多 PDF 在正文中间会突然出现 `- 195 -`。如果简单使用 `-\s*\d+\s*-` 进行删除，极易误删正文中的列表编号（如 `Item - 1 - A`）。

我们在 `clean_text_smart` 函数中使用了 **Lookahead 断言** 来解决这个问题：

```python
def clean_text_smart(text):
    # ... 省略其他清洗 ...

    # [核心技巧] 去除嵌在文本中间的页码 (如 "- 195 -")
    # 逻辑：匹配前后是破折号的数字，但必须确保后面跟着空格或换行
    # (?=\s|\n|$) 是 Lookahead 断言，防止误删 "Item-1-A" 这种编号
    text = re.sub(r'(?:^|\s|\n)[-—–－]\s*\d+\s*[-—–－](?=\s|\n|$)', ' ', text)

    # [核心技巧] 修复中文断词
    # 场景：PDF中 "法 律 规 定" 会被识别为带空格，需合并
    pattern_broken_zh = r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])'
    # 执行两次以处理连续断词 (如 "A B C" -> "AB C" -> "ABC")
    text = re.sub(pattern_broken_zh, r'\1\2', text)
    text = re.sub(pattern_broken_zh, r'\1\2', text) 
    
    return text.strip()

```

#### 阶段二：多样化指令合成 (Diversity & CoT)

为了避免模型变成只会背法条的书呆子，我们在 `generate_instructions.py` 中设计了**任务池（Task Pool）和概率采样**机制。我们定义了三种任务，并赋予不同的权重。

**1. 任务类型与权重配置**

| 任务类型 | 侧重点 | 权重 | Prompt 关键指令 |
| --- | --- | --- | --- |
| **复杂案情分析** | 深度推理 | 60% | "构造一个包含多方冲突的具体案情...必须使用 CoT" |
| **法律文书起草** | 专业写作 | 20% | "起草一份基于该法条的文书（如合同条款、律师函）" |
| **法律概念辨析** | 知识普及 | 20% | "小白用户提出的概念性问题...用大白话解释" |

**2. 加权轮盘赌算法实现**

为了严格控制数据分布，我们使用累积概率算法选择 Prompt：

```python
# 任务权重配置
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
    if rand <= cumulative_prob: # 命中区间
        selected_task_name = name
        selected_prompt_tpl = tpl
        break

```

#### 阶段三：格式化与 CoT 显性化

模型生成的 JSON 往往包含隐式的推理过程。为了让微调后的模型学会“先想后说”，我们需要在后处理阶段将这些思考过程显性化，拼接成 Markdown 格式。

```python
# generate_instructions.py 后处理逻辑
if isinstance(raw_output, dict):
    # 提取思考过程 (兼容不同的 key)
    thought = raw_output.get("思考过程") or raw_output.get("analysis")
    answer = raw_output.get("法律建议") or raw_output.get("conclusion")
    
    # [关键] 强制构建思维链格式
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
| **Output** | **#### 🧠 思考过程**<br>1. **分析法律关系**：用户在对方欺诈下签订了合同（实施民事法律行为）。<br>2. **适用法条**：根据《民法典》第一百五十条，这是"可撤销"的民事法律行为，而非直接"无效"或"不认账"。<br>3. **操作路径**：受欺诈方需行使撤销权，且必须通过法院或仲裁机构进行。<br><br>**#### 📝 专家建议**<br>您好，根据您描述的情况，您在违背真实意思的情况下签订了合同，这属于**可撤销的民事法律行为**。您不能直接"不认账"或单方面撕毁合同，而是建议采取以下措施：<br>1. **行使撤销权**：您可以向人民法院提起诉讼或向仲裁机构申请仲裁，请求撤销该合同。<br>2. ... |


**数据分布分析：**
在生成的 1000 条样本中，我们验证了分布比例，基本符合预设的设定，有效避免了模型能力的偏科。

### 5. 成本与优化 (Cost & Optimization)

#### 资源消耗

* **API 成本**：使用 DeepSeek-V3，生成 1000 条高质量 CoT 数据约为 **$0.5 - $1.0**。由于 CoT 导致输出 Token 较长，Output 费用占比约为 80%。
* **时间成本**：当前代码为单线程运行，处理速度约为 **2秒/条**。生成 1万条数据约需 5.5 小时。

#### 扩展性思考：从 Demo 到生产环境

代码中目前的 `time.sleep(0.1)` 是为了防止 API Rate Limit 的临时方案。若要扩展至百万级数据处理，建议进行以下改造：

1. **并发加速 (AsyncIO)**：
将 `client.chat.completions.create` 替换为异步调用，并配合 `asyncio.Semaphore(10)` 控制并发数。实测可将吞吐量提升至 **20-50 条/秒**。
2. **质量控制 (Reward Model)**：
目前仅依赖 Prompt 约束 JSON 格式，但在大规模生成时，约有 3-5% 的样本会解析失败。建议增加一步规则过滤器，例如：过滤掉 `output` 长度小于 50 字符的样本。

3. **正则的局限性**：
对于极度复杂的 PDF（如双栏排版、包含复杂表格），`pdfplumber` + 正则可能失效。此时应考虑接入 OCR 大模型（如 Qwen-VL）进行端到端解析，虽然成本上升，但清洗质量将有质的飞跃。