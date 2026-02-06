# 项目二：垂直领域专家 SFT (法律)

> **场景**：基于非结构化 PDF 文档构建行业专家微调数据。
> **核心技术**：Self-Instruct 构造指令、CoT 推理增强、数据多样性平衡。
> **输出**：`domain_expert.jsonl` 指令微调集。

### 1. 项目背景 (Project Brief)

- **任务定义：** 从非结构化的法律法规 PDF 文档中提取知识，利用大模型 Self-Instruct 技术构建具备"思维链（CoT）"能力的垂直领域指令微调数据集。
- **输入与输出：**
  - **Input:** 原始 PDF 文档（如《民法典》、《刑法》等，包含页眉、页脚、水印干扰）。
  - **Output:** `domain_expert.jsonl`，包含 Instruction（用户指令）与 Output（包含思考过程的专家回复）。
- **难点分析：**
  1. **PDF 噪音清洗**：法律文档中常见的引用标号（如 `[1]`）、被换行符切断的中文词汇（如`法 律`）、以及嵌入正文的页码（如 `- 195 -`）极难清理。
  2. **数据单一性**：简单的"法条解释"不足以训练专家模型，需要构造复杂的案情分析、文书写作等多样化任务。
  3. **推理能力缺失**：普通 QA 对缺乏逻辑推导，需强制模型生成 CoT（Chain of Thought）。

### 2. 架构设计 (Architecture Design)

**数据流水线图：**

![图2：构建垂直领域专家 SFT](../images/chapter6/图2_构建垂直领域专家SFT数据流水线图.png)


- **技术栈清单：**
  - **PDF 解析 (pdfplumber)**：相比 PyPDF2，pdfplumber 提供更精准的 Bounding Box 控制，方便切除页眉页脚（代码中设定切除上下 5%）。
  - **清洗引擎 (Regex)**：针对中文断词和引用标号的"胶水代码"，是提升数据质量的关键。
  - **生成模型 (DeepSeek-V3)**：利用其强大的逻辑推理能力和低成本 API 进行 Self-Instruct 数据合成。
  - **编排逻辑 (Python)**：使用加权轮盘赌（Weighted Roulette Wheel）算法实现任务类型的多样性平衡。

### 3. Step-by-Step 实战 (Implementation)

#### 阶段一：数据获取与智能清洗 (The Dirty Work)

PDF 提取最大的痛点在于格式错乱。代码 `data_processing.py` 中的 `clean_text_smart` 函数是处理这一问题的核心。我们重点解决了"中文假性空格"和"嵌入式页码"问题。

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
    ("case_analysis", PROMPT_CASE_ANALYSIS, 0_6),
    # 任务B: 法律文书起草 (侧重生成) - 权重 20%
    ("doc_drafting", PROMPT_DOCUMENT_DRAFTING, 0_2),
    # 任务C: 法律概念辨析 (侧重知识) - 权重 20%
    ("concept_explain", PROMPT_CONCEPT_EXPLAIN, 0_2)
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

在 Prompt 中，我们明确要求模型返回 JSON，并强制包含"思考过程"。在后处理阶段，我们将隐式的思维链显性化，拼接成最终的训练目标格式。

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
| **Output** | **#### 🧠 思考过程**<br>1. **分析法律关系**：用户在对方欺诈下签订了合同（实施民事法律行为）。<br>2. **适用法条**：根据《民法典》第一百五十条，这是"可撤销"的民事法律行为，而非直接"无效"或"不认账"。<br>3. **操作路径**：受欺诈方需行使撤销权，且必须通过法院或仲裁机构进行。<br><br>**#### 📝 专家建议**<br>您好，根据您描述的情况，您在违背真实意思的情况下签订了合同，这属于**可撤销的民事法律行为**。您不能直接"不认账"或单方面撕毁合同，而是建议采取以下措施：<br>1. **行使撤销权**：您可以向人民法院提起诉讼或向仲裁机构申请仲裁，请求撤销该合同。<br>2. ... |

**数据分布分析：**
- **案情分析 (60%)**：提升模型处理复杂逻辑的能力。
- **文书起草 (20%)**：提升模型的专业写作能力。
- **概念解释 (20%)**：巩固基础领域知识。

### 5. 成本与优化 (Cost & Optimization)

- **资源消耗：**
  - **API 成本**：使用 DeepSeek-V3，生成 1000 条高质量 CoT 数据约为 $0_5 - $1_0（输入输出 token 较长）。
  - **时间成本**：单线程处理约 2秒/条。
- **扩展性思考：**
  - **并发加速**：当前代码为单线程（`time.sleep`），生产环境应使用 `asyncio` + `Semaphore` 实现并发请求，效率可提升 10-20 倍。
  - **质量控制**：目前仅依赖 Prompt 约束，建议增加一步"Reward Model 打分"或"规则过滤器"，剔除生成过短或 JSON 解析失败的样本。
