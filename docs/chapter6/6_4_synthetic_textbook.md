# 项目四：合成数学/代码教科书 


> **场景**：提升小模型的逻辑推理能力。
>
> **核心技术**：Evol-Instruct 进化策略、Python 代码执行沙箱 (Sandbox) 验证、PoT (Program of Thought) 数据格式化。
>
> **输出**：经过验证的高质量合成推理数据集。

### 1. 项目背景 (Project Brief)

*   **任务定义：** 构建一个高质量的"思维程序"（Program of Thought, PoT）数据集。我们将利用大模型（DeepSeek-V3）将简单的数学问题"进化"为复杂应用题，并生成相应的 Python 代码解法，最后通过代码执行沙箱验证答案的正确性。
*   **输入与输出：**
    *   **Input:** 基础数学数据集（如 GSM8K, MBPP）的原始 JSONL 文件。
    *   **Output:** 包含 `question`（进化后的问题）、`thought_process`（代码解题思路）、`execution_output`（执行结果）的清洗版 JSONL 数据集。
*   **难点分析：** 本项目最大的难点在于**"幻觉消除"**。大模型生成的代码经常看似正确但无法运行（语法错误或逻辑漏洞）。我们需要构建一个自动化的"沙箱（Sandbox）"来清洗掉无法执行的样本，确保"教科书"的严谨性。

### 2. 架构设计 (Architecture Design)

### 数据流水线图
![图5：合成数学/代码教科书](../images/chapter6/图5_合成数学代码教科书数据流水线图.png)

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
2.  从中随机采样作为"进化"的基石。

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

这是本项目的核心。我们不能只做简单的"问答对"，我们需要让模型像人类专家一样思考。

**流程逻辑：**
1.  **Evol (进化):** 将简单问题（如"1+1=?"）重写为复杂场景（如"小明有1个苹果，受到通货膨胀影响..."），增加约束条件。
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

生成的数据有大量"坏死"样本（Syntax Error, Timeout, Loop）。必须通过执行验证。

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
| **PoT 解法** | `def solve(): total = 5 * 12; sold = total * 0_4; ... return remaining` |
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
    *   如果数据量扩大到百万级，单机脚本将无法支撑。需要引入 `RabbitMQ` 或 `Kafka` 进行任务分发，构建分布式的"生成-验证"集群。
