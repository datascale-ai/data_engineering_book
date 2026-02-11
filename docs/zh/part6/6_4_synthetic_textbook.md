# 项目四：合成数学/代码教科书

> **场景**：提升小模型（SLM）的逻辑推理与数学解题能力。
> **核心技术**：Evol-Instruct 进化策略、PoT (Program of Thought) 数据格式化、Python 代码执行沙箱 (Sandbox)。
> **输出**：一本经过代码执行验证的、无幻觉的高质量合成推理习题集。

### 1. 项目背景 (Project Brief)


* **任务定义：** 构建一个高质量的"思维程序"（Program of Thought, PoT）数据集。我们将利用大模型（DeepSeek-V3）将简单的数学种子问题"进化"为复杂的应用场景题，并强制模型生成 Python 代码来解题，最后通过执行代码来验证答案的正确性。
* **输入与输出：**
    * **Input:** 基础数学数据集（GSM8K）的原始 JSONL 文件。
    * **Output:** 清洗后的 `verified_textbook.jsonl`，包含 `question`（进化后的问题）、`thought_process`（代码解题思路）、`execution_output`（执行结果）。

* **难点分析：**
    * **幻觉消除：** 大模型生成的代码经常看似逻辑通顺，实则存在语法错误或逻辑漏洞（如除以零、变量未定义）。
    * **去重复杂度：** 在海量文本中找出“语义重复”而非“完全一致”的文档（Fuzzy Deduplication），计算量呈指数级增长。
    * **复杂性控制：** 如何确保进化后的问题既增加了难度，又保持了数学上的可解性。


### 2. 架构设计 (Architecture Design)

### 数据流水线图
![图5：合成数学/代码教科书](../../images/part6/图5_合成数学代码教科书数据流水线图.png)
我们将整个流程划分为三个核心阶段：**种子采样 (Sampler)** -> **进化与生成 (Evol-Generator)** -> **沙箱验证 (Sandbox-Validator)**。
**技术栈清单：**

* **数据源:** `GSM8K` (逻辑种子)。
* **生成引擎:** `DeepSeek-V3` (via SiliconFlow API) —— 选择理由：在代码生成和数学推理上表现优异，且 API 成本极低。
* **编排逻辑:** Python 原生脚本 —— 轻量级任务编排。
* **验证环境:** Python `subprocess` —— 本地沙箱环境，用于捕捉执行错误和超时。

### 3. Step-by-Step 实战 (Implementation)

#### 阶段一：种子数据获取 (Seed Preparation)

一切始于高质量的种子。我们不需要海量数据，只需要具有代表性的逻辑内核。

**关键动作：** 从 GSM8K 训练集中随机抽取样本，清洗掉原有的思维链（CoT），只保留问题本身作为"种子"。

**核心代码 (Sampler):**

```python
# 引用自 sampler.py
def sample_data():
    # ... (读取原始数据)
    # 随机打乱并抽取 SAMPLE_SIZE 条
    sampled = random.sample(data, SAMPLE_SIZE)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in sampled:
            # 我们只需要 question 作为进化种子，丢弃原始 answer
            seed_entry = {
                "id": random.randint(1000, 9999), 
                "seed_question": entry['question'],
                "original_answer": entry['answer']
            }
            f.write(json.dumps(seed_entry, ensure_ascii=False) + '\n')

```

*代码解析：我们特意丢弃了原始答案，因为进化后的问题数值可能会发生变化，旧答案不再适用。*

#### 阶段二：Evol-Instruct 与 PoT 生成 (Evolution & Generation)

这是本项目的核心。我们通过两次 API 调用完成从"简单问题"到"复杂代码解法"的蜕变。

**1. Prompt Engineering (进化指令):**
我们需要模型扮演"数学竞赛命题专家"，并明确要求"增加约束条件"、"场景化"和"增加推理深度"。

```python
# 引用自 evol.py
def get_evol_prompt(seed_question):
    return f"""
    你是一个专业的数学竞赛命题专家。请将下面这个基础数学问题重写为一个更复杂、逻辑更严密的问题。
    【原题】: {seed_question}
    【重写要求】:
    1. 增加约束条件：引入更多变量或限制。
    2. 增加推理深度：不要直接给出数字，让数字之间存在逻辑关联。
    3. 场景化：将抽象的数字放入具体的物理或商业场景中。
    4. 保持可解性：确保问题依然有明确的数学解。
    ...
    """

```

**2. API 调用与容错 (Debug 复盘):**
在实战中，我们发现长思维链生成容易导致 HTTP 超时。因此，我们在 `call_siliconflow` 函数中做了针对性优化。

```python
# 引用自 evol.py
def call_siliconflow(prompt, model=MODEL_NAME, max_retries=3):
    # ... (Header 配置)
    payload = {
        # ...
        "max_tokens": 4096, # 增加 token 上限，防止代码写一半截断
        "timeout": 180      # 关键修改：timeout 延长到 180 秒，适应长推理
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(BASE_URL, json=payload, headers=headers, timeout=180)
            if response.status_code == 200:
                # ... 成功返回
            else:
                print(f"  [API Error] {response.status_code}...")
                time.sleep(5) # 出错后指数退避
        except requests.exceptions.Timeout:
            print(f"  [Timeout] 请求超时 (超过180秒)，正在重试...")
            # ...

```

*经验总结：生成代码通常比生成普通文本慢得多，必须显式设置较长的 `timeout`，否则 requests 库默认会较快断开连接。*

#### 阶段三：沙箱验证 (Sandbox Verification)

生成的数据中约有 20%-30% 是不可执行的（死循环、依赖缺失、语法错误）。沙箱的作用就是充当过滤器。

**1. 代码提取：**
大模型输出的代码块格式不统一，需要用正则兼容 ````python` 和 ````` 两种格式。

```python
# 引用自 sandbox.py
def extract_python_code(text):
    # 优先匹配 python 标签
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match: return match.group(1)
    
    # 兜底匹配通用代码块
    pattern_generic = r"```\s*(.*?)\s*```"
    # ...

```

**2. 执行与超时控制：**
这是最危险的环节。为了防止模型生成的代码包含死循环（`while True`）卡死整个流水线，必须设置 `subprocess` 的超时时间。

```python
# 引用自 sandbox.py
def execute_code(code, timeout=5):
    try:
        # 使用 subprocess 启动一个新的 Python 进程执行代码
        result = subprocess.run(
            ['python3', '-c', code],
            capture_output=True,
            text=True,
            timeout=timeout # 核心安全配置：5秒内跑不完直接杀进程
        )
        
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, f"Error: {result.stderr.strip()}"
            
    except subprocess.TimeoutExpired:
        return False, "Error: Execution Timed Out"

```

### 4. 效果展示 (Showcase)

经过沙箱清洗后，我们得到了 `verified_textbook.jsonl`。以下是一条典型的进化样本对比：

| 维度 | 内容示例 |
| --- | --- |
| **原始种子** | 一列火车每小时行驶 60 英里，行驶 240 英里需要多久？ |
| **Evol 进化** | 一列货运列车不仅受速度限制，还受复杂的停靠计划影响。列车基础速度为 60 英里/小时，但每行驶 100 英里必须停靠 15 分钟进行安全检查。请编写程序计算行驶 240 英里所需的总分钟数。 |
| **PoT 解法** | `def solve(): distance = 240; speed = 60; travel_time = (distance/speed)*60; stops = int(distance/100); total_time = travel_time + (stops * 15); return total_time` |
| **验证结果** | `SUCCESS: 270.0` (代码逻辑正确，无语法错误，成功存入数据集) |

**数据质量统计：**
在一次 1000 条样本的跑测中，通过率为 **82%**。

* 18% 的失败原因主要包括：
* `SyntaxError`: 生成了不完整的代码块。
* `Timeout`: 暴力穷举算法导致运行超过 5 秒。
* `LogicError`: 引用了不存在的变量（幻觉）。
通过剔除这 18% 的"有毒"数据，我们保证了教科书的严谨性。

<<<<<<< HEAD:docs/part6/6_4_synthetic_textbook.md
=======
| 阶段 | 内容示例 |
| :--- | :--- |
| **原始种子** | 珍妮有5个苹果，吃了2个，还剩几个？ |
| **Evol 进化** | 珍妮经营一家水果店，库存5箱苹果（每箱12个）。周一她卖出了库存的40%，且由于存储不当损耗了2个单品。请计算剩余可售卖的苹果具体数量。 |
| **PoT 解法** | `def solve(): total = 5 * 12; sold = total * 0.4; ... return remaining` |
| **执行结果** | `34` (验证通过，存入数据集) |
>>>>>>> main:docs/zh/part6/6_4_synthetic_textbook.md


### 5. 成本与优化 (Cost & Optimization)

#### 资源消耗
* **时间成本：** 生成一条复杂数据（Evol+Code）约需 30-60 秒。
* **计算成本：** DeepSeek-V3 的价格极低，生成 1000 条高质量验证数据的 API 成本不到 $1。
* **存储成本：** 纯文本 JSONL，几乎可以忽略不计。


#### 性能瓶颈与优化
* **串行 vs 并行：** 当前 `evol.py` 使用单线程 `tqdm` 循环，处理速度较慢。
* **优化方案：** 生产环境应引入 Python 的 `ThreadPoolExecutor` 或使用 `Celery` + `RabbitMQ` 将生成任务分发到多个 Worker 节点，将吞吐量提升 10-50 倍。


#### 安全性 (Critical)
* 当前代码在本地直接运行 (`python3 -c code`)。如果模型生成恶意代码（如 `os.system("rm -rf /")`），将造成灾难性后果。
* **生产级改造：** 必须将 `sandbox.py` 封装进 **Docker 容器** 或 **gVisor** 中，并禁用容器的网络权限，实现真正的"沙箱隔离"。


