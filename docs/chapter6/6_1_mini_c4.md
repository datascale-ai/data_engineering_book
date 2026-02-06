# 项目一：构建"Mini-C4"预训练集

### 1. 项目背景 (Project Brief)

*   **任务定义：** 构建一个微缩版 C4 (Colossal Clean Crawled Corpus) 数据集流水线。我们的目标是将杂乱无章的原始网页数据（Common Crawl）转化为低噪、去重、高质量的纯文本数据，可直接用于大模型预训练。
*   **输入与输出：**
    *   **Input:** Common Crawl 的原始 WARC 压缩包（包含 HTTP 响应头、HTML 源码、乱码等）。
    *   **Output:** 分类分级后的 JSONL 文件（如 `data_en.jsonl`, `final_data.jsonl`），包含纯净文本及其质量评分。
*   **难点分析：**
    *   **信噪比极低：** 原始网页中 90% 以上是导航栏、广告、JavaScript 代码和无意义的占位符。
    *   **计算密集：** 在大规模语料中进行两两比对去重（Deduplication）极其消耗资源。
    *   **质量量化：** 如何让机器自动判断一句话是"人类高质量语言"还是"机器生成的垃圾"？

### 2. 架构设计 (Architecture Design)

为了处理非结构化的 Web 数据，我们设计了如下的漏斗型（Funnel）处理架构：

**数据流水线图：**

![图1：构建"Mini-C4"预训练集数据流水线图](../images/chapter6/图1_构建Mini_C4预训练集数据流水线图.png)
<!-- ![图1：构建"Mini-C4"预训练集数据流水线图](images/实战项目/图1_构建Mini_C4预训练集数据流水线图.png) -->

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
    if symbol_count / len(text) > 0_1: # 代码符号过多
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

清洗后的数据混合了多种语言且质量参差不齐。我们先用 FastText 分流语言，再用 KenLM 计算困惑度（Perplexity）。困惑度越低，代表句子越通顺、越像"人话"。

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
PERPLEXITY_THRESHOLD = -6_0  # 经验阈值：低于此值通常为低质量文本

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
> *KenLM Score:* -4_82 (优于阈值 -6_0)

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
    2.  **并行策略：** 将 Ray 任务从"单机多核"扩展到"多机集群"。
    3.  **IO 优化：** 数据读取需从本地文件系统迁移至 S3，并使用 PyArrow 进行流式列存处理。
