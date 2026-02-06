# 第4章：清洗与去噪

---

## 本章摘要

从互联网获取的原始数据就像未经加工的矿石，其中真正有价值的"精矿"可能只占很小的比例。本章将深入探讨预训练数据清洗的三大核心技术：启发式过滤规则用于剔除低质量文档，大规模去重技术用于消除重复内容，隐私数据清洗用于保护用户信息。掌握这些技术后，读者将能够构建工业级的数据清洗流水线，将原始网页数据转化为高质量的预训练语料。

---

## 场景引入

经过上一章的努力，你的团队成功从 Common Crawl 提取了 50TB 的中文网页文本。然而，当你随机抽样查看数据时，发现了各种令人头疼的问题：大量页面只有几个字的导航文本，完全没有实质内容；有些页面全是 JavaScript 代码或 CSS 样式的残留；某些网站的内容被重复爬取了数百次；还有大量带有用户邮箱、手机号码的敏感信息。

更糟糕的是，训练工程师告诉你，上次用未经清洗的数据训练的模型有严重的"复读机"问题——模型会反复输出相同的句子，甚至能够背诵出某些网站的完整内容。这显然是数据重复导致的。

如何系统性地解决这些问题？本章将给出完整的答案。

---

## 4_1 启发式过滤规则

启发式过滤（Heuristic Filtering）是数据清洗的第一道防线。它基于一系列可量化的规则，快速筛选出明显低质量的文档。虽然这些规则看起来简单，但在实践中能够过滤掉大部分噪声数据，是性价比极高的清洗手段。

![图4-1：数据清洗流水线](../images/chapter2/图4_1_数据清洗流水线.png)

*图4-1：数据清洗流水线架构 —— 从原始数据到清洁语料的八阶段处理流程*

### 4_1.1 语言识别

语言识别是多语言数据处理的基础步骤。对于训练中文模型而言，首先需要从 Common Crawl 的海量数据中筛选出中文内容，这就需要准确的语言识别能力。

**FastText 语言识别器**是目前最常用的工具。它由 Facebook AI Research 开发，预训练模型支持 176 种语言的识别，速度极快，准确率也相当高。FastText 提供两个预训练模型：`lid.176.bin` 是完整版本，准确率更高但体积较大（约 126MB）；`lid.176.ftz` 是压缩版本，体积小（约 917KB）但准确率略低。对于大规模数据处理，建议使用完整版本。

```python
import fasttext

# 加载语言识别模型
lang_model = fasttext.load_model('lid.176.bin')

def detect_language(text: str, min_confidence: float = 0_8) -> tuple:
    """
    识别文本语言
    
    Args:
        text: 待识别的文本
        min_confidence: 最低置信度阈值
    
    Returns:
        (语言代码, 置信度) 或 (None, 0) 如果置信度不足
    """
    # 预处理：移除换行符，截取前 1000 字符
    text = text.replace('\n', ' ')[:1000]
    
    # 预测
    predictions = lang_model.predict(text, k=1)
    lang = predictions[0][0].replace('__label__', '')
    confidence = predictions[1][0]
    
    if confidence >= min_confidence:
        return lang, confidence
    return None, confidence

def filter_by_language(documents: list, target_lang: str = 'zh') -> list:
    """过滤指定语言的文档"""
    results = []
    for doc in documents:
        lang, conf = detect_language(doc['text'])
        if lang == target_lang:
            doc['detected_lang'] = lang
            doc['lang_confidence'] = conf
            results.append(doc)
    return results
```

语言识别在实践中会遇到一些边界情况。混合语言的文档（如中英文混杂的技术博客）可能被错误分类。短文本的识别准确率较低，建议对长度不足 50 字符的文本跳过语言过滤。代码片段可能被识别为各种语言，需要结合内容类型进行判断。

### 4_1.2 文本质量评分

语言识别只能确保文档是目标语言，但无法判断内容质量。一段语法正确的垃圾广告和一篇优质的技术文章，在语言识别上可能得到相同的分数。这就需要更精细的质量评估机制。

**困惑度（Perplexity）过滤**是一种基于语言模型的质量评估方法。困惑度衡量的是语言模型对文本的"惊讶程度"——如果一段文本与模型训练数据的分布相似，困惑度就低；如果文本包含大量噪声、乱码或不自然的表达，困惑度就高。

KenLM 是计算困惑度最常用的工具。它基于 n-gram 语言模型，速度极快，适合大规模数据处理。

```python
import kenlm

class PerplexityFilter:
    def __init__(self, model_path: str, max_perplexity: float = 500):
        """
        初始化困惑度过滤器
        
        Args:
            model_path: KenLM 模型路径 (.arpa 或 .bin)
            max_perplexity: 困惑度阈值，超过此值的文档将被过滤
        """
        self.model = kenlm.Model(model_path)
        self.max_perplexity = max_perplexity
    
    def compute_perplexity(self, text: str) -> float:
        """计算文本的困惑度"""
        # KenLM 返回的是 log10 概率
        log_prob = self.model.score(text, bos=True, eos=True)
        # 转换为困惑度
        num_words = len(text.split()) + 1  # +1 for EOS
        perplexity = 10 ** (-log_prob / num_words)
        return perplexity
    
    def filter(self, documents: list) -> list:
        """过滤高困惑度文档"""
        results = []
        for doc in documents:
            ppl = self.compute_perplexity(doc['text'])
            if ppl <= self.max_perplexity:
                doc['perplexity'] = ppl
                results.append(doc)
        return results
```

困惑度阈值的设定需要根据具体数据进行调优。一般而言，高质量的新闻和百科文本困惑度在 100-200 之间，普通网页内容在 200-500 之间，低质量内容（如乱码、机器翻译）通常超过 500。建议先在小规模样本上分析困惑度分布，再确定合适的阈值。

### 4_1.3 启发式规则集

除了语言识别和困惑度过滤，还有一系列简单但有效的启发式规则，可以快速剔除明显的低质量内容。这些规则的设计来源于对大量数据的观察和经验总结。

**长度过滤**是最基本的规则。过短的文档（如只有几个词的导航文本）没有训练价值，应该直接移除。过长的文档可能需要截断或分段处理。典型的阈值设定是：最小长度 200 字符或 50 词，最大长度 100,000 字符。

**特殊字符比例**可以识别出大量噪声内容。如果一个文档中非字母数字字符的比例过高，很可能是代码残留、乱码或格式错误。类似地，数字比例过高可能表示是日志文件或数据表格。

**重复行比例**可以检测出模板化的低质量页面。如果一个文档中有大量完全相同的行（如导航栏在页面多处重复），说明内容质量较低。

**词汇多样性**衡量文档的信息丰富程度。一个只使用 10 个不同词汇的文档显然不如使用 500 个不同词汇的文档有价值。常用的指标是 Type-Token Ratio（TTR），即唯一词数与总词数的比值。

以下是一个综合的启发式过滤器实现：

```python
import re
from collections import Counter

class HeuristicFilter:
    def __init__(self, config: dict = None):
        """
        初始化启发式过滤器
        
        默认配置适用于中文预训练数据
        """
        self.config = config or {
            'min_length': 200,           # 最小字符数
            'max_length': 100000,        # 最大字符数
            'min_words': 50,             # 最小词数
            'max_special_ratio': 0_3,    # 最大特殊字符比例
            'max_digit_ratio': 0_3,      # 最大数字比例
            'max_duplicate_line_ratio': 0_3,  # 最大重复行比例
            'min_avg_word_length': 2,    # 最小平均词长
            'max_avg_word_length': 20,   # 最大平均词长
            'min_unique_word_ratio': 0_1 # 最小词汇多样性
        }
    
    def check_length(self, text: str) -> bool:
        """检查文档长度"""
        length = len(text)
        return self.config['min_length'] <= length <= self.config['max_length']
    
    def check_special_chars(self, text: str) -> bool:
        """检查特殊字符比例"""
        if len(text) == 0:
            return False
        special = len(re.findall(r'[^\w\s]', text, re.UNICODE))
        ratio = special / len(text)
        return ratio <= self.config['max_special_ratio']
    
    def check_digit_ratio(self, text: str) -> bool:
        """检查数字比例"""
        if len(text) == 0:
            return False
        digits = len(re.findall(r'\d', text))
        ratio = digits / len(text)
        return ratio <= self.config['max_digit_ratio']
    
    def check_duplicate_lines(self, text: str) -> bool:
        """检查重复行比例"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) == 0:
            return False
        unique_lines = len(set(lines))
        duplicate_ratio = 1 - (unique_lines / len(lines))
        return duplicate_ratio <= self.config['max_duplicate_line_ratio']
    
    def check_vocabulary_diversity(self, text: str) -> bool:
        """检查词汇多样性"""
        words = text.split()
        if len(words) < self.config['min_words']:
            return False
        unique_ratio = len(set(words)) / len(words)
        return unique_ratio >= self.config['min_unique_word_ratio']
    
    def filter(self, text: str) -> tuple:
        """
        应用所有过滤规则
        
        Returns:
            (是否通过, 失败原因或 None)
        """
        checks = [
            (self.check_length, 'length'),
            (self.check_special_chars, 'special_chars'),
            (self.check_digit_ratio, 'digit_ratio'),
            (self.check_duplicate_lines, 'duplicate_lines'),
            (self.check_vocabulary_diversity, 'vocabulary_diversity')
        ]
        
        for check_func, name in checks:
            if not check_func(text):
                return False, name
        
        return True, None
```

### 4_1.4 质量分层策略

在实践中，将数据简单地二分为"保留"和"丢弃"往往过于粗暴。更精细的做法是对数据进行质量分层，为不同质量层级的数据赋予不同的采样权重。

一种常见的分层策略是：将数据分为高、中、低三个质量层级。高质量数据（如来自权威网站、通过所有启发式检查、困惑度低）赋予较高的采样权重；中等质量数据赋予正常权重；低质量但仍可接受的数据赋予较低权重。这种策略可以在保证数据多样性的同时，让高质量数据在训练中发挥更大作用。

RefinedWeb 的论文详细记录了他们的分层策略，将数据分为五个层级，每个层级使用不同的过滤阈值。这种精细化的质量管理是构建高质量预训练数据集的关键。

![图4-2：质量过滤漏斗](../images/chapter2/图4_2_质量过滤漏斗.png)

*图4-2：数据质量过滤漏斗 —— 从100%原始数据到最终4%清洁语料的逐层过滤过程*

---

## 4_2 大规模去重

数据重复是预训练数据的大敌。Common Crawl 中，同一篇文章可能被多个网站转载，同一个网页可能在不同月份被反复抓取，导致大量重复内容。研究表明，未经去重的数据会导致模型过拟合于重复内容，产生"复读机"现象，严重影响模型质量。

去重可以分为两个层次：精确去重（Exact Deduplication）移除完全相同的文档；模糊去重（Fuzzy Deduplication）移除高度相似但不完全相同的文档（如转载时略有修改的文章）。在 TB 级数据上，这两种去重都需要高效的算法和分布式实现。

### 4_2.1 精确去重：哈希方法

精确去重的核心思想是为每个文档计算一个指纹（fingerprint），相同指纹的文档视为重复。最简单的方法是使用 MD5 或 SHA256 等哈希函数。

```python
import hashlib

def compute_hash(text: str) -> str:
    """计算文本的 SHA256 哈希"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def exact_dedup(documents: list) -> list:
    """精确去重：保留每个哈希值的第一个文档"""
    seen_hashes = set()
    results = []
    
    for doc in documents:
        doc_hash = compute_hash(doc['text'])
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            doc['hash'] = doc_hash
            results.append(doc)
    
    return results
```

对于分布式场景，可以使用 Spark 或 Ray 进行并行去重：

```python
import ray

@ray.remote
def compute_hashes_batch(documents: list) -> list:
    """批量计算哈希"""
    return [(compute_hash(doc['text']), doc) for doc in documents]

def distributed_exact_dedup(documents_path: str, output_path: str):
    """分布式精确去重"""
    ds = ray.data.read_parquet(documents_path)
    
    # 计算哈希
    ds = ds.map(lambda doc: {**doc, 'hash': compute_hash(doc['text'])})
    
    # 按哈希分组，每组保留第一个
    ds = ds.groupby('hash').map_groups(lambda group: group.head(1))
    
    # 保存结果
    ds.write_parquet(output_path)
```

精确去重效率很高，但只能处理完全相同的文档。对于略有差异的重复内容（如同一篇新闻在不同网站的转载，可能有不同的页眉页脚），精确去重无能为力。

### 4_2.2 模糊去重：MinHash LSH

模糊去重的目标是识别"高度相似但不完全相同"的文档。这是一个计算复杂度很高的问题——朴素地比较任意两个文档需要 O(n²) 的时间复杂度，对于数十亿文档的数据集完全不可行。

MinHash LSH（Locality-Sensitive Hashing）是解决这一问题的核心算法。它的基本思想是：先将文档转换为 n-gram 集合，然后使用 MinHash 技术将集合压缩为固定长度的签名，最后使用 LSH 将相似的签名聚集到同一个桶中。只有落入同一个桶的文档对才需要进行精细比较，大大减少了计算量。

理解 MinHash LSH 需要分三步来看：

**第一步是 n-gram 分解。** 将文档视为 n-gram（连续 n 个字符或词）的集合。例如，"大模型数据" 的 3-gram 集合为 {"大模型", "模型数", "型数据"}。使用 n-gram 而非整个文档，可以更好地捕捉局部相似性。

**第二步是 MinHash 签名。** MinHash 是一种将集合压缩为固定长度签名的技术。两个集合的 Jaccard 相似度可以通过它们 MinHash 签名的匹配程度来近似估计。签名长度越长，估计越准确，但存储和计算开销也越大。

**第三步是 LSH 分桶。** 将 MinHash 签名分成若干个 band，每个 band 包含若干个 hash 值。如果两个文档在任意一个 band 中的所有 hash 值都相同，则它们被放入同一个桶。调整 band 的数量和每个 band 的大小，可以控制相似度阈值和召回率。

以下是一个完整的 MinHash LSH 实现：

![图4-3：MinHash LSH算法](../images/chapter2/图4_3_MinHash_LSH算法.png)

*图4-3：MinHash LSH 算法三步骤 —— N-gram分解、MinHash签名计算、LSH分桶，将复杂度从O(n²)降至O(n)*

```python
import hashlib
import struct
from typing import Set, List, Tuple
import numpy as np

class MinHashLSH:
    def __init__(self, 
                 num_hashes: int = 128,
                 num_bands: int = 16,
                 ngram_size: int = 5,
                 threshold: float = 0_8):
        """
        初始化 MinHash LSH
        
        Args:
            num_hashes: MinHash 签名长度
            num_bands: LSH band 数量
            ngram_size: n-gram 大小
            threshold: 相似度阈值（参考值，实际阈值由 band 参数决定）
        """
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.ngram_size = ngram_size
        
        # 生成哈希函数的随机参数
        self.hash_params = [
            (np.random.randint(1, 2**31), np.random.randint(0, 2**31))
            for _ in range(num_hashes)
        ]
        
        # LSH 桶
        self.buckets = [{} for _ in range(num_bands)]
    
    def get_ngrams(self, text: str) -> Set[str]:
        """提取 n-gram 集合"""
        text = text.lower().replace(' ', '')
        ngrams = set()
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.add(text[i:i + self.ngram_size])
        return ngrams
    
    def compute_minhash(self, ngrams: Set[str]) -> np.ndarray:
        """计算 MinHash 签名"""
        signature = np.full(self.num_hashes, np.inf)
        
        for ngram in ngrams:
            # 计算 ngram 的基础哈希值
            h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
            
            # 使用多个哈希函数
            for i, (a, b) in enumerate(self.hash_params):
                hash_val = (a * h + b) % (2**31 - 1)
                if hash_val < signature[i]:
                    signature[i] = hash_val
        
        return signature.astype(np.uint32)
    
    def get_bands(self, signature: np.ndarray) -> List[str]:
        """将签名分割为 bands"""
        bands = []
        for i in range(self.num_bands):
            start = i * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]
            band_hash = hashlib.md5(band.tobytes()).hexdigest()
            bands.append(band_hash)
        return bands
    
    def insert(self, doc_id: str, text: str):
        """插入文档到 LSH 索引"""
        ngrams = self.get_ngrams(text)
        if len(ngrams) == 0:
            return
        
        signature = self.compute_minhash(ngrams)
        bands = self.get_bands(signature)
        
        for band_idx, band_hash in enumerate(bands):
            if band_hash not in self.buckets[band_idx]:
                self.buckets[band_idx][band_hash] = []
            self.buckets[band_idx][band_hash].append(doc_id)
    
    def find_candidates(self, text: str) -> Set[str]:
        """查找候选相似文档"""
        ngrams = self.get_ngrams(text)
        if len(ngrams) == 0:
            return set()
        
        signature = self.compute_minhash(ngrams)
        bands = self.get_bands(signature)
        
        candidates = set()
        for band_idx, band_hash in enumerate(bands):
            if band_hash in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][band_hash])
        
        return candidates

def jaccard_similarity(set1: Set, set2: Set) -> float:
    """计算 Jaccard 相似度"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0
```

### 4_2.3 分布式去重实践

在 TB 级数据上运行 MinHash LSH 需要精心设计的分布式策略。一个典型的流程包括：

**阶段一：签名计算。** 并行遍历所有文档，为每个文档计算 MinHash 签名。这一步是完全并行的，可以充分利用分布式计算资源。

**阶段二：Band 分组。** 将每个文档按 band 值进行分组。相同 band 值的文档被分配到同一个分区，便于后续比较。

**阶段三：组内去重。** 在每个分区内，对候选重复文档对进行精细的相似度计算，确定真正的重复关系。

**阶段四：传递闭包。** 如果文档 A 与 B 重复，B 与 C 重复，则 A、B、C 都应视为一组重复。需要计算重复关系的传递闭包。

**阶段五：选择保留文档。** 在每组重复文档中选择一个代表（通常选择质量最高或长度最长的）保留，其他删除。

```python
import ray

def distributed_fuzzy_dedup(input_path: str, output_path: str, 
                            threshold: float = 0_8):
    """
    分布式模糊去重流水线
    """
    # 读取数据
    ds = ray.data.read_parquet(input_path)
    
    # 阶段一：计算 MinHash 签名
    def compute_signature(doc):
        lsh = MinHashLSH()
        ngrams = lsh.get_ngrams(doc['text'])
        signature = lsh.compute_minhash(ngrams)
        bands = lsh.get_bands(signature)
        return {**doc, 'signature': signature.tolist(), 'bands': bands}
    
    ds = ds.map(compute_signature)
    
    # 阶段二：按 band 值分组，找候选对
    # （这里简化处理，实际实现需要更复杂的分组逻辑）
    
    # 阶段三&四：组内精确比较，建立重复关系图
    # ...
    
    # 阶段五：选择保留文档
    # ...
    
    # 保存结果
    ds.write_parquet(output_path)
```

实际工程中，推荐使用现成的工具。**text-dedup** 是一个开源的文本去重库，实现了多种去重算法，包括 MinHash LSH、SimHash、Suffix Array 等，并提供了 Spark 和 Ray 的分布式实现。**Dolma** 的去重模块也是一个高质量的参考实现。

### 4_2.4 文档内去重

除了文档级别的去重，还需要处理文档内部的重复内容。常见的情况包括：网页中反复出现的导航栏、页眉页脚；由于 JavaScript 渲染问题导致的内容重复；某些 CMS 系统生成的模板化重复段落。

文档内去重的策略相对简单：将文档按段落或固定长度分块，计算每个块的哈希值，移除重复的块。

```python
def remove_duplicate_paragraphs(text: str, min_length: int = 50) -> str:
    """移除文档内的重复段落"""
    paragraphs = text.split('\n\n')
    seen_hashes = set()
    unique_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if len(para) < min_length:
            unique_paragraphs.append(para)
            continue
        
        para_hash = hashlib.md5(para.encode()).hexdigest()
        if para_hash not in seen_hashes:
            seen_hashes.add(para_hash)
            unique_paragraphs.append(para)
    
    return '\n\n'.join(unique_paragraphs)

def remove_duplicate_ngrams(text: str, n: int = 10, threshold: int = 3) -> str:
    """移除文档内高频重复的 n-gram"""
    words = text.split()
    ngram_counts = Counter()
    
    # 计算 n-gram 频次
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        ngram_counts[ngram] += 1
    
    # 标记需要移除的位置
    remove_positions = set()
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        if ngram_counts[ngram] >= threshold:
            # 保留第一次出现，移除后续重复
            for j in range(i + n, len(words) - n + 1):
                if tuple(words[j:j + n]) == ngram:
                    for k in range(j, min(j + n, len(words))):
                        remove_positions.add(k)
    
    # 重建文本
    result_words = [w for i, w in enumerate(words) if i not in remove_positions]
    return ' '.join(result_words)
```

---

## 4_3 隐私数据清洗 (PII Removal)

预训练数据中不可避免地包含个人身份信息（Personally Identifiable Information, PII），如邮箱地址、电话号码、身份证号、银行卡号、家庭住址等。在数据合规要求日益严格的今天（如 GDPR、CCPA、《个人信息保护法》），清洗 PII 不仅是道德责任，也是法律义务。

### 4_3.1 PII 的类型与风险

PII 可以分为直接标识符和准标识符两类。直接标识符可以单独识别个人身份，如姓名、身份证号、社会保障号、电话号码、电子邮箱。准标识符单独难以识别个人，但组合使用可能导致识别，如出生日期、邮政编码、职业、工作单位。

在预训练数据中保留 PII 存在多重风险。首先是隐私泄露风险：模型可能"记住"训练数据中的敏感信息，在推理时被恶意提取。其次是合规风险：违反数据保护法规可能导致巨额罚款。最后是声誉风险：如果模型输出他人隐私信息，会严重损害企业形象。

![图4-4：PII类型与风险](../images/chapter2/图4_4_PII类型与风险.png)

*图4-4：PII类型与风险等级 —— 直接标识符（高风险）与准标识符（中风险）的分类*

### 4_3.2 Microsoft Presidio

Presidio 是微软开源的 PII 识别和匿名化工具包，支持多种语言和多种 PII 类型。它采用模块化设计，包含两个核心组件：Analyzer 负责在文本中识别 PII 实体，Anonymizer 负责对识别出的 PII 进行处理（如替换、掩码、删除）。

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# 初始化引擎
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def analyze_pii(text: str, language: str = 'en') -> list:
    """
    识别文本中的 PII
    
    Returns:
        PII 实体列表，包含类型、位置和置信度
    """
    results = analyzer.analyze(
        text=text,
        language=language,
        entities=[
            'EMAIL_ADDRESS', 'PHONE_NUMBER', 'CREDIT_CARD',
            'IP_ADDRESS', 'PERSON', 'LOCATION', 'DATE_TIME'
        ]
    )
    return results

def anonymize_pii(text: str, language: str = 'en') -> str:
    """
    匿名化文本中的 PII
    
    将识别出的 PII 替换为占位符
    """
    # 先识别
    analyzer_results = analyzer.analyze(text=text, language=language)
    
    # 定义匿名化策略
    operators = {
        'EMAIL_ADDRESS': OperatorConfig('replace', {'new_value': '<EMAIL>'}),
        'PHONE_NUMBER': OperatorConfig('replace', {'new_value': '<PHONE>'}),
        'CREDIT_CARD': OperatorConfig('replace', {'new_value': '<CREDIT_CARD>'}),
        'IP_ADDRESS': OperatorConfig('replace', {'new_value': '<IP>'}),
        'PERSON': OperatorConfig('replace', {'new_value': '<PERSON>'}),
        'LOCATION': OperatorConfig('replace', {'new_value': '<LOCATION>'}),
        'DATE_TIME': OperatorConfig('keep', {})  # 日期时间通常可以保留
    }
    
    # 匿名化
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators=operators
    )
    
    return anonymized.text
```

### 4_3.3 中文 PII 处理

Presidio 对中文的支持相对有限。对于中文预训练数据，通常需要补充基于正则表达式的规则匹配。

```python
import re

class ChinesePIIFilter:
    """中文 PII 过滤器"""
    
    patterns = {
        'phone': [
            r'1[3-9]\d{9}',  # 手机号
            r'0\d{2,3}-?\d{7,8}',  # 固定电话
        ],
        'id_card': [
            r'\d{17}[\dXx]',  # 18位身份证
            r'\d{15}',  # 15位身份证
        ],
        'email': [
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        ],
        'bank_card': [
            r'\d{16,19}',  # 银行卡号（需要结合上下文判断）
        ],
        'ip_address': [
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
        ],
        'qq': [
            r'[Qq][Qq][:：]?\s*\d{5,11}',
            r'[Qq][:：]?\s*\d{5,11}',
        ],
        'wechat': [
            r'[Vv][Xx][:：]?\s*[a-zA-Z0-9_-]{6,20}',
            r'微信[:：]?\s*[a-zA-Z0-9_-]{6,20}',
        ],
    }
    
    def __init__(self):
        self.compiled_patterns = {}
        for pii_type, patterns in self.patterns.items():
            self.compiled_patterns[pii_type] = [
                re.compile(p) for p in patterns
            ]
    
    def find_pii(self, text: str) -> list:
        """查找所有 PII"""
        findings = []
        for pii_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    findings.append({
                        'type': pii_type,
                        'value': match.group(),
                        'start': match.start(),
                        'end': match.end()
                    })
        return findings
    
    def anonymize(self, text: str) -> str:
        """匿名化 PII"""
        findings = self.find_pii(text)
        
        # 按位置倒序处理，避免替换影响后续位置
        findings.sort(key=lambda x: x['start'], reverse=True)
        
        for finding in findings:
            placeholder = f"<{finding['type'].upper()}>"
            text = text[:finding['start']] + placeholder + text[finding['end']:]
        
        return text
```

### 4_3.4 PII 处理策略的权衡

PII 处理面临准确率与召回率的权衡。过于激进的过滤可能误伤正常内容（如将普通数字序列误判为电话号码），过于保守则可能遗漏真正的敏感信息。

在实践中，建议采用分层策略。对于高风险 PII（如身份证号、银行卡号），使用较严格的匹配规则，宁可误删也不遗漏。对于中风险 PII（如电话号码、邮箱），使用适中的阈值，平衡准确率和召回率。对于低风险信息（如日期、地点），可以根据具体场景决定是否处理。

另一个重要决策是替换策略。常见的选择包括：完全删除，简单但可能破坏语句流畅性；固定占位符替换（如 `<EMAIL>`），保留语义信息但可能引入不自然的模式；随机生成替换（如用随机邮箱替换真实邮箱），最接近原始分布但实现复杂。大多数预训练数据集采用占位符替换策略，作为准确率和复杂度的平衡。

---

## 4_4 完整清洗流水线

将前面介绍的各个组件串联起来，构建一个完整的数据清洗流水线。

### 4_4.1 流水线架构

一个工业级的清洗流水线通常包括以下阶段，按顺序执行：

**阶段一：格式标准化。** 将各种来源的数据转换为统一格式，处理编码问题，提取必要的元数据。

**阶段二：语言过滤。** 使用 FastText 识别语言，保留目标语言的文档。对于混合语言文档，根据主要语言进行分类。

**阶段三：启发式过滤。** 应用长度、特殊字符、重复行等启发式规则，快速过滤明显的低质量内容。

**阶段四：文档内去重。** 移除文档内部的重复段落和重复 n-gram。

**阶段五：PII 清洗。** 识别并匿名化敏感个人信息。

**阶段六：质量评分。** 计算困惑度等质量指标，为后续的质量分层提供依据。

**阶段七：文档间去重。** 使用 MinHash LSH 进行大规模模糊去重，移除高度相似的文档。

**阶段八：质量分层与采样。** 根据质量评分将数据分层，确定各层的采样权重。

```python
import ray
from dataclasses import dataclass
from typing import Optional

@dataclass
class CleaningConfig:
    """清洗配置"""
    target_language: str = 'zh'
    min_length: int = 200
    max_length: int = 100000
    max_perplexity: float = 500
    dedup_threshold: float = 0_8
    anonymize_pii: bool = True

class DataCleaningPipeline:
    def __init__(self, config: CleaningConfig):
        self.config = config
        self.lang_filter = LanguageFilter(config.target_language)
        self.heuristic_filter = HeuristicFilter()
        self.perplexity_filter = PerplexityFilter(max_ppl=config.max_perplexity)
        self.pii_filter = ChinesePIIFilter() if config.target_language == 'zh' else None
        self.deduplicator = MinHashLSH(threshold=config.dedup_threshold)
    
    def process_document(self, doc: dict) -> Optional[dict]:
        """处理单个文档"""
        text = doc.get('text', '')
        
        # 阶段二：语言过滤
        lang, conf = self.lang_filter.detect(text)
        if lang != self.config.target_language:
            return None
        
        # 阶段三：启发式过滤
        passed, reason = self.heuristic_filter.filter(text)
        if not passed:
            return None
        
        # 阶段四：文档内去重
        text = remove_duplicate_paragraphs(text)
        
        # 阶段五：PII 清洗
        if self.config.anonymize_pii and self.pii_filter:
            text = self.pii_filter.anonymize(text)
        
        # 阶段六：质量评分
        perplexity = self.perplexity_filter.compute_perplexity(text)
        if perplexity > self.config.max_perplexity:
            return None
        
        return {
            **doc,
            'text': text,
            'language': lang,
            'lang_confidence': conf,
            'perplexity': perplexity
        }
    
    def run(self, input_path: str, output_path: str):
        """运行完整流水线"""
        # 读取数据
        ds = ray.data.read_parquet(input_path)
        
        # 阶段一到六：单文档处理
        ds = ds.map(self.process_document)
        ds = ds.filter(lambda x: x is not None)
        
        # 阶段七：文档间去重
        ds = self.deduplicator.deduplicate(ds)
        
        # 保存结果
        ds.write_parquet(output_path)
```

### 4_4.2 质量监控与迭代

清洗流水线不是一次性任务，而是需要持续监控和迭代优化的过程。建议建立以下监控机制：

**过滤率监控**：统计每个阶段的过滤率。如果某个阶段突然过滤掉大量数据，可能是阈值设置不当或数据分布发生变化。

**样本抽检**：定期人工抽检清洗结果，评估过滤规则的准确性。误删的好样本和漏删的坏样本都需要关注。

**下游反馈**：模型训练后的评测结果是最终的质量验证。如果模型表现不佳，需要回溯分析数据是否存在问题。

---

## 4_5 本章小结

本章系统介绍了预训练数据清洗的核心技术。

在启发式过滤方面，语言识别使用 FastText 快速筛选目标语言文档，困惑度过滤使用 KenLM 评估文本质量，启发式规则集涵盖长度、特殊字符、重复行、词汇多样性等多个维度。质量分层策略将数据划分为不同等级，为后续采样提供依据。

在大规模去重方面，精确去重使用哈希方法快速移除完全相同的文档，模糊去重使用 MinHash LSH 算法识别高度相似的内容。分布式实现是处理 TB 级数据的必要手段。文档内去重处理段落和 n-gram 级别的重复。

在隐私清洗方面，PII 识别可以使用 Presidio 或自定义正则规则，匿名化策略需要在准确率和信息保留之间权衡。中文 PII 处理需要特别设计的规则集。

完整的清洗流水线将各个组件串联，按照格式标准化、语言过滤、启发式过滤、文档内去重、PII 清洗、质量评分、文档间去重、质量分层的顺序执行。持续的质量监控和迭代优化是保证数据质量的关键。

![图4-5：本章知识结构](../images/chapter2/图4_5_本章知识结构.png)

*图4-5：第4章知识结构 —— 启发式过滤、大规模去重、PII清洗三大核心主题*

---

## 延伸阅读

关于数据清洗的深入内容，以下资源值得参考：

RefinedWeb 论文详细记录了从 Common Crawl 构建高质量预训练集的完整清洗流程。Dolma 数据集的技术报告介绍了 Allen AI 的清洗策略和工具。text-dedup 开源库（github.com/ChenghaoMou/text-dedup）提供了多种去重算法的实现。Microsoft Presidio 文档（microsoft.github.io/presidio）是 PII 处理的权威参考。CCNet 论文介绍了 Facebook 处理 Common Crawl 数据的方法，特别是困惑度过滤的细节。

---

## 下一章预告

在下一章《分词与序列化》中，我们将探讨预训练数据准备的最后一个关键步骤：如何将清洗后的文本转换为模型可以理解的 Token 序列。你将学习 BPE、WordPiece、Unigram 等分词算法的原理与选择，如何为特定领域扩充词表，以及数据混合与课程学习的采样策略。

带着这个问题进入下一章：如果你要训练一个专门处理代码的模型，标准的 GPT-2 分词器会遇到什么问题？
