# 第5章：清洗、去重与去污染

## 摘要

本章讨论文本预训练数据从原始语料转化为训练语料的关键步骤，覆盖规则过滤、模型质量评分、文本标准化、精确去重、模糊去重、语义去重、PII 脱敏和基准去污染。章节首先说明重复、低信息密度、隐私泄露和评测污染如何在训练阶段被放大，随后给出规则、模型和人工抽检协同的清洗框架。去重部分从 SHA-256 精确匹配扩展到 MinHash LSH 和 Embedding 相似度，强调去重不足与去重过度的双重风险。隐私和去污染部分分别讨论结构化 PII、命名实体、API Key 等机密检测，以及评测集 N-gram 指纹隔离。最后，章节通过匿名化复合案例和三档团队配置说明清洗管线的落地路径。读者应能够根据数据规模、团队资源和目标模型能力，设计具备可追溯、可抽检、可迭代能力的文本清洗方案。

## 关键词

数据清洗；去重；MinHash；PII 脱敏；基准去污染；质量评分；文本标准化；人工抽检

## 学习目标

- 能够解释重复、低质量文本、PII 和基准污染对预训练模型的影响。
- 能够组合规则过滤、模型评分和人工抽检形成分层清洗流程。
- 能够区分精确去重、模糊去重和语义去重的适用边界。
- 能够设计结构化 PII、API Key 和评测污染的检测与隔离策略。
- 能够根据团队规模选择轻量级、标准或平台级清洗方案。

## 开篇：一批"看起来很干净"的数据，为何让模型开始复读？

以下为匿名化复合案例。某团队在完成了第4章描述的数据采集之后，启动了一个中文基座模型的预训练。训练过程表面稳定——Loss 曲线平滑下降，GPU 利用率也达到项目基线。直到第一次基准评测的结果出来，团队发现了一个令人困惑的现象：模型在续写任务中会反复生成完全相同的句子，有时一段回复里同一句话重复多次；更奇怪的是，给模型一个简单的触发词，它竟然能够背出某电商平台商品描述的完整格式，逐字不差。

这是典型的**数据重复导致的过拟合**。在排查过程中，工程师发现训练数据中有一个大型电商平台的商品描述语料，通过不同的 URL 路径被爬取了数十次，导致同一类商品描述格式在训练集中出现了数万次之多。尽管在数据接入时做了基于 URL 的精确去重，但这些内容的 URL 不同（不同商品的 URL、不同时间爬取同一商品的归档 URL），因此精确去重完全没有发现这个问题。

这个案例说明了一个核心命题：**清洗不是简单删除低质量数据，而是构建训练数据质量上限的工程体系。** 从这个意义上讲，清洗章是文本预训练数据工程的核心章节之一。

---

## 5.1 为什么清洗决定训练数据质量上限

### 5.1.1 清洗投入与训练收益的非线性关系

FineWeb 项目（Penedo et al. 2024）给出了一个量化答案：针对同样规模的 Common Crawl 数据，不同的清洗策略会带来显著训练效果差异。使用精细多阶段清洗管线处理的数据，在下游基准评测上的表现优于简单规则清洗的数据。具体收益会随模型规模、语料结构和评测集而变化，不能脱离实验设置直接复用。

这一发现修正了早期"数据量足够大，质量不那么重要"的粗放思维。在算力资源有限的情况下，**把有限算力用在高质量数据上，通常优于把同等算力用在低质量数据上**。从这个视角来看，数据清洗工程投资的 ROI，是 LLM 研发链路中较高的环节之一。

### 5.1.2 上游缺陷如何在训练阶段被指数放大

上游数据管线的细微缺陷，经过训练过程的梯度累积，会在下游呈现出数量级的放大效应。几个典型的放大路径包括：

**重复内容的"记忆化"效应**：一段文本如果在训练集中出现 100 次以上，模型就会以极高的概率将这段文本"背诵"下来，并在接受相关触发词时逐字输出。这种行为本质上是对训练集的过拟合，既破坏了模型的泛化能力，也带来了严重的版权和隐私风险（模型可能原文背出用户的个人信息）。

**PII 数据的"幻觉"激活**：训练数据中未被脱敏的个人信息（手机号、邮箱、身份证号），会被模型以统计关联的方式学习到。当用户问到某人的电话号码类问题时，模型可能生成看似合理但实际指向真实个人的信息，形成 PII 泄露风险。

**基准污染的"分数虚高"**：如果训练数据中混入了测试集的题目和答案（Benchmark Contamination），模型在基准测试上的表现会被人为虚高。这是行业内极为敏感的诚信问题——多家机构因此在发布模型时不得不追加声明和重新评测。

---

## 5.2 规则、模型与人工协同的清洗框架

面对上述问题，工业界实践证明，没有任何单一手段能够独立完成高质量的数据清洗——有效的清洗体系必然是**规则过滤、模型过滤和人工抽检**三类方法的协同组合，每类方法覆盖不同的缺陷类型，各有其最优使用场景（见图5-1）。

![图5-1：清洗与去污染全景流程图](../../images/part2/cleaning_pipeline_overview.png)

*图5-1：清洗与去污染全景流程图 —— 多阶段质量闸门从原始语料逐步精炼为候选训练语料，图中比例仅为示意，真实留存率取决于来源质量、过滤阈值和合规要求。来源：本书自绘；Alt text：清洗与去污染全景流程图，展示规则过滤、模型评分、去重、PII 脱敏、去污染和人工抽检的顺序关系。*

### 5.2.1 第一道闸门：规则过滤

规则过滤（Rule-Based Filtering）是清洗流水线的第一道防线，也是性价比最高的一关。它基于一系列可量化的启发式规则，在不需要运行任何模型的前提下，快速剔除明显低质量文档。过滤比例取决于来源质量，不能跨项目复用。

**语言识别**是多语言语料库清洗的必要起点。FastText (Joulin et al. 2017) 的多语言识别模型（`lid.176.bin`，支持 176 种语言）是常用工程选择。实践中需要注意的是，置信度阈值应基于人工抽检校准；低置信度的混合语言文档（如中英夹杂的技术博客）可以保留并单独处理，而非直接丢弃。

**长度与字符比例过滤**是最基础的规则集。典型做法是为文档最小长度、最大长度、特殊字符占比和数字字符占比设置项目阈值。过短内容通常是导航栏或标签文字，超长文档可能是被合并的多页内容，需要分段处理；日志、代码和数据表格类内容应使用独立阈值，避免被通用文本规则误杀。

**重复行比例过滤**专门针对"模板噪声"——许多低质量网页会在同一页面内多次重复导航栏、版权声明、广告区域等内容，导致文档内大量行完全一致。若重复行占比超过项目阈值，文档可被视为低质量候选，触发进一步审查或直接丢弃。

代码清单5-1展示了多规则启发式质量过滤器的示意实现。

*代码清单5-1：多规则启发式质量过滤示意代码。阈值为演示配置，生产环境应按语种、来源和人工抽检结果校准。*

```python
import re
from typing import Tuple

class HeuristicQualityFilter:
    """多规则启发式质量过滤器（中文预训练数据优化版）"""

    def __init__(self):
        self.rules = {
            'min_chars': 200,
            'max_chars': 100_000,
            'max_special_ratio': 0.30,
            'max_digit_ratio': 0.30,
            'max_dup_line_ratio': 0.30,
            'min_unique_token_ratio': 0.10,
        }

    def basic_tokens(self, text: str) -> list[str]:
        """面向中英文混合文本的轻量 token 化：中文按字，英文/数字按连续片段。"""
        return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text.lower())

    def run(self, text: str) -> Tuple[bool, str]:
        n = len(text)
        if not (self.rules['min_chars'] <= n <= self.rules['max_chars']):
            return False, 'length'
        if len(re.findall(r'[^\w\s]', text, re.UNICODE)) / n > self.rules['max_special_ratio']:
            return False, 'special_chars'
        if len(re.findall(r'\d', text)) / n > self.rules['max_digit_ratio']:
            return False, 'digit_ratio'
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines and (1 - len(set(lines)) / len(lines)) > self.rules['max_dup_line_ratio']:
            return False, 'dup_lines'
        tokens = self.basic_tokens(text)
        if tokens and len(set(tokens)) / len(tokens) < self.rules['min_unique_token_ratio']:
            return False, 'low_diversity'
        return True, 'pass'
```

### 5.2.2 第二道闸门：基于模型的质量评分

规则过滤能快速过滤"明显"的低质量内容，但面对一段语法完全正确、格式也没有问题、但实质上是无意义的广告堆砌或 SEO 软文，规则过滤往往无能为力。这时需要**模型过滤（Model-Based Filtering）**——利用训练好的评分模型，对文档的语言质量进行更细粒度的判断。

**困惑度过滤（Perplexity Filter）**是目前最广泛采用的模型过滤方法。使用 **KenLM (Heafield 2011) n-gram 语言模型**（而非神经网络模型）计算困惑度时，可以将文本质量量化，但 PPL 阈值必须绑定训练 KenLM 的语料、分词方式和目标语言：高质量新闻、百科、普通网页、乱码和广告堆砌样本会呈现不同分布，生产系统应先在人工确认样本上建立各来源基线，再用分位数或 Z-score 异常检测设置拦截水位。值得注意的是，困惑度并非越低越好——困惑度异常偏低的文本可能是高度同质化的样板文本（如格式几乎固定的法律条文或商品说明），也需要额外关注。（注：若改用神经网络参照模型计算，相同文本的 PPL 数值分布会改变，两者不可混用阈值。）

**质量分类器（Quality Classifier）**是 RefinedWeb (Penedo et al. 2023)、Dolma (Soldaini et al. 2024) 等代表性数据集采用的进阶手段（注意：Nait Saada et al. 2025 的实证研究表明，分类器过滤实质上起到的是"领域选择"而非"绝对质量"筛选的作用，需配合人工抽检验证）：用一个经过人工标注的高质量文档 vs 低质量文档数据集，微调一个 fastText 或轻量级 BERT 分类器，将质量打分做成强监督的二分类或五分类问题。这种方法在覆盖"规则和困惑度都无法发现但人类能判断"的质量问题上有显著优势，代价是需要一定量的人工标注成本来构建训练集。

### 5.2.3 三阶段协同：规则、模型、人工的合理分工

从实际工程的成本效益角度来看，三类方法的合理分工是：

**规则过滤**处理体量最大、最明显的问题，速度极快，成本极低，但误判率较高（既会有漏网之鱼，也会有误杀），适合"宽松过滤"的第一阶段。

**模型过滤**在规则过滤之后处理剩余的模糊地带，精度高于规则但速度慢、成本高（需要运行模型推理），适合中等精度要求的"精细过滤"第二阶段。

**人工抽检**不是全量处理，而是以"质量审计"的方式对清洗结果进行抽样验证：每批次随机抽取 500-1000 条数据，由数据工程师进行人工判读，发现系统性错误（如某类有价值的内容被规则误杀），并将发现反馈给规则和模型的迭代优化。这是质量闭环的关键闭合点。

### 5.2.4 文本标准化：让数据"说同一种话"

在完成质量过滤之后，还有一类工作经常被忽视，但对下游 Tokenizer 和模型训练有深远影响——**文本标准化（Text Normalization）**。来自不同来源的数据往往在编码格式、标点习惯、空白字符使用等方面存在大量差异，如果不在清洗阶段统一处理，这些细节差异会被分词器（Tokenizer）识别为成千上万种"不同的 token"，不仅增大了词表压力，还会干扰模型对语义等价内容的统一表示。

**Unicode 规范化**是最基础的一步。同一个汉字可能以 NFC（组合形式）或 NFD（分解形式）编码，导致字符级别的文本看起来一样，但实际字节序列不同，在精确去重时会被当作不同文档处理。推荐统一将所有文本归一化为 `unicodedata.normalize('NFC', text)`，这在 Python 中只需一行代码，但能消除大量来自不同操作系统和编辑器的编码差异。

**全半角字符统一**是中文数据预处理的特有挑战。中文互联网语料中，全角字符（如"，。！""（）"）和半角字符（`,. ! "" ()`）混用极为普遍。对于大多数大模型训练场景，建议统一为半角标点——这与绝大多数高质量训练数据（学术论文、程序文档、英文维基）的标点习惯一致，避免模型在中文回复中混用全半角造成视觉混乱。

**多余空白字符清理**包括连续空格压缩为单空格、移除行首/行尾多余空格、将 Windows 风格换行（`\r\n`）统一为 Unix 风格（`\n`）、以及清除零宽度不可见字符（如 `\u200b`、`\ufeff` BOM 标记）。这些隐形字符来自不同平台的文本粘贴和格式转换，在模型 tokenization 阶段可能产生意外的 `<unk>` token，影响训练稳定性。

**繁简体处理策略**对中文大模型有特别的重要性。训练全参数的多方言中文模型时，繁体字和简体字应当共存；但若目标是一个专注于大陆简体中文的模型，建议对繁体语料进行简体转换（opencc 库可以实现高质量的繁简转换）。需注意：机械的繁简转换会丢失一些台湾、香港等地区特有的词汇和表达，对于涉及这些地区的垂直领域模型，需要谨慎处理而非一刀切简繁归一。

代码清单5-2展示了文本标准化的示意实现。

*代码清单5-2：文本标准化处理示意代码。生产环境应按目标语种、标点策略和繁简体策略建立可回滚配置。*

```python
import unicodedata, re

def normalize_text(text: str, to_simplified: bool = False) -> str:
    """
    文本标准化：Unicode 归一化 + 标点统一 + 空白清理
    to_simplified: 是否将繁体转为简体（需额外安装 opencc-python-reimplemented）
    """
    # 1. Unicode NFC 归一化
    text = unicodedata.normalize('NFC', text)
    # 2. 移除零宽字符和 BOM
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\u00ad]', '', text)
    # 3. Windows 换行统一为 Unix 换行
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # 4. 压缩连续空格和 Tab
    text = re.sub(r'[ \t]+', ' ', text)
    # 5. 去除行首行尾空格
    text = '\n'.join(line.strip() for line in text.split('\n'))
    # 6. 压缩连续空行（超过 2 行的空行压缩为 1 行）
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 7. 可选：繁简转换
    if to_simplified:
        try:
            import opencc
            converter = opencc.OpenCC('t2s')
            text = converter.convert(text)
        except ImportError:
            pass  # opencc 未安装时跳过
    return text.strip()
```

文本标准化虽然在单条数据上的效果不显眼，但在 TB 级语料库的整体层面，它能有效减少分词器的词表碎片化程度，降低训练集中语义等价内容的 token 表示多样性，使模型的注意力更集中于真正的语义学习而非格式差异。这是清洗管线中成本极低、收益稳定的一环，建议在任何规模的项目中都作为标配步骤纳入。



---

## 5.3 去重：从精确匹配到语义近似查重

### 5.3.1 为什么去重是清洗体系中最具挑战性的环节

去重（Deduplication）是清洗体系中工程复杂度最高、计算成本最重的环节。假设一个教学估算口径下的 50TB、约 50 亿文档语料库：精确去重需要为每个文档计算哈希值并进行全局比对，仅 I/O 成本就已相当可观；模糊去重则需要在海量候选文档对之间计算相似度，朴素实现的 O(n²) 复杂度在这个规模上完全不可行。真实文档数会随平均文档长度、压缩率和切分策略变化，应以数据接入元信息重新估算。

研究表明，未经充分去重的预训练语料会带来两类破坏性影响：一是模型对重复内容产生过拟合（开篇案例的根因）；二是模型在续写时倾向于"重复自身"，产生"复读机"现象，严重降低生成质量的多样性和流畅性。

### 5.3.2 精确去重：SHA-256 哈希

精确去重（Exact Deduplication）通过为每个文档计算一个固定长度的哈希指纹（通常用 SHA-256），相同哈希值的文档视为完全重复，只保留第一次出现的版本。这种方法实现简单、速度极快，可以在 O(n) 时间内完成，但**只能处理完全字符级相同的文档**，无法发现"略有差别"的近似重复（如同一篇文章在不同网站转载时加了不同的页眉页脚）。

在分布式场景下，可以用 Ray Data 或 Spark 的 groupBy 算子高效实现：将文档哈希值作为分组键，每组只保留一个文档。在百亿文档的规模下，精确去重可以在 8-16 节点的 CPU 集群上以数小时内完成。

### 5.3.3 模糊去重：MinHash LSH 的三步原理与工程实现

模糊去重（Fuzzy Deduplication）的目标是识别"相似度超过阈值（如 Jaccard 相似度 > 0.8）"的文档对，并从中只保留一个版本。MinHash LSH (Broder 1997; Indyk and Motwani 1998) 是目前处理 TB 级数据规模下模糊去重的工业标准算法，其核心思想是在极大降低计算量的前提下，以高概率识别出真正相似的文档对。

**第一步：N-gram 分解**。将文档转化为字符级或词级 n-gram 的集合（通常使用 5-gram）。两个文档的 Jaccard 相似度定义为它们 n-gram 集合的交集与并集之比。

**第二步：MinHash 签名压缩**。使用 k 个随机哈希函数，为每个文档的 n-gram 集合生成一个长度为 k 的 MinHash 签名向量（通常 k=128）。MinHash 的关键性质是：两个文档签名的匹配率，是它们 Jaccard 相似度的无偏估计。

**第三步：LSH 分桶**。将 128 维签名向量分成 b 个 band（每 band 含 r = 128/b 个维度）。两个文档只要在任意一个 band 内的签名完全匹配，就被放入同一个"候选桶"，后续只需对同桶内的文档对进行精确相似度计算。调节 b 和 r 可以控制实际的相似度检测阈值与召回率之间的权衡。

代码清单5-3展示了 MinHash LSH 模糊去重的示意实现，生产环境应将桶结构持久化到分布式存储或流式计算框架中。

*代码清单5-3：MinHash LSH 模糊去重示意代码。该片段用于说明算法结构，生产环境应将桶结构、候选对和复核结果持久化到分布式存储。*

```python
import hashlib
import re
import numpy as np
from typing import Set

class MinHashLSH:
    """MinHash LSH 模糊去重实现（适用于中文 5-gram）"""

    def __init__(self, num_hashes=128, num_bands=16, ngram=5, threshold=0.8):
        self.num_hashes  = num_hashes
        self.num_bands   = num_bands
        self.rows        = num_hashes // num_bands
        self.ngram       = ngram
        # 随机参数（线性哈希族）
        rng = np.random.default_rng(42)
        self.a = rng.integers(1, 2**31, num_hashes)
        self.b = rng.integers(0, 2**31, num_hashes)
        self.p = (1 << 31) - 1              # 梅森素数
        self.buckets = [{} for _ in range(num_bands)]
        self.shingles_by_doc = {}
        self.signatures_by_doc = {}

    def stable_hash(self, value: str) -> int:
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.p

    def ngrams(self, text: str) -> Set[int]:
        t = re.sub(r"\s+", "", text.lower())
        if len(t) < self.ngram:
            return {self.stable_hash(t)} if t else set()
        return {self.stable_hash(t[i:i+self.ngram]) for i in range(len(t)-self.ngram+1)}

    def signature(self, shingles: Set[int]) -> np.ndarray:
        if not shingles:
            return np.full(self.num_hashes, self.p, dtype=np.int64)
        sig = np.full(self.num_hashes, np.inf)
        for s in shingles:
            h = (self.a * s + self.b) % self.p
            sig = np.minimum(sig, h)
        return sig.astype(np.int64)

    def jaccard(self, a: Set[int], b: Set[int]) -> float:
        return len(a & b) / max(len(a | b), 1)

    def insert(self, doc_id: str, text: str) -> list[str]:
        """插入文档，返回经真实 Jaccard 相似度确认的重复文档 ID 列表。"""
        shingles = self.ngrams(text)
        sig = self.signature(shingles)
        candidates = set()
        for i in range(self.num_bands):
            band_key = tuple(sig[i*self.rows:(i+1)*self.rows])
            if band_key in self.buckets[i]:
                candidates.update(self.buckets[i][band_key])
            self.buckets[i].setdefault(band_key, []).append(doc_id)
        duplicates = [
            other_id for other_id in candidates
            if self.jaccard(shingles, self.shingles_by_doc[other_id]) >= self.threshold
        ]
        self.shingles_by_doc[doc_id] = shingles
        self.signatures_by_doc[doc_id] = sig
        return duplicates
```

### 5.3.4 语义去重：超越字面匹配的 Embedding 相似度

无论是精确的哈希去重还是基于 N-gram 的 MinHash LSH，本质上都是在捕捉文本的“字面特征”。但如果同一篇新闻报道被两家媒体用完全不同的词汇改写（如同义词替换、句子结构重组），字面重叠度极低，此时传统的 LSH 将彻底失效。为了应对这种情况，**语义去重（Semantic Deduplication）** 引入了 Embedding 模型。

在实现上，通常使用轻量级的 Embedding 模型（如 `BGE-M3` 或 `text2vec`）将每篇文档编码为稠密向量，然后通过向量数据库（如 Milvus 或 FAISS）构建近似最近邻（ANN）索引。若两个文档向量之间的余弦相似度（Cosine Similarity）超过项目阈值，即使它们的字面完全不同，也会被识别为高度语义同质化内容并进入复核或去重流程。
由于计算 Embedding 需要消耗可观的 GPU 推理算力，工业界通常将其作为**去重管线的最后一环**：先用极低成本的哈希去重和 MinHash 筛掉明显冗余，再对剩下的高价值语料进行语义去重，在算力成本与去重精度之间取得平衡。

### 5.3.5 去重的双重风险：过度与不足

去重并非越激进越好。**去重过度**（阈值设置过低，如 Jaccard > 0.5，或语义余弦相似度阈值过低）会导致大量内容相关但表达不同的文档被错误地视为重复而删除，损害数据多样性——这在专业领域尤为危险，因为领域内的高质量知识本来就有相当的主题重叠。**去重不足**（阈值过高或只做精确去重）则会遗留大量近似重复，导致前述的过拟合和复读机问题。

实践中推荐的起点阈值是 Jaccard 相似度 0.7-0.8，语义相似度 0.9-0.95，并在代理模型评测（Proxy Model Evaluation）上对不同阈值配置进行消融实验，找出当前语料和目标任务下的最优点。

---

## 5.4 PII 脱敏与个人隐私保护

### 5.4.1 常见 PII 类型与危害

个人可识别信息（Personally Identifiable Information，PII）是训练数据中隐藏最深、危害最滞后的一类缺陷。与噪声和重复不同，PII 的存在不会直接影响 Loss 指标或基准评测分数，但会在模型上线后形成严重的隐私泄露风险。

中文训练语料中最常见的 PII 类型包括：手机号（11 位数字，以 1 开头）、身份证号（18 位，含出生日期和地区码）、电子邮件地址、家庭住址和邮政编码、姓名（高频人名与组织名）、账号密码和 API Token（在代码仓库、技术论坛中尤为常见）。其中，账号和 Token 类 PII 的危害最为直接——模型可能在生成代码示例时，将训练集中看到的真实 API Key 原文输出，导致即时的安全事故。

### 5.4.2 检测与脱敏方案

PII 检测通常采用**规则 + NER 模型**的组合方案：

**正则表达式规则**对于结构化 PII（手机号、邮箱、身份证、IP 地址等）有极高的查全率，且运行速度极快，适合作为第一道检测层：

代码清单5-4展示了结构化 PII 检测与脱敏的示意实现。该示例只适合作为第一层召回规则，生产环境还应叠加校验和、上下文 allowlist、NER/Secrets 检测器与人工抽检，避免把格式相似但非敏感的字符串误删，或漏掉变体写法。

*代码清单5-4：结构化 PII 检测与脱敏示意代码。该示例只适合作为第一层召回规则，生产环境应叠加上下文校验、NER/Secrets 检测器和人工抽检。*

```python
import re

from dataclasses import dataclass
from typing import Optional

@dataclass
class PiiMatch:
    pii_type: str
    start: int
    end: int
    original: str
    redacted: str = ""
    confidence: str = "high"  # high=结构化规则，medium=NER

# ── 第一层：结构化正则（固定格式 PII，高精度） ─────────────────────
_STRUCTURAL_PATTERNS: dict = {
    "phone_cn":   r"(?<!\d)(?:(?:\+?86)[\s\-]?)?1[3-9]\d{9}(?!\d)",
    "id_card_cn": r"(?<!\d)[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)",
    "email":      r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "ip_v4":      r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
}

# ── 第二层：密钥/凭证检测（代码语料最常见泄露源） ─────────────────────
_SECRET_PATTERNS: dict = {
    "openai_key":     r"sk-[A-Za-z0-9]{20,}",
    "github_token":   r"ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{82}",
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "pem_key":        r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----",
    "generic_secret": r"(?i)(?:api[_\-]?key|access[_\-]?token|secret[_\-]?key)[\s]*[=:\"']+[\s]*[A-Za-z0-9_\-]{16,}",
}

# 上下文 allowlist：命中此模式时跳过脱敏（placeholder/示例值）
_ALLOWLIST = [
    r"(?i)example\.|placeholder|your[-_]?api|<YOUR",
    r"xxx|XXXX|\*{4,}|\[REDACTED\]",
    r"12345678901|test@example",
]

def _is_allowlisted(window: str) -> bool:
    return any(re.search(p, window) for p in _ALLOWLIST)

def _apply_pattern_layer(text: str, patterns: dict, found: list,
                          confidence: str = "high") -> str:
    """按规则字典逐一扫描，逆序替换以维护字符偏移正确性。"""
    for pii_type, pattern in patterns.items():
        for m in reversed(list(re.finditer(pattern, text))):
            window = text[max(0, m.start()-40): m.end()+40]
            if _is_allowlisted(window):
                continue
            tag = f"[{pii_type.upper()}_REDACTED]"
            found.append(PiiMatch(pii_type, m.start(), m.end(),
                                  m.group(), tag, confidence))
            text = text[:m.start()] + tag + text[m.end():]
    return text

def detect_and_redact_pii(
    text: str,
    ner_model=None,
    audit_log: Optional[list] = None,
) -> tuple[str, list]:
    """
    三阶段 PII 检测与脱敏管线。

    - 阶段1（结构化正则）：手机号、身份证、邮箱、IPv4，高精度高速。
    - 阶段2（密钥/凭证）：API Key、PEM、GitHub Token，覆盖代码语料泄露。
    - 阶段3（NER 命名实体）：人名/地点/机构，需传入 spaCy 模型；
      推荐使用 ``spacy.load("zh_core_web_trf")``，未传入时本阶段跳过。

    Args:
        text:       待脱敏原文
        ner_model:  已加载的 spaCy 模型（可选）
        audit_log:  若传入 list，追加 {text_hash, pii_types, count} 供审计

    Returns:
        (脱敏后文本, PiiMatch 列表)

    生产注意：正则对格式相近的示例数字存在误报风险，通过 allowlist 可缓解；
    NER 对单字人名和非标准地址召回有限，建议配合人工抽检（抽检率 ≥ 0.5%）。
    """
    found: list = []
    text = _apply_pattern_layer(text, _STRUCTURAL_PATTERNS, found, "high")
    text = _apply_pattern_layer(text, _SECRET_PATTERNS, found, "high")

    if ner_model is not None:
        doc = ner_model(text)
        for ent in reversed(doc.ents):
            if ent.label_ not in {"PER", "LOC", "ORG", "PERSON", "GPE"}:
                continue
            window = text[max(0, ent.start_char-40): ent.end_char+40]
            if _is_allowlisted(window):
                continue
            tag = f"[NER_{ent.label_}_REDACTED]"
            found.append(PiiMatch(ent.label_, ent.start_char, ent.end_char,
                                  ent.text, tag, "medium"))
            text = text[:ent.start_char] + tag + text[ent.end_char:]

    if audit_log is not None and found:
        import hashlib
        audit_log.append({
            "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "pii_types": sorted({m.pii_type for m in found}),
            "count":     len(found),
        })
    return text, found
```

**命名实体识别（NER）模型**则覆盖规则难以枚举的 PII 类型，如真实人名、地址和机构名。推荐使用 spaCy (Honnibal et al. 2023) 的中文模型（`zh_core_web_trf`）或 HuggingFace 上开源的中文 NER 模型，对人名（PER）、地点（LOC）、机构（ORG）等命名实体进行识别，再根据上下文判断是否需要脱敏。

---

## 5.5 基准污染检测与去污染

Benchmark Contamination（基准污染），是指训练数据中意外混入了测试/评测集的题目和答案，导致模型在这些评测集上的表现被人为虚高的现象。这是 LLM 训练数据质量治理中最敏感的诚信问题，也是近年来随着大模型评测体系日趋成熟而被越来越重视的工程挑战。

### 5.5.1 污染的传播路径

污染的发生路径通常出人意料：评测集题目出现在某个公开的技术博客中，博客被爬虫纳入训练语料；学术社区讨论评测集结果的帖子包含了原题，通过 Reddit / 知乎等论坛数据进入语料库；评测集的早期版本作为示例被纳入了公开的 GitHub 仓库，通过代码数据进入语料库。这些路径的共同特点是**间接性**——没有人故意把测试集放进训练数据，但污染还是自然地发生了。

### 5.5.2 检测与隔离方案

目前最常用的去污染方案是 **N-gram 重叠检测**：将所有评测集（MMLU (Hendrycks et al. 2021)、GSM8K (Cobbe et al. 2021)、HumanEval (Chen et al. 2021)、CEVAL 等）的题目和答案预先计算 N-gram 指纹集合，然后对训练数据中的每个文档进行扫描；当重叠率显著高于背景噪声或超过项目预先设定的污染风险阈值时，就将该文档移入隔离区（不是直接删除，而是先隔离，以便后续审查）。N 的取值和重叠阈值需要随评测集长度、语言、分词方式和误报成本校准，不能作为通用常数套用：

代码清单5-5展示了评测集 N-gram 指纹构建与重叠率计算的示意实现。

*代码清单5-5：评测集 N-gram 指纹与污染率计算示意代码。N 值和阈值需随评测集长度、语言、分词方式和误报成本校准。*

```python
from collections import Counter

import re

def _tokenize(text: str) -> list[str]:
    """语言自适应分词：中文字符级切分，其余空格切分后合并。
    适用于中英混合语料的 N-gram 污染检测。"""
    # 将中文字符切为单字，英文/数字保留为整词
    tokens = []
    for segment in re.split(r'([一-鿿㐀-䶿]+)', text.lower()):
        if re.fullmatch(r'[一-鿿㐀-䶿]+', segment):
            tokens.extend(list(segment))      # 中文：字符级
        else:
            tokens.extend(segment.split())    # 英文：空格切分
    return tokens

def build_eval_ngrams(eval_texts: list[str], n=13) -> set[str]:
    """构建评测集的 N-gram 指纹集合（支持中英混合文本）"""
    ngrams = set()
    for text in eval_texts:
        tokens = _tokenize(text)
        ngrams.update(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return ngrams

def contamination_score(doc: str, eval_ngrams: set[str], n=13) -> float:
    """计算文档与评测集的 N-gram 重叠率（支持中英混合文本）"""
    tokens = _tokenize(doc)
    if len(tokens) < n:
        return 0.0
    doc_ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    if not doc_ngrams:
        return 0.0
    hits = sum(1 for g in doc_ngrams if g in eval_ngrams)
    return hits / len(doc_ngrams)
```

去污染工作应在建立正式训练集之前系统完成，而不是边训练边修补。由于评测集范围会随时间扩大，新评测集持续出现，工程团队需要定期更新指纹库并重新扫描。

---

## 5.6 质量评分、抽检与闭环迭代

### 5.6.1 多维质量评分与分层采样

清洗不应该是"非黑即白"的二元判断，而应当对每个文档给出一个多维质量评分向量，用于后续的分层采样：

代码清单5-6展示了多维文档质量评分对象的示意定义。

*代码清单5-6：多维文档质量评分对象示意代码。示例阈值用于说明分层思想，生产环境应基于历史分布、代理模型评测和人工审计共同设定。*

```python
from dataclasses import dataclass

@dataclass
class DocumentQualityScore:
    doc_id: str
    noise_score: float      # 噪声分（特殊字符比例等），越低越好
    ppl_score: float        # 困惑度，越低质量越高
    dedup_status: str       # "unique" / "near-duplicate" / "exact-duplicate"
    pii_found: list[str]    # 发现的 PII 类型列表，空列表表示干净
    contamination_rate: float  # 基准污染率，越低越好

    @property
    def quality_tier(self) -> str:
        """基于多维评分确定质量层级"""
        if self.ppl_score < 200 and not self.pii_found and self.contamination_rate < 0.05:
            return "high"
        if self.ppl_score < 500 and len(self.pii_found) <= 1:
            return "medium"
        return "low"
```

分层采样策略：High 层数据在训练中给予 2x 采样权重，Medium 层 1x，Low 层 0.3x（而非直接丢弃，保持多样性）。

### 5.6.2 人工抽检闭环

质量闭环的设计思路是**人工审计驱动规则迭代**，而非"人工处理每条数据"（后者在 PB 级语料下根本不可行），完整流程见图5-2。

![图5-2：质量过滤漏斗与抽检闭环图](../../images/part2/quality_filter_funnel_loop.png)

*图5-2：质量过滤漏斗与抽检闭环 —— 左侧漏斗展示每阶段的数据留存率，右侧闭环展示人工抽检如何驱动过滤规则的持续迭代优化。来源：本书自绘；Alt text：质量过滤漏斗与抽检闭环图，展示规则过滤、模型评分、去重、人工抽检和规则回写之间的循环关系。*

每个清洗批次完成后，固定执行以下"质量快照"流程：随机抽取一批样本，由数据工程师进行人工标注（OK / 噪声 / PII 遗漏 / 误杀的高质量内容 / 近似重复漏网），统计各类错误的发生率，并追踪是哪个过滤步骤导致了该错误（误杀 or 漏检）。当某类错误率连续多个批次超过项目水位线，必须触发对应规则或模型阈值的审查和更新。这套机制将清洗管线从"一次性工程产物"转变为"持续迭代的质量引擎"。

---

## 5.7 常见缺陷、检测方法与代价对照

*表5-1：常见缺陷、检测方法与代价表。来源：本书整理，检测代价为相对工程复杂度描述，实际成本取决于数据规模、模型调用和基础设施配置。*

| 缺陷类型 | 典型表现 | 检测方法 | 漏检代价 | 推荐阈值/工具 |
| :--- | :--- | :--- | :--- | :--- |
| **HTML/噪声残留** | 标签如 `<div>`、CSS、JS 代码混入正文 | 特殊字符比例分位数；正则规则 | 模型输出乱码/标签 | 以人工确认样本校准项目水位 |
| **语言错误** | 目标语言外的内容混入 | FastText 语言识别及置信度分布 | 模型学习错误语言分布 | 按目标语言和来源设置置信度水位 |
| **低信息密度** | SEO 关键词堆砌、广告文案、无意义重复 | KenLM PPL 分布；质量分类器 | 模型输出空洞、凑字数文本 | 在目标语料上建立 PPL 分位数基线 |
| **精确重复** | 相同文档被多次爬取 | SHA-256 哈希全局去重 | 模型过拟合特定内容 | 相同哈希仅保留 1 份 |
| **近似重复** | 同一文章在不同网站转载（略有改动）| MinHash LSH（Jaccard 相似度） | "复读机"、泛化差 | 通过消融实验选择 Jaccard 判重水位 |
| **PII 泄露** | 手机号、身份证、邮箱、API Key | 正则规则 + NER 模型 + 人工抽检 | 上线后隐私事故 | 零容忍，人工复核 |
| **基准污染** | 测试集题目混入训练集 | 13-gram 与评测集比对 | 评测分虚高，诚信风险 | 高重叠样本隔离并人工复核 |
| **低词汇多样性** | Type-Token Ratio 极低（循环体文本）| TTR 分布异常 | 模型词汇使用僵化 | 按语言和内容类型设置 TTR 基线 |

*表5-2：清洗动作对训练效果影响对照。来源：本书整理，影响方向为工程经验归纳，具体收益需通过同配置训练或代理模型实验验证。*

注：表5-2用于说明清洗动作与风险缓解方向的对应关系，不给出跨项目固定收益。实际效果取决于语料结构、模型规模、评测集、清洗阈值和训练配置，应通过消融实验验证。

| 清洗动作 | 不做时的典型模型症状 | 完整做时的风险缓解方向 | 成本周期 |
| :--- | :--- | :--- | :--- |
| 语言过滤 | 模型混用语言；中文回答夹杂英文 | 语言一致性提升 | CPU，数小时 |
| 启发式规则过滤 | 模型输出格式混乱（HTML标签/广告词） | 降低格式噪声和模板文本污染 | CPU，数小时 |
| PPL 困惑度过滤 | 模型倾向于生成空洞、凑字数内容 | 提高语料信息密度，隔离乱码和机器生成垃圾 | CPU+小模型，数天 |
| MinHash 模糊去重 | "复读机"现象；生成内容重复率高 | 降低重复样本对概率分布的过度强化 | CPU分布式，数天 |
| PII 脱敏 | 上线后隐私泄露事故、模型背出用户信息 | 隐私合规达标，避免法律风险 | CPU+GPU NER，数天 |
| 基准去污染 | 评测分虚高；真实用户体验与评测分脱钩 | 评测诚信达标；真实场景表现更可预期 | CPU，数小时 |
| 质量分层采样 | 高低质量数据同权重稀释高质量效果 | 让有限训练预算更多消耗在高价值样本上 | 无额外计算成本 |

---

## 5.8 大规模工程案例与问题复盘

以下案例均为匿名化复合案例，数据规模、比例和周期用于说明工程口径；截至 2026-06，实际数值会随语料来源、清洗规则、模型规模和评测方法变化。

### 案例一：清洗过度导致知识损失——一次"阈值调过头"的代价（匿名化复合案例）

**背景**：某团队在完成了首轮基座模型预训练后，计划对数据清洗管线进行升级，目标是进一步提升训练数据的"纯净度"。团队将启发式过滤规则提升了标准：最小文档长度显著提高，PPL 阈值大幅收紧，MinHash 相似度阈值也变得更激进。处理后，语料库规模明显缩减。

**T+0（发现问题）**：使用新版语料训练的模型在通用评测基准上表现下降——尤其是在医学、法律等专业领域，回答质量明显不如旧版。工程师起初怀疑是训练超参数问题。

**T+5（根因定位）**：经过对新旧语料的差异分析，发现大量专业领域文档（如医学科普、法规条文、技术标准）被过度清洗所淘汰：医学科普文章常常短而密集，低于新的最低长度阈值；法律条文语言高度规范，可能在 PPL 过滤中被误判；不同网站转载的同一法规内容相似度高，又容易被激进 MinHash 阈值大量删除，导致整个法条知识库残缺不全。

**关键教训**：清洗阈值应当**分领域、分内容类型进行差异化配置**，而非全局统一调参。专业领域内容（医学、法律、科学）的知识密度高，但其文档特征（长度分布、语言规范性、内容相似度）与通用网页有本质区别，用设计给通用网页的阈值来过滤专业内容，必然产生严重的知识损失。正确的做法是：先对语料进行领域分类，再为不同领域配置独立的清洗阈值参数，并对每个领域的处理结果进行独立的人工抽检验证。

---

### 案例二：PII 遗漏引发的安全风险（匿名化复合案例）

**背景**：某公司在将一套面向企业用户的 AI 助手产品上线后，很快收到用户反馈：在询问某些技术相关问题时，模型会在生成的代码示例中输出看起来像真实 API Key 的字符串（格式为 `sk-xxxxxxxxxxxxxxxxxxxxxxxx`）。

**T+0（风险确认）**：安全团队立即展开排查，确认模型输出与真实密钥格式高度一致，疑似来自训练数据中某个 GitHub 仓库里被提交的硬编码密钥。由于 PII 脱敏管线只覆盖了手机号、邮箱、身份证等常规类型，未将 API Key 纳入检测范围，这批密钥在训练过程中被完整学习。

**T+1（紧急处置）**：安全团队通知密钥所属服务商进行紧急吊销，同时下线模型接受审查。经排查，代码语料中存在一批包含 API Key 或密码硬编码的提交记录，覆盖云服务密钥、代码托管平台 Token、数据库连接字符串等多种类型，均未被现有脱敏管线捕获。

**T+7（修复完成）**：数据团队补充了针对 API Key 和密码等结构化秘密的正则规则（参考 GitGuardian 的开源规则集），同时引入了专门检测代码中机密泄露的工具（如 truffleHog、detect-secrets），对代码语料进行了全量重扫和重新脱敏。

**关键教训**：PII 的定义应当随语料类型的扩展而扩展。代码语料中的"机密"（API Key、密码、SSH 私钥）与普通文本中的"隐私"（手机号、邮箱）是不同类型的 PII，前者危害更直接、更立竿见影，但更容易被忽视。任何接入代码语料的团队，必须在 PII 脱敏管线中专门为"机密检测"（Secrets Detection）增加独立规则集。

---

## 5.9 生产级清洗管线的最小可行组合

在理解了本章所有技术模块之后，一个现实的问题随之产生：**如果资源有限，哪些步骤是不可省略的？** 并非所有团队都有足够的工程资源在第一版就完整实现六阶段清洗管线。以下给出针对三种不同资源水位的"最小可行清洗组合"，供工程团队根据自身情况参考选用。

### 5.9.1 轻量级方案（1-3 人数据团队，数据规模 < 100GB）

轻量级方案聚焦于"守住底线"，用最少的工程投入过滤掉危害最大的缺陷：

*表5-3：轻量级清洗方案最小可行组合。来源：本书整理，方案组合为起步建议，生产环境应按风险等级、语料来源和合规要求扩展。*

| 步骤 | 实现方案 | 工具 | 是否必须 |
|:--- |:--- |:--- |:--- |
| 语言过滤 | FastText 识别，置信度阈值按语种校准 | fasttext | ★ 必须 |
| 规则过滤 | 长度、特殊字符、重复行 | 自实现 Python | ★ 必须 |
| 精确去重 | SHA-256 哈希全局去重 | hashlib | ★ 必须 |
| PII 脱敏 | 正则规则（手机/邮箱/身份证/API Key） | re | ★ 必须 |
| 文本标准化 | Unicode NFC + 空白清理 | unicodedata | ★ 必须 |
| 困惑度过滤 | KenLM（可选，有时间再做） | kenlm | △ 推荐 |
| MinHash 去重 | 可选（数据规模小时效果有限） | datasketch | ○ 可选 |
| 基准去污染 | 必须在正式训练前完成 | 自实现 | ★ 必须 |

这套组合通常可以作为早期实验阶段的起步方案，覆盖"必须有"的底线防护。代价是会遗漏相当比例的近似重复内容和低信息密度文档，因此不应直接外推到正式预训练数据发布。

### 5.9.2 标准方案（4-10 人数据团队，数据规模 100GB - 10TB）

标准方案在轻量级基础上增加了模型评分和模糊去重，覆盖了工业实践中的主流质量需求：

在轻量级方案的基础上，补充：**KenLM 困惑度过滤**（拟合 5-gram 语言模型，针对目标语言训练，并在人工确认样本上设定 PPL 分位数水位）；**MinHash LSH 模糊去重**（Jaccard 阈值、签名维度和 band 数通过样本复核与代理训练消融确定）；**NER 模型辅助 PII**（spaCy 中文模型或同类 NER 模型，覆盖人名/地址/机构等规则难以枚举的 PII 类型）；**领域分层阈值**（为代码、学术论文等特殊内容类型配置独立过滤参数，避免统一阈值误删）。这套方案需要的工程周期和 GPU 资源取决于语料规模、工具链成熟度和复核强度，可作为中型团队的完整基线方案。

### 5.9.3 平台级方案（10+ 人数据平台团队，数据规模 > 10TB）

平台级方案面向工业级大规模数据处理，在标准方案之上进一步引入：**分布式处理架构**（Ray Data 或 Spark on Kubernetes 实现所有步骤的完全分布式化，支持多节点水平扩展）；**自定义质量分类器**（用人工标注的高/低质量样本对，微调 BERT 或 fastText 分类器，将文档质量判断做为强监督的分类任务）；**全量评测集去污染**（维护包含所有主流评测集的 N-gram 指纹库，并定期更新）；**自动化质量快照仪表盘**（每批次清洗完成后自动生成质量报告，展示各阶段过滤率、质量分分布、PII 发现率等关键指标）。完整平台级方案的搭建周期取决于团队规模、平台基础和合规要求；一旦建立，可以支持公司所有大模型项目的语料质量基础设施共享复用。



---

## 本章小结

本章围绕"清洗为何构成训练数据质量上限"展开，按照清洗生命周期的顺序，系统介绍了规则过滤、模型评分、精确去重、MinHash 模糊去重、PII 脱敏与基准去污染的完整技术体系。两张表格（表5-1 缺陷-检测-代价矩阵、表5-2 清洗动作效果对照）为工程师提供了可直接参考的决策工具。两个匿名化复合案例——"清洗过度导致知识损失"和"PII 遗漏引发安全风险"——从正反两个方向印证了清洗体系的精细化配置要求。

完成清洗、去重与去污染后，原始语料才具备进入训练输入组织阶段的基础条件。下一章将在清洗完成的数据上，继续讨论预训练数据工程的最后一公里：**第6章 分词、序列化与高效加载**，即如何把干净文本转化为 GPU 可以高效消费的 Token 序列。

## 参考文献

Broder A Z (1997) On the Resemblance and Containment of Documents. In: Proceedings of the Compression and Complexity of Sequences, pp 21-29.

Heafield K (2011) KenLM: Faster and Smaller Language Model Queries. In: Proceedings of the Sixth Workshop on Statistical Machine Translation, pp 187-197.

Honnibal M, Montani I, Van Landeghem S, Boyd A (2023) explosion/spaCy: v3.7.2: Fixes for APIs and requirements. Zenodo. <https://doi.org/10.5281/zenodo.1212303>.

Indyk P, Motwani R (1998) Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality. In: Proceedings of the 30th Annual ACM Symposium on Theory of Computing, pp 604-613.

Joulin A, Grave E, Bojanowski P, Douze M, Jegou H, Mikolov T (2017) FastText.zip: Compressing Text Classification Models. arXiv preprint arXiv:1612.03651.

Penedo G, Kydlíček H, Ben Allal L, Lozhkov A, Mitchell M, Raffel C, von Werra L, Wolf T (2024) The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. arXiv preprint arXiv:2406.17557.

Penedo G, Malartic Q, Hesslow D, Cojocaru R, Cappelli A, Alobeidli H, Pannier B, Almazrouei E, Launay J (2023) The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data Only. In: Advances in Neural Information Processing Systems 36.

Soldaini L, Kinney R, Bhagia A, Schwenk D, Atkinson D, Authur R, Bogin B, Chandu K, Dumas L, Elazar Y, others (2024) Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research. arXiv preprint arXiv:2402.00159.


Cobbe K, Kosaraju V, Bavarian M, Chen M, Jun H, Kaiser L, Plappert M, Tworek J, Hilton J, Nakano R, Hesse C, Schulman J (2021) Training Verifiers to Solve Math Word Problems (GSM8K). arXiv preprint arXiv:2110.14168.

Hendrycks D, Burns C, Basart S, Zou A, Mazeika M, Song D, Steinhardt J (2021) Measuring Massive Multitask Language Understanding (MMLU). In: International Conference on Learning Representations.

Chen M, Tworek J, Jun H, Yuan Q, Pinto H P d O, Kaplan J, Edwards H, Burda Y, Joseph N, Brockman G, others (2021) Evaluating Large Language Models Trained on Code (HumanEval). arXiv preprint arXiv:2107.03374.

Nait Saada T, Bethune L, Klein M, Grangier D, Cuturi M, Ablin P (2025) The Data-Quality Illusion: Rethinking Classifier-Based Quality Filtering for LLM Pretraining. arXiv preprint arXiv:2510.00866.
