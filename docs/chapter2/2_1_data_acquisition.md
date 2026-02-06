# 第3章：数据获取与采集

---

## 本章摘要

预训练数据是大模型的"燃料"，其质量和规模直接决定了模型的基础能力。本章将深入探讨预训练数据的获取策略，从 Common Crawl 等开源数据集的解构与使用，到高性能网页爬虫系统的设计与实现，再到代码、论文、书籍等特种数据的获取技巧。掌握这些内容后，读者将具备构建 TB 级预训练语料库的能力。

---

## 场景引入

你的团队决定训练一个 7B 参数的中文基座模型。按照 Chinchilla 的最优配比，这需要约 140B Token 的高质量中文语料——换算成原始文本大约是 280TB。去哪里找这么多数据？直接从网上爬取显然不现实，一个小团队不可能在短时间内爬完整个中文互联网。

这时有人提议使用 Common Crawl——这是一个每月抓取数十亿网页的开源项目，累计数据量超过 PB 级。听起来是完美的解决方案，但当你真正下载一个月的数据后，发现原始的 WARC 文件完全无法直接使用：充斥着 HTML 标签、JavaScript 代码、导航栏、广告，真正有价值的正文内容可能不到 10%。

如何从这片"数据沼泽"中提取出可用于训练的"黄金语料"？这正是本章要解决的核心问题。

---

## 3_1 开源数据集解构

在动手爬取数据之前，首先应该充分利用已有的开源数据集。这些数据集经过社区的精心处理，可以大大降低预训练数据准备的时间和成本。理解它们的构成和处理方法，也是设计自己数据管线的重要参考。

### 3_1.1 Common Crawl：互联网的快照

Common Crawl 是一个非营利组织，自 2008 年起持续抓取互联网网页，并免费提供给研究和商业使用。它是目前绝大多数大规模预训练数据集的上游来源，无论是 GPT 系列、LLaMA 还是国内的各种大模型，都或多或少地使用了 Common Crawl 的数据。

Common Crawl 的数据以"抓取批次"（Crawl）为单位组织，每月发布一个新批次，每个批次包含数十亿个网页。数据以三种格式提供：WARC（Web ARChive）文件包含原始的 HTTP 响应，包括响应头和完整的 HTML 内容，是最原始也最完整的格式；WAT 文件是 WARC 的元数据提取版本，包含 URL、响应头、链接关系等结构化信息；WET 文件是纯文本提取版本，已经去除了 HTML 标签，只保留正文文本。

| 格式 | 内容 | 单月数据量 | 适用场景 |
|------|------|-----------|----------|
| WARC | 原始 HTTP 响应 + HTML | 约 80TB 压缩 | 需要完整内容或自定义解析 |
| WAT | 结构化元数据 | 约 3TB 压缩 | URL 分析、链接图研究 |
| WET | 纯文本提取 | 约 15TB 压缩 | 快速获取文本、初步实验 |

![图3-1：Common Crawl数据流水线](../images/chapter2/图3_1_CommonCrawl数据流水线.png)

*图3-1：Common Crawl 数据流水线 —— 从互联网抓取到清洁语料的完整处理流程*

对于预训练数据工程而言，WARC 和 WET 是最常用的两种格式。WET 文件看似方便，因为已经提取了纯文本，但实际上 Common Crawl 的默认文本提取质量较差，会保留大量噪声（如导航栏、页脚、JavaScript 文本）。因此，专业的数据处理流程通常从 WARC 文件开始，使用更高质量的解析器（如 Trafilatura）重新提取正文。

使用 Common Crawl 数据需要注意几个关键点。首先是版本选择：每月的抓取批次质量略有差异，一般建议使用较新的版本（如 2023 年以后的批次），因为 Common Crawl 持续改进其抓取策略。其次是语言过滤：Common Crawl 以英文网页为主，中文、日文等非英语内容占比较低（通常不到 10%），需要额外的语言识别步骤进行筛选。最后是法律合规：虽然 Common Crawl 是开放数据，但其中抓取的网页内容可能涉及版权问题，使用时需要根据当地法律进行评估。

### 3_1.2 RefinedWeb：高质量英文语料

RefinedWeb 是 Falcon 模型团队（UAE 的 TII 实验室）发布的高质量英文预训练数据集，包含约 5T Token。与直接使用 Common Crawl 不同，RefinedWeb 经过了严格的清洗和去重处理，被认为是目前公开可用的最高质量英文预训练语料之一。

RefinedWeb 的处理流程具有很强的参考价值。其核心步骤包括：URL 过滤（移除成人网站、垃圾站点等）、文本提取（使用 Trafilatura 进行高质量正文提取）、语言识别（使用 FastText 保留英文内容）、质量过滤（基于启发式规则移除低质量文档）、模糊去重（使用 MinHash LSH 在大规模数据上进行近似去重）。

RefinedWeb 论文详细记录了每个步骤的实现细节和效果评估，是学习预训练数据处理的绝佳教材。值得注意的是，虽然 RefinedWeb 公开了数据集的一个子集（约 600B Token），但完整版本仍保留为 Falcon 模型的独家使用。

### 3_1.3 The Pile：多元化的数据混合

The Pile 是 EleutherAI 发布的开源预训练数据集，规模约 800GB（解压后），包含约 300B Token。与 RefinedWeb 专注于网页数据不同，The Pile 的设计理念是多元化——它混合了来自 22 个不同来源的数据，覆盖网页、书籍、代码、论文、法律文件等多个领域。

The Pile 的数据来源构成反映了预训练数据多样性的重要性。其中 Pile-CC 是经过清洗的 Common Crawl 子集，占比约 50%；PubMed Central 提供生物医学论文；ArXiv 提供科学预印本；GitHub 提供开源代码；Books3 提供书籍文本；StackExchange 提供技术问答；Wikipedia 提供百科知识。这种多来源混合策略被证明能够提升模型在各类下游任务上的表现，后来的 LLaMA、Mistral 等模型的数据配方都借鉴了类似的思路。

然而，The Pile 在法律层面存在争议。其中的 Books3 子集包含大量版权书籍，已引发多起诉讼。在使用 The Pile 时，建议根据自身的法律风险承受能力进行评估，或者选择性地排除争议子集。

### 3_1.4 中文数据集概览

对于训练中文大模型，可用的开源数据集相对较少，但近年来有所改善。

WuDaoCorpora 是智源研究院发布的大规模中文语料库，包含约 3TB 的中文文本，涵盖百科、新闻、论坛、问答等多种来源。数据需要通过申请获取，使用时需遵守相关协议。ChineseCrawl 是中文版的 Common Crawl 子集提取，有多个社区版本可用。CLUECorpus 是 CLUE 基准团队发布的中文语料，规模约 100GB，适合中小规模实验。

相比英文数据集的丰富程度，中文预训练数据仍然是一个"卖方市场"。这意味着中文大模型团队往往需要自行从 Common Crawl 或其他来源获取和处理中文数据，无法完全依赖现有开源数据集。

---

## 3_2 高性能网页解析

从 Common Crawl 或自有爬虫获取原始 HTML 后，下一个关键步骤是从中提取正文文本。这看似简单，实际上是整个数据管线中最关键的环节之一——解析质量直接决定了最终语料的质量。

### 3_2.1 网页解析的挑战

现代网页远比表面看起来复杂。一个典型的网页可能包含：HTML 结构标签、CSS 样式定义、JavaScript 代码（包括内联和外部引用）、导航栏和页脚、广告和推广内容、评论区和用户生成内容、页面侧边栏和推荐内容。而我们需要的只是"正文"——即页面的主体内容部分。

传统的解析方法（如简单地去除所有 HTML 标签）效果很差，因为它无法区分正文和噪声。更高级的方法需要理解网页的语义结构，识别哪些部分是真正有价值的内容。

### 3_2.2 Trafilatura：工业级的解析库

Trafilatura 是目前最受推荐的网页正文提取库，被 RefinedWeb、Dolma 等主流数据集采用。它的核心优势在于：精心调优的提取算法，在多个评测数据集上表现优异；良好的多语言支持，特别是对中文、日文等亚洲语言的处理；丰富的配置选项，可以根据需求调整提取策略；合理的性能表现，适合大规模数据处理。

使用 Trafilatura 的基本流程如下：

```python
import trafilatura

# 从 HTML 中提取正文
def extract_content(html: str, url: str = None) -> dict:
    """
    从 HTML 提取正文内容
    
    Args:
        html: 原始 HTML 字符串
        url: 可选的 URL，用于解析相对链接
    
    Returns:
        包含正文和元数据的字典
    """
    # 核心提取
    result = trafilatura.extract(
        html,
        url=url,
        include_comments=False,    # 排除评论区
        include_tables=True,       # 保留表格内容
        no_fallback=False,         # 允许使用备选算法
        favor_precision=True,      # 优先保证精确度
        output_format='txt'        # 输出纯文本
    )
    
    # 提取元数据
    metadata = trafilatura.extract_metadata(html)
    
    return {
        'text': result,
        'title': metadata.title if metadata else None,
        'author': metadata.author if metadata else None,
        'date': metadata.date if metadata else None,
        'url': url
    }
```

Trafilatura 提供了丰富的配置选项，不同的参数组合适用于不同的场景。`include_comments` 控制是否保留页面评论区内容，对于论坛类网站可以设为 True，对于新闻网站通常设为 False。`include_tables` 控制是否保留表格，对于数据类页面（如 Wikipedia）应该设为 True。`favor_precision` 和 `favor_recall` 是一对权衡参数，前者优先保证提取内容的准确性（宁可漏掉也不要错误），后者优先保证提取的完整性（宁可有噪声也要完整）。对于预训练数据，通常选择 `favor_precision=True`，因为噪声数据对模型训练有害。

### 3_2.3 其他解析工具对比

除了 Trafilatura，还有几个常用的网页解析工具，各有特点。

**Readability** 最初由 Mozilla 开发，用于 Firefox 的阅读模式。它的算法相对简单，速度快，但对复杂页面的处理效果一般。Python 生态中有 readability-lxml 等移植版本。

**Newspaper3k** 专门针对新闻网站优化，能够较好地提取文章标题、正文、发布日期、作者等信息。但对非新闻类网站的效果较差，且项目维护不够活跃。

**Justext** 是一个专注于"样板文本去除"（boilerplate removal）的库，算法基于文本块的链接密度和文本密度。它在学术研究中较常引用，但工程实用性不如 Trafilatura。

| 工具 | 优势 | 劣势 | 推荐场景 |
|------|------|------|----------|
| Trafilatura | 综合效果最佳，多语言支持好 | 速度中等 | 通用场景，首选 |
| Readability | 速度快，算法简单 | 复杂页面效果差 | 快速原型 |
| Newspaper3k | 新闻网站效果好 | 泛化能力弱 | 新闻语料专项 |
| Justext | 学术验证充分 | 工程适配较少 | 研究场景 |

![图3-2：解析器质量对比](../images/chapter2/图3_2_解析器质量对比.png)

*图3-2：网页解析器质量对比 —— Trafilatura 在 F1 分数上领先，是 LLM 数据处理的首选工具*

在实际项目中，一个常见的策略是使用 Trafilatura 作为主解析器，当提取结果为空或过短时，回退到 Readability 进行尝试。这种"主备"策略可以提高整体的提取成功率。

### 3_2.4 分布式解析架构

单机处理能力有限，面对 TB 级的 WARC 文件，需要构建分布式解析系统。以下是一个基于 Ray Data 的分布式解析示例：

```python
import ray
import trafilatura
from warcio.archiveiterator import ArchiveIterator
import gzip

ray.init()

def parse_warc_record(record):
    """解析单条 WARC 记录"""
    if record.rec_type != 'response':
        return None
    
    url = record.rec_headers.get_header('WARC-Target-URI')
    content_type = record.http_headers.get_header('Content-Type', '')
    
    # 只处理 HTML 页面
    if 'text/html' not in content_type:
        return None
    
    try:
        html = record.content_stream().read().decode('utf-8', errors='ignore')
        text = trafilatura.extract(html, url=url, favor_precision=True)
        
        if text and len(text) > 200:  # 过滤过短的内容
            return {
                'url': url,
                'text': text,
                'length': len(text)
            }
    except Exception as e:
        return None
    
    return None

def process_warc_file(warc_path: str):
    """处理单个 WARC 文件"""
    results = []
    
    with gzip.open(warc_path, 'rb') as f:
        for record in ArchiveIterator(f):
            result = parse_warc_record(record)
            if result:
                results.append(result)
    
    return results

# 获取所有 WARC 文件路径
warc_files = [...]  # S3 或本地路径列表

# 分布式并行处理
ds = ray.data.from_items(warc_files)
ds = ds.flat_map(process_warc_file)

# 保存结果
ds.write_parquet("s3://bucket/parsed_data/")
```

这个架构的关键设计点包括：使用 Ray Data 的 `flat_map` 算子实现文件级并行；在解析函数内部进行错误处理，避免单条数据的失败影响整批处理；通过 `len(text) > 200` 等条件进行早期过滤，减少下游处理量；输出为 Parquet 格式，便于后续的去重和过滤步骤。

---

## 3_3 特种数据获取

除了通用的网页数据，预训练语料库通常还需要包含代码、学术论文、书籍等专门领域的数据。这些"特种数据"的获取和处理有其独特的挑战和技巧。

### 3_3.1 代码数据：GitHub 与 The Stack

代码能力是现代大模型的核心竞争力之一，而高质量代码数据的获取是实现这一能力的基础。目前最主要的代码数据来源是 GitHub。

直接从 GitHub API 获取代码是可行的，但效率较低且有请求限制。更常见的做法是使用 GitHub 的公开数据镜像。Google BigQuery 托管了 GitHub 公开仓库的完整快照，可以使用 SQL 进行查询和导出。Software Heritage 是一个致力于保存人类软件遗产的组织，维护着 GitHub 的完整归档。

对于大规模代码数据需求，最便捷的选择是使用 BigCode 项目发布的 The Stack 数据集。这个数据集从 GitHub 爬取了超过 300 种编程语言的代码，总规模约 3TB。The Stack 的处理流程包括：基于许可证过滤（只保留开源许可证允许使用的代码）、去重（移除重复的文件和代码片段）、PII 清洗（移除敏感信息）。

使用代码数据时需要特别注意以下几点：

**许可证合规**是首要问题。不同的开源许可证对代码使用有不同的限制。The Stack 数据集提供了许可证标签，可以根据需求进行筛选。对于商业模型训练，建议只使用 MIT、Apache 2_0 等宽松许可证的代码。

**代码质量差异巨大**。GitHub 上既有 Linux 内核这样的高质量项目，也有大量学生作业和个人实验代码。常用的质量过滤策略包括：按仓库 star 数筛选（保留 star > 10 的仓库）、按文件长度过滤（移除过短或过长的文件）、基于 AST 解析检测语法错误。

**处理代码数据的示例**：

```python
import ast
from typing import Optional

def is_valid_python(code: str) -> bool:
    """检查 Python 代码是否语法正确"""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def extract_functions(code: str) -> list:
    """提取代码中的函数定义"""
    try:
        tree = ast.parse(code)
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(ast.unparse(node))
        return functions
    except:
        return []

def filter_code_quality(code: str, 
                        min_lines: int = 10, 
                        max_lines: int = 1000,
                        require_docstring: bool = True) -> Optional[str]:
    """代码质量过滤"""
    lines = code.split('\n')
    
    # 长度过滤
    if not (min_lines <= len(lines) <= max_lines):
        return None
    
    # 语法检查
    if not is_valid_python(code):
        return None
    
    # Docstring 检查（可选）
    if require_docstring and '"""' not in code and "'''" not in code:
        return None
    
    return code
```

### 3_3.2 学术论文：ArXiv 与 S2ORC

学术论文是高质量知识的重要来源，对于提升模型的推理能力和专业知识水平有显著帮助。

**ArXiv** 是最重要的开放获取预印本平台，涵盖物理、数学、计算机科学、生物学等领域的学术论文。ArXiv 提供批量下载服务，可以通过其 S3 桶获取 LaTeX 源文件和 PDF。LaTeX 源文件是更理想的数据来源，因为它保留了论文的结构信息（章节、公式、引用等），且是纯文本格式，易于处理。

处理 ArXiv LaTeX 数据的主要挑战在于 LaTeX 语法的复杂性。一篇论文可能包含数十个 `.tex` 文件，使用自定义的宏和样式。一个实用的简化策略是：只提取主文件（通常是 `main.tex` 或与论文标题相关的文件），使用正则表达式去除图表和复杂公式环境，保留正文文本和简单的数学表达式。

```python
import re
import tarfile
from pathlib import Path

def extract_arxiv_text(latex_content: str) -> str:
    """从 LaTeX 中提取纯文本"""
    text = latex_content
    
    # 移除注释
    text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)
    
    # 移除图表环境
    text = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{table\}.*?\\end\{table\}', '', text, flags=re.DOTALL)
    
    # 简化引用
    text = re.sub(r'\\cite\{[^}]+\}', '[CITATION]', text)
    text = re.sub(r'\\ref\{[^}]+\}', '[REF]', text)
    
    # 移除常见的命令但保留参数
    commands_to_strip = ['textbf', 'textit', 'emph', 'section', 'subsection', 
                         'paragraph', 'title', 'author']
    for cmd in commands_to_strip:
        text = re.sub(rf'\\{cmd}\{{([^}}]+)\}}', r'\1', text)
    
    # 移除其他命令
    text = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # 清理空白
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()
```

**Semantic Scholar Open Research Corpus (S2ORC)** 是 Allen AI 发布的大规模学术论文数据集，包含超过 80M 篇论文的元数据和约 8M 篇论文的全文。相比 ArXiv，S2ORC 覆盖的领域更广，包括 PubMed、ACL Anthology 等多个来源。S2ORC 的全文已经处理为结构化的 JSON 格式，省去了 LaTeX/PDF 解析的麻烦，是快速获取论文数据的便捷选择。

### 3_3.3 书籍数据：版权与替代方案

书籍是高质量长文本的重要来源，对于训练模型的长程理解能力和知识深度有重要价值。然而，书籍数据的版权问题尤为敏感。

**The Pile 中的 Books3** 数据集包含约 20 万本书籍，来自 Bibliotik 等来源。这个数据集已引发多起版权诉讼，多家公司因使用该数据集训练模型而被起诉。在当前的法律环境下，直接使用 Books3 存在显著的法律风险。

**合规的替代方案**包括：

![图3-3：预训练数据来源混合](../images/chapter2/图3_3_预训练数据来源混合.png)

*图3-3：预训练数据来源混合 —— 多源混合策略可提升模型的综合能力*

Project Gutenberg 是一个志愿者项目，提供版权过期的经典书籍。主要是 1928 年前出版的英文书籍，约 70,000 本。数据质量高，但时代较远，现代用语覆盖不足。

Internet Archive 的 Open Library 提供可借阅的电子书。使用时需要遵守其借阅协议，大规模批量获取可能违反服务条款。

维基文库（Wikisource）提供公共领域的文学作品，覆盖多种语言，包括大量中文古籍。

学术教材和开放教育资源（OER）如 OpenStax 提供高质量的教材内容，适合用于构建教育类预训练数据。

对于商业模型训练，最安全的策略是只使用明确获得授权或处于公共领域的书籍数据。虽然这会限制数据规模，但可以避免潜在的法律风险。

### 3_3.4 多语言数据平衡

训练多语言模型时，不同语言的数据可用性差异巨大。英语数据最为丰富，中文、日文、德文等主要语言次之，而小语种的高质量数据则极为稀缺。

数据不平衡会导致模型能力的不均衡。如果简单按原始比例混合，英语将占据绝对优势，小语种几乎学不到。常见的解决策略包括：

**上采样小语种**是最直接的方法，通过重复采样提高小语种数据的权重。但过度重复可能导致过拟合。

**温度采样**是更精细的方法。设定一个温度参数 T，语言 L 的采样概率为 $p_L \propto n_L^{1/T}$，其中 $n_L$ 是该语言的原始数据量。T=1 时退化为原始比例，T→∞ 时接近均匀分布。LLaMA 2 使用了 T=0_3 左右的温度采样。

**质量代替数量**的思路也值得考虑。对于数据稀缺的小语种，可以使用翻译或合成的方式增强数据，但需要注意翻译质量和翻译腔问题。

---

## 3_4 数据获取的工程实践

将上述各个环节串联起来，构建一个完整的数据获取流水线，需要考虑工程层面的诸多细节。

### 3_4.1 爬虫架构设计

对于需要自主爬取数据的场景（而非使用 Common Crawl），分布式爬虫架构的设计至关重要。一个典型的架构包括以下组件：

**URL 管理器**负责维护待爬取 URL 的队列，记录已爬取 URL 的状态，处理 URL 的去重和优先级排序。常用的实现方式包括 Redis 队列配合 Bloom Filter 去重。

![图3-4：分布式爬虫架构](../images/chapter2/图3_4_分布式爬虫架构.png)

*图3-4：分布式网页爬取系统架构 —— URL管理器、下载器集群、解析器与存储层的协作*

**下载器集群**负责实际的 HTTP 请求。关键考虑因素包括：并发控制（避免对目标网站造成过大压力）、代理池管理（应对反爬虫机制）、重试策略（处理网络波动和临时故障）、robots.txt 遵守（尊重网站的爬虫规则）。

**解析器**负责从下载的 HTML 中提取正文和元数据，这部分在上一节已详细讨论。

**存储层**负责持久化爬取结果。对于大规模爬取，建议直接写入对象存储（S3/MinIO），采用 WARC 或 Parquet 格式。

```python
# 简化的爬虫示例
import asyncio
import aiohttp
from urllib.parse import urlparse
import trafilatura

class SimpleCrawler:
    def __init__(self, max_concurrent: int = 100):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.visited = set()
    
    async def fetch(self, session: aiohttp.ClientSession, url: str) -> dict:
        """异步获取并解析单个 URL"""
        if url in self.visited:
            return None
        self.visited.add(url)
        
        async with self.semaphore:
            try:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        return None
                    html = await response.text()
                    text = trafilatura.extract(html, url=url)
                    return {'url': url, 'text': text} if text else None
            except Exception:
                return None
    
    async def crawl(self, urls: list) -> list:
        """批量爬取 URL 列表"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch(session, url) for url in urls]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]
```

### 3_4.2 增量更新策略

预训练数据不是一次性任务。随着时间推移，需要持续吸收新数据以保持模型知识的时效性。增量更新的关键挑战在于：

**识别新数据**：对于 Common Crawl，每月发布新批次，可以直接处理新批次。对于自主爬取，需要维护更新时间戳，定期重新访问已知 URL 检查更新。

**避免重复处理**：已经处理过的数据不应重复进入流水线。可以通过 URL 指纹、内容哈希等方式进行去重。

**版本管理**：每个处理批次应该有明确的版本标识，便于追溯和回滚。这与第二章讨论的数据版本控制（DVC/LakeFS）紧密相关。

### 3_4.3 质量监控与反馈

数据获取流水线需要建立完善的质量监控机制。关键的监控指标包括：

**下载成功率**：失败请求的比例。如果突然升高，可能是目标网站封禁了爬虫。

**解析成功率**：成功提取正文的比例。如果降低，可能是目标网站改变了页面结构。

**平均文档长度**：正文的平均字符数。异常波动可能表示解析器有问题。

**语言分布**：各语言数据的比例。确保符合预期的语言配比。

**重复率**：与历史数据的重复比例。过高的重复率意味着新增数据的边际价值降低。

建议将这些指标集成到监控系统中（如 Prometheus + Grafana），设置告警阈值，及时发现和处理异常。

---

## 3_5 常见陷阱与最佳实践

在数据获取阶段，有几个常见的陷阱值得警惕。

**第一个陷阱是过度依赖单一数据源。** 如果预训练数据全部来自 Common Crawl，模型可能会继承网页数据的偏见和噪声。合理的做法是混合多种来源：网页数据提供广度，书籍和论文提供深度，代码数据提供逻辑能力。The Pile 式的多源混合策略被证明是有效的。

**第二个陷阱是忽视数据的时效性。** Common Crawl 的历史批次虽然量大，但可能包含过时信息。对于需要时效性的应用（如新闻、时事），应该优先使用最近的数据批次。同时，过老的数据可能包含已失效的链接、已更正的错误信息等。

**第三个陷阱是低估合规风险。** 版权问题、隐私问题、robots.txt 违规等，可能在项目后期引发严重的法律问题。最佳实践是在数据获取阶段就建立完善的元数据记录——记录每条数据的来源 URL、获取时间、声称的许可证等信息，为日后可能的审计留下依据。

**第四个陷阱是重采集轻处理。** 许多团队花费大量精力扩大数据采集规模，却在解析和清洗环节草草了事。正如第一章所述，数据质量的重要性远超数据数量。宁可少采集一些，也要确保每份数据经过严格的质量把关。

---

## 3_6 本章小结

本章系统介绍了预训练数据获取的方法论和工程实践。

在开源数据集方面，Common Crawl 是最重要的上游数据源，提供 WARC、WAT、WET 三种格式。RefinedWeb 和 The Pile 是经过精心处理的高质量数据集，其处理方法值得学习借鉴。中文数据集相对稀缺，往往需要自行从 Common Crawl 提取。

在网页解析方面，Trafilatura 是目前最推荐的工业级解析库，能够从复杂的 HTML 中准确提取正文内容。分布式解析架构（如基于 Ray Data）是处理 TB 级数据的必要手段。

在特种数据方面，代码数据可通过 The Stack 或 GitHub BigQuery 获取，需注意许可证合规；学术论文可通过 ArXiv 和 S2ORC 获取；书籍数据的版权风险较高，建议使用公共领域资源。多语言数据需要通过温度采样等策略进行平衡。

在工程实践方面，完整的数据获取流水线需要考虑爬虫架构设计、增量更新策略和质量监控机制。核心原则是多源混合、质量优先、合规先行。

![图3-5：本章知识结构](../images/chapter2/图3_5_本章知识结构.png)

*图3-5：第3章知识结构 —— 涵盖开源数据集、网页解析、特种数据、工程实践四大主题*

---

## 延伸阅读

关于预训练数据获取的深入内容，以下资源值得参考：

Common Crawl 官方文档（commoncrawl.org/the-data）详细介绍了数据格式和获取方式。RefinedWeb 论文（Falcon LLM: A Large Language Model for High-Quality Web Data）详细记录了从 Common Crawl 构建高质量预训练集的完整流程。The Pile 论文（The Pile: An 800GB Dataset of Diverse Text for Language Modeling）介绍了多源混合的数据构建策略。Trafilatura 文档（trafilatura.readthedocs.io）提供了全面的 API 说明和使用示例。The Stack 论文（StarCoder: May the Source Be with You!）介绍了大规模代码数据集的构建方法。

---

## 下一章预告

获取原始数据只是第一步。在下一章《清洗与去噪》中，我们将深入探讨如何从海量原始数据中筛选出高质量内容。你将学习启发式过滤规则（语言识别、困惑度过滤、长度分布）、大规模去重技术（MinHash LSH 的原理与分布式实现），以及隐私数据清洗（PII 识别与移除）。

带着这个问题进入下一章：如果两篇文档有 80% 的内容相同，你如何高效地识别并处理它们？
