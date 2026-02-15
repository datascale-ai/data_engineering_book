# Chapter 3: Data Acquisition and Collection

---

## Chapter Summary

Pre-training data is the "fuel" of large models, and its quality and scale directly determine the model's foundational capabilities. This chapter delves into pre-training data acquisition strategies, from deconstructing and using open-source datasets like Common Crawl, to designing and implementing high-performance web crawler systems, and to acquiring specialized data such as code, papers, and books. After mastering this content, readers will possess the ability to build TB-level pre-training corpora.

---

## Scenario Introduction

Your team has decided to train a 7B parameter Chinese base model. According to Chinchilla's optimal ratio, this requires approximately 140B tokens of high-quality Chinese corpus—translating to about 280TB of raw text. Where can you find this much data? Directly crawling from the web is clearly impractical; a small team cannot crawl the entire Chinese internet in a short time.

At this point, someone suggests using Common Crawl—an open-source project that crawls billions of web pages each month, with cumulative data exceeding PB scale. It sounds like a perfect solution, but when you actually download one month's data, you discover that the raw WARC files are completely unusable: filled with HTML tags, JavaScript code, navigation bars, and ads—the truly valuable body content may be less than 10%.

How do you extract usable "golden corpus" for training from this "data swamp"? This is the core problem this chapter aims to solve.

---

## 3.1 Deconstruction of Open-Source Datasets

Before starting to crawl data yourself, you should first fully leverage existing open-source datasets. These datasets have been carefully processed by the community and can significantly reduce the time and cost of preparing pre-training data. Understanding their composition and processing methods is also an important reference for designing your own data pipeline.

### 3.1.1 Common Crawl: A Snapshot of the Internet

Common Crawl is a non-profit organization that has been continuously crawling internet web pages since 2008 and provides them free of charge for research and commercial use. It is currently the upstream source for the vast majority of large-scale pre-training datasets—whether GPT series, LLaMA, or various Chinese large models, they all use Common Crawl data to varying degrees.

Common Crawl data is organized in "crawl batches," releasing a new batch each month, with each batch containing billions of web pages. The data is provided in three formats: WARC (Web ARChive) files contain raw HTTP responses, including response headers and complete HTML content, making it the most original and complete format; WAT files are metadata-extracted versions of WARC, containing structured information such as URLs, response headers, and link relationships; WET files are plain text extracted versions that have removed HTML tags, preserving only body text.

| Format | Content | Monthly Data Volume | Use Case |
|--------|---------|---------------------|----------|
| WARC | Raw HTTP Response + HTML | ~80TB compressed | Need complete content or custom parsing |
| WAT | Structured metadata | ~3TB compressed | URL analysis, link graph research |
| WET | Plain text extraction | ~15TB compressed | Quick text acquisition, preliminary experiments |

![Figure 3-1: Common Crawl Data Pipeline](../../images/part2/图3_1_CommonCrawl数据流水线.png)

*Figure 3-1: Common Crawl Data Pipeline — Complete processing flow from internet crawling to clean corpus*

For pre-training data engineering, WARC and WET are the two most commonly used formats. WET files seem convenient because text has already been extracted, but in reality Common Crawl's default text extraction quality is poor, retaining large amounts of noise (such as navigation bars, footers, JavaScript text). Therefore, professional data processing pipelines typically start from WARC files and use higher-quality parsers (such as Trafilatura) to re-extract the body content.

Several key points need attention when using Common Crawl data. First is version selection: each monthly crawl batch varies slightly in quality, and it is generally recommended to use more recent versions (such as batches from after 2023), as Common Crawl continuously improves its crawling strategies. Second is language filtering: Common Crawl is predominantly English web pages, with non-English content like Chinese and Japanese accounting for relatively low proportions (typically less than 10%), requiring additional language identification steps for screening. Finally is legal compliance: although Common Crawl is open data, the web page content it crawls may involve copyright issues and needs to be evaluated according to local laws when using.

### 3.1.2 RefinedWeb: High-Quality English Corpus

RefinedWeb is a high-quality English pre-training dataset released by the Falcon model team (TII Lab of UAE), containing approximately 5T tokens. Unlike directly using Common Crawl, RefinedWeb has undergone strict cleaning and deduplication processing and is considered one of the highest quality publicly available English pre-training corpora.

RefinedWeb's processing pipeline has strong reference value. Its core steps include: URL filtering (removing adult websites, spam sites, etc.), text extraction (using Trafilatura for high-quality body extraction), language identification (using FastText to retain English content), quality filtering (removing low-quality documents based on heuristic rules), and fuzzy deduplication (using MinHash LSH for approximate deduplication at large scale).

The RefinedWeb paper details the implementation and effectiveness evaluation of each step, making it an excellent textbook for learning pre-training data processing. It's worth noting that although RefinedWeb has publicly released a subset of the dataset (~600B tokens), the complete version remains exclusive to the Falcon model.

### 3.1.3 The Pile: Diversified Data Mixture

The Pile is an open-source pre-training dataset released by EleutherAI, approximately 800GB in size (uncompressed), containing about 300B tokens. Unlike RefinedWeb's focus on web data, The Pile's design philosophy is diversification—it mixes data from 22 different sources, covering web pages, books, code, papers, legal documents, and other domains.

The Pile's data source composition reflects the importance of pre-training data diversity. Among them, Pile-CC is a cleaned Common Crawl subset, accounting for approximately 50%; PubMed Central provides biomedical papers; ArXiv provides scientific preprints; GitHub provides open-source code; Books3 provides book text; StackExchange provides technical Q&A; Wikipedia provides encyclopedic knowledge. This multi-source mixing strategy has been proven to improve model performance across various downstream tasks, with later models like LLaMA and Mistral borrowing similar approaches in their data recipes.

However, The Pile faces legal controversies. Its Books3 subset contains large amounts of copyrighted books and has triggered multiple lawsuits. When using The Pile, it is recommended to evaluate based on your own legal risk tolerance, or selectively exclude controversial subsets.

### 3.1.4 Overview of Chinese Datasets

For training Chinese large models, available open-source datasets are relatively scarce, though there has been improvement in recent years.

WuDaoCorpora is a large-scale Chinese corpus released by BAAI, containing approximately 3TB of Chinese text, covering encyclopedias, news, forums, Q&A, and other sources. Data must be obtained through application, and usage must comply with relevant agreements. ChineseCrawl is a Chinese Common Crawl subset extraction, with multiple community versions available. CLUECorpus is Chinese corpus released by the CLUE benchmark team, approximately 100GB in size, suitable for small to medium-scale experiments.

Compared to the richness of English datasets, Chinese pre-training data remains a "seller's market." This means Chinese large model teams often need to acquire and process Chinese data themselves from Common Crawl or other sources, and cannot rely entirely on existing open-source datasets.

---

## 3.2 High-Performance Web Parsing

After obtaining raw HTML from Common Crawl or your own crawler, the next critical step is extracting body text from it. This seems simple but is actually one of the most critical steps in the entire data pipeline—parsing quality directly determines the quality of the final corpus.

### 3.2.1 Challenges of Web Parsing

Modern web pages are far more complex than they appear on the surface. A typical web page may contain: HTML structural tags, CSS style definitions, JavaScript code (including inline and external references), navigation bars and footers, ads and promotional content, comment sections and user-generated content, page sidebars and recommended content. What we need is only the "body"—the main content section of the page.

Traditional parsing methods (such as simply removing all HTML tags) work poorly because they cannot distinguish between body content and noise. More advanced methods need to understand the semantic structure of web pages and identify which parts are truly valuable content.

### 3.2.2 Trafilatura: Industrial-Grade Parsing Library

Trafilatura is currently the most recommended web body extraction library, adopted by mainstream datasets like RefinedWeb and Dolma. Its core advantages lie in: finely-tuned extraction algorithms with excellent performance on multiple evaluation datasets; good multilingual support, especially for Asian languages like Chinese and Japanese; rich configuration options that can adjust extraction strategies according to needs; and reasonable performance suitable for large-scale data processing.

The basic workflow for using Trafilatura is as follows:

```python
import trafilatura

# Extract body content from HTML
def extract_content(html: str, url: str = None) -> dict:
    """
    Extract body content from HTML
    
    Args:
        html: Raw HTML string
        url: Optional URL for resolving relative links
    
    Returns:
        Dictionary containing body content and metadata
    """
    # Core extraction
    result = trafilatura.extract(
        html,
        url=url,
        include_comments=False,    # Exclude comment section
        include_tables=True,       # Preserve table content
        no_fallback=False,         # Allow fallback algorithms
        favor_precision=True,      # Prioritize precision
        output_format='txt'        # Output plain text
    )
    
    # Extract metadata
    metadata = trafilatura.extract_metadata(html)
    
    return {
        'text': result,
        'title': metadata.title if metadata else None,
        'author': metadata.author if metadata else None,
        'date': metadata.date if metadata else None,
        'url': url
    }
```

Trafilatura provides rich configuration options, with different parameter combinations suitable for different scenarios. `include_comments` controls whether to preserve page comment section content—can be set to True for forum-type websites, usually set to False for news websites. `include_tables` controls whether to preserve tables—should be set to True for data-type pages (such as Wikipedia). `favor_precision` and `favor_recall` are a pair of trade-off parameters; the former prioritizes ensuring extraction accuracy (better to miss than include errors), the latter prioritizes ensuring extraction completeness (better to have noise than be incomplete). For pre-training data, typically choose `favor_precision=True`, as noisy data harms model training.

### 3.2.3 Comparison of Other Parsing Tools

Besides Trafilatura, there are several commonly used web parsing tools, each with their own characteristics.

**Readability** was originally developed by Mozilla for Firefox's reading mode. Its algorithm is relatively simple and fast, but performance on complex pages is average. The Python ecosystem has ported versions like readability-lxml.

**Newspaper3k** is specifically optimized for news websites and can effectively extract article titles, body text, publication dates, authors, and other information. However, it performs poorly on non-news sites, and the project is not actively maintained.

**Justext** is a library focused on "boilerplate removal," with algorithms based on link density and text density of text blocks. It is commonly cited in academic research but has less engineering practicality than Trafilatura.

| Tool | Advantages | Disadvantages | Recommended Scenarios |
|------|-----------|---------------|----------------------|
| Trafilatura | Best overall performance, good multilingual support | Medium speed | General scenarios, first choice |
| Readability | Fast, simple algorithm | Poor on complex pages | Rapid prototyping |
| Newspaper3k | Good for news sites | Weak generalization | News corpus specialty |
| Justext | Academically well-validated | Less engineering adaptation | Research scenarios |

![Figure 3-2: Parser Quality Comparison](../../images/part2/图3_2_解析器质量对比.png)

*Figure 3-2: Web Parser Quality Comparison — Trafilatura leads in F1 score, making it the preferred tool for LLM data processing*

In actual projects, a common strategy is to use Trafilatura as the primary parser, and when extraction results are empty or too short, fall back to Readability for attempts. This "primary-backup" strategy can improve overall extraction success rate.

### 3.2.4 Distributed Parsing Architecture

Single-machine processing capacity is limited; facing TB-level WARC files requires building distributed parsing systems. Here's an example of distributed parsing based on Ray Data:

```python
import ray
import trafilatura
from warcio.archiveiterator import ArchiveIterator
import gzip

ray.init()

def parse_warc_record(record):
    """Parse single WARC record"""
    if record.rec_type != 'response':
        return None
    
    url = record.rec_headers.get_header('WARC-Target-URI')
    content_type = record.http_headers.get_header('Content-Type', '')
    
    # Only process HTML pages
    if 'text/html' not in content_type:
        return None
    
    try:
        html = record.content_stream().read().decode('utf-8', errors='ignore')
        text = trafilatura.extract(html, url=url, favor_precision=True)
        
        if text and len(text) > 200:  # Filter overly short content
            return {
                'url': url,
                'text': text,
                'length': len(text)
            }
    except Exception as e:
        return None
    
    return None

def process_warc_file(warc_path: str):
    """Process single WARC file"""
    results = []
    
    with gzip.open(warc_path, 'rb') as f:
        for record in ArchiveIterator(f):
            result = parse_warc_record(record)
            if result:
                results.append(result)
    
    return results

# Get all WARC file paths
warc_files = [...]  # S3 or local path list

# Distributed parallel processing
ds = ray.data.from_items(warc_files)
ds = ds.flat_map(process_warc_file)

# Save results
ds.write_parquet("s3://bucket/parsed_data/")
```

Key design points of this architecture include: using Ray Data's `flat_map` operator to achieve file-level parallelism; performing error handling inside the parsing function to avoid single data failures affecting batch processing; early filtering through conditions like `len(text) > 200` to reduce downstream processing volume; outputting in Parquet format for convenient subsequent deduplication and filtering steps.

---

## 3.3 Specialized Data Acquisition

Besides general web data, pre-training corpora typically also need to include specialized domain data such as code, academic papers, and books. These "specialized data" have unique challenges and techniques for acquisition and processing.

### 3.3.1 Code Data: GitHub and The Stack

Code capability is one of the core competencies of modern large models, and acquiring high-quality code data is the foundation for achieving this capability. Currently, the most important source of code data is GitHub.

Obtaining code directly from the GitHub API is feasible but inefficient and has request limits. A more common approach is to use public data mirrors of GitHub. Google BigQuery hosts complete snapshots of GitHub public repositories and allows querying and exporting using SQL. Software Heritage is an organization dedicated to preserving humanity's software heritage and maintains a complete archive of GitHub.

For large-scale code data needs, the most convenient choice is to use The Stack dataset released by the BigCode project. This dataset crawls code in over 300 programming languages from GitHub, with a total size of about 3TB. The Stack's processing pipeline includes: license-based filtering (retaining only code allowed by open-source licenses), deduplication (removing duplicate files and code snippets), and PII cleaning (removing sensitive information).

Special attention is needed when using code data:

**License compliance** is the primary concern. Different open-source licenses have different restrictions on code usage. The Stack dataset provides license tags and can be filtered according to needs. For commercial model training, it is recommended to only use code with permissive licenses like MIT and Apache 2.0.

**Code quality varies greatly**. GitHub contains both high-quality projects like the Linux kernel and large amounts of student assignments and personal experimental code. Common quality filtering strategies include: filtering by repository star count (retaining repositories with star > 10), filtering by file length (removing overly short or long files), and detecting syntax errors based on AST parsing.

**Example of processing code data**:

```python
import ast
from typing import Optional

def is_valid_python(code: str) -> bool:
    """Check if Python code is syntactically correct"""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def extract_functions(code: str) -> list:
    """Extract function definitions from code"""
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
    """Code quality filtering"""
    lines = code.split('\n')
    
    # Length filtering
    if not (min_lines <= len(lines) <= max_lines):
        return None
    
    # Syntax check
    if not is_valid_python(code):
        return None
    
    # Docstring check (optional)
    if require_docstring and '"""' not in code and "'''" not in code:
        return None
    
    return code
```

### 3.3.2 Academic Papers: ArXiv and S2ORC

Academic papers are an important source of high-quality knowledge and significantly help improve models' reasoning abilities and professional knowledge levels.

**ArXiv** is the most important open-access preprint platform, covering academic papers in fields like physics, mathematics, computer science, and biology. ArXiv provides bulk download services, with LaTeX source files and PDFs accessible through its S3 bucket. LaTeX source files are a more ideal data source because they preserve the structural information of papers (sections, formulas, citations, etc.) and are in plain text format, easy to process.

The main challenge in processing ArXiv LaTeX data lies in the complexity of LaTeX syntax. A paper may contain dozens of `.tex` files, using custom macros and styles. A practical simplification strategy is: extract only the main file (usually `main.tex` or files related to the paper title), use regular expressions to remove figure-table and complex formula environments, and preserve body text and simple mathematical expressions.

```python
import re
import tarfile
from pathlib import Path

def extract_arxiv_text(latex_content: str) -> str:
    """Extract plain text from LaTeX"""
    text = latex_content
    
    # Remove comments
    text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)
    
    # Remove figure environments
    text = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{table\}.*?\\end\{table\}', '', text, flags=re.DOTALL)
    
    # Simplify citations
    text = re.sub(r'\\cite\{[^}]+\}', '[CITATION]', text)
    text = re.sub(r'\\ref\{[^}]+\}', '[REF]', text)
    
    # Remove common commands but preserve arguments
    commands_to_strip = ['textbf', 'textit', 'emph', 'section', 'subsection', 
                         'paragraph', 'title', 'author']
    for cmd in commands_to_strip:
        text = re.sub(rf'\\{cmd}\{{([^}}]+)\}}', r'\1', text)
    
    # Remove other commands
    text = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Clean whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()
```

**Semantic Scholar Open Research Corpus (S2ORC)** is a large-scale academic paper dataset released by Allen AI, containing metadata for over 80M papers and full text for approximately 8M papers. Compared to ArXiv, S2ORC covers broader fields, including multiple sources like PubMed and ACL Anthology. S2ORC's full text has been processed into structured JSON format, eliminating the hassle of LaTeX/PDF parsing, making it a convenient choice for quickly acquiring paper data.

### 3.3.3 Book Data: Copyright and Alternative Solutions

Books are an important source of high-quality long texts and are valuable for training models' long-range comprehension abilities and knowledge depth. However, the copyright issues of book data are particularly sensitive.

**Books3 in The Pile** contains approximately 200,000 books from sources like Bibliotik. This dataset has triggered multiple copyright lawsuits, with several companies being sued for using this dataset to train models. In the current legal environment, directly using Books3 poses significant legal risks.

**Compliant alternatives** include:

![Figure 3-3: Pre-training Data Source Mixture](../../images/part2/图3_3_预训练数据来源混合.png)

*Figure 3-3: Pre-training Data Source Mixture — Multi-source mixing strategy can improve models' comprehensive capabilities*

Project Gutenberg is a volunteer project providing copyright-expired classic books. Primarily English books published before 1928, about 70,000 volumes. High data quality, but from a distant era with insufficient coverage of modern language.

Internet Archive's Open Library provides borrowable e-books. Usage must comply with its borrowing agreements; large-scale batch acquisition may violate terms of service.

Wikisource provides public domain literary works, covering multiple languages including large amounts of classical Chinese texts.

Academic textbooks and Open Educational Resources (OER) like OpenStax provide high-quality textbook content suitable for building education-focused pre-training data.

For commercial model training, the safest strategy is to only use book data that has been explicitly authorized or is in the public domain. Although this limits data scale, it can avoid potential legal risks.

### 3.3.4 Multilingual Data Balancing

When training multilingual models, data availability varies greatly across languages. English data is most abundant, major languages like Chinese, Japanese, and German are second, while high-quality data for minor languages is extremely scarce.

Data imbalance leads to uneven model capabilities. If simply mixing according to original proportions, English will dominate overwhelmingly and minor languages will barely be learned. Common solution strategies include:

**Upsampling minor languages** is the most direct method, increasing the weight of minor language data through repeated sampling. However, excessive repetition may lead to overfitting.

**Temperature sampling** is a more refined method. Set a temperature parameter T, where the sampling probability for language L is $p_L \propto n_L^{1/T}$, where $n_L$ is the original data volume for that language. When T=1, it degenerates to original proportions; as T→∞, it approaches uniform distribution. LLaMA 2 used temperature sampling around T=0.3.

The **quality over quantity** approach is also worth considering. For minor languages with scarce data, translation or synthesis methods can be used to augment data, but attention must be paid to translation quality and unnatural translation issues.

---

## 3.4 Engineering Practices for Data Acquisition

Connecting all the above steps to build a complete data acquisition pipeline requires considering many engineering details.

### 3.4.1 Crawler Architecture Design

For scenarios requiring independent data crawling (rather than using Common Crawl), distributed crawler architecture design is crucial. A typical architecture includes the following components:

**URL Manager** is responsible for maintaining queues of URLs to be crawled, recording the status of crawled URLs, and handling URL deduplication and priority sorting. Common implementation methods include Redis queues combined with Bloom Filter deduplication.

![Figure 3-4: Distributed Crawler Architecture](../../images/part2/图3_4_分布式爬虫架构.png)

*Figure 3-4: Distributed Web Crawling System Architecture — Collaboration between URL Manager, downloader cluster, parser, and storage layer*

**Downloader Cluster** is responsible for actual HTTP requests. Key considerations include: concurrency control (avoiding excessive pressure on target websites), proxy pool management (dealing with anti-crawler mechanisms), retry strategies (handling network fluctuations and temporary failures), and robots.txt compliance (respecting website crawler rules).

**Parser** is responsible for extracting body content and metadata from downloaded HTML, which was discussed in detail in the previous section.

**Storage Layer** is responsible for persisting crawl results. For large-scale crawling, it is recommended to write directly to object storage (S3/MinIO) in WARC or Parquet format.

```python
# Simplified crawler example
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
        """Asynchronously fetch and parse single URL"""
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
        """Batch crawl URL list"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch(session, url) for url in urls]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]
```

### 3.4.2 Incremental Update Strategy

Pre-training data is not a one-time task. Over time, new data needs to be continuously absorbed to maintain the timeliness of model knowledge. The key challenges of incremental updates are:

**Identifying new data**: For Common Crawl, new batches are released monthly and can be processed directly. For independent crawling, update timestamps need to be maintained, periodically revisiting known URLs to check for updates.

**Avoiding duplicate processing**: Already processed data should not re-enter the pipeline. Deduplication can be performed through URL fingerprints, content hashes, and other methods.

**Version management**: Each processing batch should have a clear version identifier for traceability and rollback. This is closely related to the data version control (DVC/LakeFS) discussed in Chapter 2.

### 3.4.3 Quality Monitoring and Feedback

Data acquisition pipelines need to establish comprehensive quality monitoring mechanisms. Key monitoring metrics include:

**Download success rate**: Proportion of failed requests. If it suddenly increases, the target website may have blocked the crawler.

**Parsing success rate**: Proportion of successful body text extractions. If it decreases, the target website may have changed its page structure.

**Average document length**: Average character count of body text. Abnormal fluctuations may indicate parser problems.

**Language distribution**: Proportion of data in each language. Ensure it matches the expected language ratio.

**Duplication rate**: Proportion of duplication with historical data. An excessively high duplication rate means the marginal value of new data is declining.

It is recommended to integrate these metrics into monitoring systems (such as Prometheus + Grafana), set alert thresholds, and promptly detect and handle anomalies.

---

## 3.5 Common Pitfalls and Best Practices

In the data acquisition phase, several common pitfalls deserve caution.

**The first pitfall is over-reliance on a single data source.** If pre-training data all comes from Common Crawl, the model may inherit the biases and noise of web data. A reasonable approach is to mix multiple sources: web data provides breadth, books and papers provide depth, and code data provides logical capabilities. The Pile-style multi-source mixing strategy has proven effective.

**The second pitfall is ignoring data timeliness.** Historical batches of Common Crawl, though voluminous, may contain outdated information. For applications requiring timeliness (such as news and current events), more recent data batches should be prioritized. Additionally, very old data may contain defunct links, corrected erroneous information, etc.

**The third pitfall is underestimating compliance risks.** Copyright issues, privacy issues, robots.txt violations, etc., may trigger serious legal problems in later project stages. The best practice is to establish comprehensive metadata records during the data acquisition phase—recording each data item's source URL, acquisition time, claimed license, and other information, leaving evidence for possible future audits.

**The fourth pitfall is emphasizing collection over processing.** Many teams expend great effort expanding data collection scale but rush through parsing and cleaning steps. As stated in Chapter 1, data quality is far more important than data quantity. It's better to collect less data but ensure each piece undergoes strict quality control.

---

## 3.6 Chapter Summary

This chapter systematically introduces the methodology and engineering practices of pre-training data acquisition.

In terms of open-source datasets, Common Crawl is the most important upstream data source, providing three formats: WARC, WAT, and WET. RefinedWeb and The Pile are carefully processed high-quality datasets whose processing methods are worth learning from. Chinese datasets are relatively scarce, often requiring independent extraction from Common Crawl.

In terms of web parsing, Trafilatura is currently the most recommended industrial-grade parsing library, capable of accurately extracting body content from complex HTML. Distributed parsing architectures (such as based on Ray Data) are necessary for processing TB-level data.

In terms of specialized data, code data can be acquired through The Stack or GitHub BigQuery, with attention to license compliance; academic papers can be acquired through ArXiv and S2ORC; book data carries higher copyright risks, and public domain resources are recommended. Multilingual data needs to be balanced through strategies like temperature sampling.

In terms of engineering practices, a complete data acquisition pipeline needs to consider crawler architecture design, incremental update strategies, and quality monitoring mechanisms. Core principles are multi-source mixing, quality first, and compliance foremost.

![Figure 3-5: Chapter Knowledge Structure](../../images/part2/图3_5_本章知识结构.png)

*Figure 3-5: Chapter 3 Knowledge Structure — Covering four major themes: open-source datasets, web parsing, specialized data, and engineering practices*

---

## Further Reading

For in-depth content on pre-training data acquisition, the following resources are worth consulting:

Common Crawl official documentation (commoncrawl.org/the-data) provides detailed introductions to data formats and acquisition methods. The RefinedWeb paper (Falcon LLM: A Large Language Model for High-Quality Web Data) details the complete process of building high-quality pre-training sets from Common Crawl. The Pile paper (The Pile: An 800GB Dataset of Diverse Text for Language Modeling) introduces multi-source mixed data construction strategies. Trafilatura documentation (trafilatura.readthedocs.io) provides comprehensive API documentation and usage examples. The Stack paper (StarCoder: May the Source Be with You!) introduces methods for building large-scale code datasets.

---

## Next Chapter Preview

Acquiring raw data is only the first step. In the next chapter, "Cleaning and Denoising," we will delve into how to screen high-quality content from massive raw data. You will learn heuristic filtering rules (language identification, perplexity filtering, length distribution), large-scale deduplication techniques (principles and distributed implementation of MinHash LSH), and privacy data cleaning (PII identification and removal).

Enter the next chapter with this question: If two documents have 80% identical content, how can you efficiently identify and handle them?
