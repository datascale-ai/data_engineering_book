# Project One: Building a "Mini-C4" Pre-training Dataset

### 1. Project Background (Project Brief)

*   **Task Definition:** Build a miniature C4 (Colossal Clean Crawled Corpus) dataset pipeline. Our goal is to transform messy raw web data (Common Crawl) into low-noise, deduplicated, high-quality pure text data directly usable for large model pre-training.
*   **Input and Output:**
    *   **Input:** Raw WARC archives from Common Crawl (containing HTTP headers, HTML source, garbled text, etc.).
    *   **Output:** Classified and graded JSONL files (e.g., `data_en.jsonl`, `final_data.jsonl`), containing clean text and quality scores.
*   **Challenge Analysis:**
    *   **Very low signal-to-noise ratio:** Over 90% of raw web content is navigation bars, ads, JavaScript code, and meaningless placeholders.
    *   **Compute intensive:** Pairwise comparison deduplication in large-scale corpus is extremely resource-consuming.
    *   **Quality quantification:** How to let machines automatically judge whether a sentence is "high-quality human language" or "machine-generated garbage"?

### 2. Architecture Design (Architecture Design)

To handle unstructured web data, we designed the following funnel-shaped processing architecture:

**Data Pipeline Diagram:**

![Figure 1: Mini-C4 Pre-training Dataset Pipeline](../../images/part6/图1_构建Mini_C4预训练集数据流水线图.png)
<!-- ![Figure 1: Mini-C4 Pre-training Dataset Pipeline](images/Practical Projects/图1_构建Mini_C4预训练集数据流水线图.png) -->

**Technology Stack:**

*   **Parsing Layer: Trafilatura**
    *   *Decision rationale:* Compared to traditional BeautifulSoup, Trafilatura is optimized for web article extraction, automatically removing navigation, footers, and boilerplate text, with higher extraction efficiency and accuracy.
*   **Compute Layer: Ray**
    *   *Decision rationale:* Python's native multiprocessing struggles with big data. Ray provides extremely simple distributed primitives, letting us parallelize MinHash computation across multi-core CPU or even clusters with few lines of code.
*   **Quality Layer: KenLM**
    *   *Decision rationale:* Lightweight N-gram language model library. Both GPT-3 and CCNet papers use KenLM perplexity as core metric for measuring text naturalness.

### 3. Step-by-Step Implementation

#### Phase 1: Extracting Main Content from HTML Mess (Extraction & Cleaning)

Raw WARC files contain large amounts of non-text noise. We first use `warcio` for streaming read of compressed archives, then use `trafilatura` to extract core content. Subsequently apply heuristic rules for initial filtering.

**Core Code: Parsing and Heuristic Cleaning**

```python
import trafilatura
from warcio.archiveiterator import ArchiveIterator

# 1. Extraction logic (from 2_process_warc.py)
def extract_text(content_stream):
    text = trafilatura.extract(
        content_stream, 
        include_comments=False, 
        include_tables=False
    )
    return text

# 2. Heuristic cleaning rules (from 3_clean_data.py)
def is_high_quality(text):
    # Rule A: Length and average word length filter
    words = text.split()
    if not words:
        # Empty or whitespace-only text, treat as low quality
        return False
    mean_word_len = sum(len(w) for w in words) / len(words)
    if mean_word_len > 15: # Overly long words usually garbage or code
        return False
        
    # Rule B: Symbol density (Symbol Ratio)
    code_symbols = {'{', '}', '[', ']', '<', '>', '\\'}
    symbol_count = sum(1 for char in text if char in code_symbols)
    if len(text) > 0 and (symbol_count / len(text) > 0.1): # Too many code symbols
        return False
        
    # Rule C: Blacklist keywords
    bad_phrases = ["lorem ipsum", "enable cookies", "403 forbidden"]
    if any(p in text.lower() for p in bad_phrases):
        return False
        
    return True
```

#### Phase 2: Distributed MinHash Deduplication

The internet has large amounts of repeated content (reposts, mirrors). We use Ray for parallel MinHash computation, combining LSH (Locality Sensitive Hashing) to reduce $O(N^2)$ complexity to $O(N)$.

**Core Code: Ray Parallel Signature Computation**

```python
import ray
from datasketch import MinHash

# Initialize Ray to use all CPU cores
ray.init()

@ray.remote
def process_batch(lines, num_perm=128):
    """Ray Worker: Parallel MinHash fingerprint computation for a batch"""
    results = []
    for line in lines:
        item = json.loads(line)
        m = MinHash(num_perm=num_perm)
        # Shingling: Update hash by words
        for w in item['text'].split():
            m.update(w.encode('utf8'))
        results.append((item['url'], m, item['text']))
    return results

# Main flow: Map-Reduce style
# Map: Distribute compute tasks
futures = [process_batch.remote(batch) for batch in batches]
# Reduce: Collect results and build LSH index
results = ray.get(futures)
# ...continue with MinHashLSH index building...
```

#### Phase 3: Language Identification and Perplexity Filtering (Quality Filtering)

Cleaned data mixes multiple languages with varying quality. We first use FastText for language routing, then KenLM for perplexity. Lower perplexity means more fluent, more "human-like" sentences.

**Core Code: KenLM Scoring**

```python
import kenlm
import fasttext

# 1. Language routing (from 5_split_lang.py)
lid_model = fasttext.load_model('lid.176.ftz')
def predict_lang(text):
    # k=1 takes highest probability language
    predictions = lid_model.predict(text, k=1)
    return predictions[0][0].replace('__label__', '')

# 2. Perplexity filtering (from 6_quality_filter.py)
kenlm_model = kenlm.Model('en.arpa.bin')
PERPLEXITY_THRESHOLD = -6.0  # Experience threshold: below this usually low-quality text

def filter_by_perplexity(text):
    words = text.split()
    if not words:
        # Empty text as low quality; avoid division by zero
        return False, -10.0
    # Compute normalized score (Log Score / Length)
    log_score = kenlm_model.score(text)
    normalized_score = log_score / len(words)
    
    if normalized_score > PERPLEXITY_THRESHOLD:
        return True, normalized_score
    return False, normalized_score
```

### 4. Results Showcase (Showcase)

After this pipeline, data landscape fundamentally changed:

**Case 1: Navigation Bar Noise (Removed)**
> *Raw:* "Home | About Us | Contact | Enable Cookies | Copyright 2023..."
> *Result:* **[Discarded]** (Triggered short text and keyword blacklist rules)

**Case 2: Code Snippet (Removed)**
> *Raw:* "function(x) { return x > 0 ? true : false; } var a = [1,2,3];"
> *Result:* **[Discarded]** (Triggered symbol density > 10% rule)

**Case 3: High-Quality Article (Retained and Scored)**
> *Raw:* "The James Webb Space Telescope has captured a new image of the Pillars of Creation..."
> *Result:* **[Retained]**
> *KenLM Score:* -4.82 (Better than threshold -6.0)

**Data Statistics:**
In single-crawl sample test:
*   **Raw records:** 10,000
*   **Extracted valid text:** ~4,500 (HTML parsing loss)
*   **Remaining after cleaning:** ~2,800 (heuristic filter loss)
*   **Remaining after dedup:** ~2,100 (~25% duplicate rate)
*   **Final high-quality set:** ~1,800 (KenLM filter)

### 5. Cost and Optimization (Cost & Optimization)

*   **Resource consumption:**
    *   **Compute:** This project code on single machine 16-core CPU, 64G RAM, processing 1GB WARC takes ~5-8 minutes.
    *   **Bottleneck:** `MinHashLSH` index building is currently single-threaded (in `4_deduplicate.py`), and fully memory-dependent.

*   **Scaling considerations:**
    If data scales to TB level (like real C4), current architecture needs upgrade:
    1.  **LSH storage:** Can't use in-memory `MinHashLSH`; need Redis or Cassandra for hash buckets.
    2.  **Parallel strategy:** Scale Ray tasks from "single-machine multi-core" to "multi-machine cluster."
    3.  **I/O optimization:** Data reading must migrate from local filesystem to S3, using PyArrow for streaming columnar processing.
