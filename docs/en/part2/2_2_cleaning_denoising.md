# Chapter 4: Cleaning and Quality Control (Deduplication, PII Masking, Benchmark Decontamination)

---

## Chapter Summary

Raw data obtained from the internet is like unprocessed ore, where the truly valuable "concentrate" may only account for a small proportion. This chapter delves into the three core technologies of pre-training data cleaning: heuristic filtering rules for removing low-quality documents, large-scale deduplication for eliminating duplicate content, and privacy data cleaning for protecting user information. After mastering these techniques, readers will be able to build industrial-grade data cleaning pipelines to transform raw web data into high-quality pre-training corpora.

---

## Scenario Introduction

After the efforts of the previous chapter, your team successfully extracted 50TB of Chinese web text from Common Crawl. However, when you randomly sample and inspect the data, you discover various troubling issues: many pages contain only a few words of navigation text with no substantive content; some pages are full of JavaScript code or CSS style remnants; content from certain websites was crawled repeatedly hundreds of times; and there are large amounts of sensitive information including user emails and phone numbers.

Worse yet, the training engineer tells you that the model trained on uncleaned data last time had a serious "parrot" problem—the model would repeatedly output the same sentences and could even recite complete content from certain websites. This is clearly caused by data duplication.

How to systematically solve these problems? This chapter provides the complete answer.

---

## 4.1 Heuristic Filtering Rules

Heuristic filtering is the first line of defense in data cleaning. It uses a set of quantifiable rules to quickly filter out obviously low-quality documents. Although these rules may seem simple, they can filter out most noisy data in practice, making them a highly cost-effective cleaning approach.

![Figure 4-1: Data Cleaning Pipeline](../../images/part2/图4_1_数据清洗流水线.png)

*Figure 4-1: Data Cleaning Pipeline Architecture — Eight-stage processing flow from raw data to clean corpus*

### 4.1.1 Language Detection

Language detection is a fundamental step in multilingual data processing. For training Chinese models, we first need to filter Chinese content from Common Crawl's massive data, which requires accurate language detection capability.

**FastText Language Detector** is currently the most commonly used tool. Developed by Facebook AI Research, its pre-trained model supports 176 languages with extremely fast speed and high accuracy. FastText provides two pre-trained models: `lid.176.bin` is the full version with higher accuracy but larger size (~126MB); `lid.176.ftz` is the compressed version with smaller size (~917KB) but slightly lower accuracy. For large-scale data processing, the full version is recommended.

```python
import fasttext

# Load language detection model
lang_model = fasttext.load_model('lid.176.bin')

def detect_language(text: str, min_confidence: float = 0_8) -> tuple:
    """
    Detect text language
    
    Args:
        text: Text to be detected
        min_confidence: Minimum confidence threshold
    
    Returns:
        (language code, confidence) or (None, 0) if confidence insufficient
    """
    # Preprocessing: remove newlines, take first 1000 characters
    text = text.replace('\n', ' ')[:1000]
    
    # Predict
    predictions = lang_model.predict(text, k=1)
    lang = predictions[0][0].replace('__label__', '')
    confidence = predictions[1][0]
    
    if confidence >= min_confidence:
        return lang, confidence
    return None, confidence

def filter_by_language(documents: list, target_lang: str = 'zh') -> list:
    """Filter documents by specified language"""
    results = []
    for doc in documents:
        lang, conf = detect_language(doc['text'])
        if lang == target_lang:
            doc['detected_lang'] = lang
            doc['lang_confidence'] = conf
            results.append(doc)
    return results
```

Language detection encounters some edge cases in practice. Mixed-language documents (e.g., technical blogs with Chinese and English) may be misclassified. Short text has lower detection accuracy; it is recommended to skip language filtering for text shorter than 50 characters. Code snippets may be identified as various languages and require content type judgment.

### 4.1.2 Text Quality Scoring

Language detection only ensures that documents are in the target language but cannot judge content quality. A grammatically correct spam ad and a high-quality technical article may receive the same language detection score. This requires a more refined quality assessment mechanism.

**Perplexity Filtering** is a quality assessment method based on language models. Perplexity measures the language model's "surprise" at the text—if text is similar to the model's training data distribution, perplexity is low; if text contains much noise, gibberish, or unnatural expressions, perplexity is high.

KenLM is the most commonly used tool for computing perplexity. It is based on n-gram language models, extremely fast, and suitable for large-scale data processing.

```python
import kenlm

class PerplexityFilter:
    def __init__(self, model_path: str, max_perplexity: float = 500):
        """
        Initialize perplexity filter
        
        Args:
            model_path: KenLM model path (.arpa or .bin)
            max_perplexity: Perplexity threshold; documents exceeding this value will be filtered
        """
        self.model = kenlm.Model(model_path)
        self.max_perplexity = max_perplexity
    
    def compute_perplexity(self, text: str) -> float:
        """Compute text perplexity"""
        # KenLM returns log10 probability
        log_prob = self.model.score(text, bos=True, eos=True)
        # Convert to perplexity
        num_words = len(text.split()) + 1  # +1 for EOS
        perplexity = 10 ** (-log_prob / num_words)
        return perplexity
    
    def filter(self, documents: list) -> list:
        """Filter high-perplexity documents"""
        results = []
        for doc in documents:
            ppl = self.compute_perplexity(doc['text'])
            if ppl <= self.max_perplexity:
                doc['perplexity'] = ppl
                results.append(doc)
        return results
```

Perplexity threshold setting needs to be tuned based on specific data. Generally, high-quality news and encyclopedia text have perplexity between 100-200, ordinary web content between 200-500, and low-quality content (e.g., gibberish, machine translation) typically exceeds 500. It is recommended to analyze perplexity distribution on a small sample first, then determine the appropriate threshold.

### 4.1.3 Heuristic Rule Set

Besides language detection and perplexity filtering, there is a set of simple but effective heuristic rules that can quickly remove obviously low-quality content. These rules are designed based on observations and experience from large amounts of data.

**Length filtering** is the most basic rule. Documents that are too short (e.g., navigation text with only a few words) have no training value and should be removed directly. Documents that are too long may need truncation or segmentation. Typical thresholds: minimum length 200 characters or 50 words, maximum length 100,000 characters.

**Special character ratio** can identify noisy content. If the proportion of non-alphanumeric characters in a document is too high, it may be code remnants, gibberish, or format errors. Similarly, an excessively high digit ratio may indicate log files or data tables.

**Duplicate line ratio** can detect templated low-quality pages. If a document has many identical lines (e.g., navigation bars repeated in multiple places), the content quality is low.

**Vocabulary diversity** measures the information richness of a document. A document using only 10 different words is clearly less valuable than one using 500. A common metric is Type-Token Ratio (TTR), the ratio of unique words to total words.

Below is a comprehensive heuristic filter implementation:

```python
import re
from collections import Counter

class HeuristicFilter:
    def __init__(self, config: dict = None):
        """
        Initialize heuristic filter
        
        Default config suitable for Chinese pre-training data
        """
        self.config = config or {
            'min_length': 200,           # Minimum character count
            'max_length': 100000,        # Maximum character count
            'min_words': 50,             # Minimum word count
            'max_special_ratio': 0_3,    # Maximum special character ratio
            'max_digit_ratio': 0_3,      # Maximum digit ratio
            'max_duplicate_line_ratio': 0_3,  # Maximum duplicate line ratio
            'min_avg_word_length': 2,    # Minimum average word length
            'max_avg_word_length': 20,   # Maximum average word length
            'min_unique_word_ratio': 0_1 # Minimum vocabulary diversity
        }
    
    def check_length(self, text: str) -> bool:
        """Check document length"""
        length = len(text)
        return self.config['min_length'] <= length <= self.config['max_length']
    
    def check_special_chars(self, text: str) -> bool:
        """Check special character ratio"""
        if len(text) == 0:
            return False
        special = len(re.findall(r'[^\w\s]', text, re.UNICODE))
        ratio = special / len(text)
        return ratio <= self.config['max_special_ratio']
    
    def check_digit_ratio(self, text: str) -> bool:
        """Check digit ratio"""
        if len(text) == 0:
            return False
        digits = len(re.findall(r'\d', text))
        ratio = digits / len(text)
        return ratio <= self.config['max_digit_ratio']
    
    def check_duplicate_lines(self, text: str) -> bool:
        """Check duplicate line ratio"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) == 0:
            return False
        unique_lines = len(set(lines))
        duplicate_ratio = 1 - (unique_lines / len(lines))
        return duplicate_ratio <= self.config['max_duplicate_line_ratio']
    
    def check_vocabulary_diversity(self, text: str) -> bool:
        """Check vocabulary diversity"""
        words = text.split()
        if len(words) < self.config['min_words']:
            return False
        unique_ratio = len(set(words)) / len(words)
        return unique_ratio >= self.config['min_unique_word_ratio']
    
    def filter(self, text: str) -> tuple:
        """
        Apply all filtering rules
        
        Returns:
            (passed or not, failure reason or None)
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

### 4.1.4 Quality Stratification Strategy

In practice, simply binary classifying data as "retain" or "discard" is often too crude. A more refined approach is to stratify data by quality, assigning different sampling weights to different quality tiers.

A common stratification strategy: divide data into high, medium, and low quality tiers. High-quality data (e.g., from authoritative sites, passing all heuristic checks, low perplexity) receives higher sampling weights; medium-quality data receives normal weights; low-quality but acceptable data receives lower weights. This strategy can ensure data diversity while allowing high-quality data to play a larger role in training.

The RefinedWeb paper documents their stratification strategy in detail, dividing data into five tiers with different filtering thresholds for each. This refined quality management is key to building high-quality pre-training datasets.

![Figure 4-2: Quality Filtering Funnel](../../images/part2/图4_2_质量过滤漏斗.png)

*Figure 4-2: Data Quality Filtering Funnel — Layered filtering process from 100% raw data to final 4% clean corpus*

---

## 4.2 Large-Scale Deduplication: Exact vs Fuzzy Deduplication

Data duplication is the enemy of pre-training data. In Common Crawl, the same article may be reprinted by multiple websites, and the same webpage may be crawled repeatedly in different months, leading to large amounts of duplicate content. Research shows that non-deduplicated data causes models to overfit on repeated content, producing a "parrot" phenomenon that seriously affects model quality.

Deduplication can be divided into two levels: exact deduplication removes identical documents; fuzzy deduplication removes highly similar but not identical documents (e.g., articles slightly modified when reprinted). On TB-scale data, both types require efficient algorithms and distributed implementation.

### 4.2.1 Exact Deduplication: Hash Methods

The core idea of exact deduplication is to compute a fingerprint for each document; documents with the same fingerprint are considered duplicates. The simplest method uses hash functions like MD5 or SHA256.

```python
import hashlib

def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def exact_dedup(documents: list) -> list:
    """Exact deduplication: keep first document for each hash value"""
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

For distributed scenarios, Spark or Ray can be used for parallel deduplication:

```python
import ray

@ray.remote
def compute_hashes_batch(documents: list) -> list:
    """Batch compute hashes"""
    return [(compute_hash(doc['text']), doc) for doc in documents]

def distributed_exact_dedup(documents_path: str, output_path: str):
    """Distributed exact deduplication"""
    ds = ray.data.read_parquet(documents_path)
    
    # Compute hashes
    ds = ds.map(lambda doc: {**doc, 'hash': compute_hash(doc['text'])})
    
    # Group by hash, keep first per group
    ds = ds.groupby('hash').map_groups(lambda group: group.head(1))
    
    # Save results
    ds.write_parquet(output_path)
```

Exact deduplication is efficient but can only handle identical documents. For slightly different duplicate content (e.g., same news reposted on different sites with different headers/footers), exact deduplication is powerless.

### 4.2.2 Fuzzy Deduplication: MinHash LSH

The goal of fuzzy deduplication is to identify documents that are "highly similar but not identical." This is a computationally complex problem—naively comparing any two documents requires O(n²) time complexity, completely infeasible for billions of documents.

MinHash LSH (Locality-Sensitive Hashing) is the core algorithm for solving this problem. The basic idea: first convert documents to n-gram sets, then use MinHash to compress sets into fixed-length signatures, finally use LSH to cluster similar signatures into the same buckets. Only document pairs in the same bucket need fine comparison, greatly reducing computation.

Understanding MinHash LSH requires three steps:

**Step 1: n-gram decomposition.** Treat a document as a set of n-grams (consecutive n characters or words). For example, the 3-gram set of "大模型数据" is {"大模型", "模型数", "型数据"}. Using n-grams rather than entire documents better captures local similarity.

**Step 2: MinHash signature.** MinHash is a technique for compressing sets into fixed-length signatures. Jaccard similarity between two sets can be approximated by the matching degree of their MinHash signatures. Longer signatures give more accurate estimates but higher storage and computation cost.

**Step 3: LSH bucketing.** Divide the MinHash signature into several bands, each containing several hash values. If two documents have identical hash values in all positions of any band, they are placed in the same bucket. Adjusting the number of bands and band size controls the similarity threshold and recall rate.

Below is a complete MinHash LSH implementation:

![Figure 4-3: MinHash LSH Algorithm](../../images/part2/图4_3_MinHash_LSH算法.png)

*Figure 4-3: MinHash LSH Algorithm Three Steps — N-gram decomposition, MinHash signature computation, LSH bucketing, reducing complexity from O(n²) to O(n)*

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
        Initialize MinHash LSH
        
        Args:
            num_hashes: MinHash signature length
            num_bands: Number of LSH bands
            ngram_size: n-gram size
            threshold: Similarity threshold (reference value, actual threshold determined by band params)
        """
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.ngram_size = ngram_size
        
        # Generate random parameters for hash functions
        self.hash_params = [
            (np.random.randint(1, 2**31), np.random.randint(0, 2**31))
            for _ in range(num_hashes)
        ]
        
        # LSH buckets
        self.buckets = [{} for _ in range(num_bands)]
    
    def get_ngrams(self, text: str) -> Set[str]:
        """Extract n-gram set"""
        text = text.lower().replace(' ', '')
        ngrams = set()
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.add(text[i:i + self.ngram_size])
        return ngrams
    
    def compute_minhash(self, ngrams: Set[str]) -> np.ndarray:
        """Compute MinHash signature"""
        signature = np.full(self.num_hashes, np.inf)
        
        for ngram in ngrams:
            # Compute base hash value for ngram
            h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
            
            # Use multiple hash functions
            for i, (a, b) in enumerate(self.hash_params):
                hash_val = (a * h + b) % (2**31 - 1)
                if hash_val < signature[i]:
                    signature[i] = hash_val
        
        return signature.astype(np.uint32)
    
    def get_bands(self, signature: np.ndarray) -> List[str]:
        """Split signature into bands"""
        bands = []
        for i in range(self.num_bands):
            start = i * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]
            band_hash = hashlib.md5(band.tobytes()).hexdigest()
            bands.append(band_hash)
        return bands
    
    def insert(self, doc_id: str, text: str):
        """Insert document into LSH index"""
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
        """Find candidate similar documents"""
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
    """Compute Jaccard similarity"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0
```

### 4.2.3 Distributed Deduplication Practice

Running MinHash LSH on TB-scale data requires carefully designed distributed strategies. A typical flow includes:

**Phase 1: Signature computation.** Traverse all documents in parallel, computing MinHash signatures for each. This phase is fully parallelizable and can fully utilize distributed compute resources.

**Phase 2: Band grouping.** Group each document by band value. Documents with the same band value are allocated to the same partition for subsequent comparison.

**Phase 3: Intra-group deduplication.** Within each partition, perform fine similarity computation on candidate duplicate pairs to determine true duplication relationships.

**Phase 4: Transitive closure.** If document A duplicates B and B duplicates C, then A, B, C should all be considered one duplicate group. Need to compute the transitive closure of duplicate relationships.

**Phase 5: Select retained documents.** Within each duplicate group, select one representative (typically the highest quality or longest) to retain, delete the rest.

```python
import ray

def distributed_fuzzy_dedup(input_path: str, output_path: str, 
                            threshold: float = 0_8):
    """
    Distributed fuzzy deduplication pipeline
    """
    # Read data
    ds = ray.data.read_parquet(input_path)
    
    # Phase 1: Compute MinHash signatures
    def compute_signature(doc):
        lsh = MinHashLSH()
        ngrams = lsh.get_ngrams(doc['text'])
        signature = lsh.compute_minhash(ngrams)
        bands = lsh.get_bands(signature)
        return {**doc, 'signature': signature.tolist(), 'bands': bands}
    
    ds = ds.map(compute_signature)
    
    # Phase 2: Group by band value, find candidate pairs
    # (Simplified here, actual implementation needs more complex grouping logic)
    
    # Phase 3&4: Intra-group exact comparison, build duplicate relationship graph
    # ...
    
    # Phase 5: Select retained documents
    # ...
    
    # Save results
    ds.write_parquet(output_path)
```

In actual engineering, using existing tools is recommended. **text-dedup** is an open-source text deduplication library implementing various algorithms including MinHash LSH, SimHash, Suffix Array, with Spark and Ray distributed implementations. **Dolma**'s deduplication module is also a high-quality reference implementation.

### 4.2.4 Intra-Document Deduplication

Besides document-level deduplication, intra-document duplicate content also needs handling. Common cases include: navigation bars, headers, and footers that appear repeatedly across a webpage; content duplication due to JavaScript rendering issues; templated duplicate paragraphs generated by certain CMS systems.

Intra-document deduplication strategy is relatively simple: divide documents into paragraphs or fixed-length chunks, compute hash for each chunk, remove duplicate chunks.

```python
def remove_duplicate_paragraphs(text: str, min_length: int = 50) -> str:
    """Remove duplicate paragraphs within document"""
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
    """Remove high-frequency duplicate n-grams within document"""
    words = text.split()
    ngram_counts = Counter()
    
    # Compute n-gram frequencies
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        ngram_counts[ngram] += 1
    
    # Mark positions to remove
    remove_positions = set()
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        if ngram_counts[ngram] >= threshold:
            # Keep first occurrence, remove subsequent duplicates
            for j in range(i + n, len(words) - n + 1):
                if tuple(words[j:j + n]) == ngram:
                    for k in range(j, min(j + n, len(words))):
                        remove_positions.add(k)
    
    # Rebuild text
    result_words = [w for i, w in enumerate(words) if i not in remove_positions]
    return ' '.join(result_words)
```

---

## 4.3 Privacy Data Cleaning (PII Removal)

Pre-training data inevitably contains Personally Identifiable Information (PII), such as email addresses, phone numbers, ID numbers, bank card numbers, home addresses, etc. With increasingly strict data compliance requirements today (e.g., GDPR, CCPA, Personal Information Protection Law), cleaning PII is not only a moral responsibility but also a legal obligation.

### 4.3.1 Types and Risks of PII

PII can be divided into direct identifiers and quasi-identifiers. Direct identifiers can identify individuals alone, such as names, ID numbers, social security numbers, phone numbers, and email addresses. Quasi-identifiers alone have difficulty identifying individuals but may lead to identification when combined, such as birth dates, postal codes, occupation, and employer.

Retaining PII in pre-training data carries multiple risks. First is privacy leakage risk: models may "memorize" sensitive information in training data and be maliciously extracted during inference. Second is compliance risk: violating data protection regulations may result in huge fines. Finally is reputation risk: if a model outputs others' private information, it will seriously damage the company's image.

![Figure 4-4: PII Types and Risks](../../images/part2/图4_4_PII类型与风险.png)

*Figure 4-4: PII Types and Risk Levels — Classification of direct identifiers (high risk) vs. quasi-identifiers (medium risk)*

### 4.3.2 Microsoft Presidio

Presidio is Microsoft's open-source PII detection and anonymization toolkit, supporting multiple languages and PII types. It uses a modular design with two core components: Analyzer is responsible for identifying PII entities in text, Anonymizer is responsible for processing identified PII (e.g., replacement, masking, deletion).

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Initialize engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def analyze_pii(text: str, language: str = 'en') -> list:
    """
    Identify PII in text
    
    Returns:
        List of PII entities with type, position, and confidence
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
    Anonymize PII in text
    
    Replace identified PII with placeholders
    """
    # First identify
    analyzer_results = analyzer.analyze(text=text, language=language)
    
    # Define anonymization strategy
    operators = {
        'EMAIL_ADDRESS': OperatorConfig('replace', {'new_value': '<EMAIL>'}),
        'PHONE_NUMBER': OperatorConfig('replace', {'new_value': '<PHONE>'}),
        'CREDIT_CARD': OperatorConfig('replace', {'new_value': '<CREDIT_CARD>'}),
        'IP_ADDRESS': OperatorConfig('replace', {'new_value': '<IP>'}),
        'PERSON': OperatorConfig('replace', {'new_value': '<PERSON>'}),
        'LOCATION': OperatorConfig('replace', {'new_value': '<LOCATION>'}),
        'DATE_TIME': OperatorConfig('keep', {})  # Dates/times can usually be kept
    }
    
    # Anonymize
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators=operators
    )
    
    return anonymized.text
```

### 4.3.3 Chinese PII Handling

Presidio has relatively limited support for Chinese. For Chinese pre-training data, it is usually necessary to supplement rule-based matching with regular expressions.

```python
import re

class ChinesePIIFilter:
    """Chinese PII filter"""
    
    patterns = {
        'phone': [
            r'1[3-9]\d{9}',  # Mobile number
            r'0\d{2,3}-?\d{7,8}',  # Landline
        ],
        'id_card': [
            r'\d{17}[\dXx]',  # 18-digit ID
            r'\d{15}',  # 15-digit ID
        ],
        'email': [
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        ],
        'bank_card': [
            r'\d{16,19}',  # Bank card number (needs context judgment)
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
        """Find all PII"""
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
        """Anonymize PII"""
        findings = self.find_pii(text)
        
        # Process in reverse order by position to avoid affecting subsequent positions
        findings.sort(key=lambda x: x['start'], reverse=True)
        
        for finding in findings:
            placeholder = f"<{finding['type'].upper()}>"
            text = text[:finding['start']] + placeholder + text[finding['end']:]
        
        return text
```

### 4.3.4 PII Processing Strategy Trade-offs

PII processing faces trade-offs between accuracy and recall. Overly aggressive filtering may incorrectly remove normal content (e.g., misidentifying ordinary number sequences as phone numbers), while overly conservative filtering may miss true sensitive information.

In practice, a stratified strategy is recommended. For high-risk PII (e.g., ID numbers, bank card numbers), use stricter matching rules—better to over-filter than miss. For medium-risk PII (e.g., phone numbers, emails), use moderate thresholds to balance accuracy and recall. For low-risk information (e.g., dates, locations), decide whether to process based on specific scenarios.

Another important decision is the replacement strategy. Common choices include: complete deletion, simple but may disrupt sentence fluency; fixed placeholder replacement (e.g., `<EMAIL>`), preserves semantic information but may introduce unnatural patterns; random generation replacement (e.g., replace real email with random email), closest to original distribution but complex to implement. Most pre-training datasets use placeholder replacement as a balance between accuracy and complexity.

---

## 4.4 Benchmark Decontamination

When evaluating the true capabilities of large models, a key question is: did the model truly "learn" to solve problems, or did it merely "memorize" the test questions? If training data contains original questions from benchmarks like GSM8K, MMLU, or HumanEval, high scores on these tests are meaningless.

This is the "Benchmark Contamination" problem. As the scale of LLM training data explodes, benchmark test content is repeatedly reposted, discussed, and analyzed on the internet, easily mixing into web-crawled training corpora. This is a critical engineering step for evaluating a model's true capabilities.

### 4.4.1 Types and Risks of Contamination

Benchmark contamination can be categorized into two types:

**Direct contamination**: Training data contains original questions or slight variants from benchmark test sets. For example, GSM8K math problems reposted on educational websites, or MMLU multiple-choice questions appearing on an online quiz platform.

**Indirect contamination**: Training data contains detailed explanations and answers to benchmark test questions. While not the original questions themselves, models may "indirectly memorize" answers through these explanations.

### 4.4.2 Decontamination Detection Methods

The core approach to decontamination is: match training data against known benchmark test sets and remove matched content.

**N-gram overlap detection** is the most commonly used method. Compute the n-gram overlap ratio between training documents and benchmark test samples; when overlap exceeds a threshold, mark the document as contaminated. GPT-3, LLaMA, and other models' training all adopted this method.

```python
from collections import Counter
from typing import Set, List

class BenchmarkDecontaminator:
    """Benchmark test set decontaminator"""
    
    def __init__(self, ngram_size: int = 13, threshold: float = 0.8):
        """
        Args:
            ngram_size: n-gram size, GPT-3 uses 13-gram
            threshold: Overlap ratio threshold, above this is considered contamination
        """
        self.ngram_size = ngram_size
        self.threshold = threshold
        self.benchmark_ngrams: Set[tuple] = set()
    
    def load_benchmarks(self, benchmark_datasets: dict):
        """
        Load benchmark test sets
        
        Args:
            benchmark_datasets: {"name": [sample text list]}
        """
        for name, samples in benchmark_datasets.items():
            for sample in samples:
                ngrams = self._extract_ngrams(sample)
                self.benchmark_ngrams.update(ngrams)
        
        print(f"Loaded {len(self.benchmark_ngrams)} unique {self.ngram_size}-grams")
    
    def _extract_ngrams(self, text: str) -> Set[tuple]:
        """Extract n-gram set from text"""
        text = ' '.join(text.lower().split())
        words = text.split()
        
        ngrams = set()
        for i in range(len(words) - self.ngram_size + 1):
            ngram = tuple(words[i:i + self.ngram_size])
            ngrams.add(ngram)
        
        return ngrams
    
    def check_contamination(self, document: str) -> dict:
        """
        Check if a single document is contaminated
        
        Returns:
            {
                'is_contaminated': bool,
                'overlap_ratio': float,
                'matched_ngrams': int
            }
        """
        doc_ngrams = self._extract_ngrams(document)
        
        if len(doc_ngrams) == 0:
            return {'is_contaminated': False, 'overlap_ratio': 0.0, 'matched_ngrams': 0}
        
        matched = doc_ngrams & self.benchmark_ngrams
        overlap_ratio = len(matched) / len(doc_ngrams)
        
        return {
            'is_contaminated': overlap_ratio > self.threshold,
            'overlap_ratio': overlap_ratio,
            'matched_ngrams': len(matched)
        }
    
    def decontaminate(self, documents: list) -> list:
        """Batch decontamination filtering"""
        clean_docs = []
        contaminated_count = 0
        
        for doc in documents:
            result = self.check_contamination(doc['text'])
            if not result['is_contaminated']:
                clean_docs.append(doc)
            else:
                contaminated_count += 1
        
        print(f"Removed {contaminated_count} contaminated documents "
              f"({contaminated_count/len(documents)*100:.2f}%)")
        return clean_docs

# Usage example
decontaminator = BenchmarkDecontaminator(ngram_size=13, threshold=0.8)

# Load common benchmark test sets
benchmarks = {
    'gsm8k': ["Janet's ducks lay 16 eggs per day...", ...],
    'mmlu': ["What is the capital of France? A) London B) Paris...", ...],
    'humaneval': ["def has_close_elements(numbers: List[float]...", ...],
}
decontaminator.load_benchmarks(benchmarks)

# Decontaminate training data
clean_data = decontaminator.decontaminate(training_documents)
```

### 4.4.3 Engineering Best Practices

1. **Maintain a benchmark library**: Maintain a library containing all common benchmark test sets, including GSM8K, MMLU, HumanEval, MBPP, HellaSwag, ARC, WinoGrande, etc. Every time new data is processed, it must be checked against this library.
2. **Multi-granularity detection**: Besides n-gram overlap, also use MinHash LSH from the previous section for fuzzy matching to catch rewritten test questions.
3. **Regular updates**: New benchmark test sets keep emerging; the decontamination library needs regular updates.
4. **Documentation and reporting**: Clearly disclose decontamination methods and results in model technical reports — this is a basic requirement for responsible AI research (see LLaMA 3 and DeepSeek technical reports).

---

## 4.5 Model-based Quality Scoring

In Section 4.1, we introduced heuristic rule-based quality filtering. These rules are fast and effective but cannot capture deeper quality differences. For example, an advertising article that passes all heuristic checks and a high-quality technical article that also passes may receive the same score under heuristic rules.

**Model-based quality scoring** uses lightweight machine learning models for more refined quality assessment. This approach was widely adopted in LLaMA 2 training — Meta's team used a fastText classifier to identify "textbook-quality" web pages, significantly improving the overall quality of pre-training data.

### 4.5.1 fastText Quality Classifier

fastText is the most commonly used quality scoring tool because it has extremely fast inference speed and can run efficiently on TB-scale data. The core approach is:

1. **Build training set**: Sample positive examples from high-quality sources (e.g., Wikipedia, academic journals, curated websites), and negative examples from low-quality sources (spam pages, ad pages).
2. **Train classifier**: Train a binary classification model using fastText.
3. **Batch scoring**: Score all data to be processed, then filter or stratified-sample based on scores.

```python
import fasttext
import random

def build_quality_training_data(
    high_quality_texts: list,
    low_quality_texts: list,
    output_path: str
):
    """
    Build fastText quality classification training data
    
    Args:
        high_quality_texts: High-quality text list (e.g., Wikipedia articles)
        low_quality_texts: Low-quality text list (e.g., spam pages)
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        for text in high_quality_texts:
            clean_text = ' '.join(text.split()[:500])  # Take first 500 words
            f.write(f"__label__hq {clean_text}\n")
        
        for text in low_quality_texts:
            clean_text = ' '.join(text.split()[:500])
            f.write(f"__label__lq {clean_text}\n")

def train_quality_classifier(training_data_path: str, model_path: str):
    """Train quality classifier"""
    model = fasttext.train_supervised(
        input=training_data_path,
        lr=0.1,
        epoch=25,
        wordNgrams=2,
        dim=100,
        loss='softmax'
    )
    model.save_model(model_path)
    return model

class ModelBasedQualityScorer:
    """Model-based quality scorer"""
    
    def __init__(self, model_path: str):
        self.model = fasttext.load_model(model_path)
    
    def score(self, text: str) -> float:
        """
        Score text quality
        
        Returns:
            Quality score between 0-1, higher is better
        """
        text = ' '.join(text.split()[:500])
        labels, probs = self.model.predict(text, k=2)
        
        for label, prob in zip(labels, probs):
            if label == '__label__hq':
                return prob
        return 0.0
    
    def filter_by_quality(self, documents: list, 
                         min_score: float = 0.5) -> list:
        """Filter documents by quality score"""
        filtered = []
        for doc in documents:
            score = self.score(doc['text'])
            if score >= min_score:
                doc['quality_score'] = score
                filtered.append(doc)
        return filtered
```

### 4.5.2 BERT-based Fine-grained Quality Assessment

For higher-precision quality assessment needs, BERT or its variants can be used for classification. This approach is more accurate than fastText but slower for inference, suitable for fine-grained classification of borderline samples after fastText coarse filtering.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BERTQualityScorer:
    """BERT-based fine-grained quality scorer"""
    
    def __init__(self, model_name: str = 'bert-base-chinese'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.model.eval()
    
    def score_batch(self, texts: list) -> list:
        """Batch scoring"""
        encodings = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)
            quality_scores = probs[:, 1].tolist()  # High-quality class probability
        
        return quality_scores
```

### 4.5.3 Hierarchical Application of Quality Scoring

In practice, a "coarse screening + fine screening" two-stage strategy is recommended:

1. **Stage one**: Use fastText for rapid scoring of all data, removing obviously low-quality content (e.g., score < 0.3).
2. **Stage two**: For borderline samples (e.g., scores between 0.3-0.7), use BERT for fine-grained classification.
3. **Quality stratification**: Divide data into high, medium, and low tiers based on final scores, assigning different sampling weights during training.

This stratification strategy was adopted by Meta in LLaMA 2 training — by increasing sampling weights for high-quality data, model performance on various benchmarks was significantly improved.

---

## 4.6 Complete Cleaning Pipeline

Connect the components introduced above to build a complete data cleaning pipeline.

### 4.6.1 Pipeline Architecture

An industrial-grade cleaning pipeline typically includes the following stages, executed in order:

**Phase 1: Format standardization.** Convert data from various sources to unified format, handle encoding issues, extract necessary metadata.

**Phase 2: Language filtering.** Use FastText for language detection, retain documents in target language. For mixed-language documents, classify by primary language.

**Phase 3: Heuristic filtering.** Apply heuristic rules for length, special characters, duplicate lines, etc., to quickly filter obviously low-quality content.

**Phase 4: Intra-document deduplication.** Remove duplicate paragraphs and n-grams within documents.

**Phase 5: PII cleaning.** Identify and anonymize sensitive personal information.

**Phase 7: Benchmark decontamination.** Use N-gram overlap detection to remove documents highly overlapping with benchmark test sets.

**Phase 8: Quality scoring.** Use fastText/BERT quality classifiers for refined quality scoring of data.

**Phase 9: Perplexity scoring.** Compute quality metrics like perplexity to provide basis for subsequent quality stratification.

**Phase 10: Inter-document deduplication.** Use MinHash LSH for large-scale fuzzy deduplication to remove highly similar documents.

**Phase 11: Quality stratification and sampling.** Stratify data by quality score, determine sampling weights for each tier.

```python
import ray
from dataclasses import dataclass
from typing import Optional

@dataclass
class CleaningConfig:
    """Cleaning configuration"""
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
        """Process single document"""
        text = doc.get('text', '')
        
        # Phase 2: Language filtering
        lang, conf = self.lang_filter.detect(text)
        if lang != self.config.target_language:
            return None
        
        # Phase 3: Heuristic filtering
        passed, reason = self.heuristic_filter.filter(text)
        if not passed:
            return None
        
        # Phase 4: Intra-document deduplication
        text = remove_duplicate_paragraphs(text)
        
        # Phase 5: PII cleaning
        if self.config.anonymize_pii and self.pii_filter:
            text = self.pii_filter.anonymize(text)
        
        # Phase 6: Quality scoring
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
        """Run complete pipeline"""
        # Read data
        ds = ray.data.read_parquet(input_path)
        
        # Phases 1-6: Single document processing
        ds = ds.map(self.process_document)
        ds = ds.filter(lambda x: x is not None)
        
        # Phase 7: Inter-document deduplication
        ds = self.deduplicator.deduplicate(ds)
        
        # Save results
        ds.write_parquet(output_path)
```

### 4.6.2 Quality Monitoring and Iteration

The cleaning pipeline is not a one-time task but a process requiring continuous monitoring and iterative optimization. The following monitoring mechanisms are recommended:

**Filter rate monitoring**: Statistics filter rate for each stage. If a stage suddenly filters out a large amount of data, it may indicate improper threshold settings or changed data distribution.

**Sample inspection**: Regular manual inspection of cleaning results to evaluate accuracy of filtering rules. Both incorrectly deleted good samples and missed bad samples need attention.

**Downstream feedback**: Model evaluation results after training are the final quality validation. If model performance is poor, need to trace back and analyze whether data has issues.

---

## 4.7 Chapter Summary

This chapter systematically introduced the core technologies of pre-training data cleaning and quality control.

In heuristic filtering: language detection uses FastText to quickly filter target language documents; perplexity filtering uses KenLM to evaluate text quality; the heuristic rule set covers multiple dimensions including length, special characters, duplicate lines, and vocabulary diversity. Quality stratification strategy divides data into different tiers, providing basis for subsequent sampling.

In large-scale deduplication: we clearly distinguished exact and fuzzy deduplication as two technical approaches. Exact deduplication uses hash methods to quickly remove identical documents; fuzzy deduplication uses the MinHash LSH algorithm to identify highly similar content. Distributed implementation is necessary for TB-scale data. Intra-document deduplication handles paragraph and n-gram level duplicates.

In privacy cleaning: PII detection can use Presidio or custom regex rules; anonymization strategy requires trade-offs between accuracy and information retention. Chinese PII handling requires specially designed rule sets.

In benchmark decontamination: this is a critical engineering step for ensuring evaluation validity. Through N-gram overlap detection and fuzzy matching, training data content containing benchmark test sets (GSM8K, MMLU, HumanEval, etc.) is removed, preventing models from "memorizing answers" rather than truly learning.

In quality scoring: model-based quality assessment (fastText/BERT) complements heuristic rules by more precisely distinguishing "textbook-quality" content from ordinary web pages. The hierarchical strategy (coarse screening + fine screening) is the best engineering practice for balancing efficiency and precision.

The complete cleaning pipeline connects each component, executing in the order of format standardization, language filtering, heuristic filtering, intra-document deduplication, PII cleaning, benchmark decontamination, quality scoring, perplexity scoring, inter-document deduplication, and quality stratification. Continuous quality monitoring and iterative optimization are key to ensuring data quality.

![Figure 4-5: Chapter Knowledge Structure](../../images/part2/图4_5_本章知识结构.png)

*Figure 4-5: Chapter 4 Knowledge Structure — Three core themes: Heuristic Filtering, Large-Scale Deduplication, PII Cleaning*

---

## Further Reading

For in-depth content on data cleaning, the following resources are worth referencing:

The RefinedWeb paper documents the complete cleaning flow for building high-quality pre-training sets from Common Crawl. The Dolma dataset technical report introduces Allen AI's cleaning strategy and tools. The text-dedup open-source library (github.com/ChenghaoMou/text-dedup) provides implementations of various deduplication algorithms. Microsoft Presidio documentation (microsoft.github.io/presidio) is the authoritative reference for PII processing. The CCNet paper introduces Facebook's method for processing Common Crawl data, especially details on perplexity filtering.

---

## Next Chapter Preview

In the next chapter "Tokenization and Serialization," we will explore the final critical step in pre-training data preparation: how to convert cleaned text into token sequences that models can understand. You will learn the principles and selection of tokenization algorithms like BPE, WordPiece, and Unigram; how to expand vocabularies for specific domains; and data mixing and curriculum learning sampling strategies.

Consider this question as you enter the next chapter: If you are training a model specialized for code, what problems would the standard GPT-2 tokenizer encounter?
