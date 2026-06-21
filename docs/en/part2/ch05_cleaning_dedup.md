# Chapter 5: Cleaning, Deduplication, and Decontamination

<div class="chapter-authors">Jun Yu; Ke Wang; Changwen Chen</div>

## Abstract

This chapter discusses the critical steps for transforming raw corpora into training-ready text pretraining data, covering rule-based filtering, model quality scoring, text normalization, exact deduplication, fuzzy deduplication, semantic deduplication, PII redaction, and benchmark decontamination. The chapter begins by explaining how repetition, low information density, privacy leakage, and evaluation contamination are amplified during training, then presents a cleaning framework that combines rules, models, and manual spot-checks. The deduplication section extends from SHA-256 exact matching to MinHash LSH and embedding similarity, emphasizing the dual risks of under- and over-deduplication. The privacy and decontamination sections address the detection of structured PII, named entities, and secrets such as API keys, as well as N-gram fingerprint isolation of evaluation sets. Finally, the chapter illustrates practical deployment paths for cleaning pipelines through anonymized composite case studies and three-tier team configurations. Readers should be able to design text cleaning schemes with traceability, spot-check capability, and iterative refinement, tailored to their data scale, team resources, and target model capabilities.

## Keywords

Data cleaning; deduplication; MinHash; PII redaction; benchmark decontamination; quality scoring; text normalization; manual spot-check

## Learning Objectives

- Explain the impact of repetition, low-quality text, PII, and benchmark contamination on pretrained models.
- Combine rule-based filtering, model scoring, and manual spot-checks to form a layered cleaning pipeline.
- Distinguish the applicable boundaries of exact deduplication, fuzzy deduplication, and semantic deduplication.
- Design detection and isolation strategies for structured PII, API keys, and evaluation contamination.
- Select lightweight, standard, or platform-level cleaning solutions based on team size.

## Opening: A Dataset That "Looked Clean" — Why Did the Model Start Repeating Itself?

The following is an anonymized composite case study. After completing the data collection described in Chapter 4, a team launched pretraining for a Chinese base model. On the surface, training was stable: the loss curve descended smoothly, and GPU utilization reached the project baseline. Then the first benchmark evaluation results arrived, and the team discovered a puzzling phenomenon: during continuation tasks, the model repeatedly generated identical sentences, sometimes repeating the same sentence many times within a single response. Even stranger, given a simple trigger phrase, the model could recite the complete template of a product description from a major e-commerce platform, word for word.

This is a classic case of **overfitting caused by data repetition**. During the investigation, engineers discovered that the training data contained a corpus of product descriptions from a large e-commerce platform that had been crawled dozens of times through different URL paths. This caused the same product description format to appear tens of thousands of times in the training set. Although URL-based exact deduplication had been applied at data ingestion time, the content's URLs differed (different product URLs, archived URLs for the same product crawled at different times), so exact deduplication completely failed to catch this problem.

This case illustrates a core proposition: **cleaning is not simply deleting low-quality data — it is an engineering system that establishes the quality ceiling of training data.** In this sense, the cleaning chapter is one of the central chapters of text pretraining data engineering.

---

## 5.1 Why Cleaning Determines the Quality Ceiling of Training Data

### 5.1.1 The Non-Linear Relationship Between Cleaning Investment and Training Return

The FineWeb project (Penedo et al. 2024) provides a quantitative answer: for Common Crawl data at the same scale, different cleaning strategies lead to significantly different training outcomes. Data processed through a fine-grained multi-stage cleaning pipeline outperforms data cleaned with simple rules on downstream benchmarks. The specific gains vary with model scale, corpus structure, and evaluation sets and cannot be directly reused without considering the experimental setup.

This finding corrects the early coarse-grained assumption that "as long as data volume is large enough, quality matters less." When compute resources are limited, **allocating limited compute to high-quality data typically outperforms allocating the same compute to low-quality data**. From this perspective, the ROI of data cleaning engineering investment is among the highest in the LLM development pipeline.

### 5.1.2 How Upstream Defects Are Exponentially Amplified During Training

Minor defects in upstream data pipelines, accumulated through the gradient updates of training, manifest as order-of-magnitude amplification effects downstream. Several typical amplification pathways include:

**The "memorization" effect of repeated content**: If a passage appears more than 100 times in the training set, the model will memorize it with very high probability and reproduce it verbatim when exposed to related trigger phrases. This behavior is fundamentally overfitting to the training set — it not only degrades the model's generalization ability but also introduces serious copyright and privacy risks (the model may recite users' personal information verbatim).

**"Hallucination" activation of PII data**: Personal information that has not been redacted in training data (phone numbers, email addresses, national ID numbers) is learned by the model through statistical association. When a user asks questions related to someone's phone number, the model may generate information that appears plausible but actually points to a real individual, creating PII leakage risks.

**"Score inflation" from benchmark contamination**: If the training data contains questions and answers from test sets (Benchmark Contamination), the model's performance on those benchmarks will be artificially inflated. This is an extremely sensitive integrity issue in the industry — multiple organizations have been forced to issue retractions and conduct re-evaluations after model releases.

---

## 5.2 A Collaborative Cleaning Framework: Rules, Models, and Human Review

Faced with the problems above, industrial practice has demonstrated that no single technique can independently achieve high-quality data cleaning — an effective cleaning system must be a **collaborative combination of rule-based filtering, model-based filtering, and manual spot-checks**, with each method covering different defect types and having its own optimal use case.

![Figure 5-1: Overview Flowchart of the Cleaning and Decontamination Pipeline](../../images/part2/cleaning_pipeline_overview.svg)

*Figure 5-1: Overview Flowchart of the Cleaning and Decontamination Pipeline — A multi-stage quality gate gradually refines raw corpus into candidate training corpus. The proportions in the figure are illustrative only; real retention rates depend on source quality, filtering thresholds, and compliance requirements. Source: original illustration from this book.*

### 5.2.1 The First Gate: Rule-Based Filtering

Rule-Based Filtering is the first line of defense in the cleaning pipeline and the most cost-effective stage. Based on a set of quantifiable heuristic rules, it rapidly eliminates obviously low-quality documents without running any model. The filtering proportion depends on source quality and cannot be reused across projects.

**Language identification** is the essential starting point for cleaning multilingual corpora. The FastText multilingual identification model (Joulin et al. 2017) (`lid.176.bin`, supporting 176 languages) is a common engineering choice. In practice, confidence thresholds should be calibrated through manual spot checks; low-confidence mixed-language documents (such as technical blog posts mixing Chinese and English) can be retained and handled separately rather than discarded directly.

**Length and character ratio filtering** forms the most basic rule set. The typical approach is to set project thresholds for minimum document length, maximum document length, special-character ratio, and digit-character ratio. Overly short content is usually navigation bars or tag text; overly long documents may be merged multi-page content and require segmentation. Logs, code, and data tables should use independent thresholds to avoid being mistakenly removed by general-text rules.

**Duplicate line ratio filtering** specifically targets "template noise" — many low-quality web pages repeat navigation bars, copyright notices, and advertisement areas multiple times on the same page, resulting in a large number of identical lines within the document. If the duplicate-line ratio exceeds the project threshold, the document can be treated as a low-quality candidate, triggering further review or direct discarding.

Listing 5-1 shows a sample implementation of a multi-rule heuristic quality filter.

*Listing 5-1: Example code for multi-rule heuristic quality filtering. Thresholds are demonstration configurations; production environments should calibrate them by language, source, and manual spot-check results.*

```python
import re
from typing import Tuple

class HeuristicQualityFilter:
    """Multi-rule heuristic quality filter (optimized for Chinese pretraining data)"""

    def __init__(self):
        self.rules = {
            'min_chars': 200,
            'max_chars': 100_000,
            'max_special_ratio': 0.30,
            'max_digit_ratio': 0.30,
            'max_dup_line_ratio': 0.30,
            'min_unique_word_ratio': 0.10,
        }

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
        words = text.split()
        if words and len(set(words)) / len(words) < self.rules['min_unique_word_ratio']:
            return False, 'low_diversity'
        return True, 'pass'
```

### 5.2.2 The Second Gate: Model-Based Quality Scoring

Rule-based filtering can quickly eliminate "obviously" low-quality content, but it is generally powerless against a passage that is grammatically correct and well-formatted yet is essentially meaningless keyword stuffing or SEO filler text. This is where **Model-Based Filtering** is needed — using trained scoring models to make finer-grained judgments about the linguistic quality of documents.

**Perplexity Filtering** is the most widely adopted model-based filtering method. Computing perplexity with a **KenLM (Heafield 2011) n-gram language model** (rather than a neural network model) can quantify text quality, but PPL thresholds must be bound to the corpus used to train KenLM, the tokenization method, and the target language: high-quality news, encyclopedias, ordinary web pages, garbled text, and advertisement stuffing all exhibit different distributions. Production systems should first establish per-source baselines on manually confirmed samples, then set blocking waterlines through percentiles or Z-score anomaly detection. It is worth noting that lower perplexity is not always better: abnormally low perplexity may indicate highly homogeneous boilerplate text, such as nearly fixed-format legal clauses or product descriptions, and also deserves attention. (Note: if a neural-network reference model is used to compute perplexity, the numeric distribution changes; thresholds from the two approaches must not be mixed.)

**Quality Classifiers** are the advanced technique adopted by representative datasets such as RefinedWeb (Penedo et al. 2023) and Dolma (Soldaini et al. 2024). Note that empirical research by Nait Saada et al. (2025) shows classifier filtering effectively performs "domain selection" rather than "absolute quality" selection, so it must be verified through manual spot checks. The method fine-tunes a fastText or lightweight BERT classifier on a human-annotated dataset of high-quality versus low-quality documents, casting quality scoring as a supervised binary or five-class classification problem. This approach has a significant advantage in covering quality issues that "neither rules nor perplexity can detect, but humans can judge," at the cost of some manual annotation effort to build the training set.

### 5.2.3 Three-Stage Collaboration: Appropriate Division of Labor Among Rules, Models, and Humans

From a practical cost-effectiveness perspective, the appropriate division of labor among the three methods is:

**Rule-based filtering** handles the largest volume and most obvious problems, operating at extremely high speed and very low cost, but with a higher error rate (both false negatives and false positives can occur); it is suited for the "broad filtering" first stage.

**Model-based filtering** handles the remaining ambiguous cases after rule filtering, with higher precision than rules but slower speed and higher cost (requiring model inference); it is suited for the "fine filtering" second stage with medium precision requirements.

**Manual spot-checks** do not process every record but instead serve as "quality audits" through sampled verification of cleaning results: randomly sampling 500–1,000 records per batch for human review by data engineers, identifying systemic errors (such as a category of valuable content being erroneously discarded by rules), and feeding findings back into iterative optimization of rules and models. This is the critical closure point of the quality feedback loop.

### 5.2.4 Text Normalization: Making Data "Speak the Same Language"

After completing quality filtering, there is another class of work that is often overlooked but has far-reaching impact on downstream tokenizers and model training — **Text Normalization**. Data from different sources often exhibits large variations in encoding format, punctuation conventions, and whitespace usage. If these differences are not unified during the cleaning stage, tokenizers will recognize them as thousands of "different tokens," not only increasing vocabulary pressure but also interfering with the model's unified representation of semantically equivalent content.

**Unicode normalization** is the most fundamental step. The same Chinese character may be encoded in NFC (composed form) or NFD (decomposed form), causing text that appears identical at the character level to have different underlying byte sequences, which will be treated as different documents during exact deduplication. It is recommended to normalize all text uniformly to `unicodedata.normalize('NFC', text)` — this requires only a single line of Python code but eliminates a large number of encoding differences arising from different operating systems and editors.

**Full-width/half-width character unification** is a unique challenge in preprocessing Chinese data. In Chinese web corpora, full-width characters (such as `，。！""（）`) and half-width characters (`,. ! "" ()`) are mixed extremely commonly. For most large model training scenarios, it is recommended to unify to half-width punctuation — this is consistent with the punctuation conventions of the vast majority of high-quality training data (academic papers, technical documentation, English Wikipedia) and prevents models from mixing full-width and half-width punctuation in Chinese responses, causing visual inconsistency.

**Extraneous whitespace cleanup** includes compressing consecutive spaces into a single space, removing leading and trailing spaces from lines, converting Windows-style line endings (`\r\n`) to Unix-style (`\n`), and removing zero-width invisible characters (such as `\u200b` and `\ufeff` BOM markers). These invisible characters originate from text pasting and format conversion across different platforms, and may produce unexpected `<unk>` tokens during model tokenization, affecting training stability.

**Traditional/simplified Chinese handling strategy** is of particular importance for Chinese large models. When training a fully parametrized multi-dialect Chinese model, traditional and simplified characters should coexist; however, if the target is a model focused on Mainland simplified Chinese, it is advisable to convert traditional Chinese corpora to simplified Chinese (the `opencc` library can perform high-quality traditional-to-simplified conversion). Note: mechanical traditional-to-simplified conversion will lose some vocabulary and expressions unique to Taiwan, Hong Kong, and other regions. For vertical domain models involving these regions, careful handling is required rather than blanket unification.

Listing 5-2 shows a sample implementation of text normalization.

**Listing 5-2: Sample Code for Text Normalization**

```python
import unicodedata, re

def normalize_text(text: str, to_simplified: bool = False) -> str:
    """
    Text normalization: Unicode normalization + punctuation unification + whitespace cleanup.
    to_simplified: whether to convert traditional Chinese to simplified
                   (requires opencc-python-reimplemented to be installed).
    """
    # 1. Unicode NFC normalization
    text = unicodedata.normalize('NFC', text)
    # 2. Remove zero-width characters and BOM
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\u00ad]', '', text)
    # 3. Normalize Windows line endings to Unix
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # 4. Compress consecutive spaces and tabs
    text = re.sub(r'[ \t]+', ' ', text)
    # 5. Strip leading and trailing spaces from each line
    text = '\n'.join(line.strip() for line in text.split('\n'))
    # 6. Compress consecutive blank lines (more than 2 blank lines → 1 blank line)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 7. Optional: traditional-to-simplified conversion
    if to_simplified:
        try:
            import opencc
            converter = opencc.OpenCC('t2s')
            text = converter.convert(text)
        except ImportError:
            pass  # skip if opencc is not installed
    return text.strip()
```

Although text normalization has an inconspicuous effect on individual documents, at the scale of a TB-level corpus it effectively reduces vocabulary fragmentation in the tokenizer, decreases the diversity of token representations for semantically equivalent content in the training set, and allows the model to focus its attention on genuine semantic learning rather than formatting differences. This is a step in the cleaning pipeline with extremely low cost and consistently stable returns; it is recommended as a standard step in projects of any scale.

---

## 5.3 Deduplication: From Exact Matching to Semantic Near-Duplicate Detection

### 5.3.1 Why Deduplication Is the Most Challenging Step in the Cleaning System

Deduplication is the step in the cleaning system with the highest engineering complexity and the heaviest computational cost. Consider a 50 TB corpus: exact deduplication requires computing a hash value for every document and performing a global comparison — at the scale of 5 billion documents, the I/O cost alone is already substantial. Fuzzy deduplication requires computing similarity between billions of document pairs; a naïve O(n²) implementation is completely infeasible at this scale.

Research shows that inadequately deduplicated pretraining corpora lead to two types of destructive effects: first, the model overfits to repeated content (the root cause in the opening case study); second, the model tends to "repeat itself" during continuation, producing a "broken-record" phenomenon that severely degrades the diversity and fluency of generated output.

### 5.3.2 Exact Deduplication: SHA-256 Hashing

Exact Deduplication computes a fixed-length hash fingerprint (typically SHA-256) for each document; documents with the same hash value are treated as exact duplicates, and only the first occurrence is retained. This approach is simple to implement and extremely fast, completing in O(n) time, but it **can only handle documents that are character-for-character identical** and cannot detect "slightly different" near-duplicates (e.g., the same article republished on a different website with a different header and footer added).

In distributed settings, this can be efficiently implemented using the `groupBy` operator in Ray Data or Spark: the document hash value serves as the grouping key, and only one document is retained per group. At the scale of tens of billions of documents, exact deduplication can complete in a matter of hours on a CPU cluster of 8–16 nodes.

### 5.3.3 Fuzzy Deduplication: The Three-Step Principle and Engineering Implementation of MinHash LSH

The goal of Fuzzy Deduplication is to identify document pairs whose "similarity exceeds a threshold (e.g., Jaccard similarity > 0.8)" and retain only one version from each such pair. MinHash LSH (Broder 1997; Indyk and Motwani 1998) is the industrial standard algorithm for fuzzy deduplication at TB-scale data volumes. Its core idea is to identify truly similar document pairs with high probability while dramatically reducing computational cost.

**Step 1: N-gram decomposition**. Convert each document into a set of character-level or word-level n-grams (typically 5-grams). The Jaccard similarity between two documents is defined as the ratio of the intersection to the union of their n-gram sets.

**Step 2: MinHash signature compression**. Using k random hash functions, generate a MinHash signature vector of length k for the n-gram set of each document (typically k = 128). The key property of MinHash is that the match rate between two documents' signatures is an unbiased estimate of their Jaccard similarity.

**Step 3: LSH banding**. Divide the 128-dimensional signature vector into b bands (each band containing r = 128/b dimensions). Two documents are placed into the same "candidate bucket" as long as their signatures match exactly within any single band; only document pairs within the same bucket then require an exact similarity computation. Adjusting b and r controls the trade-off between the effective similarity detection threshold and recall rate.

Listing 5-3 shows a sample implementation of MinHash LSH fuzzy deduplication. In production environments, the bucket structure should be persisted to distributed storage or a stream processing framework.

*Listing 5-3: Example code for MinHash LSH fuzzy deduplication. This snippet illustrates the algorithm structure; production environments should persist bucket structures, candidate pairs, and review results to distributed storage.*

```python
import hashlib
import re
import numpy as np
from typing import Set

class MinHashLSH:
    """MinHash LSH fuzzy deduplication implementation (suited for Chinese 5-grams)"""

    def __init__(self, num_hashes=128, num_bands=16, ngram=5, threshold=0.8):
        self.num_hashes  = num_hashes
        self.num_bands   = num_bands
        self.rows        = num_hashes // num_bands
        self.ngram       = ngram
        # Random parameters (linear hash family)
        rng = np.random.default_rng(42)
        self.a = rng.integers(1, 2**31, num_hashes)
        self.b = rng.integers(0, 2**31, num_hashes)
        self.p = (1 << 31) - 1              # Mersenne prime
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
        """Insert a document and return duplicate IDs confirmed by true Jaccard similarity."""
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

### 5.3.4 Semantic Deduplication: Embedding Similarity Beyond Literal Matching

Both exact hash deduplication and N-gram-based MinHash LSH are fundamentally capturing the "literal features" of text. But if the same news story is rewritten by two media outlets using entirely different vocabulary (synonym substitution, sentence restructuring), with very low literal overlap, traditional LSH will fail entirely. To address this, **Semantic Deduplication** introduces embedding models.

In practice, a lightweight embedding model (such as `BGE-M3` or `text2vec`) is used to encode each document as a dense vector, and an Approximate Nearest Neighbor (ANN) index is built using a vector database (such as Milvus or FAISS). If the cosine similarity between two document vectors exceeds 0.95, they are identified as highly semantically homogeneous content and deduplicated — even if their literal content is completely different.

Since computing embeddings requires significant GPU inference compute, the industry typically treats it as the **last stage of the deduplication pipeline**: hash deduplication and MinHash first filter obvious redundancy at extremely low cost, and then semantic deduplication is applied to the remaining high-value corpus, achieving a balance between computational cost and deduplication precision.

### 5.3.5 The Dual Risks of Deduplication: Over- and Under-Deduplication

More aggressive deduplication is not always better. **Over-deduplication** (thresholds set too low, e.g., Jaccard > 0.5, or semantic cosine similarity threshold too low) will cause large numbers of documents that are topically related but expressed differently to be erroneously treated as duplicates and deleted, harming data diversity — this is especially dangerous in specialized domains, where high-quality domain knowledge inherently has substantial topical overlap. **Under-deduplication** (thresholds too high, or performing only exact deduplication) will leave large amounts of near-duplicates, leading to the overfitting and broken-record problems described above.

In practice, the recommended starting thresholds are a Jaccard similarity of 0.7–0.8 and a semantic similarity of 0.9–0.95, with ablation experiments on proxy model evaluations across different threshold configurations to find the optimal point for the current corpus and target task.

---

## 5.4 PII Redaction and Personal Privacy Protection

### 5.4.1 Common PII Types and Their Harms

Personally Identifiable Information (PII) is the most deeply hidden and most latently harmful category of defects in training data. Unlike noise and repetition, the presence of PII does not directly affect the loss metric or benchmark scores, but creates serious privacy leakage risks once the model is deployed.

The most common types of PII in Chinese training corpora include: mobile phone numbers (11-digit numbers beginning with 1), national ID card numbers (18 digits, containing date of birth and region code), email addresses, home addresses and postal codes, names (high-frequency personal names and organization names), account passwords and API tokens (especially prevalent in code repositories and technical forums). Among these, account credentials and token PII are the most immediately harmful — the model may reproduce real API keys verbatim from the training set when generating code examples, causing immediate security incidents.

### 5.4.2 Detection and Redaction Approaches

PII detection typically adopts a **rules + NER model** combined approach:

**Regular expression rules** achieve very high recall for structured PII (phone numbers, email addresses, ID card numbers, IP addresses, etc.) and run at extremely high speed, making them well-suited as the first detection layer:

Listing 5-4 shows a sample implementation of structured PII detection and redaction.

**Listing 5-4: Sample Code for Structured PII Detection and Redaction**

```python
import re

PII_PATTERNS = {
    "phone_cn":   r"(?<!\d)1[3-9]\d{9}(?!\d)",         # Chinese mobile phone number
    "id_card_cn": r"\d{17}[\dXx]",                      # National ID card number
    "email":      r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "api_key":    r"(?i)(sk-|api[_\-]?key|token)[a-zA-Z0-9]{16,}",
    "ip_addr":    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}

def detect_and_redact_pii(text: str) -> tuple[str, list]:
    """Detect and redact PII in text; return the redacted text and a list of PII types found."""
    found = []
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            found.append(pii_type)
            text = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", text)
    return text, found
```

**Named Entity Recognition (NER) models** cover PII types that are difficult to enumerate with rules, such as real personal names, addresses, and organization names. It is recommended to use spaCy (Honnibal et al. 2020) with its Chinese model (`zh_core_web_trf`) or open-source Chinese NER models available on HuggingFace to identify named entities such as persons (PER), locations (LOC), and organizations (ORG), then determine based on context whether redaction is necessary.

---

## 5.5 Benchmark Contamination Detection and Decontamination

Benchmark Contamination refers to the phenomenon whereby questions and answers from test or evaluation sets are accidentally mixed into the training data, causing the model's performance on those evaluation sets to be artificially inflated. This is the most sensitive integrity issue in LLM training data quality governance, and an engineering challenge that has received increasing attention in recent years as the LLM evaluation ecosystem has matured.

### 5.5.1 Contamination Propagation Pathways

Contamination pathways are often surprising: evaluation set questions appear in a public technical blog, which is then crawled into the training corpus by a web scraper; forum posts on academic community sites discussing evaluation set results contain original questions, entering the corpus through Reddit, Zhihu, or similar platform data; early versions of evaluation sets included as examples in public GitHub repositories enter the corpus through code data. The common characteristic of these pathways is **indirectness** — no one deliberately placed test sets in the training data, yet contamination still occurred naturally.

### 5.5.2 Detection and Isolation Approaches

The most commonly used decontamination approach is **N-gram overlap detection**: pre-compute 13-gram fingerprint sets for all evaluation sets (MMLU (Hendrycks et al. 2021), GSM8K (Cobbe et al. 2021), HumanEval (Chen et al. 2021), CEVAL, etc.), then scan each document in the training data; any document whose 13-gram match rate with any evaluation set exceeds 50% is flagged as a "contamination risk" and moved to an isolation zone (not deleted directly, but quarantined first for subsequent review):

Listing 5-5 shows a sample implementation of evaluation set N-gram fingerprint construction and overlap rate computation.

**Listing 5-5: Sample Code for Evaluation Set N-gram Fingerprints and Contamination Rate Computation**

```python
from collections import Counter

def build_eval_ngrams(eval_texts: list[str], n=13) -> set[str]:
    """Build the N-gram fingerprint set for an evaluation set."""
    ngrams = set()
    for text in eval_texts:
        tokens = text.lower().split()
        ngrams.update(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return ngrams

def contamination_score(doc: str, eval_ngrams: set[str], n=13) -> float:
    """Compute the N-gram overlap rate between a document and the evaluation set."""
    tokens = doc.lower().split()
    if len(tokens) < n:
        return 0.0
    doc_ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    if not doc_ngrams:
        return 0.0
    hits = sum(1 for g in doc_ngrams if g in eval_ngrams)
    return hits / len(doc_ngrams)
```

Decontamination work should be systematically completed before the formal training set is established, not patched incrementally during training. Because the scope of evaluation sets expands over time as new benchmarks continually emerge, engineering teams need to periodically update the fingerprint database and re-scan.

---

## 5.6 Quality Scoring, Spot-Checks, and Closed-Loop Iteration

### 5.6.1 Multi-Dimensional Quality Scoring and Stratified Sampling

Cleaning should not be a binary black-or-white judgment; instead, each document should be assigned a multi-dimensional quality score vector for use in subsequent stratified sampling:

Listing 5-6 shows a sample definition of a multi-dimensional document quality score object.

**Listing 5-6: Sample Code for a Multi-Dimensional Document Quality Score Object**

```python
from dataclasses import dataclass

@dataclass
class DocumentQualityScore:
    doc_id: str
    noise_score: float      # Noise score (special character ratio, etc.); lower is better
    ppl_score: float        # Perplexity; lower means higher quality
    dedup_status: str       # "unique" / "near-duplicate" / "exact-duplicate"
    pii_found: list[str]    # List of PII types found; empty list means clean
    contamination_rate: float  # Benchmark contamination rate; lower is better

    @property
    def quality_tier(self) -> str:
        """Determine quality tier based on multi-dimensional scores."""
        if self.ppl_score < 200 and not self.pii_found and self.contamination_rate < 0.05:
            return "high"
        if self.ppl_score < 500 and len(self.pii_found) <= 1:
            return "medium"
        return "low"
```

Stratified sampling strategy: High-tier data is given a 2x sampling weight during training, Medium-tier 1x, and Low-tier 0.3x (rather than discarding entirely, preserving diversity).

### 5.6.2 Manual Spot-Check Feedback Loop

The quality feedback loop is designed around **human-audit-driven rule iteration**, not "human processing of every record" (the latter is completely infeasible at PB-scale corpora).

![Figure 5-2: Quality Filtering Funnel and Spot-Check Feedback Loop](../../images/part2/quality_filter_funnel_loop.svg)

*Figure 5-2: Quality Filtering Funnel and Spot-Check Feedback Loop — The funnel on the left shows the data retention rate at each stage; the feedback loop on the right shows how manual spot-checks drive continuous iterative optimization of filtering rules. Source: original illustration.*

After each cleaning batch is completed, the following "quality snapshot" procedure is executed on a fixed schedule: randomly sample a batch of records for manual annotation by data engineers (OK / noise / missed PII / erroneously discarded high-quality content / near-duplicate slippage), tally the occurrence rate of each error type, and trace which filtering step caused the error (false positive or false negative). When the error rate for any category exceeds the project waterline for multiple consecutive batches, a review and update of the corresponding rule or model threshold must be triggered. This mechanism transforms the cleaning pipeline from a "one-time engineering artifact" into a "continuously iterating quality engine."

---

## 5.7 Common Defects, Detection Methods, and Cost Reference

*Table 5-1: Common defects, detection methods, and cost matrix. Source: compiled by the authors; detection cost is a relative description of engineering complexity, and actual cost depends on data scale, model calls, and infrastructure configuration.*

| Defect Type | Typical Manifestation | Detection Method | Cost of Miss | Recommended Threshold/Tool |
| :--- | :--- | :--- | :--- | :--- |
| **HTML/noise residue** | Tags such as `<div>`, CSS, and JS code mixed into body text | Special-character ratio percentiles; regex rules | Model outputs garbled text/tags | Calibrate the project waterline on manually confirmed samples |
| **Wrong language** | Content outside the target language mixed in | FastText language identification and confidence distribution | Model learns wrong language distribution | Set confidence waterlines by target language and source |
| **Low information density** | SEO keyword stuffing, ad copy, meaningless repetition | KenLM PPL distribution; quality classifier | Model generates hollow, padded text | Establish PPL percentile baselines on the target corpus |
| **Exact duplicates** | Same document crawled multiple times | SHA-256 hash global deduplication | Model overfits specific content | Keep only 1 per identical hash |
| **Near-duplicates** | Same article republished on different sites (slightly modified) | MinHash LSH (Jaccard similarity) | "Broken-record" effect; poor generalization | Choose the Jaccard duplicate waterline through ablation experiments |
| **PII leakage** | Phone numbers, ID cards, emails, API keys | Regex rules + NER model + manual spot-check | Post-deployment privacy incident | Zero tolerance; manual review |
| **Benchmark contamination** | Test set questions mixed into training set | 13-gram comparison against evaluation sets | Inflated benchmark scores; integrity risk | Isolate high-overlap samples and review manually |
| **Low lexical diversity** | Extremely low Type-Token Ratio (boilerplate text) | TTR distribution anomaly | Model vocabulary use becomes rigid | Set TTR baselines by language and content type |

*Table 5-2: Impact comparison of cleaning actions on training outcomes. Source: compiled by the authors; impact directions are engineering-experience summaries, and specific gains must be validated through same-configuration training or proxy-model experiments.*

Note: Table 5-2 is used to illustrate the correspondence between cleaning actions and risk-mitigation directions. It does not provide fixed cross-project gains. Actual effects depend on corpus structure, model scale, evaluation sets, cleaning thresholds, and training configuration, and should be validated through ablation experiments.

| Cleaning Action | Typical Model Symptoms When Skipped | Risk-Mitigation Direction When Fully Applied | Cost/Timeline |
| :--- | :--- | :--- | :--- |
| Language filtering | Model mixes languages; Chinese responses interspersed with English | Improved language consistency | CPU, hours |
| Heuristic rule filtering | Model output is format-disordered (HTML tags / ad slogans) | Reduces format noise and template-text contamination | CPU, hours |
| PPL perplexity filtering | Model tends to generate hollow, padded content | Improves corpus information density and isolates garbled or machine-generated junk | CPU + small model, days |
| MinHash fuzzy deduplication | "Broken-record" phenomenon; high repetition rate in generated content | Reduces over-reinforcement of repeated samples in the probability distribution | Distributed CPU, days |
| PII redaction | Post-deployment privacy leakage incidents; model recites user information | Privacy compliance achieved; legal risks avoided | CPU + GPU NER, days |
| Benchmark decontamination | Inflated benchmark scores; real user experience misaligned with benchmarks | Evaluation integrity achieved; real-world performance more predictable | CPU, hours |
| Quality-stratified sampling | High- and low-quality data at equal weight dilutes the effect of high-quality data | Lets limited training budget be spent more on high-value samples | No additional compute cost |

---

## 5.8 Large-Scale Engineering Case Studies and Issue Retrospectives

All of the following cases are anonymized composite case studies. Data scales, proportions, and timelines are provided to illustrate engineering scope; as of June 2026, actual figures will vary with corpus source, cleaning rules, model scale, and evaluation methodology.

### Case Study 1: Knowledge Loss from Over-Cleaning — The Cost of "Over-Tuning the Threshold" (Anonymized Composite Case)

**Background**: After completing the first round of base-model pretraining, a team planned to upgrade its data cleaning pipeline with the goal of further improving the "purity" of training data. The team raised the standards of its heuristic filtering rules: the minimum document length increased significantly, the PPL threshold tightened substantially, and the MinHash similarity threshold also became more aggressive. After processing, the corpus size shrank noticeably.

**T+0 (Problem discovered)**: The model trained on the new corpus showed a decline in general benchmark performance — especially in specialized domains such as medicine and law, where response quality was noticeably worse than the previous version. Engineers initially suspected training hyperparameter issues.

**T+5 (Root cause identified)**: Differential analysis of the old and new corpora revealed that a large number of specialized domain documents, such as medical science articles, legal regulations, and technical standards, had been eliminated by over-cleaning. Medical science articles are often short and dense, falling below the new minimum-length threshold; legal provisions use highly standardized language and may be misjudged by PPL filtering; the same regulatory content republished across different websites has high similarity and can be heavily deleted by aggressive MinHash thresholds, leaving the legal knowledge base incomplete.

**Key lesson**: Cleaning thresholds should be **configured differentially by domain and content type**, not tuned globally. Specialized domain content (medical, legal, scientific) has high knowledge density, but its document characteristics (length distribution, linguistic regularity, content similarity) are fundamentally different from general web pages. Applying thresholds designed for general web pages to filter specialized content inevitably causes serious knowledge loss. The correct approach is to first classify the corpus by domain, then configure independent cleaning threshold parameters for each domain, and perform independent manual spot-check validation on the processing results for each domain.

---

### Case Study 2: Security Risks from PII Omission (Anonymized Composite Case)

**Background**: Shortly after a company launched an AI assistant product for enterprise users, user feedback quickly arrived: when asked certain technical questions, the model would output strings that looked like real API keys (in the format `sk-xxxxxxxxxxxxxxxxxxxxxxxx`) in generated code examples.

**T+0 (Risk confirmed)**: The security team immediately launched an investigation and confirmed that the model's output was highly consistent with the format of real keys, suspected to originate from hardcoded keys committed to a GitHub repository in the training data. Because the PII redaction pipeline only covered conventional types such as phone numbers, email addresses, and ID card numbers and had not included API keys in its detection scope, these keys had been fully learned during training.

**T+1 (Emergency response)**: The security team notified the service providers owning the keys to perform emergency revocation, while taking the model offline for review. The investigation revealed a batch of code-corpus commit records containing hardcoded API keys or passwords, covering cloud-service keys, code-hosting platform tokens, database connection strings, and other types, none of which had been captured by the existing redaction pipeline.

**T+7 (Fix completed)**: The data team added regex rules targeting API keys and passwords and other structured secrets (referencing GitGuardian's open-source ruleset), and also introduced tools specifically designed to detect secret leakage in code (such as truffleHog and detect-secrets) to perform a full re-scan and re-redaction of the code corpus.

**Key lesson**: The definition of PII must expand as the corpus type expands. "Secrets" in code corpora (API keys, passwords, SSH private keys) and "privacy" in plain text (phone numbers, email addresses) are different types of PII; the former causes more immediate and direct harm but is more easily overlooked. Any team ingesting code corpora must add an independent rule set specifically for **Secrets Detection** in the PII redaction pipeline.

---

## 5.9 Minimum Viable Combinations for Production-Grade Cleaning Pipelines

After understanding all the technical modules in this chapter, a practical question naturally arises: **if resources are limited, which steps are non-negotiable?** Not every team has sufficient engineering resources to fully implement a six-stage cleaning pipeline in the first version. The following presents "minimum viable cleaning combinations" for three different resource levels, for engineering teams to reference and select according to their situation.

### 5.9.1 Lightweight Solution (1–3 Person Data Team, Data Scale < 100 GB)

The lightweight solution focuses on "holding the baseline," filtering out the most harmful defects with minimal engineering investment:

*Table 5-3: Minimum viable combination for the lightweight cleaning solution. Source: compiled by the authors; the combination is a starting recommendation, and production environments should extend it according to risk level, corpus source, and compliance requirements.*

| Step | Implementation | Tools | Required? |
|:--- |:--- |:--- |:--- |
| Language filtering | FastText identification, confidence threshold calibrated by language | fasttext | ★ Required |
| Rule-based filtering | Length, special characters, duplicate lines | Custom Python | ★ Required |
| Exact deduplication | SHA-256 hash global deduplication | hashlib | ★ Required |
| PII redaction | Regex rules (phone / email / ID card / API key) | re | ★ Required |
| Text normalization | Unicode NFC + whitespace cleanup | unicodedata | ★ Required |
| Perplexity filtering | KenLM (optional, add when time permits) | kenlm | △ Recommended |
| MinHash deduplication | Optional (limited benefit at small data scale) | datasketch | ○ Optional |
| Benchmark decontamination | Must be completed before formal training | Custom implementation | ★ Required |

This combination can usually serve as a starting plan for the early experimental phase and covers the "must-have" baseline protections. The trade-off is that it will miss a considerable proportion of near-duplicate content and low-information-density documents, so it should not be directly extrapolated to formal pretraining data releases.

### 5.9.2 Standard Solution (4–10 Person Data Team, Data Scale 100 GB – 10 TB)

The standard solution adds model scoring and fuzzy deduplication on top of the lightweight solution, covering the mainstream quality requirements of industrial practice:

Building on the lightweight solution, the following are added: **KenLM perplexity filtering** (fit a 5-gram language model trained on the target language and set PPL percentile waterlines on manually confirmed samples); **MinHash LSH fuzzy deduplication** (determine Jaccard threshold, signature dimensions, and band count through sample review and proxy-training ablations); **NER model-assisted PII detection** (spaCy Chinese model or similar NER models, covering PII types such as personal names, addresses, and organizations that are difficult to enumerate with rules); and **domain-stratified thresholds** (configure independent filtering parameters for special content types such as code and academic papers, preventing a unified threshold from erroneously discarding them). The engineering cycle and GPU resources required by this solution depend on corpus scale, toolchain maturity, and review intensity; it can serve as a complete baseline solution for medium-sized teams.

### 5.9.3 Platform-Level Solution (10+ Person Data Platform Team, Data Scale > 10 TB)

The platform-level solution targets industrial-scale large-volume data processing, introducing on top of the standard solution: **distributed processing architecture** (Ray Data or Spark on Kubernetes to fully distribute all steps and support multi-node horizontal scaling); **custom quality classifiers** (fine-tune a BERT or fastText classifier on manually annotated high/low quality sample pairs, framing document quality judgment as a supervised classification task); **comprehensive evaluation set decontamination** (maintaining an N-gram fingerprint database covering all major evaluation sets, updated periodically); and **automated quality snapshot dashboards** (automatically generating quality reports after each cleaning batch, displaying key metrics such as per-stage filter rates, quality-score distributions, and PII discovery rates). The build cycle for a complete platform-level solution depends on team size, platform foundation, and compliance requirements; once established, it can support shared reuse of corpus quality infrastructure across all large model projects within the organization.

---

## Chapter Summary

This chapter, organized around the theme of "why cleaning determines the quality ceiling of training data," systematically introduces the complete technical system of rule-based filtering, model scoring, exact deduplication, MinHash fuzzy deduplication, PII redaction, and benchmark decontamination, following the sequence of the cleaning lifecycle. Two tables (Table 5-1 defect–detection–cost matrix; Table 5-2 cleaning action effect comparison) provide engineers with directly usable decision-making tools. Two anonymized composite case studies — "knowledge loss from over-cleaning" and "security risks from PII omission" — validate the need for fine-grained configuration of the cleaning system from both a positive and negative direction.

After cleaning, deduplication, and decontamination are complete, the raw corpus is ready to enter the training input organization stage. The next chapter continues on cleaned data to discuss the final stretch of pretraining data engineering: **Chapter 6: Tokenization, Serialization, and Efficient Loading** — that is, how to transform clean text into token sequences that GPUs can efficiently consume.

## References

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

Chen M, Tworek J, Jun H, Yuan Q, Pinto H P d O, Kaplan J, Edwards H, Burda Y, Joseph N, Brockman G, Ray A, Puri R, Krueger G, Petrov M, Khlaaf H, Sastry G, Mishkin P, Chan B, Gray S, Ryder N, Pavlov M, Power A, Kaiser L, Bavarian M, Winter C, Tillet P, Such F P, Cummings D, Plappert M, Chantzis F, Barnes E, Herbert-Voss A, Guss W H, Nichol A, Paino A, Tezak N, Tang J, Babuschkin I, Balaji S, Jain S, Saunders W, Hesse C, Carr A N, Leike J, Achiam J, Misra V, Morikawa E, Radford A, Knight M, Brundage M, Murati M, Mayer K, Welinder P, McGrew B, Amodei D, Sutskever I, Zaremba W (2021) Evaluating Large Language Models Trained on Code (HumanEval). arXiv preprint arXiv:2107.03374.
