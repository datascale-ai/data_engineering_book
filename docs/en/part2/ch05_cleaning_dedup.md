# Chapter 5: Cleaning, Deduplication, and Decontamination

## Abstract

This chapter covers the critical steps that turn raw text into trainable pre-training data: rule filtering, model-based quality scoring, text normalization, exact deduplication, fuzzy deduplication, semantic deduplication, PII redaction, and benchmark decontamination. We first explain how repetition, low information density, privacy leakage, and evaluation contamination are amplified during training. We then introduce a layered cleaning framework that combines rules, models, and human review. The deduplication section moves from SHA-256 exact matching to MinHash LSH and embedding similarity, emphasizing the double risk of under-deduplication and over-deduplication. The privacy and decontamination sections discuss structured PII, named entities, API keys, and N-gram fingerprinting for benchmark isolation. Finally, anonymized composite cases and three implementation tiers show how teams can deploy cleaning pipelines according to scale and staffing. Readers should be able to design a traceable, reviewable, and iterative text-cleaning system for their own model project.

## Keywords

Data cleaning; deduplication; MinHash; PII redaction; benchmark decontamination; quality scoring; text normalization; human sampling

## Learning Objectives

- Explain how repetition, low-quality text, PII, and benchmark contamination affect pre-training models.
- Combine rule filters, model scoring, and human sampling into a layered cleaning workflow.
- Distinguish exact, fuzzy, and semantic deduplication and their operational boundaries.
- Design detection and isolation strategies for structured PII, API keys, and benchmark contamination.
- Choose lightweight, standard, or platform-level cleaning configurations according to team size and corpus scale.

## Opening: Why Did a "Clean-Looking" Corpus Make the Model Repeat Itself?

The following anonymized composite case illustrates a common failure. After careful source acquisition, a team started pre-training a 7B Chinese base model. The training run looked healthy: loss decreased smoothly, GPU utilization stayed above 90%, and infrastructure metrics were normal. The first evaluation revealed something strange. In continuation tasks, the model repeated identical sentences several times in one answer. With a simple trigger phrase, it could recite the exact template of an e-commerce product description.

This was classic overfitting caused by duplicated data. The team discovered that a large e-commerce source had been crawled through many different URL paths and archive timestamps. Similar product-description templates appeared tens of thousands of times. URL-level exact deduplication did not catch the problem because the URLs differed.

The lesson is central to this chapter: **cleaning is not merely deleting bad data; it is an engineering system that builds the quality ceiling of the training set**.

---

## 5.1 Why Cleaning Determines the Quality Ceiling

### 5.1.1 Cleaning Investment Has Nonlinear Training Returns

FineWeb (Penedo et al. 2024) shows that different cleaning strategies applied to Common Crawl-scale data can lead to large downstream differences. A multi-stage cleaning pipeline can outperform simple heuristic cleaning even when token volume is similar. The exact improvement depends on model scale, corpus structure, benchmark mix, and thresholds, so results should not be copied blindly across projects.

The important shift is conceptual. In compute-constrained pre-training, spending GPU time on high-quality data is usually better than spending the same compute on noisy or repetitive data. Cleaning therefore has a high ROI in the LLM development chain.

### 5.1.2 How Upstream Defects Are Amplified During Training

Small defects in the upstream data pipeline are magnified by gradient accumulation and repeated exposure.

**Memorization of repeated content.** If a passage appears hundreds of times, the model is likely to memorize it and reproduce it when given related prompts. This harms generalization and creates copyright and privacy risk.

**PII activation.** Unredacted phone numbers, emails, ID numbers, addresses, names, passwords, and API tokens can be learned statistically. A model may later generate plausible or real personal information in response to user questions.

**Inflated benchmark scores.** If test questions and answers appear in training data, benchmark performance is artificially high. This is a serious research-integrity issue and can force re-evaluation or public correction.

---

## 5.2 A Cleaning Framework Combining Rules, Models, and Humans

No single method can clean a production corpus. Effective systems combine rule filtering, model filtering, and human sampling, with each layer handling the defect types it is best suited for.

![Figure 5-1: Cleaning and decontamination pipeline overview](../../images/part2/cleaning_pipeline_overview.png)

*Figure 5-1: Cleaning and decontamination pipeline overview. Multi-stage quality gates refine raw corpora into final training data, with typical retention ratios shown by stage.*

### 5.2.1 First Gate: Rule-Based Filtering

Rule-based filtering is the first and cheapest defense. It can remove 40-60% of obviously poor documents without model inference.

**Language identification** is the starting point for multilingual corpora. FastText language ID (`lid.176.bin`) remains a practical choice because it supports 176 languages and can process large volumes quickly. In Chinese corpora, use confidence thresholds above 0.8. Mixed-language documents below that threshold may deserve special routing rather than immediate deletion.

**Length and character-ratio filters** catch common bad documents. Typical thresholds include a minimum length of 200 characters, a maximum length of 100,000 characters, a special-character ratio below 30%, and a digit ratio below 30%. These thresholds should be calibrated by source.

**Repeated-line filters** catch template noise such as repeated navigation bars, copyright notices, and advertising blocks. If more than 30% of non-empty lines are duplicates, the document should be reviewed or filtered.

**Listing 5-1: Multi-rule heuristic quality filter**

```python
import re
from typing import Tuple


class HeuristicQualityFilter:
    """Heuristic quality filter for Chinese pre-training text."""

    def __init__(self):
        self.rules = {
            "min_chars": 200,
            "max_chars": 100_000,
            "max_special_ratio": 0.30,
            "max_digit_ratio": 0.30,
            "max_dup_line_ratio": 0.30,
            "min_unique_word_ratio": 0.10,
        }

    def run(self, text: str) -> Tuple[bool, str]:
        n = len(text)
        if not (self.rules["min_chars"] <= n <= self.rules["max_chars"]):
            return False, "length"
        if len(re.findall(r"[^\w\s]", text, re.UNICODE)) / n > self.rules["max_special_ratio"]:
            return False, "special_chars"
        if len(re.findall(r"\d", text)) / n > self.rules["max_digit_ratio"]:
            return False, "digit_ratio"
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines and (1 - len(set(lines)) / len(lines)) > self.rules["max_dup_line_ratio"]:
            return False, "dup_lines"
        words = text.split()
        if words and len(set(words)) / len(words) < self.rules["min_unique_word_ratio"]:
            return False, "low_diversity"
        return True, "pass"
```

### 5.2.2 Second Gate: Model-Based Quality Scoring

Rules catch obvious defects, but they struggle with grammatical advertising, SEO articles, and low-density but well-formatted text. Model-based filtering provides finer judgment.

**Perplexity filtering** is widely used. With a KenLM n-gram model, high-quality news and encyclopedic text often falls roughly in the 100-300 range, ordinary web text in the 200-500 range, and gibberish or machine-translated spam above 500. Very low perplexity can also be suspicious because it may indicate boilerplate or repetitive templates. These thresholds are not interchangeable with neural reference models; a LLaMA-like reference model produces a different numeric range.

**Quality classifiers** are used in datasets such as RefinedWeb and Dolma. A fastText classifier or lightweight BERT model can be trained from human-labeled high-quality and low-quality documents. This catches quality problems that humans recognize but rules and perplexity miss, at the cost of annotation and inference.

### 5.2.3 Division of Labor Across Rules, Models, and Humans

The cost-effective division is:

- **Rules** handle the largest volume and obvious defects. They are fast and cheap but should be moderately loose to avoid over-deletion.
- **Models** handle the ambiguous middle after rule filtering. They are more accurate but more expensive.
- **Human sampling** audits batches rather than reviewing every record. A 500-1,000 document sample per batch can reveal systematic false positives and false negatives and feed rule/model updates.

### 5.2.4 Text Normalization: Making Data Speak One Format

After quality filtering, text normalization removes format variation that would otherwise fragment the tokenizer vocabulary and confuse downstream training.

**Unicode normalization** should standardize text to NFC so visually identical characters share the same byte representation.

**Full-width and half-width punctuation** should be standardized according to the target model. For most Chinese LLMs, consistent punctuation prevents visually inconsistent output and reduces token fragmentation.

**Whitespace cleanup** should normalize line endings, remove zero-width characters, strip line margins, and compress repeated blank lines.

**Traditional/simplified Chinese conversion** should match product goals. A mainland-focused model may convert traditional text to simplified Chinese; a multilingual or pan-Chinese model should keep both variants and treat region-specific vocabulary carefully.

**Listing 5-2: Text normalization**

```python
import re
import unicodedata


def normalize_text(text: str, to_simplified: bool = False) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    if to_simplified:
        try:
            import opencc

            text = opencc.OpenCC("t2s").convert(text)
        except ImportError:
            pass
    return text.strip()
```

---

## 5.3 Deduplication: From Exact Matching to Semantic Near-Deduplication

### 5.3.1 Why Deduplication Is the Hardest Cleaning Step

Deduplication is computationally heavy. A 50 TB corpus may contain billions of documents. Exact deduplication needs global hashing and comparison; fuzzy deduplication cannot naively compute pairwise similarity because O(n^2) is impossible at that scale.

Insufficient deduplication causes memorization and repetitive generation. Excessive deduplication deletes legitimate diversity, especially in specialized domains where high-quality documents naturally overlap in topic and terminology.

### 5.3.2 Exact Deduplication With SHA-256

Exact deduplication computes a fixed fingerprint, usually SHA-256, for each document. Identical hashes are exact duplicates, and only one record is retained. This is simple and O(n), but it only catches byte- or character-identical documents. It misses near-duplicates with different headers, footers, timestamps, or small edits.

In distributed pipelines, Ray Data or Spark can group by hash and retain one document per group. At tens of billions of documents, exact deduplication can still finish in hours on a moderate CPU cluster if storage and shuffle are provisioned correctly.

### 5.3.3 Fuzzy Deduplication With MinHash LSH

Fuzzy deduplication identifies document pairs whose similarity exceeds a threshold, such as Jaccard similarity above 0.8. MinHash LSH is the industrial standard for TB-scale fuzzy deduplication because it finds likely similar pairs without comparing every pair.

The method has three steps:

1. **N-gram shingling.** Convert a document into a set of character or word n-grams, often 5-grams for Chinese.
2. **MinHash signature compression.** Use many hash functions to turn the shingle set into a fixed-length signature. Signature agreement estimates Jaccard similarity.
3. **LSH banding.** Split the signature into bands. Documents that match in any band become candidate duplicates and are then verified with exact similarity.

**Listing 5-3: MinHash LSH fuzzy deduplication**

```python
import numpy as np
from typing import Set


class MinHashLSH:
    """MinHash LSH for Chinese 5-gram fuzzy deduplication."""

    def __init__(self, num_hashes=128, num_bands=16, ngram=5):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows = num_hashes // num_bands
        self.ngram = ngram
        rng = np.random.default_rng(42)
        self.a = rng.integers(1, 2**31, num_hashes)
        self.b = rng.integers(0, 2**31, num_hashes)
        self.p = (1 << 31) - 1
        self.buckets = [{} for _ in range(num_bands)]

    def ngrams(self, text: str) -> Set[int]:
        compact = text.lower().replace(" ", "")
        return {hash(compact[i : i + self.ngram]) % self.p for i in range(len(compact) - self.ngram + 1)}

    def signature(self, shingles: Set[int]) -> np.ndarray:
        sig = np.full(self.num_hashes, np.inf)
        for shingle in shingles:
            hashed = (self.a * shingle + self.b) % self.p
            sig = np.minimum(sig, hashed)
        return sig.astype(np.int64)

    def insert(self, doc_id: str, text: str) -> list[str]:
        sig = self.signature(self.ngrams(text))
        candidates = set()
        for band_idx in range(self.num_bands):
            key = tuple(sig[band_idx * self.rows : (band_idx + 1) * self.rows])
            if key in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][key])
            self.buckets[band_idx].setdefault(key, []).append(doc_id)
        return list(candidates)
```

Production implementations should persist buckets to distributed storage or run inside a distributed processing engine.

### 5.3.4 Semantic Deduplication With Embeddings

Hashing and n-grams capture surface similarity. If two articles express the same event with different wording, MinHash may miss them. Semantic deduplication uses embedding models such as BGE-M3 or text2vec and approximate-nearest-neighbor indexes such as FAISS or Milvus. If cosine similarity exceeds a high threshold, for example 0.95, the pair can be treated as semantically redundant.

Because embedding inference is GPU-expensive, semantic deduplication is usually the final stage: exact hashing and MinHash remove most redundancy first, and embeddings are applied to the remaining high-value subset.

### 5.3.5 The Double Risk: Too Much and Too Little Deduplication

Deduplication should not be maximally aggressive. If thresholds are too low, documents about the same topic but with different expression are wrongly removed, damaging diversity. If thresholds are too high or only exact deduplication is used, near-duplicates remain and cause memorization.

A practical starting point is Jaccard similarity 0.7-0.8 for MinHash and cosine similarity 0.9-0.95 for semantic deduplication. The final choice should be validated through proxy-model experiments and downstream metrics.

---

## 5.4 PII Redaction and Personal Privacy Protection

### 5.4.1 Common PII Types and Their Harm

Personally identifiable information is often invisible during loss monitoring and benchmark evaluation, yet it creates severe deployment risk. Chinese corpora commonly contain phone numbers, national ID numbers, email addresses, home addresses, names, account credentials, passwords, and API tokens. API keys and passwords are especially dangerous because a model may reproduce a real secret in generated code.

### 5.4.2 Detection and Redaction Strategy

PII detection usually combines regular expressions and NER models.

Regex rules work well for structured PII such as phone numbers, emails, IDs, IP addresses, and API tokens.

**Listing 5-4: Structured PII detection and redaction**

```python
import re

PII_PATTERNS = {
    "phone_cn": r"(?<!\d)1[3-9]\d{9}(?!\d)",
    "id_card_cn": r"\d{17}[\dXx]",
    "email": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "api_key": r"(?i)(sk-|api[_\-]?key|token)[a-zA-Z0-9]{16,}",
    "ip_addr": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}


def detect_and_redact_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for pii_type, pattern in PII_PATTERNS.items():
        if re.findall(pattern, text):
            found.append(pii_type)
            text = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", text)
    return text, found
```

NER models cover less structured entities such as real names, addresses, organizations, and locations. Chinese spaCy models or open-source HuggingFace NER models can identify PER, LOC, and ORG entities, after which context rules decide whether redaction is required.

---

## 5.5 Benchmark Contamination Detection and Decontamination

Benchmark contamination occurs when test questions and answers leak into training data, causing artificially high benchmark scores. It often enters indirectly: benchmark questions appear in blog posts, discussion threads, GitHub examples, or academic commentary that later get crawled.

### 5.5.1 Contamination Paths

The important feature is indirectness. No one intentionally adds the test set; it is absorbed through the open web. A blog explaining GSM8K, a repository containing HumanEval examples, or a forum discussion quoting MMLU questions can all create contamination.

### 5.5.2 Detection and Isolation

The standard method is N-gram overlap. Precompute fingerprint sets for evaluation data such as MMLU, GSM8K, HumanEval, CEVAL, and internal benchmarks. Scan training documents, and move any document above the overlap threshold into quarantine rather than deleting it immediately.

**Listing 5-5: Evaluation-set N-gram fingerprints**

```python
def build_eval_ngrams(eval_texts: list[str], n: int = 13) -> set[str]:
    ngrams = set()
    for text in eval_texts:
        tokens = text.lower().split()
        ngrams.update(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return ngrams


def contamination_score(doc: str, eval_ngrams: set[str], n: int = 13) -> float:
    tokens = doc.lower().split()
    if len(tokens) < n:
        return 0.0
    doc_ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not doc_ngrams:
        return 0.0
    hits = sum(1 for gram in doc_ngrams if gram in eval_ngrams)
    return hits / len(doc_ngrams)
```

Decontamination should happen before the official training set is frozen. Because benchmark sets evolve, fingerprint libraries should be updated and historical data rescanned on a schedule.

---

## 5.6 Quality Scoring, Sampling, and Iterative Loops

### 5.6.1 Multi-dimensional Quality Scores and Stratified Sampling

Cleaning should not be a binary keep/delete decision. A document should carry a quality vector that downstream samplers can use.

**Listing 5-6: Multi-dimensional document-quality score**

```python
from dataclasses import dataclass


@dataclass
class DocumentQualityScore:
    doc_id: str
    noise_score: float
    ppl_score: float
    dedup_status: str
    pii_found: list[str]
    contamination_rate: float

    @property
    def quality_tier(self) -> str:
        if self.ppl_score < 200 and not self.pii_found and self.contamination_rate < 0.05:
            return "high"
        if self.ppl_score < 500 and len(self.pii_found) <= 1:
            return "medium"
        return "low"
```

A common sampling policy is to give High-tier data 2x training weight, Medium-tier data 1x, and Low-tier data 0.3x rather than deleting all low-tier data. This preserves some diversity while emphasizing better material.

### 5.6.2 Human Sampling Loop

The quality loop is based on human audit driving rule iteration, not humans processing every document.

![Figure 5-2: Quality filter funnel and human sampling loop](../../images/part2/quality_filter_funnel_loop.png)

*Figure 5-2: Quality filter funnel and human sampling loop. The left side shows retention by stage; the right side shows human sampling feeding rule and model updates.*

After each cleaning batch, randomly sample about 500 documents. Reviewers label them as OK, noise, missed PII, false deletion of high-quality content, or missed near-duplicate. Track which filter step caused the problem. If an error type exceeds 5% for two consecutive batches, review the corresponding threshold or model.

---

## 5.7 Common Defects, Detection Methods, and Costs

**Table 5-1: Common defects and detection methods**

| Defect | Typical symptom | Detection method | Cost of missing it | Suggested threshold / tool |
| :--- | :--- | :--- | :--- | :--- |
| HTML / noise residue | Tags, CSS, JS mixed into body text | Special-character ratio, regex | Model emits markup or garbage | Special-character ratio below 0.15 |
| Wrong language | Non-target language appears | FastText language ID | Wrong language distribution | Confidence >= 0.8 |
| Low information density | SEO stuffing, ads, loops | KenLM PPL, classifier | Hollow output | PPL roughly 100-500 |
| Exact duplicate | Same document appears repeatedly | SHA-256 global dedup | Memorization | Keep one hash |
| Near duplicate | Reposted article with edits | MinHash LSH | Repetitive generation | Jaccard below 0.8 |
| PII leakage | Phone, ID, email, API key | Regex + NER + sampling | Privacy incident | Zero tolerance with review |
| Benchmark contamination | Test item in training set | 13-gram overlap | Inflated evaluation | Quarantine if overlap > 0.5 |
| Low lexical diversity | Very low type-token ratio | TTR < 0.1 | Rigid vocabulary | TTR >= 0.1 |

**Table 5-2: Impact of cleaning actions**

The gains below are engineering examples as of 2026-06. Actual gains depend on corpus, model size, evaluation set, thresholds, and training configuration.

| Cleaning action | Symptom if skipped | Potential benefit when done | Cost cycle |
| :--- | :--- | :--- | :--- |
| Language filtering | Mixed-language answers | Better language consistency | CPU, hours |
| Heuristic filtering | Markup and ad words in output | 5-10% fluency improvement in some settings | CPU, hours |
| PPL filtering | Hollow filler text | Higher information density | CPU + small model, days |
| MinHash dedup | Repetition and low diversity | 20-40% generation-diversity gain in some settings | Distributed CPU, days |
| PII redaction | Privacy leakage | Compliance baseline | CPU + GPU NER, days |
| Decontamination | Inflated scores | More trustworthy evaluation | CPU, hours |
| Stratified sampling | High-quality data diluted | 3-7% benchmark gain in some settings | No extra compute |

---

## 5.8 Large-Scale Case Review

The following anonymized composite cases use approximate scales and ratios to illustrate engineering decisions.

### Case 1: Knowledge Loss From Over-Cleaning

A team tightened every cleaning threshold after observing noisy samples. It raised minimum length, lowered MinHash similarity thresholds, deleted documents with many line breaks, and removed conversational text with nonstandard punctuation. The resulting corpus looked cleaner in dashboards, but a small-model A/B test showed worse code generation, weaker dialogue style, and degraded recall of common facts.

Root-cause review found three mistakes. First, code and Markdown were harmed by newline normalization rules designed for web prose. Second, conversational data was removed because it looked "nonstandard," even though it was valuable for dialogue. Third, aggressive MinHash thresholds deleted legitimate topical overlap. The fix was to split cleaning routes by domain: code keeps indentation and newlines, dialogue uses separate quality rules, and domain-specific high-value terms are protected during deduplication.

### Case 2: PII Misses and Security Exposure

Another team relied on regexes for phone numbers and emails but did not scan code repositories for secrets. During red-team testing, the model generated a string that resembled a real API token. The investigation found historical forum posts and sample configuration files that contained credentials.

The remediation added API-key patterns, entropy-based secret detection, NER review for names and addresses, and a quarantine queue for sensitive documents. It also introduced a policy that any redacted document must preserve a redaction log, so auditors can inspect what was removed without exposing the original secret to training.

---

## 5.9 Minimum Viable Production Cleaning Pipelines

### 5.9.1 Lightweight Setup: 1-3 People, Less Than 100 GB

Use a simple pipeline: language ID, length and character filters, Unicode normalization, SHA-256 exact deduplication, regex PII detection, and 500-sample human review per release. Store metadata in Parquet or a small relational database. This tier is enough for prototypes and small domain corpora.

### 5.9.2 Standard Setup: 4-10 People, 100 GB to 10 TB

Add KenLM or a lightweight neural quality scorer, MinHash LSH fuzzy deduplication, benchmark N-gram decontamination, quality tiers, DVC or lakeFS versioning, and a batch dashboard. Human sampling should be stratified by source and quality tier.

### 5.9.3 Platform Setup: More Than 10 People, More Than 10 TB

At platform scale, cleaning becomes a service. Use distributed processing with Ray or Spark, persistent LSH buckets, embedding-based semantic deduplication for high-value subsets, full provenance ledgers, automated alerts, policy-managed PII redaction, and release gates before data is exposed to training jobs. Every rule change should run against regression sandboxes from multiple domains.

## Chapter Summary

Cleaning is a quality system, not a one-off script. A production pipeline must combine cheap rules, model-based scoring, human audit, layered deduplication, privacy redaction, benchmark decontamination, quality tiers, and release-version governance. The goal is not to make the corpus superficially tidy; it is to preserve useful diversity while removing defects that would be amplified by training.

## References

- Broder, A. Z. (1997). On the resemblance and containment of documents.
- Heafield, K. (2011). KenLM: Faster and Smaller Language Model Queries.
- Honnibal, M. et al. (2020). spaCy: Industrial-strength Natural Language Processing in Python.
- Indyk, P., and Motwani, R. (1998). Approximate nearest neighbors.
- Joulin, A. et al. (2017). Bag of Tricks for Efficient Text Classification.
- Penedo, G. et al. (2023). The RefinedWeb Dataset for Falcon LLM.
- Penedo, G. et al. (2024). FineWeb: Decanting the Web for the Finest Text Data at Scale.
- Soldaini, L. et al. (2024). Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research.
