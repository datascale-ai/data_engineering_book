# Project 1: Building a Distributed Mini-C4 Data Pipeline with Ray

<div class="chapter-authors">Xin Xu</div>

## Abstract
P01 focuses on the engineering process of constructing a Mini-C4 training dataset from a Common Crawl shard. The chapter emphasizes not the results of a single crawl, but rather the organization of web archiving, body text extraction, deduplication and filtering, training packaging, and result validation into a reproducible data production pipeline.

This chapter can be understood along four main threads:

- **Data collection and body text extraction**: Extracting training-ready body text from web archives.
- **Cleaning, deduplication, and quality control**: Handling template noise, near-duplicate content, language mixing, and low-quality pages.
- **Training packaging and data splitting**: Organizing processed results into standardized JSONL files and training manifests.
- **Evaluation, validation, and cost boundaries**: Assessing pipeline status through inspection scripts, statistical metrics, and resource consumption.

When read in engineering order, this chapter corresponds to a complete processing chain:

**Web Archive → Body Text Extraction → Basic Cleaning → Near-Duplicate Deduplication → Language Splitting → Quality Filtering → Training Packaging → Evaluation and Validation**

This structure corresponds to the core objective: reproducing an interpretable and reusable web pretraining data pipeline under single-machine CPU and Ray Data conditions. The design of Mini-C4 draws on the fundamental idea from C4/T5 of "constructing trainable text data from web corpora" (Raffel et al. 2020), but this chapter emphasizes small-scale engineering reproduction rather than replicating the full C4 dataset.

---

## Keywords

Mini-C4; Ray Data; distributed cleaning; corpus packaging; quality acceptance

## Project Objectives and Reader Takeaways

This project uses the "Mini-C4 distributed web corpus pipeline" as its core case study, with the goal of constructing an auditable pretraining corpus sample from a Common Crawl shard. Upon completing this chapter, readers should be able to identify the key data objects in this scenario, decompose the engineering chain, set acceptance criteria, and transfer the case methodology to similar data engineering tasks.

## Scenario Constraints and Data Boundaries

Single shard, single-machine CPU, and Ray Data environment, with emphasis on validating web body text extraction, cleaning and deduplication, language splitting, and training packaging. These boundaries ensure the case can be reproduced and audited; when data scale, data sources, permission scope, or deployment environment change, sampling strategies, quality thresholds, operational costs, and compliance requirements must be re-evaluated.

## Architectural Decisions

This project adopts the architectural path of "WARC streaming parsing, heuristic cleaning, MinHash deduplication, language splitting, quality filtering, and manifest acceptance." This decision prioritizes input/output contract clarity, version traceability, anomaly localization, and result verifiability, rather than compressing all logic into a single one-off script run.

## Sample Schema / Data Flow

The core data flow can be summarized as:

Listing P01-1 provides a process or path example to illustrate the input/output relationships, structural constraints, or execution modes in this section.
```text
Common Crawl WARC -> HTML Response -> Body Text -> Deduplicated Corpus -> Language Subset -> train/val JSONL -> Manifest and Evaluation Report
```

The purpose of this snippet is to transform the above process into a checkable, structured representation.

The sample schema should retain at minimum the fields `id`, `source`, `content_or_payload`, `metadata`, `quality_signals`, `split_or_stage`, and `audit_trace`; specific fields are further refined by this project's data types, downstream tasks, and acceptance criteria.

## Core Implementation Snippets

Only key implementation snippets that illustrate design trade-offs are retained in the body text. Complete scripts, lengthy configurations, run logs, and large files should be placed in the companion repository or appendix; code presentation focuses on input/output contracts, quality thresholds, exception handling, and acceptance interfaces.

## Experimental or Acceptance Metrics

Acceptance metrics include extraction success rate, duplication rate, language distribution, quality filter retention rate, train/val integrity, smoke test pass rate, and processing cost. If the project enters production, a course, or a public reproduction experiment environment, version numbers, dependency environments, random seeds, sample spot-check results, and failed sample post-mortem records should also be logged.

## Cost, Risk, and Compliance Boundaries

Computational costs arise primarily from WARC parsing, deduplication, and quality filtering; risks concentrate on web copyright, template noise, low-quality text retention, and language identification errors. When involving external data, personal information, copyrighted content, or third-party services, source descriptions, permission status, desensitization strategies, call records, and manual review records should be retained.

## Common Failure Patterns

Common failures include input distribution drift, missing schema fields, quality thresholds that are too loose or too strict, insufficient evaluation sample coverage, unstable model calls, and non-traceable results. When troubleshooting, prioritize locating data boundaries and intermediate artifacts before inspecting models, toolchains, and deployment environments.

## Reproducibility Resource Notes

Reproduction materials should include data source descriptions, minimal samples, configuration files, run commands, metrics scripts, inspection reports, and artifact directories. The body text retains necessary snippets; complete notebooks, long scripts, and large files are maintained separately as companion resources. Data reading and shard management can reference the engineering patterns of Hugging Face Datasets (Hugging Face 2026) and Ray Data (Ray Project 2026); experiment tracking and quality checks can reference MLflow (MLflow Authors 2026) and Great Expectations (Great Expectations Contributors 2026) respectively.

## 1. Project Background: The Engineering Position of Mini-C4

In large model pretraining, web corpora have consistently been one of the most important data sources. Web data is sufficiently large in scale, sufficiently broad in coverage, and updated frequently, making it naturally suited for constructing general-purpose pretraining corpora.
However, web data also has three highly characteristic problems:

1. **Extremely low signal-to-noise ratio**: HTML pages contain large amounts of non-body content such as navigation bars, ad placements, scripts, footers, copyright notices, cookie prompts, comment sections, and table of contents pages.
2. **Extremely high duplication**: Reposts, mirror sites, aggregation pages, template pages, and partial page copies are very common.
3. **Difficult-to-control distribution**: Different websites, different languages, and different text quality levels are mixed together, easily pulling the training corpus toward a noise distribution.

Therefore, the core of pretraining data engineering is not "obtaining more text," but rather establishing an **interpretable, reproducible, and verifiable data production pipeline** that progressively refines raw web pages into text samples ready for the training system.

This is where Mini-C4's significance lies. It is not a replacement for the industrial-scale full C4, but rather a **minimally reproducible, runnable, and explainable miniature version**.
Through it, readers can reproduce end-to-end the key problems in large-scale web pretraining data processing under single-shard, single-machine CPU conditions, thereby establishing a methodological foundation for subsequent larger-scale data engineering.

---

## 2. Project Objectives and Boundaries

### 2.1 Project Objectives

The goal of this project is not simply to download Common Crawl data, but to complete end-to-end validation of the following chain within controlled boundaries:

> **Web Archive → Body Text Extraction → Basic Cleaning → Near-Duplicate Deduplication → Language Splitting → Quality Filtering → Training Packaging → Evaluation and Validation**

The final outputs include:

- `train.jsonl`
- `val.jsonl`
- `smoke_test.jsonl`
- `training_manifest.json`
- Evaluation reports and inspection reports

This project focuses on **transforming web pages into training data**, not merely stopping at the step of "transforming web pages into text."

### 2.2 Project Boundaries

To keep the project minimally reproducible and controlled, the following boundaries are explicitly defined:

- **Data scale boundary**: Only one Common Crawl shard is processed; industrial full-scale volume is not pursued.
- **Hardware boundary**: Defaults to running on a single-machine CPU environment, without GPU dependency.
- **Parallelism boundary**: Ray is used for single-machine multi-core parallelism in the deduplication stage.
- **Language boundary**: Currently covers primarily English and Chinese; English quality filtering is more complete, while the Chinese quality gate is relatively weaker.
- **Target positioning boundary**: This is an engineering practice case study, not a research project pursuing SOTA metrics.

### 2.3 The Purpose of Boundary Setting

This boundary setting provides two benefits.

First, it ensures the project remains reproducible under limited resource conditions.
If the initial target were multi-machine, massive shards, and complex scheduling, the project would quickly become dominated by infrastructure issues, obscuring the key logic of data engineering itself.

Second, it enables the team to more clearly observe the effect of each filtering step.
At smaller data scales, it is easier to inspect intermediate artifacts, conduct manual sampling, and adjust thresholds, thereby truly understanding why data is retained or removed.

---

## 3. Overall Project Architecture

![Figure P01-1](../../images/part14/p01/p01_01_mini_c4_pipeline_overview.svg)
*Figure P01-1: Mini-C4 Data Pipeline Overview*


### 3.1 Process Overview

The overall project process can be summarized in 10 steps:

1. `src/1_download_data.py`: Download Common Crawl data
2. `src/2_process_warc.py`: Parse WARC and extract body text
3. `src/3_clean_data.py`: Heuristic cleaning
4. `src/4_deduplicate.py`: MinHash deduplication
5. `src/5_split_lang.py`: Split by language
6. `src/6_quality_filter.py`: Quality filtering
7. `src/7_prepare_training_data.py`: Training data packaging
8. `src/8_evaluate_dataset.py`: Dataset evaluation
9. `src/9_training_smoke_test.py`: Training smoke test
10. `src/10_run_p1_checks.py`: Project checks and consistency validation

### 3.2 A Three-Phase Understanding

If the above process is further categorized, it can be divided into three major phases:

#### Phase One: From the Web World to the Text World

This phase primarily addresses the question of "whether text exists," with core tasks of:

- Downloading WARC
- Reading web responses
- Filtering non-HTML content
- Extracting body text from HTML

The focus of this step is to convert complex content from web archives into text as stably as possible.

#### Phase Two: From the Text World to the Corpus World

This phase addresses the question of "whether text can serve as corpus material," and primarily includes:

- Basic cleaning
- Deduplication
- Language splitting
- Quality filtering

That is, actively controlling noise, duplication, and distribution to bring text closer to the form of training corpus.

#### Phase Three: From the Corpus World to the Training Interface

This phase addresses the question of "whether the corpus can be stably fed into the training system," including:

- Deterministic train/val splitting
- Manifest construction
- Smoke test construction
- Evaluation and inspection

Only at this step does the data engineering loop truly close.

---

## 4. Data Acquisition: Engineering Choices for Common Crawl

Common Crawl is one of the most commonly used public sources for constructing web-based pretraining datasets. It stores web crawl results in WARC (Web ARChive) format, preserving HTTP responses, headers, and raw web page content.

The reasons for choosing Common Crawl are primarily threefold:

1. **Large scale**: Can cover a large number of real web page scenarios.
2. **Standardized format**: WARC is a mature web archiving format, suitable for stream processing.
3. **Close to real industrial problems**: Web noise, templates, duplication, and language mixing all appear authentically.

But precisely for this reason, Common Crawl cannot be used directly for training.
Without rigorous extraction and filtering, models would learn large amounts of HTML fragments, copyright pages, table of contents pages, and template garbage text.

Therefore, choosing Common Crawl means choosing a set of problems that more closely resemble a real industrial production environment.

---

## 5. WARC Parsing and Body Text Extraction

### 5.1 Body Text Extraction as the First Critical Gate

Web pages are not inherently equivalent to natural language text. A typical HTML page contains mixed content including:

- Navigation bars
- Breadcrumbs
- Recommendation slots
- JavaScript
- CSS
- Footer links
- Advertisements
- Copyright notices
- Comment sections
- Table layout fragments

If HTML is read directly and tags are stripped naively, what the model sees is often a pile of structural fragments rather than coherent semantic body text.

Therefore, the goal of the body text extraction phase is not "to capture as many characters as possible," but rather **to extract the main content area as accurately as possible**.

### 5.2 Core Component Selection

![Figure P01-2](../../images/part14/p01/p01_02_warc_to_text.svg)
*Figure P01-2: Parsing Path from WARC to Body Text*


| Component | Selection | Reason for Choice |
|---|---|---|
| WARC reading | `warcio` | Standard WARC reading library with streaming support, avoiding the memory pressure of loading large files all at once |
| Body text extraction | `trafilatura` | More stable extraction of main content areas; compared to simple HTML parsing approaches, offers better cleanup of navigation bars, footers, and template areas |

*Table P01-1: Component and Reason for Choice Reference Table*

### 5.3 The Engineering Value of Stream Processing

WARC files are typically large and contain many responses that are not needed.
Loading the entire file into memory at once wastes resources and is not conducive to stable long-process execution.

Therefore, this project uses **streaming traversal** to read WARC records one by one, continuing to process only HTML responses that meet the criteria.
This design reduces peak memory consumption and is more consistent with engineering practices when later scaling to multiple shards.

### 5.4 Core Implementation

Listing P01-2 provides a Python implementation snippet to illustrate the input/output relationships, structural constraints, or execution modes in this section.
```python
from warcio.archiveiterator import ArchiveIterator
import trafilatura

def extract_text_from_warc(warc_path, output_path):
    with open(warc_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type != "response":
                continue

            content_type = record.http_headers.get_header("Content-Type")
            if not content_type or "text/html" not in content_type:
                continue

            text = trafilatura.extract(
                record.content_stream().read(),
                include_comments=False,
                include_tables=False,
                no_fallback=False
            )
```

The purpose of this snippet is to transform the above process into a checkable, structured representation.

Several parameters here are intentional:

- `include_comments=False`: Avoids pulling in high-noise areas like comment sections into the body text.
- `include_tables=False`: Reduces structural noise introduced by table layouts.
- `no_fallback=False`: Allows the extraction component to perform remedial extraction when necessary, improving recall.

### 5.5 The Meaning of Results at This Stage

In the single-shard test, a total of **3,028** candidate body texts were successfully extracted.
This number conveys two things:

First, not all web responses can be converted into usable body text.
Second, body text extraction is already a significant data compression step, because large numbers of raw responses are blocked at stages such as "non-HTML," "empty content," and "extraction failure."

From an engineering perspective, this stage answers:

> In real web data, how many "text-like" candidate texts can the system stably obtain?

---

## 6. Heuristic Cleaning: First-Pass Noise Removal

![Figure P01-3](../../images/part14/p01/p01_03_cleaning_rules.svg)
*Figure P01-3: Heuristic Cleaning Rules Illustration*


### 6.1 The Necessity of Heuristic Cleaning

Even after successful body text extraction, the resulting text is far from being high-quality corpus.
A page may have extracted "body text," but that body text may still be:

- Extremely short text
- A table of contents page
- A tag cloud
- SEO-stitched text
- Code snippets
- System error pages
- Privacy and cookie notices

If such samples are fed directly into the training set, they pollute the model and waste subsequent computational resources.

Therefore, pipelines typically first design a layer of **cheap, fast, and interpretable heuristic cleaning** to intercept the most obviously low-quality text.

### 6.2 Main Cleaning Rules Used in This Project

#### 1) Length Rules

- Discard overly short text, e.g., fewer than 100 characters
- Discard overly long text, e.g., more than 2M characters

The reasons are straightforward:
Overly short text often lacks sufficient semantic information; overly long text may be the product of anomalous concatenation, page stitching, or structural corruption.

#### 2) Average Word Length Rule

If the average word length is noticeably high, such as exceeding 15 characters, the text is likely not natural language, but rather:

- Compressed code output
- URL strings
- Mixed identifiers
- Style fragments

#### 3) Symbol Density Rule

Statistics are collected on the proportion of the following symbols:

Listing P01-3 provides a process or path example to illustrate the input/output relationships, structural constraints, or execution modes in this section.
```text
{ } [ ] < > \
```

The purpose of this snippet is to transform the above process into a checkable, structured representation.

When the proportion of these symbols is too high, the text typically resembles structural fragments more than natural language paragraphs.

#### 4) Blacklist Phrase Rule

For example, intercepting:

- `lorem ipsum`
- `enable cookies`
- `403 forbidden`

These texts are either placeholder content or system prompt pages and have no practical training value.

### 6.3 Characteristics of Heuristic Cleaning

This layer of rules is not aimed at achieving "maximum accuracy," but rather at removing the most obvious problems at low cost.
Its advantages include:

- Fast execution
- Low cost
- Easy to interpret
- Easy to tune parameters
- Suitable as the front end of a funnel

In other words, this stage is not responsible for solving all quality problems, but rather for prioritizing the removal of samples that "almost certainly should not be retained."

### 6.4 Interpreting Results at This Stage

After heuristic cleaning, the sample count decreased from **3,028** to **2,425**.
This indicates that approximately one-fifth of candidate body texts can already be judged as low quality under the most basic text rules.

The significance of this stage is:

> Without relying on expensive model scoring, first compress the coarsest noise, saving resources for subsequent finer-grained processing.

---

## 7. Deduplication: Near-Duplicate Handling in Web Corpora

![Figure P01-4](../../images/part14/p01/p01_04_dedup_minhash_lsh.svg)
*Figure P01-4: MinHash + LSH Deduplication Approach*


### 7.1 How Severe Is the Duplication Problem

Internet text contains large amounts of duplication, including but not limited to:

- Reposted articles
- Aggregation pages
- Mirror sites
- Template pages
- Partial page overlap
- Different layout versions of the same content

Without deduplication, several problems emerge in training corpora:

1. Certain content is excessively repeated, causing distributional imbalance.
2. Models may over-memorize specific templates or sites.
3. Data leakage may occur during subsequent evaluation.
4. Storage and training resources are unnecessarily consumed by duplicate content.

Therefore, deduplication is not an optional enhancement but a mandatory step in web pretraining data engineering.

### 7.2 Why Avoid Pairwise Comparison

Suppose there are \(N\) texts. If pairwise similarity comparison is performed directly, the complexity approaches \(O(N^2)\).
At real data scales, this approach quickly becomes unacceptable.

Therefore, this project adopts the **MinHash + LSH** approach, transforming the problem of "finding similar texts" into "finding similar signatures," thereby reducing processing complexity to a more practical range.

### 7.3 Engineering Intuition Behind MinHash and LSH

- **MinHash**: Maps a piece of text to a shorter signature that approximately reflects the set similarity of the text.
- **LSH (Locality-Sensitive Hashing)**: Makes similar texts more likely to fall into the same candidate bucket, reducing the number of global comparisons.

The result is that the system does not need every text to be compared against all other texts, but rather makes determinations only within candidate sets that are more likely to be similar.

### 7.4 Engineering Considerations for Using Ray

Even with MinHash, generating signatures is still a computationally intensive operation.
Especially as the number of texts increases, single-threaded processing will noticeably slow down the entire pipeline.

Ray plays a very clear role here:
It is not to demonstrate the concept of "distributed computing," but to enable a single-machine multi-core CPU to run batch processing tasks in parallel.

The corresponding implementation is as follows:

Listing P01-4 provides a Python implementation snippet to illustrate the input/output relationships, structural constraints, or execution modes in this section.
```python
import ray
from datasketch import MinHash

@ray.remote
def process_batch(lines, batch_id):
    results = []
    for line in lines:
        item = json.loads(line)
        m = MinHash(num_perm=128)
        for w in item["text"].split():
            m.update(w.encode("utf8"))
        results.append((item["url"], m, item["text"]))
    return results

futures = [process_batch.remote(batch, i) for i, batch in enumerate(batches)]
processed_batches = ray.get(futures)
```

The purpose of this snippet is to transform the above process into a checkable, structured representation.

### 7.5 The Most Common Pitfall Here

The biggest common misconception in Ray parallel processing is:
**Do not dispatch a single text as an independent task.**

This creates massive overhead from small-object serialization and inter-process communication, ultimately making performance worse.
The correct approach is:

- First pack texts into batches
- Then dispatch by batch to workers

For example, batching every 1,000 items is a more reliable engineering choice.

### 7.6 Interpreting Results at This Stage

After deduplication, the sample count decreased from **2,425** to **2,305**.
This indicates that while the duplication problem exists, at this minimal experimental scale the shrinkage from deduplication is not as dramatic as from quality filtering.

However, this does not mean deduplication is unimportant.
On the contrary, the importance of deduplication is reflected in its ability to significantly improve the health of the training distribution, not merely in reducing the count.

---

## 8. Language Splitting: The Necessity of Language-Based Processing

![Figure P01-5](../../images/part14/p01/p01_05_language_split.svg)
*Figure P01-5: Language Splitting and Branch Processing*

### 8.1 Different Languages Cannot Share the Same Quality Gate

Quality judgment of web text is highly dependent on the language itself.
For example, certain perplexity thresholds, word length statistics, or grammatical naturalness rules for English do not apply to Chinese.
Conversely, the common problems with Chinese web pages are not entirely the same as those with English web pages.

Therefore, after deduplication, the project further splits text by language to make quality control more precise, rather than putting all languages through the same filter.

### 8.2 The Project's Approach

The project uses FastText's language identification model `lid.176.ftz` to predict the language of text and splits text into:

- `en`
- `zh`
- `others`

After this, subsequent quality filtering can adopt different strategies based on language.

### 8.3 Language Splitting as a Necessary Intermediate Layer

The value of language splitting is mainly reflected in three aspects:

1. **Avoiding misclassification**: The statistical characteristics of texts in different languages vary greatly.
2. **Facilitating analysis**: The retention rate and interception reasons for each language can be observed independently.
3. **Facilitating extensibility**: In the future, if more languages are added, only a language branch needs to be added at this layer without starting from scratch.

From an engineering organization perspective, language splitting elevates the pipeline from "unified processing" to "pluggable processing."

---

## 9. Quality Filtering: From "Looks Like Text" to "Suitable for Training"

![Figure P01-6](../../images/part14/p01/p01_06_quality_filter.svg)
*Figure P01-6: Quality Filtering Decision Illustration*

### 9.1 Why Quality Filtering Is the Most Critical Gate

Heuristic cleaning and deduplication resolve many explicit problems, but still cannot guarantee that text is truly suitable for training.
Because many pages superficially conform to text rules but may still be:

- Table of contents pages
- Low information density pages
- Pages stacked with repeated sentences
- Pages with fragmented language
- Machine translation residues
- Web noise with very poor grammatical naturalness

At this point, a layer of filtering closer to "language quality" is needed.

### 9.2 English Quality Gate: KenLM Perplexity

This project introduces the KenLM language model on the English side for quality filtering.
The core idea is:

- Use a language model to score text
- Use the score normalized by token count to measure text naturalness
- Filter out obviously unnatural text through thresholds

Empirically, this can be understood as:

- `> -5.0`: Generally closer to high-quality text
- `< -6.0`: Often closer to fragmented sentences, garbled text, or low-quality generated content

This does not mean lower perplexity is always better, but rather that language models can serve as a **signal closer to "natural language quality" than pure rules**.

### 9.3 Main Interception Reasons Observed in This Project

During the quality filtering stage, common interception reasons include:

- `directory_like`: Directory-type web pages with low information density
- `duplicate_lines`: Too many repeated lines within the page
- `too_few_tokens`: Too few effective tokens

These rules together with KenLM form a combined filtering strategy of "heuristic + language naturalness."

### 9.4 What the Difference in English and Chinese Retention Rates Reveals

The final results show:

- English candidate set: **846** records, **502** retained
- Chinese candidate set: **201** records, **24** retained

This difference is highly representative.
It does not simply indicate that "Chinese data is inferior," but rather exposes two more realistic problems:

1. Current Chinese quality filtering capability is significantly weaker than English.
2. The structure and noise patterns of Chinese web pages may differ from English web pages, and English rules cannot be directly applied.

This also means that in industrial-scale multilingual data engineering, language quality models must be designed with finer-grained localization.

---

## 10. Three-Round Experimental Review: The Iterative Formation of the Pipeline

![Figure P01-7](../../images/part14/p01/p01_07_three_iterations.svg)
*Figure P01-7: Three-Round Experimental Iteration Path*

If the project is understood only as a series of script calls, the trade-offs behind these design decisions are not easy to discern.
A more authentic representation of the actual engineering process is to restore it as several rounds of progressively tightened experiments.

### 10.1 Experiment One: Body Text Extraction Only

The objective of the first round of experiments was very straightforward:
First verify whether the chain "WARC → HTML → body text" can be completed stably.

This stage resolved:

- Whether WARC can be correctly traversed
- Whether obviously irrelevant responses can be filtered out
- Whether body text can be extracted from web pages

This round typically produces a batch of candidate texts quickly, but the problems are also very apparent:
High noise, many table of contents pages, abundant template content, and serious contamination from code fragments and footer text.

So the first round answered "whether text exists," not "whether these texts can be used for training."

### 10.2 Experiment Two: Adding Heuristic Cleaning and Deduplication

In the second round, the project upgraded from "extracting body text" to "initial corpus formation."

This round added:

- Length filtering
- Symbol density filtering
- Blacklist phrase filtering
- MinHash deduplication

The result was that the coarsest garbage samples and near-duplicate pages were noticeably reduced.
However, during spot-checking, many pages could still be seen that appeared to be body text but had low actual information density.

Therefore, the second round moved data from "readable" to "more corpus-like," but not yet to a level directly suitable for training.

### 10.3 Experiment Three: Adding Language Splitting and Quality Filtering

The third round introduced:

- FastText language splitting
- English KenLM quality scoring
- More stringent filtering logic for directory pages, repeated lines, and short tokens

The direct effect of this round was:
The sample count further decreased significantly, but training usability improved substantially.

The final sample count shrank from **3,028** to **526**, which appears to be a large loss, but this precisely reflects the project's active tightening of quality standards.
It indicates that the project pursues not "retaining more," but "ensuring what is retained is more worth training on."

### 10.4 The Engineering Significance of Three Rounds of Experiments

These three rounds of experiments correspond to a very typical data engineering progression:

1. **First complete chain validation**
2. **Then suppress explicit noise**
3. **Finally achieve language awareness and quality convergence**

---

## 11. Training Data Packaging: From Cleaned Results to Training Interface

### 11.1 Data Cleaning Does Not Equal Training Readiness

Even if the finally retained text is relatively clean, one still cannot directly say "it is ready for training."
Because training systems typically also require:

- Stable train/val splitting
- Metadata indexing
- Token estimation
- Small-scale smoke testing
- File-level organization

If this step is not done well, subsequent training and evaluation can easily produce inconsistencies or data leakage issues.

### 11.2 The Importance of Deterministic Splitting

The project did not use random splitting, but rather performed modulo splitting based on deterministic identifiers such as `text_sha1`.
The advantages of this approach are:

- The train/val sets remain stable across multiple re-runs
- Easier to troubleshoot differences in training results
- Facilitates dataset version management
- Promotes engineering reproducibility

It is important to emphasize:
**Reproducibility is a part of data engineering quality, not an optional add-on.**

### 11.3 The Role of Smoke Tests

The project additionally constructs `smoke_test.jsonl`.
It is not part of the formal training set, but rather an extremely small-scale, quickly loadable sample set used to:

- Validate training scripts
- Check whether the tokenizer and data interface are functioning normally
- Detect format errors, encoding issues, or missing fields early

In actual engineering, this kind of smoke test set can often save substantial debugging time.

### 11.4 The Engineering Value of Manifest

`training_manifest.json` records important metadata about the dataset, such as:

- Sample count
- Split breakdown
- Estimated token count
- File paths
- Overlap check results

Its significance lies in making the dataset not merely a collection of scattered JSONL files, but a formal artifact that can be read by systems, evaluated, and inspected.

---

## 12. Data Evaluation: Pipeline Value Assessment

![Figure P01-8](../../images/part14/p01/p01_08_funnel.svg)
*Figure P01-8: Data Retention Funnel*

### 12.1 Data Retention Funnel

The final retention funnel obtained by this project is as follows:

| Stage | Record Count | Retention Rate (based on extracted) | Typical Interception Reasons |
|---|---:|---:|---|
| Extracted | 3028 | 100.0% | HTML parsing failure, empty content |
| Cleaned | 2425 | 80.08% | Short text, excessive code symbols, blacklist |
| Dedup | 2305 | 76.12% | Mirror sites, template pages, reposts |
| Final | 526 | 17.37% | Directory pages, high perplexity, language mixing |

*Table P01-2: Stage and Typical Interception Reasons Reference Table*

### 12.2 What These Numbers Really Indicate

Looking only at the final result, 526 samples may seem like a small number.
But for data engineering, what matters more is not "how many remain," but rather **what was removed at each layer, why it was removed, and to what extent**.

These numbers indicate at minimum:

1. Raw web page noise is very high.
2. Heuristic cleaning can quickly remove the coarsest noise.
3. Deduplication improves the health of the training distribution.
4. Quality filtering is the critical stage that truly determines final data usability.

From the perspective of engineering interpretability, this conveys more than simply reporting "how many records remain in the end."

### 12.3 Data Profile

The final results also include:

- Final sample count: **526**
- Training set: **468**
- Validation set: **58**
- Train/Val overlap: **0**
- Total estimated tokens: **321,430**
- Average tokens per sample: **611.08**

This indicates that the final dataset is no longer merely a collection of texts, but a standardized corpus artifact with training interface properties and a basic statistical profile.

---

## 13. Cost Analysis: Resource Accounting and Bottlenecks

![Figure P01-9](../../images/part14/p01/p01_09_cost_breakdown.svg)
*Figure P01-9: Resource and Cost Breakdown*

In many introductory projects, developers focus more on "whether the chain can be completed" and less on "what the cost is."
But in real production environments, cost awareness and engineering awareness are bound together.

### 13.1 Storage Costs

Project statistics show:

- Total disk usage approximately **5.31 GB**
- Monthly storage cost estimated at approximately **$0.12 USD**

For a single-shard experiment, this cost is not high.
But it reminds readers: when the process scales to more shards and more intermediate artifacts, storage costs will multiply.

### 13.2 Computational Bottlenecks

The main computational bottlenecks of this project include:

- Download bandwidth
- CPU text processing
- KenLM loading and scoring
- Signature computation in the deduplication stage

In other words, even without introducing a GPU, data engineering is still not "lightweight work."
If the process design is unreasonable, CPU and I/O will quickly become real bottlenecks.


## 14. Validation Loop: Project Consistency Checks

![Figure P01-10](../../images/part14/p01/p01_10_validation_loop.svg)
*Figure P01-10: Project Validation Loop*

### 14.1 The Role of Project Checks

If a data engineering project has only output files and no inspection mechanism, it is actually difficult to say whether it is truly correct.
Because errors can come from many places:

- Scripts run but artifacts are missing
- Train/val split has leakage
- Reports and metrics are inconsistent
- Smoke test samples are not part of the training set
- Final data still contains duplicate samples

Therefore, the project specifically designs inspection scripts for consistency validation.

### 14.2 Inspection Results

The project inspection results are:

- Total check items: **14**
- Passed: **14**
- Overall status: **PASS**

### 14.3 Inspection Coverage

#### Command-Level Checks

- `py_compile`
- `dedup_unit_check`
- `training_smoke_test`
- `dataset_evaluation`

#### Data/Artifact-Level Checks

- Required files exist
- Final file count is consistent with language splitting results
- Training manifest is consistent with training file count
- Train/val has no overlap
- Smoke test belongs to train
- Final dataset has no exact duplicates
- Reports and metrics files are consistent

### 14.4 The Engineering Significance of the Validation Loop

This layer of checks is critically important.
It means the project is not "roughly correct at a glance," but has established a closed loop among code, artifacts, evaluation, and reports.


## 15. Main Limitations and Risks

Any minimally reproducible project is not a final form.
The value of Mini-C4 lies in explaining the methodology, but it also has very clear limitations.

### 15.1 Low Retention Rate

The final retention rate is only **17.37%**.
This indicates that raw web page noise is indeed very heavy and that the current quality gate is relatively strict.

This is not necessarily a bad thing, but it means that if the goal shifts toward "maximizing scale," rules and models must be further optimized to avoid removing too much potentially valuable data.

### 15.2 Low Chinese Retention Rate

Only **24** Chinese records were finally retained, exposing the problem of insufficient Chinese quality scoring capability.
This cannot be completely resolved simply by adjusting thresholds, and more likely requires:

- Data quality rules better adapted to Chinese web pages
- Language models or scoring models better suited to Chinese
- More fine-grained analysis of Chinese web page samples

### 15.3 Limited Deduplication Scalability

Current deduplication still primarily uses in-memory indexing.
When the number of shards increases, the first problems encountered will be:

- Memory pressure
- Increasing runtime
- Difficulty managing global indices

Therefore, the current solution is more suitable for minimal experiments and small-to-medium scale data processing, rather than being directly transposed to ultra-large-scale production environments.

---

## 16. Future Extension Directions

### 16.1 Deduplication Backend Upgrade

Upgrade the current in-memory LSH index to external storage, such as:

- Redis
- Cassandra
- Other distributed KV/index systems

This can support deduplication needs across more shards.

### 16.2 Chinese Quality Model Upgrade

Introduce more stable quality modeling approaches for Chinese web data, such as:

- More suitable Chinese language models
- Chinese web page quality feature engineering
- Lightweight quality classifiers

### 16.3 Pre-Extraction Domain Filtering

Performing domain-level allowlist/blocklist filtering before HTML parsing can significantly reduce subsequent unnecessary computation.
This is a critical step from "text-side cleaning" toward "crawl entry control."

### 16.4 Observability Enhancement

Add for each stage:

- Timing logs
- Throughput statistics
- Sample spot-check dashboards
- Threshold hit statistics

This way, when tuning parameters, developers know not only "results changed" but also "why they changed."

---

## 17. Engineering Practice Summary: The Methodological Value of Mini-C4

![Figure P01-11](../../images/part14/p01/p01_11_methodology_summary.svg)
*Figure P01-11: Mini-C4 Engineering Methodology Summary*

What this project truly aims to convey is not the usage of a particular library, but a more general data engineering methodology:

1. **First complete full-chain validation within controlled boundaries**
2. **Make each step an interpretable stage**
3. **Prioritize establishing a result validation loop**
4. **Observe system behavior through funnels and intermediate metrics**
5. **Ensure the methodology holds before scaling up**

The value of Mini-C4 lies not in the fact that it only processed one shard, but in that it concentrated the most core problems in web pretraining data engineering into a reproducible pipeline.

This pipeline also possesses the key elements required for a complete engineering loop:

- Clear objectives
- Complete process
- Real metrics
- Intermediate trade-offs
- Limitations and extensions
- Engineering closed loop

---

## 18. Main Deliverables Checklist

### 18.1 Intermediate Data Artifacts

- `data/processed/extracted_data.jsonl`
- `data/processed/clean_data.jsonl`
- `data/processed/deduplicated_data.jsonl`
- `data/processed/data_en.jsonl`
- `data/processed/data_zh.jsonl`
- `data/processed/final_data_en.jsonl`
- `data/processed/final_data_zh.jsonl`
- `data/processed/final_data.jsonl`

### 18.2 Training Data Artifacts

- `data/training/serialized_dataset.jsonl`
- `data/training/train.jsonl`
- `data/training/val.jsonl`
- `data/training/smoke_test.jsonl`
- `data/training/training_manifest.json`

### 18.3 Reports and Inspection Artifacts

- `data/reports/p1_metrics.json`
- `data/reports/p1_report.md`
- `data/reports/p1_test_results.json`
- `data/reports/p1_test_report.md`
---

## 19. Conclusion

For large model training, data is often harder to "clean up" than models.
Because model architectures can be reused and training frameworks can be migrated, but high-quality corpus production always depends on a solid set of data engineering capabilities.

The Mini-C4 case demonstrates one thing:
Even under very limited boundaries, a project can still articulate and completely address the key problems in pretraining data engineering, and distill them into reusable methodology.

This is also the core of why such engineering pipelines are reusable.

---

## Special Topic: Acceptance Baselines for the Mini-C4 Pipeline

Mini-C4 projects like this one are easily misread as "doing Common Crawl at a smaller scale." But from an engineering perspective, what is truly worth reusing is that pretraining data processing has been written as a chain of auditable stages. By "auditable" we mean not ending with the production of a `final_data.jsonl`, but rather having at each layer a baseline that can determine "whether to continue moving forward."

### I. Crawling and Parsing Baseline

In the earliest crawling and parsing stage, what matters most is not how many more pages were crawled, but whether the crawled pages can be stably parsed. At minimum, attention should be paid to:

* Whether WARC samples can be correctly decompressed;
* Whether main body content is retained after HTML parsing, rather than ads and navigation noise;
* Whether parsed fields are complete, for example whether URL, language, body text length, and metadata are all present;
* Whether failed-to-parse samples are recorded rather than silently discarded.

The value of this step is to expose "problems at the raw material layer" as early as possible. Because if the raw material layer is already severely distorted, subsequent cleaning, deduplication, and scoring are likely only performing increasingly expensive computations on top of noise.

### II. Cleaning and Deduplication Baseline

The cleaning and deduplication stage most easily leads teams into a trap of "metrics look strong, but unclear what was removed." A more reliable approach is to retain both quantitative metrics and sample spot-checks simultaneously.

Key baselines at this layer include:

* Whether the body text length distribution after cleaning is still reasonable;
* Whether obvious template pages, navigation pages, and script residue pages have decreased significantly;
* Whether sufficient topical diversity is retained after deduplication;
* Whether duplicates across different shards have been effectively handled;
* Whether high-value long texts have not been excessively damaged by rules.

For pretraining corpus, the difficulty of deduplication has never been only "whether there is duplication," but "what remains after deduplication." If what ultimately remains are all structurally similar short pages, then even if the retention rate looks decent, the training value may not necessarily be high.

### III. Language Splitting and Quality Scoring Baseline

Mini-C4 has already separated English and Chinese for processing, which is critically important, because different languages differ greatly in web structure, noise types, and quality signals. After language splitting, quality scoring should no longer look at a unified threshold, but should judge based on language characteristics.

At this layer, important baselines include:

* Whether language identification is stable, avoiding Chinese-English mixed pages from being incorrectly classified;
* Whether the retention rate for each language is consistent with intuitions about sample quality;
* Whether topical and length distributions of retained corpus fluctuate dramatically when quality thresholds change;
* Whether there is still obvious clustering of low-value sites in the finally retained corpus.

These baselines together determine one thing: whether what ultimately remains is "cleaner," or merely "fewer." These two are not the same thing in engineering.

---

## Special Topic: From Teaching Prototype to Large-Scale Pretraining Factory

The current form of P01 is better suited as a teaching-type minimal closed loop, but it has already clearly shown several key paths toward a large-scale factory. What is most worth emphasizing here is that scaling up cannot be understood only as "running scripts on more machines," but requires simultaneously expanding the control plane, observability, and error handling capabilities.

### I. Expand the Control Plane First, Then Expand Data Volume

Many teams want to expand data volume right away, but if the control plane is too weak, the larger the scale, the harder it is to locate problems. A more reasonable sequence is usually:

* First fill in stage-level logs and statistics;
* Then fill in sample spot-checks and rule hit distributions;
* Then expand shard count and parallelism;
* Finally pursue higher throughput and greater coverage.

Because only when the control plane is strong enough can the team still know where problems occur, why they occur, and which segment to fix first when scale increases.

### II. Pretraining Corpus Factories Need "Entry Governance"

P01 has already mentioned pre-extraction domain filtering, which is actually very important. Because much of the cost of pretraining data is not spent on high-value content, but on downloading, parsing, cleaning, and deduplicating vast numbers of low-value pages. In the future, when moving toward a more authentic factory form, entry governance will become increasingly important, including:

* Domain allowlists and blocklists;
* Site quality profiles;
* Update frequency and crawl priority;
* Differentiated strategies for sites in different languages and different regions.

As long as entry governance is done well enough, the subsequent cleaning pressure and computational costs will decrease significantly.

### III. Pretraining Projects Ultimately Compete on Continuous Production Capability

In the long term, what pretraining corpus engineering truly competes on is not how cleanly data was processed in one particular run, but whether it can continuously, stably, and auditably produce the next version of corpus. To achieve this, at minimum requires:

* Clear versioning;
* Stage-level baselines;
* Retention of anomalous samples;
* Explanations for quality changes;
* A stable interface consumable by the training side.

This is also where Mini-C4 as a project prototype has the most value. It does not pretend to already be a complete industrial system, but it has constructed the most critical skeleton of an industrial system first. Subsequently, regardless of how much scale the team wants to expand to, how many languages are added, or how many new rules are introduced, as long as this skeleton remains, the methodology has room to continue growing.

---

## Special Topic: Pre-Emptive Thinking on Corpus Mixing Ratios and Training Blend Strategies

Although this chapter of Mini-C4 focuses on data cleaning and quality control, from the complete perspective of pretraining engineering, corpus mixing ratios are also a problem worth thinking about proactively. Because "cleaned up" only resolves the question of whether data can be used, while "how to blend" determines what kind of distributional impact these corpora will have after entering training.

### I. Data Preparation Phase Should Already Retain Information Needed for Blending

If the team plans to subsequently blend data by language, source, length, or quality tiers, then the relevant fields should be retained during the data preparation phase, rather than being guessed at right before training. The most common retainable information includes:

* Language labels;
* Source domain or source type;
* Text length range;
* Quality scores or quality buckets;
* Pre/post-deduplication status.

These fields look like "deal with later" supplementary information in the current minimal project, but once entering the training mix phase, they immediately become the most valuable control handles.

### II. Training Blend Strategy Is Essentially a Continuation of Quality Control

Many people view cleaning as data engineering and blending as training engineering, but in pretraining projects, these two are actually a continuous chain. Because if high-quality long texts are retained after cleaning but are excessively diluted during training blending, the cleaning gains upstream are difficult to truly transmit to the model. Conversely, if certain low-quality but high-frequency web pages are extensively retained and have too high a proportion in training, the model will still be noticeably disturbed.

From this perspective, the structured fields and intermediate artifacts currently retained by Mini-C4 are not only serving the cleaning process, but also pre-reserving interfaces for subsequent more refined training blend strategies.

## Chapter Summary

This chapter used the "Mini-C4 distributed web corpus pipeline" as a case study to demonstrate the engineering organization of constructing auditable pretraining corpus samples from a Common Crawl shard. The main value of the case lies in placing task definition, data boundaries, architectural decisions, sample schema, metrics acceptance, and reproducibility resources in the same chain, making the project no longer merely a sequence of operational steps but an auditable case study.

The boundaries of this case must also be clearly preserved. Single shard, single-machine CPU, and Ray Data environment, with emphasis on validating web body text extraction, cleaning and deduplication, language splitting, and training packaging. In scenarios of larger scale, higher risk, or stronger compliance constraints, data sources, permission status, manual review proportions, operational costs, and failure rollback plans should be re-evaluated.

As part of Chapter 14, this chapter corresponds to the project-level empirical validation of the methods presented in earlier chapters. Readers can combine this case with the data recipes from Chapter 13, the platform governance chapters in preceding sections, and the checklists in the appendices to form a closed loop from methodological understanding to engineering delivery.

## References

1. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR, 21(140), 1-67.
2. Hugging Face. (2026). Datasets Documentation. https://huggingface.co/docs/datasets/.
3. Ray Project. (2026). Ray Data Documentation. https://docs.ray.io/en/latest/data/data.html.
4. MLflow Authors. (2026). MLflow Documentation. https://mlflow.org/docs/latest/.
5. Great Expectations Contributors. (2026). Great Expectations Documentation. https://docs.greatexpectations.io/.
