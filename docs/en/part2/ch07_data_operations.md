# Chapter 7: Data Evaluation, Quality Feedback Loops, and Operational Iteration

## Abstract

This chapter addresses the challenges of continuous evaluation, version governance, and operational iteration for pre-training data after the cleaning phase is complete. The chapter opens with an anonymized composite case study illustrating that "cleaner" data does not necessarily yield better model performance, then establishes the foundational framework for Data Operations (DataOps): offline proxy metrics, representative sampling, quality dashboards, issue sample pools, version comparison, and upstream strategy write-back. The metrics section focuses on explaining the applicable boundaries of perplexity, type-token ratio, toxicity and PII density, benchmark contamination, and domain coverage, emphasizing that these are proxy signals only and must be combined with small-scale validator models and manual spot-checks. The latter half of the chapter covers 5 Whys root-cause retrospectives, weekly operational cadences, dashboard alerting, and pathways for reusing data assets in SFT and RAG pipelines. Readers should be able to extend data processing from a one-time delivery into a traceable, rollback-capable, auditable continuous operations system.

## Keywords

Data operations; proxy evaluation; quality feedback loop; DVC; issue sample pool; A/B testing; data drift; dashboard

## Learning Objectives

- Explain why pre-training data requires continuous evaluation and versioned operations.
- Design offline proxy metrics including PPL, TTR, PII density, benchmark contamination, and domain coverage.
- Use issue sample pools, DVC versioning, and A/B testing to identify the model impact of data changes.
- Establish data quality dashboards, automated alerts, and a weekly operational cadence.
- Write evaluation findings back into collection, cleaning, SFT, and RAG data assets.

## Opening: An Unexpected "Performance Regression"

The following is an anonymized composite case study; metrics, timelines, and data scales are used solely to illustrate the retrospective methodology. As of June 2026, evaluation fluctuations in similar projects are influenced by model scale, corpus mixing ratios, training steps, and benchmark selection. In one 7B language model development project, the data team spent two months cleaning the pre-training corpus to an extremely rigorous standard. They used heuristic rules to exclude large volumes of short text, applied perplexity scoring to remove "non-standard language," and performed deduplication with a low MinHash threshold. The team believed this represented a cleaner, more controllable dataset.

However, the model trained on the new dataset (codenamed v2.0) underperformed across multiple benchmark evaluations compared to the version trained one month earlier on coarser data (v1.0). After a thorough investigation, the team uncovered several important findings:

1. Because all text containing "large numbers of newlines and symbols" had been removed, the model almost entirely lost its ability to generate code and render Markdown.
2. Because "non-standard colloquial expressions" had been removed, the model lost its capacity for empathetic responses in conversational tasks, becoming as cold and rigid as an encyclopedia.
3. Aggressive deduplication caused certain extremely high-frequency facts (such as common-sense geography and basic history) to appear too infrequently in the training set, causing the model to suffer severe "knowledge forgetting."

This retrospective delivered one critical conclusion to the team: **high quality is not a static standard—unilateral "cleanliness" divorced from model performance is insufficient.**

This chapter shifts perspective from concrete data processing code to the systems engineering level, exploring **data evaluation and operational iteration** in large language model projects. We will dismantle the traditional misconception that "handing over data means the job is done," and establish a data governance cycle driven by offline evaluation and proxy metrics, transforming data into a continuously evolving asset that grows alongside model capability.

---

## 7.1 Why Data Also Requires Continuous Operations

### 7.1.1 Breaking the Engineering Illusion of "One-Time Delivery"

In the era of traditional deep learning (e.g., image classification or sequence labeling), datasets were typically treated as static assets: collect, annotate, publish, then freeze indefinitely. This paradigm conditioned engineers to think in terms of one-time delivery.

In the pre-training of large language models (LLMs), however, the boundary between data and model becomes blurred. Different stages of model development demand fundamentally different data recipes. For example, during the cold-start phase (0–100B tokens), the model requires large-scale breadth of information to learn general grammar and foundational world knowledge; whereas during the late convergence or cooldown phase, the model requires highly dense, high-quality knowledge material (mathematical and scientific reasoning, code structure) to raise its capability ceiling. **A single static dataset used from start to finish is unlikely to support high-level model training.**

This transforms the role of data engineers from one-time data deliverers into operators who continuously adjust data recipes and quality boundaries. Accordingly, teams must establish a complete Data Operations (DataOps) system.

### 7.1.2 Why Evaluating After Training Is Already Too Late

The traditional pipeline is often sequential: the data team spends a month cleaning data → the training team spends a month running the pre-training run → the evaluation team runs benchmarks to verify performance. If final results fall short of expectations, tracing the root cause becomes difficult: was the data itself poor? Was the sampling ratio imbalanced? Or did the learning rate or optimizer hyperparameters diverge?

Because a single long pre-training run for an LLM is extraordinarily expensive, the margin for error is limited. Without front-loading the evaluation system—that is, evaluating data quality with proxy metrics and small-scale validator models before formal training begins—project teams typically discover data problems only after completing a costly training run. This necessitates the introduction of **offline proxy evaluation** and **real-time feedback operations**.

### 7.1.3 Boundary Coordination Between Data Operations and Model Operations

In a modern large-model development organization, the typical collaboration boundaries and organizational interfaces are as follows:

- **Model engineers**: responsible for monitoring compute cluster health, handling gradient anomalies (e.g., gradient norm spikes), and designing architecture tuning and cooldown strategies.
- **Data engineers**: focused on pipeline throughput, token cost, and pipeline error handling.
- **Data operations lead / evaluation lead**: this role connects both sides—identifying specific problematic data batches from transient model behavior (such as training loss spikes or sharp drops in specific model capabilities) and guiding upstream teams to cut off or update cleaning rules in a timely manner.

This cross-functional coordination is realized through the "operations flywheel."

![Figure 7-1: Data Operations Flywheel](../../images/part2/data_operations_flywheel.png)

*Figure 7-1: Data Operations Flywheel — The left side shows the high-cost startup zone; the right side shows the gradually accumulated cycle of automated, high-quality data assets formed after long-term model evaluation and root-cause analysis feedback. Source: Original illustration by the authors. Alt text: Data operations flywheel diagram showing the cyclical relationship among data production, model evaluation, root-cause analysis, rule write-back, and asset reuse.*

---

## 7.2 Offline Evaluation and Proxy Metric Design

The key mechanism for resolving evaluation latency is to design proxy metrics that can operate independently of the time-consuming main training run. This involves a systematic set of inspection actions that ensure every version of sampled data undergoes quantifiable quality checks before it is handed to the GPU.

### 7.2.1 Statistical Stratification and Representative Evaluation

The first question to address is "what to evaluate." At the scale of trillions of tokens, exhaustive full-corpus statistics are not only computationally expensive but typically unnecessary. The most fundamental technique is **stratified sampling**.

In practice, documents are stratified by category (news, Wikipedia, domain-specific forums, code repositories), and a fixed-size data sandbox (e.g., a subset of 100 million tokens) is randomly sampled at 0.1% or 0.01% before data serialization. All offline analyses are performed on this subset; if the subset's distribution exhibits a pronounced anomaly, it signals that the entire batch carries elevated risk.

Evaluating a corpus typically encompasses the following four core dimensions (proxy metrics):

### 7.2.2 Core Offline Proxy Metrics Explained (with Computation Logic)

Throughout the long engineering lifecycle of pre-training, discussing "quality" in the abstract is insufficient. Engineers must map the intuitive notion of "good vs. bad" onto objective numbers that code can compute and thresholds can enforce. The following four core proxy metrics form the backbone of a quality health report:

#### 1. Linguistic Properties and Perplexity (PPL)
- **Detection objective**: Basic fluency, knowledge density, and linguistic orthogonality of the corpus.
- **Validation method**: Use a mature but lightweight reference model (e.g., LLaMA-7B base, or a 1B validation model trained on clean data in early stages) to perform gradient-free forward passes on sampled batches.
- **Mathematical essence**: Perplexity is fundamentally the exponential form of the cross-entropy loss ($PPL = e^{Loss}$). If the model finds a sentence "very common and expected," the PPL will be low; if it finds it "confusing or garbled," PPL will spike.
- **Interpretation logic**: This is not a one-directional "lower is better" metric.
  - **Very low (PPL < 5)**: Typically indicates boilerplate code, infinitely copied disclaimers, or SEO content left behind by over-aggressive deduplication. The model learns little new information from such data.
  - **Very high (PPL > 500)**: Typically indicates format corruption, misaligned machine-translated text, or disordered character sequences.
  - **Preferred range**: Using a neural network reference model (e.g., LLaMA-7B) as the baseline, high-quality text typically falls in the **PPL 20–150** range (news tends toward the narrower end; scientific literature is somewhat broader). Note: if an n-gram language model (such as KenLM, described in Section 5.2.1 of Chapter 5) is used as the reference, PPL values for the same text will be significantly larger (approximately 100–300 for news/encyclopedia text). The two reference types use different scales and their thresholds must not be conflated.

Listing 7-1 shows a reference implementation for offline perplexity sampling computation.

**Listing 7-1: Reference Code for Offline Perplexity Sampling Computation**

```python
# Pseudocode for a typical offline perplexity sampling computation (based on PyTorch and HuggingFace API)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_perplexity_batch(texts, cache_model_path="llama-1b-ref"):
    tokenizer = AutoTokenizer.from_pretrained(cache_model_path)
    model = AutoModelForCausalLM.from_pretrained(cache_model_path).cuda()
    model.eval()

    ppl_results = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            # Filter out texts that are too short to avoid PPL computation instability
            if inputs.input_ids.shape[1] < 50:
                continue
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            ppl = torch.exp(loss)
            ppl_results.append(ppl.item())

    return ppl_results  # Returns an array for downstream histogram generation
```

#### 2. Diversity Sparsity (Type-Token Ratio, TTR & Vocabulary Coverage)
- **Detection objective**: Confirm whether the cleaning pipeline, due to overly aggressive threshold settings or excessively strict deduplication (MinHash), has permanently eliminated niche knowledge or specific long-tail vocabulary.
- **Validation method**: Compute the ratio of unique word types (distinct word stems within the vocabulary) to the total token count in the document collection described above. TTR tends to be lower across long passages, so a windowed averaging algorithm must be applied (e.g., MATTR (Covington and McFall 2010)).
- **Interpretation logic**: If the total number of non-repeated words computed from a 100-million-token document sandbox is fewer than 50,000, the dataset suffers from severe vocabulary diversity deficiency (likely stemming from large volumes of e-commerce filler content or machine-translation cycles). Over time, this tends to produce a mechanical, low-information response style in the trained model.

Listing 7-2 shows a reference implementation for offline Type-Token Ratio computation.

**Listing 7-2: Reference Code for Offline Type-Token Ratio Computation**

```python
# Pseudocode for a typical offline TTR (Type-Token Ratio) computation
def calculate_ttr(texts, tokenizer=None):
    if tokenizer is None:
        # Fall back to simple whitespace-based tokenization
        tokens = " ".join(texts).split()
    else:
        # Use a real LLM tokenizer
        tokens = tokenizer.tokenize(" ".join(texts))

    total_tokens = len(tokens)
    unique_types = len(set(tokens))

    if total_tokens == 0:
        return 0.0
    return unique_types / total_tokens
```

- **Advanced validation — Vocabulary Coverage**: Teams should compile a domain-specific vocabulary list (e.g., rare disease names, recently introduced niche code frameworks, or the complete roster of characters from a specific literary work). If the coverage of such targeted vocabulary in the sandbox falls below 5%, whitelist weights should immediately be added to the upstream crawlers for the corresponding domains.

#### 3. Toxicity and Adverse Leakage Rate (Toxicity & PII Density)
- **Detection objective**: Directly relevant to the risk-control compliance baseline for commercial deployment. Checks the stability of residual rates for harmful content (hate speech, terrorism, abuse, soft pornography) and PII (personal identity, phone numbers, password tokens) after cleaning.
- **Validation method**: This is the most computationally intensive part of the entire metric chain. It typically requires invoking a lightweight discriminator specifically fine-tuned for safety (such as an open-source offline variant of the Perspective API (Lees et al. 2022), or a 5-class discriminator trained on RoBERTa) to score sampled articles.
- **Interpretation logic**:
  - Toxicity scores should not be evaluated solely by their mean; the **P99 or P99.9 percentile** must also be monitored.
  - If the P99 percentile score in a sample exceeds the threshold (> 0.8), isolation and re-review must be triggered immediately.
  - Additionally, the trigger rate for text containing patterns such as `sk-****` (API tokens) or `13[0-9]*` (mobile number signatures) should be monitored via regular expressions to confirm that the PII masking layer has not accidentally thrown errors during updates.

#### 4. Domain Classification and Subpopulation Overlap
- **Detection objective**: This is one of the most important high-risk metrics in data operations in recent years, also known as Benchmark Contamination Prevention. Teams must confirm that randomly crawled data does not contain the evaluation questions from official benchmarks (e.g., standard answers from GSM8K (Cobbe et al. 2021) or the original English text from MMLU (Hendrycks et al. 2021)). Contamination undermines evaluation reliability and research integrity.
- **Validation method**: Hash all test set data from major benchmarks using N-gram (typically 13-gram or 15-gram) fingerprinting; then compute the intersection against sampled data pending ingestion.
- **Interpretation logic**: The overlap ratio should be as close to zero as possible. If a batch of Wikipedia expansion packages triggers a 1% 13-gram overlap, this typically indicates that some open-source library or an individual's GitHub repository has been included in the current pipeline, necessitating targeted removal.

### 7.2.3 Alignment Bias Between Proxy Metrics and True Model Performance

It is important to be vigilant: all offline metrics are merely "proxies" and do not have a 100% correlation with final generation quality. For example, large volumes of machine-translated text may look excellent on PPL and TTR, but because the corpus is persistently contaminated by translationese, it can cause the trained model to frequently hallucinate on culturally specific common knowledge. Therefore, for data of heterogeneous quality, the ultimate reliance must be on continuously building small-scale "validator models."

### 7.2.4 Mapping Evaluation Metrics to Governance Actions

Evaluation must never stop at merely "looking at metrics." A qualified evaluation report must culminate in concrete system governance actions. See the table below for reference:

**Table 7-1: Evaluation Metric to Governance Action Mapping**

| Metric Observation (Offline/Online) | Common Root Cause and Manifestation | Corresponding Governance Action |
| :--- | :--- | :--- |
| **Overall decline in sampled TTR (diversity)** | MinHash deduplication is overly aggressive, eliminating reasonable overlap in general domains | **Loosen the MinHash Jaccard threshold (e.g., 0.8→0.7); introduce domain-specific vocabulary protection** |
| **Heavy tail expansion in the high-PPL segment** | Incomplete HTML tag cleaning; newly introduced data contains large volumes of garbled symbols | **Trace back high-PPL samples; add HTML element regex filters or strengthen language identification confidence thresholds** |
| **Steep decline in code capability benchmarks** | Indentation and newlines were mechanically stripped during normalization | **Disable the global newline-merging rule; implement an independent parsing bypass for code domains** |
| **Frequent repetition loops during pre-training** | The same page from a specific site was repeatedly packaged across different time-point snapshots | **Run full sequence-level SHA256 deduplication; block the source or configure strict penalty weights** |
| **Large local oscillations in the loss function** | Severely malformed data with broken punctuation has been mixed in | **Retrieve the current shard by Batch ID; downweight anomalous data to a blacklist or filter it out** |
| **Sharp rise in Toxicity** | Forum crawler sources (e.g., Reddit) have crawled illicit or sensitive subreddits | **Update the safety filtering model; expand the stopword list or NSFW discrimination feature set; perform historical retroactive cleanup** |

This tightly coupled governance mapping enables teams to immediately translate raw data metrics into a roadmap for the next engineering optimization cycle.

---

## 7.3 Version Iteration, Spot-Checking, and Root-Cause Retrospectives

For the evaluation described above to be effective, the organization must establish an iteration mechanism and attribution tracking system based on version comparison (A/B testing). Data development for large language models is fundamentally a process of "aligning anticipated parameters with actual outcomes," and failures along the way are entirely normal—the key is to distill lessons into reusable rules.

### 7.3.1 Dataset Versioning and A/B Comparison from a DVC Perspective

Analogous to code versioned with Git, for data lakes spanning multiple terabytes we must introduce DVC (Kuprieiev et al. 2020) (Data Version Control) or a similar immutable object management strategy based on SHA-backed mounts. In large-scale experiments, raw data must never be overwritten in place; any modification at a processing node should produce a completely new incremental version or a snapshot partitioned via Delta Lake.

**A/B testing principles**: Each time a new pipeline adjustment is made (for example, newly incorporating 20 GB of data parsed from high-quality Reddit nodes with enhanced comment-tree filtering logic for that site), it should be validated before full rollout by allocating an equivalent compute budget (e.g., launching a set of parallel comparative 1B small-model training runs at a scale of 1B tokens). Only after both experimental models have completed evaluation on the core benchmark set and demonstrated that "mathematical or conversational empathy capabilities have improved significantly without meaningfully degrading general world knowledge metrics" should this strategy be deployed into the production version (e.g., upgrading from v2.1 to v2.2).

### 7.3.2 Building an Issue Sample Pool and Establishing a Traceability Loop

The most valuable asset in any retrospective is the **issue sample pool**. These error labels span every stage of the pipeline: source URL parsing errors, classifier misses, cleaning false-deletions, and benchmark contamination. Teams should actively encourage the extraction of problematic data discovered during training or manual annotation.

**Traceability chain setup**: The value of a single erroneous sample lies in its ability to expose failure points throughout the entire pipeline in reverse.

- Which crawler source did this text containing an adversarial keyword come from? (Source tracing)
- Why was it not filtered out? (Regex failure or model miss)
- Which code change introduced it into the repository? (Accountability attribution)

### 7.3.3 Distilling Lessons from Failed Experiments with the 5 Whys Root-Cause Framework

Rather than spending energy in mutual blame, teams should apply a systematic retrospective model to distill lessons. The industry-standard "5 Whys data attribution method" converts engineering incidents into reusable governance rules through successive layers of questioning:

**5 Whys Anomaly Investigation with Specific Action Steps:**

- **Why 1 (Symptom layer)**: Why does this version of the model consistently produce large amounts of meaningless whitespace when generating code?<br>
  *Action: Immediately pull online error logs; intercept and trace back the 100 most recent code corpus samples responsible for this behavior.*
- **Why 2 (Mechanism layer)**: Why is the indentation in these 100 code samples uniformly corrupted into disordered whitespace blocks?<br>
  *Action: Inspect the data parsing pipeline; discover that during `Markdown_to_Text` conversion, `\n` newlines and `\t` tabs were forcibly merged and deleted.*
- **Why 3 (Policy layer)**: Why was newline merging enforced at the conversion stage?<br>
  *Action: Review the Git commit history (blame); discover that last week, a global carriage-return removal cleaning rule was committed "to reduce PPL in novel-genre data."*
- **Why 4 (Architecture layer)**: Why did a cleaning rule targeting novels contaminate the code-domain corpus?<br>
  *Action: Confirm that the data queue architecture is flawed—different domains share the same cleaning pipeline without logical or physical isolation.*
- **Why 5 (Root cause / organizational layer)**: Why was this high-risk global rule not intercepted before merging to the main branch?<br>
  *Governance action: Due to the absence of independent automated canary review gating, immediately enforce a "multi-domain regression sandbox test" in the CI/CD pipeline to ensure that any single-domain policy change cannot cause cross-domain contamination.*

### 7.3.4 Case Retrospective: Root-Cause Localization of a Training Loss Spike

The following is an anonymized composite case study; token counts, loss values, data scales, and costs are illustrative parameters. A large-scale training job triggered a monitoring alert near the 87-billion-token mark: training loss, which had been smoothly declining to approximately 1.8, rose to 14.5 within 15 steps, and the gradient norm at that node became NaN. The model engineers paused training and rolled back to a checkpoint 100 steps earlier, then transferred the investigation to the data asset operations team.

This is a classic instance of training instability caused by anomalous data. After taking over, the data retrospective team initiated the standard root-cause localization procedure:

**Action 1: Batch Trapping**
Since the training set had been shuffled, directly locating the problematic samples in the source files was inefficient. The team retrieved from the logging system the token ID sequence most recently loaded to the GPU before the incident—specifically, Batch 86,995.

**Action 2: Reverse Detokenization from Token IDs**
Engineers decoded the token array that caused the system crash using the TikToken vocabulary. The result showed: across a sequence of 4,096 tokens, there were almost no punctuation marks or common words—the sequence was filled with truncated garbled bytes such as `\uA4\uB6\uFF\uC2` and meaningless Unicode replacement characters (placeholders). This high-entropy sequence exceeded the stable processing range of the model's attention mechanism, triggering numerical overflow in the forward computation.

**Action 3: Batch-to-Source Tracing**
Using the globally unique identifier (GUID), the data team queried the document's lineage. The result pointed to a large-scale pull of a public thesis PDF repository conducted three weeks earlier.

**Action 4: Locating the Cleaner Defect**
Investigation of the cleaning logs from that time revealed: these PDFs had been encrypted by early third-party software, and their underlying text layer was actually a byte stream that had been obfuscated. When processed by the conventional language identification model (FastText Language ID), this garbled content was misclassified as a low-resource language (confidence 0.92), thereby evading the "excessive proportion of non-standard characters" filtering funnel; subsequently, because this sequence was extremely rare, it was retained as "unique content" during the deduplication stage and ultimately entered the high-weight data queue.

**Root-Cause Action**
After establishing the facts, the team immediately executed a three-step response:

1. **Isolate the problematic source**: All PDF-parsed data originating from this source was isolated from the data lake for review—totaling 1.4 TB (approximately 350 million tokens, as an example figure).
2. **Rule patching**: Added a "valid UTF-8 / target-language character proportion detector" as a pre-filter before the FastText language model, and set a minimum punctuation density threshold for major natural languages.
3. **Safety re-review**: Applied an additional filtering gate—using a small LLaMA model to determine whether text meets basic grammatical structure—to the heterogeneous text ingested in the same batch.

This costly training interruption demonstrates that the data operations team must be able to trace from an anomalous batch back to the source document, parser version, and cleaning rule, and consolidate retrospective findings into automated checks.

**Table 7-2: Version Iteration Log Template**

In a formal business iteration system, every data batch deployed to the main training cluster must be accompanied by a release log as rigorous as a software release note. The table below provides a benchmark log template from a production pipeline.

| Evaluation Dimension | Version Log Field Example |
| :--- | :--- |
| **Basic information** | Version: v2.1 → v2.2; Operator: Zhang San (DataOps); Submission date: 2026-X-X |
| **Main changes (Changelog)** | 1. Added 30 GB of medium-to-high quality StackOverflow Q&A (via new crawler integration).<br>2. Tightened MinHash threshold (0.85→0.8) for near-duplicate removal within the Chinese Wikipedia corpus.<br>3. Fixed a regex vulnerability that incorrectly stripped preceding paragraphs on `<p>` tags. |
| **Scale change** | Expected net addition: 50 GB; actual net addition after cleaning and deduplication: 23 GB; total token count: 1.45T. |
| **A/B evaluation highlights** | In the small-scale comparative experiment, HumanEval (code evaluation) pass rate improved by 4.1 points; all other general benchmarks fluctuated by no more than 0.3%, classified as a low-risk change. |
| **Known issues and anticipated risks** | After adding StackOverflow, some very outdated answers were incorporated. Time-based heuristic filtering of old posts has not yet been applied; planned for v2.3. |
| **Final review conclusion** | √ Validation passed; approved for mounting into the v2.2 production main queue for pre-training consumption. |

---

## 7.4 Data Operations Dashboard and Organizational Coordination

Facing a dynamically complex experimental pipeline, Excel spreadsheets alone are insufficient. A unified "Data Operations Dashboard" with a consolidated visual perspective has become the scheduling core linking training and engineering.

### 7.4.1 Core Modules of the Data Quality Dashboard

An excellent quality dashboard should provide a top-down view of all metrics. It primarily includes:

1. **Overall status overview**: Ingestion volume by domain source, current inventory balance, and percentage of tokens consumed.
2. **Cleaning funnel yield rate**: Stage-by-stage retention metrics, such as the proportion blocked by language identification, the reject rate from heuristic filtering, and the removal rate from fuzzy deduplication. Any sudden drop or spike in any stage should be flagged with a red alert.
3. **Safety risk baseline monitoring**: Records the number of PII or highly sensitive harmful documents detected and the blocking logs for each cycle.
4. **Spot-check audit traffic lights**: Displays scoring trends from the weekly blind review of 1,000 randomly sampled corpus items, showing a moving average of fluency and correctness on a scale of 1 to 5.

![Figure 7-2: Data Evaluation Feedback Loop](../../images/part2/data_evaluation_loop.png)

*Figure 7-2: Data Evaluation Feedback Loop — A circular architecture proceeding from sampling-based blind review to root-cause investigation triggered by metric anomalies, followed by targeted system governance actions. Source: Original illustration by the authors. Alt text: Data evaluation feedback loop diagram showing the closed-loop relationship among sampling evaluation, metric anomalies, root-cause investigation, governance actions, and rule updates.*

### 7.4.2 Automated Quality Alert System Architecture

If the dashboard is merely a static report requiring daily manual inspection, oversight gaps are inevitable. A mature large-model data factory requires not only a static dashboard but also an active blocking and alerting mechanism. This alerting architecture is typically built on a distributed stream-processing framework (such as Apache Flink or Spark Streaming) to achieve low-latency interception of anomalous data.

**Tier 1: Data Drift Alerts**
New web crawl data, cleaned data, and even synthetic data continuously enter the data lake each day. The system extracts sample sets daily to compute distributional entropy. If the frequency of a specific word type rises by 300% in the current day's batch (possibly because a domain crawler has entered a loop, repeatedly downloading redundant navigation bar tags), a `[P1-DataDrift]` alert is triggered in Slack or Feishu. The data stream is automatically suspended until an engineer manually logs into the dashboard to lift the hold. The 300% threshold is an example; production systems should be calibrated against historical distributions and business tolerance levels.

**Tier 2: Cost and Latency Threshold Alerts**
Data preprocessing also consumes large amounts of CPU resources. If the dashboard shows that the `FastText` or `regex filtering` node has maxed out its CPU cores and throughput has plummeted from the usual 2 GB/s to 100 MB/s, such an alert almost certainly indicates that a regex pattern with catastrophic backtracking is experiencing severe time-complexity explosion when processing an extremely long document—the classic ReDoS vulnerability. When this occurs, the scheduling system can directly terminate the timed-out process to protect the main pipeline from being blocked.

**Tier 3: Multimodal Anomaly Alerts (for Next-Generation Systems)**
With the introduction of image-text multimodal data (see Chapter 8), the dashboard must also add monitoring for image broken-link rates and CLIP similarity extremes between text and images. If the proportion of semantically unrelated image-text pairs in a batch exceeds the threshold, a re-review should be triggered immediately.

### 7.4.3 Tiered Responsibility Boundary Framework

Under an automated monitoring and review mechanism, the responsibilities of different teams in a large AI development organization can be more clearly quantified, reducing ambiguity during incident diagnosis.

- The **Data Infrastructure team**'s north star metrics are stable throughput and cost per token (Cost_per_Token). They are responsible for providing downstream teams with high-speed, low-storage-I/O distributed storage architecture; if GPU utilization drops due to a `DataLoader` bottleneck, this team is responsible for diagnosis and resolution.
- The **Data Ingestion (crawling and collection) team** focuses on acquisition breadth, coverage, and legal/copyright compliance. If PII or high-risk harmful samples escape into the training corpus, they bear responsibility for upstream root-cause investigation.
- **Pre-training researchers**' primary focus is on which architecture (MoE vs. dense, MHA vs. GQA) and well-configured hyperparameters to use in order to fully utilize the hardware cluster, and whether the current data mixing ratio allows loss to decline strictly in accordance with scaling laws.
- The **Data Quality Evaluation (DataOps/Evaluator) team**: maintains the quality dashboard and alerting pipeline, determines data mixing ratios (e.g., web pages : code : papers = 6:2:2), and uses the dashboard to assess whether ingested data meets quality standards. They must also combine model evaluation results to determine whether the data recipe is effective; if evaluation performance drops significantly in any phase, they should drive training suspension, sample tracing, and rule updates.

---

## 7.5 Weekly Operational Cadence: A Practical Template

If the evolution of a large model's data infrastructure relies solely on post-hoc retrospectives, it will inevitably lag behind. Evaluation, feedback, and action must be decomposed into a highly frequent "weekly sprint" cadence. Below is a standard weekly agile template for a data operations team during a trillion-scale model sprint.

### 7.5.1 Monday: Dashboard Review and Metrics Review Meeting

**Core participants**: Data engineers, data product managers, evaluation leads.
**Key workflow**:

1. **Weekend pre-training inspection**: Review the total token count continuously fed into the main pre-training branch over the past weekend. Cross-check the `nvidia-smi` monitoring panel for GPU idle time caused by `DataLoader` stalls or storage I/O blockages (MFU below the warning threshold). If found, immediately log the I/O deficiency as the first priority item for the day.
2. **Offline detection report review**: For the T-1 batch data most recently cleaned as of Sunday evening (typically two or three 10 GB test sandboxes after sampling), extract KenLM (Heafield 2011) perplexity (PPL), type-token ratio (TTR), and text length distribution histograms.
3. **Metric anomaly alert investigation**: If the PPL mean suddenly rises above 500, this typically indicates that the most recently onboarded data source contains HTML residue that was not fully parsed. If the safety block rate (Toxicity Alert) doubles, it may be related to recently added community discussion sources. No conclusions are rushed in the meeting—only the anomalies requiring deep-dive investigation are identified.

### 7.5.2 Tuesday and Wednesday: Anomaly Tracing and Small-Scale Validation (Root Cause Analysis)

**Core participants**: Data operations lead, pre-training data engineers.
**Key workflow**:

1. **Targeted blind review and labeling**: For the quality degradation points identified in Monday's meeting, extract approximately 200 raw corpus samples. The operations evaluation team manually reads through the samples to determine whether rules are causing "false positives" (deleting good articles) or "false negatives" (junk vocabulary bypassing the regex defenses).
2. **Cleaning strategy correction**: If the issue is identified as "special indentation in certain code domains causing line-filtering logic errors," engineers will correct the `FastText` or regex script logic on Tuesday afternoon and rerun the pipeline on the affected corpus module.
3. **Mini-experiment scheduling**: Push the newly cleaned sandbox data to a 0.5B or 1B small model and launch a 12- to 24-hour controlled comparative experiment. This is where DVC (data version control) demonstrates its power: strictly controlling variables and comparing scores between only two groups, such as `v1.2_Base` vs. `v1.2_CodePatch`.

### 7.5.3 Thursday: Experimental Decision and Data Mixing

**Core participants**: Pre-training model engineers, data engineers, lead architects.
**Key workflow**:

1. **A/B outcome comparison**: On Thursday morning, the mini-experiment results are available. Model engineers report whether the validation loss curves of the two data versions intersect, and the pass rate difference on specific downstream evaluation benchmarks (e.g., the MMLU (Hendrycks et al. 2021) code subcategory or GSM8K (Cobbe et al. 2021)).
2. **Qualitative analysis**: If the new data (`v1.2_CodePatch`) improves code capability by 5% without meaningfully degrading general instruction-following ability, the cleaning patch is considered validated and the code is merged into the main pre-training cleaning repository.
3. **Data subset ratio rebalancing**: At this step, the team decides the data mix to be pushed to the large cluster in the following week. For example, if recent evaluations show that foundational reasoning capability is weak, the next million training steps starting Friday may increase the proportion of arXiv papers and high-quality books to 30% while reducing the proportion of open-web encyclopedia data to 15%. The data engine will adjust sampling probabilities (temperature sampling) accordingly. These proportions are example parameters and must be calibrated through ablation experiments.

### 7.5.4 Friday: Full Production Build and Release (Production Release)

**Core participants**: Full infrastructure team.
**Key workflow**:
1. **Weekly version freeze**: Incorporating the fix scripts validated during the week, generate the latest incremental tokens from the raw data lake. All metadata is recorded and updated, stored in the cloud-hosted environment, and the pointer to this latest corpus is updated in the DataLoader configuration file.
2. **Release smoke test**: Run a two-hour simulation on an idle 8-GPU node to ensure that the sequence incorporating the new weights—after tokenization loading, binary compressed reading, and tensor assembly—can be successfully pushed to the GPU without errors.
3. **Main training data switchover**: After confirming all is well, execute a smooth switchover on the "7B main model training cluster" on Friday evening; the model will read the latest `v1.3` data version at the next checkpoint. The entire workflow is thereby closed.

---

## 7.6 Writing Evaluation Findings Back into Collection/Cleaning Strategies

The value of an evaluation report depends entirely on its ability to exercise reverse control over upstream processes (collection and cleaning). If findings are not converted into durable governance rule assets, the monitoring system is nothing more than an alarm bell.

### 7.6.1 Writing Strategies Back to the "Data Collection Phase"

In the late stages of pre-training, operational evaluation often reveals very specific, narrow capability gaps (for example, the model performs well in a minority language such as German but frequently falls into loops when handling small languages from South Asian countries, or consistently hallucinates in the domain of quantum physics).

**Reverse collection directives**: At this point, "targeted crawling" tasks must be issued upstream. For example, specifying the directed crawl of open-source PDF books in a particular domain (such as arXiv Physics), or procuring a specific institutional financial reporting database. This practice of determining data mining directions based on identified capability gaps is an important form of data capitalization. Blindly scaling up general-purpose crawlers tends to produce only heavier accumulation of low-value data redundancy.

### 7.6.2 Writing Rules Back to the "Data Cleaning Phase"

Most benchmark evaluation failures can be attributed to one of two extreme tendencies in the data cleaning system: over-retention and over-filtering.

- **Countering over-deduplication**: When a model exhibits obvious gaps in a specific event dimension, investigation may reveal that hundreds of news articles covering that dimension were all filtered out by MinHash at a 0.85 similarity threshold because they cited the same highly similar background summary. A rule write-back is needed: establish domain whitelisting, or lower the deduplication threshold for specific topics to reclaim incorrectly eliminated documents.
- **Countering knowledge contamination**: If the security evaluation team discovers during a canary release that certain responses reference closed or high-risk URLs, tracing via source domain ID may reveal that a class of old URLs has been tampered with. The associated domains must immediately be added to a high-priority blocklist on the cleaning side, and the relevant corpus segment must be re-cleaned.

### 7.6.3 Consolidating into Reusable Assets for SFT (Instruction Fine-Tuning) and RAG

In large model engineering, the final and greatest strategic value of the pre-training corpus cleaning pipeline is the "downstream flow of assets."

Data that has undergone multiple rounds of rigorous evaluation and manual review during pre-training and has ultimately contributed to improvements in model metrics is commonly referred to as a "Golden Dataset." This high-value content (for example, well-formatted long-form expository text, or structured data with high-resolution typographic layouts) should not be used only in a single pre-training run.

During the subsequent SFT (supervised fine-tuning) phase, high-quality data from this pool can be used to reverse-generate unsupervised questions, using a teacher model (e.g., GPT-4) for instruction generation. During RAG (retrieval-augmented generation) system development, the clean, semantically aligned, and appropriately sized high-quality text chunks retained during pre-training can be converted into vector index sources for the knowledge base.

This is also why the data flywheel can continue to turn: the initially costly manual sampling tests and offline evaluations yield, over time, a deepening team-wide intuition and consensus on "what constitutes good data." The pipeline configurations and domain-specific vocabulary lists that remain as a result will become long-term enterprise knowledge capital that outlasts any single generation of trained models.

---

## Chapter Summary

This chapter has articulated the data asset governance logic centered on "data evaluation, quality feedback loops, and operational iteration." We began with the "one-time delivery engineering illusion"—an oversight that cannot be tolerated in large model training—and demonstrated that placing data testing at the end of the training pipeline significantly increases rework costs.

To address this, the chapter systematically explained the necessity of establishing "offline proxy metrics (PPL/TTR, etc.)," and from there introduced DVC version comparison, issue sample pool retention, and A/B testing, ultimately distilling everything into an agile workflow comprising four operational action cycles. This ensures that data development for large language models is no longer an isolated black-box process, but rather a quality feedback loop in which capability gaps can be traced upstream to drive improvements in collection and cleaning strategies.

From raw web pages through quality control, cleaning and deduplication, and data mixing to efficient delivery to the GPU, Part II has covered the main pipeline of text pre-training data engineering. The next chapter begins Part III, which addresses multimodal data engineering—structurally more complex, more costly, and subject to stricter alignment requirements: **Chapter 8: Image-Text Pair Data Engineering**.

## References

Chen M, Tworek J, Jun H, Yuan Q, Pinto H P d O, Kaplan J, Edwards H, Burda Y, Joseph N, Brockman G, others (2021) Evaluating Large Language Models Trained on Code (HumanEval). arXiv preprint arXiv:2107.03374.

Cobbe K, Kosaraju V, Bavarian M, Chen M, Jun H, Kaiser L, Plappert M, Tworek J, Hilton J, Nakano R, Hesse C, Schulman J (2021) Training Verifiers to Solve Math Word Problems (GSM8K). arXiv preprint arXiv:2110.14168.

Covington M A, McFall J D (2010) Cutting the Gordian Knot: The Moving-Average Type–Token Ratio (MATTR). Journal of Quantitative Linguistics 17(2):94-100.

Heafield K (2011) KenLM: Faster and Smaller Language Model Queries. In: Proceedings of the Sixth Workshop on Statistical Machine Translation, pp 187-197.

Hendrycks D, Burns C, Basart S, Zou A, Mazeika M, Song D, Steinhardt J (2021) Measuring Massive Multitask Language Understanding (MMLU). In: International Conference on Learning Representations.

Lees A, Tran V Q, Tay Y, Sorensen J, Gupta J, Metzler D, Vasserman L (2022) A New Generation of Perspective API. In: Proceedings of KDD 2022, pp 3197-3207.

Kuprieiev R, Petrov D, Shcheklein I, et al. (2020) DVC: Data Version Control - Git for Data & Models. Zenodo. https://doi.org/10.5281/zenodo.012345 (software; versioned DOI available in the Zenodo record; maintained by Iterative.ai)
