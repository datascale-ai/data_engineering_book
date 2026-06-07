# Chapter 7: Data Evaluation, Quality Loops, and Operational Iteration

## Abstract

This chapter discusses how pre-training data should be evaluated, versioned, and operated after cleaning. We begin with an anonymized composite case showing that "cleaner" data does not automatically produce a better model. We then build a DataOps framework for pre-training corpora: offline proxy metrics, representative sampling, quality dashboards, issue sample pools, version comparison, and feedback to upstream acquisition and cleaning strategies. The metric section explains perplexity, type-token ratio, toxicity and PII density, benchmark contamination, and domain coverage, while emphasizing that these are proxy signals that must be connected to small validator models and human review. The second half introduces 5 Whys root-cause review, weekly operating cadence, dashboard alerts, and reuse paths into SFT and RAG assets. Readers should be able to turn data processing from one-time delivery into a traceable, rollbackable, auditable, and continuously improving operating system.

## Keywords

Data operations; proxy evaluation; quality loop; DVC; issue sample pool; A/B testing; data drift; dashboard

## Learning Objectives

- Explain why pre-training data needs continuous evaluation and versioned operation.
- Design offline proxy metrics such as PPL, TTR, PII density, benchmark contamination, and domain coverage.
- Use issue sample pools, DVC versions, and A/B tests to locate model effects caused by data changes.
- Build quality dashboards, automated alerts, and weekly operating rhythms.
- Feed evaluation conclusions back into acquisition, cleaning, SFT, and RAG data assets.

## Opening: An Unexpected Regression

The following anonymized composite case uses approximate metrics, timelines, and data scales. A 7B language-model project spent two months cleaning its pre-training corpus very aggressively. Short texts were removed, high-perplexity documents were filtered, and MinHash thresholds were tightened. The team believed it had produced a cleaner and more controllable dataset.

The model trained on this new dataset, version v2.0, underperformed the earlier model trained on rougher data, version v1.0. The postmortem found three facts:

1. Because documents with many line breaks and symbols had been removed, the model lost much of its code and Markdown ability.
2. Because "nonstandard colloquial expressions" had been filtered, the model became worse at empathetic dialogue and sounded like a cold encyclopedia.
3. Strict deduplication lowered the frequency of some high-value common facts, causing knowledge forgetting.

The conclusion was sharp: **quality is not a static notion, and cleanliness divorced from model effect is insufficient**. This chapter moves from cleaning code to system operation: how to evaluate data before expensive training, observe data effects during model development, and feed evidence back into source and cleaning strategy.

---

## 7.1 Why Data Also Needs Continuous Operations

### 7.1.1 Breaking the One-time Delivery Illusion

In older supervised-learning workflows, a dataset was often treated as a static asset: collect it, label it, publish it, and freeze it. LLM pre-training does not work that way. The data-model boundary is dynamic. Early training needs broad data to learn grammar and world knowledge; cooldown needs high-density material such as code, math, books, and scientific text to raise the capability ceiling.

This changes the role of data engineers. They are no longer one-time data deliverers. They become operators of data recipes, quality boundaries, and release gates. The organization therefore needs DataOps for large-model data.

### 7.1.2 Why Evaluating Only After Training Is Too Late

A traditional pipeline often looks like this: data team cleans for a month, training team runs for a month, evaluation team tests at the end. If results are poor, root-cause analysis is hard. Was the data bad? Was the mix ratio wrong? Did the optimizer fail? Did the learning-rate schedule collapse?

Because long LLM training runs are expensive, evaluation must move earlier. Offline proxy metrics and small validator models should inspect data before it reaches the full training run.

### 7.1.3 Collaboration Between DataOps and ModelOps

Modern model organizations have overlapping responsibilities:

- **Model engineers** monitor cluster health, gradient behavior, architecture experiments, and training schedules.
- **Data engineers** monitor pipeline throughput, token cost, parser failures, and storage layout.
- **DataOps / evaluation owners** connect the two sides. They trace training anomalies or capability regressions back to source batches, parser versions, cleaning rules, and mix changes.

![Figure 7-1: Data operations flywheel](../../images/part2/data_operations_flywheel.png)

*Figure 7-1: Data operations flywheel. Data production, model evaluation, root-cause analysis, rule updates, and asset reuse form a continuous loop.*

---

## 7.2 Offline Evaluation and Proxy Metrics

Proxy metrics provide earlier signals than full training. They do not replace model evaluation, but they reduce the chance of discovering data failures only after a costly run.

### 7.2.1 Statistical Stratification and Representative Evaluation

At trillion-token scale, full manual or model-based evaluation is usually unnecessary and expensive. The core method is stratified sampling. Before serialization, sample fixed sandboxes from each source or domain, for example 0.1% or 0.01% of documents, or a 100M-token subset. Evaluate these sandboxes repeatedly. If the sampled distribution is abnormal, the whole batch is at risk.

The evaluation should cover four dimensions: fluency and density, lexical diversity, safety/privacy residue, and domain/benchmark overlap.

### 7.2.2 Core Offline Proxy Metrics

#### 1. Linguistic Quality and Perplexity

**Goal.** Measure fluency, density, and whether the text resembles natural target-language data.

**Method.** Use a mature reference model, such as a small base LLM or a 1B validator model, to run no-gradient forward passes on sampled documents.

**Interpretation.** Perplexity is the exponential of cross-entropy loss. It is not simply "lower is better." Extremely low PPL may indicate boilerplate, repeated disclaimers, or SEO residue. Extremely high PPL usually indicates gibberish, broken parsing, machine-translation artifacts, or encoding failure. With a neural reference model, good text might cluster around PPL 20-150; with KenLM, comparable text can have much larger values. Thresholds cannot be shared across reference models.

**Listing 7-1: Offline perplexity sampling**

```python
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
            if inputs.input_ids.shape[1] < 50:
                continue
            outputs = model(**inputs, labels=inputs.input_ids)
            ppl_results.append(torch.exp(outputs.loss).item())
    return ppl_results
```

#### 2. Type-Token Ratio and Vocabulary Coverage

**Goal.** Detect whether cleaning or deduplication removed too much lexical diversity.

**Method.** Compute the ratio of unique types to total tokens, preferably with windowed measures such as MATTR for long documents.

**Interpretation.** If a 100M-token sandbox contains surprisingly few unique terms, the corpus may be dominated by e-commerce filler, machine-generated loops, or excessively deduplicated text. Domain-specific vocabulary coverage should also be tracked: rare diseases, legal terms, code frameworks, names, or product entities may require explicit coverage lists.

**Listing 7-2: Type-token ratio**

```python
def calculate_ttr(texts, tokenizer=None):
    if tokenizer is None:
        tokens = " ".join(texts).split()
    else:
        tokens = tokenizer.tokenize(" ".join(texts))
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)
```

#### 3. Toxicity and PII Density

**Goal.** Monitor safety and privacy residues that affect deployment risk.

**Method.** Use a safety classifier for hate, harassment, sexual content, and other harmful categories, and combine it with regex/NER detection for phone numbers, IDs, email addresses, passwords, and API tokens.

**Interpretation.** Mean toxicity is not enough. Track P99 or P99.9. If P99 exceeds a threshold such as 0.8, quarantine and review the source. For PII, monitor trigger rates and inspect false negatives manually.

#### 4. Domain Classification and Benchmark Overlap

**Goal.** Prevent benchmark contamination and measure domain distribution.

**Method.** Precompute 13-gram or 15-gram hashes for benchmark test sets such as MMLU, GSM8K, HumanEval, CEVAL, and internal exams. Compare sampled documents against these fingerprints.

**Interpretation.** Overlap should approach zero. If a new batch shows meaningful overlap, a blog, repository, or discussion thread may contain benchmark items and must be removed or quarantined.

### 7.2.3 Proxy Metrics and Real Effect Can Diverge

All offline metrics are proxies. A machine-translated corpus can have attractive PPL and TTR while still damaging cultural knowledge and writing style. A code corpus may look noisy under prose-oriented rules while being valuable for training. Therefore, proxy metrics should be connected to small validator models, task-specific evaluation, and human sampling.

### 7.2.4 Mapping Metrics to Governance Actions

Evaluation must end in action, not observation.

**Table 7-1: Metric signals and governance actions**

| Metric signal | Likely cause | Required action |
| :--- | :--- | :--- |
| Sampling TTR drops | MinHash too aggressive; legitimate overlap removed | Relax threshold, protect domain terms, rerun sample review |
| PPL long tail grows | HTML residue, encoding failure, parser regression | Inspect high-PPL samples and add parser/regex rules |
| Code evaluation drops | Newline or indentation normalization damaged code | Route code through a separate parser and normalization path |
| Model repeats during pre-training | Source snapshots duplicated or exact dedup failed | Run sequence-level SHA-256 and source blocking |
| Loss spikes locally | Corrupt or high-entropy garbage in one shard | Trace batch ID to shard and quarantine source |
| Toxicity rises sharply | Forum or community source added unsafe sections | Update safety filters and purge history |

---

## 7.3 Version Iteration, Sampling, and Root-Cause Review

### 7.3.1 Dataset Versioning and A/B Comparison With DVC

Large data lakes need version control just as code does. DVC, lakeFS, Delta Lake snapshots, or immutable object-store pointers can prevent accidental in-place overwrite. Every processing change should create a new version that can be compared and rolled back.

Before a new strategy is deployed globally, run a small A/B experiment: train two small models, for example a 0.5B or 1B model on 1B tokens, with all variables fixed except the data change. Only promote the data change if target metrics improve without unacceptable regressions.

### 7.3.2 Building an Issue Sample Pool

The most valuable artifact from postmortems is the issue sample pool. It should contain examples of parser misalignment, classifier misses, false deletions, PII leakage, benchmark contamination, corrupt PDFs, and source-specific failures. Each issue sample should preserve provenance:

- Which crawler or source produced it?
- Which parser version transformed it?
- Which rule or model failed to catch it?
- Which data version introduced it?
- Which model behavior exposed it?

This pool becomes a regression sandbox for future rule changes.

### 7.3.3 5 Whys Root-Cause Review

The purpose of review is not blame; it is converting incidents into reusable rules. A typical 5 Whys chain might be:

1. Why does the model generate meaningless whitespace in code? Because some code samples lost indentation.
2. Why did they lose indentation? A Markdown-to-text conversion removed newlines and tabs.
3. Why was that rule introduced? It was added to reduce PPL for noisy fiction data.
4. Why did a fiction rule affect code? Cleaning routes were not separated by domain.
5. Why was this merged? The CI gate lacked multi-domain regression samples.

The governance action is to separate domain-specific cleaning paths and require regression sandbox tests before merging global rules.

### 7.3.4 Case Review: Root Cause of a Training Loss Spike

An anonymized large-scale training job triggered an alert near the 87B-token mark. Loss rose from about 1.8 to 14.5 within 15 steps, and gradient norm became NaN. Model engineers paused training and rolled back to an earlier checkpoint; DataOps took over root-cause analysis.

**Batch trapping.** Logs identified the last token batch loaded before the anomaly.

**Detokenization.** The token IDs decoded into a 4,096-token sequence full of broken Unicode and meaningless replacement characters.

**Batch-to-source tracing.** The global document ID pointed to a PDF batch ingested three weeks earlier from a public thesis collection.

**Cleaner defect.** The PDFs had encrypted or obfuscated text layers. FastText language ID misclassified the byte garbage as a low-resource language with high confidence, and the uniqueness of the text allowed it to survive deduplication.

**Actions.** The team quarantined the source, added a valid-UTF-8 and target-script-ratio gate before language ID, and added a small LLM judge for basic grammatical plausibility on heterogeneous text.

**Table 7-2: Data version-release record template**

| Dimension | Example field |
| :--- | :--- |
| Basic info | Version v2.1 to v2.2; owner: DataOps; date: 2026-X-X |
| Changelog | Added 30 GB StackOverflow Q&A; tightened MinHash 0.85 to 0.8; fixed a `<p>` parser bug |
| Scale change | Planned +50 GB, net +23 GB after cleaning; total 1.45T tokens |
| A/B result | HumanEval +4.1 points, common benchmarks within +/-0.3% |
| Known issue | Some old StackOverflow answers remain; timestamp filter planned in v2.3 |
| Release decision | Approved for v2.2 production training queue |

---

## 7.4 Data Operations Dashboard and Organizational Coordination

### 7.4.1 Core Dashboard Modules

A useful dashboard should include:

1. **Global inventory:** ingestion volume by source, remaining inventory, and consumed-token progress.
2. **Cleaning funnel yield:** language-ID rejection, heuristic-filter rejection, fuzzy-dedup rejection, PII quarantine, and other stage-level retention rates.
3. **Safety baseline:** counts and rates for PII, toxic documents, and blocked sensitive sources.
4. **Sampling audit traffic lights:** weekly blind-review scores for fluency, correctness, source risk, and parser quality.

![Figure 7-2: Data evaluation loop](../../images/part2/data_evaluation_loop.png)

*Figure 7-2: Data evaluation loop. Sampling, metric anomaly, root-cause analysis, governance action, and rule update form a closed loop.*

### 7.4.2 Automated Alert Architecture

Dashboards should not be passive reports. Mature data factories add active alerting and blocking, often with Flink, Spark Streaming, or similar systems.

**Level 1: data drift alerts.** Daily batches are sampled and compared with historical distributions. If a term family or source signature rises by several hundred percent, the batch should be suspended pending review.

**Level 2: cost and latency alerts.** If a regex filtering stage drops from 2 GB/s to 100 MB/s and CPU cores are saturated, catastrophic backtracking or pathological documents may be causing a ReDoS-like slowdown. The scheduler should kill timed-out tasks and isolate the input.

**Level 3: multimodal alerts.** As image-text data enters the pipeline, dashboards should track broken-image rates, CLIP similarity extremes, and caption-image mismatch rates.

### 7.4.3 Tiered Responsibility Boundaries

Clear ownership reduces confusion during incidents.

- **Data infrastructure** owns stable throughput and cost per token. It must diagnose DataLoader and storage bottlenecks.
- **Data ingestion** owns source coverage and legal/copyright safety. It investigates source-level PII or unsafe-data escape.
- **Pre-training researchers** own architecture, hyperparameters, and whether the data mix supports scaling-law behavior.
- **DataOps / evaluation** owns dashboards, alerts, mix decisions, and feedback from model evaluation to data policy.

---

## 7.5 Weekly Operating Cadence

Large-model data systems evolve too quickly for occasional postmortems. A weekly rhythm turns evaluation into action.

### 7.5.1 Monday: Metrics Review

Participants include data engineers, data product owners, and evaluation leads. Review weekend token consumption, MFU and DataLoader health, the latest cleaned-batch reports, PPL/TTR histograms, text-length distributions, safety-block rates, and any dashboard alerts. The goal is not to solve every anomaly immediately, but to pick the anomalies requiring root-cause drilldown.

### 7.5.2 Tuesday and Wednesday: Root-Cause Drilldown and Small Experiments

Extract around 200 raw samples from the suspicious area. Human reviewers label whether problems are false positives, false negatives, parser errors, PII misses, or domain-specific quality issues. Engineers patch the relevant cleaning route and run the affected batch again. The new sandbox is then used for a small A/B run, often 12-24 hours on a 0.5B or 1B model.

### 7.5.3 Thursday: Experiment Decision and Data Mixing

Model and data teams compare validation loss and downstream metrics. If a patch improves the target capability without harming general benchmarks, it is promoted. The team also decides next week's mix: for example, increasing arXiv and high-quality book share if reasoning is weak, or restoring dialogue data if interaction style is degrading.

### 7.5.4 Friday: Production Release

The accepted cleaning rules and mix changes are frozen into a weekly data version. Metadata is updated, pointers for DataLoader configuration are changed, and a realistic smoke test runs on a small GPU node. After the smoke test passes, the main training cluster reads the new version at the next checkpoint boundary.

---

## 7.6 Writing Evaluation Conclusions Back to Upstream Strategy

### 7.6.1 Feedback to Acquisition

If a domain consistently produces high-quality samples, the crawler should increase its weight or add more URLs from similar sources. If a source repeatedly produces high PPL, PII, or copyright issues, it should be downweighted, graylisted, or blocked. Acquisition is therefore not only about coverage; it is guided by downstream evidence.

### 7.6.2 Feedback to Cleaning Rules

Evaluation should update cleaning thresholds and route design. Code, prose, legal text, math, and dialogue need different normalization and filtering. Global rules are dangerous unless tested across multiple domains.

### 7.6.3 Reuse as SFT and RAG Assets

Issue samples, high-quality documents, and domain-specific reviewed examples should not disappear after pre-training. Clean explanations can become SFT seeds; trusted documents can enter RAG indexes; hard negatives can support evaluation sets; and user-feedback issues can become preference data. A good DataOps loop turns every review into reusable data capital.

## Chapter Summary

Pre-training data is a living system. It needs proxy metrics, version control, issue samples, dashboards, alerts, weekly cadence, A/B experiments, and upstream feedback. The goal is not to maximize a single cleanliness metric, but to continuously improve the relationship between data, model behavior, cost, and governance.

## References

- Cobbe, K. et al. (2021). Training Verifiers to Solve Math Word Problems.
- Covington, M. A., and McFall, J. D. (2010). Cutting the Gordian Knot: The Moving-Average Type-Token Ratio.
- Heafield, K. (2011). KenLM: Faster and Smaller Language Model Queries.
- Hendrycks, D. et al. (2021). Measuring Massive Multitask Language Understanding.
- Lees, A. et al. (2022). A new generation of Perspective API toxicity models.
- Ruslan, K. et al. (2021). Data Version Control.
