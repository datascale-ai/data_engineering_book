# Chapter 42: Dolma Pre-training Corpus Transparent Ledger

## Abstract

Most open model releases provide weights, inference code, and some evaluation results, but training data is often reduced to vague source descriptions. This is insufficient for continued pre-training, bias analysis, evaluation-contamination checks, and license audits. When a model exhibits a problem, the team can hardly answer which sources it saw, how those sources were filtered, what the sampling proportions were, whether a benchmark leaked into the training corpus, or which shard should be located when a user requests removal of personal data.

Dolma is a three-trillion-token-scale English pre-training corpus released by the Allen Institute for AI to support open language-model research and OLMo training. Its value is not only scale; it turns pre-training data into an auditable engineering record: versions, sources, source-level statistics, sample proportions, the ODC-BY license, original-source terms, a personal-data removal entry point, and the Dolma Toolkit processing flow are all documented publicly. Around this ledger, the chapter starts from the data gap in open-model research, then reviews Dolma versions and source statistics, then follows the source ledger, individual document records, and training manifest to decompose the sample structure. Finally, it discusses how Dolma Toolkit turns tag, dedup, mix, and tokenize into auditable actions, and where transparent corpora are bounded in quality control, removal, and enterprise migration.

## Keywords

Dolma; transparent pre-training corpus; OLMo; source mix; token accounting; source card; manifest; Dolma Toolkit; ODC-BY; data audit

## 42.0 Learning Objectives

After completing this chapter, readers should be able to:

- Explain why open weights cannot substitute for training-data transparency.
- Read Dolma's versions, source statistics, sampling proportions, and ODC-BY usage boundaries.
- Distinguish the roles of document records, source cards, and training manifests in audits.
- Use token-accounting formulas to describe the relationship among raw tokens, filtered tokens, sample proportions, and seen tokens.
- Understand the evidence chain of transparent corpora through the four Dolma Toolkit actions: tag, dedup, mix, and tokenize.
- Design source ledgers, removal ledgers, contamination checks, and version-freezing mechanisms for internal enterprise pre-training corpora.

## 42.1 Problem Scenario: Open Weights Still Cannot Explain the Model

Two teams obtain the same 7B open-model weights. The first team wants to continue pre-training the model to improve code and scientific-question-answering capabilities. The second team wants to analyze why the model gives outdated answers on several factual questions. The weights, inference code, and part of the evaluation scripts are downloadable, but they quickly encounter the same problem: no one can answer what data the model actually saw.

The continued-pretraining team needs to know how much code, papers, encyclopedic content, Web text, and social media were present in the original model corpus, so that it does not repeatedly oversample the same data types. The bias-analysis team needs to know whether certain Web pages, papers, forum discussions, or benchmark solutions entered training, so it can judge whether a problem comes from knowledge gaps, sampling weights, contamination, or model training itself. Without training corpora and processing records, every explanation remains guesswork.

Dolma is designed to solve exactly this problem. It turns an English pre-training corpus from an invisible data recipe into a set of downloadable, measurable, processable, removable, and auditable sources. OLMo is not a parallel protagonist in this chapter, but a downstream example of Dolma being consumed by a transparent training chain. It reminds us that open model research should not only open weights; it should open training data, processing tools, and evaluation code as far as possible.

### 42.1.1 Open-model Research Needs Data Evidence

"Open" has different levels. Releasing only weights allows users to run a model, but does not allow researchers to explain it. Releasing training code allows users to reproduce the training framework, but still does not explain what the model actually saw. Data transparency that supports scientific research must answer at least six classes of questions.

- Source: Does each document come from Common Crawl, code repositories, papers, books, encyclopedias, or social platforms?
- Version: What were the source acquisition time, processing-script version, and filtering rules?
- Scale: Are raw tokens, filtered tokens, sampled tokens, and seen tokens consistent?
- Contamination: Did evaluation sets, solutions, or customer test sets overlap with the training corpus?
- License: How do the dataset release license and original-source terms jointly constrain users?
- Removal: If a user requests removal of personal data, can the corresponding document be located and handled?

The Dolma dataset card directly reflects this design orientation: it lists versions, summary statistics, download methods, license information, and a personal-data removal entry point. The Dolma GitHub repository further provides data and tools, so transparency is not limited to paper descriptions.

For corpora like Dolma, the core object of transparency is a ledger; a single `text` field is not the only object. More important are three ledgers formed around `text`.

The first is the source ledger, recording where each data type comes from, how large it is, what cutoff date it uses, and how it was processed. The second is the processing ledger, recording how taggers, filters, deduplication strategies, mixers, and tokenizers changed raw documents. The third is the training ledger, recording which sources a training run actually sampled, what the sampling proportions were, and how seen tokens were distributed across training steps.

If these three ledgers are disconnected, transparency degrades into "downloadability." The data can be downloaded, but cannot explain the model; the model can be trained, but cannot be audited; versions can be updated, but cannot be compared.

## 42.2 Dataset Overview: Versions, Scale, and Source Structure

Dolma is not a single static file, but a corpus asset with version evolution. The Hugging Face dataset card lists versions such as `v1`, `v1_5`, `v1_5-sample`, `v1_6`, `v1_6-sample`, and `v1_7`. Among them, `v1_7` is used to train OLMo 7B-v1.7 and introduces new sources, more quality filtering, and fuzzy deduplication.

*Table 42-1 Public Dolma Versions and Uses*

| Version | Release Date | Compressed Size | Dataset-card Description | Engineering Use |
| --- | --- | ---: | --- | --- |
| `v1` | 2023-08-18 | 6.0 TB | First Dolma release | Trace the earliest public corpus form |
| `v1_5` | 2023-10-31 | 6.4 TB | Used to train OLMo-1B, about 3T tokens | Review early OLMo training corpus |
| `v1_5-sample` | 2023-10-31 | 2.9 TB | Sample of about 1.9T tokens, used for OLMo-7B | Track training samples below full scale |
| `v1_6` | 2024-01-31 | 5.4 TB | Adds partial deduplication and repeated n-gram filtering on top of v1.5 | Study filtering and deduplication evolution |
| `v1_6-sample` | 2024-01-31 | 16.4 GB | Exploratory sample of about 10B tokens | Quick debugging and data browsing |
| `v1_7` | 2024-04-15 | 4.5 TB | Used to train OLMo 7B-v1.7, with new sources, more quality filtering, and fuzzy deduplication | Current default version and transparent-training reference |

Source: Versions section of the Hugging Face `allenai/dolma` dataset card.

### 42.2.1 v1.6 Source Structure

Dolma covers Web, code, papers, social media, books, and encyclopedic sources. To avoid mixing versions, Table 42-2 uses the coarse-grained statistics from the dataset card's v1.6 summary statistics. The v1.7 sources are more fine-grained, adding Refined Web, StarCoder, arXiv, StackExchange, Flan, OpenWebMath, Algebraic Stack, MegaWika, and other sources. Subsequent writing or experiments should explicitly state which version is used.

*Table 42-2 Dolma v1.6 Source Statistics*

| Source | Document Type | UTF-8 Bytes | Documents | Unicode Words | Llama Tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Common Crawl | web pages | 9,022 GB | 3,370M | 1,775B | 2,281B |
| The Stack | code | 1,043 GB | 210M | 260B | 411B |
| C4 | web pages | 790 GB | 364M | 153B | 198B |
| Reddit | social media | 339 GB | 377M | 72B | 89B |
| PeS2o | STEM papers | 268 GB | 38.8M | 50B | 70B |
| Project Gutenberg | books | 20.4 GB | 0.056M | 4.0B | 6.0B |
| Wikipedia and Wikibooks | encyclopedic | 16.2 GB | 6.2M | 3.7B | 4.3B |
| Total | mixed | 11,519 GB | 4,367M | 2,318B | 3,059B |

Source: Hugging Face `allenai/dolma` dataset card, Summary Statistics v1.6. GB, M, and B follow the dataset-card convention.

Table 42-2 should not be read only as a scale display. It reveals three engineering facts.

First, Dolma is a source mix, not a single Web dump. Common Crawl accounts for a large share, but code, papers, social media, books, and encyclopedic content all enter the corpus in different forms. Changes in model capability cannot be vaguely attributed to "more Web data."

Second, different statistical conventions serve different questions. UTF-8 bytes help estimate storage and processing cost, document count helps observe sample granularity, while Unicode words and Llama tokens are closer to the training budget. Mixing these conventions distorts discussions of data scale.

Third, versions cannot be directly compared without care. v1.6 and v1.7 differ in source decomposition, filtering rules, and sample proportions. If a model is trained with v1.7, one cannot explain its training behavior using only the v1.6 coarse table.

## 42.3 Decomposing a Transparent Chain Through the Source Ledger

This section does not begin from an individual text, but from Dolma's source ledger to decompose the chain of a transparent pre-training corpus. The "sample" here is not an image or a question, but an evidence path from source to document and then to the training manifest.

### 42.3.1 From Source to Document

Dolma's task is not to annotate supervised labels for each text, but to make records consumed by training reconstructable. Let the source set be $S=\{s_1,\ldots,s_m\}$. Each source is processed by a function $P_s$ into a document set $D_s$:

$$
D_s=P_s(R_s, C_s)
$$

Here, $R_s$ is the raw source, and $C_s$ is the processing configuration for that source, including taggers, filters, deduplication strategy, sampling proportion, and tokenizer. The final training corpus is a mixture of multiple sources:

$$
D=\bigcup_{s \in S} Sample(D_s, r_s)
$$

Dolma's transparency lies in recording $S$, $R_s$, $C_s$, $r_s$, and version information as publicly as possible. Model training is no longer just "used three trillion tokens"; it can be traced to which sources contributed how much, how they were processed, and how they were sampled.

### 42.3.2 Token Accounting Should Track Actual Training Consumption

The easiest part of a multi-source corpus to misread is token scale. A source with many raw tokens does not necessarily contribute the same proportion during training; filtering, deduplication, sample proportion, and multiple epochs all change the final seen tokens.

The actual contribution of a source $s$ during training can be written as:

$$
T^{seen}_s = T^{filtered}_s \times r_s \times e_s
$$

where $T^{filtered}_s$ is the number of filtered tokens, $r_s$ is the sampling proportion, and $e_s$ is the epoch count or equivalent sampling count during training. The proportion of that source in the training mix is:

$$
p_s=\frac{T^{seen}_s}{\sum_j T^{seen}_j}
$$

The Dolma v1.7 dataset card lists both source token counts and sample proportions precisely so that users can distinguish "what is in the dataset" from "how much the model actually saw."

### 42.3.3 Three Ways to Read a Document

Transparent corpora do not end with packaging and uploading `text`. At least three layers of records are needed: an individual document record, a source card, and a training-version manifest. The document record supports training and location; the source card explains data origin and processing rules; the training manifest reproduces the data actually consumed by a model run.

*Table 42-3 Dolma-like Transparent Corpus Record Schema*

| Layer | Typical Fields | Source or Generation Method | Engineering Use |
| --- | --- | --- | --- |
| Document level | `id`, `source`, `text`, `text_hash` | Data reader and hash computation | Locate samples, deduplicate, read during training |
| Document level | `created_at`, `url_or_origin`, `license_hint` | Original source metadata | License review and time tracking |
| Document level | `language_tag`, `toxicity_tag`, `perplexity_score` | Dolma Toolkit taggers or custom taggers | Quality filtering and risk bucketing |
| Source level | `source_name`, `source_version`, `raw_size`, `filtered_size` | Source card and statistics scripts | Explain data composition |
| Source level | `dedup_policy`, `filter_config`, `sample_proportion` | Dolma mixer and processing configs | Reproduce source mix |
| Training level | `dolma_version`, `tokenizer`, `sample_seed`, `seen_tokens` | Training manifest | Reproduce experiments and explain metric changes |
| Governance level | `removal_status`, `known_limitations`, `release_constraints` | Dataset card and governance records | Handle removal, bias, and usage boundaries |

These fields are an engineering schema organized by the author from the Dolma dataset card, Dolma Toolkit documentation, and transparent-training audit needs. They do not imply that Dolma officially publishes each field exactly as listed.

The following is an abstract Dolma-like document record showing how a transparent corpus connects samples, sources, and training versions.

```json
{
  "id": "dolma-v1_7/common-crawl/doc-000001",
  "source": "Dolma's CC",
  "source_version": "v1_7",
  "text": "<document text>",
  "text_hash": "sha256:...",
  "tags": {
    "language": "en",
    "toxicity_bucket": "low",
    "perplexity_bucket": "normal"
  },
  "processing": {
    "tagger_config": "dolma-toolkit-config-x",
    "dedup_policy": "fuzzy_dedup",
    "sample_proportion": 0.5
  },
  "training_manifest": {
    "tokenizer": "OLMo tokenizer version",
    "dolma_version": "v1_7",
    "included_in_run": true
  }
}
```

This record cannot be read at only one layer. In Dolma, the document layer, source layer, and training layer must point back to one another. If a document has a `source` but no source card, it can locate a sample but cannot explain the origin. If a source card has statistics but no training manifest, it can explain what is in the dataset but not what the model actually saw. If a manifest has sampling proportions but no document hash, removal and contamination checks break.

## 42.4 Dolma Toolkit Makes the Evidence Chain Executable

The Dolma GitHub repository states that Dolma is both a dataset and a toolkit. Dolma Toolkit supports single-machine, cluster, and cloud environments. It includes language detection, toxicity detection, perplexity scoring, and common filtering recipes such as Gopher, C4, and OpenWebText; its deduplication component uses a Rust Bloom filter for acceleration.

### 42.4.1 Four Actions Correspond to Four Kinds of Evidence

Dolma Toolkit documentation summarizes data organization as four actions: tag, dedup, mix, and tokenize. They are not isolated scripts, but evidence-chain generators: tag records document attributes, dedup records what is retained and removed, mix records sampling proportions, and tokenize records the token convention that enters training.

*Table 42-4 Dolma Toolkit Processing Actions and Evidence Outputs*

| Order | Action | Official Documentation Description | Evidence Output | Main Risk |
| ---: | --- | --- | --- | --- |
| 1 | Taggers | Assign language, toxicity, perplexity, and other attribute tags to document spans | Document quality labels and risk labels | Tagger-version changes alter filtering results |
| 2 | Deduplication | Deduplicate documents by content or metadata | Deduplication strategy, retention priority, deletion records | Cross-source deduplication changes source mix |
| 3 | Mixer | Remove, filter, or mix documents based on attribute values | Sample proportion, source mix, data version | Opaque sample proportions cause token-accounting errors |
| 4 | Tokenization | Use Hugging Face-compatible tokenizers | Token counts, tokenizer version, training stream | Tokenizer changes alter token counts and training budget |

Source: Dolma Toolkit documentation README.

![Figure 42-1 Dolma transparent-corpus evidence chain](../../images/part12/ch42_01_dolma_evidence_chain_en.svg)

*Figure 42-1 Dolma transparent-corpus evidence chain. Source: original illustration based on AllenAI Dolma Toolkit documentation.*

The boundary between toolchains and manual audits matters. A toolchain can stably generate statistics, tags, hashes, and manifests, but it cannot replace all audits. License boundaries, PII removal, evaluation contamination, and source representativeness still require human rules, sample review, or dedicated detection tasks.

This is also the difference between a Dolma-like transparent corpus and an ordinary "collection of cleaning scripts." Ordinary scripts only answer "what did I delete"; a transparent toolchain must also answer "why was it deleted, how did the post-deletion distribution change, did training actually benefit, and can it be removed later." If a processing action cannot leave interpretable evidence, it is hard for it to support transparent training.

## 42.5 Evaluation Shifts from Highest Score to Attribution

The evaluation focus of transparent corpora is not the "highest score," but whether score changes can be explained. Dolma-like source-level corpora let model evaluation move beyond leaderboards and trace training logs back to sources, versions, and sampling proportions.

### 42.5.1 Source Ablation Locates Capability Sources

To judge whether a source affects model capability, one can run source ablation. Let full training produce model $M_{all}$, and training with source $s$ removed produce model $M_{-s}$. The average difference over task set $B$ is:

$$
\Delta_s=\frac{1}{|B|}\sum_{b \in B} \left[score(M_{all}, b)-score(M_{-s}, b)\right]
$$

When $\Delta_s$ changes clearly on code tasks, scientific QA, or long-context tasks, the data team can trace capability changes back to source mix instead of vaguely attributing them to "model parameters."

![Figure 42-2 Dolma source mix and training-diagnosis loop](../../images/part12/ch42_02_dolma_source_mix_diagnosis_en.svg)

*Figure 42-2 Dolma source mix and training-diagnosis loop. Source: original illustration based on the Dolma dataset card and OLMo training use.*

### 42.5.2 Diagnosis Checklist

*Table 42-5 Dolma-like Transparent Corpus Evaluation and Diagnosis Table*

| Evaluation Question | Required Records | Metric or Evidence | Possible Action |
| --- | --- | --- | --- |
| Which source drives a capability | Source mix, sample proportion, seen tokens | Source ablation, task-score difference $\Delta_s$ | Adjust source weight or add data |
| Whether an evaluation set is contaminated | Document hash, n-gram index, eval-set hash | Overlap rate, contamination span | Remove contaminated samples and freeze a new version |
| Training loss fluctuates abnormally | Batch source records, tokenizer version | Source-specific loss, token distribution | Inspect sampler and source shards |
| Social-media risk increases | Toxicity tag, PII tag, source card | Risk-tag rate, manual sample review | Tighten filtering or reduce sampling proportion |
| Versions are incomparable | `dolma_version`, `filter_config`, `dedup_policy` | Manifest diff | Freeze version or rerun controlled experiments |

### 42.5.3 Common Failure Modes

*Table 42-6 Common Failures and Repair Actions for Dolma-like Transparent Corpora*

| Failure Mode | Symptom | Possible Root Cause | Governance Action |
| --- | --- | --- | --- |
| Source-mix drift | A task category suddenly regresses in a new version | Sample proportion or filtering rules changed | Compare manifest diff, roll back, or resample by strata |
| Inconsistent token accounting | Reported scale does not match training seen tokens | Raw, filtered, and sampled tokens are mixed | Report raw, filtered, sampled, and seen tokens together |
| Cross-source duplication | Paper abstracts, encyclopedia mirrors, or code fragments repeat | Deduplication only within source | Add cross-source deduplication and record retention priority |
| Evaluation contamination | Benchmark scores become abnormally high | Web or forum data contains questions and solutions | Build eval hash/n-gram decontamination index |
| Removal cannot be located | PII removal request cannot be handled | Missing source/id/hash mapping | Build removal ledger and shard reverse index |
| Ambiguous license boundary | ODC-BY is mistaken for authorization from original sources | Original source terms are ignored | Preserve original terms and restrictions in source cards |

These failure modes show that transparent-corpus quality is not only about whether text is clean. A transparent training sample must pass three kinds of checks simultaneously: whether the source is explainable, whether the processing actions are reproducible, and whether training consumption is traceable. Missing any one of these checks can turn data into a corpus that is public but not auditable.

## 42.6 Reuse Boundaries of Transparent Corpora

Dolma is suitable for open language-model pre-training research, source-mix experiments, transparent data-governance teaching, training-data audit method validation, and internal enterprise manifest design. It is especially suitable for research questions such as "how does training data affect model capabilities and limitations," because it opens data, dataset cards, and processing tools together.

Dolma should not be understood as meaning that all sources are unconditionally usable for commercial purposes. The Hugging Face dataset card states that Dolma is released under ODC-BY, while users are still bound by the licenses and terms of use of the original data sources. In other words, Dolma's release license does not automatically erase the boundaries of Common Crawl, Reddit, code, papers, books, and other original sources.

Enterprises usually cannot publish training data, but they can transfer Dolma's transparency practices. A minimum viable version includes:

- Build a source card for each source, recording origin, version, license, acquisition time, filtering rules, and limitations.
- Freeze a manifest for each training version, recording source mix, sample proportion, tokenizer, `filter_config`, `dedup_policy`, and `sample_seed`.
- Build validation splits for major sources to support source-specific loss and source ablation.
- Build decontamination indexes for evaluation sets, customer test sets, and online issue repositories.
- Build a removal ledger so that URL, document id, or text hash can map to training shards.

At the same time, Dolma is not suitable as a direct quality standard for Chinese, multilingual, or vertical-industry corpora. It is primarily an English pre-training corpus. Migration to Chinese, medical, legal, or financial scenarios requires redefining sources, licenses, filters, and evaluation tasks. Dolma should also not be used to prove that open corpora are inherently safe. The meaning of transparency is that defects are visible, locatable, and repairable, not that defects do not exist.

## Chapter Summary

Dolma moves pre-training corpora from invisible data recipes toward auditable data assets. This chapter's core conclusions are threefold.

First, the basic unit of a transparent pre-training corpus is not a single `text` field, but a document record that connects source, processing configuration, sampling proportion, and training manifest. Second, source mix must be explained through token accounting rather than by reporting only raw scale. Third, the four Dolma Toolkit actions - tag, dedup, mix, and tokenize - provide a clear template for enterprises building reproducible training corpora internally.

For readers of this book, what is most transferable from Dolma is not a particular source or the three-trillion-token scale, but the way it leaves evidence for training data. Only when sources, processing actions, sampling proportions, training manifests, and governance records form a connected chain does a pre-training corpus have a foundation for continued research, error attribution, and long-term maintenance.

## References

- Soldaini, L., Kinney, R., Bhagia, A., Schwenk, D., Atkinson, D., Authur, R., et al. (2024). Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research. ACL 2024. https://arxiv.org/abs/2402.00159
- Allen Institute for AI. (2023). Ai2 Dolma: 3 trillion token open corpus for language model pretraining. https://allenai.org/blog/dolma-3-trillion-tokens-open-llm-corpus-9a0ff4b8da64
- AllenAI. (2026). allenai/dolma Dataset Card. https://huggingface.co/datasets/allenai/dolma
- AllenAI. (2026). Dolma Dataset and Toolkit Repository. https://github.com/allenai/dolma
- AllenAI. (2026). Dolma Toolkit Documentation. https://github.com/allenai/dolma/blob/main/docs/README.md
- Groeneveld, D., Beltagy, I., Walsh, P., Bhagia, A., Kinney, R., Tafjord, O., et al. (2024). OLMo: Accelerating the Science of Language Models. https://arxiv.org/abs/2402.00838
