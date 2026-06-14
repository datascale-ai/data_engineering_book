# Project 11: Mini-DeepSeek Pre-Training Reproduction

<div class="chapter-authors">Ke Wang</div>

## Abstract

This project builds a reproducible data engineering case study around "Mini-DeepSeek Pre-Training Reproduction," with an emphasis on business objectives, data boundaries, architectural decisions, core implementation, acceptance criteria, and risk controls. The chapter consolidates installation commands and script details into an engineering retrospective perspective, highlighting the relationships among sample schemas, data flows, failure modes, and deliverables, helping readers translate the methods presented earlier into auditable and extensible project assets.

## Keywords

Mini-DeepSeek; project practice; reproducible data engineering; data pipeline; acceptance criteria

## Project Objectives and Reader Outcomes

This project uses "Mini-DeepSeek Pre-Training Reproduction" as its core case study, with the goal of reproducing the key engineering stages of an open-source LLM pre-training data recipe using small-scale resources. Upon completing this chapter, readers should be able to identify the critical data objects in this scenario, decompose the engineering pipeline, set acceptance criteria, and transfer the case methodology to comparable data engineering tasks.

## Scenario Constraints and Data Boundaries

This project is positioned as a reduced-scale recipe validation exercise; it does not aim for full large-model scale or publicly reported SOTA metrics. These boundaries make the case reproducible and auditable. When data scale, data sources, access permissions, or deployment environments change, sampling strategies, quality thresholds, runtime costs, and compliance requirements must be re-evaluated.

## Architectural Decisions

This project follows an architectural path of "corpus mixing, tokenization, training-sample packing, training smoke test, metric logging, and cost analysis." This decision prioritizes input/output contracts, version traceability, anomaly localizability, and result verifiability over compressing all logic into a single one-shot script execution.

## Sample Schema / Data Flow

The core data flow can be summarized as:

```text
Candidate corpus -> Recipe sampling -> Tokenizer processing -> Packed dataset -> Training smoke test -> Loss and sample quality report
```

The sample schema should retain at minimum the fields `id`, `source`, `content_or_payload`, `metadata`, `quality_signals`, `split_or_stage`, and `audit_trace`; specific fields are further refined by the data types, downstream tasks, and acceptance methods of this project.

## Core Implementation Fragments

The body of the chapter retains only the key implementation fragments that illustrate design trade-offs. Complete scripts, lengthy configurations, execution logs, and large files should be placed in the companion repository or appendix; code presentation focuses on input/output contracts, quality thresholds, exception handling, and acceptance interfaces.

## Experiment and Acceptance Metrics

Acceptance metrics include token distribution, corpus-mix deviation, packing efficiency, training loss trend, throughput, GPU memory/cost, and failed-sample review. If the project enters production, a course, or a public reproduction experiment environment, the version number, dependency environment, random seed, sample spot-check results, and failed-sample retrospective records should also be logged.

*Table P11-1: Mini-DeepSeek Pre-Training Reproduction Publication Acceptance Table*

| Acceptance Dimension | Metric / Evidence | Publication Review Criterion |
| --- | --- | --- |
| Recipe reproduction | Corpus-mix deviation, cross-source deduplication records, and tokenizer training logs | A reduced-scale experiment must state the scale difference from the original recipe and the boundaries of non-comparability |
| Training smoke test | Packing efficiency, loss trend, throughput, and GPU memory/cost records | Report retains random seed, environment, sample scale, and failed-sample review conclusions |
| Data compliance | Data-source licenses, contamination checks, and sample deletion mechanisms | External corpora must have their origin and redistribution rights confirmed before entering public deliverables |

## Cost, Risk, and Compliance Boundaries

Costs arise primarily from training compute and data processing. Risks center on recipe misinterpretation, sample contamination, tokenizer inconsistency, and extrapolating small-scale conclusions. When external data, personal information, copyrighted content, or third-party services are involved, source documentation, permission status, anonymization strategies, call records, and manual review records should be retained.

## Common Failure Modes

Common failures include input distribution drift, missing schema fields, quality thresholds that are too loose or too tight, insufficient evaluation-sample coverage, unstable model calls, and non-traceable results. When diagnosing, prioritize locating data boundaries and intermediate artifacts before examining the model, toolchain, and deployment environment.

## Reproducible Resource Description

Reproduction materials should include data source descriptions, minimal samples, configuration files, run commands, metric scripts, inspection reports, and an artifact directory. The body of the chapter retains necessary fragments; complete notebooks, long scripts, and large files are maintained separately as companion resources.

## Background and Objectives

In pre-training data engineering, "Scaling Laws" (Kaplan et al. 2020) apply not only to model parameters but equally to the experimentation and validation of data recipes. In the earlier Project 1 (Mini-C4), we completed an end-to-end cleaning pipeline for a single-source corpus. However, real industrial-scale large models—such as DeepSeek-V3 (Liu et al. 2024)—are never trained on a single corpus; they are trained on a precise mixture of web pages, code, mathematics, academic papers, and other data sources.

Why do we need a Mini pre-training pipeline?

1. **Low-cost validation**: Experimenting on the full 14.8T tokens of real data is prohibitively expensive. Through proportional scaling, we can rapidly validate multi-source mixing strategies at the 1B-token scale.
2. **Exposing inter-source interactions**: Engineering problems such as cross-source deduplication and the effect of data-mix adjustments on the tokenizer vocabulary distribution only surface in a multi-source mixing environment.
3. **Smooth scaling curve**: A validated 1B-token data pipeline requires only replacing the underlying data-source cluster and compute nodes to scale out horizontally to 7B, 14B, or even 70B tokens.

This project aims to fully replicate the data recipe of DeepSeek-V3 using approximately 1B tokens—an amount that a single node with 8× 4090/A100 GPUs can process in tens of hours. Upon completing this project, readers will have a multi-source mixing sampler, a cross-source deduplication engine, and tokenizer training code targeting a 150K super-vocabulary, all meeting industrial-grade standards, providing a solid foundation for large-scale pre-training.

## Architecture Design

To achieve the objectives above, we designed a data pipeline consisting of four core components. The overall architecture is shown in Figure P11-1.

![Mini-DeepSeek Data Pipeline](../../images/part14/p11_mini_deepseek_arch_en.svg)
*Figure P11-1: Mini-DeepSeek Multi-Source Pre-Training Data Pipeline Architecture*

The four core components of the pipeline are:

1. **Multi-source Sampler**: Responsible for fetching multiple open-source datasets from Hugging Face (e.g., FineWeb-Edu, The Stack v2) and performing precise sampling according to the per-domain proportions disclosed in the DeepSeek-V3 report.
2. **Cross-source MinHash Deduplication Engine**: When data sources include not only ordinary web pages but also GitHub code and arXiv papers, implicit overlap may exist between sources. This component implements efficient deduplication across different data sources using the MinHash LSH algorithm (Broder 1997).
3. **Tokenizer Trainer**: Using the BPE algorithm (Sennrich et al. 2016), this component trains and constructs a super-vocabulary of 150K entries on the mixed multilingual and multi-code-domain corpus, ensuring efficient compression of both Chinese and English text as well as specialized code.
4. **Pack & Shuffle**: After tokenization, variable-length sequences are efficiently "packed" into fixed-length training sequences, globally shuffled, and output as `.arrow` format files suitable for large-scale distributed training.

Table P11-2 maps each architectural component to its code entry point, stage artifact, and review fields. Project chapters need to retain tables of this kind because they connect "the engineering narrative visible to the reader" with the scripts in `code/zh/project_11_mini_deepseek`, preventing the chapter from remaining at the level of conceptual introduction alone.

| Stage | Code Entry Point | Primary Input | Primary Output | Review Fields |
| --- | --- | --- | --- | --- |
| Multi-source sampling | `mix_sampler.py` | `RECIPE`, target document count, Hugging Face data sources | `./data/mixed_1b_raw` | `source`, sample count, recipe weight deviation |
| Cross-source deduplication | `cross_source_dedup.py` | `mixed_1b_raw` | `./data/mixed_1b_dedup` | MinHash parameters, duplicate sample count, retention ratio |
| Tokenizer training | `train_tokenizer.py` | `mixed_1b_dedup` | `mini_deepseek_tokenizer.json` | vocab size, special tokens, training sample ratio |
| Pack & Shuffle | `pack_shuffle.py` | Deduplicated corpus, tokenizer | `./data/mixed_1b_final_packed` | `SEQ_LEN`, packing efficiency, shuffle seed |
| End-to-end run | `run_pipeline.sh` | Stage scripts and local environment | Complete data directory | Logs, failed stages, artifact integrity |
| Unit tests | `tests/test_pipeline.py` | Recipe, MinHash, packing constants | Test report | Weights sum to 1, MinHash similarity, `SEQ_LEN` |

*Table P11-2: Mini-DeepSeek Data Pipeline Stage Artifacts and Code Entry Points*

The most easily overlooked column in Table P11-2 is "Review Fields." For example, `mixed_1b_raw` is simply a Hugging Face Dataset directory on its own and says nothing about whether the recipe is correct; one must additionally verify that the sample count for each `source` is consistent with the target weights. Similarly, `mixed_1b_dedup` cannot be validated merely by checking whether the directory exists—the duplicate sample ratio and threshold must also be recorded. For the tokenizer, the existence of the file does not indicate training success; special tokens, vocabulary size, Chinese/code compression ratio, and rare-character coverage must also be checked.

## Step-by-Step Implementation

### Step 1: Multi-Source Mixed Extraction and Proportioning

According to the DeepSeek-V3 report, we need to fuse multiple data sources. In this implementation, we select open-source alternative datasets:

- English web pages: FineWeb-Edu (Penedo et al. 2024)
- Chinese web pages: Wudao or open-source Chinese–English mixed data
- Code: The Stack v2 (Lozhkov et al. 2024)
- Mathematics: OpenWebMath (Paster et al. 2023)
- Academic: arXiv

We write the `mix_sampler.py` script to sample at the configured proportions.

```python
from datasets import load_dataset, concatenate_datasets

RECIPE = {
    "HuggingFaceFW/fineweb-edu": {"weight": 0.40},
    "bigcode/the-stack-v2": {"weight": 0.25},
    "open-web-math/open-web-math": {"weight": 0.15},
    "togethercomputer/RedPajama-Data-1T": {"name": "arxiv", "weight": 0.10},
    "m-a-p/WanJuan-1.0-Text": {"weight": 0.10},
}

def sample_multi_source(recipe, target_docs):
    shards = []
    for repo_id, cfg in recipe.items():
        n = int(target_docs * cfg["weight"])
        stream = load_dataset(repo_id, cfg.get("name"), split="train", streaming=True)
        rows = [normalize_text(item, source=repo_id) for item in take(stream, n)]
        shards.append(rows_to_dataset(rows))
    return concatenate_datasets(shards)

mixed = sample_multi_source(RECIPE, target_docs=500_000)
mixed.save_to_disk("./data/mixed_1b_raw")
```

### Step 2: Cross-Source MinHash LSH Deduplication

After multi-source mixing, the greatest hidden risk is duplicates between different sources (for example, code snippets in The Stack v2 duplicating code segments in arXiv papers). In Project 1 (Mini-C4), we performed MinHash deduplication only within a single source; here we need global deduplication.

```python
from datasketch import MinHash, MinHashLSH

def get_minhash(text, num_perm=128):
    sig = MinHash(num_perm=num_perm)
    for token in char_ngrams(text, n=5):
        sig.update(token.encode("utf-8"))
    return sig

def cross_source_dedup(dataset, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    keep, duplicates = [], 0
    with lsh.insertion_session() as session:
        for idx, row in enumerate(dataset):
            sig = get_minhash(row["text"])
            if lsh.query(sig):
                duplicates += 1
                continue
            session.insert(str(idx), sig)
            keep.append(idx)
    return dataset.select(keep), duplicates

unique, dup_count = cross_source_dedup(load_stage("mixed_1b_raw"))
unique.save_to_disk("./data/mixed_1b_dedup")
```

### Step 3: Training a 150K Super-Vocabulary Tokenizer

DeepSeek-V3 (Liu et al. 2024) employs a super-vocabulary of approximately 150K entries (a substantial increase over Llama-2's 32K), which makes it highly efficient at processing Chinese text and code. In this step, we train a BPE tokenizer on the mixed and deduplicated data.

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers

def train_large_tokenizer(dataset, vocab_size=150_000):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    sample = dataset.select(range(0, len(dataset), 10))
    tokenizer.train_from_iterator(batch_text(sample), trainer=trainer)
    tokenizer.save("./data/mini_deepseek_tokenizer.json")
    return tokenizer

train_large_tokenizer(load_stage("mixed_1b_dedup"))
```

### Step 4: Pack & Shuffle and `.arrow` Shard Output

To avoid having the GPU handle large amounts of padding during training, we concatenate variable-length token sequences into contiguous segments of length `4096` or `8192` (packing), inserting special separator tokens.

```python
from tokenizers import Tokenizer

SEQ_LEN = 4096

def pack_and_shuffle(dataset, tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eot = tokenizer.token_to_id("<|endoftext|>")

    def encode_batch(batch):
        stream = []
        for text in batch["text"]:
            stream.extend(tokenizer.encode(text).ids + [eot])
        usable = (len(stream) // SEQ_LEN) * SEQ_LEN
        blocks = [stream[i:i + SEQ_LEN] for i in range(0, usable, SEQ_LEN)]
        return {"input_ids": blocks}

    packed = dataset.map(encode_batch, batched=True, remove_columns=dataset.column_names)
    return packed.shuffle(seed=42)

packed = pack_and_shuffle(load_stage("mixed_1b_dedup"), "./data/mini_deepseek_tokenizer.json")
packed.save_to_disk("./data/mixed_1b_final_packed")
```

## Engineering Execution and Minimal Reproduction Path

The minimal entry point for running this project is `run_pipeline.sh`. The script chains together four stages: multi-source sampling, cross-source deduplication, tokenizer training, and packing. Its value is not merely "saving the manual execution of four commands"; it fixes the stage order, artifact paths, and failure locations. In pre-training data engineering, an incorrect stage order directly alters the data distribution. For example, if the tokenizer is trained before cross-source deduplication, the tokenizer will see duplicate samples that should have been removed; if packing precedes shuffling, subsequent recipe adjustments become difficult to trace.

Listing P11-1 gives the minimal entry point for this project. Formal reproduction experiments should record Python, datasets, tokenizers, datasketch, disk paths, and random seeds before running.

```bash
cd code/zh/project_11_mini_deepseek
bash run_pipeline.sh
```

This command sequentially generates `mixed_1b_raw`, `mixed_1b_dedup`, `mini_deepseek_tokenizer.json`, and `mixed_1b_final_packed`. If a stage fails, it is not recommended to delete the entire `data/` directory and rerun from scratch; the safer approach is to first confirm whether the failed stage and its upstream artifacts are intact, then clean only the affected stage directory. For teaching reproductions, `target_docs` can be reduced from `500000` to a smaller scale to validate contracts and tests before scaling up the data volume.

Table P11-3 lists the minimal audit information that should be recorded before and after a run.

| Category | Record Item | Purpose |
| --- | --- | --- |
| Data version | Data source repo id, split, config name, sampling time | Explains sample distribution changes |
| Recipe parameters | `RECIPE` weights, target document count | Determines whether the reduced-scale recipe is satisfied |
| Deduplication parameters | n-gram length, `num_perm`, LSH threshold | Reproduces duplicate rate and false-deletion risk |
| Tokenizer parameters | vocab size, normalizer, pre-tokenizer, special tokens | Reproduces compression ratio and compatibility |
| Packing parameters | `SEQ_LEN=4096`, shuffle seed, batch size | Reproduces training sample boundaries |
| Environment information | Python, datasets, tokenizers, datasketch versions | Diagnoses artifact differences |
| Run results | Sample count, token count, directory size, failure logs | Determines deliverability |

*Table P11-3: Mini-DeepSeek Minimal Reproduction Experiment Record Items*

## Data Quality and Recipe Acceptance

The acceptance focus for Mini-DeepSeek is not the final model score but whether the recipe is interpretable, verifiable, and extensible. A common pitfall is reporting only "1B tokens were ultimately obtained" without stating which domains those tokens came from, how many cross-source duplicates were removed, or whether the tokenizer is biased toward a particular language or code domain. For pre-training data, quantity is merely the outcome; the recipe is the core.

Table P11-4 gives the recipe-level acceptance criteria.

| Acceptance Item | Recommended Check | Non-conformance Indicator |
| --- | --- | --- |
| Weight consistency | Deviation between each data source's sample count and its `RECIPE` weight | A source is over-sampled or underestimated due to streaming interruption |
| Field completeness | Each sample retains `text` and `source` | Downstream cannot compute source statistics or perform sample review |
| Cross-source duplicates | Record duplicate count and duplication rate | Implicit duplicates among code, papers, and web pages are not cleaned |
| Text anomalies | Ratio of empty text, extremely short text, garbled content, and binary residues | Tokenizer learns meaningless tokens |
| Tokenizer coverage | Compression ratio for Chinese, English, code, and math samples | Tokens/char is noticeably abnormal for some text category |
| Packing integrity | All `input_ids` have length `4096` | Padding or length inconsistency occurs during training |
| Randomness | Fixed shuffle seed and sampling strategy | Sample order or distribution across multiple runs is inexplicable |

*Table P11-4: Mini-DeepSeek Recipe-Level Acceptance Checklist*

In practice, begin by sampling 100 records from each source for manual inspection to verify that the text type matches expectations; then spot-check near-duplicate pairs from the MinHash-deleted samples to assess whether the threshold is too strict. If the duplication rate is anomalously high, it may indicate genuine overlap between data sources, or it may indicate that 5-grams are overly sensitive to short texts. If the duplication rate is anomalously low, check whether the text field was selected correctly—some datasets use `content` rather than `text`.

## Fine-Grained Inspection of Tokenizer and Packing

The tokenizer training stage most easily produces the problem of "file creation succeeded but quality is unknown." `train_tokenizer.py` uses BPE, NFKC normalization, a ByteLevel pre-tokenizer, and a 150K vocabulary. This configuration suits mixed Chinese, English, code, and math corpora, but it introduces two risks. First, with a large vocabulary, a small-scale training sample may not provide sufficient coverage of long-tail tokens. Second, code and mathematical symbols may occupy a disproportionate share of the vocabulary space, affecting the compression ratio for ordinary text.

It is recommended to compute the metrics in Table P11-5 after training is complete.

| Metric | Computation Method | Interpretation |
| --- | --- | --- |
| Chinese tokens/char | Token count of Chinese web page samples divided by character count | Assesses Chinese compression efficiency |
| English tokens/word | Token count of English web page samples divided by word count | Assesses whether ordinary English segmentation is abnormal |
| Code tokens/char | Token count of The Stack v2 samples divided by character count | Assesses code symbol coverage |
| Math formula fragmentation rate | Ratio of short tokens in math samples | Assesses whether formulas and LaTeX are over-fragmented |
| `<|endoftext|>` presence | Whether separator tokens exist in packed data | Assesses whether document boundaries are preserved |
| OOV behavior | `<|unk|>` usage rate | Assesses whether ByteLevel coverage is normal |

*Table P11-5: Tokenizer and Packing Quality Metrics*

The packing stage also requires inspection. `pack_shuffle.py` concatenates sample tokens and truncates to an integer multiple of `SEQ_LEN`, outputting fixed-length `input_ids`. This improves training throughput but means original document boundaries are no longer directly visible. Therefore, the insertion and counting of `<|endoftext|>` is critically important. If the separator is forgotten, the model learns adjacent documents as continuous text; if separators are overly dense, short documents will dominate the context structure.

## Test Coverage and Code Isolation

`tests/test_pipeline.py` already covers several minimal contracts: `RECIPE` weights sum to 1, critical data sources exist, MinHash objects can be created, identical texts have a Jaccard similarity of 1.0, and `SEQ_LEN` equals 4096. These tests are not intended to replace comprehensive data acceptance; rather, they prevent the example code in the project chapter from losing its basic contracts after refactoring.

Table P11-6 maps tests to gaps.

| Test Item | Covered Content | Still Requires Manual or Integration Acceptance |
| --- | --- | --- |
| `test_recipe_weights` | Recipe weights sum to 1 | Actual sample count per source |
| `test_sampler_keys` | Critical data sources exist | Data source licenses, field names, and accessibility |
| `test_minhash_creation` | MinHash object is usable | Large-scale LSH memory footprint |
| `test_minhash_similarity` | Identical text similarity | Near-duplicate false-deletion/missed-deletion boundary |
| `test_tokenizer_pack` | `SEQ_LEN` constant | Length of each packed sample |
| `test_end_to_end_mock` | Test entry point exists | Real end-to-end small-sample run |

*Table P11-6: Mini-DeepSeek Test Coverage and Acceptance Gaps*

Published manuscripts should clearly distinguish "teaching example code" from "production-ready code." The code in this chapter illustrates the basic organization of multi-source sampling, MinHash, BPE, and packing, but real large-scale pre-training additionally requires distributed execution, data-source failure retries, download caching, contamination detection, sensitive-content filtering, license whitelisting, and training framework integration.

## Common Failures and Diagnostic Paths

If the sampling stage reports an error, first check the data source name, config name, and streaming split. Field names are not consistent across different Hugging Face datasets—some use `text`, some use `content`, and others require additional authentication or configuration. If a data source returns very few samples, training should not proceed immediately; instead, the gap for that source should be recorded in the report and the actual recipe recomputed.

If the deduplication stage has excessive memory usage, the current approach of holding the entire MinHash LSH in memory is approaching single-node limits. First reduce the sample volume to validate the pipeline, then migrate to Spark, Ray, or an external key-value store. If the sample count drops sharply after deduplication, spot-check duplicate pairs to determine whether over-matching is caused by short texts, template texts, or license headers.

If tokenizer training takes too long, first check the training sub-sample ratio. The current implementation samples every tenth record using `ds.select(range(0, len(ds), 10))` to train the tokenizer—a pedagogical compromise. For larger data volumes, stratified sampling by source can be used to ensure coverage of code, math, Chinese, and English. If the resulting `mini_deepseek_tokenizer.json` cannot be loaded by `pack_shuffle.py`, check whether special tokens were written and whether the file was corrupted by an interrupted write.

If the training smoke test shows an anomalous loss after packing, the diagnostic sequence should be: first confirm that all `input_ids` have length `4096`; next, sample-decode packed records to check document boundaries and anomalous characters; then verify the shuffle seed and sample order; only then suspect training hyperparameters. Pre-training data problems frequently masquerade as training problems; diagnosis should return to the data artifacts first.

## Training Smoke Test and Metric Recording

The goal of P11 is not to report a complete large-model training result but to demonstrate that data artifacts can enter the training pipeline. Therefore, the training smoke test should remain small-scale, short-duration, and reproducible. A qualified smoke test may run only a few hundred to a few thousand steps, but it must answer three questions: whether the data can be stably read by the training framework, whether the loss exhibits a reasonable downward trend, and whether throughput, GPU memory, and sample decoding meet expectations.

Table P11-7 provides a training smoke test record template.

| Record Item | Example | Review Significance |
| --- | --- | --- |
| Data directory | `./data/mixed_1b_final_packed` | Confirms that training reads from the final packed data |
| Sample length | `4096` | Confirms that the training context is consistent with the packing configuration |
| Batch configuration | Global batch, micro batch, gradient accumulation | Explains throughput and GPU memory differences |
| Model scale | e.g., 100M, 300M, 1B parameter teaching model | Makes explicit that no comparison is made with the full DeepSeek-V3 |
| Tokenizer | `mini_deepseek_tokenizer.json` | Confirms that training and packing use the same tokenizer |
| Step range | e.g., steps 0–1000 | States smoke test duration |
| Loss curve | Initial and final loss, anomalous spikes | Determines whether the data flow is clearly abnormal |
| Tokens/s | Tokens per second | Estimates cost of subsequent scale-up |
| Sample decoding | Random decode of 10 packed samples | Checks for garbled content, boundary issues, and repetition |

*Table P11-7: Mini-DeepSeek Training Smoke Test Record Template*

The smoke test report should not simply state "training runs successfully." More useful content includes: read throughput, loss over the first several steps, decoded results from several packed samples, and a record of how failed samples were handled. If the loss fails to decrease over an extended period, this may indicate a model configuration problem or the presence of large amounts of duplicated, garbled, or meaningless tokens in the data. If throughput is below expectation, the `.arrow` shards may be too small, data loading workers may be insufficient, or disk I/O may be the bottleneck.

## Data Contamination and Benchmark Leakage Inspection

Pre-training reproduction projects must pay attention to benchmark contamination. Even in a pedagogical reduced-scale context, the contamination inspection methodology should be documented. Multi-source corpora may contain questions, solutions, and answers from evaluation sets such as GSM8K, MATH, HumanEval, MMLU, or others. If such content enters the pre-training data, subsequent benchmark scores will be inflated.

Table P11-8 presents a minimal contamination inspection plan.

| Inspection Target | Method | Handling |
| --- | --- | --- |
| Exact duplicates | Hash or normalized string matching against benchmark prompts | Directly remove matched samples |
| Near-duplicates | MinHash or embedding retrieval on prompts, answers, and solutions | Manual spot-check followed by deletion |
| Code evaluation | Check for HumanEval function names, docstrings, and canonical solutions | Remove complete problems and answers |
| Math evaluation | Check for problem stems, answer choices, and final-answer patterns | Remove solutions and answer leakage |
| Forum reposts | Check for benchmark problem reposts on web pages or blogs | Remove or flag as contamination risk |
| Training report | Save match counts, deletion strategy, and sample ids | Supports subsequent benchmark claims |

*Table P11-8: Pre-Training Data Contamination Inspection Plan*

Contamination inspection should be completed before packing, because once the data enters a packed dataset, original document boundaries and source fields become harder to trace. If a teaching project does not implement complete contamination filtering, the report should explicitly state "not used for public benchmark claims" to prevent readers from mistaking smoke-test scores for publishable results.

## Deliverable Directory and Release Package Structure

A deliverable P11 project should contain not only the final data directory but also the recipe, logs, tests, and audit materials. Table P11-9 gives the recommended directory structure.

| Path | Content | Included in Public Release |
| --- | --- | --- |
| `data/mixed_1b_raw/` | Raw mixed data sampled according to the recipe | Depends on data licenses |
| `data/mixed_1b_dedup/` | Training candidate corpus after cross-source deduplication | Depends on data licenses |
| `data/mini_deepseek_tokenizer.json` | BPE tokenizer | May be made public |
| `data/mixed_1b_final_packed/` | Packed `.arrow` data | Depends on data licenses |
| `reports/source_mix.json` | Sample count and proportion per source | May be made public |
| `reports/dedup_report.json` | Duplicate count, threshold, spot-checked samples | Anonymized version may be made public |
| `reports/tokenizer_eval.md` | Compression ratio, anomalous tokens, sample decodes | May be made public |
| `reports/smoke_train.md` | Training smoke test metrics | May be made public |
| `tests/` | Unit tests and small-sample tests | May be made public |
| `LICENSES.md` | Data source license documentation | Must be made public |

*Table P11-9: Mini-DeepSeek Project Deliverable Directory*

If data sources do not permit redistribution, the recipe, scripts, tokenizer, and report templates may still be published, but raw or packed samples cannot be released directly. In this case, a reproducibility guide should be provided so that readers with the requisite authorization can rebuild the data locally.

## Scaling from 1B Tokens to Larger Scales

The reduced-scale pipeline of P11 helps readers understand the recipe, but scaling to 10B, 70B, or larger requires systematic refactoring. Table P11-10 lists the refactoring checklist for moving from the teaching implementation to a production-scale implementation.

| Module | Teaching Implementation | Production-Scale Refactoring |
| --- | --- | --- |
| Data ingestion | Hugging Face streaming + local Dataset | Connect to object storage, data lake, or distributed cache |
| Recipe sampling | Single-pass proportional sampling | Support epoch-level dynamic mixing ratios and curriculum |
| Deduplication | Single-node MinHash LSH | Distributed MinHash, SimHash, or embedding near-duplicate system |
| Tokenizer | Single-node sampled training | Stratified sampling, version freezing, and compatibility regression |
| Packing | Single-node map + shuffle | Distributed packing, fixed shard size, training framework prefetch |
| Auditing | Manual reports | Metadata service, lineage tracking, deletion-request replay |
| Testing | Unit tests + mock e2e | Small-sample real e2e, data-diff regression, contamination scan |

*Table P11-10: Mini-DeepSeek Refactoring Checklist from Teaching Implementation to Production Scale*

The most important point when scaling is not to mistake "the script runs" for "the system is scalable." Large-scale pre-training data systems must handle failure retries, checkpoint-based resumption, version freezing, sample deletion, license changes, in-training data mixing strategy adjustments, and multi-team collaboration. The value of P11 lies in presenting the minimal form of these problems rather than claiming to replace a complete industrial system.

## Data Dashboard and Continuous Monitoring

Once pre-training data engineering enters continuous iteration, one-off run logs are insufficient. Data sources update, licenses change, field schemas are adjusted, and Hugging Face datasets may change their splits or sample content due to maintenance. Therefore, P11 needs a lightweight data dashboard for comparing differences across different run batches.

Table P11-11 gives the recommended dashboard metrics.

| Dashboard Metric | Statistics Source | Observation Purpose |
| --- | --- | --- |
| Source sample count | `mixed_1b_raw` | Check for recipe proportion drift |
| Source token count | Post-tokenizer statistics | Check for consistency between sample count and token count |
| Deduplication rate | `mixed_1b_raw` vs. `mixed_1b_dedup` | Detect duplication anomalies or threshold issues |
| Text length distribution | Raw text field | Detect overly short, overly long, or templated samples |
| Language proportion | Language identification script | Check proportions of Chinese, English, and other languages |
| Code proportion | Source and code feature detection | Check whether code sources are over-sampled |
| Math sample proportion | Source and LaTeX/formula detection | Check math data coverage |
| Tokenizer compression ratio | Tokenizer evaluation script | Detect vocabulary quality changes |
| Packed shard size | `mixed_1b_final_packed` | Check training read balance |
| Contamination match count | Contamination scan | Record benchmark leakage risk |

*Table P11-11: Mini-DeepSeek Data Dashboard Metrics*

The core value of the dashboard is batch comparison. For example, if The Stack v2 sample count is unchanged in a given run but the token count rises significantly, it may indicate that code samples have become longer or that filtering conditions have changed. If the deduplication rate for OpenWebMath suddenly increases in a given run, it may indicate that templated content in math web pages has increased. Without a batch dashboard, these changes often go undetected until a training loss or benchmark anomaly is observed.

## Deletion Requests and Sample Withdrawal Mechanism

Multi-source pre-training data must have a sample withdrawal capability. Even in a pedagogical reproduction context, the following should be documented: if a data source, author, or content owner requests the deletion of samples, how does the system locate, remove, and rebuild downstream artifacts? The current P11 implementation retains the `source` field, but original document boundaries are weakened in packed data, so deletion requests are best handled before packing.

Table P11-12 gives the processing path for deletion requests.

| Step | Action | Artifact |
| --- | --- | --- |
| Receive request | Record the requester, URL, sample characteristics, and timestamp | Deletion request ticket |
| Locate samples | Search `raw`/`dedup` data by URL, hash, text fragment, or source | Affected sample ids |
| Delete upstream | Remove matched samples from `mixed_1b_raw` or `mixed_1b_dedup` | Revised dataset |
| Assess tokenizer rebuild | If the deletion proportion is large, evaluate whether to retrain the tokenizer | Tokenizer impact report |
| Rebuild packed data | Re-run packing and shuffle | Revised packed dataset |
| Update report | Update recipe proportions, token counts, and deletion notes | Release note |

*Table P11-12: Pre-Training Sample Deletion Request Processing Path*

Real production systems typically store document-level hashes, URLs, source, license, and a packed-shard reverse index. Teaching implementations do not necessarily require a complete index, but readers should understand that the earlier lineage is discarded, the higher the subsequent withdrawal cost. Retaining this point in the published text prevents readers from mistakenly believing that `.arrow` training data can circulate independently of its origin.

## Domain Transfer: From General Recipe to Domain-Specific Models

The Mini-DeepSeek recipe is dominated by web pages, code, mathematics, academic papers, and Chinese text. Transferring to legal, medical, financial, or industrial domains requires more than simply adding a domain-specific data source to `RECIPE`. Domain data typically carries stronger requirements around permissions, privacy, terminology, and temporal relevance, necessitating independently designed recipes and acceptance criteria.

Table P11-13 gives adjustment directions for domain transfer.

| Domain | Data to Add | Additional Risks | Acceptance Focus |
| --- | --- | --- | --- |
| Legal | Regulations, case law, contracts, compliance Q&A | Regional and temporal differences | Statute version, citation accuracy |
| Medical | Guidelines, papers, drug package inserts, case templates | Privacy and high-risk recommendations | Anonymization, source grade, expert review |
| Financial | Research reports, announcements, financial statements, market rules | Temporal and investment-advice risks | Date, market, calibration consistency |
| Industrial | Equipment manuals, fault records, process documents | Internal confidentiality and terminology ambiguity | Permissions, terminology glossary, fault classification |
| Education | Textbooks, exercises, solutions, course notes | Copyright and answer leakage | Copyright licenses, question-bank contamination |

*Table P11-13: Domain Transfer Considerations for the Mini-DeepSeek Recipe*

When transferring to a domain, it is recommended to retain the general corpus as a foundation and gradually increase the domain corpus proportion. If the domain data proportion is raised too rapidly, the model may acquire domain terminology capability while losing general language and coding capability. A safer approach is to design a multi-round curriculum: maintain a dominant share of general corpus in the early stages, then increase domain- and task-relevant data proportions in the mid-to-late stages, monitored by domain validation sets.

## Relationship with P01 Mini-C4

P01 focuses on single-source web page cleaning; P11 focuses on a multi-source pre-training recipe. These are not redundant; they represent an upgrade from "cleaning one type of corpus" to "organizing multiple types of corpora." Table P11-14 summarizes the differences.

| Dimension | P01 Mini-C4 | P11 Mini-DeepSeek |
| --- | --- | --- |
| Data sources | Single source or a small number of web sources | Web pages, code, mathematics, academic papers, Chinese text |
| Core problem | Cleaning quality, denoising, basic filtering | Recipe proportions, cross-source deduplication, tokenizer, and packing |
| Deduplication scope | Near-duplicates within a single source | Cross-source near-duplicates |
| Tokenizer | Can reuse an existing tokenizer | Train a 150K super-vocabulary |
| Training samples | Cleaned documents | Fixed-length packed token blocks |
| Acceptance focus | Text quality and cleaning rules | Recipe, compression ratio, contamination, smoke test |
| Scaling direction | Larger web corpus | Larger multi-source pre-training system |

*Table P11-14: Differences Between P01 Mini-C4 and P11 Mini-DeepSeek*

Understanding this relationship helps readers connect the projects in Part 14. P01 is the starting point of data cleaning; P11 organizes multiple cleaned sources into a pre-training recipe. Without the quality filtering of P01, the multi-source recipe of P11 would absorb large amounts of noise; without the recipe organization of P11, the single-source cleaning of P01 is insufficient to support modern general-purpose model training.

## Results Presentation and Analysis

The pedagogical example configuration can be set up to run on a single node (e.g., 8× 4090 GPUs), recording end-to-end elapsed time to demonstrate the organization of a pipeline acceptance report.

When running at a sampling scale of `TARGET_TOTAL_DOCS = 500,000`, the MinHash deduplication rate should be recorded in the report; an indicative figure of approximately **4.2%** implicit duplicates filtered—concentrated primarily between code and academic sources—can be used. For formal delivery, actual run logs, random seeds, and data manifests are required.

The shuffled and packed `mixed_1b_final_packed` dataset should record storage size, number of `.arrow` shards, and token statistics. An indicative report may use approximately `5 GB` and approximately **1.05B tokens** to describe the output format, but the formal version must be generated jointly from script output, the sample manifest, and the random seed.

### Tokenizer Efficiency Validation

With the vocabulary expanded to 150K entries, sampling-based validation shows that this tokenizer achieves an average Chinese web page compression ratio (tokens/char) of **0.62**, a significant improvement over Llama-2's 1.1, substantially increasing downstream pre-training throughput efficiency.

## Cost and Optimization

For the pedagogical example at the 1B-token scale, resource consumption can be recorded under the following headings:

- **Storage**: Record the actual size of raw crawled data and the packed output; indicative figures are approximately 8 GB for raw data and approximately 5 GB after packing.
- **Compute and memory**: Record peak memory usage and elapsed time for streaming extraction, parallel map operations, and cross-source MinHash deduplication; indicative figures are a peak memory of approximately 32 GB and approximately 3 hours for MinHash deduplication. Formal delivery should be based on run logs.

**Optimization Notes**:
If horizontal scaling to 70B tokens is required, single-node Python in-memory processing will become the bottleneck. It is recommended to integrate a distributed engine such as Apache Spark (Zaharia et al. 2016) or Ray (Moritz et al. 2018). For the MinHash deduplication step, memory decoupling can be achieved by storing hash buckets in an external database such as Redis.

## Extended Considerations

Scaling the Mini-DeepSeek recipe to tens-of-billions-of-tokens requires particular attention to two points:

1. **Dynamic decay of mixing ratios (Curriculum)**: In the early stages of training, foundational knowledge (web pages and academic papers) should dominate; in the mid-to-late stages, the sampling weights of code and mathematics (OpenWebMath) should be increased. `mix_sampler.py` can be refactored into a streaming module that supports epoch-level dynamic loading.
2. **Comparison with the earlier project**: Compared to P01 (Mini-C4) in Part 14, this project no longer relies on simple filtering with a single quality threshold; instead, it uses cross-source fusion and a super-vocabulary design to demonstrate how modern industrial-scale models such as DeepSeek-V3 lay their multi-task foundations.

### Data Compliance and Open-Source License Notes

When performing multi-source mixing, the open-source licenses of the original data must be strictly observed:

- **FineWeb-Edu**: CC0 license (fully open).
- **The Stack v2**: SPDX whitelist license system; only code with redistribution-permitting licenses is used.
- **OpenWebMath**: ODC-By license.
- **arXiv**: The specific distribution license chosen by each paper's authors.
- **Project Gutenberg**: Public Domain.

*(Note: The complete 1B-token data sample has been processed in compliance with applicable licenses and may be uploaded to the HuggingFace Datasets repository `dataforge-mini-deepseek-1b` for direct use in downstream fine-tuning.)*

## Chapter Summary

This chapter used "Mini-DeepSeek Pre-Training Reproduction" as a case study to demonstrate the engineering organization required to reproduce the key engineering stages of an open-source LLM pre-training data recipe using small-scale resources. The primary value of the case lies in placing task definition, data boundaries, architectural decisions, sample schema, metric acceptance, and reproduction resources on a single traceable chain, so that the project becomes not merely a set of operational steps but an auditable case study.

The boundaries of this case must also be clearly preserved. It is positioned as a reduced-scale recipe validation and does not aim for full large-model scale or publicly reported SOTA metrics. In scenarios involving larger scale, higher risk, or stricter compliance requirements, data sources, permission status, manual review proportions, runtime costs, and failure rollback plans should be re-evaluated.

As part of Part 14, this chapter corresponds to the project-level validation of the methods presented earlier. Readers can combine this case with the data recipes of Part 13, the platform governance chapters in earlier sections, and the checklists in the appendices to form a complete loop from methodological understanding to engineering delivery.

## References

Broder A Z (1997) On the Resemblance and Containment of Documents. In: Proceedings of the Compression and Complexity of Sequences, pp 21–29.

Kaplan J, McCandlish S, Henighan T, Brown T B, Chess B, Child R, Gray S, Radford A, Wu J, Amodei D (2020) Scaling Laws for Neural Language Models. arXiv preprint arXiv:2001.08361.

Liu A, Feng B, Xue B, Wang B, Wu B, Lu C, Zhao C, Deng C, Zhang C, Ruan C, others (2024) DeepSeek-V3 Technical Report. arXiv preprint arXiv:2412.19437.

Lozhkov A, Ben Allal L, von Werra L, Wolf T (2024) StarCoder 2 and The Stack v2: The Next Generation (The Stack v2). arXiv preprint arXiv:2402.19173.

Moritz P, Nishihara R, Wang S, Tumanov A, Liaw R, Liang E, Elibol M, Yang Z, Paul W, Jordan M I, Stoica I (2018) Ray: A Distributed Framework for Emerging AI Applications. In: Proceedings of the 13th USENIX Symposium on Operating Systems Design and Implementation, pp 561–577.

Paster K, Santos M D, Azerbayev Z, Ba J (2023) OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text. arXiv preprint arXiv:2310.06786.

Penedo G, Kydlicek H, de Wiele T V, Lozhkov A, Mitchell M, Raffel C, von Werra L, Wolf T (2024) The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. arXiv preprint arXiv:2406.17557.

Sennrich R, Haddow B, Birch A (2016) Neural Machine Translation of Rare Words with Subword Units (BPE). In: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, pp 1715–1725.

Zaharia M, Xin R S, Wendell P, Das T, Armbrust M, Dave A, Meng X, Rosen J, Venkataraman S, Franklin M J, Ghodsi A, Gonzalez J, Shenker S, Stoica I (2016) Apache Spark: A Unified Engine for Big Data Processing. Communications of the ACM 59(11):56–65.
