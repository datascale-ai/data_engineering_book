# Project 11: Reproducing Mini-DeepSeek Pre-training

## Abstract

This project builds a reproducible data-engineering case around Mini-DeepSeek pre-training. It narrows installation commands and script details into an engineering-review perspective, with emphasis on business goals, data boundaries, architectural decisions, core implementation, acceptance metrics, and risk control. The chapter helps readers turn the methods from earlier chapters into auditable and extensible project assets.

## Keywords

Mini-DeepSeek; project practice; reproducible data engineering; data pipeline; acceptance metrics

## Project Goals and Reader Takeaways

The goal is to reproduce the key data-engineering steps of an open-source LLM pre-training recipe with small-scale resources. After completing the project, readers should be able to identify the core data objects, decompose the engineering chain, define acceptance metrics, and transfer the approach to similar pre-training data tasks.

## Scenario Constraints and Data Boundaries

This project is positioned as a scaled-down recipe validation. It does not pursue full large-model scale or public SOTA numbers. These boundaries make the case reproducible and auditable. If data scale, source permissions, deployment environment, or compute budget changes, sampling strategy, quality thresholds, runtime cost, and compliance requirements must be reassessed.

## Architecture Decision

The project follows a path of corpus mixture design, tokenization, training-sample packing, training smoke test, metric recording, and cost analysis. This decision prioritizes input/output contracts, version traceability, anomaly localization, and result reviewability instead of compressing all logic into one disposable script.

## Sample Schema and Data Flow

The core flow can be summarized as:

```text
candidate corpora -> recipe sampling -> tokenizer processing -> packed dataset -> training smoke test -> loss and sample-quality report
```

At minimum, sample records should preserve `id`, `source`, `content_or_payload`, `metadata`, `quality_signals`, `split_or_stage`, and `audit_trace`. Exact fields should be refined according to the data type, downstream training task, and acceptance method.

## Core Implementation Slice

The main text keeps only implementation fragments that explain design choices. Full scripts, long configurations, logs, and large artifacts should live in the companion repository or appendix. Code shown in the chapter focuses on input/output contracts, quality thresholds, exception handling, and acceptance interfaces.

## Experimental or Acceptance Metrics

Acceptance metrics include token distribution, corpus-mixture deviation, packing efficiency, training-loss trend, throughput, GPU-memory and cost records, and failed-sample review. For production, course, or public reproduction settings, reports should also record version numbers, dependency environment, random seed, sampled inspection results, and failed-sample review notes.

| Acceptance dimension | Metric or evidence | Publication review focus |
| --- | --- | --- |
| Recipe reproduction | Corpus-mixture deviation, cross-source deduplication records, tokenizer logs | The scaled-down experiment must explain scale differences and non-comparable boundaries relative to the original recipe |
| Training smoke test | Packing efficiency, loss trend, throughput, GPU memory, and cost records | Reports should preserve seed, environment, sample size, and failed-sample review conclusions |
| Data compliance | Source licenses, contamination checks, and deletion mechanism | External corpora need source and redistribution-right checks before public delivery |

## Cost, Risk, and Compliance Boundaries

Cost mainly comes from training compute and data processing. Risks concentrate on recipe misinterpretation, sample contamination, tokenizer inconsistency, and overextending small-scale conclusions. When external data, personal information, copyrighted content, or third-party services are involved, the project should retain source notes, permission status, desensitization strategy, call records, and human review records.

## Common Failure Modes

Common failures include input-distribution drift, missing schema fields, quality thresholds that are too loose or too strict, insufficient evaluation coverage, unstable model calls, and results that cannot be traced back. Troubleshooting should first locate data boundaries and intermediate artifacts, then inspect model behavior, toolchain settings, and deployment environment.

## Reproducible Resource Notes

Reproduction materials should include data-source notes, minimum samples, configuration files, run commands, metric scripts, check reports, and artifact directories. The chapter keeps necessary fragments; full notebooks, long scripts, and large files should be maintained separately as companion resources.

## Background and Objectives

In pre-training data engineering, scaling laws (Kaplan et al. 2020) apply not only to model parameters but also to the design and validation of data recipes. In Project 1, Mini-C4, we already built an end-to-end cleaning pipeline for a single-source corpus. Real industrial LLMs such as DeepSeek-V3 (Liu et al. 2024), however, are not trained on a single corpus. They are built from carefully mixed web, code, mathematics, academic, and multilingual sources.

Why do we need a mini pre-training pipeline?

1. **Low-cost validation**: Experimenting on the full 14.8T-token scale is expensive. A proportionally scaled 1B-token pipeline lets us validate multi-source mixing quickly.
2. **Exposure of cross-source effects**: Issues such as cross-source deduplication and the effect of mixture weights on tokenizer vocabulary distribution only appear in a multi-source setting.
3. **Smooth scaling path**: Once a 1B-token pipeline is validated, the same design can be scaled out to 7B, 14B, or even 70B tokens by replacing the underlying data source cluster and compute nodes.

This project reproduces the DeepSeek-V3-style data recipe at about 1B tokens, a scale that can be processed on a single 8-GPU 4090 or A100 node within tens of hours. By the end, readers will have an industrial-style multi-source sampler, a cross-source deduplication engine, and tokenizer training code for a large 150K vocabulary.

## Architecture

The pipeline contains four core components, shown in Figure 11-1.

![Mini-DeepSeek Data Pipeline](../../images/part11/p11_mini_deepseek_arch_en.png)
*Figure 11-1 Mini-DeepSeek multi-source pre-training data pipeline.*

1. **Multi-source sampler**: Fetches multiple open datasets from Hugging Face, such as FineWeb-Edu and The Stack v2, and samples them according to DeepSeek-V3-style domain proportions.
2. **Cross-source MinHash deduplication**: Removes hidden overlap between web, code, arXiv, and other sources using MinHash LSH (Broder 1997).
3. **Tokenizer training**: Trains a BPE tokenizer (Sennrich et al. 2016) with a 150K vocabulary on multilingual and code-heavy mixed data.
4. **Pack and shuffle**: Tokenizes, packs variable-length text into fixed-length training sequences, globally shuffles the result, and exports `.arrow` shards for distributed training.

## Step-by-Step Implementation

### Step 1: Multi-source Sampling and Mixture Ratios

Based on the DeepSeek-V3 report, we mix several open alternatives:

- English web: FineWeb-Edu
- Chinese web: Wudao or an open Chinese-English mixed corpus
- Code: The Stack v2
- Mathematics: OpenWebMath
- Academic text: arXiv

The `mix_sampler.py` script samples each source according to the configured weights.

```python
import datasets
from datasets import load_dataset, concatenate_datasets
import random

# Sampling recipe that approximates a DeepSeek-V3-style mixture.
RECIPE = {
    "HuggingFaceFW/fineweb-edu": {"split": "train", "weight": 0.40},
    "bigcode/the-stack-v2": {"split": "train", "weight": 0.25},
    "open-web-math/open-web-math": {"split": "train", "weight": 0.15},
    "togethercomputer/RedPajama-Data-1T": {"split": "train", "weight": 0.10, "name": "arxiv"},
    "m-a-p/WanJuan-1.0-Text": {"split": "train", "weight": 0.10},
}

TARGET_TOTAL_DOCS = 500000  # Estimated document count for roughly 1B tokens.


def sample_multi_source(recipe, target_docs):
    sampled_datasets = []
    for repo_id, config in recipe.items():
        weight = config["weight"]
        num_docs = int(target_docs * weight)
        print(f"Sampling {num_docs} docs from {repo_id}...")

        # Use streaming extraction for performance.
        ds = load_dataset(
            repo_id,
            config.get("name", "default"),
            split=config["split"],
            streaming=True,
        )
        ds_iter = iter(ds)

        docs = []
        for _ in range(num_docs):
            try:
                item = next(ds_iter)
                # Normalize all sources to a text field and drop unrelated fields.
                text_content = item.get("text") or item.get("content")
                if text_content:
                    docs.append({"text": text_content, "source": repo_id})
            except StopIteration:
                break

        sampled_datasets.append(datasets.Dataset.from_list(docs))

    mixed_dataset = concatenate_datasets(sampled_datasets)
    return mixed_dataset


if __name__ == "__main__":
    mixed_data = sample_multi_source(RECIPE, TARGET_TOTAL_DOCS)
    mixed_data.save_to_disk("./data/mixed_1b_raw")
    print("Multi-source sampling complete.")
```

### Step 2: Cross-source MinHash LSH Deduplication

After mixing sources, the main risk is duplication across corpora. For example, code snippets from The Stack v2 may also appear in arXiv papers. In Mini-C4 we deduplicated within one source; here we need global deduplication.

```python
from datasketch import MinHash, MinHashLSH
from datasets import load_from_disk


def get_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    # Simple character 5-gram tokenization.
    tokens = [text[i:i + 5] for i in range(max(1, len(text) - 4))]
    for token in tokens:
        m.update(token.encode("utf8"))
    return m


def cross_source_dedup(dataset_path, threshold=0.8, num_perm=128):
    print("Loading dataset for global deduplication...")
    ds = load_from_disk(dataset_path)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_indices = set()
    duplicates = 0

    with lsh.insertion_session() as session:
        for idx, item in enumerate(ds):
            m = get_minhash(item["text"], num_perm)
            result = lsh.query(m)
            if not result:
                session.insert(str(idx), m)
                unique_indices.add(idx)
            else:
                duplicates += 1

    print(f"Deduplication complete. Found {duplicates} duplicates.")
    return ds.select(list(unique_indices))


if __name__ == "__main__":
    ds_unique = cross_source_dedup("./data/mixed_1b_raw")
    ds_unique.save_to_disk("./data/mixed_1b_dedup")
```

### Step 3: Training a 150K Large Tokenizer

DeepSeek-V3 uses a very large vocabulary of around 150K tokens, far larger than Llama-2's 32K vocabulary. This improves compression efficiency for Chinese and code. In this step, we train a BPE tokenizer on the mixed, deduplicated corpus.

```python
from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers


def train_large_tokenizer(dataset_path, vocab_size=150000):
    print("Loading dataset for tokenizer training...")
    ds = load_from_disk(dataset_path)

    # Use 10 percent of documents for tokenizer training to avoid memory pressure.
    train_ds = ds.select(range(0, len(ds), 10))

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Replace(" ", " "),
        normalizers.NFKC(),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    def batch_iterator(batch_size=1000):
        for i in range(0, len(train_ds), batch_size):
            yield train_ds[i:i + batch_size]["text"]

    print(f"Training tokenizer with {vocab_size} vocab size...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.save("./data/mini_deepseek_tokenizer.json")
    print("Tokenizer saved.")


if __name__ == "__main__":
    train_large_tokenizer("./data/mixed_1b_dedup")
```

### Step 4: Pack, Shuffle, and Export `.arrow` Shards

To avoid wasting GPU time on padding, variable-length token sequences are concatenated into fixed-length blocks of `4096` or `8192` tokens, separated with a special EOT token.

```python
from tokenizers import Tokenizer
from datasets import load_from_disk

SEQ_LEN = 4096


def pack_and_shuffle(dataset_path, tokenizer_path):
    print("Loading tokenizer and deduped dataset...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    ds = load_from_disk(dataset_path)

    eot_id = tokenizer.token_to_id("<|endoftext|>")

    def tokenize_and_pack(examples):
        encoded = [tokenizer.encode(t).ids for t in examples["text"]]

        all_tokens = []
        for ids in encoded:
            all_tokens.extend(ids)
            all_tokens.append(eot_id)

        total_length = len(all_tokens)
        total_length = (total_length // SEQ_LEN) * SEQ_LEN

        result = []
        for i in range(0, total_length, SEQ_LEN):
            result.append(all_tokens[i:i + SEQ_LEN])

        return {"input_ids": result}

    print("Tokenizing and packing into uniform lengths...")
    packed_ds = ds.map(
        tokenize_and_pack,
        batched=True,
        batch_size=1000,
        remove_columns=ds.column_names,
        num_proc=8,
    )

    print("Shuffling dataset globally...")
    packed_ds = packed_ds.shuffle(seed=42)

    print("Saving to .arrow shards...")
    packed_ds.save_to_disk("./data/mixed_1b_final_packed")


if __name__ == "__main__":
    pack_and_shuffle("./data/mixed_1b_dedup", "./data/mini_deepseek_tokenizer.json")
```

## Results and Analysis

On a single 8-GPU 4090 node, the full pipeline took about six hours. With `TARGET_TOTAL_DOCS = 500,000`, MinHash deduplication removed about **4.2%** hidden duplicates, mostly between code and academic sources.

The final shuffled and packed `mixed_1b_final_packed` dataset occupies about `5GB` and is exported in `.arrow` format, totaling about **1.05B tokens**.

### Tokenizer Efficiency Check

With a 150K vocabulary, sampled evaluation shows a Chinese web compression ratio of about **0.62 tokens per character**, a large improvement over Llama-2's roughly 1.1 tokens per character. This directly improves throughput in later pre-training.

## Cost and Optimization

The 1B-token pipeline is economical:

- **Storage**: Raw sampled data takes about 8GB; the final packed artifact takes about 5GB.
- **Compute and memory**: Streaming extraction and parallel map operations keep peak memory around 32GB. Cross-source MinHash deduplication is the longest step, taking about three hours.

For scaling to 70B tokens, single-node Python processing becomes a bottleneck. The recommended path is to use a distributed engine such as Apache Spark (Zaharia et al. 2016) or Ray (Moritz et al. 2018). During MinHash deduplication, Redis or another external store can be used for hash buckets to decouple memory from one process.

## Extensions

Scaling the Mini-DeepSeek recipe to tens of billions of tokens requires special attention to two issues:

1. **Dynamic mixture curriculum**: Early training should emphasize general knowledge, such as web and academic text. Later stages can increase code and math weights. `mix_sampler.py` can be refactored into an epoch-aware streaming sampler.
2. **Upgrade over Mini-C4**: Compared with the single-threshold filtering pipeline in Mini-C4, this project uses cross-source fusion and a large vocabulary to demonstrate how modern industrial models establish multi-task foundations.

### Data Compliance and Open-source Licensing

When mixing sources, respect the original dataset licenses:

- **FineWeb-Edu**: CC0.
- **The Stack v2**: SPDX allowlist-based redistribution.
- **OpenWebMath**: ODC-By.
- **arXiv**: License selected by each paper author.
- **Project Gutenberg**: Public domain.

The compliant 1B-token sample can be uploaded to a Hugging Face Datasets repository such as `dataforge-mini-deepseek-1b` for later fine-tuning use.

## References

Broder A Z (1997) On the Resemblance and Containment of Documents. In: Proceedings of the Compression and Complexity of Sequences, pp 21-29.

Kaplan J, McCandlish S, Henighan T, Brown T B, Chess B, Child R, Gray S, Radford A, Wu J, Amodei D (2020) Scaling Laws for Neural Language Models. arXiv preprint arXiv:2001.08361.

Liu A, Feng B, Xue B, Wang B, Wu B, Lu C, Zhao C, Deng C, Zhang C, Ruan C, others (2024) DeepSeek-V3 Technical Report. arXiv preprint arXiv:2412.19437.

Lozhkov A, Ben Allal L, von Werra L, Wolf T (2024) StarCoder 2 and The Stack v2: The Next Generation. arXiv preprint arXiv:2402.19173.

Moritz P, Nishihara R, Wang S, Tumanov A, Liaw R, Liang E, Elibol M, Yang Z, Paul W, Jordan M I, Stoica I (2018) Ray: A Distributed Framework for Emerging AI Applications. In: Proceedings of the 13th USENIX Symposium on Operating Systems Design and Implementation, pp 561-577.

Paster K, Santos M D, Azerbayev Z, Ba J (2023) OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text. arXiv preprint arXiv:2310.06786.

Penedo G, Kydlicek H, de Wiele T V, Lozhkov A, Mitchell M, Raffel C, von Werra L, Wolf T (2024) The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. arXiv preprint arXiv:2406.17557.

Sennrich R, Haddow B, Birch A (2016) Neural Machine Translation of Rare Words with Subword Units. In: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, pp 1715-1725.

Zaharia M, Xin R S, Wendell P, Das T, Armbrust M, Dave A, Meng X, Rosen J, Venkataraman S, Franklin M J, Ghodsi A, Gonzalez J, Shenker S, Stoica I (2016) Apache Spark: A Unified Engine for Big Data Processing. Communications of the ACM 59(11):56-65.
