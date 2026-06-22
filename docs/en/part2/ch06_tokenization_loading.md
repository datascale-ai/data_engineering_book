# Chapter 6: Tokenization, Serialization, and Efficient Data Loading

<div class="chapter-authors">Ke Wang; Fan Yu; Jun Yu</div>

## Abstract

This chapter discusses how cleaned text is transformed into an input pipeline suitable for efficient large-model training, covering tokenizer design, data format selection, sequence packing, multi-source mixing, DataLoader configuration, caching strategies, and distributed data reading. The chapter opens with an anonymized composite case study illustrating how I/O bottlenecks cause GPU idle time and wasted training cost. It then compares the engineering characteristics of BPE, WordPiece, and SentencePiece, and analyzes the effects of vocabulary size, domain vocabulary extension, and multilingual balancing on training efficiency and capability distribution. The serialization section compares formats such as JSONL, Parquet, Arrow, MDS, WebDataset, and memmap, emphasizing the role of offline tokenization and binary sharding on throughput. The latter half of the chapter further discusses packing, temperature sampling, curriculum learning, and smoke testing, and provides rank-aware configuration for multi-node data reading. Readers should be able to design stable, diagnosable, and cost-controlled input pipelines for pretraining tasks at different scales.

## Keywords

Tokenization; Tokenizer; Serialization; DataLoader; MDS; Packing; Data Mixing; Throughput Diagnostics

## Learning Objectives

- Be able to compare the engineering trade-offs of BPE, WordPiece, and SentencePiece in large-model input pipelines.
- Be able to explain the effects of vocabulary size, domain vocabulary extension, and multilingual sampling on model training.
- Be able to select data formats, sharding strategies, and offline tokenization schemes appropriate for pretraining scale.
- Be able to locate input bottlenecks through smoke tests, GPU utilization monitoring, I/O monitoring, and profilers.
- Be able to design multi-node distributed data reading solutions that avoid duplicate reads, NFS bottlenecks, and global shuffle failures.

## Opening: A Training Incident Where "The Data Pipeline Was Slower Than the Model"

The following is an anonymized composite case study used to illustrate the diagnostic path for input-pipeline bottlenecks; throughput and utilization should be pressure-tested on the target cluster and should not be reused across projects. When a team launched pretraining for a medium-scale base model, anomalies appeared early in training: `nvidia-smi` showed GPU utilization staying below the project baseline for a long period. Initial diagnosis suspected a model configuration issue, until an engineer opened `iostat`, DataLoader wait time, and profiler monitoring and discovered that disk I/O and online tokenization had become bottlenecks: the DataLoader could not keep up with the GPU's consumption rate.

The root cause was quickly identified: the team had stored the cleaned corpus on ordinary disk arrays, with each shard being a compressed `.jsonl.gz` file, requiring the DataLoader to decompress and tokenize in real time at runtime, causing both the CPU and disk to become bottlenecks simultaneously. The team ultimately paused training, re-tokenized all data offline and serialized it into MDS format (Mosaic Data Shard), and migrated to higher-throughput storage media before GPU utilization recovered to the project baseline range.

The cost of such an incident should be recalculated based on real cluster unit prices, pause policies, queue time, and reprocessing cost. More importantly, it could have been discovered entirely during the smoke-test phase before training launch.

This case illustrates the central thesis of this chapter: **the efficiency of the data input pipeline is one of the most underestimated engineering components in pretraining — and one with the highest cost when things go wrong.** It occupies a gray zone between "data cleaning is complete" and "training has not yet started" — belonging neither to the focus of data engineering nor to the tuning scope of the training system, and as a result it is often overlooked by both sides until real compute waste forces it to be addressed.

---

## 6.1 Why the Input Pipeline Determines the Training Ceiling

### 6.1.1 The Hidden Cost of GPU Idle Time

In large-scale pretraining scenarios, GPU cluster rental costs are typically charged by the hour, and prices fluctuate significantly with region, provider, instance specification, and procurement agreement. Under this cost structure, "GPU utilization" is no longer merely a performance metric; it is an economic metric that directly translates into financial loss. Any low utilization caused by waiting for data will extend the time required to reach the same number of effective training tokens.

The more precise metric is **Model FLOPS Utilization (MFU)**. MFU is jointly affected by model architecture, parallel strategy, communication topology, batch size, mixed precision, and kernel implementation, and cannot be judged with a single universal threshold. If MFU or GPU utilization stays below the project's historical baseline for a long period, DataLoader wait time, storage throughput, network reads, and online preprocessing should all be investigated.

### 6.1.2 End-to-End Latency Breakdown from Data Format to GPU

Moving data from disk to GPU memory involves the following stages, each of which can become a bottleneck:

**Disk read**: Reading raw bytes from shard files stored on HDD/SSD/network storage (NFS/S3). Throughput varies greatly across media and cloud-provider object storage, and actual speed is also affected by concurrency, file size, network topology, and cache hit rate. Therefore, before training, pressure tests should be run on the target cluster using the same file format as production shards rather than applying generic bandwidth numbers.

**Decompression and deserialization**: If data is stored in compressed formats such as `.gz` or `.zst`, CPU decompression is required; if stored in text formats like `.jsonl`, JSON parsing is also required. Both steps are compute-intensive CPU operations that consume significant time in DataLoader worker processes.

**Online tokenization (if offline tokenization was not performed)**: Tokenizing text in real time within the DataLoader is one of the most CPU-intensive operations. A `tiktoken` or SentencePiece tokenizer processing a single 1,000-character document takes approximately 0.5–2 ms; when processing concurrently across 8 DataLoader workers, this is sufficient to become a significant bottleneck.

**CPU-to-GPU transfer (PCIe/NVLink)**: Transferring an assembled tensor batch from CPU memory to GPU memory. PCIe 4.0 peak bandwidth is approximately 32 GB/s, but non-contiguous tensor memory layouts can cause actual transfer efficiency to drop substantially.

Understanding this pipeline is a prerequisite for making correct optimization decisions.

---

## 6.2 Tokenization, Serialization, and Data Format Trade-offs

### 6.2.1 Tokenization Algorithms: Engineering Selection Among Three Major Paradigms

Tokenization is the starting point of the training input pipeline and the only component in the entire input processing chain with an "irreversible" property — once the vocabulary and tokenization model are determined, it is difficult to replace them without re-tokenizing all data. Therefore, vocabulary selection decisions must be made carefully before large-scale tokenization begins in earnest.

The tokenization algorithms used by mainstream large models today fall into three main categories:

**BPE (Byte Pair Encoding)** (Sennrich et al. 2016) is the most widely used algorithm; the GPT series (including GPT-3 (Brown et al. 2020), ChatGPT, and GPT-4) are all based on it. Its core idea is to start from the character (or byte) level and repeatedly merge the most frequently occurring adjacent token pairs.

Listing 6-1 presents simplified pseudocode for the BPE merge process.

*Listing 6-1: Simplified pseudocode for the BPE merge process. This snippet explains the merge idea and is not a production-grade tokenizer training implementation.* Note: traditional BPE is not aware of morpheme boundaries; in 2025, MorphBPE (Asgari et al. 2025) explored improving tokenization efficiency and training performance in morphologically rich languages by constraining merge rules not to cross morpheme boundaries.

```python
# BPE merge process pseudocode
def bpe_train(corpus, num_merges):
    vocab = get_initial_characters(corpus)
    for _ in range(num_merges):
        pairs = get_stats(vocab)  # Count the frequency of all adjacent token pairs
        best = max(pairs, key=pairs.get) # Select the most frequent pair
        vocab = merge_vocab(best, vocab) # Merge it into a new token
    return vocab
```
The byte-level variant of BPE (Byte-level BPE, such as GPT-2's tiktoken) completely resolves the OOV problem by using raw bytes rather than Unicode characters as the base unit, and has been widely adopted by models such as LLaMA 2/3 and Mistral.

**WordPiece** is BERT's tokenization scheme. It is similar to BPE but the merge criterion is not absolute frequency; instead it uses **maximum likelihood estimation based on a language model**. When WordPiece merges $A$ and $B$, it evaluates the score $\frac{P(AB)}{P(A)P(B)}$ (similar to mutual information). This means that if $A$ and $B$ each have relatively low standalone probabilities, but their co-occurrence probability is high, WordPiece will tend to merge them.

**The OOV (Out-of-Vocabulary) Problem**:
In the traditional era of word-level tokenizers, when encountering rare characters or words not recorded in the vocabulary, the model would typically emit a `<UNK>` (out-of-vocabulary) placeholder representing the unknown token. This is catastrophic in specialized fields such as medicine and law: a passage containing complex chemical formulas would become a screen full of `<UNK>` tokens. Subword-based approaches like BPE and WordPiece, when encountering previously unseen words, decompose them further into more basic subwords or even individual characters/bytes. Although this increases sequence length, true OOV truncation never occurs, ensuring lossless information delivery.

**SentencePiece (Unigram)** (Kudo and Richardson 2018) is Google's promoted approach, and rather than following a "merge from small to large" route, it takes a "prune from large to small" approach. Unigram starts from a very large base vocabulary and iteratively computes and removes the tokens whose removal causes the smallest drop in overall corpus likelihood. It is more amenable to languages without explicit word boundaries such as Chinese and Japanese.

For Chinese large models, **Byte-level BPE** (implemented via tiktoken) is recommended as the base approach, with a vocabulary size between **64K and 100K** — this range achieves a reasonable balance between Chinese character coverage (Chinese has approximately 50,000 characters, with about 3,500 common-use characters) and embedding matrix parameter count. A vocabulary that is too small (32K) will cause many Chinese characters to be split into multiple byte-level tokens, significantly increasing sequence length; a vocabulary that is too large (200K+) will make the embedding matrix parameter count excessively large, reducing training efficiency.

Listing 6-2 presents a sample implementation for offline batch tokenization using `tiktoken`.

*Listing 6-2: Example code for offline batch tokenization. Production environments should add shard validation, failed retries, vocabulary-version records, and output-consistency checks.*

```python
# Offline batch tokenization using tiktoken (recommended for preprocessing)
import tiktoken, json
from pathlib import Path

enc = tiktoken.get_encoding("cl100k_base")   # GPT-4's base BPE vocabulary

def tokenize_document(doc: dict, max_length: int = 4096) -> dict | None:
    """
    Tokenize a single document and attach metadata.
    Returns None if the document is too short after tokenization to be included in training.
    """
    token_ids = enc.encode(doc["text"], disallowed_special=())
    if len(token_ids) < 64:        # Filter out documents that are too short
        return None
    return {
        "token_ids": token_ids,
        "num_tokens": len(token_ids),
        "source": doc.get("source", "unknown"),
        "quality_tier": doc.get("quality_tier", "medium"),
    }
```

### 6.2.2 Vocabulary Design and Domain Adaptation: More Than "Good Enough"

The vocabulary is the core output of the tokenizer and also the only component of the overall large-model architecture that is nearly impossible to change after training begins. Once the vocabulary is determined, all subsequent data processing, model embedding matrices, and output logit layers are tightly bound to it — changing the vocabulary means re-tokenizing all training data and re-initializing the embedding matrix (discarding the pretrained embedding weights), at enormous cost. Therefore, vocabulary design decisions must be completed before the entire engineering effort begins, rather than being corrected mid-training when a problem is discovered.

**Vocabulary size trade-offs** are the primary decision. A larger vocabulary, such as at the 100K scale, can preserve more high-frequency words and domain-specific terminology as single tokens, reducing sequence length and lowering Transformer computation (because attention complexity is quadratic in sequence length); but a larger embedding matrix increases parameter count, and rare tokens encounter fewer training samples, resulting in lower embedding quality. LLaMA-3 (Grattafiori et al. 2024) dramatically expanded the vocabulary from LLaMA-2's 32K to 128K and identifies the larger vocabulary as an important design for improving multilingual and code capabilities. Additional embedding parameters can be estimated as "number of added tokens x hidden size," while memory overhead also depends on parameter precision, tied embeddings, and optimizer state; it should not be given as a fixed GB value without the model configuration.

**Domain Vocabulary Extension** is a common requirement for vertical-domain large models. When the base vocabulary has insufficient coverage of domain-specific terminology (e.g., molecular formulas in medical terminology, proper nouns in legal terminology, keyword combinations in programming languages), these terms are split into multiple sub-tokens, leading to: first, increased sequence length, reducing the amount of domain information the model's context window can accommodate; and second, the model needing to reconstruct semantics from fragmented tokens at higher learning cost.

The standard approach for domain vocabulary extension is: collect large amounts of professional text from the target domain, count the terms with the highest "segmentation ratio" (i.e., most commonly represented as multiple tokens) under the base vocabulary, select the top-K terms to add to the vocabulary, and correspondingly expand the embedding matrix (the embedding vectors for newly added tokens are typically initialized as the mean of their sub-token embeddings, to reduce distributional shift at the start of training). Chinese vertical-domain versions of LLaMA (such as Chinese-LLaMA) have broadly adopted the strategy of adding 20K–30K Chinese characters and vocabulary items on top of the original vocabulary (32K), effectively improving Chinese generation quality and inference efficiency.

**Cross-lingual vocabulary balancing** is another critical challenge for multilingual foundation models (such as BLOOM, mT5, and Qwen). If the vocabulary is trained jointly on multilingual corpora directly, high-resource languages (English) will occupy more vocabulary space due to their higher frequency, severely compressing the vocabulary of low-resource languages (such as Thai or Arabic), resulting in what is known as the "vocabulary curse" — the text of these languages appears to the model as nearly meaningless byte fragments, causing the understanding and generation capabilities of low-resource languages to fall far below those of high-resource languages.

The solution is to **upsample and balance** the corpora of different languages when training the tokenizer: sample the training text for each target language to approximately the same token count (or use a temperature parameter T=3–5), ensuring each language receives sufficient vocabulary "seats"; at the same time, use SentencePiece's `character_coverage=0.9999` parameter to ensure that the basic character set of each language (even if low in frequency) is included in the vocabulary. This is a common engineering practice in the vocabulary design of multilingual models such as mT5 and BLOOM.

Listing 6-3 presents a sample configuration for SentencePiece multilingual vocabulary training.

*Listing 6-3: SentencePiece multilingual vocabulary training configuration snippet. Parameters are configuration examples only; production environments should tune them jointly through language coverage, OOV/UNK rate, and downstream evaluation.*

```python
# SentencePiece multilingual vocabulary training (illustrative)
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="multilingual_corpus_balanced.txt",
    model_prefix="tokenizer_multilingual_100k",
    vocab_size=100_000,
    model_type="bpe",
    character_coverage=0.9999,   # Ensure full coverage of each language's basic character set
    byte_fallback=True,           # Fall back to byte representation for Unicode characters not in vocabulary
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    # Upsample low-resource languages (sampling weights specified per line in a weights file)
    input_sentence_size=20_000_000,
    shuffle_input_sentence=True,
)
```



### 6.2.3 Data Formats and Serialization: A Performance-Defining Choice

The choice of data format has a direct, order-of-magnitude impact on DataLoader throughput. The following summarizes the performance and engineering trade-offs of mainstream formats:

*Table 6-1: Data format, compression, and access pattern comparison. Source: compiled by the authors; performance should be validated through pressure tests on target hardware, storage backend, compression method, and DataLoader implementation.*

| Format | Type | Sequential Read Speed | Random Access | Compression Support | Cross-Framework Support | Applicable Scenarios |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **JSONL (.jsonl)** | Text lines | Slow (requires JSON parsing) | Not supported | ✗ (requires .gz combination) | Excellent | Data exchange, debugging |
| **Parquet** | Columnar binary | Fast (column pruning) | Supported (row-group level) | √ Snappy/Zstd | Very good (Spark/pandas) | Batch analytics, Chapter 5 output |
| **Apache Arrow / Feather** | Row-oriented binary | Very fast (zero-copy) | Supported | √ LZ4/Zstd | Good (PyArrow) | CPU→GPU intermediate layer |
| **MDS (Mosaic)** | Shard binary | Very fast | Shard level | √ Zstd | Good (Streaming Datasets) | Preferred for LLM pretraining |
| **WebDataset (.tar)** | Tar archive | Fast (streaming) | Shard level | √ (internal file compression) | Good (Torchvision) | Multimodal training |
| **Raw .bin (Token IDs)** | Binary integer | Very fast (memory-mapped) | Supported (byte offset) | ✗ | Requires custom implementation | Very large-scale pretraining |

For LLM pretraining scenarios, **MDS format** (Mosaic AI Research 2022) is currently the most recommended choice — it is designed specifically for streaming multi-node reads, supports concurrent conflict-free reading of the same dataset by multiple GPU nodes, has a built-in shuffle buffer, and supports direct streaming reads from object storage such as S3/GCS without requiring a full download. The second choice is **Raw .bin memory-mapped format** (the approach used by Megatron-LM), which writes the token ID array directly as a binary file and uses `np.memmap` for memory mapping at read time, achieving near-memory read speeds on local NVMe SSDs.

### 6.2.4 Sharding Strategy and Global Shuffle

The dataset should be split into a large number of equal-sized shard files rather than stored as a single large file. The recommended shard size is between **256 MB and 1 GB**; this range is chosen because shards that are too small lead to excessive file metadata overhead (file open, seek operations), while shards that are too large cause load imbalance in cross-node distribution, and corruption of a single shard results in a larger amount of data becoming unavailable.

Shuffle is another critical step in pretraining data preparation. Unshuffled data is arranged in source order, with data from the same source appearing in concentrated bursts, causing the model to encounter continuous "local distributional shifts" during training that affect the smoothness of loss convergence. Global Shuffle requires random permutation across all shards — this is easy to implement on a single machine, but requires dedicated design in distributed training (MDS format has built-in support for streaming shuffle buffers across shards and is the recommended approach).

---

## 6.3 Packing, Mixing, and Curriculum Strategies

### 6.3.1 Sequence Packing: Eliminating the "Compute Tax" of Padding

In standard DataLoader implementations, all samples in a batch are padded to the same length. When the training set contains a large number of short documents (such as many QA pairs or brief code snippets), the proportion of padding tokens can reach 30–50% — meaning 30–50% of GPU compute resources are wasted on ineffective attention computations over `<pad>` tokens.

**Sequence Packing** is the standard engineering technique for addressing this: multiple short documents are concatenated within the same sequence, with a special `[EOS]` token serving as a document boundary, so that the proportion of effective tokens in each sequence approaches 100%. The attention mask correspondingly breaks cross-document attention at the `[EOS]` position (preventing the end of document A from influencing the attention of the beginning of document B), preserving the semantic independence of each document.

Listing 6-4 presents a sample implementation of greedy sequence packing.

*Listing 6-4: Example code for greedy sequence packing. This snippet shows the basic strategy; production environments should add sample boundaries, label masks, and reproducible experiment records.*

```python
def greedy_pack_sequences(
    token_id_lists: list[list[int]],
    max_seq_len: int = 4096,
    eos_token_id: int = 2
) -> list[dict]:
    """
    Greedy bin-packing: pack multiple document token lists into fixed-length sequences.
    Returns each packed sequence along with its attention mask.
    """
    packed, current_seq, current_mask = [], [], []
    doc_count = 0

    for token_ids in token_id_lists:
        token_ids = token_ids + [eos_token_id]   # End-of-document marker
        if len(current_seq) + len(token_ids) > max_seq_len:
            if current_seq:
                # Pad to max_seq_len with 0 (pad)
                pad_len = max_seq_len - len(current_seq)
                packed.append({
                    "input_ids":      current_seq + [0] * pad_len,
                    "attention_mask": current_mask + [0] * pad_len,
                    "num_docs":       doc_count
                })
            current_seq, current_mask, doc_count = [], [], 0

        current_seq.extend(token_ids)
        current_mask.extend([1] * len(token_ids))
        doc_count += 1

    return packed
```

For training sets containing many short documents, enabling packing can usually improve effective token throughput (tokens/s). Actual gains depend on document length distribution, max sequence length, attention-mask implementation, and hardware configuration, and should be verified on the target dataset using both padding ratio and tokens/s.

### 6.3.2 Multi-Source Mixing: Temperature Weighting and Domain Ratio Control

When training data comes from multiple heterogeneous sources (web corpora, code, academic papers, books, enterprise data), how to control the sampling proportion of each source during training is a critical decision that directly affects the distribution of model capabilities.

The most commonly used approach is **temperature sampling**: the data volume $n_i$ for each source is transformed by a temperature parameter $T$ via exponentiation and then normalized into sampling weights:

$$p_i = \frac{n_i^{1/T}}{\sum_j n_j^{1/T}}$$

When $T = 1$, weights are proportional to data volume and large sources completely dominate; as $T \to \infty$, all source weights approach uniform. In practice, $T = 2$ is commonly used (the multilingual sampling setting of mT5 (Xue et al. 2021)), upsampling small sources while avoiding excessive deviation from the original data distribution.

*Table 6-2: Comparison of sampling and mixing strategy benefits. Source: compiled by the authors; benefit descriptions are summaries of common patterns, and actual effects should be confirmed through data-recipe ablation experiments.*

| Mixing Strategy | Principle | Advantages | Disadvantages | Applicable Scenarios |
| :--- | :--- | :--- | :--- | :--- |
| **Proportional sampling** (T=1) | Proportional to original data volume | Closest to true data distribution | Small sources overwhelmed by large ones; code/papers are underrepresented | General corpus pretraining (early stages) |
| **Uniform sampling** (T→∞) | Equal probability per source | Full coverage of all sources | Model skews toward minority source styles; general capability degrades | Specific coverage experiments |
| **Temperature sampling** (T=2) | Power-smoothing of data volume | Balances large and small sources, enhances diversity | Requires hyperparameter tuning | Multilingual, multi-domain mixing (recommended) |
| **Fixed-ratio mixing** | Manually specify mixing ratio per source | Fully controllable, directly aligns with business objectives | Requires manual design; costly if misconfigured | Custom training with clear business objectives (recommended) |
| **Curriculum learning** | Use simple/general data first, introduce complex data later | More stable convergence, more efficient targeted capability improvement | Requires designing difficulty metrics; complex to implement | Long-term large-scale training |

### 6.3.3 Curriculum Learning

Curriculum learning is a strategy that **dynamically adjusts the data recipe** during training (Bengio et al. 2009): in the early stages of model training, "simpler" data (shorter sentences, more fluent language, more general domains) is used, with progressively longer and more complex samples introduced as training proceeds. This mimics the cognitive principle that humans learn "easy before hard."

In engineering implementation, difficulty metrics for curriculum learning can come from multiple dimensions: token sequence length (short → long), perplexity score (low perplexity → high perplexity), and quality tier (High → Medium → Low). The Llama 3 technical report (Grattafiori et al. 2024) explicitly mentions substantially increasing the weight of high-quality curated data (code, mathematical reasoning, books) during the pretraining cooldown phase — this is essentially a **data quality curriculum**: first using massive general data to establish broad world knowledge, then using high-quality curated data in the final phase to strengthen specific capabilities.

---

## 6.4 Efficient Loading, Caching, and Throughput Diagnostics

### 6.4.1 Key DataLoader Configuration Parameters

PyTorch's `DataLoader` provides several parameters that directly affect I/O throughput. The following are the most engineering-significant for large-scale pretraining scenarios:

**`num_workers`**: Controls the number of subprocesses for parallel data reading. This is the most common tuning point. The general rule is `num_workers = 4–8 × number of GPUs`, but the actual optimal value must be determined through experimentation (too many workers can actually reduce throughput due to process management overhead and IPC contention). For MDS format read from high-speed NVMe SSDs, `num_workers=8–16` is typically sufficient to saturate disk utilization.

**`pin_memory=True`**: When enabled, the DataLoader allocates pinned memory on the CPU side for storing batches, allowing subsequent CPU-to-GPU transfers to use DMA (Direct Memory Access), significantly improving PCIe transfer efficiency. The actual gain depends on batch size, tensor layout, hardware, and transfer frequency and should be measured with profiling.

**`prefetch_factor`**: The number of batches each worker prefetches in advance (default is 2). Increasing this moderately (e.g., to 4–8) can hide disk read latency, but increases CPU memory usage.

Listing 6-5 presents a DataLoader configuration example based on MosaicML Streaming Dataset.

*Listing 6-5: MosaicML Streaming Dataset DataLoader configuration snippet. Production environments should pressure-test object-storage bandwidth, cache strategy, and node failure-recovery capability together.*

```python
from torch.utils.data import DataLoader
from streaming import StreamingDataset  # MosaicML Streaming Datasets

dataset = StreamingDataset(
    local="./data/shards/",
    remote="s3://my-bucket/shards/",   # Supports streaming reads from S3 during training
    shuffle=True,
    shuffle_seed=42,
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=12,        # Adjust based on CPU core count and number of GPUs
    pin_memory=True,       # Enable pinned memory to accelerate CPU→GPU transfer
    prefetch_factor=4,     # Prefetch 4 batches per worker
    persistent_workers=True,  # Avoid worker restart overhead between epochs
)
```

Listing 6-6 presents a sample implementation of a binary token ID dataset based on `np.memmap`.

*Listing 6-6: Example code for an `np.memmap`-based token ID dataset. Production environments should add dtype, file-integrity, index-boundary, and cross-platform compatibility checks.*

```python
# Pseudocode for a memmap binary loader optimized for tens-of-millions of small-file I/O
import numpy as np
class MemmapDataset(torch.utils.data.Dataset):
    def __init__(self, bin_path, seq_len=4096):
        # Use np.memmap to map a large binary .bin file (Raw Token IDs)
        # Completely avoids loading the entire dataset into memory; relies on OS Page Cache for fast random access
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.total_tokens = len(self.data)
        self.num_samples = self.total_tokens // self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        # Slicing is lightweight, handled by underlying C code and OS memory paging for high throughput
        chunk = self.data[start_idx : start_idx + self.seq_len]
        return torch.from_numpy(chunk.astype(np.int64))
```

### 6.4.2 Throughput Bottleneck Diagnostics: A Three-Step Systematic Approach

When GPU utilization falls below expectations, follow these systematic steps to diagnose:

![Figure 6-1: Throughput Bottleneck Diagnosis Flowchart](../../images/part2/io_bottleneck_diagnosis_flow.svg)

*Figure 6-1: Throughput bottleneck diagnosis flowchart — starting from abnormal GPU utilization, a three-level decision tree is used to locate disk I/O bottlenecks, CPU preprocessing bottlenecks, and PCIe transfer bottlenecks, with corresponding remediation steps. Source: original illustration from this book.*

**Step 1 — Confirm whether the GPU is waiting for data**: Run `nvidia-smi dmon -s u` to monitor SM utilization; if SM utilization periodically drops to 0 and `sm_active` is intermittently 0, the GPU is waiting. Also check the MFU (Model FLOPS Utilization) metric and compare it with the project's historical baseline.

**Step 2 — Locate the I/O level**: Run `iostat -x 1` to monitor disk I/O and compare utilization, queue depth, and wait time with the storage baseline. Simultaneously use `top` or `htop` to check CPU utilization of DataLoader worker processes; if all CPU cores are saturated, online tokenization or decompression is the bottleneck.

**Step 3 — Check PCIe transfer**: Use PyTorch's Profiler to record the time proportion of `cudaMemcpyH2D`; if H2D transfer time takes an abnormal share relative to GPU kernel execution time, PCIe transfer is the bottleneck, requiring `pin_memory` to be enabled or tensor memory layout to be optimized.

### 6.4.3 Pre-Training Smoke Test: 30-Minute Automated Verification Before Launch

Before formally launching a long-running pretraining job, it is strongly recommended to execute a brief **smoke test** — using the complete data pipeline (real shard files, real DataLoader configuration) but running only a small number of training steps, specifically to verify the following metrics:

- **DataLoader throughput**: Whether Tokens/s meets the target value (which can be pre-calculated based on GPU count and MFU target)
- **GPU utilization**: Whether it is stable around the project baseline
- **Initial loss value**: Whether it is within a reasonable range (for LLMs, the loss at random initialization is approximately ln(vocab_size); for a vocabulary of 100K this is approximately 11.5)
- **No abnormal crashes**: No DataLoader worker crashes, no CUDA OOM

This short smoke test can detect many configuration issues and avoid the high-cost mistake from the opening case study of discovering the pipeline bottleneck only after formal training has already started.

### 6.4.4 Multi-Node Distributed Data Reading: Avoiding "Data Silos"

When training scales to multi-machine multi-GPU configurations (e.g., 8 servers × 8 GPUs = 64 GPUs), data loading faces new challenges not encountered in single-machine scenarios: **how to enable all nodes to read the same dataset efficiently and without conflict, while ensuring the correctness of the global data distribution (no duplicates, no omissions, globally consistent shuffle randomness)?**

The most common mistake is "shared NFS mounting" — all nodes mount the same NFS filesystem, with each node's DataLoader reading shards directly from NFS. This approach is simple to configure, but under large-scale concurrent reads the NFS server quickly becomes a bandwidth bottleneck, and NFS random access latency is usually higher than local SSDs, seriously degrading DataLoader throughput.

**Recommended Option 1: Local SSD + Data Pre-staging**. Before training begins, distribute shard files to the local NVMe SSD of each node in advance (via rsync or S3 batch download); during training, each node reads only from local disk. This option delivers the best I/O performance but requires additional storage space and pre-staging time, which depends on data volume, network bandwidth, and copy concurrency. The shard allocation strategy is recommended to use "static allocation + global ordering": shuffle all shards in global random order, then distribute them evenly across nodes (node 0 gets shards 0, 8, 16..., node 1 gets shards 1, 9, 17...), ensuring each node's data partition is an equal share of the globally shuffled dataset.

**Recommended Option 2: MosaicML Streaming from S3**. This has become an increasingly popular approach among large teams in recent years — the dataset is stored in object storage such as S3/GCS, and each node downloads shards on demand via `StreamingDataset` during training (download one shard, delete it after training on it is complete, then download the next), with local disk serving only as a cache layer (cache size is configurable). The advantage of this approach is that the dataset does not need to be pre-copied to each node, and new nodes can join training immediately; the limitation is that it requires stable object-storage read bandwidth and a low-jitter network, making it unsuitable for scenarios with very small shards or high latency sensitivity.

Listing 6-7 presents a sample DataLoader configuration for multi-node distributed training.

*Listing 6-7: Multi-node distributed DataLoader configuration snippet. Production environments should combine rank-aware sharding, global shuffle, and token-count consistency checks.*

```python
# DataLoader configuration for multi-node distributed training
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from streaming import StreamingDataset

# Initialize distributed
rank = dist.get_rank()
world_size = dist.get_world_size()

# MosaicML Streaming: S3 streaming reads with automatic inter-node shard coordination
dataset = StreamingDataset(
    local=f"/nvme/cache/rank_{rank}/",     # Independent local cache directory per node
    remote="s3://my-bucket/pretrain_shards/",
    shuffle=True,
    shuffle_seed=42,
    num_canonical_nodes=world_size,         # Shard distribution based on total node count
)

# Each rank automatically receives a non-overlapping subset of shards
dataloader = DataLoader(
    dataset,
    batch_size=8,          # Micro-batch size per GPU
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)
```

**Avoiding duplicate reads** is a common pitfall in multi-node distributed DataLoaders: if the DataLoaders across nodes are not properly partitioned in a rank-aware manner, each node will independently read the complete dataset, causing all nodes to see the same data order — gradient updates are effectively performed on repeated data, equivalent to batch size not correctly scaling with the number of nodes. For custom datasets that do not use `StreamingDataset`, `DistributedSampler` must be used, and `sampler.set_epoch(epoch)` must be called at the beginning of each epoch to ensure shuffle randomness differs across epochs.

**Global token count consistency check**: In distributed training, the number of tokens actually processed by each rank should be nearly identical (allowing for ±1 batch of error). If token counts across ranks differ significantly (more than 5%), it indicates a problem with data distribution or DataLoader configuration, which should be verified during the smoke test phase by using `dist.all_reduce` to aggregate batch counts across ranks.



---

## 6.5 Engineering Case Studies and Performance Optimization Checklist

### Figures and Case Studies

![Figure 6-2: Training Input Pipeline Layer Diagram](../../images/part2/training_input_pipeline_layers.svg)

*Figure 6-2: LLM training input pipeline layered architecture — the complete five-stage path from tokenization, serialization, data mixing, and packing to DataLoader GPU feeding, with the two highest-frequency bottleneck risk points (disk I/O and CPU-GPU transfer) annotated at the bottom. Source: original illustration from this book.*

### Case Study: Migration Benefits from JSONL + Online Tokenization to MDS + Offline Tokenization

Continuing the anonymized composite case from the opening, the following gives a pressure-test comparison template after completing the storage-format migration. The table deliberately avoids fixed numbers to prevent results from one cluster from being misread as universal gains; actual results depend on hardware, storage, data format, batch size, and framework implementation.

**Before migration** (JSONL.gz, HDD, online tokenization):

- Record disk read speed, object-storage request latency, and DataLoader wait time.
- Record end-to-end tokens/s, GPU utilization/MFU, and CPU decompression/tokenization usage.
- Record read failures, retries, and bad-sample ratios for each shard.

**After migration** (MDS, NVMe SSD, offline pre-tokenization):

- Record end-to-end tokens/s under the same batch size and model configuration.
- Record whether GPU wait time decreases and whether the DataLoader remains a bottleneck.
- Record additional storage cost, preprocessing time, and data-validation cost introduced by the migration.

**Core-benefit accounting method**: Compare end-to-end tokens/s, DataLoader wait time, GPU wait time, and total preprocessing cost before and after migration, then convert them into the GPU hours required to reach the same number of training tokens. Only under the same model configuration, batch size, and training objective are pre- and post-migration gains comparable.

### 6.5.1 Input Pipeline Optimization Checklist

The following is a directly usable input pipeline optimization checklist, recommended to be reviewed item by item before launching each new training job:

**Storage and Format**

- [ ] Data is stored in binary format (MDS / .bin memmap), not JSONL
- [ ] Stored on high-speed devices (NVMe SSD / high-performance network storage), not HDD or NFS
- [ ] Shard size is between 256 MB and 1 GB
- [ ] Data has completed checksum verification to ensure no corrupted shards

**Tokenization and Serialization**

- [ ] Tokenization was completed offline during the preprocessing phase; the DataLoader does not perform online tokenization
- [ ] Sequences have been packed; padding token proportion is below the project baseline
- [ ] Global shuffle has been enabled (randomness across shards)
- [ ] Sequence length distribution matches the target max_seq_len

**DataLoader Configuration**

- [ ] `num_workers` has been tuned through the smoke test rather than copied from a fixed multiple
- [ ] `pin_memory=True`
- [ ] `persistent_workers=True` (avoid restart overhead between epochs)
- [ ] Smoke test has been run, confirming that GPU util, MFU, and DataLoader wait time reach the project baseline

**Monitoring and Observability**

- [ ] Tokens/s, GPU util, and DataLoader wait time are continuously recorded during training
- [ ] DataLoader timeout alerts have been configured (e.g., alert triggers if batch wait > 10 s)
- [ ] Complete data source metadata has been retained to support issue tracing

---

## Chapter Summary

This chapter used an anonymized composite case study to illustrate how I/O bottlenecks cause GPU idle time and cost waste, systematically establishing a complete technical understanding of the training input pipeline. We examined tokenization algorithm selection in detail (the engineering trade-offs of BPE/SentencePiece), data format choices (the performance trade-offs from JSONL to MDS/Arrow), packing strategies (reducing ineffective computation caused by padding), temperature sampling and curriculum learning-based mixing strategies (Table 6-2), and a systematic three-step I/O bottleneck diagnosis method (Figure 6-1). The "Input Pipeline Optimization Checklist" in this section can be used directly as a pre-launch verification tool for production-grade pretraining jobs.

This chapter echoes the cost governance perspective of Chapter 3: in pretraining projects where GPU compute costs are extremely high, the engineering quality of the input pipeline directly affects effective training time and total budget. Moving into the next chapter, we shift our perspective from "how to feed data into the model" to "how to evaluate what the model has learned from this data": **Chapter 7: Data Evaluation, Quality Feedback Loops, and Operational Iteration**.

## References

Bengio Y, Louradour J, Collobert R, Weston J (2009) Curriculum Learning. In: Proceedings of the 26th Annual International Conference on Machine Learning, pp 41-48.

Brown T B, Mann B, Ryder N, Subbiah M, Kaplan J, Dhariwal P, Neelakantan A, Shyam P, Sastry G, Askell A, Agarwal S, Herbert-Voss A, Krueger G, Henighan T, Child R, Ramesh A, Ziegler D M, Wu J, Winter C, Hesse C, Chen M, Sigler E, Litwin M, Gray S, Chess B, Clark J, Berner C, McCandlish S, Radford A, Sutskever I, Amodei D (2020) Language Models are Few-Shot Learners. In: Advances in Neural Information Processing Systems 33, pp 1877-1901. arXiv:2005.14165.

Grattafiori A, Dubey A, Jauhri A, Pandey A, Kadian A, Al-Dahle A, Letman A, Mathur A, Schelten A, Vaughan A, others (2024) The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.

Kudo T, Richardson J (2018) SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. In: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp 66-71.

Mosaic AI Research (2022) MosaicML Streaming. GitHub repository. https://github.com/mosaicml/streaming.

Sennrich R, Haddow B, Birch A (2016) Neural Machine Translation of Rare Words with Subword Units (BPE). In: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, pp 1715-1725.

Xue L, Constant N, Roberts A, Kale M, Al-Rfou R, Siddhant A, Barua A, Raffel C (2021) mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer. In: Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics, pp 483-498.

Asgari E, El Kheir Y, Javaheri M A S (2025) MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies. arXiv preprint arXiv:2502.00894.
