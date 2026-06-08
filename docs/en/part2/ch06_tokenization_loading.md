# Chapter 6: Tokenization, Serialization, and Efficient Loading

## Abstract

This chapter explains how cleaned text becomes an efficient training input pipeline for large models. It covers tokenizer design, data-format selection, sequence packing, multi-source mixing, DataLoader configuration, caching, throughput diagnosis, and distributed reading. We begin with an anonymized training incident in which an inefficient input pipeline caused expensive GPU idle time. We then compare BPE, WordPiece, and SentencePiece; analyze vocabulary size, domain vocabulary extension, and multilingual balance; and compare JSONL, Parquet, Arrow, MDS, WebDataset, and memmap formats. The second half of the chapter discusses packing, temperature sampling, curriculum learning, smoke tests, and rank-aware distributed data loading. Readers should be able to design input pipelines that are stable, diagnosable, and cost-controlled at different pre-training scales.

## Keywords

Tokenization; tokenizer; serialization; DataLoader; MDS; sequence packing; data mixing; throughput diagnosis

## Learning Objectives

- Compare BPE, WordPiece, and SentencePiece for large-model input pipelines.
- Explain how vocabulary size, domain vocabulary extension, and multilingual sampling affect training efficiency and capability distribution.
- Choose suitable data formats, shard strategies, and offline tokenization plans for different training scales.
- Use smoke tests, GPU utilization, I/O monitoring, and profilers to locate input bottlenecks.
- Design distributed reading schemes that avoid duplicate reads, NFS bottlenecks, and broken global shuffle.

## Opening: When the Data Pipeline Was Slower Than the Model

The following anonymized composite case uses approximate costs and throughput figures as of 2026-06. A team launched pre-training for a 13B model on a 64-A100 cluster. At high-end cloud prices, this cluster cost roughly RMB 16,000 per hour. Two hours into training, `nvidia-smi` showed GPU utilization stuck near 38%, far below the expected 85%+. The first suspicion was model configuration. Then engineers opened `iostat` and saw the real issue: disk I/O was saturated, yet the DataLoader still could not feed the GPUs.

The root cause was simple. The cleaned corpus was stored on ordinary HDD arrays. Each shard was compressed `.jsonl.gz`; the DataLoader decompressed and tokenized text online during training. CPU and disk both became bottlenecks. The team paused training for 18 hours, re-tokenized all data offline, serialized it into Mosaic Data Shard (MDS), and moved the shards to NVMe SSDs. GPU utilization recovered to about 88%.

The waste was roughly RMB 30,000 in compute plus 18 hours of engineering delay. A 30-minute smoke test before launch could have caught the problem. The lesson is the premise of this chapter: **input-pipeline efficiency is one of the most underestimated and most expensive failure points in pre-training**.

---

## 6.1 Why the Input Pipeline Determines the Training Ceiling

### 6.1.1 Hidden Cost of GPU Idling

In large-scale pre-training, GPU cost is billed by time. Utilization is therefore a financial metric, not only a performance metric. Every 10% drop in utilization means 10% of the compute budget is waiting for data rather than producing gradients.

For modern LLM training, Model FLOPS Utilization (MFU) is a more precise metric than raw GPU utilization. Sustained MFU below roughly 30% is a strong signal that data loading, communication, or kernel efficiency is limiting the system. Data-pipeline bottlenecks are especially common because they sit between data engineering and training infrastructure.

### 6.1.2 Latency From Format to GPU Memory

Data travels through several stages before it reaches GPU memory:

**Disk read.** Bytes are read from HDD, SSD, NFS, or object storage. HDDs may deliver about 200 MB/s sequential read, NVMe SSDs can reach several GB/s, and object storage depends on concurrency and network bandwidth.

**Decompression and deserialization.** `.gz` and `.zst` need CPU decompression. JSONL requires text parsing. These operations can dominate DataLoader worker time.

**Online tokenization.** If tokenization is performed inside the DataLoader, CPU time grows quickly. A tokenizer may spend 0.5-2 ms per 1,000-character document; multiplied by many samples per step, this can starve GPUs.

**CPU-to-GPU transfer.** Tensor batches move over PCIe or NVLink. Non-contiguous tensors and missing pinned memory reduce transfer efficiency.

Understanding this chain is necessary before optimizing the wrong layer.

---

## 6.2 Tokenization, Serialization, and Format Trade-offs

### 6.2.1 Tokenization Algorithms: Three Engineering Families

Tokenization is the start of the input pipeline and one of its few nearly irreversible decisions. Once a tokenizer and vocabulary are fixed, changing them requires re-tokenizing the dataset and changing the embedding/output layers.

**BPE (Byte Pair Encoding)** is the most widely used family. GPT-style tokenizers use variants of BPE. BPE starts from characters or bytes and repeatedly merges the most frequent adjacent token pairs.

**Listing 6-1: Simplified BPE merge process**

```python
def bpe_train(corpus, num_merges):
    vocab = get_initial_characters(corpus)
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    return vocab
```

Byte-level BPE, as used by GPT-2-style `tiktoken`, avoids true out-of-vocabulary failure by falling back to bytes. LLaMA and Mistral-family models rely on similar subword ideas.

**WordPiece** is associated with BERT. It resembles BPE, but the merge criterion is likelihood-oriented rather than pure frequency. It favors pairs whose joint occurrence is informative relative to the separate occurrence of each part.

**SentencePiece / Unigram** starts from a large candidate vocabulary and prunes tokens that contribute least to corpus likelihood. It is friendly to languages without explicit word boundaries, including Chinese and Japanese.

Traditional word-level tokenizers suffer from OOV failure: rare words or domain terms collapse into `<UNK>`. Subword tokenizers avoid this by decomposing unseen words into smaller units. The price is longer sequences, but information is not lost.

For Chinese LLMs, Byte-level BPE is a strong default, with vocabulary size often in the 64K-100K range. A 32K vocabulary can split too many Chinese characters into multiple byte tokens; a 200K+ vocabulary inflates the embedding matrix and leaves rare tokens undertrained.

**Listing 6-2: Offline tokenization with tiktoken**

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


def tokenize_document(doc: dict, max_length: int = 4096) -> dict | None:
    token_ids = enc.encode(doc["text"], disallowed_special=())
    if len(token_ids) < 64:
        return None
    return {
        "token_ids": token_ids[:max_length],
        "num_tokens": min(len(token_ids), max_length),
        "source": doc.get("source", "unknown"),
        "quality_tier": doc.get("quality_tier", "medium"),
    }
```

### 6.2.2 Vocabulary Design and Domain Adaptation

Vocabulary size is a major architectural decision. Larger vocabularies preserve more frequent words and domain terms as single tokens, reducing sequence length and attention cost. They also enlarge embedding and output matrices and make rare-token embeddings harder to train. LLaMA 3 expanded its vocabulary sharply compared with LLaMA 2, improving multilingual and code handling but increasing embedding-layer cost.

**Domain vocabulary extension** is common for medical, legal, chemical, and code-heavy models. If core domain terms are repeatedly split into many sub-tokens, the sequence becomes longer and the model must reconstruct meaning from fragments. A practical extension workflow is:

1. Collect a large domain corpus.
2. Identify terms that are frequently split into many tokens by the base tokenizer.
3. Add the top terms to the vocabulary.
4. Initialize new embeddings from the average of their sub-token embeddings.
5. Continue training with domain data to stabilize the new embeddings.

**Multilingual balance** is another hard problem. If the tokenizer is trained on raw multilingual text, high-resource languages consume vocabulary slots and low-resource languages degrade into byte fragments. Multilingual tokenizers should sample languages with smoothing or upsampling and set character coverage high enough to include basic scripts.

**Listing 6-3: SentencePiece multilingual tokenizer configuration**

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="multilingual_corpus_balanced.txt",
    model_prefix="tokenizer_multilingual_100k",
    vocab_size=100_000,
    model_type="bpe",
    character_coverage=0.9999,
    byte_fallback=True,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    input_sentence_size=20_000_000,
    shuffle_input_sentence=True,
)
```

### 6.2.3 Data Formats and Serialization

Format choice can change DataLoader throughput by orders of magnitude.

**Table 6-1: Data formats, compression, and access patterns**

| Format | Type | Sequential read | Random access | Compression | Cross-framework support | Typical use |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| JSONL | Text lines | Slow due to parsing | No | External `.gz` / `.zst` | Excellent | Exchange and debugging |
| Parquet | Columnar binary | Fast with column pruning | Row group level | Snappy / Zstd | Strong | Batch analytics and cleaning output |
| Arrow / Feather | Binary | Very fast | Yes | LZ4 / Zstd | Good | CPU-to-GPU intermediate layers |
| MDS | Binary shards | Very fast | Shard level | Zstd | Good with streaming datasets | LLM pre-training |
| WebDataset | Tar shards | Fast streaming | Shard level | Internal file compression | Good | Multimodal training |
| Raw `.bin` token IDs | Integer binary | Extremely fast | Byte-offset access | No | Custom | Very large pre-training |

For LLM pre-training, MDS is a strong default because it is designed for streaming multi-node reads, shard-level shuffle, and object-store access. Raw `.bin` plus `np.memmap` is also extremely fast on local NVMe and is used by Megatron-LM-style pipelines.

### 6.2.4 Shard Strategy and Global Shuffle

Datasets should be split into many similarly sized shards rather than one large file. A practical shard size is 256 MB to 1 GB. Smaller shards create too much metadata overhead; larger shards create load imbalance and make corruption more expensive.

Global shuffle prevents local distribution shift. If all data from one source appears contiguously, training sees long stretches of one style or domain. Distributed global shuffle needs explicit support; MDS provides cross-shard streaming shuffle buffers, while custom systems must coordinate shard order and sample order across ranks.

---

## 6.3 Packing, Mixing, and Curriculum

### 6.3.1 Sequence Packing: Removing the Padding Tax

In a standard batch, all samples are padded to the same length. When the corpus contains many short documents, 30-50% of compute can be spent on padding tokens. Sequence packing concatenates multiple documents into one fixed-length sequence separated by EOS markers. Attention masks should prevent cross-document leakage when the training objective requires separation.

**Listing 6-4: Greedy sequence packing**

```python
def greedy_pack_sequences(
    token_id_lists: list[list[int]],
    max_seq_len: int = 4096,
    eos_token_id: int = 2,
) -> list[dict]:
    packed = []
    current_seq = []
    current_mask = []
    doc_count = 0

    for token_ids in token_id_lists:
        token_ids = token_ids + [eos_token_id]
        if len(current_seq) + len(token_ids) > max_seq_len:
            if current_seq:
                pad_len = max_seq_len - len(current_seq)
                packed.append({
                    "input_ids": current_seq + [0] * pad_len,
                    "attention_mask": current_mask + [0] * pad_len,
                    "num_docs": doc_count,
                })
            current_seq = []
            current_mask = []
            doc_count = 0

        current_seq.extend(token_ids)
        current_mask.extend([1] * len(token_ids))
        doc_count += 1

    if current_seq:
        pad_len = max_seq_len - len(current_seq)
        packed.append({
            "input_ids": current_seq + [0] * pad_len,
            "attention_mask": current_mask + [0] * pad_len,
            "num_docs": doc_count,
        })
    return packed
```

For corpora with average document length below 512 tokens, packing can substantially increase effective token throughput. The exact benefit depends on the length distribution and implementation.

### 6.3.2 Multi-source Mixing: Temperature and Fixed Ratios

When data comes from web, code, papers, books, and enterprise corpora, sampling ratios shape capability. Temperature sampling smooths source sizes:

$$p_i = \frac{n_i^{1/T}}{\sum_j n_j^{1/T}}$$

At $T=1$, sampling follows raw volume. As $T$ rises, smaller sources receive more weight. A value around $T=2$ is common in multilingual and multi-domain settings, but business-specific models often use fixed ratios to directly express product priorities.

**Table 6-2: Mixing strategy comparison**

| Strategy | Principle | Strength | Weakness | Use case |
| :--- | :--- | :--- | :--- | :--- |
| Proportional sampling | Follow source size | Natural distribution | Small sources disappear | Early general pre-training |
| Uniform sampling | Equal source probability | Covers all sources | Distorts distribution | Coverage experiments |
| Temperature sampling | Smooth source sizes | Balances diversity and scale | Needs tuning | Multilingual or multi-domain corpora |
| Fixed-ratio mixing | Manual recipe | Direct business control | Requires expert design | Custom product models |
| Curriculum | Change mix over time | Stable capability shaping | More complex | Long training runs |

### 6.3.3 Curriculum Learning

Curriculum learning changes the data recipe during training. Early phases use shorter, cleaner, and broader data to establish language and world knowledge. Later phases introduce longer, denser, or harder data such as code, math, scientific text, and high-quality books. LLaMA 3's cooldown upweighting of curated data can be viewed as a data-quality curriculum.

Difficulty can be measured by sequence length, perplexity, quality tier, domain complexity, or human rating. Curriculum should be tested with small runs because an overly narrow late-stage mix can improve target tasks while harming general capability.

---

## 6.4 Efficient Loading, Caching, and Throughput Diagnosis

### 6.4.1 Key DataLoader Configuration

PyTorch `DataLoader` has several parameters with direct throughput impact.

**`num_workers`** controls parallel reader processes. A rough starting point is 4-8 workers per GPU, but the optimum depends on CPU cores, storage, and preprocessing.

**`pin_memory=True`** allocates pinned CPU memory, enabling faster CPU-to-GPU transfer through DMA. This can reduce host-to-device transfer time for large batches.

**`prefetch_factor`** controls how many batches each worker prepares in advance. Increasing it can hide disk latency but consumes more CPU memory.

**`persistent_workers=True`** avoids worker restart overhead between epochs.

**Listing 6-5: MosaicML Streaming Dataset DataLoader configuration**

```python
from torch.utils.data import DataLoader
from streaming import StreamingDataset

dataset = StreamingDataset(
    local="./data/shards/",
    remote="s3://my-bucket/shards/",
    shuffle=True,
    shuffle_seed=42,
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=12,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
)
```

**Listing 6-6: Memmap token-ID dataset**

```python
import numpy as np
import torch


class MemmapDataset(torch.utils.data.Dataset):
    def __init__(self, bin_path: str, seq_len: int = 4096):
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)
        self.num_samples = self.total_tokens // self.seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len]
        return torch.from_numpy(chunk.astype(np.int64))
```

### 6.4.2 Three-step Throughput Diagnosis

![Figure 6-2: Throughput bottleneck diagnosis flow](../../images/part2/io_bottleneck_diagnosis_flow.png)

*Figure 6-2: Throughput bottleneck diagnosis flow. Starting from low GPU utilization, the decision tree locates disk I/O, CPU preprocessing, and PCIe transfer bottlenecks.*

**Step 1: Confirm whether GPUs are waiting for data.** Use `nvidia-smi dmon -s u`, MFU metrics, and training-step timing. Periodic drops to zero SM utilization usually indicate waiting.

**Step 2: Locate the I/O layer.** Use `iostat -x 1`. If disk `%util` stays above 80%, storage is saturated. Use `top` or `htop` to see whether DataLoader workers are CPU-bound from decompression or online tokenization.

**Step 3: Inspect host-to-device transfer.** Use PyTorch Profiler and examine `cudaMemcpyH2D`. If transfer time is more than 10% of kernel execution time, enable pinned memory and fix tensor layout.

### 6.4.3 Pre-training Smoke Test

Before launching a long run, execute a short smoke test with the real shards and real DataLoader configuration for 100-200 training steps. Check:

- **DataLoader throughput:** tokens per second should match the expected target.
- **GPU utilization / MFU:** utilization should be stable and close to the target range.
- **Initial loss:** for a randomly initialized LLM, loss should roughly match `ln(vocab_size)`.
- **No crashes:** no worker crashes, CUDA OOMs, corrupt shards, or shape mismatches.

This 30-minute test catches most configuration and throughput failures before the expensive run begins.

### 6.4.4 Multi-node Distributed Reading

At multi-node scale, every rank must read efficiently without duplicating data or breaking shuffle. A common bad pattern is mounting one shared NFS volume on all nodes. It is easy to configure but quickly saturates the NFS server and introduces high random-access latency.

**Option 1: local SSD plus pre-staging.** Copy shards to each node's NVMe SSD before training. This gives excellent I/O performance but costs storage and preparation time. Shards should be globally shuffled and statically distributed across nodes.

**Option 2: streaming from S3/GCS.** Store the dataset in object storage and let each node download shards on demand into a local cache. This avoids full pre-copy and supports elastic nodes, but requires stable network bandwidth and well-sized shards.

**Listing 6-7: Multi-node distributed DataLoader configuration**

```python
import torch.distributed as dist
from torch.utils.data import DataLoader
from streaming import StreamingDataset

rank = dist.get_rank()
world_size = dist.get_world_size()

dataset = StreamingDataset(
    local=f"/nvme/cache/rank_{rank}/",
    remote="s3://my-bucket/pretrain_shards/",
    shuffle=True,
    shuffle_seed=42,
    num_canonical_nodes=world_size,
)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)
```

For custom datasets, use `DistributedSampler` and call `sampler.set_epoch(epoch)` at every epoch to maintain correct shuffle behavior. During the smoke test, aggregate token counts across ranks with `dist.all_reduce`; if counts differ by more than a few percent, the distribution logic is wrong.

---

## 6.5 Engineering Case and Performance Checklist

### Figure and Case

The opening case illustrates the typical migration path from runtime tokenization and compressed JSONL to offline tokenization plus binary shards. The key gain is not only speed; it is diagnostic clarity. Once token IDs are precomputed and stored in predictable shards, throughput can be measured at each layer.

### Case: Migrating From JSONL + Online Tokenization to MDS + Offline Tokenization

A team began with cleaned JSONL because it was easy to inspect. For experimentation, this was reasonable. For pre-training, it was expensive: each batch required gzip decompression, JSON parsing, tokenizer execution, sequence construction, and tensor conversion. After migration to offline tokenization and MDS shards, the DataLoader only needed shard streaming, decompression, and tensor assembly. GPU utilization increased from below 40% to roughly 88% in the composite case.

### 6.5.1 Input Pipeline Optimization Checklist

- Tokenize offline before full-scale pre-training.
- Store token IDs in binary or streaming-friendly shards.
- Keep shard sizes between roughly 256 MB and 1 GB.
- Avoid shared NFS for large multi-node training.
- Enable `pin_memory`, tune `num_workers`, and use `persistent_workers`.
- Run a smoke test with the real dataset and full DataLoader configuration.
- Track token throughput per rank, not just global samples per second.
- Preserve source and quality metadata after tokenization so mixing remains controllable.
- Use packing when short documents create high padding ratios.
- Keep the tokenizer immutable once large-scale training begins.

## Chapter Summary

The input pipeline is the bridge between cleaned data and model training. Poor choices in tokenizer design, data format, sharding, online preprocessing, or distributed reading can waste large amounts of compute. A robust pipeline uses a stable tokenizer, offline tokenization, efficient shard formats, sequence packing, controlled mixing, tuned DataLoader settings, smoke tests, and rank-aware distributed reading.

## References

- Dubey, A. et al. (2024). The Llama 3 Herd of Models.
- Kudo, T., and Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer.
- MosaicML Research. (2022). StreamingDataset and Mosaic Data Shard.
- Sennrich, R., Haddow, B., and Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units.
- Xue, L. et al. (2021). mT5: A massively multilingual pre-trained text-to-text transformer.
