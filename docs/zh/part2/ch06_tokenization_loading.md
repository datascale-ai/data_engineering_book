# 第6章 分词、序列化与高效加载

<div class="chapter-authors">王柯（Ke Wang）</div>

## 摘要

本章讨论清洗后的文本如何被转换为可供大模型高效训练的输入管道，覆盖分词器设计、数据格式选择、序列 Packing、多源混采、DataLoader 配置、缓存策略和分布式读取。章节首先通过匿名化复合案例说明 I/O 瓶颈如何导致 GPU 空转和训练成本浪费，随后比较 BPE、WordPiece 与 SentencePiece 的工程特性，并分析词表大小、领域词表扩充和多语言平衡对训练效率与能力分布的影响。序列化部分比较 JSONL、Parquet、Arrow、MDS、WebDataset 与 memmap 等格式，强调离线分词和二进制 shard 对吞吐的作用。后半章进一步讨论 Packing、温度采样、课程学习和 Smoke Test，并给出多节点读取的 rank-aware 配置。读者应能够为不同规模的预训练任务设计稳定、可诊断、成本可控的输入管道。

## 关键词

分词；Tokenizer；序列化；DataLoader；MDS；Packing；数据混采；吞吐诊断

## 学习目标

- 能够比较 BPE、WordPiece 和 SentencePiece 在大模型输入管道中的工程取舍。
- 能够解释词表大小、领域词表扩充和多语言采样对模型训练的影响。
- 能够选择适合预训练规模的数据格式、shard 策略和离线分词方案。
- 能够通过 Smoke Test、GPU 利用率、I/O 监控和 Profiler 定位输入瓶颈。
- 能够设计多节点分布式读取方案，避免重复读取、NFS 瓶颈和全局 shuffle 失效。

## 开篇：一次"数据管线比模型慢"的训练事故

以下为匿名化复合案例，用于说明输入管道瓶颈的排查路径；其中吞吐和利用率应在目标集群上压测，不应跨项目复用。某团队在启动中等规模基座模型预训练时，训练早期出现异常：`nvidia-smi` 显示 GPU 利用率长期低于项目基线。初步排查认为是模型配置问题，直到工程师打开 `iostat`、DataLoader wait time 和 profiler 监控，才发现磁盘 I/O 与在线分词成为瓶颈，DataLoader 追不上 GPU 的消费速度。

根本原因很快被定位：团队将清洗好的语料存放在普通磁盘阵列上，每个 shard 是一个压缩的 `.jsonl.gz` 文件，DataLoader 需要在运行时实时解压和分词，导致 CPU 和磁盘双双成为瓶颈。最终，该团队暂停训练，重新对所有数据进行离线分词并序列化为 MDS 格式（Mosaic Data Shard），迁移到更高吞吐的存储介质后，GPU 利用率恢复到项目基线区间。

这类事故的代价应按真实集群单价、暂停策略、排队时间和重处理成本重新核算。更重要的是，它完全可以在训练启动前的 smoke test 阶段被发现。

这个案例说明了本章的核心命题：**数据输入管道（Input Pipeline）的效率，是预训练中最容易被低估、一旦出问题代价最高的工程环节之一。** 它处于"清洗已完成，训练还没开始"的灰色地带——既不属于数据工程的关注重点，也不属于训练系统的调优范围，结果往往被双方忽视，直到真实的算力浪费产生才被迫正视。

---

## 6.1 为什么输入管道决定训练上限

### 6.1.1 GPU 空转的隐性成本

在大规模预训练场景下，GPU 集群的租用成本通常以小时计，且价格会因地区、供应商、实例规格和采购协议大幅波动。在这种成本结构下，"GPU 利用率"不再只是一个性能指标，而是直接换算为财务损耗的经济指标：任何由数据等待造成的低利用率，都会拉长达到同等有效训练 token 所需的时间。

更精确的指标是 **Model FLOPS Utilization，MFU**。MFU 受模型结构、并行策略、通信拓扑、batch size、混合精度和 kernel 实现共同影响，不能用单一阈值判断好坏。如果 MFU 或 GPU utilization 长期低于项目历史基线，就需要同时排查 DataLoader wait time、存储吞吐、网络读取和在线预处理。

### 6.1.2 从数据格式到 GPU 的全链路延迟拆解

数据从磁盘到 GPU 显存，需要经历以下环节，每个环节都可能成为瓶颈：

**磁盘读取**：从 HDD/SSD/网络存储（NFS/S3）读取 shard 文件的原始字节。不同介质与云厂商对象存储的吞吐差异很大，实际速度还会受到并发数、文件大小、网络拓扑和缓存命中率影响。因此，训练前应在目标集群上用与生产 shard 相同的文件格式做压测，而不是直接套用通用带宽数字。

**解压与反序列化**：如果数据以 `.gz` 或 `.zst` 压缩格式存储，需要 CPU 解压；如果存储为 `.jsonl` 等文本格式，还需要 JSON 解析。这两步都是计算密集型的 CPU 操作，在 DataLoader 的 worker 进程中占用大量时间。

**在线分词（如果未离线分词）**：在 DataLoader 时实时对文本进行分词，是最耗 CPU 的操作之一。`tiktoken` 或 SentencePiece 的单条文本处理耗时取决于字符串长度、语言分布、词表大小、CPU 型号和 worker 并发数；在高吞吐训练中，它足以成为显著瓶颈，必须通过 profiler 而不是经验数字确认。

**CPU 到 GPU 传输（PCIe/NVLink）**：将组装好的 tensor batch 从 CPU 内存传输到 GPU 显存。总线规格给出的峰值带宽通常高于训练中的有效吞吐；不合理的 tensor 布局（非连续内存）、同步拷贝和 worker 调度都会导致实际传输效率大幅下降。

理解这条链路，是做出正确优化决策的前提。

---

## 6.2 分词、序列化与数据格式权衡

### 6.2.1 分词算法：三大流派的工程选型

分词（Tokenization）是训练输入管道的起点，也是整个输入处理链路中唯一具有"不可逆"特性的环节——一旦词表和分词模型确定，就难以在不重新分词全部数据的前提下更换。因此，词表选型决策必须在正式开始大规模分词之前慎重做出。

目前主流大模型采用的分词算法以三种为主：

**BPE（Byte Pair Encoding）** (Sennrich et al. 2016) 是最广泛使用的算法，GPT 系列（包括 ChatGPT、GPT-4）均基于此。其核心思想是从字符（或字节）级别出发，反复合并出现频率最高的相邻 token 对。

代码清单6-1展示了 BPE 合并过程的简化伪代码。

*代码清单6-1：BPE 合并过程简化伪代码。该片段用于解释合并思想，非生产级 tokenizer 训练实现。* 注：传统 BPE 不感知词素边界；2025 年 MorphBPE (Asgari et al. 2025) 探索通过约束合并规则不跨越词素边界，改善形态丰富语言上的分词效率与训练表现。

```python
# BPE 合并原理伪代码
def bpe_train(corpus, num_merges):
    vocab = get_initial_characters(corpus)
    for _ in range(num_merges):
        pairs = get_stats(vocab)  # 统计所有相邻 token 对的频率
        best = max(pairs, key=pairs.get) # 选出频率最高的一对
        vocab = merge_vocab(best, vocab) # 将其合并为一个新的 token
    return vocab
```
BPE 的字节级变体（Byte-level BPE，如 GPT-2 的 tiktoken）通过将原始字节而非 Unicode 字符作为起始单元，显著降低了 `<UNK>` 式未登录词风险，被 LLaMA 2/3、Mistral 等模型广泛采用。

**WordPiece** 是 BERT 的分词方案，与 BPE 较为相似，但合并标准并非绝对频率，而是**基于语言模型的最大似然估计（Likelihood）**。WordPiece 在合并 $A$ 和 $B$ 时，考察的是 $\frac{P(AB)}{P(A)P(B)}$ 的得分（类似互信息）。这意味着如果 $A$ 和 $B$ 各自单独出现的概率较低，但它们一起出现的概率较高，WordPiece 会倾向于将它们合并。

**OOV（Out-of-Vocabulary）危机与未登录词问题**：
在传统的基于词级（Word-level）的旧时代分词器中，如果遇到未被记录在词表中的生僻字或罕见词，模型通常会抛出一个代表未知的 `<UNK>`（Out-of-Vocabulary）占位符。这在医学、法律等专业领域风险很高：一段含有复杂化学式的文本可能被大量 `<UNK>` 稀释。而 BPE 和 WordPiece 这种基于 Subword 的方案，在遇到未见过的单词时，会继续向下拆分为更基础的子词甚至单字母/单字节。字节级方案还可以退化到字节表示，从而减少信息被 `<UNK>` 直接替代的情况；代价是序列长度可能明显增加，且稀有字节片段的语义学习仍需要足够样本支持。

**SentencePiece（Unigram）** (Kudo and Richardson 2018) 则是 Google 推广的方案，它不走“从小到大合并”的路线，而是“从大到小裁剪”。Unigram 从超大基础词表出发，逐步计算并删除对整体语料似然度下降最小的 token。它对中文、日文等无显式词边界的语言更为友好。

对于中文大模型，推荐以 **Byte-level BPE**（tiktoken 实现）为基础方案，词表大小建议在 **64K-100K** 之间——这一区间在中文字符覆盖率（中文汉字约 5 万字，基础常用字约 3500 字）和嵌入矩阵参数量之间取得了合理平衡。词表过小（32K）会导致大量中文汉字被切分为字节级别的多个 token，严重增加序列长度；词表过大（200K+）则会使嵌入矩阵参数量过于庞大，影响训练效率。

代码清单6-2展示了使用 `tiktoken` 进行离线批量分词的示意实现。

*代码清单6-2：离线批量分词示意代码。生产环境应补充分片校验、失败重试、词表版本记录和输出一致性检查。*

```python
# 使用 tiktoken 进行离线批量分词（推荐用于预处理阶段）
import tiktoken, json
from pathlib import Path

enc = tiktoken.get_encoding("cl100k_base")   # GPT-4 的基础 BPE 词表

def tokenize_document(doc: dict, max_length: int = 4096) -> dict | None:
    """
    对单条文档进行分词，并附加元数据。
    返回 None 表示文档在分词后长度过短，不纳入训练集。
    """
    token_ids = enc.encode(doc["text"], disallowed_special=())
    if len(token_ids) < 64:        # 过短文档过滤
        return None
    return {
        "token_ids": token_ids,
        "num_tokens": len(token_ids),
        "source": doc.get("source", "unknown"),
        "quality_tier": doc.get("quality_tier", "medium"),
    }
```

### 6.2.2 词表设计与领域适配：不只是"够用就好"

词表（Vocabulary）是分词器的核心产出，也是大模型整体架构中唯一在训练开始后几乎无法更改的组件。一旦词表确定，后续所有的数据处理、模型嵌入矩阵、输出 logit 层都与之强绑定——更换词表意味着重新分词所有训练数据、重新初始化嵌入矩阵（丢失预训练权重的嵌入部分），代价极高。因此，词表设计决策必须在整个工程启动之前完成，而不是在训练中途发现问题再回来修正。

**词表大小的权衡**是首要决策。较大的词表（如 100K 量级）能够将更多高频词和领域专业术语保留为单个 token，减少序列长度，降低 Transformer 的计算量（因为 attention 复杂度是序列长度的平方）；但更大的嵌入矩阵会增加参数量，且稀有 token 在训练中见到的样本更少，嵌入质量较低。LLaMA-3 (Grattafiori et al. 2024) 将词表从 LLaMA-2 的 32K 大幅扩展至 128K，并在技术报告中把更大的词表列为改进多语言与代码能力的重要设计之一。嵌入层新增参数可按“新增 token 数 × hidden size”估算，显存开销还取决于参数精度、是否 tied embedding 以及优化器状态，不能脱离模型配置给出固定 GB 数。

**领域词表扩充（Domain Vocabulary Extension）** 是垂直领域大模型的常见需求。当基础词表对特定领域的专业术语覆盖不足时（如医学术语的分子式、法律术语的专有名称、代码语言的关键字组合），这些词会被切分为多个子 token，导致：一是序列长度增长，模型上下文窗口中能容纳的领域信息减少；二是模型需要从碎片化的 token 中重建语义，学习成本更高。

领域词表扩充的常见方案是：收集目标领域的大量专业文本，统计在基础词表下分词后"切割比例最高"（即最常以多 token 表示）的词汇，选取 Top-K 词汇加入词表，并对应扩展嵌入矩阵（新增 token 的嵌入向量通常以其子 token 嵌入的均值初始化，以减少训练初期的分布偏移）。一些中文 LLaMA 衍生项目采用过在原始词表基础上新增中文汉字和词汇的策略；是否能提升中文生成质量和推理效率，需要通过目标任务的消融实验确认。

**跨语言词表平衡** 对多语言基座模型（如 BLOOM、mT5、Qwen）是另一个关键挑战。若词表直接在多语言语料上联合训练，高资源语言（英文）会因其高频率占据更多词表空间，低资源语言（如泰语、阿拉伯语）的词汇被严重压缩，出现所谓"词表诅咒"（Vocabulary Curse）——这些语言的文本在模型看来是一串几乎无意义的字节碎片，导致低资源语言的理解和生成能力远低于高资源语言。

解决方案是在训练分词器时对不同语言的语料进行**上采样均衡**：将每种目标语言的训练文本采样到大致相同的 token 数量（或使用温度参数 T=3-5），确保每种语言都获得足够的词表"席位"；同时通过 SentencePiece 的 `character_coverage=0.9999` 参数，确保每种语言的基本字符集（哪怕频率很低）都被纳入词表。这是 mT5、BLOOM 等多语言模型词表设计中的常见工程实践。

代码清单6-3展示了 SentencePiece 多语言词表训练的示意配置。

*代码清单6-3：SentencePiece 多语言词表训练配置片段。参数仅为配置示例，生产环境应通过语种覆盖、OOV/UNK 率和下游评测共同调参。*

```python
# SentencePiece 多语言词表训练（示意）
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="multilingual_corpus_balanced.txt",
    model_prefix="tokenizer_multilingual_100k",
    vocab_size=100_000,
    model_type="bpe",
    character_coverage=0.9999,   # 确保每种语言基本字符集全覆盖
    byte_fallback=True,           # Unicode 字符不在词表时退化为字节表示
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    # 对小语种上采样（权重文件中指定每行的采样权重）
    input_sentence_size=20_000_000,
    shuffle_input_sentence=True,
)
```



### 6.2.3 数据格式与序列化：性能决定性选择

数据格式的选择对 DataLoader 的吞吐量有直接的数量级级别的影响。以下是主流格式的性能与工程权衡：

*表6-1：数据格式、压缩与访问模式对照表。来源：本书整理，性能表现需以目标硬件、存储后端、压缩方式和 DataLoader 实现压测为准。*

| 格式 | 类型 | 顺序读速度 | 随机访问 | 压缩支持 | 跨框架支持 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **JSONL (.jsonl)** | 文本行 | 慢（需 JSON 解析） | 不支持 | ✗（需 .gz 组合）| 极好 | 数据交换、调试 |
| **Parquet** | 列式二进制 | 快（列裁剪） | 支持（行组级） | √ Snappy/Zstd | 很好（Spark/pandas）| 批处理分析、第5章输出 |
| **Apache Arrow / Feather** | 行式二进制 | 极快（零拷贝） | 支持 | √ LZ4/Zstd | 好（PyArrow）| CPU→GPU 中间层 |
| **MDS（Mosaic）** | Shard二进制 | 极快 | Shard级 | √ Zstd | 好（Streaming Datasets）| 流式多节点训练的候选方案 |
| **WebDataset (.tar)** | Tar打包 | 快（流式）| Shard级 | √（内部文件压缩）| 好（Torchvision）| 多模态训练 |
| **Raw .bin（Token IDs）** | 二进制整型 | 极快（内存映射）| 支持（byte offset）| ✗ | 需自实现 | 超大规模预训练 |

对于需要对象存储流式读取和多节点弹性扩缩的 LLM 预训练场景，**MDS 格式** (Mosaic AI Research 2022) 是值得优先评估的候选方案——它专为流式多节点读取设计，支持多 GPU 节点并发读取同一数据集，内置 shuffle 缓冲区，并支持从 S3/GCS 等对象存储直接流式读取而无需完整下载。若训练集已经完成离线分词且主要依赖本地 NVMe，**Raw .bin 内存映射格式**（Megatron-LM 使用方案）同样常见：它将 token ID 数组直接写为二进制文件，读取时使用 `np.memmap` 进行内存映射，在本地高性能存储上可获得很低的解析开销。

### 6.2.4 Shard 策略与全局 Shuffle

数据集应当被切分为大量等大小的 shard 文件，而不是存储为单个大文件。Shard 大小需要结合对象存储请求开销、文件系统元数据压力、单节点缓存容量和跨节点负载均衡压测决定：shard 过小会导致文件元数据（文件打开、seek）的开销占比过大；shard 过大会导致跨节点分配时的负载不均衡，且单 shard 损坏会导致更大量的数据不可用。

Shuffle 是预训练数据准备中的另一个关键步骤。未经 shuffle 的数据按来源顺序排列，同一来源的数据集中出现，会导致模型在训练时遇到连续的"局部分布偏移"，影响 Loss 收敛的平滑性。全局 Shuffle（Global Shuffle）要求在所有 shard 之间进行随机打乱——这在单机上容易实现，但在分布式训练中需要专门的设计（MDS 格式内置支持跨 shard 的流式 shuffle 缓冲区，推荐使用）。

---

## 6.3 Packing、Mixing 与 Curriculum 策略

### 6.3.1 序列 Packing：消除 Padding 的"算力税"

在标准的 DataLoader 实现中，一个 batch 内的所有样本被填充（Padding）到同一长度。当训练集中存在大量短文档时（如大量 QA 对、简短代码片段），Padding token 的比例可能显著升高，这意味着一部分 GPU 计算资源被浪费在对 `<pad>` token 做无效的注意力计算上。

**序列 Packing（Sequence Packing）** 是解决这一问题的标准工程手段：将多个短文档拼接在同一个序列中，用特殊的 `[EOS]` token 作为文档边界，提高每个序列中的有效 token 比例。Attention Mask 中对应地在 `[EOS]` 处切断跨文档的注意力（避免文档 A 的末尾影响文档 B 的开头的 attention），保持各文档的语义独立性。

代码清单6-4展示了贪心序列 Packing 的示意实现。

*代码清单6-4：贪心序列 Packing 示意代码。该片段展示基本策略，生产环境应补充样本边界、标签 mask 和可复现实验记录。*

```python
def greedy_pack_sequences(
    token_id_lists: list[list[int]],
    max_seq_len: int = 4096,
    eos_token_id: int = 2
) -> list[dict]:
    """
    贪心 Bin-Packing：将多个文档 token 列表打包到固定长度序列中。
    返回每个 packed 序列及其 attention mask。
    """
    packed, current_seq, current_mask = [], [], []
    doc_count = 0

    for token_ids in token_id_lists:
        token_ids = token_ids + [eos_token_id]   # 文档结束标记
        if len(current_seq) + len(token_ids) > max_seq_len:
            if current_seq:
                # 用 0 (pad) 填充到 max_seq_len
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

对于包含大量短文档的训练集，启用 Packing 通常可以提高有效 Token 吞吐量（Tokens/s）。实际收益取决于文档长度分布、max sequence length、attention mask 实现和硬件配置，应在目标数据集上用 padding ratio 与 tokens/s 共同验证。

### 6.3.2 多源混采：温度权重与领域比例控制

当训练数据来自多个异构来源（网页语料、代码、学术论文、书籍、企业数据）时，如何在训练过程中控制每种来源的采样比例，是一个直接影响模型能力分布的关键决策。

最常用的方案是**温度采样（Temperature Sampling）**：将每个来源的数据量 $n_i$ 经过温度参数 $T$ 的幂次变换后，归一化为采样权重：

$$p_i = \frac{n_i^{1/T}}{\sum_j n_j^{1/T}}$$

当 $T = 1$ 时，权重与数据量等比，大来源完全主导；当 $T \to \infty$ 时，所有来源权重趋于均匀。实践中常用 $T = 2$（mT5 (Xue et al. 2021) 的多语言采样设置），在上采样小来源的同时避免过度偏离原始数据分布。

*表6-2：采样与混采策略收益对照表。来源：本书整理，收益描述为常见模式归纳，实际效果需通过数据配方消融实验确认。*

| 混采策略 | 原理 | 优点 | 缺点 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **等比例采样** (T=1) | 按原始数据量等比 | 最接近真实数据分布 | 小来源被大来源淹没，代码/论文等稀少 | 通用语料预训练初期 |
| **均匀采样** (T→∞) | 每个来源等概率 | 充分覆盖所有来源 | 模型偏向少数来源风格，通用能力下降 | 特定覆盖率实验 |
| **温度采样** (T=2) | 对数据量做幂次平滑 | 平衡大小来源，增强多样性 | 需要调参 | 多语言、多领域混合（推荐）|
| **固定比例混采** | 手动设定每源配比 | 完全可控，直接对应业务目标 | 需要人工设计，配比调错代价大 | 有明确业务目标的定制训练（推荐）|
| **课程学习（Curriculum）** | 先用简单/通用数据，后引入复杂数据 | 收敛更稳，特定能力提升更高效 | 需设计难度度量，实现复杂 | 长期大规模训练 |

### 6.3.3 课程学习（Curriculum Learning）

课程学习（Curriculum Learning）是一种在训练过程中**动态调整数据配方**的策略：模型训练的早期阶段使用更"简单"（句子更短、语言更通顺、领域更通用）的数据，随着训练进行逐步引入更长、更复杂的样本，以模拟"先易后难"的渐进式学习路径 (Bengio et al. 2009)。

在工程实现上，课程学习的难度度量可以来自多个维度：token 序列长度（短→长）、困惑度分数（低困惑度→高困惑度）、质量层级（High→Medium→Low）。LLaMA-3 (Grattafiori et al. 2024) 的技术报告明确提到，在预训练的 Cooldown 阶段大幅提升高质量精选数据（代码、数学推理、书籍）的权重，这本质上正是一种**数据质量课程**——先用海量通用数据建立广泛的世界知识，再用高质量精选数据在最后阶段强化特定能力。

---

## 6.4 高效加载、缓存与吞吐诊断

### 6.4.1 DataLoader 的关键配置

PyTorch 的 `DataLoader` 提供了多个直接影响 I/O 吞吐的参数，以下是对大规模预训练场景最有工程影响的几个：

**`num_workers`**：控制并行读取数据的子进程数量。这是最常见的调优点。可以从每 GPU 2-4 个 worker 或每节点 8-16 个 worker 的量级开始压测，但实际最优值必须由 CPU 核数、存储介质、shard 大小、batch size 和预处理开销共同决定。过多 worker 反而会因为进程管理开销和 IPC 争用而降低吞吐。

**`pin_memory=True`**：启用后，DataLoader 会在 CPU 端分配固定内存（Pinned Memory）用于存放 batch，使后续的 CPU→GPU 传输可以使用 DMA（直接内存访问），有机会提升 PCIe 传输效率。收益取决于 batch 大小、tensor 布局、CPU 内存压力和 GPU 拷贝调度，应通过 PyTorch Profiler 验证。

**`prefetch_factor`**：每个 worker 提前预加载的 batch 数量（默认为 2）。适当增大（如 4-8）可以隐藏磁盘读取延迟，但会增加 CPU 内存占用。

代码清单6-5展示了基于 MosaicML Streaming Dataset 的 DataLoader 配置示例。

*代码清单6-5：MosaicML Streaming Dataset DataLoader 配置片段。生产环境应结合对象存储带宽、缓存策略和节点故障恢复能力压测。*

```python
from torch.utils.data import DataLoader
from streaming import StreamingDataset  # MosaicML Streaming Datasets

dataset = StreamingDataset(
    local="./data/shards/",
    remote="s3://my-bucket/shards/",   # 支持从 S3 流式读取，边下边训
    shuffle=True,
    shuffle_seed=42,
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=12,        # 根据 CPU 核数和 GPU 数量调整
    pin_memory=True,       # 启用固定内存，加速 CPU→GPU 传输
    prefetch_factor=4,     # 每 worker 预取 4 个 batch
    persistent_workers=True,  # 避免 epoch 间 worker 重启的开销
)
```

代码清单6-6展示了基于 `np.memmap` 的二进制 Token ID 数据集示意实现。

*代码清单6-6：基于 np.memmap 的 Token ID 数据集示意代码。生产环境应补充 dtype、文件完整性、索引边界和跨平台兼容性校验。*

```python
# 应对千万级小文件 IO 优化的 Memmap 二进制加载器伪代码
import numpy as np
class MemmapDataset(torch.utils.data.Dataset):
    def __init__(self, bin_path, seq_len=4096):
        # 使用 np.memmap 映射大型二进制 .bin 文件 (Raw Token IDs)
        # 完全避免了将整个数据集加载进内存，依靠 OS 的 Page Cache 极速随机读取
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.total_tokens = len(self.data)
        self.num_samples = self.total_tokens // self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        # 切片操作较轻量，由底层 C 代码和 OS 内存分页完成，吞吐较高
        chunk = self.data[start_idx : start_idx + self.seq_len]
        return torch.from_numpy(chunk.astype(np.int64))
```

### 6.4.2 吞吐瓶颈诊断：三步系统化排查

当 GPU 利用率不达预期时，按以下系统化步骤排查（见图6-1）：

![图6-1：吞吐瓶颈诊断流程图](../../images/part2/io_bottleneck_diagnosis_flow.svg)

*图6-1：吞吐瓶颈诊断流程图 —— 从 GPU 利用率异常出发，通过三级决策树定位磁盘 I/O 瓶颈、CPU 预处理瓶颈和 PCIe 传输瓶颈，并给出对应的修复方案。来源：本书自绘；Alt text：吞吐瓶颈诊断流程图，展示从 GPU 利用率异常到磁盘 I/O、CPU 预处理和 PCIe 传输排查的决策路径。*

**Step 1 - 确认 GPU 是否在等数据**：运行 `nvidia-smi dmon -s u` 监控 SM 利用率；若 SM 利用率周期性下降且 `sm_active` 间歇性接近 0，说明 GPU 可能正在等待。同时检查 MFU（Model FLOPS Utilization）和 DataLoader wait time，判断问题是否来自数据管线。

**Step 2 - 定位 I/O 层级**：运行 `iostat -x 1` 监控磁盘 I/O；如果磁盘利用率、等待时间或队列长度长期高于项目基线，说明磁盘可能是瓶颈。同时用 `top` 或 `htop` 检查 DataLoader worker 进程的 CPU 占用，若 CPU 核心长期打满，说明在线分词/解压可能是瓶颈。

**Step 3 - 检查 PCIe 传输**：使用 PyTorch 的 Profiler 记录 `cudaMemcpyH2D` 的时间占比；若 H2D 传输在 step 时间中占比异常偏高，说明 PCIe 传输可能是瓶颈，需要启用 `pin_memory` 或优化 tensor 内存布局。

### 6.4.3 训练前 Smoke Test：上线前 30 分钟的自动检验

在正式启动长周期预训练任务之前，强烈建议执行一个简短的 **Smoke Test**——用完整的数据管线（真实 shard 文件，真实 DataLoader 配置），但只运行 100-200 步训练，专门用于检验以下指标：

- **DataLoader 吞吐**：Token/s 是否达到目标值（可根据 GPU 数量和 MFU 目标预计算）
- **GPU 利用率**：是否达到项目基线，且不存在周期性等待数据的波动
- **Loss 初始值**：是否在合理范围内（对于 LLM，随机初始化的 Loss 约等于 ln(vocab_size)，如 vocab 为 100K 则 Loss 初始值约 11.5）
- **无异常崩溃**：DataLoader 没有 worker crash，CUDA 没有 OOM

这类 Smoke Test 可以在正式训练前暴露大量配置问题，避免开篇案例中"训练跑起来后才发现管线是瓶颈"的高代价错误。

### 6.4.4 多节点分布式读取：避免"数据孤岛"

当训练规模扩展到多机多卡（如 8 台服务器 × 8 GPU = 64 GPU）时，数据加载面临单机场景不会遇到的新挑战：**如何让所有节点高效、无冲突地读取同一个数据集，同时保证全局数据分布的正确性（不重复、不遗漏、shuffle 随机性全局一致）？**

最常见的错误做法是"共享 NFS 挂载"——所有节点挂载同一个 NFS 文件系统，每个节点的 DataLoader 直接从 NFS 读取 shard。这种方案配置简单，但在大规模并发读取时，NFS 服务器很快成为带宽瓶颈，且 NFS 随机访问的延迟通常高于本地 SSD，严重影响 DataLoader 的吞吐。

**推荐方案一：本地 SSD + 数据预拷贝（Data Pre-staging）**。在训练开始前，将 shard 文件预先分发（rsync 或 S3 批量下载）到每个节点的本地 NVMe SSD 上，训练期间各节点只读本地磁盘。这种方案的 I/O 性能最佳，但需要额外的存储空间和预拷贝时间，耗时取决于数据量、网络带宽和拷贝并发。Shard 的分配策略推荐采用"静态分配+全局排列"：将所有 shard 按全局顺序打乱后，均匀分配给各节点（节点 0 取 shard 0, 8, 16..., 节点 1 取 shard 1, 9, 17...），确保每个节点的数据分片是全局 shuffle 后的等份。

**推荐方案二：MosaicML Streaming 从 S3 流式读取**。这是近年来大型团队越来越流行的方案——数据集存放在 S3/GCS 等对象存储上，每个节点在训练期间通过 `StreamingDataset` 按需下载 shard（下载一个 shard，训练完毕后删除，再下载下一个），本地磁盘只作为缓存层（缓存大小可配置）。这种方案的优势是数据集无需预先拷贝到各节点，新节点可以立即加入训练；局限是需要稳定的对象存储读取带宽和低抖动网络，不适合 shard 极小或对延迟敏感的场景。

代码清单6-7展示了多节点分布式训练中的 DataLoader 配置示意。

*代码清单6-7：多节点分布式 DataLoader 配置片段。生产环境应结合 rank-aware 分片、全局 shuffle 和 token 计数一致性检验。*

```python
# 多节点分布式训练中的 DataLoader 配置
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from streaming import StreamingDataset

# 初始化分布式
rank = dist.get_rank()
world_size = dist.get_world_size()

# MosaicML Streaming：S3 流式读取，节点间自动协调 shard 分配
dataset = StreamingDataset(
    local=f"/nvme/cache/rank_{rank}/",     # 每节点独立的本地缓存目录
    remote="s3://my-bucket/pretrain_shards/",
    shuffle=True,
    shuffle_seed=42,
    num_canonical_nodes=world_size,         # 按总节点数进行 shard 分发
)

# 每个 rank 自动接收不重叠的 shard 子集
dataloader = DataLoader(
    dataset,
    batch_size=8,          # 每 GPU 的 micro-batch size
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)
```

**避免重复读取**是多节点分布式 DataLoader 的一个常见易错点：如果各节点的 DataLoader 没有进行适当的 rank-aware 划分，每个节点会独立读取完整数据集，导致所有节点看到相同的数据顺序，梯度更新实际上是在重复数据上进行的——等效于 batch size 没有随着节点数增加而正确扩展。对于非 `StreamingDataset` 的自定义数据集，需要使用 `DistributedSampler`，并在每个 epoch 开始时调用 `sampler.set_epoch(epoch)` 以确保不同 epoch 之间的 shuffle 随机性。

**全局 Token 计数一致性检验**：在分布式训练中，每个 rank 实际处理的 token 数量应当接近。若各 rank 的 token 计数出现明显差异，说明数据分发或 DataLoader 配置存在问题，需要在 Smoke Test 阶段通过 `dist.all_reduce` 对各 rank 的 batch 计数进行汇总检验。



---

## 6.5 工程案例与性能优化清单

### 图与案例

![图6-2：训练输入管道分层图](../../images/part2/training_input_pipeline_layers.svg)

*图6-2：LLM 训练输入管道分层架构 —— 从分词、序列化、数据混采、Packing 到 DataLoader GPU 馈送的五阶段完整路径，底部标注了两个最高频的瓶颈风险点（磁盘 I/O 和 CPU↔GPU 传输）。来源：本书自绘；Alt text：训练输入管道分层图，展示分词、序列化、混采、Packing、DataLoader 和 GPU 馈送之间的顺序关系。*

### 案例：从 JSONL+在线分词 到 MDS+离线分词 的迁移收益

接续开篇匿名化复合案例，下面给出完成存储格式迁移后的压测对比模板。表中不写固定数值，是为了避免将某个集群的结果误读为通用收益；实际结果取决于硬件、存储、数据格式、batch size 和框架实现。

**迁移前**（JSONL.gz，HDD，在线分词）：

- 记录磁盘读取速度、对象存储请求延迟和 DataLoader wait time。
- 记录端到端 tokens/s、GPU utilization/MFU 和 CPU 解压/分词占用。
- 记录每个 shard 的读取失败、重试和坏样本比例。

**迁移后**（MDS，NVMe SSD，离线预分词）：

- 记录同一 batch size、同一模型配置下的端到端 tokens/s。
- 记录 GPU wait time 是否下降，DataLoader 是否仍是瓶颈。
- 记录迁移带来的额外存储成本、预处理耗时和数据校验成本。

**核心收益的核算方式**：比较迁移前后的端到端 tokens/s、DataLoader wait time、GPU wait time 和预处理总成本，再折算为达到同等训练 token 所需的 GPU 小时。只有在同一模型配置、同一 batch size 和同一训练目标下，迁移前后的收益才具备可比性。

### 6.5.1 输入管道优化检查清单

以下是一份可直接使用的输入管道优化检查清单，建议在每个新训练任务启动前逐项核对：

**存储与格式**

- [ ] 数据以二进制格式存储（MDS / .bin memmap），而非 JSONL
- [ ] 存储在高速设备（NVMe SSD / 高性能网络存储），而非 HDD 或 NFS
- [ ] Shard 大小已通过目标集群压测确定
- [ ] 数据已完成校验和（checksum）验证，确保无损坏 shard

**分词与序列化**

- [ ] 分词已于预处理阶段离线完成，DataLoader 不做在线分词
- [ ] 序列已打包（Packing），padding token 比例低于项目基线
- [ ] 已启用全局 Shuffle（跨 shard 的随机性）
- [ ] 序列长度分布与目标 max_seq_len 匹配

**DataLoader 配置**

- [ ] `num_workers` 已通过 Smoke Test 调参，而非直接套用固定倍数
- [ ] `pin_memory=True`
- [ ] `persistent_workers=True`（避免 epoch 间重启开销）
- [ ] 已运行 Smoke Test，并确认 GPU util、MFU 和 DataLoader wait time 达到项目基线

**监控与可观测性**

- [ ] 训练过程中持续记录 tokens/s、GPU util、DataLoader wait time
- [ ] 设置了 DataLoader 超时告警（如 batch 等待 > 10s 触发告警）
- [ ] 保留了完整的数据来源元数据，支持问题回溯

---

## 本章小结

本章以一个匿名化复合案例说明 I/O 瓶颈如何导致 GPU 空转和成本浪费，系统建立了训练输入管道的完整技术认知。我们详细梳理了分词算法选型（BPE/SentencePiece 的工程权衡）、数据格式选择（从 JSONL 到 MDS/Arrow 的性能取舍）、Packing 策略（减少 Padding 带来的无效计算）、温度采样与课程学习的混采策略（表6-2），以及系统化的 I/O 瓶颈诊断三步法（图6-1）。附录中的"输入管道优化检查清单"可直接作为生产级预训练任务的启动前核验工具。

本章与第3章成本治理的视角呼应：在 GPU 算力成本极高的预训练项目中，输入管道的工程质量会直接影响有效训练时间和总预算。进入下一章，我们将视角从"如何把数据送进模型"转向"如何评价模型用这些数据学到了什么"：**第7章 数据评估、质量闭环与运营迭代**。

## 参考文献

Bengio Y, Louradour J, Collobert R, Weston J (2009) Curriculum Learning. In: Proceedings of the 26th Annual International Conference on Machine Learning, pp 41-48.

Brown T B, Mann B, Ryder N, Subbiah M, Kaplan J, Dhariwal P, Neelakantan A, Shyam P, Sastry G, Askell A, Agarwal S, Herbert-Voss A, Krueger G, Henighan T, Child R, Ramesh A, Ziegler D M, Wu J, Winter C, Hesse C, Chen M, Sigler E, Litwin M, Gray S, Chess B, Clark J, Berner C, McCandlish S, Radford A, Sutskever I, Amodei D (2020) Language Models are Few-Shot Learners. In: Advances in Neural Information Processing Systems 33, pp 1877-1901.

Grattafiori A, Dubey A, Jauhri A, Pandey A, Kadian A, Al-Dahle A, Letman A, Mathur A, Schelten A, Vaughan A, others (2024) The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.

Kudo T, Richardson J (2018) SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. In: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp 66-71.

Mosaic AI Research (2022) MosaicML Streaming. GitHub repository. <https://github.com/mosaicml/streaming>.

Sennrich R, Haddow B, Birch A (2016) Neural Machine Translation of Rare Words with Subword Units (BPE). In: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, pp 1715-1725.

Xue L, Constant N, Roberts A, Kale M, Al-Rfou R, Siddhant A, Barua A, Raffel C (2021) mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer. In: Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics, pp 483-498.

Asgari E, El Kheir Y, Javaheri M A S (2025) MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies. arXiv preprint arXiv:2502.00894.
