# 第6章 分词、序列化与高效加载

## 开篇：一次"数据管线比模型慢"的训练事故

某团队在启动 13B 参数模型的预训练时，申请了 64 张 A100 GPU 组成的计算集群，按照 H100 价格的折算，这套集群的租用成本约为每小时 1.6 万元人民币。训练在第 2 小时开始出现异常：`nvidia-smi` 显示 GPU 利用率稳定在 **38%** 左右，而不是预期的 85% 以上。初步排查认为是模型配置问题——直到工程师打开 `iostat` 监控，才发现磁盘 I/O 已经跑满：读速度维持在磁盘上限的 100%，但 DataLoader 仍然追不上 GPU 的消费速度。

根本原因很快被定位：团队将清洗好的语料存放在普通 HDD 阵列上，每个 shard 是一个压缩的 `.jsonl.gz` 文件，DataLoader 需要在运行时实时解压和分词，导致 CPU 和磁盘双双成为瓶颈。最终，该团队将训练暂停了整整 18 个小时，重新对所有数据进行离线分词和序列化为 MDS 格式（Mosaic Data Shard），迁移到 NVMe SSD 存储，才将 GPU 利用率恢复到 88%。

**代价：约 3 万元算力浪费，外加 18 小时的工程延误。** 而这个问题完全可以在训练启动前 1 天的 smoke test 阶段被发现。

这个案例说明了本章的核心命题：**数据输入管道（Input Pipeline）的效率，是预训练中最容易被低估、一旦出问题代价最高的工程环节之一。** 它处于"清洗已完成，训练还没开始"的灰色地带——既不属于数据工程的关注重点，也不属于训练系统的调优范围，结果往往被双方忽视，直到真实的算力爆炸才被迫正视。

---

## 6.1 为什么输入管道决定训练上限

### 6.1.1 GPU 空转的隐性成本

在大规模预训练场景下，GPU 集群的租用成本通常以小时计，且居高不下（H100 SXM 单卡按需价格约 $3-4/小时，80 卡集群每小时成本超过 240 美元）。在这种成本结构下，"GPU 利用率"不再只是一个性能指标，而是直接换算为财务损耗的经济指标——每降低 10% 的 GPU 利用率，就意味着有 10% 的算力支出被浪费在"等待数据"上，没有产生任何实际的梯度更新。

理论上，一个配置合理的 LLM 训练系统，其 GPU 利用率（更精确的指标是 **Model FLOPS Utilization，MFU**）应当保持在 40-50% 以上（考虑到通信和计算重叠后，顶级基础设施也很少超过 60%）。如果 MFU 持续低于 30%，几乎可以断定数据管线是瓶颈之一。

### 6.1.2 从数据格式到 GPU 的全链路延迟拆解

数据从磁盘到 GPU 显存，需要经历以下环节，每个环节都可能成为瓶颈：

**磁盘读取**：从 HDD/SSD/网络存储（NFS/S3）读取 shard 文件的原始字节。HDD 顺序读速度约 200MB/s，NVMe SSD 可达 5-7GB/s，S3 通过多线程并发可达 2-5GB/s。这是大多数 I/O 瓶颈的第一发生源。

**解压与反序列化**：如果数据以 `.gz` 或 `.zst` 压缩格式存储，需要 CPU 解压；如果存储为 `.jsonl` 等文本格式，还需要 JSON 解析。这两步都是计算密集型的 CPU 操作，在 DataLoader 的 worker 进程中占用大量时间。

**在线分词（如果未离线分词）**：在 DataLoader 时实时对文本进行分词，是最耗 CPU 的操作之一。一个 `tiktoken` 或 SentencePiece 分词器处理单条 1000 字符的文本约需 0.5-2ms，在 8 worker 的 DataLoader 中并发处理，足以成为显著瓶颈。

**CPU 到 GPU 传输（PCIe/NVLink）**：将组装好的 tensor batch 从 CPU 内存传输到 GPU 显存。PCIe 4.0 的峰值带宽约 32GB/s，但不合理的 tensor 布局（非连续内存）会导致实际传输效率大幅下降。

理解这条链路，是做出正确优化决策的前提。

---

## 6.2 分词、序列化与数据格式权衡

### 6.2.1 分词算法：三大流派的工程选型

分词（Tokenization）是训练输入管道的起点，也是整个输入处理链路中唯一具有"不可逆"特性的环节——一旦词表和分词模型确定，就难以在不重新分词全部数据的前提下更换。因此，词表选型决策必须在正式开始大规模分词之前慎重做出。

目前主流大模型采用的分词算法以三种为主：

**BPE（Byte Pair Encoding）** 是最广泛使用的算法，GPT 系列（包括 ChatGPT、GPT-4）均基于此。其核心思想是从字符级别出发，反复合并出现频率最高的相邻 token 对，直到词表达到目标大小。BPE 的字节级变体（Byte-level BPE，如 GPT-2 的 tiktoken）通过将原始字节而非 Unicode 字符作为起始单元，彻底解决了 OOV 问题，被 LLaMA 2/3、Mistral 等模型广泛采用。

**SentencePiece（Unigram）** 是 Google 推广的方法，T5、XLNet、mT5 等模型使用此方案。Unigram 语言模型从大词表出发，逐步删除最不重要的 token，直到达到目标词表大小，通常能取得更高的压缩率和更平滑的 token 概率分布。SentencePiece 库还原生支持无预分词（无空格依赖）的方式，对中文、日文等无显式词边界的语言更为友好。

**WordPiece** 是 BERT 的分词方案，与 BPE 类似，但以最大似然为合并标准而非频率。目前较少在新的 LLM 预训练中使用，但仍在大量 BERT 系微调场景中作为历史遗产延续。

对于中文大模型，推荐以 **Byte-level BPE**（tiktoken 实现）为基础方案，词表大小建议在 **64K-100K** 之间——这一区间在中文字符覆盖率（中文汉字约 5 万字，基础常用字约 3500 字）和嵌入矩阵参数量之间取得了合理平衡。词表过小（32K）会导致大量中文汉字被切分为字节级别的多个 token，严重增加序列长度；词表过大（200K+）则会使嵌入矩阵参数量过于庞大，影响训练效率。

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

**词表大小的权衡**是首要决策。较大的词表（100K-150K）能够将更多高频词和领域专业术语保留为单个 token，减少序列长度，降低 Transformer 的计算量（因为 attention 复杂度是序列长度的平方）；但更大的嵌入矩阵会增加参数量（词表从 32K 增至 100K，嵌入矩阵增加约 3× 参数），且稀有 token 在训练中见到的样本更少，嵌入质量较低。LLaMA-3 将词表从 LLaMA-2 的 32K 大幅扩展至 128K，并被证明在多语言理解和代码任务上有显著收益，代价是嵌入层参数约增加 12GB（对于 7B 模型而言这已是不可忽视的额外开销）。

**领域词表扩充（Domain Vocabulary Extension）** 是垂直领域大模型的常见需求。当基础词表对特定领域的专业术语覆盖不足时（如医学术语的分子式、法律术语的专有名称、代码语言的关键字组合），这些词会被切分为多个子 token，导致：一是序列长度增长，模型上下文窗口中能容纳的领域信息减少；二是模型需要从碎片化的 token 中重建语义，学习成本更高。

领域词表扩充的标准方案是：收集目标领域的大量专业文本，统计在基础词表下分词后"切割比例最高"（即最常以多 token 表示）的词汇，选取 Top-K 词汇加入词表，并对应扩展嵌入矩阵（新增 token 的嵌入向量通常以其子 token 嵌入的均值初始化，以减少训练初期的分布偏移）。LLaMA 的中文垂直领域版本（如 Chinese-LLaMA）普遍采用了在原始词表（32K）基础上新增 20K-30K 中文汉字和词汇的策略，有效提升了中文生成质量和推理效率。

**跨语言词表平衡** 对多语言基座模型（如 BLOOM、mT5、Qwen）是另一个关键挑战。若词表直接在多语言语料上联合训练，高资源语言（英文）会因其高频率占据更多词表空间，低资源语言（如泰语、阿拉伯语）的词汇被严重压缩，出现所谓"词表诅咒"（Vocabulary Curse）——这些语言的文本在模型看来是一串几乎无意义的字节碎片，导致低资源语言的理解和生成能力远低于高资源语言。

解决方案是在训练分词器时对不同语言的语料进行**上采样均衡**：将每种目标语言的训练文本采样到大致相同的 token 数量（或使用温度参数 T=3-5），确保每种语言都获得足够的词表"席位"；同时通过 SentencePiece 的 `character_coverage=0.9999` 参数，确保每种语言的基本字符集（哪怕频率极低）都被纳入词表。这是 mT5、BLOOM 等顶级多语言模型词表设计的核心工程实践。

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



### 6.2.2 数据格式与序列化：性能决定性选择

数据格式的选择对 DataLoader 的吞吐量有直接的数量级级别的影响。以下是主流格式的性能与工程权衡：

**表6-1：数据格式、压缩与访问模式对照表**

| 格式 | 类型 | 顺序读速度 | 随机访问 | 压缩支持 | 跨框架支持 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **JSONL (.jsonl)** | 文本行 | 慢（需 JSON 解析） | 不支持 | ❌（需 .gz 组合）| 极好 | 数据交换、调试 |
| **Parquet** | 列式二进制 | 快（列裁剪） | 支持（行组级） | ✅ Snappy/Zstd | 很好（Spark/pandas）| 批处理分析、Ch05 输出 |
| **Apache Arrow / Feather** | 行式二进制 | 极快（零拷贝） | 支持 | ✅ LZ4/Zstd | 好（PyArrow）| CPU→GPU 中间层 |
| **MDS（Mosaic）** | Shard二进制 | 极快 | Shard级 | ✅ Zstd | 好（Streaming Datasets）| LLM 预训练首选 |
| **WebDataset (.tar)** | Tar打包 | 快（流式）| Shard级 | ✅（内部文件压缩）| 好（Torchvision）| 多模态训练 |
| **Raw .bin（Token IDs）** | 二进制整型 | 极快（内存映射）| 支持（byte offset）| ❌ | 需自实现 | 超大规模预训练 |

对于 LLM 预训练场景，**MDS 格式**（由 MosaicML 开发，现为 Databricks 开源）是目前最推荐的选择——它专为流式多节点读取设计，支持多 GPU 节点并发无冲突读取同一数据集，内置 shuffle 缓冲区，并支持从 S3/GCS 等对象存储直接流式读取而无需完整下载。其次选择是 **Raw .bin 内存映射格式**（Megatron-LM 使用方案），将 token ID 数组直接写为二进制文件，读取时使用 `np.memmap` 进行内存映射，在本地 NVMe SSD 上读取速度接近内存速度。

### 6.2.3 Shard 策略与全局 Shuffle

数据集应当被切分为大量等大小的 shard 文件，而不是存储为单个大文件。推荐的 shard 大小在 **256MB-1GB** 之间，这一区间的选择依据是：shard 过小会导致文件元数据（文件打开、seek）的开销占比过大；shard 过大会导致跨节点分配时的负载不均衡，且单 shard 损坏会导致更大量的数据不可用。

Shuffle 是预训练数据准备中的另一个关键步骤。未经 shuffle 的数据按来源顺序排列，同一来源的数据集中出现，会导致模型在训练时遇到连续的"局部分布偏移"，影响 Loss 收敛的平滑性。全局 Shuffle（Global Shuffle）要求在所有 shard 之间进行随机打乱——这在单机上容易实现，但在分布式训练中需要专门的设计（MDS 格式内置支持跨 shard 的流式 shuffle 缓冲区，推荐使用）。

---

## 6.3 Packing、Mixing 与 Curriculum 策略

### 6.3.1 序列 Packing：消除 Padding 的"算力税"

在标准的 DataLoader 实现中，一个 batch 内的所有样本被填充（Padding）到同一长度。当训练集中存在大量短文档时（如大量 QA 对、简短代码片段），Padding token 的比例可能高达 30-50%——这意味着 30-50% 的 GPU 计算资源被浪费在对 `<pad>` token 做无效的注意力计算上。

**序列 Packing（Sequence Packing）** 是解决这一问题的标准工程手段：将多个短文档拼接在同一个序列中，用特殊的 `[EOS]` token 作为文档边界，使每个序列的有效 token 比例接近 100%。Attention Mask 中对应地在 `[EOS]` 处切断跨文档的注意力（避免文档 A 的末尾影响文档 B 的开头的 attention），保持各文档的语义独立性。

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

实践数据表明，对于包含大量短文档（平均长度 < 512 token）的训练集，启用 Packing 可以使有效 Token 吞吐量（Tokens/s）提升 **40-80%**，折算成算力等效节省相当显著。

### 6.3.2 多源混采：温度权重与领域比例控制

当训练数据来自多个异构来源（网页语料、代码、学术论文、书籍、企业数据）时，如何在训练过程中控制每种来源的采样比例，是一个直接影响模型能力分布的关键决策。

最常用的方案是**温度采样（Temperature Sampling）**：将每个来源的数据量 $n_i$ 经过温度参数 $T$ 的幂次变换后，归一化为采样权重：

$$p_i = \frac{n_i^{1/T}}{\sum_j n_j^{1/T}}$$

当 $T = 1$ 时，权重与数据量等比，大来源完全主导；当 $T \to \infty$ 时，所有来源权重趋于均匀。实践中常用 $T = 2$（mT5 的多语言采样设置），在上采样小来源的同时避免过度偏离原始数据分布。

**表6-2：采样与混采策略收益对照表**

| 混采策略 | 原理 | 优点 | 缺点 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **等比例采样** (T=1) | 按原始数据量等比 | 最接近真实数据分布 | 小来源被大来源淹没，代码/论文等稀少 | 通用语料预训练初期 |
| **均匀采样** (T→∞) | 每个来源等概率 | 充分覆盖所有来源 | 模型偏向少数来源风格，通用能力下降 | 特定覆盖率实验 |
| **温度采样** (T=2) | 对数据量做幂次平滑 | 平衡大小来源，增强多样性 | 需要调参 | 多语言、多领域混合 ✅ 推荐 |
| **固定比例混采** | 手动设定每源配比 | 完全可控，直接对应业务目标 | 需要人工设计，配比调错代价大 | 有明确业务目标的定制训练 ✅ 推荐 |
| **课程学习（Curriculum）** | 先用简单/通用数据，后引入复杂数据 | 收敛更稳，特定能力提升更高效 | 需设计难度度量，实现复杂 | 长期大规模训练 |

### 6.3.3 课程学习（Curriculum Learning）

课程学习（Curriculum Learning）是一种在训练过程中**动态调整数据配方**的策略：模型训练的早期阶段使用更"简单"（句子更短、语言更通顺、领域更通用）的数据，随着训练进行逐步引入更长、更复杂的样本。这模拟了人类学习"先易后难"的认知规律。

在工程实现上，课程学习的难度度量可以来自多个维度：token 序列长度（短→长）、困惑度分数（低困惑度→高困惑度）、质量层级（High→Medium→Low）。LLaMA-3 的技术报告明确提到，在预训练的 Cooldown 阶段大幅提升高质量精选数据（代码、数学推理、书籍）的权重，这本质上正是一种**数据质量课程**——先用海量通用数据建立广泛的世界知识，再用高质量精选数据在最后阶段强化特定能力。

---

## 6.4 高效加载、缓存与吞吐诊断

### 6.4.1 DataLoader 的关键配置

PyTorch 的 `DataLoader` 提供了多个直接影响 I/O 吞吐的参数，以下是对大规模预训练场景最有工程影响的几个：

**`num_workers`**：控制并行读取数据的子进程数量。这是最常见的调优点。一般规则是：`num_workers = 4-8 × GPU数量`，但实际最优值需要通过实验确定（过多 worker 反而会因为进程管理开销和 IPC 争用而降低吞吐）。对于从高速 NVMe SSD 读取的 MDS 格式，通常 `num_workers=8-16` 即可将磁盘利用率跑满。

**`pin_memory=True`**：启用后，DataLoader 会在 CPU 端分配固定内存（Pinned Memory）用于存放 batch，使后续的 CPU→GPU 传输可以使用 DMA（直接内存访问），显著提升 PCIe 传输效率。对于大 batch 的高频数据传输，启用 `pin_memory` 通常可以将 CPU→GPU 传输时间降低 20-30%。

**`prefetch_factor`**：每个 worker 提前预加载的 batch 数量（默认为 2）。适当增大（如 4-8）可以隐藏磁盘读取延迟，但会增加 CPU 内存占用。

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

### 6.4.2 吞吐瓶颈诊断：三步系统化排查

当 GPU 利用率不达预期时，按以下系统化步骤排查：

![图6-2：吞吐瓶颈诊断流程图](../../images/part1/io_bottleneck_diagnosis_flow.png)

*图6-2：吞吐瓶颈诊断流程图 —— 从 GPU 利用率异常出发，通过三级决策树定位磁盘 I/O 瓶颈、CPU 预处理瓶颈和 PCIe 传输瓶颈，并给出对应的修复方案。*

**Step 1 - 确认 GPU 是否在等数据**：运行 `nvidia-smi dmon -s u` 监控 SM 利用率；若 SM 利用率周期性下降到 0 且 `sm_active` 间歇性为 0，说明 GPU 正在等待。同时检查 MFU（Model FLOPS Utilization）指标，若 MFU < 30%，极大概率是数据管线瓶颈。

**Step 2 - 定位 I/O 层级**：运行 `iostat -x 1` 监控磁盘 I/O，若 `%util` 持续 > 80%，说明磁盘是瓶颈；同时用 `top` 或 `htop` 检查 DataLoader worker 进程的 CPU 占用，若 CPU 核心全满，说明在线分词/解压是瓶颈。

**Step 3 - 检查 PCIe 传输**：使用 PyTorch 的 Profiler 记录 `cudaMemcpyH2D` 的时间占比；若 H2D 传输时间超过 GPU Kernel 执行时间的 10%，说明 PCIe 传输是瓶颈，需要启用 `pin_memory` 或优化 tensor 内存布局。

### 6.4.3 训练前 Smoke Test：上线前 30 分钟的自动检验

在正式启动长周期预训练任务之前，强烈建议执行一个简短的 **Smoke Test**——用完整的数据管线（真实 shard 文件，真实 DataLoader 配置），但只运行 100-200 步训练，专门用于检验以下指标：

- **DataLoader 吞吐**：Token/s 是否达到目标值（可根据 GPU 数量和 MFU 目标预计算）
- **GPU 利用率**：是否稳定在 80% 以上  
- **Loss 初始值**：是否在合理范围内（对于 LLM，随机初始化的 Loss 约等于 ln(vocab_size)，如 vocab 为 100K 则 Loss 初始值约 11.5）
- **无异常崩溃**：DataLoader 没有 worker crash，CUDA 没有 OOM

这 30 分钟的 Smoke Test，可以发现 90% 以上的配置问题，避免开篇案例中"训练跑了 2 小时才发现管线是瓶颈"的高代价错误。

### 6.4.4 多节点分布式读取：避免"数据孤岛"

当训练规模扩展到多机多卡（如 8 台服务器 × 8 GPU = 64 GPU）时，数据加载面临单机场景不会遇到的新挑战：**如何让所有节点高效、无冲突地读取同一个数据集，同时保证全局数据分布的正确性（不重复、不遗漏、shuffle 随机性全局一致）？**

最常见的错误做法是"共享 NFS 挂载"——所有节点挂载同一个 NFS 文件系统，每个节点的 DataLoader 直接从 NFS 读取 shard。这种方案配置简单，但在大规模并发读取时，NFS 服务器很快成为带宽瓶颈（NFS 服务器的聚合带宽通常在 1-10GB/s，对 64 GPU 的集群来说几乎立刻被打满），且 NFS 随机访问的延迟远高于本地 SSD，严重影响 DataLoader 的吞吐。

**推荐方案一：本地 SSD + 数据预拷贝（Data Pre-staging）**。在训练开始前，将 shard 文件预先分发（rsync 或 S3 批量下载）到每个节点的本地 NVMe SSD 上，训练期间各节点只读本地磁盘。这种方案的 I/O 性能最佳，但需要额外的存储空间和预拷贝时间（通常需要 30 分钟至数小时，取决于数据量）。Shard 的分配策略推荐采用"静态分配+全局排列"：将所有 shard 按全局顺序打乱后，均匀分配给各节点（节点 0 取 shard 0, 8, 16..., 节点 1 取 shard 1, 9, 17...），确保每个节点的数据分片是全局 shuffle 后的等份。

**推荐方案二：MosaicML Streaming 从 S3 流式读取**。这是近年来大型团队越来越流行的方案——数据集存放在 S3/GCS 等对象存储上，每个节点在训练期间通过 `StreamingDataset` 按需下载 shard（下载一个 shard，训练完毕后删除，再下载下一个），本地磁盘只作为缓存层（缓存大小可配置）。这种方案的优势是数据集无需预先拷贝到各节点，新节点可以立即加入训练；局限是需要稳定的网络带宽（每个节点约需 1-2GB/s 的 S3 读取带宽），且网络延迟比本地 SSD 高 5-20 倍，不适合 shard 极小或对延迟敏感的场景。

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

**避免重复读取**是多节点分布式 DataLoader 的一个常见坑：如果各节点的 DataLoader 没有进行适当的 rank-aware 划分，每个节点会独立读取完整数据集，导致所有节点看到相同的数据顺序，梯度更新实际上是在重复数据上进行的——等效于 batch size 没有随着节点数增加而正确扩展。对于非 `StreamingDataset` 的自定义数据集，需要使用 `DistributedSampler`，并在每个 epoch 开始时调用 `sampler.set_epoch(epoch)` 以确保不同 epoch 之间的 shuffle 随机性。

**全局 Token 计数一致性检验**：在分布式训练中，每个 rank 实际处理的 token 数量应当几乎相同（允许 ±1 个 batch 的误差）。若各 rank 的 token 计数出现较大差异（超过 5%），说明数据分发或 DataLoader 配置存在问题，需要在 Smoke Test 阶段通过 `dist.all_reduce` 对各 rank 的 batch 计数进行汇总检验。



---

## 6.5 工程案例与性能优化清单

### 图与案例

![图6-1：训练输入管道分层图](../../images/part1/training_input_pipeline_layers.png)

*图6-1：LLM 训练输入管道分层架构 —— 从分词、序列化、数据混采、Packing 到 DataLoader GPU 馈送的五阶段完整路径，底部标注了两个最高频的瓶颈风险点（磁盘 I/O 和 CPU↔GPU 传输）。*

### 案例：从 JSONL+在线分词 到 MDS+离线分词 的迁移收益

接续开篇案例，详细记录该团队完成存储格式迁移后的量化收益对比：

**迁移前**（JSONL.gz，HDD，在线分词）：
- 磁盘读取速度（IPC 读取）：约 180MB/s（HDD 上限）
- DataLoader 吞吐：约 12,000 tokens/s（64 GPU）
- GPU 利用率：38%
- 每 1B token 处理时间：约 83,333 秒（约 23 小时）

**迁移后**（MDS，NVMe SSD，离线预分词）：
- 磁盘读取速度：约 4.5GB/s（NVMe RAID）
- DataLoader 吞吐：约 380,000 tokens/s
- GPU 利用率：88%
- 每 1B token 处理时间：约 2,632 秒（约 43 分钟）

**核心收益**：相同训练目标（1T token，7B 模型），迁移前预计耗时约 966 天 GPU 小时，迁移后约 440 天 GPU 小时，**实際算力成本降低约 54%**，工程改动耗时 18 小时（离线重分词 + 存储迁移）ROI 极高。

### 6.5.1 输入管道优化检查清单

以下是一份可直接使用的输入管道优化检查清单，建议在每个新训练任务启动前逐项核对：

**存储与格式**
- [ ] 数据以二进制格式存储（MDS / .bin memmap），而非 JSONL
- [ ] 存储在高速设备（NVMe SSD / 高性能网络存储），而非 HDD 或 NFS
- [ ] Shard 大小在 256MB-1GB 之间
- [ ] 数据已完成校验和（checksum）验证，确保无损坏 shard

**分词与序列化**
- [ ] 分词已于预处理阶段离线完成，DataLoader 不做在线分词
- [ ] 序列已打包（Packing），padding token 比例 < 5%
- [ ] 已启用全局 Shuffle（跨 shard 的随机性）
- [ ] 序列长度分布与目标 max_seq_len 匹配

**DataLoader 配置**
- [ ] `num_workers` ≥ 2 × GPU 数量
- [ ] `pin_memory=True`
- [ ] `persistent_workers=True`（避免 epoch 间重启开销）
- [ ] 已运行 Smoke Test（100 步，确认 GPU util ≥ 80%）

**监控与可观测性**
- [ ] 训练过程中持续记录 tokens/s、GPU util、DataLoader wait time
- [ ] 设置了 DataLoader 超时告警（如 batch 等待 > 10s 触发告警）
- [ ] 保留了完整的数据来源元数据，支持问题回溯

---

## 本章小结

本章以一次真实 I/O 瓶颈导致 GPU 空转 62% 的训练事故开篇，系统建立了训练输入管道的完整技术认知。我们详细梳理了分词算法选型（BPE/SentencePiece 的工程权衡）、数据格式选择（从 JSONL 到 MDS/Arrow 的性能跃升）、Packing 策略（消除 Padding 带来的 40-80% 吞吐提升）、温度采样与课程学习的混采策略（表6-2），以及系统化的 I/O 瓶颈诊断三步法（图6-2）。附录中的"输入管道优化检查清单"可直接作为生产级预训练任务的启动前核验工具。

本章与 Ch03（成本治理）的成本视角深度呼应——在 GPU 算力成本极高的预训练专案中，输入管道的工程质量差异可以直接决定数十乃至数百万人民币的算力成本节省。进入下一章，我们将视角从"如何把数据送进模型"转向"如何评价模型用这些数据学到了什么"：**第7章 数据评估、质量闭环与运营迭代**。

## 参考文献

<!-- 待补充：本章引用的论文、博客、工具与官方文档。补全策略见 publishing/citations_progress.md。 -->
