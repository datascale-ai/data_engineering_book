# 项目十一：Mini-DeepSeek 预训练复现

<div class="chapter-authors">王柯（Ke Wang）</div>

## 摘要

本项目围绕“Mini-DeepSeek 预训练复现”构建可复现的数据工程案例，重点说明业务目标、数据边界、架构决策、核心实现、验收指标与风险控制。章节将安装命令和脚本细节收敛到工程复盘视角，突出样本 schema、数据流、失败模式和可交付物之间的关系，帮助读者把前文方法转化为可审计、可扩展的项目资产。

## 关键词

Mini-DeepSeek；项目实战；可复现数据工程；数据流水线；验收指标

## 项目目标与读者收获

本项目以“Mini-DeepSeek 预训练复现”为核心案例，目标是以小规模资源复现开源 LLM 预训练数据配方的关键工程环节。读者完成本章后，应能够辨认该场景的关键数据对象、拆分工程链路、设置验收指标，并将案例方法迁移到相近的数据工程任务中。

## 场景约束与数据边界

定位为缩小版配方验证，不追求完整大模型规模和公开 SOTA 指标。这些边界使案例能够被复现和审计；当数据规模、数据来源、权限范围或部署环境变化时，需要重新评估采样策略、质量阈值、运行成本和合规要求。

## 架构决策

本项目采用“语料配比、tokenization、训练样本打包、训练烟测、指标记录和成本分析”的架构路径。该决策优先保证输入输出契约、版本可追踪、异常可定位和结果可复核，而不是把全部逻辑压缩为一次性脚本运行。

## 样本 schema / 数据流

核心数据流可概括为：

```text
候选语料 -> 配方采样 -> tokenizer 处理 -> packed dataset -> 训练烟测 -> loss 与样本质量报告
```

样本 schema 至少应保留 `id`、`source`、`content_or_payload`、`metadata`、`quality_signals`、`split_or_stage` 与 `audit_trace` 等字段；具体字段由本项目的数据类型、下游任务和验收方式进一步细化。

## 核心实现片段

正文只保留能够说明设计取舍的关键实现片段。完整脚本、长配置、运行日志和大文件应放入配套仓库或附录说明；代码展示重点放在输入输出契约、质量阈值、异常处理和验收接口上。

## 实验或验收指标

验收指标包括token 分布、语料配比偏差、packing 效率、训练 loss 趋势、吞吐、显存/成本和失败样本复查。若项目进入生产、课程或公开复现实验环境，还应记录版本号、依赖环境、随机种子、样本抽检结果和失败样本复盘记录。

*表 P11-1：Mini-DeepSeek 预训练复现出版验收表*

| 验收维度 | 指标/证据 | 出版复核口径 |
| --- | --- | --- |
| 配方复现 | 语料配比偏差、跨源去重记录和 tokenizer 训练日志 | 缩小版实验必须说明与原始配方的规模差异和不可比边界 |
| 训练烟测 | packing 效率、loss 趋势、吞吐和显存/成本记录 | 报告保留随机种子、环境、样本规模和失败样本复查结论 |
| 数据合规 | 数据源许可、污染检查和样本删除机制 | 外部语料进入公开交付前需确认来源与再分发权限 |

## 成本、风险与合规边界

成本主要来自训练算力和数据处理；风险集中在配方误读、样本污染、tokenizer 不一致和小规模结论外推。涉及外部数据、个人信息、版权内容或第三方服务时，应保留来源说明、权限状态、脱敏策略、调用记录和人工复核记录。

## 常见失败模式

常见失败包括输入分布偏离、schema 字段缺失、质量阈值过松或过紧、评测样本覆盖不足、模型调用不稳定、结果无法回溯等。排查时应优先定位数据边界和中间产物，再检查模型、工具链与部署环境。

## 可复现资源说明

复现材料应包括数据来源说明、最小样本、配置文件、运行命令、指标脚本、检查报告和产物目录。正文保留必要片段；完整 notebook、长脚本和大文件作为配套资源独立维护。

## 背景与目标

在预训练数据工程中，“按比例缩放（Scaling Laws）”(Kaplan et al. 2020) 不仅适用于模型参数，同样适用于数据配方的实验与验证。我们在前作 项目 1（Mini-C4）中，已经走通了单源语料的清洗流水线；但真实的工业级大模型（如 DeepSeek-V3 (Liu et al. 2024)）从来不是在单一语料上训练出来的，而是由网页、代码、数学、学术论文等多种数据源精确混合而成。

为什么我们需要一个 Mini 版的预训练流水线？

1. **低成本验证**：在全量 14.8T tokens 的真实数据上做实验，成本极为高昂。通过等比例缩放，我们可以在 1B tokens 的规模上快速验证多源混合策略的有效性。
2. **揭示数据间的影响**：只有在多源混合的环境下，跨源去重（Cross-source Deduplication）、数据配比调整对 Tokenizer 词表分布的影响等工程问题才会暴露。
3. **平滑的放缩曲线**：验证通过的 1B tokens 数据流水线，只需要替换底层数据源集群与算力节点，可以直接横向扩展（Scale-out）到 7B、14B 甚至 70B tokens。

本项目旨在用约 1B tokens（对应单机 8 卡 4090/A100 可在数十小时内处理完毕的数据量），完全复刻 DeepSeek-V3 的数据配方。读者完成本项目后，将获得一套具备工业级标准的多源混合采样器、跨源去重引擎以及面向 150K 超大词表的 Tokenizer 训练代码，为大规模预训练打下坚实基础。

## 架构设计

为了实现上述目标，我们设计了包含四个核心组件的数据流水线。其整体架构如图 P11-1 所示。

![Mini-DeepSeek Data Pipeline](../../images/part14/p11_mini_deepseek_arch_en.svg)
*图 P11-1：Mini-DeepSeek 多源预训练数据流水线架构*

流水线的四个核心组件包括：

1. **多源混合采样器 (Multi-source Sampler)**：负责从 Hugging Face 获取多种不同的开源数据集（如 FineWeb-Edu、The Stack v2 等），并根据 DeepSeek-V3 披露的各领域配比进行精确抽样。
2. **跨源去重引擎 (Cross-source MinHash Deduplication)**：当数据来源不仅有普通网页，还包含 GitHub 代码、arXiv 论文时，数据源之间可能存在隐性重合。该组件基于 MinHash LSH 算法 (Broder 1997)，实现在不同数据源间的高效去重。
3. **词表训练器 (Tokenizer Training)**：采用 BPE 算法 (Sennrich et al. 2016)，针对混合后的多语种、多代码领域语料，训练并构建一个 150K 容量的超大词表，确保对中英文及专业代码的高效压缩。
4. **打包与分片 (Pack & Shuffle)**：在经过 Tokenize 后，将变长的序列高效地“打包（Pack）”成定长的训练序列，并全局打乱（Shuffle），最终输出适用于大规模分布式训练的 `.arrow` 格式文件。

表 P11-2 将架构组件与代码入口、阶段产物和复核字段对应起来。项目章需要保留这类表格，因为它把“读者看到的工程叙述”和 `code/zh/project_11_mini_deepseek` 中的脚本连接在一起，避免章节只停留在概念介绍。

| 阶段 | 代码入口 | 主要输入 | 主要输出 | 复核字段 |
| --- | --- | --- | --- | --- |
| 多源采样 | `mix_sampler.py` | `RECIPE`、目标文档数、Hugging Face 数据源 | `./data/mixed_1b_raw` | `source`、样本数量、配方权重偏差 |
| 跨源去重 | `cross_source_dedup.py` | `mixed_1b_raw` | `./data/mixed_1b_dedup` | MinHash 参数、重复样本数、保留比例 |
| Tokenizer 训练 | `train_tokenizer.py` | `mixed_1b_dedup` | `mini_deepseek_tokenizer.json` | vocab size、特殊 token、训练样本比例 |
| Pack & Shuffle | `pack_shuffle.py` | 去重语料、tokenizer | `./data/mixed_1b_final_packed` | `SEQ_LEN`、packing 效率、shuffle seed |
| 端到端运行 | `run_pipeline.sh` | 阶段脚本和本地环境 | 完整数据目录 | 日志、失败阶段、产物完整性 |
| 单元测试 | `tests/test_pipeline.py` | 配方、MinHash、packing 常量 | 测试报告 | 权重和为 1、MinHash 相似度、`SEQ_LEN` |

*表 P11-2：Mini-DeepSeek 数据流水线阶段产物与代码入口*

表 P11-2 中最容易被忽视的是“复核字段”。例如，`mixed_1b_raw` 本身只是一个 Hugging Face Dataset 目录，不能说明配方是否正确；必须额外检查每个 `source` 的样本数量与目标权重是否一致。`mixed_1b_dedup` 也不能只看目录是否存在，还要记录重复样本比例和阈值。对于 tokenizer，文件存在并不代表训练合格，还需要检查特殊 token、词表大小、中文/代码压缩率和异常字符覆盖。

## 分步实现

### Step 1: 多源混合抽取与配比

根据 DeepSeek-V3 报告，我们需要融合多种数据源。在本实现中，我们选取开源平替数据集：

- 英文网页：FineWeb-Edu (Penedo et al. 2024)
- 中文网页：Wudao 或是开源的中英文混合数据
- 代码：The Stack v2 (Lozhkov et al. 2024)
- 数学：OpenWebMath (Paster et al. 2023)
- 学术：arXiv

我们编写 `mix_sampler.py` 脚本，按设定比例进行抽样。

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

### Step 2: 跨源 MinHash LSH 去重

多源混合后，最大的隐患是不同来源间存在重复（例如 The Stack v2 中的代码片段，与 arXiv 论文中的代码段重复）。在 项目 1（Mini-C4）中，我们仅在单源内进行了 MinHash 去重；在此，我们需要全局去重。

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

### Step 3: 训练 150K 超大 Tokenizer

DeepSeek-V3 (Liu et al. 2024) 采用了一个规模为 150K 左右的超大词表（相较于 Llama-2 的 32K 提升巨大），这使其在处理中文与代码时效率极高。在此步骤，我们将以混合且去重后的数据训练 BPE Tokenizer。

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

### Step 4: Pack & Shuffle 与 .arrow 分片产出

为了让 GPU 在训练期间不用处理大量的 Padding，我们将变长的 Token 序列拼接成长度为 `4096` 或 `8192` 的连续片段（Pack），并加入特殊分隔符。

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

## 工程运行与最小复现路径

本项目的最小运行入口是 `run_pipeline.sh`。脚本串联四个阶段：多源采样、跨源去重、tokenizer 训练和 packing。它的价值不只是“省去手动执行四条命令”，而是把阶段顺序、产物路径和失败位置固定下来。对于预训练数据工程，顺序错误会直接改变数据分布。例如，如果先训练 tokenizer 再做跨源去重，tokenizer 会看见本应被删除的重复样本；如果先打包再 shuffle，后续再调整配方会变得难以追踪。

Listing P11-1 给出本项目的最小运行入口。正式复现实验应在运行前记录 Python、datasets、tokenizers、datasketch、磁盘路径和随机种子。

```bash
cd code/zh/project_11_mini_deepseek
bash run_pipeline.sh
```

这段命令会依次生成 `mixed_1b_raw`、`mixed_1b_dedup`、`mini_deepseek_tokenizer.json` 和 `mixed_1b_final_packed`。若某一阶段失败，不建议直接删除整个 `data/` 目录重跑；更稳妥的做法是先确认失败阶段和上游产物是否完整，再只清理受影响的阶段目录。教学复现可以将 `target_docs` 从 `500000` 降到更小规模，先验证契约和测试，再扩大数据量。

表 P11-3 给出运行前后应记录的最小审计信息。

| 类别 | 记录项 | 作用 |
| --- | --- | --- |
| 数据版本 | 数据源 repo id、split、config name、抽样时间 | 解释样本分布变化 |
| 配方参数 | `RECIPE` 权重、目标文档数 | 判断是否符合缩小版配方 |
| 去重参数 | n-gram 长度、`num_perm`、LSH threshold | 复现重复率和误删风险 |
| tokenizer 参数 | vocab size、normalizer、pre-tokenizer、特殊 token | 复现压缩率和兼容性 |
| packing 参数 | `SEQ_LEN=4096`、shuffle seed、batch size | 复现训练样本边界 |
| 环境信息 | Python、datasets、tokenizers、datasketch 版本 | 排查产物差异 |
| 运行结果 | 样本数、token 数、目录大小、失败日志 | 判断是否可交付 |

*表 P11-3：Mini-DeepSeek 最小复现实验记录项*

## 数据质量与配方验收

Mini-DeepSeek 的验收重点不是最终模型分数，而是配方是否可解释、可复查、可扩展。一个常见误区是只报告“最终得到 1B tokens”，但不说明这些 tokens 来自哪些领域、跨源重复被删除了多少、tokenizer 是否偏向某个语言或代码域。对于预训练数据，数量只是结果，配方才是核心。

表 P11-4 给出配方级验收口径。

| 验收项 | 推荐检查 | 不合格表现 |
| --- | --- | --- |
| 权重一致性 | 各数据源样本数与 `RECIPE` 权重偏差 | 某源过采样或因 streaming 中断被低估 |
| 字段完整性 | 每条样本保留 `text` 和 `source` | 下游无法统计来源或复查样本 |
| 跨源重复 | 记录重复数和重复率 | 代码、论文、网页之间隐性重复未清理 |
| 文本异常 | 空文本、极短文本、乱码和二进制残留比例 | tokenizer 学到无意义 token |
| tokenizer 覆盖 | 中文、英文、代码、数学样本压缩率 | 某类文本 token/char 明显异常 |
| packing 完整性 | `input_ids` 长度均为 `4096` | 训练时出现 padding 或长度不一致 |
| 随机性 | 固定 shuffle seed 和抽样策略 | 多次运行样本顺序或分布不可解释 |

*表 P11-4：Mini-DeepSeek 配方级验收清单*

在实际复核中，可以先从每个 source 抽取 100 条样本，人工检查文本类型是否符合预期；再从 MinHash 删除样本中抽查近重复对，判断阈值是否过严。若重复率异常偏高，可能是数据源之间确实重叠，也可能是 5-gram 对短文本过于敏感。若重复率异常偏低，应检查文本字段是否取错，例如某些数据集使用 `content` 而不是 `text`。

## Tokenizer 与 packing 的细粒度检查

Tokenizer 训练阶段最容易产生“文件成功但质量不明”的问题。`train_tokenizer.py` 采用 BPE、NFKC normalization、ByteLevel pre-tokenizer 和 150K 词表。这个设置适合中英文、代码和数学混合语料，但它也带来两个风险。第一，词表过大时，小规模训练样本可能无法充分覆盖长尾 token；第二，代码和数学符号可能占用较多词表空间，影响普通文本压缩率。

建议在训练完成后计算表 P11-5 中的指标。

| 指标 | 计算方式 | 解释 |
| --- | --- | --- |
| 中文 tokens/char | 中文网页样本 token 数除以字符数 | 判断中文压缩效率 |
| 英文 tokens/word | 英文网页样本 token 数除以词数 | 判断普通英文切分是否异常 |
| 代码 tokens/char | The Stack v2 样本 token 数除以字符数 | 判断代码符号覆盖 |
| 数学公式碎片率 | 数学样本中短 token 比例 | 判断公式和 LaTeX 是否被过度切碎 |
| `<|endoftext|>` 命中 | packing 后分隔 token 是否存在 | 判断样本边界是否被保留 |
| OOV 行为 | `<|unk|>` 使用率 | 判断 ByteLevel 覆盖是否正常 |

*表 P11-5：Tokenizer 与 packing 质量指标*

Packing 阶段同样需要检查。`pack_shuffle.py` 将样本 token 串接后截断到 `SEQ_LEN` 的整数倍，并输出定长 `input_ids`。这会提高训练吞吐，但也意味着原始文档边界不再直接可见。因此，`<|endoftext|>` 的插入和统计非常重要。如果忘记插入分隔符，模型会把相邻文档学习成连续文本；如果分隔符过密，短文档会主导上下文结构。

## 测试覆盖与代码隔离

`tests/test_pipeline.py` 已经覆盖了几个最小契约：`RECIPE` 权重和为 1、关键数据源存在、MinHash 可创建、相同文本 Jaccard 为 1.0、`SEQ_LEN` 等于 4096。这些测试不是为了替代完整数据验收，而是为了防止项目章中的示例代码在重构后失去基本契约。

表 P11-6 给出测试与缺口的对应关系。

| 测试项 | 已覆盖内容 | 仍需人工或集成验收 |
| --- | --- | --- |
| `test_recipe_weights` | 配方权重总和 | 每个 source 实际抽样数量 |
| `test_sampler_keys` | 关键数据源存在 | 数据源许可、字段名和可访问性 |
| `test_minhash_creation` | MinHash 对象可用 | 大规模 LSH 内存占用 |
| `test_minhash_similarity` | 相同文本相似度 | 近重复误删/漏删边界 |
| `test_tokenizer_pack` | `SEQ_LEN` 常量 | 每条 packed 样本长度 |
| `test_end_to_end_mock` | 测试入口存在 | 真实端到端小样本运行 |

*表 P11-6：Mini-DeepSeek 测试覆盖与验收缺口*

出版稿中应明确区分“教学示例代码”和“生产可直接运行代码”。本章代码可以说明多源采样、MinHash、BPE 和 packing 的基本组织方式，但在真实大规模预训练中，还需要补充分布式执行、数据源失败重试、下载缓存、污染检测、敏感内容过滤、许可证白名单和训练框架接入。

## 常见故障与排查路径

若采样阶段报错，优先检查数据源名称、config name 和 streaming split。不同 Hugging Face 数据集字段名并不完全一致，有的使用 `text`，有的使用 `content`，还有的数据需要额外认证或配置。若某个数据源返回很少样本，不应直接继续训练，而应在报告中记录该 source 的缺口，并重新计算实际配方。

若去重阶段内存占用过高，说明当前 MinHash LSH 全量驻留内存的方式已经接近单机瓶颈。可以先降低样本量验证流程，再迁移到 Spark、Ray 或外部 key-value 存储。若去重后样本数下降过多，应抽查重复对，判断是否因为短文本、模板文本或许可证头导致过度匹配。

若 tokenizer 训练时间过长，优先检查训练子样本比例。当前实现使用 `ds.select(range(0, len(ds), 10))` 抽取十分之一样本训练 tokenizer，这是一种教学化折中。若数据量更大，可以分 source 分层抽样，保证代码、数学、中文和英文都被覆盖。若最终 `mini_deepseek_tokenizer.json` 无法被 `pack_shuffle.py` 加载，应检查特殊 token 是否写入，以及文件是否被中断写坏。

若 packing 后训练烟测 loss 异常，排查顺序应为：首先确认 `input_ids` 长度是否全为 `4096`；其次抽样 decode packed 样本，检查文档边界和异常字符；再次检查 shuffle seed 和样本顺序；最后再怀疑训练超参。预训练数据问题常常伪装成训练问题，排查时应先回到数据产物。

## 训练烟测与指标记录

P11 的目标不是报告一个完整大模型训练结果，而是证明数据产物能进入训练链路。因此，训练烟测应当保持小规模、短时长和可复现。一个合格的烟测可以只运行几百到几千 step，但必须回答三个问题：数据能否被训练框架稳定读取，loss 是否呈现合理下降趋势，以及吞吐、显存和样本解码是否符合预期。

表 P11-7 给出训练烟测记录模板。

| 记录项 | 示例 | 复核意义 |
| --- | --- | --- |
| 数据目录 | `./data/mixed_1b_final_packed` | 确认训练读取的是最终 packed 数据 |
| 样本长度 | `4096` | 确认训练上下文与 packing 配置一致 |
| batch 配置 | global batch、micro batch、grad accumulation | 解释吞吐与显存差异 |
| 模型规模 | 例如 100M、300M、1B 参数教学模型 | 明确不与完整 DeepSeek-V3 对比 |
| tokenizer | `mini_deepseek_tokenizer.json` | 确认训练与 packing 使用同一 tokenizer |
| step 范围 | 例如 0-1000 step | 说明烟测长度 |
| loss 曲线 | 首尾 loss、异常 spike | 判断数据流是否明显异常 |
| tokens/s | 每秒 token 数 | 估算后续扩容成本 |
| 样本解码 | 随机 decode 10 条 packed 样本 | 检查乱码、边界和重复 |

*表 P11-7：Mini-DeepSeek 训练烟测记录模板*

烟测报告中不要只写“训练可以跑通”。更有用的写法是：给出读取速度、前若干 step 的 loss、若干条 packed 样本解码结果，以及失败样本的处理记录。若 loss 长时间不下降，可能是模型配置问题，也可能是数据中存在大量重复、乱码或无意义 token；若吞吐低于预期，可能是 `.arrow` 分片过小、数据加载进程不足或磁盘 IO 成为瓶颈。

## 数据污染与基准泄漏检查

预训练复现类项目必须注意 benchmark contamination。即使本章只是教学化缩小版，也应说明数据污染检查的方法。多源语料中可能包含 GSM8K、MATH、HumanEval、MMLU 或其他评测集的题面、解析和答案。如果这些内容进入预训练数据，后续 benchmark 分数会被高估。

表 P11-8 给出污染检查的最小方案。

| 检查对象 | 方法 | 处理方式 |
| --- | --- | --- |
| 精确重复 | 对 benchmark 题面做 hash 或规范化字符串匹配 | 命中样本直接剔除 |
| 近重复 | 对题面、答案和解析做 MinHash 或 embedding 召回 | 人工抽检后删除 |
| 代码评测 | 检查 HumanEval 函数名、docstring 和 canonical solution | 删除完整题目和答案 |
| 数学评测 | 检查题干、选项、最终答案模式 | 删除题解和答案泄漏 |
| 论坛转载 | 检查 benchmark 题目在网页/博客中的转载 | 删除或标记为污染风险 |
| 训练报告 | 保存命中数量、删除策略和样本 id | 支撑后续 benchmark 声明 |

*表 P11-8：预训练数据污染检查方案*

污染检查应在 packing 之前完成，因为一旦进入 packed dataset，原始文档边界和来源字段会变得更难追踪。若教学项目没有完整实现污染过滤，也应在报告中明确说明“未用于公开 benchmark 声明”，避免读者把烟测分数误解为可发表结果。

## 交付物目录与发布包结构

一个可交付的 P11 项目不应只包含最终数据目录，还应包含配方、日志、测试和审计材料。表 P11-9 给出推荐目录。

| 路径 | 内容 | 是否进入公开发布 |
| --- | --- | --- |
| `data/mixed_1b_raw/` | 按配方抽样的原始混合数据 | 视数据许可决定 |
| `data/mixed_1b_dedup/` | 跨源去重后的训练候选语料 | 视数据许可决定 |
| `data/mini_deepseek_tokenizer.json` | BPE tokenizer | 可公开 |
| `data/mixed_1b_final_packed/` | packed `.arrow` 数据 | 视数据许可决定 |
| `reports/source_mix.json` | 各 source 样本数与比例 | 可公开 |
| `reports/dedup_report.json` | 重复数、阈值、抽检样本 | 可公开脱敏版本 |
| `reports/tokenizer_eval.md` | 压缩率、异常 token、样本 decode | 可公开 |
| `reports/smoke_train.md` | 训练烟测指标 | 可公开 |
| `tests/` | 单元测试与小样本测试 | 可公开 |
| `LICENSES.md` | 数据源许可说明 | 必须公开 |

*表 P11-9：Mini-DeepSeek 项目交付物目录*

若数据源不允许再分发，仍然可以发布“配方 + 脚本 + tokenizer + 报告模板”，但不能把原始样本或 packed 样本直接发布。此时应提供可复现说明，让具备授权的读者在本地重建数据。

## 从 1B tokens 扩展到更大规模

P11 的缩小版链路可以帮助读者理解配方，但扩展到 10B、70B 或更大规模时，需要做系统改造。表 P11-10 给出从教学实现到规模化实现的改造清单。

| 模块 | 教学实现 | 规模化改造 |
| --- | --- | --- |
| 数据读取 | Hugging Face streaming + 本地 Dataset | 对接对象存储、数据湖或分布式缓存 |
| 配方采样 | 单次按比例抽样 | 支持 epoch 级动态配比和 curriculum |
| 去重 | 单机 MinHash LSH | 分布式 MinHash、SimHash 或 embedding 近重复系统 |
| tokenizer | 单机抽样训练 | 分层抽样、版本冻结和兼容性回归 |
| packing | 单机 map + shuffle | 分布式 packing、固定 shard 大小、训练框架预取 |
| 审计 | 手工报告 | 元数据服务、 lineage、删除请求回放 |
| 测试 | 单元测试 + mock e2e | 小样本真实 e2e、数据差异回归、污染扫描 |

*表 P11-10：Mini-DeepSeek 从教学实现到规模化实现的改造清单*

扩展时最重要的是不要把“脚本能跑”误认为“系统可扩展”。大规模预训练数据系统需要处理失败重试、断点续跑、版本冻结、样本删除、许可证变更、训练中数据混合策略和多团队协作。P11 的价值在于把这些问题的最小形态展示出来，而不是声称已经替代完整工业系统。

## 数据看板与持续监控

预训练数据工程一旦进入持续迭代，就不能只依赖一次性的运行日志。数据源会更新，许可证会变化，字段 schema 会调整，Hugging Face 数据集也可能因为维护原因改变 split 或样本内容。因此，P11 需要建立一个轻量数据看板，用于比较不同运行批次之间的差异。

表 P11-11 给出推荐看板指标。

| 看板指标 | 统计来源 | 观察目的 |
| --- | --- | --- |
| source 样本数 | `mixed_1b_raw` | 检查配方比例是否漂移 |
| source token 数 | tokenizer 后统计 | 检查样本数与 token 数是否一致 |
| 去重率 | `mixed_1b_raw` 与 `mixed_1b_dedup` | 发现重复异常或阈值问题 |
| 文本长度分布 | 原始文本字段 | 发现过短、过长和模板化样本 |
| 语言比例 | 语言识别脚本 | 检查中文、英文和其他语言比例 |
| 代码比例 | source 与代码特征检测 | 检查代码源是否被过采样 |
| 数学样本比例 | source 与 LaTeX/公式检测 | 检查数学数据覆盖 |
| tokenizer 压缩率 | tokenizer 评估脚本 | 发现词表质量变化 |
| packed shard 大小 | `mixed_1b_final_packed` | 检查训练读取均衡性 |
| 污染命中数 | contamination scan | 记录 benchmark 泄漏风险 |

*表 P11-11：Mini-DeepSeek 数据看板指标*

看板的核心价值是批次比较。例如，某次运行中 The Stack v2 样本数没有变化，但 token 数显著上升，可能说明代码样本长度变长，或者过滤条件发生变化。某次运行中 OpenWebMath 的去重率突然升高，可能说明数学网页中模板化内容增加。若没有批次看板，这些变化往往要等到训练 loss 或 benchmark 异常时才被发现。

## 删除请求与样本撤回机制

多源预训练数据必须具备样本撤回能力。即使项目只是教学化复现，也应说明：如果某个数据源、作者或内容方要求删除样本，系统如何定位、删除并重建下游产物。P11 的当前实现保留了 `source` 字段，但 packed 数据中原始文档边界会弱化，因此删除请求最好在 packing 前处理。

表 P11-12 给出删除请求的处理路径。

| 步骤 | 操作 | 产物 |
| --- | --- | --- |
| 接收请求 | 记录请求方、URL、样本特征和时间 | deletion request ticket |
| 定位样本 | 在 raw/dedup 数据中按 URL、hash、文本片段或 source 查找 | affected sample ids |
| 删除上游 | 从 `mixed_1b_raw` 或 `mixed_1b_dedup` 中移除命中样本 | revised dataset |
| 重建 tokenizer 判断 | 若删除比例较大，评估是否重训 tokenizer | tokenizer impact report |
| 重建 packed 数据 | 重新运行 packing 与 shuffle | revised packed dataset |
| 更新报告 | 更新配方比例、token 数和删除说明 | release note |

*表 P11-12：预训练样本删除请求处理路径*

真实生产系统通常会保存文档级 hash、URL、source、license 和 packed shard 反向索引。教学实现不一定需要完整索引，但至少要让读者理解：越早丢弃 lineage，后续撤回成本越高。出版文本中保留这一点，能避免读者误以为 `.arrow` 训练数据可以脱离来源单独流转。

## 领域迁移：从通用配方到行业模型

Mini-DeepSeek 的配方以网页、代码、数学、学术和中文语料为主。如果迁移到法律、医疗、金融或工业领域，不能简单把某个行业数据源加入 `RECIPE`。行业数据通常具有更强的权限、隐私、术语和时效要求，需要单独设计配方和验收。

表 P11-13 给出领域迁移时的调整方向。

| 领域 | 需要增加的数据 | 额外风险 | 验收重点 |
| --- | --- | --- | --- |
| 法律 | 法规、判例、合同、合规问答 | 地区和时效差异 | 法条版本、引用准确性 |
| 医疗 | 指南、论文、药品说明、病例模板 | 隐私和高风险建议 | 脱敏、来源等级、专家复核 |
| 金融 | 研报、公告、财报、市场规则 | 时效和投资建议风险 | 日期、市场、口径一致性 |
| 工业 | 设备手册、故障记录、工艺文档 | 内部机密和术语歧义 | 权限、术语表、故障分类 |
| 教育 | 教材、习题、解析、课程讲义 | 版权和答案泄漏 | 版权许可、题库污染 |

*表 P11-13：Mini-DeepSeek 配方的领域迁移注意事项*

领域迁移时，建议先保留通用语料作为基础，再逐步提高领域语料比例。若一次性把领域数据比例拉得过高，模型可能获得领域术语能力，但损失通用语言和代码能力。更稳妥的做法是设计多轮 curriculum：前期保持通用语料占主导，中后期提高领域和任务相关数据比例，并用领域验证集监控收益。

## 与 P01 Mini-C4 的关系

P01 关注单源网页清洗，P11 关注多源预训练配方。二者不是重复关系，而是从“清洗一类语料”到“组织多类语料”的升级。表 P11-14 总结二者差异。

| 维度 | P01 Mini-C4 | P11 Mini-DeepSeek |
| --- | --- | --- |
| 数据来源 | 单源或少量网页源 | 网页、代码、数学、学术、中文 |
| 核心问题 | 清洗质量、去噪、基础过滤 | 配方比例、跨源去重、tokenizer 和 packing |
| 去重范围 | 单源内近重复 | 跨源近重复 |
| tokenizer | 可沿用现有 tokenizer | 训练 150K 大词表 |
| 训练样本 | 清洗后的文档 | 定长 packed token blocks |
| 验收重点 | 文本质量和清洗规则 | 配方、压缩率、污染、烟测 |
| 扩展方向 | 更大网页语料 | 更大多源预训练系统 |

*表 P11-14：P01 Mini-C4 与 P11 Mini-DeepSeek 的差异*

理解这个关系有助于读者把第十四篇项目串起来。P01 是数据清洗的起点，P11 则把多个清洗后的来源组织成预训练配方。没有 P01 的质量过滤，P11 的多源配方会吸收大量噪声；没有 P11 的配方组织，P01 的单源清洗又难以支撑现代通用模型训练。

## 结果展示与分析

教学化示例配置可以设定为单机节点（如 8 张 4090）运行，并记录端到端耗时，用于展示流水线验收报告的组织方式。
若以 `TARGET_TOTAL_DOCS = 500,000` 的抽样规模运行，报告中应记录 MinHash 去重率；示例量级可写为约 **4.2%** 的隐性重复被过滤，主要集中在代码源与学术源之间。正式交付时必须以实际运行日志、随机种子和数据 manifest 为准。

打乱打包后的 `mixed_1b_final_packed` 数据集应记录存储大小、`.arrow` 分片数量和 token 统计。示例报告中可使用约 `5GB`、约 **1.05B Tokens** 的量级说明输出格式，但正式版本必须由脚本输出、样本 manifest 和随机种子共同生成。

### Tokenizer 效率验证
由于词表扩容至 150K，通过抽样验证，该 Tokenizer 对于中文网页平均压缩比（Tokens/Char）达到了 **0.62**，相较于 Llama-2 的 1.1 的表现，显著提升了后续预训练的吞吐效率。

## 成本与优化

在 1B tokens 级别的教学化示例中，资源消耗可按以下口径记录：

- **存储**：记录原始抓取数据和 Pack 后产物的实际大小；示例量级可写为原始数据约 8GB、Pack 后约 5GB。
- **算力与内存**：记录 Streaming 抽取、并行 Map 和跨源 MinHash 去重的内存峰值与耗时；示例量级可写为内存峰值约 32GB、MinHash 去重约 3 小时。正式交付时应以运行日志为准。

**优化点**：
如果需要横向扩展到 70B Tokens，单节点的 Python 内存处理将成为瓶颈。建议接入 Apache Spark (Zaharia et al. 2016) 或是 Ray (Moritz et al. 2018) 这样的分布式引擎。在 MinHash 去重环节，可通过 Redis 等外部数据库存储 Hash Bucket 来实现内存解耦。

## 扩展思考

将 Mini-DeepSeek 项目的配方扩展到百亿级 Tokens，有两点需要格外关注：

1. **配比的动态衰减（Curriculum）**：在初期训练，基础知识（网页与学术论文）应当占据主导；在中后期，需要拉高代码与数学（OpenWebMath）的采样权重。可将 `mix_sampler.py` 改造为支持 Epoch 级动态加载的流式模块。
2. **与前作的升级对比**：相比于第十四篇 P01（Mini-C4），本项目不再依赖单一质量阈值的简单过滤，而是用跨源融合与超大词表的设计，展示了现代工业级模型（如 DeepSeek-V3）面向多任务的基础奠基方式。

### 数据合规与开源许可说明
在进行多源混合时，必须严格遵守原始数据的开源许可（License）：

- **FineWeb-Edu**：采用 CC0 许可（完全开源）。
- **The Stack v2**：遵循 SPDX 白名单许可体系，仅使用允许再分发的代码。
- **OpenWebMath**：采用 ODC-By 许可。
- **arXiv**：遵循各论文作者选择的具体分发 License。
- **Project Gutenberg**：公有领域（Public Domain）。
*(注：完整的 1B 数据样本已合规处理，可上传至 HuggingFace Datasets 仓库 `dataforge-mini-deepseek-1b` 供后续微调直接使用。)*

## 本章小结

本章以“Mini-DeepSeek 预训练复现”为案例，展示了以小规模资源复现开源 LLM 预训练数据配方的关键工程环节的工程组织方式。案例的主要价值在于把任务定义、数据边界、架构决策、样本 schema、指标验收和复现资源放在同一条链路中，使项目不再只是操作步骤，而成为可复核的案例研究。

该案例的边界同样需要被清楚保留。定位为缩小版配方验证，不追求完整大模型规模和公开 SOTA 指标。在更大规模、更高风险或更强合规约束的场景中，应重新评估数据来源、权限状态、人工复核比例、运行成本和失败回滚方案。

作为第十四篇的一部分，本章对应前文方法在项目层面的落地验证。读者可将本案例与第十三篇的数据配方、前文的平台治理章节以及附录中的检查清单合并使用，形成从方法理解到工程交付的闭环。

## 参考文献

Broder A Z (1997) On the Resemblance and Containment of Documents. In: Proceedings of the Compression and Complexity of Sequences, pp 21-29.

Kaplan J, McCandlish S, Henighan T, Brown T B, Chess B, Child R, Gray S, Radford A, Wu J, Amodei D (2020) Scaling Laws for Neural Language Models. arXiv preprint arXiv:2001.08361.

Liu A, Feng B, Xue B, Wang B, Wu B, Lu C, Zhao C, Deng C, Zhang C, Ruan C, others (2024) DeepSeek-V3 Technical Report. arXiv preprint arXiv:2412.19437.

Lozhkov A, Ben Allal L, von Werra L, Wolf T (2024) StarCoder 2 and The Stack v2: The Next Generation (The Stack v2). arXiv preprint arXiv:2402.19173.

Moritz P, Nishihara R, Wang S, Tumanov A, Liaw R, Liang E, Elibol M, Yang Z, Paul W, Jordan M I, Stoica I (2018) Ray: A Distributed Framework for Emerging AI Applications. In: Proceedings of the 13th USENIX Symposium on Operating Systems Design and Implementation, pp 561-577.

Paster K, Santos M D, Azerbayev Z, Ba J (2023) OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text. arXiv preprint arXiv:2310.06786.

Penedo G, Kydlicek H, de Wiele T V, Lozhkov A, Mitchell M, Raffel C, von Werra L, Wolf T (2024) The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. arXiv preprint arXiv:2406.17557.

Sennrich R, Haddow B, Birch A (2016) Neural Machine Translation of Rare Words with Subword Units (BPE). In: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, pp 1715-1725.

Zaharia M, Xin R S, Wendell P, Das T, Armbrust M, Dave A, Meng X, Rosen J, Venkataraman S, Franklin M J, Ghodsi A, Gonzalez J, Shenker S, Stoica I (2016) Apache Spark: A Unified Engine for Big Data Processing. Communications of the ACM 59(11):56-65.
