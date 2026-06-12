# 第42章：Dolma 预训练语料透明账本

## 摘要

多数开放模型发布时会给出权重、推理代码和若干评测结果，但训练数据往往只留下含混的来源描述。对继续预训练、偏差分析、评测污染排查和许可审计来说，这种发布方式不够。模型出了问题，团队很难回答它见过哪些 source、这些 source 被怎样过滤、采样比例是多少、某个 benchmark 是否泄漏进训练语料，以及用户请求移除个人数据时应定位到哪个 shard。

Dolma 是 Allen Institute for AI 发布的三万亿 token 级英文预训练语料，用于支撑开放语言模型研究和 OLMo 训练。它的价值不只是规模，而是把预训练语料写成一本可查账的工程记录：版本、来源、source-level statistics、sample proportion、ODC-BY 许可、原始来源条款、个人数据移除入口和 Dolma Toolkit 处理流程都被公开说明。围绕这本账，本章先从开放模型研究中的数据缺口进入，再梳理 Dolma 的版本和来源统计，随后沿 source 账本、单条文档记录和训练 manifest 拆解样本结构；最后讨论 Dolma Toolkit 如何把 tag、dedup、mix、tokenize 变成可审计动作，以及透明语料在质量控制、撤回和企业内部迁移中的边界。

## 关键词

Dolma；透明预训练语料；OLMo；source mix；token accounting；source card；manifest；Dolma Toolkit；ODC-BY；数据审计

## 42.0 学习目标

通过本章学习，读者应能够：

- 解释为什么权重开放不能替代训练数据透明。
- 读懂 Dolma 的版本、来源统计、采样比例和 ODC-BY 使用边界。
- 区分文档记录、source card 和训练 manifest 在审计中的作用。
- 用 token accounting 公式描述 raw tokens、filtered tokens、sample proportion 和 seen tokens 的关系。
- 按 Dolma Toolkit 的 tag、dedup、mix、tokenize 四个动作理解透明语料的证据链。
- 设计企业内部预训练语料的 source 账本、撤回账本、污染检查和版本冻结机制。

## 42.1 问题场景：权重开放后仍然无法解释模型

两个团队都拿到了同一个 7B 开放模型权重。第一个团队想继续预训练，让模型增强代码和科学问答能力；第二个团队想分析模型为什么在若干事实题上给出过时答案。权重、推理代码和部分评测脚本都能下载，但他们很快遇到同一个问题：模型到底见过什么数据，没人能回答。

继续预训练团队需要知道原模型语料中代码、论文、百科、Web 和社交媒体分别占多少，避免在同类数据上重复过采样。偏差分析团队需要知道某些网页、论文、论坛讨论或 benchmark 题解是否进入过训练，判断问题来自知识缺失、采样权重、污染还是模型训练本身。没有训练语料和处理记录，所有解释都只能停留在猜测。

Dolma 要解决的正是这个问题。它把英文预训练语料从不可见的数据配方，变成一组可下载、可统计、可处理、可撤回、可审计的 source。OLMo 在本章中不是并列主角，而是 Dolma 被透明训练链路消费的下游例子：它提醒我们，开放模型研究不应只开放权重，还应尽可能开放训练数据、处理工具和评估代码。

### 42.1.1 开放模型研究需要数据证据

“开放”有不同层次。只发布权重，可以让用户运行模型，却不能让研究者解释模型。发布训练代码，可以让用户复现训练框架，却仍然不能说明模型实际看过什么。真正支撑科学研究的数据透明，至少需要回答六类问题。

- 来源问题：每个 document 来自 Common Crawl、代码仓库、论文、书籍、百科还是社交平台。
- 版本问题：来源获取时间、处理脚本版本和过滤规则是什么。
- 规模问题：raw tokens、filtered tokens、sampled tokens 和 seen tokens 是否一致。
- 污染问题：评测集、题解、客户测试集是否与训练语料重叠。
- 许可问题：数据集发布许可与原始来源条款如何共同约束使用者。
- 撤回问题：若用户请求移除个人数据，能否定位并处理对应文档。

Dolma 数据卡直接体现了这种设计取向：它列出版本、summary statistics、下载方式、许可信息，并提供个人数据移除入口。Dolma GitHub 仓库进一步给出数据和工具，使透明不只停留在论文描述中。

对 Dolma 这类语料来说，透明的核心对象是账本，单条 `text` 不是唯一对象。更重要的是围绕 `text` 形成三本账。

第一本是 source 账，记录某类数据来自哪里、规模多大、截止日期是什么、处理方法是什么。第二本是处理账，记录 tagger、过滤器、去重策略、mixer 和 tokenizer 如何改变原始文档。第三本是训练账，记录某次训练实际采样了哪些 source、采样比例是多少、seen tokens 如何分布到训练 step。

如果这三本账断开，透明就会退化为“可下载”。数据可以下载，但不能解释；模型可以训练，但不能审计；版本可以更新，但不能比较。

## 42.2 数据集概览：版本、规模与来源结构

Dolma 不是单一静态文件，而是带版本演进的语料资产。Hugging Face 数据卡列出 `v1`、`v1_5`、`v1_5-sample`、`v1_6`、`v1_6-sample` 和 `v1_7` 等版本；其中 `v1_7` 用于训练 OLMo 7B-v1.7，并引入新来源、更多质量过滤和 fuzzy deduplication。

*表42-1 Dolma 公开版本和用途*

| 版本 | 发布时间 | 压缩体积 | 数据卡说明 | 工程用途 |
| --- | --- | ---: | --- | --- |
| `v1` | 2023-08-18 | 6.0 TB | Dolma 第一版 | 追溯最早公开语料形态 |
| `v1_5` | 2023-10-31 | 6.4 TB | 用于训练 OLMo-1B，约 3T tokens | 复查 OLMo 早期训练语料 |
| `v1_5-sample` | 2023-10-31 | 2.9 TB | 约 1.9T tokens 的样本，用于 OLMo-7B | 低于 full 版本的训练样本追踪 |
| `v1_6` | 2024-01-31 | 5.4 TB | 在 v1.5 基础上增加部分去重和重复 n-gram 过滤 | 研究过滤和去重演进 |
| `v1_6-sample` | 2024-01-31 | 16.4 GB | 约 10B tokens 的探索样本 | 快速调试和数据浏览 |
| `v1_7` | 2024-04-15 | 4.5 TB | 用于训练 OLMo 7B-v1.7，新来源、更多质量过滤、fuzzy deduplication | 当前默认版本和透明训练参照 |

数据来源 Hugging Face `allenai/dolma` 数据卡 Versions 小节。

### 42.2.1 v1.6 来源结构

Dolma 的来源覆盖 Web、代码、论文、社交媒体、书籍和百科。为了避免不同版本混淆，表42-2 使用数据卡中 v1.6 summary statistics 的大类统计。v1.7 的来源更加细分，新增 Refined Web、StarCoder、arXiv、StackExchange、Flan、OpenWebMath、Algebraic Stack、MegaWika 等 source；后续写作或实验应明确使用哪个版本。

*表42-2 Dolma v1.6 来源统计*

| 来源 | 文档类型 | UTF-8 bytes | 文档数 | Unicode words | Llama tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Common Crawl | web pages | 9,022 GB | 3,370M | 1,775B | 2,281B |
| The Stack | code | 1,043 GB | 210M | 260B | 411B |
| C4 | web pages | 790 GB | 364M | 153B | 198B |
| Reddit | social media | 339 GB | 377M | 72B | 89B |
| PeS2o | STEM papers | 268 GB | 38.8M | 50B | 70B |
| Project Gutenberg | books | 20.4 GB | 0.056M | 4.0B | 6.0B |
| Wikipedia and Wikibooks | encyclopedic | 16.2 GB | 6.2M | 3.7B | 4.3B |
| Total | mixed | 11,519 GB | 4,367M | 2,318B | 3,059B |

数据来源 Hugging Face `allenai/dolma` 数据卡 Summary Statistics v1.6。表中 GB、M、B 均沿用数据卡口径。

表42-2 不应只被读成规模展示。它提示三类工程事实。

第一，Dolma 是 source mix，而不是单一 Web dump。Common Crawl 占比很高，但代码、论文、社交媒体、书籍和百科都以不同形式进入语料。模型能力变化不能只笼统归因于“Web 数据更多”。

第二，不同统计口径服务不同问题。UTF-8 bytes 适合估算存储和处理成本，document count 适合观察样本颗粒度，Unicode words 和 Llama tokens 则更接近训练预算。把这些口径混在一起，会让数据规模讨论失真。

第三，版本之间不能直接横比。v1.6 和 v1.7 的来源拆分、过滤规则和 sample proportion 不同。如果一个模型用 v1.7 训练，不能只拿 v1.6 的大类表解释训练行为。

## 42.3 以 source 账本拆解一条透明链路

本节不从单条文本开讲，而是从 Dolma 的 source 账本拆解透明预训练语料的链路。这里的“样本”不是一张图或一道题，而是一条从 source 到文档、再到训练 manifest 的证据路径。

### 42.3.1 从 source 到 document

Dolma 的任务不是给每条文本标注监督标签，而是让训练消费记录可以被复原。设 source 集合为 $S=\{s_1,\ldots,s_m\}$，每个 source 经过处理函数 $P_s$ 后得到文档集合 $D_s$：

$$
D_s=P_s(R_s, C_s)
$$

其中 $R_s$ 是原始来源，$C_s$ 是该 source 的处理配置，包括 tagger、过滤器、去重策略、采样比例和 tokenizer。最终训练语料是多个 source 的混合：

$$
D=\bigcup_{s \in S} Sample(D_s, r_s)
$$

Dolma 的透明性体现在：$S$、$R_s$、$C_s$、$r_s$ 和版本信息都尽量被公开记录。这样模型训练不再只是“用了三万亿 token”，而是可以追踪到哪些 source 贡献了多少、如何处理、如何采样。

### 42.3.2 token 账要看训练实际消费量

多来源语料最容易被误读的是 token 规模。一个 source 的 raw tokens 很大，不代表它在训练中贡献同等比例；过滤、去重、sample proportion 和多轮 epoch 都会改变最终 seen tokens。

可以把某个 source $s$ 在训练中的实际贡献写成：

$$
T^{seen}_s = T^{filtered}_s \times r_s \times e_s
$$

其中，$T^{filtered}_s$ 是过滤后的 token 数，$r_s$ 是采样比例，$e_s$ 是训练中被重复看到的 epoch 或等价采样次数。训练 mix 中该 source 的比例为：

$$
p_s=\frac{T^{seen}_s}{\sum_j T^{seen}_j}
$$

Dolma v1.7 数据卡中同时列出 source token 数和 sample proportion，正是为了让使用者区分“数据集里有什么”和“训练实际看了多少”。

### 42.3.3 一条文档的三层读法

透明语料不是把 `text` 打包上传就结束。至少需要三层记录：单条文档记录、source card 和训练版本 manifest。单条文档用于训练和定位，source card 用于解释数据来源和处理规则，训练 manifest 用于复现某次模型训练实际消费的数据。

*表42-3 Dolma 类透明语料记录 schema*

| 层级 | 典型字段 | 来源或生成方式 | 工程用途 |
| --- | --- | --- | --- |
| 文档级 | `id`、`source`、`text`、`text_hash` | 数据读取和哈希计算 | 定位样本、去重、训练读取 |
| 文档级 | `created_at`、`url_or_origin`、`license_hint` | 原始 source 元数据 | 授权复核和时间追踪 |
| 文档级 | `language_tag`、`toxicity_tag`、`perplexity_score` | Dolma Toolkit taggers 或自定义 tagger | 质量过滤和风险分桶 |
| source 级 | `source_name`、`source_version`、`raw_size`、`filtered_size` | source card 与统计脚本 | 解释数据组成 |
| source 级 | `dedup_policy`、`filter_config`、`sample_proportion` | Dolma mixer 和处理配置 | 复现 source mix |
| 训练级 | `dolma_version`、`tokenizer`、`sample_seed`、`seen_tokens` | 训练 manifest | 复现实验和解释指标变化 |
| 治理级 | `removal_status`、`known_limitations`、`release_constraints` | 数据卡和治理记录 | 处理撤回、偏差和使用边界 |

表中字段是作者根据 Dolma 数据卡、Dolma Toolkit 文档和透明训练审计需求整理的工程 schema，不表示 Dolma 官方逐项发布了这些字段。

下面是一个抽象化的 Dolma 类文档记录，用于说明透明语料如何把样本、source 和训练版本连接起来。

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

这条记录不能只看单层字段。放到 Dolma 中，文档层、source 层和训练层必须能相互指回。若文档有 `source` 但没有 source card，只能定位样本，不能解释来源；若 source card 有统计但没有训练 manifest，只能说明数据集里有什么，不能说明模型实际看了什么；若 manifest 有采样比例但没有文档 hash，撤回和污染检查就会断链。

## 42.4 Dolma Toolkit 让证据链可执行

Dolma GitHub 仓库说明，Dolma 同时是数据集和工具包。Dolma Toolkit 支持单机、集群和云环境，内置语言检测、毒性检测、perplexity scoring，以及 Gopher、C4、OpenWebText 等常见过滤 recipe；去重部分使用 Rust Bloom filter 加速。

### 42.4.1 四个动作对应四类证据

Dolma Toolkit 文档把数据整理概括为四个动作：tag、dedup、mix、tokenize。它们不是孤立脚本，而是证据链的生成器：tag 记录文档属性，dedup 记录保留和删除，mix 记录采样比例，tokenize 记录进入训练的 token 口径。

*表42-4 Dolma Toolkit 处理动作与证据输出*

| 顺序 | 动作 | 官方文档说明 | 证据输出 | 主要风险 |
| ---: | --- | --- | --- | --- |
| 1 | Taggers | 给文档 span 打语言、毒性、perplexity 等属性标签 | 文档质量标签和风险标签 | tagger 版本变化会改变过滤结果 |
| 2 | Deduplication | 基于内容或元数据对文档去重 | 去重策略、保留优先级、删除记录 | 跨 source 去重会改变 source mix |
| 3 | Mixer | 根据属性值移除、过滤或混合文档 | sample proportion、source mix、数据版本 | sample proportion 不透明会导致 token accounting 错误 |
| 4 | Tokenization | 使用 Hugging Face 兼容 tokenizer | token 计数、tokenizer 版本、训练流 | tokenizer 变化会改变 token 数和训练预算 |

数据来源 Dolma Toolkit documentation README。

![图42-1 Dolma 透明语料证据链](../../images/part12/ch42_01_dolma_evidence_chain.svg)

*图42-1 Dolma 透明语料证据链。Source: original illustration based on AllenAI Dolma Toolkit documentation.*

这里要注意工具链和人工审计的边界。工具链能稳定地产生统计、标签、hash 和 manifest，但它不能替代所有审计。许可边界、PII removal、评测污染和 source 代表性仍需要人工规则、抽样复核或专门的检测任务介入。

这也是 Dolma 类透明语料和普通“清洗脚本集合”的区别。普通脚本只回答“我删掉了什么”，透明工具链还要回答“为什么删、删后分布怎么变、训练是否真的受益、后续能否撤回”。如果某个处理动作不能留下可解释证据，它就很难支撑透明训练。

## 42.5 评估从最高分转向可归因

透明语料的评估重点不是“最高分”，而是“分数变化能不能解释”。Dolma 这类 source-level 语料让模型评估不再停在 leaderboard 上，而是能顺着训练日志回到 source、版本和采样比例。

### 42.5.1 source ablation 定位能力来源

如果要判断某个 source 是否影响模型能力，可以做 source ablation。设完整训练得到模型 $M_{all}$，移除 source $s$ 后得到模型 $M_{-s}$，在任务集合 $B$ 上的平均差异为：

$$
\Delta_s=\frac{1}{|B|}\sum_{b \in B} \left[score(M_{all}, b)-score(M_{-s}, b)\right]
$$

当 $\Delta_s$ 在代码任务、科学问答或长文本任务上明显变化时，数据团队才能把能力变化回溯到 source mix，而不是泛泛归因于“模型参数”。

![图42-2 Dolma source mix 与训练诊断回路](../../images/part12/ch42_02_dolma_source_mix_diagnosis.svg)

*图42-2 Dolma source mix 与训练诊断回路。Source: original illustration based on Dolma dataset card and OLMo training use.*

### 42.5.2 诊断清单

*表42-5 Dolma 类透明语料评估和诊断表*

| 评估问题 | 所需记录 | 指标或证据 | 可能动作 |
| --- | --- | --- | --- |
| 某类能力来自哪个 source | source mix、sample proportion、seen tokens | source ablation、任务得分差 $\Delta_s$ | 调整 source 权重或补充数据 |
| 某个评测是否被污染 | 文档 hash、n-gram index、eval set hash | overlap rate、contamination span | 删除污染样本并冻结新版本 |
| 训练 loss 异常波动 | batch source 记录、tokenizer 版本 | source-specific loss、token distribution | 回查采样器和 source shard |
| 社交媒体风险升高 | toxicity tag、PII tag、source card | risk tag rate、人工抽检 | 收紧过滤或降低采样比例 |
| 版本间不可比 | dolma_version、filter_config、dedup_policy | manifest diff | 固定版本或重跑对照实验 |

### 42.5.3 常见失败模式

*表42-6 Dolma 类透明语料常见失败与修复动作*

| 失败模式 | 表现 | 可能根因 | 治理方式 |
| --- | --- | --- | --- |
| source mix 漂移 | 新版本某类任务突然退化 | sample proportion 或过滤规则变化 | 比较 manifest diff，回滚或分层重采样 |
| token accounting 不一致 | 报告规模与训练 seen tokens 对不上 | raw、filtered、sampled tokens 混用 | 同时报告 raw、filtered、sampled、seen tokens |
| 跨来源重复 | 论文摘要、百科镜像、代码片段重复出现 | 只做 source 内去重 | 增加跨 source 去重并记录保留优先级 |
| 评测污染 | benchmark 分数异常偏高 | Web 或论坛含题目和题解 | 建立 eval hash/n-gram 去污染索引 |
| 撤回不可定位 | PII removal 请求无法处理 | 缺少 source/id/hash 映射 | 建立 removal ledger 和 shard 反向索引 |
| 许可边界模糊 | 误把 ODC-BY 当作原始来源授权 | 忽略原始 source terms | 在 source card 中保留原始条款和限制 |

这些失败模式说明，透明语料的质量不只取决于文本是否干净。一个透明训练样本至少要同时通过三类检查：source 是否可解释，处理动作是否可复现，训练消费是否可追踪。缺少任何一类检查，数据都可能成为“公开但不可审计”的语料。

## 42.6 透明语料的复用边界

Dolma 适合用于开放语言模型预训练研究、source mix 实验、透明数据治理教学、训练数据审计方法验证，以及企业内部 manifest 设计参照。它尤其适合回答“训练数据如何影响模型能力和局限”这类研究问题，因为它把数据、数据卡和处理工具一起公开。

Dolma 不应被理解为所有来源都可无条件商用。Hugging Face 数据卡说明 Dolma 按 ODC-BY 发布，同时使用者仍受原始数据来源的许可协议和使用条款约束。也就是说，Dolma 的发布许可不自动抹平 Common Crawl、Reddit、代码、论文、书籍等原始来源的边界。

企业内部通常不能公开训练数据，但可以迁移 Dolma 的透明做法。最低可行版本包括：

- 为每个 source 建 source card，记录来源、版本、许可、获取时间、过滤规则和限制。
- 为每个训练版本冻结 manifest，记录 source mix、sample proportion、tokenizer、filter_config、dedup_policy 和 sample_seed。
- 为主要 source 建 validation split，支持 source-specific loss 和 source ablation。
- 为评测集、客户测试集和线上问题库建立去污染索引。
- 建立 removal ledger，使 URL、document id 或 text hash 能映射到训练 shard。

同时，Dolma 不适合直接作为中文、多语或垂直行业语料的质量标准。它主要是英文预训练语料，迁移到中文、医疗、法律或金融场景时，需要重新定义 source、许可、过滤器和评估任务。Dolma 也不应被用来证明开放语料天然安全。透明的意义是缺陷可见、可定位、可修复，而不是宣称缺陷不存在。

## 本章小结

Dolma 把预训练语料从不可见的数据配方推进到有账可查的数据资产。本章的核心结论有三点。

第一，透明预训练语料的基本单位不是单条 `text`，而是能连接 source、处理配置、采样比例和训练 manifest 的文档记录。第二，source mix 必须用 token accounting 解释，不能只报告原始规模。第三，Dolma Toolkit 的 tag、dedup、mix、tokenize 四个动作，为企业内部构建可复现训练语料提供了清晰模板。

对本书读者而言，Dolma 最值得迁移的不是某个具体 source，也不是三万亿 token 的规模，而是把训练数据留下证据的方式。只要 source、处理动作、采样比例、训练 manifest 和治理记录能连成链，预训练语料才具备继续研究、错误归因和长期维护的基础。

## 参考文献

- Soldaini, L., Kinney, R., Bhagia, A., Schwenk, D., Atkinson, D., Authur, R., et al. (2024). Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research. ACL 2024. https://arxiv.org/abs/2402.00159
- Allen Institute for AI. (2023). Ai2 Dolma: 3 trillion token open corpus for language model pretraining. https://allenai.org/blog/dolma-3-trillion-tokens-open-llm-corpus-9a0ff4b8da64
- AllenAI. (2026). allenai/dolma Dataset Card. https://huggingface.co/datasets/allenai/dolma
- AllenAI. (2026). Dolma Dataset and Toolkit Repository. https://github.com/allenai/dolma
- AllenAI. (2026). Dolma Toolkit Documentation. https://github.com/allenai/dolma/blob/main/docs/README.md
- Groeneveld, D., Beltagy, I., Walsh, P., Bhagia, A., Kinney, R., Tafjord, O., et al. (2024). OLMo: Accelerating the Science of Language Models. https://arxiv.org/abs/2402.00838
