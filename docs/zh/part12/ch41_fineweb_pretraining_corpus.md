# 第41章：FineWeb 预训练语料数据工程

## 摘要

FineWeb 是 Hugging Face 团队基于 Common Crawl 构建的大规模英文 Web 预训练语料。它的价值不只在于规模，而在于把“如何把网页快照变成训练数据”这件事拆成了可复现的工程链路：从 WARC 原始网页读取、URL 风险过滤、Trafilatura 正文抽取、FastText 语言识别、Gopher/C4/FineWeb 质量过滤，到按 crawl 独立 MinHash 去重、PII 格式化和版本发布。FineWeb 论文同时公开了处理代码、DataTrove 处理库和消融模型，使数据处理选择可以被训练结果回验，而不是停留在人工抽检或经验判断上。

FineWeb 这一章的主线是 Web 预训练语料的“炼制过程”。Common Crawl 给出的只是网页快照，离可训练 token stream 还差正文抽取、语言识别、质量过滤、去重、隐私处理和评估消融。FineWeb 的价值不只在 15T tokens 的规模，而在它把这些选择做成可执行代码和可比较实验：正文抽取和过滤要由下游训练结果验证，去重范围要看分布影响而不是只看删除比例，公开语料要同时交代数据、代码、字段注释和使用边界。

## 关键词

FineWeb；Common Crawl；DataTrove；WARC；Trafilatura；FastText；MinHash；质量过滤；预训练语料；数据消融

## 41.0 学习目标

通过本章学习，读者应能够：

- 区分 Common Crawl WARC 文件、WET 文本文件、抽取后的网页正文、过滤后的 FineWeb 文档和最终 token stream。
- 解释 FineWeb 为什么选择从 WARC 重新抽取正文，而不是直接使用 Common Crawl WET 文本。
- 理解 FineWeb 官方处理脚本中 `WarcReader`、`URLFilter`、`Trafilatura`、`LanguageFilter`、`GopherRepetitionFilter`、`GopherQualityFilter`、`C4QualityFilter`、`FineWebQualityFilter`、`MinhashDedup*` 和 `PIIFormatter` 的数据工程位置。
- 设计 Web 预训练文档记录 schema，使每条样本能追溯到来源、过滤、去重、token 统计和隐私处理状态。
- 用固定模型规模、固定 token 预算、固定评测集和重复随机种子比较不同数据处理策略。
- 将 FineWeb 的处理方式迁移到企业或研究项目时，识别版权、隐私、撤回、评测污染和跨语言迁移边界。

## 场景引入

一个团队准备训练 7B 参数级英文基础模型。第一版数据方案很直接：下载最近几年的 Common Crawl WET 文件，过滤掉非英文网页，做一次简单去重，然后把文本送进 tokenizer。离线抽样看起来还不错，很多页面确实是自然语言，token 数量也足够大。团队据此启动小模型预训练，却在几周后遇到三个难解释的问题。

第一，模型在若干常识题上没有随训练步数稳定提升。抽样回看训练片段，团队发现不少文本其实是网页菜单、页脚、cookie 横幅、SEO 关键词列表和自动生成的站内推荐。第二，同一类模板网页在不同月份和不同站点反复出现，训练 loss 看似平稳，但模型输出越来越容易复读模板化短句。第三，法务要求定位某个域名的样本和疑似邮箱地址，数据团队却只能在已经打散的 token shard 中模糊搜索，无法还原“这条文本来自哪个 crawl dump、哪条 URL、经过哪些过滤器、是否被去重保留”。

这三个问题说明，Common Crawl 只是网页快照，不是训练集。真正的数据工程任务不是“下载更多网页”，而是把每个网页样本转化为一条可追溯、可过滤、可去重、可评估、可撤回的训练记录。FineWeb 正是围绕这个问题建立的公开案例。

## 41.1 Common Crawl 是网页快照，不是训练语料

Common Crawl 的 WARC 文件保留网页抓取时的原始响应，包括 HTML、请求元数据和页面结构。WET 文件则是 Common Crawl 提供的文本抽取版本。对预训练数据工程来说，WET 的吸引力很强：它省去了 HTML 解析成本，体积更接近模型训练需要的文本。但 FineWeb 的实验发现，直接使用 WET 会留下过多 boilerplate、菜单文本和页面噪声，因此选择从 WARC 重新抽取正文。

### 41.1.1 网页快照到训练文本的处理层

从网页快照到训练文本，至少隔着五层变换。

第一层是 URL 层过滤。某些域名、路径或子词模式本身就带有高风险，例如恶意站点、成人内容站点或明显垃圾页面。FineWeb 官方数据卡把 URL filtering 放在流程第一步，用 block-list 和 subword detection 移除来自恶意和 NSFW 网站的文档。

第二层是正文抽取。HTML 页面不是正文，页面中会混入导航、页脚、脚本、推荐列表和广告。FineWeb 使用 Trafilatura 从 WARC 原始 HTML 中抽取主文本，并在论文中通过消融实验比较 WARC+Trafilatura 与 WET 的差异。

第三层是语言识别。FineWeb 是英文语料，因此使用 FastText 语言过滤，保留英文得分达到阈值的文档。官方数据卡说明，FineWeb 移除 `en` language score 低于 0.65 的文档。

第四层是质量过滤。网页文本中常见重复 n-gram、异常行长、过短行、列表化页面和格式错误。FineWeb 使用 Gopher repetition / quality filters、部分 C4 filters，以及 FineWeb 自定义过滤器共同控制这些问题。

第五层是去重与隐私处理。FineWeb 使用按 crawl 独立执行的 MinHash 去重，并在公开发布时使用 PIIFormatter 匿名化邮箱和公网 IP 地址。

这些步骤构成的不是“清洗脚本集合”，而是一组训练前的数据契约。每一步都应有输入、输出、失败记录和可复查的参数。

### 41.1.2 过滤强度决定训练信号

Web 语料清洗最容易出现两种相反错误。

过滤太少时，模型会吸收模板、乱码、广告、重复页面和非自然语言。它们可能在 token 统计上贡献很大，但对下游任务没有帮助，甚至会损伤语言分布。过滤太多时，语料规模下降，内容覆盖面变窄，一些长尾知识、论坛问答和非标准写作会被误删。对预训练来说，过滤器不是越严格越好，而是要在保留 token 预算和提升训练信号之间寻找可验证的平衡。

可用一个简单的训练收益函数描述这种权衡：

$$
U(F)=S_{eval}(D_F)-\lambda \cdot R_{risk}(D_F)-\mu \cdot \max(0, T_{target}-T_F)
$$

其中，$F$ 表示过滤策略，$D_F$ 是过滤后数据集，$S_{eval}$ 是固定评测协议下的模型得分，$R_{risk}$ 是隐私、版权、毒性和污染风险，$T_F$ 是保留 token 数，$T_{target}$ 是训练预算需要的 token 下限。这个公式不是 FineWeb 论文中的原始公式，而是对 FineWeb 实验思路的工程化抽象：最终选择过滤器时，不能只看样本“干净不干净”，还要看固定训练预算下模型是否变好，以及风险是否下降。

## 41.2 FineWeb 的数据定义和公开形态

FineWeb 的公开形态包括完整数据集、按 Common Crawl dump 切分的数据配置，以及较小的 sample 版本。官方数据卡说明，可以加载全量数据，也可以指定某个 crawl/dump；dump 名称采用 `CC-MAIN-(year)-(week number)` 格式。样本版本包括约 350B、100B 和 10B GPT-2 tokens 的随机子集，便于研究者在较低成本下复现实验或调试处理代码。

*表41-1 FineWeb 公开数据形态和工程用途*

| 形态 | 公开口径 | 工程用途 | 使用注意 |
| --- | ---: | --- | --- |
| FineWeb full dataset | 论文初版报告 15T tokens，官方数据卡持续列出后续 dump | 大规模英文 Web 预训练、数据消融、过滤策略研究 | 数据持续更新，引用规模时要说明数据卡访问时间和口径 |
| Per-dump config | 以 `CC-MAIN-YYYY-WW` 组织 | 按时间窗口抽样、复现实验、定位分布变化 | 不同 dump 的站点覆盖和质量不同，不能默认同分布 |
| `sample-350BT` | 约 350B GPT-2 tokens | 中等规模数据实验、去重和过滤策略验证 | 适合较大 ablation，不等于完整 FineWeb |
| `sample-100BT` | 约 100B GPT-2 tokens | 原型训练、快速评估、成本受限实验 | 需要记录抽样来源和随机性 |
| `sample-10BT` | 约 10B GPT-2 tokens | 管线调试、字段检查、读写性能测试 | 不适合得出最终数据质量结论 |

数据来源 Hugging Face FineWeb 数据卡的下载配置、sample 版本说明和 dump 命名规则；FineWeb 论文的初版规模口径。

### 41.2.1 任务定义

FineWeb 的任务不是标注一个监督学习标签，而是为自回归语言模型构建预训练 token stream。给定 Common Crawl 的网页快照集合 $C=\{c_i\}$，目标是学习一条数据处理函数：

$$
P_\theta: C \rightarrow D=\{d_j\}
$$

其中每条输出文档 $d_j$ 至少包含抽取文本、来源元数据、语言信息、token 统计、过滤状态和去重状态。随后 tokenizer 将文档集合映射为训练序列：

$$
\tau(D)=\left[x_1,x_2,\ldots,x_N\right]
$$

预训练模型的标准目标仍是最小化 next-token 负对数似然：

$$
\mathcal{L}(\theta)=-\sum_{t=1}^{N}\log p_\theta(x_t|x_{<t})
$$

FineWeb 关注的是这个目标函数之前的部分：如何选择 $P_\theta$，使得同样的模型、同样的训练 token 和同样的评测集下，训练出的模型更好。

FineWeb 留给工程团队的判断主要有三类。第一类是正文抽取问题。WET 文本已经是“文本”，但未必是“可训练正文”。FineWeb 用 WARC+Trafilatura 取代 WET，说明正文抽取本身需要被视为影响模型能力的核心变量。

第二类是去重粒度问题。全局去重看起来更彻底，但 FineWeb 的消融结果显示，对所有 crawl 做全局 MinHash 去重并不一定更好；按 crawl 独立去重反而表现更强。这提醒数据工程团队，去重目标不是最大限度删除重复，而是删除会损害训练的重复。

第三类是过滤器验证问题。FineWeb 没有只凭人工规则决定过滤器，而是训练多组数据消融模型，在固定评测集上比较分数。过滤器阈值、C4 规则取舍和自定义启发式规则，都是在训练回验中确定的。

## 41.3 FineWeb 文档记录的关键字段

FineWeb 官方数据卡说明，样本会带有 `language`、`language_score` 和 `token_count` 注释；这些字段分别来自语言过滤器和 GPT-2 tokenizer 统计。若在企业内部复刻 FineWeb 类流程，还需要把处理状态、来源、去重和风险字段一起保留下来。否则一旦训练结果异常，就无法判断问题来自抽取、过滤、去重还是采样。

*表41-2 FineWeb 类 Web 文档记录 schema*

| 字段组 | 典型字段 | 来源或生成方式 | 工程用途 |
| --- | --- | --- | --- |
| 来源字段 | `url`、`dump`、`warc_record_id`、`fetch_time` | WARC 元数据和读取器补充 | 追溯原始网页、定位 crawl、响应撤回 |
| 文本字段 | `text`、`raw_html_hash`、`text_hash` | Trafilatura 抽取与哈希计算 | 支持训练读取、抽取质量检查和精确定位 |
| 语言字段 | `language`、`language_score` | FastText LanguageFilter | 控制英文语料边界，排查语言识别误差 |
| 质量字段 | `gopher_flags`、`c4_flags`、`fineweb_flags` | Gopher、C4 和 FineWeb filters | 解释样本为何被保留或移除 |
| 去重字段 | `minhash_signature`、`dedup_cluster_id`、`dedup_keep` | MinHash 去重阶段 | 控制近似重复和复查被删除样本 |
| 统计字段 | `token_count`、`char_count`、`line_count` | TokensCounter 和文档统计 | 估算训练预算，分析过滤影响 |
| 隐私字段 | `pii_email_replaced`、`pii_ip_replaced` | PIIFormatter | 记录邮箱和公网 IP 匿名化状态 |

表中 `gopher_flags`、`c4_flags`、`fineweb_flags` 等字段是作者为解释工程结构补充的字段组，不代表 FineWeb 官方数据卡逐项发布了这些列。FineWeb 官方明确发布的注释包括 `language`、`language_score` 和 `token_count`。

### 41.3.1 一条样本不能只保存 text

下面是一条抽象化的 FineWeb 类文档记录。它不是 FineWeb 原始样本，而是根据 FineWeb 数据卡和 DataTrove 管线整理的工程示例。

```json
{
  "id": "CC-MAIN-2023-50/segment-x/warc-record-y",
  "url": "https://example.org/article",
  "dump": "CC-MAIN-2023-50",
  "text": "<Trafilatura 抽取后的正文>",
  "language": "en",
  "language_score": 0.94,
  "token_count": 1267,
  "filters": {
    "url_filter": "kept",
    "gopher_repetition": "kept",
    "gopher_quality": "kept",
    "c4_quality": "kept",
    "fineweb_quality": "kept"
  },
  "dedup": {
    "method": "minhash",
    "scope": "per_dump",
    "ngram": 5,
    "buckets": 14,
    "hashes_per_bucket": 8,
    "keep": true
  },
  "pii": {
    "email_formatted": true,
    "public_ip_formatted": true
  }
}
```

这个例子展示了 FineWeb 类语料的基本思想：`text` 是训练入口，但它不能单独解释样本质量。真正支撑复查的是来源、语言分数、过滤状态、去重范围和隐私处理记录。

### 41.3.2 schema 与训练评估的关系

FineWeb 的评估不是直接评估单条样本，而是评估由某个处理策略生成的数据版本。设某个处理版本 $v$ 对应数据集 $D_v$，训练得到模型 $M_v$。若评测集合为 $B=\{b_1,\ldots,b_k\}$，每个任务得分为 $s(M_v,b_i)$，则可以定义一个聚合得分：

$$
S(M_v)=\frac{1}{k}\sum_{i=1}^{k}s(M_v,b_i)
$$

FineWeb 论文采用固定模型、固定训练 token、固定评测任务、重复随机样本和不同初始化种子的方式比较数据版本。工程上可以进一步记录每个数据版本的处理 manifest：

$$
Manifest(v)=\{code\_commit, dump\_set, filter\_params, dedup\_params, tokenizer, sample\_seed\}
$$

没有这个 manifest，即使复现了评测脚本，也无法复现“到底训练了哪个数据版本”。

## 41.4 FineWeb 的代码化处理流程

FineWeb 的一个重要特点是处理过程有公开脚本。DataTrove 仓库中的 `examples/fineweb.py` 声明该文件用于处理和创建 FineWeb 数据集。脚本分为两大部分：先对每个 dump 做主处理，再对处理后的输出做 MinHash 去重和 PII 格式化。

### 41.4.1 主处理流水线

主处理流水线可抽象为以下顺序。类名来自 DataTrove 的 FineWeb 示例脚本，文字说明为本章整理。

*表41-3 FineWeb 主处理流水线中的关键模块*

| 顺序 | DataTrove 模块 | 输入 | 输出 | 作用 |
| ---: | --- | --- | --- | --- |
| 1 | `WarcReader` | Common Crawl WARC segments | 原始 HTML 文档流 | 从 `s3://commoncrawl/crawl-data/.../warc/` 读取网页快照 |
| 2 | `URLFilter` | URL 和原始文档 | 保留或移除文档 | 移除恶意、NSFW 或 block-list 命中的来源 |
| 3 | `Trafilatura` | 原始 HTML | 抽取正文文本 | 减少菜单、页脚和页面模板噪声 |
| 4 | `LanguageFilter` | 文本 | 英文文档流和非英文排除日志 | 保留英文得分达到阈值的文档 |
| 5 | `GopherRepetitionFilter` | 英文文本 | 重复模式过滤结果 | 移除重复 n-gram 和异常重复内容 |
| 6 | `GopherQualityFilter` | 文本统计 | 质量过滤结果 | 应用 MassiveText/Gopher 风格质量规则 |
| 7 | `C4QualityFilter` | 文本统计 | C4 规则过滤结果 | 应用 FineWeb 采用的 C4 规则子集 |
| 8 | `FineWebQualityFilter` | 文本统计 | 自定义过滤结果 | 移除列表化、重复行和异常换行文档 |
| 9 | `JsonlWriter` | 保留文档 | JSONL 分片 | 写出进入去重阶段的文档 |

数据来源 DataTrove `examples/fineweb.py` 的导入模块和主处理 pipeline；FineWeb 数据卡的 data processing steps。

主处理阶段的代码结构可以概括为：

```python
pipeline = [
    WarcReader(common_crawl_warc_path),
    URLFilter(...),
    Trafilatura(favour_precision=True),
    LanguageFilter(...),
    GopherRepetitionFilter(...),
    GopherQualityFilter(...),
    C4QualityFilter(...),
    FineWebQualityFilter(...),
    JsonlWriter(base_processing_output)
]
```

这段是概念化伪代码，用于说明 FineWeb 示例脚本中的模块顺序；真实参数、日志目录、S3 路径、任务数和 Slurm 资源配置以 DataTrove 仓库脚本为准。

### 41.4.2 去重和隐私处理流水线

FineWeb 使用 MinHash 做近似去重。MinHash 的目标是近似估计两个文档的 Jaccard 相似度。若文档 $A$ 和 $B$ 被表示为 5-gram 集合，则相似度可写为：

$$
J(A,B)=\frac{|G_5(A)\cap G_5(B)|}{|G_5(A)\cup G_5(B)|}
$$

MinHash 用多个哈希函数近似这个相似度。FineWeb 论文说明其去重参数为 5-grams、112 个哈希函数，拆成 14 个 bucket，每个 bucket 8 个 hash；任一 bucket 的 8 个 MinHash 相同即可判为重复候选。DataTrove 示例脚本中的 `MinhashConfig` 也对应 `n_grams=5`、`num_buckets=14`、`hashes_per_bucket=8`。

![图41-1 FineWeb MinHash 去重和 PII 处理流程](../../images/part12/ch41_01_fineweb_minhash_pii_flow.svg)

*图41-1 FineWeb MinHash 去重和 PII 处理流程。Source: original illustration based on Hugging Face DataTrove `examples/fineweb.py` and FineWeb dataset card.*

### 41.4.3 FineWeb 按 crawl 独立去重的判断

直觉上，全局去重似乎更彻底：把 96 个 crawl 放在一起，删除所有近似重复文档。FineWeb 的消融实验却给了相反信号。论文描述了一个关键现象：从最新 crawl 开始向旧 crawl 做全局去重时，旧 crawl 会被大量删除；在某个旧 snapshot 中，保留下来的 10% 数据反而比被删除的 90% 更差，包含更多广告、关键词列表和格式异常文本。最终，FineWeb 选择对每个 crawl 独立 MinHash 去重。

这个结果对工程实践很重要。去重不是数学上越彻底越好，而是要看它如何改变数据分布。全局去重会让新旧 crawl 之间的时间分布、站点覆盖和重复簇结构发生复杂变化；如果只看“删除了多少重复”，可能误删更有价值的样本，保留低质量长尾。

![图41-2 FineWeb 数据处理选择的消融评估回路](../../images/part12/ch41_02_fineweb_ablation_loop.svg)

*图41-2 FineWeb 数据处理选择的消融评估回路。Source: original illustration based on FineWeb paper Section 3.1.*

## 41.5 FineWeb 的数据处理选择评估

FineWeb 的评估方法与一般数据集介绍不同。它把数据处理步骤当成实验变量，通过训练 ablation models 比较不同数据版本。论文说明，数据消融模型在模型参数、架构超参数、训练 token 数和训练步数上保持一致；为了降低随机抽样影响，每个数据版本训练两个模型，使用不同随机子集和不同初始化种子，然后比较平均分。

### 41.5.1 固定变量

FineWeb 的评估协议可以抽象为表41-4。

*表41-4 FineWeb 数据消融评估协议*

| 控制项 | FineWeb 论文做法 | 数据工程意义 |
| --- | --- | --- |
| 模型规模 | ablation 模型为 1.82B 参数，Llama 架构 | 避免模型规模变化掩盖数据差异 |
| tokenizer | GPT-2 tokenizer | 固定 token 统计口径 |
| 训练预算 | 过滤消融约 28B tokens，部分去重和累计改进实验为 350B tokens | 区分快速筛选和高成本验证 |
| 重复实验 | 每个数据版本训练两个模型，随机子集和初始化种子不同 | 降低抽样和初始化噪声 |
| 训练框架 | Nanotron | 固定训练实现 |
| 评测框架 | lighteval | 固定评测实现 |
| 评测任务 | CommonSense QA、HellaSwag、OpenBook QA、PIQA、SIQA、WinoGrande、ARC、MMLU | 用多任务信号评估数据处理效果 |

数据来源 FineWeb paper Section 3.1 Experimental setup。

若第 $v$ 个数据版本训练两次，得到模型 $M_{v,1}$ 和 $M_{v,2}$，每个模型在 $k$ 个任务上得分，则版本得分可以写为：

$$
\bar{S}_v=\frac{1}{2k}\sum_{r=1}^{2}\sum_{i=1}^{k}s(M_{v,r},b_i)
$$

这个公式同样是对 FineWeb 评估协议的工程化表达。它强调评估对象不是单个样本，而是“处理策略生成的数据版本”。

过滤器不能只靠规则直觉一次决定。FineWeb 的过滤器选择可以分为三步。

第一步，建立基础过滤。FineWeb 从 WARC 抽取文本后，先做 URL block-list、英文语言识别和 Gopher/MassiveText 风格质量过滤。论文报告，在对 96 个 snapshot 的 WARC 抽取文本应用这些基础步骤后，得到约 36T GPT-2 tokens 的数据。

第二步，比较已有规则。FineWeb 研究 C4 规则时发现，terminal punctuation 规则单独带来明显提升，但会删除约 30% tokens；最终 FineWeb 采用除 terminal punctuation 外的 C4 规则子集，因为后者删除更少数据并取得更合适的训练收益。

第三步，设计自定义过滤器。FineWeb 收集 50 多个文档级和跨文档统计指标，比较“较高质量”和“较低质量”数据分布，选择能区分两者的阈值，再用 28B token ablation runs 验证。最终被采用的自定义过滤器关注三类问题：行尾标点比例过低、重复行字符比例过高、短行比例异常。

### 41.5.3 常见失败和修复动作

FineWeb 的经验可以转化为一张 Web 预训练语料错误归因表。它不是 FineWeb 官方表格，而是本章根据 FineWeb 论文和数据卡整理的工程复盘。

*表41-5 FineWeb 类 Web 语料常见失败与修复动作*

| 错误类型 | 现象 | 可能根因 | 数据工程修复动作 |
| --- | --- | --- | --- |
| 页面模板残留 | 模型复读菜单、页脚、cookie 文案 | 直接使用 WET 或正文抽取质量差 | 回到 WARC，用 Trafilatura 或同类工具重新抽取，并抽样检查模板残留 |
| 非英文混入 | 英文模型训练中出现多语乱码和混排 | 语言识别阈值过宽或段落混语未处理 | 保留 `language_score`，按分桶抽检，必要时段落级过滤 |
| 重复簇过大 | loss 虚高稳定但下游无提升 | 模板站点、镜像站点、跨月重复 | 使用 MinHash 去重，并记录重复簇和去重范围 |
| 全局去重伤害分布 | 删除大量旧 crawl 内容后模型不变好 | 全局去重改变时间和质量分布 | 比较 per-crawl 与 global dedup，固定训练预算回验 |
| 过滤器过严 | token 规模下降，长尾知识被删 | 单条规则删除比例过高 | 记录每个过滤器 token removal rate，用 ablation 决定阈值 |
| 隐私样本残留 | 邮箱、公网 IP 等可识别信息进入发布数据 | PII 处理缺失或误检漏检 | 使用 PIIFormatter 或同类规则，记录替换策略和边界 |

## 41.6 公开 Web 语料的使用边界

FineWeb 是开放 Web 预训练语料工程的强案例，但它不能被简单理解成“可以直接商用的一切网页文本”。公开数据集、开放代码和 ODC-By 许可证降低了研究复现门槛，却不自动消除使用者所在司法辖区、业务场景和下游模型发布中的版权、隐私、安全与撤回责任。

FineWeb 适合用于英文基础模型预训练、Web 数据过滤策略研究、去重策略消融、DataTrove 类大规模文本处理管线调试，以及预训练语料版本治理教学。它尤其适合回答“某个数据处理步骤是否让模型变好”这类问题，因为 FineWeb 的公开资料提供了代码、数据卡、论文消融和评估协议。

FineWeb 不适合直接回答所有语言、所有领域、所有合规环境下的训练数据问题。它主要由 Common Crawl 英文 Web 内容构成，不等价于中文语料、专业版权语料、医疗法律财务领域语料，也不等价于面向对话助手的 SFT 或偏好数据。

企业复刻 FineWeb 思路时，最值得迁移的不是某个固定阈值，而是四类工程对象。

第一，处理代码要版本化。`code_commit`、过滤器参数、tokenizer、抽样种子和 dump 列表都应进入 manifest。第二，过滤器要有排除日志。每个被删除样本最好能说明被哪个规则删除。第三，去重要保留范围和参数。per-dump、per-domain、global dedup 的影响不同，不能只记录“已去重”。第四，评估要固定变量。模型结构、训练 token、训练步数、评测集和随机种子不固定，数据处理结论就不可比。

FineWeb 不应被用作无审查商业训练全集。对于商业模型，仍需做许可证、robots/terms、数据撤回、隐私和敏感内容审查。FineWeb 也不应被当作“高质量英文语料”的唯一标准；它的质量定义来自固定 ablation 模型和一组学术 benchmark，不必然覆盖产品真实使用中的帮助性、安全性、事实更新和指令遵循需求。

对于中文或多语训练，不能直接照搬 FineWeb 的英文 FastText 阈值、英文 tokenizer 统计和英文页面格式假设。迁移时需要重新校准语言识别、繁简处理、站点模板、域名分布、低质量页面规则和评估任务。

## 本章小结

FineWeb 讲清楚了一个常被低估的问题：Web 预训练语料不是 Common Crawl 的下载结果，而是代码、过滤器、去重策略、评估协议和发布说明共同构成的数据资产。本章的核心结论有三点。

第一，网页正文抽取是模型能力变量。FineWeb 从 WARC 使用 Trafilatura 重新抽取正文，正是因为 WET 文本会残留过多模板和菜单噪声。第二，去重策略必须用训练结果验证。FineWeb 的实验表明，全局 MinHash 去重不一定优于按 crawl 独立去重，删除更多重复不等于得到更好的训练数据。第三，过滤器选择应当进入固定评估协议。FineWeb 通过同构 ablation 模型、固定 token 预算、lighteval 评测和重复随机种子，把数据处理选择变成可复查的工程实验。

对本书读者而言，FineWeb 最值得学习的不是照搬某个 token 规模，而是把预训练语料工程做成一套可追溯、可复现、可评估、可审计的系统。

## 参考文献

- Penedo, G., Kydlíček, H., Allal, L. B., Lozhkov, A., Mitchell, M., Raffel, C., von Werra, L., & Wolf, T. (2024). The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. NeurIPS 2024 Datasets and Benchmarks Track. https://arxiv.org/abs/2406.17557
- Hugging Face. (2026). HuggingFaceFW/fineweb Dataset Card. https://huggingface.co/datasets/HuggingFaceFW/fineweb
- Hugging Face. (2026). DataTrove FineWeb Processing Script. https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py
- Penedo, G., Kydlíček, H., Cappelli, A., Sasko, M., & Wolf, T. (2024). DataTrove large scale data processing. https://github.com/huggingface/datatrove
- Luccioni, S., & Viviano, J. (2021). What's in the Box? A Preliminary Analysis of Undesirable Content in the Common Crawl Corpus. https://arxiv.org/abs/2105.02732
