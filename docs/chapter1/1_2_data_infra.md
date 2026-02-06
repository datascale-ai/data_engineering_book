# 第2章：数据基础设施选型

---

## 本章摘要

工欲善其事，必先利其器。在处理 TB 级甚至 PB 级的 LLM 训练数据之前，选择正确的基础设施是决定项目成败的第一步。本章将从**存储、计算、格式、版本控制**四个维度，系统性地介绍现代数据栈的技术选型，帮助读者建立一套高效、可扩展、可复现的数据处理平台。

---

## 场景引入

你刚加入一家 AI 初创公司，负责搭建 LLM 预训练数据处理平台。团队的现状令人担忧：数据散落在 50 台机器的本地硬盘上，格式五花八门，包括 `.txt`、`.json`、`.csv`、`.parquet` 等各种类型。每次处理数据，都要手动编写 Python 脚本，在单机上运行三天才能跑完。上周有人不小心覆盖了一个关键数据集，而这份数据没有任何备份和版本记录。老板问你："一个月后我们要开始训练，数据平台能不能 ready？"

你面临的第一个决策是：是用团队熟悉的 Spark，还是转向号称"AI 原生"的 Ray？是自建 MinIO 集群，还是直接上云用 S3？这些问题没有"标准答案"，但有明确的决策框架。本章将为你提供这个框架。

---

## 2_1 现代数据栈 (Modern Data Stack)

### 2_1.1 什么是现代数据栈？

"现代数据栈"（Modern Data Stack, MDS）是近年来数据工程领域的热门概念，指的是一套云原生、模块化、解耦合的数据基础设施组合。与传统的一体化数据平台相比，现代数据栈的核心理念是将存储、计算、编排等功能拆分到独立的组件中，每个组件可以根据需求独立替换和扩展。

![图2-1：现代数据栈架构](../images/chapter1/图2_1_现代数据栈架构.png)

*图2-1：现代数据栈架构 —— 从存储层到应用层的5层解耦架构，每层可独立替换*

传统数据平台往往部署在本地机房，采用一体化系统，存储与计算紧密绑定。以 Hadoop 生态为例，HDFS 与 MapReduce 的耦合使得更换任何一个组件都非常困难。数据格式也常常是私有的，导致严重的厂商锁定问题。扩展方式以垂直扩展为主，即通过购买更强大的单机来提升性能，前期投入成本高昂。

| 特征 | 传统方案 | 现代数据栈 |
|------|----------|-----------|
| **部署模式** | 本地机房，一体化系统 | 云原生，按需弹性伸缩 |
| **组件耦合** | 存储计算绑定（如 HDFS + MapReduce） | 存储计算分离，各层可独立替换 |
| **数据格式** | 私有格式、厂商锁定 | 开放格式（Parquet、ORC） |
| **扩展性** | 垂直扩展为主 | 水平扩展，近乎无限 |
| **成本模式** | 固定投入，前期成本高 | 按用量付费，弹性成本 |

现代数据栈的出现改变了这一局面。云原生的部署模式允许按需弹性伸缩，存储与计算完全分离使得各层可以独立演进。开放的数据格式（如 Parquet、ORC）消除了厂商锁定风险。水平扩展能力使得系统可以处理近乎无限的数据量，而按用量付费的成本模式则大大降低了项目启动的门槛。

### 2_1.2 存储层：对象存储与数据湖

对象存储是现代数据平台的事实标准底座。无论是 AWS S3、Google Cloud Storage、Azure Blob，还是开源的 MinIO，它们的核心理念相同：采用扁平化命名空间，没有真正的目录层级，只有 `bucket/key` 的二元结构；理论上可存储无限数据；提供极高的数据持久性（S3 声称达到 11 个 9，即 99_999999999%）；按照实际使用量计费，无需前期大额投入。

在具体选型时，需要考虑部署模式、兼容性、成本等多个因素。AWS S3 是公有云托管的标杆产品，生态最为成熟，适合绝大多数生产环境。MinIO 是 S3 兼容的开源替代方案，适合有数据合规要求需要私有部署的场景，或用于开发测试环境。Google Cloud Storage 和 Azure Blob 分别适合已经深度使用 GCP 或 Azure 生态的用户。

| 特性 | AWS S3 | MinIO | Google GCS | Azure Blob |
|------|--------|-------|------------|------------|
| **部署模式** | 公有云托管 | 自建/私有云 | 公有云托管 | 公有云托管 |
| **S3 兼容性** | 原生 | 100% 兼容 | 需适配层 | 需适配层 |
| **冷热分层** | Glacier | Tiering | Nearline/Coldline | Cool/Archive |
| **最低成本** | $0_023/GB/月 | 硬件成本 | $0_020/GB/月 | $0_018/GB/月 |
| **适用场景** | 生产环境首选 | 私有部署/开发测试 | GCP 生态用户 | Azure 生态用户 |

对象存储解决了"存储"问题，但缺乏事务性和元数据管理能力。直接在 S3 上操作 Parquet 文件会遇到诸多困难：无法进行 ACID 事务，并发写入可能损坏数据；无法高效查询，每次都要扫描所有文件的元数据；无法进行时间旅行，一旦数据被覆盖就无法回滚到历史版本。

数据湖表格式（Table Format）正是为解决这些问题而生。它在对象存储之上增加了一层元数据管理，提供了数据仓库级别的能力。Apache Iceberg、Apache Hudi 和 Delta Lake 是目前最主流的三种数据湖格式。

![图2-2：数据湖仓架构](../images/chapter1/图2_2_数据湖仓架构.png)

*图2-2：数据湖仓架构 —— 表格式层提供ACID事务、时间旅行、Schema演进等能力*

Apache Iceberg 由 Netflix 开发并贡献给 Apache 基金会，最大的优势是引擎中立性——它可以与 Spark、Flink、Trino、Dremio、DuckDB 等多种计算引擎良好配合。对于 LLM 数据工程场景，Iceberg 是最推荐的选择。Apache Hudi 由 Uber 开发，特点是对流批一体和实时更新的支持较好，如果有大量实时更新需求（如 RAG 知识库的持续更新），可以考虑 Hudi。Delta Lake 由 Databricks 开发，与 Spark 的集成最为紧密，如果已经深度使用 Databricks 生态，选择 Delta Lake 可以获得最佳体验。

| 特性 | Apache Iceberg | Apache Hudi | Delta Lake |
|------|----------------|-------------|------------|
| **背后厂商** | Netflix → Apache | Uber → Apache | Databricks |
| **开源程度** | 完全开源 | 完全开源 | 核心开源，部分功能商业 |
| **引擎兼容性** | Spark, Flink, Trino, DuckDB | Spark, Flink, Presto | 主要 Spark |
| **适用场景** | 多引擎混用、厂商中立 | 流批一体、实时更新 | Databricks 生态用户 |

在实际选型时，可以按照以下决策树进行：首先判断数据规模是否超过 100TB。如果超过，进一步考虑是否需要 ACID 事务和时间旅行能力——如果需要，且有多引擎访问需求，推荐 Iceberg + S3；如果只用 Spark，可以选择 Delta Lake 或 Hudi。如果不需要 ACID 能力，直接使用 S3/MinIO + Parquet 即可。对于数据量在 100TB 以下的场景，如果团队规模较小（少于 5 人），本地磁盘 + Parquet 足以满足原型验证需求；随着规模增长，再逐步迁移到 S3 + Parquet 的方案。

![图2-3：存储层选型决策树](../images/chapter1/图2_3_存储层选型决策树.png)

*图2-3：存储层选型决策树 —— 根据数据规模、ACID需求、多引擎访问等因素选择最佳方案*

---

### 2_1.3 计算层：Spark vs Ray Data

这是 LLM 数据工程中最常见的"二选一"难题。两者都是分布式计算框架，但设计哲学和适用场景截然不同。理解它们的差异，对于做出正确的技术选型至关重要。

Apache Spark 诞生于 2009 年的 Berkeley AMPLab，经过十五年发展，已成为大数据处理的"瑞士军刀"。Spark 的核心优势在于其成熟稳定——经过 PB 级生产验证，文档和社区资源极其丰富。Spark SQL 的存在使得数据分析师也能编写分布式处理逻辑，降低了使用门槛。Structured Streaming 支持实时数据处理，实现了流批一体。然而，Spark 也有明显的劣势：由于核心是 JVM 实现，Python UDF 需要跨 JVM-Python 序列化，性能损耗较大；对 GPU 和 PyTorch/TensorFlow 的集成支持较弱，不够"AI 原生"；算子之间必须物化中间结果，内存压力较大。

Ray 诞生于 2017 年的 Berkeley RISELab，最初是一个分布式强化学习框架，后演变为通用的 AI 应用基础设施。Ray Data 是其数据处理模块，专为 AI 工作负载设计。Ray Data 的核心优势在于 Python 原生——没有 JVM 开销，与 PyTorch、HuggingFace 等 AI 生态无缝集成。它天然支持流水线式执行，内存效率高；内置 GPU 调度，轻松调用 CUDA 算子；Actor 模型适合有状态的复杂处理，如需要加载 ML 模型的推理任务。不过，Ray 相对年轻，文档和最佳实践不如 Spark 丰富；SQL 支持较弱，没有 Spark SQL 那样成熟的 SQL 接口；与传统大数据生态（Hive、Iceberg）的集成需要额外工作。

| 维度 | Apache Spark | Ray Data |
|------|--------------|----------|
| **语言** | Scala/Java 核心，Python API | Python 原生 |
| **运行时** | JVM | Python (Arrow-based) |
| **数据抽象** | DataFrame (批处理思维) | Dataset (流处理思维) |
| **GPU 支持** | 需要 RAPIDS 插件 | 原生支持 |
| **PyTorch 集成** | 繁琐 | 一等公民 |
| **SQL 支持** | 非常成熟 | 有限 |
| **典型用户** | 传统大数据团队 | AI/ML 团队 |

为了更直观地理解两者的差异，我们来看一个具体的代码对比。假设任务是：读取 Parquet 文件，过滤短文本，计算文本长度，保存结果。

使用 Spark 实现：

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import length, col

# 初始化 Spark Session
spark = SparkSession.builder \
    .appName("TextFilter") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# 读取 → 过滤 → 计算 → 保存
df = spark.read.parquet("s3://my-bucket/raw_data/")
df_filtered = df.filter(length(col("text")) > 100) \
                .withColumn("text_length", length(col("text")))
df_filtered.write.parquet("s3://my-bucket/processed_data/")

spark.stop()
```

使用 Ray Data 实现：

```python
import ray

# 初始化 Ray（自动检测集群资源）
ray.init()

# 定义处理函数
def filter_and_compute(batch):
    mask = batch["text"].str.len() > 100
    filtered = batch[mask].copy()
    filtered["text_length"] = filtered["text"].str.len()
    return filtered

# 读取 → 处理 → 保存（流水线执行）
ds = ray.data.read_parquet("s3://my-bucket/raw_data/")
ds_processed = ds.map_batches(filter_and_compute, batch_format="pandas")
ds_processed.write_parquet("s3://my-bucket/processed_data/")
```

可以看到，Spark 需要显式配置 Executor 内存，使用声明式 DataFrame API；而 Ray 自动发现资源，使用函数式 `map_batches` 接口。Spark 中的自定义逻辑需要定义 UDF，存在序列化开销；Ray 中直接使用普通 Python 函数，更加自然。

![图2-4：计算框架选型决策树](../images/chapter1/图2_4_计算框架选型决策树.png)

*图2-4：计算框架选型决策树 —— Spark适合SQL/ETL场景，Ray适合GPU/ML场景*

在实际决策时，可以按照以下逻辑进行：如果数据处理需要使用 GPU（如调用 BERT 模型进行质量评分），Ray Data 是更自然的选择。如果有大量 SQL 和 BI 查询需求，Spark 的 SQL 生态更为成熟。如果已有大量 Spark 基础设施和代码资产，需要评估迁移成本——成本高则保留 Spark，成本低可考虑逐步引入 Ray。如果是新项目，团队背景是决定性因素：传统大数据团队选 Spark 更容易上手，AI/ML 团队选 Ray 更加顺畅。

值得一提的是，在实际大型项目中，Spark 和 Ray 往往共存而非互斥。一种常见的混合策略是：Spark 负责与数据湖/数据仓库的交互，包括读写 Iceberg/Hive 表、执行 SQL 分析等 ETL 任务；Ray Data 负责 ML 密集型处理，如调用大模型进行推理、使用 GPU 进行批量处理。两者通过共享的对象存储（S3 上的 Parquet 文件）进行数据交换，各司其职，相得益彰。

---

## 2_2 数据格式与 I/O 优化

选对了存储和计算，接下来要选择数据的序列化格式。格式选择看似是一个技术细节，实际上直接影响存储成本、读取速度和工具兼容性。不同格式的压缩率差异可达十倍之多，列式与行式格式的查询性能差异同样巨大，而且并非所有框架都支持所有格式。

### 2_2.1 主流数据格式对比

**Parquet** 是目前大规模结构化数据的事实标准。它采用列式存储，相同列的数据物理上连续存放，这带来两个显著优势：一是利于压缩，相同类型的数据聚集在一起可以获得更高的压缩率；二是利于向量化读取，查询特定列时无需扫描整个文件。Parquet 文件是自描述的，Schema 嵌入在文件中，无需外部元数据定义。它还支持嵌套类型，可以存储类似 JSON 的复杂结构，并且天然支持目录分区。Parquet 是预训练语料存储的首选格式，特别是需要按列过滤的分析查询场景，与 Spark、DuckDB、Pandas 等工具都能良好配合。

**JSONL**（JSON Lines）是另一种常见格式，每行是一个独立的 JSON 对象。它的最大优势是人类可读——可以用 `head`、`cat` 等命令直接查看内容。同时它支持流式处理，可以逐行读取而无需加载整个文件到内存。Schema 非常灵活，每行可以有不同的字段结构。JSONL 特别适合 SFT 指令数据，因为这类数据需要频繁人工查看和编辑。它也常用于数据交换和小规模数据集（10GB 以下）。然而，JSONL 的劣势也很明显：无压缩时体积是 Parquet 的三到五倍，读取速度较慢（需要解析每一行的 JSON 字符串）。

**WebDataset** 是 NVIDIA 主导开发的格式，专为图文、视频等多模态数据设计。它的核心思想是将相关文件（如一张图片和对应的描述文本）打包成 TAR 归档。这种设计支持流式读取，无需解压即可顺序读取内容；同时对分布式处理非常友好，每个 TAR 是独立的数据分片。WebDataset 是 LAION 风格图文对数据集和视频数据集的最佳选择，适用于任何需要多文件关联的多模态数据。

| 特性 | Parquet | JSONL | WebDataset |
|------|---------|-------|------------|
| **存储效率** | 高（列式压缩） | 低（文本冗余大） | 中（无压缩但紧凑） |
| **读取速度** | 快（向量化） | 慢（逐行解析） | 中（顺序读取） |
| **人类可读** | 否 | 是 | 否 |
| **多模态支持** | 弱（需编码） | 弱 | 强（原生支持） |
| **典型用例** | 预训练文本语料 | SFT 指令数据 | 图文对、视频数据 |

### 2_2.2 压缩算法选型

无论选择何种格式，压缩算法都会显著影响存储成本和读取速度。正确的压缩策略需要在空间效率和时间效率之间找到平衡。

Snappy 是最常用的默认选择。它的压缩率中等，但压缩和解压速度都很快，适合读写均衡的场景。LZ4 追求极致的读取速度，解压性能甚至比 Snappy 更快，压缩率略低，适合对读取延迟敏感的场景。Zstandard（ZSTD）提供最高的压缩率，尤其是高级别（如 level 19）时，但压缩速度较慢，适合存储成本敏感的归档场景。Gzip 是兼容性最好的选择，几乎所有工具都支持，适合需要与外部系统交换数据的场景。

| 算法 | 压缩率 | 压缩速度 | 解压速度 | 适用场景 |
|------|--------|----------|----------|----------|
| **Snappy** | 中等 | 快 | 快 | 默认选择，读写均衡 |
| **LZ4** | 较低 | 极快 | 极快 | 极致读取速度 |
| **ZSTD** | 高 | 中等 | 快 | 存储成本敏感 |
| **Gzip** | 高 | 慢 | 中等 | 兼容性要求高 |

在实践中，可以采用分层策略：冷数据（归档存储，长期不读取）使用 ZSTD level 19 以获得最大压缩率；热数据（频繁读取处理）使用 Snappy 或 LZ4 以减少解压开销；网络传输场景使用 ZSTD level 3，在压缩率和速度之间取得平衡。

### 2_2.3 I/O 优化实战技巧

在大规模数据处理中，I/O 往往是性能瓶颈。以下三个技巧可以显著提升 I/O 效率。

**合理设置文件大小**是第一个关键点。常见的错误是生成大量小文件——例如 10 万个 1MB 的文件。这会导致元数据开销巨大，S3 的 ListObjects 操作变得极慢。正确的做法是将数据合并成少量大文件，每个 Parquet 文件的大小应在 128MB 到 1GB 之间。太小会导致元数据膨胀和并行度不足，太大会影响任务的负载均衡。

```python
# 错误：生成大量小文件
df.write.parquet("s3://bucket/data/", maxRecordsPerFile=1000)

# 正确：生成少量大文件（推荐 128MB - 1GB）
df.coalesce(100).write.parquet("s3://bucket/data/")
```

**分区裁剪（Partition Pruning）**是第二个重要技巧。通过在写入时按特定列进行分区，读取时可以只扫描需要的分区，避免全表扫描。分区列应选择低基数（Cardinality）的列，如日期、语言、数据来源；应避免高基数列（如用户 ID），否则会产生海量小目录。

```python
# 写入时按日期分区
df.write.partitionBy("date").parquet("s3://bucket/data/")

# 读取时只扫描需要的分区
spark.read.parquet("s3://bucket/data/date=2024-01-01/")
```

**列裁剪（Column Pruning）**是第三个技巧。列式存储的最大优势就是只读取需要的列。确保在查询语句中尽早进行列选择，避免先读取全部列再过滤。

```python
# 错误：读取全部列
df = spark.read.parquet("s3://bucket/data/")  # 如果有 100 列，全部加载

# 正确：只读取需要的列
df = spark.read.parquet("s3://bucket/data/").select("text", "length")
```

![图2-5：I/O优化效果对比](../images/chapter1/图2_5_IO优化效果对比.png)

*图2-5：I/O优化效果对比 —— 分区裁剪+列裁剪可减少91%查询时间和92%数据扫描量*

综合使用这三个技巧，可以将查询时间从 55 秒降低到 5 秒，数据扫描量从 100GB 降低到 8GB，效果非常显著。

---

## 2_3 数据版本控制 (DataOps)

代码有 Git，机器学习模型有 MLflow，那么 TB 级数据集如何进行版本控制？这是 LLM 数据工程中经常被忽视但极其重要的问题。

### 2_3.1 为什么数据需要版本控制？

考虑这样一个场景：六个月前训练的模型效果特别好，老板要求复现。你翻遍服务器，发现当时的训练数据已被清理——"谁让你删的？""它占了 10TB 啊！"数据处理脚本倒是还在，但依赖的上游数据也变了。重新跑一遍处理流程，发现结果和之前不一样。结论：无法复现。

这个场景在实际工作中屡见不鲜。数据版本控制正是为了解决这类问题而存在。它的核心价值体现在四个方面：可复现性——任意时刻可以精确还原当时的数据状态；可追溯性——追踪数据从原始输入到最终输出的完整链路；协作安全——多人同时修改数据不会产生冲突；回滚能力——发现数据问题时可以快速回到之前的版本。

### 2_3.2 工具选型：DVC vs LakeFS

目前最主流的两个数据版本控制工具是 DVC 和 LakeFS，它们的设计哲学截然不同。

**DVC（Data Version Control）**的设计哲学是"Git for Data"——让数据版本控制的体验尽可能接近 Git。其工作原理是：数据文件本身存储在远程存储（S3/GCS），Git 仓库只保存数据的元数据文件（`.dvc` 文件），通过 `dvc push/pull` 命令同步实际数据。

```bash
# 初始化 DVC
dvc init

# 将数据集纳入版本控制
dvc add data/training_corpus.parquet
# 会生成 data/training_corpus.parquet.dvc 和 .gitignore

# 提交到 Git
git add data/training_corpus.parquet.dvc .gitignore
git commit -m "Add training corpus v1"

# 推送数据到远程存储
dvc push

# 切换到历史版本
git checkout v1_0
dvc checkout  # 同步对应版本的数据
```

DVC 的优势在于与现有 Git 工作流无缝集成，学习曲线平缓，支持 ML 流水线定义（通过 `dvc.yaml`），适合文件级别的版本控制场景。其劣势是每个数据集需要单独的 `.dvc` 文件管理，不支持细粒度的"表级"操作（如回滚某个分区）。

**LakeFS** 的设计哲学是"Git for Data Lake"——在对象存储之上提供 Git 风格的分支和提交。其工作原理是：LakeFS 作为对象存储的代理层，所有读写请求通过 LakeFS 的 S3 网关，系统支持分支（Branch）、提交（Commit）、合并（Merge）等 Git 风格的操作。

```bash
# 创建开发分支
lakectl branch create lakefs://repo/dev --source lakefs://repo/main

# 在开发分支上修改数据（通过 S3 协议）
aws s3 cp new_data.parquet s3://lakefs-repo/dev/data/

# 提交更改
lakectl commit lakefs://repo/dev -m "Add new training data"

# 验证通过后合并到主分支
lakectl merge lakefs://repo/dev lakefs://repo/main
```

LakeFS 的核心优势是零拷贝分支——创建分支不复制数据，只记录元数据，这对于 TB 级数据湖来说至关重要。它完全 S3 兼容，现有工具（Spark/Ray）无需修改即可使用。其劣势是需要部署额外的服务（LakeFS Server），学习曲线比 DVC 略陡。

| 特性 | DVC | LakeFS |
|------|-----|--------|
| **设计理念** | Git 的数据扩展 | 对象存储的版本层 |
| **粒度** | 文件级 | 对象级（更细） |
| **分支开销** | 需复制 .dvc 文件 | 零拷贝 |
| **S3 兼容** | 需要 dvc 命令 | 原生 S3 API |
| **部署复杂度** | 低（CLI 工具） | 中（需要服务端） |
| **适合场景** | ML 实验管理、少量数据 | 数据湖管理、大规模数据 |

![图2-6：DVC vs LakeFS架构对比](../images/chapter1/图2_6_DVC与LakeFS架构对比.png)

*图2-6：DVC vs LakeFS架构对比 —— DVC基于Git的文件级版本控制，LakeFS提供零拷贝分支的对象级版本控制*

选型建议非常明确：如果数据量在 1TB 以下，团队熟悉 Git 工作流，主要用于 ML 实验管理，选择 DVC；如果数据量在 TB 级以上，需要数据湖级别的版本控制，有多个团队并行操作，选择 LakeFS。

### 2_3.3 数据血缘追踪 (Data Lineage)

版本控制解决了"数据是什么"的问题，血缘追踪则解决了"数据从哪来"的问题。血缘追踪记录的信息包括：这份数据是由哪些上游数据处理得来的？使用了什么处理脚本和参数？何时、由谁执行的处理？

实现血缘追踪有多种方案。如果使用 Spark，可以通过 OpenLineage 集成获得自动化的血缘追踪。如果使用 Airflow 等编排工具，Marquez 是一个很好的选择。对于企业级数据治理需求，DataHub 和 Apache Atlas 提供了更完善的功能。对于简单场景，手动埋点生成元数据文件也是一种轻量级的解决方案：

```python
import json
from datetime import datetime

metadata = {
    "version": "v2_0",
    "created_at": datetime.now().isoformat(),
    "created_by": "data-pipeline-v3_2",
    "inputs": [
        {"path": "s3://bucket/raw/crawl_2024_01.parquet", "version": "abc123"},
        {"path": "s3://bucket/raw/crawl_2024_02.parquet", "version": "def456"}
    ],
    "processing": {
        "script": "cleaning_pipeline.py",
        "git_commit": "789xyz",
        "params": {"min_length": 100, "dedup_threshold": 0_9}
    },
    "outputs": [
        {"path": "s3://bucket/processed/clean_2024_q1.parquet", "records": 1000000}
    ]
}

with open("clean_2024_q1.metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

---

## 2_4 常见错误与避坑指南

在基础设施选型过程中，即使是经验丰富的工程师也容易犯一些典型错误。这里总结三个最常见的问题，希望读者能够引以为戒。

**第一个常见错误是过早优化、过度工程。** 有些团队规模只有五人，数据量仅 500GB，却搭建了 Spark 集群 + Iceberg + Airflow + LakeFS 的"全栈"基础设施。结果是 80% 的时间花在维护基础设施上，只有 20% 的时间用于实际数据处理。正确的做法是从简单开始，按需演进。500GB 的数据量，单机 + Parquet + DVC 完全够用，等数据量增长到 10TB 时再考虑分布式方案也不迟。

**第二个常见错误是盲目追新、忽视生态。** 有些团队看了几篇博客，决定抛弃 Spark 全面转向 Ray，结果发现公司的 Hive 表、Iceberg 表全部无法直接读取。最终需要额外编写大量数据转换脚本，增加了数据一致性风险。正确的做法是在做技术选型前，充分评估现有数据资产和上下游依赖。技术选型不是单点决策，而是系统工程，需要考虑整体生态的兼容性。

**第三个常见错误是存储成本优化过激。** 有些团队为了节省存储费用，把所有数据压缩到 ZSTD level 22，并存入 S3 Glacier Deep Archive。结果每次需要读取数据，要等 12 小时解冻，解压又要 4 小时，模型训练一次要排期一周。正确的做法是区分冷热数据。活跃处理的数据放在 S3 Standard + Snappy 压缩；六个月以上不使用的归档数据再放入 Glacier。存储成本和访问效率需要找到平衡点。

---

## 2_5 本章小结

本章系统介绍了 LLM 数据工程的基础设施选型，涵盖存储、计算、格式和版本控制四个核心维度。

在存储选型方面，对象存储（S3/MinIO）是现代数据栈的基础设施，数据湖格式（Iceberg/Hudi/Delta）解决了 ACID 事务、时间旅行等问题。对于 LLM 场景，推荐组合是 S3 + Iceberg，因为 Iceberg 的引擎中立性最好。

在计算选型方面，Spark 以其成熟稳定和强大的 SQL 生态著称，适合传统大数据团队；Ray Data 是 Python 原生的 AI 友好框架，适合 ML/AI 团队。两者并不互斥，可以混用：Spark 负责 ETL，Ray 负责 ML 处理。

在数据格式方面，Parquet 是结构化数据的默认选择，JSONL 适合需要人工查看的小规模数据，WebDataset 是多模态数据的最佳格式。压缩算法和 I/O 优化技巧可以显著影响性能和成本。

在版本控制方面，DVC 轻量级且与 Git 紧密集成，适合 ML 实验；LakeFS 提供数据湖级别的版本控制，适合大规模生产环境。

贯穿始终的核心原则是：从简单开始，按需演进，避免过度工程。技术选型应该服务于业务目标，而非为了追求技术先进性。

![图2-7：基础设施选型速查表](../images/chapter1/图2_7_基础设施选型速查表.png)

*图2-7：数据基础设施选型速查表 —— 存储、表格式、计算、版本控制四象限决策指南*

---

## 延伸阅读

对于希望深入了解本章内容的读者，以下资源值得参考：

Ray Data 官方文档（docs.ray.io）提供了 Ray Data 的最佳实践和详细 API 说明。Apache Iceberg 官方文档（iceberg.apache.org）包含表格式的详细规范和各引擎集成指南。DVC 官方教程（dvc.org/doc）是快速入门的好起点。LakeFS 官方文档（docs.lakefs.io）详细介绍了架构设计和部署方案。

Databricks 发布的数据湖选型白皮书对 Delta、Iceberg、Hudi 三种格式进行了深度对比分析。Uber 发表的"Scaling MLOps at Uber"一文介绍了如何在 PB 级规模管理 ML 数据。这些资料可以帮助读者建立更全面的技术视野。

---

## 下一章预告

在下一章《数据获取与采集》中，我们将正式进入预训练数据的处理流程。你将学习如何获取和解析 Common Crawl、The Pile 等开源数据集，如何使用 Trafilatura 构建高性能网页解析器，以及从 GitHub、ArXiv 抓取代码和论文的特种策略。

带着这个问题进入下一章：Common Crawl 每月新增 3-5PB 数据，你如何从中高效提取需要的内容？
