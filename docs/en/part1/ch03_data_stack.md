# Chapter 3: AI-Native Data Stack and Cost Governance

<div class="chapter-authors">Ke Wang</div>

## Abstract

This chapter discusses the AI-native data stack and cost-governance methods that support large-model data engineering. Unlike traditional data warehouses, which mainly serve analytical queries, an LLM data stack is designed to provide traceable, evaluable, versioned data assets to training and application pipelines at stable and low cost. The chapter first compares traditional data warehouses and large-model data stacks in their goals, workloads, and cost constraints. It then decomposes the data stack into five layers: ingestion and access, processing orchestration, storage and indexing, evaluation operations, and governance and security. It also discusses the appropriate boundaries of tools such as Spark, Ray Data, Iceberg, object storage, vector databases, DVC, and MLflow. Finally, the chapter provides a cost model, ROI decision framework, and three team-architecture patterns to help readers choose infrastructure that matches startup, mid-sized, and large-organization stages.

## Keywords

AI-native data stack; data infrastructure; Ray Data; Apache Iceberg; object storage; cost governance; DataOps

## Learning Objectives

- Distinguish traditional data warehouses from LLM data stacks in goals and workloads.
- Understand the five-layer architecture of an AI-native data stack and its key interfaces.
- Understand the cost components of data processing, storage, annotation, inference services, and training I/O.
- Choose a lightweight stack, platform stack, or multi-tenant data platform according to team scale.

## Opening Scenario: Risks When a Data Platform Lacks Governance

You have just joined a Series B large-model startup as the head of data. During your first week, you audit the company's data infrastructure and discover that corpus data is scattered across more than 50 engineers' local disks. File formats include `.txt`, `.json`, `.csv`, and `.parquet`, with no unified standard. Every time new data is processed, engineers manually write Python scripts and run them on a single machine; a 500 GB file taking three days is normal. Three months earlier, an incorrect file path overwrote a critical SFT dataset, and that dataset had no backup or version record. The CEO asks: the company plans to begin its first base-model pretraining run at the scale of tens of billions of parameters in one month. Can the data platform support it?

This scenario is not an exaggerated edge case. In many fast-moving large-model teams, this is common during the early stage. Algorithms and model architectures usually receive the most attention, while data infrastructure is treated as work that can be postponed, until critical data is lost, the training cluster is bottlenecked by data I/O, or a compliance audit finds unclear training-data provenance. Only then does the team rush to build infrastructure.

Reactive construction is expensive because the team must fill platform gaps while facing business pressure, training-window constraints, and compliance risk at the same time.

## 3.1 Why an AI-Native Data Stack Differs from a Traditional Data Warehouse

For many engineers, "data infrastructure" means the classic big-data stack of Hadoop, Hive, and Spark, plus relational databases, data warehouses, and BI reporting systems. Over the past decade, that stack has supported countless analytics and machine-learning workloads. It is mature and production-proven.

However, if this stack is moved unchanged into LLM data engineering, it fails to match almost every key dimension.

### 3.1.1 Different Goals: From Analytical Queries to Training Supply

The core goal of a traditional data warehouse is **analysis and insight**: gather business data and use SQL queries and BI tools to help decision makers answer "what happened" and "what is the current business state?" Under this goal, the core value of data lies in **query performance** and **consistency**: data must be accurate, well structured, and quickly joinable and aggregatable.

An LLM data stack serves a different core goal: **training a neural network model**. The final consumer is not a human but attention computation running on GPUs. Under this goal, the core value of data lies in **training throughput** and **signal-to-noise ratio**. GPU utilization must remain high, and tokens must be filtered to reduce the risk that the model learns incorrect knowledge or low-quality language patterns.

This difference in goals leads directly to a full divergence in technical choices.

### 3.1.2 Essential Workload Differences

The typical workload of a traditional data warehouse is **read-heavy, write-light OLAP queries**: many users concurrently execute `SELECT` queries, with occasional batch ETL writes. Data scale usually ranges from GB to TB, and peak performance aims for sub-second query latency.

The workload of an LLM data stack has a very different structure.

**Pretraining data processing** is the heaviest workload in the system. Tens of TB to PB of raw corpus data must pass through serial steps such as deduplication, filtering, format conversion, and tokenization. Each step is CPU-intensive and must run across thousands of cores to complete in a reasonable time window. These workloads require the compute layer to schedule and track hundreds of millions of file- or document-level tasks efficiently.

**I/O alignment with GPU training** is another key constraint. During model training, the DataLoader must feed data to GPUs at high frequency, often every step. Any I/O wait causes GPU idle time and increases compute cost. The storage layer must therefore provide read bandwidth that matches GPU training demand. This imposes strict requirements on object-storage access patterns, such as large sequential files versus random small files, and on network bandwidth such as InfiniBand or 100GbE Ethernet.

**Coexistence of online inference and real-time feedback** adds further complexity. When the model serves inference traffic, user interaction data, including conversations, satisfaction feedback, and valuable correction cases, must flow back into SFT or RLHF pipelines after cleaning and annotation. The stack must therefore support both offline batch processing, such as pretraining-corpus cleaning, and low-latency online writes, such as user-feedback ingestion. These workloads have different storage and compute requirements and need explicit isolation and coordination in system design.

### 3.1.3 Multidimensional Cost Constraints

In a traditional BI system, cost mainly means **storage cost** and **query compute cost**. In an LLM data system, four major cost categories create a multidimensional constraint.

First is **compute cost** for GPU/TPU training clusters. GPU cloud prices, rental prices, and self-built depreciation change rapidly with region, contract, supply and demand, model type, and utilization, so a manuscript should not hard-code unit prices. Engineering budgets should be calculated as "GPU hours x unit GPU-hour cost x effective-utilization loss." A pretraining or retraining run typically consumes a large number of GPU hours, which means that any training interruption or restart caused by data issues is expensive. This is the fundamental reason an LLM data stack must emphasize stability and rollback.

Second is **data-processing cost**. Large-scale corpus cleaning tasks run on CPU or GPU clusters, and their cost is determined jointly by data scale, parsing complexity, deduplication algorithms, quality-model inference, failed retries, and intermediate-data write amplification. Reducing per-run cost through refined scheduling, such as Spot instances, and algorithmic optimization, such as approximate MinHash deduplication (Broder 1997) instead of full pairwise comparison, is a core engineering concern.

Third is **annotation and human cost**. High-quality SFT, preference, and safety samples require annotators with professional backgrounds to write or review them manually. Unit cost depends on domain difficulty, sample length, QA ratio, rework rate, and regional labor prices. In high-risk domains such as medicine, law, and finance, expert review can easily become the largest single item in the data-engineering budget.

Fourth is **storage cost**. The prices of object storage, block storage, and archive storage vary with cloud provider, region, discount, access pattern, and retrieval fee. Budgets should distinguish hot data, intermediate artifacts, reproducible experiment snapshots, and long-term archive data, and should separately calculate capacity charges, request charges, cross-region traffic charges, and retrieval charges. Planning hot, warm, and cold tiers across the full data lifecycle, and avoiding active-tier charges for historical data that is no longer used, is an important cost-governance topic discussed further in Section 3.3.

---

## 3.2 Five Layers of the Data Stack

After clarifying the fundamental difference between an AI-native data stack and a traditional data warehouse, we can build the architecture systematically. A complete LLM data stack can be decomposed into five functional layers from bottom to top. Each layer has core responsibilities and key technical choices.

![Figure 3-1: Five-layer architecture of an AI-native data stack](../../images/part1/ai_data_stack_architecture.png)

*Figure 3-1: Five-layer architecture of an AI-native data stack. Source: original illustration from this book. The figure shows how ingestion and access, processing orchestration, storage and indexing, evaluation operations, and governance and security layers jointly move data from raw corpus to trainable datasets; Alt text: five-layer architecture of an AI-native data stack showing the data flow among ingestion and access, processing orchestration, storage and indexing, evaluation operations, and governance and security layers.*

### 3.2.1 Ingestion and Access Layer: Turning "Data Everywhere" into "Traceable Data"

The ingestion and access layer is the entry point of the stack. It brings data scattered across sources into platform management in a unified and controlled way. It may sound ordinary, but it is one of the easiest layers to neglect and one of the most expensive to fix later.

LLM data engineering has highly diverse sources: public web pages from Common Crawl snapshots or custom crawlers, GitHub code repositories through API datasets or incremental mirrors, academic papers from arXiv, PubMed, and Semantic Scholar, high-quality books from sources such as Books3 or licensed channels, private enterprise documents, and online user feedback. Each source has distinct crawling methods, file formats, and update frequency.

The core challenge of the ingestion layer is not "how to get data," which is mainly crawler and API development. It is **how to create a complete metadata record the moment data enters the platform**. When a batch lands, the platform must record key information: source, such as source URL or API endpoint; ingestion timestamp for freshness management; file format and version; data owner or license type for compliance audit; original file size and count; and the ingestion task ID for resume and traceability. Without this metadata, data becomes a black box after entering the platform. You know data exists, but you do not know where it came from, when it arrived, or which cleaning-pipeline version processed it. When a problem appears, root-cause analysis is impossible.

```python
# Example metadata registration: automatically write a record during ingestion
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class DataIngestionRecord:
    ingestion_id: str          # Unique task ID
    source_name: str           # Source name, such as "common_crawl_2024_04"
    source_url: str            # Original crawl URL or API endpoint
    ingestion_timestamp: str   # Ingestion timestamp, ISO 8601
    license_type: str          # License, such as "CC-BY-4.0", "Unknown", "Proprietary"
    file_format: str           # File format, such as "jsonl", "parquet", "warc"
    raw_file_count: int        # Number of raw files
    raw_size_bytes: int        # Raw data size in bytes
    s3_prefix: str             # Object-storage prefix
    crawl_agent: str           # Crawler script version

def register_ingestion(record: DataIngestionRecord, metadata_db_path: str):
    """Write an ingestion record to a metadata DB."""
    record_dict = asdict(record)
    record_dict["ingestion_timestamp"] = datetime.now().isoformat()
    # In production, write to PostgreSQL, DynamoDB, or another transactional store.
    with open(metadata_db_path, "a") as f:
        f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")
```

*Listing 3-1: Example data-ingestion metadata registration. In production, this should write to a transactional metadata database and include permission, audit, and idempotency controls.*

### 3.2.2 Processing Orchestration Layer: The Scheduler of the Data Factory

After data enters the platform, it must go through serial processing steps to become training-ready. These steps usually include HTML tag stripping and text extraction, language identification and filtering, rule-based noise filtering, such as removing URL-dense or advertisement-heavy spans, approximate deduplication with MinHash LSH, exact deduplication, quality scoring with PPL or classifiers, and final tokenization and serialization.

The processing orchestration layer solves the core engineering problem: how to run these steps efficiently across thousands of parallel compute nodes while preserving observability, recoverability, and reproducibility.

Two mainstream industrial choices are **Apache Spark** (Zaharia et al. 2016) and **Ray Data** (Moritz et al. 2018). They differ fundamentally in design philosophy and use cases.

*Table 3-1: Core feature comparison of Apache Spark vs. Ray Data. Source: compiled by the authors based on public documentation of open-source frameworks and LLM data-processing practice.*

| Dimension | Apache Spark | Ray Data |
| :--- | :--- | :--- |
| **Core language runtime** | Scala/Java core; Python through PySpark, with JVM-Python serialization overhead | Python-native, no JVM overhead, seamless with PyTorch and Hugging Face |
| **Data abstraction** | DataFrame, batch-processing mindset, emphasizes full-batch materialization | Dataset, pipeline mindset, streams data between operators |
| **GPU support** | Requires NVIDIA RAPIDS plugins such as cuDF/cuML; integration is more complex | Native GPU scheduling; operators can directly call CUDA kernels or deploy ML models |
| **Memory pattern** | Intermediate results between operators must materialize to disk or memory, increasing pressure | Pipeline execution allows upstream and downstream overlap, improving memory efficiency |
| **SQL and BI ecosystem** | Very mature Spark SQL with Hive metadata compatibility | Weaker SQL support and no mature SQL query interface |
| **Fault tolerance and stability** | More than a decade of PB-scale production validation | Younger system; fewer large-scale best practices than Spark |
| **Typical code style** | Functional chained calls such as `map`, `filter`, and `groupBy` | Declarative pipeline topology such as chained `ds.map_batches()` |

The right choice depends on team background and workload. If the team has traditional big-data experience and the pipeline contains much SQL logic or depends heavily on Hive/Iceberg, Spark is the more robust choice. If the team comes from AI/ML and frequently calls ML models during data processing, such as PPL classifiers or NER models for PII detection, Ray Data's Python-native runtime and GPU scheduling advantages are clear. Many large teams eventually use a hybrid: Spark handles massive coarse filtering, such as language identification and rule deduplication, while Ray Data handles fine-grained steps that require ML inference, such as quality scoring and benchmark-contamination detection.

The following is a typical Ray Data cleaning pipeline.

```python
import ray
from ray.data import read_parquet

ray.init()

def remove_html_noise(batch):
    """Remove HTML tags and compute a simple noise score."""
    import re
    texts = batch["text"].tolist()
    cleaned = [re.sub(r"<[^>]+>", "", t) for t in texts]
    noise_scores = [
        (len(re.findall(r"<[^>]+>", t)) * 10) / max(len(t), 1)
        for t in texts
    ]
    batch["text"] = cleaned
    batch["noise_score"] = noise_scores
    return batch

def filter_quality(batch):
    """Filter low-quality samples."""
    mask = (batch["noise_score"] < 0.1) & (batch["text"].str.len() > 100)
    return batch[mask]

# Build a cleaning pipeline.
ds = (
    read_parquet("s3://my-bucket/raw/common_crawl/")
    .map_batches(remove_html_noise, batch_format="pandas")
    .map_batches(filter_quality, batch_format="pandas")
)

# Write output to object storage.
ds.write_parquet("s3://my-bucket/processed/cc_cleaned/")
```

*Listing 3-2: Ray Data cleaning pipeline example. Production systems should add retry logic, metric reporting, data versions, and output validation.*

### 3.2.3 Storage and Indexing Layer: Different Strategies for Three Data Types

An LLM data stack must manage three very different kinds of data. Each has distinct storage requirements, so the storage and indexing layer needs differentiated technical choices.

**Text corpora and structured annotation data** are the largest category. They include cleaned pretraining corpora in Parquet format, SFT samples in JSONL format, and preference-comparison datasets. Their read/write pattern is typical batch processing: large sequential writes after processing and large sequential reads when training DataLoaders scan the data. Object storage such as AWS S3 or MinIO is usually the recommended choice. On top of it, a lakehouse table format is needed to provide version management and ACID transactions.

The three mainstream lakehouse formats each have their own appropriate scenarios.

*Table 3-2: Lakehouse table format selection comparison: Apache Iceberg vs. Delta Lake vs. Apache Hudi. Source: compiled by the authors based on public documentation of open-source projects and lakehouse architecture practice.*

| Feature | Apache Iceberg | Delta Lake | Apache Hudi |
| :--- | :--- | :--- | :--- |
| **Core maintainer** | Netflix to Apache Foundation | Databricks, core commercial open source | Uber to Apache Foundation |
| **Engine compatibility** | Spark / Flink / Trino / DuckDB / StarRocks; broadest | Mainly Spark; others need adaptation layers | Spark / Flink / Presto |
| **ACID transactions** | Full support | Full support | Supported, strongest for upsert/delete |
| **Time-travel queries** | Full support by version or timestamp | Full support | Supported, lighter-weight |
| **Schema evolution** | Supports add, rename, and delete columns | Supported | Supported |
| **Recommended scenarios** | Multi-engine use and vendor-neutral teams | Teams deeply using the Databricks ecosystem | High-frequency upsert needs, such as real-time knowledge-base updates |

For LLM data-engineering scenarios that require multi-engine collaboration and object-storage scalability, **Apache Iceberg** (Apache Software Foundation 2024) **+ S3** is a common priority candidate because it is engine-neutral. It allows engineering teams to use Spark for massive cleaning, DuckDB for lightweight exploration, and other engines without migrating data or being locked into one vendor.

**Vector data, or embeddings**, are the second storage need, mainly serving RAG. A vector database converts massive text chunks into high-dimensional dense vectors, builds indexes, and supports efficient approximate nearest-neighbor retrieval (Malkov and Yashunin 2020). Mainstream vector databases include Milvus, which is open source and suited for large distributed deployments; Qdrant, implemented in Rust and friendly for lightweight high-performance deployment; and Weaviate, which has built-in multimodal vector support and friendly schema management. The key selection factors are vector scale, such as below one million versus hundreds of millions; whether hybrid search is needed, combining dense vectors and BM25 sparse retrieval (Robertson and Zaragoza 2009); and whether the operations team can manage distributed systems.

**Model checkpoints and experiment artifacts** are the third category. They include model weight files saved during training, often hundreds of GB, TensorBoard or W&B logs, tokenizer configuration, and related artifacts. These data are large and have uneven access frequency: frequent writes during training, mostly read-only afterward. Object storage is suitable as primary storage, with DVC (Data Version Control) (DVC 2024) or MLflow Artifacts for version tracking.

```bash
# Use DVC to track a dataset version.
dvc init
dvc add datasets/sft/v2.3/  # Track the dataset directory with DVC.
git add datasets/sft/v2.3.dvc .gitignore
git commit -m "feat: add SFT dataset v2.3 (84k samples, legal domain)"
dvc push  # Push actual data to S3 remote storage.
# Other engineers can use dvc pull to fetch the exact same data.
```

*Listing 3-3: Example DVC commands for dataset version tracking. Production environments should combine this with object-storage permissions, data-hash validation, and release approval.*

### 3.2.4 Evaluation Operations Layer: Making Data Quality Visible

The evaluation operations layer provides observability for the entire data platform. It lets teams see pipeline status, data-quality trends, and experiment records in real time. It is the dashboard that keeps the data flywheel turning.

A mature evaluation operations layer should cover at least four dimensions. A **data quality dashboard** displays the core quality metrics of each batch, such as deduplication rate, PPL distribution, noise ratio, and benchmark contamination rate, and compares them with historical baselines. If any metric deviates beyond a threshold, an alert is triggered immediately. **Pipeline runtime monitoring** tracks task completion rate, processing speed in documents per second, retry count, and resource consumption across CPU, memory, and network, ensuring that there are no silent failures, such as a task quietly skipping a large amount of data without raising an alert. **Data sampling tools** provide random sampling and human review interfaces, allowing data engineers to periodically inspect outputs from each node and find systematic quality problems that automated tests miss. **Experiment tracking** binds dataset versions to model-training experiments, ensuring every training run can accurately trace which dataset version and cleaning configuration it used. This is the basis for reproducibility.

### 3.2.5 Governance and Security Layer: Data Needs Compliance Records

As privacy regulations such as GDPR and China's Personal Information Protection Law become stricter, and as LLM training-data copyright disputes lead to lawsuits, including media organizations suing OpenAI and Getty Images suing Stability AI, governance and security have moved from optional future planning to a current requirement.

Core functions include **copyright and license management**, which records the license type of each data batch, such as CC-BY, CC-BY-SA, CC-BY-NC, commercial license, or unknown source, and filters high-risk sources at ingestion through technical methods such as `robots.txt` compliance checks and license classifiers; **PII detection and de-identification**, which uses NER models or regular expressions to identify names, phone numbers, email addresses, ID numbers, and other sensitive information, then de-identifies them before storage; **permission and access control**, which applies strict RBAC policies to different data sensitivity levels, such as user-conversation feedback data, so only authorized engineers can access or process them; and **data lineage tracking**, which records the full transformation chain from original source to final training dataset and provides complete provenance evidence during compliance audits.

---

## 3.3 Cost Model and Budget Governance

The cost structure of large-model data engineering is much more complex than that of traditional data engineering. Many teams include only GPU rental in the project budget and put data processing, annotation, and storage into miscellaneous expenses. Midway through a project, data-side cost often exceeds the original budget.

### 3.3.1 Five Cost Dimensions

LLM data-engineering cost can be decomposed into five main dimensions. Understanding the cost drivers of each dimension is the foundation of a reasonable budget.

**Data acquisition cost** is the first. It includes crawler server rental and bandwidth, purchased commercial corpora such as professional databases in a domain, and API call fees for structured web content from services such as Diffbot or Apify. Cloud Spot instances are usually cheaper than on-demand instances, but the discount varies with region, instance type, and supply-demand conditions; budgets should explicitly record the price source and effective date.

**Data processing cost** is the second and is the easiest to underestimate. The cost of a complete corpus-cleaning pipeline, including parsing, filtering, deduplication, and quality scoring, depends on processing complexity, data scale, regional pricing, retry count, and intermediate-result write amplification. If ML inference is used during processing, such as running a quality classifier on GPUs, GPU instance cost is added.

**Annotation cost** is the third and often the largest single item in the data budget. High-quality SFT samples in professional domains such as medicine, law, and finance require experts with relevant backgrounds. Actual cost must be recalculated based on annotation guidelines, sample length, QA ratio, rework rate, regional labor prices, and confidentiality requirements.

**Storage cost** is the fourth. Actual pricing should be based on the cloud provider, region, discount, and access pattern. Without hot/cold tier management, intermediate artifacts that are no longer used continue occupying active storage and accumulate long-term cost.

**Inference-service cost** is the fifth. It is incurred when strong models are called during data processing for quality evaluation, synthetic data generation, or automated annotation. Budgets should jointly estimate input tokens, output tokens, image/audio/video unit prices, retry rate, cache hit rate, and concurrency limits. When massive numbers of samples are scored for quality, API fees alone may become a major cost, so sampling, cascaded evaluation, and caching strategies must be planned.

### 3.3.2 Cost Accounting Before, During, and After Training

A more practical cost framework decomposes the training lifecycle into three stages.

In the **pre-training stage before model training**, data acquisition and processing dominate cost. The key budget decisions are target data scale, usually measured in tokens, and whether to build a processing cluster or use cloud Spot instances. For projects below roughly 10 billion tokens, cloud Spot instances, such as Ray on AWS Spot, are often the most cost-effective. For projects above hundreds of billions of tokens, dedicated CPU clusters, whether owned or leased, can reduce processing cost per token.

During the **training stage**, compute cost dominates, but data I/O quality directly affects GPU utilization and therefore actual compute cost. If the training process waits for the DataLoader or remote storage for long periods, the same effective compute requires more wall-clock time. Data I/O optimization therefore often has high ROI.

In the **post-training stage**, model evaluation, data-version archiving, and knowledge-base maintenance, such as RAG updates, become continuous costs. The focus is a clear data lifecycle strategy: datasets that enter official releases can move to infrequent-access storage; temporary intermediate datasets can have 30-day auto-deletion policies; only final datasets with clear labels and descriptions should be retained. Prices and retrieval fees for infrequent-access storage change quickly and should be checked against cloud-provider announcements after 2026-06.

### 3.3.3 ROI Decision Framework

When choosing among several data-quality improvement plans, engineers need a quantified ROI framework to avoid intuition-driven spending such as "this feels useful":

$$\text{Data Engineering ROI} = \frac{\Delta\text{Model Performance} \times \text{Model Business Value}}{\text{Data Processing Cost} + \text{Annotation Cost} + \text{Storage Cost}}$$

For example, if an investment in legal-domain SFT data brings a statistically significant improvement in user satisfaction and that improvement can be mapped to retention, conversion, or reduced manual review cost, the team can compare the incremental return against data acquisition, annotation, training, and launch costs to estimate the payback period. This type of quantitative thinking is essential for avoiding a situation where data engineering "does a lot" without contributing to model capability.

Connecting planning, monitoring, evaluation, optimization, and postmortem into a continuous loop forms the **cost-governance loop** of a large-model team, as shown below.

![Figure 3-2: Training-data cost-governance loop](../../images/part1/cost_governance_loop.png)

*Figure 3-2: Training-data cost-governance loop. Source: original illustration from this book. The figure shows a cross-version iteration cycle that starts from budget planning, passes through cost monitoring, ROI evaluation, and optimization decisions, and returns to budget review; Alt text: training-data cost-governance loop showing the cycle of budget planning, cost monitoring, ROI evaluation, optimization decisions, and budget review.*

---

## 3.4 Three Team Architecture Patterns

No data-stack architecture fits every team size and stage. By team scale and business maturity, LLM data-engineering teams can be roughly divided into three categories, each with a recommended architecture pattern.

### 3.4.1 Startup Lightweight Stack: Validate Quickly and Avoid Over-Design

**Applicable scenario**: a 1-5 person data team in proof-of-concept or angel-to-Pre-A stage. The main task is to validate data-recipe hypotheses quickly and prove that the model has core capabilities.

At this stage, the biggest engineering trap is **over-design**. Spending three months building an "industrial-grade" distributed data platform may leave the team two months behind in the right direction by the time the platform is ready. The correct startup strategy is to validate core hypotheses at minimum cost and introduce distributed capabilities only when real data volume reaches single-machine limits.

The recommended lightweight stack is: **S3 / MinIO + Parquet** for storage, without Iceberg at first and with manual version directories; **DuckDB** for compute, suitable for single-machine or small-scale data exploration with friendly SQL syntax, plus Python pandas or polars; **DVC** for version control; and **Prefect** or **Dagster** for pipeline orchestration. The engineering setup time for this stack depends on team experience, permission workflows, and data-source complexity.

DuckDB deserves special emphasis because it is often underestimated in startups. It can directly read and write Parquet files on S3 without downloading them locally, and it supports standard SQL syntax so engineers unfamiliar with PySpark can start quickly. Actual time and cost depend on region, instance discount, data format, compression method, and query complexity; production use should be preceded by pressure testing on representative target-data samples.

### 3.4.2 Platform Construction for Mid-Sized Teams

**Applicable scenario**: a 6-20 person data/algorithm team, usually Series A to B, that has validated the core direction, started pretraining at tens-of-billions-of-tokens scale, and maintains several vertical-domain SFT pipelines.

The core challenge at this stage is **standardization and reuse** of data pipelines. Cleaning scripts written by different engineers lack unified interfaces and cannot be reused; data-version management is chaotic; and it is hard to trace which data version a given experiment used. As data volume grows, single-machine processing no longer meets timeliness requirements.

The recommended mid-sized-team stack is: upgrade compute to **Ray Data**, which supports multi-machine distributed execution and is Python-native, or **Spark on EMR** if the team has Spark experience; introduce **Apache Iceberg + S3** for data version management; use **Weights & Biases** or **MLflow** for experiment tracking; and build a basic Grafana dashboard for data-quality monitoring based on the scorecard system described in Section 3.2.4. Data-processing code should be modularized into reusable operators, forming an operator library.

### 3.4.3 Multi-Tenant Collaboration for Large Organizations

**Applicable scenario**: a data-engineering team of more than 20 people, usually after Series B or inside a technology company with large AI businesses, supporting multiple large-model projects at the same time, such as a general foundation model, vertical industry customization, and multimodal initiatives. The data infrastructure must share resources across projects while preserving data isolation and permission control.

The core challenge is **multi-tenancy**. How can different LLM projects share the underlying processing cluster while ensuring that dirty data from Project A does not pollute Project B's pipeline? How should resources and priorities be scheduled across dozens of parallel processing jobs?

The recommended large-team pattern centers on a **unified data platform**. At the bottom, Kubernetes manages compute resources, including CPU and GPU nodes. Ray on Kubernetes or Spark on Kubernetes provides scheduling. Data isolation is implemented through S3 IAM policies at bucket level, with independent buckets for projects or shared buckets with strict prefix isolation. Metadata management uses data-catalog tools such as Apache Atlas or AWS Glue Data Catalog to manage company-wide data lineage. The platform team maintains a standardized data operator library, and business lines build their cleaning pipelines by calling its interfaces, ensuring unified quality-check logic.

An important lesson is that large-team data platforms should be built in **three stages**, not all at once. Stage 1, taking one to three months, should connect the core path: storage ingestion, basic cleaning operators, and version management, so data can flow in a controlled way. Stage 2, taking three to six months, should build observability: quality dashboards, experiment tracking, and alerting systems, turning the platform into a transparent system. Stage 3, after six months, should add more complex multi-tenant isolation, cross-project lineage insight, and resource quota management. Entering Stage 3 too early can make platform complexity exceed real needs and reduce core data-flow efficiency.

*Table 3-3: Quick selection matrix for data stacks across three team types. Source: compiled by the authors; setup cycles are empirical planning ranges, and actual cycles depend on team experience, permission workflows, and data-source complexity.*

| Team size | Recommended storage | Recommended compute | Recommended orchestration | Recommended version management | Estimated build cycle |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1-5 people, startup | S3/MinIO + Parquet | DuckDB / Polars | Prefect / Dagster | DVC | 1-2 weeks |
| 6-20 people, mid-sized | S3 + Apache Iceberg | Ray Data or Spark | Airflow / Dagster | DVC + MLflow | 1-2 months |
| 20+ people, large | S3 + Iceberg with multi-bucket isolation | Ray on K8s / Spark on K8s | Airflow + Argo | DVC + Glue Catalog + Atlas | 3-6 months |

---

## 3.5 Platform Interfaces and Connections to Later Chapters

The AI-native data stack built in this chapter is not an isolated menu of technical options. It is the **shared infrastructure foundation** for the engineering content in the rest of the book. Understanding this helps readers map later technical details to the corresponding layer in the architecture blueprint.

In **Part II: Text Pretraining Data Engineering**, large-scale MinHash deduplication in Chapter 5, KenLM perplexity filtering in Chapter 6, and DataLoader I/O optimization in Chapter 7 all run on the **processing orchestration layer** described in Section 3.2.2, using Ray Data or Spark clusters, and depend on the **metadata ingestion layer** in Section 3.2.1 for lineage tracking.

In **Part III: Multimodal Data Engineering**, image-text cleaning and video slicing require GPU scheduling, which is supported by the GPU-native scheduling capability of Ray Data discussed in Section 3.2.2.

In **Part VII: RAG Application Data Engineering**, real-time knowledge-base update pipelines rely on the vector-database choices in Section 3.2.3, such as Milvus or Qdrant, to host vector indexes, and on the compliance-audit capability in Section 3.2.5 to ensure documents entering the knowledge base do not carry copyright risk.

**Part VIII: DataOps Platform Development** is an expanded version of this chapter: it will build on the five-layer architecture to discuss end-to-end observability for data pipelines, automated governance of data assets, and deep integration of quality scorecards with CI/CD pipelines, upgrading the platform from a manual workshop to an intelligent data factory.

The core principle for capability boundaries is: **capabilities needed by multiple projects or multiple data stages should be platformized**, such as deduplication operator libraries, quality-scorecard frameworks, and data-version management; **capabilities highly customized to one business scenario should remain project-specific**, such as entity-recognition rules for one vertical domain or parsing logic for one source. Bigger platform boundaries are not automatically better. Over-abstraction reduces iteration speed and forces project teams to adapt to platform interfaces instead of letting the platform serve real needs.

---

**Chapter Summary**

This chapter established a full architecture blueprint for an AI-native data stack. It first analyzed why traditional warehouse technology stacks designed for BI cannot be directly moved into LLM data engineering, using three dimensions: goal differences, workload characteristics, and cost constraints. On that basis, it decomposed the stack into five functional layers: ingestion and access, processing orchestration, storage and indexing, evaluation operations, and governance and security. Each layer included mainstream industrial choices and concrete comparison criteria. The cost-model section revealed five cost dimensions, stage-based accounting, and a quantified ROI decision framework. Finally, the chapter provided differentiated architecture patterns for startup, mid-sized, and large teams, helping readers avoid introducing platform complexity beyond team capability and business need during early stages.

With this infrastructure blueprint in place, the next chapter begins Part II: Text Pretraining Data Engineering, where we will discuss how to build trainable, traceable, and evaluable pretraining datasets from large-scale public corpora on top of this stack.

## Chapter Summary

This chapter organized the core issues, process flows, and acceptance criteria for AI-native data stacks and cost governance in large-model data engineering. It places concepts, data objects, quality signals, and engineering delivery in the same narrative so that readers can judge which links must be explicitly recorded and which results require sampling, evaluation, or audit.

The methods in this chapter should be applied according to data source, business objective, model capability, cost budget, and compliance requirements. In scenarios involving sensitive information, cross-system calls, automated decisions, or public release, teams should retain human review, version freezing, access control, and rollback mechanisms rather than extrapolating illustrative workflows into production promises.

Within the structure of the book, this chapter belongs to the foundational-framework layer and connects previous concepts to text and multimodal data processing. Readers can combine the framework with the figures, references, and appendix checklists to turn the methods into reproducible, inspectable, and deliverable engineering workflows.

## References

Zaharia M, Xin R S, Wendell P, Das T, Armbrust M, Dave A, Meng X, Rosen J, Venkataraman S, Franklin M J, Ghodsi A, Gonzalez J, Shenker S, Stoica I (2016) Apache Spark: A Unified Engine for Big Data Processing. Communications of the ACM 59(11):56-65.

Moritz P, Nishihara R, Wang S, Tumanov A, Liaw R, Liang E, Elibol M, Yang Z, Paul W, Jordan M I, Stoica I (2018) Ray: A Distributed Framework for Emerging AI Applications. In: Proceedings of the 13th USENIX Symposium on Operating Systems Design and Implementation, pp 561-577.

Broder A Z (1997) On the Resemblance and Containment of Documents. In: Proceedings of the Compression and Complexity of Sequences, pp 21-29.

Heafield K (2011) KenLM: Faster and Smaller Language Model Queries. In: Proceedings of the Sixth Workshop on Statistical Machine Translation, pp 187-197.

Robertson S, Zaragoza H (2009) The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval 3(4):333-389.

Malkov Y A, Yashunin D A (2020) Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs (HNSW). IEEE Transactions on Pattern Analysis and Machine Intelligence 42(4):824-836.

Apache Software Foundation (2024) Apache Iceberg: Table Specification and Documentation. <https://iceberg.apache.org/spec/> (accessed 2024-11).

DVC Team and Contributors (2024) DVC: Data Version Control - Git for Data & Models. Documentation: <https://dvc.org/doc>. Source repository: <https://github.com/iterative/dvc>.

Penedo G, Kydlíček H, Ben Allal L, Lozhkov A, Mitchell M, Raffel C, von Werra L, Wolf T (2024) The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. arXiv preprint arXiv:2406.17557.

Together Computer (2023) RedPajama: An Open Dataset for Training Large Language Models. GitHub repository. <https://github.com/togethercomputer/RedPajama-Data>.

Sculley D, Holt G, Golovin D, Davydov E, Phillips T, Ebner D, Chaudhary V, Young M, Crespo J F, Dennison D (2015) Hidden Technical Debt in Machine Learning Systems. Advances in Neural Information Processing Systems 28:2503-2511.

Longpre S, Mahari R, Chen A, Obeng-Marnu N, Sileo D, Brannon W, Muennighoff N, Khazam N, Kabbara J, Perisetla K, Wu X, Shippole E, Bollacker K, Wu T, Villa L, Pentland S, Hooker S (2024) The Data Provenance Initiative: A Large Scale Audit of Dataset Licensing & Attribution in AI. Nature Machine Intelligence 6(8):975-987.

Chen D, Huang Y, Pan X, Jiang N, Wang H, Zhang Y, Ge C, Chen Y, Zhang W, Ma Z, Huang J, Lin W, Li Y, Ding B, Zhou J (2025) Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models. arXiv preprint arXiv:2501.14755.
