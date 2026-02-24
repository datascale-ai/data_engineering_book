# Chapter 2: AI-Native Data Stack (Vector DB, Object Storage, Ray/Spark Distributed Computing)

---

## Chapter Summary

A craftsman must sharpen his tools before he can do his work well. Before processing TB-level or even PB-level LLM training data, choosing the right infrastructure is the first step that determines project success or failure. This chapter systematically introduces AI-native data stack technology selection from five dimensions—**storage, compute, vector databases, format, and version control**. We pay special attention to distributed data processing frameworks (Ray Data, Apache Spark, Dask) in large-scale token processing, as well as GPU training I/O bottleneck optimization strategies, helping readers build an efficient, scalable, and reproducible data processing platform.

---

## Scenario Introduction

You just joined an AI startup, responsible for building the LLM pre-training data processing platform. The team's situation is concerning: data is scattered across local disks on 50 machines, in various formats including `.txt`, `.json`, `.csv`, `.parquet`, and more. Every time data is processed, Python scripts must be manually written and run on a single machine for three days to complete. Last week someone accidentally overwrote a critical dataset, and there was no backup or version record. Your boss asks: "We're starting training in a month—can the data platform be ready?"

Your first decision: Use the team-familiar Spark, or switch to the "AI-native" Ray? Build a self-hosted MinIO cluster, or go straight to cloud with S3? There are no "standard answers" to these questions, but there are clear decision frameworks. This chapter provides that framework.

---

## 2.1 Modern Data Stack (MDS)

### 2.1.1 What Is the Modern Data Stack?

The "Modern Data Stack" (MDS) is a hot concept in data engineering in recent years, referring to a cloud-native, modular, decoupled combination of data infrastructure. Compared with traditional integrated data platforms, the core philosophy of the modern data stack is to split storage, compute, orchestration, and other functions into independent components, each of which can be independently replaced and scaled according to needs.

![Figure 2-1: Modern Data Stack Architecture](../../images/part1/图2_1_现代数据栈架构.png)

*Figure 2-1: Modern Data Stack Architecture — 5-layer decoupled architecture from storage to application layer, each layer independently replaceable*

Traditional data platforms are often deployed in local data centers with integrated systems, storage tightly coupled with compute. Taking the Hadoop ecosystem as an example, HDFS and MapReduce coupling makes replacing any component very difficult. Data formats are often proprietary, leading to serious vendor lock-in. Scaling is mainly vertical—improving performance by purchasing more powerful single machines—with high upfront costs.

| Feature | Traditional Approach | Modern Data Stack |
|------|----------|-----------|
| **Deployment Mode** | Local data center, integrated system | Cloud-native, elastic scaling on demand |
| **Component Coupling** | Storage-compute bound (e.g., HDFS + MapReduce) | Storage-compute separation, each layer independently replaceable |
| **Data Format** | Proprietary format, vendor lock-in | Open format (Parquet, ORC) |
| **Scalability** | Mainly vertical scaling | Horizontal scaling, nearly unlimited |
| **Cost Model** | Fixed investment, high upfront cost | Pay-per-use, elastic cost |

The emergence of the modern data stack changed this landscape. Cloud-native deployment allows elastic scaling on demand; complete storage-compute separation enables each layer to evolve independently. Open data formats (e.g., Parquet, ORC) eliminate vendor lock-in risk. Horizontal scaling enables the system to handle nearly unlimited data volume, while pay-per-use cost model greatly lowers project startup barriers.

### 2.1.2 Storage Layer: Object Storage and Data Lake

Object storage is the de facto standard foundation for modern data platforms. Whether AWS S3, Google Cloud Storage, Azure Blob, or open-source MinIO, their core philosophy is the same: flat namespace with no true directory hierarchy, only `bucket/key` binary structure; theoretically unlimited storage; extremely high data durability (S3 claims 11 nines, i.e., 99.999999999%); billed by actual usage with no large upfront investment.

When selecting among options, deployment mode, compatibility, cost, and other factors must be considered. AWS S3 is the benchmark for public cloud hosting with the most mature ecosystem, suitable for most production environments. MinIO is an S3-compatible open-source alternative, suitable for private deployment scenarios with data compliance requirements or for development/test environments. Google Cloud Storage and Azure Blob are respectively suitable for users already deeply in GCP or Azure ecosystems.

| Feature | AWS S3 | MinIO | Google GCS | Azure Blob |
|------|--------|-------|------------|------------|
| **Deployment Mode** | Public cloud hosted | Self-hosted/private cloud | Public cloud hosted | Public cloud hosted |
| **S3 Compatibility** | Native | 100% compatible | Requires adapter layer | Requires adapter layer |
| **Cold/Hot Tiering** | Glacier | Tiering | Nearline/Coldline | Cool/Archive |
| **Lowest Cost** | $0.023/GB/month | Hardware cost | $0.020/GB/month | $0.018/GB/month |
| **Typical Use Case** | Production default | Private deployment/dev-test | GCP ecosystem users | Azure ecosystem users |

Object storage solves the "storage" problem but lacks transactional and metadata management capabilities. Operating directly on Parquet files in S3 encounters many difficulties: no ACID transactions, concurrent writes may corrupt data; no efficient querying, must scan all file metadata each time; no time travel, once data is overwritten it cannot be rolled back to historical versions.

Data lake table formats emerged precisely to solve these problems. They add a metadata management layer on top of object storage, providing data warehouse-level capabilities. Apache Iceberg, Apache Hudi, and Delta Lake are currently the three most mainstream data lake formats.

![Figure 2-2: Data Lakehouse Architecture](../../images/part1/图2_2_数据湖仓架构.png)

*Figure 2-2: Data Lakehouse Architecture — Table format layer provides ACID transactions, time travel, schema evolution, and other capabilities*

Apache Iceberg was developed by Netflix and contributed to the Apache Foundation; its biggest advantage is engine neutrality—it works well with Spark, Flink, Trino, Dremio, DuckDB, and other compute engines. For LLM data engineering scenarios, Iceberg is the most recommended choice. Apache Hudi was developed by Uber, with strengths in streaming-batch unification and real-time updates; if there are substantial real-time update needs (e.g., continuous RAG knowledge base updates), Hudi can be considered. Delta Lake was developed by Databricks with the tightest Spark integration; if already deeply in the Databricks ecosystem, choosing Delta Lake provides the best experience.

| Feature | Apache Iceberg | Apache Hudi | Delta Lake |
|------|----------------|-------------|------------|
| **Backing Vendor** | Netflix → Apache | Uber → Apache | Databricks |
| **Open Source Degree** | Fully open source | Fully open source | Core open source, some features commercial |
| **Engine Compatibility** | Spark, Flink, Trino, DuckDB | Spark, Flink, Presto | Primarily Spark |
| **Typical Use Case** | Multi-engine mixed use, vendor neutral | Stream-batch unification, real-time updates | Databricks ecosystem users |

When making actual selections, the following decision tree can be used: First determine whether data scale exceeds 100TB. If yes, further consider whether ACID transactions and time travel are needed—if yes, and there are multi-engine access needs, recommend Iceberg + S3; if only using Spark, Delta Lake or Hudi can be chosen. If ACID capability is not needed, use S3/MinIO + Parquet directly. For scenarios with data volume below 100TB, if team size is small (fewer than 5 people), local disk + Parquet is sufficient for prototype validation; migrate to S3 + Parquet as scale grows.

![Figure 2-3: Storage Layer Selection Decision Tree](../../images/part1/图2_3_存储层选型决策树.png)

*Figure 2-3: Storage Layer Selection Decision Tree — Choose best solution based on data scale, ACID needs, multi-engine access, and other factors*

---

### 2.1.3 Compute Layer: Spark vs Ray Data

This is the most common "either-or" dilemma in LLM data engineering. Both are distributed compute frameworks, but with distinctly different design philosophies and use cases. Understanding their differences is crucial for making the right technical selection.

Apache Spark was born in Berkeley AMPLab in 2009; after fifteen years of development, it has become the "Swiss Army knife" of big data processing. Spark's core strength is its maturity and stability—production-validated at PB scale, with extremely rich documentation and community resources. Spark SQL enables data analysts to also write distributed processing logic, lowering the barrier to entry. Structured Streaming supports real-time data processing, achieving stream-batch unification. However, Spark also has clear disadvantages: core is JVM implementation, Python UDF requires cross JVM-Python serialization with significant performance overhead; weaker integration support for GPU and PyTorch/TensorFlow, not "AI-native"; operators must materialize intermediate results between them, creating significant memory pressure.

Ray was born in Berkeley RISELab in 2017, initially a distributed reinforcement learning framework, later evolving into general AI application infrastructure. Ray Data is its data processing module, designed specifically for AI workloads. Ray Data's core strength is Python-native—no JVM overhead, seamless integration with PyTorch, HuggingFace, and other AI ecosystems. It natively supports pipeline execution with high memory efficiency; built-in GPU scheduling for easy CUDA operator invocation; Actor model suits stateful complex processing, such as inference tasks requiring loaded ML models. However, Ray is relatively young with less rich documentation and best practices than Spark; weak SQL support, no mature SQL interface like Spark SQL; integration with traditional big data ecosystem (Hive, Iceberg) requires additional work.

| Dimension | Apache Spark | Ray Data |
|------|--------------|----------|
| **Language** | Scala/Java core, Python API | Python native |
| **Runtime** | JVM | Python (Arrow-based) |
| **Data Abstraction** | DataFrame (batch thinking) | Dataset (stream thinking) |
| **GPU Support** | Requires RAPIDS plugin | Native support |
| **PyTorch Integration** | Cumbersome | First-class citizen |
| **SQL Support** | Very mature | Limited |
| **Typical Users** | Traditional big data teams | AI/ML teams |

To more intuitively understand their differences, consider a concrete code comparison. Assume the task: read Parquet files, filter short text, compute text length, save results.

Spark implementation:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import length, col

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("TextFilter") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# Read → Filter → Compute → Save
df = spark.read.parquet("s3://my-bucket/raw_data/")
df_filtered = df.filter(length(col("text")) > 100) \
                .withColumn("text_length", length(col("text")))
df_filtered.write.parquet("s3://my-bucket/processed_data/")

spark.stop()
```

Ray Data implementation:

```python
import ray

# Initialize Ray (auto-detect cluster resources)
ray.init()

# Define processing function
def filter_and_compute(batch):
    mask = batch["text"].str.len() > 100
    filtered = batch[mask].copy()
    filtered["text_length"] = filtered["text"].str.len()
    return filtered

# Read → Process → Save (pipeline execution)
ds = ray.data.read_parquet("s3://my-bucket/raw_data/")
ds_processed = ds.map_batches(filter_and_compute, batch_format="pandas")
ds_processed.write.parquet("s3://my-bucket/processed_data/")
```

As can be seen, Spark requires explicit Executor memory configuration and uses declarative DataFrame API; Ray auto-discovers resources and uses functional `map_batches` interface. Custom logic in Spark requires defining UDF with serialization overhead; Ray uses ordinary Python functions directly, more natural.

![Figure 2-4: Compute Framework Selection Decision Tree](../../images/part1/图2_4_计算框架选型决策树.png)

*Figure 2-4: Compute Framework Selection Decision Tree — Spark suits SQL/ETL scenarios, Ray suits GPU/ML scenarios*

When making actual decisions, the following logic can be used: If data processing requires GPU (e.g., calling BERT model for quality scoring), Ray Data is the more natural choice. If there are substantial SQL and BI query needs, Spark’s SQL ecosystem is more mature. If there is already extensive Spark infrastructure and code assets, migration cost must be evaluated—high cost then keep Spark, low cost consider gradually introducing Ray. If a new project, team background is decisive: traditional big data teams find Spark easier to adopt, AI/ML teams find Ray smoother.

Worth mentioning: in actual large projects, Spark and Ray often coexist rather than being mutually exclusive. A common hybrid strategy: Spark handles interaction with data lake/data warehouse, including reading/writing Iceberg/Hive tables, executing SQL analysis and other ETL tasks; Ray Data handles ML-intensive processing, such as invoking large models for inference, using GPU for batch processing. The two exchange data through shared object storage (Parquet files on S3), each performing its role, complementing each other.

#### Dask: A Python-Native Third Option

Besides Spark and Ray, **Dask** is another noteworthy distributed computing framework, especially suited for teams with existing Pandas/NumPy code. Dask’s core principle is "parallelize the PyData ecosystem"—its API is nearly identical to Pandas/NumPy, allowing single-machine code to scale to clusters with minimal changes.

**Dask’s core strengths**:

- **Zero learning cost**: `dask.dataframe` API is nearly identical to Pandas; teams don’t need to learn new syntax.
- **Flexible scheduling**: Can run on a single machine with multiple cores (replacing multiprocessing) or scale to distributed clusters.
- **Integration with scientific computing ecosystem**: Good integration with scikit-learn, XGBoost, and other ML libraries.
- **Low deployment barrier**: No JVM required (unlike Spark), no complex cluster management needed (simpler than Ray).

**Dask’s weaknesses**:

- **Large-scale performance inferior to Spark**: At PB-level data processing, Dask’s optimizer and shuffle performance are less mature than Spark.
- **No native GPU support**: Unlike Ray’s native GPU scheduling (requires Dask-CUDA plugin).
- **Smaller community**: Not as active as Spark and Ray communities.

```python
import dask.dataframe as dd
import dask

# Dask vs Pandas: nearly identical API
def process_with_dask(input_path: str, output_path: str):
    """Distributed text processing with Dask"""
    # Read (auto-partitioned, lazy execution)
    ddf = dd.read_parquet(input_path)
    
    # Filter short text (API identical to Pandas)
    ddf_filtered = ddf[ddf['text'].str.len() > 100]
    
    # Add computed column
    ddf_filtered = ddf_filtered.assign(
        text_length=ddf_filtered['text'].str.len()
    )
    
    # Save (triggers actual computation)
    ddf_filtered.to_parquet(output_path)

# Advanced: Using Dask Bag for unstructured data
import dask.bag as db

def process_jsonl_with_dask(input_pattern: str):
    """Process JSONL files with Dask Bag"""
    bag = db.read_text(input_pattern).map(json.loads)
    
    # Chained processing
    result = (
        bag
        .filter(lambda x: len(x.get('text', '')) > 100)
        .map(lambda x: {**x, 'text_length': len(x['text'])})
    )
    
    # Convert to DataFrame and save
    result.to_dataframe().to_parquet('output/')
```

**Three-Framework Selection Summary**:

| Dimension | Apache Spark | Ray Data | Dask |
|------|-------------|----------|------|
| **Best Scenario** | SQL/ETL, data lake | GPU/ML inference | Pandas parallelization |
| **Learning Curve** | Medium (need Spark API) | Medium (need Ray API) | Very low (zero barrier for Pandas users) |
| **PB-level Performance** | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **GPU Support** | Plugin | Native | Plugin |
| **Target Audience** | Data engineers | AI/ML engineers | Data scientists |

For typical LLM data engineering scenarios (TB-level text data + occasional GPU inference), the recommended combination is: **Spark for ETL, Ray for ML inference, Dask for rapid prototyping and medium-scale processing**.

---

### 2.1.4 Vector Database Selection

With the rise of RAG (Retrieval-Augmented Generation) and multimodal search, vector databases have become an indispensable component of the AI data stack. Vector databases are purpose-built for storing and retrieving high-dimensional vectors (embeddings), serving as the bridge between data engineering and model inference.

#### Core Concepts

The core operation of a vector database is **Approximate Nearest Neighbor (ANN) search**. Given a query vector $q$, find the $k$ most similar vectors in the database. Exact search is prohibitively expensive in high-dimensional spaces, so practical systems use approximate algorithms, trading off between **Recall** and **Query Throughput (QPS)**.

Mainstream ANN indexing algorithms include:

- **HNSW (Hierarchical Navigable Small World)**: Graph-based algorithm with high recall and fast queries, but large memory footprint. Best for scenarios requiring very high recall.
- **IVF (Inverted File Index)**: Clustering-based algorithm that partitions the vector space into Voronoi regions, searching only the nearest regions at query time. Good memory efficiency, suitable for large-scale data.
- **ScaNN (Scalable Nearest Neighbors)**: Developed by Google, combining quantization and pruning techniques for excellent QPS-Recall balance.

#### Vector Database Comparison

| Feature | Milvus | Qdrant | Weaviate | Pinecone | FAISS |
|------|--------|--------|----------|----------|-------|
| **Deployment** | Self-hosted/Cloud | Self-hosted/Cloud | Self-hosted/Cloud | Pure SaaS | Library (not a DB) |
| **Open Source** | Yes | Yes | Yes | No | Yes |
| **Index Algorithms** | HNSW, IVF, DiskANN | HNSW | HNSW | Proprietary | HNSW, IVF, PQ |
| **Distributed** | Native support | Supported | Supported | Managed | Manual sharding |
| **Hybrid Search** | Supported | Supported | Supported | Supported | Not supported |
| **Suitable Scenario** | Large-scale production | Small-medium, high perf | Full-stack semantic search | Quick start, zero ops | Research prototypes |

**Selection Decision Points**:

- **QPS vs Recall tradeoff**: For pre-training data deduplication, high Recall (>0.99) is needed but lower QPS is tolerable; for online RAG retrieval, high QPS (>1000) is needed with slightly lower Recall acceptable.
- **Data scale**: Under millions of vectors, Qdrant or FAISS suffice; for tens of millions to billions, Milvus’s distributed architecture has advantages.
- **Operations capability**: If the team lacks ops experience, Pinecone’s fully managed mode is the lowest-risk choice.

```python
# Milvus vector retrieval example
from pymilvus import (
    connections, Collection, FieldSchema,
    CollectionSchema, DataType, utility
)

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define Schema
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, description="Document embeddings")

# Create Collection
collection = Collection("documents", schema)

# Create HNSW index (high recall configuration)
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {
        "M": 16,               # Connections per node; higher = better recall but more memory
        "efConstruction": 256  # Search width during construction
    }
}
collection.create_index("embedding", index_params)

# Search
collection.load()
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 128}},
    limit=10,
    output_fields=["text"]
)
```

#### Object Storage High-Throughput Read Optimization

In GPU training scenarios, data loading speed often becomes the bottleneck. When training data is stored on S3/MinIO, network I/O latency and throughput limits can leave GPUs in a "starving" state—compute units waiting for data arrival. Key optimization strategies include:

**Prefetching and pipelining**: While GPU processes the current batch, CPU prefetches next batch data, overlapping compute and I/O.

**Local SSD caching**: Cache frequently accessed hot data on local NVMe SSD. First read pulls from S3, subsequent reads hit local cache. Tools like Alluxio and JuiceFS provide transparent caching layers.

**Multi-threaded concurrent reads**: S3 supports Range Requests; multiple concurrent segment requests can fully utilize network bandwidth.

**Data format optimization**: Use columnar formats (Parquet) with column pruning to load only training-needed columns; use Arrow IPC format for zero-copy reads.

```python
import ray

def optimized_data_loading(s3_path: str, num_workers: int = 8):
    """Optimized S3 data loading with Ray parallel prefetch"""
    
    # Use Ray Data streaming reads with automatic prefetch management
    ds = ray.data.read_parquet(
        s3_path,
        parallelism=num_workers * 4,  # Prefetch multiplier
        columns=["input_ids", "attention_mask"],  # Column pruning
    )
    
    # Pipelining: read and process simultaneously
    pipe = ds.iter_batches(
        batch_size=1024,
        prefetch_batches=4,  # Prefetch 4 batches
        local_shuffle_buffer_size=10000  # Local shuffle
    )
    
    return pipe
```

## 2.2 Data Format and I/O Optimization

With storage and compute selected, next is choosing data serialization format. Format selection may seem a technical detail but actually directly affects storage cost, read speed, and tool compatibility. Compression ratio differences between formats can reach tenfold; columnar vs row format query performance differences are equally huge; and not all frameworks support all formats.

### 2.2.1 Mainstream Data Format Comparison

**Parquet** is the de facto standard for large-scale structured data. It uses columnar storage—data from the same column is physically contiguous—bringing two significant advantages: First, it facilitates compression—same-type data gathered together achieves higher compression ratio; second, it facilitates vectorized reads, when querying specific columns there's no need to scan the entire file. Parquet files are self-describing with Schema embedded in the file, no external metadata definition needed. It also supports nested types for JSON-like complex structures and native directory partitioning. Parquet is the preferred format for pre-training corpus storage, especially for analysis queries requiring column filtering; works well with Spark, DuckDB, Pandas, and other tools.

**JSONL** (JSON Lines) is another common format, each line an independent JSON object. Its biggest advantage is human readability—can directly view content with `head`, `cat`, and other commands. It also supports streaming processing, can read line by line without loading entire file to memory. Schema is very flexible, each line can have different field structure. JSONL is especially suitable for SFT instruction data, as this type of data requires frequent manual viewing and editing. It's also commonly used for data exchange and small-scale datasets (under 10GB). However, JSONL's disadvantages are also clear: without compression volume is three to five times Parquet, read speed is slow (must parse each line's JSON string).

**WebDataset** is a format spearheaded by NVIDIA, designed specifically for image-text, video, and other multimodal data. Its core idea is packaging related files (e.g., one image and its caption) into TAR archives. This design supports streaming reads—can sequentially read content without decompression; also very friendly to distributed processing—each TAR is an independent data shard. WebDataset is the best choice for LAION-style image-text pair datasets and video datasets, suitable for any multimodal data requiring multi-file association.

| Feature | Parquet | JSONL | WebDataset |
|------|---------|-------|------------|
| **Storage Efficiency** | High (columnar compression) | Low (text redundancy) | Medium (no compression but compact) |
| **Read Speed** | Fast (vectorized) | Slow (line-by-line parsing) | Medium (sequential read) |
| **Human Readable** | No | Yes | No |
| **Multimodal Support** | Weak (requires encoding) | Weak | Strong (native support) |
| **Typical Use Case** | Pre-training text corpus | SFT instruction data | Image-text pairs, video data |

### 2.2.2 Compression Algorithm Selection

Regardless of format choice, compression algorithm significantly affects storage cost and read speed. Correct compression strategy requires finding balance between space efficiency and time efficiency.

Snappy is the most common default choice. Its compression ratio is moderate but compression and decompression speeds are both fast, suitable for read-write balanced scenarios. LZ4 pursues extreme read speed—decompression performance even faster than Snappy, slightly lower compression ratio, suitable for read latency sensitive scenarios. Zstandard (ZSTD) provides highest compression ratio, especially at high levels (e.g., level 19), but compression speed is slower, suitable for storage cost sensitive archival scenarios. Gzip is the most compatible choice—almost all tools support it—suitable for scenarios requiring data exchange with external systems.

| Algorithm | Compression Ratio | Compression Speed | Decompression Speed | Typical Use Case |
|------|--------|----------|----------|----------|
| **Snappy** | Medium | Fast | Fast | Default choice, read-write balanced |
| **LZ4** | Lower | Very fast | Very fast | Extreme read speed |
| **ZSTD** | High | Medium | Fast | Storage cost sensitive |
| **Gzip** | High | Slow | Medium | High compatibility requirement |

In practice, a layered strategy can be adopted: cold data (archived storage, rarely read long-term) use ZSTD level 19 for maximum compression ratio; hot data (frequently read and processed) use Snappy or LZ4 to reduce decompression overhead; network transfer scenarios use ZSTD level 3 for balance between compression ratio and speed.

### 2.2.3 I/O Optimization Practical Tips

In large-scale data processing, I/O is often the performance bottleneck. The following three tips can significantly improve I/O efficiency.

**Reasonable file size settings** is the first key point. A common mistake is generating many small files—e.g., 100,000 files of 1MB each. This causes huge metadata overhead and extremely slow S3 ListObjects operations. The correct approach is consolidating data into fewer large files, each Parquet file should be 128MB to 1GB. Too small causes metadata bloat and insufficient parallelism; too large affects task load balancing.

```python
# Wrong: generate many small files
df.write.parquet("s3://bucket/data/", maxRecordsPerFile=1000)

# Correct: generate fewer large files (recommend 128MB - 1GB)
df.coalesce(100).write.parquet("s3://bucket/data/")
```

**Partition Pruning** is the second important tip. By partitioning by specific columns when writing, only needed partitions need to be scanned when reading, avoiding full table scan. Partition columns should be low cardinality (e.g., date, language, data source); avoid high cardinality columns (e.g., user ID) or you'll get massive small directories.

```python
# Partition by date when writing
df.write.partitionBy("date").parquet("s3://bucket/data/")

# Only scan needed partitions when reading
spark.read.parquet("s3://bucket/data/date=2024-01-01/")
```

**Column Pruning** is the third tip. The biggest advantage of columnar storage is only reading needed columns. Ensure column selection happens early in query statements, avoid reading all columns then filtering.

```python
# Wrong: read all columns
df = spark.read.parquet("s3://bucket/data/")  # If 100 columns, all loaded

# Correct: only read needed columns
df = spark.read.parquet("s3://bucket/data/").select("text", "length")
```

![Figure 2-5: I/O Optimization Effect Comparison](../../images/part1/图2_5_IO优化效果对比.png)

*Figure 2-5: I/O Optimization Effect Comparison — Partition pruning + column pruning can reduce query time by 91% and data scan volume by 92%*

Combined use of these three tips can reduce query time from 55 seconds to 5 seconds, data scan volume from 100GB to 8GB—very significant effect.

---

## 2.3 Data Version Control (DataOps)

Code has Git, machine learning models have MLflow—so how do you version control TB-level datasets? This is an often overlooked but extremely important question in LLM data engineering.

### 2.3.1 Why Does Data Need Version Control?

Consider this scenario: a model trained six months ago performed particularly well, and the boss wants reproduction. You search through servers and find the training data was already cleaned up—"Who told you to delete it?" "It took 10TB!" The data processing scripts are still there, but dependent upstream data has changed. Re-running the processing flow yields different results. Conclusion: cannot reproduce.

This scenario is common in actual work. Data version control exists precisely to solve such problems. Its core value is reflected in four aspects: Reproducibility—exactly restore data state at any moment; Traceability—track complete chain from raw input to final output; Collaboration safety—multiple people modifying data simultaneously won't conflict; Rollback capability—quickly return to previous version when data issues are discovered.

### 2.3.2 Tool Selection: DVC vs LakeFS

The two most mainstream data version control tools today are DVC and LakeFS, with distinctly different design philosophies.

**DVC (Data Version Control)** design philosophy is "Git for Data"—making data version control experience as close to Git as possible. Its working principle: data files themselves are stored in remote storage (S3/GCS), Git repo only stores data metadata files (`.dvc` files), actual data synced via `dvc push/pull` commands.

```bash
# Initialize DVC
dvc init

# Add dataset to version control
dvc add data/training_corpus.parquet
# Generates data/training_corpus.parquet.dvc and .gitignore

# Commit to Git
git add data/training_corpus.parquet.dvc .gitignore
git commit -m "Add training corpus v1"

# Push data to remote storage
dvc push

# Switch to historical version
git checkout v1_0
dvc checkout  # Sync corresponding version data
```

DVC's advantage is seamless integration with existing Git workflow, gentle learning curve, ML pipeline definition support (via `dvc.yaml`), suitable for file-level version control scenarios. Its disadvantage is each dataset needs separate `.dvc` file management, no support for fine-grained "table-level" operations (e.g., rolling back a partition).

**LakeFS** design philosophy is "Git for Data Lake"—providing Git-style branches and commits on top of object storage. Its working principle: LakeFS acts as object storage proxy layer, all read/write requests go through LakeFS S3 gateway, system supports Branch, Commit, Merge and other Git-style operations.

```bash
# Create development branch
lakectl branch create lakefs://repo/dev --source lakefs://repo/main

# Modify data on dev branch (via S3 protocol)
aws s3 cp new_data.parquet s3://lakefs-repo/dev/data/

# Commit changes
lakectl commit lakefs://repo/dev -m "Add new training data"

# Merge to main branch after validation
lakectl merge lakefs://repo/dev lakefs://repo/main
```

LakeFS's core advantage is zero-copy branching—creating branches doesn't copy data, only records metadata, crucial for TB-level data lakes. It's fully S3 compatible; existing tools (Spark/Ray) work without modification. Its disadvantage is requiring deployment of additional service (LakeFS Server), slightly steeper learning curve than DVC.

**Pachyderm** is a third noteworthy data version control tool, unique in that it **integrates data version control with data pipelines**. Pachyderm is built on Kubernetes, where each data processing step runs in a container, and the system automatically tracks the correspondence between input data, processing code, and output data.

```bash
# Pachyderm workflow example

# Create data repository (similar to Git repo)
pachctl create repo raw_data

# Upload data (auto-versioned)
pachctl put file raw_data@master:/corpus.parquet -f corpus.parquet

# Create processing pipeline (declarative YAML)
pachctl create pipeline -f cleaning_pipeline.json
# Pipeline defines: input repo, processing container, output repo
# Pachyderm automatically tracks the complete input→processing→output lineage

# View data lineage
pachctl inspect commit cleaned_data@master
# Output shows which commit from raw_data, through which pipeline, generated this data
```

Pachyderm’s core advantage is **automated lineage tracking**—when input data is updated, downstream pipelines automatically trigger incremental processing, with the system naturally recording complete data lineage relationships. This is very valuable in LLM projects that require frequent iteration of data processing flows. Its downside is requiring a Kubernetes cluster (highest deployment complexity) and the steepest learning curve.

| Feature | DVC | LakeFS | Pachyderm |
|------|-----|--------|----------|
| **Design Philosophy** | Git extension for data | Version layer for object storage | Data pipeline + version control |
| **Granularity** | File-level | Object-level (finer) | File/directory-level |
| **Branch Overhead** | Need to copy .dvc files | Zero-copy | Zero-copy |
| **S3 Compatibility** | Requires dvc commands | Native S3 API | Native S3 API |
| **Lineage Tracking** | Manual | Manual/integration | **Automatic** |
| **Incremental Processing** | Manual | Manual | **Auto-triggered** |
| **Deployment Complexity** | Low (CLI tool) | Medium (requires server) | High (requires Kubernetes) |
| **Suitable Scenario** | ML experiment management, small data | Data lake management, large-scale data | End-to-end data pipelines |

![Figure 2-6: DVC vs LakeFS Architecture Comparison](../../images/part1/图2_6_DVC与LakeFS架构对比.png)

*Figure 2-6: DVC vs LakeFS Architecture Comparison — DVC provides file-level version control based on Git, LakeFS provides zero-copy object-level version control with branching*

Selection recommendations: If data volume under 1TB, team familiar with Git workflow, mainly for ML experiment management, choose **DVC**; if data volume TB-level or above, need data lake-level version control, multiple teams operating in parallel, choose **LakeFS**; if you need end-to-end data pipeline management and the team has Kubernetes operations capability, choose **Pachyderm**.

### 2.3.3 Data Lineage Tracking

Version control solves "what is the data"; lineage tracking solves "where did the data come from." Lineage tracking records: which upstream data was this data derived from? What processing scripts and parameters were used? When and by whom was the processing executed?

There are multiple approaches to implement lineage tracking. If using Spark, automated lineage tracking can be obtained through OpenLineage integration. If using orchestration tools like Airflow, Marquez is a good choice. For enterprise data governance needs, DataHub and Apache Atlas provide more complete functionality. For simple scenarios, manual instrumentation to generate metadata files is a lightweight solution:

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
        "params": {"min_length": 100, "dedup_threshold": 0.9}
    },
    "outputs": [
        {"path": "s3://bucket/processed/clean_2024_q1.parquet", "records": 1000000}
    ]
}

with open("clean_2024_q1.metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

---

## 2.4 Common Mistakes and Pitfall Guide

In the infrastructure selection process, even experienced engineers easily make some typical mistakes. Here we summarize three most common issues, hoping readers can take heed.

**The first common mistake is premature optimization and over-engineering.** Some teams have only five people and 500GB of data, yet build a "full-stack" infrastructure of Spark cluster + Iceberg + Airflow + LakeFS. Result: 80% of time spent maintaining infrastructure, only 20% on actual data processing. The correct approach is start simple, evolve on demand. For 500GB data volume, single machine + Parquet + DVC is completely sufficient; consider distributed solutions when data volume grows to 10TB.

**The second common mistake is blindly chasing new technology while ignoring ecosystem.** Some teams read a few blog posts and decide to abandon Spark entirely for Ray, only to find company Hive tables and Iceberg tables cannot be read directly. Finally need to write substantial data conversion scripts, increasing data consistency risk. The correct approach is fully evaluate existing data assets and upstream-downstream dependencies before technical selection. Technical selection is not a single-point decision but systems engineering—overall ecosystem compatibility must be considered.

**The third common mistake is overly aggressive storage cost optimization.** Some teams compress all data to ZSTD level 22 and store in S3 Glacier Deep Archive to save storage costs. Result: every time data needs to be read, wait 12 hours for thaw, decompression takes another 4 hours, model training has to be scheduled a week in advance. The correct approach is distinguish cold and hot data. Actively processed data goes in S3 Standard + Snappy compression; archival data unused for six months or more goes to Glacier. Storage cost and access efficiency need to find a balance point.

---

## 2.5 Chapter Summary

This chapter systematically introduced AI-native data stack technology selection, covering five core dimensions: storage, compute, vector databases, format, and version control.

For storage selection: object storage (S3/MinIO) is the foundation of modern data stack; data lake formats (Iceberg/Hudi/Delta) solve ACID transactions, time travel, and other issues. For LLM scenarios, the recommended combination is S3 + Iceberg, because Iceberg has the best engine neutrality. For GPU training scenarios, I/O bottlenecks need optimization through prefetch pipelining, local SSD caching, and concurrent reads.

For compute selection: Spark is known for maturity and stability and powerful SQL ecosystem, suitable for traditional big data teams; Ray Data is Python-native AI-friendly framework, suitable for ML/AI teams; Dask offers a zero-learning-curve option for teams with existing Pandas code. The three are not mutually exclusive—can be mixed according to needs.

For vector databases: Milvus, Qdrant, Weaviate, and other systems provide foundational capabilities for RAG and semantic retrieval. Selection requires balancing QPS and Recall based on data scale and operations capability.

For data format: Parquet is the default for structured data, JSONL suitable for small-scale data requiring manual viewing, WebDataset is the best format for multimodal data. Compression algorithms and I/O optimization tips can significantly affect performance and cost.

For version control: DVC is lightweight and tightly integrated with Git, suitable for ML experiments; LakeFS provides data lake-level version control, suitable for large-scale production environments; Pachyderm integrates version control with data pipelines, suitable for teams needing end-to-end lineage tracking.

The core principle throughout: start simple, evolve on demand, avoid over-engineering. Technical selection should serve business goals, not pursue technical advancement for its own sake.

![Figure 2-7: Infrastructure Selection Quick Reference](../../images/part1/图2_7_基础设施选型速查表.png)

*Figure 2-7: Data Infrastructure Selection Quick Reference — Four-quadrant decision guide for storage, table format, compute, and version control*

---

## Further Reading

For readers wishing to deepen understanding of this chapter's content, the following resources are worth referencing:

Ray Data official documentation (docs.ray.io) provides Ray Data best practices and detailed API reference. Apache Iceberg official documentation (iceberg.apache.org) contains table format detailed specifications and engine integration guides. DVC official tutorial (dvc.org/doc) is a good starting point for quick start. LakeFS official documentation (docs.lakefs.io) details architecture design and deployment options.

Databricks' published data lake selection white paper provides in-depth comparative analysis of Delta, Iceberg, and Hudi formats. Uber's published "Scaling MLOps at Uber" article introduces how to manage ML data at PB scale. These materials can help readers build more comprehensive technical perspective.

---

## Preview of Next Chapter

In the next chapter *Data Acquisition and Collection*, we will formally enter the pre-training data processing flow. You will learn how to obtain and parse open-source datasets like Common Crawl and The Pile, how to use Trafilatura to build high-performance web page parsers, and specialized strategies for crawling code and papers from GitHub and ArXiv.

Take this question into the next chapter: Common Crawl adds 3-5PB of data monthly—how do you efficiently extract the content you need from it?
