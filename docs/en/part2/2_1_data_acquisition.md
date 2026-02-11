## Chapter 6: Image-Text Pairs Processing

### Chapter Summary

In building the next generation of Foundation Models, the focus of data engineering has shifted from simple text cleaning to capturing, aligning, and reconstructing multi-dimensional signals from the physical world. If language model data engineering is about "denoising," multimodal data engineering is about "correlation" and "alignment." With the emergence of GPT-4V, Gemini, and Sora, we have come to realize that single-modality data can no longer satisfy models' desire to understand the world.

This chapter provides an in-depth analysis of the complete engineering pipeline for building billion-scale multimodal datasets. This is far more than writing a few scripts to download images—it is a comprehensive campaign involving network protocols, distributed storage, heterogeneous computing, and aesthetic evaluation. We will explore the underlying logic of data paradigms, analyze how to use distributed computing frameworks to solve the challenge of high-concurrency acquisition of massive images, and leverage GPU hardware acceleration to break through the I/O bottleneck of image preprocessing. Additionally, we will build an automated cleaning loop based on semantic and aesthetic criteria to ensure that data fed to the model is both relevant and safe.

**Learning Objectives**:
* Deeply understand the training benefits and engineering challenges of LAION-5B (image-text pairs) and OBELICS (interleaved documents) paradigms, and master the design methods for hybrid data strategies.
* Be able to write distributed downloaders based on PySpark and Ray Data, handle DNS bottlenecks and long-tail latency, and achieve throughput of 10,000+ img/s.
* Master NVIDIA DALI pipeline design, solve CPU decoding bottlenecks, and optimize data loading using GPU Direct concepts.
* Build a multi-stage cleaning funnel including CLIP semantic filtering, aesthetic scoring, and safety detection, and master threshold tuning strategies for different business scenarios.

**Scenario Introduction**:
> "Imagine this scenario: Your crawler team has just extracted 2 billion raw URLs from Common Crawl, stored in thousands of Parquet files. Your task is to transform this data into a high-quality dataset suitable for GPT-4V pre-training within two weeks. When you try to download with the traditional Python requests library on a single machine, you find the estimated time is a staggering 15 years—a classic network I/O blocking problem. Worse yet, preliminary sampling shows that 30% of downloaded images are e-commerce ads (full of noise), 15% have severe watermarks, and there is even serious NSFW content. If we use this data directly, we will not only waste millions of dollars in compute, but the trained model may also face legal risks due to generating prohibited content. We need an industrial-grade, high-throughput, intelligent data engineering solution to meet this challenge."

### 6.1 Data Paradigms: Image-Text Pairs (LAION-5B) vs. Interleaved Documents (OBELICS/MMC4)

Before designing the data pipeline, our first responsibility is to clarify the organizational form of the data. This is not only about storage structure but also directly determines the training objective and emergent capabilities of downstream models. Different data forms are essentially different abstractions of "how knowledge exists in the world."

#### 6.1.1 Core Concepts and Principles

**Image-Text Pairs**
are the cornerstone of multimodal learning, represented by CLIP, ALIGN, and LAION-5B.
* **Theoretical Analysis**: This paradigm assumes a strong semantic correlation between image $I$ and text $T$, and this correlation is independent and atomic. The training objective is typically to maximize the cosine similarity of $I$ and $T$ in a shared embedding space (Contrastive Learning). Its advantage lies in extremely high "signal-to-noise ratio" refinement potential—through contrastive learning, the model can learn direct mapping between objects and vocabulary.
* **Engineering Perspective**: The data structure is simple, typically represented as flattened records of `(url, caption, metadata)`. This data is extremely easy to shard and randomly shuffle. During training, due to sample independence, we can easily implement Global Batch Shuffling to improve contrastive learning effectiveness.

**Interleaved Image-Text Documents**
are the key fuel for the new generation of multimodal large models (e.g., Flamingo, GPT-4V, MM1), represented by OBELICS and MMC4.
* **Theoretical Analysis**: This paradigm preserves the original DOM structure order of web pages, with data presented as sequences of `<text>, <image>, <text>...`. This forces the model to learn "multimodal contextual dependencies" (Multimodal In-Context Learning). For example, in a "how to make a cake" webpage, the relationship between the first image (ingredients) and the fifth image (final product), as well as their logical connection to surrounding text, cannot be provided by image-text pairs. It simulates the cognitive process of humans reading image-text mixed documents.
* **Engineering Perspective**: The data pipeline is extremely complex. Because single samples (documents) have variable lengths and may contain multiple images, batch assembly becomes difficult. Traditional Collators require complex padding strategies. Furthermore, when cleaning, care must be taken to maintain document integrity—arbitrarily deleting a low-quality image may break contextual logic and cause the model to learn incorrect referential relationships.

#### 6.1.2 Architectural Decisions: Paradigm Comparison Table

With limited resources, how do we balance these two data paradigms? This is not a simple binary choice but involves deep trade-offs between model architecture, training cost, and ultimate application scenarios.

In early multimodal research (before 2021), the industry widely believed that sufficient data volume (e.g., CLIP's 400 million pairs) would suffice for models to learn everything. However, with the emergence of GPT-4V, we found that models trained solely on image-text pairs, while able to accurately identify "this is a cat," cannot answer "what might this cat in the image do" because they lack the context for logical reasoning. Conversely, interleaved documents, though rich in logic, are sparse in data and have extremely high processing costs.

The table below compares the core differences between the two paradigms at the engineering implementation level, helping architects make technical selections based on actual requirements:

| Dimension | Image-Text Pairs (LAION-style) | Interleaved Documents (OBELICS-style) | In-depth Analysis & Recommendation |
| :--- | :--- | :--- | :--- |
| **Training Objective** | Contrastive Learning (CLIP), Text-to-Image (Stable Diffusion) | Next-Token Prediction, Multimodal Dialogue (GPT-4V) | **Hybrid strategy is the way to go**. Research shows that training visual encoders solely with interleaved documents is inefficient (images are not dense enough), while using only image-text pairs lacks reasoning ability. A Curriculum Learning strategy is recommended. |
| **Data Source Parsing** | Simple: only need to extract `<img>` tags and Alt-text | Complex: need to parse DOM tree, filter ads/sidebars, preserve main content logic | **Engineering complexity warning**. Building interleaved documents requires handling extremely complex HTML rendering logic. It is recommended to initially use Common Crawl's WET files for construction, or directly use OBELICS open-source dataset for augmentation, rather than attempting to re-clean the entire internet from scratch. |
| **Storage Cost** | Medium: metadata only in CSV/Parquet, images stored separately | High: need to save document topology, recommend WebDataset or TFRecord | **I/O performance bottleneck**. For interleaved documents, sharded storage must be used to avoid small file fragmentation. Reading requires pre-reading entire documents, placing higher demands on memory bandwidth. |
| **Cleaning Challenges** | Point-wise: each image judged independently, easy to parallelize | Contextual: need to consider text coherence and image quality simultaneously, cleaning logic coupled | **Strategy choice**. When processing interleaved documents, if an image is deemed NSFW, recommend replacing with a special `<BLOCKED_IMAGE>` token rather than deleting it directly, to maintain positional embedding accuracy. |
| **Model Benefits** | Extremely strong visual-semantic alignment, strong Zero-shot classification | Strong Few-shot Learning, supports multi-turn dialogue and logical reasoning | **Business-oriented**. If the scenario is "image search," image-text pairs suffice; if the business involves complex document understanding (e.g., research report analysis, long-form story generation), interleaved documents must be introduced. |

> **Tips:**
> In cutting-edge research like MM1 and Idefics2, best practice is not to choose one or the other, but to mix. It is typically recommended to use **80% image-text pairs** in the early pre-training phase to establish a solid visual-language mapping foundation, while mixing in **20% interleaved documents**; in the late pre-training phase (Annealing Phase), significantly increase the proportion of interleaved documents to stimulate the model's long-context reasoning ability. This "foundation first, logic later" strategy maximizes compute utilization.

### 6.2 Image Acquisition and Preprocessing

Once the data manifest is determined, the next step is to build a high-throughput download and preprocessing pipeline. This is a typical I/O-intensive task, with main bottlenecks in network bandwidth, DNS resolution latency, and disk writes for massive small files.

#### 6.2.1 img2dataset High-Concurrency Download in Practice

`img2dataset` is currently the community-recognized best practice tool. It is not merely a download script but a distributed data processing framework based on MapReduce principles.

Why use a specialized tool rather than writing a simple `requests.get` loop? Because the internet environment is extremely harsh. Links expire (Link Rot), servers rate-limit, and DNS times out. When processing billions of URLs, any tiny long-tail latency gets amplified into weeks of time cost.

**Core Principles**:
1.  **Sharding**: Split 1 billion URLs into tens of thousands of small tasks (Shards). This is the foundation of distributed computing.
2.  **Async I/O**: Use Python's aiohttp or Go's goroutines to concurrently initiate hundreds of network requests per core, masking network latency.
3.  **Streaming Archival**: Downloaded images are not written to disk; they are directly assembled into tar packages (WebDataset format) in memory, then streamed to object storage (S3/HDFS). This avoids exhausting the filesystem inode when creating millions of small files in one directory—a pitfall that new practitioners often encounter.

**Engineering Implementation: PySpark Distributed Download Script**

When processing PB-scale data, single-machine multiprocessing mode is insufficient; a Spark cluster must be used.

```python
# Recommended environment: PySpark 3_2+, img2dataset 1_41+
# Run command: spark-submit --master yarn --deploy-mode cluster...

from img2dataset import download
import shutil
import os

def run_distributed_download():
    """
    Configuration tuning is key to throughput.
    process_count: Number of processes per Spark Executor.
    thread_count: Number of async threads per process.
    For nodes with 10Gbps NIC, typically recommend total_concurrency around 1000.
    """
    
    # Define output path (S3 or HDFS)
    output_dir = "s3a://multimodal-lake/raw-images/laion-5b-subset"
    
    # Clean old data (use with caution, production recommends versioning)
    if os.path.exists(output_dir): 
        # shutil.rmtree(output_dir) # Dangerous operation, commented out
        pass

    download(
        processes_count=4,          # 4 CPU cores per node
        thread_count=64,            # 64 download threads per core
        url_list="s3a://multimodal-lake/meta/laion-urls.parquet",
        image_size=256,             # 256x256 sufficient for pre-training, saves bandwidth
        resize_only_if_bigger=True, # Avoid blur from upscaling small images
        resize_mode="keep_ratio",   # Maintain aspect ratio, pad or center crop
        skip_reencode=True,         # If original is JPG and size suitable, store directly, saves CPU
        output_folder=output_dir,
        output_format="webdataset", # Force WebDataset format
        input_format="parquet",
        url_col="url",
        caption_col="caption",
        enable_wandb=True,          # Strongly recommended for monitoring download rate and error rate
        number_sample_per_shard=10000, # 10k images per tar, ~200-300MB, easy to transfer
        distributor="pyspark",      # Use Spark for task distribution
        save_additional_columns=["similarity", "hash"], # Preserve original metadata
        timeout=10                  # Short timeout, fail fast, long-tail requests not worth waiting
    )

if __name__ == "__main__":
    # Initialize Spark Session (usually handled by spark-submit, but declare explicitly for IDE debugging)
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("Img2Dataset-Production") \
        .config("spark.executor.memory", "8g") \
        .config("spark.task.maxFailures", "10") \
        .getOrCreate()
    
    run_distributed_download()
```

**Pro Tips**:
* **DNS Caching**: Under high concurrency, DNS resolution can become a bottleneck or even get blocked by providers. Deploy local DNS cache (e.g., dnsmasq) on worker nodes, or maintain a domain-to-IP mapping table in code.
* **User-Agent Rotation**: Though an "open" secret, rotating User-Agent can reduce 403 Forbidden rates.
* **Error Handling**: Monitor success_rate in the WandB dashboard. If below 80%, it usually means the URL list is severely stale or your IP pool is contaminated.

#### 6.2.2 The Pitfalls of Visual Preprocessing: Cropping and Semantic Alignment

After solving the challenge of acquiring massive data (Getting bytes), we immediately face the second challenge: data usability. Raw internet images have wildly varying aspect ratios, while models typically require fixed resolution input (e.g., 224x224 or 512x512).

Many novice engineering solutions habitually use brute-force random preprocessing to unify dimensions, but this is often the root of the model's "invisible performance ceiling." We must not only focus on "getting the image in" but also "what is being put in."



![Figure 6-1: Cropping and Semantic Alignment in Image Preprocessing](../../images/part3/图6_1_图片预处理中裁剪与语义对齐问题.png)
*Figure 6-1: Cropping and Semantic Alignment in Image Preprocessing*

* **Bad Case (Left - The Cost of Mechanical Cropping)**:
    Traditional `RandomCrop` or `CenterCrop` have no awareness of composition. When processing a portrait photo in vertical composition, center cropping can easily cut off key features (e.g., the head), leaving only the torso. At this point, if the text label is still "a smiling man," the model is forced to establish incorrect mappings (mistaking torso features for "smiling person"), causing severe visual hallucinations in the trained model.

* **Good Case (Right - Semantic Integrity)**:
    High-quality data engineering pursues "image-text consistency."
    1.  **Smart Resize**: Prefer `Resize with Padding` (maintain aspect ratio, pad with black/white borders) to preserve complete visual subjects. Though this introduces invalid pixels, it ensures semantic integrity.
    2.  **Aspect Ratio Bucketing**: This is an advanced technique commonly used by generation models like SDXL and Midjourney. Group images with similar aspect ratios into the same batch for training, avoiding cropping while reducing padding waste.
    3.  **Recaptioning**: As detailed in Chapter 7, using VLM to generate high-density descriptions allows text to precisely correspond to details in the image (e.g., sign text, background objects), maximizing training value of the data.

#### 6.2.3 GPU-Accelerated Decoding and Transformation (NVIDIA DALI)

In the deep learning model training phase, most researchers and developers focus their attention on model architecture design, hyperparameter tuning, loss function improvements, and other modules that directly affect model accuracy, while easily overlooking the data loading (DataLoader) phase—yet in practice, it often becomes the "invisible performance killer" that constrains training efficiency, even preventing full utilization of high-end GPU compute and causing serious hardware waste.

To understand this pain point, we must first clarify the complete logic of the deep learning training flow: model training's core compute relies on GPU's massive parallel computing capability; GPUs can efficiently process massive tensor operations and complete backpropagation and parameter updates. But before data reaches the GPU, it must go through a series of preprocessing operations, among which the most basic and time-consuming are image decoding and resizing. In the traditional PyTorch training flow, these critical preprocessing operations are entirely performed by the CPU, creating a contradiction between "CPU preprocessing bottleneck" and "GPU compute redundancy."

Specifically, the traditional PyTorch Dataset workflow is: first read image files stored on disk (mostly JPEG format) via CPU, then the CPU performs JPEG decoding—this process requires complex computation such as Huffman decoding and inverse discrete cosine transform (IDCT) on compressed image binary data, a typical CPU-intensive task; after decoding, the CPU performs Resize, normalization, color space conversion, and other preprocessing, and finally transfers the processed image tensor to the GPU for model training via data copy.

More critically, CPU architecture is designed for serial computation and logic control, with parallel computing capability far inferior to GPU. However, decoding and Resize in image preprocessing are highly parallelizable and can improve efficiency through multi-threading or multi-core processing. But even with DataLoader's num_workers parameter to increase CPU parallelism, traditional PyTorch Dataset struggles to break through the CPU's compute ceiling—especially when the training dataset is large (e.g., millions of images) and single-image resolution is high (e.g., 1080P and above), CPU preprocessing speed will seriously lag behind GPU training speed, causing the GPU to frequently idle waiting for data, significantly reducing GPU utilization, and ultimately dragging down overall training efficiency. This is why data loading is called the "neglected performance killer."

To address this core pain point, NVIDIA introduced DALI (Data Loading Library), a GPU-accelerated data preprocessing library optimized for deep learning training. Its core goal is to migrate image decoding, resizing, and other intensive preprocessing operations that originally relied on CPU to the GPU for parallel execution, breaking the data loading performance bottleneck and unleashing GPU compute.


![Figure 6-2: Data Decoding and Transformation with vs. without DALI](../../images/part3/图6_2_使用DALI与不使用DALI下数据解码与变换的区别.png)
*Figure 6-2: Data Decoding and Transformation with vs. without DALI*

**Code Walkthrough: High-Performance Pipeline Based on DALI**

```python
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def


@pipeline_def(batch_size=256, num_threads=8, device_id=0)
def webdataset_gpu_pipeline(shard_id, num_shards):
    """
    Define end-to-end GPU data loading pipeline
    Input: WebDataset (Tar) -> Output: GPU Tensor
    """
    
    # Step 1: Read WebDataset (CPU stage)
    # Using index_paths is required; otherwise init phase needs to traverse entire tar, extremely slow [5]
    jpegs, captions = fn.readers.webdataset(
        paths=["/data/shards/shard-{:05d}.tar".format(i) for i in range(100)],
        index_paths=["/data/indices/shard-{:05d}.idx".format(i) for i in range(100)],
        ext=["jpg", "txt"],
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=True,
        initial_fill=10000,      # Shuffle buffer size, larger = more random but slower startup
        pad_last_batch=True,     # Ensure all batches have consistent size
        name="Reader",
        read_ahead=True          # Enable prefetch
    )

    # Step 2: GPU Decoding (core acceleration point)
    # device="mixed" means input in Host memory, output in Device memory
    # output_type=types.RGB handles color space conversion automatically
    images = fn.decoders.image(
        jpegs,
        device="mixed",
        output_type=types.RGB,
        # Fault tolerance for corrupted images
        # In production, never let one bad image crash training
    )

    # Step 3: GPU transformation pipeline
    # resize: scale while maintaining aspect ratio
    images = fn.resize(
        images,
        resize_x=224,
        resize_y=224,
        interp_type=types.INTERP_LINEAR
    )
    
    # crop_mirror_normalize: random crop + flip + normalize (fused operator)
    # This step converts uint8 to float and subtracts mean, divides by std
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(224, 224),
        mean=[0_485 * 255, 0_456 * 255, 0_406 * 255],
        std=[0_229 * 255, 0_224 * 255, 0_225 * 255],
        mirror=fn.random.coin_flip(probability=0_5)
    )

    # Text data typically processed directly on CPU or passed to Tokenizer
    # Here we only return raw bytes for subsequent PyTorch processing
    return images, captions

# Use DALIGenericIterator integrated with PyTorch
from nvidia.dali.plugin.pytorch import DALIGenericIterator

pipe = webdataset_gpu_pipeline(shard_id=0, num_shards=1)
pipe.build()
dataloader = DALIGenericIterator(pipe, ["images", "captions"], reader_name="Reader")

# Benchmark: On A100, this pipeline typically achieves 3000-5000 FPS, 5-10x CPU Loader
```

### 6.3 Multimodal Cleaning Pipeline

Massive data comes with massive noise. In raw LAION-5B data, truly high-quality samples may account for less than 10%. We need to build a multi-stage cleaning funnel to improve data density while losing as little data diversity as possible. So-called "data cleaning" is essentially **Data Diet**—feeding the model less but better.

#### 6.3.1 Architecture Design: Ray Data Distributed Cleaning

Why choose Ray over Spark for the cleaning phase? Because cleaning is no longer simple ETL but includes substantial **deep learning inference (Model Inference)**. Compared to Spark's MapReduce paradigm, Ray provides a more flexible Actor mechanism, allowing us to keep GPU models (e.g., CLIP, Safety Checker) resident, avoiding the huge overhead of reloading multi-GB models for each small batch.

Ray Data is suitable for handling this mixed workload of both CPU-intensive (decompression, hashing, Regex) and GPU-intensive (CLIP Embedding inference) tasks. Below is a typical three-stage pipeline design:
* **Stage 1 (CPU)**: Quick filtering. Directly remove samples with insufficient resolution (<256px), too short text, non-English (if training English-only model), or abnormal aspect ratio.
* **Stage 2 (GPU)**: Deep feature extraction. Use CLIP model to generate Embeddings, and compute image-text similarity and aesthetic score based on Embeddings.
* **Stage 3 (CPU/Mixed)**: Logical judgment and deduplication. Comprehensive thresholding based on safety (NSFW), aesthetic score, and image-text relevance, plus semantic deduplication.

**Data Flow Diagram**

![Figure 6-3: Ray Data Distributed Cleaning Data Flow](../../images/part3/图6_3_Ray_Data分布式清洗数据流向图.png)
*Figure 6-3: Ray Data Distributed Cleaning Data Flow*

#### 6.3.2 Core Algorithm Implementation

Cleaning is not just deletion but also quantification of data value. We need multi-dimensional metrics to measure the "gold content" of an image and its corresponding text.

1.  **Aesthetics Scoring**
    * **Principle**: Datasets are filled with invoices, screenshots, blurry surveillance footage—these are useless for generating beautiful images. LAION-Aesthetics Predictor is typically used.
    * **Technical Details**: This is a simple MLP (multi-layer perceptron); input is CLIP Image Embedding, output is a 1-10 score. Training data comes from the AVA dataset (containing scores from professional photographers).
    * **Suggested Threshold**: For base pre-training, retain data with Score > 4_5; for fine-tuning high-quality generation models (SFT phase), recommend Score > 6_0, or even 6_5.

2.  **Image-Text Alignment Filtering**
    * **Principle**: Many Alt-texts are SEO garbage word stacking or filenames ("DSC_001.jpg"), unrelated to image content.
    * **Technical Details**: Compute cosine similarity (Dot Product) between CLIP Image Embedding and Text Embedding.
    * **Gotcha**: Different CLIP versions (e.g., OpenAI ViT-L/14 vs OpenCLIP ViT-G/14) have different embedding space distributions; scores are not directly comparable. Must recalibrate thresholds based on specific model. Common practice is to compute similarity distribution across the entire dataset, then retain Top 50% or Top 70%.

3.  **Safety Detection**
    * **Principle**: Must remove pornography, violence, and images with obvious brand watermarks.
    * **Strategy**: Use specially trained classifier heads (also based on CLIP Embedding) to detect NSFW and watermarks. For watermark detection: if the goal is training generation models (e.g., SDXL), must be extremely strict (Recall priority), as generation models easily overfit watermark features; if the goal is training understanding models (e.g., GPT-4V), can be relaxed, as understanding models need to recognize "there is a watermark in the image."

**Code Implementation: Ray Data Cleaning Operators**

```python
import ray
import torch
import open_clip
import numpy as np
from PIL import Image
import io

# Define Ray Actor class to ensure model is loaded only once
class QualityScorer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load CLIP model (ViT-B-32 fast, suitable for cleaning)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device
        )
        # Load aesthetic scoring head (Linear Layer)
        self.aesthetic_head = torch.nn.Linear(512, 1).to(self.device)
        self.aesthetic_head.load_state_dict(torch.load("sac+logos+ava1-l14-linearMSE.pth"))
        self.aesthetic_head.eval()

    def __call__(self, batch: dict) -> dict:
        """
        Process one batch of data. Ray automatically shards and transfers data to Actor.
        """
        images = []
        valid_indices = []
        
        # Preprocess images (CPU operation)
        for idx, img_bytes in enumerate(batch["jpg"]):
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img_tensor = self.preprocess(img)
                images.append(img_tensor)
                valid_indices.append(idx)
            except Exception:
                # Log bad image but don't interrupt
                continue
        
        if not images:
            return {"aesthetic_score": [], "clip_score": []}

        image_input = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            # 1. Extract features
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # 2. Compute aesthetic score
            aesthetic_scores = self.aesthetic_head(image_features).squeeze().cpu().numpy()
            
            # 3. Compute image-text match (assuming batch has text field)
            # text_tokens = self.tokenizer(batch["txt"]).to(self.device)
            # text_features = self.model.encode_text(text_tokens)
            #... compute cosine similarity
            
        # Return results (note alignment with original batch indices)
        return {"aesthetic_score": aesthetic_scores}

# Orchestrate Ray pipeline
ray.init()
ds = ray.data.read_webdataset("s3://raw-bucket/{00000..00099}.tar")

# map_batches automatically schedules GPU resources
# num_gpus=0_25 means one GPU can run 4 Actors concurrently, improving throughput
scored_ds = ds.map_batches(
    QualityScorer, 
    compute=ray.data.ActorPoolStrategy(size=8), 
    num_gpus=0_25, 
    batch_size=128
)

# Final filtering
filtered_ds = scored_ds.filter(lambda row: row["aesthetic_score"] > 4_5)
filtered_ds.write_webdataset("s3://clean-bucket/")
```

### 6.4 Pitfalls & Troubleshooting

When building billion-scale multimodal datasets, engineering teams often stumble on details. Here are lessons learned from painful experience:

* **Parquet Metadata Explosion**:
    * **Error**: Habitually reading Parquet files containing 2 billion rows directly in pandas.
    * **Consequence**: Out of memory (OOM), because pandas tries to load the entire index into memory even when only reading one column.
    * **Fix**: Use Polars or PySpark's lazy evaluation mode; or strictly split Parquet files by row count (e.g., 1 million rows) into smaller files to avoid processing single giant metadata files.

* **Insufficient WebDataset Shuffle**:
    * **Error**: During download, data written in domain order; during training, relying only on DataLoader's buffer shuffle (typically buffer of 10k).
    * **Consequence**: Model may see 100k e-commerce images consecutively, then 100k landscape images. Small buffer cannot break this "temporal correlation," causing violent training curve oscillation or even divergence.
    * **Fix**: Before writing WebDataset, must perform**Global Shuffle** on the URL list. Can use Spark's `orderBy(rand())`.

* **Accidentally Deleting Long-tail Data**:
    * **Error**: Chasing extreme aesthetic scores, deleting all images with Score < 4_5.
    * **Consequence**: Model becomes "narrow," only recognizing art photos and wallpapers, not real-world (possibly ugly) photos like medical images, street scenes, handwritten notes. Greatly reduces model generalization.
    * **Fix**: Use stratified sampling. Retain 5%-10% low-score data as "regularization," or establish whitelists for specific domains (e.g., OCR, charts) that bypass the aesthetic filter.

* **The Hidden Danger of Duplicate Data (Deduplication)**:
    * **Error**: Ignoring the large amount of duplicate images on the internet (e.g., Memes, viral news images).
    * **Consequence**: Model overfits specific samples, even "memorizing" training set images during generation, leading to serious copyright issues.
    * **Fix**: Must add**semantic deduplication** to the cleaning pipeline. Compute Embeddings for all images, use Faiss or MinHashLSH for clustering, and retain only one image per highly similar group.
