## Chapter 6: Image-Text Pairs Data Processing

### Chapter Summary

In the journey of building next-generation Foundation Models, the focus of data engineering has shifted from mere text cleaning to capturing, aligning, and reconstructing multi-dimensional signals from the physical world. If language model data engineering is about "denoising," then multimodal data engineering is about "association" and "alignment." With the emergence of GPT-4V, Gemini, and Sora, we have come to realize that single-modality data can no longer satisfy models' appetite for understanding the world.

This chapter provides an in-depth analysis of the complete engineering pipeline for building billion-scale multimodal datasets. This is far more than writing a few scripts to download images—it is a comprehensive campaign involving network protocols, distributed storage, heterogeneous computing, and aesthetic evaluation. We will explore the underlying logic of data paradigms, analyze how to leverage distributed computing frameworks to solve the high-concurrency acquisition challenges of massive images, and use GPU hardware acceleration to break through the I/O bottleneck of image preprocessing. Furthermore, we will build an automated cleaning pipeline based on semantics and aesthetics to ensure that data fed into the model is both relevant and safe.

**Learning Objectives**:
* Understand the training benefits and engineering challenges of LAION-5B (image-text pairs) and OBELICS (interleaved documents) paradigms, and master the design of hybrid data strategies.
* Be able to write distributed downloaders based on PySpark and Ray Data, handle DNS bottlenecks and long-tail latency, and achieve throughput of 10,000+ img/s.
* Master NVIDIA DALI pipeline design, solve CPU decoding bottlenecks, and optimize data loading using GPU Direct principles.
* Build a multi-stage cleaning funnel that includes CLIP semantic filtering, aesthetic scoring, and safety detection, and master threshold tuning strategies for different business scenarios.

**Scenario Introduction**:
> "Imagine this scenario: Our crawler team has just extracted 2 billion raw URLs from Common Crawl, stored in thousands of Parquet files. Your task is to transform this data into a high-quality dataset suitable for GPT-4V pre-training within two weeks. When you try to download using the traditional Python requests library on a single machine, you find the estimated time is as high as 15 years—a classic network I/O blocking problem. Worse, preliminary sampling shows that 30% of downloaded images are e-commerce ads (full of noise), 15% have severe watermarks, and there is even serious NSFW content. If we use this data directly, we will not only waste millions of dollars in compute, but the trained model may face legal risks due to generating objectionable content. We need an industrial-grade, high-throughput, intelligent data engineering solution to meet this challenge."

### 6.1 Data Paradigms: Image-Text Pairs (LAION-5B) vs Interleaved Documents (OBELICS/MMC4)

Before designing the data pipeline, our first responsibility is to clarify the data organization format. This is not only about storage structure, but also directly determines the training objective and emergent capabilities of downstream models. Different data forms are essentially different abstractions of "how knowledge exists in the world."

#### 6.1.1 Core Concepts and Principles

**Image-Text Pairs**
are the foundation of multimodal learning, represented by CLIP, ALIGN, and LAION-5B.
* **Theoretical Analysis**: This paradigm assumes strong semantic association between image $I$ and text $T$, and this association is independent and atomic. The training objective is typically to maximize the cosine similarity of $I$ and $T$ in a shared embedding space (Contrastive Learning). Its advantage lies in extremely high "signal-to-noise ratio" refinement potential—through contrastive learning, the model learns direct mapping between objects and vocabulary.
* **Engineering Perspective**: Data structure is simple, typically represented as flattened records of `(url, caption, metadata)`. This data is extremely easy to shard and randomly shuffle. During training, since samples are independent, we can easily implement Global Batch Shuffling to improve contrastive learning effectiveness.

**Interleaved Image-Text Documents**
are the key fuel for next-generation multimodal large models (such as Flamingo, GPT-4V, MM1), represented by OBELICS and MMC4.
* **Theoretical Analysis**: This paradigm preserves the original DOM structure order of web pages, with data presented as sequences of `<text>, <image>, <text>...`. This forces the model to learn "multimodal context dependency" (Multimodal In-Context Learning). For example, in a "how to make a cake" web page, the relationship between Image 1 (ingredients) and Image 5 (finished product), and their logical connection with surrounding text, cannot be provided by image-text pairs. It simulates the cognitive process of humans reading illustrated books.
* **Engineering Perspective**: The data pipeline is extremely complex. Since individual samples (documents) have variable length and may contain multiple images, batch assembly becomes difficult. Traditional Collators require complex padding strategies. Additionally, document integrity must be carefully maintained during cleaning—arbitrarily deleting a low-quality image may break context logic and cause the model to learn incorrect referential relationships.

#### 6.1.2 Architecture Decision: Comparison Table

With limited resources, how do we weigh these two data paradigms? This is not a simple binary choice, but involves deep trade-offs among model architecture, training cost, and final application scenarios.

In early multimodal research (before 2021), the industry widely believed that as long as data volume was sufficient (e.g., 400M pairs for CLIP), models could learn everything. However, with the emergence of GPT-4V, we found that models trained only on image-text pairs, while able to accurately identify "this is a cat," cannot answer "what might this cat in the image do," because they lack logical reasoning context. Conversely, while interleaved documents are rich in logic, the data is sparse and processing cost is extremely high.

The table below compares the core differences between the two paradigms at the engineering implementation level, aimed at helping architects make technical choices based on actual needs:

| Dimension | Image-Text Pairs (LAION-style) | Interleaved Documents (OBELICS-style) | In-depth Analysis & Recommendation |
| :--- | :--- | :--- | :--- |
| **Training Objective** | Contrastive Learning (CLIP), Text-to-Image (Stable Diffusion) | Next-Token Prediction, Multimodal Dialogue (GPT-4V) | **Hybrid strategy is King**. Research shows that training visual encoders only with interleaved documents is inefficient (images not dense enough), while using only image-text pairs lacks reasoning capability. Recommend Curriculum Learning strategy. |
| **Data Source Parsing** | Simple: only need to extract `<img>` tags and Alt-text | Complex: need to parse DOM tree, filter ads and sidebars, preserve main content logic | **Engineering complexity warning**. Building interleaved documents requires handling extremely complex HTML rendering logic. Recommend initially using Common Crawl WET files, or directly using OBELICS open-source set for augmentation—don't try to clean the entire internet from scratch. |
| **Storage Cost** | Medium: metadata is CSV/Parquet only, images stored separately | High: need to save document topology, recommend WebDataset or TFRecord encapsulation | **I/O performance bottleneck**. For interleaved documents, must use sharded storage to avoid small file fragmentation. Reading requires pre-loading entire documents, placing higher demands on memory bandwidth. |
| **Cleaning Challenges** | Single-point cleaning: each image judged independently, easy to parallelize | Context cleaning: must consider text coherence and image quality simultaneously, cleaning logic coupled | **Strategy selection**. When processing interleaved documents, if an image is judged NSFW, recommend replacing with special `<BLOCKED_IMAGE>` token rather than deleting, to maintain Positional Embedding accuracy. |
| **Model Benefits** | Strong visual-semantic alignment, strong Zero-shot classification | Powerful Few-shot Learning, supports multi-turn dialogue and logical reasoning | **Business-oriented**. If the scenario is "image search," image-text pairs suffice; if it involves complex document understanding (e.g., research report analysis, long-form story generation), interleaved documents must be introduced. |

> **Tips:**
> In cutting-edge research like MM1 and Idefics2, best practice is not either-or but proportioning. Typically recommend using **80% image-text pairs** in the early pre-training phase to establish solid visual-language mapping foundation, while mixing **20% interleaved documents**; in the late pre-training phase (Annealing Phase), significantly increase the proportion of interleaved documents to stimulate model long-context reasoning capability. This "foundation first, logic later" strategy maximizes compute utilization.

### 6.2 Image Acquisition and Preprocessing

Once the data manifest is determined, the next step is to build a high-throughput download and preprocessing pipeline. This is a typical I/O-intensive task, with main bottlenecks in network bandwidth, DNS resolution latency, and disk writes of massive small files.

#### 6.2.1 img2dataset High-Concurrency Download in Practice

`img2dataset` is currently the community-recognized best practice tool. It is not just a download script, but a distributed data processing framework based on MapReduce principles.

Why do we need specialized tools instead of writing a simple `requests.get` loop? Because the internet environment is extremely harsh. Links expire (Link Rot), servers rate-limit, DNS times out. When processing billions of URLs, any tiny long-tail latency is amplified into weeks of time cost.

**Core Principles**:
1.  **Sharding**: Split 1 billion URLs into tens of thousands of small tasks (Shards). This is the foundation of distributed computing.
2.  **Async I/O**: Use Python's aiohttp or Go coroutines to concurrently initiate hundreds of network requests on a single core, masking network latency.
3.  **Streaming Archival**: Downloaded images don't hit disk; they are directly assembled into tar packages (WebDataset format) in memory, then streamed to object storage (S3/HDFS). This avoids the file system inode exhaustion problem from creating millions of small files in one directory—a pitfall newcomers often encounter.

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
    For 10Gbps NIC nodes, typically recommend total_concurrency around 1000.
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
        resize_mode="keep_ratio",   # Maintain aspect ratio, black padding or center crop
        skip_reencode=True,         # If original is JPG and size acceptable, store directly, saves CPU
        output_folder=output_dir,
        output_format="webdataset", # Force WebDataset format
        input_format="parquet",
        url_col="url",
        caption_col="caption",
        enable_wandb=True,          # Strongly recommended for monitoring download rate and error rate
        number_sample_per_shard=10000, # 10k images per tar, ~200-300MB, for easy transfer
        distributor="pyspark",      # Use Spark for task distribution
        save_additional_columns=["similarity", "hash"], # Preserve original metadata
        timeout=10                  # Shorter timeout for fast failure, long-tail requests not worth waiting
    )

if __name__ == "__main__":
    # Initialize Spark Session (usually handled by spark-submit, but explicit for IDE debugging)
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("Img2Dataset-Production") \
        .config("spark.executor.memory", "8g") \
        .config("spark.task.maxFailures", "10") \
        .getOrCreate()
    
    run_distributed_download()
```

**Pro Tips**:
* **DNS Caching**: Under high concurrency, DNS resolution can become a bottleneck or even get blocked by providers. Recommend deploying local DNS cache (e.g., dnsmasq) on worker nodes, or maintaining a domain-to-IP mapping table in code.
* **User-Agent Rotation**: Though an "open" secret, rotating User-Agent can reduce 403 Forbidden rates.
* **Error Handling**: Monitor success_rate in WandB dashboard. If below 80%, usually means URL list is severely stale or your IP pool is contaminated.

#### 6.2.2 Visual Preprocessing Pitfalls: Cropping and Semantic Alignment

After solving the challenge of acquiring massive data (Getting bytes), we immediately face the second challenge: data usability. Raw internet images have wildly varying aspect ratios, while models typically require fixed resolution input (e.g., 224x224 or 512x512).

Many novice engineering solutions habitually use simple brute-force random preprocessing to unify dimensions, but this is often the root of the model's "invisible performance ceiling." We must not only focus on "fitting the image in," but also on "what exactly is being put in."



![Figure 6-1: Cropping and Semantic Alignment in Image Preprocessing](../../images/part3/图6_1_图片预处理中裁剪与语义对齐问题.png)
*Figure 6-1: Cropping and Semantic Alignment in Image Preprocessing*

* **Bad Case (Left image - Cost of Naive Cropping)**:
    Traditional `RandomCrop` or `CenterCrop` has no awareness of composition. When processing a portrait photo in vertical composition, center cropping easily cuts off key features (such as the head), leaving only the torso. At this point, if the text label is still "a smiling man," the model is forced to establish incorrect mapping (mistaking torso features for "smiling person"), causing the trained model to produce severe visual hallucinations.

* **Good Case (Right image - Semantic Completeness)**:
    High-quality data engineering pursues "image-text consistency."
    1.  **Smart Resize**: Prefer `Resize with Padding` (maintain aspect ratio, black/white padding) to preserve complete visual subject. Though this introduces invalid pixels, it guarantees semantic completeness.
    2.  **Aspect Ratio Bucketing**: An advanced technique commonly used by SDXL and Midjourney. Group images with similar aspect ratios into the same batch for training, avoiding cropping while reducing padding waste.
    3.  **Recaptioning**: As detailed in Chapter 7 below, using VLM to generate high-density descriptions allows text to precisely correspond to on-screen details (e.g., sign text, background objects), maximizing data training value.

#### 6.2.3 GPU-Accelerated Decoding and Transformation (NVIDIA DALI)

In the deep learning model training phase, most researchers and developers focus their attention on model architecture design, hyperparameter tuning, loss function improvement—modules that directly affect model accuracy—yet easily overlook the data loading (DataLoader) foundation. In reality, it often becomes the "invisible performance killer" that constrains training efficiency, even preventing full utilization of high-end GPU compute and causing serious hardware waste.

To understand this pain point, we must first clarify the complete logic of the deep learning training flow: model training's core compute relies on GPU's massive parallel computing capability; GPU can efficiently process massive tensor operations and complete backpropagation and parameter updates. But before data enters the GPU, it must go through a series of preprocessing operations, the most basic and time-consuming of which is image decoding and resizing. In traditional PyTorch training flow, these critical preprocessing operations are entirely done on CPU, forming the contradiction between "CPU preprocessing bottleneck" and "GPU compute redundancy."

Specifically, the traditional PyTorch Dataset workflow is: first read image files (mostly JPEG) from disk via CPU, then CPU completes JPEG decoding—this process requires Huffman decoding, inverse discrete cosine transform (IDCT) and other complex computations on compressed image binary data, a typical CPU-intensive task. After decoding, CPU executes Resize, normalization, color space conversion and other preprocessing, finally copying the processed image tensor to GPU for model training.

More critically, CPU architecture is better suited for serial computation and logic control; its parallel computing capability is far inferior to GPU. Yet image preprocessing operations like decoding and Resize are inherently highly parallelizable and can improve efficiency through multi-threading or multi-core parallelism. But traditional PyTorch Dataset, even with DataLoader's num_workers to improve CPU parallelism, can hardly break through CPU's own compute ceiling—especially when the training dataset is massive (millions of images) and single image resolution is high (1080p+), CPU preprocessing speed will severely lag behind GPU training speed, causing GPU to frequently idle "waiting for data," with significantly reduced GPU utilization, ultimately dragging down the entire training efficiency. This is why data loading is called the "neglected performance killer."

Addressing this core pain point, NVIDIA introduced DALI (Data Loading Library), a GPU-accelerated data preprocessing library optimized for deep learning training. Its core goal is to migrate CPU-intensive operations like image decoding and resizing to GPU for parallel execution, breaking the data loading performance bottleneck and enabling full GPU utilization.



![Figure 6-2: Data Decoding and Transformation With vs Without DALI](../../images/part3/图6_2_使用DALI与不使用DALI下数据解码与变换的区别.png)
*Figure 6-2: Data Decoding and Transformation With vs Without DALI*

**Code Walkthrough: High-Performance DALI Pipeline**

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
    # Using index_paths is necessary, otherwise initialization requires traversing entire tar, extremely slow [5]
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
        read_ahead=True          # Enable read-ahead
    )

    # Step 2: GPU Decoding (core acceleration point)
    # device="mixed" means input in Host memory, output in Device memory
    # output_type=types.RGB handles color space conversion automatically
    images = fn.decoders.image(
        jpegs,
        device="mixed",
        output_type=types.RGB,
        # Error handling for corrupted images
        # In production, never let a single bad image crash training
    )

    # Step 3: GPU transformation pipeline
    # resize: maintain aspect ratio scaling
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

# Benchmark test
# On A100, this pipeline typically achieves 3000-5000 FPS, 5-10x CPU Loader
```

### 6.3 Multimodal Cleaning Pipeline

Massive data comes with massive noise. In raw LAION-5B data, truly high-quality samples may be less than 10%. We need to establish a multi-stage cleaning funnel to improve data density while minimizing loss of data diversity. So-called "data cleaning" is essentially **Data Diet**—feeding the model less but better.

#### 6.3.1 Architecture Design: Ray Data Distributed Cleaning

Why choose Ray over Spark for the cleaning phase? Because cleaning is no longer simple ETL but contains large amounts of **deep learning inference (Model Inference)**. Compared to Spark's MapReduce paradigm, Ray provides more flexible Actor mechanism, allowing us to keep GPU models (e.g., CLIP, Safety Checker) resident, avoiding the huge overhead of reloading multi-GB models for each small batch.

Ray Data is suitable for this mixed workload with both CPU-intensive (decompression, hashing, Regex) and GPU-intensive (CLIP Embedding inference) tasks. Below is a typical three-stage pipeline design:
* **Stage 1 (CPU)**: Fast filtering. Directly remove samples with insufficient resolution (<256px), too short text, non-English (if only training English model), or abnormal aspect ratio.
* **Stage 2 (GPU)**: Deep feature extraction. Use CLIP model to generate Embeddings, compute image-text similarity and aesthetic score based on Embeddings.
* **Stage 3 (CPU/Mixed)**: Logic evaluation and deduplication. Apply final threshold cutoff based on safety (NSFW), aesthetic score, and image-text relevance, and perform semantic deduplication.



**Data Flow Diagram**

![Figure 6-3: Ray Data Distributed Cleaning Data Flow](../../images/part3/图6_3_Ray_Data分布式清洗数据流向图.png)
*Figure 6-3: Ray Data Distributed Cleaning Data Flow*

#### 6.3.2 Core Algorithm Implementation

Cleaning is not just deletion—it is also quantization of data value. We need multi-dimensional metrics to measure the "gold content" of an image and its corresponding text.

1.  **Aesthetics Scoring**
    * **Principle**: Datasets are filled with invoices, screenshots, blurry surveillance footage—these are useless for generating beautiful images. Typically use LAION-Aesthetics Predictor.
    * **Technical Details**: A simple MLP (multi-layer perceptron) with CLIP Image Embedding as input, outputting 1-10 score. Training data from AVA dataset (professional photographer human ratings).
    * **Recommended Threshold**: For base pre-training, keep Score > 4_5; for fine-tuning high-quality generation models (SFT stage), recommend Score > 6_0 or even 6_5.

2.  **Image-Text Alignment Filtering**
    * **Principle**: Many Alt-texts are SEO garbage word stacking or filenames ("DSC_001.jpg"), unrelated to image content.
    * **Technical Details**: Compute cosine similarity (Dot Product) of CLIP Image Embedding and Text Embedding.
    * **Pitfall**: Different CLIP versions (e.g., OpenAI ViT-L/14 vs OpenCLIP ViT-G/14) have different embedding space distributions—scores are not directly comparable. Must recalibrate thresholds for specific model. Common approach: compute similarity distribution over entire dataset, then keep Top 50% or Top 70%.

3.  **Safety Detection (Safety & Watermark)**
    * **Principle**: Must remove pornographic, violent, and prominently watermarked images.
    * **Strategy**: Use specially trained classifier heads (also based on CLIP Embedding) for NSFW and watermark detection. For watermark detection: if target is training generation models (e.g., SDXL), must be extremely strict (Recall priority) because generation models easily overfit watermark features; if target is training understanding models (e.g., GPT-4V), can relax somewhat, because understanding models need to recognize "there is a watermark in the image."

**Code Implementation: Ray Data Cleaning Operator**

```python
import ray
import torch
import open_clip
import numpy as np
from PIL import Image
import io

# Define Ray Actor class to ensure model loaded only once
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
        Process a batch of data. Ray will automatically partition and transfer data to Actor.
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
            
        # Return results (must align with original batch indices)
        return {"aesthetic_score": aesthetic_scores}

# Orchestrate Ray pipeline
ray.init()
ds = ray.data.read_webdataset("s3://raw-bucket/{00000..00099}.tar")

# map_batches will automatically schedule GPU resources
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

In building billion-scale multimodal datasets, engineering teams often stumble on details. Here are several lessons learned:

* **Parquet Metadata Explosion**:
    * **Error**: Habitually reading Parquet files with 2 billion rows directly in pandas.
    * **Consequence**: Memory overflow (OOM), because pandas tries to load entire index into memory even when reading just one column.
    * **Fix**: Use Polars or PySpark lazy evaluation mode; or strictly split Parquet files by row count (e.g., 1M rows) into small files to avoid processing single giant metadata files.

* **WebDataset Insufficient Shuffle**:
    * **Error**: Data written in domain order during download, training relies only on DataLoader buffer shuffle (typically buffer only 10k).
    * **Consequence**: Model may see 100k e-commerce images consecutively, then 100k landscape images consecutively. Small buffer cannot break this "temporal correlation," causing violent training curve oscillation or even divergence.
    * **Fix**: Must perform **Global Shuffle** on URL list before writing WebDataset. Can use Spark's `orderBy(rand())`.

* **Accidentally Deleting Long-Tail Data**:
    * **Error**: For pursuit of extreme aesthetic score, deleting all images with Score < 4_5.
    * **Consequence**: Model becomes "specialized"—only recognizes art photos and wallpapers, not real-world (possibly ugly) photos like medical imaging, street views, handwritten notes. Greatly reduces model generalization.
    * **Fix**: Use stratified sampling strategy. Keep 5%-10% low-score data as "regularization," or set special whitelists for specific domains (e.g., OCR, charts) that bypass aesthetic filter.

* **Duplicate Data Hazards (Deduplication)**:
    * **Error**: Ignoring massive duplicate images on the internet (e.g., Memes, viral news images).
    * **Consequence**: Model overfits specific samples, even "memorizing" training set images during generation, causing serious copyright risks.
    * **Fix**: Must add **semantic deduplication** to cleaning pipeline. Compute Embeddings for all images, use Faiss or MinHashLSH for clustering, keep only one per highly similar image group.
