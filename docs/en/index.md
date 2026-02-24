# **"Data Engineering for Large Models: Architecture, Algorithms, and Project Practice"**

------

# Book Overview

- **Part 1: Infrastructure & Core Concepts** (Building the Data Foundation)
- **Part 2: Large-Scale Text Pre-training Engineering** (Cleaning, Deduplication & Quality Control)
- **Part 3: Multimodal Data Engineering** (Image-Text, Video & Audio)
- **Part 4: Alignment & Synthetic Data Engineering** (Instructions & Quality)
- **Part 5: Application-level Data Engineering** (RAG & Agent)
- **Part 6: Capstone Projects** (End-to-end Code Implementations)

------

# Detailed Outline

## Part 1: Infrastructure & Core Concepts

> **Goal:** Establish a Data-Centric AI mindset and set up a high-performance data processing environment.

### Chapter 1: Data Revolution in the LLM Era (From Data Ops to AI Ops)

- 1_1 **Insights from Scaling Laws:** Data quality > quantity — the paradigm shift from "big data" to "high-quality data."
- 1_2 **LLM Data Lifecycle:** Pre-training $\rightarrow$ SFT $\rightarrow$ RLHF $\rightarrow$ RAG.
- 1_3 **Challenges & Opportunities:** The interplay of heterogeneous multimodality, copyright compliance, and compute costs.

### Chapter 2: AI-Native Data Stack

- 2_1 **AI-Native Data Stack:**
  - Storage: Object storage (S3/MinIO) vs Data lakes (Iceberg/Hudi).
  - Compute: Spark vs **Ray Data** vs **Dask** — three distributed frameworks compared.
  - Vector Databases: Milvus / Qdrant / Weaviate / Pinecone selection and QPS vs Recall tradeoffs.
- 2_2 **Data Formats & I/O Optimization:**
  - Parquet vs JSONL vs WebDataset (multimodal scenarios).
  - Compression algorithms and read performance optimization, GPU training I/O bottleneck optimization.
- 2_3 **Data Version Control (DataOps):** Managing PB-scale datasets with DVC, LakeFS, and **Pachyderm**.

------

## Part 2: Large-Scale Text Pre-training Engineering

> **Goal:** Process massive unstructured text to build the model's linguistic cognitive foundation.

### Chapter 3: Data Acquisition (CommonCrawl Parsing & High-Concurrency Crawling)

- 3_1 **Deconstructing Open-source Datasets:** Deep analysis of Common Crawl, C4, RefinedWeb, and The Pile.
- 3_2 **High-performance Web Crawling:** Application of `Trafilatura` parsing library and distributed crawler architecture design.
- 3_3 **Specialized Data Acquisition:** Extraction strategies for code (GitHub), papers (ArXiv/S2ORC), and book data.

### Chapter 4: Cleaning & Quality Control

- 4_1 **Heuristic Filtering Rules:** Language identification (FastText), perplexity filtering, length and punctuation distribution.
- 4_2 **Large-scale Deduplication (Exact vs Fuzzy):**
  - **Exact Deduplication:** Hash methods for removing identical documents.
  - **Fuzzy Deduplication:** MinHash LSH algorithm principles and distributed implementation.
  - **Intra-document Deduplication:** Eliminating repeated paragraphs and navigation bars.
- 4_3 **Privacy Cleaning (PII Removal):** Using Presidio to identify and mask emails, IPs, phone numbers, and addresses.
- 4_4 **Benchmark Decontamination:** Ensuring training data doesn't contain GSM8K, MMLU test questions.
- 4_5 **Model-based Quality Scoring:** Using fastText/BERT for "textbook quality" scoring (following LLaMA 2 approach).

### Chapter 5: Tokenization, Serialization & Efficient Loading (DataLoader Optimization)

- 5_1 **Tokenizer Principles:** BPE, WordPiece, Unigram and **Byte-Level BPE** deep analysis.
- 5_2 **Efficient Vocabulary Construction:** Domain-specific vocabulary expansion and **LLaMA Chinese vocabulary extension engineering**.
- 5_3 **Data Mixing:** Dynamic sampling strategies and Curriculum Learning data arrangement.

------

## Part 3: Multimodal Data Engineering

> **Goal:** Process images, video, and audio to support training of GPT-4V/Sora-class models.

### Chapter 6: Image-Text Pair Processing

- 6_1 **Data Paradigms:** Image-text pairs (LAION-5B) vs Interleaved documents (OBELICS/MMC4).
- 6_2 **Image Acquisition & Preprocessing:**
  - `img2dataset` high-concurrency download in practice.
  - GPU-accelerated decoding and transforms (NVIDIA DALI).
- 6_3 **Multimodal Cleaning Pipeline:**
  - **Aesthetic Scoring:** Using CLIP-Score to filter high-aesthetic images.
  - **Image-text Alignment Filtering:** Removing mismatched samples.
  - **Safety Detection:** NSFW and watermark detection.

### Chapter 7: Recaptioning

- 7_1 **Limitations of Alt-text:** Why raw web descriptions are unusable.
- 7_2 **Synthetic Caption Factory:**
  - Using BLIP-2 / LLaVA / CogVLM to regenerate detailed captions.
  - **Prompt Strategies:** Controlling caption granularity (brief vs detailed).
- 7_3 **OCR Enhancement:** Extracting in-image text and fusing it into text descriptions (critical for document understanding).

### Chapter 8: Video & Audio Data

- 8_1 **Video Processing Pipeline:** Scene Detection and keyframe extraction strategies.
- 8_2 **Video Tokenization:** Video compression and discrete representation.
- 8_3 **Audio Alignment:** Large-scale ASR with Whisper and Force Alignment (timestamp alignment).

------

## Part 4: Alignment & Synthetic Data Engineering

> **Goal:** Make models follow instructions and break through the bottleneck of human data.

### Chapter 9: Instruction Fine-tuning Data (SFT Data)

- 9_1 **Prompt Engineering for Data Production:** Writing robust System Prompts.
- 9_2 **Automated Construction Methods:**
  - **Self-Instruct:** Leveraging strong models to generate instructions.
  - **Evol-Instruct:** Evolutionary strategies for instruction complexity.
- 9_3 **Chain-of-Thought (CoT) Data:** Constructing step-by-step reasoning samples.

### Chapter 10: Synthetic Data

- 10_1 **Textbook-quality Data (Textbooks Are All You Need):** Synthesizing high-quality domain knowledge.
- 10_2 **Code & Math Synthesis:**
  - **PoT (Program of Thought):** Generating code, executing it, and verifying data correctness via execution results.
- 10_3 **Multimodal Instruction Synthesis:** Using GPT-4o to construct complex image-based reasoning Q&A.

### Chapter 11: Human Preference Data (RLHF/DPO)

- 11_1 **Preference Data Format:** Constructing chosen vs rejected sample pairs.
- 11_2 **Annotation Platforms & Quality Control:** Crowdsourcing management and IAA (Inter-Annotator Agreement) analysis.
- 11_3 **RLAIF (AI Feedback):** Using LLMs to replace human preference scoring.

------

## Part 5: Application-level Data Engineering (RAG & Agent)

> **Goal:** Enterprise-oriented solutions for external knowledge base parsing and retrieval.

### Chapter 12: RAG Data Pipeline

- 12_1 **Deep Document Parsing:**
  - Complex PDF processing: table reconstruction, multi-column recognition (`Unstructured`, `LlamaParse`).
- 12_2 **Chunking Strategies:** Semantic chunking, recursive chunking, and Parent-Child Indexing.
- 12_3 **Vectorization & Storage:** Embedding model fine-tuning and vector database optimization.

### Chapter 13: Multimodal RAG

- 13_1 **Cross-modal Retrieval:** Using CLIP/SigLIP for "text-to-image" and "image-to-text" search.
- 13_2 **ColPali Architecture in Practice:** Vision-language model based document retrieval (skip OCR, directly understand document images).

------

## Part 6: Capstone Projects

> **Goal:** Through 5 end-to-end projects, integrate all technical topics from the book with runnable code repositories.

### Project 1: Building a "Mini-C4" Pre-training Set

- **Scenario:** From Common Crawl raw data (WARC) to high-quality Parquet data.
- **Core Technologies:** Trafilatura parsing, Spark/Ray distributed MinHash deduplication, KenLM quality filtering.
- **Output:** Cleaned plain text corpus and processing pipeline.

### Project 2: Domain Expert SFT (Legal/Medical)

- **Scenario:** Building industry expert fine-tuning data from unstructured PDF documents.
- **Core Technologies:** Self-Instruct instruction generation, CoT reasoning enhancement, data diversity balancing.
- **Output:** `domain_expert.jsonl` instruction fine-tuning dataset.

### Project 3: Building a LLaVA Multimodal Instruction Set

- **Scenario:** Training a multimodal model that can understand images.
- **Core Technologies:** Using GPT-4o API for multi-turn image-text dialogues, Bounding Box data alignment, multi-image interleaved format processing.
- **Output:** Image-text dataset with visual instructions.

### Project 4: Synthetic Math/Code Textbook

- **Scenario:** Improving small model logical reasoning capabilities.
- **Core Technologies:** Evol-Instruct evolutionary strategies, Python code execution sandbox verification, PoT data formatting.
- **Output:** Verified high-quality synthetic reasoning dataset.

### Project 5: Multimodal RAG Financial Report Assistant

- **Scenario:** Retrieving and answering questions about annual reports with complex charts.
- **Core Technologies:** PDF table and chart parsing, multi-route recall (hybrid retrieval), ColPali visual retrieval.
- **Output:** A RAG knowledge base system supporting chart Q&A.

------

## Appendix

- **Appendix A:** Common Tools Quick Reference (Hugging Face Datasets, LangChain, Ray Data).
- **Appendix B:** Data Compliance Checklist (Copyright, GDPR, robots.txt).
- **Appendix C:** Compute Cost Estimation (GPU/CPU consumption reference for different data processing scales).
