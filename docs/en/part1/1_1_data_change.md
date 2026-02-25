# Chapter 1: Data Transformation in the Era of Large Language Models (From Data Ops to AI Ops)

---

## Chapter Summary

The rise of Large Language Models (LLMs) has reshaped the paradigm of artificial intelligence. However, innovation in model architecture has converged, and what truly determines the upper limit of model capability is **data quality**. This chapter establishes the concept of "data as core asset" from the perspective of Scaling Laws, systematically introduces the full lifecycle of LLM data, and explores practical challenges such as heterogeneous multimodal data, copyright compliance, and compute cost.

---

## Scenario Introduction

You are the data lead at an AI startup. The team has just spent three months crawling 50TB of Chinese text from the public web and is confidently starting to pre-train a 7B parameter base model.

However, two weeks into training, the loss curve suddenly "flattens" at a certain point, and the model output is filled with ad copy, repetitive SEO junk, and even recites user agreements from certain websites. In the post-mortem meeting, a senior engineer raises a sharp question: "After spending 1 million on compute for training, are we building a language model, or a compressed index of internet garbage?"

This scenario is not fabricated. OpenAI, Google, Meta, and other top labs have already reached consensus: in an era of converging model architectures, data quality is the core variable that determines the intelligence ceiling of models.

---

## 1.1 Implications of Scaling Laws: Paradigm Shift from "Big Data" to "High-Quality Data"

### 1.1.1 What Are Scaling Laws?

In 2020, OpenAI published the landmark paper *Scaling Laws for Neural Language Models*, revealing a simple yet profound regularity: model performance (measured by Loss) follows a power law relationship with three core factors—model parameter count $N$, dataset size $D$, and compute $C$.

$$
L(N, D, C) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

Where $L$ is the model's cross-entropy loss, $N_c, D_c, C_c$ are constants, and $\alpha$ are power law exponents. In plain terms: to make a model smarter, you either increase model scale, add more data, or invest more compute. These three factors constrain each other, forming the "impossible triangle" of the LLM era.

This discovery sent shockwaves through academia and industry. Before this, the development of deep learning was more like "alchemy"—researchers relied on intuition and experience to tune model architectures, hoping for accidental breakthroughs. The emergence of Scaling Laws shifted large model training from art to engineering science for the first time, enabling companies to plan resource investment and expected outcomes based on precise mathematical models.

### 1.1.2 The "Hidden Variable" of Data Quality

However, the original Scaling Laws had a critical blind spot: it assumed that all data "quality" was uniform. This assumption clearly does not hold in reality. Text quality on the internet varies widely, from carefully written academic papers to comments riddled with typos, and the value difference between data can span orders of magnitude.

In 2022, DeepMind's Chinchilla paper shattered this assumption. The research team conducted a large-scale experiment: under the same compute budget, compare the effects of different model sizes and data volume ratios. The results shocked the industry—Chinchilla (70B parameters), with only one-quarter the parameters of Gopher, surpassed the 280B parameter Gopher on almost all evaluation tasks by using four times the high-quality training data (1.4T tokens).

| Model | Parameters | Training Tokens | Final Performance |
|------|--------|---------------|----------|
| Gopher | 280B | 300B tokens | Baseline |
| **Chinchilla** | **70B** | **1.4T tokens** | **Outperforms Gopher** |

This research revealed a long-neglected fact: the industry had previously overfitted on model scale while undervaluing data volume. The Chinchilla paper recommended an "optimal ratio": for each additional parameter, approximately 20 tokens of training data should be allocated. This means training a 7B model theoretically requires about 140B tokens of high-quality corpus—a number far beyond what many teams initially expected.

### 1.1.3 Quality vs. Quantity: The Extreme Experiment of the Phi Series

If Chinchilla proved that "data volume was underestimated," then Microsoft's Phi series demonstrated an even more radical view: data quality can override the scaling laws.

In 2023, Microsoft Research released the Phi-1 model. This is a "small" model with only 1.3B parameters, trained on merely 7B tokens—a stark contrast to the mainstream approach of hundreds of billions or even trillions of tokens at the time. Yet this seemingly "malnourished" model surpassed competitors with ten times its parameters on code generation tasks.

Phi-1's secret weapon was not massive crawled data, but carefully designed synthetic data with pedagogical value. The researchers used GPT-4 to generate large amounts of structured, step-by-step programming tutorials, from basic syntax to advanced algorithms, forming a complete "artificial textbook." These synthetic data had advantages that real web data could hardly match: no noise, no errors, clear logic, and progressive difficulty. Small models trained on these "artificial textbooks" showed remarkable capability in solving real programming problems.

![Figure 1-1: Data Quality Comparison](../../images/part1/图1_1_数据质量对比.png)

*Figure 1-1: Relationship between Data Quality and Model Performance — Low-quality data (40% noise) after cleansing becomes high-quality data (5% noise)*

The success of the Phi series sparked heated discussion in the industry about "synthetic data." If carefully designed synthetic data could produce such excellent results, does the traditional "crawl-clean-train" paradigm need to be overturned? We will explore synthetic data methodology and best practices in detail in Chapter 7.

### 1.1.4 The Core Mission of Data Engineers

Synthesizing the above research, we can distill the core mission of data engineers in the LLM era. Traditional wisdom held that "more data is better"; the new paradigm emphasizes that data quality determines the performance ceiling, and noisy data is not only unhelpful but harmful. Past belief was that "model architecture is the core competitive advantage"; today architecture has converged, and data has become the key differentiator. The old view that "cleaning is a minor preprocessing step" is outdated—data cleaning is now seen as the foundation of successful model training. Similarly, the traditional belief that "human annotation is irreplaceable" has been broken; high-quality synthetic data can surpass real data in certain scenarios.

It is important to emphasize that "quality over quantity" should not be misinterpreted as "only quality, no quantity." Scaling Laws still hold—the priority under the same compute budget should be investing in data quality. An extreme example: a model trained on 100 perfect data points can never surpass a model trained on 1T high-quality data points. Quality and quantity are not opposed; they require finding the optimal balance under practical constraints.

---

## 1.2 Full Lifecycle of LLM Data

From "birth" to "deployment," a large language model undergoes multiple training stages, each with distinctly different data requirements. Understanding this full lifecycle is the first step toward becoming a qualified data engineer.

### 1.2.1 Four-Stage Paradigm: Pre-train → SFT → RLHF → RAG

![Figure 1-2: LLM Data Lifecycle](../../images/part1/图1_2_LLM数据生命周期.png)

*Figure 1-2: LLM Training Data Pipeline — From TB-level pre-training to KB-level RAG, data volume decreases while quality requirements increase*

**Pre-training** is the starting point of a large model's life. At this stage, the model learns the essence of language from massive amounts of unlabeled text—grammatical structure, commonsense knowledge, and how the world works. Pre-training data scale is typically at the TB level, containing trillions of tokens, with sources from web pages, books, code, academic papers, and other text types. The core challenges at this stage are deduplication, denoising, quality filtering, and diversity balance. Typical pre-training datasets include Common Crawl, The Pile, and RefinedWeb.

**Supervised Fine-Tuning (SFT)** is the critical period when the model learns to "follow instructions." Although pre-trained models possess strong language understanding capability, they do not know how to interact effectively with humans. The SFT stage uses paired instruction-response data to teach the model to understand user intent and provide helpful responses. Data scale at this stage drops to GB level, containing hundreds of thousands to millions of carefully constructed dialogue samples. Key challenges include ensuring instruction diversity, response quality, and format standardization. Alpaca, ShareGPT, and OpenAssistant are commonly used datasets for this stage.

**Reinforcement Learning from Human Feedback (RLHF/DPO)** aims to make model output more "aligned" with human preferences. Alignment means making the model safer (not producing harmful content), more helpful (actually solving user problems), and more honest (not fabricating facts). This stage uses preference comparison data where human annotators choose the better of two candidate responses. Data scale further shrinks from hundreds of thousands to tens of thousands of samples, but annotation quality requirements are extremely high. Inter-Annotator Agreement, accuracy of preference signals, and coverage of sample distribution are all factors that must be carefully controlled.

**Retrieval-Augmented Generation (RAG)** addresses model knowledge updates and hallucination. No matter how thorough pre-training is, a model's knowledge has a cutoff date and cannot completely avoid "sounding authoritative while talking nonsense." RAG injects external information such as enterprise knowledge bases and latest documents into the generation process by letting the model "consult external sources." Data sources at this stage are typically enterprise-private, including PDF documents, database records, web content, and other structured or semi-structured data. Key challenges include document parsing, semantic chunking, vectorization, and retrieval accuracy optimization.

### 1.2.2 The "Funnel Model" of Data Flow

Another perspective on understanding the data lifecycle is the "funnel model." From raw web data to final high-quality corpus usable for training, data volume undergoes dramatic reduction.

![Figure 1-3: Data Funnel Model](../../images/part1/图1_3_数据漏斗模型.png)

*Figure 1-3: Data filtration funnel — From 100PB raw data to 10GB SFT data, retention rate only 0.00001%*

For a typical pre-training data processing workflow: assume we obtain 100PB of raw web data from Common Crawl. After URL deduplication and exact-content deduplication, data volume may drop to 30PB (30% retention). Next, language identification and quality filtering—removing non-target languages, pornographic/violent content, extremely short text, garbled text—further reduces data to 5PB (5% retention). Then finer quality assessment using perplexity, repetition rate, information density, and other metrics yields final pre-training corpus of perhaps only 1PB (1% retention). And if SFT data needs to be extracted or constructed from this, the final output may be only 10GB—relative to raw data, retention rate as low as one hundred-thousandth.

In practical engineering, understanding this ratio is crucial. It directly affects planning of crawl scale, storage cost budgeting, and processing pipeline design. Many teams underestimated this "attrition rate" at project start, leading to insufficient data reserves and mid-project rework.

---

## 1.3 Challenges and Opportunities: The Game of Heterogeneous Multimodal, Copyright Compliance, and Compute Cost

### 1.3.1 Challenge One: Complexity of Processing Heterogeneous Multimodal Data

With the emergence of multimodal models like GPT-4V, Gemini, and Sora, data engineering complexity has increased exponentially. In the pure text era, all data was UTF-8 encoded strings; processing toolchains were mature and standardized. In the multimodal era, data formats have exploded: images have JPEG, PNG, WebP; video has MP4, AVI, MKV; audio has WAV, MP3, FLAC; documents have PDF, Word, HTML. Each format has its own parsing methods and quality assessment standards.

| Dimension | Pure Text Era | Multimodal Era |
|------|-----------|-----------|
| **Data Format** | Unified UTF-8 text | Multiple formats: images, video, audio, documents, web pages |
| **Storage Requirements** | TB level | PB level (images/video dominate) |
| **Alignment Difficulty** | None (pure text needs no alignment) | Very high (image-text alignment, audio-video sync, cross-modal semantic consistency) |
| **Cleaning Toolchain** | Mature (FastText, KenLM) | Fragmented (each modality has independent toolchain) |
| **Quality Assessment** | Perplexity, deduplication rate | Aesthetic score, CLIP-Score, OCR accuracy, ASR error rate, etc. |

Even more challenging is the cross-modal alignment problem. An image paired with a caption—how to ensure the text accurately describes the image content? How do voice and visuals in a video stay precisely synchronized? These alignment problems did not exist in the pure text era but are core challenges in multimodal data engineering.

![Figure 1-4: Multimodal Processing Complexity](../../images/part1/图1_4_多模态处理复杂度.png)

*Figure 1-4: Multimodal Data Processing Complexity — Line thickness indicates processing difficulty, video (5/5) and PDF (4/5) are most complex*

From an engineering practice perspective, multimodal data processing requires modular design—each modality processed independently before cross-modal alignment. At the same time, priority should be given to unified data format standards such as WebDataset, TFDS, etc., to reduce downstream processing complexity. Establishing modality-aware quality assessment systems is also critical: images need aesthetic scores and CLIP similarity, speech needs ASR accuracy and speaker separation quality, PDFs need OCR accuracy and layout analysis precision.

### 1.3.2 Challenge Two: The Gray Zone of Copyright Compliance

Copyright issues with large model training data have evolved from technical discussion to legal battlefield. In 2023, Getty Images sued Stability AI, alleging unauthorized use of millions of copyrighted images to train Stable Diffusion. In 2024, *The New York Times* sued OpenAI and Microsoft, alleging GPT models infringed copyright. Regulatory agencies in multiple countries have begun requiring AI companies to disclose training data sources.

The outcomes of these lawsuits will profoundly affect the development direction of the AI industry, but for data engineers, waiting for matters to be resolved is clearly not wise. Current compliance strategies can proceed from several aspects: First, source traceability—record complete source metadata for each training data point, including URL, crawl time, original copyright declaration, etc.; Second, license filtering—prioritize data with explicit permission for AI training use, such as CC0, CC-BY and other open license content; Third, respect robots.txt protocol, honoring website owners' crawler restriction declarations; Fourth, data desensitization—anonymize or delete content involving personal privacy.

It is worth noting that "Fair Use" boundaries in AI training scenarios remain unclear, and applicable legal standards vary by country. Therefore, it is recommended to reserve copyright filtering interfaces in the data pipeline so that data processing strategies can be quickly adjusted once the legal environment is clarified.

### 1.3.3 Challenge Three: The "Hidden Killer" of Compute Cost

Compute cost for data processing is often severely underestimated. Many teams only calculate GPU training time when budgeting model training costs, but neglect that data preprocessing may consume equal or more resources.

Consider a practical scenario: processing 10TB of raw web data. Assume each data point requires five steps on average: HTML parsing, text extraction, language identification, quality scoring, deduplication check. Each step takes 100 milliseconds per item (already quite fast). Total processing time would be approximately: 10TB ÷ 10KB/item × 0.5 sec/item ≈ 1.5 million GPU hours. At cloud provider pricing, this could mean hundreds of thousands of dollars—comparable to model training costs itself.

| Scenario | Training Data Volume | Training Cost (A100 hours) | Model Performance |
|------|-----------|----------------------|----------|
| No deduplication, train directly | 10T Token | 10,000 hours | Baseline (includes "repeater" issues from repetition) |
| Train after deduplication | 7T Token | 7,000 hours | Better than baseline (repeated data reduces learning efficiency) |
| **Net benefit** | - | **Save 3,000 hours + better model** | - |

However, from another perspective, the return on investment of data engineering can be extremely high. The table above shows a simplified case: by investing resources in data deduplication, not only 30% of training cost is saved, but better model performance is achieved. This is the "leverage effect" of data engineering—at current compute prices (A100 ~$2/hour), 1 hour of data processing work that can eliminate 10 hours of ineffective training yields 20x ROI.

![Figure 1-5: Data Quality and Training Efficiency](../../images/part1/图1_5_数据质量与训练效率.png)

*Figure 1-5: Relationship between Data Quality and Training Efficiency — Green curve (curated data) most efficient, 30-70% quality range has maximum ROI*

### 1.3.4 Opportunity: Three Trends Lowering the Barrier

Despite challenges, for data engineers, now is the golden time to enter. Three major trends are significantly lowering the barrier to large model data engineering.

The first trend is the emergence of open-source high-quality datasets. Unlike early closed commercial data, recent years have seen many high-quality open-source pre-training datasets. HuggingFace's FineWeb provides carefully cleaned 15T token English web data. RedPajama open-sourced the complete reproduction of LLaMA training data. DCLM provides domain-optimized datasets. These open-source datasets greatly reduce the cost for small and medium teams to build pre-training corpora, making resources that were previously only accessible to large companies within reach.

The second trend is the maturation of AI-native data tools. Traditional big data tools (like Hadoop, Spark) are mature but not designed for AI training scenarios. In recent years, a batch of tools designed for LLM data engineering have matured. Alibaba's Data-Juicer provides modular data processing pipelines with dozens of out-of-the-box cleaning operators. Dolma is an Allen AI-developed large-scale text processing toolkit for pre-training data optimization. These tools enable data engineers to stand on the shoulders of giants without building from scratch.

The third trend is the mainstreaming of synthetic data. As mentioned, Microsoft Phi series success demonstrated the enormous potential of synthetic data. Today, using powerful models like GPT-4, Claude to generate training data has become industry standard. This "strong-leading-weak" strategy makes high-quality data acquisition no longer completely dependent on human annotation, greatly accelerating data preparation efficiency. However, synthetic data also brings new challenges: How to avoid model collapse? How to ensure data diversity? These questions will be explored in detail in subsequent chapters.

---

## 1.4 Common Misconceptions and Pitfall Guide

Before entering specific technical practice, it is necessary to clarify several common misconceptions. These misconceptions can lead to resource waste at best, and project failure at worst.

### Misconception One: "More Data Is Better"

This is the most common and most harmful misconception. Many teams' first reaction when starting a project is to "crawl as much data as possible." However, as mentioned, raw data retention rate is typically only 5-20%. Blindly pursuing data volume means 80%+ of crawl, storage, and initial processing costs are wasted. Worse, if subsequent quality filtering is not strict enough and low-quality data mixed into the training set, it may negatively impact model performance.

The correct approach: first validate the entire processing pipeline's effectiveness with small-scale data, confirm quality standards and filtering strategy, then proceed with large-scale data collection. "Quality first, quantity second" should be a fundamental principle of data engineering.

### Misconception Two: "Process All Data Once"

Data processing is an iterative process, not a one-time task. Model training will expose data issues (e.g., excess samples of certain types leading to bias), evaluation results will guide data filtering strategy (e.g., need to augment data in certain domains), legal compliance requirements may change (e.g., need to remove data from certain sources).

Therefore, data processing pipelines must be designed for reproducible execution, incremental updates, and version rollback. Chapter 2 will detail how to use tools like DVC, LakeFS for data version control.

### Misconception Three: "Only Focus on Pre-training Data"

Pre-training data is indeed the foundation of LLMs, but neglecting data from subsequent stages is equally dangerous. A base model well pre-trained on large corpus may produce mediocre dialogue models if SFT data quality is poor. Similarly, if RLHF preference data has inconsistent annotations, the model may exhibit unstable behavior.

Full lifecycle data quality management is equally important. Pre-training, SFT, RLHF, and RAG data should each have their own quality standards and evaluation systems.

### Misconception Four: "Open-Source Data Is Ready to Use"

Open-source datasets greatly lower the entry barrier, but using open-source data directly without secondary review is dangerous. Open-source datasets may have: outdated data (crawled earlier, missing latest knowledge), bias (uneven data distribution), noise (some samples not meeting quality standards), copyright risk (unclear license).

The correct approach is to treat open-source data as "raw material" rather than "finished product." Secondary filtering, augmentation, and balancing based on your own task requirements are needed to maximize the value of open-source data.

---

## 1.5 Chapter Summary

This chapter systematically discussed the core concepts of data engineering in the LLM era from the perspective of Scaling Laws. We first explored the paradigm shift of "quality over quantity," from Chinchilla's optimal ratio to the extreme experiment of the Phi series, demonstrating the decisive role of high-quality data in model performance.

We then introduced the four stages of the LLM data lifecycle: pre-training, SFT, RLHF, and RAG. Each stage has distinctly different data requirements, from TB-level unlabeled text to tens of thousands of preference comparison data points—data volume decreases while quality requirements increase. Understanding this "funnel model" is the foundation for planning data resources.

In the challenges and opportunities section, we analyzed three practical issues: multimodal complexity, copyright compliance, and compute cost, while also seeing opportunities from three trends: open-source datasets, AI-native tools, and synthetic data.

Finally, by clarifying four common misconceptions, we established the correct data engineering mindset for readers: quality before quantity, iterative rather than one-time, full-cycle management, and secondary processing of open-source data.

![Figure 1-6: Knowledge Structure Mind Map](../../images/part1/图1_6_知识结构思维导图.png)

*Figure 1-6: Chapter 1 Knowledge Structure — Covering four major themes: Scaling Laws, Data Lifecycle, Challenges and Opportunities*

---

## Further Reading

**Core Papers**

*Scaling Laws for Neural Language Models* by Kaplan et al. (2020) is the foundational work for understanding large model resource allocation. This OpenAI paper first systematically revealed the power law relationship between model scale, data volume, and compute, providing theoretical guidance for subsequent large model development.

*Training Compute-Optimal Large Language Models* (the Chinchilla paper) by Hoffmann et al. (2022) from DeepMind pointed out the industry's widespread "over-parameterization" problem and provided compute-optimal model-data ratio recommendations.

*Textbooks Are All You Need* by Gunasekar et al. (2023) from Microsoft Research is the pioneering work of the Phi series, proving that high-quality synthetic data can enable small models to match large model performance.

**Open-Source Datasets**

FineWeb released by Penedo et al. is HuggingFace's open-source large-scale high-quality English web dataset, containing approximately 15T tokens, suitable as a base corpus for pre-training. RedPajama released by Together AI is the open-source reproduction of LLaMA training data, valuable for teams wanting to reproduce the LLaMA training process.

---

## Preview of Next Chapter

In the next chapter *Data Infrastructure Selection*, we will move from conceptual to engineering level. You will learn how to choose appropriate storage solutions (S3 vs MinIO), compute frameworks (Spark vs Ray), data formats (Parquet vs JSONL vs WebDataset), and version control tools (DVC vs LakeFS).

Take this question into the next chapter: With limited resources, how do you design a data infrastructure that can support current needs while scaling smoothly?
