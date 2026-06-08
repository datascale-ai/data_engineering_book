# Part 2: Text Pre-training Data Engineering

## Positioning

Part 2 focuses on the production, governance, and continuous iteration of text pre-training corpora. It covers the full engineering chain from source acquisition, copyright boundaries, cleaning and deduplication, tokenization and serialization, to quality evaluation and operational feedback loops. This part builds on Part 1's discussion of data infrastructure and quality frameworks, turning the abstract data lifecycle into an executable text pre-training pipeline. It also provides the baseline for Part 3: only after understanding how text data is acquired, filtered, packaged, and evaluated can we judge why images, documents, video, and audio introduce more complex representation, alignment, and cost problems.

The goal is not to list tools, but to establish a reusable engineering judgment framework. Readers should be able to answer three questions: which sources can enter a pre-training corpus and which should be isolated or removed; how cleaning, deduplication, decontamination, and PII redaction jointly define training-data quality; and how tokenization, serialization, loading, and operational evaluation turn "usable data" into "trainable data."

## Learning Objectives

- Build an end-to-end governance view of text pre-training data from source, acquisition, parsing, and evidence preservation.
- Master key engineering methods for cleaning, deduplication, PII redaction, and benchmark decontamination.
- Understand how tokenization, serialization, packing, mixing, and efficient data loaders affect training throughput.
- Design offline proxy metrics, data version management, and quality loops for continuous data operations.
- Distinguish the risk boundaries of public web data, code, academic papers, books, enterprise internal data, and user feedback.

## Prerequisites

Readers should understand the data lifecycle, quality dimensions, cost governance, and stack layering introduced in Part 1. Readers with traditional ETL, data lake, or machine-learning data processing experience may focus on the differences introduced here: copyright and license boundaries for training corpora, benchmark contamination, offline tokenization, sequence packing, cross-node data loaders, and data-version rollback.

## Chapter Logic

Chapter 4 answers where data comes from and whether it can be used, focusing on source profiling, acquisition pipelines, copyright licenses, and metadata evidence. Chapter 5 answers how raw corpora become high-quality corpora, focusing on rule filtering, model scoring, deduplication, PII redaction, and benchmark decontamination. Chapter 6 answers how high-quality corpora efficiently enter training systems, focusing on tokenization, serialization, sharding, packing, mixing, and data-loader throughput. Chapter 7 answers how to judge whether a data version really improves a model, focusing on offline proxy metrics, data versioning, problem-sample libraries, root-cause review, and operating cadence.

This part grounds the quality framework from Part 1 in the production flow of text pre-training corpora and provides a comparison baseline for multimodal data engineering in Part 3. Many issues that look like "text cleaning" problems, such as source licensing, deduplication, contamination, packaging, and evaluation, become even more complex in image, video, and audio settings.

## Part Contents

- [Chapter 4: Data Sources, Acquisition, and Copyright](ch04_data_sources.md)
- [Chapter 5: Cleaning, Deduplication, and Decontamination](ch05_cleaning_dedup.md)
- [Chapter 6: Tokenization, Serialization, and Efficient Loading](ch06_tokenization_loading.md)
- [Chapter 7: Data Evaluation, Quality Closed Loop, and Operational Iteration](ch07_data_operations.md)

Suggested order: read Chapters 4 through 7 sequentially. Readers with pre-training experience may read Chapter 7 first to establish the quality-loop view, then return to Chapters 4 through 6 to check whether their own pipelines are traceable, diagnosable, and reproducible.
