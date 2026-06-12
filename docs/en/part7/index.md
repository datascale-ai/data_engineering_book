# Part VII: Application-Level Data Engineering

## Scope of This Part

Part VII approaches data engineering from an application-systems perspective, covering RAG, visual retrieval, multimodal evidence fusion, online feedback loops, and knowledge updates—focusing on data engineering design for real-world production use cases.

## Terminology Conventions

Throughout this part, "Retrieval-Augmented Generation (RAG)" refers to the complete pipeline encompassing document ingestion, chunking, indexing, retrieval, reranking, and generation with citations. "Evidence" refers to context fragments that can be cited, located, and audited by the model. "Online feedback" refers to evaluation signals, corrections, and updates derived from real user interactions. RAG corpora, knowledge bases, retrieval indexes, and feedback samples should be managed separately and not conflated with static document repositories.

## Learning Objectives

After completing this part, readers should be able to:

- Design a RAG data pipeline from document ingestion, chunking, indexing, and retrieval to answer citation.
- Establish basic engineering conventions for multimodal evidence localization, visual retrieval, and cross-modal evaluation.
- Turn online feedback, user corrections, and knowledge updates into a traceable data loop.
- Judge the evidence quality, update cost, and release risk of application-level data systems.

## Prerequisites

Before reading this part, readers should understand text processing from Part 2, multimodal samples from Part 3, and tool/agent data from Part 6. Readers from business application teams may focus on evidence objects, version updates, and feedback write-back rather than reducing RAG to vector-database integration.

## Chapter Logic

Chapter 21 establishes the basic pipeline for document RAG, focusing on ingestion, chunking, retrieval, reranking, and citation. Chapter 22 extends this into multimodal RAG and visual retrieval, handling image, table, and document evidence. Chapter 23 discusses online feedback and knowledge updates, connecting application logs, correction signals, and version governance into a long-term loop.

## Table of Contents

- [Chapter 21: RAG Data Pipelines](ch21_rag_pipeline.md)
- [Chapter 22: Multimodal RAG and Visual Retrieval](ch22_multimodal_rag_visual_retrieval.md)
- [Chapter 23: Online Feedback Loops and Knowledge Updates](ch23_online_feedback_knowledge_update.md)

## Recommended Reading Order

- Start with Chapter 21 to understand document ingestion, chunking, indexing, and retrieval workflows.
- Proceed to Chapter 22 to explore multimodal RAG, visual evidence localization, and evaluation attribution.
- Finish with Chapter 23 to integrate online feedback, knowledge backfilling, and version updates into the closed loop.
