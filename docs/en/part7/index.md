# Part VII: Application-Level Data Engineering

## Scope of This Part

Part VII approaches data engineering from an application-systems perspective, covering RAG, visual retrieval, multimodal evidence fusion, online feedback loops, and knowledge updates—focusing on data engineering design for real-world production use cases.

## Terminology Conventions

Throughout this part, "Retrieval-Augmented Generation (RAG)" refers to the complete pipeline encompassing document ingestion, chunking, indexing, retrieval, reranking, and generation with citations. "Evidence" refers to context fragments that can be cited, located, and audited by the model. "Online feedback" refers to evaluation signals, corrections, and updates derived from real user interactions. RAG corpora, knowledge bases, retrieval indexes, and feedback samples should be managed separately and not conflated with static document repositories.

## Table of Contents

- [Chapter 21: RAG Data Pipelines](ch21_rag_pipeline.md)
- [Chapter 22: Multimodal RAG and Visual Retrieval](ch22_multimodal_rag_visual_retrieval.md)
- [Chapter 23: Online Feedback Loops and Knowledge Updates](ch23_online_feedback_knowledge_update.md)

## Recommended Reading Order

- Start with Chapter 21 to understand document ingestion, chunking, indexing, and retrieval workflows.
- Proceed to Chapter 22 to explore multimodal RAG, visual evidence localization, and evaluation attribution.
- Finish with Chapter 23 to integrate online feedback, knowledge backfilling, and version updates into the closed loop.
