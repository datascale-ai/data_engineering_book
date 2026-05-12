# Data Engineering for Large Models: Architecture, Algorithms & Projects

## Full Table of Contents Overview

The Chinese 2026 edition is the mainline of this book, covering 33 chapters and 14 end-to-end project chapters. The English and Japanese editions are being translated incrementally. Chapters not yet translated will display a notice; please use the language switcher at the top to view the Chinese edition.

- [Preface](preface.md)
- [Chinese 2026 Edition Translation Status](translation-status.md)
- Part 1: Overview and Infrastructure
- Part 2: Text Pre-training Data Engineering
- Part 3: Multimodal Data Engineering
- Part 4: Instruction Fine-tuning and Preference Data
- Part 5: Synthetic Data Engineering
- Part 6: Reasoning and Agent Data Engineering
- Part 7: Application-Level Data Engineering
- Part 8: Data Operations and Platform Development
- Part 9: Privacy Compliance and Data Security
- Part 10: Practical Projects
- Part 11: Open-Source LLM Data Engineering in Practice

## Part 1: Overview and Infrastructure

Establishes the core cognitive framework for LLM data engineering, covering the data lifecycle, quality evaluation, platform stack, and cost governance.

- [Chapter 1: The Data Revolution in the Era of Large Models](part1/ch01_data_change.md)
- [Chapter 2: LLM Data Lifecycle and Quality Evaluation Framework](part1/ch02_quality_framework.md)
- [Chapter 3: AI-Native Data Stack and Cost Governance](part1/ch03_data_stack.md)

## Part 2: Text Pre-training Data Engineering

Targets large-scale text corpora, covering data sources, acquisition and copyright, cleaning and deduplication, tokenization and serialization, efficient loading, and the quality closed loop.

- [Chapter 4: Data Sources, Acquisition, and Copyright](part2/ch04_data_sources.md)
- [Chapter 5: Cleaning, Deduplication, and Decontamination](part2/ch05_cleaning_dedup.md)
- [Chapter 6: Tokenization, Serialization, and Efficient Loading](part2/ch06_tokenization_loading.md)
- [Chapter 7: Data Evaluation, Quality Closed Loop, and Operational Iteration](part2/ch07_data_operations.md)

## Part 3: Multimodal Data Engineering

Handles image-text, document, video, audio, and cross-modal alignment data, focusing on sample structure, quality control, annotation augmentation, and fusion training.

- [Chapter 8: Image-Text Pair Data Engineering](part3/ch08_multimodal_image.md)
- [Chapter 9: Re-captioning and Document Understanding](part3/ch09_recaptioning_ocr.md)
- [Chapter 10: Video and Audio Data Engineering](part3/ch10_video_audio.md)
- [Chapter 11: Cross-modal Alignment and Fusion](part3/ch11_cross_modal_alignment.md)

## Part 4: Instruction Fine-tuning and Preference Data

Centers on model alignment data, covering the SFT instruction system, preference data, reward signals, annotation platforms, and quality operations.

- [Chapter 12: SFT Data Design and Instruction System](part4/ch12_sft.md)
- [Chapter 13: Preference Data and Reward Signals](part4/ch13_preference.md)
- [Chapter 14: Annotation Platforms, QA Systems, and Data Operations](part4/ch14_qa.md)

## Part 5: Synthetic Data Engineering

Walks from seed samples to a synthetic data factory, including knowledge distillation, model collaboration, quality control, and the risk of model collapse.

- [Chapter 15: Synthetic Data Factory: From Seed to Verification](part5/ch15_data_synthesis.md)
- [Chapter 16: Knowledge Distillation and Model Collaboration](part5/ch16_distillation.md)
- [Chapter 17: Synthetic Data Quality Control and Model Collapse](part5/ch17_quality.md)

## Part 6: Reasoning and Agent Data Engineering

Covers the construction and validation of chain-of-thought reasoning traces, Tool-Use, function calling, agent memory, and multi-turn interaction data.

- [Chapter 18: Chain-of-Thought and Reasoning Data Engineering](part6/ch18_cot.md)
- [Chapter 19: Tool-Use and Function Calling Data](part6/ch19_tool.md)
- [Chapter 20: Agent Memory and Multi-turn Interaction Data](part6/ch20_agent.md)

## Part 7: Application-Level Data Engineering

Targets RAG and online knowledge systems, including document parsing, visual retrieval, multimodal RAG, online feedback loops, and knowledge updates.

- [Chapter 21: RAG Data Pipeline](part7/ch21_rag_pipeline.md)
- [Chapter 22: Multimodal RAG and Visual Retrieval](part7/ch22_multimodal_rag_visual_retrieval.md)
- [Chapter 23: Online Feedback Loop and Knowledge Update](part7/ch23_online_feedback_knowledge_update.md)

## Part 8: Data Operations and Platform Development

Builds sustainable data platform capabilities from the perspectives of team organization, version management, experiment tracking, and observability.

- [Chapter 24: DataOps Flywheel and Team Organization](part8/ch24_dataops_flywheel_team.md)
- [Chapter 25: Data Version Management and Experiment Tracking](part8/ch25_data_versioning_experiment_tracking.md)
- [Chapter 26: Data Platform Observability](part8/ch26_data_platform_observability.md)

## Part 9: Privacy Compliance and Data Security

Discusses data compliance and governance, privacy protection, federated learning, and security boundaries, emphasizing compliance gates in engineering workflows.

- [Chapter 27: Data Compliance Framework and Governance](part9/ch27_compliance_framework_and_governance.md)
- [Chapter 28: Federated Learning and Privacy-Preserving Technologies](part9/ch28_federated_learning_and_privacy_preserving_technologies.md)

## Part 10: Practical Projects

Ten runnable projects that string together acquisition, cleaning, synthesis, RAG, Agent, DataOps, privacy protection, and the data flywheel into end-to-end practice.

- [Project 1: Building a Distributed Mini-C4 Data Pipeline with Ray](part10/10_1_mini_c4.md)
- [Project 2: Vertical-Domain Expert SFT (Legal)](part10/10_2_legal_sft.md)
- [Project 3: LLaVA Multimodal Instruction Data Factory](part10/10_3_llava_instruct.md)
- [Project 4: Synthetic Math and Code Textbook Factory](part10/10_4_synthetic_textbook.md)
- [Project 5: Multimodal RAG Enterprise Financial Report Assistant](part10/10_5_mm_rag.md)
- [Project 6: CoT Reasoning Dataset Construction and PRM Training](part10/10_6_PRM.md)
- [Project 7: Agent Tool-Use Data Factory](part10/10_7_Agent_Tooluse.md)
- [Project 8: Enterprise DataOps Platform: From Data Projects to Organizational Governance](part10/10_8_dataops.md)
- [Project 9: Privacy-Preserving Data Pipeline](part10/10_9_privacy_pipeline.md)
- [Project 10: End-to-End LLM Data Flywheel](part10/10_10_flywheel.md)

## Part 11: Open-Source LLM Data Engineering in Practice

- [Chapter 29: LLM Pre-training Data Recipes](part11/ch29_pretrain_recipes.md)
- [Chapter 30: LLM Post-training Data Engineering: SFT and Preference Alignment](part11/ch30_posttrain_recipes.md)
- [Chapter 31: Reasoning Models and RL Data Engineering: R1 / QwQ Paradigm](part11/ch31_rl_reasoning_data.md)
- [Chapter 32: Multimodal Understanding VLM](part11/ch32_multimodel.md)
- [Chapter 33: Multimodal Generative Model Data Engineering — T2I and T2V Data Pipelines](part11/ch33_t21_t2v.md)
- [Project 11: Mini-DeepSeek Pre-training Reproduction](part11/projects/p11_mini_deepseek.md)
- [Project 12: R1 Reasoning Flywheel](part11/projects/p12_r1_reasoning_flywheel.md)
- [Project 13: Multimodal Instruction Factory](part11/projects/p13_multimodel_ins.md)
- [Project 14: Video Generation Dataset — From Video Source to T2V Training Pipeline](part11/projects/p14_vedio_gen.md)
