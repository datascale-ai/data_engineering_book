# Part XIV: Project Case Studies

## Overview

Part XIV translates the methods, platforms, and recipes from the preceding thirteen parts into fifteen reproducible case studies. Rather than providing step-by-step replication instructions, this part focuses on why each project is justified, how data boundaries are defined, how architectural decisions are formed, how sample schemas are designed, how acceptance criteria are established, and how failure modes and compliance risks are documented.

The fifteen projects span pre-training corpora, domain-specific SFT, multimodal instruction tuning, synthetic textbooks, multimodal RAG, process supervision, Agent Tool-Use, DataOps, privacy protection, data flywheels, open-source model pre-training replication, a pedagogical R1 reasoning data flywheel, a multimodal instruction factory, a video generation data pipeline, and an enterprise-grade semantic question-answering assistant. Together they map the key capabilities from the preceding thirteen parts, forming the final validation layer bridging methodological understanding and engineering delivery.

All project chapters in this part follow a unified case-study structure: abstract, keywords, project objectives and reader takeaways, scenario constraints and data boundaries, architectural decisions, sample schema/data flow, core implementation excerpts, experimental or acceptance metrics, cost and compliance boundaries, common failure modes, reproducibility resource notes, chapter summary, and references. The main text retains essential code snippets; long scripts and complete notebooks belong in the companion resources.

## Terminology

Throughout this part, "project case study" denotes a runnable, verifiable, and auditable engineering delivery unit; "end-to-end closed loop" describes a project workflow in which inputs, processing, outputs, acceptance criteria, costs, and risk boundaries are all traceable; and "training candidate data" refers to data artifacts that still require quality, compliance, and task-fitness checks. Project chapters do not use the launching of a prototype demonstration as the completion criterion; instead, the primary judgment criteria are deliverables, acceptance conditions, reproduction paths, and applicable boundaries.

## Table of Contents

- [Project 1: Building a Distributed Mini-C4 Data Pipeline with Ray](p01_mini_c4.md)
- [Project 2: Vertical Domain Expert SFT (Legal)](p02_legal_sft.md)
- [Project 3: LLaVA Multimodal Instruction Data Factory](p03_llava_instruct.md)
- [Project 4: Synthetic Mathematics and Code Textbook Factory](p04_synthetic_textbook.md)
- [Project 5: Multimodal RAG Enterprise Financial Report Assistant](p05_mm_rag.md)
- [Project 6: CoT Reasoning Dataset Construction and PRM Training](p06_prm.md)
- [Project 7: Agent Tool-Use Data Factory](p07_agent_tooluse.md)
- [Project 8: Building an Enterprise-Grade DataOps Platform: From Data Projects to Organizational Governance Capability](p08_dataops.md)
- [Project 9: Privacy-Preserving Data Pipeline](p09_privacy_pipeline.md)
- [Project 10: End-to-End LLM Data Flywheel](p10_flywheel.md)
- [Project 11: Mini-DeepSeek Pre-training Replication](p11_mini_deepseek.md)
- [Project 12: Pedagogical R1 Reasoning Data Flywheel](p12_r1_reasoning_flywheel.md)
- [Project 13: Multimodal Instruction Factory](p13_multimodal_instruction_factory.md)
- [Project 14: Video Generation Dataset: From Raw Video Sources to a T2V-Ready Training Data Pipeline](p14_video_generation.md)
- [Project 15: Building an Enterprise-Grade Semantic Question-Answering Assistant with DataAgent](p15_dataagent_semantic_nl2sql_agent.md)

## Capability Mapping

P01–P04 correspond to raw data materials, domain instruction tuning, multimodal instruction tuning, and synthetic data capabilities, primarily building on Parts II through V.

P05–P07 correspond to multimodal RAG, process supervision, and Agent Tool-Use, primarily building on Parts VI and VII.

P08–P10 correspond to DataOps, privacy protection, and data flywheels, primarily building on Parts VIII through XI.

P11–P15 correspond to open-source model data recipes, reasoning flywheels, multimodal instruction tuning, video generation, and enterprise-grade Agent applications, primarily building on Part XIII and extending into the delivery templates in the appendices.

## Reading Order

Readers seeking a comprehensive engineering perspective may read in sequence from P01 to P15. This path traces data engineering from raw material processing, training signals, and application systems through platform governance to enterprise deployment.

Readers approaching by task may locate the relevant capability first: for pre-training and web corpora, read P01 and P11; for domain SFT, read P02; for multimodal and visual data, read P03, P05, P13, and P14; for reasoning and process supervision, read P06 and P12; for Agent and enterprise applications, read P07 and P15; for platform governance and compliance, read P08, P09, and P10.

Readers implementing a course or team boot camp are advised to select P01, P02, P05, P08, P09, P12, and P15 to form a tiered lab bundle, and to use the environment, cost, and delivery checklists in the appendices for acceptance testing.
