# Data Engineering for Large Models: Architecture, Algorithms & Projects

## Full Table of Contents

The Chinese 2026 Springer edition is the canonical mainline of this book, covering 14 parts, 48 chapters, 15 project case studies, and 3 appendices, with front matter and an afterword in the site edition. The English edition follows the same structure and is being brought into release quality in batches: existing complete English chapters are preserved, missing chapters are translated against the Chinese mainline, and the website navigation is checked with each batch.

- [Abbreviations](abbreviations.md)
- [Preface](preface.md)
- [English Edition Status](translation-status.md)
- [Front-Matter Guide: Book Structure, Reading Paths, and Edition Notes](front_matter_guide.md)
- [Part 1: Overview and Infrastructure](part1/index.md)
- [Part 2: Text Pre-training Data Engineering](part2/index.md)
- [Part 3: Multimodal Data Engineering](part3/index.md)
- [Part 4: Instruction Fine-tuning and Preference Data](part4/index.md)
- [Part 5: Synthetic Data Engineering](part5/index.md)
- [Part 6: Reasoning and Agent Data Engineering](part6/index.md)
- [Part 7: Application-Level Data Engineering](part7/index.md)
- [Part 8: Data Operations and Platform Development](part8/index.md)
- [Part 9: Data Assets, Data Products, and Data Contracts](part9/index.md)
- [Part 10: Agentic Data Engineering and Data Engineering Agents](part10/index.md)
- [Part 11: Privacy, Compliance, and Data Security](part11/index.md)
- [Part 12: Specialized Datasets and Data Engineering Practice](part12/index.md)
- [Part 13: Open-Source Large-Model Data Engineering Recipes and Paradigms](part13/index.md)
- [Part 14: Hands-on Projects](part14/index.md)
- [Appendix A: Tools and Frameworks Quick Reference](appendix_a_tools_and_frameworks_quick_reference.md)
- [Appendix B: Compliance and Release Checklist](appendix_b_compliance_and_release_checklist.md)
- [Appendix C: Cost Estimation and Resource Templates](appendix_c_cost_estimation_and_resource_templates.md)
- [Afterword](afterword.md)

## Part 1: Overview and Infrastructure

This part establishes the core framework for large-model data engineering: how the data lifecycle, quality evaluation, AI-native data stack, and cost governance fit together.

- [Part Contents](part1/index.md)
- [Chapter 1: The Data Revolution in the Era of Large Models](part1/ch01_data_change.md)
- [Chapter 2: LLM Data Lifecycle and Quality Evaluation Framework](part1/ch02_quality_framework.md)
- [Chapter 3: AI-Native Data Stack and Cost Governance](part1/ch03_data_stack.md)

## Part 2: Text Pre-training Data Engineering

This part focuses on large-scale text corpora, including data sources, acquisition and copyright, cleaning, deduplication, decontamination, tokenization, serialization, efficient loading, and quality operations.

- [Part Contents](part2/index.md)
- [Chapter 4: Data Sources, Acquisition, and Copyright](part2/ch04_data_sources.md)
- [Chapter 5: Cleaning, Deduplication, and Decontamination](part2/ch05_cleaning_dedup.md)
- [Chapter 6: Tokenization, Serialization, and Efficient Loading](part2/ch06_tokenization_loading.md)
- [Chapter 7: Data Evaluation, Quality Closed Loop, and Operational Iteration](part2/ch07_data_operations.md)

## Part 3: Multimodal Data Engineering

This part covers image-text, document, video, audio, and cross-modal alignment data, with attention to sample structure, quality control, annotation augmentation, and fusion training.

- [Part Contents](part3/index.md)
- [Chapter 8: Image-Text Pair Data Engineering](part3/ch08_multimodal_image.md)
- [Chapter 9: Re-captioning and Document Understanding](part3/ch09_recaptioning_ocr.md)
- [Chapter 10: Video and Audio Data Engineering](part3/ch10_video_audio.md)
- [Chapter 11: Cross-modal Alignment and Fusion](part3/ch11_cross_modal_alignment.md)

## Part 4: Instruction Fine-tuning and Preference Data

This part centers on model alignment data, covering SFT instruction systems, preference data, reward signals, annotation platforms, QA, and data operations.

- [Part Contents](part4/index.md)
- [Chapter 12: SFT Data Design and Instruction Systems](part4/ch12_sft.md)
- [Chapter 13: Preference Data and Reward Signals](part4/ch13_preference.md)
- [Chapter 14: Annotation Platforms, QA Systems, and Data Operations](part4/ch14_qa.md)

## Part 5: Synthetic Data Engineering

This part explains the path from seed samples to a synthetic data factory, including knowledge distillation, model collaboration, quality control, and model-collapse risks.

- [Part Contents](part5/index.md)
- [Chapter 15: Synthetic Data Factory: From Seed to Verification](part5/ch15_data_synthesis.md)
- [Chapter 16: Knowledge Distillation and Model Collaboration](part5/ch16_distillation.md)
- [Chapter 17: Synthetic Data Quality Control and Model Collapse](part5/ch17_quality.md)

## Part 6: Reasoning and Agent Data Engineering

This part covers chain-of-thought data, reasoning traces, tool use, function calling, agent memory, and multi-turn interaction data.

- [Part Contents](part6/index.md)
- [Chapter 18: Chain-of-Thought and Reasoning Data Engineering](part6/ch18_cot.md)
- [Chapter 19: Tool-Use and Function Calling Data](part6/ch19_tool.md)
- [Chapter 20: Agent Memory and Multi-turn Interaction Data](part6/ch20_agent.md)

## Part 7: Application-Level Data Engineering

This part targets RAG and online knowledge systems, including document parsing, visual retrieval, multimodal RAG, online feedback loops, and knowledge updates.

- [Part Contents](part7/index.md)
- [Chapter 21: RAG Data Pipeline](part7/ch21_rag_pipeline.md)
- [Chapter 22: Multimodal RAG and Visual Retrieval](part7/ch22_multimodal_rag_visual_retrieval.md)
- [Chapter 23: Online Feedback Closed Loop and Knowledge Update](part7/ch23_online_feedback_knowledge_update.md)

## Part 8: Data Operations and Platform Development

This part builds sustainable data platform capabilities through team organization, version management, experiment tracking, and observability.

- [Part Contents](part8/index.md)
- [Chapter 24: DataOps Flywheel and Team Organization](part8/ch24_dataops_flywheel_team.md)
- [Chapter 25: Data Version Management and Experiment Tracking](part8/ch25_data_versioning_experiment_tracking.md)
- [Chapter 26: Data Platform Observability](part8/ch26_data_platform_observability.md)

## Part 9: Data Assets, Data Products, and Data Contracts

This part turns data pipelines into discoverable, reusable, auditable organizational assets through catalogs, metadata governance, data products, contracts, valuation, reuse, and internal data markets.

- [Part Contents](part9/index.md)
- [Chapter 27: Data Catalogs and Metadata Governance](part9/ch27_data_catalog_and_metadata_governance.md)
- [Chapter 28: Data Productization and Data Contracts](part9/ch28_data_productization_and_data_contracts.md)
- [Chapter 29: Data Valuation and Reuse Mechanisms](part9/ch29_data_valuation_and_reuse.md)
- [Chapter 30: Internal Data Markets and Sharing Governance](part9/ch30_internal_data_market_and_sharing_governance.md)

## Part 10: Agentic Data Engineering and Data Engineering Agents

This part discusses how data engineering agents participate in acquisition, parsing, cleaning, annotation, synthesis, evaluation, DataOps, security, permissions, and human-AI collaboration.

- [Part Contents](part10/index.md)
- [Chapter 31: Data Engineering Agent Architecture and Task Boundaries](part10/ch31_agent_architecture.md)
- [Chapter 32: Automated Acquisition, Parsing, and Cleaning Agents](part10/ch32_auto_collection_parsing_cleaning.md)
- [Chapter 33: Annotation, Synthesis, and Evaluation Agents](part10/ch33_labeling_synthesis_evaluation.md)
- [Chapter 34: DataOps Agents and Platform Autonomy](part10/ch34_dataops_agent.md)
- [Chapter 35: Security, Permissions, and Human-AI Collaboration for Data Engineering Agents](part10/ch35_security_permission_collaboration.md)

## Part 11: Privacy, Compliance, and Data Security

This part focuses on compliance frameworks, privacy protection, federated learning, security boundaries, and auditable controls across the data lifecycle.

- [Part Contents](part11/index.md)
- [Chapter 36: Data Compliance Frameworks and Governance](part11/ch36_compliance_framework_and_governance.md)
- [Chapter 37: Federated Learning and Privacy-Preserving Technologies](part11/ch37_federated_learning_and_privacy_preserving_technologies.md)

## Part 12: Specialized Datasets and Data Engineering Practice

This part uses representative specialized datasets to show how data engineering methods are organized around task definitions, schemas, build pipelines, quality control, evaluation protocols, and compliance risks.

- [Part Contents](part12/index.md)
- [Chapter 38: StructBill-CN Bill Document Understanding Data Engineering](part12/ch38_structbill_cn_dataset.md)
- [Chapter 39: SparseTable-Bench Table-Structure Robustness Data Engineering](part12/ch39_sparse_table_bench_dataset.md)
- [Chapter 40: Multi-Chart Infographic Reasoning Data Engineering](part12/ch40_multi_chart_infographic_reasoning_dataset.md)
- [Chapter 41: MedImage-ToolVQA Medical Image Tool-Use VQA Data Engineering](part12/ch41_medimage_tool_vqa_dataset.md)
- [Chapter 42: VoiceStyleControl Controllable Voice Interaction Data Engineering](part12/ch42_voice_style_control_dataset.md)
- [Chapter 43: Latent-Switch-69K Implicit/Explicit Reasoning Data Engineering](part12/ch43_latent_switch_69k.md)

## Part 13: Open-Source Large-Model Data Engineering Recipes and Paradigms

This part focuses on data recipes, training paradigms, and engineering organization for open-source large models, covering pre-training, post-training, reasoning RL, VLMs, and T2I/T2V generation.

- [Part Contents](part13/index.md)
- [Chapter 44: LLM Pre-training Data Engineering in Practice: From Recipes to Delivery](part13/ch44_pretrain_recipes.md)
- [Chapter 45: LLM Post-training Data Engineering: SFT and Preference Alignment](part13/ch45_posttrain_recipes.md)
- [Chapter 46: Reasoning Models and RL Data Engineering: R1 / QwQ Paradigms](part13/ch46_rl_reasoning_data.md)
- [Chapter 47: VLM Data Recipes: From Pre-training to Visual Alignment](part13/ch47_vlm_data_recipes.md)
- [Chapter 48: Multimodal Generative Model Data Engineering: T2I and T2V Data Pipelines](part13/ch48_t2i_t2v.md)

## Part 14: Hands-on Projects

This part connects acquisition, cleaning, synthesis, RAG, agents, DataOps, privacy, data flywheels, open-source model reproduction, video-generation data pipelines, and enterprise semantic data agents into runnable projects.

- [Part Contents](part14/index.md)
- [Project 1: Building a Distributed Mini-C4 Data Pipeline with Ray](part14/p01_mini_c4.md)
- [Project 2: Vertical-Domain Expert SFT for Legal Data](part14/p02_legal_sft.md)
- [Project 3: LLaVA Multimodal Instruction Data Factory](part14/p03_llava_instruct.md)
- [Project 4: Synthetic Math and Code Textbook Factory](part14/p04_synthetic_textbook.md)
- [Project 5: Multimodal RAG Enterprise Financial Report Assistant](part14/p05_mm_rag.md)
- [Project 6: CoT Reasoning Dataset Construction and PRM Training](part14/p06_prm.md)
- [Project 7: Agent Tool-Use Data Factory](part14/p07_agent_tooluse.md)
- [Project 8: Enterprise DataOps Platform: From Data Projects to Organizational Governance](part14/p08_dataops.md)
- [Project 9: Privacy-Preserving Data Pipeline](part14/p09_privacy_pipeline.md)
- [Project 10: End-to-End LLM Data Flywheel](part14/p10_flywheel.md)
- [Project 11: Mini-DeepSeek Pre-training Reproduction](part14/p11_mini_deepseek.md)
- [Project 12: R1 Reasoning Flywheel](part14/p12_r1_reasoning_flywheel.md)
- [Project 13: Multimodal Instruction Factory](part14/p13_multimodal_instruction_factory.md)
- [Project 14: Video Generation Dataset: From Video Sources to a T2V Training Pipeline](part14/p14_video_generation.md)
- [Project 15: Building an Enterprise Semantic Data Assistant with DataAgent](part14/p15_dataagent_semantic_nl2sql_agent.md)
