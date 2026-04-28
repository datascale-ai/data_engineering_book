# 《大模型数据工程：架构、算法及项目实战》

## 全书目录概览

本书当前中文主线采用 2026 版结构，正文覆盖 28 章与 10 个端到端项目。英文版与日文版正在跟进翻译，当前以翻译状态页说明新版范围。

- [序言](preface.md)
- 第一部分：总论与基础设施
- 第二部分：文本预训练数据工程
- 第三部分：多模态数据工程
- 第四部分：指令微调与偏好数据
- 第五部分：合成数据工程
- 第六部分：推理与 Agent 数据工程
- 第七部分：应用级数据工程
- 第八部分：数据运营与平台建设
- 第九部分：隐私合规与数据安全
- 第十部分：项目实战

## 第一部分：总论与基础设施

建立大模型数据工程的核心认知，说明数据生命周期、质量评估、平台栈和成本治理的基本框架。

- [第1章：大模型时代的数据变革](part1/ch01_data_change.md)
- [第2章：LLM数据生命周期与质量评估框架](part1/ch02_quality_framework.md)
- [第3章：AI原生数据栈与成本治理](part1/ch03_data_stack.md)

## 第二部分：文本预训练数据工程

面向大规模文本语料，覆盖数据来源、采集版权、清洗去重、分词序列化、高效加载和质量闭环。

- [第4章：数据源、采集与版权](part2/ch04_data_sources.md)
- [第5章：清洗、去重与去污染](part2/ch05_cleaning_dedup.md)
- [第6章：分词、序列化与高效加载](part2/ch06_tokenization_loading.md)
- [第7章：数据评估、质量闭环与运营迭代](part2/ch07_data_operations.md)

## 第三部分：多模态数据工程

处理图文、文档、视频、音频与跨模态对齐数据，关注样本结构、质量控制、标注增强和融合训练。

- [第8章：图文对数据工程](part3/ch08_multimodal_image.md)
- [第9章：重标注与文档理解](part3/ch09_recaptioning_ocr.md)
- [第10章：视频与音频数据工程](part3/ch10_video_audio.md)
- [第11章：跨模态对齐与融合](part3/ch11_cross_modal_alignment.md)

## 第四部分：指令微调与偏好数据

围绕模型对齐数据，展开 SFT 指令体系、偏好数据、奖励信号、标注平台与质量运营。

- [第12章：SFT数据设计与指令体系](part4/4_1_SFT.md)
- [第13章：偏好数据与奖励信号](part4/4_2_preference.md)
- [第14章：标注平台、QA体系与数据运营](part4/4_3_qa.md)

## 第五部分：合成数据工程

讲解从种子样本到合成数据工厂的流程，包括知识蒸馏、模型协作、质量控制和模型坍缩风险。

- [第15章：合成数据工厂：从种子到验证](part5/5_1_data_synthesis.md)
- [第16章：知识蒸馏与模型协作](part5/5_2_distillation.md)
- [第17章：合成数据质量控制与模型坍缩](part5/5_3_quality.md)

## 第六部分：推理与 Agent 数据工程

覆盖思维链、推理轨迹、Tool-Use、函数调用、Agent 记忆和多轮交互数据的构建与验证。

- [第18章：思维链与推理数据工程](part6/6_1_COT.md)
- [第19章：Tool-Use 与函数调用数据](part6/6_2_tool.md)
- [第20章：Agent 记忆与多轮交互数据](part6/6_3_agent.md)

## 第七部分：应用级数据工程

面向 RAG 和在线知识系统，说明文档解析、视觉检索、多模态 RAG、在线反馈闭环与知识更新。

- [第21章：RAG 数据流水线](part7/7_1_rag_pipeline.md)
- [第22章：多模态 RAG 与视觉检索](part7/7_2_multimodal_rag_visual_retrieval.md)
- [第23章：在线反馈闭环与知识更新](part7/7_3_online_feedback_knowledge_update.md)

## 第八部分：数据运营与平台建设

从团队组织、版本管理、实验追踪和可观测性角度，构建可持续演进的数据平台能力。

- [第24章：DataOps 飞轮与团队组织](part8/8_1_dataops_flywheel_team.md)
- [第25章：数据版本管理与实验追踪](part8/8_2_data_versioning_experiment_tracking.md)
- [第26章：数据平台可观测性](part8/8_3_data_platform_observability.md)

## 第九部分：隐私合规与数据安全

说明数据合规治理、隐私保护、联邦学习和安全边界，强调工程流程中的合规门禁。

- [第27章：数据合规框架与治理](part9/Ch27_compliance_framework_and_governance.md)
- [第28章：联邦学习与隐私保护技术](part9/Ch28_federated_learning_and_privacy_preserving_technologies.md)

## 第十部分：项目实战

通过 10 个可运行项目，把采集、清洗、合成、RAG、Agent、DataOps、隐私保护和数据飞轮串成端到端实践。

- [项目一：基于 Ray 构建分布式 Mini-C4 数据流水线](part10/10_1_mini_c4.md)
- [项目二：垂直领域专家 SFT（法律）](part10/10_2_legal_sft.md)
- [项目三：LLaVA 多模态指令数据工厂](part10/10_3_llava_instruct.md)
- [项目四：合成数学与代码教材工厂](part10/10_4_synthetic_textbook.md)
- [项目五：多模态 RAG 企业财报助手](part10/10_5_mm_rag.md)
- [项目六：CoT 推理数据集构建与 PRM 训练](part10/10_6_PRM.md)
- [项目七：Agent Tool-Use 数据工厂](part10/10_7_Agent_Tooluse.md)
- [项目八：企业级 DataOps 平台搭建：从数据项目到组织级治理能力](part10/10_8_dataops.md)
- [项目九：隐私保护数据流水线](part10/10_9_privacy_pipline.md)
- [项目十：端到端 LLM 数据飞轮](part10/10_10_flywheel.md)
