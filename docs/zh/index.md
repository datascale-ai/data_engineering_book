# 《大模型数据工程：架构、算法及项目实战》

## 全书目录概览

本书当前中文主线采用 2026 Springer 尺寸版结构，正文覆盖 48 章、15 个端到端项目与 7 个附录（A–G）。为降低跨篇阅读门槛，本版在前置部分新增统一的缩写表，并为每一篇补充了分册目录页。

- [缩写表](abbreviations.md)
- [序言](preface.md)
- [卷前导读：全书结构、阅读路径与版本说明](front_matter_guide.md)
- [第一篇：总论与基础设施](part1/index.md)
- [第二篇：文本预训练数据工程](part2/index.md)
- [第三篇：多模态数据工程](part3/index.md)
- [第四篇：指令微调与偏好数据](part4/index.md)
- [第五篇：合成数据工程](part5/index.md)
- [第六篇：推理与 Agent 数据工程](part6/index.md)
- [第七篇：应用级数据工程](part7/index.md)
- [第八篇：数据运营与平台建设](part8/index.md)
- [第九篇：数据资产、数据产品与数据契约](part9/index.md)
- [第十篇：智能化数据工程与 Data Engineering Agent](part10/index.md)
- [第十一篇：隐私合规与数据安全](part11/index.md)
- [第十二篇：专项数据集与数据工程实践](part12/index.md)
- [第十三篇：开源大模型数据工程配方与范式](part13/index.md)
- [第十四篇：项目案例研究](part14/index.md)

## 分篇目录

## 第一篇：总论与基础设施

建立大模型数据工程的核心认知，说明数据生命周期、质量评估、平台栈和成本治理的基本框架。

- [本篇目录](part1/index.md)
- [第1章：大模型时代的数据变革](part1/ch01_data_change.md)
- [第2章：LLM数据生命周期与质量评估框架](part1/ch02_quality_framework.md)
- [第3章：AI原生数据栈与成本治理](part1/ch03_data_stack.md)

## 第二篇：文本预训练数据工程

面向大规模文本语料，覆盖数据来源、采集版权、清洗去重、分词序列化、高效加载和质量闭环。

- [本篇目录](part2/index.md)
- [第4章：数据源、采集与版权](part2/ch04_data_sources.md)
- [第5章：清洗、去重与去污染](part2/ch05_cleaning_dedup.md)
- [第6章：分词、序列化与高效加载](part2/ch06_tokenization_loading.md)
- [第7章：数据评估、质量闭环与运营迭代](part2/ch07_data_operations.md)

## 第三篇：多模态数据工程

处理图文、文档、视频、音频与跨模态对齐数据，关注样本结构、质量控制、标注增强和融合训练。

- [本篇目录](part3/index.md)
- [第8章：图文对数据工程](part3/ch08_multimodal_image.md)
- [第9章：重标注与文档理解](part3/ch09_recaptioning_ocr.md)
- [第10章：视频与音频数据工程](part3/ch10_video_audio.md)
- [第11章：跨模态对齐与融合](part3/ch11_cross_modal_alignment.md)

## 第四篇：指令微调与偏好数据

围绕模型对齐数据，展开 SFT 指令体系、偏好数据、奖励信号、标注平台与质量运营。

- [本篇目录](part4/index.md)
- [第12章：SFT数据设计与指令体系](part4/ch12_sft.md)
- [第13章：偏好数据与奖励信号](part4/ch13_preference.md)
- [第14章：标注平台、QA体系与数据运营](part4/ch14_qa.md)

## 第五篇：合成数据工程

讲解从种子样本到合成数据工厂的流程，包括知识蒸馏、模型协作、质量控制和模型坍缩风险。

- [本篇目录](part5/index.md)
- [第15章：合成数据工厂：从种子到验证](part5/ch15_data_synthesis.md)
- [第16章：知识蒸馏与模型协作](part5/ch16_distillation.md)
- [第17章：合成数据质量控制与模型坍缩](part5/ch17_quality.md)

## 第六篇：推理与 Agent 数据工程

覆盖思维链、推理轨迹、Tool-Use、函数调用、Agent 记忆和多轮交互数据的构建与验证。

- [本篇目录](part6/index.md)
- [第18章：思维链与推理数据工程](part6/ch18_cot.md)
- [第19章：Tool-Use 与函数调用数据](part6/ch19_tool.md)
- [第20章：Agent 记忆与多轮交互数据](part6/ch20_agent.md)

## 第七篇：应用级数据工程

面向 RAG 和在线知识系统，说明文档解析、视觉检索、多模态 RAG、在线反馈闭环与知识更新。

- [本篇目录](part7/index.md)
- [第21章：RAG 数据流水线](part7/ch21_rag_pipeline.md)
- [第22章：多模态 RAG 与视觉检索](part7/ch22_multimodal_rag_visual_retrieval.md)
- [第23章：在线反馈闭环与知识更新](part7/ch23_online_feedback_knowledge_update.md)

## 第八篇：数据运营与平台建设

从团队组织、版本管理、实验追踪和可观测性角度，构建可持续演进的数据平台能力。

- [本篇目录](part8/index.md)
- [第24章：DataOps 飞轮与团队组织](part8/ch24_dataops_flywheel_team.md)
- [第25章：数据版本管理与实验追踪](part8/ch25_data_versioning_experiment_tracking.md)
- [第26章：数据平台可观测性](part8/ch26_data_platform_observability.md)

## 第九篇：数据资产、数据产品与数据契约

第九篇补齐数据资产化、数据产品、价值评估和企业内部共享治理，使前文的数据流水线进一步沉淀为可发现、可复用、可审计的组织级资产。

- [本篇目录](part9/index.md)
- [第27章：数据资产目录与元数据治理](part9/ch27_data_catalog_and_metadata_governance.md)
- [第28章：数据产品化与数据契约](part9/ch28_data_productization_and_data_contracts.md)
- [第29章：数据资产价值评估与复用机制](part9/ch29_data_valuation_and_reuse.md)
- [第30章：企业内部数据市场与共享治理](part9/ch30_internal_data_market_and_sharing_governance.md)

## 第十篇：智能化数据工程与 Data Engineering Agent

第十篇讨论 Agentic Data Engineering，聚焦数据工程 Agent 如何参与采集、解析、清洗、标注、合成、评测、DataOps 与安全协同，并以 DataAgent 作为贯穿式工程参照，将抽象架构连接到可运行的数据工程 Agent 系统。

- [本篇目录](part10/index.md)
- [第31章：数据工程 Agent 的架构与任务边界](part10/ch31_agent_architecture.md)
- [第32章：自动化采集、解析与清洗 Agent](part10/ch32_auto_collection_parsing_cleaning.md)
- [第33章：标注、合成与评测 Agent](part10/ch33_labeling_synthesis_evaluation.md)
- [第34章：DataOps Agent 与平台自治](part10/ch34_dataops_agent.md)
- [第35章：数据工程 Agent 的安全、权限与人机协同](part10/ch35_security_permission_collaboration.md)

## 第十一篇：隐私合规与数据安全

第十一篇聚焦数据合规、隐私保护、联邦学习与安全边界，强调在工程流程中把法规要求、风险控制与可审计能力前置到数据生命周期。

- [本篇目录](part11/index.md)
- [第36章：数据合规框架与治理](part11/ch36_compliance_framework_and_governance.md)
- [第37章：联邦学习与隐私保护技术](part11/ch37_federated_learning_and_privacy_preserving_technologies.md)

## 第十二篇：专项数据集与数据工程实践

第十二篇以若干具有代表性的专项数据集为线索，讨论数据工程方法在真实任务中的组织方式。各章围绕任务定义、样本 schema、构建流水线、质量控制、评测协议和合规风险展开，并向后连接项目案例研究与开源模型数据配方。

- [本篇目录](part12/index.md)
- [第38章：StructBill-CN 票据文档理解数据工程](part12/ch38_structbill_cn_dataset.md)
- [第39章：SparseTable-Bench 表格结构鲁棒性数据工程](part12/ch39_sparse_table_bench_dataset.md)
- [第40章：多图表信息图推理数据工程](part12/ch40_multi_chart_infographic_reasoning_dataset.md)
- [第41章：MedImage-ToolVQA 医学图像工具调用数据工程](part12/ch41_medimage_tool_vqa_dataset.md)
- [第42章：VoiceStyleControl 可控语音交互数据工程](part12/ch42_voice_style_control_dataset.md)
- [第43章：Latent-Switch-69K 隐式/显式推理数据工程](part12/ch43_latent_switch_69k.md)

## 第十三篇：开源大模型数据工程配方与范式

第十三篇聚焦开源大模型的数据配方、训练范式与工程化组织方式，覆盖预训练、后训练、推理强化学习、VLM、T2I/T2V 等关键方向。

- [本篇目录](part13/index.md)
- [第44章：LLM 预训练数据工程实战：从配方到落地](part13/ch44_pretrain_recipes.md)
- [第45章：LLM 后训练数据工程实战：SFT 与偏好对齐](part13/ch45_posttrain_recipes.md)
- [第46章：推理模型与 RL 数据工程：R1 / QwQ 范式](part13/ch46_rl_reasoning_data.md)
- [第47章：多模态大模型（VLM）数据配方：从预训练到视觉对齐](part13/ch47_vlm_data_recipes.md)
- [第48章：多模态生成模型数据工程：T2I 与 T2V 数据流水线](part13/ch48_t2i_t2v.md)

## 第十四篇：项目案例研究

通过 15 个可运行项目案例，把采集、清洗、合成、RAG、Agent、DataOps、隐私保护、数据飞轮、开源模型复现、视频生成数据流水线与企业级语义问数助手串成端到端实践。

- [本篇目录](part14/index.md)
- [项目一：基于 Ray 构建分布式 Mini-C4 数据流水线](part14/p01_mini_c4.md)
- [项目二：垂直领域专家 SFT（法律）](part14/p02_legal_sft.md)
- [项目三：LLaVA 多模态指令数据工厂](part14/p03_llava_instruct.md)
- [项目四：合成数学与代码教材工厂](part14/p04_synthetic_textbook.md)
- [项目五：多模态 RAG 企业财报助手](part14/p05_mm_rag.md)
- [项目六：CoT 推理数据集构建与 PRM 训练](part14/p06_prm.md)
- [项目七：Agent Tool-Use 数据工厂](part14/p07_agent_tooluse.md)
- [项目八：企业级 DataOps 平台搭建：从数据项目到组织级治理能力](part14/p08_dataops.md)
- [项目九：隐私保护数据流水线](part14/p09_privacy_pipeline.md)
- [项目十：端到端 LLM 数据飞轮](part14/p10_flywheel.md)
- [项目十一：Mini-DeepSeek 预训练复现](part14/p11_mini_deepseek.md)
- [项目十二：R1 推理飞轮](part14/p12_r1_reasoning_flywheel.md)
- [项目十三：多模态指令工厂](part14/p13_multimodal_instruction_factory.md)
- [项目十四：视频生成数据集：从视频源到可用于 T2V 训练的数据流水线](part14/p14_video_generation.md)
- [项目十五：基于 DataAgent 构建企业级语义问数助手](part14/p15_dataagent_semantic_nl2sql_agent.md)

## 附录

- [附录A：工具与框架速查表](appendix_a_tools_and_frameworks_quick_reference.md)
- [附录B：合规与上线检查清单](appendix_b_compliance_and_release_checklist.md)
- [附录C：成本估算与资源模板](appendix_c_cost_estimation_and_resource_templates.md)
- [附录D：论文到工程化转换指南](appendix_d_paper_to_implementation_guide.md)
- [附录E：常见数据工程 Bug 调试手册](appendix_e_common_bug_debugging_manual.md)
- [附录F：术语表与中英文对照](appendix_f_terminology_and_chinese_english_mapping.md)
- [附录G：MindSpore 简介与致谢](appendix_g_mindspore_note.md)
