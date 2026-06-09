# Springer 出版项目管理总表

本表为当前交付控制台，口径固定为 `14 篇、48 章、15 个项目、3 个附录`。历史 `publishing/chapters/`、`chapters_v2/`、`outlines_v2/` 和 `part11-handbook/` 保留作过程资料，不再作为当前交付结构依据。

## 一、状态定义

| 状态 | 含义 |
| --- | --- |
| 可交付 | 已通过机器校验，进入人工终稿抽检 |
| 待统稿 | 内容可构建，但仍需人工语言、逻辑和图表审校 |
| 高优先级复核 | Part 10、Part 12、Part 14 等高风险内容，终稿前优先抽检 |


## 二、当前质量门槛

- `mkdocs build --strict --clean` 必须通过。
- `python3 scripts/publish_lint.py` 目标为 `ERROR=0`、`WARN=0`。
- `python3 scripts/xref_scan.py` 目标为 `ERROR=0`；体例类 warning 已清零。
- `python3 scripts/final_publication_audit.py --report-dir publishing/final_review --fail-on-blocker` 必须通过，且报告包进入人工签核流程。

## 三、章节 / 项目状态总表

| 单元 | 所属篇 | 标题 | 文件 | 元数据 | 参考文献 | 图表状态 | 统稿状态 | 交付状态 | 优先级 | 下一步 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ch01 | 第一篇：总论与基础设施 | 第1章 大模型时代的数据变革 | `docs/zh/part1/ch01_data_change.md` | 已补摘要/关键词 | 已补 | 2 图 / 6 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch02 | 第一篇：总论与基础设施 | 第2章：LLM数据生命周期与质量评估框架 | `docs/zh/part1/ch02_quality_framework.md` | 已补摘要/关键词 | 已补 | 3 图 / 1 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch03 | 第一篇：总论与基础设施 | 第3章 AI原生数据栈与成本治理 | `docs/zh/part1/ch03_data_stack.md` | 已补摘要/关键词 | 已补 | 2 图 / 3 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch04 | 第二篇：文本预训练数据工程 | 第4章 数据源、采集与版权 | `docs/zh/part2/ch04_data_sources.md` | 已补摘要/关键词 | 已补 | 2 图 / 2 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch05 | 第二篇：文本预训练数据工程 | 第5章 清洗、去重与去污染 | `docs/zh/part2/ch05_cleaning_dedup.md` | 已补摘要/关键词 | 已补 | 2 图 / 2 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch06 | 第二篇：文本预训练数据工程 | 第6章 分词、序列化与高效加载 | `docs/zh/part2/ch06_tokenization_loading.md` | 已补摘要/关键词 | 已补 | 2 图 / 2 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch07 | 第二篇：文本预训练数据工程 | 第7章 数据评估、质量闭环与运营迭代 | `docs/zh/part2/ch07_data_operations.md` | 已补摘要/关键词 | 已补 | 2 图 / 2 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch08 | 第三篇：多模态数据工程 | 第8章 图文对数据工程 | `docs/zh/part3/ch08_multimodal_image.md` | 已补摘要/关键词 | 已补 | 3 图 / 2 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch09 | 第三篇：多模态数据工程 | 第9章 重标注与文档理解 | `docs/zh/part3/ch09_recaptioning_ocr.md` | 已补摘要/关键词 | 已补 | 2 图 / 2 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch10 | 第三篇：多模态数据工程 | 第10章 视频与音频数据工程 | `docs/zh/part3/ch10_video_audio.md` | 已补摘要/关键词 | 已补 | 4 图 / 2 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch11 | 第三篇：多模态数据工程 | 第11章 跨模态对齐与融合 | `docs/zh/part3/ch11_cross_modal_alignment.md` | 已补摘要/关键词 | 已补 | 2 图 / 3 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch12 | 第四篇：指令微调与偏好数据 | 第12章 SFT数据设计与指令体系 | `docs/zh/part4/ch12_sft.md` | 已补摘要/关键词 | 已补 | 2 图 / 2 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch13 | 第四篇：指令微调与偏好数据 | 第13章 偏好数据与奖励信号 | `docs/zh/part4/ch13_preference.md` | 已补摘要/关键词 | 已补 | 2 图 / 2 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch14 | 第四篇：指令微调与偏好数据 | 第14章 标注平台、QA体系与数据运营 | `docs/zh/part4/ch14_qa.md` | 已补摘要/关键词 | 已补 | 2 图 / 2 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch15 | 第五篇：合成数据工程 | 第15章 合成数据工厂：从种子到验证 | `docs/zh/part5/ch15_data_synthesis.md` | 已补摘要/关键词 | 已补 | 2 图 / 0 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch16 | 第五篇：合成数据工程 | 第16章 知识蒸馏与模型协作 | `docs/zh/part5/ch16_distillation.md` | 已补摘要/关键词 | 已补 | 2 图 / 0 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch17 | 第五篇：合成数据工程 | 第17章 合成数据质量控制与模型坍缩 | `docs/zh/part5/ch17_quality.md` | 已补摘要/关键词 | 已补 | 2 图 / 0 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch18 | 第六篇：推理与 Agent 数据工程 | 第18章 思维链与推理数据工程 | `docs/zh/part6/ch18_cot.md` | 已补摘要/关键词 | 已补 | 2 图 / 0 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch19 | 第六篇：推理与 Agent 数据工程 | 第19章 Tool-Use 与函数调用数据 | `docs/zh/part6/ch19_tool.md` | 已补摘要/关键词 | 已补 | 2 图 / 0 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch20 | 第六篇：推理与 Agent 数据工程 | 第20章 Agent记忆与多轮交互数据 | `docs/zh/part6/ch20_agent.md` | 已补摘要/关键词 | 已补 | 2 图 / 0 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch21 | 第七篇：应用级数据工程 | 第21章：RAG 数据流水线 | `docs/zh/part7/ch21_rag_pipeline.md` | 已补摘要/关键词 | 已补 | 9 图 / 8 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch22 | 第七篇：应用级数据工程 | 第22章：多模态 RAG 与视觉检索 | `docs/zh/part7/ch22_multimodal_rag_visual_retrieval.md` | 已补摘要/关键词 | 已补 | 5 图 / 8 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch23 | 第七篇：应用级数据工程 | 第23章：在线反馈闭环与知识更新 | `docs/zh/part7/ch23_online_feedback_knowledge_update.md` | 已补摘要/关键词 | 已补 | 4 图 / 12 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch24 | 第八篇：数据运营与平台建设 | 第24章：DataOps 飞轮与团队组织 | `docs/zh/part8/ch24_dataops_flywheel_team.md` | 已补摘要/关键词 | 已补 | 3 图 / 30 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch25 | 第八篇：数据运营与平台建设 | 第25章：数据版本管理与实验追踪 | `docs/zh/part8/ch25_data_versioning_experiment_tracking.md` | 已补摘要/关键词 | 已补 | 2 图 / 8 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch26 | 第八篇：数据运营与平台建设 | 第26章：数据平台可观测性 | `docs/zh/part8/ch26_data_platform_observability.md` | 已补摘要/关键词 | 已补 | 2 图 / 8 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch27 | 第九篇：数据资产、数据产品与数据契约 | 第27章：数据资产目录与元数据治理 | `docs/zh/part9/ch27_data_catalog_and_metadata_governance.md` | 已补摘要/关键词 | 已补 | 4 图 / 5 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch28 | 第九篇：数据资产、数据产品与数据契约 | 第28章：数据产品化与数据契约 | `docs/zh/part9/ch28_data_productization_and_data_contracts.md` | 已补摘要/关键词 | 已补 | 4 图 / 4 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch29 | 第九篇：数据资产、数据产品与数据契约 | 第29章：数据资产价值评估与复用机制 | `docs/zh/part9/ch29_data_valuation_and_reuse.md` | 已补摘要/关键词 | 已补 | 3 图 / 9 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch30 | 第九篇：数据资产、数据产品与数据契约 | 第30章：企业内部数据市场与共享治理 | `docs/zh/part9/ch30_internal_data_market_and_sharing_governance.md` | 已补摘要/关键词 | 已补 | 2 图 / 3 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch31 | 第十篇：智能化数据工程与 Data Engineering Agent | 第31章：数据工程 Agent 的架构与任务边界 | `docs/zh/part10/ch31_agent_architecture.md` | 已补摘要/关键词 | 已补 | 0 图 / 9 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch32 | 第十篇：智能化数据工程与 Data Engineering Agent | 第32章：自动化采集、解析与清洗 Agent | `docs/zh/part10/ch32_auto_collection_parsing_cleaning.md` | 已补摘要/关键词 | 已补 | 3 图 / 5 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch33 | 第十篇：智能化数据工程与 Data Engineering Agent | 第33章：标注、合成与评测 Agent | `docs/zh/part10/ch33_labeling_synthesis_evaluation.md` | 已补摘要/关键词 | 已补 | 2 图 / 8 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch34 | 第十篇：智能化数据工程与 Data Engineering Agent | 第34章：DataOps Agent 与平台自治 | `docs/zh/part10/ch34_dataops_agent.md` | 已补摘要/关键词 | 已补 | 3 图 / 5 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch35 | 第十篇：智能化数据工程与 Data Engineering Agent | 第35章：数据工程 Agent 的安全、权限与人机协同 | `docs/zh/part10/ch35_security_permission_collaboration.md` | 已补摘要/关键词 | 已补 | 3 图 / 6 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch36 | 第十一篇：隐私合规与数据安全 | 第36章：数据合规框架与治理 | `docs/zh/part11/ch36_compliance_framework_and_governance.md` | 已补摘要/关键词 | 已补 | 8 图 / 0 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch37 | 第十一篇：隐私合规与数据安全 | 第37章：联邦学习与隐私保护技术 | `docs/zh/part11/ch37_federated_learning_and_privacy_preserving_technologies.md` | 已补摘要/关键词 | 已补 | 12 图 / 0 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch38 | 第十二篇：专项数据集与数据工程实践 | 第38章：StructBill-CN 票据文档理解数据工程 | `docs/zh/part12/ch38_structbill_cn_dataset.md` | 已补摘要/关键词 | 已补 | 0 图 / 3 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch39 | 第十二篇：专项数据集与数据工程实践 | 第39章：SparseTable-Bench 表格结构鲁棒性数据工程 | `docs/zh/part12/ch39_sparse_table_bench_dataset.md` | 已补摘要/关键词 | 已补 | 3 图 / 0 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch40 | 第十二篇：专项数据集与数据工程实践 | 第40章：多图表信息图推理数据工程 | `docs/zh/part12/ch40_multi_chart_infographic_reasoning_dataset.md` | 已补摘要/关键词 | 已补 | 5 图 / 0 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch41 | 第十二篇：专项数据集与数据工程实践 | 第41章：MedImage-ToolVQA 医学图像工具调用数据工程 | `docs/zh/part12/ch41_medimage_tool_vqa_dataset.md` | 已补摘要/关键词 | 已补 | 5 图 / 0 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch42 | 第十二篇：专项数据集与数据工程实践 | 第42章：VoiceStyleControl 可控语音交互数据工程 | `docs/zh/part12/ch42_voice_style_control_dataset.md` | 已补摘要/关键词 | 已补 | 3 图 / 4 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch43 | 第十二篇：专项数据集与数据工程实践 | 第43章：Latent-Switch-69K 隐式/显式推理数据工程 | `docs/zh/part12/ch43_latent_switch_69k.md` | 已补摘要/关键词 | 已补 | 5 图 / 0 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch44 | 第十三篇：开源大模型数据工程配方与范式 | 第44章：LLM 预训练数据工程实战：从配方到落地 | `docs/zh/part13/ch44_pretrain_recipes.md` | 已补摘要/关键词 | 已补 | 5 图 / 5 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch45 | 第十三篇：开源大模型数据工程配方与范式 | 第45章：LLM 后训练数据工程实战：SFT 与偏好对齐 | `docs/zh/part13/ch45_posttrain_recipes.md` | 已补摘要/关键词 | 已补 | 3 图 / 3 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch46 | 第十三篇：开源大模型数据工程配方与范式 | 第46章：推理模型与 RL 数据工程：R1 / QwQ 范式 | `docs/zh/part13/ch46_rl_reasoning_data.md` | 已补摘要/关键词 | 已补 | 3 图 / 2 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch47 | 第十三篇：开源大模型数据工程配方与范式 | 第47章：多模态大模型（VLM）数据配方：从预训练到视觉对齐 | `docs/zh/part13/ch47_vlm_data_recipes.md` | 已补摘要/关键词 | 已补 | 4 图 / 3 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Ch48 | 第十三篇：开源大模型数据工程配方与范式 | 第48章：多模态生成模型数据工程：T2I 与 T2V 数据流水线 | `docs/zh/part13/ch48_t2i_t2v.md` | 已补摘要/关键词 | 已补 | 3 图 / 3 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| P01 | 第十四篇：项目实战 | 项目一：基于 Ray 构建分布式 Mini-C4 数据流水线 | `docs/zh/part14/p01_mini_c4.md` | 已补摘要/关键词 | 已补 | 11 图 / 0 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| P02 | 第十四篇：项目实战 | 项目二：垂直领域专家 SFT（法律） | `docs/zh/part14/p02_legal_sft.md` | 已补摘要/关键词 | 已补 | 20 图 / 0 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| P03 | 第十四篇：项目实战 | 项目三：LLaVA 多模态指令数据工厂 | `docs/zh/part14/p03_llava_instruct.md` | 已补摘要/关键词 | 已补 | 8 图 / 0 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| P04 | 第十四篇：项目实战 | 项目四：合成数学与代码教材工厂 | `docs/zh/part14/p04_synthetic_textbook.md` | 已补摘要/关键词 | 已补 | 11 图 / 1 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| P05 | 第十四篇：项目实战 | 项目五：多模态 RAG 企业财报助手 | `docs/zh/part14/p05_mm_rag.md` | 已补摘要/关键词 | 已补 | 9 图 / 1 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| P06 | 第十四篇：项目实战 | 项目六：CoT 推理数据集构建与 PRM 训练 | `docs/zh/part14/p06_prm.md` | 已补摘要/关键词 | 已补 | 10 图 / 1 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| P07 | 第十四篇：项目实战 | 项目七：Agent Tool-Use 数据工厂 | `docs/zh/part14/p07_agent_tooluse.md` | 已补摘要/关键词 | 已补 | 14 图 / 1 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| P08 | 第十四篇：项目实战 | 项目八：企业级 DataOps 平台搭建：从数据项目到组织级治理能力 | `docs/zh/part14/p08_dataops.md` | 已补摘要/关键词 | 已补 | 11 图 / 1 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| P09 | 第十四篇：项目实战 | 项目九：隐私保护数据流水线 | `docs/zh/part14/p09_privacy_pipeline.md` | 已补摘要/关键词 | 已补 | 12 图 / 1 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| P10 | 第十四篇：项目实战 | 项目十：端到端 LLM 数据飞轮 | `docs/zh/part14/p10_flywheel.md` | 已补摘要/关键词 | 已补 | 10 图 / 1 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| P11 | 第十四篇：项目实战 | 项目十一：Mini-DeepSeek 预训练复现 | `docs/zh/part14/p11_mini_deepseek.md` | 已补摘要/关键词 | 已补 | 1 图 / 1 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| P12 | 第十四篇：项目实战 | 项目十二：R1 推理飞轮 | `docs/zh/part14/p12_r1_reasoning_flywheel.md` | 已补摘要/关键词 | 已补 | 1 图 / 1 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| P13 | 第十四篇：项目实战 | 项目十三：多模态指令工厂 | `docs/zh/part14/p13_multimodal_instruction_factory.md` | 已补摘要/关键词 | 已补 | 1 图 / 1 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| P14 | 第十四篇：项目实战 | 项目十四：视频生成数据集：从视频源到可用于 T2V 训练的数据流水线 | `docs/zh/part14/p14_video_generation.md` | 已补摘要/关键词 | 已补 | 2 图 / 0 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| P15 | 第十四篇：项目实战 | 项目十五：基于 DataAgent 构建企业级语义问数助手 | `docs/zh/part14/p15_dataagent_semantic_nl2sql_agent.md` | 已补摘要/关键词 | 已补 | 2 图 / 0 表；已入台账 | 高优先级复核 | 可交付 | 高 | 终稿抽检语言、图源、alt text 与代码长度 |
| Appendix a_tools_and_frameworks_quick_reference | 附录 | 附录A：工具与框架速查表 | `docs/zh/appendix_a_tools_and_frameworks_quick_reference.md` | 已补摘要/关键词 | 已补 | 0 图 / 1 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Appendix b_compliance_and_release_checklist | 附录 | 附录B：合规与上线检查清单 | `docs/zh/appendix_b_compliance_and_release_checklist.md` | 已补摘要/关键词 | 已补 | 0 图 / 2 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |
| Appendix c_cost_estimation_and_resource_templates | 附录 | 附录C：成本估算与资源模板 | `docs/zh/appendix_c_cost_estimation_and_resource_templates.md` | 已补摘要/关键词 | 已补 | 0 图 / 3 表；已入台账 | 常规终稿复核 | 可交付 | 常规 | 终稿抽检语言、图源、alt text 与代码长度 |

## 四、高优先级抽检清单

| 范围 | 抽检重点 | 当前动作 |
| --- | --- | --- |
| Part 10 | Agent 自动化、权限、安全、人机协同 | 复核图源、权限边界、案例语气和参考文献口径 |
| Part 12 | 专项数据集真实性、图表版权、评测指标 | 抽检第 40 章和所有外部素材，确认图源和数据集链接 |
| Part 14 | 15 个项目从教程转为案例研究 | 抽检 P09、P12、P15 的代码长度、失败模式和复现边界 |


## 五、最终交付动作

1. 再跑四项机器校验：MkDocs、publish lint、xref scan、final publication audit。
2. 生成并复核 `publishing/final_review/` 报告包。
3. 抽检第 12、16、21、24、29、40 章与 P11、P12、P13、P15。
4. 将图表高清源文件、AI 使用声明、alt text 表和参考文献清单打包给出版社。
5. 确认 `outputs/`、PPT 产物、内部规划材料不进入出版构建。
