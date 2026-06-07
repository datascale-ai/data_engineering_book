# Part 12: Specialized Datasets and Data Engineering Practice

## Positioning

Part 12 uses representative specialized datasets to discuss how data engineering methods are organized in real tasks. Unlike earlier methodological and platform chapters, this part starts from concrete data objects and clarifies the relationships among task definition, sample structure, construction pipeline, quality control, evaluation protocol, risk boundary, and reproducibility.

This part carries forward the earlier discussions of multimodal data, document understanding, tool use, voice interaction, reasoning data, data-version governance, and privacy compliance. It grounds those capabilities in typical scenarios such as visual documents, sparse tables, compound charts, medical VQA, controllable speech, and reasoning data. It also connects forward to Part 13 and Part 14 by providing case references for multimodal RAG, Agent Tool-Use, VLM data recipes, voice generation, and reasoning flywheels.

## Part Contents

- [Chapter 38: StructBill-CN Bill Document Understanding Data Engineering](ch38_structbill_cn_dataset.md)
- [Chapter 39: SparseTable-Bench Table-Structure Robustness Data Engineering](ch39_sparse_table_bench_dataset.md)
- [Chapter 40: Multi-Chart Infographic Reasoning Data Engineering](ch40_multi_chart_infographic_reasoning_dataset.md)
- [Chapter 41: MedImage-ToolVQA Medical Image Tool-Use VQA Data Engineering](ch41_medimage_tool_vqa_dataset.md)
- [Chapter 42: VoiceStyleControl Controllable Voice Interaction Data Engineering](ch42_voice_style_control_dataset.md)
- [Chapter 43: Latent-Switch-69K Implicit/Explicit Reasoning Data Engineering](ch43_latent_switch_69k.md)

## Suggested Reading Order

- Chapters 38 through 40 progress from visual documents to sparse tables and compound charts, making them a natural continuation of OCR, document understanding, multimodal alignment, and evidence organization.
- Chapters 41 and 42 move into medical image tool-use VQA and controllable voice interaction, connecting to agent data, Data Engineering Agents, and compliance/security.
- Chapter 43 closes with reasoning-data compression and implicit/explicit reasoning switches, connecting forward to reasoning models, RL data engineering, and the R1 reasoning flywheel project.

## Unified Reading Notes

- Focus on the concrete data engineering problem each case solves, not only the dataset name or model task.
- Read each chapter through four lenses: sample schema, construction pipeline, quality control, and evaluation protocol.
- For cases involving bills, medical images, voice identity, and reasoning traces, pay attention to privacy, authorization, audit, and misuse risks.
- These cases can serve both as implementations of earlier methods and as prerequisites for later practical projects and open-source model data recipes.
