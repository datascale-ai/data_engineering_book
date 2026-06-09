# Part XII: Specialized Datasets and Data Engineering Practice

## Positioning of This Part

Part XII serves as the methodology validation function for the book as a whole. The preceding eleven parts have established frameworks covering the data lifecycle, text and multimodal processing, alignment data, reasoning data, RAG, DataOps, data asset management, Agent automation, and compliance governance. This part grounds those frameworks in concrete data objects, examining their applicability to scenarios such as invoice documents, sparse tables, compound charts, medical images, controllable speech, and reasoning traces.

This part does not aim to enumerate dataset names; rather, its thread is "how specialized datasets are defined, constructed, evaluated, published, and reproduced." Each chapter must answer four questions: what is the structure of the data object, how does the construction pipeline control quality, how does the evaluation protocol support task conclusions, and how do permission and risk boundaries affect public release.

Looking backward, this part builds on Part III (multimodal data), Part V (synthetic data), Part VI (tool and reasoning data), Part VIII (data operations), and Part XI (compliance governance). Looking forward, it provides citable engineering evidence for Part XIII (open-source model data recipes) and Part XIV (project case studies).

## Terminology Conventions

Throughout this part, "specialized dataset" consistently refers to data assets constructed around a specific task, scenario, or evaluation protocol; "sample schema" refers to the structural convention covering fields, inputs, outputs, supervision signals, and quality labels; and "evaluation protocol" refers to metrics, splits, baselines, error attribution, and reproduction conditions. Each case study should make explicit the engineering problem the dataset addresses, rather than merely introducing its name, scale, or model task.

## Table of Contents for This Part

- [Chapter 38: StructBill-CN Invoice Document Understanding Data Engineering](ch38_structbill_cn_dataset.md)
- [Chapter 39: SparseTable-Bench Table Structure Robustness Data Engineering](ch39_sparse_table_bench_dataset.md)
- [Chapter 40: Multi-Chart Infographic Reasoning Data Engineering](ch40_multi_chart_infographic_reasoning_dataset.md)
- [Chapter 41: MedImage-ToolVQA Medical Image Tool-Calling Data Engineering](ch41_medimage_tool_vqa_dataset.md)
- [Chapter 42: VoiceStyleControl Controllable Speech Interaction Data Engineering](ch42_voice_style_control_dataset.md)
- [Chapter 43: Latent-Switch-69K Implicit/Explicit Reasoning Data Engineering](ch43_latent_switch_69k.md)

## Reading Order

Chapters 38 through 40 revolve around visual documents, sparse tables, and compound charts, and are best read in conjunction with Part III's coverage of OCR, multimodal imagery, and cross-modal alignment.

Chapters 41 and 42 move into medical image tool-calling and controllable speech interaction respectively, and are best connected with Part VI's Agent data, Part X's Data Engineering Agent, and Part XI's privacy compliance.

Chapter 43 concludes with implicit/explicit reasoning switching and reasoning trace compression, leading naturally into Part XIII's reasoning models, RL data engineering, and the R1 reasoning flywheel case study in Part XIV.

## Unified Review Criteria

When reading this part, priority should be given to examining each case study's data definition, schema, construction pipeline, quality control, evaluation protocol, chart provenance, references, and reproduction boundaries.

For cases involving invoices, medical images, voice identity, and reasoning traces, data authorization, privacy protection, misuse risks, and human review mechanisms must also be documented. Only when these boundaries are clearly articulated does a specialized dataset meet the conditions for inclusion in a published manuscript, course experiment, or public reproduction effort.
