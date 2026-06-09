# English Translation Style Guide

This guide is the editorial baseline for the English edition of *Data Engineering for Large Models: Architecture, Algorithms & Projects*. It is not part of the reader-facing table of contents; it exists to keep chapter translations, navigation labels, captions, and project pages consistent.

## Editorial Goals

- Translate for technical readers, not word-for-word bilingual comparison.
- Preserve the Chinese edition's structure, examples, and argument order unless a sentence must be reshaped for natural English.
- Use concise technical prose. Avoid inflated phrasing such as "empower," "ecological construction," or "landing implementation" unless the Chinese text explicitly discusses business positioning.
- Keep Markdown structure stable: headings, code blocks, tables, admonitions, image paths, and internal links must remain valid after translation.
- Keep existing high-quality English chapters unless they conflict with the 2026 structure or terminology.

## Core Terms

| Chinese | Preferred English |
| --- | --- |
| 大模型 | large model |
| 大语言模型 | large language model / LLM |
| 大模型数据工程 | large-model data engineering |
| 数据生命周期 | data lifecycle |
| 质量闭环 | quality closed loop |
| 数据飞轮 | data flywheel |
| 数据栈 | data stack |
| 成本治理 | cost governance |
| 数据资产 | data asset |
| 数据产品 | data product |
| 数据契约 | data contract |
| 数据目录 | data catalog |
| 元数据治理 | metadata governance |
| 指令微调 | instruction fine-tuning |
| 偏好数据 | preference data |
| 奖励信号 | reward signal |
| 合成数据 | synthetic data |
| 知识蒸馏 | knowledge distillation |
| 思维链 | chain-of-thought |
| 推理轨迹 | reasoning trace |
| 工具调用 | tool use / tool calling |
| 函数调用 | function calling |
| 多轮交互 | multi-turn interaction |
| 应用级数据工程 | application-level data engineering |
| 在线反馈闭环 | online feedback closed loop |
| 数据运营 | data operations |
| 平台建设 | platform development |
| 可观测性 | observability |
| 智能化数据工程 | agentic data engineering |
| 数据工程 Agent | data engineering agent |
| 人机协同 | human-AI collaboration |
| 隐私合规 | privacy and compliance |
| 数据安全 | data security |
| 联邦学习 | federated learning |
| 隐私保护技术 | privacy-preserving technologies |
| 专项数据集 | specialized dataset |
| 项目实战 | hands-on projects |

## Structure Terms

Use `Part` for 篇 and `Chapter` for 章. Use `Project` for 项目. Avoid `Article`, `Volume`, or `Title` in navigation labels.

Use these fixed part titles:

| ID | English title |
| --- | --- |
| Part 1 | Overview and Infrastructure |
| Part 2 | Text Pre-training Data Engineering |
| Part 3 | Multimodal Data Engineering |
| Part 4 | Instruction Fine-tuning and Preference Data |
| Part 5 | Synthetic Data Engineering |
| Part 6 | Reasoning and Agent Data Engineering |
| Part 7 | Application-Level Data Engineering |
| Part 8 | Data Operations and Platform Development |
| Part 9 | Data Assets, Data Products, and Data Contracts |
| Part 10 | Agentic Data Engineering and Data Engineering Agents |
| Part 11 | Privacy, Compliance, and Data Security |
| Part 12 | Specialized Datasets and Data Engineering Practice |
| Part 13 | Open-Source Large-Model Data Engineering Recipes and Paradigms |
| Part 14 | Hands-on Projects |

## Quality Gates

Every translated batch must pass:

```bash
python3 scripts/audit_english_release.py
```

The full site must also build before release:

```bash
python3 -m mkdocs build --clean
```

When the audit reports Chinese characters in examples, bibliographic titles, dataset names, or image filenames, either translate the surrounding prose and keep the literal name intentionally, or add the case to the audit script's allowed exceptions with a short comment.
