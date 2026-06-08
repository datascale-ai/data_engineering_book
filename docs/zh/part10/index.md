# 第十篇：智能化数据工程与 Data Engineering Agent

## 本篇定位

第十篇讨论 Agentic Data Engineering，聚焦数据工程 Agent 如何参与采集、解析、清洗、标注、合成、评测、DataOps 与安全协同。本篇以 DataAgent 作为贯穿式工程参照：第31章先把它放进架构与边界框架，后续章节再分别讨论采集清洗、评测、DataOps 和安全协同如何接入这种 Agent 数据工程底座。

## 术语口径

本篇统一使用“数据工程 Agent”描述能够在权限边界内执行数据任务、调用工具、记录过程并接受审计的智能组件；使用“DataAgent”描述贯穿本篇的工程参照系统；使用“人机协同”描述 Agent、人工审核、平台策略和安全门禁之间的分工。Agent 自动化不等同于无人值守，涉及采集、清洗、合成、评测和安全操作时必须说明权限、回滚和审计机制。

## 本篇目录

- [第31章：数据工程 Agent 的架构与任务边界](ch31_agent_architecture.md)
- [第32章：自动化采集、解析与清洗 Agent](ch32_auto_collection_parsing_cleaning.md)
- [第33章：标注、合成与评测 Agent](ch33_labeling_synthesis_evaluation.md)
- [第34章：DataOps Agent 与平台自治](ch34_dataops_agent.md)
- [第35章：数据工程 Agent 的安全、权限与人机协同](ch35_security_permission_collaboration.md)

## 建议阅读顺序

- 先读第31章，明确数据工程 Agent 的架构、任务边界和自动化等级。
- 再读第32章到第34章，理解采集清洗、标注合成评测和 DataOps 自治的核心闭环。
- 最后读第35章，把权限、安全、审计和人机协同作为上线前门禁。
