# 第41章：Ophiuchus 工具集成医学 VQA 数据工程

## 源文件

- 原始材料：`Ophiuchus-Tool-Integrated-VQA-Dataset.docx`

## 章节定位

本章用于把 Ophiuchus Tool-Integrated Medical VQA Dataset 整理为“医学图像 Agent 工具调用数据”的章节。它不是普通医学 VQA 数据集，而是面向会主动调用视觉工具的医疗图像智能体，记录问题、局部视觉证据、工具轨迹、观察结果和回答监督。

本章承接第三篇多模态数据工程、第六篇 Tool-Use 与 Agent 数据、第十篇 Data Engineering Agent 的工具边界和安全协同、第十一篇隐私合规与医学数据风险。向后支撑第十三篇 VLM/多模态指令数据配方和第十四篇 Agent Tool-Use、多模态指令工厂项目。

## 写作目标

- 讲清该数据集的核心差异：answer supervision + tool-behavior supervision。
- 讲清数据规模和字段：final parquet、record 数、data_source、ability、source、raw_images 等。
- 讲清工具链：BioMedParse、Zoom-in、SAM2 等工具如何生成 ROI、mask、bbox 和观察图像。
- 讲清轨迹生成：assistant tool call、user observation image、多轮 continuation 和最终答案。
- 讲清医学场景的安全边界：隐私、诊断风险、工具错误、区域误定位和人工复核。

## 素材使用要求

- 必须以 docx 中描述的脚本链路和记录规模为基础，例如 merge、make_vqa、verify、makereasoning、make_sft 等步骤。
- 不要把医学内容写成诊疗建议；本章只讨论数据工程和模型训练/评测。
- 所有涉及医学图像和工具调用的内容，都要补充合规、脱敏和审计说明。

## 建议结构

1. 问题场景：医学 VQA 为什么需要工具集成轨迹。
2. 数据集概览：规模、字段、图像来源、ROI 样本和任务类型。
3. 工具链与视觉证据：mask、bbox、segmentation、zoom-in 和 observation image。
4. 轨迹合成流水线：QA 生成、校验、工具路径合成、SFT 记录形成。
5. 质量控制：schema 校验、区域 grounding、选项质量、答案一致性、工具调用有效性。
6. 安全与合规：医学隐私、误诊风险、工具越权、人工复核门禁。
7. 与前后章节回链：连接 Ch19-Ch20、Ch31-Ch35、Ch36-Ch37、Ch47、P07 和 P13。

## 篇幅要求

- 目标篇幅：16-18 页。
- 正文建议 10,000-12,000 中文字。
- 至少包含 3 张流程/架构图、2 个表格、1 个多轮工具轨迹样例。

## 必须交付的图表

- Ophiuchus 数据构建流水线图。
- 医学 VQA 工具调用轨迹结构图。
- 工具类型、输入输出和风险边界表。
- 质量校验与人工复核流程图。
- SFT 多轮记录 schema 示例。

## 写作验收清单

- 是否讲清“医学 VQA + 工具轨迹”与普通 VQA 的差异。
- 是否把工具调用、观察图像和最终回答组织成训练样本逻辑。
- 是否明确医学数据的隐私和安全边界。
- 是否能自然衔接 Agent 数据工程和后续项目实战。
