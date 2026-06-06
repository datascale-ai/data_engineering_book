# 第39章：SparseTable-Bench 表格结构鲁棒性数据工程

## 源文件

- 原始材料：`SparseTable_Bench_Dataset.docx`

## 章节定位

本章用于把 SparseTable-Bench 整理为“表格识别鲁棒性与结构评测”的数据工程章节。它应强调表格不是普通 OCR 文本，而是同时包含 HTML 结构、单元格文本、空间 bbox、空单元格和遮挡压力测试的复合对象。

本章承接第三篇的文档理解和跨模态对齐，承接第十二篇上一章的票据/表格抽取案例，并为第十三篇 VLM 数据配方中的文档、表格和视觉对齐任务提供数据集级支撑。

## 写作目标

- 讲清 SparseTable-Bench 的核心任务：table structure recognition、geometry-aware annotation、mask-stress robustness。
- 讲清三类监督信号：HTML structure sequence、cell-level textual content、fine-grained spatial bounding boxes。
- 讲清空单元格和稀疏布局为什么会破坏传统 TSR benchmark 的评测可靠性。
- 讲清 STB-Mask-Stress 的压力测试设计，以及它如何检验模型在严重信息缺失下的结构恢复能力。
- 讲清 TEDS/TEDS-S 等指标的使用边界和错误解释方法。

## 素材使用要求

- 必须保留 docx 中的数据规模、数据切分、标注对象、STB-Mask-Stress 和评价协议。
- 可以补充表格识别领域常见术语，但不要把本章写成 OCR 或 TSR 文献综述。
- 所有算法和指标描述都要回到“这个数据集如何被构建、评测和复现”。

## 建议结构

1. 问题场景：为什么表格结构识别需要专门 benchmark。
2. 数据集概览：样本规模、数据切分、标准测试与压力测试。
3. 标注 schema：HTML、文本、bbox、空单元格和几何拓扑。
4. 构建流水线：表格收集、结构标注、bbox 校验、空单元格处理、质检。
5. STB-Mask-Stress：mask 构造、压力测试目的和评测解释。
6. 评测协议：TEDS、TEDS-S、结构错误、文本错误和空间错误。
7. 与前后章节回链：连接 Ch09、Ch11、Ch38、Ch47 和 P03/P05。

## 篇幅要求

- 目标篇幅：12-14 页。
- 正文建议 8,000-10,000 中文字。
- 至少包含 3 个表格、2 张结构/流程图、1 个标注样例。

## 必须交付的图表

- SparseTable-Bench 四阶段构建流水线图。
- 表格样本三类监督信号结构图。
- STB-Mask-Stress 遮挡生成与评测流程图。
- 标注字段与质量检查规则表。
- TEDS/TEDS-S 适用边界和失败类型对照表。

## 写作验收清单

- 是否区分“表格内容识别”和“表格结构识别”。
- 是否说明 bbox、HTML 和空单元格三者为什么必须同时存在。
- 是否给出压力测试的工程意义，而不只是列出测试名称。
- 是否说明该数据集如何用于 VLM/文档模型的训练和评测。
