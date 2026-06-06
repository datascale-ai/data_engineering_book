# 第43章：Latent-Switch-69K 隐式/显式推理数据工程

## 源文件

- 原始材料：`Latent-Switch-69K.docx`

## 章节定位

本章用于把 Latent-Switch-69K 整理为“推理数据压缩、隐式规划和显式推导切换”的数据工程案例。它要说明该数据集不是普通 Long-CoT 语料，而是为 LaTER 这类 latent-then-explicit reasoning 系统构建的监督语料，重点在于如何保留高层问题求解意图，同时压缩可见推理链。

本章承接第五篇合成数据工程和第六篇 CoT/推理数据工程，向后衔接第十三篇推理模型与 RL 数据工程，并为第十四篇 P06/P10/P12 的推理飞轮提供数据集级前置。

## 写作目标

- 讲清 Latent-Switch-69K 的任务目标：训练 latent-then-explicit reasoning，而不是简单收集长思维链。
- 讲清数据规模和分布：69,745 examples、难度分布、数学/代码/科学/指令等领域构成。
- 讲清蒸馏流程：从 Dolci-Think-SFT-32B 推理轨迹出发，压缩 CoT，保留 solution intuition。
- 讲清 latent placeholder 和 student sequence：`<latent_think>`、`<think>`、answer tokens 的组织方式。
- 讲清监督 mask：哪些 token 参与 loss，哪些位置只作为隐式规划槽位。

## 素材使用要求

- 必须保留 docx 中的数据规模、难度比例、领域比例、压缩率和 sequence 结构。
- 不要把 LaTER 训练细节写成完整算法复现；本章重点是数据构造、监督格式和质量控制。
- 如果补充公式或训练目标，要服务于解释样本 schema 和 mask，不要堆叠模型推导。

## 建议结构

1. 问题场景：Long-CoT 数据为什么需要压缩和隐式规划。
2. 数据集概览：规模、难度、领域和来源。
3. 蒸馏与记录形成：teacher trace、solution intuition、CoT 压缩和样本保留标准。
4. Latent budget：placeholder 数量、长度上限和序列渲染。
5. Supervision masks：loss 作用位置、边界 token、显式 reasoning 和 answer。
6. 质量控制：压缩过度、推理断裂、答案不一致、领域偏置。
7. 与前后章节回链：连接 Ch15-Ch20、Ch45-Ch46、P06、P10 和 P12。

## 篇幅要求

- 目标篇幅：14-16 页。
- 正文建议 9,000-11,000 中文字。
- 至少包含 3 张图表、2 个表格、1 个完整样本序列示例。

## 必须交付的图表

- Latent-Switch-69K 构建流水线图。
- 原始 CoT、压缩 CoT 与 latent placeholder 对比图。
- 难度和领域分布表。
- Supervision mask 示意图。
- 数据质量风险与修复动作表。

## 写作验收清单

- 是否明确 latent reasoning 数据与普通 CoT/SFT 数据的差异。
- 是否讲清 `<latent_think>` 和 `<think>` 的样本组织方式。
- 是否把数据压缩、mask 和训练目标之间的关系讲明白。
- 是否能承接前文推理数据工程，并自然导向第十三篇推理模型数据配方。
