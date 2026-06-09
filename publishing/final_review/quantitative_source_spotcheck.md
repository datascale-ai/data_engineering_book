# 量化数据来源抽样核验记录

核验时间：2026-06-08。

本记录补充 `quantitative_data_audit.md`，用于保存出版前已经做过的外部来源抽样核验。它不替代全书逐条人工终审。

## 已核验来源

| 对象 | 涉及章节 | 核验结果 | 出版处理 |
| --- | --- | --- | --- |
| `Tioe/LATENT-SWITCH-69K` | `docs/zh/part12/ch43_latent_switch_69k.md` | Hugging Face API 返回公开数据集，license 为 Apache-2.0，format 为 Parquet，数据卡描述中列出 `Samples: 69,745`、`File: sft_train.parquet`、`Columns: 31`、`Generated at: 2026-04`。 | 支撑 69,745 条样本这一规模描述；终稿参考文献中保留 Hugging Face 数据集 URL。 |
| `xychen-zh/multi-chart-infographic-reasoning-dataset` | `docs/zh/part12/ch40_multi_chart_infographic_reasoning_dataset.md` | GitHub 仓库可访问，但 `gh api repos/.../contents` 返回空仓库；当前没有数据卡、样本 manifest、许可证文件或发布文件可核对 `354` 张图、`1917` 条问题、`28` 个领域等数字。 | 正文已改为“数据构建案例”口径；公开仓库、数据卡、manifest 和许可证补齐前，不应表述为已正式发布的公开 benchmark。 |

## 本地项目数值处理

本轮 repo 内检索没有发现能独立支撑以下项目章运行数字的 JSON/CSV 日志或 manifest：

- `docs/zh/part14/p02_legal_sft.md`：`2577`、`7737`、`3882`、`1710`、`855`、`50` 条抽样等。
- `docs/zh/part14/p04_synthetic_textbook.md`：`30-60 秒`、`1000 条`、`1 美元`等。
- `docs/zh/part14/p05_mm_rag.md`：召回页码 `49, 91, 130, 8`。
- `docs/zh/part14/p07_agent_tooluse.md`：`22` 条原始轨迹、`103` 条训练记录。
- `docs/zh/part14/p11_mini_deepseek.md`：`500,000` 抽样、`4.2%` 去重率、`5GB`、`1.05B Tokens` 等。
- `docs/zh/part14/p13_multimodal_instruction_factory.md`：`4.3 / 5.0` 平均分等。

这些正文已改为“示例”“教学化示例”“正式交付需以 manifest / 日志 / API 账单替换”的口径。若后续恢复为真实项目结果，必须把运行脚本、随机种子、样本 ID、评分结果、账单或数据 manifest 纳入交付包。
