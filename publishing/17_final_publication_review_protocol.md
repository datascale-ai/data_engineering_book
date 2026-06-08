# Springer 终稿人工复核协议

本文件把终稿阶段的人工工作转为可执行、可追踪的流程。机器校验只负责发现候选问题和生成证据包，最终签核仍需主编、资料编辑、图表编辑和参考文献编辑共同完成。

## 一、执行命令

```bash
python3 scripts/final_publication_audit.py --report-dir publishing/final_review
```

生成文件：

- `publishing/final_review/style_report.md`
- `publishing/final_review/chapter_style_checklist.md`
- `publishing/final_review/figure_rights_report.md`
- `publishing/final_review/reference_audit_report.md`
- `publishing/final_review/manual_review_checklist.md`
- `publishing/final_review/final_publication_audit.json`

## 二、人工签核口径

| 工作项 | 机器输出 | 人工确认标准 | 签核角色 |
| --- | --- | --- | --- |
| 全书逐章统稿和风格精修 | `chapter_style_checklist.md`、`style_report.md` | 逐章确认语言、逻辑、术语、图表引用、参考文献和出版体例；判断口语化、博客化、课程化表达是否需要改写，保留必要技术解释，不机械替换 | 主编 / 章节作者 |
| 图表版权、AI 声明和高清源文件 | `figure_rights_report.md` | 每张图确认来源、授权、AI 生成或辅助情况、高清源文件、可出版分辨率和替换方案 | 图表编辑 / 资料编辑 |
| 参考文献 DOI / Springer 样式 / 真实性 | `reference_audit_report.md` | 每条参考文献确认真实存在、年份、作者、题名、DOI/URL/arXiv、出版社样式和正文引用关系 | 参考文献编辑 |
| 指定章节抽检 | `manual_review_checklist.md` | 第 12、16、21、24、29、40 章，P11、P12、P13、P15，以及 Part 10/12/14 高风险范围完成签核 | 主编 / 领域审校 |

## 三、完成定义

只有同时满足以下条件，才可声明“出版社终稿级人工复核完成”：

- `mkdocs build --strict --clean` 通过。
- `python3 scripts/publish_lint.py` 为 `ERROR=0, WARN=0`。
- `python3 scripts/xref_scan.py` 为 `ERROR=0, WARN=0`。
- `python3 scripts/final_publication_audit.py --report-dir publishing/final_review --fail-on-blocker` 通过。
- `publishing/final_review/manual_review_checklist.md` 中指定章节和高风险范围完成线下签核。
- 图表高清源、AI 使用声明、参考文献最终清单和作者元数据已进入出版社交付包。
