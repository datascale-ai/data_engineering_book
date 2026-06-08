# 《大模型数据工程》Springer 出版控制台

本目录当前以 Springer 中文交付稿为准，冻结口径为 `14 篇、48 章、15 个项目、3 个附录`。历史规划材料仍保留作过程追溯，但不再代表当前目录结构。

## 当前主线

- [01_editorial_blueprint.md](01_editorial_blueprint.md)：当前 Springer 交付蓝图。
- [12_figures_tables_register.md](12_figures_tables_register.md)：图表、图源、权限和 alt text 台账。
- [13_references_register.md](13_references_register.md)：全书参考文献台账。
- [15_final_delivery_checklist.md](15_final_delivery_checklist.md)：出版社交付前检查清单。
- [16_project_management_dashboard.md](16_project_management_dashboard.md)：48 章、15 项目、3 附录状态总表。
- [17_final_publication_review_protocol.md](17_final_publication_review_protocol.md)：终稿人工复核协议。
- [final_review/](final_review/)：终稿人工复核报告包，由 `scripts/final_publication_audit.py` 生成。

## 历史资料边界

- [chapters/](chapters/)、[chapters_v2/](chapters_v2/)、[outlines_v2/](outlines_v2/)、[writing_plans/](writing_plans/) 与 [part11-handbook/](part11-handbook/) 保留为历史规划和任务书档案。
- 上述历史目录中的 28 章、10 项目、6 附录或旧 Part XI 口径不作为当前交付依据。
- 当前交付验收只看 `docs/zh/`、`mkdocs.yml`、`scripts/publish_lint.py`、`scripts/xref_scan.py` 与本目录的主动台账文件。

## 使用方式

1. 先运行 `mkdocs build --strict --clean`，确认构建范围、导航、链接和 i18n 正常。
2. 运行 `python3 scripts/publish_lint.py`，确认摘要、关键词、参考文献、断图和图号全部清零。
3. 运行 `python3 scripts/xref_scan.py`，确认篇章、章节和项目引用无 dangling 目标。
4. 运行 `python3 scripts/final_publication_audit.py --report-dir publishing/final_review`，生成逐章统稿、图表、参考文献和高风险抽检证据包。
5. 按 `15_final_delivery_checklist.md` 和 `17_final_publication_review_protocol.md` 做人工签核与出版社交付包整理。
