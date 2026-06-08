# 出版社交付前总检查清单

本文件用于 Springer 交付前终稿阶段做最后核对；当前交付口径为 14 篇、48 章、15 个项目、3 个附录。

## 一、结构检查

- [ ] 目录、章节编号、标题一致
- [ ] 48 章 + 15 项目 + 3 附录全部齐全
- [ ] 各章前后顺序无错位
- [ ] 章节间引用关系已核对，`scripts/xref_scan.py` 无 ERROR
- [ ] 中文主线、MkDocs 导航、出版台账和宣传文案口径一致

## 二、内容检查

- [ ] 每章均有 SpringerLink 友好的摘要与 3-6 个关键词
- [ ] 每章均有参考文献或官方资料来源
- [ ] 第 12 篇专项数据集能回扣前文方法论
- [ ] 第 13 篇开源配方突出长期范式，不只堆叠热点模型
- [ ] 第 14 篇项目章已改成案例研究写法，不是教程写法
- [ ] 重点抽检第 12、16、21、24、29、40 章与 P11、P12、P13、P15

## 三、图表检查

- [ ] 图表编号连续，且图号/表号与所在章节一致
- [ ] 图题、表题完整，正文中有首次引用
- [ ] 图表来源、自绘/改绘状态和权限已登记
- [ ] 清晰度、字号、线条和色彩区分满足纸书要求
- [ ] 每章图表均有 alt text 台账

## 四、引用与版权检查

- [ ] 参考文献台账完整，无“待补”字段
- [ ] 图源台账完整，外部图表已核查可出版性
- [ ] AI 生成或 AI 辅助图像使用情况已披露并按 Springer 政策处理
- [ ] 所有直接引用和关键事实均可回溯

## 五、文字与风格检查

- [ ] 术语写法统一，缩写首次出现有中英文全称
- [ ] 没有明显网页教程式、课程讲义式或口语化残留表达
- [ ] `publishing/final_review/chapter_style_checklist.md` 已逐章签核
- [ ] `publishing/final_review/style_report.md` 中候选表达已完成保留/改写判断
- [ ] 大段代码已外置，正文代码只保留关键片段
- [ ] 章首问题意识、章末小结和适用边界完整
- [ ] 常见误区、风险边界和合规提示未缺失

## 六、配套资源检查

- [ ] 每章都有配套资源或复现说明
- [ ] 仓库中的章节映射关系清楚
- [ ] 附录 C 已说明正文与仓库的分工
- [ ] `docs/superpowers/`、`outputs/`、PPT 产物不进入出版构建

## 七、对外材料检查

- [ ] 作者名单、单位、ORCID 和简介齐全
- [ ] 书籍简介、卖点摘要、关键词和 SEO 元数据可单独抽出
- [ ] 封底文案 / 宣传语已有终稿
- [ ] 试读反馈关键结论已沉淀并处理

## 八、最终交付件

- [ ] 正文终稿
- [ ] 图表终稿与高清源文件
- [ ] `publishing/final_review/figure_rights_report.md` 中每张图的版权、AI 生成/辅助声明和高清源文件已签核
- [ ] 参考文献清单
- [ ] `publishing/final_review/reference_audit_report.md` 中每条参考文献 DOI / URL / 年份 / Springer 样式 / 真实性已终审
- [ ] `publishing/final_review/manual_review_checklist.md` 中第 12、16、21、24、29、40 章，P11、P12、P13、P15，以及 Part 10/12/14 高风险内容已签核
- [ ] 作者信息清单
- [ ] 配套资源说明
- [ ] alt text 表
- [ ] AI 使用声明
- [ ] Springer 要求的其他元数据

## 九、使用规则

- 所有复选项必须由主编最终确认。
- 有任何一项未完成，不建议直接交出版社。
- 交付前必须通过 `mkdocs build --strict --clean`、`python3 scripts/publish_lint.py`、`python3 scripts/xref_scan.py` 和 `python3 scripts/final_publication_audit.py --report-dir publishing/final_review --fail-on-blocker`。
