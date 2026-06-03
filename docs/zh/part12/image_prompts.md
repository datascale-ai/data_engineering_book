# 第十二篇配图 Prompt（中英双语，图内文字英文版）

本文档为第十二篇全部配图提供一一对应的绘图 prompt，便于后续统一补图。默认建议所有图片采用同一视觉规范：

- 白色背景，学术出版风格，干净专业的信息图。
- 以红、黑、灰为主色，少量浅蓝或浅金作为辅助高亮。
- 除 `chj_02_dataset_card_template.png` 外，其余所有图片的图内文字、标题、图例、坐标轴、表头、气泡说明均使用英文。
- 强调高可读性、清晰网格、细线条、矢量感、可印刷、无装饰噪声。
- 默认横向版式，适合书页插图，推荐 16:9 或 4:3。

## ChJ 数据工程数据集：从 Dataset Card 到 Benchmark Card

### 图J-1 / `chj_01_four_cards_system.png`

对应位置：ChJ 开篇“四卡体系”总览图。

中文 Prompt：
绘制一张专业学术信息图，白底，红黑灰配色，展示四卡体系。左侧是原始数据集与数据来源，中间依次是 Dataset Card、Algorithm Card、Experiment Card、Benchmark Card 四张卡片，右侧是工程资产落地，包括 training、evaluation、governance、open release。四张卡片之间用箭头连接形成一条治理链。每张卡片下方用英文标注职责，如 task and schema、method interface、controls and attribution、open governance and leaderboard operations。图内所有可见文字必须是英文，主标题使用英文 “From Dataset Description to Engineering Assets: The Four-Card System”，整体布局横向，风格简洁、矢量化、适合教材印刷。

English Prompt:
Create a clean academic infographic on a white background with a red, black, and gray palette. Show a horizontal pipeline of four cards: Dataset Card, Algorithm Card, Experiment Card, and Benchmark Card. On the left, place raw datasets and data sources; on the right, place engineering outcomes such as training, evaluation, governance, and open release. Connect the four cards with arrows to form one governance chain. Add English role labels under each card, such as task and schema, method interface, controls and attribution, and open governance and leaderboard operations. All visible text in the image must be in English, and the main title should be “From Dataset Description to Engineering Assets: The Four-Card System”. Use a vector-like textbook style, high legibility, and minimal decoration.

### 图J-2 / `chj_02_dataset_card_template.png`

对应位置：ChJ 的 Dataset Card 字段模板。

中文 Prompt：
绘制一张数据集卡片模板信息图，白底、印刷友好风格，中间是一张大卡片，分成七个模块：任务范围、样本结构、来源与许可、构造流程、统计分布、已知缺陷、推荐使用方式。每个模块用浅灰框分区，关键字段用深红标题，内部用简洁项目符号显示示例字段，如 schema、license、difficulty、known issues、recommended metrics。右侧可补充小图标表示文本、文档、语音、多模态等数据类型。整体像标准表单模板，标题为“Dataset Card 字段模板”，强调结构化、工程化、规范化。

English Prompt:
Design a structured template infographic for a Dataset Card on a white background, print-friendly and academic. Put one large central card divided into seven modules: task scope, sample structure, source and license, construction process, statistical distribution, known defects, and recommended usage. Use light gray panel boxes, dark red section headers, and concise bullet-style field examples such as schema, license, difficulty, known issues, and recommended metrics. Optionally place small icons for text, documents, speech, and multimodal data on the side. The visual should feel like a formal engineering template, with the Chinese title “Dataset Card 字段模板”.

### 图J-3 / `chj_03_data_algorithm_mapping.png`

对应位置：ChJ 中“数据形状与算法接口匹配图”。

中文 Prompt：
绘制一张映射关系图，展示不同数据形状与算法接口的匹配关系。左列用英文列出六类数据：receipt structuring、multimodal tables、tool trajectories、RAG evidence、reasoning switch、speech style control；右列用英文列出对应算法接口：schema-bound generation、geometry-aware parsing、tool-aware policy learning、retrieval-plus-citation、latent-switch reasoning、speaker-emotion controlled generation。中间用箭头和不同线型连接，突出“同样是数据，不同 shape 对应不同算法接口”。图内所有文字必须为英文，主标题用英文 “Data Shapes and Algorithm Interface Mapping”。

English Prompt:
Create a mapping diagram showing how different data shapes match different algorithm interfaces. In the left column, list six data families in English: receipt structuring, multimodal tables, tool trajectories, RAG evidence, reasoning switch, and speech style control. In the right column, list the corresponding algorithm interfaces in English: schema-bound generation, geometry-aware parsing, tool-aware policy learning, retrieval-plus-citation, latent-switch reasoning, and speaker-emotion controlled generation. Use arrows and varied line styles between columns to emphasize that one data shape does not imply one universal algorithm. All visible text must be in English, and the title should be “Data Shapes and Algorithm Interface Mapping”.

### 图J-4 / `chj_04_benchmark_governance_loop.png`

对应位置：ChJ 中 Benchmark Card 与 Leaderboard 治理闭环。

中文 Prompt：
绘制一张闭环流程图，展示 Benchmark Card 驱动的开放治理循环。环路节点包括 task definition、data split、submission format、automated evaluation、anomaly review、leaderboard publication、community feedback、version update。节点按照圆环或椭圆环形排布，用箭头形成持续循环。中心放置 “Benchmark Card” 主卡片，周围附上 contamination rules、resource declaration、baseline bundle、release notes 等英文标签。图内所有文字使用英文，整体制度化、流程化、长期维护导向。

English Prompt:
Draw a governance loop diagram centered on a Benchmark Card. Arrange the loop nodes in a circular or oval flow: task definition, data split, submission format, automated evaluation, anomaly review, leaderboard publication, community feedback, and version update. Place the “Benchmark Card” in the center and add small surrounding tags such as contamination rules, resource declaration, baseline bundle, and release notes. All visible text must be in English. The style should communicate institutional maintenance and long-term governance.

### 图J-5 / `chj_05_experiment_card_layers.png`

对应位置：ChJ 中 Experiment Card 的四层结构。

中文 Prompt：
绘制一张四层结构信息图，像一张分层卡片或四层楼板，自上而下分别是 hypothesis、control relation、evaluation slices、write-back actions。每一层都用英文写一句简短说明和小例子，例如 validate whether schema reward improves logical consistency、fix the model and only change reward、slice by risk fields and hard cases、write findings back to sampling and annotation。图内所有文字为英文，主标题使用英文 “The Four Layers of an Experiment Card”。

English Prompt:
Create a four-layer infographic that looks like a stacked card or layered platform. From top to bottom, show hypothesis, control relation, evaluation slices, and write-back actions. Add short English explanations and small examples such as validating whether schema reward improves logical consistency, fixing the model while only changing reward, slicing by risk fields and hard cases, and writing findings back to sampling and annotation. All visible text must be in English, and the title should be “The Four Layers of an Experiment Card”.

### 图J-6 / `chj_06_paper_vs_engineering_doc.png`

对应位置：ChJ 中论文式说明与工程式说明对比图。

中文 Prompt：
绘制一张左右对照表风格的信息图。左侧标题使用英文 “Paper-Style Documentation”，右侧标题使用英文 “Engineering-Style Documentation”。对照维度全部用英文，包括 task definition、field schema、defect description、recommended algorithms、evaluation slices、version governance、open rules。左侧用较少字段和模糊描述表现 concise but not operational，右侧用更完整结构表现 reusable、traceable、governable。图内所有文字必须为英文。

English Prompt:
Design a side-by-side comparison infographic. The left side should be titled “Paper-Style Documentation” and the right side “Engineering-Style Documentation”. Compare them across task definition, field schema, defect description, recommended algorithms, evaluation slices, version governance, and open rules. Make the left side visually sparse and abstract to suggest concise but incomplete documentation, while the right side is more structured and actionable to suggest reusable, traceable, and governable engineering documentation. All visible text must be in English.

## ChK 数据质量评估、修复算法与主动学习

### 图K-1 / `chk_01_quality_four_layer_loop.png`

对应位置：ChK 开篇“质量工程四层闭环”。

中文 Prompt：
绘制一张四阶段闭环流程图，节点依次为 quality scoring、defect detection、repair suggestion、active learning，四个节点用环形箭头相连，形成持续优化闭环。每个节点旁边用英文小注释说明：scoring sets priority、detection identifies defect types、repair chooses treatment path、active learning allocates human effort。中心放置英文标题 “Data Quality Engineering”。图内全部文字为英文。

English Prompt:
Create a four-stage closed-loop infographic with the nodes quality scoring, defect detection, repair suggestion, and active learning. Connect them with circular arrows to show continuous optimization. Add English side notes for each node: scoring sets priority, detection identifies defect types, repair chooses treatment path, and active learning allocates human effort. Place a central label “Data Quality Engineering”. All visible text must be in English.

### 图K-2 / `chk_02_quality_scoring_axes.png`

对应位置：ChK 中质量评分三维坐标图。

中文 Prompt：
绘制一张三维坐标系信息图，三个轴分别用英文标注 correctness、task fitness、learning value。在坐标空间中分布多个样本点，并用颜色区分 high-quality high-value、correct but low-value、high-risk for repair 等区域。右侧使用英文图例解释不同颜色和点形状。图内所有文字使用英文，标题为 “Three-Dimensional Quality Scoring”。

English Prompt:
Draw a three-dimensional scoring diagram with three axes labeled correctness, task fitness, and learning value. Place multiple sample points in the space and use color to distinguish regions such as high-quality high-value, correct but low-value, and high-risk for repair. Add an English legend on the right to explain colors and marker shapes. All visible text must be in English, and the title should be “Three-Dimensional Quality Scoring”.

### 图K-3 / `chk_03_repair_decision_tree.png`

对应位置：ChK 中质量修复分层决策树。

中文 Prompt：
绘制一张从上到下的决策树图，根节点使用英文 “Quality Issue Detected”，分叉到四类处理路径：rule-based repair、statistical repair、representation-learning repair、LLM-as-Judge triage。每条路径再延伸出英文示例，如 schema validation、anomaly detection、embedding clustering、semantic judging。底部汇总到 automatic repair、manual review、deferred handling。图内全部文字必须为英文。

English Prompt:
Create a top-down decision tree beginning with “Quality Issue Detected”. Branch it into four repair paths: rule-based repair, statistical repair, representation-learning repair, and LLM-as-Judge triage. Extend each branch with English examples such as schema validation, anomaly detection, embedding clustering, and semantic judging. Merge the bottom into three final outcomes: automatic repair, manual review, and deferred handling. All visible text must be in English.

### 图K-4 / `chk_04_scorers_by_dataset.png`

对应位置：ChK 中六类数据的质量评分器组合。

中文 Prompt：
绘制一张矩阵热力图，纵轴用英文列出 StructBill-CN、SparseTable-Bench、Ophiuchus、Latent-Switch-69K、VoiceStyleControl、multi-chart；横轴用英文列出 token correctness、structural consistency、logical verification、geometric stability、tool behavior、style consistency、uncertainty、disagreement。用深浅色块表示各数据集重点依赖哪些评分器，标题用英文 “Scorer Combinations Across Six Dataset Types”。图内文字全部为英文。

English Prompt:
Design a matrix heatmap. Put six datasets on the vertical axis in English: StructBill-CN, SparseTable-Bench, Ophiuchus, Latent-Switch-69K, VoiceStyleControl, and multi-chart. Put scorer types on the horizontal axis in English: token correctness, structural consistency, logical verification, geometric stability, tool behavior, style consistency, uncertainty, and disagreement. Use shaded cells to show which scorers matter most for each dataset. All visible text must be in English, and the title should be “Scorer Combinations Across Six Dataset Types”.

### 图K-5 / `chk_05_scoring_to_active_learning_queue.png`

对应位置：ChK 中从质量评分到主动学习的闭环队列图。

中文 Prompt：
绘制一张四段式队列流程图，按横向排列四个模块：cheap filters、semantic triage、repair execution、post-repair evaluation。每个模块内部列出 2 到 4 个英文动作，如 schema checks、embedding triage、manual review、re-scoring after repair。顶部增加英文入口 “Quality Scores Enter Queue”，底部增加英文出口 “Active Learning Priority Panel”。图内所有文字为英文。

English Prompt:
Create a four-stage queue pipeline laid out horizontally: cheap filters, semantic triage, repair execution, and post-repair evaluation. Inside each module, list 2 to 4 representative actions in English such as schema checks, embedding triage, manual review, and re-scoring after repair. Add an English entry at the top, “Quality Scores Enter Queue”, and an English exit at the bottom, “Active Learning Priority Panel”. All visible text must be in English.

### 图K-6 / `chk_06_detector_repairer_matrix.png`

对应位置：ChK 中检测器与修复器对应表。

中文 Prompt：
绘制一张表格式矩阵图，纵轴是英文缺陷类型：logical defect、geometric defect、broken evidence chain、behavior defect、reasoning-switch defect、style-control defect；横轴是 detector、repairer、human review needed、repair cost。所有单元格中的条目都使用英文，整体像教材中的工程决策表，标题用英文 “Detector and Repairer Matrix by Defect Type”。

English Prompt:
Make a table-style matrix diagram. Rows represent defect types in English: logical defect, geometric defect, broken evidence chain, behavior defect, reasoning-switch defect, and style-control defect. Columns represent detector, repairer, human review needed, and repair cost. Fill each row with short English entries so the image feels like an engineering decision table in a textbook. All visible text must be in English, and the title should be “Detector and Repairer Matrix by Defect Type”.

### 图K-7 / `chk_07_active_learning_priority_scatter.png`

对应位置：ChK 中主动学习优先级散点图。

中文 Prompt：
绘制一张二维散点图，横轴使用英文 “Repair Cost”，纵轴使用英文 “Repair Gain”，点的颜色表示 business risk level，点的大小表示 sample count 或 sample weight。右上区域标注 “Priority Investment”，左上区域标注 “High Gain, Low Cost”，右下区域标注 “Handle with Caution”。图内所有坐标、图例、标签均使用英文。

English Prompt:
Draw a two-dimensional scatter plot where the x-axis is “Repair Cost” and the y-axis is “Repair Gain”. Let point color indicate business risk level and point size indicate sample count or sample weight. Label the upper-right region “Priority Investment”, the upper-left “High Gain, Low Cost”, and the lower-right “Handle with Caution”. All visible text, axes, and legend entries must be in English.

## ChL 清洗、去重、去污染与隐私数据集

### 图L-1 / `chl_01_four_way_tension.png`

对应位置：ChL 开篇“四向张力”图。

中文 Prompt：
绘制一张四向张力示意图，可采用菱形或十字坐标布局，四个方向分别使用英文 quality、generalization、evaluation credibility、compliance。中心写英文 “Data Processing Decisions”，四个方向有箭头向外拉伸，表示清洗、去重、去污染、隐私治理之间的平衡关系。主标题使用英文 “Four-Way Tension in Cleaning, Deduplication, Decontamination, and Privacy Governance”。图内所有文字为英文。

English Prompt:
Create a four-direction tension diagram using a diamond or cross-axis layout. The four directions should be labeled in English: quality, generalization, evaluation credibility, and compliance. Put “Data Processing Decisions” in the center and use outward arrows to show the balancing force among cleaning, deduplication, decontamination, and privacy governance. The title should be “Four-Way Tension in Cleaning, Deduplication, Decontamination, and Privacy Governance”. All visible text must be in English.

### 图L-2 / `chl_02_dedup_actions.png`

对应位置：ChL 中重复样本处理三分法。

中文 Prompt：
绘制一张三分支处理流程图，从英文起点 “Duplicate Cluster Detected” 出发，分成三条英文路径：delete、merge、keep with down-weighting。每条路径下放置英文说明和适用场景，如 exact duplicate、semantic duplicate、near duplicate with minor variation。图内标签、箭头说明、标题全部为英文，标题建议使用 “Three Ways to Handle Duplicate Samples”。

English Prompt:
Draw a three-way handling flowchart starting from “Duplicate Cluster Detected”. Split it into three English paths: delete, merge, and keep with down-weighting. Under each path, add short English explanations and applicable cases such as exact duplicate, semantic duplicate, and near duplicate with minor variation. All visible text must be in English, and the title should be “Three Ways to Handle Duplicate Samples”.

### 图L-3 / `chl_03_privacy_action_matrix.png`

对应位置：ChL 中隐私字段分级与处理动作矩阵。

中文 Prompt：
绘制一张矩阵表，纵轴用英文列出 direct identifiers、quasi-identifiers、business-critical fields、non-sensitive structural fields；横轴用英文列出 delete、irreversible masking、generalization、controlled retention、publication restriction。用不同色块和勾叉标记呈现每类字段适合的动作。图内全部文字必须为英文，标题使用 “Privacy Field Tiers and Treatment Actions”。

English Prompt:
Create a matrix table for privacy treatment. Rows should be labeled in English: direct identifiers, quasi-identifiers, business-critical fields, and non-sensitive structural fields. Columns should be labeled in English: delete, irreversible masking, generalization, controlled retention, and publication restriction. Use different colored cells plus check and cross marks to show which actions fit each field type. All visible text must be in English, and the title should be “Privacy Field Tiers and Treatment Actions”.

### 图L-4 / `chl_04_decontamination_workflow.png`

对应位置：ChL 中去污染工作流。

中文 Prompt：
绘制一张线性加分叉的工作流图，步骤全部使用英文，包括 assign unique IDs to raw sources、source-level clustering、template-level clustering、cross-split retrieval before splitting、manual spot checks、freeze report、version-upgrade recheck。关键警示标签也使用英文，例如 “Run Before Split” 和 “Cluster Migration Beats Single-Item Patching”。图内全部文字为英文。

English Prompt:
Draw a linear workflow with a few branch notes for decontamination. Use English for every step: assign unique IDs to raw sources, source-level clustering, template-level clustering, cross-split retrieval before splitting, manual spot checks, freeze report, and version-upgrade recheck. Add English warning notes at key stages, such as “Run Before Split” and “Cluster Migration Beats Single-Item Patching”. All visible text must be in English.

## ChM 多模态解析、RAG 与 Agent 轨迹数据集

### 图M-1 / `chm_01_three_data_families.png`

对应位置：ChM 开篇“三类数据家族统一视角”。

中文 Prompt：
绘制一张三栏结构图，分别代表 Document Parsing and Chart Understanding、RAG Evidence Data、Agent Trajectory Data。每栏下方列出英文监督对象：structure and logic、evidence and citation、behavior and observation。三栏底部用一条共同横带连接，写英文 “Shared Core: Evidence Chain Integrity”。图内所有标题、栏目名、说明均为英文。

English Prompt:
Create a three-column overview diagram representing Document Parsing and Chart Understanding, RAG Evidence Data, and Agent Trajectory Data. Under each column, list the core supervision target in English: structure and logic, evidence and citation, and behavior and observation. Connect the three columns with a shared bottom band labeled “Shared Core: Evidence Chain Integrity”. All visible text must be in English.

### 图M-2 / `chm_02_document_schema.png`

对应位置：ChM 中文档理解数据三层 schema。

中文 Prompt：
绘制一张三层嵌套架构图，从外到内或从上到下分别是 Page Layer、Structure Layer、Logic Layer。页面层包含 image、page number、resolution；结构层包含 tables、paragraphs、fields、bounding boxes、reading order；逻辑层包含 constraint rules、aggregation relations、business semantics。图内所有标题和字段项都使用英文，主标题为 “Three-Layer Schema for Document Understanding Data”。

English Prompt:
Draw a three-layer nested schema diagram showing Page Layer, Structure Layer, and Logic Layer. The page layer includes image, page number, and resolution; the structure layer includes tables, paragraphs, fields, bounding boxes, and reading order; the logic layer includes constraint rules, aggregation relations, and business semantics. All visible text must be in English, and the title should be “Three-Layer Schema for Document Understanding Data”.

### 图M-3 / `chm_03_multichart_region_dependencies.png`

对应位置：ChM 中复合信息图区域依赖示意图。

中文 Prompt：
绘制一张复合信息图解析示意图，主体是一张由多个子图组成的信息图页面，包括 bar chart、line chart、legend area、title area、annotation area。用半透明框标出不同区域，并用箭头显示一个问题如何依赖多个区域来完成推理。问题气泡也使用英文，例如 “Which quarter grew the fastest for Product A according to the legend?”。图内所有文字必须为英文。

English Prompt:
Create an explanatory diagram of a composite infographic page that contains multiple subcharts, such as a bar chart, line chart, legend area, title area, and annotation area. Mark different regions with translucent boxes and show arrows indicating how one question depends on multiple regions for reasoning. Add an English question bubble, for example “Which quarter grew the fastest for Product A according to the legend?”. All visible text must be in English.

### 图M-3 / `chm_03_rag_evidence_card.png`

对应位置：ChM 中 RAG evidence card 结构图。

中文 Prompt：
绘制一张卡片式结构图，中心是 “RAG Evidence Card”，分为 query、canonical answer、evidence bundle、hard negatives、freshness、access level、citation rule 七个模块。右侧展示 evidence bundle 的展开结构，包含 document snippets、page numbers、source markers、permission tags。图内所有字段名、说明、标题必须为英文。

English Prompt:
Design a card-structured infographic centered on “RAG Evidence Card”. Divide it into seven modules: query, canonical answer, evidence bundle, hard negatives, freshness, access level, and citation rule. On the right side, show an expanded view of the evidence bundle with document snippets, page numbers, source markers, and permission tags. All visible text must be in English.

### 图M-4 / `chm_04_agent_trajectory_schema.png`

对应位置：ChM 中 Agent 轨迹状态转移图。

中文 Prompt：
绘制一张状态转移图，从英文 “User Goal” 开始，进入 “state_0”，然后沿着 thought、tool_call、observation、state_update 组成的循环前进，必要时分出 failure 和 recovery_action，再汇总到 final_outcome。用不同颜色区分 thinking、acting、observing、recovering。图内所有文字均使用英文。

English Prompt:
Create a state-transition diagram that starts from “User Goal”, moves into “state_0”, and then cycles through thought, tool_call, observation, and state_update. Add branches for failure and recovery_action, and finally converge into final_outcome. Use different colors to distinguish thinking, acting, observing, and recovering. All visible text must be in English.

### 表M-1 / `chm_04_chart_question_evidence_table.png`

对应位置：ChM 中图表问题类型与证据类型对照表。

中文 Prompt：
绘制一张规范表格，左列和表头全部使用英文。问题类型包括 reading values、comparison、ranking、aggregation、trend analysis、conditional reasoning、unanswerable；证据类型包括 local numbers、cross-region legends、timelines、explanatory text、missing conditions。图内所有表头、单元格、标题均为英文，标题使用 “Question Types and Required Evidence in Chart Reasoning”。

English Prompt:
Create a clean analytical table. The left column should list question types in English: reading values, comparison, ranking, aggregation, trend analysis, conditional reasoning, and unanswerable. The evidence columns should also be in English, including local numbers, cross-region legends, timelines, explanatory text, and missing conditions. All visible text must be in English, and the title should be “Question Types and Required Evidence in Chart Reasoning”.

## ChN 数据实验设计、评测方法与效果归因

### 图N-1 / `chn_01_experiment_feedback_loop.png`

对应位置：ChN 开篇“实验设计与策略回写闭环”。

中文 Prompt：
绘制一张闭环图，节点依次使用英文 data hypothesis、experiment design、control variables、slice evaluation、effect attribution、policy write-back，最后再回到 data hypothesis。中心放置英文 “Data Experiment System”。每个节点旁用英文小字说明，例如 define controls、refresh slice matrix、write back to collection and cleaning。图内所有文字必须为英文。

English Prompt:
Draw a closed-loop diagram with the nodes data hypothesis, experiment design, control variables, slice evaluation, effect attribution, and policy write-back, looping back to data hypothesis. Place “Data Experiment System” in the center. Add small English notes near each node, such as define controls, refresh slice matrix, and write back to collection and cleaning. All visible text must be in English.

### 图N-2 / `chn_02_slice_vs_average.png`

对应位置：ChN 中平均分与切片评测差异示意。

中文 Prompt：
绘制一张对比图，左边是一条平稳上升的 average score 曲线，右边或下方是一组 slice score 柱状图，其中部分关键切片明显下降。所有切片标签都使用英文，例如 amount fields、cross-chart multi-hop、fear-sad boundary、occlusion stress。图中用英文明确表达 “Rising Overall Score Does Not Guarantee Stronger Key Capabilities”。图内全部文字为英文。

English Prompt:
Create a comparison graphic where one side shows a steadily improving average score line, while the other side or lower panel shows slice-specific bars with several critical slices dropping. Label all slices in English, such as amount fields, cross-chart multi-hop, fear-sad boundary, and occlusion stress. Make the message visually explicit with an English statement like “Rising Overall Score Does Not Guarantee Stronger Key Capabilities”. All visible text must be in English.

### 图N-3 / `chn_03_attribution_to_policy_rewrite.png`

对应位置：ChN 中归因到策略回写的五条路径。

中文 Prompt：
绘制一张中心辐射图，中心是英文 “Attribution Findings”，向外辐射五条路径，分别通向 collection、cleaning、annotation、data mixture、evaluation。每条路径旁边加英文例子，如 collect more high-risk samples、adjust cleaning rules、rewrite schema guidelines、increase hard-sample ratio、upgrade slice metrics。图内所有文字为英文。

English Prompt:
Design a radial diagram with “Attribution Findings” in the center and five outward paths leading to collection, cleaning, annotation, data mixture, and evaluation. Add a short English example next to each path, such as collect more high-risk samples, adjust cleaning rules, rewrite schema guidelines, increase hard-sample ratio, and upgrade slice metrics. All visible text must be in English.

### 图N-4 / `chn_04_unified_attribution_radar.png`

对应位置：ChN 中六个数据集统一归因雷达图。

中文 Prompt：
绘制一张多系列雷达图，六个维度全部使用英文：structural correctness、logical consistency、evidence completeness、behavior stability、style control、robustness。用 3 到 6 组不同颜色或线型表示不同数据集或不同实验方案。右侧加英文图例，主标题使用英文 “Unified Attribution Radar Across Six Datasets”。图内全部文字为英文。

English Prompt:
Create a multi-series radar chart with six axes labeled in English: structural correctness, logical consistency, evidence completeness, behavior stability, style control, and robustness. Use 3 to 6 colored or differently styled polygons to represent different datasets or experimental setups. Add an English legend on the right. All visible text must be in English, and the title should be “Unified Attribution Radar Across Six Datasets”.

## ChO 开放基准、排行榜与教学实验

### 图O-1 / `cho_01_benchmark_lifecycle.png`

对应位置：ChO 开篇“从数据集到开放基准的生命周期”。

中文 Prompt：
绘制一张生命周期环图，节点全部使用英文，包括 dataset construction、Benchmark Card、baseline bundle、open submission、leaderboard maintenance、community feedback、course labs、version upgrades。中间放英文主标题 “Open Benchmark Asset Lifecycle”。整体应表现一个不断演进的生态，图内所有文字为英文。

English Prompt:
Draw a lifecycle ring diagram with the stages dataset construction, Benchmark Card, baseline bundle, open submission, leaderboard maintenance, community feedback, course labs, and version upgrades. Put the central title “Open Benchmark Asset Lifecycle” in the middle. The design should communicate an evolving ecosystem rather than a one-time release. All visible text must be in English.

### 图O-2 / `cho_02_benchmark_packaging_map.png`

对应位置：ChO 中六类数据集开放基准包装地图。

中文 Prompt：
绘制一张二维分类地图，横轴和纵轴都使用英文，例如 task output form 与 evaluation focus，把六类数据集放入不同象限，例如 structure-and-logic benchmarks、complex-reasoning benchmarks、behavior-and-control benchmarks。每个数据集节点附上英文主榜单指标。图内所有象限标题、坐标轴、卡片标签全部使用英文。

English Prompt:
Create a two-dimensional packaging map where the horizontal axis represents task output form and the vertical axis represents evaluation focus, both in English. Place six datasets in different quadrants such as structure-and-logic benchmarks, complex-reasoning benchmarks, and behavior-and-control benchmarks. Represent each dataset as a card-like node with its recommended main leaderboard metric attached in English. All visible text must be in English.

### 图O-3 / `cho_03_teaching_experiments_roadmap.png`

对应位置：ChO 中教学实验路线图。

中文 Prompt：
绘制一张课程路线图，从左到右排列五个实验模块：StructBill-CN、SparseTable-Bench、multi-chart、Ophiuchus、VoiceStyleControl。每个模块下方写英文说明，如 structured extraction、occlusion robustness、cross-chart reasoning、tool trajectory supervision、style control。上方也使用英文 semester timeline 或 week markers。图内所有文字必须为英文。

English Prompt:
Design a course roadmap running from left to right with five experiment modules: StructBill-CN, SparseTable-Bench, multi-chart, Ophiuchus, and VoiceStyleControl. Under each module, add a short English description such as structured extraction, occlusion robustness, cross-chart reasoning, tool trajectory supervision, and style control. Optionally add an English semester timeline or week markers along the top. All visible text must be in English.

### 图O-4 / `cho_04_leaderboard_workflow.png`

对应位置：ChO 中 Leaderboard 维护流程图。

中文 Prompt：
绘制一张六步流程图，步骤全部使用英文：receive submission、format validation、automatic evaluation、anomaly screening、manual review、official ranking publication，并从正式上榜再引出 delisting、relabeling、version locking 等英文后续治理动作。图内所有步骤名、分支说明、标题均使用英文。

English Prompt:
Create a six-step workflow diagram with the stages receive submission, format validation, automatic evaluation, anomaly screening, manual review, and official ranking publication. From the publication step, branch out follow-up governance actions such as delisting, relabeling, and version locking. All visible text must be in English.

### 图O-5 / `cho_05_benchmark_card_mockup.png`

对应位置：ChO 中 Benchmark Card 模板页示意。

中文 Prompt：
绘制一张仿页面式样的 mockup，页面中的模块全部使用英文，包括 task definition、input-output format、official metrics、slice metrics、submission format、contamination rules、resource declaration、maintenance roles。整体像正式 benchmark 首页缩略图，图内所有可见文字使用英文。

English Prompt:
Create a page-like mockup resembling a benchmark homepage or a textbook template page. Include sections in English for task definition, input-output format, official metrics, slice metrics, submission format, contamination rules, resource declaration, and maintenance roles. All visible text must be in English.

### 图O-6 / `cho_06_teaching_lab_roadmap.png`

对应位置：ChO 末尾教学实验路线图。

中文 Prompt：
绘制一张更偏教学管理视角的路线图，按时间或层级展示五类实验：structured extraction、robustness analysis、composite reasoning、tool behavior、controllable generation。每个阶段用英文标注 target capability、suggested baseline、expected deliverable，形成可执行的教学实验主线。图内全部文字为英文。

English Prompt:
Design a teaching-management roadmap organized by time or level, showing five experiment families: structured extraction, robustness analysis, composite reasoning, tool behavior, and controllable generation. For each stage, annotate target capability, suggested baseline, and expected deliverable in English, so the roadmap becomes an executable teaching-lab mainline. All visible text must be in English.
