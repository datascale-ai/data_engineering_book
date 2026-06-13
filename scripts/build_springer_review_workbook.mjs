import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const outDir = path.join(root, "outputs", "springer_review");
const outFile = path.join(outDir, "springer_zh_submission_gap_review.xlsx");

const finalAudit = JSON.parse(
  await fs.readFile(path.join(root, "publishing/final_review/final_publication_audit.json"), "utf8"),
);
const refAudit = JSON.parse(
  await fs.readFile(path.join(root, "publishing/final_review/reference_integrity_audit.json"), "utf8"),
);

const partNames = {
  part1: "第一篇：总论与基础设施",
  part2: "第二篇：文本预训练数据工程",
  part3: "第三篇：多模态数据工程",
  part4: "第四篇：指令微调与偏好数据",
  part5: "第五篇：合成数据工程",
  part6: "第六篇：推理与 Agent 数据工程",
  part7: "第七篇：应用级数据工程",
  part8: "第八篇：数据运营与平台建设",
  part9: "第九篇：数据资产、数据产品与数据契约",
  part10: "第十篇：智能化数据工程与 Data Engineering Agent",
  part11: "第十一篇：隐私合规与数据安全",
  part12: "第十二篇：专项数据集与数据工程实践",
  part13: "第十三篇：开源大模型数据工程配方与范式",
  part14: "第十四篇：项目案例研究",
};

const roleByCategory = {
  "构建/断链": "技术编辑",
  "图表/权限": "图表编辑",
  "参考文献": "参考文献编辑",
  "统稿风格": "责任编辑/章节作者",
  "人工签核": "主编/章节作者",
  "结构/元数据": "主编/项目管理",
  "交付包": "主编/制稿负责人",
  "合规声明": "主编/法务合规",
};

function firstHeading(markdown) {
  const line = markdown.split(/\r?\n/).find((item) => item.startsWith("# "));
  return line ? line.replace(/^#\s+/, "").trim() : "";
}

function classifyFile(file) {
  const base = path.basename(file);
  const parts = file.split("/");
  const partKey = parts.find((p) => /^part\d+$/.test(p));
  let section = partKey ? partNames[partKey] : "附录/卷前卷末";
  let unit = "其他";
  let order = 9000;

  const ch = base.match(/^ch(\d{2})_/);
  const proj = base.match(/^p(\d{2})_/);
  const app = base.match(/^appendix_([a-g])_/);
  if (ch) {
    unit = `Ch${ch[1]}`;
    order = Number(ch[1]);
  } else if (proj) {
    unit = `P${proj[1]}`;
    order = 1000 + Number(proj[1]);
  } else if (app) {
    unit = `附录${app[1].toUpperCase()}`;
    order = 2000 + app[1].charCodeAt(0);
  } else if (base === "index.md" && partKey) {
    unit = "篇首页";
    order = Number(partKey.replace("part", "")) * 100 - 1;
  } else if (base === "preface.md") {
    unit = "序言";
    order = 3001;
  } else if (base === "front_matter_guide.md") {
    unit = "卷前导读";
    order = 3002;
  } else if (base === "index.md") {
    unit = "全书总目录";
    order = 3003;
  } else if (base === "abbreviations.md") {
    unit = "缩写表";
    order = 3004;
  } else if (base === "afterword.md") {
    unit = "后记";
    order = 3005;
  }
  return { section, unit, order };
}

function countByFile(items, issueKey = "issues") {
  const result = new Map();
  for (const item of items) {
    const file = item.file;
    if (!file?.startsWith("docs/zh/")) continue;
    if (!result.has(file)) result.set(file, { total: 0, issues: new Map(), examples: [] });
    const bucket = result.get(file);
    bucket.total += 1;
    for (const issue of item[issueKey] || []) {
      bucket.issues.set(issue, (bucket.issues.get(issue) || 0) + 1);
    }
    if (bucket.examples.length < 3) {
      const line = item.line || item.entry_no || "";
      const detail = item.src || item.entry || item.context || "";
      bucket.examples.push(line ? `L${line}: ${detail}` : detail);
    }
  }
  return result;
}

function styleByFile() {
  const result = new Map();
  for (const hit of finalAudit.style_hits || []) {
    if (!hit.file?.startsWith("docs/zh/")) continue;
    if (!result.has(hit.file)) result.set(hit.file, { total: 0, examples: [] });
    const bucket = result.get(hit.file);
    bucket.total += 1;
    if (bucket.examples.length < 3) bucket.examples.push(`L${hit.line}: ${hit.phrase}`);
  }
  return result;
}

function refIntegrityByFile(list) {
  const result = new Map();
  for (const item of list || []) {
    if (!item.file?.startsWith("docs/zh/")) continue;
    if (!result.has(item.file)) result.set(item.file, []);
    result.get(item.file).push(item);
  }
  return result;
}

function issueCountsText(map) {
  return [...map.entries()].map(([k, v]) => `${k}=${v}`).join("; ");
}

function countAuditIssue(items, issue) {
  return (items || []).filter((item) => (item.issues || []).includes(issue)).length;
}

function priorityFor(category, countsText) {
  if (category === "构建/断链") return "P0";
  if (countsText.includes("missing-file") || countsText.includes("missing-figure-register")) return "P1";
  if (category === "参考文献" || category === "图表/权限") return "P1";
  if (category === "人工签核") return "P1";
  return "P2";
}

async function listZhMarkdownFiles() {
  const files = [];
  async function walk(dir) {
    for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) await walk(full);
      else if (entry.isFile() && entry.name.endsWith(".md")) {
        files.push(path.relative(root, full).split(path.sep).join("/"));
      }
    }
  }
  await walk(path.join(root, "docs/zh"));
  return files.sort((a, b) => classifyFile(a).order - classifyFile(b).order || a.localeCompare(b));
}

const figureByFile = countByFile(finalAudit.figures || []);
const referenceByFile = countByFile(finalAudit.references || []);
const styleCounts = styleByFile();
const missingSameChapter = refIntegrityByFile(refAudit.missing_same_chapter_references || []);
const uncitedRefs = refIntegrityByFile(refAudit.uncited_references || []);

const zhFiles = await listZhMarkdownFiles();
const fileMeta = new Map();
for (const file of zhFiles) {
  const text = await fs.readFile(path.join(root, file), "utf8");
  const meta = classifyFile(file);
  fileMeta.set(file, {
    ...meta,
    title: firstHeading(text),
    hasAbstract: text.includes("## 摘要"),
    hasKeywords: text.includes("## 关键词"),
    hasReferences: text.includes("## 参考文献"),
  });
}

const missingRegisterFiles = [...figureByFile.entries()]
  .filter(([, v]) => v.issues.has("missing-figure-register"))
  .map(([f]) => fileMeta.get(f)?.unit || f)
  .join("; ");

const chapterScope = "Ch01-Ch51; P01-P15; 附录A-G; 序言/目录/缩写表/后记";
const officialSubmit = "Springer 提交页：目录/篇章/标题编号正确；每章源文件含参考文献、图注、表格；原始图文件单独提交；正文和图件完整终稿；第三方权限齐备";
const officialPolicy = "Springer 图书政策：第三方内容权限、AI 使用披露、AI 不列作者、利益冲突披露、引用可靠性与可核验性";
const officialSubmitUrl = "https://www.springernature.com/gp/authors/publish-a-book/submitting-your-manuscript";
const officialPolicyUrl = "https://www.springernature.com/gp/policies/book-publishing-policies";
const figureStats = {
  highRes: countAuditIssue(finalAudit.figures, "needs-high-res-confirmation"),
  rightsAi: countAuditIssue(finalAudit.figures, "needs-rights-ai-review"),
  missingRegister: countAuditIssue(finalAudit.figures, "missing-figure-register"),
  missingFile: countAuditIssue(finalAudit.figures, "missing-file"),
};
const referenceStats = {
  authenticity: countAuditIssue(finalAudit.references, "needs-authenticity-review"),
  missingTraceId: countAuditIssue(finalAudit.references, "missing-doi-url-arxiv"),
  missingYear: countAuditIssue(finalAudit.references, "missing-year"),
};
const refFormatStats = refAudit.summary.format_issue_counts || {};

const unifiedRows = [
  ["U-001", "P0", figureStats.missingFile ? "未完成" : "当前审计未发现", "构建/断链", "消除会导致构建失败的图片缺失或路径错误", "中文文档 docs/zh", `final_publication_audit: missing-file=${figureStats.missingFile}`, officialSubmit, "持续执行 publish_lint 与 strict build 双重检查；若出现图片路径失效，应修正相对路径或补回源文件。", "技术编辑", "missing-file=0；publish_lint ERROR=0；mkdocs strict build 通过。", "publish_lint; mkdocs strict"],
  ["U-002", "P1", "当前审计通过", "构建/断链", "保持严格构建通过状态", "中文文档 docs/zh", "当前 mkdocs build --strict --clean 已通过；后续 PR 合并仍需作为门禁。", officialSubmit, "每次合并 PR 后复跑 strict build；nav、链接、图片 warning 应逐项处理至无警告。", "技术编辑", "mkdocs strict clean 完成且 warnings=0。", "mkdocs strict"],
  ["U-003", "P1", "未完成", "图表/权限", "全部图表高清源文件确认", `${finalAudit.summary.figures} 张中文正文图片`, `final_publication_audit: needs-high-res-confirmation=${figureStats.highRes}`, officialSubmit, "建立图表源文件包；逐图确认分辨率、字体、线条、纸书灰度可读性和源文件位置", "图表编辑", "每张图在台账中标记高清源已确认并可交付", "figure_rights_report"],
  ["U-004", "P1", "未完成", "图表/权限", "图表权属与 AI 使用声明终审", `${finalAudit.summary.figures} 张中文正文图片`, `final_publication_audit: needs-rights-ai-review=${figureStats.rightsAi}`, officialPolicy, "区分自绘、改绘、第三方、AI 辅助；补证据与声明；无法确认的外部图改绘", "图表编辑/法务合规", "图表台账每张图都有权属、AI 使用、第三方权限结论", "figure_rights_report"],
  ["U-005", "P1", figureStats.missingRegister ? "未完成" : "当前审计未发现", "图表/权限", "补齐缺失图表台账登记", missingRegisterFiles || "当前审计未发现缺登记图片", `final_publication_audit: missing-figure-register=${figureStats.missingRegister}`, officialSubmit, "将缺登记图片逐张加入 `publishing/12_figures_tables_register.md`，补标题、来源、权限、alt text、高清源。", "图表编辑", "missing-figure-register=0。", "figure_rights_report"],
  ["U-006", "P1", figureStats.missingFile ? "未完成" : "当前审计未发现", "图表/权限", "补齐或修正缺失图片文件", "见 figure_rights_report", `final_publication_audit: missing-file=${figureStats.missingFile}`, officialSubmit, "确认图片实际存在于 docs/images，并统一路径；如缺源文件，应重新导出。", "技术编辑/图表编辑", "missing-file=0；PDF/HTML 均能渲染。", "figure_rights_report"],
  ["U-007", "P1", "未完成", "参考文献", "全书参考文献真实性终审", `${refAudit.summary.files} 个中文扫描文件`, `final_publication_audit: references=${finalAudit.summary.references}; needs-authenticity-review=${referenceStats.authenticity}`, officialPolicy, "逐条核验作者、题名、年份、venue、DOI/arXiv/URL；删除或替换无法核验的引用。", "参考文献编辑", "所有条目完成人工签核；external_checks 不再是 not-checked。", "reference_audit_report"],
  ["U-008", "P1", "未完成", "参考文献", "补 DOI / URL / arXiv 等可追溯标识", "全书参考文献", `final_publication_audit: missing-doi-url-arxiv=${referenceStats.missingTraceId}; reference_integrity_audit=${refFormatStats["missing-doi-arxiv-url"] || 0}`, officialPolicy, "优先补 DOI；论文无 DOI 时补 arXiv/ACL Anthology/官方出版页；文档补官方 URL 与访问日期。", "参考文献编辑", "missing-doi-url-arxiv=0，或逐条保留豁免说明。", "reference_audit_report"],
  ["U-009", "P1", "未完成", "参考文献", "补齐参考文献年份", "全书参考文献", `final_publication_audit: missing-year=${referenceStats.missingYear}; reference_integrity_audit=${refFormatStats["missing-year"] || 0}`, officialPolicy, "按 Springer 样式补年份；滚动文档使用发布年份或访问年份并统一标注。", "参考文献编辑", "missing-year=0。", "reference_audit_report"],
  ["U-010", "P2", "未完成", "参考文献", "统一参考文献句末标点和 URL 尾部标点", "全书参考文献", `reference_integrity_audit: missing-terminal-period=${refFormatStats["missing-terminal-period"] || 0}; url-trailing-punctuation=${refFormatStats["url-trailing-punctuation"] || 0}`, officialPolicy, "统一 Springer 参考文献标点；修正 URL 尾部误吸收的句点或尖括号。", "参考文献编辑", "format_issue_counts 相关项全部消除。", "reference_integrity_audit"],
  ["U-011", "P1", "未完成", "参考文献", "处理正文引用与本章参考文献不匹配", `涉及 ${refAudit.summary.missing_same_chapter_references} 个同章缺失引用项`, `reference_integrity_audit: missing_same_chapter_references=${refAudit.summary.missing_same_chapter_references}`, officialPolicy, "补本章参考文献或删除/改写正文引用；确保作者-年份引用能在本章文末找到", "章节作者/参考文献编辑", "missing_same_chapter_references=0", "reference_integrity_audit"],
  ["U-012", "P2", "未完成", "参考文献", "处理未被正文引用的参考文献", `涉及 ${refAudit.summary.uncited_references} 条未引文献`, `reference_integrity_audit: uncited_references=${refAudit.summary.uncited_references}`, officialPolicy, "删除非必要列入的文献，或在正文中补充明确引用语境。", "章节作者/参考文献编辑", "uncited_references=0，或完成人工豁免标记。", "reference_integrity_audit"],
  ["U-013", "P2", "未完成", "交叉引用", "人工复核长距离章节引用", "全书 22 项 long-range-ref", "xref_scan: ERROR=0 WARN=0 INFO=22", officialSubmit, "确认长距离引用是否必要；必要时改为篇章综述或增加就近回扣句", "责任编辑", "long-range-ref 完成逐项人工确认", "xref_scan"],
  ["U-014", "P2", "未完成", "统稿风格", "全书非正式表达精修", chapterScope, `final_publication_audit: style_hits=${finalAudit.summary.style_hits}`, officialSubmit, "减少非正式、课程讲义式和网页教程式表达；统一为正式技术书语体。", "责任编辑/章节作者", "chapter_style_checklist 逐章签核。", "style_report"],
  ["U-015", "P1", "未完成", "人工签核", "逐章统稿签核", chapterScope, "chapter_style_checklist 当前均为待人工签核", officialSubmit, "逐章确认摘要、关键词、术语、图表引用、参考文献和 Springer 体例", "主编/章节作者", "所有行状态从待人工签核改为已签核", "chapter_style_checklist"],
  ["U-016", "P1", "未完成", "人工签核", "高风险章节与抽检章节签核", "Ch12, Ch16, Ch21, Ch24, Ch29, Ch40, P11, P12, P13, P15; Part10/12/14", "manual_review_checklist 当前均为待人工签核", officialSubmit, "重点复核安全/合规边界、图表权属、参考文献真实性和案例复现边界。", "主编/章节作者/法务合规", "manual_review_checklist 全部完成。", "manual_review_checklist"],
  ["U-017", "P1", "需持续验证", "结构/元数据", "统一交付口径：附录数量与结构冻结说明", "README; publishing 控制台; 目录; 台账", "当前中文主线口径为 14 篇、51 章、15 项目、7 附录 A-G", officialSubmit, "统一为当前中文主线口径：14 篇、51 章、15 项目、7 附录 A-G；后续 PR 合并后同步 README、MkDocs、index 与台账", "主编/项目管理", "所有控制台、台账、README、交付清单口径一致", "README; publishing"],
  ["U-018", "P1", "未完成", "交付包", "整理 Springer 可生产源文件包", "全书", "当前主要是 Markdown、预览 PDF/HTML；未见正式 Word/LaTeX 分章源文件交付包", officialSubmit, "按 Springer 要求导出每章源文件，包含参考文献、图注、表格；命名规则固定", "制稿负责人", "源文件包、图源包、PDF 样稿、清单齐备", "本地文件扫描"],
  ["U-019", "P1", "未完成", "交付包", "生成最终中文合并 PDF 并做版面 QA", "全书中文主线", "当前可见分篇预览与内部工作 PDF，未确认最终中文合并交付 PDF", officialSubmit, "生成最终合并 PDF；检查字体嵌入、目录、页码、图表跨页、图中文字字号", "制稿负责人", "最终中文 PDF 可交付且与源文件一致", "output/publishing 扫描"],
  ["U-020", "P1", "未完成", "合规声明", "补齐 AI 使用、第三方权限、利益冲突、作者信息声明", "卷前/卷末/交付包", "当前未见完整出版声明包；图表台账仍要求 AI/权限终审", officialPolicy, "在前言/致谢/声明文件中披露 AI 工具使用、人类责任、第三方内容、利益冲突、ORCID/单位", "主编/法务合规", "声明文件齐备并与台账一致", "Springer policies; local audit"],
  ["U-021", "P2", "未完成", "结构/元数据", "补齐作者、单位、ORCID、简介、通讯作者元数据", "卷前材料; README", "README 仅有团队级作者信息，未形成 Springer 元数据清单", officialSubmit, "准备作者名单、单位、ORCID、邮箱、简介、贡献说明、通讯作者", "主编/项目管理", "作者元数据表可直接提交出版社", "README; checklist"],
  ["U-022", "P2", "未完成", "结构/元数据", "确认附录是否需要摘要/关键词", "附录A-G", "正文和项目章均有摘要/关键词；附录A-G 当前缺摘要/关键词", officialSubmit, "若附录作为独立章节进入 SpringerLink，补摘要与关键词；若作为 back matter，需在交付说明中声明", "主编/责任编辑", "附录处理口径明确且全书一致", "Markdown 结构扫描"],
  ["U-023", "P2", "未完成", "交付包", "配套代码与复现资源说明终审", "P01-P15; 附录C; README", "项目章需要配套资源或复现说明；P15 当前无独立代码目录", officialSubmit, "补项目-代码-数据-环境映射；说明不可复现或需模拟数据的边界", "项目作者/技术编辑", "每个项目均有资源说明和边界声明", "checklist; repo scan"],
  ["U-024", "P2", "未完成", "图表/权限", "alt text 与无障碍描述复核", "全书图表", "台账已含 alt text 字段，但仍需终稿复核", officialPolicy, "复核 alt text 是否描述信息而非重复图名；复杂图补长说明或正文解释", "图表编辑/责任编辑", "alt text 台账全部签核", "12_figures_tables_register"],
  ["U-025", "P2", "未完成", "结构/元数据", "统一书名、术语、英文缩写首次出现规则", "全书", "局部存在“算法及/算法与”、中英文术语和缩写口径差异风险", officialSubmit, "以术语表和封面/合同书名为准全书替换；首次出现补中英文全称", "责任编辑", "术语表、目录、正文、README 一致", "README; abbreviations"],
  ["U-026", "P1", "未完成", "合规声明", "补齐竞争利益声明", "卷前材料；作者元数据；交付包", "当前表中仅笼统写入利益冲突披露，未形成可提交声明文本。", officialPolicy, "按 Springer 政策为全体作者准备 competing interests 声明；无竞争利益时也应有明确的否定声明。", "主编/法务合规", "竞争利益声明文件齐备，并与作者名单、单位和资助信息一致。", officialPolicyUrl],
  ["U-027", "P1", "未完成", "合规声明", "补齐 AI 使用声明并排除 AI 作者署名", "卷前/卷末；图表台账；交付包", "当前表中已有 AI 使用终审，但未单列 AI 不列作者、AI 辅助写作责任归属和 AI 图像限制。", officialPolicy, "披露生成式 AI 在写作、翻译、图表或代码辅助中的使用范围；明确人类作者承担内容责任；AI 工具不列为作者。", "主编/法务合规", "AI 使用声明、作者贡献说明和图表台账三者一致。", officialPolicyUrl],
  ["U-028", "P1", "未完成", "图表/权限", "排查并替换不符合政策的 AI 生成图像", "全书图表", "图表台账仅要求 AI 使用声明，尚未单列 AI 生成图片的政策边界。", officialPolicy, "逐图确认是否为 AI 生成图片；若属于政策不允许或权属不可证明的图片，应替换为自绘矢量图、作者原创图或可授权图片。", "图表编辑/法务合规", "图表台账标明 AI 生成/AI 辅助/非 AI；不合规图片已替换并复核。", officialPolicyUrl],
  ["U-029", "P1", "未完成", "合规声明", "补齐伦理、知情同意与敏感数据说明", "Ch36-Ch37；Part12；项目章；交付包", "医疗影像、票据、语音、人脸/图文候选池等内容涉及隐私、个人数据或敏感数据风险。", officialPolicy, "逐章确认是否涉及人类参与者、个人数据、医疗数据、儿童、人脸、语音或第三方数据；需要时补伦理批准、知情同意、匿名化和数据使用边界说明。", "主编/法务合规/章节作者", "伦理与隐私说明覆盖所有相关章节；无法公开的数据有替代说明和限制说明。", officialPolicyUrl],
  ["U-030", "P1", "未完成", "交付包", "补齐数据可用性与代码可复现性声明", "P01-P15；附录C；全书数据集章节", "当前仅提示配套代码资源说明，尚未形成 Springer 可审查的数据可用性/代码可用性声明。", officialPolicy, "为项目章和数据集章补充 data/code availability 说明，列出公开仓库、不可公开原因、模拟数据方案、许可证和运行环境。", "项目作者/技术编辑", "每个项目和数据集章节均有数据/代码可用性声明，且与 README、附录C、仓库目录一致。", officialPolicyUrl],
  ["U-031", "P2", "未完成", "统稿风格", "执行包容性语言与歧视性表达复核", "全书", "现有风格检查关注非正式表达，尚未单列包容性语言和潜在偏见表达。", officialPolicy, "复核涉及性别、地域、职业、年龄、医疗、残障、民族/国家等表述，避免歧视性、标签化或不必要的群体概括。", "责任编辑/法务合规", "包容性语言检查完成；争议表达有替换或说明。", officialPolicyUrl],
  ["U-032", "P2", "未完成", "图表/权限", "补齐可访问性和色彩可读性复核", "全书图表；PDF 样稿", "alt text 已列入，但尚未单列色彩对比、灰度打印、复杂图长说明和表格可读性。", officialPolicy, "复核图中文字字号、灰度打印、色彩对比、图例、复杂图说明和表格跨页；必要时提供长描述或正文解释。", "图表编辑/制稿负责人", "图表在 HTML/PDF/纸书灰度场景下均可读，alt text 与长说明已签核。", officialPolicyUrl],
  ["U-033", "P1", "未完成", "交付包", "建立 Springer 提交包目录结构", "全书交付包", "当前表中提到源文件包和图源包，但未明确提交包目录和文件冻结规则。", officialSubmit, "按出版社提交习惯整理 manuscript/source、figures、permissions、supplementary、metadata、forms 等目录，并冻结版本号和校验清单。", "制稿负责人", "提交包可直接交付出版社；每个文件有命名、版本、责任人和校验状态。", officialSubmitUrl],
  ["U-034", "P1", "未完成", "合规声明", "准备 License to Publish 与作者授权材料", "作者元数据；交付包", "当前表中未列 License to Publish、作者确认和署名顺序冻结。", officialSubmit, "整理作者署名顺序、单位、ORCID、邮箱、通讯作者、贡献说明和 License to Publish 所需信息。", "主编/项目管理", "LTP/作者授权所需信息齐备，作者顺序和单位信息已冻结。", officialSubmitUrl],
  ["U-035", "P2", "未完成", "结构/元数据", "补齐出版社营销与元数据材料", "书名；副标题；封面；简介；关键词；作者简介", "当前主表关注正文和交付文件，尚未覆盖出版社常需的元数据和营销文本。", officialSubmit, "准备英文书名、副标题、中文书名、简介、关键词、作者简介、目标读者、卖点、封面说明和系列信息。", "主编/项目管理", "元数据表可直接用于出版社系统录入，并与 README、封面和合同口径一致。", officialSubmitUrl],
];

const templateAdjustedRows = new Map([
  ["U-018", ["已建立规范，待导出交付包", "已新增 `publishing/18_springer_submission_package.md`；仍需按出版社模板实际导出 Word/LaTeX 或其他生产源文件。"]],
  ["U-020", ["已建立模板，待人工填写", "已新增 `publishing/19_declarations_and_metadata_templates.md`；仍需主编、作者和法务确认 AI 使用、第三方权限、利益冲突、ORCID/单位等内容。"]],
  ["U-021", ["已建立模板，待人工填写", "已新增作者元数据模板；仍需作者本人确认姓名、单位、ORCID、邮箱、简介、署名顺序和通讯作者。"]],
  ["U-026", ["已建立模板，待作者确认", "已新增竞争利益声明模板；未代填有/无竞争利益结论，需全体作者确认。"]],
  ["U-027", ["已建立模板，待人工填写", "已新增 AI 使用声明模板；仍需确认工具、用途、影响范围、图表台账一致性和 AI 不列作者。"]],
  ["U-029", ["已建立模板，待法务/作者确认", "已新增伦理、知情同意与敏感数据模板；仍需逐章确认是否涉及个人数据、医疗数据、语音、人脸、票据或第三方数据。"]],
  ["U-030", ["已建立模板，待项目作者填写", "已新增数据可用性与代码可用性模板；仍需 P01-P15、附录C 和数据集章节填写公开仓库、许可证、环境和不可公开说明。"]],
  ["U-031", ["已建立复核模板，待人工统稿", "已新增包容性语言复核模板；正文候选表达仍需责任编辑逐章判断，未自动大规模改写。"]],
  ["U-032", ["已建立复核模板，待逐图确认", "已新增可访问性、色彩可读性和长说明复核模板；仍需 PDF/纸书/HTML 逐图检查。"]],
  ["U-033", ["已建立规范，待整理交付包", "已新增 Springer 提交包目录、命名和冻结规则；仍需实际整理文件、版本号、校验和和责任人签核。"]],
  ["U-034", ["已建立清单，待作者授权", "已新增 License to Publish 准备清单；仍需作者顺序、单位、贡献说明和授权表单人工确认。"]],
  ["U-035", ["已建立模板，待主编填写", "已新增出版社元数据模板，并写入英文主标题和副标题；仍需中文书名、简介、关键词、作者简介、目标读者和封面说明终稿。"]],
]);

for (const row of unifiedRows) {
  const adjusted = templateAdjustedRows.get(row[0]);
  if (!adjusted) continue;
  row[2] = adjusted[0];
  row[8] = `${row[8]}；本次已补充对应出版控制文件或模板。`;
}

function remainingForUnified(row) {
  const [id, , status, category] = row;
  if (templateAdjustedRows.has(id)) return templateAdjustedRows.get(id)[1];
  if (status === "当前审计通过" || status === "当前审计未发现") return "本次未发现需调整项；后续 PR 合并后继续复跑检查。";
  if (id === "U-001" || id === "U-002") return "本次无需人工结论；后续合并后需持续验证。";
  if (category === "图表/权限") return "需逐图确认高清源、权属、AI 使用和第三方许可；本文未代替图表/法务签核。";
  if (category === "参考文献") return "需参考文献编辑逐条核验真实性、DOI/URL/arXiv、年份和引用必要性；本文未代签核。";
  if (category === "统稿风格") return "需责任编辑和章节作者逐章判断口语化/课程化候选表达；本次未做大规模自动改写。";
  if (category === "人工签核") return "需主编、章节作者或法务按人工复核清单签核；本文未代签核。";
  if (category === "结构/元数据") return "需主编确认最终出版口径、书名、术语和元数据；本次仅补充可确认的模板和口径。";
  if (category === "交付包") return "需制稿负责人实际导出源文件、PDF、图源包和校验清单；本次未生成最终提交包。";
  if (category === "合规声明") return "需作者、主编和法务确认声明结论；本文仅提供模板，不代填法律或伦理结论。";
  return "需对应责任人复核并签核。";
}

const detailRows = [];
let detailId = 1;
function addDetail(file, category, description, evidence, action, acceptance, priority = undefined) {
  const meta = fileMeta.get(file);
  if (!meta) return;
  const pr = priority || priorityFor(category, evidence);
  detailRows.push([
    `D-${String(detailId++).padStart(4, "0")}`,
    pr,
    "未完成",
    meta.section,
    meta.unit,
    meta.title,
    file,
    category,
    description,
    evidence,
    action,
    roleByCategory[category] || "责任编辑",
    acceptance,
  ]);
}

for (const file of zhFiles) {
  const meta = fileMeta.get(file);
  const isMainUnit = /^Ch\d+|^P\d+|^附录/.test(meta.unit);
  if (!isMainUnit) continue;

  if (meta.unit.startsWith("附录") && (!meta.hasAbstract || !meta.hasKeywords)) {
    addDetail(
      file,
      "结构/元数据",
      "确认并补齐附录摘要/关键词口径",
      `hasAbstract=${meta.hasAbstract}; hasKeywords=${meta.hasKeywords}; hasReferences=${meta.hasReferences}`,
      "若附录按独立章节进入交付，补摘要与3-6个关键词；若按 back matter 处理，交付说明中统一声明。",
      "附录 A-G 口径一致，目录和交付清单同步。",
      "P2",
    );
  }

  const figures = figureByFile.get(file);
  if (figures) {
    const counts = issueCountsText(figures.issues);
    if (figures.issues.has("missing-file")) {
      addDetail(
        file,
        "构建/断链",
        "修复会导致构建失败的图片缺失/路径错误",
        `${counts}; examples=${figures.examples.join(" | ")}`,
        "改为正确相对路径或补回源文件；同时检查 PDF/HTML 双端渲染。",
        "missing-file=0；publish_lint ERROR=0。",
        "P0",
      );
    }
    addDetail(
      file,
      "图表/权限",
      "完成本章图表台账、权属、AI 声明和高清源复核",
      `${counts || "figures-with-review"}; 图片数=${figures.total}; examples=${figures.examples.join(" | ")}`,
      "逐图补登记、图题、来源、权限、alt text、高清源路径；优先处理缺失台账项。",
      "本章图表无 missing-register/missing-file；所有图权属和高清源均已签核。",
    );
  }

  const refs = referenceByFile.get(file);
  if (refs) {
    addDetail(
      file,
      "参考文献",
      "完成本章参考文献真实性、DOI/URL/arXiv、年份和样式复核",
      `${issueCountsText(refs.issues)}; 问题条目=${refs.total}; examples=${refs.examples.join(" | ")}`,
      "逐条核验真实存在与引用必要性；补 DOI/URL/arXiv/年份；统一 Springer 样式。",
      "本章 reference_audit 问题全部处理，或保留经人工签核的豁免说明。",
    );
  }

  const missing = missingSameChapter.get(file);
  if (missing?.length) {
    addDetail(
      file,
      "参考文献",
      "处理正文作者-年份引用未在本章参考文献出现的问题",
      `missing_same_chapter_references=${missing.length}`,
      "补本文献条目、调整正文引用或删除无法支撑的引用。",
      "missing_same_chapter_references=0。",
      "P1",
    );
  }

  const uncited = uncitedRefs.get(file);
  if (uncited?.length) {
    addDetail(
      file,
      "参考文献",
      "处理本章参考文献未被正文引用的问题",
      `uncited_references=${uncited.length}`,
      "删除非必要列入的文献，或在正文中补充明确引用语境。",
      "uncited_references=0，或人工签核保留。",
      "P2",
    );
  }

  const styles = styleCounts.get(file);
  if (styles?.total) {
    addDetail(
      file,
      "统稿风格",
      "精修本章非正式表达候选项",
      `style_hits=${styles.total}; examples=${styles.examples.join(" | ")}`,
      "逐条判断保留或改写，减少课程化、非正式和重复修辞表达，保持正式技术书语体。",
      "chapter_style_checklist 本章状态已签核。",
      styles.total >= 35 ? "P1" : "P2",
    );
  }

  if (/^Ch(12|16|21|24|29|40)$/.test(meta.unit) || /^P(11|12|13|15)$/.test(meta.unit)) {
    addDetail(
      file,
      "人工签核",
      "完成 Springer 抽检章节人工复核",
      "manual_review_checklist 标记为指定抽检章节/项目",
      "复核摘要/关键词、术语、图表权属、参考文献真实性、代码长度、案例边界、章末小结。",
      "manual_review_checklist 对应行签核完成。",
      "P1",
    );
  }

  if (/part10|part12|part14/.test(file)) {
    addDetail(
      file,
      "人工签核",
      "完成高风险范围安全/合规/复现边界复核",
      "manual_review_checklist 将 Part10/Part12/Part14 标为高风险范围",
      "重点复核数据来源、隐私/版权、Agent 权限、医疗/票据/语音数据、项目复现边界和风险提示。",
      "高风险复核清单对应项签核完成。",
      "P1",
    );
  }
}

detailRows.sort((a, b) => {
  const pr = { P0: 0, P1: 1, P2: 2, P3: 3 };
  return (pr[a[1]] - pr[b[1]]) || a[3].localeCompare(b[3], "zh-Hans-CN") || a[4].localeCompare(b[4], "zh-Hans-CN") || a[0].localeCompare(b[0]);
});

function remainingForDetail(row) {
  const category = row[7];
  if (category === "构建/断链") return "需技术编辑修复对应文件后复跑 publish_lint 与 strict build。";
  if (category === "图表/权限") return "需逐图人工确认权属、AI 使用、高清源、alt text 和可访问性；本文未代签核。";
  if (category === "参考文献") return "需章节作者和参考文献编辑逐条核验或删改；本文未代签核。";
  if (category === "统稿风格") return "需责任编辑判断候选表达是否应改写；本次未自动大规模改正文。";
  if (category === "人工签核") return "需主编、章节作者或法务完成清单签核；本文未代签核。";
  if (category === "结构/元数据") return "需主编确认附录是否按独立章节处理；本次未自动补附录摘要/关键词。";
  return "需对应责任人复核。";
}

function remainingForSummary(row) {
  const metric = row[0];
  if (metric === "检查范围") return "英文翻译质量本次未评审。";
  if (metric === "提交就绪结论") return "需作者、主编、法务、图表编辑和参考文献编辑继续签核。";
  if (metric === "publish_lint") return "后续 PR 合并后需复跑。";
  if (metric === "xref_scan") return "长距离引用是否必要需责任编辑判断。";
  if (metric === "最终出版审计") return "风格候选、图表权属和参考文献真实性仍需人工签核。";
  if (metric === "图表问题") return "需逐图确认高清源、权属和 AI 使用。";
  if (metric === "参考文献问题") return "需逐条核验真实性和可追溯标识。";
  if (metric === "参考文献完整性") return "需补引、删引或人工豁免。";
  if (metric === "统稿风格") return "本次未自动大规模改写正文措辞。";
  if (metric === "人工签核") return "需主编、章节作者和法务确认。";
  if (metric === "Springer 要求覆盖") return "已补模板和规范，声明结论仍需人工填写。";
  return "需人工复核。";
}

function remainingForCoverage(row) {
  const requirement = row[0];
  if (requirement.includes("目录")) return "后续 PR 合并后需继续复跑结构测试。";
  if (requirement.includes("源文件")) return "已新增提交包规范；未实际导出 Word/LaTeX 生产源文件。";
  if (requirement.includes("原始图件")) return "逐图高清源和可访问性未签核。";
  if (requirement.includes("第三方材料")) return "权属和授权结论需法务/图表编辑确认。";
  if (requirement.includes("最终正文")) return "未生成最终 PDF 和文件校验清单。";
  if (requirement.includes("作者、单位")) return "作者个人信息和授权材料未填写。";
  if (requirement.includes("License")) return "正式 LTP 表单需作者或授权代表完成。";
  if (requirement.includes("AI 使用披露")) return "AI 使用范围和图表台账一致性需人工确认。";
  if (requirement.includes("AI 生成图片")) return "逐图 AI 生成/辅助/非 AI 结论未签核。";
  if (requirement.includes("竞争利益")) return "未代填有/无竞争利益结论，需全体作者确认。";
  if (requirement.includes("伦理")) return "伦理、知情同意、匿名化和不适用说明需人工确认。";
  if (requirement.includes("数据可用性")) return "公开仓库、许可证、运行环境和不可公开原因未逐项确认。";
  if (requirement.includes("引用可靠性")) return "需参考文献编辑逐条核验。";
  if (requirement.includes("包容性语言")) return "需责任编辑逐章判断，未自动大规模改写。";
  if (requirement.includes("可访问性")) return "需逐图逐表视觉和无障碍复核。";
  if (requirement.includes("出版社元数据")) return "中文书名、简介、关键词、作者简介和营销文案未终审。";
  return "需对应责任人确认。";
}

const summaryRows = [
  ["检查范围", "中文主线 docs/zh", "仅关注中文，英文后续统一翻译", "用户最新指令"],
  ["提交就绪结论", "未达到正式提交标准", "无构建断链 blocker；仍有 P1 图表/参考文献/政策声明/人工签核缺口", "综合审计"],
  ["publish_lint", "ERROR=0, WARN=0", "当前审计未发现出版 lint 错误", "uv run python scripts/publish_lint.py"],
  ["xref_scan", "ERROR=0, WARN=0, INFO=22", "22 个长距离引用需人工复核", "uv run python scripts/xref_scan.py"],
  ["最终出版审计", `style_hits=${finalAudit.summary.style_hits}; figures=${finalAudit.summary.figures}; broken_figures=${finalAudit.summary.broken_figures}`, finalAudit.summary.broken_figures ? "未通过 fail-on-blocker" : "无图片断链 blocker，仍需人工签核", "final_publication_audit.json"],
  ["图表问题", `needs-high-res=${figureStats.highRes}; needs-rights-ai-review=${figureStats.rightsAi}; missing-register=${figureStats.missingRegister}; missing-file=${figureStats.missingFile}`, "图表台账与权限是提交前主线工作", "figure_rights_report"],
  ["参考文献问题", `references=${finalAudit.summary.references}; issue_rows=${finalAudit.summary.reference_issue_rows}; authenticity-review=${referenceStats.authenticity}; missing DOI/URL/arXiv=${referenceStats.missingTraceId}`, "引用可靠性未达出版标准", "reference_audit_report"],
  ["参考文献完整性", `missing_same_chapter=${refAudit.summary.missing_same_chapter_references}; uncited=${refAudit.summary.uncited_references}`, "需要章节作者和文献编辑共同处理", "reference_integrity_audit.json"],
  ["统稿风格", `style_hits=${finalAudit.summary.style_hits}`, "候选项不等于全部错误，但需逐章签核", "style_report"],
  ["人工签核", "逐章与高风险章节均待完成", "Ch12/16/21/24/29/40、P11/P12/P13/P15、Part10/12/14 重点处理", "manual_review_checklist"],
  ["Springer 要求覆盖", "新增官方要求覆盖清单", "提交页与图书政策要求已映射到统一整改项 U-001 至 U-035", "Springer 官方页面"],
];

const springerCoverageRows = [
  ["提交前最终目录、篇章与标题编号一致", "已覆盖", "U-017; 中文结构测试", "当前口径为 14 篇、51 章、15 项目、7 附录；后续 PR 后需复核。", officialSubmitUrl],
  ["每章源文件包含正文、参考文献、图注和表格", "已覆盖", "U-018; U-033", "仍需导出 Springer 可生产源文件包。", officialSubmitUrl],
  ["原始图件单独提交，并满足格式、分辨率和可读性要求", "已覆盖", "U-003; U-032", "图源、高清、灰度打印和图中文字需逐图签核。", officialSubmitUrl],
  ["第三方材料权限齐备", "已覆盖", "U-004; U-005; U-028", "需区分自绘、改绘、第三方、AI 生成或 AI 辅助。", officialPolicyUrl],
  ["最终正文与图件完整，提交后不再频繁替换", "新增覆盖", "U-019; U-033", "需冻结提交包版本和校验清单。", officialSubmitUrl],
  ["作者、单位、ORCID、通讯作者和署名顺序完整", "已覆盖", "U-021; U-034", "需与 License to Publish 材料一致。", officialSubmitUrl],
  ["License to Publish / 作者授权材料", "新增覆盖", "U-034", "原表未单列，已新增整改项。", officialSubmitUrl],
  ["AI 使用披露，且 AI 工具不列为作者", "增强覆盖", "U-020; U-027", "原表有 AI 声明，现补充 AI 不列作者和人类责任归属。", officialPolicyUrl],
  ["AI 生成图片或 AI 辅助图件的政策边界", "新增覆盖", "U-028", "需排查不符合政策或权属不可证明的 AI 生成图。", officialPolicyUrl],
  ["竞争利益声明", "新增覆盖", "U-026", "需为全体作者准备明确声明。", officialPolicyUrl],
  ["伦理审批、知情同意、隐私与敏感数据处理", "新增覆盖", "U-029", "医疗、票据、语音、图文候选池等章节需重点复核。", officialPolicyUrl],
  ["数据可用性、代码可用性和复现边界", "增强覆盖", "U-023; U-030", "项目章和数据集章需要形成可审查声明。", officialPolicyUrl],
  ["引用可靠性、参考文献真实性和格式", "已覆盖", "U-007 至 U-012", "仍是当前主要 P1 缺口。", officialPolicyUrl],
  ["包容性语言与潜在偏见表达", "新增覆盖", "U-031", "原表仅有非正式表达检查，现补入政策性语言复核。", officialPolicyUrl],
  ["可访问性、alt text、图表长说明和色彩可读性", "增强覆盖", "U-024; U-032", "从 alt text 扩展到 PDF/纸书可读性。", officialPolicyUrl],
  ["出版社元数据、书名、副标题、关键词和作者简介", "新增覆盖", "U-035", "与最新英文书名和副标题同步。", officialSubmitUrl],
];

function writeSheet(sheet, rows) {
  if (!rows.length) return;
  const cols = rows[0].length;
  const endCol = columnName(cols);
  const used = sheet.getRange(`A1:${endCol}${rows.length}`);
  used.values = rows;
  used.format = {
    font: { name: "Arial", size: 10, color: "#111827" },
    borders: { preset: "all", style: "thin", color: "#D1D5DB" },
    verticalAlignment: "top",
    wrapText: true,
  };
  sheet.getRange(`A1:${endCol}1`).format = {
    fill: "#E5EEF9",
    font: { name: "Arial", size: 10, bold: true, color: "#111827" },
    borders: { preset: "all", style: "thin", color: "#9CA3AF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  used.format.autofitColumns();
  used.format.autofitRows();
}

function columnName(n) {
  let name = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    name = String.fromCharCode(65 + rem) + name;
    n = Math.floor((n - 1) / 26);
  }
  return name;
}

const workbook = Workbook.create();
const summarySheet = workbook.worksheets.add("摘要");
const unifiedSheet = workbook.worksheets.add("全书统一整改");
const detailSheet = workbook.worksheets.add("逐篇逐章重点");
const coverageSheet = workbook.worksheets.add("Springer要求覆盖清单");

writeSheet(summarySheet, [["指标", "当前结果", "结论/含义", "来源", "未确认/未调整事项"], ...summaryRows.map((row) => [...row, remainingForSummary(row)])]);
writeSheet(unifiedSheet, [[
  "ID",
  "优先级",
  "状态",
  "问题类别",
  "整改项目",
  "涉及章节/范围",
  "证据/当前发现",
  "Springer要求对应",
  "建议动作",
  "建议责任人",
  "验收标准",
  "来源",
  "未确认/未调整事项",
], ...unifiedRows.map((row) => [...row, remainingForUnified(row)])]);
writeSheet(detailSheet, [[
  "ID",
  "优先级",
  "状态",
  "篇/部分",
  "单元",
  "标题",
  "文件",
  "问题类别",
  "整改项目",
  "证据/数量",
  "建议动作",
  "建议责任人",
  "验收标准",
  "未确认/未调整事项",
], ...detailRows.map((row) => [...row, remainingForDetail(row)])]);
writeSheet(coverageSheet, [[
  "Springer官方要求",
  "覆盖状态",
  "对应整改项",
  "当前判断/仍需动作",
  "官方来源",
  "未确认/未调整事项",
], ...springerCoverageRows.map((row) => [...row, remainingForCoverage(row)])]);

await fs.mkdir(outDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outFile);
console.log(outFile);
