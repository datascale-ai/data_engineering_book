import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DOCS_EN = path.join(ROOT, "docs", "en");
const OUT_DIR = path.join(ROOT, "output", "indexes");
const OUT_XLSX = path.join(OUT_DIR, "springer_source_audit_index_en.xlsx");
const PREVIEW_DIR = path.join(OUT_DIR, "previews");

const IMAGE_HEADERS = [
  "ID",
  "Source chapter",
  "Source file",
  "Line",
  "Image syntax",
  "Alt text",
  "Caption / nearby label",
  "Target",
  "Local file exists",
  "Check: caption/source credit",
  "Check: third-party permission/license",
  "Check: AI-generated image or AI manipulation disclosure",
  "Check: accessibility alt text/contrast",
  "Check: ethics/privacy/sensitive content",
  "Check: data/code support for result figure",
  "Check status",
  "Review note",
];

const TABLE_HEADERS = [
  "ID",
  "Source chapter",
  "Source file",
  "Start line",
  "End line",
  "Caption / nearby label",
  "Columns",
  "Data rows",
  "Header preview",
  "Check: caption/source credit",
  "Check: third-party data/material permission",
  "Check: data/code availability statement needed",
  "Check: accessibility/readability",
  "Check: ethics/privacy/sensitive content",
  "Check status",
  "Review note",
];

const REF_HEADERS = [
  "ID",
  "Source chapter",
  "Source file",
  "Line",
  "Reference section",
  "Reference text",
  "Detected DOI / URL / arXiv",
  "Check: supports claim/relevance",
  "Check: source read and bibliographic accuracy",
  "Check: original/peer-reviewed source preferred",
  "Check: DOI/URL present and verified",
  "Check: not hallucinated/retracted/unreliable",
  "Check: citation balance/no excessive self-citation",
  "Check status",
  "Review note",
];

const CHAPTER_CHECK_HEADERS = [
  "ID",
  "Source chapter",
  "Source file",
  "Images",
  "Tables",
  "References",
  "Code blocks",
  "External URLs",
  "AI/model mentions",
  "Privacy/ethics mentions",
  "Permission/source mentions",
  "Check: AI disclosure complete",
  "Check: third-party rights complete",
  "Check: citations accurate/relevant",
  "Check: data/code availability",
  "Check: accessibility",
  "Check: ethics/privacy/harm boundary",
  "Check status",
  "Review note",
];

const AI_DISCLOSURE_HEADERS = [
  "ID",
  "Source chapter",
  "Source file",
  "Images",
  "AI/model mentions",
  "Synthetic/generated mentions",
  "Check: no prohibited AI-generated images",
  "Check: allowed AI-image exception labelled in image field",
  "Check: AI-assisted text/code verified and referenced",
  "Check: AI-assisted citation changes reviewed",
  "Check: non-generative image manipulation disclosed in caption",
  "Check status",
  "Review note",
];

const PERMISSIONS_HEADERS = [
  "ID",
  "Source chapter",
  "Source file",
  "Images",
  "Tables",
  "External URLs",
  "Source/license/copyright mentions",
  "Logo/screenshot/trademark mentions",
  "Check: third-party materials inventory complete",
  "Check: written permission or licence recorded",
  "Check: required credit wording in caption/table note",
  "Check: reuse from prior publication cleared",
  "Check: OA/print/eBook territory rights covered",
  "Check status",
  "Review note",
];

const DATA_CODE_HEADERS = [
  "ID",
  "Source chapter",
  "Source file",
  "Tables",
  "Images",
  "Code blocks",
  "Data/code/repository mentions",
  "Check: data availability statement present or not applicable",
  "Check: code availability statement present or not applicable",
  "Check: repository/DOI/URL resolves",
  "Check: privacy/security access limits stated",
  "Check: reproducibility metadata sufficient",
  "Check status",
  "Review note",
];

const ACCESSIBILITY_HEADERS = [
  "ID",
  "Source chapter",
  "Source file",
  "Images",
  "Images with missing/short alt",
  "Tables",
  "Wide tables (>6 columns)",
  "Code blocks",
  "Check: meaningful alt text",
  "Check: figure text/contrast/readability",
  "Check: table headers and reading order",
  "Check: code blocks readable in print/eBook",
  "Check: headings/navigation structure",
  "Check status",
  "Review note",
];

const ETHICS_PRIVACY_HEADERS = [
  "ID",
  "Source chapter",
  "Source file",
  "Privacy/PII mentions",
  "Medical/human-subject mentions",
  "Safety/harmful-content mentions",
  "Territory/map/political mentions",
  "Check: privacy/publicity rights not violated",
  "Check: anonymization/consent/permission addressed",
  "Check: harmful or dual-use content bounded",
  "Check: medical/legal/financial advice boundaries clear",
  "Check: territorial neutrality issues reviewed",
  "Check: no defamatory/obscene/infringing content",
  "Check status",
  "Review note",
];

function rel(file) {
  return path.relative(ROOT, file).split(path.sep).join("/");
}

async function listMarkdownFiles(dir) {
  const out = [];
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...(await listMarkdownFiles(full)));
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      out.push(full);
    }
  }
  return out.sort((a, b) => rel(a).localeCompare(rel(b)));
}

function inFenceAt(lines, index) {
  let inFence = false;
  for (let i = 0; i <= index; i += 1) {
    if (/^\s*```/.test(lines[i])) {
      inFence = !inFence;
    }
  }
  return inFence;
}

function chapterTitle(lines, fallback) {
  const title = lines.find((line) => /^#\s+/.test(line));
  return title ? title.replace(/^#\s+/, "").trim() : fallback;
}

function nearbyCaption(lines, index, kind) {
  const pattern = kind === "table" ? /\btable\s+[A-Za-z0-9_.-]+/i : /\bfigure\s+[A-Za-z0-9_.-]+/i;
  for (let offset = 1; offset <= 4; offset += 1) {
    const line = lines[index + offset];
    if (line && isTableLine(line)) continue;
    if (line && pattern.test(line)) return cleanupInline(line);
  }
  for (let offset = 1; offset <= 3; offset += 1) {
    const line = lines[index - offset];
    if (line && isTableLine(line)) continue;
    if (line && pattern.test(line)) return cleanupInline(line);
  }
  return "";
}

function cleanupInline(value) {
  return value
    .replace(/<\/?[^>]+>/g, "")
    .replace(/^[*_`>\s-]+|[*_`\s]+$/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function resolveImageTarget(sourceFile, target) {
  if (/^(?:https?:|data:|file:|#)/i.test(target)) return { exists: "", normalized: target };
  const clean = target.split("#")[0].split("?")[0];
  const full = path.resolve(path.dirname(sourceFile), clean);
  return { exists: "", normalized: rel(full) };
}

async function localExistsFromTarget(sourceFile, target) {
  if (/^(?:https?:|data:|file:|#)/i.test(target)) return "";
  const clean = target.split("#")[0].split("?")[0];
  try {
    await fs.access(path.resolve(path.dirname(sourceFile), clean));
    return "yes";
  } catch {
    return "no";
  }
}

async function extractImages(sourceFile, lines, sourceChapter, startId) {
  const rows = [];
  let id = startId;
  const mdImage = /!\[([^\]]*)\]\(([^)]+)\)/g;
  const htmlImage = /<img\b[^>]*\bsrc=(["'])(.*?)\1[^>]*>/gi;

  for (let i = 0; i < lines.length; i += 1) {
    if (inFenceAt(lines, i)) continue;
    let match;
    mdImage.lastIndex = 0;
    while ((match = mdImage.exec(lines[i])) !== null) {
      const target = match[2].trim();
      const resolved = resolveImageTarget(sourceFile, target);
      rows.push([
        `IMG-${String(id).padStart(4, "0")}`,
        sourceChapter,
        rel(sourceFile),
        i + 1,
        "Markdown",
        cleanupInline(match[1]),
        nearbyCaption(lines, i, "figure"),
        resolved.normalized,
        await localExistsFromTarget(sourceFile, target),
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
      ]);
      id += 1;
    }

    htmlImage.lastIndex = 0;
    while ((match = htmlImage.exec(lines[i])) !== null) {
      const target = match[2].trim();
      const resolved = resolveImageTarget(sourceFile, target);
      rows.push([
        `IMG-${String(id).padStart(4, "0")}`,
        sourceChapter,
        rel(sourceFile),
        i + 1,
        "HTML img",
        "",
        nearbyCaption(lines, i, "figure"),
        resolved.normalized,
        await localExistsFromTarget(sourceFile, target),
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
      ]);
      id += 1;
    }
  }
  return rows;
}

function isTableSeparator(line) {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function isTableLine(line) {
  return /^\s*\|.*\|\s*$/.test(line);
}

function splitCells(line) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cleanupInline(cell));
}

function extractTables(sourceFile, lines, sourceChapter, startId) {
  const rows = [];
  let id = startId;
  let i = 0;
  while (i < lines.length - 1) {
    if (inFenceAt(lines, i) || !isTableLine(lines[i]) || !isTableSeparator(lines[i + 1])) {
      i += 1;
      continue;
    }
    const start = i;
    const header = splitCells(lines[i]);
    i += 2;
    let dataRows = 0;
    while (i < lines.length && isTableLine(lines[i]) && !inFenceAt(lines, i)) {
      dataRows += 1;
      i += 1;
    }
    const end = i;
    rows.push([
      `TBL-${String(id).padStart(4, "0")}`,
      sourceChapter,
      rel(sourceFile),
      start + 1,
      end,
      nearbyCaption(lines, end - 1, "table"),
      header.length,
      dataRows,
      header.join(" | "),
      "",
      "",
      "",
      "",
      "",
      "",
      "",
    ]);
    id += 1;
  }
  return rows;
}

function markerFromReference(text) {
  const markers = [];
  const doi = text.match(/\b10\.\d{4,9}\/[-._;()/:A-Z0-9]+/i);
  const arxiv = text.match(/\barXiv:\s*\d{4}\.\d{4,5}\b/i);
  const url = text.match(/https?:\/\/\S+/i);
  if (doi) markers.push(`DOI ${doi[0]}`);
  if (arxiv) markers.push(arxiv[0].replace(/\s+/g, ""));
  if (url) markers.push(url[0].replace(/[).,;]+$/, ""));
  return markers.join("; ");
}

function extractReferences(sourceFile, lines, sourceChapter, startId) {
  const rows = [];
  let id = startId;
  for (let i = 0; i < lines.length; i += 1) {
    const heading = lines[i].match(/^(#{1,6})\s+(References|Bibliography)\s*$/i);
    if (!heading) continue;
    const sectionTitle = cleanupInline(lines[i].replace(/^#{1,6}\s+/, ""));
    let paragraph = [];
    let paragraphStart = i + 2;
    for (let j = i + 1; j <= lines.length; j += 1) {
      const line = lines[j] ?? "";
      if (j === lines.length || /^#{1,6}\s+/.test(line)) {
        if (paragraph.length) {
          const text = cleanupInline(paragraph.join(" "));
          rows.push([
            `REF-${String(id).padStart(4, "0")}`,
            sourceChapter,
            rel(sourceFile),
            paragraphStart,
            sectionTitle,
            text,
            markerFromReference(text),
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
          ]);
          id += 1;
        }
        break;
      }
      if (!line.trim()) {
        if (paragraph.length) {
          const text = cleanupInline(paragraph.join(" "));
          rows.push([
            `REF-${String(id).padStart(4, "0")}`,
            sourceChapter,
            rel(sourceFile),
            paragraphStart,
            sectionTitle,
            text,
            markerFromReference(text),
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
          ]);
          id += 1;
          paragraph = [];
        }
        paragraphStart = j + 2;
      } else {
        if (!paragraph.length) paragraphStart = j + 1;
        paragraph.push(line.replace(/^[-*]\s+/, ""));
      }
    }
  }
  return rows;
}

function countMatches(text, regex) {
  return (text.match(regex) || []).length;
}

function countCodeBlocks(lines) {
  return Math.floor(lines.filter((line) => /^\s*```/.test(line)).length / 2);
}

function countExternalUrls(text) {
  return countMatches(text, /https?:\/\/[^\s)]+/gi);
}

function countKeywordGroup(text, words) {
  return words.reduce((sum, word) => sum + countMatches(text, new RegExp(`\\b${word}\\b`, "gi")), 0);
}

function extractChapterChecks(files, meta, imageRows, tableRows, refRows) {
  const imagesByFile = new Map();
  const tablesByFile = new Map();
  const refsByFile = new Map();
  for (const row of imageRows) {
    const rows = imagesByFile.get(row[2]) || [];
    rows.push(row);
    imagesByFile.set(row[2], rows);
  }
  for (const row of tableRows) tablesByFile.set(row[2], (tablesByFile.get(row[2]) || 0) + 1);
  for (const row of refRows) refsByFile.set(row[2], (refsByFile.get(row[2]) || 0) + 1);

  const chapterRows = [];
  const aiRows = [];
  const permissionRows = [];
  const dataCodeRows = [];
  const accessibilityRows = [];
  const ethicsRows = [];

  files.forEach((file, index) => {
    const sourceFile = rel(file);
    const item = meta.get(sourceFile);
    const text = item.text;
    const imageList = imagesByFile.get(sourceFile) || [];
    const imageCount = imageList.length;
    const tableCount = tablesByFile.get(sourceFile) || 0;
    const refCount = refsByFile.get(sourceFile) || 0;
    const codeCount = countCodeBlocks(item.lines);
    const externalUrlCount = countExternalUrls(text);
    const aiMentions = countKeywordGroup(text, [
      "AI",
      "LLM",
      "VLM",
      "Qwen",
      "Qwen-VL",
      "GPT",
      "generative",
      "generated",
      "synthetic",
    ]);
    const syntheticMentions = countKeywordGroup(text, ["synthetic", "generated", "generation", "AI-generated"]);
    const privacyMentions = countKeywordGroup(text, [
      "privacy",
      "PII",
      "personal",
      "sensitive",
      "consent",
      "anonymization",
      "medical",
      "patient",
      "harm",
      "safety",
      "copyright",
    ]);
    const permissionMentions = countKeywordGroup(text, [
      "source",
      "license",
      "licence",
      "permission",
      "copyright",
      "authorization",
      "attribution",
      "credit",
    ]);
    const logoScreenshotTrademarkMentions = countKeywordGroup(text, ["logo", "screenshot", "trademark", "brand"]);
    const dataCodeMentions = countKeywordGroup(text, [
      "data",
      "dataset",
      "code",
      "repository",
      "GitHub",
      "DOI",
      "artifact",
      "reproducible",
      "availability",
    ]);
    const medicalHumanMentions = countKeywordGroup(text, ["medical", "patient", "human", "clinical", "health"]);
    const safetyMentions = countKeywordGroup(text, ["safety", "harm", "dangerous", "abuse", "attack", "red-team", "jailbreak"]);
    const territoryMentions = countKeywordGroup(text, ["map", "territory", "jurisdiction", "country", "region", "political"]);
    const missingAlt = imageList.filter((row) => String(row[5] || "").trim().length < 20).length;
    const wideTables = tableRows.filter((row) => row[2] === sourceFile && Number(row[6]) > 6).length;
    const rowId = `CH-${String(index + 1).padStart(3, "0")}`;

    chapterRows.push([
      rowId,
      item.title,
      sourceFile,
      imageCount,
      tableCount,
      refCount,
      codeCount,
      externalUrlCount,
      aiMentions,
      privacyMentions,
      permissionMentions,
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
    ]);
    aiRows.push([
      `AI-${String(index + 1).padStart(3, "0")}`,
      item.title,
      sourceFile,
      imageCount,
      aiMentions,
      syntheticMentions,
      "",
      "",
      "",
      "",
      "",
      "",
      "",
    ]);
    permissionRows.push([
      `PERM-${String(index + 1).padStart(3, "0")}`,
      item.title,
      sourceFile,
      imageCount,
      tableCount,
      externalUrlCount,
      permissionMentions,
      logoScreenshotTrademarkMentions,
      "",
      "",
      "",
      "",
      "",
      "",
      "",
    ]);
    dataCodeRows.push([
      `DC-${String(index + 1).padStart(3, "0")}`,
      item.title,
      sourceFile,
      tableCount,
      imageCount,
      codeCount,
      dataCodeMentions,
      "",
      "",
      "",
      "",
      "",
      "",
      "",
    ]);
    accessibilityRows.push([
      `ACC-${String(index + 1).padStart(3, "0")}`,
      item.title,
      sourceFile,
      imageCount,
      missingAlt,
      tableCount,
      wideTables,
      codeCount,
      "",
      "",
      "",
      "",
      "",
      "",
      "",
    ]);
    ethicsRows.push([
      `ETH-${String(index + 1).padStart(3, "0")}`,
      item.title,
      sourceFile,
      privacyMentions,
      medicalHumanMentions,
      safetyMentions,
      territoryMentions,
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
    ]);
  });

  return {
    chapterRows,
    aiRows,
    permissionRows,
    dataCodeRows,
    accessibilityRows,
    ethicsRows,
  };
}

function summarize(files, imageRows, tableRows, refRows) {
  const map = new Map();
  for (const file of files) {
    map.set(rel(file), {
      sourceFile: rel(file),
      sourceChapter: "",
      images: 0,
      tables: 0,
      references: 0,
    });
  }
  for (const row of imageRows) {
    const item = map.get(row[2]);
    item.sourceChapter = row[1];
    item.images += 1;
  }
  for (const row of tableRows) {
    const item = map.get(row[2]);
    item.sourceChapter = row[1];
    item.tables += 1;
  }
  for (const row of refRows) {
    const item = map.get(row[2]);
    item.sourceChapter = row[1];
    item.references += 1;
  }
  return [...map.values()]
    .filter((item) => item.images || item.tables || item.references)
    .map((item) => [item.sourceChapter, item.sourceFile, item.images, item.tables, item.references]);
}

async function collectIndexRows() {
  const files = await listMarkdownFiles(DOCS_EN);
  const imageRows = [];
  const tableRows = [];
  const refRows = [];
  const meta = new Map();

  for (const file of files) {
    const text = await fs.readFile(file, "utf8");
    const lines = text.split(/\r?\n/);
    const title = chapterTitle(lines, rel(file));
    meta.set(rel(file), { title, text, lines });
    imageRows.push(...(await extractImages(file, lines, title, imageRows.length + 1)));
    tableRows.push(...extractTables(file, lines, title, tableRows.length + 1));
    refRows.push(...extractReferences(file, lines, title, refRows.length + 1));
  }
  const chapterChecks = extractChapterChecks(files, meta, imageRows, tableRows, refRows);

  return {
    files,
    images: imageRows,
    tables: tableRows,
    references: refRows,
    summary: summarize(files, imageRows, tableRows, refRows),
    ...chapterChecks,
  };
}

function writeSheet(sheet, headers, rows, widths) {
  const matrix = [headers, ...rows];
  sheet.getRangeByIndexes(0, 0, matrix.length, headers.length).values = matrix;
  sheet.getRangeByIndexes(0, 0, 1, headers.length).format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
  };
  sheet.getRangeByIndexes(0, 0, matrix.length, headers.length).format.wrapText = true;
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;
  widths.forEach((width, index) => {
    sheet.getRangeByIndexes(0, index, 1, 1).format.columnWidth = width;
  });
}

async function buildWorkbook() {
  const rows = await collectIndexRows();
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });

  const workbook = Workbook.create();
  const summary = workbook.worksheets.add("Summary");
  const images = workbook.worksheets.add("Images");
  const tables = workbook.worksheets.add("Tables");
  const references = workbook.worksheets.add("References");
  const chapterChecklist = workbook.worksheets.add("Chapter_Checklist");
  const aiDisclosure = workbook.worksheets.add("AI_Disclosure");
  const permissions = workbook.worksheets.add("Permissions");
  const dataCode = workbook.worksheets.add("Data_Code");
  const accessibility = workbook.worksheets.add("Accessibility");
  const ethicsPrivacy = workbook.worksheets.add("Ethics_Privacy");

  const summaryRows = [
    ["Metric", "Count"],
    ["Markdown files scanned", rows.files.length],
    ["Images found", rows.images.length],
    ["Tables found", rows.tables.length],
    ["References found", rows.references.length],
    ["Chapter policy rows", rows.chapterRows.length],
    [],
    ["Source chapter", "Source file", "Images", "Tables", "References"],
    ...rows.summary,
  ];
  summary.getRangeByIndexes(0, 0, summaryRows.length, 5).values = summaryRows.map((row) => {
    const filled = [...row];
    while (filled.length < 5) filled.push("");
    return filled;
  });
  summary.getRange("A1:B1").format = { fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" } };
  summary.getRange("A7:E7").format = { fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" } };
  summary.getRangeByIndexes(0, 0, summaryRows.length, 5).format.wrapText = true;
  summary.freezePanes.freezeRows(7);
  summary.showGridLines = false;
  [28, 58, 12, 12, 14].forEach((width, index) => {
    summary.getRangeByIndexes(0, index, 1, 1).format.columnWidth = width;
  });

  writeSheet(images, IMAGE_HEADERS, rows.images, [14, 42, 55, 10, 16, 30, 55, 55, 16, 24, 28, 34, 28, 28, 30, 18, 28]);
  writeSheet(tables, TABLE_HEADERS, rows.tables, [14, 42, 55, 12, 12, 55, 10, 10, 60, 24, 32, 34, 26, 28, 18, 28]);
  writeSheet(references, REF_HEADERS, rows.references, [14, 42, 55, 10, 18, 100, 36, 28, 34, 34, 32, 34, 36, 18, 28]);
  writeSheet(chapterChecklist, CHAPTER_CHECK_HEADERS, rows.chapterRows, [12, 42, 55, 10, 10, 12, 12, 12, 14, 16, 18, 26, 28, 30, 26, 22, 30, 18, 30]);
  writeSheet(aiDisclosure, AI_DISCLOSURE_HEADERS, rows.aiRows, [12, 42, 55, 10, 16, 18, 32, 34, 34, 32, 36, 18, 30]);
  writeSheet(permissions, PERMISSIONS_HEADERS, rows.permissionRows, [14, 42, 55, 10, 10, 14, 22, 24, 34, 32, 34, 30, 30, 18, 30]);
  writeSheet(dataCode, DATA_CODE_HEADERS, rows.dataCodeRows, [12, 42, 55, 10, 10, 12, 24, 36, 36, 28, 34, 30, 18, 30]);
  writeSheet(accessibility, ACCESSIBILITY_HEADERS, rows.accessibilityRows, [12, 42, 55, 10, 22, 10, 20, 12, 26, 30, 28, 30, 26, 18, 30]);
  writeSheet(ethicsPrivacy, ETHICS_PRIVACY_HEADERS, rows.ethicsRows, [12, 42, 55, 18, 22, 24, 24, 32, 34, 30, 34, 30, 34, 18, 30]);

  for (const sheetName of [
    "Summary",
    "Images",
    "Tables",
    "References",
    "Chapter_Checklist",
    "AI_Disclosure",
    "Permissions",
    "Data_Code",
    "Accessibility",
    "Ethics_Privacy",
  ]) {
    const preview = await workbook.render({
      sheetName,
      range: sheetName === "Summary" ? "A1:E28" : "A1:O25",
      scale: 1,
      format: "png",
    });
    await fs.writeFile(
      path.join(PREVIEW_DIR, `${sheetName}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }

  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(OUT_XLSX);
  return rows;
}

const rows = await buildWorkbook();
console.log(`xlsx=${OUT_XLSX}`);
console.log(`markdown_files=${rows.files.length}`);
console.log(`images=${rows.images.length}`);
console.log(`tables=${rows.tables.length}`);
console.log(`references=${rows.references.length}`);
