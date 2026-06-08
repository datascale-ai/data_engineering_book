#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate final Springer handoff audit reports.

This script does not replace human editorial review. It creates the evidence
pack reviewers need for final line editing, figure rights / AI disclosure,
reference DOI/style checks, and high-risk chapter spot checks.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ZH_ROOT = ROOT / "docs" / "zh"
FIGURE_REGISTER = ROOT / "publishing" / "12_figures_tables_register.md"
DEFAULT_REPORT_DIR = ROOT / "publishing" / "final_review"

STYLE_PATTERNS: list[tuple[str, str]] = [
    ("colloquial-you", r"你可以|你会发现|你会看到"),
    ("weak-booster", r"真正|很容易|显然|毫无疑问"),
    ("blog-transition", r"换句话说|简单来说|说白了|最怕"),
    ("rhetorical-not-but", r"不是.{0,18}而是"),
    ("slang", r"踩坑|搞定|一把梭|玄学|爽点|杀手锏"),
]

REVIEW_TARGETS: list[tuple[str, str, str]] = [
    ("sample-long", "Ch12", "docs/zh/part4/ch12_sft.md"),
    ("sample-long", "Ch16", "docs/zh/part5/ch16_distillation.md"),
    ("sample-long", "Ch21", "docs/zh/part7/ch21_rag_pipeline.md"),
    ("sample-long", "Ch24", "docs/zh/part8/ch24_dataops_flywheel_team.md"),
    ("sample-long", "Ch29", "docs/zh/part9/ch29_data_valuation_and_reuse.md"),
    ("sample-risk", "Ch40", "docs/zh/part12/ch40_multi_chart_infographic_reasoning_dataset.md"),
    ("sample-project", "P11", "docs/zh/part14/p11_mini_deepseek.md"),
    ("sample-project", "P12", "docs/zh/part14/p12_r1_reasoning_flywheel.md"),
    ("sample-project", "P13", "docs/zh/part14/p13_multimodal_instruction_factory.md"),
    ("sample-project", "P15", "docs/zh/part14/p15_dataagent_semantic_nl2sql_agent.md"),
]

HIGH_RISK_PARTS = {
    "Part 10": ROOT / "docs" / "zh" / "part10",
    "Part 12": ROOT / "docs" / "zh" / "part12",
    "Part 14": ROOT / "docs" / "zh" / "part14",
}

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
URL_RE = re.compile(r"https?://\S+")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
REF_HEADER_RE = re.compile(r"^##\s*参考文献\s*$")
NEXT_H2_RE = re.compile(r"^##\s+")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


@dataclass
class StyleHit:
    file: str
    line: int
    kind: str
    phrase: str
    context: str


@dataclass
class FigureAudit:
    file: str
    line: int
    alt: str
    src: str
    resolved: str
    exists: bool
    size_kb: float | None
    dimensions: str
    registry_status: str
    high_res_status: str
    issues: list[str]


@dataclass
class ReferenceAudit:
    file: str
    entry_no: int
    entry: str
    year: str
    doi: str
    url: str
    issues: list[str]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def book_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(ZH_ROOT.rglob("*.md")):
        if "superpowers" in path.parts or path.name in {"translation-status.md", "index.md"}:
            continue
        name = path.name
        if name.startswith("ch") or re.match(r"p\d+", name) or "appendix" in name:
            files.append(path)
    return files


def title_of(path: Path) -> str:
    for line in read_text(path).splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def scan_style(files: Iterable[Path]) -> list[StyleHit]:
    hits: list[StyleHit] = []
    for path in files:
        for lineno, line in enumerate(read_text(path).splitlines(), 1):
            for kind, pattern in STYLE_PATTERNS:
                m = re.search(pattern, line)
                if not m:
                    continue
                context = line.strip()
                if len(context) > 120:
                    context = context[:117] + "..."
                hits.append(StyleHit(rel(path), lineno, kind, m.group(0), context))
    return hits


def image_dimensions(path: Path) -> str:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return "unknown"
    try:
        with Image.open(path) as image:
            return f"{image.width}x{image.height}"
    except Exception:
        return "unknown"


def normalized_image_key(src: str, doc_path: Path | None = None) -> str:
    if src.startswith(("http://", "https://")):
        return src
    if doc_path is not None:
        candidate = (doc_path.parent / src).resolve()
        try:
            return rel(candidate)
        except ValueError:
            return str(candidate)
    normalized = src.strip().strip("`")
    while normalized.startswith("../"):
        normalized = normalized[3:]
    if normalized.startswith("images/"):
        normalized = f"docs/{normalized}"
    return normalized


def parse_figure_register() -> dict[str, tuple[str, str]]:
    if not FIGURE_REGISTER.exists():
        return {}
    mapping: dict[str, tuple[str, str]] = {}
    for line in read_text(FIGURE_REGISTER).splitlines():
        if not line.startswith("| 图"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 9:
            continue
        src_match = re.search(r"`([^`]+)`", cells[5])
        if not src_match:
            continue
        key = normalized_image_key(src_match.group(1))
        mapping[key] = (cells[6], cells[8])
    return mapping


def scan_figures(files: Iterable[Path]) -> list[FigureAudit]:
    register = parse_figure_register()
    rows: list[FigureAudit] = []
    for path in files:
        for lineno, line in enumerate(read_text(path).splitlines(), 1):
            for match in IMAGE_RE.finditer(line):
                alt, src = match.group(1).strip(), match.group(2).strip()
                if src.startswith(("http://", "https://")):
                    rows.append(FigureAudit(rel(path), lineno, alt, src, src, True, None, "remote", "external URL", "manual", ["external-url-rights"]))
                    continue
                resolved_path = (path.parent / src).resolve()
                exists = resolved_path.exists()
                key = normalized_image_key(src, path)
                status, high_res = register.get(key, ("missing-register", "unknown"))
                issues: list[str] = []
                if not exists:
                    issues.append("missing-file")
                if status == "missing-register":
                    issues.append("missing-figure-register")
                if "复核" in status or "AI 使用声明" in status:
                    issues.append("needs-rights-ai-review")
                if high_res in {"是", "unknown"}:
                    issues.append("needs-high-res-confirmation")
                size_kb = resolved_path.stat().st_size / 1024 if exists else None
                dims = image_dimensions(resolved_path) if exists else "missing"
                rows.append(FigureAudit(rel(path), lineno, alt, src, key, exists, size_kb, dims, status, high_res, issues))
    return rows


def extract_references(path: Path) -> list[str]:
    lines = read_text(path).splitlines()
    in_refs = False
    entries: list[str] = []
    for line in lines:
        if REF_HEADER_RE.match(line.strip()):
            in_refs = True
            continue
        if in_refs and NEXT_H2_RE.match(line.strip()):
            break
        if not in_refs:
            continue
        item = line.strip()
        if not item or item in {"---", "----"}:
            continue
        item = re.sub(r"^[-*]\s+", "", item)
        entries.append(item)
    return entries


def scan_references(files: Iterable[Path]) -> list[ReferenceAudit]:
    rows: list[ReferenceAudit] = []
    for path in files:
        for idx, entry in enumerate(extract_references(path), 1):
            doi_match = DOI_RE.search(entry)
            url_match = URL_RE.search(entry)
            year_match = YEAR_RE.search(entry)
            issues: list[str] = ["needs-authenticity-review"]
            if not year_match:
                issues.append("missing-year")
            if not doi_match and not url_match and "arXiv:" not in entry:
                issues.append("missing-doi-url-arxiv")
            if entry.startswith("[") or "](" in entry:
                issues.append("markdown-link-style")
            if not entry.endswith((".", "。")):
                issues.append("missing-terminal-period")
            rows.append(
                ReferenceAudit(
                    rel(path),
                    idx,
                    entry,
                    year_match.group(0) if year_match else "",
                    doi_match.group(0) if doi_match else "",
                    url_match.group(0) if url_match else "",
                    issues,
                )
            )
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_style_report(path: Path, hits: list[StyleHit], max_per_file: int) -> None:
    by_file: dict[str, list[StyleHit]] = {}
    for hit in hits:
        by_file.setdefault(hit.file, []).append(hit)
    out = ["# 全书逐章统稿与风格精修报告", "", f"- 风格候选命中：{len(hits)}", "- 说明：本报告列出需要人工判断的候选表达，不表示一定必须删除。", ""]
    for file, file_hits in sorted(by_file.items()):
        out.append(f"## {file}")
        for hit in file_hits[:max_per_file]:
            out.append(f"- `{hit.line}` `{hit.kind}` `{hit.phrase}`：{hit.context}")
        if len(file_hits) > max_per_file:
            out.append(f"- 其余 {len(file_hits) - max_per_file} 条见 JSON 明细。")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def write_chapter_checklist(path: Path, files: list[Path], hits: list[StyleHit]) -> None:
    hit_counts: dict[str, int] = {}
    for hit in hits:
        hit_counts[hit.file] = hit_counts.get(hit.file, 0) + 1
    out = [
        "# 全书逐章统稿签核矩阵",
        "",
        "本表用于主编和章节作者逐章确认语言、逻辑、术语、图表引用、参考文献和出版体例；脚本只给出候选风险数量，最终状态需人工填写。",
        "",
        "| 文件 | 标题 | 风格候选 | 统稿重点 | 状态 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    axes = "语言风格；术语统一；章节逻辑；图表引用；参考文献；Springer 体例"
    for file in files:
        file_key = rel(file)
        title = title_of(file).replace("|", "\\|")
        out.append(f"| `{file_key}` | {title} | {hit_counts.get(file_key, 0)} | {axes} | 待人工签核 |")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_figure_report(path: Path, rows: list[FigureAudit]) -> None:
    issue_rows = [row for row in rows if row.issues]
    out = [
        "# 图表版权、AI 声明与高清源文件终审报告",
        "",
        f"- 图片引用总数：{len(rows)}",
        f"- 需要人工确认：{len(issue_rows)}",
        "- 说明：每一行都需要图表编辑最终确认；`needs-rights-ai-review` 表示台账仍要求终稿阶段复核权属或 AI 使用声明；`needs-high-res-confirmation` 表示需确认高清源文件或出版社可用分辨率。",
        "",
        "| 文件 | 行 | 图片 | 尺寸 | 大小 KB | 台账权属状态 | 高清源状态 | 问题 |",
        "| --- | ---: | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        size = "" if row.size_kb is None else f"{row.size_kb:.1f}"
        issues = ", ".join(row.issues) if row.issues else "manual-final-confirmation"
        out.append(f"| `{row.file}` | {row.line} | `{row.resolved}` | {row.dimensions} | {size} | {row.registry_status} | {row.high_res_status} | {issues} |")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_reference_report(path: Path, rows: list[ReferenceAudit], max_rows: int) -> None:
    issue_rows = [row for row in rows if len(row.issues) > 1]
    out = [
        "# 参考文献 DOI / Springer 样式 / 真实性终审报告",
        "",
        f"- 参考文献条目：{len(rows)}",
        f"- DOI/URL/年份/样式候选问题：{len(issue_rows)}",
        "- 所有条目默认需要人工真实性终审；脚本列出 DOI、URL、年份和样式候选问题，但不替代逐条查证。",
        "",
        "| 文件 | 序号 | 年份 | DOI | URL | 问题 | 条目 |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows[:max_rows]:
        entry = row.entry.replace("|", "\\|")
        if len(entry) > 180:
            entry = entry[:177] + "..."
        out.append(f"| `{row.file}` | {row.entry_no} | {row.year} | {row.doi} | {row.url} | {', '.join(row.issues)} | {entry} |")
    if len(rows) > max_rows:
        out.append(f"\n> 其余 {len(rows) - max_rows} 条见 JSON 明细。")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_manual_checklist(path: Path) -> None:
    out = [
        "# 高风险章节与抽检章节人工复核清单",
        "",
        "## 一、指定抽检章节",
        "",
        "| 类型 | 单元 | 文件 | 复核项 | 状态 |",
        "| --- | --- | --- | --- | --- |",
    ]
    axes = "摘要/关键词；术语；图表权属；参考文献真实性；代码长度；案例边界；章末小结"
    for kind, unit, file in REVIEW_TARGETS:
        out.append(f"| {kind} | {unit} | `{file}` | {axes} | 待人工签核 |")
    out.extend(["", "## 二、Part 10 / Part 12 / Part 14 高风险范围", ""])
    for part_name, root in HIGH_RISK_PARTS.items():
        out.append(f"### {part_name}")
        for file in sorted(root.glob("*.md")):
            if file.name == "index.md":
                continue
            out.append(f"- [ ] `{rel(file)}`：安全/合规边界、图表权属、参考文献真实性、案例复现边界。")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final publication audit reports.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for generated reports.")
    parser.add_argument("--max-style-per-file", type=int, default=8, help="Maximum style hits shown per file in Markdown.")
    parser.add_argument("--max-reference-rows", type=int, default=300, help="Maximum reference rows shown in Markdown.")
    parser.add_argument("--fail-on-blocker", action="store_true", help="Exit 1 when broken images or missing references are found.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    files = book_files()
    style_hits = scan_style(files)
    figures = scan_figures(files)
    references = scan_references(files)
    missing_reference_files = [rel(path) for path in files if not extract_references(path)]
    broken_figures = [row for row in figures if not row.exists]

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at_utc": generated_at,
        "scanned_files": len(files),
        "style_hits": [asdict(hit) for hit in style_hits],
        "figures": [asdict(row) for row in figures],
        "references": [asdict(row) for row in references],
        "missing_reference_files": missing_reference_files,
        "summary": {
            "style_hits": len(style_hits),
            "figures": len(figures),
            "figures_needing_review": sum(1 for row in figures if row.issues),
            "broken_figures": len(broken_figures),
            "references": len(references),
            "reference_issue_rows": sum(1 for row in references if len(row.issues) > 1),
            "missing_reference_files": len(missing_reference_files),
        },
    }

    write_json(report_dir / "final_publication_audit.json", payload)
    write_style_report(report_dir / "style_report.md", style_hits, args.max_style_per_file)
    write_chapter_checklist(report_dir / "chapter_style_checklist.md", files, style_hits)
    write_figure_report(report_dir / "figure_rights_report.md", figures)
    write_reference_report(report_dir / "reference_audit_report.md", references, args.max_reference_rows)
    write_manual_checklist(report_dir / "manual_review_checklist.md")
    (report_dir / "README.md").write_text(
        "\n".join(
            [
                "# Springer 终稿人工复核报告包",
                "",
                f"生成时间：{generated_at}",
                "",
                "- `chapter_style_checklist.md`：全书逐章统稿签核矩阵。",
                "- `style_report.md`：逐章统稿与风格候选表达。",
                "- `figure_rights_report.md`：图表权属、AI 声明与高清源文件确认清单。",
                "- `reference_audit_report.md`：参考文献 DOI / URL / 年份 / 样式候选问题。",
                "- `manual_review_checklist.md`：指定抽检章节与高风险篇章人工签核清单。",
                "- `final_publication_audit.json`：完整机器可读明细。",
                "",
                "本报告包用于终稿阶段人工确认，不表示所有人工审校已经完成。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Final publication audit generated")
    for key, value in payload["summary"].items():
        print(f"- {key}: {value}")

    if args.fail_on_blocker and (broken_figures or missing_reference_files):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
