#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
publish_lint.py — Springer 出版就绪度校验

针对 docs/zh/ 的章节，逐项检查并输出一份完整清单：
  1. 断链图        : 引用的本地图片在磁盘上不存在，或为本机绝对路径
  2. 图号-章号一致 : 正文图号 图X-Y / 图X_Y 的前缀 X 与所在章号不一致
  3. 缺失 Abstract : 章节正文未出现 摘要 / Abstract
  4. 缺失 关键词    : 章节正文未出现 ## 关键词 / ## Keywords
  5. 缺失 参考文献  : 章节未出现 ## 参考文献 / References 节
  6. 缺失本章小结   : 章节正文未出现 ## 本章小结
  7. 项目章结构     : 项目章缺少案例研究关键模块
  8. 项目章图表     : P04-P13 至少包含 1 张图和 1 张出版验收表
  9. 疑似未成稿     : 含写作任务书标记（写作目标/篇幅要求/建议结构…）或正文过短

用法:
  python scripts/publish_lint.py                # 控制台彩色摘要 + 详表
  python scripts/publish_lint.py --json out.json
  python scripts/publish_lint.py --md report.md # 输出 Markdown 报告
退出码: 发现任何 ERROR 级问题 -> 1, 否则 0
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZH = ROOT / "docs" / "zh"

# ---- 阈值（与 publishing/01_editorial_blueprint.md 的配额对齐）----
MIN_CHAPTER_CHARS = 6000   # 正文章建议 1.5-2万字；低于此判定 "疑似未成稿"
MIN_PROJECT_CHARS = 5000   # 项目章建议 1-1.3万字
STUB_MARKERS = ["写作目标", "篇幅要求", "建议结构", "写作验收清单",
                "目标篇幅", "素材使用要求", "必须交付的图表"]

CN_NUM = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,
          "十一":11,"十二":12,"十三":13,"十四":14}

# 章号 -> 所属篇号（来自冻结结构；用于校验交叉引用 "第N章" 是否落在合理篇）
# 这里只用于报告维度，不强制
CH_TITLE_RE = re.compile(r"第\s*(\d+)\s*[章]")
PART_TITLE_RE = re.compile(r"第\s*([一二三四五六七八九十]+)\s*篇")
FIG_RE = re.compile(r"图\s*(\d+)\s*[-_]\s*(\d+)")
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
REF_HEADER_RE = re.compile(r"^#{1,4}\s*(参考文献|References)\s*$", re.M)
ABSTRACT_RE = re.compile(r"(##\s*摘\s*要|^\s*\*\*摘\s*要\*\*|Abstract)", re.M)
KEYWORDS_RE = re.compile(r"^#{1,4}\s*(关键词|Keywords)\s*$", re.M)
SUMMARY_RE = re.compile(r"^##\s*本章小结\s*$", re.M)
WINPATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
TABLE_CAPTION_RE = re.compile(r"^\*表\s+P?\d{1,2}-\d+：", re.M)
PROJECT_CASE_TERMS = [
    ("project-goal", r"项目目标"),
    ("scenario-boundary", r"场景约束|数据边界"),
    ("architecture", r"架构决策"),
    ("schema-flow", r"样本\s*schema|数据流"),
    ("implementation", r"核心实现"),
    ("metrics", r"验收指标|实验指标"),
    ("risk-compliance", r"成本|风险|合规边界"),
    ("failure-mode", r"常见失败模式"),
    ("reproducibility", r"可复现资源"),
]


@dataclass
class Issue:
    file: str
    level: str        # ERROR / WARN / INFO
    kind: str         # broken-image / fig-number / no-abstract / ...
    detail: str
    line: int = 0


@dataclass
class ChapterReport:
    file: str
    part: str
    declared_chapter: int | None
    kind: str         # chapter / project / index / appendix / other
    chars: int
    issues: list = field(default_factory=list)


def detect_part_no(path: Path) -> int | None:
    """从 docs/zh/partN/ 推出篇号数字。"""
    m = re.search(r"part(\d+)", str(path))
    return int(m.group(1)) if m else None


def detect_declared_chapter(text: str) -> int | None:
    """扫描前 40 行找 '第N章'（标题可能在 BOM/篇前导读/--- 之后）。"""
    head = "\n".join(text.splitlines()[:40])
    m = re.search(r"^\ufeff?\s*#\s*第\s*(\d+)\s*[章]", head, re.M)
    if m:
        return int(m.group(1))
    m = CH_TITLE_RE.search(head)
    return int(m.group(1)) if m else None


def classify(path: Path) -> str:
    name = path.name
    if name == "index.md":
        return "index"
    if name.startswith("ch"):
        return "chapter"
    if re.match(r"p\d+", name):
        return "project"
    if "appendix" in name:
        return "appendix"
    return "other"


def detect_project_no(path: Path) -> int | None:
    m = re.match(r"p(\d+)", path.name)
    return int(m.group(1)) if m else None


def lint_file(path: Path) -> ChapterReport:
    text = path.read_text(encoding="utf-8", errors="replace")
    rel = str(path.relative_to(ROOT))
    part_no = detect_part_no(path)
    part = f"第{part_no}篇" if part_no else "-"
    kind = classify(path)
    declared = detect_declared_chapter(text)
    chars = len(text)
    rep = ChapterReport(file=rel, part=part, declared_chapter=declared,
                        kind=kind, chars=chars)

    lines = text.splitlines()

    # --- 1. 断链图 / 本机绝对路径 ---
    for i, line in enumerate(lines, 1):
        for m in IMG_RE.finditer(line):
            src = m.group(1).strip()
            if src.startswith("http://") or src.startswith("https://"):
                continue
            if WINPATH_RE.match(src) or src.startswith("/Users/") or "\\" in src:
                rep.issues.append(Issue(rel, "ERROR", "broken-image",
                    f"本机绝对路径图片，纸书/构建必断链: {src}", i))
                continue
            resolved = (path.parent / src).resolve()
            if not resolved.exists():
                rep.issues.append(Issue(rel, "ERROR", "broken-image",
                    f"图片文件不存在: {src}", i))

    # --- 2. 图号-章号一致性（仅对有章号的正文章生效）---
    if kind == "chapter" and declared is not None:
        seen = set()
        for i, line in enumerate(lines, 1):
            for m in FIG_RE.finditer(line):
                fig_chap = int(m.group(1))
                tag = f"图{m.group(1)}-{m.group(2)}"
                if fig_chap != declared and tag not in seen:
                    seen.add(tag)
                    rep.issues.append(Issue(rel, "WARN", "fig-number",
                        f"图号 {tag} 的章前缀 {fig_chap} ≠ 本章 {declared}", i))

    # --- 3/4. Abstract / Keywords（正文章 + 项目章要求）---
    if kind in ("chapter", "project"):
        if not ABSTRACT_RE.search(text):
            rep.issues.append(Issue(rel, "WARN", "no-abstract",
                "缺少章首 摘要/Abstract（SpringerLink 索引必需）"))
        if not KEYWORDS_RE.search(text):
            rep.issues.append(Issue(rel, "WARN", "no-keywords",
                "缺少 ## 关键词 / ## Keywords 标题"))
        if not SUMMARY_RE.search(text):
            rep.issues.append(Issue(rel, "WARN", "no-summary",
                "缺少 ## 本章小结"))

    # --- 5. 参考文献 ---
    if kind in ("chapter", "project"):
        if not REF_HEADER_RE.search(text):
            rep.issues.append(Issue(rel, "WARN", "no-references",
                "缺少 ## 参考文献 / References 节"))

    # --- 6. 项目章案例研究结构 / 代码块长度 ---
    if kind == "project":
        project_no = detect_project_no(path)
        for issue_kind, pattern in PROJECT_CASE_TERMS:
            if not re.search(pattern, text, re.I):
                rep.issues.append(Issue(rel, "WARN", issue_kind,
                    f"项目章缺少案例研究模块: {issue_kind}"))
        if project_no is not None and 4 <= project_no <= 13:
            if not IMG_RE.search(text):
                rep.issues.append(Issue(rel, "WARN", "project-figure",
                    "P04-P13 项目章应至少包含 1 张流程/架构图"))
            if not TABLE_CAPTION_RE.search(text):
                rep.issues.append(Issue(rel, "WARN", "project-table",
                    "P04-P13 项目章应至少包含 1 张出版验收表"))
        in_code = False
        code_start = 0
        code_lines = 0
        for i, line in enumerate(lines, 1):
            if line.startswith("```"):
                if not in_code:
                    in_code = True
                    code_start = i
                    code_lines = 0
                else:
                    if code_lines > 25:
                        rep.issues.append(Issue(rel, "WARN", "long-code-block",
                            f"项目章代码块 {code_lines} 行 > 25 行，建议外置到配套资源", code_start))
                    in_code = False
            elif in_code:
                code_lines += 1

    # --- 7. 疑似未成稿 ---
    if kind in ("chapter", "project"):
        marker_hits = [k for k in STUB_MARKERS if k in text]
        if len(marker_hits) >= 2:
            rep.issues.append(Issue(rel, "ERROR", "stub-draft",
                f"含写作任务书标记 {marker_hits[:4]}，疑似大纲/未成稿"))
        min_chars = MIN_PROJECT_CHARS if kind == "project" else MIN_CHAPTER_CHARS
        if chars < min_chars:
            rep.issues.append(Issue(rel, "WARN" if marker_hits else "WARN",
                "too-short", f"正文 {chars} 字 < 配额下限 {min_chars} 字"))

    return rep


def collect(zh: Path) -> list[ChapterReport]:
    reports = []
    for path in sorted(zh.rglob("*.md")):
        # 跳过翻译状态、superpowers spec 等非成书内容
        if any(seg in path.parts for seg in ("superpowers",)):
            continue
        if path.name == "translation-status.md":
            continue
        reports.append(lint_file(path))
    return reports


# ---------- 输出 ----------
C = dict(red="\033[31m", yel="\033[33m", grn="\033[32m", dim="\033[2m",
         bold="\033[1m", end="\033[0m")


def render_console(reports: list[ChapterReport]) -> int:
    counts = {"ERROR": 0, "WARN": 0}
    by_kind: dict[str, int] = {}
    print(f"\n{C['bold']}=== Springer 出版就绪度校验报告 ==={C['end']}\n")
    for rep in reports:
        if not rep.issues:
            continue
        errs = [x for x in rep.issues if x.level == "ERROR"]
        warns = [x for x in rep.issues if x.level == "WARN"]
        head_color = C["red"] if errs else C["yel"]
        print(f"{head_color}{rep.file}{C['end']} "
              f"{C['dim']}[{rep.part} · {rep.kind} · {rep.chars}字]{C['end']}")
        for iss in rep.issues:
            counts[iss.level] = counts.get(iss.level, 0) + 1
            by_kind[iss.kind] = by_kind.get(iss.kind, 0) + 1
            col = C["red"] if iss.level == "ERROR" else C["yel"]
            loc = f":{iss.line}" if iss.line else ""
            print(f"  {col}{iss.level:5}{C['end']} {iss.kind:14} {iss.detail}{C['dim']}{loc}{C['end']}")
        print()

    print(f"{C['bold']}---- 汇总 ----{C['end']}")
    print(f"  {C['red']}ERROR{C['end']}: {counts.get('ERROR',0)}   "
          f"{C['yel']}WARN{C['end']}: {counts.get('WARN',0)}")
    print(f"  按类型: " + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))
    print(f"  扫描文件: {len(reports)}")
    return 1 if counts.get("ERROR", 0) else 0


def render_md(reports: list[ChapterReport]) -> str:
    out = ["# Springer 出版就绪度校验报告\n"]
    counts = {"ERROR": 0, "WARN": 0}
    by_kind: dict[str, int] = {}
    for rep in reports:
        for iss in rep.issues:
            counts[iss.level] = counts.get(iss.level, 0) + 1
            by_kind[iss.kind] = by_kind.get(iss.kind, 0) + 1
    out.append("## 汇总\n")
    out.append(f"- ERROR: **{counts.get('ERROR',0)}**  WARN: **{counts.get('WARN',0)}**")
    out.append(f"- 扫描文件: {len(reports)}")
    out.append("- 按问题类型: " + ", ".join(f"`{k}`={v}" for k, v in sorted(by_kind.items())) + "\n")
    out.append("## 明细\n")
    out.append("| 文件 | 篇 | 类型 | 字数 | 级别 | 问题 | 详情 | 行 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for rep in reports:
        for iss in rep.issues:
            d = iss.detail.replace("|", "\\|")
            out.append(f"| {rep.file} | {rep.part} | {rep.kind} | {rep.chars} | "
                       f"{iss.level} | {iss.kind} | {d} | {iss.line or ''} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH", help="输出 JSON")
    ap.add_argument("--md", metavar="PATH", help="输出 Markdown 报告")
    args = ap.parse_args()

    reports = collect(ZH)
    code = render_console(reports)

    if args.json:
        data = [asdict(r) for r in reports]
        Path(args.json).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON 已写入 {args.json}")
    if args.md:
        Path(args.md).write_text(render_md(reports), encoding="utf-8")
        print(f"Markdown 报告已写入 {args.md}")
    sys.exit(code)


if __name__ == "__main__":
    main()
