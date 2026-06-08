#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit quantitative claims for Springer fact-checking.

The report is intentionally conservative: it highlights numeric statements
that need source or reproducibility evidence before publication. It does not
prove authenticity by itself.
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
DEFAULT_REPORT_DIR = ROOT / "publishing" / "final_review"

REF_HEADER_RE = re.compile(r"^##\s*参考文献\s*$")
NEXT_H2_RE = re.compile(r"^##\s+")
HEADING_RE = re.compile(r"^#{1,6}\s+")
TABLE_DIVIDER_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")

NUMBER_RE = re.compile(
    r"(?<![\w./-])"
    r"(?:[~≈约近超超过大于小于不足至少至多上下约为]*\s*)?"
    r"(?:[<>≤≥=+\-±]\s*)?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?:\s*(?:%|％|‰|K|M|B|T|GB|TB|PB|MB|KB|GiB|TiB|ms|s|Tokens?|tokens?|token|Token|"
    r"参数|词表|样本|文档|图片|图像|表格|问题|问答|条|张|篇|个|轮|步|卡|小时|分钟|秒|毫秒|"
    r"天|周|月|年|元|美元|A100|H100|4090|4090/A100|倍|万|亿))?"
)

ARXIV_RE = re.compile(r"arXiv:\d{4}\.\d{4,5}", re.I)
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
URL_RE = re.compile(r"https?://[^\s)）]+")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
AUTHOR_YEAR_RE = re.compile(r"\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?\s*(?:19|20)\d{2}\)")
CN_SOURCE_RE = re.compile(r"来源[:：]|根据|报告|论文|数据集说明|公开入口|Available at|doi\.org", re.I)
CITATION_PARENS_RE = re.compile(r"\([^)]*(?:19|20)\d{2}[^)]*\)")

NUMERIC_CONTEXT_RE = re.compile(
    r"%|％|‰|Token|token|tokens|参数|词表|样本|数据集|训练数据|语料|规模|占比|比例|"
    r"提升|降低|下降|增长|超过|少于|吞吐|显存|成本|耗时|SLA|A100|H100|4090|"
    r"GB|TB|PB|MB|B\b|M\b|K\b|万|亿|张|条|篇|问答|表格|图片|图像|准确率|召回|"
    r"precision|recall|accuracy|loss|Loss|BLEU|ROUGE|TEDS|MMLU|GSM8K",
    re.I,
)

EXTERNAL_CONTEXT_RE = re.compile(
    r"论文|报告|公开|官方|数据集|benchmark|基准|模型|参数|Tokens?|token|训练数据|"
    r"GPT|LLaMA|Llama|DeepSeek|Chinchilla|Gopher|Phi|FineWeb|Dolma|The Pile|"
    r"HumanEval|MMLU|GSM8K|ChartQA|PlotQA|DocVQA|PubTables|SparseTable|Latent-Switch|"
    r"STB|VQA|COCO|Spider|法律|法规",
    re.I,
)

EXAMPLE_CONTEXT_RE = re.compile(r"假设|匿名化|复合案例|示例|教学化|例如|可设|建议|目标|SLA|验收|阈值|通常|常用经验|约")
LOCAL_PROJECT_RE = re.compile(r"本项目|本实现|我们|项目\s*\d+|Mini-|复现|烟测|产出|运行|抽样")
LOCAL_RESULT_RE = re.compile(
    r"当前运行结果|一次真实运行|最终在|最终训练集|最终.*产出|产出约|抽样规模|"
    r"在这次|固定随机种子|运行记录|耗时约|滤除|召回了|形成\s*`?\d|"
    r"成本不到|训练日志|验证了本套流水线"
)
STRONG_LOCAL_RESULT_RE = re.compile(
    r"当前运行结果|一次真实运行记录|本项目当前产物|项目在固定随机种子|我们最终|"
    r"耗时约|成本不到|验证了本套流水线"
)
PERFORMANCE_RE = re.compile(r"%|％|提升|降低|下降|增长|超过|更优|准确率|召回|loss|Loss|分数|SOTA", re.I)
RESOURCE_RE = re.compile(r"GB|TB|PB|MB|A100|H100|4090|GPU|显存|内存|算力|成本|耗时|小时|分钟|秒|卡|美元", re.I)


@dataclass
class QuantitativeClaim:
    file: str
    line: int
    section: str
    numbers: list[str]
    category: str
    source_status: str
    risk: str
    action: str
    context: str


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def book_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(ZH_ROOT.rglob("*.md")):
        if "superpowers" in path.parts or path.name in {"translation-status.md", "index.md"}:
            continue
        if path.name.startswith("ch") or re.match(r"p\d+", path.name) or "appendix" in path.name:
            files.append(path)
    return files


def split_review_lines(path: Path) -> list[tuple[int, str, str, list[str]]]:
    rows: list[tuple[int, str, str, list[str]]] = []
    section = ""
    in_code = False
    in_refs = False
    lines = read_text(path).splitlines()
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if REF_HEADER_RE.match(stripped):
            in_refs = True
            continue
        if in_refs:
            if NEXT_H2_RE.match(stripped):
                in_refs = False
            else:
                continue
        if in_code or not stripped or TABLE_DIVIDER_RE.match(stripped):
            continue
        if HEADING_RE.match(stripped):
            section = re.sub(r"^#{1,6}\s+", "", stripped)
            # Keep headings only when they carry a domain number, not chapter
            # numbering such as "1.2.3".
            if not NUMERIC_CONTEXT_RE.search(stripped):
                continue
        window = lines[max(0, lineno - 3) : min(len(lines), lineno + 2)]
        rows.append((lineno, line, section, window))
    return rows


def normalize_context(line: str) -> str:
    context = re.sub(r"\s+", " ", line.strip())
    context = context.replace("|", "\\|")
    return context[:260] + ("..." if len(context) > 260 else "")


def strip_structural_numbers(line: str) -> str:
    text = CITATION_PARENS_RE.sub("", line)
    text = re.sub(r"^\s*\d+[.、]\s*", "", text)
    text = re.sub(r"\b(?:Ch|P)\d+\b", "", text)
    text = re.sub(r"第\s*\d+\s*(?:章|篇|节|至第\s*\d+\s*篇)", "", text)
    text = re.sub(r"[图表]\s*\d+(?:[-‑–—]\d+)?", "", text)
    text = re.sub(r"§\s*\d+(?:\.\d+)*", "", text)
    text = re.sub(r"\b\d+(?:\.\d+)*\s*节", "", text)
    return text


def is_relevant_numeric_line(line: str, numbers: list[str]) -> bool:
    stripped = line.strip()
    if not numbers:
        return False
    if HEADING_RE.match(stripped) and not NUMERIC_CONTEXT_RE.search(stripped):
        return False
    if re.fullmatch(r"[\s|:.\-#\d]+", stripped):
        return False
    joined = " ".join(numbers)
    if re.search(r"[,％%]|[A-Za-z]*B\b|[A-Za-z]*T\b|GB|TB|PB|MB|Token|token|万|亿", joined):
        return True
    return bool(NUMERIC_CONTEXT_RE.search(line))


def source_status(line: str, window: Iterable[str], file_has_refs: bool) -> str:
    text = "\n".join(window)
    line_has_source = any(regex.search(line) for regex in (ARXIV_RE, DOI_RE, URL_RE, AUTHOR_YEAR_RE, CN_SOURCE_RE))
    if line_has_source:
        return "inline-source-signal"
    nearby_has_source = any(regex.search(text) for regex in (ARXIV_RE, DOI_RE, URL_RE, AUTHOR_YEAR_RE, CN_SOURCE_RE))
    if nearby_has_source:
        return "nearby-source-signal"
    if file_has_refs:
        return "chapter-reference-section-only"
    return "no-source-signal"


def categorize(line: str) -> str:
    if PERFORMANCE_RE.search(line):
        return "performance-or-percentage"
    if RESOURCE_RE.search(line):
        return "cost-resource-or-scale"
    if re.search(r"样本|数据集|训练数据|语料|问答|表格|图片|图像|张|条|篇", line):
        return "dataset-size-or-composition"
    if re.search(r"参数|词表|Token|token|tokens|模型", line, re.I):
        return "model-or-token-scale"
    if re.search(r"SLA|验收|阈值|目标|建议", line):
        return "internal-target-or-threshold"
    return "other-quantitative"


def assess_risk(line: str, section: str, category: str, status: str) -> tuple[str, str]:
    scope_text = f"{section}\n{line}"
    external = bool(EXTERNAL_CONTEXT_RE.search(line))
    example = bool(EXAMPLE_CONTEXT_RE.search(scope_text))
    local_project = bool(LOCAL_PROJECT_RE.search(line))
    local_result = bool(LOCAL_RESULT_RE.search(line))
    sourced = status in {"inline-source-signal", "nearby-source-signal"}

    if sourced and not local_result:
        return "low", "保留；出版终稿阶段核对引用条目格式和 DOI/URL。"
    if example:
        return "medium", "已按示例/匿名化案例口径处理；出版前确认不会被误读为真实统计。"
    if local_result and (local_project or STRONG_LOCAL_RESULT_RE.search(line)):
        return "high", "补充可复现实验日志、数据产物路径或改写为示例值；没有证据时不要作为真实结果发布。"
    if external and status == "chapter-reference-section-only":
        return "medium", "核对本节数值是否能对应到参考文献；必要时补内联来源或脚注。"
    if local_project and category in {"dataset-size-or-composition", "cost-resource-or-scale", "performance-or-percentage"}:
        return "medium", "判断是否为项目实测值；若是，补充日志/产物证据，否则改写为示例或建议口径。"
    if external and not sourced:
        return "high", "补充权威来源并逐项核验；无法核验时删除或改为非定量表述。"
    if example:
        return "medium", "明确标注为假设/教学示例，避免被出版社当作真实统计。"
    if category == "internal-target-or-threshold":
        return "medium", "标注为建议阈值或项目 SLA，避免暗示行业通用事实。"
    if status == "no-source-signal":
        return "medium", "判断是否为外部事实；若是外部事实需补来源。"
    return "low", "人工抽检即可。"


def scan_claims(files: Iterable[Path]) -> list[QuantitativeClaim]:
    claims: list[QuantitativeClaim] = []
    for path in files:
        text = read_text(path)
        file_has_refs = any(REF_HEADER_RE.match(line.strip()) for line in text.splitlines())
        for lineno, line, section, window in split_review_lines(path):
            claim_line = strip_structural_numbers(line)
            numbers = [match.group(0).strip() for match in NUMBER_RE.finditer(claim_line)]
            numbers = [
                number
                for number in numbers
                if number and (not re.fullmatch(r"\d+", number.strip()) or NUMERIC_CONTEXT_RE.search(claim_line))
            ]
            if not is_relevant_numeric_line(claim_line, numbers):
                continue
            status = source_status(line, window, file_has_refs)
            category = categorize(claim_line)
            risk, action = assess_risk(line, section, category, status)
            claims.append(
                QuantitativeClaim(
                    file=rel(path),
                    line=lineno,
                    section=section,
                    numbers=numbers[:12],
                    category=category,
                    source_status=status,
                    risk=risk,
                    action=action,
                    context=normalize_context(line),
                )
            )
    return claims


def count_by(rows: Iterable[QuantitativeClaim], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = getattr(row, attr)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, claims: list[QuantitativeClaim], generated_at: str, max_rows: int) -> None:
    high = [row for row in claims if row.risk == "high"]
    medium = [row for row in claims if row.risk == "medium"]
    source_weak = [row for row in claims if row.source_status in {"chapter-reference-section-only", "no-source-signal"}]
    out = [
        "# 全书量化数据真实性 Review 报告",
        "",
        f"生成时间：{generated_at}",
        "",
        "## 一、结论摘要",
        "",
        f"- 量化表述候选：{len(claims)} 条。",
        f"- 高风险：{len(high)} 条，主要是本项目/本实现结果、数据集规模、性能百分比或资源成本但缺少可复现实验日志或内联来源。",
        f"- 中风险：{len(medium)} 条，主要是章节参考文献可支撑但未内联到具体数值，或属于教学示例/建议阈值。",
        f"- 弱来源信号：{len(source_weak)} 条，出版前需要人工确认是否为外部事实。",
        "",
        "## 二、处理口径",
        "",
        "- 外部论文、官方报告、公开数据集、法规、标准中的数值：应能回溯到 DOI、arXiv、官方文档或数据集主页，关键数值建议补内联来源。",
        "- 书中自建项目、复现实验、成本、去重率、吞吐、显存和训练时长：必须有可复现实验日志、脚本输出或数据产物路径；否则改成教学示例值。",
        "- SLA、阈值和验收指标：如果是作者建议，需明确写成建议口径，不能写成行业统计事实。",
        "- 匿名化复合案例和教学化示例：可以保留数值，但应明确其示例属性，避免被误读为真实项目统计。",
        "",
        "## 三、分布统计",
        "",
        "### 按风险级别",
        "",
    ]
    for key, value in count_by(claims, "risk").items():
        out.append(f"- {key}: {value}")
    out.extend(["", "### 按类型", ""])
    for key, value in count_by(claims, "category").items():
        out.append(f"- {key}: {value}")
    out.extend(["", "### 按来源信号", ""])
    for key, value in count_by(claims, "source_status").items():
        out.append(f"- {key}: {value}")

    out.extend(
        [
            "",
            "## 四、高风险与中风险明细",
            "",
            "| 风险 | 文件 | 行 | 类型 | 来源信号 | 数值 | 处理建议 | 原文 |",
            "| --- | --- | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    selected = [row for row in claims if row.risk in {"high", "medium"}]
    risk_order = {"high": 0, "medium": 1, "low": 2}
    selected.sort(key=lambda row: (risk_order[row.risk], row.file, row.line))
    for row in selected[:max_rows]:
        nums = ", ".join(row.numbers).replace("|", "\\|")
        out.append(
            f"| {row.risk} | `{row.file}` | {row.line} | {row.category} | {row.source_status} | "
            f"{nums} | {row.action} | {row.context} |"
        )
    if len(selected) > max_rows:
        out.append(f"\n> 其余 {len(selected) - max_rows} 条见 `quantitative_data_audit.json`。")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit quantitative claims for publication review.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for generated reports.")
    parser.add_argument("--max-rows", type=int, default=400, help="Maximum high/medium rows shown in Markdown.")
    parser.add_argument("--fail-on-high", action="store_true", help="Exit 1 when high-risk quantitative claims exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    files = book_files()
    claims = scan_claims(files)
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at_utc": generated_at,
        "scanned_files": len(files),
        "claims": [asdict(row) for row in claims],
        "summary": {
            "claims": len(claims),
            "by_risk": count_by(claims, "risk"),
            "by_category": count_by(claims, "category"),
            "by_source_status": count_by(claims, "source_status"),
        },
    }
    write_json(args.report_dir / "quantitative_data_audit.json", payload)
    write_markdown(args.report_dir / "quantitative_data_audit.md", claims, generated_at, args.max_rows)

    print("Quantitative data audit generated")
    print(f"- scanned_files: {len(files)}")
    for key, value in payload["summary"].items():
        print(f"- {key}: {value}")
    high_count = payload["summary"]["by_risk"].get("high", 0)
    if args.fail_on_high and high_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
