#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xref_scan.py — 全书交叉引用扫描（检测指向旧篇号/旧章号的错误回链）

背景:
  正文经历 28章 -> 48章 的扩展，许多 "第N章 / 第N篇 / ChNN / 项目N / PNN" 回链
  可能仍指向旧编号或不存在的目标。本脚本以【文件名】为权威建立 章->篇 映射
  （ch08_*.md => 第8章 => 第3篇），逐文件扫描正文引用并校验：

  1. dangling-chapter : 引用 "第N章"/"ChNN" 但全书无第N章（N>48 或缺号）
  2. dangling-part    : 引用 "第N篇" 但 N>14
  3. dangling-project : 引用 "项目N"/"PNN" 但 N>15
  4. self-stale-hint  : 引用的章号所属篇与该章自身所属篇产生明显跨度（仅提示，供人工核对回链是否还正确）
  5. en-cn-mismatch   : 同一处 ChNN 与 第N章 混用风格（统计，供统稿）

用法:
  python scripts/xref_scan.py
  python scripts/xref_scan.py --md report.md
  python scripts/xref_scan.py --json out.json
退出码: 发现 dangling-* (ERROR) -> 1
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZH = ROOT / "docs" / "zh"

# ---- 权威结构（来自实际文件 + 冻结目录）----
# 章 -> 篇（按文件名 chNN 推断；这是 source of truth）
CH_TO_PART = {
    1:1, 2:1, 3:1,
    4:2, 5:2, 6:2, 7:2,
    8:3, 9:3, 10:3, 11:3,
    12:4, 13:4, 14:4,
    15:5, 16:5, 17:5,
    18:6, 19:6, 20:6,
    21:7, 22:7, 23:7,
    24:8, 25:8, 26:8,
    27:9, 28:9, 29:9, 30:9,
    31:10, 32:10, 33:10, 34:10, 35:10,
    36:11, 37:11,
    38:12, 39:12, 40:12, 41:12, 42:12, 43:12,
    44:13, 45:13, 46:13, 47:13, 48:13,
}
MAX_CHAPTER = 48
MAX_PART = 14
MAX_PROJECT = 15   # P01-P15（第14篇项目实战）

CN_NUM = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,
          "十一":11,"十二":12,"十三":13,"十四":14,
          "十五":15,"十六":16,"十七":17,"十八":18,"十九":19,"二十":20,
          "二十一":21,"二十二":22,"二十三":23,"二十四":24,"二十五":25,
          "二十六":26,"二十七":27,"二十八":28,"二十九":29,"三十":30}
NUM_CN = {v: k for k, v in CN_NUM.items()}

# 每篇主题关键词（取自实际 docs/zh/partN/index.md 标题），用于 topic-drift 检测：
# 当正文在引用『第N篇』时附带了一个主题词，但该主题词与第N篇实际主题不符，
# 说明这是旧编号体系（28章/10篇）遗留的错误回链。
PART_TOPIC_KEYWORDS = {
    1:  ["总论", "基础设施", "范式", "总纲"],
    2:  ["预训练", "文本"],
    3:  ["多模态"],
    4:  ["指令微调", "偏好", "SFT", "对齐"],
    5:  ["合成数据"],
    6:  ["推理", "Agent", "CoT", "工具", "Tool"],
    7:  ["应用级", "RAG", "检索"],
    8:  ["数据运营", "平台", "DataOps", "可观测"],
    9:  ["数据资产", "数据产品", "数据契约", "价值评估"],
    10: ["智能化", "Data Engineering Agent", "Agent"],
    11: ["隐私", "合规", "安全", "联邦"],
    12: ["专项数据集", "实践"],
    13: ["开源", "数据配方", "VLM", "多模态", "T2I", "T2V", "生成"],
    14: ["项目实战", "项目", "Mini-C4", "DataAgent", "RAG", "DataOps"],
}
# 这些主题词若出现，强烈指示某个 *特定* 旧篇号（用于反查 "说错了篇号"）
TOPIC_TO_CURRENT_PART = {
    "隐私": 11, "合规": 11, "联邦": 11,
    "项目实战": 14, "十五个项目": 14, "工业级项目实战": 14,
    "DataOps": 8, "可观测": 8,
    "RAG": 7,
    "合成数据": 5,
    "多模态": 3,
}

# 引用模式
RE_CH_CN   = re.compile(r"第\s*(\d+)\s*章")
RE_CH_EN   = re.compile(r"\bCh\.?\s*(\d+)\b")
RE_PART_CN = re.compile(r"第\s*([一二三四五六七八九十]+)\s*篇")
RE_PART_NUM= re.compile(r"第\s*(\d+)\s*篇")
RE_PROJ_CN = re.compile(r"项目\s*([一二三四五六七八九十]+)")
RE_PROJ_EN = re.compile(r"\bP(\d{1,2})\b")


@dataclass
class XIssue:
    file: str
    level: str
    kind: str
    detail: str
    line: int
    snippet: str = ""


def self_chapter(path: Path) -> int | None:
    m = re.search(r"ch(\d+)", path.name)
    return int(m.group(1)) if m else None


def check_topic_drift(part_n: int, span_text: str, is_label: bool,
                      rel: str, line_no: int, snip: str) -> XIssue | None:
    """topic-drift 检测。仅当 span 是“该篇标签”(is_label) 时才据主题词判错，
    避免把『第N篇的<下游内容>』里的主题词误配到篇号上。"""
    if not is_label or part_n not in PART_TOPIC_KEYWORDS:
        return None
    if any(k in span_text for k in PART_TOPIC_KEYWORDS[part_n]):
        return None  # 标签已含本篇正确主题词
    for topic, real_part in TOPIC_TO_CURRENT_PART.items():
        if topic in span_text and real_part != part_n:
            return XIssue(rel, "ERROR", "stale-part-topic",
                f"引用『第{part_n}篇』并标注主题『{topic}』，但『{topic}』现属第{real_part}篇"
                f"（疑似沿用旧10篇编号，回链需改篇号或措辞）", line_no, snip)
    return None


def nearest_span(line: str, m: re.Match) -> tuple[str, bool]:
    """返回 (span_text, is_label)。

    只有当『第N篇』后紧跟的文字看起来像“该篇的标题/标签”时，topic-drift 才可信：
      - 括号紧随：第N篇（隐私、合规…）           -> is_label=True
      - 表格单元：| 第N篇 隐私与合规 |             -> is_label=True
    其它情况（第十三篇的多模态 RAG…，是下游内容描述而非篇标签）-> is_label=False，
    返回但不据此报 ERROR，避免跨句/下游名词短语误报。
    """
    after = line[m.end(): m.end() + 24]
    pm = re.match(r"\s*[（(]([^）)]{0,30})[）)]", after)
    if pm:
        return pm.group(1), True
    # 表格单元：本 match 落在 | … | 之间，且其后直到下一个 | 为短标签
    before = line[:m.start()]
    if before.rstrip().endswith("|") or "|" in after:
        cell = re.split(r"\||→", after)[0]
        return cell, True
    # 形如『第N篇隐私与合规』『第N篇 隐私……』紧贴（无“的/中”等领属字）也算标签
    tight = re.match(r"\s*([^的中里下游，。；、：,.;:\s]{0,12})", after)
    label = tight.group(1) if tight else ""
    # 若紧跟“的/中”，多半是“第N篇的<下游内容>”，非标签
    is_label = bool(label) and not after.lstrip().startswith(("的", "中"))
    return label, is_label




def scan_file(path: Path) -> list[XIssue]:
    rel = str(path.relative_to(ROOT))
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    self_ch = self_chapter(path)
    self_part = CH_TO_PART.get(self_ch) if self_ch else None
    issues: list[XIssue] = []
    in_code = False

    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        snip = line.strip()[:90]

        # 第N章
        for m in RE_CH_CN.finditer(line):
            n = int(m.group(1))
            if n < 1 or n > MAX_CHAPTER:
                issues.append(XIssue(rel, "ERROR", "dangling-chapter",
                    f"引用『第{n}章』但全书仅 1-{MAX_CHAPTER} 章", i, snip))
            elif self_part is not None:
                tgt_part = CH_TO_PART.get(n)
                # 跨 >=6 篇的回链，旧版编号迁移后最易指错 -> 提示人工核对
                if tgt_part and abs(tgt_part - self_part) >= 6:
                    issues.append(XIssue(rel, "INFO", "long-range-ref",
                        f"本章(第{self_part}篇)引用『第{n}章』(第{tgt_part}篇)，跨度大，请核对回链是否仍准确", i, snip))

        # ChNN
        for m in RE_CH_EN.finditer(line):
            n = int(m.group(1))
            if n < 1 or n > MAX_CHAPTER:
                issues.append(XIssue(rel, "ERROR", "dangling-chapter",
                    f"引用『Ch{n}』但全书仅 Ch1-Ch{MAX_CHAPTER}", i, snip))

        # 第N篇（数字写法，正规应为中文）
        for m in RE_PART_NUM.finditer(line):
            n = int(m.group(1))
            if n > MAX_PART:
                issues.append(XIssue(rel, "ERROR", "dangling-part",
                    f"引用『第{n}篇』但全书仅 {MAX_PART} 篇", i, snip))
            else:
                issues.append(XIssue(rel, "WARN", "part-style",
                    f"『第{n}篇』用了阿拉伯数字，体例应为中文『第{NUM_CN.get(n, n)}篇』", i, snip))
                sp, lab = nearest_span(line, m)
                drift = check_topic_drift(n, sp, lab, rel, i, snip)
                if drift:
                    issues.append(drift)

        # 第N篇（中文）
        for m in RE_PART_CN.finditer(line):
            n = CN_NUM.get(m.group(1))
            if n and n > MAX_PART:
                issues.append(XIssue(rel, "ERROR", "dangling-part",
                    f"引用『第{m.group(1)}篇』(={n}) 但全书仅 {MAX_PART} 篇", i, snip))
            elif n:
                sp, lab = nearest_span(line, m)
                drift = check_topic_drift(n, sp, lab, rel, i, snip)
                if drift:
                    issues.append(drift)

        # 项目N（中文）
        for m in RE_PROJ_CN.finditer(line):
            n = CN_NUM.get(m.group(1))
            if n and n > MAX_PROJECT:
                issues.append(XIssue(rel, "ERROR", "dangling-project",
                    f"引用『项目{m.group(1)}』(={n}) 但全书仅 P01-P{MAX_PROJECT}", i, snip))

        # PNN（仅当像引用，避免误伤 P95 等指标）—限制 P01-P20 区间外报 dangling
        for m in RE_PROJ_EN.finditer(line):
            n = int(m.group(1))
            # 只在明显的项目语境下判定（前后含 项目/复现/实战/章），避免 P50/P99 分位误判
            ctx = line[max(0, m.start()-6):m.end()+6]
            if re.search(r"项目|复现|实战|P0[0-9]|P1[0-5]", ctx):
                if n > MAX_PROJECT or n == 0:
                    issues.append(XIssue(rel, "ERROR", "dangling-project",
                        f"引用『P{n:02d}』但全书仅 P01-P{MAX_PROJECT}", i, snip))

    return issues


def collect() -> list[XIssue]:
    out = []
    for path in sorted(ZH.rglob("*.md")):
        if any(seg in path.parts for seg in ("superpowers",)):
            continue
        if path.name == "translation-status.md":
            continue
        out.extend(scan_file(path))
    return out


C = dict(red="\033[31m", yel="\033[33m", blu="\033[34m", dim="\033[2m", bold="\033[1m", end="\033[0m")
LVLCOL = {"ERROR": C["red"], "WARN": C["yel"], "INFO": C["blu"]}


def render_console(issues: list[XIssue]) -> int:
    by_file = defaultdict(list)
    for x in issues:
        by_file[x.file].append(x)
    counts = defaultdict(int); kinds = defaultdict(int)
    print(f"\n{C['bold']}=== 全书交叉引用扫描 ==={C['end']}\n")
    for f in sorted(by_file):
        items = by_file[f]
        # 文件级只在有 ERROR/WARN 时打印（INFO 折叠进汇总）
        shown = [x for x in items if x.level in ("ERROR", "WARN")]
        if not shown:
            for x in items:
                counts[x.level]+=1; kinds[x.kind]+=1
            continue
        print(f"{C['bold']}{f}{C['end']}")
        for x in items:
            counts[x.level]+=1; kinds[x.kind]+=1
            if x.level == "INFO":
                continue
            col = LVLCOL[x.level]
            print(f"  {col}{x.level:5}{C['end']} {x.kind:17} {x.detail}  {C['dim']}L{x.line}: {x.snippet}{C['end']}")
        print()
    print(f"{C['bold']}---- 汇总 ----{C['end']}")
    print(f"  {C['red']}ERROR{C['end']}={counts['ERROR']}  {C['yel']}WARN{C['end']}={counts['WARN']}  {C['blu']}INFO{C['end']}={counts['INFO']}")
    print("  按类型: " + ", ".join(f"{k}={v}" for k,v in sorted(kinds.items())))
    return 1 if counts["ERROR"] else 0


def render_md(issues: list[XIssue]) -> str:
    out = ["# 全书交叉引用扫描报告\n", "## 明细\n",
           "| 文件 | 级别 | 类型 | 详情 | 行 | 原文 |",
           "|---|---|---|---|---|---|"]
    for x in sorted(issues, key=lambda y:(y.file, y.line)):
        d=x.detail.replace("|","\\|"); s=x.snippet.replace("|","\\|")
        out.append(f"| {x.file} | {x.level} | {x.kind} | {d} | {x.line} | {s} |")
    return "\n".join(out)+"\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json"); ap.add_argument("--md")
    a = ap.parse_args()
    issues = collect()
    code = render_console(issues)
    if a.json:
        Path(a.json).write_text(json.dumps([asdict(x) for x in issues], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON -> {a.json}")
    if a.md:
        Path(a.md).write_text(render_md(issues), encoding="utf-8")
        print(f"Markdown -> {a.md}")
    sys.exit(code)


if __name__ == "__main__":
    main()
