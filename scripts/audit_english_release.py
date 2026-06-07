#!/usr/bin/env python3
"""Audit the English documentation release against the Chinese mainline."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
ZH_ROOT = ROOT / "docs" / "zh"
EN_ROOT = ROOT / "docs" / "en"
PLACEHOLDER = "The English translation of this chapter is not yet available"
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
FENCED_BLOCK_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
LINK_TARGET_RE = re.compile(r"(?<!!)\[[^\]]+\]\([^)]+\)|!\[[^\]]*\]\([^)]+\)")


@dataclass
class Finding:
    kind: str
    path: Path
    detail: str

    def format(self) -> str:
        rel = self.path.relative_to(ROOT)
        return f"{self.kind}: {rel}: {self.detail}"


def markdown_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md"))


def relative_zh_paths() -> set[Path]:
    return {path.relative_to(ZH_ROOT) for path in markdown_files(ZH_ROOT)}


def relative_en_paths() -> set[Path]:
    return {path.relative_to(EN_ROOT) for path in markdown_files(EN_ROOT)}


def should_check_link(target: str) -> bool:
    target = target.strip()
    if not target or target.startswith("#"):
        return False
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
        return False
    return True


def resolve_link(source: Path, target: str) -> Path:
    clean = unquote(target.split("#", 1)[0].strip())
    return (source.parent / clean).resolve()


def audit_missing_files(findings: list[Finding]) -> None:
    zh_paths = relative_zh_paths()
    en_paths = relative_en_paths()
    legacy_only = {
        Path("translation-style-guide.md"),
        Path("translation-status.md"),
        Path("part10/10_1_mini_c4.md"),
        Path("part10/10_2_legal_sft.md"),
        Path("part10/10_3_llava_instruct.md"),
        Path("part10/10_4_synthetic_textbook.md"),
        Path("part10/10_5_mm_rag.md"),
        Path("part10/10_6_PRM.md"),
        Path("part10/10_7_Agent_Tooluse.md"),
        Path("part10/10_8_dataops.md"),
        Path("part10/10_9_privacy_pipeline.md"),
        Path("part10/10_10_flywheel.md"),
        Path("part11/ch29_pretrain_recipes.md"),
        Path("part11/ch30_posttrain_recipes.md"),
        Path("part11/ch31_rl_reasoning_data.md"),
        Path("part11/ch32_multimodel.md"),
        Path("part11/ch33_t21_t2v.md"),
        Path("part11/projects/p11_mini_deepseek.md"),
        Path("part11/projects/p12_r1_reasoning_flywheel.md"),
        Path("part11/projects/p13_multimodel_ins.md"),
        Path("part11/projects/p14_vedio_gen.md"),
        Path("part9/ch27_compliance_framework_and_governance.md"),
        Path("part9/ch28_federated_learning_and_privacy_preserving_technologies.md"),
    }

    for rel in sorted(zh_paths - en_paths):
        findings.append(Finding("missing-en", EN_ROOT / rel, "no English counterpart"))

    for rel in sorted(en_paths - zh_paths - legacy_only):
        findings.append(Finding("extra-en", EN_ROOT / rel, "not present in Chinese mainline"))


def audit_file(path: Path, findings: list[Finding]) -> None:
    text = path.read_text(encoding="utf-8-sig")
    text_without_code = FENCED_BLOCK_RE.sub("", text)

    if PLACEHOLDER in text:
        findings.append(Finding("placeholder", path, "translation placeholder remains"))

    if path.name != "translation-style-guide.md":
        text_without_link_targets = LINK_TARGET_RE.sub("", text_without_code)
        match = CHINESE_RE.search(text_without_link_targets)
        if match:
            line_no = text_without_link_targets[: match.start()].count("\n") + 1
            findings.append(Finding("chinese", path, f"Chinese character remains near line {line_no}"))

    for link in LINK_RE.finditer(text_without_code):
        target = link.group(1)
        if not should_check_link(target):
            continue
        resolved = resolve_link(path, target)
        if not resolved.exists():
            line_no = text[: link.start()].count("\n") + 1
            findings.append(Finding("broken-link", path, f"line {line_no}: {target}"))


def main() -> int:
    findings: list[Finding] = []
    audit_missing_files(findings)

    for path in markdown_files(EN_ROOT):
        audit_file(path, findings)

    if findings:
        for finding in findings:
            print(finding.format())
        print(f"\n{len(findings)} finding(s)")
        return 1

    print("English release audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
