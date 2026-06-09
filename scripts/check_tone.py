#!/usr/bin/env python3
"""Scan Chinese documentation for banned phrases that violate the book's voice guideline."""
from __future__ import annotations

import argparse
from pathlib import Path


BANNED_PHRASES: list[str] = [
    "核聚变",
    "屁股后面",
    "淬炼狱",
    "炼丹",
    "血一样",
    "毛骨悚然",
    "庞然大物",
    "刺客",
    "碾覆",
    "屁话",
    "杀得昏天黑地",
    "如同核",
    "灭顶之灾",
    "死寂",
    "大放异彩",
    "至暗",
    "秒懂",
    "玄学",
    "混合面糊",
    "面糊",
    "数据食谱",
    "一锅烩",
    "踩坑",
    "深水炸弹",
    "翻车",
    "拉垮",
    "拉挂",
    "裸用",
    "词穷",
    "冰冷的账本题",
    "魔咒",
    "毁灭性",
    "无所遁形",
    "骨子里",
    "联姻",
    "极其壮烈",
    "翻天覆地",
    "甘于",
    "开箱即用",
    "小股试错",
    "对撞实验",
    "宣告胜利",
    "无感热切",
    "滚雪球式",
]

DEFAULT_ROOT = Path("docs/zh")
CONTEXT_TRUNCATE = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan docs/zh markdown files for banned phrases that violate the book's voice guideline.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory to scan recursively for .md files.",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Report hits but always exit 0 (use during transition periods).",
    )
    parser.add_argument(
        "--list-phrases",
        action="store_true",
        help="Print the banned phrase list and exit 0.",
    )
    return parser.parse_args()


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return a list of (line_number, phrase, context) tuples for each hit."""
    hits: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return hits
    for lineno, line in enumerate(text.splitlines(), start=1):
        for phrase in BANNED_PHRASES:
            if phrase in line:
                context = line.strip()
                if len(context) > CONTEXT_TRUNCATE:
                    context = context[:CONTEXT_TRUNCATE] + "..."
                hits.append((lineno, phrase, context))
    return hits


def main() -> int:
    args = parse_args()

    if args.list_phrases:
        print("Banned phrases:")
        for phrase in BANNED_PHRASES:
            print(f"- {phrase}")
        return 0

    total_hits = 0
    files_with_hits: set[Path] = set()
    for path in sorted(args.root.rglob("*.md")):
        if not path.is_file():
            continue
        hits = scan_file(path)
        if not hits:
            continue
        files_with_hits.add(path)
        for lineno, phrase, context in hits:
            print(f"{path}:{lineno}: {phrase}: {context}")
            total_hits += 1

    if total_hits == 0:
        print("No tone violations found")
        return 0

    print(f"Found {total_hits} tone-violation hits across {len(files_with_hits)} files")
    return 0 if args.warn_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
