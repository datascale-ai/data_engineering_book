from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def flatten_nav(items):
    for item in items:
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str):
                    yield value
                elif isinstance(value, list):
                    yield from flatten_nav(value)


class ChinesePublicationStructureTest(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
        self.zh_nav = next(
            lang["nav"]
            for lang in self.config["plugins"][2]["i18n"]["languages"]
            if lang["locale"] == "zh"
        )

    def test_all_chinese_chapter_files_are_in_mkdocs_nav(self):
        chapter_paths = sorted(
            path.relative_to(ROOT / "docs/zh").as_posix()
            for path in (ROOT / "docs/zh").glob("part*/ch*.md")
        )
        nav_paths = set(flatten_nav(self.zh_nav))

        missing = [path for path in chapter_paths if path not in nav_paths]

        self.assertEqual(missing, [])

    def test_all_chinese_chapter_files_are_in_book_index(self):
        index_text = (ROOT / "docs/zh/index.md").read_text(encoding="utf-8")
        chapter_paths = sorted(
            path.relative_to(ROOT / "docs/zh").as_posix()
            for path in (ROOT / "docs/zh").glob("part*/ch*.md")
        )

        missing = [path for path in chapter_paths if f"]({path})" not in index_text]

        self.assertEqual(missing, [])

    def test_chinese_chapter_numbers_are_unique(self):
        seen: dict[int, str] = {}
        duplicates: list[tuple[int, str, str]] = []

        for path in sorted((ROOT / "docs/zh").glob("part*/ch*.md")):
            lines = path.read_text(encoding="utf-8-sig").splitlines()
            first_line = next((line for line in lines if line.startswith("# ")), "")
            match = re.search(r"第(\d+)章", first_line)
            self.assertIsNotNone(match, f"{path} missing chapter number in title")
            chapter_no = int(match.group(1))
            rel = path.relative_to(ROOT).as_posix()
            if chapter_no in seen:
                duplicates.append((chapter_no, seen[chapter_no], rel))
            else:
                seen[chapter_no] = rel

        self.assertEqual(duplicates, [])


if __name__ == "__main__":
    unittest.main()
