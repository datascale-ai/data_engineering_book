from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADING_WITH_ZERO_NUMBER = re.compile(r"^#{2,6}\s+(?:[A-Z]|\d+(?:\.\d+)*)\.0\b")


class HeadingNumberingTest(unittest.TestCase):
    def test_markdown_headings_do_not_use_zero_numbered_sections(self):
        offenders: list[str] = []

        for path in sorted((ROOT / "docs").glob("**/*.md")):
            if not any(part in {"en", "zh"} for part in path.relative_to(ROOT / "docs").parts):
                continue
            for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
                if HEADING_WITH_ZERO_NUMBER.match(line):
                    offenders.append(f"{path.relative_to(ROOT).as_posix()}:{line_no}: {line}")

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
