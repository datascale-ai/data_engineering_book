from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reference_integrity_audit.py"


def load_audit():
    spec = importlib.util.spec_from_file_location("reference_integrity_audit", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReferenceIntegrityAuditTest(unittest.TestCase):
    def test_extract_body_citations_avoids_overlapping_author_year_matches(self):
        audit = load_audit()

        with tempfile.TemporaryDirectory(dir=ROOT / "output") as tmp:
            path = Path(tmp) / "chapter.md"
            path.write_text(
                "\n".join(
                    [
                        "# Chapter",
                        "综述见 Nait Saada et al. 2025。",
                        "结构化推理见 Yao, Zhao et al. 2023; Kim, Shin et al. 2024。",
                        "## 参考文献",
                    ]
                ),
                encoding="utf-8",
            )

            keys = [citation.key for citation in audit.extract_body_citations(path)]

        self.assertIn("naitsaada:2025", keys)
        self.assertIn("yao:2023", keys)
        self.assertIn("kim:2024", keys)
        self.assertNotIn("saada:2025", keys)
        self.assertNotIn("zhao:2023", keys)
        self.assertNotIn("shin:2024", keys)


if __name__ == "__main__":
    unittest.main()
