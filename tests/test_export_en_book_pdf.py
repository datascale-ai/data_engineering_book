from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_en_book_pdf.py"
LATEX_SCRIPT = ROOT / "scripts" / "export_en_book_latex.py"


def load_exporter():
    spec = importlib.util.spec_from_file_location("export_en_book_pdf", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_latex_exporter():
    spec = importlib.util.spec_from_file_location("export_en_book_latex", LATEX_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExportEnglishBookPdfTest(unittest.TestCase):
    def embedded_font_gaps(self, pdf_path: Path) -> list[tuple[int, str, str]]:
        from pypdf import PdfReader

        def deref(obj):
            return obj.get_object() if hasattr(obj, "get_object") else obj

        gaps: list[tuple[int, str, str]] = []
        reader = PdfReader(str(pdf_path))
        for page_no, page in enumerate(reader.pages, 1):
            resources = deref(page.get("/Resources", {}))
            fonts = deref(resources.get("/Font", {})) if resources else {}
            for font_ref in fonts.values():
                font = deref(font_ref)
                base = str(font.get("/BaseFont", ""))
                subtype = str(font.get("/Subtype", ""))
                embedded = False
                descriptor = deref(font.get("/FontDescriptor")) if font.get("/FontDescriptor") else None
                if descriptor:
                    embedded = any(key in descriptor for key in ["/FontFile", "/FontFile2", "/FontFile3"])
                descendants = font.get("/DescendantFonts")
                if descendants:
                    for child_ref in deref(descendants):
                        child = deref(child_ref)
                        child_descriptor = (
                            deref(child.get("/FontDescriptor")) if child.get("/FontDescriptor") else None
                        )
                        if child_descriptor and any(
                            key in child_descriptor for key in ["/FontFile", "/FontFile2", "/FontFile3"]
                        ):
                            embedded = True
                if not embedded and subtype != "/Type3":
                    gaps.append((page_no, base, subtype))
        return gaps

    def test_english_exporter_uses_english_nav_and_16k_outputs(self):
        exporter = load_exporter()
        config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))

        items = exporter.flatten_nav(exporter.find_en_nav(config))
        paths = [item.path for item in items]

        self.assertEqual(exporter.OUT_HTML.name, "data_engineering_book_en_16k_compact.html")
        self.assertEqual(exporter.OUT_PDF.name, "data_engineering_book_en_16k_compact.pdf")
        self.assertIn("preface.md", paths)
        self.assertIn("contributors.md", paths)
        self.assertNotIn("about_authors.md", paths)
        self.assertIn("part14/p15_dataagent_semantic_nl2sql_agent.md", paths)
        self.assertIn("appendix_f_terminology_and_chinese_english_mapping.md", paths)

    def test_english_exporter_writes_english_html_shell(self):
        exporter = load_exporter()
        config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
        items = exporter.flatten_nav(exporter.find_en_nav(config))[:1]

        html_doc, stats = exporter.build_book_html(items, include_cover_toc=False)

        self.assertIn('<html lang="en">', html_doc)
        self.assertIn("<title>Data Engineering for Large Foundation Models: A Handbook - 16K PDF</title>", html_doc)
        self.assertEqual(stats["files"], 1)

    def test_section_opening_uses_title_and_author_without_number_block(self):
        exporter = load_exporter()
        html_body = (
            "<h1>Chapter 1: The Data Revolution in the Era of Large Models</h1>\n"
            '<div class="chapter-authors">Ke Wang</div>'
        )

        transformed = exporter.transform_section_opening(html_body, "part1/ch01_data_change.md")

        self.assertIn("<h1>The Data Revolution in the Era of Large Models</h1>", transformed)
        self.assertIn('<div class="chapter-authors">Ke Wang</div>', transformed)
        self.assertNotIn("chapter-number", transformed)

    def test_submission_pdfs_are_limited_to_actual_manuscript_units(self):
        exporter = load_exporter()
        config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
        items = exporter.flatten_nav(exporter.find_en_nav(config))

        paths = [item.path for item in exporter.submission_pdf_items(items)]

        self.assertIn("part10/ch31_agent_architecture.md", paths)
        self.assertIn("part14/p15_dataagent_semantic_nl2sql_agent.md", paths)
        self.assertIn("appendix_a_tools_and_frameworks_quick_reference.md", paths)
        self.assertIn("afterword.md", paths)
        self.assertNotIn("preface.md", paths)
        self.assertNotIn("contributors.md", paths)
        self.assertNotIn("part1/index.md", paths)
        self.assertNotIn("index.md", paths)

    def test_formal_pdf_front_matter_excludes_web_reading_guide(self):
        exporter = load_exporter()
        config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
        items = exporter.flatten_nav(exporter.find_en_nav(config))

        paths = [item.path for item in exporter.prepare_pdf_items(items)]

        self.assertIn("preface.md", paths)
        self.assertIn("acknowledgments.md", paths)
        self.assertIn("contributors.md", paths)
        self.assertIn("abbreviations.md", paths)
        self.assertNotIn("front_matter_guide.md", paths)

    def test_formal_pdf_front_matter_follows_springer_manuscript_order(self):
        exporter = load_exporter()
        front_html = exporter.generated_front_matter_html(include_toc=True)

        title_index = front_html.index("Title Page")
        preface_index = front_html.index(">Preface<")
        acknowledgments_index = front_html.index(">Acknowledgments<")
        contents_index = front_html.index(">Contents<")
        contributors_index = front_html.index(">Contributors<")
        abbreviations_index = front_html.index(">Abbreviations<")

        self.assertLess(title_index, preface_index)
        self.assertLess(preface_index, acknowledgments_index)
        self.assertLess(acknowledgments_index, contents_index)
        self.assertLess(contents_index, contributors_index)
        self.assertLess(contributors_index, abbreviations_index)
        self.assertNotIn("Half Title", front_html)
        self.assertNotIn("Copyright page placeholder", front_html)
        self.assertNotIn("Declaration of Competing Interests", front_html)
        self.assertNotIn("Ethics Approval", front_html)

    def test_springer_reference_pdf_sets_include_front_and_back_matter(self):
        exporter = load_exporter()
        config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
        items = exporter.flatten_nav(exporter.find_en_nav(config))

        front_paths = [item.path for item in exporter.front_matter_pdf_items(items)]
        back_paths = [item.path for item in exporter.back_matter_pdf_items(items)]

        self.assertIn("preface.md", front_paths)
        self.assertIn("acknowledgments.md", front_paths)
        self.assertIn("contributors.md", front_paths)
        self.assertIn("abbreviations.md", front_paths)
        self.assertIn("afterword.md", back_paths)
        self.assertNotIn("part10/ch31_agent_architecture.md", front_paths)
        self.assertNotIn("part10/ch31_agent_architecture.md", back_paths)

    def test_locate_item_pages_stops_after_all_items_are_found(self):
        exporter = load_exporter()

        class Page:
            def __init__(self, text: str):
                self.text = text
                self.calls = 0

            def extract_text(self):
                self.calls += 1
                return self.text

        class Reader:
            def __init__(self):
                self.pages = [
                    Page("Chapter 1: The Data Revolution in the Era of Large Models"),
                    Page("large image-only page"),
                    Page("large image-only page"),
                ]

        reader = Reader()
        item = exporter.NavItem(
            title="Chapter 1: The Data Revolution in the Era of Large Models",
            path="part1/ch01_data_change.md",
            level=2,
            group="Part 1",
            group_slug="part-1",
        )

        self.assertEqual({"part1/ch01_data_change.md": 0}, exporter.locate_item_pages(reader, [item]))
        self.assertEqual([1, 0, 0], [page.calls for page in reader.pages])

    def test_locate_item_pages_does_not_scan_for_part_overview_pages(self):
        exporter = load_exporter()

        class Page:
            def __init__(self, text: str):
                self.text = text
                self.calls = 0

            def extract_text(self):
                self.calls += 1
                return self.text

        class Reader:
            def __init__(self):
                self.pages = [
                    Page("Part I: Overview and Infrastructure"),
                    Page("Chapter 1: The Data Revolution in the Era of Large Models"),
                    Page("image-heavy page"),
                ]

        reader = Reader()
        items = [
            exporter.NavItem(
                title="Part Overview",
                path="part1/index.md",
                level=2,
                group="Part 1",
                group_slug="part-1",
            ),
            exporter.NavItem(
                title="Chapter 1: The Data Revolution in the Era of Large Models",
                path="part1/ch01_data_change.md",
                level=2,
                group="Part 1",
                group_slug="part-1",
            ),
        ]

        self.assertEqual(
            {"part1/index.md": 0, "part1/ch01_data_change.md": 1},
            exporter.locate_item_pages(reader, items),
        )
        self.assertEqual([0, 1, 0], [page.calls for page in reader.pages])

    def test_locate_item_pages_matches_rendered_titles_without_number_prefix(self):
        exporter = load_exporter()

        class Page:
            def __init__(self, text: str):
                self.text = text
                self.calls = 0

            def extract_text(self):
                self.calls += 1
                return self.text

        class Reader:
            def __init__(self):
                self.pages = [
                    Page("The Data Revolution in the Era of Large Models"),
                    Page("image-heavy page"),
                    Page("image-heavy page"),
                ]

        reader = Reader()
        item = exporter.NavItem(
            title="Chapter 1: The Data Revolution in the Era of Large Models",
            path="part1/ch01_data_change.md",
            level=2,
            group="Part 1",
            group_slug="part-1",
        )

        self.assertEqual({"part1/ch01_data_change.md": 0}, exporter.locate_item_pages(reader, [item]))
        self.assertEqual([1, 0, 0], [page.calls for page in reader.pages])

    def test_locate_item_pages_reuses_text_cache_for_same_cache_key(self):
        exporter = load_exporter()
        exporter.PDF_TEXT_CACHE.clear()

        class Page:
            def __init__(self, text: str):
                self.text = text
                self.calls = 0

            def extract_text(self):
                self.calls += 1
                return self.text

        class Reader:
            def __init__(self):
                self.pages = [Page("The Data Revolution in the Era of Large Models")]

        item = exporter.NavItem(
            title="Chapter 1: The Data Revolution in the Era of Large Models",
            path="part1/ch01_data_change.md",
            level=2,
            group="Part 1",
            group_slug="part-1",
        )
        first = Reader()
        second = Reader()

        self.assertEqual(
            {"part1/ch01_data_change.md": 0},
            exporter.locate_item_pages(first, [item], cache_key="/tmp/same-part.pdf"),
        )
        self.assertEqual(
            {"part1/ch01_data_change.md": 0},
            exporter.locate_item_pages(second, [item], cache_key="/tmp/same-part.pdf"),
        )
        self.assertEqual([1], [page.calls for page in first.pages])
        self.assertEqual([0], [page.calls for page in second.pages])

    def test_generated_pdf_front_matter_embeds_fonts(self):
        exporter = load_exporter()

        with tempfile.TemporaryDirectory(dir=ROOT / "output") as tmp:
            tmp_path = Path(tmp)
            opening = tmp_path / "opening.pdf"
            contents = tmp_path / "contents.pdf"

            exporter.generate_opening_front_pdf(opening, {"files": 1})
            exporter.generate_contents_pdf(contents, [("Chapter 1", 1, "1")], 2)

            self.assertEqual([], self.embedded_font_gaps(opening))
            self.assertEqual([], self.embedded_font_gaps(contents))

    def test_contents_pdf_handles_long_titles_without_hanging(self):
        exporter = load_exporter()

        with tempfile.TemporaryDirectory(dir=ROOT / "output") as tmp:
            contents = Path(tmp) / "contents.pdf"
            long_title = (
                "Chapter 99: "
                + "Very Long Multimodal Data Engineering Chapter Title " * 12
            )

            pages = exporter.generate_contents_pdf(contents, [(long_title, 2, "123")], 2)

            self.assertEqual(1, pages)
            self.assertGreater(contents.stat().st_size, 1_000)


class ExportEnglishBookLatexTest(unittest.TestCase):
    def test_submission_latex_items_exclude_web_only_part_overviews(self):
        exporter = load_latex_exporter()
        config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
        items = exporter.prepare_latex_items(exporter.flatten_nav(exporter.find_en_nav(config)))

        paths = [item.path for item in exporter.submission_latex_items(items)]

        self.assertIn("part1/ch01_data_change.md", paths)
        self.assertIn("part14/p15_dataagent_semantic_nl2sql_agent.md", paths)
        self.assertIn("appendix_a_tools_and_frameworks_quick_reference.md", paths)
        self.assertIn("afterword.md", paths)
        self.assertNotIn("part1/index.md", paths)
        self.assertNotIn("front_matter_guide.md", paths)
        self.assertNotIn("index.md", paths)

    def test_split_export_writes_one_latex_file_per_submission_unit(self):
        exporter = load_latex_exporter()
        config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
        items = exporter.prepare_latex_items(exporter.flatten_nav(exporter.find_en_nav(config)))

        with tempfile.TemporaryDirectory(dir=ROOT / "output") as tmp:
            tmp_path = Path(tmp)
            exporter.ASSET_DIR = tmp_path / "assets"
            exporter.PARTS_DIR = tmp_path / "parts"
            exporter.CHAPTERS_DIR = tmp_path / "chapters"
            exporter.OUT_WARNINGS = tmp_path / "warnings.txt"

            exporter.export_split(items, compile_output=False, timeout=30)

            chapter_files = sorted(exporter.CHAPTERS_DIR.glob("*.tex"))
            chapter_readme = (exporter.CHAPTERS_DIR / "README.md").read_text(encoding="utf-8")
            main_part_text = (exporter.PARTS_DIR / "00-manuscript.tex").read_text(encoding="utf-8")

            self.assertEqual(len(exporter.submission_latex_items(items)), len(chapter_files))
            self.assertTrue((exporter.CHAPTERS_DIR / "01-part1-ch01-data-change.tex").exists())
            self.assertTrue((exporter.CHAPTERS_DIR / "63-part14-p15-dataagent-semantic-nl2sql-agent.tex").exists())
            self.assertIn("part1/ch01_data_change.md", chapter_readme)
            self.assertIn(r"\input{../chapters/01-part1-ch01-data-change.tex}", main_part_text)

    def test_latex_asset_manager_converts_svg_to_png(self):
        exporter = load_latex_exporter()

        with tempfile.TemporaryDirectory(dir=ROOT / "output") as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "chapter.md"
            source.write_text("![sample](diagram.svg)\n", encoding="utf-8")
            svg = tmp_path / "diagram.svg"
            svg.write_text(
                """<svg xmlns="http://www.w3.org/2000/svg" width="120" height="80" viewBox="0 0 120 80">
<rect x="4" y="4" width="112" height="72" fill="#ffffff" stroke="#111111" stroke-width="2"/>
<line x1="20" y1="58" x2="100" y2="22" stroke="#2b6cb0" stroke-width="4"/>
<text x="18" y="26" font-size="14" fill="#111111">SVG</text>
</svg>
""",
                encoding="utf-8",
            )
            stats = exporter.ExportStats()
            assets = exporter.AssetManager(tmp_path / "assets", stats)
            assets.reset()

            rel = assets.register(svg, source)

            self.assertIsNotNone(rel)
            self.assertTrue(str(rel).endswith(".png"))
            written = tmp_path / rel
            self.assertTrue(written.exists())
            self.assertEqual(exporter.detect_image_suffix(written), ".png")
            self.assertEqual(stats.unsupported_images, 0)
            self.assertEqual(stats.warnings, [])


if __name__ == "__main__":
    unittest.main()
