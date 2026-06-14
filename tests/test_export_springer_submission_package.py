from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_springer_submission_package.py"


def load_exporter():
    spec = importlib.util.spec_from_file_location("export_springer_submission_package", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SpringerSubmissionPackageTest(unittest.TestCase):
    def test_export_package_creates_core_folders_and_manifest(self):
        exporter = load_exporter()

        with tempfile.TemporaryDirectory(dir=ROOT / "output") as tmp:
            package_dir = exporter.export_package(Path(tmp), include_pdfs=False, include_figures=False)

            self.assertTrue((package_dir / "Metadata").is_dir())
            self.assertTrue((package_dir / "Source_Files").is_dir())
            self.assertTrue((package_dir / "Permissions").is_dir())
            self.assertTrue((package_dir / "Checksums" / "manifest.json").exists())
            manifest = json.loads((package_dir / "Checksums" / "manifest.json").read_text(encoding="utf-8"))
            rel_paths = {row["relative_path"] for row in manifest["files"]}
            self.assertIn("Metadata/18_springer_submission_package.md", rel_paths)
            self.assertIn("Permissions/README.md", rel_paths)

    def test_export_package_includes_chapter_level_latex_sources(self):
        exporter = load_exporter()

        with tempfile.TemporaryDirectory(dir=ROOT / "output") as tmp:
            tmp_path = Path(tmp)
            latex_chapters = tmp_path / "latex_chapters"
            latex_parts = tmp_path / "latex_parts"
            latex_assets = tmp_path / "latex_assets"
            latex_chapters.mkdir()
            latex_parts.mkdir()
            latex_assets.mkdir()
            (latex_chapters / "01-part1-ch01-data-change.tex").write_text(
                r"\chapter{One}\includegraphics{../latex_assets_en/asset_0001.png}",
                encoding="utf-8",
            )
            (latex_chapters / "README.md").write_text("# Chapter sources\n", encoding="utf-8")
            (latex_parts / "00-manuscript.tex").write_text(
                "\n".join(
                    [
                        r"\input{../data_engineering_book_en_16k_latex_chapters/01-part1-ch01-data-change.tex}",
                        r"\includegraphics{../latex_assets_en/asset_0001.png}",
                    ]
                ),
                encoding="utf-8",
            )
            (latex_assets / "asset_0001.png").write_bytes(b"png")
            latex_root = tmp_path / "data_engineering_book_en_16k_latex.tex"
            latex_root.write_text(r"\includegraphics{latex_assets_en/asset_0001.png}", encoding="utf-8")
            exporter.LATEX_CHAPTERS_DIR = latex_chapters
            exporter.LATEX_PARTS_DIR = latex_parts
            exporter.LATEX_ASSETS_DIR = latex_assets
            exporter.PDF_DIR = tmp_path

            package_dir = exporter.export_package(tmp_path, include_pdfs=False, include_figures=False)

            chapter_source = (
                package_dir
                / "Source_Files"
                / "LaTeX"
                / "chapters"
                / "01-part1-ch01-data-change.tex"
            )
            self.assertTrue(chapter_source.exists())
            self.assertIn("../assets/asset_0001.png", chapter_source.read_text(encoding="utf-8"))
            manuscript = package_dir / "Source_Files" / "LaTeX" / "parts" / "00-manuscript.tex"
            self.assertIn(
                r"\input{../chapters/01-part1-ch01-data-change.tex}",
                manuscript.read_text(encoding="utf-8"),
            )
            self.assertIn("../assets/asset_0001.png", manuscript.read_text(encoding="utf-8"))
            root_source = package_dir / "Source_Files" / "LaTeX" / "data_engineering_book_en_16k_latex.tex"
            self.assertIn("assets/asset_0001.png", root_source.read_text(encoding="utf-8"))
            self.assertIn("LaTeX/chapters", (package_dir / "README.md").read_text(encoding="utf-8"))

    def test_export_package_generates_missing_latex_sources_before_copying(self):
        exporter = load_exporter()

        with tempfile.TemporaryDirectory(dir=ROOT / "output") as tmp:
            tmp_path = Path(tmp)
            latex_chapters = tmp_path / "missing_chapters"
            latex_parts = tmp_path / "missing_parts"
            latex_assets = tmp_path / "missing_assets"
            latex_root = tmp_path / "data_engineering_book_en_16k_latex.tex"
            exporter.PDF_DIR = tmp_path
            exporter.LATEX_CHAPTERS_DIR = latex_chapters
            exporter.LATEX_PARTS_DIR = latex_parts
            exporter.LATEX_ASSETS_DIR = latex_assets

            calls: list[tuple[str, ...]] = []

            def fake_run_latex_export(args: list[str]) -> None:
                calls.append(tuple(args))
                if args == ["--split"]:
                    latex_chapters.mkdir(parents=True, exist_ok=True)
                    latex_parts.mkdir(parents=True, exist_ok=True)
                    latex_assets.mkdir(parents=True, exist_ok=True)
                    (latex_chapters / "01-part1-ch01-data-change.tex").write_text(
                        r"\chapter{One}\includegraphics{../latex_assets_en/asset_0001.png}",
                        encoding="utf-8",
                    )
                    (latex_parts / "00-manuscript.tex").write_text(
                        r"\input{../data_engineering_book_en_16k_latex_chapters/01-part1-ch01-data-change.tex}",
                        encoding="utf-8",
                    )
                    (latex_assets / "asset_0001.png").write_bytes(b"png")
                elif args == []:
                    latex_root.write_text(
                        r"\includegraphics{latex_assets_en/asset_0001.png}",
                        encoding="utf-8",
                    )

            exporter.run_latex_export = fake_run_latex_export

            package_dir = exporter.export_package(tmp_path, include_pdfs=False, include_figures=False)

            self.assertEqual([("--split",), ()], calls)
            self.assertTrue(
                (
                    package_dir
                    / "Source_Files"
                    / "LaTeX"
                    / "chapters"
                    / "01-part1-ch01-data-change.tex"
                ).exists()
            )
            self.assertTrue((package_dir / "Source_Files" / "LaTeX" / "parts" / "00-manuscript.tex").exists())
            self.assertTrue((package_dir / "Source_Files" / "LaTeX" / "assets" / "asset_0001.png").exists())
            self.assertTrue(
                (
                    package_dir
                    / "Source_Files"
                    / "LaTeX"
                    / "data_engineering_book_en_16k_latex.tex"
                ).exists()
            )

    def test_export_package_writes_publisher_facing_readme(self):
        exporter = load_exporter()

        with tempfile.TemporaryDirectory(dir=ROOT / "output") as tmp:
            package_dir = exporter.export_package(Path(tmp), include_pdfs=False, include_figures=False)

            readme = package_dir / "README.md"
            self.assertTrue(readme.exists())
            text = readme.read_text(encoding="utf-8")
            self.assertIn("Data Engineering for Large Foundation Models: A Handbook", text)
            self.assertIn("Source_Files/LaTeX", text)
            self.assertIn("Full_PDF", text)
            self.assertIn("Chapter_PDFs", text)
            self.assertIn("human-only", text)
            self.assertIn("License to Publish", text)

    def test_copy_pdfs_keeps_front_and_back_matter_with_chapter_pdf_set(self):
        exporter = load_exporter()

        with tempfile.TemporaryDirectory(dir=ROOT / "output") as tmp:
            tmp_path = Path(tmp)
            pdf_dir = tmp_path / "submission_pdfs"
            pdf_dir.mkdir()
            for name in [
                "00_full_book_pagenumbered.pdf",
                "00_front_matter.pdf",
                "01-part1-ch01_data_change.pdf",
                "99_back_matter.pdf",
            ]:
                (pdf_dir / name).write_bytes(b"%PDF-1.4\n%test\n")
            (pdf_dir / "README.md").write_text("# PDFs\n", encoding="utf-8")
            exporter.SUBMISSION_PDF_DIR = pdf_dir

            package_dir = tmp_path / "package"
            exporter.copy_pdfs(package_dir)

            full_names = {path.name for path in (package_dir / "Full_PDF").glob("*.pdf")}
            chapter_names = {path.name for path in (package_dir / "Chapter_PDFs").glob("*.pdf")}

            self.assertIn(f"{exporter.BOOK_SLUG}_00_full_book_pagenumbered.pdf", full_names)
            self.assertNotIn(f"{exporter.BOOK_SLUG}_00_front_matter.pdf", full_names)
            self.assertIn(f"{exporter.BOOK_SLUG}_00_front_matter.pdf", chapter_names)
            self.assertIn(f"{exporter.BOOK_SLUG}_99_back_matter.pdf", chapter_names)

    def test_create_zip_archive_preserves_package_root_and_manifest(self):
        exporter = load_exporter()

        with tempfile.TemporaryDirectory(dir=ROOT / "output") as tmp:
            package_dir = exporter.export_package(Path(tmp), include_pdfs=False, include_figures=False)
            zip_path = exporter.create_zip_archive(package_dir)

            self.assertTrue(zip_path.exists())
            self.assertEqual(zip_path.suffix, ".zip")
            with zipfile.ZipFile(zip_path) as archive:
                names = set(archive.namelist())

            root = f"{exporter.BOOK_SLUG}/"
            self.assertIn(root + "README.md", names)
            self.assertIn(root + "Checksums/manifest.json", names)
            self.assertIn(root + "Source_Files/mkdocs.yml", names)


if __name__ == "__main__":
    unittest.main()
