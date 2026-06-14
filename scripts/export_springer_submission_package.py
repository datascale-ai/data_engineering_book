#!/usr/bin/env python3
"""Export a deterministic Springer submission package.

The script organizes the current manuscript, PDFs, figures, permissions,
declarations, audit reports, and checksums into a publisher-facing folder.
It does not fabricate legal proof files; it copies `publishing/permissions`
as provided by the author/editor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOOK_SLUG = "Data_Engineering_for_Large_Foundation_Models_A_Handbook"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "springer_submission"
PDF_DIR = ROOT / "output" / "pdf"
SUBMISSION_PDF_DIR = PDF_DIR / "data_engineering_book_en_16k_compact_submission_pdfs"
LATEX_PARTS_DIR = PDF_DIR / "data_engineering_book_en_16k_latex_parts"
LATEX_CHAPTERS_DIR = PDF_DIR / "data_engineering_book_en_16k_latex_chapters"
LATEX_ASSETS_DIR = PDF_DIR / "latex_assets_en"
ACCESSIBILITY_DIR = ROOT / "publishing" / "accessibility"
LATEX_EXPORT_SCRIPT = ROOT / "scripts" / "export_en_book_latex.py"


@dataclass
class ManifestRow:
    relative_path: str
    size_bytes: int
    sha256: str
    source_path: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def latex_root_tex() -> Path:
    return PDF_DIR / "data_engineering_book_en_16k_latex.tex"


def has_tex_files(path: Path) -> bool:
    return path.exists() and any(item.is_file() for item in path.glob("*.tex"))


def latex_sources_complete() -> bool:
    return (
        latex_root_tex().exists()
        and has_tex_files(LATEX_CHAPTERS_DIR)
        and has_tex_files(LATEX_PARTS_DIR)
        and LATEX_ASSETS_DIR.exists()
    )


def run_latex_export(args: list[str]) -> None:
    subprocess.run([sys.executable, str(LATEX_EXPORT_SCRIPT), *args], cwd=ROOT, check=True)


def ensure_latex_sources() -> None:
    if not has_tex_files(LATEX_CHAPTERS_DIR) or not has_tex_files(LATEX_PARTS_DIR) or not LATEX_ASSETS_DIR.exists():
        run_latex_export(["--split"])
    if not latex_root_tex().exists():
        run_latex_export([])
    if not latex_sources_complete():
        missing: list[str] = []
        if not latex_root_tex().exists():
            missing.append(str(latex_root_tex()))
        if not has_tex_files(LATEX_CHAPTERS_DIR):
            missing.append(str(LATEX_CHAPTERS_DIR))
        if not has_tex_files(LATEX_PARTS_DIR):
            missing.append(str(LATEX_PARTS_DIR))
        if not LATEX_ASSETS_DIR.exists():
            missing.append(str(LATEX_ASSETS_DIR))
        raise RuntimeError("LaTeX source export incomplete; missing or empty: " + ", ".join(missing))


def rewrite_latex_package_paths(package_dir: Path) -> None:
    latex_root = package_dir / "Source_Files" / "LaTeX"
    replacements = {
        "../latex_assets_en/": "../assets/",
        "latex_assets_en/": "assets/",
        "../data_engineering_book_en_16k_latex_chapters/": "../chapters/",
    }
    for tex_file in sorted(latex_root.rglob("*.tex")):
        text = tex_file.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        tex_file.write_text(text, encoding="utf-8")


def copy_markdown_sources(package_dir: Path) -> None:
    source_root = package_dir / "Source_Files"
    copy_tree(ROOT / "docs" / "en", source_root / "Markdown" / "docs_en")
    copy_tree(ROOT / "docs" / "zh", source_root / "Markdown" / "docs_zh_reference")
    copy_file(ROOT / "mkdocs.yml", source_root / "mkdocs.yml")
    copy_file(ROOT / "README.md", source_root / "README.md")
    copy_file(ROOT / "publishing" / "12_figures_tables_register.md", source_root / "12_figures_tables_register.md")
    copy_tree(LATEX_PARTS_DIR, source_root / "LaTeX" / "parts")
    copy_tree(LATEX_CHAPTERS_DIR, source_root / "LaTeX" / "chapters")
    copy_tree(LATEX_ASSETS_DIR, source_root / "LaTeX" / "assets")
    copy_file(latex_root_tex(), source_root / "LaTeX" / "data_engineering_book_en_16k_latex.tex")
    rewrite_latex_package_paths(package_dir)


def copy_metadata(package_dir: Path) -> None:
    metadata = package_dir / "Metadata"
    declarations = package_dir / "Declarations"
    copy_file(ROOT / "publishing" / "18_springer_submission_package.md", metadata / "18_springer_submission_package.md")
    copy_file(ROOT / "publishing" / "15_final_delivery_checklist.md", metadata / "15_final_delivery_checklist.md")
    copy_file(ROOT / "publishing" / "19_declarations_and_metadata_templates.md", declarations / "19_declarations_and_metadata_templates.md")


def copy_permissions_and_audits(package_dir: Path) -> None:
    copy_tree(ROOT / "publishing" / "permissions", package_dir / "Permissions")
    copy_tree(ROOT / "publishing" / "final_review", package_dir / "Audit_Reports")


def copy_accessibility(package_dir: Path) -> None:
    copy_tree(ACCESSIBILITY_DIR, package_dir / "Accessibility")


def copy_pdfs(package_dir: Path) -> None:
    full_dir = package_dir / "Full_PDF"
    chapter_dir = package_dir / "Chapter_PDFs"
    full_dir.mkdir(parents=True, exist_ok=True)
    chapter_dir.mkdir(parents=True, exist_ok=True)
    if not SUBMISSION_PDF_DIR.exists():
        return
    for pdf in sorted(SUBMISSION_PDF_DIR.glob("*.pdf")):
        if pdf.name == "00_full_book_pagenumbered.pdf":
            copy_file(pdf, full_dir / f"{BOOK_SLUG}_{pdf.name}")
        else:
            copy_file(pdf, chapter_dir / f"{BOOK_SLUG}_{pdf.name}")
    copy_file(SUBMISSION_PDF_DIR / "README.md", chapter_dir / "README.md")


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob(pattern) if item.is_file())


def write_package_readme(package_dir: Path) -> None:
    chapter_pdf_count = count_files(package_dir / "Chapter_PDFs", "*.pdf")
    full_pdf_count = count_files(package_dir / "Full_PDF", "*.pdf")
    latex_source = "Source_Files/LaTeX/data_engineering_book_en_16k_latex.tex"
    readme = f"""# Data Engineering for Large Foundation Models: A Handbook

This folder is the publisher-facing Springer submission package generated from the current repository state.

## Package Contents

| Folder | Purpose |
| --- | --- |
| `Source_Files/Markdown` | English manuscript sources plus Chinese reference sources. |
| `Source_Files/LaTeX` | LaTeX export sources, including `{latex_source}`, `chapters/` independent chapter/contribution `.tex` files, `parts/` review-group `.tex` files, and LaTeX assets. |
| `Full_PDF` | Complete paginated review PDF. Current PDF count: {full_pdf_count}. |
| `Chapter_PDFs` | Springer reference PDF set: front matter PDF, chapter/project/appendix PDFs, and back matter PDF when applicable. Current PDF count: {chapter_pdf_count}. |
| `Figures` | Figure files referenced by the English manuscript, with `figures_manifest.csv`. |
| `Accessibility` | Springer alt-text Excel workbook plus CSV/JSON sidecars for all manuscript images. |
| `Permissions` | Author/editor-provided third-party permission evidence copied as-is. |
| `Declarations` | Declaration and metadata templates for publisher workflow completion. |
| `Audit_Reports` | Machine audit reports plus human signoff/exception notes. |
| `Checksums` | SHA-256 manifests for package integrity verification. |

## Submission Notes

- Submit `Source_Files/LaTeX`, especially `Source_Files/LaTeX/chapters`, and the referenced figures/assets as the editable source package when LaTeX source is requested.
- Submit `Full_PDF` for whole-book layout review.
- Submit `Chapter_PDFs` as the individual PDF set requested by Springer guidelines: front matter, chapters/contributions, appendices, and back matter when applicable.
- Submit `Accessibility/springer_alt_text_inventory.xlsx` with the final manuscript to satisfy Springer Nature's alt-text accessibility requirement for figures, illustrations, and images.
- Keep `Permissions` with the package. The export script does not fabricate permission letters or publisher forms.

## Human-Only Items

The following human-only items cannot be generated by this repository and must be supplied or confirmed by the author/editor or Springer production workflow when requested:

- signed License to Publish forms;
- publisher-approved copyright page and imprint metadata;
- original third-party permission correspondence beyond the signoff notes already present;
- final author/editor approval of any proof-stage changes.

## Verification Pointers

- Figure rights status: `Audit_Reports/figure_rights_signoff.md`.
- Quantitative-data status: `Audit_Reports/quantitative_data_signoff.md` and `Audit_Reports/quantitative_source_spotcheck.md`.
- Reference status: `Audit_Reports/reference_integrity_audit.md` and `Audit_Reports/reference_doi_url_exceptions.md`.
- Package checksums: `Checksums/manifest.json` and `Checksums/manifest.csv`.
"""
    (package_dir / "README.md").write_text(readme, encoding="utf-8")


def markdown_image_targets(markdown_path: Path) -> set[Path]:
    text = markdown_path.read_text(encoding="utf-8", errors="replace")
    urls = set(re.findall(r"!\[[^\]]*]\(([^)]+)\)", text))
    urls.update(re.findall(r"<img[^>]+src=[\"']([^\"']+)[\"']", text, flags=re.I))
    targets: set[Path] = set()
    for raw in urls:
        url = raw.strip().split("#", 1)[0].split("?", 1)[0]
        if not url or re.match(r"^(?:https?:|data:|file:|#)", url):
            continue
        path = (markdown_path.parent / url).resolve()
        if path.exists() and path.is_file() and path.is_relative_to(ROOT):
            targets.add(path)
    return targets


def copy_figures(package_dir: Path) -> None:
    figure_root = package_dir / "Figures"
    manifest_rows: list[dict[str, str]] = []
    targets: set[Path] = set()
    for markdown_path in sorted((ROOT / "docs" / "en").rglob("*.md")):
        targets.update(markdown_image_targets(markdown_path))
    for src in sorted(targets):
        rel = src.relative_to(ROOT)
        dst = figure_root / rel
        copy_file(src, dst)
        manifest_rows.append({"source": rel.as_posix(), "package_path": dst.relative_to(package_dir).as_posix()})
    if manifest_rows:
        figure_root.mkdir(parents=True, exist_ok=True)
        with (figure_root / "figures_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source", "package_path"])
            writer.writeheader()
            writer.writerows(manifest_rows)


def collect_manifest(package_dir: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.relative_to(package_dir).as_posix().startswith("Checksums/"):
            continue
        rel = path.relative_to(package_dir).as_posix()
        rows.append(
            ManifestRow(
                relative_path=rel,
                size_bytes=path.stat().st_size,
                sha256=sha256(path),
                source_path="",
            )
        )
    return rows


def write_manifest(package_dir: Path) -> None:
    rows = collect_manifest(package_dir)
    checksums = package_dir / "Checksums"
    checksums.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "book": "Data Engineering for Large Foundation Models: A Handbook",
        "generated_at_utc": generated_at,
        "files": [asdict(row) for row in rows],
    }
    (checksums / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (checksums / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256", "source_path"])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def create_zip_archive(package_dir: Path) -> Path:
    zip_path = package_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if not path.is_file():
                continue
            archive.write(path, path.relative_to(package_dir.parent).as_posix())
    return zip_path


def export_package(output_root: Path = DEFAULT_OUTPUT_ROOT, *, include_pdfs: bool = True, include_figures: bool = True) -> Path:
    package_dir = output_root / BOOK_SLUG
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "Metadata",
        "Source_Files",
        "Chapter_PDFs",
        "Full_PDF",
        "Figures",
        "Accessibility",
        "Permissions",
        "Declarations",
        "Checksums",
        "Audit_Reports",
    ]:
        (package_dir / name).mkdir(parents=True, exist_ok=True)
    ensure_latex_sources()
    copy_metadata(package_dir)
    copy_markdown_sources(package_dir)
    copy_permissions_and_audits(package_dir)
    copy_accessibility(package_dir)
    if include_pdfs:
        copy_pdfs(package_dir)
    if include_figures:
        copy_figures(package_dir)
    write_package_readme(package_dir)
    write_manifest(package_dir)
    return package_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the Springer submission package.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-pdfs", action="store_true", help="Skip copying generated PDF outputs.")
    parser.add_argument("--no-figures", action="store_true", help="Skip copying referenced figure files.")
    parser.add_argument("--zip", action="store_true", help="Also create a ZIP archive next to the package folder.")
    args = parser.parse_args()
    package_dir = export_package(args.output_root, include_pdfs=not args.no_pdfs, include_figures=not args.no_figures)
    print(f"[ok] Springer submission package written: {package_dir}")
    print(f"[ok] Manifest: {package_dir / 'Checksums' / 'manifest.json'}")
    if args.zip:
        zip_path = create_zip_archive(package_dir)
        print(f"[ok] ZIP archive written: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
