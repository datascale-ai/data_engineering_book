"""Export the Chinese manuscript to a spacious 16K PDF.

The script reads the zh navigation from mkdocs.yml, concatenates the current
delivery manuscript in book order, writes a print-oriented HTML file, and uses
local Google Chrome headless to produce a PDF.

Usage:
    python3 scripts/export_zh_book_pdf.py
    python3 scripts/export_zh_book_pdf.py --no-pdf

Outputs:
    output/pdf/data_engineering_book_zh_16k_spacious.html
    output/pdf/data_engineering_book_zh_16k_spacious.pdf
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import markdown
import yaml


ROOT = Path(__file__).resolve().parents[1]
MKDOCS = ROOT / "mkdocs.yml"
DOCS_ZH = ROOT / "docs" / "zh"
OUT_DIR = ROOT / "output" / "pdf"
OUT_HTML = OUT_DIR / "data_engineering_book_zh_16k_spacious.html"
OUT_PDF = OUT_DIR / "data_engineering_book_zh_16k_spacious.pdf"
PARTS_DIR = OUT_DIR / "data_engineering_book_zh_16k_spacious_parts"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


CSS = r"""
@page {
  size: 185mm 260mm;
  margin: 22mm 18mm 21mm 18mm;
}

@media print {
  html, body {
    width: 185mm;
  }
  a {
    color: inherit;
    text-decoration: none;
  }
}

:root {
  --ink: #1d2430;
  --muted: #5f6876;
  --line: #d9dee7;
  --soft: #f6f8fb;
  --soft-2: #eef3f8;
  --accent: #245a8d;
  --accent-2: #6b4f9b;
  --code-bg: #f4f6f8;
}

* {
  box-sizing: border-box;
}

body {
  font-family:
    "Source Han Serif SC", "Noto Serif CJK SC", "Songti SC", "SimSun",
    "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", serif;
  font-size: 10.4pt;
  line-height: 1.82;
  letter-spacing: 0;
  color: var(--ink);
  margin: 0;
  background: #fff;
  word-break: break-word;
  overflow-wrap: anywhere;
  counter-reset: chapter;
}

.cover {
  min-height: 205mm;
  display: flex;
  flex-direction: column;
  justify-content: center;
  border-bottom: 1.5pt solid var(--ink);
  page-break-after: always;
}

.cover .kicker {
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 10pt;
  letter-spacing: 0;
  color: var(--muted);
  margin-bottom: 18mm;
}

.cover h1 {
  font-size: 26pt;
  line-height: 1.28;
  margin: 0 0 6mm;
  border: 0;
  padding: 0;
  page-break-before: auto;
}

.cover .subtitle {
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 13pt;
  color: var(--accent);
  margin-bottom: 22mm;
}

.cover .meta {
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 9.5pt;
  color: var(--muted);
  line-height: 1.9;
}

.front-note {
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
  color: var(--muted);
  border: 1px solid var(--line);
  background: var(--soft);
  padding: 4mm 5mm;
  margin: 0 0 8mm;
  font-size: 9.2pt;
}

.print-toc {
  page-break-after: always;
}

.print-toc h1 {
  page-break-before: auto;
}

.print-toc ol {
  columns: 2;
  column-gap: 12mm;
  margin: 0;
  padding-left: 5mm;
  font-size: 9pt;
  line-height: 1.65;
}

.print-toc li {
  break-inside: avoid;
  margin: 0 0 1.5mm;
}

.source-file {
  page-break-before: always;
}

.source-file.front-section {
  page-break-before: always;
}

.source-file.part-index {
  page-break-before: always;
}

.source-file.chapter,
.source-file.project,
.source-file.appendix {
  page-break-before: always;
}

.source-file h1:first-child {
  margin-top: 0;
}

h1, h2, h3, h4, h5, h6 {
  font-family:
    "Source Han Sans SC", "Noto Sans CJK SC", "PingFang SC",
    "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  font-weight: 650;
  letter-spacing: 0;
  color: #182336;
  break-after: avoid;
  page-break-after: avoid;
}

h1 {
  font-size: 19pt;
  line-height: 1.35;
  margin: 0 0 8mm;
  padding-bottom: 3mm;
  border-bottom: 1.2pt solid #1f2c3f;
}

h2 {
  font-size: 14.2pt;
  line-height: 1.42;
  margin: 9mm 0 4mm;
  padding-left: 3mm;
  border-left: 3pt solid var(--accent);
}

h3 {
  font-size: 12.2pt;
  line-height: 1.45;
  margin: 7mm 0 3mm;
  color: #214a73;
}

h4 {
  font-size: 11.1pt;
  margin: 5mm 0 2mm;
  color: #2e3b4e;
}

h5, h6 {
  font-size: 10.4pt;
  margin: 4mm 0 1.5mm;
}

p {
  margin: 0 0 3.2mm;
  text-align: justify;
  text-justify: inter-ideograph;
  widows: 2;
  orphans: 2;
}

ul, ol {
  margin: 2.5mm 0 4mm;
  padding-left: 6mm;
}

li {
  margin: 1.2mm 0;
  break-inside: avoid;
}

blockquote {
  margin: 4mm 0;
  padding: 3mm 4mm;
  border-left: 3pt solid var(--accent);
  background: var(--soft);
  color: #354052;
  break-inside: avoid;
}

hr {
  border: 0;
  border-top: 1px solid var(--line);
  margin: 6mm 0;
}

strong {
  font-weight: 700;
}

em {
  color: #3f4857;
}

a {
  color: #1c5d99;
  text-decoration: none;
}

img {
  display: block;
  max-width: 100%;
  max-height: 148mm;
  width: auto;
  height: auto;
  object-fit: contain;
  margin: 5mm auto 4mm;
  break-inside: avoid;
  page-break-inside: avoid;
}

p:has(img) {
  text-align: center;
  break-inside: avoid;
  page-break-inside: avoid;
  margin: 6mm 0;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 5mm 0 6mm;
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 8.6pt;
  line-height: 1.5;
  table-layout: fixed;
  break-inside: auto;
}

thead {
  display: table-header-group;
}

tr {
  break-inside: avoid;
  page-break-inside: avoid;
}

th, td {
  border: 0.6pt solid #cbd2dc;
  padding: 2.1mm 2.3mm;
  vertical-align: top;
  overflow-wrap: anywhere;
}

th {
  background: var(--soft-2);
  color: #1f4d78;
  font-weight: 700;
}

tbody tr:nth-child(even) td {
  background: #fbfcfe;
}

code {
  font-family: "SF Mono", "Menlo", "Consolas", "Liberation Mono", monospace;
  font-size: 8.7pt;
  background: var(--code-bg);
  color: #9a2f4f;
  padding: 0.2mm 0.8mm;
  border-radius: 2px;
}

pre {
  font-family: "SF Mono", "Menlo", "Consolas", "Liberation Mono", monospace;
  font-size: 8.5pt;
  line-height: 1.48;
  color: #263241;
  background: var(--code-bg);
  border: 0.6pt solid #dde3ea;
  border-left: 2.6pt solid var(--accent-2);
  border-radius: 3px;
  padding: 3.5mm 4mm;
  margin: 4.5mm 0 6mm;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  break-inside: avoid;
  page-break-inside: avoid;
}

pre code {
  color: inherit;
  background: transparent;
  padding: 0;
  border-radius: 0;
  font-size: inherit;
}

.mermaid-source {
  font-family: "SF Mono", "Menlo", "Consolas", monospace;
  background: #f7f7fa;
  border: 0.6pt solid #d9dbe4;
  padding: 3mm 4mm;
  margin: 4mm 0;
  white-space: pre-wrap;
  font-size: 8.3pt;
  line-height: 1.45;
}

.MathJax {
  font-size: 95% !important;
}

.footnote,
.reference,
.references {
  font-size: 9pt;
  line-height: 1.62;
}

.file-label {
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
  color: var(--muted);
  font-size: 8.3pt;
  margin: -4mm 0 6mm;
}
"""


MATHJAX = r"""
<script>
window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)'], ['$', '$']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true
  },
  svg: { fontCache: 'none' },
  startup: {
    ready: () => {
      MathJax.startup.defaultReady();
      MathJax.startup.promise.then(() => {
        document.body.setAttribute('data-mathjax-ready', 'true');
      });
    }
  }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
"""


@dataclass
class NavItem:
    title: str
    path: str
    level: int
    group: str
    group_slug: str


def find_zh_nav(config: dict[str, Any]) -> list[Any]:
    for plugin in config.get("plugins", []):
        if isinstance(plugin, dict) and "i18n" in plugin:
            for lang in plugin["i18n"].get("languages", []):
                if lang.get("locale") == "zh":
                    nav = lang.get("nav")
                    if isinstance(nav, list):
                        return nav
    raise ValueError("Cannot find zh navigation in mkdocs.yml")


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", value).strip("-")
    return slug[:48] or "section"


def flatten_nav(nodes: list[Any], level: int = 1, group: str = "卷前", group_slug: str = "front") -> list[NavItem]:
    items: list[NavItem] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        for title, value in node.items():
            if isinstance(value, str) and value.endswith(".md"):
                if value == "afterword.md":
                    item_group = "卷末"
                    item_group_slug = "back-matter"
                elif level == 1 and not re.search(r"part\d+/", value) and not value.startswith("appendix_"):
                    item_group = "卷前"
                    item_group_slug = "front-matter"
                else:
                    item_group = group
                    item_group_slug = group_slug
                items.append(NavItem(str(title), value, level, item_group, item_group_slug))
            elif isinstance(value, list):
                child_group = str(title) if level == 1 else group
                child_slug = slugify(child_group) if level == 1 else group_slug
                items.extend(flatten_nav(value, level + 1, child_group, child_slug))
    return items


def classify_path(path: str) -> str:
    if re.search(r"part\d+/ch\d+_", path):
        return "chapter"
    if re.search(r"part14/p\d+_", path):
        return "project"
    if re.search(r"part\d+/index\.md$", path):
        return "part-index"
    if path.startswith("appendix_"):
        return "appendix"
    return "front-section"


def rewrite_image_paths(markdown_text: str, source_file: Path) -> str:
    def repl(match: re.Match[str]) -> str:
        alt = match.group(1)
        raw_url = match.group(2).strip()
        if re.match(r"^(?:https?:|data:|file:|#)", raw_url):
            return match.group(0)
        url, suffix = split_url_suffix(raw_url)
        abs_path = (source_file.parent / url).resolve()
        return f"![{alt}]({abs_path.as_uri()}{suffix})"

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, markdown_text)

    def repl_img(match: re.Match[str]) -> str:
        tag = match.group(0)
        src_match = re.search(r'\bsrc=(["\'])(.*?)\1', tag)
        if not src_match:
            return tag
        src = src_match.group(2)
        if re.match(r"^(?:https?:|data:|file:|#)", src):
            return tag
        url, suffix = split_url_suffix(src)
        abs_path = (source_file.parent / url).resolve()
        return tag[: src_match.start(2)] + abs_path.as_uri() + suffix + tag[src_match.end(2) :]

    return re.sub(r"<img\b[^>]*>", repl_img, text, flags=re.I)


def split_url_suffix(url: str) -> tuple[str, str]:
    for sep in ("#", "?"):
        if sep in url:
            base, rest = url.split(sep, 1)
            return base, sep + rest
    return url, ""


def normalize_mermaid_blocks(markdown_text: str) -> str:
    """Keep Mermaid source readable when no renderer is available."""

    def repl(match: re.Match[str]) -> str:
        body = html.escape(match.group(1).strip())
        return f'<pre class="mermaid-source"><code>{body}</code></pre>'

    return re.sub(r"```mermaid\s*\n([\s\S]*?)\n```", repl, markdown_text, flags=re.I)


def make_markdown_converter() -> markdown.Markdown:
    return markdown.Markdown(
        extensions=[
            "tables",
            "fenced_code",
            "sane_lists",
            "attr_list",
            "md_in_html",
        ],
        output_format="html5",
    )


def build_book_html(
    items: list[NavItem],
    *,
    title_suffix: str = "",
    include_mathjax: bool = False,
    include_cover_toc: bool = True,
    toc_items: list[NavItem] | None = None,
    cover_stats: dict[str, int] | None = None,
) -> tuple[str, dict[str, int]]:
    md = make_markdown_converter()
    sections: list[str] = []
    included: set[str] = set()
    stats = {
        "files": 0,
        "missing": 0,
        "images": 0,
        "code_blocks": 0,
        "tables": 0,
    }

    for item in items:
        if item.path in included:
            continue
        included.add(item.path)
        src = DOCS_ZH / item.path
        if not src.exists():
            stats["missing"] += 1
            print(f"[WARN] missing nav source: {src}", file=sys.stderr)
            continue

        text = src.read_text(encoding="utf-8")
        stats["images"] += len(re.findall(r"!\[[^\]]*\]\([^)]+\)|<img\b", text, flags=re.I))
        stats["code_blocks"] += len(re.findall(r"^```", text, flags=re.M)) // 2
        stats["tables"] += len(
            re.findall(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", text, flags=re.M)
        )
        text = rewrite_image_paths(text, src)
        text = normalize_mermaid_blocks(text)
        html_body = md.convert(text)
        md.reset()
        file_class = classify_path(item.path)
        label = html.escape(f"{item.title} | {item.path}")
        sections.append(
            f'<section class="source-file {file_class}" data-source="{html.escape(item.path)}">\n'
            f'<div class="file-label">{label}</div>\n'
            f"{html_body}\n"
            "</section>"
        )
        stats["files"] += 1

    cover = build_cover(cover_stats or stats) if include_cover_toc else ""
    toc = build_print_toc(toc_items or items) if include_cover_toc else ""
    mathjax = MATHJAX if include_mathjax else ""
    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>大模型数据工程：架构、算法及项目实战 - 16K PDF{html.escape(title_suffix)}</title>
<style>{CSS}</style>
{mathjax}
</head>
<body>
{cover}
{toc}
<main>
{''.join(sections)}
</main>
</body>
</html>
"""
    return html_doc, stats


def build_cover(stats: dict[str, int]) -> str:
    return f"""
<section class="cover">
  <div class="kicker">Springer Chinese Manuscript Export | 16K Spacious Layout</div>
  <h1>大模型数据工程：<br>架构、算法及项目实战</h1>
  <div class="subtitle">中文首版排版样稿 PDF</div>
  <div class="meta">
    页面规格：16K，185mm x 260mm<br>
    排版口径：图表/代码保留较舒展，章节、项目、附录独立起页<br>
    导出范围：中文导航稿件，共 {stats.get("files", 0)} 个文件<br>
    生成工具：scripts/export_zh_book_pdf.py
  </div>
</section>
"""


def build_print_toc(items: list[NavItem]) -> str:
    lis: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item.path in seen:
            continue
        seen.add(item.path)
        indent = max(0, item.level - 1)
        style = f"margin-left:{indent * 3.5}mm"
        lis.append(f'<li style="{style}">{html.escape(item.title)}</li>')
    return f"""
<section class="print-toc">
  <h1>全书目录</h1>
  <div class="front-note">本目录按 mkdocs.yml 中文导航生成，用于排版样稿快速核对章节顺序；最终页码以出版社排版系统为准。</div>
  <ol>
    {''.join(lis)}
  </ol>
</section>
"""


def write_html(path: Path, html_doc: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding="utf-8")
    print(f"[ok] HTML written: {path}")


def export_pdf(html_path: Path, pdf_path: Path, timeout: int) -> None:
    if not CHROME.exists():
        raise FileNotFoundError(f"Google Chrome not found: {CHROME}")

    cmd = [
        str(CHROME),
        "--headless=new",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=1000",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    print("[run] " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Chrome PDF export failed with rc={proc.returncode}")
    if not pdf_path.exists() or pdf_path.stat().st_size < 100_000:
        raise RuntimeError("PDF was not produced or is suspiciously small")
    print(f"[ok] PDF written: {pdf_path} ({pdf_path.stat().st_size / 1024 / 1024:.1f} MB)")


def group_items(items: list[NavItem]) -> list[tuple[str, str, list[NavItem]]]:
    groups: list[tuple[str, str, list[NavItem]]] = []
    current_slug = ""
    current_title = ""
    current_items: list[NavItem] = []
    for item in items:
        if item.group_slug != current_slug:
            if current_items:
                groups.append((current_slug, current_title, current_items))
            current_slug = item.group_slug
            current_title = item.group
            current_items = []
        current_items.append(item)
    if current_items:
        groups.append((current_slug, current_title, current_items))
    return groups


def merge_pdfs(parts: list[Path], output: Path, items: list[NavItem] | None = None) -> None:
    try:
        from pypdf import PdfReader, PdfWriter
    except Exception as exc:  # pragma: no cover - dependency check
        raise RuntimeError("pypdf is required to merge part PDFs") from exc

    writer = PdfWriter()
    offsets: list[int] = []
    page_offset = 0
    for part in parts:
        offsets.append(page_offset)
        page_offset += len(PdfReader(str(part)).pages)
        writer.append(str(part))
    if items:
        add_bookmarks(writer, parts, offsets, items)
        try:
            writer.page_mode = "/UseOutlines"
        except Exception:
            pass
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)
    print(f"[ok] merged PDF written: {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")


def add_bookmarks(writer: Any, parts: list[Path], offsets: list[int], items: list[NavItem]) -> None:
    groups = group_items(items)
    if len(groups) != len(parts):
        print("[WARN] bookmark group count does not match part PDFs", file=sys.stderr)
        return
    writer.add_outline_item("封面", 0)
    writer.add_outline_item("全书目录", 1)
    for index, ((_, title, grouped), part, offset) in enumerate(zip(groups, parts, offsets), 1):
        part_reader = None
        try:
            part_reader = __import__("pypdf").PdfReader(str(part))
        except Exception:
            part_reader = None
        parent = writer.add_outline_item(title, offset)
        local_pages = locate_item_pages(part_reader, grouped, start_after_toc=index == 1) if part_reader else {}
        for item in grouped:
            local_page = local_pages.get(item.path, 0)
            writer.add_outline_item(item.title, offset + local_page, parent=parent)


def locate_item_pages(reader: Any, items: list[NavItem], start_after_toc: bool = False) -> dict[str, int]:
    def normalize(text: str) -> str:
        import unicodedata

        text = unicodedata.normalize("NFKC", text or "")
        return re.sub(r"[\W_]+", "", text)

    result: dict[str, int] = {}
    if reader is None:
        return result
    texts = [normalize(page.extract_text() or "") for page in reader.pages]
    current = 3 if start_after_toc and len(texts) > 3 else 0
    for item in items:
        needle = normalize(item.title)
        found = None
        for idx in range(current, len(texts)):
            if needle and needle in texts[idx]:
                found = idx
                break
        if found is None:
            found = min(current, max(0, len(texts) - 1))
        result[item.path] = found
        current = min(found + 1, max(0, len(texts) - 1))
    return result


def export_split_pdf(items: list[NavItem], timeout: int, include_mathjax: bool, global_stats: dict[str, int]) -> None:
    if PARTS_DIR.exists():
        shutil.rmtree(PARTS_DIR)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    part_pdfs: list[Path] = []
    groups = group_items(items)
    width = len(str(len(groups)))
    for index, (slug, title, grouped) in enumerate(groups, 1):
        prefix = f"{index:0{width}d}-{slug}"
        html_path = PARTS_DIR / f"{prefix}.html"
        pdf_path = PARTS_DIR / f"{prefix}.pdf"
        html_doc, stats = build_book_html(
            grouped,
            title_suffix=f" - {title}",
            include_mathjax=include_mathjax,
            include_cover_toc=index == 1,
            toc_items=items,
            cover_stats=global_stats,
        )
        write_html(html_path, html_doc)
        print(
            "[stats] "
            + title
            + " | "
            + ", ".join(
                [
                    f"files={stats['files']}",
                    f"missing={stats['missing']}",
                    f"images={stats['images']}",
                    f"code_blocks={stats['code_blocks']}",
                    f"tables={stats['tables']}",
                ]
            )
        )
        export_pdf(html_path, pdf_path, timeout)
        part_pdfs.append(pdf_path)
    merge_pdfs(part_pdfs, OUT_PDF, items)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export zh manuscript to 16K spacious PDF")
    parser.add_argument("--no-pdf", action="store_true", help="Only write the intermediate HTML")
    parser.add_argument("--split", action="store_true", help="Export by book part and merge PDFs")
    parser.add_argument("--mathjax", action="store_true", help="Load MathJax from CDN before printing")
    parser.add_argument("--timeout", type=int, default=1200, help="Chrome export timeout in seconds")
    args = parser.parse_args()

    config = yaml.safe_load(MKDOCS.read_text(encoding="utf-8"))
    items = flatten_nav(find_zh_nav(config))
    html_doc, stats = build_book_html(items, include_mathjax=args.mathjax)
    write_html(OUT_HTML, html_doc)
    print(
        "[stats] "
        + ", ".join(
            [
                f"files={stats['files']}",
                f"missing={stats['missing']}",
                f"images={stats['images']}",
                f"code_blocks={stats['code_blocks']}",
                f"tables={stats['tables']}",
            ]
        )
    )
    if not args.no_pdf:
        if args.split:
            export_split_pdf(items, args.timeout, args.mathjax, stats)
        else:
            export_pdf(OUT_HTML, OUT_PDF, args.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
