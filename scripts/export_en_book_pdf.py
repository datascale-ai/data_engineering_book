"""Export the English manuscript to a compact 16K PDF.

The script reads the en navigation from mkdocs.yml, concatenates the current
delivery manuscript in book order, writes a print-oriented HTML file, and uses
local Google Chrome headless to produce a PDF.

Usage:
    uv run python scripts/export_en_book_pdf.py --split
    uv run python scripts/export_en_book_pdf.py --no-pdf

Outputs:
    output/pdf/data_engineering_book_en_16k_compact.html
    output/pdf/data_engineering_book_en_16k_compact.pdf
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
DOCS_EN = ROOT / "docs" / "en"
OUT_DIR = ROOT / "output" / "pdf"
OUT_HTML = OUT_DIR / "data_engineering_book_en_16k_compact.html"
OUT_PDF = OUT_DIR / "data_engineering_book_en_16k_compact.pdf"
PARTS_DIR = OUT_DIR / "data_engineering_book_en_16k_compact_parts"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
FRONT_PDF = PARTS_DIR / "00-book-front-matter.pdf"
OPENING_FRONT_PDF = PARTS_DIR / "00a-opening-front-matter.pdf"
CONTENTS_PDF = PARTS_DIR / "00b-contents.pdf"
SUBMISSION_PDF_DIR = OUT_DIR / "data_engineering_book_en_16k_compact_submission_pdfs"
PDF_TEXT_CACHE: dict[str, dict[int, str]] = {}

EXCLUDED_FROM_FORMAL_PDF = {"title_page.md", "index.md", "translation-status.md", "front_matter_guide.md"}
PRE_CONTENTS_FRONT_PATHS = {"preface.md", "acknowledgments.md"}
POST_CONTENTS_FRONT_PATHS = {"contributors.md", "abbreviations.md"}

BOOK_AUTHORS = "Jun Yu, Ran Zhang, Wenzhuo Du, Guanlin Mu, ZhiLi Wang, Ke Wang, and Xin Xu"
PDF_FONT_REGULAR = "DataEngineeringBook-Regular"
PDF_FONT_BOLD = "DataEngineeringBook-Bold"
PDF_FONT_REGISTERED = False
PDF_FONT_CANDIDATES = {
    PDF_FONT_REGULAR: [
        Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
        Path("/Library/Fonts/Times New Roman.ttf"),
    ],
    PDF_FONT_BOLD: [
        Path("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
        Path("/Library/Fonts/Times New Roman Bold.ttf"),
    ],
}
SECTION_AUTHORS = {
    **{f"part1/ch{chapter:02d}_": "Ke Wang" for chapter in range(1, 4)},
    **{f"part2/ch{chapter:02d}_": "Ke Wang" for chapter in range(4, 8)},
    **{f"part3/ch{chapter:02d}_": "Ke Wang" for chapter in range(8, 12)},
    **{f"part4/ch{chapter:02d}_": "Ran Zhang" for chapter in range(12, 15)},
    **{f"part5/ch{chapter:02d}_": "Ran Zhang" for chapter in range(15, 18)},
    **{f"part6/ch{chapter:02d}_": "Ran Zhang" for chapter in range(18, 21)},
    **{f"part7/ch{chapter:02d}_": "Wenzhuo Du" for chapter in range(21, 24)},
    **{f"part8/ch{chapter:02d}_": "Wenzhuo Du" for chapter in range(24, 27)},
    **{f"part9/ch{chapter:02d}_": "Wenzhuo Du" for chapter in range(27, 31)},
    **{f"part10/ch{chapter:02d}_": "ZhiLi Wang" for chapter in range(31, 36)},
    "part11/ch36_": "ZhiLi Wang; Xin Xu",
    "part11/ch37_": "ZhiLi Wang; Xin Xu",
    "part12/ch38_": "Guanlin Mu",
    "part12/ch39_": "Guanlin Mu",
    "part12/ch40_": "Guanjun Liu; Yuefeng Zou",
    "part12/ch41_": "Lin Xu; Xinyu Chen",
    "part12/ch42_": "Fengxin Chen; Xuan Li",
    "part12/ch43_": "Fengxin Chen; Xuan Li",
    "part13/ch44_": "Ke Wang",
    "part13/ch45_": "Xin Xu",
    "part13/ch46_": "Xin Xu",
    "part13/ch47_": "Ke Wang",
    "part13/ch48_": "Ran Zhang",
    **{f"part14/p{project:02d}_": "Xin Xu" for project in range(1, 11)},
    "part14/p11_": "Ke Wang",
    "part14/p12_": "Xin Xu",
    "part14/p13_": "Ke Wang",
    "part14/p14_": "Ran Zhang",
    "part14/p15_": "caoxuhong",
    "appendix_a_": "Xin Xu",
    "appendix_b_": "Xin Xu",
    "appendix_c_": "Xin Xu",
    "appendix_d_": "ZhiLi Wang",
    "appendix_e_": "ZhiLi Wang",
    "appendix_f_": "ZhiLi Wang",
    "appendix_g_": "Xuhong Cao",
    "appendix_h_": "Xin Xu",
}


def register_pdf_fonts() -> tuple[str, str]:
    """Register embedded TrueType fonts for generated PDF pages and overlays."""

    global PDF_FONT_REGISTERED
    if PDF_FONT_REGISTERED:
        return PDF_FONT_REGULAR, PDF_FONT_BOLD
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception as exc:  # pragma: no cover - dependency check
        raise RuntimeError("reportlab font support is required for PDF generation") from exc

    for font_name, candidates in PDF_FONT_CANDIDATES.items():
        for candidate in candidates:
            if candidate.exists():
                pdfmetrics.registerFont(TTFont(font_name, str(candidate)))
                break
        else:
            raise FileNotFoundError(f"Cannot find a TrueType font for {font_name}: {candidates}")
    PDF_FONT_REGISTERED = True
    return PDF_FONT_REGULAR, PDF_FONT_BOLD


def author_line_for_path(path: str) -> str:
    for prefix, authors in SECTION_AUTHORS.items():
        if path.startswith(prefix):
            return authors
    return ""


CSS = r"""
@page {
  size: 185mm 260mm;
  margin: 21mm 18mm 19mm 18mm;
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
    "Times New Roman", "Times", "Nimbus Roman", serif;
  font-size: 11pt;
  line-height: 1.45;
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
  font-family: "Times New Roman", "Times", "Nimbus Roman", serif;
  font-size: 10pt;
  letter-spacing: 0;
  color: var(--muted);
  margin-bottom: 18mm;
}

.cover h1 {
  font-size: 24pt;
  line-height: 1.28;
  margin: 0 0 6mm;
  border: 0;
  padding: 0;
  page-break-before: auto;
}

.cover .subtitle {
  font-family: "Times New Roman", "Times", "Nimbus Roman", serif;
  font-size: 13pt;
  color: var(--accent);
  margin-bottom: 22mm;
}

.cover .meta {
  font-family: "Times New Roman", "Times", "Nimbus Roman", serif;
  font-size: 9.5pt;
  color: var(--muted);
  line-height: 1.9;
}

.front-note {
  font-family: "Times New Roman", "Times", "Nimbus Roman", serif;
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
  font-size: 8.4pt;
  line-height: 1.35;
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
    "Times New Roman", "Times", "Nimbus Roman", serif;
  font-weight: 700;
  letter-spacing: 0;
  color: #182336;
  break-after: avoid;
  page-break-after: avoid;
}

h1 {
  font-size: 17pt;
  line-height: 1.25;
  margin: 0 0 5.5mm;
  padding-bottom: 2.2mm;
  border-bottom: 1.2pt solid #1f2c3f;
}

h2 {
  font-size: 13pt;
  line-height: 1.3;
  margin: 5.2mm 0 2.2mm;
  padding-left: 3mm;
  border-left: 3pt solid var(--accent);
}

h3 {
  font-size: 11.5pt;
  line-height: 1.3;
  margin: 4.2mm 0 1.8mm;
  color: #214a73;
}

h4 {
  font-size: 11.2pt;
  margin: 5mm 0 2mm;
  color: #2e3b4e;
}

h5, h6 {
  font-size: 11pt;
  margin: 4mm 0 1.5mm;
}

p {
  margin: 0 0 2.1mm;
  text-align: justify;
  text-justify: inter-ideograph;
  widows: 2;
  orphans: 2;
}

ul, ol {
  margin: 1.8mm 0 2.8mm;
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
  max-height: 102mm;
  width: auto;
  height: auto;
  object-fit: contain;
  margin: 1.4mm auto;
  break-inside: auto;
  page-break-inside: auto;
}

p:has(img) {
  text-align: center;
  break-inside: auto;
  page-break-inside: auto;
  margin: 1.6mm 0;
}

figure {
  margin: 1.6mm 0 2mm;
  break-inside: auto;
}

figcaption {
  font-size: 8.4pt;
  line-height: 1.3;
  color: var(--muted);
  margin-top: 1.2mm;
}

div[align="center"] {
  margin: 1mm 0 1.6mm;
  line-height: 1.25;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 3mm 0 3.4mm;
  font-family: "Times New Roman", "Times", "Nimbus Roman", serif;
  font-size: 7.8pt;
  line-height: 1.3;
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
  padding: 1.4mm 1.7mm;
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
  font-family: "Courier New", "Courier", "Liberation Mono", monospace;
  font-size: 8.8pt;
  background: var(--code-bg);
  color: #9a2f4f;
  padding: 0.2mm 0.8mm;
  border-radius: 2px;
}

pre {
  font-family: "Courier New", "Courier", "Liberation Mono", monospace;
  font-size: 8pt;
  line-height: 1.34;
  color: #263241;
  background: var(--code-bg);
  border: 0.6pt solid #dde3ea;
  border-left: 2.6pt solid var(--accent-2);
  border-radius: 3px;
  padding: 2.4mm 3mm;
  margin: 2.8mm 0 3.2mm;
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
  font-family: "Courier New", "Courier", "Liberation Mono", monospace;
  background: #f7f7fa;
  border: 0.6pt solid #d9dbe4;
  padding: 3mm 4mm;
  margin: 4mm 0;
  white-space: pre-wrap;
  font-size: 8.3pt;
  line-height: 1.45;
}

.MathJax {
  font-size: 96% !important;
}

mjx-container,
mjx-container[jax="SVG"],
mjx-container[jax="SVG"][display="true"] {
  max-width: 100%;
  overflow: hidden;
}

mjx-container[jax="SVG"][display="true"] {
  margin: 2.2mm 0 !important;
  text-align: center !important;
}

mjx-container[jax="SVG"][display="true"] svg {
  max-width: 100%;
  height: auto;
}

.footnote,
.reference,
.references {
  font-size: 9pt;
  line-height: 1.62;
}

.file-label {
  font-family: "Times New Roman", "Times", "Nimbus Roman", serif;
  color: var(--muted);
  font-size: 8.3pt;
  margin: -4mm 0 6mm;
}

.chapter-authors {
  font-family: "Times New Roman", "Times", "Nimbus Roman", serif;
  font-size: 10.2pt;
  line-height: 1.35;
  color: #2e3b4e;
  margin: -2.2mm 0 5.4mm;
  text-align: left;
  break-after: avoid;
  page-break-after: avoid;
}

.chapter-authors::before {
  content: "Author: ";
  font-weight: 700;
}
"""


MATHJAX = r"""
<script>
window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)'], ['$', '$']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
    ignoreHtmlClass: 'tex2jax_ignore',
    processHtmlClass: 'tex2jax_process'
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
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-svg.js"></script>
"""


@dataclass
class NavItem:
    title: str
    path: str
    level: int
    group: str
    group_slug: str


def find_en_nav(config: dict[str, Any]) -> list[Any]:
    for plugin in config.get("plugins", []):
        if isinstance(plugin, dict) and "i18n" in plugin:
            for lang in plugin["i18n"].get("languages", []):
                if lang.get("locale") == "en":
                    nav = lang.get("nav")
                    if isinstance(nav, list):
                        return nav
    raise ValueError("Cannot find en navigation in mkdocs.yml")


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", value).strip("-")
    return slug[:48] or "section"


def flatten_nav(nodes: list[Any], level: int = 1, group: str = "Front Matter", group_slug: str = "front") -> list[NavItem]:
    items: list[NavItem] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        for title, value in node.items():
            if isinstance(value, str) and value.endswith(".md"):
                if value == "afterword.md":
                    item_group = "Back Matter"
                    item_group_slug = "back-matter"
                elif level == 1 and not re.search(r"part\d+/", value) and not value.startswith("appendix_"):
                    item_group = "Front Matter"
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
        src = DOCS_EN / item.path
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
        html_body = transform_section_opening(html_body, item.path)
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
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Engineering for Large Foundation Models: A Handbook - 16K PDF{html.escape(title_suffix)}</title>
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


def transform_section_opening(html_body: str, source_path: str) -> str:
    """Use a Springer-like title block in the PDF while keeping source headings stable."""

    if not re.search(r"(?:part\d+/ch\d+_|part14/p\d+_|appendix_)", source_path):
        return html_body

    patterns = [
        (r"<h1>Chapter\s+\d+:\s*([^<]+)</h1>", r"<h1>\1</h1>"),
        (r"<h1>Project\s+\d+:\s*([^<]+)</h1>", r"<h1>\1</h1>"),
        (r"<h1>Appendix\s+[A-G]:\s*([^<]+)</h1>", r"<h1>\1</h1>"),
    ]
    for pattern, replacement in patterns:
        html_body, count = re.subn(pattern, replacement, html_body, count=1)
        if count:
            return html_body
    return html_body


def build_cover(stats: dict[str, int]) -> str:
    return f"""
<section class="cover">
  <div class="kicker">Springer English Manuscript Export | 16K Compact Layout</div>
  <h1>Data Engineering for Large Foundation Models</h1>
  <div class="subtitle">A Handbook</div>
  <div class="meta">
    Page size: 16K, 185mm x 260mm<br>
    Layout: compact figure/code flow, reduced blank space before image-heavy pages<br>
    Scope: English navigation manuscript, {stats.get("files", 0)} files<br>
    Generator: scripts/export_en_book_pdf.py
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
  <h1>Table of Contents</h1>
  <div class="front-note">This table of contents is generated from the English mkdocs.yml navigation for layout review; final page numbers should be verified in the publisher production system.</div>
  <ol>
    {''.join(lis)}
  </ol>
</section>
"""


def generated_title_page_html() -> str:
    return f"""
<section class="source-file front-section generated-front-section">
  <h1>Title Page</h1>
  <p><strong>Data Engineering for Large Foundation Models</strong></p>
  <p><em>A Handbook</em></p>
  <p>{html.escape(BOOK_AUTHORS)}</p>
  <p>Springer manuscript review PDF.</p>
</section>
"""


def generated_front_matter_html(*, include_toc: bool = True) -> str:
    toc = """
<section class="source-file front-section generated-front-section">
  <h1>Contents</h1>
  <p>The final table of contents and page numbers should be verified in the publisher production system.</p>
</section>
""" if include_toc else ""
    return (
        generated_title_page_html()
        + """
<section class="source-file front-section generated-front-section"><h1>Preface</h1></section>
<section class="source-file front-section generated-front-section"><h1>Acknowledgments</h1></section>
"""
        + toc
        + """
<section class="source-file front-section generated-front-section"><h1>Contributors</h1></section>
<section class="source-file front-section generated-front-section"><h1>Abbreviations</h1></section>
"""
    )


def prepare_pdf_items(items: list[NavItem]) -> list[NavItem]:
    """Return the formal PDF sequence, excluding web-only front-matter pages."""

    filtered = [item for item in items if item.path not in EXCLUDED_FROM_FORMAL_PDF]
    front_order = {
        "preface.md": 0,
        "acknowledgments.md": 1,
        "front_matter_guide.md": 2,
        "contributors.md": 3,
        "abbreviations.md": 4,
    }
    front = [item for item in filtered if item.group_slug == "front-matter"]
    rest = [item for item in filtered if item.group_slug != "front-matter"]
    front.sort(key=lambda item: (front_order.get(item.path, 99), item.path))
    return front + rest


def is_submission_pdf_item(item: NavItem) -> bool:
    path = item.path
    return bool(
        re.match(r"part\d+/ch\d+_", path)
        or re.match(r"part14/p\d+_", path)
        or path.startswith("appendix_")
        or path == "afterword.md"
    )


def submission_pdf_items(items: list[NavItem]) -> list[NavItem]:
    return [item for item in prepare_pdf_items(items) if is_submission_pdf_item(item)]


def front_matter_pdf_items(items: list[NavItem]) -> list[NavItem]:
    paths = PRE_CONTENTS_FRONT_PATHS | POST_CONTENTS_FRONT_PATHS
    return [item for item in prepare_pdf_items(items) if item.path in paths]


def back_matter_pdf_items(items: list[NavItem]) -> list[NavItem]:
    return [item for item in prepare_pdf_items(items) if item.path == "afterword.md"]


def to_roman(number: int) -> str:
    values = [
        (1000, "m"),
        (900, "cm"),
        (500, "d"),
        (400, "cd"),
        (100, "c"),
        (90, "xc"),
        (50, "l"),
        (40, "xl"),
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    ]
    result: list[str] = []
    remaining = number
    for value, numeral in values:
        while remaining >= value:
            result.append(numeral)
            remaining -= value
    return "".join(result)


def build_page_number_label(page_number: int, first_body_page: int) -> str:
    if page_number < first_body_page:
        return to_roman(page_number)
    return str(page_number - first_body_page + 1)


def generate_opening_front_pdf(path: Path, stats: dict[str, int]) -> int:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import mm
        from reportlab.pdfgen import canvas
    except Exception as exc:  # pragma: no cover - dependency check
        raise RuntimeError("reportlab is required to generate formal front matter") from exc

    regular_font, bold_font = register_pdf_fonts()
    path.parent.mkdir(parents=True, exist_ok=True)
    page_width = 185 * mm
    page_height = 260 * mm
    left = 18 * mm
    right = 18 * mm
    top = 22 * mm
    bottom = 20 * mm
    usable_width = page_width - left - right

    c = canvas.Canvas(str(path), pagesize=(page_width, page_height), initialFontName=regular_font)
    page_count = 0

    def draw_wrapped(text: str, x: float, y: float, font: str, size: float, max_width: float, leading: float) -> float:
        c.setFont(font, size)
        line = ""
        for word in text.split():
            candidate = (line + " " + word).strip()
            if c.stringWidth(candidate, font, size) > max_width and line:
                c.drawString(x, y, line)
                y -= leading
                line = word
            else:
                line = candidate
        if line:
            c.drawString(x, y, line)
            y -= leading
        return y

    def footer(label: str | None = None) -> None:
        nonlocal page_count
        page_count += 1
        if label:
            c.setFont(regular_font, 9)
            c.setFillColor(colors.HexColor("#5f6876"))
            c.drawCentredString(page_width / 2, 10 * mm, label)
        c.showPage()

    # Title page.
    c.setFillColor(colors.HexColor("#182336"))
    c.setFont(bold_font, 24)
    y = page_height - top - 42 * mm
    c.drawString(left, y, "Data Engineering for")
    y -= 10 * mm
    c.drawString(left, y, "Large Foundation Models")
    y -= 13 * mm
    c.setFont(regular_font, 15)
    c.setFillColor(colors.HexColor("#245a8d"))
    c.drawString(left, y, "A Handbook")
    y -= 20 * mm
    c.setFillColor(colors.HexColor("#333333"))
    c.setFont(bold_font, 11)
    c.drawString(left, y, "Authors")
    y -= 6 * mm
    y = draw_wrapped(BOOK_AUTHORS, left, y, regular_font, 10, usable_width, 5 * mm)
    y -= 12 * mm
    c.setFont(regular_font, 9.5)
    c.setFillColor(colors.HexColor("#5f6876"))
    draw_wrapped("Springer manuscript review PDF", left, y, regular_font, 9.5, usable_width, 4.8 * mm)
    footer(None)

    c.save()
    return page_count


def generate_contents_pdf(path: Path, toc_entries: list[tuple[str, int, str]], start_page_number: int) -> int:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import mm
        from reportlab.pdfgen import canvas
    except Exception as exc:  # pragma: no cover - dependency check
        raise RuntimeError("reportlab is required to generate formal contents") from exc

    regular_font, bold_font = register_pdf_fonts()
    path.parent.mkdir(parents=True, exist_ok=True)
    page_width = 185 * mm
    page_height = 260 * mm
    left = 18 * mm
    right = 18 * mm
    top = 22 * mm
    bottom = 20 * mm
    usable_width = page_width - left - right
    c = canvas.Canvas(str(path), pagesize=(page_width, page_height), initialFontName=regular_font)
    page_count = 0

    def footer(label: str | None = None) -> None:
        nonlocal page_count
        page_count += 1
        if label:
            c.setFont(regular_font, 9)
            c.setFillColor(colors.HexColor("#5f6876"))
            c.drawCentredString(page_width / 2, 10 * mm, label)
        c.showPage()

    def write_contents_page_header(page_no: int) -> float:
        c.setFillColor(colors.HexColor("#182336"))
        c.setFont(bold_font, 17)
        c.drawString(left, page_height - top, "Contents")
        c.setStrokeColor(colors.HexColor("#d9dee7"))
        c.setLineWidth(0.7)
        c.line(left, page_height - top - 4 * mm, page_width - right, page_height - top - 4 * mm)
        c.setFont(regular_font, 9)
        c.setFillColor(colors.HexColor("#5f6876"))
        c.drawRightString(page_width - right, page_height - top, f"Page {page_no}")
        return page_height - top - 12 * mm

    contents_page_no = start_page_number
    y = write_contents_page_header(contents_page_no)
    min_y = bottom + 7 * mm
    for title, level, page_label in toc_entries:
        if y < min_y:
            footer(to_roman(contents_page_no))
            contents_page_no += 1
            y = write_contents_page_header(contents_page_no)
        indent = max(0, level - 1) * 5 * mm
        font = bold_font if level == 1 else regular_font
        size = 9.4 if level <= 2 else 8.7
        c.setFont(font, size)
        c.setFillColor(colors.HexColor("#182336") if level <= 2 else colors.HexColor("#333333"))
        max_title_width = usable_width - indent - 19 * mm
        text = title
        if c.stringWidth(text, font, size) > max_title_width and len(text) > 18:
            available = max(1, len(text) - 3)
            while available > 15:
                candidate = text[:available].rstrip() + "..."
                if c.stringWidth(candidate, font, size) <= max_title_width:
                    text = candidate
                    break
                available -= 1
            else:
                text = text[:15].rstrip() + "..."
        x = left + indent
        c.drawString(x, y, text)
        page_x = page_width - right
        c.drawRightString(page_x, y, page_label)
        dots_start = x + c.stringWidth(text, font, size) + 2 * mm
        dots_end = page_x - c.stringWidth(page_label, font, size) - 2 * mm
        if dots_end > dots_start:
            c.setStrokeColor(colors.HexColor("#c8ced8"))
            c.setDash(1, 2)
            c.line(dots_start, y + 1.2, dots_end, y + 1.2)
            c.setDash()
        y -= 5.0 * mm if level <= 2 else 4.4 * mm
    footer(to_roman(contents_page_no))
    c.save()
    return page_count


def generate_book_front_pdf(
    path: Path,
    toc_entries: list[tuple[str, int, str]],
    stats: dict[str, int],
) -> int:
    """Backward-compatible front matter generator for non-Springer split flows."""

    opening_pages = generate_opening_front_pdf(path, stats)
    contents_pages = generate_contents_pdf(CONTENTS_PDF, toc_entries, opening_pages + 1)
    try:
        from pypdf import PdfWriter
    except Exception as exc:  # pragma: no cover - dependency check
        raise RuntimeError("pypdf is required to merge generated front matter") from exc

    writer = PdfWriter()
    writer.append(str(path))
    writer.append(str(CONTENTS_PDF))
    with path.open("wb") as handle:
        writer.write(handle)
    return opening_pages + contents_pages


def write_html(path: Path, html_doc: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding="utf-8")
    print(f"[ok] HTML written: {path}")


def export_pdf(html_path: Path, pdf_path: Path, timeout: int, *, min_size: int = 100_000) -> None:
    if not CHROME.exists():
        raise FileNotFoundError(f"Google Chrome not found: {CHROME}")

    cmd = [
        str(CHROME),
        "--headless=new",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=5000",
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
    if not pdf_path.exists() or pdf_path.stat().st_size < min_size:
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


def formal_groups(items: list[NavItem]) -> list[tuple[str, str, list[NavItem]]]:
    pre_contents = [
        item for item in items if item.group_slug == "front-matter" and item.path in PRE_CONTENTS_FRONT_PATHS
    ]
    post_contents = [
        item for item in items if item.group_slug == "front-matter" and item.path in POST_CONTENTS_FRONT_PATHS
    ]
    body_items = [item for item in items if item.group_slug != "front-matter"]

    groups: list[tuple[str, str, list[NavItem]]] = []
    if pre_contents:
        groups.append(("front-matter-before-contents", "Front Matter Before Contents", pre_contents))
    if post_contents:
        groups.append(("front-matter-after-contents", "Front Matter After Contents", post_contents))
    groups.extend(group_items(body_items))
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


def merge_formal_book_pdf(
    opening_pdf: Path,
    contents_pdf: Path,
    parts: list[Path],
    output: Path,
    groups: list[tuple[str, str, list[NavItem]]],
    *,
    first_body_page: int,
    contents_after_group: int,
) -> None:
    try:
        from pypdf import PdfReader, PdfWriter
    except Exception as exc:  # pragma: no cover - dependency check
        raise RuntimeError("pypdf is required to merge the formal book PDF") from exc

    writer = PdfWriter()
    writer.append(str(opening_pdf))
    offsets: list[int] = []
    page_offset = len(PdfReader(str(opening_pdf)).pages)
    for index, part in enumerate(parts):
        offsets.append(page_offset)
        page_offset += len(PdfReader(str(part)).pages)
        writer.append(str(part))
        if index == contents_after_group:
            writer.append(str(contents_pdf))
            page_offset += len(PdfReader(str(contents_pdf)).pages)

    add_formal_bookmarks(
        writer,
        opening_pdf,
        contents_pdf,
        parts,
        offsets,
        groups,
        contents_after_group=contents_after_group,
    )
    try:
        writer.page_mode = "/UseOutlines"
    except Exception:
        pass

    numbered = add_page_number_overlay(writer, first_body_page=first_body_page)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        numbered.write(handle)
    print(f"[ok] merged formal PDF written: {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")


def pdf_text_cache_key(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


def add_formal_bookmarks(
    writer: Any,
    opening_pdf: Path,
    contents_pdf: Path,
    parts: list[Path],
    offsets: list[int],
    groups: list[tuple[str, str, list[NavItem]]],
    *,
    contents_after_group: int,
) -> None:
    opening_pages = len(__import__("pypdf").PdfReader(str(opening_pdf)).pages)
    contents_pages = len(__import__("pypdf").PdfReader(str(contents_pdf)).pages)
    writer.add_outline_item("Title Page", 0)
    contents_offset = offsets[contents_after_group] + len(__import__("pypdf").PdfReader(str(parts[contents_after_group])).pages)
    writer.add_outline_item("Contents", contents_offset)
    if len(groups) != len(parts):
        print("[WARN] bookmark group count does not match part PDFs", file=sys.stderr)
        return
    for (_, title, grouped), part, offset in zip(groups, parts, offsets):
        part_reader = __import__("pypdf").PdfReader(str(part))
        parent = writer.add_outline_item(title, offset)
        local_pages = locate_item_pages(
            part_reader,
            grouped,
            start_after_toc=False,
            cache_key=pdf_text_cache_key(part),
        )
        for item in grouped:
            local_page = local_pages.get(item.path, 0)
            writer.add_outline_item(item.title, offset + local_page, parent=parent)
    if opening_pages < 1 or contents_pages < 1:
        print("[WARN] generated front matter has fewer pages than expected", file=sys.stderr)


def add_page_number_overlay(writer: Any, *, first_body_page: int) -> Any:
    try:
        from io import BytesIO

        from pypdf import PdfReader, PdfWriter
        from reportlab.lib import colors
        from reportlab.pdfgen import canvas
    except Exception as exc:  # pragma: no cover - dependency check
        raise RuntimeError("pypdf and reportlab are required to add page numbers") from exc

    regular_font, _ = register_pdf_fonts()
    numbered = PdfWriter()
    total_pages = len(writer.pages)
    for index, page in enumerate(writer.pages, 1):
        if index == 1:
            numbered.add_page(page)
            continue
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        packet = BytesIO()
        c = canvas.Canvas(packet, pagesize=(width, height), initialFontName=regular_font)
        c.setFont(regular_font, 8.5)
        c.setFillColor(colors.HexColor("#5f6876"))
        label = build_page_number_label(index, first_body_page)
        c.drawCentredString(width / 2, 18, label)
        if index >= first_body_page:
            c.setFont(regular_font, 7.8)
            c.drawString(42, 18, "Data Engineering for Large Foundation Models")
            c.drawRightString(width - 42, 18, "A Handbook")
        c.save()
        packet.seek(0)
        overlay = PdfReader(packet).pages[0]
        page.merge_page(overlay)
        numbered.add_page(page)
    if len(numbered.pages) != total_pages:
        raise RuntimeError("page-number overlay changed the page count")
    return numbered


def add_bookmarks(writer: Any, parts: list[Path], offsets: list[int], items: list[NavItem]) -> None:
    groups = group_items(items)
    if len(groups) != len(parts):
        print("[WARN] bookmark group count does not match part PDFs", file=sys.stderr)
        return
    writer.add_outline_item("Cover", 0)
    writer.add_outline_item("Table of Contents", 1)
    for index, ((_, title, grouped), part, offset) in enumerate(zip(groups, parts, offsets), 1):
        part_reader = None
        try:
            part_reader = __import__("pypdf").PdfReader(str(part))
        except Exception:
            part_reader = None
        parent = writer.add_outline_item(title, offset)
        local_pages = (
            locate_item_pages(
                part_reader,
                grouped,
                start_after_toc=index == 1,
                cache_key=pdf_text_cache_key(part),
            )
            if part_reader
            else {}
        )
        for item in grouped:
            local_page = local_pages.get(item.path, 0)
            writer.add_outline_item(item.title, offset + local_page, parent=parent)


def locate_item_pages(
    reader: Any,
    items: list[NavItem],
    start_after_toc: bool = False,
    cache_key: str | None = None,
) -> dict[str, int]:
    def normalize(text: str) -> str:
        import unicodedata

        text = unicodedata.normalize("NFKC", text or "")
        return re.sub(r"[\W_]+", "", text)

    def is_part_overview(item: NavItem) -> bool:
        return bool(re.search(r"part\d+/index\.md$", item.path))

    def title_needles(title: str) -> list[str]:
        candidates = [title]
        stripped = re.sub(r"^(?:Chapter|Project)\s+\d+\s*:\s*", "", title)
        stripped = re.sub(r"^Appendix\s+[A-Z]\s*:\s*", "", stripped)
        if stripped != title:
            candidates.append(stripped)
        needles: list[str] = []
        for candidate in candidates:
            needle = normalize(candidate)
            if needle and needle not in needles:
                needles.append(needle)
        return needles

    result: dict[str, int] = {}
    if reader is None:
        return result
    pages = list(reader.pages)
    page_count = len(pages)
    reader_name = cache_key or str(getattr(getattr(reader, "stream", None), "name", "") or "")
    if reader_name:
        text_cache = PDF_TEXT_CACHE.setdefault(reader_name, {})
    else:
        text_cache = getattr(reader, "_data_engineering_text_cache", {})
        try:
            setattr(reader, "_data_engineering_text_cache", text_cache)
        except Exception:
            pass

    def page_text(index: int) -> str:
        if index not in text_cache:
            text_cache[index] = normalize(pages[index].extract_text() or "")
        return text_cache[index]

    current = 3 if start_after_toc and page_count > 3 else 0
    for item in items:
        if is_part_overview(item):
            result[item.path] = min(current, max(0, page_count - 1))
            current = min(current + 1, max(0, page_count - 1))
            continue
        needles = title_needles(item.title)
        found = None
        for idx in range(current, page_count):
            text = page_text(idx)
            if any(needle in text for needle in needles):
                found = idx
                break
        if found is None:
            found = min(current, max(0, page_count - 1))
        result[item.path] = found
        current = min(found + 1, max(0, page_count - 1))
    return result


def compute_toc_entries(
    items: list[NavItem],
    parts: list[Path],
    *,
    front_pages: int,
    first_body_page: int,
) -> list[tuple[str, int, str]]:
    from pypdf import PdfReader

    entries: list[tuple[str, int, str]] = []
    groups = group_items(items)
    content_offset = front_pages
    for (_, title, grouped), part in zip(groups, parts):
        reader = PdfReader(str(part))
        entries.append((title, 1, build_page_number_label(content_offset + 1, first_body_page)))
        local_pages = locate_item_pages(
            reader,
            grouped,
            start_after_toc=False,
            cache_key=pdf_text_cache_key(part),
        )
        for item in grouped:
            local_page = local_pages.get(item.path, 0)
            absolute_page = content_offset + local_page + 1
            entries.append((item.title, item.level, build_page_number_label(absolute_page, first_body_page)))
        content_offset += len(reader.pages)
    return entries


def compute_formal_toc_entries(
    groups: list[tuple[str, str, list[NavItem]]],
    parts: list[Path],
    *,
    opening_pages: int,
    contents_pages: int,
    first_body_page: int,
    contents_after_group: int,
) -> list[tuple[str, int, str]]:
    from pypdf import PdfReader

    entries: list[tuple[str, int, str]] = []
    content_offset = opening_pages
    for index, ((_, title, grouped), part) in enumerate(zip(groups, parts)):
        reader = PdfReader(str(part))
        is_artificial_front_group = title in {"Front Matter Before Contents", "Front Matter After Contents"}
        if not is_artificial_front_group:
            entries.append((title, 1, build_page_number_label(content_offset + 1, first_body_page)))
        local_pages = locate_item_pages(
            reader,
            grouped,
            start_after_toc=False,
            cache_key=pdf_text_cache_key(part),
        )
        for item in grouped:
            local_page = local_pages.get(item.path, 0)
            absolute_page = content_offset + local_page + 1
            entry_level = item.level if not is_artificial_front_group else 1
            entries.append((item.title, entry_level, build_page_number_label(absolute_page, first_body_page)))
        content_offset += len(reader.pages)
        if index == contents_after_group:
            content_offset += contents_pages
    return entries


def first_body_page_number(
    groups: list[tuple[str, str, list[NavItem]]],
    parts: list[Path],
    *,
    opening_pages: int,
    contents_pages: int,
    contents_after_group: int,
) -> int:
    from pypdf import PdfReader

    page_offset = opening_pages
    for index, ((slug, _, _), part) in enumerate(zip(groups, parts)):
        if slug not in {"front-matter-before-contents", "front-matter-after-contents"}:
            return page_offset + 1
        page_offset += len(PdfReader(str(part)).pages)
        if index == contents_after_group:
            page_offset += contents_pages
    return page_offset + 1


def export_split_pdf(items: list[NavItem], timeout: int, include_mathjax: bool, global_stats: dict[str, int]) -> None:
    if PARTS_DIR.exists():
        shutil.rmtree(PARTS_DIR)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    part_pdfs: list[Path] = []
    groups = formal_groups(items)
    width = len(str(len(groups)))
    for index, (slug, title, grouped) in enumerate(groups, 1):
        prefix = f"{index:0{width}d}-{slug}"
        html_path = PARTS_DIR / f"{prefix}.html"
        pdf_path = PARTS_DIR / f"{prefix}.pdf"
        html_doc, stats = build_book_html(
            grouped,
            title_suffix=f" - {title}",
            include_mathjax=include_mathjax,
            include_cover_toc=False,
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
        export_pdf(html_path, pdf_path, timeout, min_size=10_000)
        part_pdfs.append(pdf_path)

    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - dependency check
        raise RuntimeError("pypdf is required to compute page-numbered contents") from exc

    opening_pages = generate_opening_front_pdf(OPENING_FRONT_PDF, global_stats)
    contents_after_group = 0
    contents_pages = 1
    for _ in range(4):
        first_body_page = first_body_page_number(
            groups,
            part_pdfs,
            opening_pages=opening_pages,
            contents_pages=contents_pages,
            contents_after_group=contents_after_group,
        )
        toc_entries = compute_formal_toc_entries(
            groups,
            part_pdfs,
            opening_pages=opening_pages,
            contents_pages=contents_pages,
            first_body_page=first_body_page,
            contents_after_group=contents_after_group,
        )
        actual_contents_pages = generate_contents_pdf(
            CONTENTS_PDF,
            toc_entries,
            start_page_number=opening_pages + len(PdfReader(str(part_pdfs[0])).pages) + 1,
        )
        if actual_contents_pages == contents_pages:
            break
        contents_pages = actual_contents_pages
    else:
        print("[WARN] contents page count did not stabilize after TOC generation", file=sys.stderr)

    first_body_page = first_body_page_number(
        groups,
        part_pdfs,
        opening_pages=opening_pages,
        contents_pages=contents_pages,
        contents_after_group=contents_after_group,
    )
    toc_entries = compute_formal_toc_entries(
        groups,
        part_pdfs,
        opening_pages=opening_pages,
        contents_pages=contents_pages,
        first_body_page=first_body_page,
        contents_after_group=contents_after_group,
    )
    generate_contents_pdf(
        CONTENTS_PDF,
        toc_entries,
        start_page_number=opening_pages + len(PdfReader(str(part_pdfs[0])).pages) + 1,
    )
    merge_formal_book_pdf(
        OPENING_FRONT_PDF,
        CONTENTS_PDF,
        part_pdfs,
        OUT_PDF,
        groups,
        first_body_page=first_body_page,
        contents_after_group=contents_after_group,
    )


def extract_main_html(html_doc: str) -> str:
    match = re.search(r"<main>\s*(.*?)\s*</main>", html_doc, flags=re.S)
    return match.group(1) if match else ""


def build_reference_pdf_html(
    body_html: str,
    *,
    title_suffix: str,
    include_mathjax: bool,
) -> str:
    mathjax = MATHJAX if include_mathjax else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Engineering for Large Foundation Models: A Handbook - {html.escape(title_suffix)}</title>
<style>{CSS}</style>
{mathjax}
</head>
<body>
<main>
{body_html}
</main>
</body>
</html>
"""


def build_front_matter_reference_html(items: list[NavItem], include_mathjax: bool) -> str:
    prepared = prepare_pdf_items(items)
    pre_items = [item for item in prepared if item.path in PRE_CONTENTS_FRONT_PATHS]
    post_items = [item for item in prepared if item.path in POST_CONTENTS_FRONT_PATHS]
    pre_doc, _ = build_book_html(pre_items, include_mathjax=include_mathjax, include_cover_toc=False)
    post_doc, _ = build_book_html(post_items, include_mathjax=include_mathjax, include_cover_toc=False)
    body = (
        generated_title_page_html()
        + extract_main_html(pre_doc)
        + """
<section class="source-file front-section generated-front-section">
  <h1>Contents</h1>
  <p>The final table of contents and page numbers should be verified in the publisher production system.</p>
</section>
"""
        + extract_main_html(post_doc)
    )
    return build_reference_pdf_html(body, title_suffix="Front Matter", include_mathjax=include_mathjax)


def export_submission_pdfs(items: list[NavItem], timeout: int, include_mathjax: bool) -> None:
    """Export a Springer-style PDF folder: full book plus one PDF per manuscript unit."""

    if SUBMISSION_PDF_DIR.exists():
        shutil.rmtree(SUBMISSION_PDF_DIR)
    SUBMISSION_PDF_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_PDF.exists():
        shutil.copy2(OUT_PDF, SUBMISSION_PDF_DIR / "00_full_book_pagenumbered.pdf")
    else:
        print("[WARN] full book PDF does not exist yet; submission folder will contain contribution PDFs only")

    manifest_lines = [
        "# English Springer PDF Submission Folder",
        "",
        "This folder contains the full paginated book PDF plus reference PDFs for front matter, chapters, project chapters, appendices, and back matter.",
        "Springer Nature's manuscript guidelines ask for a single ZIP that includes source files and a PDF set with chapter PDFs plus front/back matter PDFs when applicable.",
        "",
        "| No. | Title | Source | PDF |",
        "| --- | --- | --- | --- |",
    ]
    front_html_path = SUBMISSION_PDF_DIR / "00_front_matter.html"
    front_pdf_path = SUBMISSION_PDF_DIR / "00_front_matter.pdf"
    write_html(front_html_path, build_front_matter_reference_html(items, include_mathjax))
    export_pdf(front_html_path, front_pdf_path, timeout, min_size=10_000)
    manifest_lines.append(
        f"| Front | Front matter | generated title page plus `{', '.join(item.path for item in front_matter_pdf_items(items))}` | `{front_pdf_path.name}` |"
    )

    manuscript_items = submission_pdf_items(items)
    width = len(str(len(manuscript_items)))
    for index, item in enumerate(manuscript_items, 1):
        slug = slugify(Path(item.path).with_suffix("").as_posix().replace("/", "-"))
        prefix = f"{index:0{width}d}-{slug}"
        html_path = SUBMISSION_PDF_DIR / f"{prefix}.html"
        pdf_path = SUBMISSION_PDF_DIR / f"{prefix}.pdf"
        html_doc, stats = build_book_html(
            [item],
            title_suffix=f" - {item.title}",
            include_mathjax=include_mathjax,
            include_cover_toc=False,
        )
        write_html(html_path, html_doc)
        print(
            "[stats] submission "
            + item.title
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
        export_pdf(html_path, pdf_path, timeout, min_size=10_000)
        safe_title = item.title.replace("|", "\\|")
        manifest_lines.append(
            f"| {index} | {safe_title} | `{item.path}` | `{pdf_path.name}` |"
        )

    back_items = back_matter_pdf_items(items)
    if back_items:
        back_html_path = SUBMISSION_PDF_DIR / "99_back_matter.html"
        back_pdf_path = SUBMISSION_PDF_DIR / "99_back_matter.pdf"
        back_doc, _ = build_book_html(
            back_items,
            title_suffix=" - Back Matter",
            include_mathjax=include_mathjax,
            include_cover_toc=False,
        )
        write_html(back_html_path, back_doc)
        export_pdf(back_html_path, back_pdf_path, timeout, min_size=10_000)
        manifest_lines.append(
            f"| Back | Back matter | `{', '.join(item.path for item in back_items)}` | `{back_pdf_path.name}` |"
        )
    manifest = SUBMISSION_PDF_DIR / "README.md"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"[ok] submission PDF folder written: {SUBMISSION_PDF_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export en manuscript to 16K compact PDF")
    parser.add_argument("--no-pdf", action="store_true", help="Only write the intermediate HTML")
    parser.add_argument("--split", action="store_true", help="Export by book part and merge PDFs")
    parser.add_argument(
        "--submission-pdfs",
        action="store_true",
        help="Also export a Springer-style folder with the full PDF and one PDF per navigation unit",
    )
    parser.add_argument(
        "--submission-pdfs-only",
        action="store_true",
        help="Only export the Springer-style submission PDF folder from the current sources",
    )
    parser.add_argument("--no-mathjax", action="store_true", help="Disable MathJax formula rendering")
    parser.add_argument("--timeout", type=int, default=1200, help="Chrome export timeout in seconds")
    args = parser.parse_args()

    config = yaml.safe_load(MKDOCS.read_text(encoding="utf-8"))
    items = prepare_pdf_items(flatten_nav(find_en_nav(config)))
    include_mathjax = not args.no_mathjax
    html_doc, stats = build_book_html(items, include_mathjax=include_mathjax)
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
    if args.submission_pdfs_only:
        export_submission_pdfs(items, args.timeout, include_mathjax)
    elif not args.no_pdf:
        if args.split:
            export_split_pdf(items, args.timeout, include_mathjax, stats)
        else:
            export_pdf(OUT_HTML, OUT_PDF, args.timeout)
        if args.submission_pdfs:
            export_submission_pdfs(items, args.timeout, include_mathjax)
    return 0


if __name__ == "__main__":
    sys.exit(main())
