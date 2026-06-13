"""Export the English manuscript to a 16K LaTeX source package.

The exporter reads the English navigation from mkdocs.yml, concatenates the
book-order Markdown files, and writes a print-oriented LaTeX manuscript. It can
also invoke Tectonic or XeLaTeX to build a PDF. It is intentionally independent
from Pandoc so the repository can produce Springer-reviewable source files from
the same navigation used by the English PDF export.

Usage:
    uv run python scripts/export_en_book_latex.py
    uv run python scripts/export_en_book_latex.py --compile
    uv run python scripts/export_en_book_latex.py --limit 3
    uv run python scripts/export_en_book_latex.py --split --compile

Outputs:
    output/pdf/data_engineering_book_en_16k_latex.tex
    output/pdf/data_engineering_book_en_16k_latex.pdf
    output/pdf/data_engineering_book_en_16k_latex_warnings.txt
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[1]
MKDOCS = ROOT / "mkdocs.yml"
DOCS_EN = ROOT / "docs" / "en"
OUT_DIR = ROOT / "output" / "pdf"
OUT_TEX = OUT_DIR / "data_engineering_book_en_16k_latex.tex"
OUT_PDF = OUT_DIR / "data_engineering_book_en_16k_latex.pdf"
OUT_WARNINGS = OUT_DIR / "data_engineering_book_en_16k_latex_warnings.txt"
ASSET_DIR = OUT_DIR / "latex_assets_en"
PARTS_DIR = OUT_DIR / "data_engineering_book_en_16k_latex_parts"
CHAPTERS_DIR = OUT_DIR / "data_engineering_book_en_16k_latex_chapters"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
UNSUPPORTED_IMAGE_SUFFIXES = {".gif", ".webp"}
PIL_FORMAT_SUFFIXES = {
    "JPEG": ".jpg",
    "PNG": ".png",
}


@dataclass
class NavItem:
    title: str
    path: str
    level: int


@dataclass
class ExportStats:
    files: int = 0
    missing: int = 0
    images: int = 0
    unsupported_images: int = 0
    code_blocks: int = 0
    tables: int = 0
    warnings: list[str] = field(default_factory=list)


class AssetManager:
    def __init__(self, asset_dir: Path, stats: ExportStats) -> None:
        self.asset_dir = asset_dir
        self.stats = stats
        self._seen: dict[Path, str] = {}
        self._counter = 0

    def reset(self) -> None:
        if self.asset_dir.exists():
            shutil.rmtree(self.asset_dir)
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self._seen.clear()
        self._counter = 0

    def register(self, image_path: Path, source_file: Path) -> str | None:
        self.stats.images += 1
        image_path = image_path.resolve()
        suffix = image_path.suffix.lower()
        if suffix in UNSUPPORTED_IMAGE_SUFFIXES:
            self.stats.unsupported_images += 1
            self.stats.warnings.append(
                f"unsupported image format skipped: {source_file.relative_to(ROOT)} -> {image_path}"
            )
            return None
        if suffix not in SUPPORTED_IMAGE_SUFFIXES:
            self.stats.unsupported_images += 1
            self.stats.warnings.append(
                f"unknown image format skipped: {source_file.relative_to(ROOT)} -> {image_path}"
            )
            return None
        if not image_path.exists():
            self.stats.unsupported_images += 1
            self.stats.warnings.append(
                f"missing image skipped: {source_file.relative_to(ROOT)} -> {image_path}"
            )
            return None
        if suffix == ".svg":
            if image_path in self._seen:
                return self._seen[image_path]
            self._counter += 1
            target_name = f"asset_{self._counter:04d}.png"
            target = self.asset_dir / target_name
            try:
                convert_svg_to_png(image_path, target)
            except Exception as exc:
                self.stats.unsupported_images += 1
                self.stats.warnings.append(
                    f"svg conversion failed: {source_file.relative_to(ROOT)} -> {image_path} ({exc})"
                )
                return None
            rel = f"{self.asset_dir.name}/{target_name}"
            self._seen[image_path] = rel
            return rel
        actual_suffix = detect_image_suffix(image_path)
        if actual_suffix is None:
            self.stats.unsupported_images += 1
            self.stats.warnings.append(
                f"invalid image skipped: {source_file.relative_to(ROOT)} -> {image_path}"
            )
            return None
        suffix = actual_suffix
        if image_path in self._seen:
            return self._seen[image_path]
        self._counter += 1
        target_name = f"asset_{self._counter:04d}{suffix}"
        target = self.asset_dir / target_name
        shutil.copy2(image_path, target)
        rel = f"{self.asset_dir.name}/{target_name}"
        self._seen[image_path] = rel
        return rel


def detect_image_suffix(image_path: Path) -> str | None:
    if image_path.suffix.lower() == ".pdf":
        return ".pdf"
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            image.verify()
            return PIL_FORMAT_SUFFIXES.get(image.format)
    except Exception:
        return None


def svg_dimensions(svg_path: Path) -> tuple[int, int]:
    root = ET.parse(svg_path).getroot()

    def parse_length(value: str | None) -> float | None:
        if not value:
            return None
        match = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", value)
        return float(match.group(1)) if match else None

    width = parse_length(root.get("width"))
    height = parse_length(root.get("height"))
    if width is None or height is None:
        view_box = root.get("viewBox")
        if view_box:
            parts = [float(part) for part in re.split(r"[\s,]+", view_box.strip()) if part]
            if len(parts) == 4:
                width = width or parts[2]
                height = height or parts[3]
    return max(1, round(width or 1200)), max(1, round(height or 800))


def convert_svg_to_png(svg_path: Path, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = svg_dimensions(svg_path)
    sips = shutil.which("sips")
    if sips:
        proc = subprocess.run(
            [sips, "-s", "format", "png", str(svg_path), "--out", str(png_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0 and detect_image_suffix(png_path) == ".png":
            return
    if CHROME.exists():
        try:
            with tempfile.TemporaryDirectory() as profile:
                cmd = [
                    str(CHROME),
                    "--headless=new",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--no-first-run",
                    "--no-default-browser-check",
                    f"--user-data-dir={profile}",
                    f"--screenshot={png_path}",
                    f"--window-size={width},{height}",
                    svg_path.resolve().as_uri(),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode == 0 and detect_image_suffix(png_path) == ".png":
                return
        except subprocess.TimeoutExpired:
            pass
    render_simple_svg_to_png(svg_path, png_path, width, height)
    if detect_image_suffix(png_path) != ".png":
        raise RuntimeError("SVG conversion did not produce a valid PNG")


def render_simple_svg_to_png(svg_path: Path, png_path: Path, width: int, height: int) -> None:
    try:
        from PIL import Image, ImageColor, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("Pillow is required for SVG fallback rendering") from exc

    def color(value: str | None, default: str | None = None):
        value = value or default
        if not value or value == "none":
            return None
        try:
            return ImageColor.getrgb(value)
        except ValueError:
            return ImageColor.getrgb(default or "#000000")

    def number(value: str | None, default: float = 0.0) -> float:
        if value is None:
            return default
        match = re.match(r"\s*(-?[0-9]+(?:\.[0-9]+)?)", value)
        return float(match.group(1)) if match else default

    def local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    root = ET.parse(svg_path).getroot()
    for node in root.iter():
        tag = local_name(node.tag)
        if tag == "rect":
            x = number(node.get("x"))
            y = number(node.get("y"))
            w = number(node.get("width"), width)
            h = number(node.get("height"), height)
            draw.rectangle(
                [x, y, x + w, y + h],
                fill=color(node.get("fill"), "#ffffff"),
                outline=color(node.get("stroke")),
                width=max(1, round(number(node.get("stroke-width"), 1))),
            )
        elif tag == "line":
            draw.line(
                [
                    (number(node.get("x1")), number(node.get("y1"))),
                    (number(node.get("x2")), number(node.get("y2"))),
                ],
                fill=color(node.get("stroke"), "#000000"),
                width=max(1, round(number(node.get("stroke-width"), 1))),
            )
        elif tag == "text":
            text = "".join(node.itertext()).strip()
            if text:
                font_size = max(8, round(number(node.get("font-size"), 14)))
                try:
                    font = ImageFont.truetype("Arial.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()
                draw.text((number(node.get("x")), number(node.get("y")) - font_size), text, fill=color(node.get("fill"), "#000000"), font=font)
    image.save(png_path, "PNG")


def find_en_nav(config: dict[str, Any]) -> list[Any]:
    for plugin in config.get("plugins", []):
        if isinstance(plugin, dict) and "i18n" in plugin:
            for lang in plugin["i18n"].get("languages", []):
                if lang.get("locale") == "en":
                    nav = lang.get("nav")
                    if isinstance(nav, list):
                        return nav
    raise ValueError("Cannot find en navigation in mkdocs.yml")


def flatten_nav(nodes: list[Any], level: int = 1) -> list[NavItem]:
    items: list[NavItem] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        for title, value in node.items():
            if isinstance(value, str) and value.endswith(".md"):
                items.append(NavItem(str(title), value, level))
            elif isinstance(value, list):
                items.extend(flatten_nav(value, level + 1))
    return items


def prepare_latex_items(items: list[NavItem]) -> list[NavItem]:
    """Return the formal LaTeX sequence, excluding web-only pages."""

    excluded = {"index.md", "translation-status.md"}
    filtered = [item for item in items if item.path not in excluded]
    front_order = {
        "title_page.md": 0,
        "preface.md": 1,
        "acknowledgments.md": 2,
        "front_matter_guide.md": 3,
        "contributors.md": 4,
        "abbreviations.md": 5,
    }
    front = [
        item
        for item in filtered
        if not re.search(r"part\d+/", item.path)
        and not item.path.startswith("appendix_")
        and item.path != "afterword.md"
    ]
    rest = [item for item in filtered if item not in front]
    front.sort(key=lambda item: (front_order.get(item.path, 99), item.path))
    return front + rest


def part_key(item: NavItem) -> str:
    match = re.search(r"(part\d+)/", item.path)
    if match:
        return match.group(1)
    if item.path.startswith("appendix_"):
        return "appendix"
    if item.path == "afterword.md":
        return "back-matter"
    return "front-back"


def group_items(items: list[NavItem]) -> list[tuple[str, list[NavItem]]]:
    groups: list[tuple[str, list[NavItem]]] = []
    current_key: str | None = None
    current_items: list[NavItem] = []
    for item in items:
        key = part_key(item)
        if current_key is None:
            current_key = key
        if key != current_key:
            groups.append((current_key, current_items))
            current_key = key
            current_items = []
        current_items.append(item)
    if current_key is not None:
        groups.append((current_key, current_items))
    return groups


def is_submission_latex_unit(item: NavItem) -> bool:
    if item.path in {"index.md", "front_matter_guide.md", "translation-status.md"}:
        return False
    if not re.search(r"part\d+/", item.path) and not item.path.startswith("appendix_") and item.path != "afterword.md":
        return False
    if re.search(r"part\d+/index\.md$", item.path):
        return False
    return True


def submission_latex_items(items: list[NavItem]) -> list[NavItem]:
    return [item for item in items if is_submission_latex_unit(item)]


def latex_unit_slug(item: NavItem, index: int) -> str:
    stem = Path(item.path).with_suffix("").as_posix()
    stem = stem.replace("/", "-").replace("_", "-")
    stem = re.sub(r"[^A-Za-z0-9-]+", "-", stem).strip("-").lower()
    return f"{index:02d}-{stem}.tex"


def strip_url_suffix(url: str) -> str:
    for sep in ("#", "?"):
        if sep in url:
            return url.split(sep, 1)[0]
    return url


def resolve_asset_url(raw_url: str, source_file: Path) -> Path | None:
    url = strip_url_suffix(raw_url.strip())
    if re.match(r"^(?:https?:|data:|file:|#)", url):
        return None
    return (source_file.parent / url).resolve()


def escape_plain(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def latex_url(url: str) -> str:
    return url.replace("\\", "/").replace("%", r"\%")


def protect(replacements: list[str], latex: str) -> str:
    token = f"@@LATEX_BLOCK_{len(replacements)}@@"
    replacements.append(latex)
    return token


def inline_to_latex(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</?(?:span|div|center|b|strong|em|i)[^>]*>", "", text, flags=re.I)
    text = re.sub(r"<sup>(.*?)</sup>", r"^\1", text, flags=re.I)
    text = re.sub(r"<sub>(.*?)</sub>", r"_\1", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)

    replacements: list[str] = []

    def code_repl(match: re.Match[str]) -> str:
        value = escape_plain(match.group(1).strip())
        return protect(replacements, rf"\texttt{{{value}}}")

    text = re.sub(r"`([^`\n]+)`", code_repl, text)

    def image_link_repl(match: re.Match[str]) -> str:
        return match.group(1)

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image_link_repl, text)

    def link_repl(match: re.Match[str]) -> str:
        label = escape_plain(re.sub(r"\s+", " ", match.group(1)).strip())
        url = latex_url(match.group(2).strip())
        if not label:
            label = escape_plain(url)
        return protect(replacements, rf"\href{{{url}}}{{{label}}}")

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_repl, text)

    def bold_repl(match: re.Match[str]) -> str:
        value = escape_plain(match.group(1).strip())
        return protect(replacements, rf"\textbf{{{value}}}")

    text = re.sub(r"\*\*([^*\n][\s\S]*?[^*\n])\*\*", bold_repl, text)

    def italic_repl(match: re.Match[str]) -> str:
        value = escape_plain(match.group(1).strip())
        return protect(replacements, rf"\emph{{{value}}}")

    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", italic_repl, text)
    text = re.sub(r"__([^_\n]+)__", bold_repl, text)
    text = re.sub(r"~~([^~\n]+)~~", lambda m: escape_plain(m.group(1)), text)

    escaped = escape_plain(text)
    escaped = escaped.replace("\n", r"\newline{} ")
    for index, latex in enumerate(replacements):
        escaped = escaped.replace(rf"@@LATEX\_BLOCK\_{index}@@", latex)
        escaped = escaped.replace(f"@@LATEX_BLOCK_{index}@@", latex)
    return escaped.strip()


def is_table_separator(line: str) -> bool:
    line = line.strip()
    if "|" not in line:
        return False
    cells = split_table_row(line)
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def split_table_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for ch in line:
        if escaped:
            current.append(ch)
            escaped = False
        elif ch == "\\":
            current.append(ch)
            escaped = True
        elif ch == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    cells.append("".join(current).strip())
    return cells


def render_table(lines: list[str], stats: ExportStats) -> str:
    stats.tables += 1
    header = split_table_row(lines[0])
    rows = [split_table_row(line) for line in lines[2:]]
    cols = max(1, len(header))
    width = max(0.10, min(0.32, 0.90 / cols))
    spec = "|" + "|".join([rf">{{\RaggedRight\arraybackslash}}p{{{width:.3f}\textwidth}}" for _ in range(cols)]) + "|"

    def normalize(row: list[str]) -> list[str]:
        row = row[:cols]
        return row + [""] * (cols - len(row))

    rendered: list[str] = [rf"\begin{{longtable}}{{{spec}}}", r"\hline"]
    rendered.append(
        r"\rowcolor{tablehead} "
        + " & ".join(rf"\textbf{{{inline_to_latex(cell)}}}" for cell in normalize(header))
        + r" \\ \hline"
    )
    rendered.append(r"\endfirsthead")
    rendered.append(
        r"\rowcolor{tablehead} "
        + " & ".join(rf"\textbf{{{inline_to_latex(cell)}}}" for cell in normalize(header))
        + r" \\ \hline"
    )
    rendered.append(r"\endhead")
    for row in rows:
        rendered.append(" & ".join(inline_to_latex(cell) for cell in normalize(row)) + r" \\ \hline")
    rendered.append(r"\end{longtable}")
    return "\n".join(rendered)


def render_image(match: re.Match[str], source_file: Path, assets: AssetManager, tex_dir: Path) -> str:
    alt = match.group(1).strip()
    raw_url = match.group(2).strip()
    image_path = resolve_asset_url(raw_url, source_file)
    caption = inline_to_latex(alt) if alt else "Figure"
    if image_path is None:
        return rf"\begin{{quote}}\small External image not embedded: {inline_to_latex(raw_url)}\end{{quote}}"
    rel = assets.register(image_path, source_file)
    if rel is None:
        return (
            r"\begin{quote}\small "
            + f"Image placeholder: {caption} (source: {inline_to_latex(str(image_path))})"
            + r"\end{quote}"
        )
    image_ref = Path(rel)
    if not image_ref.is_absolute():
        image_ref = (OUT_DIR / image_ref).resolve()
    image_ref_text = Path(os.path.relpath(image_ref, tex_dir)).as_posix()
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=\linewidth,height=0.55\textheight,keepaspectratio]{{{image_ref_text}}}",
            rf"\caption*{{{caption}}}",
            r"\end{figure}",
        ]
    )


def preprocess_markdown(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = re.sub(
        r'<div\s+class=["\']chapter-authors["\']>\s*(.*?)\s*</div>',
        lambda m: f"**Author:** {html.unescape(m.group(1)).strip()}",
        text,
        flags=re.I | re.S,
    )
    text = re.sub(
        r'<div\s+align=["\']center["\']>\s*<b>(.*?)</b>\s*</div>',
        lambda m: f"**{html.unescape(m.group(1)).strip()}**",
        text,
        flags=re.I | re.S,
    )
    text = re.sub(r"</?div[^>]*>", "", text, flags=re.I)
    return text


def block_starts(line: str, next_line: str | None = None) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if re.match(r"^#{1,6}\s+", stripped):
        return True
    if stripped.startswith("```"):
        return True
    if stripped.startswith("!["):
        return True
    if stripped.startswith(">"):
        return True
    if re.match(r"^\s*[-*]\s+", line):
        return True
    if re.match(r"^\s*\d+\.\s+", line):
        return True
    if next_line is not None and "|" in stripped and is_table_separator(next_line):
        return True
    if stripped == "---":
        return True
    return False


def render_code_block(language: str, body: list[str], stats: ExportStats) -> str:
    stats.code_blocks += 1
    label = inline_to_latex(language.strip()) if language.strip() else "code"
    code = "\n".join(body).replace(r"\end{printcode}", r"\textbackslash{}end{printcode}")
    return "\n".join(
        [
            rf"\noindent\textsf{{{label}}}",
            r"\begin{printcode}",
            code,
            r"\end{printcode}",
        ]
    )


def render_list(lines: list[str], ordered: bool) -> str:
    env = "enumerate" if ordered else "itemize"
    rendered = [rf"\begin{{{env}}}"]
    pattern = r"^\s*\d+\.\s+(.*)$" if ordered else r"^\s*[-*]\s+(.*)$"
    for line in lines:
        match = re.match(pattern, line)
        if not match:
            continue
        rendered.append(rf"\item {inline_to_latex(match.group(1))}")
    rendered.append(rf"\end{{{env}}}")
    return "\n".join(rendered)


def render_heading(line: str) -> str:
    match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
    if not match:
        return inline_to_latex(line)
    level = len(match.group(1))
    title = inline_to_latex(match.group(2).strip())
    commands = {
        1: "chapter",
        2: "section",
        3: "subsection",
        4: "subsubsection",
        5: "paragraph",
        6: "subparagraph",
    }
    command = commands[level]
    return rf"\{command}{{{title}}}"


def markdown_to_latex(text: str, source_file: Path, assets: AssetManager, stats: ExportStats, tex_dir: Path) -> str:
    text = preprocess_markdown(text)
    lines = text.splitlines()
    out: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        next_line = lines[i + 1] if i + 1 < len(lines) else None

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            out.append(r"\par\noindent\rule{\linewidth}{0.3pt}\par")
            i += 1
            continue

        if stripped.startswith("```"):
            language = stripped[3:].strip()
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            out.append(render_code_block(language, body, stats))
            continue

        image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            out.append(render_image(image_match, source_file, assets, tex_dir))
            i += 1
            continue

        if re.match(r"^#{1,6}\s+", stripped):
            out.append(render_heading(stripped))
            i += 1
            continue

        if next_line is not None and "|" in stripped and is_table_separator(next_line):
            table_lines = [line, next_line]
            i += 2
            while i < len(lines) and "|" in lines[i].strip() and lines[i].strip():
                table_lines.append(lines[i])
                i += 1
            out.append(render_table(table_lines, stats))
            continue

        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(re.sub(r"^>\s?", "", lines[i].strip()))
                i += 1
            out.append(r"\begin{quote}" + "\n" + inline_to_latex(" ".join(quote_lines)) + "\n" + r"\end{quote}")
            continue

        if re.match(r"^\s*[-*]\s+", line):
            list_lines: list[str] = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                list_lines.append(lines[i])
                i += 1
            out.append(render_list(list_lines, ordered=False))
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            list_lines = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                list_lines.append(lines[i])
                i += 1
            out.append(render_list(list_lines, ordered=True))
            continue

        paragraph_lines = [stripped]
        i += 1
        while i < len(lines):
            lookahead = lines[i + 1] if i + 1 < len(lines) else None
            if block_starts(lines[i], lookahead):
                break
            paragraph_lines.append(lines[i].strip())
            i += 1
        paragraph = " ".join(paragraph_lines)

        def inline_image_repl(match: re.Match[str]) -> str:
            return "\n" + render_image(match, source_file, assets, tex_dir) + "\n"

        if re.search(r"!\[[^\]]*\]\([^)]+\)", paragraph):
            pieces = re.split(r"(\!\[[^\]]*\]\([^)]+\))", paragraph)
            rendered_pieces: list[str] = []
            for piece in pieces:
                match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", piece)
                if match:
                    rendered_pieces.append(render_image(match, source_file, assets, tex_dir))
                elif piece.strip():
                    rendered_pieces.append(inline_to_latex(piece))
            out.append("\n".join(rendered_pieces))
        else:
            out.append(inline_to_latex(paragraph) + "\n")

    return "\n\n".join(out)


def latex_preamble(stats: ExportStats) -> str:
    return rf"""
\documentclass[openany,10pt]{{book}}
\usepackage[paperwidth=185mm,paperheight=260mm,top=22mm,bottom=21mm,left=18mm,right=18mm,headheight=14pt]{{geometry}}
\usepackage{{fontspec}}
\usepackage{{xeCJK}}
\setmainfont{{Times New Roman}}
\setsansfont{{Arial}}
\setCJKmainfont{{Songti SC}}
\setCJKsansfont{{PingFang SC}}
\setCJKmonofont{{PingFang SC}}
\setmonofont{{Menlo}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{caption}}
\usepackage{{longtable}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{ragged2e}}
\usepackage[table]{{xcolor}}
\usepackage{{fancyvrb}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true,linkcolor=black,urlcolor=blue,citecolor=black}}
\definecolor{{tablehead}}{{RGB}}{{238,243,248}}
\definecolor{{codeframe}}{{RGB}}{{190,198,210}}
\DefineVerbatimEnvironment{{printcode}}{{Verbatim}}{{fontsize=\scriptsize,frame=single,framesep=2mm,rulecolor=\color{{codeframe}}}}
\setlength{{\parindent}}{{2em}}
\setlength{{\parskip}}{{0.25em}}
\linespread{{1.18}}
\captionsetup{{font=small,labelformat=empty}}
\sloppy

\title{{Data Engineering for Large Foundation Models\\A Handbook}}
\author{{datascale-ai}}
\date{{Springer 16K LaTeX review source\\Files: {stats.files}; images: {stats.images}; code blocks: {stats.code_blocks}; tables: {stats.tables}}}
"""


def build_latex_document(items: list[NavItem], assets: AssetManager, stats: ExportStats, tex_dir: Path = OUT_DIR) -> str:
    body: list[str] = []
    included: set[str] = set()

    for item in items:
        if item.path in included:
            continue
        included.add(item.path)
        source_file = DOCS_EN / item.path
        if not source_file.exists():
            stats.missing += 1
            stats.warnings.append(f"missing nav source: {source_file.relative_to(ROOT)}")
            continue
        stats.files += 1
        text = source_file.read_text(encoding="utf-8")
        body.append(r"\cleardoublepage")
        body.append(
            r"\noindent{\small\textsf{"
            + inline_to_latex(f"{item.title} | {item.path}")
            + r"}}\par\vspace{1.5mm}"
        )
        body.append(markdown_to_latex(text, source_file, assets, stats, tex_dir))

    preamble = latex_preamble(stats)
    # Stats are updated while rendering the body, so regenerate the preamble once.
    preamble = latex_preamble(stats)
    return "\n".join(
        [
            preamble,
            r"\begin{document}",
            r"\frontmatter",
            r"\maketitle",
            r"\tableofcontents",
            r"\mainmatter",
            "\n\n".join(body),
            r"\end{document}",
            "",
        ]
    )


def build_latex_body(items: list[NavItem], assets: AssetManager, stats: ExportStats, tex_dir: Path) -> str:
    body: list[str] = []
    included: set[str] = set()

    for item in items:
        if item.path in included:
            continue
        included.add(item.path)
        source_file = DOCS_EN / item.path
        if not source_file.exists():
            stats.missing += 1
            stats.warnings.append(f"missing nav source: {source_file.relative_to(ROOT)}")
            continue
        stats.files += 1
        text = source_file.read_text(encoding="utf-8")
        body.append(r"\cleardoublepage")
        body.append(
            r"\noindent{\small\textsf{"
            + inline_to_latex(f"{item.title} | {item.path}")
            + r"}}\par\vspace{1.5mm}"
        )
        body.append(markdown_to_latex(text, source_file, assets, stats, tex_dir))

    return "\n\n".join(body)


def build_latex_wrapper(input_paths: list[Path], stats: ExportStats, base_dir: Path) -> str:
    includes = [rf"\input{{{Path(os.path.relpath(path, base_dir)).as_posix()}}}" for path in input_paths]
    return "\n".join(
        [
            latex_preamble(stats),
            r"\begin{document}",
            r"\frontmatter",
            r"\maketitle",
            r"\tableofcontents",
            r"\mainmatter",
            "\n\n".join(includes),
            r"\end{document}",
            "",
        ]
    )


def write_outputs(tex: str, stats: ExportStats, tex_path: Path = OUT_TEX, warnings_path: Path = OUT_WARNINGS) -> None:
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex, encoding="utf-8")
    warnings_path.write_text("\n".join(stats.warnings) + ("\n" if stats.warnings else ""), encoding="utf-8")
    print(f"[ok] LaTeX written: {tex_path}")
    print(f"[ok] warnings written: {warnings_path} ({len(stats.warnings)} warnings)")


def compile_pdf(tex_path: Path, pdf_path: Path, timeout: int) -> None:
    tectonic = shutil.which("tectonic")
    xelatex = shutil.which("xelatex")
    if tectonic:
        tex_dir = tex_path.parent
        cmd = [tectonic, "--keep-logs", "-o", ".", tex_path.name]
        print("[run] " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=tex_dir)
        emit_process_output(proc.stdout, stream=sys.stdout)
        emit_process_output(proc.stderr, stream=sys.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"Tectonic failed with rc={proc.returncode}")
    elif xelatex:
        tex_dir = tex_path.parent
        cmd = [
            xelatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory=.",
            tex_path.name,
        ]
        for _ in range(2):
            print("[run] " + " ".join(cmd))
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=tex_dir)
            emit_process_output(proc.stdout, stream=sys.stdout)
            emit_process_output(proc.stderr, stream=sys.stderr)
            if proc.returncode != 0:
                raise RuntimeError(f"XeLaTeX failed with rc={proc.returncode}")
    else:
        raise RuntimeError("tectonic or xelatex is required to compile the LaTeX PDF")
    if not pdf_path.exists() or pdf_path.stat().st_size < 100_000:
        raise RuntimeError("PDF was not produced or is suspiciously small")
    print(f"[ok] PDF written: {pdf_path} ({pdf_path.stat().st_size / 1024 / 1024:.1f} MB)")


def emit_process_output(text: str, stream: Any, max_lines: int = 30) -> None:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return
    if len(lines) <= max_lines:
        print("\n".join(lines), file=stream)
        return
    omitted = len(lines) - max_lines
    print(f"[log] omitted {omitted} lines; showing last {max_lines}:", file=stream)
    print("\n".join(lines[-max_lines:]), file=stream)


def merge_pdfs(part_pdfs: list[Path], output_pdf: Path) -> None:
    if not part_pdfs:
        raise RuntimeError("no part PDFs to merge")
    pdfunite = shutil.which("pdfunite")
    if pdfunite is None:
        raise RuntimeError("pdfunite is required to merge split PDFs")
    cmd = [pdfunite, *map(str, part_pdfs), str(output_pdf)]
    print("[run] " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"pdfunite failed with rc={proc.returncode}")
    print(f"[ok] merged PDF written: {output_pdf} ({output_pdf.stat().st_size / 1024 / 1024:.1f} MB)")


def export_single(items: list[NavItem], compile_output: bool, timeout: int) -> None:
    stats = ExportStats()
    assets = AssetManager(ASSET_DIR, stats)
    assets.reset()
    tex = build_latex_document(items, assets, stats)
    write_outputs(tex, stats)
    print_stats(stats)
    if compile_output:
        compile_pdf(OUT_TEX, OUT_PDF, timeout)


def export_split(items: list[NavItem], compile_output: bool, timeout: int) -> None:
    if PARTS_DIR.exists():
        shutil.rmtree(PARTS_DIR)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    if CHAPTERS_DIR.exists():
        shutil.rmtree(CHAPTERS_DIR)
    CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    groups = group_items(items)
    width = len(str(len(groups)))
    part_pdfs: list[Path] = []
    total = ExportStats()
    shared_assets = AssetManager(ASSET_DIR, total)
    shared_assets.reset()
    manifest_lines = [
        "# English Springer LaTeX Source Parts",
        "",
        "Each file is generated from the English MkDocs navigation and can be reviewed as an editable source unit.",
        "",
        "| No. | Group | LaTeX | Source files |",
        "| --- | --- | --- | --- |",
    ]

    chapter_manifest_lines = [
        "# English Springer LaTeX Chapter Sources",
        "",
        "Each `.tex` file is one chapter, project, appendix, or front/back matter contribution in the original source format.",
        "",
        "| No. | LaTeX | Source file | Title |",
        "| --- | --- | --- | --- |",
    ]
    chapter_paths: list[Path] = []
    for index, item in enumerate(submission_latex_items(items), 1):
        chapter_stats = ExportStats()
        shared_assets.stats = chapter_stats
        tex_path = CHAPTERS_DIR / latex_unit_slug(item, index)
        tex_body = build_latex_body([item], shared_assets, chapter_stats, tex_path.parent)
        write_outputs(tex_body + "\n", chapter_stats, tex_path, tex_path.with_suffix(".warnings.txt"))
        chapter_paths.append(tex_path)
        chapter_manifest_lines.append(
            f"| {index} | `{tex_path.name}` | `{item.path}` | {item.title} |"
        )
    (CHAPTERS_DIR / "README.md").write_text("\n".join(chapter_manifest_lines) + "\n", encoding="utf-8")

    main_stats = ExportStats()
    main_stats.files = len(chapter_paths)
    main_tex = build_latex_wrapper(chapter_paths, main_stats, PARTS_DIR)
    write_outputs(main_tex, main_stats, PARTS_DIR / "00-manuscript.tex", PARTS_DIR / "00-manuscript.warnings.txt")

    for index, (key, grouped_items) in enumerate(groups, 1):
        part_stats = ExportStats()
        shared_assets.stats = part_stats
        prefix = f"{index:0{width}d}-{key}"
        tex_path = PARTS_DIR / f"{prefix}.tex"
        pdf_path = PARTS_DIR / f"{prefix}.pdf"
        warnings_path = PARTS_DIR / f"{prefix}.warnings.txt"
        tex = build_latex_document(grouped_items, shared_assets, part_stats, tex_path.parent)
        write_outputs(tex, part_stats, tex_path, warnings_path)
        print_stats(part_stats, label=prefix)
        source_files = "<br>".join(f"`{item.path}`" for item in grouped_items)
        manifest_lines.append(f"| {index} | {key} | `{tex_path.name}` | {source_files} |")
        total.files += part_stats.files
        total.missing += part_stats.missing
        total.images += part_stats.images
        total.unsupported_images += part_stats.unsupported_images
        total.code_blocks += part_stats.code_blocks
        total.tables += part_stats.tables
        total.warnings.extend(part_stats.warnings)
        if compile_output:
            compile_pdf(tex_path, pdf_path, timeout)
            part_pdfs.append(pdf_path)

    OUT_WARNINGS.write_text("\n".join(total.warnings) + ("\n" if total.warnings else ""), encoding="utf-8")
    (PARTS_DIR / "README.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print_stats(total, label="total")
    if compile_output:
        merge_pdfs(part_pdfs, OUT_PDF)


def print_stats(stats: ExportStats, label: str = "") -> None:
    prefix = f"[stats] {label} | " if label else "[stats] "
    print(
        prefix
        + ", ".join(
            [
                f"files={stats.files}",
                f"missing={stats.missing}",
                f"images={stats.images}",
                f"unsupported_images={stats.unsupported_images}",
                f"code_blocks={stats.code_blocks}",
                f"tables={stats.tables}",
            ]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export en manuscript to a 16K LaTeX source package")
    parser.add_argument("--compile", action="store_true", help="Compile PDF with Tectonic or XeLaTeX after writing .tex")
    parser.add_argument("--split", action="store_true", help="Write and optionally compile one .tex per book part")
    parser.add_argument("--timeout", type=int, default=1800, help="LaTeX compiler timeout in seconds")
    parser.add_argument("--limit", type=int, default=0, help="Only export the first N nav files for smoke testing")
    parser.add_argument("--only", action="append", default=[], help="Only export nav paths containing this text")
    args = parser.parse_args()

    config = yaml.safe_load(MKDOCS.read_text(encoding="utf-8"))
    items = prepare_latex_items(flatten_nav(find_en_nav(config)))
    if args.only:
        needles = tuple(args.only)
        items = [item for item in items if any(needle in item.path or needle in item.title for needle in needles)]
    if args.limit:
        items = items[: args.limit]

    if args.split:
        export_split(items, args.compile, args.timeout)
    else:
        export_single(items, args.compile, args.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
