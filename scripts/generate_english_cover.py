from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)

REF_NETWORK = Path(
    "/Users/wangcong/Library/Containers/com.tencent.xinWeChat/Data/Documents/"
    "xwechat_files/jie48324_a15a/temp/RWTemp/2026-05/"
    "aa1b19db33d7074d94cd2cff3d2eec0e/3bb4042862810386c45fe77f1a82557c.png"
)
REF_POLY = Path(
    "/Users/wangcong/Library/Containers/com.tencent.xinWeChat/Data/Documents/"
    "xwechat_files/jie48324_a15a/temp/RWTemp/2026-05/"
    "aa1b19db33d7074d94cd2cff3d2eec0e/f1741e656ddbbff445c991083234fae9.png"
)

DPI = 300
FRONT_W = 6 * DPI
SPINE_W = int(0.58 * DPI)
H = 9 * DPI
SPREAD_W = FRONT_W * 2 + SPINE_W

COLORS = {
    "paper": (246, 248, 246),
    "ink": (244, 250, 250),
    "muted": (177, 210, 210),
    "teal": (6, 57, 73),
    "teal_dark": (3, 36, 50),
    "teal_deep": (2, 28, 40),
    "cyan": (34, 194, 226),
    "blue": (62, 146, 204),
    "magenta": (231, 0, 111),
    "lime": (157, 218, 83),
    "orange": (247, 160, 56),
}

FONT_DIRS = [
    Path("/System/Library/Fonts"),
    Path("/System/Library/Fonts/Supplemental"),
    Path("/Library/Fonts"),
]


def font_path(name: str) -> str:
    for base in FONT_DIRS:
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    return str(Path("/System/Library/Fonts/Supplemental/Arial.ttf"))


FONT_HEAD = font_path("DIN Condensed Bold.ttf")
FONT_BODY = font_path("Arial.ttf")
FONT_BOLD = font_path("Arial Bold.ttf")
FONT_NARROW = font_path("Arial Narrow.ttf")
FONT_NARROW_BOLD = font_path("Arial Narrow Bold.ttf")
FONT_MONO = font_path("SFNSMono.ttf")


def f(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def gradient(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    w, h = size
    im = Image.new("RGB", size)
    px = im.load()
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        for x in range(w):
            px[x, y] = (r, g, b)
    return im


def reference_image(path: Path, size: tuple[int, int], crop_bias: tuple[float, float] = (0.5, 0.5)) -> Image.Image | None:
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    return ImageOps.fit(im, size, method=Image.Resampling.LANCZOS, centering=crop_bias)


def tint_reference(im: Image.Image, opacity: int = 255, contrast: float = 1.04, brightness: float = 0.72) -> Image.Image:
    im = ImageEnhance.Contrast(im).enhance(contrast)
    im = ImageEnhance.Brightness(im).enhance(brightness)
    overlay = Image.new("RGB", im.size, COLORS["teal_deep"])
    im = Image.blend(im, overlay, 0.20)
    rgba = im.convert("RGBA")
    rgba.putalpha(opacity)
    return rgba


def draw_reference_band(canvas: Image.Image, box: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = box
    ref = reference_image(REF_NETWORK, (x1 - x0, y1 - y0), crop_bias=(0.50, 0.44))
    if ref is None:
        return False
    layer = tint_reference(ref, opacity=255, contrast=1.18, brightness=0.68)
    shade = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    sd.rectangle((0, int(layer.height * 0.56), layer.width, layer.height), fill=(2, 28, 40, 72))
    sd.rectangle((0, 0, layer.width, layer.height), outline=(34, 194, 226, 52), width=3)
    layer = Image.alpha_composite(layer, shade)
    canvas.alpha_composite(layer, (x0, y0))
    return True


def draw_polyhedron_wash(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    opacity: int = 72,
    crop_bias: tuple[float, float] = (0.50, 0.50),
) -> bool:
    x0, y0, x1, y1 = box
    ref = reference_image(REF_POLY, (x1 - x0, y1 - y0), crop_bias=crop_bias)
    if ref is None:
        return False
    ref = ImageEnhance.Contrast(ref).enhance(1.15)
    ref = ImageEnhance.Brightness(ref).enhance(0.55)
    overlay = Image.new("RGB", ref.size, COLORS["teal_dark"])
    ref = Image.blend(ref, overlay, 0.42)
    rgba = ref.convert("RGBA").filter(ImageFilter.GaussianBlur(0.4))
    rgba.putalpha(opacity)
    canvas.alpha_composite(rgba, (x0, y0))
    return True


def wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_w or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_w: int,
    leading: int,
) -> int:
    x, y = xy
    for line in wrap(draw, text, font, max_w):
        draw.text((x, y), line, font=font, fill=fill)
        y += leading
    return y


def draw_pipeline_art(canvas: Image.Image, box: tuple[int, int, int, int], seed: int = 7) -> None:
    random.seed(seed)
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # Technical image band: faint grid, data cards, and flow lines.
    for i in range(-h, w, 70):
        d.line([(i, 0), (i + h, h)], fill=(255, 255, 255, 24), width=2)
    for y in range(0, h, 85):
        d.line([(0, y), (w, y)], fill=(42, 205, 232, 35), width=1)
    for x in range(0, w, 110):
        d.line([(x, 0), (x, h)], fill=(255, 255, 255, 18), width=1)

    nodes = []
    for i in range(42):
        px = int(w * (0.08 + random.random() * 0.84))
        py = int(h * (0.18 + random.random() * 0.70))
        nodes.append((px, py))
    nodes.sort()
    for a, b in zip(nodes, nodes[1:]):
        if random.random() < 0.72:
            color = random.choice([(34, 194, 226, 110), (157, 218, 83, 85), (255, 255, 255, 75)])
            d.line([a, b], fill=color, width=random.choice([2, 2, 3]))
    for px, py in nodes:
        r = random.choice([5, 7, 9])
        d.ellipse((px - r, py - r, px + r, py + r), fill=(236, 252, 255, 210), outline=(34, 194, 226, 210))

    card_font = f(FONT_MONO, 22)
    labels = ["RAW", "CLEAN", "DEDUP", "ALIGN", "RAG", "EVAL"]
    for i, label in enumerate(labels):
        cx = int(w * (0.10 + i * 0.145))
        cy = int(h * (0.64 + 0.16 * math.sin(i)))
        d.rounded_rectangle((cx, cy, cx + 132, cy + 54), radius=6, fill=(4, 45, 61, 190), outline=(34, 194, 226, 140), width=2)
        d.text((cx + 18, cy + 15), label, font=card_font, fill=(219, 252, 255, 230))

    # A soft glow on the right gives the top image depth without becoming busy.
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for r, alpha in [(560, 22), (380, 32), (230, 42)]:
        gd.ellipse((w - r // 2, -r // 3, w + r // 2, r), fill=(34, 194, 226, alpha))
    glow = glow.filter(ImageFilter.GaussianBlur(28))
    layer = Image.alpha_composite(layer, glow)
    canvas.alpha_composite(layer, (x0, y0))


def draw_front(canvas: Image.Image, x: int, y: int, w: int, h: int, use_references: bool = False) -> None:
    d = ImageDraw.Draw(canvas)
    d.rectangle((x, y, x + w, y + h), fill=COLORS["teal_dark"])

    top_h = int(h * 0.34)
    if not (use_references and draw_reference_band(canvas, (x, y, x + w, y + top_h))):
        band = gradient((w, top_h), (52, 171, 198), (3, 43, 56)).convert("RGBA")
        canvas.alpha_composite(band, (x, y))
        draw_pipeline_art(canvas, (x, y, x + w, y + top_h), seed=12)
    if use_references:
        draw_polyhedron_wash(
            canvas,
            (x + int(w * 0.48), y + top_h + 420, x + w - 36, y + h - 430),
            opacity=48,
            crop_bias=(0.48, 0.50),
        )

    # Reference-inspired accent block near the spine.
    accent_x = x
    accent_y = y + top_h - 8
    d.rectangle((accent_x, accent_y, accent_x + 245, accent_y + 245), fill=COLORS["cyan"])
    d.rectangle((accent_x + 42, accent_y + 42, accent_x + 203, accent_y + 203), outline=(255, 255, 255), width=7)
    for i in range(5):
        px = accent_x + 72 + i * 25
        d.line((px, accent_y + 160, px + 36, accent_y + 96), fill=COLORS["ink"], width=5)
        d.ellipse((px + 31, accent_y + 91, px + 44, accent_y + 104), fill=COLORS["ink"])
    d.rectangle((accent_x + 245, accent_y + 86, accent_x + w, accent_y + 99), fill=(255, 255, 255, 58))

    left = x + 330
    title_font = f(FONT_HEAD, 166)
    sub_title_font = f(FONT_NARROW, 60)
    meta_font = f(FONT_NARROW, 38)
    small_font = f(FONT_BODY, 28)
    micro_font = f(FONT_MONO, 22)

    d.text((left, y + top_h + 105), "Data Engineering", font=title_font, fill=COLORS["ink"])
    d.text((left, y + top_h + 275), "for Large Models", font=title_font, fill=COLORS["ink"])
    d.text((left, y + top_h + 475), "Architecture, Algorithms & Projects", font=sub_title_font, fill=(207, 238, 240))

    d.rectangle((left, y + top_h + 585, left + 980, y + top_h + 591), fill=COLORS["cyan"])
    d.text((left, y + top_h + 630), "A complete guide to LLM data pipelines, multimodal alignment, RAG,", font=small_font, fill=(198, 225, 226))
    d.text((left, y + top_h + 672), "synthetic data, DataOps, and privacy-compliant governance.", font=small_font, fill=(198, 225, 226))

    chips = [("28 CHAPTERS", COLORS["magenta"]), ("10 CAPSTONE PROJECTS", COLORS["cyan"]), ("2026 EDITION", COLORS["lime"])]
    chip_x = left
    chip_y = y + h - 520
    for label, color in chips:
        tw = d.textbbox((0, 0), label, font=micro_font)[2]
        d.rounded_rectangle((chip_x, chip_y, chip_x + tw + 54, chip_y + 62), radius=5, fill=color)
        d.text((chip_x + 27, chip_y + 19), label, font=micro_font, fill=(2, 28, 40))
        chip_x += tw + 82

    d.text((left, y + h - 320), "Jun Yu", font=meta_font, fill=COLORS["ink"])
    d.text((left, y + h - 258), "DataScale AI Open Book Series", font=small_font, fill=COLORS["muted"])

    # Small imprint mark, custom and non-publisher-specific.
    mark_x = x + w - 310
    mark_y = y + h - 270
    d.rectangle((mark_x, mark_y, mark_x + 72, mark_y + 72), outline=COLORS["ink"], width=5)
    d.line((mark_x + 16, mark_y + 50, mark_x + 36, mark_y + 24, mark_x + 58, mark_y + 50), fill=COLORS["cyan"], width=5)
    d.text((mark_x + 96, mark_y + 13), "DataScale", font=f(FONT_BOLD, 36), fill=COLORS["ink"])
    d.text((mark_x + 98, mark_y + 53), "AI", font=f(FONT_BODY, 28), fill=COLORS["muted"])


def draw_spine(canvas: Image.Image, x: int, y: int, w: int, h: int) -> None:
    d = ImageDraw.Draw(canvas)
    d.rectangle((x, y, x + w, y + h), fill=(4, 47, 62))
    d.rectangle((x, y, x + 15, y + h), fill=COLORS["cyan"])
    d.rectangle((x + w - 15, y, x + w, y + h), fill=(2, 31, 43))
    d.rectangle((x, y + int(h * 0.34) - 8, x + w, y + int(h * 0.34) + 235), fill=(5, 68, 84))

    text = "Data Engineering for Large Models"
    font = f(FONT_NARROW_BOLD, 52)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    txt = Image.new("RGBA", (tw + 12, th + 12), (0, 0, 0, 0))
    td = ImageDraw.Draw(txt)
    td.text((6, 6), text, font=font, fill=COLORS["ink"])
    rot = txt.rotate(90, expand=True)
    canvas.alpha_composite(rot, (x + (w - rot.width) // 2, y + (h - rot.height) // 2))

    author = "Jun Yu"
    afont = f(FONT_NARROW, 38)
    abox = d.textbbox((0, 0), author, font=afont)
    atxt = Image.new("RGBA", (abox[2] - abox[0] + 12, abox[3] - abox[1] + 12), (0, 0, 0, 0))
    ad = ImageDraw.Draw(atxt)
    ad.text((6, 6), author, font=afont, fill=COLORS["muted"])
    arot = atxt.rotate(90, expand=True)
    canvas.alpha_composite(arot, (x + (w - arot.width) // 2, y + 170))


def draw_back(canvas: Image.Image, x: int, y: int, w: int, h: int, use_references: bool = False) -> None:
    d = ImageDraw.Draw(canvas)
    d.rectangle((x, y, x + w, y + h), fill=COLORS["teal"])
    if use_references:
        draw_polyhedron_wash(canvas, (x, y, x + w, y + int(h * 0.46)), opacity=92, crop_bias=(0.50, 0.52))
        d.rectangle((x, y, x + w, y + int(h * 0.46)), fill=(2, 28, 40, 88))
    d.rectangle((x, y, x + 20, y + h), fill=(8, 86, 105))
    d.rectangle((x + 48, y + 105, x + w - 130, y + 112), fill=COLORS["cyan"])

    title_font = f(FONT_NARROW_BOLD, 58)
    body_font = f(FONT_BODY, 30)
    body_bold = f(FONT_BOLD, 31)
    small_font = f(FONT_BODY, 25)
    mono = f(FONT_MONO, 23)

    left = x + 120
    top = y + 170
    d.text((left, top), "Data Engineering for Large Models", font=title_font, fill=COLORS["ink"])
    top += 100
    paragraph = (
        "Data quality determines the upper bound of model performance. This book turns that principle "
        "into a practical engineering system, connecting raw corpora, multimodal assets, alignment data, "
        "retrieval pipelines, DataOps platforms, and privacy-aware governance."
    )
    top = draw_wrapped(d, (left, top), paragraph, body_font, (211, 232, 232), w - 260, 45) + 42

    d.text((left, top), "Inside the 2026 edition", font=body_bold, fill=COLORS["ink"])
    top += 64
    bullets = [
        "28 chapters across the complete LLM data lifecycle",
        "10 runnable capstone projects, from Mini-C4 to LLM data flywheel",
        "Pre-training data cleaning, deduplication, tokenization, and evaluation",
        "Multimodal data alignment for image, document, video, and audio workloads",
        "RAG, synthetic data, reasoning data, agent data, DataOps, and governance",
    ]
    for bullet in bullets:
        d.ellipse((left, top + 13, left + 14, top + 27), fill=COLORS["cyan"])
        top = draw_wrapped(d, (left + 38, top), bullet, body_font, (219, 238, 238), w - 320, 42) + 18

    # Bottom technical table, echoing the production-book reference layouts.
    table_y = y + h - 590
    d.rectangle((left, table_y, x + w - 120, table_y + 320), fill=(3, 42, 55), outline=(58, 135, 150), width=2)
    rows = [
        ("Core stack", "Ray Data, Spark, DVC, LakeFS, MLflow, Airflow"),
        ("Data formats", "Parquet, WebDataset, vector indexes, audit logs"),
        ("Learning path", "Theory, architecture, algorithms, projects"),
        ("Audience", "Researchers, ML engineers, data platform teams"),
    ]
    row_h = 80
    for i, (k, v) in enumerate(rows):
        yy = table_y + i * row_h
        if i:
            d.line((left, yy, x + w - 120, yy), fill=(58, 135, 150), width=2)
        d.text((left + 32, yy + 25), k.upper(), font=mono, fill=COLORS["cyan"])
        d.text((left + 330, yy + 24), v, font=small_font, fill=(221, 238, 238))

    d.text((left, y + h - 170), "Computer Science / Artificial Intelligence / Data Engineering", font=small_font, fill=COLORS["muted"])
    d.text((left, y + h - 122), "datascale-ai.github.io/data_engineering_book", font=mono, fill=COLORS["ink"])

    # Barcode placeholder to preserve the book-cover affordance without inventing an ISBN.
    bx, by = x + w - 430, y + h - 232
    d.rectangle((bx, by, bx + 260, by + 112), fill=(246, 248, 246))
    rng = random.Random(3)
    xx = bx + 18
    while xx < bx + 240:
        bw = rng.choice([3, 4, 5, 7])
        d.rectangle((xx, by + 16, xx + bw, by + 92), fill=(12, 24, 28))
        xx += bw + rng.choice([4, 5, 7])
    d.text((bx + 20, by + 92), "ISBN placeholder", font=f(FONT_BODY, 18), fill=(12, 24, 28))


def add_cover_shadow(im: Image.Image) -> Image.Image:
    # Thin production guides and panel separation, like a full wrap placed on a white page.
    d = ImageDraw.Draw(im)
    d.line((FRONT_W, 0, FRONT_W, H), fill=(0, 0, 0, 90), width=3)
    d.line((FRONT_W + SPINE_W, 0, FRONT_W + SPINE_W, H), fill=(0, 0, 0, 90), width=3)
    d.rectangle((0, 0, SPREAD_W - 1, H - 1), outline=(1, 24, 33), width=5)
    return im


def save_pdf_from_png(png_path: Path, pdf_path: Path) -> None:
    image = Image.open(png_path).convert("RGB")
    image.save(pdf_path, "PDF", resolution=DPI)


def render_cover(prefix: str, use_references: bool = False) -> None:
    spread = Image.new("RGBA", (SPREAD_W, H), COLORS["teal_deep"] + (255,))
    draw_back(spread, 0, 0, FRONT_W, H, use_references=use_references)
    draw_spine(spread, FRONT_W, 0, SPINE_W, H)
    draw_front(spread, FRONT_W + SPINE_W, 0, FRONT_W, H, use_references=use_references)
    spread = add_cover_shadow(spread)
    spread_rgb = spread.convert("RGB")
    spread_png = OUT / f"{prefix}_cover_spread.png"
    spread_pdf = OUT / f"{prefix}_cover_spread.pdf"
    spread_rgb.save(spread_png, quality=96, dpi=(DPI, DPI))
    save_pdf_from_png(spread_png, spread_pdf)

    front = spread.crop((FRONT_W + SPINE_W, 0, SPREAD_W, H)).convert("RGB")
    front_png = OUT / f"{prefix}_front_cover.png"
    front_pdf = OUT / f"{prefix}_front_cover.pdf"
    front.save(front_png, quality=96, dpi=(DPI, DPI))
    save_pdf_from_png(front_png, front_pdf)

    print(spread_png)
    print(spread_pdf)
    print(front_png)
    print(front_pdf)


def main() -> None:
    render_cover("data_engineering_large_models", use_references=False)
    render_cover("data_engineering_large_models_fusion", use_references=True)


if __name__ == "__main__":
    main()
