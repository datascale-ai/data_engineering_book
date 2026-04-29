#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_LIMIT_MB = 5
DEFAULT_ROOT = Path("docs/images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail when documentation images exceed a size budget.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Image directory to scan.")
    parser.add_argument("--limit-mb", type=float, default=DEFAULT_LIMIT_MB, help="Maximum allowed size per image.")
    parser.add_argument(
        "--total-budget-mb",
        type=float,
        default=None,
        help=(
            "Optional aggregate size budget in MB for all matched images under --root. "
            "When omitted, no aggregate check is performed and only per-image size is enforced."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    limit_bytes = int(args.limit_mb * 1024 * 1024)
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

    oversized: list[tuple[Path, int]] = []
    total_bytes = 0
    image_count = 0
    for path in sorted(args.root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            size = path.stat().st_size
            total_bytes += size
            image_count += 1
            if size > limit_bytes:
                oversized.append((path, size))

    per_image_failed = bool(oversized)
    if per_image_failed:
        print(f"Per-image check: FAIL — {len(oversized)} image(s) larger than {args.limit_mb:g} MB:")
        for path, size in oversized:
            print(f"- {path}: {size / 1024 / 1024:.2f} MB")
    else:
        print(f"Per-image check: PASS — all {image_count} image(s) under {args.root} are <= {args.limit_mb:g} MB.")

    total_failed = False
    if args.total_budget_mb is not None:
        budget_bytes = int(args.total_budget_mb * 1024 * 1024)
        total_mb = total_bytes / 1024 / 1024
        if total_bytes > budget_bytes:
            total_failed = True
            print(
                f"Total budget: {total_mb:.2f} MB / {args.total_budget_mb:g} MB (FAIL)"
            )
        else:
            print(
                f"Total budget: {total_mb:.2f} MB / {args.total_budget_mb:g} MB (PASS)"
            )

    return 1 if (per_image_failed or total_failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
