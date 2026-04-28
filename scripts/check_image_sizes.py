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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    limit_bytes = int(args.limit_mb * 1024 * 1024)
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

    oversized: list[tuple[Path, int]] = []
    for path in sorted(args.root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            size = path.stat().st_size
            if size > limit_bytes:
                oversized.append((path, size))

    if oversized:
        print(f"Found {len(oversized)} image(s) larger than {args.limit_mb:g} MB:")
        for path, size in oversized:
            print(f"- {path}: {size / 1024 / 1024:.2f} MB")
        return 1

    print(f"All images under {args.root} are <= {args.limit_mb:g} MB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
