#!/usr/bin/env python3
"""Step 1: load Pexels-licensed videos from /data0/book and emit normalized metadata.

Reads the existing pexels_manifest.jsonl produced by /data0/pexels_video_downloader,
re-probes each mp4 with ffprobe to fill duration / fps / resolution / nb_frames,
and writes stage1_output/source_videos.jsonl.

Resumable: re-running just appends rows for video_ids not yet present in the output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.io_utils import ffprobe, stage_paths
from utils.resume import SafeJsonlWriter, repair_tail, scan_done_ids


def _read_pexels_manifest(src_dir: Path) -> list[dict]:
    """Read pexels_manifest.jsonl if present; otherwise build a minimal record from filenames."""
    mf = src_dir / "pexels_manifest.jsonl"
    out: list[dict] = []
    if mf.exists():
        with open(mf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append(rec)
        return out
    # fallback: glob filenames
    for p in sorted(src_dir.glob("pexels_*.mp4")):
        try:
            vid = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        out.append({"video_id": vid, "saved_as": str(p), "user": {}, "page_url": None})
    return out


def _normalize(rec: dict, src_dir: Path) -> dict | None:
    vid = rec.get("video_id")
    if vid is None:
        return None
    saved_as = rec.get("saved_as")
    p = Path(saved_as) if saved_as else None
    if (p is None) or (not p.exists()):
        # try to find by id in src_dir
        cands = list(src_dir.glob(f"pexels_{int(vid)}_*.mp4"))
        if not cands:
            return None
        p = cands[0]
    info = ffprobe(p)
    if info is None:
        return None
    user = rec.get("user") or {}
    return {
        "video_id": int(vid),
        "path": str(p),
        "page_url": rec.get("page_url"),
        "author_name": user.get("name"),
        "author_url": user.get("url"),
        "license": "pexels",
        "duration": info.duration,
        "fps": info.fps,
        "width": info.width,
        "height": info.height,
        "nb_frames": info.nb_frames,
        "file_size": info.file_size,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data0/book", help="source directory containing pexels_*.mp4")
    ap.add_argument("--out", required=True, help="stage1_output directory")
    args = ap.parse_args()

    src = Path(args.src)
    sp = stage_paths(args.out)
    sp["root"].mkdir(parents=True, exist_ok=True)

    out_path = sp["source_videos"]
    repair_tail(out_path)
    done = scan_done_ids(out_path, key="video_id")
    print(f"[step1] resume: {len(done)} videos already in {out_path}")

    raw = _read_pexels_manifest(src)
    print(f"[step1] manifest entries: {len(raw)}")

    n_new = 0
    n_skip = 0
    n_miss = 0
    with SafeJsonlWriter(out_path) as w:
        for rec in raw:
            vid = rec.get("video_id")
            if vid is None:
                continue
            if str(vid) in done:
                n_skip += 1
                continue
            norm = _normalize(rec, src)
            if norm is None:
                n_miss += 1
                continue
            w.append(norm)
            n_new += 1
    total = len(done) + n_new
    print(f"[step1] new={n_new} skipped={n_skip} missing={n_miss} total={total}")


if __name__ == "__main__":
    main()
