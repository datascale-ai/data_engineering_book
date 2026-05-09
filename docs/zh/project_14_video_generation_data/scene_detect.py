#!/usr/bin/env python3
"""Step 2: PySceneDetect ContentDetector + split_video_ffmpeg.

Per-source-video pipeline:
  1. mkdir shots/<vid>/
  2. detect scenes using a 480p-ish proxy (downscale flag)
  3. split with ffmpeg `-c copy` (keyframe-aligned)
  4. append per-shot rows to stage2_scenes.shard{worker_id}.jsonl
  5. touch shots/<vid>/_DONE

Crash-safety:
- If _DONE missing but shots/<vid>/ has .mp4 files we WIPE the dir before re-cutting,
  to avoid mixing partial cuts with manifest rows.
- Each worker writes its own shard JSONL; merge_shards rebuilds stage2_scenes.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scenedetect import ContentDetector, SceneManager, open_video
from scenedetect.video_splitter import split_video_ffmpeg

from utils.io_utils import iter_jsonl, shard_path, shot_id, stage_paths
from utils.resume import SafeJsonlWriter, repair_tail, scan_done_ids


def _detect_scenes(video_path: str, threshold: float, downscale: int):
    """Return (FrameTimecode-list, [(start_sec, end_sec)]).

    When the detector finds no scene boundaries (e.g. a single-shot stock clip),
    we fall back to treating the entire video as one shot.
    """
    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    if downscale and downscale > 1:
        try:
            sm.auto_downscale = False
            sm.downscale = int(downscale)
        except Exception:
            pass
    sm.detect_scenes(video=video, show_progress=False)
    scene_list_ftc = sm.get_scene_list()
    if scene_list_ftc:
        secs = [(float(s.get_seconds()), float(e.get_seconds())) for s, e in scene_list_ftc]
        return scene_list_ftc, secs
    # fallback: treat whole video as one shot
    try:
        start = video.base_timecode  # FrameTimecode at t=0
        end = video.duration
        if end is None or end == start:
            return [], []
        full = (start, end)
        return [full], [(0.0, float(end.get_seconds()))]
    except Exception:
        return [], []


def _process_one(arg: dict) -> dict:
    """Worker entry. Returns summary stats per video."""
    vid = int(arg["video_id"])
    src = arg["path"]
    out_root = Path(arg["out_root"])
    threshold = float(arg["threshold"])
    downscale = int(arg["downscale"])
    min_len = float(arg["min_shot_len"])
    shard_jsonl = Path(arg["shard_jsonl"])
    worker_id = int(arg["worker_id"])

    shot_dir = out_root / "shots" / f"pexels_{vid}"
    done_marker = shot_dir / "_DONE"

    if done_marker.exists():
        # already finished previously
        return {"video_id": vid, "status": "skipped_done"}

    # if dir has stale partial cuts, wipe it
    if shot_dir.exists():
        try:
            shutil.rmtree(shot_dir)
        except OSError:
            pass
    shot_dir.mkdir(parents=True, exist_ok=True)

    try:
        scene_ftc, scene_secs = _detect_scenes(src, threshold=threshold, downscale=downscale)
    except Exception:
        return {
            "video_id": vid,
            "status": "detect_error",
            "error": traceback.format_exc(limit=2),
        }

    # filter very short shots in lock-step
    keep = [(ftc, secs) for ftc, secs in zip(scene_ftc, scene_secs) if (secs[1] - secs[0]) >= min_len]
    if not keep:
        done_marker.touch()
        return {"video_id": vid, "status": "no_shots"}
    scene_ftc = [k[0] for k in keep]
    scene_secs = [k[1] for k in keep]

    # split with ffmpeg -c copy; keyframe-aligned (fast, slight boundary drift accepted)
    output_template = str(shot_dir / "shot_$SCENE_NUMBER.mp4")
    try:
        split_video_ffmpeg(
            input_video_path=src,
            scene_list=scene_ftc,
            output_file_template=output_template,
            arg_override="-map 0:v:0 -map 0:a? -c:v copy -c:a copy -avoid_negative_ts make_zero",
            show_progress=False,
        )
    except Exception:
        return {
            "video_id": vid,
            "status": "split_error",
            "error": traceback.format_exc(limit=2),
        }

    produced = sorted(shot_dir.glob("shot_*.mp4"))
    if not produced:
        return {"video_id": vid, "status": "no_segments"}

    rows = []
    for idx0, fp in enumerate(produced):
        if idx0 < len(scene_secs):
            s, e = scene_secs[idx0]
        else:
            s, e = 0.0, 0.0
        sid = shot_id(vid, idx0)
        rows.append(
            {
                "shot_id": sid,
                "video_id": vid,
                "idx": idx0,
                "start_ts": float(s),
                "end_ts": float(e),
                "n_frames": int(round((e - s) * float(arg.get("fps") or 30.0))),
                "segment_path": str(fp),
            }
        )

    # append rows atomically; mark done only after writes flushed
    with SafeJsonlWriter(shard_jsonl) as w:
        for r in rows:
            w.append(r)
    done_marker.touch()
    return {"video_id": vid, "status": "ok", "n_shots": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source_videos.jsonl from Step 1")
    ap.add_argument("--out", required=True, help="stage1_output directory")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=27.0)
    ap.add_argument("--proxy-downscale", type=int, default=4)
    ap.add_argument("--min-shot-len", type=float, default=1.0, help="seconds; drop shots shorter than this")
    ap.add_argument("--max-samples", type=int, default=0, help="optional cap on total shots; 0 = unbounded")
    args = ap.parse_args()

    sp = stage_paths(args.out)
    sp["stages"].mkdir(parents=True, exist_ok=True)

    # already-done video_ids: any pexels_<vid>/_DONE
    shots_root = sp["shots"]
    shots_root.mkdir(parents=True, exist_ok=True)
    done_videos = set()
    for d in shots_root.glob("pexels_*"):
        if (d / "_DONE").exists():
            try:
                done_videos.add(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                continue

    src_records = list(iter_jsonl(args.src))
    todo = [r for r in src_records if int(r["video_id"]) not in done_videos]
    print(f"[step2] {len(src_records)} videos total, {len(done_videos)} done, {len(todo)} todo")

    # round-robin assign shard files by worker_id; we don't pre-assign per video,
    # we let the Pool dispatch in submission order and group writes by (worker_id, video).
    n_workers = max(1, int(args.workers))

    # pre-build the per-worker shard target so writes are mutually exclusive
    args_list = []
    for i, rec in enumerate(todo):
        worker_id = i % n_workers
        sjsonl = shard_path(sp["stage2"], worker_id)
        args_list.append(
            {
                "video_id": rec["video_id"],
                "path": rec["path"],
                "fps": rec.get("fps"),
                "out_root": str(sp["root"]),
                "threshold": args.threshold,
                "downscale": args.proxy_downscale,
                "min_shot_len": args.min_shot_len,
                "shard_jsonl": str(sjsonl),
                "worker_id": worker_id,
            }
        )

    # repair tails of all shards before launching
    for w in range(n_workers):
        repair_tail(shard_path(sp["stage2"], w))

    n_ok = n_err = n_no = 0
    n_shots = 0
    with Pool(processes=n_workers) as pool:
        for res in pool.imap_unordered(_process_one, args_list):
            status = res.get("status", "?")
            if status == "ok":
                n_ok += 1
                n_shots += int(res.get("n_shots", 0))
            elif status in ("no_shots", "no_segments"):
                n_no += 1
            elif status == "skipped_done":
                pass
            else:
                n_err += 1
                print(f"[step2] err video={res.get('video_id')} status={status}", flush=True)
            if args.max_samples and n_shots >= args.max_samples:
                print(f"[step2] reached max_samples={args.max_samples}, terminating workers")
                pool.terminate()
                break

    print(f"[step2] ok={n_ok} no_shots={n_no} errors={n_err} new_shots={n_shots}")


if __name__ == "__main__":
    main()
