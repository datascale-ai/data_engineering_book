#!/usr/bin/env python3
"""Step 5: Qwen2.5-VL-3B multi-frame English caption (>= 50 words).

Sharded across 8 GPUs. Each process loads the model once, then iterates its
shard of shots. For each shot:
  * sample N frames (default 8) at long_edge (default 448)
  * save them under frames/<vid>/shot_<NNNN>/f0..f{N-1}.jpg (reused by Step 6 + downstream NSFW)
  * run Qwen2.5-VL with a video-style prompt; require >= MIN_WORDS English words
  * up to 2 retries with higher temperature; mark caption_short=True if still short

Global sample cap: each shard periodically counts successful captions across
ALL shard files; if >= --max-samples, it exits cleanly. Up to ~num_shards
overshoot is tolerated."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from utils.gpu_safety import (
    DegradePolicy,
    OOMExhausted,
    free_memory,
    safe_call,
    setup_cuda_alloc_env,
)
from utils.io_utils import iter_jsonl, shard_path, parse_shot_id, stage_paths
from utils.resume import SafeJsonlWriter, count_done_with, repair_tail, scan_done_ids
from utils.video_io import sample_frames_pil, save_pil_jpgs


CAPTION_PROMPT = (
    "You are a professional video captioner. The frames below are sampled in time order from a single shot. "
    "Write ONE single-paragraph English caption of AT LEAST 60 words describing this shot as a whole. "
    "Cover: the main subjects, the setting/scene, the actions or movement happening, the camera framing, "
    "the lighting and color mood, and the overall atmosphere. Do NOT enumerate frames or say 'frame 1', "
    "'frame 2', etc. Do NOT mention filenames, indices, or quotes. Output the caption text only, no preamble."
)


def _count_global_pass(stage5_path: Path, num_shards: int) -> int:
    """Count successful captions across all sibling shard files."""
    total = 0
    for s in range(num_shards):
        total += count_done_with(
            shard_path(stage5_path, s),
            predicate=lambda r: r.get("status") == "ok",
        )
    return total


def _frames_dir(out_root: Path, shot_id: str) -> Path:
    p = parse_shot_id(shot_id)
    return out_root / "frames" / f"pexels_{p['video_id']}" / f"shot_{p['idx']:04d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="stage4_aesthetic.jsonl (merged) preferred")
    ap.add_argument("--scenes", required=True, help="stage2_scenes.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--qwen-path", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--long-edge", type=int, default=448)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--min-words", type=int, default=50)
    ap.add_argument("--require-pass-aesthetic", action="store_true",
                    help="if set, only caption shots with pass_aesthetic=True (default: caption all)")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="global cap on successful captions across all shards; 0 = unbounded")
    ap.add_argument("--global-check-every", type=int, default=20,
                    help="check the global cap every N shots")
    args = ap.parse_args()

    setup_cuda_alloc_env()

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        raise SystemExit(f"qwen_vl_utils not available: {e}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES is set externally
    dtype = torch.bfloat16

    sp = stage_paths(args.out)
    sp["stages"].mkdir(parents=True, exist_ok=True)
    sp["frames"].mkdir(parents=True, exist_ok=True)

    seg_map = {r["shot_id"]: r["segment_path"] for r in iter_jsonl(args.scenes)}
    aest_rows = list(iter_jsonl(args.inp))
    if args.require_pass_aesthetic:
        aest_rows = [r for r in aest_rows if r.get("pass_aesthetic")]

    shotlist = sorted(aest_rows, key=lambda r: r["shot_id"])
    mine = [r for i, r in enumerate(shotlist) if (i % args.num_shards) == args.shard_id]

    out_shard = shard_path(sp["stage5"], args.shard_id)
    repair_tail(out_shard)
    done = scan_done_ids(out_shard)
    todo = [r for r in mine if r["shot_id"] not in done and r["shot_id"] in seg_map]
    logging.info("step5 g=%d input=%d mine=%d done=%d todo=%d",
                 args.shard_id, len(shotlist), len(mine), len(done), len(todo))

    # Honor pre-existing global cap before loading the heavy model
    if args.max_samples > 0:
        already = _count_global_pass(sp["stage5"], args.num_shards)
        if already >= args.max_samples:
            logging.info("g=%d global cap already reached: %d >= %d; nothing to do",
                         args.shard_id, already, args.max_samples)
            return

    if not todo:
        return

    logging.info("g=%d loading Qwen2.5-VL from %s", args.shard_id, args.qwen_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.qwen_path)
    logging.info("g=%d model loaded", args.shard_id)

    policy = DegradePolicy(configs=[
        {"frames_n": args.frames, "long_edge": args.long_edge, "max_new_tokens": args.max_new_tokens},
        {"frames_n": args.frames, "long_edge": 384, "max_new_tokens": args.max_new_tokens},
        {"frames_n": 6, "long_edge": 384, "max_new_tokens": 200},
        {"frames_n": 4, "long_edge": 384, "max_new_tokens": 180},
        {"frames_n": 4, "long_edge": 336, "max_new_tokens": 160},
        {"frames_n": 2, "long_edge": 336, "max_new_tokens": 140},
    ])

    @torch.inference_mode()
    def _generate(*, frame_paths: list, frames_n: int, long_edge: int,
                  max_new_tokens: int, temperature: float, do_sample: bool):
        # frame_paths is the on-disk list (already saved at long_edge)
        # we may downsample to `frames_n` and re-resize if degraded
        sel = frame_paths
        if len(sel) > frames_n:
            import numpy as np
            ix = np.linspace(0, len(sel) - 1, num=frames_n).round().astype(int).tolist()
            sel = [sel[i] for i in ix]
        # pass as a Qwen video using file paths; processor will read+resize
        msgs = [{
            "role": "user",
            "content": [
                {"type": "video", "video": [f"file://{p}" for p in sel]},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        # When degrading long_edge we resize the in-memory frames manually
        if long_edge != args.long_edge and video_inputs:
            from PIL import Image
            new_videos = []
            for vid in video_inputs:
                # vid may be a list of PIL.Image OR a torch tensor; handle the common PIL list case
                if isinstance(vid, list):
                    resized = []
                    for im in vid:
                        if isinstance(im, Image.Image):
                            w, h = im.size
                            m = max(w, h)
                            if m > 0 and m != long_edge:
                                s = long_edge / m
                                im = im.resize((max(1, int(round(w * s))),
                                                max(1, int(round(h * s)))), Image.BICUBIC)
                            resized.append(im)
                        else:
                            resized.append(im)
                    new_videos.append(resized)
                else:
                    new_videos.append(vid)
            video_inputs = new_videos
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=0.9 if do_sample else 1.0,
        )
        trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen_ids)]
        out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return out[0].strip() if out else ""

    writer = SafeJsonlWriter(out_shard)
    n_ok = n_short = n_oom = n_err = 0
    try:
        for shot_idx, rec in enumerate(todo):
            sid = rec["shot_id"]
            seg = seg_map[sid]
            try:
                # 1. sample + save frames at the policy's current long_edge for re-use
                cur_le = int(policy.current["long_edge"])
                cur_frames = int(policy.current["frames_n"])
                pil_frames = sample_frames_pil(seg, k=max(args.frames, cur_frames), long_edge=cur_le)
                if not pil_frames:
                    writer.append({"shot_id": sid, "status": "no_frames"})
                    n_err += 1
                    continue
                fdir = _frames_dir(sp["root"], sid)
                fpaths = save_pil_jpgs(pil_frames[:args.frames], fdir)
                if not fpaths:
                    writer.append({"shot_id": sid, "status": "save_frames_failed"})
                    n_err += 1
                    continue

                # 2. generate, retry up to 2 times if too short
                caption = ""
                retries = 0
                short = False
                temps = [(False, 1.0), (True, 0.5), (True, 0.7)]
                for attempt, (do_sample, temp) in enumerate(temps):
                    try:
                        cap = safe_call(
                            lambda **cfg: _generate(
                                frame_paths=fpaths,
                                temperature=temp,
                                do_sample=do_sample,
                                **cfg,
                            ),
                            policy=policy,
                        )
                    except OOMExhausted:
                        n_oom += 1
                        cap = ""
                        break
                    cap = (cap or "").strip()
                    n_words = len(cap.split())
                    caption = cap
                    retries = attempt
                    if n_words >= args.min_words:
                        short = False
                        break
                    short = True

                n_words = len(caption.split())
                rec_out = {
                    "shot_id": sid,
                    "caption_en": caption,
                    "n_words": int(n_words),
                    "frame_paths": fpaths,
                    "retries": int(retries),
                    "caption_short": bool(short and n_words < args.min_words),
                    "status": "ok" if caption else ("oom" if n_oom else "empty"),
                    "cfg": dict(policy.current),
                }
                writer.append(rec_out)
                if rec_out["status"] == "ok":
                    n_ok += 1
                    if rec_out["caption_short"]:
                        n_short += 1
                else:
                    n_err += 1
            except Exception as e:
                logging.exception("error on shot %s", sid)
                writer.append({"shot_id": sid, "status": "error", "error": str(e)[:300]})
                n_err += 1

            # housekeeping
            if (shot_idx + 1) % 50 == 0:
                free_memory()
            if (shot_idx + 1) % 25 == 0:
                logging.info("g=%d progress %d/%d ok=%d short=%d oom=%d err=%d cfg=%s",
                             args.shard_id, shot_idx + 1, len(todo), n_ok, n_short, n_oom, n_err,
                             policy.current)

            # global cap check
            if args.max_samples > 0 and ((shot_idx + 1) % args.global_check_every == 0):
                global_ok = _count_global_pass(sp["stage5"], args.num_shards)
                if global_ok >= args.max_samples:
                    logging.info("g=%d global cap reached (%d >= %d); stopping",
                                 args.shard_id, global_ok, args.max_samples)
                    break
    finally:
        writer.close()
        try:
            del model, processor
        except Exception:
            pass
        free_memory()

    logging.info("step5 g=%d done: ok=%d short=%d oom=%d err=%d",
                 args.shard_id, n_ok, n_short, n_oom, n_err)


if __name__ == "__main__":
    main()
