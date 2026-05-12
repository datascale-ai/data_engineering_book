#!/usr/bin/env python3
"""Step 6: shot-language tagging.

Two parallel sub-tasks per shot, joined into one stage6 row:
  (a) Qwen2.5-VL with a strict-JSON prompt over a fixed controlled vocabulary
      (shot_size / camera_angle / composition / lighting / color_palette / style).
  (b) OpenCV global optical flow -> classify camera motion as
      static / pan_left / pan_right / tilt_up / tilt_down / zoom_in / zoom_out
      / jitter / complex (with motion_strength, pan_speed, tilt_speed, zoom, jitter).

Re-uses the per-shot frame jpgs already written by Step 5; falls back to
re-sampling from the segment mp4 if those are missing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from utils.flow_utils import classify_camera_motion
from utils.gpu_safety import (
    DegradePolicy,
    OOMExhausted,
    free_memory,
    safe_call,
    setup_cuda_alloc_env,
)
from utils.io_utils import iter_jsonl, shard_path, parse_shot_id, stage_paths
from utils.resume import SafeJsonlWriter, repair_tail, scan_done_ids
from utils.video_io import sample_frames_pil, save_pil_jpgs


VOCAB = {
    "shot_size": ["extreme_wide", "wide", "medium", "close_up", "extreme_close_up"],
    "camera_angle": ["eye_level", "high_angle", "low_angle", "dutch", "overhead"],
    "composition": ["rule_of_thirds", "centered", "symmetrical", "leading_lines", "framing", "negative_space"],
    "lighting": ["high_key", "low_key", "natural", "golden_hour", "backlit", "silhouette", "artificial", "mixed"],
    "color_palette": ["warm", "cool", "neutral", "monochrome", "saturated", "desaturated"],
    "style": ["cinematic", "documentary", "vlog", "commercial", "artistic"],
}

TAG_PROMPT = (
    "You are a film analyst. Given the following video frames sampled in time order from a single shot, "
    "classify the shot using strictly the controlled vocabulary below and return ONE JSON object only, "
    "no prose, no markdown fences.\n\n"
    "Allowed values per field:\n"
    + "\n".join([f"- {k}: one of {v}" for k, v in VOCAB.items()])
    + "\n\nIf you cannot decide, use the value \"unknown\". "
    "Output exactly this JSON schema (single line is fine):\n"
    '{"shot_size": "...", "camera_angle": "...", "composition": "...", '
    '"lighting": "...", "color_palette": "...", "style": "..."}'
)


def _frames_dir(out_root: Path, shot_id: str) -> Path:
    p = parse_shot_id(shot_id)
    return out_root / "frames" / f"pexels_{p['video_id']}" / f"shot_{p['idx']:04d}"


def _sanitize_json(text: str) -> dict:
    """Best-effort JSON extraction from a possibly-noisy VLM response."""
    if not text:
        return {}
    # strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text)
    # find first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    s = m.group(0) if m else text
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # fall back: regex per field
        out = {}
        for k in VOCAB:
            mm = re.search(rf'"{k}"\s*:\s*"([^"]+)"', text)
            if mm:
                out[k] = mm.group(1)
        return out


def _coerce_vocab(d: dict) -> dict:
    out = {}
    for k, allowed in VOCAB.items():
        v = str(d.get(k, "unknown")).strip().lower().replace(" ", "_")
        out[k] = v if v in allowed else "unknown"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="stage5_captions.jsonl (merged) preferred")
    ap.add_argument("--scenes", required=True, help="stage2_scenes.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--qwen-path", required=True)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--long-edge", type=int, default=448)
    ap.add_argument("--max-new-tokens", type=int, default=128)
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
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    sp = stage_paths(args.out)
    sp["stages"].mkdir(parents=True, exist_ok=True)

    seg_map = {r["shot_id"]: r["segment_path"] for r in iter_jsonl(args.scenes)}

    cap_rows = [r for r in iter_jsonl(args.inp) if r.get("status") == "ok"]
    shotlist = sorted(cap_rows, key=lambda r: r["shot_id"])
    mine = [r for i, r in enumerate(shotlist) if (i % args.num_shards) == args.shard_id]

    out_shard = shard_path(sp["stage6"], args.shard_id)
    repair_tail(out_shard)
    done = scan_done_ids(out_shard)
    todo = [r for r in mine if r["shot_id"] not in done and r["shot_id"] in seg_map]
    logging.info("step6 g=%d input=%d mine=%d done=%d todo=%d",
                 args.shard_id, len(shotlist), len(mine), len(done), len(todo))

    if not todo:
        return

    logging.info("g=%d loading Qwen2.5-VL", args.shard_id)
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
        {"frames_n": 4, "long_edge": 336, "max_new_tokens": 128},
        {"frames_n": 2, "long_edge": 336, "max_new_tokens": 96},
    ])

    @torch.inference_mode()
    def _generate(*, frame_paths: list, frames_n: int, long_edge: int, max_new_tokens: int):
        sel = frame_paths
        if len(sel) > frames_n:
            import numpy as np
            ix = np.linspace(0, len(sel) - 1, num=frames_n).round().astype(int).tolist()
            sel = [sel[i] for i in ix]
        msgs = [{
            "role": "user",
            "content": [
                {"type": "video", "video": [f"file://{p}" for p in sel]},
                {"type": "text", "text": TAG_PROMPT},
            ],
        }]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen_ids)]
        return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    writer = SafeJsonlWriter(out_shard)
    n_ok = n_oom = n_err = 0
    try:
        for shot_idx, rec in enumerate(todo):
            sid = rec["shot_id"]
            seg = seg_map[sid]

            # ---- (a) Qwen tags ----
            try:
                fdir = _frames_dir(sp["root"], sid)
                fpaths = sorted(str(p) for p in fdir.glob("f*.jpg"))
                if not fpaths:
                    pil_frames = sample_frames_pil(seg, k=args.frames, long_edge=args.long_edge)
                    if pil_frames:
                        fpaths = save_pil_jpgs(pil_frames, fdir)
                if not fpaths:
                    raw_tags = ""
                else:
                    try:
                        raw_tags = safe_call(
                            lambda **cfg: _generate(frame_paths=fpaths, **cfg),
                            policy=policy,
                        )
                    except OOMExhausted:
                        raw_tags = ""
                        n_oom += 1
                vlm_tags = _coerce_vocab(_sanitize_json(raw_tags or ""))
            except Exception as e:
                vlm_tags = _coerce_vocab({})
                logging.exception("vlm tag error on %s: %s", sid, e)

            # ---- (b) Camera motion via optical flow ----
            try:
                cm = classify_camera_motion(seg)
                cam = {
                    "class": cm.cls,
                    "motion_strength": cm.motion_strength,
                    "pan_speed": cm.pan_speed,
                    "tilt_speed": cm.tilt_speed,
                    "zoom_factor": cm.zoom_factor,
                    "jitter_score": cm.jitter_score,
                    "n_pairs": cm.n_pairs,
                }
            except Exception as e:
                cam = {"class": "unknown", "error": str(e)[:200]}

            writer.append({
                "shot_id": sid,
                "vlm_tags": vlm_tags,
                "camera_motion": cam,
                "status": "ok",
            })
            n_ok += 1
            if (shot_idx + 1) % 50 == 0:
                free_memory()
            if (shot_idx + 1) % 25 == 0:
                logging.info("g=%d step6 %d/%d ok=%d oom=%d err=%d cfg=%s",
                             args.shard_id, shot_idx + 1, len(todo), n_ok, n_oom, n_err, policy.current)
    finally:
        writer.close()
        try:
            del model, processor
        except Exception:
            pass
        free_memory()

    logging.info("step6 g=%d done: ok=%d oom=%d err=%d", args.shard_id, n_ok, n_oom, n_err)


if __name__ == "__main__":
    main()
