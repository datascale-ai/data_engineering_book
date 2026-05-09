#!/usr/bin/env python3
"""Step 4: CLIP ViT-L/14 + LAION-Aesthetic MLP score, sharded across 8 GPUs.

Per shot: sample K=4 evenly-spaced frames -> CLIP image_embeds (768-d, L2-norm)
-> MLP -> score in [1,10]. Average across the 4 frames; pass if >= threshold.

The MLP architecture matches /data0/improved-aesthetic-predictor/simple_inference.py:40-61
exactly so the published .pth state_dict loads cleanly.
"""

from __future__ import annotations

import argparse
import gc
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
from utils.io_utils import iter_jsonl, shard_path, stage_paths
from utils.resume import SafeJsonlWriter, repair_tail, scan_done_ids
from utils.video_io import sample_frames_pil


def _build_mlp(input_size: int = 768):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_size, 1024),
        nn.Dropout(0.2),
        nn.Linear(1024, 128),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.Dropout(0.1),
        nn.Linear(64, 16),
        nn.Linear(16, 1),
    )


def _load_aesthetic_mlp(pth_path: str, device, dtype):
    """Load LAION aesthetic MLP from a LightningModule-style checkpoint.

    The .pth's keys are 'layers.{0,2,4,6,7}.{weight,bias}' (shape 1024/128/64/16/1)
    because the upstream class wraps Sequential under self.layers. We strip the
    'layers.' prefix so it loads cleanly into a top-level Sequential."""
    import torch
    mlp = _build_mlp(768)
    sd = torch.load(pth_path, map_location="cpu", weights_only=True)
    sd = {(k[len("layers."):] if k.startswith("layers.") else k): v for k, v in sd.items()}
    mlp.load_state_dict(sd, strict=True)
    mlp.eval()
    mlp.to(device=device, dtype=dtype)
    return mlp


def _load_clip(clip_path: str, device, dtype):
    from transformers import CLIPImageProcessor, CLIPModel
    proc = CLIPImageProcessor.from_pretrained(clip_path)
    model = CLIPModel.from_pretrained(clip_path).to(device=device, dtype=dtype)
    model.eval()
    return proc, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="stage3_motion.jsonl (merged) OR pass shards")
    ap.add_argument("--scenes", required=True, help="stage2_scenes.jsonl (for shot->segment_path lookup)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--clip-path", required=True)
    ap.add_argument("--mlp-path", required=True)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--require-pass-motion", action="store_true",
                    help="if set, only score shots with pass_motion=True")
    args = ap.parse_args()

    setup_cuda_alloc_env()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES is set externally
    dtype = torch.bfloat16

    sp = stage_paths(args.out)
    sp["stages"].mkdir(parents=True, exist_ok=True)

    # build shot_id -> segment_path map from stage2 (kept regardless of motion result)
    seg_map = {}
    for r in iter_jsonl(args.scenes):
        seg_map[r["shot_id"]] = r["segment_path"]

    motion_rows = list(iter_jsonl(args.inp))
    # Optional: filter to only shots that pass motion. By default we still score
    # everything so the manifest is complete; downstream picks via pass_aesthetic.
    if args.require_pass_motion:
        motion_rows = [r for r in motion_rows if r.get("pass_motion")]

    # deterministic sharding: sort by shot_id, modulo on enumerate index
    shotlist = sorted(motion_rows, key=lambda r: r["shot_id"])
    mine = [r for i, r in enumerate(shotlist) if (i % args.num_shards) == args.shard_id]

    logging.info("step4 g=%d shard=%d/%d input_shots=%d mine=%d",
                 args.shard_id, args.shard_id, args.num_shards, len(shotlist), len(mine))

    out_shard = shard_path(sp["stage4"], args.shard_id)
    repair_tail(out_shard)
    done = scan_done_ids(out_shard)
    logging.info("resume: %d already done in shard", len(done))

    todo = [r for r in mine if r["shot_id"] not in done and r["shot_id"] in seg_map]
    logging.info("todo: %d", len(todo))

    if not todo:
        return

    # load models
    proc, clip = _load_clip(args.clip_path, device, dtype)
    mlp = _load_aesthetic_mlp(args.mlp_path, device, dtype)

    policy = DegradePolicy(configs=[
        {"batch": args.batch},
        {"batch": max(args.batch // 2, 1)},
        {"batch": max(args.batch // 4, 1)},
        {"batch": max(args.batch // 8, 1)},
        {"batch": 4},
        {"batch": 2},
        {"batch": 1},
    ])

    @torch.no_grad()
    def _score_batch(images_pil: list, *, batch: int):
        """Batch-encode pil images and return aesthetic scores (1D float list)."""
        n = len(images_pil)
        scores: List[float] = []
        for i in range(0, n, batch):
            chunk = images_pil[i : i + batch]
            inputs = proc(images=chunk, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
            feats = clip.get_image_features(pixel_values=pixel_values)  # (B, 768)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            preds = mlp(feats).squeeze(-1).float().detach().cpu().tolist()  # (B,)
            scores.extend(preds)
        return scores

    writer = SafeJsonlWriter(out_shard)
    n_done = 0
    n_pass = 0
    n_err = 0
    n_oom = 0
    try:
        for rec in todo:
            sid = rec["shot_id"]
            seg = seg_map.get(sid)
            try:
                frames = sample_frames_pil(seg, k=args.frames)
                if len(frames) == 0:
                    writer.append({
                        "shot_id": sid,
                        "aesthetic_score": 0.0,
                        "per_frame_scores": [],
                        "pass_aesthetic": False,
                        "status": "no_frames",
                    })
                    n_err += 1
                    continue
                try:
                    per_frame = safe_call(
                        lambda batch: _score_batch(frames, batch=batch),
                        policy=policy,
                    )
                except OOMExhausted:
                    n_oom += 1
                    writer.append({
                        "shot_id": sid,
                        "aesthetic_score": 0.0,
                        "per_frame_scores": [],
                        "pass_aesthetic": False,
                        "status": "oom",
                    })
                    free_memory()
                    continue
                avg = float(sum(per_frame) / len(per_frame))
                writer.append({
                    "shot_id": sid,
                    "aesthetic_score": avg,
                    "per_frame_scores": [float(x) for x in per_frame],
                    "pass_aesthetic": bool(avg >= args.threshold),
                    "status": "ok",
                })
                n_done += 1
                if avg >= args.threshold:
                    n_pass += 1
            except Exception as e:
                logging.exception("error on shot %s", sid)
                writer.append({
                    "shot_id": sid,
                    "aesthetic_score": 0.0,
                    "per_frame_scores": [],
                    "pass_aesthetic": False,
                    "status": "error",
                    "error": str(e)[:200],
                })
                n_err += 1
            if (n_done + n_err + n_oom) % 50 == 0:
                free_memory()
            if (n_done + n_err + n_oom) % 200 == 0:
                logging.info("g=%d progress %d/%d pass=%d oom=%d err=%d cur_batch=%s",
                             args.shard_id, n_done + n_err + n_oom, len(todo), n_pass, n_oom, n_err, policy.current.get("batch"))
    finally:
        writer.close()
        free_memory()

    logging.info("step4 g=%d done: scored=%d pass=%d oom=%d err=%d",
                 args.shard_id, n_done, n_pass, n_oom, n_err)


if __name__ == "__main__":
    main()
