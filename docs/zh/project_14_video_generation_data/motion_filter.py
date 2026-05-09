#!/usr/bin/env python3
"""Step 3: optical-flow motion filter.

For each shot in stage2, compute Farneback mean magnitude at 480x270 and decide
whether the shot is sufficiently dynamic (`motion_strength >= threshold`).

Resumable per shot_id; sharded by worker for non-conflicting writes.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.flow_utils import compute_motion_magnitude
from utils.io_utils import iter_jsonl, shard_path, stage_paths
from utils.resume import SafeJsonlWriter, repair_tail, scan_done_ids


def _process_one(arg: dict) -> dict:
    sid = arg["shot_id"]
    seg = arg["segment_path"]
    threshold = float(arg["threshold"])
    proxy_w = int(arg["proxy_w"])
    proxy_h = int(arg["proxy_h"])
    stride = int(arg["stride"])
    max_pairs = int(arg["max_pairs"])
    try:
        mm = compute_motion_magnitude(
            seg,
            proxy_wh=(proxy_w, proxy_h),
            stride=stride,
            max_pairs=max_pairs,
        )
        return {
            "shot_id": sid,
            "motion_strength": mm.motion_strength,
            "n_pairs": mm.n_pairs,
            "pass_motion": bool(mm.motion_strength >= threshold and mm.n_pairs > 0),
            "status": "ok",
        }
    except Exception:
        return {
            "shot_id": sid,
            "motion_strength": 0.0,
            "n_pairs": 0,
            "pass_motion": False,
            "status": "error",
            "error": traceback.format_exc(limit=2),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="stage2_scenes.jsonl")
    ap.add_argument("--out", required=True, help="stage1_output directory")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--proxy-w", type=int, default=480)
    ap.add_argument("--proxy-h", type=int, default=270)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-pairs", type=int, default=60)
    args = ap.parse_args()

    sp = stage_paths(args.out)
    sp["stages"].mkdir(parents=True, exist_ok=True)

    shots = [r for r in iter_jsonl(args.inp) if Path(r.get("segment_path", "")).exists()]
    print(f"[step3] input shots: {len(shots)}")

    n_workers = max(1, int(args.workers))

    # repair shard tails and scan done ids across all shards
    done = set()
    for w in range(n_workers):
        sf = shard_path(sp["stage3"], w)
        repair_tail(sf)
        done |= scan_done_ids(sf)
    print(f"[step3] resume: {len(done)} shots already done")

    todo = [s for s in shots if s["shot_id"] not in done]
    print(f"[step3] todo: {len(todo)}")

    args_list = []
    for i, rec in enumerate(todo):
        worker_id = i % n_workers
        args_list.append(
            {
                "shot_id": rec["shot_id"],
                "segment_path": rec["segment_path"],
                "threshold": args.threshold,
                "proxy_w": args.proxy_w,
                "proxy_h": args.proxy_h,
                "stride": args.stride,
                "max_pairs": args.max_pairs,
                "worker_id": worker_id,
            }
        )

    # group by worker so each worker writes its own shard
    writers = {w: SafeJsonlWriter(shard_path(sp["stage3"], w)) for w in range(n_workers)}
    try:
        n_ok = n_pass = n_err = 0
        with Pool(processes=n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_process_one, args_list)):
                # we don't track which worker produced which result, so we just
                # round-robin distribute results to shard writers; that's fine
                # because the merge step dedupes by shot_id anyway.
                w_idx = i % n_workers
                writers[w_idx].append(res)
                if res.get("status") == "ok":
                    n_ok += 1
                    if res.get("pass_motion"):
                        n_pass += 1
                else:
                    n_err += 1
                if (i + 1) % 200 == 0:
                    print(f"[step3] processed {i+1}/{len(todo)} pass={n_pass} err={n_err}")
        print(f"[step3] done. ok={n_ok} pass={n_pass} err={n_err}")
    finally:
        for w in writers.values():
            w.close()


if __name__ == "__main__":
    main()
