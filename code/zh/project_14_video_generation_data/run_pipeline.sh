#!/usr/bin/env bash
# End-to-end Stage-1 video preprocessing pipeline (Pexels -> shots -> caption + tags).
#
# All steps are resumable: re-running this script picks up where it left off.
# GPU shards crash-recover via a 3-attempt supervisor; per-shot OOMs are absorbed
# inside each script via gpu_safety.safe_call.

set -euo pipefail

ROOT=${ROOT:-/data0/book_code}
OUT=${OUT:-${ROOT}/stage1_output}
SRC=${SRC:-/data0/book}
N_GPU=${N_GPU:-8}
MAX_SAMPLES=${MAX_SAMPLES:-5000}
CLIP_PATH=${CLIP_PATH:-/data0/vit-large}
MLP_PATH=${MLP_PATH:-/data0/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth}
QWEN_PATH=${QWEN_PATH:-/data0/qwen-vl}

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

mkdir -p "$OUT/stages" "$OUT/logs" "$OUT/shots" "$OUT/frames"

ts() { date +"%Y-%m-%d %H:%M:%S"; }

run_shard() {
  # $1 = python script  $2 = gpu_id  rest = extra args
  local script=$1 gpu=$2; shift 2
  local name; name=$(basename "$script" .py)
  local logf="$OUT/logs/${name}_g${gpu}.log"
  local attempt=0 max=3
  while [ $attempt -lt $max ]; do
    attempt=$((attempt+1))
    echo "[$(ts)] [sup] $name g=$gpu attempt=$attempt -> $logf" >&2
    if CUDA_VISIBLE_DEVICES=$gpu python "$script" \
        --shard-id $gpu --num-shards $N_GPU \
        "$@" >>"$logf" 2>&1; then
      return 0
    fi
    echo "[$(ts)] [sup] $name g=$gpu FAILED on attempt=$attempt; sleep 5" >&2
    sleep 5
  done
  echo "[$(ts)] [sup] $name g=$gpu gave up after $max attempts" >&2
  return 1
}

echo "[$(ts)] === Step 1: load_pexels ==="
python "$ROOT/load_pexels.py" --src "$SRC" --out "$OUT" 2>&1 | tee -a "$OUT/logs/step1.log"

echo "[$(ts)] === Step 2: scene_detect (8 CPU workers) ==="
python "$ROOT/scene_detect.py" \
  --src "$OUT/source_videos.jsonl" \
  --out "$OUT" \
  --workers 8 \
  --threshold 27.0 \
  --proxy-downscale 4 \
  --min-shot-len 1.0 \
  2>&1 | tee -a "$OUT/logs/step2.log"
python -m utils.merge_shards --stage stage2_scenes --out "$OUT" 2>&1 | tee -a "$OUT/logs/step2.log"

echo "[$(ts)] === Step 3: motion_filter (8 CPU workers) ==="
python "$ROOT/motion_filter.py" \
  --in "$OUT/stages/stage2_scenes.jsonl" \
  --out "$OUT" \
  --workers 8 \
  --threshold 0.5 \
  2>&1 | tee -a "$OUT/logs/step3.log"
python -m utils.merge_shards --stage stage3_motion --out "$OUT" 2>&1 | tee -a "$OUT/logs/step3.log"

echo "[$(ts)] === Step 4: aesthetic_filter (8 GPUs) ==="
pids=()
for g in $(seq 0 $((N_GPU-1))); do
  ( run_shard "$ROOT/aesthetic_filter.py" $g \
      --in "$OUT/stages/stage3_motion.jsonl" \
      --scenes "$OUT/stages/stage2_scenes.jsonl" \
      --out "$OUT" \
      --clip-path "$CLIP_PATH" \
      --mlp-path "$MLP_PATH" \
      --batch 64 --threshold 5.0 \
      --require-pass-motion ) &
  pids+=($!)
done
wait "${pids[@]}" || true
python -m utils.merge_shards --stage stage4_aesthetic --out "$OUT" 2>&1 | tee -a "$OUT/logs/step4.log"

echo "[$(ts)] === Step 5: caption_with_vlm (8 GPUs, max_samples=$MAX_SAMPLES) ==="
pids=()
for g in $(seq 0 $((N_GPU-1))); do
  ( run_shard "$ROOT/caption_with_vlm.py" $g \
      --in "$OUT/stages/stage4_aesthetic.jsonl" \
      --scenes "$OUT/stages/stage2_scenes.jsonl" \
      --out "$OUT" \
      --qwen-path "$QWEN_PATH" \
      --frames 8 --long-edge 448 --min-words 50 \
      --require-pass-aesthetic \
      --max-samples "$MAX_SAMPLES" --global-check-every 20 ) &
  pids+=($!)
done
wait "${pids[@]}" || true
python -m utils.merge_shards --stage stage5_captions --out "$OUT" 2>&1 | tee -a "$OUT/logs/step5.log"

echo "[$(ts)] === Step 6: shot_language_tagger (8 GPUs) ==="
pids=()
for g in $(seq 0 $((N_GPU-1))); do
  ( run_shard "$ROOT/shot_language_tagger.py" $g \
      --in "$OUT/stages/stage5_captions.jsonl" \
      --scenes "$OUT/stages/stage2_scenes.jsonl" \
      --out "$OUT" \
      --qwen-path "$QWEN_PATH" \
      --frames 4 --long-edge 448 ) &
  pids+=($!)
done
wait "${pids[@]}" || true
python -m utils.merge_shards --stage stage6_shot_language --out "$OUT" 2>&1 | tee -a "$OUT/logs/step6.log"

echo "[$(ts)] === Build manifest ==="
python -m utils.build_manifest --out "$OUT" 2>&1 | tee -a "$OUT/logs/manifest.log"

echo "[$(ts)] DONE."
