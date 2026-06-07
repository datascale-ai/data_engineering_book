# Project 14: Video Generation Dataset: From Video Source to T2V-ready Data Pipeline

## Abstract

This project builds a reproducible data-engineering case for a video generation dataset, from raw video sources to a T2V-ready training pipeline. It explains business goals, data boundaries, architecture decisions, core implementation, acceptance metrics, and risk control from an engineering-review perspective. The chapter highlights sample schema, data flow, failure modes, and deliverables so readers can turn video-data processing methods into auditable and extensible project assets.

## Keywords

video generation dataset; project practice; reproducible data engineering; T2V data pipeline; acceptance metrics

## Project Goals and Reader Takeaways

The goal is to organize video sources, shot splitting, captions, quality scoring, filtering, and training packaging into T2V data assets. After completing the project, readers should be able to identify the key data objects, split the pipeline into runnable stages, set acceptance metrics, and migrate the approach to related video-data engineering tasks.

## Scenario Constraints and Data Boundaries

The project uses public video samples and a small-scale pipeline. It does not cover full commercial copyright management or a large-scale video platform. These boundaries make the case reproducible and auditable. If data scale, source type, permission scope, or deployment environment changes, sampling strategy, quality thresholds, storage cost, and compliance requirements must be reassessed.

## Architecture Decision

The architecture follows video acquisition, shot splitting and frame sampling, caption generation, quality scoring, deduplication and filtering, and T2V sample packaging. This path prioritizes input/output contracts, version traceability, anomaly localization, and reviewable outputs instead of compressing all work into a one-off script.

## Sample Schema and Data Flow

The core flow can be summarized as:

```text
video source -> shot splitting -> frames/subtitles/motion features -> caption and quality scoring -> filtering/deduplication -> T2V training sample
```

At minimum, sample records should preserve `id`, `source`, `content_or_payload`, `metadata`, `quality_signals`, `split_or_stage`, and `audit_trace`. For T2V data, important additional fields include `shot_id`, start and end time, motion strength, aesthetic score, caption, camera motion, shot size, lighting, and license metadata.

## Core Implementation Slice

The main text keeps implementation fragments that explain design choices: source-manifest normalization, scene splitting, motion filtering, aesthetic scoring, multi-frame captioning, and shot-language tagging. Full scripts, long configurations, logs, and large video files should stay in the companion repository or artifact directory.

## Experimental or Acceptance Metrics

Acceptance metrics include usable-clip rate, caption consistency, motion and sharpness distribution, deduplication rate, safety-filter rate, cost per video minute, and training-package completeness. For production, course, or public reproduction settings, reports should also record version numbers, dependency environment, random seed, sample inspection results, and failed-sample review notes.

## Cost, Risk, and Compliance Boundaries

Cost mainly comes from video download, transcoding, visual-model calls, and storage. Risks concentrate on copyright, portrait rights, sensitive content, and misuse of generated video models. When external data, personal information, copyrighted content, or third-party services are involved, the project should retain source notes, permission status, desensitization strategy, call records, and human review records.

## Common Failure Modes

Common failures include source metadata loss, scene splitting that cuts actions midstream, motion thresholds that remove useful slow shots, caption models that ignore temporal order, aesthetic filters that overselect one visual style, and training packages that cannot trace clips back to source videos. Troubleshooting should first inspect data boundaries and intermediate artifacts before changing downstream model settings.

## Reproducible Resource Notes

Reproduction materials should include source descriptions, minimum video samples, configuration files, run commands, metric scripts, check reports, and artifact directories. The chapter keeps necessary fragments; full notebooks, long scripts, and large media files should be maintained separately.

## Background and Objectives

Text-to-video (T2V) data engineering is usually harder than text-to-image (T2I) data engineering. T2I data uses an image-text pair as the basic unit, and cleaning focuses on image quality, text alignment, safety filtering, and resolution bucketing. T2V data consists of continuous frames. Training samples must describe not only static visual content, but also motion, temporal order, camera movement, and scene continuity. Whether a video clip is useful for training cannot be judged from one clear frame alone; the whole clip must have meaningful motion, stable visuals, complete actions, and consistent subjects over time.

Before videos enter a training set, they therefore need more detailed processing. A practical pipeline first selects usable videos from public sources, internal captures, or stock libraries, then cuts long videos into semantically coherent single-shot clips. It then uses optical flow, motion strength, blur, shake, watermark, and OCR-area indicators to filter low-quality clips. Video captions also need to go beyond "what is in the frame." They should describe the order of actions, subject movement, camera motion, and cinematic language such as dolly, pan, overhead view, close-up, and wide shot.

This project turns raw videos into training samples with structured captions, temporal-spatial alignment, and quality tags. The output records subjects and scenes as well as action processes, spatial relations, and cinematic presentation, providing supervision for video generation models to learn both how things move and how shots are filmed.

## 2. Architecture: Six-component Video Generation Data Pipeline

This section organizes a rerunnable, auditable, and scalable T2V data production pipeline on public videos. The input is a batch of Pexels videos. The output is shot-level samples for video generation training. Each sample contains the clip, source metadata, motion strength, aesthetic score, multi-frame caption, shot-language tags, and camera-motion information. This structure is more detailed than ordinary video classification data because a T2V model learns joint correspondences among text, visuals, actions, and shots rather than a single class label.

The pipeline consists of six components. The first is **video source loading**. It reads a Pexels manifest or local filenames, reprobes duration, fps, resolution, frame count, and file size, and writes author, page link, and license fields into a unified manifest. This creates a base record for source tracing, resolution statistics, duration statistics, and license checks.

The second is **scene splitting**. T2V models usually do not train directly on raw long videos, because a long video may contain multiple shots, scene transitions, and semantic breaks. The pipeline uses PySceneDetect's ContentDetector (Castellano 2012) to detect scene boundaries and ffmpeg to cut videos into single-shot clips. Each clip receives a `shot_id` and records its start time, end time, source video, segment index, and local path.

The third is **motion filtering**. Video generation models need temporal change, so the training set should not contain too many static clips. The component computes Farneback optical-flow magnitude (Farneback 2003) at proxy resolution and uses `motion_strength` to decide `pass_motion`. It does not classify action semantics; it separates dynamic clips from nearly static image-like clips.

The fourth is **quality filtering**. A clip that passes motion filtering can still be blurry, over-compressed, overexposed, or poorly composed. This project uses CLIP ViT-L/14 (Radford et al. 2021) to extract multi-frame visual features, then feeds them to a LAION-Aesthetic MLP (Schuhmann et al. 2022). Clip-level score is the average over multiple frames.

The fifth is **multi-frame captioning**. A T2V caption should cover subjects, scene, action, camera framing, light, color, and atmosphere. A single-frame caption struggles to explain how an action unfolds, so the pipeline samples frames in temporal order and uses Qwen2.5-VL (Wang et al. 2024) or InternVL3 (Chen et al. 2024) to generate one video-level caption, with minimum word count and retry logic.

The sixth is **shot-language tagging**. Ordinary captions describe content; shot-language tags describe filming style. This component uses a controlled vocabulary for shot size, camera angle, composition, lighting, color, and style, and uses optical flow to estimate camera motion such as static, pan, tilt, zoom, jitter, and complex.

---

## 3. Step-by-Step Implementation: From Pexels Videos to a Trainable T2V Dataset

### Step 1: Load 1000+ Videos from an Open Pexels Subset

The source-loading stage builds a reliable manifest and does not yet perform training or filtering. Pexels (Pexels 2014) video files are typically downloaded locally and paired with `pexels_manifest.jsonl`. The manifest stores video ID, page link, author information, and local path. If the manifest is missing, minimal records can be recovered from filenames such as `pexels_*.mp4`. To avoid relying on stale download metadata, the script reprobes each mp4 with `ffprobe` to fill `duration`, `fps`, `width`, `height`, `nb_frames`, and `file_size`. The resulting `source_videos.jsonl` becomes the stable input for downstream stages.

```python
from pathlib import Path
import json


def read_pexels_manifest(src_dir: Path) -> list[dict]:
    manifest = src_dir / "pexels_manifest.jsonl"
    records = []

    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    # Recover minimal metadata from filenames when the manifest is missing.
    for video_path in sorted(src_dir.glob("pexels_*.mp4")):
        video_id = int(video_path.stem.split("_")[1])
        records.append({
            "video_id": video_id,
            "saved_as": str(video_path),
            "page_url": None,
            "user": {},
        })
    return records


def normalize_video_record(raw: dict, src_dir: Path) -> dict | None:
    video_id = raw.get("video_id")
    video_path = Path(raw.get("saved_as", ""))

    if not video_path.exists():
        candidates = list(src_dir.glob(f"pexels_{int(video_id)}_*.mp4"))
        if not candidates:
            return None
        video_path = candidates[0]

    info = ffprobe(video_path)
    if info is None:
        return None

    user = raw.get("user") or {}
    return {
        "video_id": int(video_id),
        "path": str(video_path),
        "page_url": raw.get("page_url"),
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
```

Two details matter. First, all source videos are normalized into consistent JSONL rows. Downstream stages read this manifest instead of scanning the mp4 directory directly. Second, loading supports resume: written `video_id` values are skipped, and new videos are appended. At 1000+ videos, this is more stable than rerunning everything.

---

### Step 2: Split Shots with PySceneDetect

T2V training samples are normally organized by shot, not by raw video boundary. A Pexels video may be one long shot or may contain multiple cuts. Without splitting, captions can merge multiple scenes and create text-video mismatch. We use PySceneDetect's ContentDetector (Castellano 2012) and keep the whole video as one shot if no boundary is found. Segments shorter than one second are usually discarded.

```python
from scenedetect import open_video, SceneManager, ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg


def detect_scenes(video_path: str, threshold: float = 27.0, downscale: int = 4):
    video = open_video(video_path)

    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))

    if downscale > 1:
        manager.auto_downscale = False
        manager.downscale = downscale

    manager.detect_scenes(video=video, show_progress=False)
    scenes = manager.get_scene_list()

    if scenes:
        return scenes, [
            (float(start.get_seconds()), float(end.get_seconds()))
            for start, end in scenes
        ]

    start = video.base_timecode
    end = video.duration
    if end is None or end == start:
        return [], []

    return [(start, end)], [(0.0, float(end.get_seconds()))]


def split_one_video(record: dict, out_root: Path, min_shot_len: float = 1.0):
    video_id = record["video_id"]
    video_path = record["path"]
    shot_dir = out_root / "shots" / f"pexels_{video_id}"
    shot_dir.mkdir(parents=True, exist_ok=True)

    scenes, time_ranges = detect_scenes(video_path)

    kept = [
        (scene, ts)
        for scene, ts in zip(scenes, time_ranges)
        if ts[1] - ts[0] >= min_shot_len
    ]

    if not kept:
        return []

    scenes = [x[0] for x in kept]
    time_ranges = [x[1] for x in kept]

    split_video_ffmpeg(
        input_video_path=video_path,
        scene_list=scenes,
        output_file_template=str(shot_dir / "shot_$SCENE_NUMBER.mp4"),
        arg_override="-map 0:v:0 -map 0:a? -c:v copy -c:a copy -avoid_negative_ts make_zero",
        show_progress=False,
    )

    rows = []
    for idx, segment_path in enumerate(sorted(shot_dir.glob("shot_*.mp4"))):
        start_ts, end_ts = time_ranges[idx]
        rows.append({
            "shot_id": f"pexels_{video_id}_shot_{idx:04d}",
            "video_id": video_id,
            "idx": idx,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "n_frames": int(round((end_ts - start_ts) * float(record.get("fps") or 30.0))),
            "segment_path": str(segment_path),
        })
    return rows
```

With 1000 source videos and an average of 8-15 shots per video, this stage can produce around 10,000 clips. The exact count is not fixed. PySceneDetect threshold strongly affects the number of segments: a lower threshold creates finer cuts; a higher threshold is more conservative. Sample and inspect results before fixing the threshold.

---

### Step 3: Filter Static Clips with Motion Detection

After scene splitting, clips are still only training candidates. Many public clips are clear but almost static, such as landscapes, product shots, posed portraits, or very slow fixed-camera scenes. Too many of them can bias a model toward static videos. The third step estimates motion strength with optical flow.

The script scales video to a 480 x 270 proxy resolution, samples frame pairs by stride, and caps maximum pairs to avoid slow long-video processing. It outputs `motion_strength`, `n_pairs`, and `pass_motion`.

```python
def motion_filter_one(
    shot: dict,
    threshold: float = 0.5,
    proxy_wh: tuple[int, int] = (480, 270),
    stride: int = 2,
    max_pairs: int = 60,
) -> dict:
    """
    Given one shot record, return motion-filtering results.
    compute_motion_magnitude may use Farneback optical flow:
    1. Decode frames.
    2. Resize to proxy_wh.
    3. Compute optical flow between sampled adjacent frames.
    4. Use average magnitude as motion_strength.
    """
    try:
        motion = compute_motion_magnitude(
            shot["segment_path"],
            proxy_wh=proxy_wh,
            stride=stride,
            max_pairs=max_pairs,
        )

        return {
            "shot_id": shot["shot_id"],
            "motion_strength": motion.motion_strength,
            "n_pairs": motion.n_pairs,
            "pass_motion": bool(
                motion.motion_strength >= threshold and motion.n_pairs > 0
            ),
            "status": "ok",
        }

    except Exception as e:
        return {
            "shot_id": shot["shot_id"],
            "motion_strength": 0.0,
            "n_pairs": 0,
            "pass_motion": False,
            "status": "error",
            "error": str(e)[:200],
        }
```

Do not set the motion threshold once by intuition. First plot the `motion_strength` distribution, then sample low-, medium-, and high-score clips. For ordinary public videos, a threshold around 0.5 is a reasonable starting point. For data dominated by slow motion, natural scenery, or subtle actions, lower the threshold to avoid deleting useful clips. It is often better to keep `pass_motion=False` records in the manifest and decide during training which buckets to use.

---

### Step 4: CLIP Aesthetic Scoring and LAION-Aesthetic Filtering

Motion filtering checks whether temporal change exists; quality filtering checks whether the visual content is worth keeping. A clip may have strong motion but still be overexposed, blurry, highly compressed, or poorly composed. Since generation models inherit the visual distribution of their training data, samples should be stratified by quality.

This project uses CLIP ViT-L/14 to extract features and a LAION-Aesthetic MLP to predict aesthetic score. Each shot samples four frames uniformly and averages their scores. Multi-frame averaging is more stable than using only the first or middle frame because a clip may contain brief blur, occlusion, or exposure changes.

```python
import torch
import torch.nn as nn


def build_aesthetic_mlp(input_size: int = 768) -> nn.Sequential:
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


@torch.no_grad()
def score_shot_aesthetic(
    segment_path: str,
    clip_model,
    clip_processor,
    aesthetic_mlp,
    frames: int = 4,
    threshold: float = 5.0,
    device: str = "cuda",
) -> dict:
    images = sample_frames_pil(segment_path, k=frames)
    if not images:
        return {
            "aesthetic_score": 0.0,
            "per_frame_scores": [],
            "pass_aesthetic": False,
            "status": "no_frames",
        }

    inputs = clip_processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    feats = clip_model.get_image_features(pixel_values=pixel_values)

    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    scores = aesthetic_mlp(feats).squeeze(-1).float().cpu().tolist()

    avg_score = float(sum(scores) / len(scores))
    return {
        "aesthetic_score": avg_score,
        "per_frame_scores": [float(x) for x in scores],
        "pass_aesthetic": bool(avg_score >= threshold),
        "status": "ok",
    }
```

The implementation should also handle multi-GPU sharding and memory fallback. A deterministic shard strategy is simple: sort by `shot_id`, take the row index modulo `num_shards`, and let each GPU process its own shard. This avoids duplicate computation and supports resume after failure. Aesthetic scores should not only be deletion gates; they can also become sampling weights or buckets during training.

---

### Step 5: Multi-frame Captioning with Qwen2.5-VL / InternVL3

After motion and quality filtering, retained shots need video-level captions. A T2V caption is language supervision, not a short title. It should cover subjects, setting, action, camera framing, lighting, color, and atmosphere. The prompt explicitly asks for one English paragraph and forbids enumerations such as "frame 1" and "frame 2".

The script samples eight frames in temporal order and saves them under `frames/pexels_<video_id>/shot_<idx>/`. Saved frames make caption review easier and can be reused by Step 6 shot-language tagging.

```python
CAPTION_PROMPT = """
You are a professional video captioner. The frames below are sampled in time order
from a single shot. Write ONE single-paragraph English caption of AT LEAST 60 words
describing this shot as a whole. Cover the main subjects, the setting, the actions
or movement, the camera framing, the lighting and color mood, and the overall
atmosphere. Do NOT enumerate frames. Output the caption text only.
""".strip()


@torch.inference_mode()
def generate_video_caption(
    frame_paths: list[str],
    model,
    processor,
    frames_n: int = 8,
    max_new_tokens: int = 220,
    min_words: int = 50,
    device: str = "cuda",
) -> dict:
    if len(frame_paths) > frames_n:
        import numpy as np
        indices = np.linspace(0, len(frame_paths) - 1, frames_n).round().astype(int)
        frame_paths = [frame_paths[i] for i in indices]

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": [f"file://{p}" for p in frame_paths]},
            {"type": "text", "text": CAPTION_PROMPT},
        ],
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)

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
        do_sample=False,
    )
    trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen_ids)]
    caption = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    n_words = len(caption.split())
    return {
        "caption_en": caption,
        "n_words": n_words,
        "caption_short": bool(n_words < min_words),
        "status": "ok" if caption else "empty",
    }
```

If the first caption is too short, increase temperature and retry twice, but avoid infinite retries. A longer caption is not necessarily more accurate. Keep the `caption_short` flag for batch post-processing, so the model is not pressured to invent details just to satisfy length.

---

### Step 6: Shot-language Tags for Camera Movement, Composition, and Lighting

Multi-frame captions describe content; shot-language tags describe filming style. This step uses two parallel paths. The first asks a VLM to output structured tags under a controlled vocabulary: shot size, camera angle, composition, lighting, color palette, and style. The second estimates camera motion from optical flow: static, pan, tilt, zoom, jitter, or complex. The merged sample records both semantic caption and cinematographic attributes.

```python
VOCAB = {
    "shot_size": [
        "extreme_wide", "wide", "medium", "close_up", "extreme_close_up",
    ],
    "camera_angle": [
        "eye_level", "high_angle", "low_angle", "dutch", "overhead",
    ],
    "composition": [
        "rule_of_thirds", "centered", "symmetrical",
        "leading_lines", "framing", "negative_space",
    ],
    "lighting": [
        "high_key", "low_key", "natural", "golden_hour",
        "backlit", "silhouette", "artificial", "mixed",
    ],
    "color_palette": [
        "warm", "cool", "neutral", "monochrome", "saturated", "desaturated",
    ],
    "style": [
        "cinematic", "documentary", "vlog", "commercial", "artistic",
    ],
}


def tag_shot_language(shot_id: str, segment_path: str, frame_paths: list[str]) -> dict:
    """
    Output structured shot-language tags:
    1. A VLM selects shot size, angle, composition, lighting, color, and style.
    2. An optical-flow module estimates camera motion.
    3. Both are merged into a stage-6 record.
    """

    raw_json = vlm_generate_json(
        frames=frame_paths,
        allowed_vocab=VOCAB,
        task="classify shot language with controlled vocabulary",
    )
    vlm_tags = sanitize_and_coerce_to_vocab(raw_json, VOCAB)

    camera_motion = classify_camera_motion(segment_path)
    camera_motion_record = {
        "class": camera_motion.cls,
        "motion_strength": camera_motion.motion_strength,
        "pan_speed": camera_motion.pan_speed,
        "tilt_speed": camera_motion.tilt_speed,
        "zoom_factor": camera_motion.zoom_factor,
        "jitter_score": camera_motion.jitter_score,
        "n_pairs": camera_motion.n_pairs,
    }

    return {
        "shot_id": shot_id,
        "vlm_tags": vlm_tags,
        "camera_motion": camera_motion_record,
        "status": "ok",
    }
```

A controlled vocabulary is recommended. Free-form tags look richer, but they are hard to use for retrieval, bucketing, and training sampling. Controlled tags are narrower but group similar samples into shared fields. For example, `close_up`, `medium`, and `wide` directly support shot-size stratification; `golden_hour`, `backlit`, and `low_key` support lighting statistics; and `pan_left`, `zoom_in`, and `jitter` support camera-control data construction.

After Step 6, a final sample can be organized as:

```python
final_sample = {
    "shot_id": "pexels_123456_shot_0003",
    "source": {
        "video_id": 123456,
        "license": "pexels",
        "page_url": "https://www.pexels.com/video/...",
        "author_name": "...",
    },
    "video": {
        "segment_path": "shots/pexels_123456/shot_0003.mp4",
        "start_ts": 12.48,
        "end_ts": 17.92,
        "n_frames": 163,
    },
    "filters": {
        "motion_strength": 1.37,
        "pass_motion": True,
        "aesthetic_score": 6.21,
        "pass_aesthetic": True,
    },
    "caption": {
        "caption_en": "A person walks through a sunlit urban street...",
        "n_words": 67,
        "caption_short": False,
    },
    "shot_language": {
        "shot_size": "medium",
        "camera_angle": "eye_level",
        "composition": "centered",
        "lighting": "natural",
        "color_palette": "warm",
        "style": "cinematic",
        "camera_motion": "pan_right",
    },
}
```

This structure is already close to trainable data. Later stages can add NSFW filtering, OCR/watermark filtering, deduplication, category resampling, and WebDataset packaging. The key idea is that each stage adds learnable supervision: scene splitting provides temporal boundaries, motion filtering provides dynamic quality, aesthetic filtering provides visual quality, multi-frame captions provide semantic supervision, and shot-language tags provide controllable filming information.

## Results and Analysis

| frame1 | frame2 |
|---|---|
| ![frame1](../../images/part10/14_f0.jpg) | ![frame2](../../images/part10/14_f1.jpg) |

The sampled frames show a coastline captured from a high aerial view. Deep blue water strikes rugged rocks, white foam forms along dark rock edges, and small patches of green vegetation add visual layers to the harsh natural setting. The multi-frame caption covers subject, scene, lighting, atmosphere, cool-toned water, clear rock texture, and wave motion. The shot-language tag identifies the clip as `extreme_wide`, `high_angle`, `rule_of_thirds`, `natural` lighting, `cool` color palette, and `cinematic` style. The camera-motion module classifies it as `zoom_in`; `motion_strength=0.8974` indicates stable motion signal suitable for learning natural-scene movement, aerial perspective, and coastline shot language in T2V training.

## References

Castellano B (2012) PySceneDetect: Python and OpenCV-based Scene Cut/Transition Detection Program. Available at: https://www.brettcastellano.com/post/pyscenedetect.

Chen Z, Wang W, Tian H, Ye S, Gao Z, Cui E, Tong X, Hu J, Luo J, Ma S, others (2024) InternVL3: Exploring Advanced Training and Test-Time Scaling for Vision-Language Models. arXiv preprint arXiv:2504.10479.

Farneback G (2003) Two-Frame Motion Estimation Based on Polynomial Expansion. In: Proceedings of the 13th Scandinavian Conference on Image Analysis, pp 363-370.

Pexels (2014) Pexels: Free Stock Photos, Royalty Free Images & Videos. Available at: https://www.pexels.com.

Radford A, Kim J W, Hallacy C, Ramesh A, Goh G, Agarwal S, Askell A, Mishkin P, Clark J, others (2021) Learning Transferable Visual Models from Natural Language Supervision. In: Proceedings of the 38th International Conference on Machine Learning, pp 8748-8763.

Schuhmann C, Beaumont R, Vencu R, Gordon C, Wightman R, Cherti M, Coombes T, Katta A, Mullis C, Wortsman M, others (2022) LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models. In: Advances in Neural Information Processing Systems 35:25278-25294.

Wang P, Bai S, Tan S, Wang S, Fan Z, Bai J, Chen K, Liu X, Wang J, Ge W, others (2024) Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv preprint arXiv:2409.12191.
