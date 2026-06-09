# Project 14: Video Generation Dataset — From Video Sources to a T2V-Training-Ready Data Pipeline

## Abstract

This project constructs a reproducible data engineering case study around "Video Generation Dataset: From Video Sources to a T2V-Training-Ready Data Pipeline," with emphasis on business objectives, data boundaries, architectural decisions, core implementation, acceptance metrics, and risk controls. Installation commands and script details are consolidated into an engineering retrospective perspective, highlighting the relationships among sample schemas, data flows, failure modes, and deliverables — helping readers translate the methods presented earlier in this book into auditable, extensible project assets.

This project builds a production pipeline from public video sources to a trainable T2V dataset, covering six stages: video source loading, shot segmentation, motion filtering, aesthetic filtering, multi-frame captioning, and cinematic language annotation. The project goal is not to train a video generation model directly, but to organize raw video into shot-level samples, retaining for each sample its provenance, temporal boundaries, motion intensity, aesthetic score, structured caption, cinematic language tags, and quality status. Upon completing this project, readers should be able to describe the input, processing, output, and acceptance conditions for a T2V data sample; understand the engineering relationships among text, motion, shots, and licensing information; and evaluate this pipeline's boundaries with respect to scalability, compliance, and training compatibility.

**Keywords**: T2V data engineering; video generation dataset; shot segmentation; motion filtering; multi-frame captioning; cinematic language annotation

**Project Objectives**

- Organize public video sources into a traceable `source_videos.jsonl`.
- Use PySceneDetect to segment long videos into semantically coherent shot-level samples.
- Filter low-value clips via motion intensity, aesthetic scores, and quality labels.
- Use multi-frame sampling to generate captions covering subjects, actions, framing, and atmosphere.
- Output training candidate samples containing provenance, video segments, filter results, captions, and cinematic language tags.

## 1. Background and Objectives

Data engineering for text-to-video (T2V) generation models is generally more difficult than for text-to-image (T2I) generation. T2I data typically treats "single image — text description" as the basic unit, with data cleaning centered on image quality, text relevance, safety filtering, and resolution bucketing. T2V must deal with continuous frames: in addition to static visual content, training samples must capture motion changes, temporal ordering, camera movement, and scene continuity. Determining whether a video clip is suitable for training requires more than checking whether a single frame is sharp — the entire clip must be examined for valid motion, stable imagery, complete actions, and temporal consistency of subjects.

Accordingly, video must undergo finer processing before entering a training set. The pipeline first selects usable videos from public sources, proprietary crawled data, or stock footage libraries, then uses shot segmentation to decompose long videos into semantically coherent single-shot clips. Subsequent filtering based on optical flow, motion intensity, blur, jitter, watermarks, and OCR coverage removes low-quality clips, reducing the interference of static frames, severe jitter, and invalid motion on training. Video captions also cannot remain at the level of "what is in the frame" — they must clearly describe the sequence of actions, subject position changes, camera movement style, and cinematic language such as push/pull, pan, tilt, overhead, close-up, and wide shot.

This project covers the process from video sources to a trainable T2V dataset, transforming raw video into training samples with structured captions, spatiotemporal alignment information, and quality labels. Output samples record not only subjects and scenes, but also action sequences, spatial relationships, and cinematic presentation — providing the supervision needed for video generation models to learn "how to move" and "how to shoot."

## Keywords

Video generation dataset; project practice; reproducible data engineering; data pipeline; acceptance metrics

## Project Objectives and Reader Outcomes

This project uses the "video generation data pipeline" as its core case study, with the goal of organizing video sources, clip segmentation, captioning, quality scoring, and training packaging into T2V data assets. After completing this chapter, readers should be able to identify the key data objects in this scenario, decompose the engineering workflow, set acceptance metrics, and transfer the case methodology to similar data engineering tasks.

## Scenario Constraints and Data Boundaries

The focus is on public video samples and a small-scale pipeline, not covering full commercial copyright management or large-scale video platforms. These boundaries make the case reproducible and auditable; when data scale, data sources, permission scope, or deployment environment change, sampling strategy, quality thresholds, operational cost, and compliance requirements must be reassessed.

## Architectural Decisions

This project follows an architectural path of "video acquisition, segmentation and frame extraction, subtitle/description generation, quality scoring, deduplication and filtering, and T2V sample packaging." This decision prioritizes well-defined input/output contracts, version traceability, anomaly localization, and result verifiability, rather than compressing all logic into a single one-off script execution.

## Sample Schema / Data Flow

The core data flow can be summarized as:

```text
Video sources -> Clip segmentation -> Frames/subtitles/motion features -> Caption and quality scoring -> Filtering and deduplication -> T2V training samples
```

The sample schema should retain at minimum the fields `id`, `source`, `content_or_payload`, `metadata`, `quality_signals`, `split_or_stage`, and `audit_trace`; specific fields are further refined by the data types, downstream tasks, and acceptance methods of this project.

## Core Implementation Excerpts

The main text retains only the key implementation excerpts that illustrate design trade-offs. Complete scripts, long configurations, execution logs, and large files should be placed in the companion repository or appendix; code presentation focuses on input/output contracts, quality thresholds, exception handling, and acceptance interfaces.

## Experimental or Acceptance Metrics

Acceptance metrics include clip usability rate, caption consistency, motion/sharpness distribution, deduplication rate, safety filter rate, cost per minute, and training packaging completeness. If the project enters production, a course, or a public reproducibility experiment environment, version numbers, dependency environments, random seeds, sample spot-check results, and failure sample post-mortem records should also be logged.

## Cost, Risk, and Compliance Boundaries

Costs arise primarily from video downloading, transcoding, visual model inference, and storage; risks center on copyright, portrait rights, sensitive content, and generative model misuse. When external data, personal information, copyrighted content, or third-party services are involved, source documentation, permission status, anonymization strategy, call records, and manual review records must be retained.

## Common Failure Modes

Common failures include input distribution drift, missing schema fields, quality thresholds that are too loose or too strict, insufficient evaluation sample coverage, unstable model calls, and non-reproducible results. Troubleshooting should prioritize locating data boundaries and intermediate artifacts before examining models, toolchains, and deployment environments.

## Reproducibility Resources

Reproducibility materials should include data source descriptions, minimal samples, configuration files, run commands, metric scripts, inspection reports, and artifact directories. The main text retains necessary excerpts; complete notebooks, long scripts, and large files are maintained separately as companion resources.

## 2. Architecture Design: A Six-Component Video Generation Data Pipeline

This section focuses on how to organize a rerunnable, auditable, and extensible T2V data production pipeline on top of public video. The input side is a batch of Pexels open-license videos; the output side is shot-level samples for video generation training, where each sample simultaneously contains a video clip, provenance information, motion intensity, aesthetic score, multi-frame caption, cinematic language tags, and camera motion information. This data structure is more detailed than a typical video classification dataset, because T2V models learn the joint correspondence among text, frames, motion, and shots — the target is no longer a single category label. This section is based on six processing scripts: video source loading, shot segmentation, motion filtering, aesthetic filtering, multi-frame captioning, and cinematic language annotation.

Figure P14-1 shows the English-annotated architecture diagram for this project. The upper half of the diagram represents data stages; the lower half represents engineering controls. This layout deliberately places "caption generation" in the later stage rather than treating it as the sole core: the quality of a video generation dataset depends on the collective coordination of source, segmentation, motion, aesthetics, captioning, cinematic language, and release gates.

![P14 Video Generation Data Pipeline](../../images/part14/p14_video_generation_pipeline_en.png)

*Figure P14-1: English architecture diagram of the video generation data pipeline*

The entire pipeline can be broken down into six components. The first is **video source loading**. It reads a Pexels manifest or local video filenames, re-probes each video for duration, fps, resolution, frame count, and file size, and writes the author, page URL, and license fields into a unified manifest. This component provides the foundational information for subsequent provenance tracing, resolution statistics, duration statistics, and authorization status checks.

The second is **shot segmentation**. T2V models typically do not train directly on raw long videos, because long videos may contain multiple shots, scene cuts, and semantic discontinuities. The pipeline uses PySceneDetect's ContentDetector to detect shot boundaries, then uses ffmpeg to split videos into single-shot clips; interface and parameter details should follow the official PySceneDetect documentation (PySceneDetect Contributors 2026). Each clip is assigned a `shot_id` and records its start and end timestamps, parent video, clip index, and local path. All subsequent filtering, captioning, and shot tagging are organized around `shot_id` as the primary key.

The third is **motion filtering**. Video generation models need to learn temporal change, so the training set must not contain too many static clips. This component computes the Farneback (Farnebäck 2003) optical flow mean at a proxy resolution, uses `motion_strength` to measure the motion intensity within a clip, and generates `pass_motion` via a threshold. This step does not perform action semantic classification; it primarily distinguishes "dynamic clips with training value" from "nearly static image-like clips."

The fourth is **quality filtering**. Passing motion filtering does not guarantee a clip is suitable for training. Blurry, compressed, exposure-abnormal, or poorly composed video pollutes the visual distribution of a generative model. This project uses CLIP ViT-L/14 (Radford et al. 2021) to extract multi-frame visual features, then feeds them into the LAION-Aesthetic MLP (Schuhmann et al. 2022) to predict aesthetic scores. The clip-level score is obtained by averaging across multiple frames, avoiding single-frame randomness while preserving the overall visual impression of the clip.

The fifth is **multi-frame captioning**. T2V captions typically need to cover subjects, scenes, actions, camera framing, lighting, and atmosphere. A single-frame description can only state "what is in the frame" and struggles to convey "how the action unfolds." Therefore, the pipeline samples multiple frames in temporal order and passes them to Qwen2.5-VL (Bai et al. 2025) or InternVL3 (Zhu et al. 2025) to generate a video-level caption, with a minimum word count and retry mechanism to reduce excessively short descriptions.

The sixth is **cinematic language annotation**. A regular caption describes content; a cinematic language tag describes the shooting style. This component uses a controlled vocabulary to annotate shot size, camera angle, composition, lighting, color, and style on one hand; on the other, it uses optical flow to estimate camera motion, such as static, pan, tilt, zoom, jitter, and complex. The final sample retains both "what happens in the video" and "how it was filmed."

Table P14-1 summarizes the artifacts of each stage. The publication manuscript must retain these filenames, as they serve as the index by which readers map the main text, code, and reproduced artifacts to one another.

| Stage | Code Entry Point | Primary Input | Primary Output | Key Fields |
| --- | --- | --- | --- | --- |
| Video source loading | `load_pexels.py` | `pexels_manifest.jsonl` or `pexels_*.mp4` | `source_videos.jsonl` | `video_id`, `path`, `license`, `duration`, `fps`, `width`, `height` |
| Shot segmentation | `scene_detect.py` | `source_videos.jsonl` | `stage2_scenes.jsonl`, `shots/` | `shot_id`, `start_ts`, `end_ts`, `segment_path` |
| Motion filtering | `motion_filter.py` | `stage2_scenes.jsonl` | `stage3_motion.jsonl` | `motion_strength`, `n_pairs`, `pass_motion` |
| Aesthetic filtering | `aesthetic_filter.py` | `stage3_motion.jsonl`, `stage2_scenes.jsonl` | `stage4_aesthetic.jsonl` | `aesthetic_score`, `per_frame_scores`, `pass_aesthetic` |
| Multi-frame captioning | `caption_with_vlm.py` | `stage4_aesthetic.jsonl`, `stage2_scenes.jsonl` | `stage5_captions.jsonl`, `frames/` | `caption_en`, `n_words`, `caption_short`, `frame_paths` |
| Cinematic language annotation | `shot_language_tagger.py` | `stage5_captions.jsonl`, `stage2_scenes.jsonl` | `stage6_shot_language.jsonl` | `vlm_tags`, `camera_motion`, `status` |
| Final manifest construction | `utils.build_manifest` | Per-stage JSONL | final manifest | Provenance, clips, filters, captions, shot tags, and audit information |

*Table P14-1: Stage artifacts and field contracts of the video generation data pipeline*

An important implication of Table P14-1 is that this project does not store intermediate results temporarily in memory but saves each stage as JSONL or directory assets. This approach consumes somewhat more disk space, but yields three engineering benefits. First, a failure can be resumed from the last completed stage; second, spot-checks can trace back to a specific `shot_id`, source video, and frame image; third, subsequent safety filtering, deduplication, resampling, and WebDataset packaging can iterate independently without re-running the VLM.

---

## 3. Step-by-Step Implementation: From Pexels Videos to a Trainable T2V Dataset

### Step 1: Load 1,000+ Videos from the Pexels Open-Source Subset

The video loading stage first establishes a reliable source data manifest, without yet entering training or filtering. Pexels (Pexels 2014) video files are typically already downloaded to a local directory, accompanied by a `pexels_manifest.jsonl`. The manifest stores video IDs, page URLs, author information, and local save paths; if the manifest is missing, minimal records can also be recovered from `pexels_*.mp4` filenames. To avoid relying on stale metadata from the download stage, the script re-runs `ffprobe` on each mp4 to fill in duration, fps, width, height, nb_frames, and file_size. The resulting `source_videos.jsonl` serves as the stable entry point for the downstream pipeline.

```python
from pathlib import Path
import json

def load_source_videos(src_dir: Path) -> list[dict]:
    manifest = read_jsonl(src_dir / "pexels_manifest.jsonl")
    if not manifest:
        manifest = [{"saved_as": str(p), "video_id": parse_id(p)} for p in src_dir.glob("pexels_*.mp4")]

    records = []
    for raw in manifest:
        video_path = resolve_video_path(raw, src_dir)
        info = ffprobe(video_path)
        if info is None:
            continue
        records.append(normalize_video_record(raw, video_path, info))
    return records
```

Two points deserve attention here. First, all source videos are organized into structurally consistent JSONL rows; subsequent stages no longer scan the mp4 directory directly but instead read from this manifest. Second, the loading process supports checkpoint resumption: `video_id` entries already written are not reprocessed, and new videos are only appended to the end of the file. At the scale of 1,000+ videos, this approach is more robust than a full rerun and makes it easier to remove corrupted videos mid-process.

---

### Step 2: Shot Segmentation with PySceneDetect

T2V training samples are typically organized by shot rather than by the original video boundaries. A Pexels video may have a single long shot or may contain multiple cut points. Without segmentation, captions easily conflate multiple scenes, causing text-frame mismatches during training. This step uses PySceneDetect's ContentDetector for shot detection (PySceneDetect Contributors 2026) and treats the entire video as a single shot when no boundary is detected. The segmentation stage also filters out excessively short clips — shots shorter than one second are generally discarded.

```python
from scenedetect import open_video, SceneManager, ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg

def split_one_video(record: dict, out_root: Path, min_shot_len: float = 1.0):
    video = open_video(record["path"])
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=27.0))
    manager.detect_scenes(video=video, show_progress=False)

    scenes = manager.get_scene_list() or [(video.base_timecode, video.duration)]
    kept = [s for s in scenes if seconds(s[1] - s[0]) >= min_shot_len]
    if not kept:
        return []

    shot_dir = out_root / "shots" / f"pexels_{record['video_id']}"
    split_video_ffmpeg(record["path"], kept, str(shot_dir / "shot_$SCENE_NUMBER.mp4"))
    return [build_shot_record(record, idx, scene, path) for idx, (scene, path) in enumerate(zip(kept, sorted(shot_dir.glob("*.mp4"))))]
```

In practice, if 1,000 videos average 8 to 15 shots each, approximately 10,000 clips are produced. This figure is only a reference for experimental scale, not a fixed requirement. Note that PySceneDetect's threshold significantly affects the number of clips: a lower threshold yields finer segmentation; a higher threshold is more conservative. It is advisable to first inspect a sample of segmentation results and then fix the threshold, to avoid producing large numbers of semantically incomplete fragments.

---

### Step 3: Motion Detection to Filter Static Clips

After shot segmentation, the resulting clips are still only "candidate training clips" and cannot yet be treated as final training data. Among public videos, many clips are sharp but contain almost no motion — static landscapes, fixed product shots, posed subjects, and slow-motion stills. They offer limited benefit for T2V temporal modeling, and an excess of them will bias the model toward generating static video. Therefore, the third step uses optical flow to estimate clip motion intensity.

Rather than performing complex action recognition, the script simply computes the average optical flow magnitude between consecutive frames. Videos are scaled to a proxy resolution of 480×270, frames are sampled at a given stride with a cap on the maximum number of frame pairs to avoid slow processing of long videos. The final output is `motion_strength`, `n_pairs`, and `pass_motion`.

```python
def motion_filter_one(shot: dict, threshold: float = 0.5) -> dict:
    try:
        motion = compute_motion_magnitude(
            shot["segment_path"],
            proxy_wh=(480, 270),
            stride=2,
            max_pairs=60,
        )
        passed = motion.motion_strength >= threshold and motion.n_pairs > 0
        return {
            "shot_id": shot["shot_id"],
            "motion_strength": motion.motion_strength,
            "n_pairs": motion.n_pairs,
            "pass_motion": bool(passed),
            "status": "ok",
        }
    except Exception as exc:
        return failed_motion_record(shot["shot_id"], str(exc))
```

The motion threshold should not be fixed solely by intuition in a single pass. It is advisable to first compute the `motion_strength` distribution across all clips, then spot-check low-, mid-, and high-scoring samples. For typical public videos, starting experiments around a threshold of 0.5 is reasonable; if the data contains many slow-motion, natural scenery, or micro-motion clips, the threshold should be lowered to avoid discarding valid samples. At this stage, failed clips need not be immediately deleted — records with `pass_motion=False` can be retained and the decision deferred to the training stage.

---

### Step 4: CLIP Aesthetic Scoring and LAION-Aesthetic Filtering

Motion filtering examines whether temporal change exists in a clip; quality filtering examines whether the visual content is worth retaining. A clip may have strong motion yet suffer from overexposure, blur, severe compression, or disorganized composition. Generative models inherit the visual distribution of their training set, so quality scoring is needed to stratify samples.

This project uses CLIP ViT-L/14 to extract image features, then feeds them into the LAION-Aesthetic MLP to compute aesthetic scores. Each shot uniformly samples 4 frames, which are individually scored and then averaged. Compared to using only the first or middle frame, multi-frame averaging is more stable, since a video clip may contain brief blurriness, subject occlusion, or exposure variation.

```python
import torch
import torch.nn as nn

def build_aesthetic_mlp(input_size: int = 768) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, 1024), nn.Dropout(0.2),
        nn.Linear(1024, 128), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.Linear(64, 16), nn.Linear(16, 1),
    )

@torch.no_grad()
def score_shot_aesthetic(segment_path, clip_model, clip_processor, aesthetic_mlp):
    images = sample_frames_pil(segment_path, k=4)
    if not images:
        return {"aesthetic_score": 0.0, "pass_aesthetic": False, "status": "no_frames"}
    feats = encode_clip_images(images, clip_model, clip_processor)
    scores = aesthetic_mlp(torch.nn.functional.normalize(feats, p=2, dim=-1))
    avg = float(scores.squeeze(-1).mean().cpu())
    return {"aesthetic_score": avg, "pass_aesthetic": avg >= 5.0, "status": "ok"}
```

Engineering implementation must also handle multi-GPU sharding and GPU memory degradation. In this project, scripts use deterministic sharding: samples are first sorted by `shot_id`, then assigned by index modulo `num_shards`, with each GPU processing only its own shard. This sharding approach avoids redundant computation and facilitates resumption from a specific shard upon failure. Aesthetic thresholds should not be used solely as deletion criteria. Scores can be written into the manifest first, with sampling weights or data buckets set by score at the training stage.

---

### Step 5: Multi-Frame Sampling and Caption Generation with Qwen2.5-VL / InternVL3

After motion and quality filtering, the retained shots require video-level captions. The caption here serves as the language supervision signal in T2V training, not merely a brief title for the video. It should cover subjects, scenes, actions, camera framing, lighting and color, and overall atmosphere. To avoid frame-by-frame enumeration, the prompt explicitly requires a single English paragraph output and disallows enumeration such as "frame 1," "frame 2."

In implementation, the script samples 8 frames from each shot in temporal order and saves them to `frames/pexels_<video_id>/shot_<idx>/`. Saving frames allows verification of caption inputs during review, and also allows Step 6's cinematic language annotation to reuse the same frame set, avoiding redundant video decoding.

```python
CAPTION_PROMPT = """
Write one English paragraph describing the whole shot: subjects, setting,
actions, camera framing, lighting, color mood, and atmosphere. Do not enumerate frames.
""".strip()

@torch.inference_mode()
def generate_video_caption(frame_paths, model, processor, frames_n=8):
    selected = sample_frames_in_time_order(frame_paths, k=frames_n)
    messages = [{"role": "user", "content": [
        {"type": "video", "video": [f"file://{p}" for p in selected]},
        {"type": "text", "text": CAPTION_PROMPT},
    ]}]
    inputs = build_vlm_inputs(messages, processor).to("cuda")
    gen_ids = model.generate(**inputs, max_new_tokens=220, do_sample=False)
    caption = decode_new_tokens(gen_ids, inputs.input_ids, processor)
    return {"caption_en": caption, "n_words": len(caption.split()), "caption_short": len(caption.split()) < 50}
```

If a caption is too short on the first attempt, increasing the temperature and retrying up to twice is acceptable, but unlimited retries are not recommended. Excessive retries may make captions longer without improving accuracy. For training data, it is preferable to retain the `caption_short` flag and handle it uniformly in a post-processing stage, preventing the model from fabricating details not present in the frame in order to fill word count. InternVL3 integration is similar to Qwen2.5-VL — only the model loading and input organization interfaces need to be replaced; the data-level workflow remains "sample multiple frames in temporal order → generate a single video description."

---

### Step 6: Cinematic Language Annotation — Camera Movement, Composition, and Lighting

Multi-frame captioning focuses on describing video content; cinematic language annotation supplements the shooting style. This step uses two parallel paths. The first path has the VLM output structured tags from a controlled vocabulary, covering shot size, camera angle, composition, lighting, color, and style. The second path estimates camera motion from optical flow, outputting categories such as static, pan, tilt, zoom, jitter, or complex. After merging both components, the sample carries both a semantic caption and a cinematic language tag set.

```python
VOCAB = {
    "shot_size": ["extreme_wide", "wide", "medium", "close_up"],
    "camera_angle": ["eye_level", "high_angle", "low_angle", "overhead"],
    "composition": ["rule_of_thirds", "centered", "symmetrical", "framing"],
    "lighting": ["high_key", "low_key", "natural", "backlit"],
    "style": ["cinematic", "documentary", "vlog", "commercial"],
}

def tag_shot_language(shot_id: str, segment_path: str, frame_paths: list[str]) -> dict:
    raw = vlm_generate_json(frames=frame_paths, allowed_vocab=VOCAB)
    tags = sanitize_and_coerce_to_vocab(raw, VOCAB)
    motion = classify_camera_motion(segment_path)
    return {
        "shot_id": shot_id,
        "vlm_tags": tags,
        "camera_motion": summarize_camera_motion(motion),
        "status": "ok",
    }
```

Using a controlled vocabulary is strongly recommended here, to avoid letting the model generate tags freely. Free-text tags appear richer but are difficult to use for retrieval, bucketing, and training sampling; controlled tags have a limited expressive range but group similar samples under consistent fields. For example, `close_up`, `medium`, and `wide` can be used directly for shot-size stratification; `golden_hour`, `backlit`, and `low_key` can be used for lighting distribution statistics; `pan_left`, `zoom_in`, and `jitter` can be used to construct camera motion control samples. In T2V training, these structured fields enter engineering workflows more readily than a polished but uncontrollable prose description.

After completing Step 6, the final sample can be organized as follows:

```python
final_sample = {
    "shot_id": "pexels_123456_shot_0003",
    "source": {"video_id": 123456, "license": "pexels", "page_url": "https://..."},
    "video": {"segment_path": "shots/pexels_123456/shot_0003.mp4", "start_ts": 12.48, "end_ts": 17.92},
    "filters": {"motion_strength": 1.37, "pass_motion": True, "aesthetic_score": 6.21},
    "caption": {"caption_en": "A person walks through a sunlit urban street...", "caption_short": False},
    "shot_language": {
        "shot_size": "medium",
        "camera_angle": "eye_level",
        "lighting": "natural",
        "style": "cinematic",
        "camera_motion": "pan_right",
    },
}
```

This structure already has the basic form of trainable data. Safety filtering (NSFW), OCR/watermark filtering, deduplication, class resampling, and WebDataset packaging can all be added subsequently. The key principle to keep in mind throughout this pipeline is the sample organization approach: video samples progressively accumulate learnable supervision signals at each stage, and are finally organized into training data. Shot segmentation provides temporal boundaries, motion filtering provides dynamic quality, aesthetic filtering provides visual quality, multi-frame captioning provides semantic supervision, and cinematic language annotation provides shooting control information. Used together, these fields make it substantially easier for T2V models to establish stable "text—action—shot" correspondences.

## 4. Engineering Execution: Resumable Processing, Sharding, and GPU Memory Degradation

The code in this project is not a one-off notebook but is organized as a production pipeline. `run_pipeline.sh` chains the six stages into an end-to-end workflow, controlled via environment variables for data directory, output directory, GPU count, sample limit, and model paths. Default configuration includes `ROOT`, `OUT`, `SRC`, `N_GPU`, `MAX_SAMPLES`, `CLIP_PATH`, `MLP_PATH`, and `QWEN_PATH`. These variables should be written into the experiment record to avoid retaining only the final JSONL while losing the runtime environment.

Listing P14-1 shows the core form of the run entry point. This command is not the only deployment method, but it illustrates the minimum engineering boundary of this project: input directory, output directory, GPU count, sample limit, and model paths must all be provided explicitly.

```bash
ROOT=/data0/book_code \
OUT=/data0/book_code/stage1_output \
SRC=/data0/book \
N_GPU=8 \
MAX_SAMPLES=5000 \
CLIP_PATH=/data0/vit-large \
MLP_PATH=/data0/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth \
QWEN_PATH=/data0/qwen-vl \
bash run_pipeline.sh
```

This command serves to fix the runtime context rather than to demonstrate all parameters. A true production run must also record the versions of CUDA, PyTorch, Transformers, PySceneDetect, ffmpeg, CLIP weights, Qwen2.5-VL weights, and the LAION-Aesthetic MLP weights.

Three categories of engineering controls in the script deserve individual explanation. The first is **checkpoint resumption**. Each stage uses `repair_tail` to fix potentially corrupted JSONL tails, `scan_done_ids` to scan already-completed `video_id` or `shot_id` entries, and `SafeJsonlWriter` to append new results. This design allows scripts to resume after a mid-run failure without discarding already-processed clips.

The second is **sharded execution**. CPU stages use multi-process workers; GPU stages partition samples deterministically by `shard-id` and `num-shards`. Aesthetic scoring, captioning, and cinematic language annotation can all be distributed across multiple GPUs running in parallel. Each GPU writes to an independent shard, which is subsequently merged via `utils.merge_shards`. This avoids multiple processes writing to the same file simultaneously and makes it easy to identify which GPU or shard encountered an anomaly.

The third is **GPU memory degradation**. `aesthetic_filter.py`, `caption_with_vlm.py`, and `shot_language_tagger.py` all incorporate `DegradePolicy` and `safe_call`. When an OOM error occurs, the script progressively reduces batch size, frame count, maximum resolution, or generation length rather than terminating the entire pipeline. For video tasks, this is important because the frame complexity and model input length vary greatly across shots, and a statically fixed batch size is easily disrupted by individual outlier samples.

Table P14-2 summarizes the key runtime parameters and their effects.

| Parameter | Default or Example | Scope of Effect | Tuning Recommendation |
| --- | --- | --- | --- |
| `scene_detect.py --threshold` | `27.0` | Number and completeness of detected shots | Inspect a sample of segmentation results before fixing the threshold |
| `scene_detect.py --min-shot-len` | `1.0` seconds | Whether short clips are retained | T2V training typically discards excessively short shots |
| `motion_filter.py --threshold` | `0.5` | Recall rate of dynamic clips | Lower for slow-motion or scenic footage |
| `aesthetic_filter.py --frames` | `4` | Stability and compute cost of aesthetic scoring | Increase frame count when quality varies widely |
| `aesthetic_filter.py --threshold` | `5.0` | Visual quality gate | Recommend retaining scores first, then bucketing for training |
| `caption_with_vlm.py --frames` | `8` | Temporal information in captions | Increase for clips with complex action |
| `caption_with_vlm.py --min-words` | `50` | Caption verbosity | Word count cannot substitute for factual accuracy |
| `MAX_SAMPLES` | `5000` | Experiment scale | Set a small value first for teaching or smoke testing |

*Table P14-2: Key runtime parameters of the video generation data pipeline*

## 5. Quality Acceptance and Release Gates

Accepting a video generation dataset cannot be reduced to "checking whether captions were generated." For T2V training, at minimum, provenance, clips, motion, visual quality, text, cinematic tags, and safety boundaries must all be simultaneously verified. Table P14-3 provides the publication-grade acceptance criteria for this project.

| Acceptance Dimension | Metric / Evidence | Pre-Release Check |
| --- | --- | --- |
| Provenance compliance | `license`, `page_url`, `author_name`, source file path | Randomly sample and trace back to original pages; confirm authorization and attribution fields are complete |
| Segmentation quality | Shot duration distribution, empty clip rate, segmentation failure log | Spot-check whether boundaries cross scenes or produce excessively short fragments |
| Motion quality | `motion_strength` distribution, `pass_motion` ratio | Inspect low- and high-scoring samples; verify threshold does not incorrectly discard slow-motion clips |
| Visual quality | `aesthetic_score` distribution, proportion of anomalous status | Separately spot-check low-scoring, high-scoring, and boundary samples |
| Caption quality | `n_words`, `caption_short`, manual consistency spot-check | Check for frame enumeration, fabricated subjects, or missing actions |
| Cinematic language | Controlled vocabulary hit rate, `unknown` proportion, camera motion distribution | Verify tags are usable for retrieval, bucketing, and control training |
| Resumability | Shard files, logs, `_DONE` markers, merge results | Confirm that failed reruns do not produce duplicate writes or lost samples |
| Safety boundary | NSFW, watermark, OCR, portrait rights, and sensitive scene checks | Safety filter records must be supplemented before publication or public release |

*Table P14-3: Publication acceptance checklist for the video generation data pipeline*

During acceptance, it is recommended to divide samples into three spot-check groups. The first group consists of samples that pass all filters, used to confirm final training set quality. The second group consists of samples just below threshold — boundary cases used to determine whether the threshold is too strict. The third group consists of failed samples, used to determine whether failures originate from the data itself, model calls, GPU OOM, missing paths, or corrupted JSONL. Inspecting only passing samples while ignoring failed ones consistently leads to overestimating pipeline quality.

## 6. Common Failures and Localization Paths

Common issues in P14 can be localized by stage. If the count in `source_videos.jsonl` is anomalous, first check whether `pexels_manifest.jsonl` exists, whether video paths are accessible, and whether `ffprobe` is installed. If `stage2_scenes.jsonl` is empty, first check the PySceneDetect threshold, video encoding format, and ffmpeg split output. If the motion filter pass rate is too low, examine `proxy_w`, `proxy_h`, `stride`, and the threshold, rather than immediately concluding that the videos are invalid.

GPU-stage failures typically fall into three categories. The first is incorrect weight paths, such as `CLIP_PATH`, `MLP_PATH`, or `QWEN_PATH` not existing. The second is insufficient GPU memory — in this case, observe whether the log shows `DegradePolicy` has already reduced to a smaller batch, lower resolution, or fewer frames. The third is single-sample anomalies, such as a video that cannot be frame-sampled, a corrupted frame file, or an unexpected VLM response format. For the third category, the entire shard should not be allowed to fail — the affected sample should be marked `status="error"` and deferred to a subsequent post-mortem.

When final manifest construction fails, the cause is typically not a model issue but inconsistent primary keys across stages. All stages must use `shot_id` as the join key. If a given stage used a different naming convention, downstream joins will produce missing fields. Troubleshooting should start from `stage2_scenes.jsonl` and check, stage by stage, the count of `shot_id` entries, duplicates, and missing values.

## 7. Final Manifest and Training Integration

After the first six stages are complete, the project must merge the distributed provenance, clip, filter, caption, and cinematic language fields into a manifest that can be consumed by the training side. This step is handled in the code by `utils.build_manifest`. Its responsibility is not to generate new content but to organize multi-stage evidence into a stable sample contract. For T2V training, the final manifest must at minimum answer five questions: where is the video clip, what is the text supervision, do the quality signals pass, what are the cinematic language tags, and can the sample be traced back to its source.

Table P14-4 presents the recommended field structure for the final manifest.

| Field Group | Fields | Source Stage | Purpose |
| --- | --- | --- | --- |
| Identity fields | `shot_id`, `video_id`, `idx` | scene detect | Sample primary key and join key |
| Provenance fields | `license`, `page_url`, `author_name`, `source_path` | source load | Authorization, attribution, and retraction |
| Temporal fields | `start_ts`, `end_ts`, `duration` | scene detect | Training clip boundaries |
| File fields | `segment_path`, `frame_paths` | scene detect / caption | Training read access and manual spot-checks |
| Motion fields | `motion_strength`, `pass_motion`, `camera_motion` | motion / shot tags | Dynamic quality and camera motion control |
| Visual quality | `aesthetic_score`, `pass_aesthetic`, `per_frame_scores` | aesthetic filter | Quality bucketing and sampling weights |
| Text supervision | `caption_en`, `n_words`, `caption_short` | VLM caption | T2V text conditioning |
| Cinematic language | `shot_size`, `camera_angle`, `lighting`, `style` | shot tags | Control training and retrieval |
| Audit fields | `status`, `error`, `run_id`, `model_versions` | all stages | Post-mortem and release gates |

*Table P14-4: Recommended field structure for the T2V final manifest*

When integrating with training, it is not advisable to immediately write all fields into the model input. A more prudent approach is to use the manifest in three layers. The first layer is training-essential fields, including `segment_path` and `caption_en`. The second layer is sampling control fields, including `motion_strength`, `aesthetic_score`, `shot_size`, and `camera_motion`. The third layer is audit fields, including provenance, authorization, run batch, and error status. The training data loader needs to read only the first layer and part of the second; the auditing system and data dashboard require the complete set of fields.

Table P14-5 presents three common training integration approaches.

| Integration Approach | Data Organization | Applicable Scenario | Notes |
| --- | --- | --- | --- |
| JSONL manifest + local video | JSONL pointing to `segment_path` | Single-machine teaching, internal experiments | Manifest must be rewritten if paths are migrated |
| WebDataset tar shards | `.tar` containing video, caption, metadata | Large-scale distributed training | Shard size, sample ordering, and index must be fixed |
| Object storage URI | Manifest pointing to S3/OSS/HDFS URI | Multi-machine training and platform deployments | Requires permissions, caching, and failure retry |

*Table P14-5: Training integration approaches for video generation datasets*

If the training framework only accepts a simple "video path + caption" format, the remaining fields should not be discarded. A narrow-format training file can be generated while retaining the wide-format audit manifest. The narrow format serves training throughput; the wide format serves data governance. Both are linked via `shot_id`.

## 8. Deliverable Directory and Version Freezing

P14 deliverables should not consist solely of the final manifest. A video data pipeline involves video clips, sampled frames, model outputs, quality scores, and logs — any missing component will impair reproducibility. Table P14-6 presents the recommended directory structure.

| Path | Contents | Publication Recommendation |
| --- | --- | --- |
| `source_videos.jsonl` | Source video manifest and ffprobe metadata | Publish a de-identified version |
| `shots/` | Single-shot clips | Determine based on Pexels or internal licensing |
| `frames/` | Sampled frames for each shot | Usable for spot-checks; confirm authorization before public release |
| `stages/stage2_scenes.jsonl` | Shot segmentation results | Publishable |
| `stages/stage3_motion.jsonl` | Motion filtering results | Publishable |
| `stages/stage4_aesthetic.jsonl` | Aesthetic scoring results | Publishable |
| `stages/stage5_captions.jsonl` | Multi-frame caption results | Publish after spot-check review |
| `stages/stage6_shot_language.jsonl` | Cinematic language and camera motion tags | Publishable |
| `logs/` | Per-stage logs and GPU shard logs | Publish a de-identified version |
| `manifest/final_manifest.jsonl` | Final sample manifest for training | Core release artifact |
| `reports/quality_report.md` | Quality, failure, and spot-check report | Core release evidence |
| `reports/license_audit.md` | Provenance, authorization, and takedown mechanism description | Required for public release |

*Table P14-6: Deliverable directory for the video generation data pipeline*

Version freezing must include at minimum four categories of information. The first is code version, including the commit or release package version of each script. The second is model version, including the path and weight hash for CLIP, the LAION-Aesthetic MLP, Qwen2.5-VL, or InternVL3. The third is tool version, including ffmpeg, PySceneDetect, OpenCV, Transformers, and PyTorch. The fourth is threshold version, including scene threshold, motion threshold, aesthetic threshold, minimum word count, and sampling frame count. Without these records, even with access to the same batch of videos, it is very difficult to reproduce the same set of training samples.

## 9. Data Dashboard and Continuous Iteration

Before a video dataset enters training, it is advisable to establish a lightweight data dashboard. The dashboard need not be a complex system — even a set of statistical scripts and a Markdown report can significantly improve review efficiency. Table P14-7 presents the recommended dashboard metrics.

| Dashboard Metric | Computed From | Purpose |
| --- | --- | --- |
| Number of video sources | `source_videos.jsonl` | Check source material coverage |
| Shot duration distribution | `stage2_scenes.jsonl` | Determine whether segmentation is too fragmented or too coarse |
| Motion distribution | `stage3_motion.jsonl` | Adjust the proportion of dynamic samples |
| Aesthetic distribution | `stage4_aesthetic.jsonl` | Adjust the visual quality threshold |
| Caption length distribution | `stage5_captions.jsonl` | Identify excessively short or templated descriptions |
| `unknown` tag proportion | `stage6_shot_language.jsonl` | Check controlled vocabulary and VLM tag quality |
| Failure status counts | Per-stage `status` field | Locate systemic failures |
| Spot-check pass rate | Manual review records | Decide whether to release |

*Table P14-7: Dashboard metrics for the video generation data pipeline*

During continuous iteration, avoid adjusting thresholds based solely on final training outcomes. The more prudent approach is to first observe distributions on the data dashboard, then manually spot-check boundary samples, and finally validate with a small-scale training run. Directly adjusting thresholds by large amounts based on model performance risks attributing training hyperparameter issues to data filtering, or concealing data biases within model metrics.

## 10. Sample Retraction and Copyright Response

Video data requires retraction mechanisms more urgently than plain text or single-image data. A video clip may involve the original author's license, platform permissions, portrait rights, location information, trademarks, watermarks, and on-screen text. Even though this project uses Pexels open-license video as its example, publication should still preserve the path from every final sample back to the original page and author information.

Table P14-9 presents the video sample retraction workflow.

| Step | Action | Affected Objects |
| --- | --- | --- |
| Log the request | Record video URL, author, requester, reason, and timestamp | request ticket |
| Locate source video | Find via `video_id`, `page_url`, or file hash | `source_videos.jsonl` |
| Locate derived shots | Find all `shot_id` entries under the same `video_id` | `stage2_scenes.jsonl` |
| Delete intermediate artifacts | Remove corresponding clips, frames, captions, and tags | `shots/`, `frames/`, stages |
| Rebuild manifest | Re-merge final manifest and statistical reports | manifest / reports |
| Publish release note | Record deletion scope, new version number, and impact statistics | release note |

*Table P14-9: Video generation data sample retraction workflow*

The retraction workflow requires the final manifest to retain `video_id`, `page_url`, `author_name`, and `license`. If only `segment_path` and `caption_en` are stored, it becomes very difficult to confirm which original video a training sample came from. For public video platforms, authorization status may also change over time; therefore, published versions should freeze a provenance snapshot and specify the policy for responding to subsequent retraction requests.

## 11. Domain Transfer: From Public Video to Vertical Video Assets

The P14 pipeline can be transferred to scenarios such as e-commerce, education, industrial, medical, transportation, and film/advertising footage, but different domains define "trainable clips" differently. Public video emphasizes natural scenes and general motion; industrial video emphasizes defects, processes, and equipment motion; educational video emphasizes whiteboard content, gestures, and demonstration steps; transportation video emphasizes object trajectories, viewpoints, and safety events.

Table P14-10 presents the adjustment directions for domain transfer.

| Domain | Video Type | Stages Requiring Adjustment | Additional Acceptance |
| --- | --- | --- | --- |
| E-commerce | Product showcases, livestream clips | Watermark/OCR, product subject stability, brand authorization | Check for product attribute claims and exaggerated descriptions |
| Education | Lectures, whiteboards, lab demonstrations | OCR, spoken subtitles, step alignment | Copyright, answer leakage, and knowledge accuracy |
| Industrial | Production lines, equipment, defect video | Shot segmentation, slow-motion threshold, anomalous action labels | Internal permissions and expert defect review |
| Transportation | Dashcam, intersection, drone video | Object trajectories, occlusion, weather and time-of-day labels | Privacy, license plate, and face anonymization |
| Medical | Surgery, imaging dynamics, rehabilitation motion | High-risk content filtering, expert captioning | Patient privacy and medical expert review |
| Film/Advertising | Raw footage, commercial clips | Cinematic language, style, copyright metadata | Copyright chain and redistribution scope |

*Table P14-10: Domain transfer considerations for the video generation data pipeline*

General thresholds should not be reused directly when transferring to a new domain. For example, many valid industrial videos may show low-speed mechanical motion — an excessively high `motion_strength` threshold would incorrectly discard them. Educational whiteboard videos may show relatively little frame change, yet OCR content and sequential narration carry training value. Transportation videos may have strong motion, yet privacy and safety filtering are more critical. Accordingly, thresholds should be determined jointly by domain spot-checks and downstream training objectives.

## 12. Integration with the P13 Instruction Factory

P14 produces high-quality video shots and structured video metadata; P13 produces multimodal instruction data. The two can be combined into Video-Instruct or Video-QA datasets: use P14 to generate `segment_path`, `frame_paths`, `caption_en`, `shot_language`, and `camera_motion`, then use P13's templates, judges, multilingual expansion, and packaging mechanisms to produce instruction samples for video understanding or video generation control.

Table P14-11 presents the integration approach between the two.

| P14 Field | How P13 Can Use It | Example Task |
| --- | --- | --- |
| `caption_en` | As base video description or answer draft | "Describe the video in detail." |
| `frame_paths` | As multi-frame visual input | "What changes from the first frame to the last frame?" |
| `camera_motion` | Construct cinematic language Q&A | "Is the camera panning or zooming?" |
| `shot_size` | Construct composition and framing tasks | "Identify the shot size and explain why." |
| `lighting` | Construct lighting style descriptions | "Describe the lighting and mood." |
| `motion_strength` | Control sample difficulty and sampling weights | High-dynamic samples used for action understanding |
| `aesthetic_score` | Control visual quality bucketing | High-quality samples used for generation training |

*Table P14-11: Integration of P14 video fields into the P13 instruction factory*

This integration avoids rebuilding video instruction data from scratch. P14 handles video material and temporal quality; P13 handles instruction diversity and language quality. If the scope later expands to video generation control training, the `shot_language` fields can also be converted into prompt conditions — for example, "a cinematic wide shot with natural lighting and a slow zoom-in over ocean cliffs" — and paired with video clips for T2V training.

## Results and Analysis

| frame1 | frame2 |
|---|---|
| ![frame1](../../images/part10/14_f0.jpg) | ![frame2](../../images/part10/14_f1.jpg) |

*Table P14-12: Example of multi-frame sampling from a video clip*

The figure shows two sampled frames from the output data. The clip presents a coastal scene filmed from a high-altitude aerial perspective: deep blue water continuously crashes against rugged reef formations, white foam forms clearly at the edges of the dark rock faces, and sparse green vegetation distributed among the rocks preserves a degree of visual layering within the natural environment. The multi-frame caption covers the clip's subjects, scene, lighting, and atmosphere, describing natural illumination, cool-toned ocean surface, clear rock textures, and the dynamic quality imparted by wave motion. From the cinematic language annotation results, this shot is identified as `extreme_wide` shot size, `high_angle` camera angle, `rule_of_thirds` composition, `natural` lighting type, `cool` overall color tone, and `cinematic` style tag. The camera motion module classifies it as `zoom_in`, indicating a noticeable push-in or scale change in the frame; `motion_strength=0.8974` indicates that the clip carries a stable motion signal and is suitable for T2V training to learn natural scene motion, aerial perspectives, and coastal cinematic language.

For formal acceptance, the project must satisfy at least four conditions: first, all samples can be traced back through `shot_id` to the source video, author, page URL, and license status; second, video clip paths, start/end timestamps, and frame counts are consistent with the actual files; third, motion intensity, aesthetic scores, captions, and cinematic language tags can be read by downstream training or sampling scripts; fourth, samples with low quality, low motion, unclear provenance, or unverified licensing do not enter the formal training set. Only when all these conditions are simultaneously satisfied does the pipeline output meet the basic prerequisites for advancing from prototype validation to the training data candidate pool.

## Chapter Summary

This chapter uses the "video generation data pipeline" as a case study to demonstrate the engineering organization of video sources, clip segmentation, captioning, quality scoring, and training packaging into T2V data assets. The principal value of the case lies in placing task definition, data boundaries, architectural decisions, sample schema, metric acceptance, and reproducibility resources on a single chain — making the project not merely a sequence of operational steps but a verifiable case study.

The boundaries of this case must also be clearly retained. The focus is on public video samples and a small-scale pipeline, not covering full commercial copyright management or large-scale video platforms. In scenarios with larger scale, higher risk, or stricter compliance requirements, data sources, permission status, manual review proportions, operational costs, and failure rollback strategies must all be reassessed.

As part of Chapter 14, this chapter corresponds to the project-level validation of the methods presented earlier in this book. Readers may combine this case with the data recipes in Chapter 13, the platform governance chapters earlier in the book, and the checklists in the appendix to form a closed loop from methodological understanding to engineering delivery.

## References

PySceneDetect Contributors (2026) PySceneDetect Documentation. Available at: https://www.scenedetect.com/docs/latest/.

Bai S, Chen K, Liu X, Wang J, Ge W, Song S, Dang K, Wang P, Wang S, Tang J, others (2025) Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923.

Zhu J, Wang W, Chen Z, Liu Z, Ye S, Gu L, Duan Y, Tian H, Su W, Shao J, others (2025) InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models. arXiv preprint arXiv:2504.10479.

Farnebäck G (2003) Two-Frame Motion Estimation Based on Polynomial Expansion. In: Proceedings of the 13th Scandinavian Conference on Image Analysis, pp 363–370.

Pexels (2014) Pexels: Free Stock Photos, Royalty Free Images & Videos. Available at: https://www.pexels.com.

Radford A, Kim J W, Hallacy C, Ramesh A, Goh G, Agarwal S, Sastry G, Askell A, Mishkin P, Clark J, others (2021) Learning Transferable Visual Models from Natural Language Supervision (CLIP). In: Proceedings of the 38th International Conference on Machine Learning, pp 8748–8763.

Schuhmann C, Beaumont R, Vencu R, Gordon C, Wightman R, Cherti M, Coombes T, Katta A, Mullis C, Wortsman M, others (2022) LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models. In: Advances in Neural Information Processing Systems 35:25278–25294.
