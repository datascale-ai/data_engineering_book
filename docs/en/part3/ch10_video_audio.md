# Chapter 10: Video and Audio Data Engineering

<div class="chapter-authors">Ke Wang</div>

## Abstract

This chapter discusses the core problems of video and audio data engineering, emphasizing why long-temporal multimodal data is much harder than static image-text data in usable-sample ratio, decoding cost, temporal alignment, and quality evaluation. It first analyzes why video data "looks abundant but yields few usable samples," including dimensional growth, static redundancy, background noise, audio-video separation, and decoding I/O bottlenecks. It then builds a three-track parallel pipeline across visual, acoustic, and text streams: shot-boundary detection, key-frame extraction, ASR transcription, denoising, speaker diarization, subtitle correction, and timestamp alignment. The second half discusses event labels, audio-video mismatch detection, cost models, and hardware decoding strategies such as NVDEC and DALI. An anonymized composite case shows how temporal offset can destroy audio-video learning signals. Readers should be able to design auditable video and audio preprocessing pipelines that can slice, transcribe, align, and evaluate long-temporal samples.

## Keywords

Video data; audio data; ASR; WhisperX; shot-boundary detection; temporal alignment; NVDEC; multimodal quality evaluation

## Learning Objectives

- Explain the special challenges of video and audio data in dimensionality, redundancy, noise, and decoding cost.
- Design a three-track pipeline for shot segmentation, key-frame extraction, ASR, denoising, and speaker diarization.
- Explain why timestamps, audio-video synchronization, and cross-modal semantic consistency determine sample quality.
- Build video/audio quality evaluation, severity classification, and isolation strategies.
- Estimate the main cost sources in decoding, frame extraction, ASR, and packaging.

After the processing chain from natural-language text in Parts I and II to static image-text parsing in Chapters 8 and 9, this chapter enters **temporal video and audio data engineering**.

In training based on image-text pairs or frame snapshots, a model can learn object categories, scenes, and static relationships. It still struggles to understand the motion trajectory, sound-image synchronization, and temporal causality contained in an apple "falling from a table, rolling under a bed, and making an impact sound." To train models such as Sora (Brooks et al. 2024) or Gemini 1.5 Pro (Gemini Team 2024), which can process long temporal input, one must build samples that express time, action, and sound association.

That means the data-engineering problem expands from two-dimensional image-text data to time and audio dimensions. Cost, quality, and alignment difficulty all rise sharply.

## 10.1 Why Audio and Video Data Looks Abundant but Yields Few Usable Samples

Architects new to multimodal projects often assume that the web's daily supply of public video naturally forms a trainable data pool. In real pretraining preprocessing, however, teams often find that only a portion of the raw video library can enter the training framework. The usable ratio depends on source licensing, content type, resolution, audio-video synchronization, quality thresholds, and sampling standards, and cannot be estimated directly from raw storage volume.

The gap comes mainly from three sources.

### 10.1.1 The Dimensionality Problem: From 2D Space to 4D Spacetime

For a pure image, even a high-resolution 4K AnyRes input is mainly a two-dimensional tensor: $(W \times H \times C)$. Video adds a time dimension: $(T \times W \times H \times C)$, where $T$ is the number of timesteps. A one-minute short video at 30 FPS produces 1,800 consecutive images. If CLIP Score or visual-token compression is computed on every frame, GPU memory and compute quickly exceed budget. Video data engineering must therefore design a strict **key-frame sampling** system that keeps only a small number of key frames and discards a large amount of redundant frames. Systematic experiments (Zohar et al. 2025, CVPR) show that FPS-style rate sampling significantly outperforms fixed-frame-count sampling on long-video understanding tasks, and that this conclusion is consistent across model scales.

### 10.1.2 Surface Richness: Useless Oversampling and High Modal Noise

The disk may indeed contain a large amount of video, but a substantial portion may be:

1. **Static redundancy**: in a two-hour online lecture, one hour of video may show almost the same static slide background with a small face in the corner. Encoding thousands of nearly identical frames dilutes the training signal. This data adds little cognitive value and may reduce the effective sample ratio.
2. **Background noise and audio-video separation**: many lifestyle vlogs contain wind noise, mechanical hum, or scenes where the video shows someone playing golf while the background audio is pop music. For a model that must learn physical causality, such as matching breaking-glass video to breaking-glass sound, this wild audio-video data provides false supervision.

### 10.1.3 Underestimated Decoding Compute and Storage Bottlenecks

As discussed in Section 6.4, storing text requires only byte reads, while storing and dynamically loading long videos during training stresses the file system much more heavily.

Videos are usually stored in compressed formats such as H.264, H.265, or VP9. To extract pixel-frame sequences and audio samples usable by a model, the first loading step must decode the stream. When 100 H100 GPUs wait for batches, the CPU DataLoader and cluster I/O bandwidth can become bottlenecks because of concurrent MP4 decompression. This hardware scale is illustrative; production teams should re-benchmark on their own cluster.

---

## 10.2 A Three-Track Pipeline for Slicing, Transcription, and Temporal Alignment

For long-temporal data, the cleaning factory cannot reuse the old image-text-pair pattern of "one image, one sentence." We need an automated audio-video sample-building platform that separates and processes visual, acoustic, and text tracks.

![Figure 10-1: Distributed audio-video alignment pipeline](../../images/part3/av_sample_pipeline.png)

*Figure 10-1: Distributed audio-video alignment pipeline. Raw mixed videos in the video lake are split into visual and acoustic tracks. Visual-frame extractors and acoustic separators extract features independently before the streams meet in a temporal alignment engine, which produces aligned multimodal JSONL samples with closed timestamp constraints. Source: drawn for this book. Alt text: an audio-video alignment pipeline showing a raw video split into visual, audio, and text tracks and transformed into JSONL by a temporal alignment engine.*

### 10.2.1 Visual Extraction: Shot-Boundary Detection and Scene Slicing

Before training, very long videos, such as two-hour films, must be cut into clips of 10 to 30 seconds that are logically and visually continuous. Simple fixed-time splitting, such as cutting every 10 seconds, is undesirable because it may cut an action or sentence in half and create semantic incompleteness.

1. **Shot-boundary detection**

The visual pipeline needs a fast detection node, such as dual-threshold color-histogram comparison, with a high threshold for hard cuts and a low threshold for fades or dissolves, or lightweight optical-flow difference between adjacent frames. The goal is to capture hard and soft transitions caused by camera movement or editing. Only continuous frames within the same shot are suitable as one complete knowledge concept, or event grounding unit, for pretraining a visual model.

![Figure 10-2: Adaptive shot-boundary detection and semantic leakage prevention](../../images/part3/av_shot_boundary_hsv.png)

*Figure 10-2: Adaptive shot-boundary detection and semantic leakage prevention. The upper track extracts aggregated HSV-channel color-space differences, while the lower track extracts optical-flow pixel displacement to capture subtle motion posture. The two tensor differences flow into dual-threshold triage. When the jump score $\Delta$ exceeds the hard-cut threshold, the engine splits the clip and prevents semantic leakage across scenes. Source: drawn for this book. Alt text: adaptive shot-boundary detection showing HSV difference, optical-flow difference, and dual-threshold routing.*

2. **Adaptive subsampling**

After slicing, a 20-second shot may be logically continuous but visually almost static. The factory deploys small models to continuously measure the displacement between the current frame and the last retained frame in dense visual features, such as DINOv2 (Oquab et al. 2023) embeddings. Only frames whose Euclidean distance exceeds a preset threshold are retained. A slice that originally contains many adjacent frames is ultimately compressed into a smaller number of key frames. The compression ratio depends on frame rate, motion density, and threshold settings, and should be verified through sampled playback to ensure that key actions were not cut off.

### 10.2.2 Acoustic Separation: ASR, Denoising, and Speaker Diarization

The lower track running in parallel with visual frame extraction extracts acoustic semantics. It first performs **audio stripping**, then passes the stream through three filters.

#### A. Core semantic extraction: large-scale WhisperX ASR

For speech tracks, a common approach is to use open-source Whisper (Radford et al. 2023), WhisperX (Bain et al. 2023), or similar frameworks to transcribe accented, noisy, and pause-filled audio into structured text with timestamps.

![Figure 10-3: Large-scale ASR extraction and temporal calibration](../../images/part3/asr_whisperx_comparison.png)

*Figure 10-3: Large-scale ASR extraction and temporal calibration. Traditional ASR can suffer cumulative temporal drift and semantic errors, such as mishearing `I love apples.` as `maples.` WhisperX uses VAD slicing, multi-path acoustic decoding, and a DTW phoneme-level forced-alignment matrix for temporal calibration. The bottom shows word tokens aligned with waveform troughs through vertical dashed lines. Source: drawn for this book. Alt text: ASR extraction and temporal calibration showing traditional ASR drift, WhisperX calibration, and word-level timestamp alignment.*

#### B. Denoiser layer

Not every video has studio-grade isolation. Large amounts of field data contain strong wind noise or mechanical resonance. Demucs (Défossez et al. 2019) and deep-learning source-separation algorithms can separate background music, environmental noise, and vocals from reverberant spectra.

#### C. Speaker diarization: who is speaking?

For podcasts or multi-person meeting videos, compressing all speech into a single string prevents the model from distinguishing who asks and who answers. Diarization splits a long audio stream into speaker segments such as `[Speaker A]: 01:23-01:30` and `[Speaker B]: 01:31-01:40`.

#### D. LLM-driven subtitle correction

Raw ASR often misrecognizes specialized vocabulary in code, medicine, finance, and other domains. Industrial pipelines commonly add an LLM correction step after WhisperX output. A strong LLM receives timestamped raw ASR and a prompt such as "repair typos and punctuation according to context, but never change the original timestamps." This can reduce final word error rate. Concrete gains depend on language, noise, domain vocabulary, and ASR model version, and must be evaluated on a small sample with human-transcribed gold labels.

### 10.2.3 Multi-Track Temporal Alignment: Binding Subtitles, Speech, and Frames

After the visual key-frame array, ASR subtitles, and audio waveform are collected, the hard part is binding these signals to the same timeline: **cross-modal geometric and temporal lock**.

An ASR subtitle may say "Hello World!", but for a 10-second temporal segment, which milliseconds, frames, and lip shapes correspond to that sound must be specified with temporal anchors. Without this binding, a model cannot learn audiovisual synchronization or lip-motion prediction.

![Figure 10-4: Cross-modal temporal calibration and geometric alignment](../../images/part3/av_alignment_diagram.png)

*Figure 10-4: Cross-modal temporal calibration and geometric alignment. The cyan top track is visual key frames, the gray middle track is acoustic features, and the coral bottom track is discrete text tokens. At `t=4.2s`, the temporal lock binds the visual action "raising a cup," waveform features near the trough, and `<start:4.2s> "Water cup"` text into a unified mixed-token pipeline or JSONL sample. Source: drawn for this book. Alt text: a cross-modal temporal calibration diagram showing visual frames, audio waveform, and text tokens bound by a shared timeline anchor.*

Large teams usually deploy a **multi-modal temporal alignment engine** based on timestamp matrices. Once a front-end recognizer emits coordinate bounds such as `<start:2.1s><end:4.5s>`, code must use floating-point logic to slice the corresponding video frames. The final alignment information is not handed to the model only as video; it is transformed into a structured JSONL sequence with metadata tags and HTML-like multi-track mixed tokens, then passed to the training DataLoader.

---

## 10.3 Event Labeling and Evaluation Funnels

Although Section 10.2 binds visual and audio tracks in time, these raw structured samples still need higher-level event-grounding signals and mismatch detection before entering the pretraining engine.

### 10.3.1 Event Detection and Grounding

A wild video should not contain only frames and ASR text. It also needs a description of physical-world action flow. Inside large pipelines, behavior-understanding video models such as LLaVA-Video (Zhang et al. 2024), Video-LLaMA (Zhang H et al. 2023), and similar side models are called asynchronously on aligned short clips.

These models should provide not only a one-sentence summary such as "a young man tries a backflip at a skate park and falls," but also **dynamic event tags** and **detailed temporal captions**:

1. **Coarse event tags**: structured labels such as `[Sports]`, `[Skateboarding]`, `[Accident]`, and `[Impact_Sound]`, which help data mixing.
2. **Fine-grained dense video captions**:
   - `<time: 01.2s-03.5s>`: the boy runs up and uses momentum to enter the U-shaped ramp...
   - `<time: 03.5s-05.1s>`: he tries to complete a 360-degree spin in the air, but his back loses balance...
   - `<time: 05.1s-06.8s>`: his back hits the concrete ramp heavily, producing a dull low-frequency impact sound.

These reinforced labels and category tags, which include cause, process, and result, are injected into the aligned JSONL built in the previous section, giving video samples explicit spatiotemporal semantics.

### 10.3.2 Audio-Video Mismatch Detection

One of the most serious alignment failures in long-temporal data is unrelated audio and video. For example, the video may show a quiet giraffe eating grass while the editor has mixed in electronic music or unrelated game commentary. If such samples enter foundation-model training, the model may incorrectly associate giraffes with unrelated music or commentary and form cross-modal hallucinations.

Strict mismatch detection and review are therefore required.

*Table 10-1: Temporal audio-video data defects and multi-layer detection/remediation strategies. Source: compiled by the authors; detection and remediation strategies are engineering patterns, and thresholds should be calibrated through sampled playback and downstream evaluation.*

| Defect type and manifestation | Root cause | Detection and remediation | Severity |
| :--- | :--- | :--- | :--- |
| **Severe audio-video mismatch**: a silent forest view while the speech track explains an FPS game. | Incorrect secondary editing or audio track bleeding during automated encoding. | **Compute feature cosine scores with pretrained discriminators**: extract CLIP visual vectors from middle frames and semantic vectors from speech/audio. If cross-modal similarity is below the warning threshold, isolate the segment and discard that time-window label. | P0: must not enter the data lake |
| **Flicker, black screen, or extreme mosaic** | Original bitrate is too low, or transmission loss is severe. | Calculate brightness-histogram range and sharpness score, such as Laplacian variance, over the segment. If long-term black screen or blur overflow is detected, block it, log the anomaly, and trace frame extraction and decoding operators. | P1: isolate for review |
| **Background noise overwhelms speech** | Broken microphone or high-frequency mechanical noise that cannot be separated. | Use a small model on the full-band spectrogram for SNR estimation. Speech tracks below the floor threshold should be downweighted or removed; dialogue projects usually discard them. | P2: depends on use case |

---

## 10.4 Cost Model, Quantitative Design, and Throughput Optimization

Compared with pure-text processing, long-temporal multimodal pipelines sharply increase costs for cloud GPUs, object storage, network bandwidth, and decoding.

In text processing, Markdown or web-body parsing is usually far lighter than video decoding. In video cleaning, even at an illustrative scale such as 10,000 hours of HD MP4 files, reading and decoding video into tensors for feature extraction quickly consumes CPU, memory, PCIe, and storage bandwidth. This scale is illustrative; actual throughput depends on video codec, resolution, concurrency, and hardware.

### 10.4.1 Decoder Compute and I/O Bandwidth

The key question is what hardware should decode video frames.

1. **Limits of pure CPU software decoding**: early designs may use high-end CPUs, multithreaded ffmpeg, or Python OpenCV. Under high concurrency, memory movement and PCIe/RAM bandwidth quickly become bottlenecks.
2. **Hardware video decoders such as NVDEC**: a more scalable solution is to offload decoding to dedicated hardware. Calling the GPU's video decoding module, such as the NVDEC API, reduces CPU decoding pressure and improves throughput. Although GPU instances cost more, they are often the core cost-reduction method at large cleaning scale.

### 10.4.2 Audio-Video Quality Assessment

To decide whether a decompressed video is worth sending to the next stage, we need an automated quality-metric set:

- **Aesthetic and sharpness score**: score extracted key frames with models such as LAION-Aesthetic and filter heavy blur or mosaic.
- **Motion blur and optical-flow overload**: a violently shaking camera has high optical-flow displacement variance, lowering visual encoding quality; such clips should be removed or downweighted.
- **SNR and clipping ratio**: detect whether environmental noise masks speech and remove harsh distorted clips.

### 10.4.3 Industrial Processing Cost Breakdown

Data engineers need a clear view of unit cost at each layer.

*Table 10-2: Long-temporal audio-video processing cost drivers and cost-reduction strategies. Source: compiled by the authors; cost drivers should be recalculated according to cloud pricing, hardware specifications, concurrency limits, and caching strategies.*

Note: Table 10-2 does not provide universal cost shares, because actual bills depend on cloud pricing, GPU model, video resolution, sampled frame rate, ASR model, cache hit rate, and object-storage billing. A more robust engineering approach is to identify cost drivers first, then run small-batch stress tests in the target environment.

| Processing stage | Resource profile | Cost drivers | Cost-reduction strategy |
| :--- | :--- | :--- | :--- |
| **1. Raw long-stream crawling and block download** | High-bandwidth network and massive object-storage block I/O. | Cross-region traffic, object-storage request count, cache hit rate | Add edge caching gateways and preload fragments to fast NVMe near the GPU to avoid direct reads from slow storage. |
| **2. Hardware decoding and intelligent frame extraction** | NVDEC hardware decoding module, GPU memory, and PCIe bandwidth pressure. | Resolution, codec, sampled frame rate, concurrent decoding lanes | Use DALI or hardware decoding instead of Python OpenCV; combine shot-boundary detection and key-frame filtering to avoid decoding useless frames. |
| **3. ASR and dense recaptioning with WhisperX/LLaVA** | High memory consumption and GPU-intensive inference. | Audio duration, model size, batching efficiency, domain-correction strategy | Use quantized models and dynamic batching to reduce padding waste. |
| **4. Sequence merge and package writing** | Backend NAS/S3 small-file concurrent write pressure. | Shard size, small-file count, checksums, and indexing strategy | Use WebDataset TAR format and aggregate into GB-scale contiguous shards to reduce small-file overhead. |

---

## 10.5 Anonymized Composite Case

### 10.5.1 Postmortem of a Large-Scale Video Data Pipeline Failure

The following is an anonymized composite case used to illustrate the risk profile of missing temporal calibration. In one internal video project, a team accumulated a large amount of HD mixed video material, but the dataset construction work ultimately fell short of expectations.

The root cause was that the architecture skipped several critical temporal-calibration steps. The audio-feature separation interface had a reading offset bug. After multiple slicing and merging operations, this offset accumulated so that in later-stage slices, the actor's audio track systematically led or lagged lip motion and physical action.

When this temporally misaligned data was used for model training, audio-video association ability noticeably degraded: in benchmark tests, whenever the model saw a specific human action, it produced an acoustic prediction unrelated to the image.

This case reinforces the core conclusion of Chapter 1: **without strict data preprocessing, algorithmic investment cannot compensate for fundamental data defects**.

### 10.5.2 Chapter Summary and Bridge

From text cleaning in Parts I and II, to image-pixel alignment in Chapters 8 and 9, and now long-temporal video and audio data, we have systematically covered preprocessing methods for heterogeneous data: video frame extraction, ASR transcription, audio-video alignment, event labeling, and quality filtering.

Video and audio pipelines solve slicing, transcription, and temporal synchronization for long-temporal samples. Multimodal training still needs to answer another question: how do image, text, audio, and video signals form stable correspondences in the same semantic space? The next chapter, **Chapter 11: Cross-Modal Alignment and Fusion**, discusses object-level, segment-level, and document-level aligned sample construction.

---

## 10.6 Appendix: Frequent Audio-Video Pipeline Error Logs and Troubleshooting

> The following anonymized error logs cover five core links in large-scale audio-video preprocessing: I/O, decoding, ASR alignment, diarization, and storage writing. Host names, paths, batch IDs, and metrics are illustrative and do not correspond to public incidents. Each category includes root-cause analysis and remediation, followed by a quick-reference table.

### 10.6.1 I/O Avalanche: S3 Concurrent Streaming Limit Causes DataLoader Deadlock [TMP_ERR_CODE_1001]

**Symptom**: when a large GPU cluster starts, hundreds of DataLoader workers simultaneously request large MP4 chunks from S3 object storage, exhausting backbone bandwidth and node file descriptors. Training processes enter a wait state.

Listing 10-1 shows an example error log for S3 concurrent streaming overload.

*Listing 10-1: S3 concurrent streaming overload error log example. The log content is anonymized; metrics and paths do not correspond to a public incident.*

```bash
[FATAL] node-001.gpu-cluster.internal:
Connection reset by peer. Timeout extracting frame chunk from blob: /bucket-v/dataset/vid_slice_0001.mp4
File descriptor limits exceeded (Too many open files).
RuntimeError: Multiprocessing synchronization lock stuck at DataLoader worker 1.
AVSync_Module: Subtitle timestamp [1.21s] completely drifts out of matched acoustic window bounds.
```

**Root cause and fix**

- **Root cause**: no randomized exponential backoff; all workers request large chunks in the same millisecond.
- **Fix**: add `jitter_sleep(0-500ms)` retry logic in the PyTorch DataLoader; raise file descriptor limits with `ulimit -n 1048576`; reduce S3 block size from 128 MB to 2 MB and prewarm data to local NVMe through an edge caching layer.

### 10.6.2 NVDEC OOM: GPU Hardware Decoder Out of Memory [TMP_ERR_CODE_2001]

**Symptom**: when NVIDIA NVDEC decodes high-resolution 4K videos concurrently, GPU memory is exhausted, decoding stops, and training nodes are affected.

*Listing 10-2: NVDEC concurrent decoding OOM log example. The log content is anonymized; hardware limits must be based on actual device specifications and stress-test results.*

```bash
[FATAL] node-007.gpu-cluster.internal:
NVDecCreateDecoder failed: CUDA_ERROR_OUT_OF_MEMORY (error 2)
Video resolution 3840x2160 exceeds NVDEC hardware capability on A100-40GB.
cudaMemcpy failed during frame copy: cudaErrorIllegalAddress
Decoder context invalidated. All queued frames dropped (estimated loss: 2.3TB).
```

**Root cause and fix**

- **Root cause**: 4K resolution exceeds the per-instance NVDEC capacity; concurrent decoding has no memory quota isolation.
- **Fix**: force downsampling to 1080p before decoding, such as `-vf scale=1920:1080`; limit maximum concurrent decode streams per GPU, with H100 often using <= 24 1080p streams as an illustrative guideline; use DALI `VideoReader` instead of OpenCV to gain built-in memory quota management.

### 10.6.3 ASR Temporal Drift: WhisperX Timestamp Offset in Long Videos [TMP_ERR_CODE_3001]

**Symptom**: when ASR is run on videos longer than 30 minutes, WhisperX timestamps drift in the second half, sometimes by 8-12 seconds, making audio-video alignment fail.

*Listing 10-3: WhisperX timestamp drift error log example. The log content is anonymized; drift thresholds should be calibrated through sampled playback and downstream evaluation.*

```bash
[WARN] whisperx_worker_3: Timestamp drift detected at segment 847.
Expected anchor: [1823.4s], Model output: [1831.8s]. Delta: +8.4s.
[ERROR] TemporalAligner: Cross-modal lock failed - audio anchor outside visual frame window.
Alignment quality score: 0.23 (threshold: 0.75). Segment rejected and quarantined.
```

**Root cause and fix**

- **Root cause**: WhisperX uses VAD to slice speech; silence may be skipped incorrectly, causing cumulative timestamp drift. BGM mixing in long videos interferes with VAD decisions.
- **Fix**: force videos longer than 15 minutes into 10-minute subsegments before transcription; run Demucs vocal separation before VAD; use a 30-second sliding window to validate timestamp anchors and trigger realignment when drift exceeds 0.5 seconds.

### 10.6.4 Diarization Crash: Memory Leak Causes OOM [TMP_ERR_CODE_4001]

**Symptom**: when pyannote-audio (Bredin et al. 2020) diarization runs in bulk for a long time, process memory increases linearly by batch. After roughly four hours, the system OOM killer terminates the process and all processed results are lost.

*Listing 10-4: Diarization memory-leak error log example. The log content is anonymized; memory watermarks and batch sizes should be stress-tested according to node configuration.*

```bash
[ERROR] diarization_worker_12: Killed by OOM Killer (signal 9).
Process memory at kill time: 187.3 GB / 192 GB RAM.
pyannote.audio: SpeakerDiarization pipeline not released between batches.
torch.nn.Module references retained in embedding cache (est. leak: 2.1 GB/batch).
Unprocessed queue depth at crash: 3,421 audio segments (est. 68h audio).
```

**Root cause and fix**

- **Root cause**: the pyannote pipeline object is not explicitly destroyed between batches, embedding cache accumulates, and PyTorch graphs are not released.
- **Fix**: after each batch, call `del pipeline; torch.cuda.empty_cache(); gc.collect()`; run each diarization batch in an independent subprocess with `multiprocessing.spawn` so the process exit releases memory; cap audio duration per batch, for example at two hours.

### 10.6.5 WebDataset Write Collision: Concurrent Writes Corrupt a Shard [TMP_ERR_CODE_5001]

**Symptom**: in final packaging, multiple worker processes concurrently write to the same `.tar` shard, corrupting its structure. Training-time DataLoader then fails to parse it.

*Listing 10-5: WebDataset shard corruption error log example. The log content is anonymized; production environments should combine shard write locks, checksums, and retry strategies.*

```bash
[ERROR] training_node_44: WebDataset TarReader failed on shard: /data/processed/shard_0023.tar
tarfile.ReadError: invalid header magic bytes at offset 2147483392.
Estimated corrupted samples in shard: ~4,200 (approx 12.3GB of aligned multimodal data).
DataLoader worker 0: Pipe broken, resetting shard iterator. Skipping shard.
```

**Root cause and fix**

- **Root cause**: no write lock or per-shard assignment; multiple processes write to one file, interleaving byte streams.
- **Fix**: allocate an independent shard file per worker, named by `worker_id`; merge in the main process after writing or upload directly to S3; use `wids` (WebDataset Indexed Shards) instead of `.tar` when safe random writes and indexing are needed.

## 10.6.6 Frequent Error Quick Reference

*Table 10-3: Frequent audio-video pipeline error types and remediation strategies. Source: compiled by the authors; error codes and remediation strategies are anonymized engineering patterns.*

| Error code | Error type | Trigger | One-line fix |
| :--- | :--- | :--- | :--- |
| TMP_ERR_CODE_1XXX | S3/I/O timeout | Large-scale concurrent streaming without jittered backoff | Add jitter sleep and edge-cache prewarming |
| TMP_ERR_CODE_2XXX | NVDEC OOM | Unlimited concurrent 4K decoding | Downsample to 1080p and cap stream concurrency |
| TMP_ERR_CODE_3XXX | ASR temporal drift | Long-video VAD incorrectly skips silence | Segment transcription and sliding-window timestamp validation |
| TMP_ERR_CODE_4XXX | Diarization OOM | Pipeline object not released between batches | Subprocess isolation and explicit `gc.collect` per batch |
| TMP_ERR_CODE_5XXX | Shard corruption | Multiple processes write one `.tar` | One shard per worker, then main-process merge |
| TMP_ERR_CODE_6XXX | Audio-video mismatch hallucination | BGM mixed into training corpus | Set CLIP/SigLIP cross-modal cosine filtering watermarks according to the project baseline |
| TMP_ERR_CODE_7XXX | Decoded frames out of order | ffmpeg seek precision issue | Put `-ss` before the input argument |
| TMP_ERR_CODE_8XXX | Low-SNR audio track | Field noise exceeds the project noise watermark | Demucs separation and task-specific SNR discard thresholds |

## Chapter Summary

This chapter extended data engineering from static image-text data to long-temporal data with time and audio dimensions. The core conclusion is that video and audio "look abundant but yield few usable samples": dimensionality grows from $(W{\times}H{\times}C)$ to $(T{\times}W{\times}H{\times}C)$, and static redundancy, audio-video separation, and decoding I/O bottlenecks make the truly usable share of raw material quite low. To address this, the chapter built a three-track parallel pipeline across visual, acoustic, and text streams: on the visual side, shot-boundary detection slices semantically continuous segments and DINOv2 feature displacement drives adaptive frame sampling; on the acoustic side, WhisperX transcription, Demucs denoising, speaker diarization, and LLM subtitle correction extract and clean speech; finally, temporal anchors lock subtitles, waveforms, and key frames onto the same timeline and package them as multi-track mixed JSONL samples.

On this basis, the chapter added dense event labeling, audio-video mismatch detection, and engineering governance around decoding hardware (NVDEC/DALI), visual/acoustic quality scoring, and staged cost drivers. The anonymized postmortem of a 30 ms reading-offset accumulation showed how missing temporal alignment can damage training signals. The three-track samples in this chapter solve slicing and synchronization for long-temporal signals, but how images, text, audio, and video form stable correspondences in one semantic space still requires dedicated alignment and fusion design. That is the topic of the next chapter.

## References

Bain M, Huh J, Han T, Zisserman A (2023) WhisperX: Time-Accurate Speech Transcription of Long-Form Audio. arXiv preprint arXiv:2303.00747.

Bredin H, Yin R, Coria J M, Gelly G, Korshunov P, Lavechin M, Fustes D, Titeux H, Bouaziz W, Gill M P (2020) pyannote.audio: Neural Building Blocks for Speaker Diarization. In: IEEE International Conference on Acoustics, Speech and Signal Processing, pp 7124-7128.

Brooks T, Peebles B, Holmes C, DePue W, Guo Y, Jing L, Schnurr D, Taylor J, Luhman T, Luhman E, others (2024) Video Generation Models as World Simulators (Sora). OpenAI Technical Report.

Défossez A, Usunier N, Bottou L, Bach F (2019) Music Source Separation in the Waveform Domain (Demucs). arXiv preprint arXiv:1911.13254.

Oquab M, Darcet T, Moutakanni T, Vo H, Szafraniec M, Khalidov V, Fernandez P, Haziza D, Massa F, El-Nouby A, others (2023) DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research.

Radford A, Kim J W, Xu T, Brockman G, McLeavey C, Sutskever I (2023) Robust Speech Recognition via Large-Scale Weak Supervision (Whisper). In: Proceedings of the 40th International Conference on Machine Learning, pp 28492-28518.

Gemini Team (2024) Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530.

Zhang Y, Li Z, Liu C, Chen K, Ma L, Sun Y, Dou Q, Ouyang W, Yang M H, others (2024) Video Instruction Tuning with Synthetic Data (LLaVA-Video). arXiv preprint arXiv:2410.02713.

Zhang H, Li X, Bing L (2023) Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding. arXiv preprint arXiv:2306.02858.

Zohar O, Wang X, Dubois Y, Mehta N, Xiao T, Hansen-Estruch P, Yu L, Wang X F, Juefei-Xu F, Zhang N, Yeung-Levy S, Xia X (2025) Apollo: An Exploration of Video Understanding in Large Multimodal Models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. arXiv preprint arXiv:2412.10360.
