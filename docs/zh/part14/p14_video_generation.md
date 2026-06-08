# 项目十四：视频生成数据集：从视频源到可用于 T2V 训练的数据流水线

## 摘要

本项目围绕“视频生成数据集：从视频源到可用于 T2V 训练的数据流水线”构建可复现的数据工程案例，重点说明业务目标、数据边界、架构决策、核心实现、验收指标与风险控制。章节将安装命令和脚本细节收敛到工程复盘视角，突出样本 schema、数据流、失败模式和可交付物之间的关系，帮助读者把前文方法转化为可审计、可扩展的项目资产。



本项目构建一条从公开视频源到可训练 T2V 数据集的生产流水线，覆盖视频源加载、镜头切分、运动过滤、美学过滤、多帧 caption 和镜头语言标注六个环节。项目目标不是直接训练视频生成模型，而是把原始视频整理为 shot-level 样本，并为每个样本保留来源信息、时间边界、运动强度、美学分数、结构化 caption、镜头语言标签和质量状态。完成本项目后，应能够说明 T2V 数据样本的输入、处理、输出和验收条件，理解文本、动作、镜头和授权信息之间的工程关系，并评估该流水线在规模化、合规和训练适配上的边界。

**关键词**：T2V 数据工程；视频生成数据集；镜头切分；运动过滤；多帧 caption；镜头语言标注

**项目目标**
- 将公开视频源整理为可追溯的 `source_videos.jsonl`。
- 基于 PySceneDetect 将长视频切分为语义相对完整的 shot-level 样本。
- 通过运动强度、美学分数和质量标签过滤低价值片段。
- 使用多帧采样生成覆盖主体、动作、镜头和氛围的 caption。
- 输出包含来源、视频片段、过滤结果、caption 和镜头语言标签的训练候选样本。

## 背景与目标

文本到视频生成（T2V）模型的数据工程，通常比文本到图像生成（T2I）更难处理。T2I 数据通常以“单张图像—文本描述”为基本单元，数据清洗主要围绕图像质量、文本匹配度、安全过滤和分辨率分桶展开。T2V 面对的是连续画面，训练样本除了静态视觉内容，还要包含动作变化、时间顺序、镜头运动和场景连续性。判断一个视频片段能否用于训练，不能只看某一帧是否清晰，还要检查整段视频是否存在有效运动，画面是否稳定，动作是否完整，主体在时间维度上是否保持一致。

因此，视频进入训练集之前需要经过更细的处理。流程首先从公开视频源、自建采集数据或素材库中筛选可用视频，再通过镜头切分把长视频拆成语义相对完整的 single-shot clips。随后，利用光流、运动强度、模糊度、抖动、水印和 OCR 面积等指标过滤低质量片段，减少静止画面、剧烈抖动和无效运动对训练的干扰。视频 caption 也不能停留在“画面里有什么”，还要写清动作发生顺序、主体位置变化、相机运动方式和镜头语言，例如推拉、平移、俯拍、特写、远景等信息。

本项目围绕从视频源到可训练 T2V 数据集的处理过程展开，将原始视频转化为带有结构化 caption、时空对齐信息和质量标签的训练样本。输出样本不仅记录主体和场景，也覆盖动作过程、空间关系和镜头表现，从而为视频生成模型学习“如何运动”和“如何拍摄”提供监督。

## 关键词

视频生成数据集；项目实战；可复现数据工程；数据流水线；验收指标

## 项目目标与读者收获

本项目以“视频生成数据流水线”为核心案例，目标是把视频源、片段切分、caption、质量评分和训练封装组织成 T2V 数据资产。读者完成本章后，应能够辨认该场景的关键数据对象、拆分工程链路、设置验收指标，并将案例方法迁移到相近的数据工程任务中。

## 场景约束与数据边界

以公开视频样本和小规模流水线为主，不覆盖完整商业版权管理和大规模视频平台。这些边界使案例能够被复现和审计；当数据规模、数据来源、权限范围或部署环境变化时，需要重新评估采样策略、质量阈值、运行成本和合规要求。

## 架构决策

本项目采用“视频采集、切分抽帧、字幕/描述生成、质量评分、去重过滤和 T2V 样本封装”的架构路径。该决策优先保证输入输出契约、版本可追踪、异常可定位和结果可复核，而不是把全部逻辑压缩为一次性脚本运行。

## 样本 schema / 数据流

核心数据流可概括为：

```text
视频源 -> 片段切分 -> 帧/字幕/运动特征 -> caption 与质量评分 -> 过滤去重 -> T2V 训练样本
```

样本 schema 至少应保留 `id`、`source`、`content_or_payload`、`metadata`、`quality_signals`、`split_or_stage` 与 `audit_trace` 等字段；具体字段由本项目的数据类型、下游任务和验收方式进一步细化。

## 核心实现片段

正文只保留能够说明设计取舍的关键实现片段。完整脚本、长配置、运行日志和大文件应放入配套仓库或附录说明；代码展示重点放在输入输出契约、质量阈值、异常处理和验收接口上。

## 实验或验收指标

验收指标包括片段可用率、caption 一致性、运动/清晰度分布、去重率、安全过滤率、成本/分钟和训练封装完整性。若项目进入生产、课程或公开复现实验环境，还应记录版本号、依赖环境、随机种子、样本抽检结果和失败样本复盘记录。

## 成本、风险与合规边界

成本主要来自视频下载、转码、视觉模型调用和存储；风险集中在版权、肖像权、敏感内容和生成模型滥用。涉及外部数据、个人信息、版权内容或第三方服务时，应保留来源说明、权限状态、脱敏策略、调用记录和人工复核记录。

## 常见失败模式

常见失败包括输入分布偏离、schema 字段缺失、质量阈值过松或过紧、评测样本覆盖不足、模型调用不稳定、结果无法回溯等。排查时应优先定位数据边界和中间产物，再检查模型、工具链与部署环境。

## 可复现资源说明

复现材料应包括数据来源说明、最小样本、配置文件、运行命令、指标脚本、检查报告和产物目录。正文保留必要片段；完整 notebook、长脚本和大文件作为配套资源独立维护。

## 2. 架构设计：六组件视频生成数据流水线

本节关注如何在公开视频基础上组织一条可复跑、可审计、可扩展的 T2V 数据生产流水线。输入端是一批 Pexels 开源视频；输出端则是面向视频生成训练的 shot-level 样本，每个样本同时包含视频片段、来源信息、运动强度、美学分数、多帧 caption、镜头语言标签和相机运动信息。这种数据结构比普通视频分类数据集更细，因为 T2V 模型学习的是文本、画面、动作与镜头之间的联合对应关系，目标不再是单一类别标签。编写依据来自六个处理脚本：视频源加载、镜头切分、运动过滤、美学过滤、多帧 caption 和镜头语言标注。

整条流水线可以拆成六个组件。第一是**视频源加载**。它读取 Pexels manifest 或本地视频文件名，重新探测视频时长、fps、分辨率、帧数和文件大小，并将作者、页面链接、许可字段写入统一清单。该组件为后续来源回溯、分辨率统计、时长统计和授权状态检查提供基础信息。

第二是**镜头切分**。T2V 模型通常不直接使用原始长视频训练，因为长视频内部可能包含多个镜头、场景跳转和语义断裂。流水线使用 PySceneDetect (Castellano 2012) 的 ContentDetector 检测镜头边界，再用 ffmpeg 将视频切成 single-shot clips。每个片段被赋予 `shot_id`，并记录起止时间、所属视频、片段序号和本地路径。此后，所有过滤、caption 和镜头标签都以 `shot_id` 为主键展开。

第三是**运动过滤**。视频生成模型需要学习时间变化，因此训练集中不能混入过多静态片段。该组件在代理分辨率上计算 Farneback (Farnebäck 2003) 光流均值，用 `motion_strength` 衡量片段内部的运动强度，再通过阈值生成 `pass_motion`。该步骤暂不判断动作语义，主要用于区分“有训练价值的动态片段”和“近似静止的图片式片段”。

第四是**质量过滤**。通过运动过滤并不意味着片段一定适合训练。模糊、压缩、曝光异常和构图较差的视频会污染生成模型的视觉分布。本项目采用 CLIP ViT-L/14 (Radford et al. 2021) 提取多帧视觉特征，再送入 LAION-Aesthetic MLP (Schuhmann et al. 2022) 预测审美分。片段级分数由多帧平均得到，既避免单帧偶然性，也保留了视频整体观感。

第五是**多帧 caption**。T2V caption 通常需要覆盖主体、场景、动作、镜头构图、光线和氛围。单帧描述只能说明“画面里有什么”，很难说明“动作如何展开”。因此，流水线按时间顺序采样多帧，交给 Qwen2.5-VL (Wang et al. 2024) 或 InternVL3 (Chen et al. 2024) 生成一段视频级 caption，并设置最小词数与重试机制，减少过短描述。

第六是**镜头语言标注**。普通 caption 描述内容，镜头语言标签描述拍摄方式。该组件一方面使用受控词表标注景别、机位、构图、光照、色彩和风格；另一方面利用光流估计相机运动，如 static、pan、tilt、zoom、jitter 和 complex。最终样本同时保留“视频中发生了什么”和“它是怎样被拍摄出来的”两类信息。

---

## 3. 分步实现：从 Pexels 视频到可训练 T2V 数据集

### Step 1：从 Pexels 开源子集加载 1000+ 视频

视频加载阶段先建立一份可靠的源数据清单，暂不进入训练或过滤环节。Pexels (Pexels 2014) 视频文件通常已经下载到本地目录，同时配有 `pexels_manifest.jsonl`。manifest 中保存视频 ID、页面链接、作者信息和本地保存路径；如果 manifest 缺失，也可以从 `pexels_*.mp4` 文件名中恢复最小记录。为了避免依赖下载阶段的旧元数据，脚本会对每个 mp4 重新执行 `ffprobe`，补齐 duration、fps、width、height、nb_frames 和 file_size。由此生成的 `source_videos.jsonl` 可以作为后续流水线的稳定入口。

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

这里需要注意两点。第一，所有源视频都被整理成结构一致的 JSONL 行，后续阶段不再直接扫描 mp4 目录，而是读取这份 manifest。第二，加载过程支持断点续跑：已经写入的 `video_id` 不重复处理，新视频只追加到文件末尾。在 1000+ 视频规模下，这种方式比一次性全量重跑更稳，也便于中途清除损坏视频。

---

### Step 2：PySceneDetect 切分镜头

T2V 训练样本通常按镜头组织，不直接沿用原始视频边界。一个 Pexels 视频可能只有一个长镜头，也可能包含多个剪辑点。若不做切分，caption 很容易把多个场景揉在一起，训练时文本和画面之间会出现错配。这里使用 PySceneDetect (Castellano 2012) 的 ContentDetector 做镜头检测，并在检测不到边界时把整段视频作为一个 shot。切分阶段还要过滤过短片段，例如小于 1 秒的镜头通常不保留。

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

实际运行时，1000 条视频如果平均切出 8 到 15 个镜头，就能得到约 10000 个片段。这个数量只是实验规模上的参考，并不是固定要求。需要注意的是，PySceneDetect 的阈值会明显影响片段数量：阈值低，切分更细；阈值高，切分更保守。建议先抽样观察切分结果，再固定阈值，避免产生大量语义不完整的碎片。

---

### Step 3：运动检测过滤静态片段

镜头切分后得到的仍是“可训练候选片段”，还不能直接视为最终训练数据。公开视频中有不少片段虽然清晰，但几乎没有运动，例如静态风景、固定产品图、人物摆拍和慢速定帧。它们对视频生成的时间建模帮助有限，数量过多还会让模型偏向生成静态视频。因此第三步用光流估计片段运动强度。

这里不做复杂动作识别，只计算连续帧之间的平均光流幅值。脚本将视频缩放到 480×270 的代理分辨率，按 stride 抽取帧对，并限制最大帧对数量，避免长视频计算过慢。最终输出 `motion_strength`、`n_pairs` 和 `pass_motion`。

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

运动阈值不宜只凭经验一次确定。可以先统计所有片段的 `motion_strength` 分布，再抽查低分、中分和高分样本。对于普通公开视频，阈值可以从 0.5 附近开始试验；若数据中包含大量慢镜头、自然风光或微动作，需要降低阈值，避免误删有效样本。这个阶段的输出不一定马上删除失败片段，也可以保留 `pass_motion=False` 的记录，后续按训练阶段决定是否使用。

---

### Step 4：CLIP 美学打分与 LAION-Aesthetic 过滤

运动过滤关注片段是否存在时间变化，质量过滤关注画面是否值得保留。一个片段可能动作明显，但画面过曝、模糊、压缩严重或构图混乱。生成模型会继承训练集中的视觉分布，因此需要用质量评分对样本分层。

本项目使用 CLIP ViT-L/14 提取图像特征，再接入 LAION-Aesthetic MLP 计算审美分。每个 shot 均匀采样 4 帧，分别打分后取平均值。相比只取首帧或中间帧，多帧平均更稳定，因为视频片段内部可能存在短暂模糊、主体遮挡或曝光变化。

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

工程实现中还需要处理多 GPU 分片和显存退化。在本项目中脚本采用确定性分片：先按 `shot_id` 排序，再根据样本序号对 `num_shards` 取模，每个 GPU 只处理自己的 shard。这种分片方式可以避免重复计算，也便于失败后从指定 shard 继续恢复。审美阈值也不应只作为删除条件使用。可以先把分数写入 manifest，再在训练阶段按分数设置采样权重或数据分桶。

---

### Step 5：多帧采样与 Qwen2.5-VL / InternVL3 生成 caption

经过运动和质量过滤后，保留下来的 shot 需要生成视频级 caption。这里的 caption 承担 T2V 训练中的语言监督作用，而不是给视频取一个简短标题。它应该覆盖主体、场景、动作、相机 framing、光照色彩和整体氛围。为了避免逐帧描述，提示词明确要求输出单段英文，不允许出现 “frame 1”“frame 2” 这样的枚举。

实现上，脚本先从 shot 中按时间顺序采样 8 帧，并保存到 `frames/pexels_<video_id>/shot_<idx>/`。保存帧便于在 caption 阶段复查输入，也能让 Step 6 的镜头语言标注复用同一组帧，避免重复解码视频。

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

若第一次 caption 过短，可以提高 temperature 重试两次，但不建议无限重试。过度重试可能让 caption 变长，但不一定提高准确性。对于训练数据，可以保留 `caption_short` 标记，在后处理阶段统一处理，避免模型为了凑长度补写画面中不存在的细节。InternVL3 的接入方式与 Qwen2.5-VL 类似，只需要替换模型加载和输入组织接口；数据层面的流程仍保持为“按时间顺序采样多帧 → 生成单段视频描述”。

---

### Step 6：镜头语言标注：运镜、构图与光线

多帧 caption 侧重描述视频内容，镜头语言标注则补充拍摄方式。这一步使用两条并行路径：第一条路径由 VLM 按受控词表输出结构化标签，包括景别、机位、构图、光照、色彩和风格；第二条路径由光流估计相机运动，输出 static、pan、tilt、zoom、jitter 或 complex 等类别。两部分合并后，样本就同时具备语义 caption 和拍摄语言标签。

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

这里建议使用受控词表，避免让模型自由生成标签。自由文本标签看起来更丰富，但很难用于检索、分桶和训练采样；受控标签的表达范围有限，却能把同类样本归到同一字段下。例如，`close_up`、`medium`、`wide` 可以直接用于景别分层；`golden_hour`、`backlit`、`low_key` 可以用于光照分布统计；`pan_left`、`zoom_in`、`jitter` 可以用于相机运动控制样本构建。在 T2V 训练中，这类结构化字段比一段漂亮但不可控的描述更容易进入工程流程。

完成 Step 6 后，最终样本可以组织为如下形式：

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

这个结构已经具备可训练数据的基本形态。后续可以继续加入 NSFW 过滤、OCR/水印过滤、去重、类别重采样和 WebDataset 打包。在这条流水线中，需要把握的是样本组织方式：视频样本会在每个阶段逐步积累可学习的监督信号，最后再被组织成训练数据。镜头切分提供时间边界，运动过滤提供动态质量，美学过滤提供视觉质量，多帧 caption 提供语义监督，镜头语言标注提供拍摄控制信息。这些字段配合使用后，T2V 模型更容易建立稳定的“文本—动作—镜头”对应关系。

## 结果展示与分析

| frame1 | frame2 |
|---|---|
| ![frame1](../../images/part10/14_f0.jpg) | ![frame2](../../images/part10/14_f1.jpg) |

图中展示了产出数据中的两帧采样结果。该片段展示了一段从高空视角拍摄的海岸线画面：深蓝色海水不断冲击崎岖礁石，浪花在暗色岩壁边缘形成明显的白色泡沫，岩石间分布着少量绿色植被，使画面在自然环境中保留了一定层次感。多帧 caption 覆盖该片段的主体、场景、光照和氛围，描述自然光照、冷色调海面、清晰岩石纹理，以及海浪运动带来的动态感。从镜头语言标注结果看，该 shot 被识别为 `extreme_wide` 景别、`high_angle` 高角度视角，构图方式为 `rule_of_thirds`，光照类型为 `natural`，整体色彩倾向为 `cool`，风格标签为 `cinematic`。相机运动模块将其判定为 `zoom_in`，说明画面存在较明显的推进或尺度变化；`motion_strength=0.8974` 表明该片段具有稳定运动信号，可用于 T2V 训练中学习自然场景运动、航拍视角和海岸镜头语言。

正式验收时，项目至少需要满足四项条件：第一，所有样本都能通过 `shot_id` 追溯到源视频、作者、页面链接和许可状态；第二，视频片段路径、起止时间和帧数与实际文件一致；第三，运动强度、美学分数、caption 和镜头语言标签能够被下游训练或采样脚本读取；第四，低质量、低运动、来源不明或许可不清的样本不进入正式训练集。只有这些条件同时满足，流水线产物才具备从原型验证转向训练数据候选池的基础条件。

## 本章小结

本章以“视频生成数据流水线”为案例，展示了把视频源、片段切分、caption、质量评分和训练封装组织成 T2V 数据资产的工程组织方式。案例的主要价值在于把任务定义、数据边界、架构决策、样本 schema、指标验收和复现资源放在同一条链路中，使项目不再只是操作步骤，而成为可复核的案例研究。

该案例的边界同样需要被清楚保留。以公开视频样本和小规模流水线为主，不覆盖完整商业版权管理和大规模视频平台。在更大规模、更高风险或更强合规约束的场景中，应重新评估数据来源、权限状态、人工复核比例、运行成本和失败回滚方案。

作为第十四篇的一部分，本章对应前文方法在项目层面的落地验证。读者可将本案例与第十三篇的数据配方、前文的平台治理章节以及附录中的检查清单合并使用，形成从方法理解到工程交付的闭环。

## 参考文献

Castellano B (2012) PySceneDetect: Python and OpenCV-based Scene Cut/Transition Detection Program. Available at: https://www.brettcastellano.com/post/pyscenedetect.

Chen Z, Wang W, Tian H, Ye S, Gao Z, Cui E, Tong X, Hu J, Luo J, Ma S, others (2024) InternVL3: Exploring Advanced Training and Test-Time Scaling for Vision-Language Models. arXiv preprint arXiv:2504.10479.

Farnebäck G (2003) Two-Frame Motion Estimation Based on Polynomial Expansion. In: Proceedings of the 13th Scandinavian Conference on Image Analysis, pp 363-370.

Pexels (2014) Pexels: Free Stock Photos, Royalty Free Images & Videos. Available at: https://www.pexels.com.

Radford A, Kim J W, Hallacy C, Ramesh A, Goh G, Agarwal S, Sastry G, Askell A, Mishkin P, Clark J, others (2021) Learning Transferable Visual Models from Natural Language Supervision (CLIP). In: Proceedings of the 38th International Conference on Machine Learning, pp 8748-8763.

Schuhmann C, Beaumont R, Vencu R, Gordon C, Wightman R, Cherti M, Coombes T, Katta A, Mullis C, Wortsman M, others (2022) LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models. In: Advances in Neural Information Processing Systems 35:25278-25294.

Wang P, Bai S, Tan S, Wang S, Fan Z, Bai J, Chen K, Liu X, Wang J, Ge W, others (2024) Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv preprint arXiv:2409.12191.
