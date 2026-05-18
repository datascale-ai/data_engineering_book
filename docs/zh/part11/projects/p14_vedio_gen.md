## 项目标题：视频生成数据集——从视频源到可用于 T2V 训练的数据流水线

### 背景与目标

文本到视频生成（T2V）模型的数据工程，通常比文本到图像生成（T2I）更难处理。T2I 数据通常以“单张图像—文本描述”为基本单元，数据清洗主要围绕图像质量、文本匹配度、安全过滤和分辨率分桶展开。T2V 面对的是连续画面，训练样本除了静态视觉内容，还要包含动作变化、时间顺序、镜头运动和场景连续性。判断一个视频片段能否用于训练，不能只看某一帧是否清晰，还要检查整段视频是否存在有效运动，画面是否稳定，动作是否完整，主体在时间维度上是否保持一致。

因此，视频进入训练集之前需要经过更细的处理。流程首先从公开视频源、自建采集数据或素材库中筛选可用视频，再通过镜头切分把长视频拆成语义相对完整的 single-shot clips。随后，利用光流、运动强度、模糊度、抖动、水印和 OCR 面积等指标过滤低质量片段，减少静止画面、剧烈抖动和无效运动对训练的干扰。视频 caption 也不能停留在“画面里有什么”，还要写清动作发生顺序、主体位置变化、相机运动方式和镜头语言，例如推拉、平移、俯拍、特写、远景等信息。

本项目围绕从视频源到可训练 T2V 数据集的处理过程展开，将原始视频转化为带有结构化 caption、时空对齐信息和质量标签的训练样本。输出样本不仅记录主体和场景，也覆盖动作过程、空间关系和镜头表现，从而为视频生成模型学习“如何运动”和“如何拍摄”提供监督。

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

def read_pexels_manifest(src_dir: Path) -> list[dict]:
    manifest = src_dir / "pexels_manifest.jsonl"
    records = []

    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    # manifest 缺失时，从文件名中恢复最小元数据
    for video_path in sorted(src_dir.glob("pexels_*.mp4")):
        video_id = int(video_path.stem.split("_")[1])
        records.append({
            "video_id": video_id,
            "saved_as": str(video_path),
            "page_url": None,
            "user": {}
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

    info = ffprobe(video_path)  # 返回 duration / fps / width / height / nb_frames 等字段
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

这里需要注意两点。第一，所有源视频都被整理成结构一致的 JSONL 行，后续阶段不再直接扫描 mp4 目录，而是读取这份 manifest。第二，加载过程支持断点续跑：已经写入的 `video_id` 不重复处理，新视频只追加到文件末尾。在 1000+ 视频规模下，这种方式比一次性全量重跑更稳，也便于中途清除损坏视频。

---

### Step 2：PySceneDetect 切分镜头

T2V 训练样本通常按镜头组织，不直接沿用原始视频边界。一个 Pexels 视频可能只有一个长镜头，也可能包含多个剪辑点。若不做切分，caption 很容易把多个场景揉在一起，训练时文本和画面之间会出现错配。这里使用 PySceneDetect (Castellano 2012) 的 ContentDetector 做镜头检测，并在检测不到边界时把整段视频作为一个 shot。切分阶段还要过滤过短片段，例如小于 1 秒的镜头通常不保留。

```python
from scenedetect import open_video, SceneManager, ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg

def detect_scenes(video_path: str, threshold: float = 27.0, downscale: int = 4):
    video = open_video(video_path)

    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))

    # 在低分辨率代理上检测边界，降低计算开销
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

    # 对没有剪辑点的素材，保留为一个完整 single-shot clip
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

    # 丢弃过短镜头
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

实际运行时，1000 条视频如果平均切出 8 到 15 个镜头，就能得到约 10000 个片段。这个数量只是实验规模上的参考，并不是固定要求。需要注意的是，PySceneDetect 的阈值会明显影响片段数量：阈值低，切分更细；阈值高，切分更保守。建议先抽样观察切分结果，再固定阈值，避免产生大量语义不完整的碎片。

---

### Step 3：运动检测过滤静态片段

镜头切分后得到的仍是“可训练候选片段”，还不能直接视为最终训练数据。公开视频中有不少片段虽然清晰，但几乎没有运动，例如静态风景、固定产品图、人物摆拍和慢速定帧。它们对视频生成的时间建模帮助有限，数量过多还会让模型偏向生成静态视频。因此第三步用光流估计片段运动强度。

这里不做复杂动作识别，只计算连续帧之间的平均光流幅值。脚本将视频缩放到 480×270 的代理分辨率，按 stride 抽取帧对，并限制最大帧对数量，避免长视频计算过慢。最终输出 `motion_strength`、`n_pairs` 和 `pass_motion`。

```python
def motion_filter_one(
    shot: dict,
    threshold: float = 0.5,
    proxy_wh: tuple[int, int] = (480, 270),
    stride: int = 2,
    max_pairs: int = 60,
) -> dict:
    """
    输入一个 shot 记录，输出运动过滤结果。
    compute_motion_magnitude 内部可使用 Farneback 光流：
    1. 解码视频帧；
    2. 缩放到 proxy_wh；
    3. 计算相邻采样帧之间的光流；
    4. 统计平均幅值作为 motion_strength。
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
    # 1. 从 shot 中均匀采样多帧
    images = sample_frames_pil(segment_path, k=frames)
    if not images:
        return {
            "aesthetic_score": 0.0,
            "per_frame_scores": [],
            "pass_aesthetic": False,
            "status": "no_frames",
        }

    # 2. CLIP 编码
    inputs = clip_processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    feats = clip_model.get_image_features(pixel_values=pixel_values)

    # 3. L2 归一化后送入 LAION-Aesthetic MLP
    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    scores = aesthetic_mlp(feats).squeeze(-1).float().cpu().tolist()

    # 4. 片段级平均分
    avg_score = float(sum(scores) / len(scores))
    return {
        "aesthetic_score": avg_score,
        "per_frame_scores": [float(x) for x in scores],
        "pass_aesthetic": bool(avg_score >= threshold),
        "status": "ok",
    }
```

工程实现中还需要处理多 GPU 分片和显存退化。在本项目中脚本采用确定性分片：先按 `shot_id` 排序，再根据样本序号对 `num_shards` 取模，每个 GPU 只处理自己的 shard。这种分片方式可以避免重复计算，也便于失败后从指定 shard 继续恢复。审美阈值也不应只作为删除条件使用。可以先把分数写入 manifest，再在训练阶段按分数设置采样权重或数据分桶。

---

### Step 5：多帧采样与 Qwen2.5-VL / InternVL3 生成 caption

经过运动和质量过滤后，保留下来的 shot 需要生成视频级 caption。这里的 caption 承担 T2V 训练中的语言监督作用，而不是给视频取一个简短标题。它应该覆盖主体、场景、动作、相机 framing、光照色彩和整体氛围。为了避免逐帧描述，提示词明确要求输出单段英文，不允许出现 “frame 1”“frame 2” 这样的枚举。

实现上，脚本先从 shot 中按时间顺序采样 8 帧，并保存到 `frames/pexels_<video_id>/shot_<idx>/`。保存帧便于在 caption 阶段复查输入，也能让 Step 6 的镜头语言标注复用同一组帧，避免重复解码视频。

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
    # 1. 按时间顺序均匀选择多帧
    if len(frame_paths) > frames_n:
        import numpy as np
        indices = np.linspace(0, len(frame_paths) - 1, frames_n).round().astype(int)
        frame_paths = [frame_paths[i] for i in indices]

    # 2. 构造 Qwen2.5-VL / InternVL3 的视频式输入
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

    # 3. 生成 caption
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

    # 4. 记录质量状态，过短 caption 留给后续重试或复核
    n_words = len(caption.split())
    return {
        "caption_en": caption,
        "n_words": n_words,
        "caption_short": bool(n_words < min_words),
        "status": "ok" if caption else "empty",
    }
```

若第一次 caption 过短，可以提高 temperature 重试两次，但不建议无限重试。过度重试可能让 caption 变长，但不一定提高准确性。对于训练数据，可以保留 `caption_short` 标记，在后处理阶段统一处理，避免模型为了凑长度补写画面中不存在的细节。InternVL3 的接入方式与 Qwen2.5-VL 类似，只需要替换模型加载和输入组织接口；数据层面的流程仍保持为“按时间顺序采样多帧 → 生成单段视频描述”。

---

### Step 6：镜头语言标注：运镜、构图与光线

多帧 caption 侧重描述视频内容，镜头语言标注则补充拍摄方式。这一步使用两条并行路径：第一条路径由 VLM 按受控词表输出结构化标签，包括景别、机位、构图、光照、色彩和风格；第二条路径由光流估计相机运动，输出 static、pan、tilt、zoom、jitter 或 complex 等类别。两部分合并后，样本就同时具备语义 caption 和拍摄语言标签。

```python
VOCAB = {
    "shot_size": [
        "extreme_wide", "wide", "medium", "close_up", "extreme_close_up"
    ],
    "camera_angle": [
        "eye_level", "high_angle", "low_angle", "dutch", "overhead"
    ],
    "composition": [
        "rule_of_thirds", "centered", "symmetrical",
        "leading_lines", "framing", "negative_space"
    ],
    "lighting": [
        "high_key", "low_key", "natural", "golden_hour",
        "backlit", "silhouette", "artificial", "mixed"
    ],
    "color_palette": [
        "warm", "cool", "neutral", "monochrome", "saturated", "desaturated"
    ],
    "style": [
        "cinematic", "documentary", "vlog", "commercial", "artistic"
    ],
}


def tag_shot_language(shot_id: str, segment_path: str, frame_paths: list[str]) -> dict:
    """
    输出结构化镜头语言标签：
    1. VLM 根据固定词表判断景别、机位、构图、光照、色彩和风格；
    2. 光流模块估计相机运动；
    3. 二者合并为 stage6 记录。
    """

    # A. VLM 标签，输出必须是 JSON
    raw_json = vlm_generate_json(
        frames=frame_paths,
        allowed_vocab=VOCAB,
        task="classify shot language with controlled vocabulary",
    )
    vlm_tags = sanitize_and_coerce_to_vocab(raw_json, VOCAB)

    # B. 光流相机运动
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

这里建议使用受控词表，避免让模型自由生成标签。自由文本标签看起来更丰富，但很难用于检索、分桶和训练采样；受控标签的表达范围有限，却能把同类样本归到同一字段下。例如，`close_up`、`medium`、`wide` 可以直接用于景别分层；`golden_hour`、`backlit`、`low_key` 可以用于光照分布统计；`pan_left`、`zoom_in`、`jitter` 可以用于相机运动控制样本构建。在 T2V 训练中，这类结构化字段比一段漂亮但不可控的描述更容易进入工程流程。

完成 Step 6 后，最终样本可以组织为如下形式：

```python
final_sample = {
    "shot_id": "pexels_123456_shot_0003",
    "source": {
        "video_id": 123456,
        "license": "pexels",
        "page_url": "https://www.pexels.com/video/...",
        "author_name": "..."
    },
    "video": {
        "segment_path": "shots/pexels_123456/shot_0003.mp4",
        "start_ts": 12.48,
        "end_ts": 17.92,
        "n_frames": 163
    },
    "filters": {
        "motion_strength": 1.37,
        "pass_motion": True,
        "aesthetic_score": 6.21,
        "pass_aesthetic": True
    },
    "caption": {
        "caption_en": "A person walks through a sunlit urban street...",
        "n_words": 67,
        "caption_short": False
    },
    "shot_language": {
        "shot_size": "medium",
        "camera_angle": "eye_level",
        "composition": "centered",
        "lighting": "natural",
        "color_palette": "warm",
        "style": "cinematic",
        "camera_motion": "pan_right"
    }
}
```

这个结构已经具备可训练数据的基本形态。后续可以继续加入 NSFW 过滤、OCR/水印过滤、去重、类别重采样和 WebDataset 打包。在这条流水线中，需要把握的是样本组织方式：视频样本会在每个阶段逐步积累可学习的监督信号，最后再被组织成训练数据。镜头切分提供时间边界，运动过滤提供动态质量，美学过滤提供视觉质量，多帧 caption 提供语义监督，镜头语言标注提供拍摄控制信息。这些字段配合使用后，T2V 模型更容易建立稳定的“文本—动作—镜头”对应关系。

## 结果展示与分析

| frame1 | frame2 |
|---|---|
| ![frame1](../../images/part10/14_f0.jpg) | ![frame2](../../images/part10/14_f1.jpg) |

图中展示了产出数据中的两帧采样结果。该片段展示了一段从高空视角拍摄的海岸线画面：深蓝色海水不断冲击崎岖礁石，浪花在暗色岩壁边缘形成明显的白色泡沫，岩石间分布着少量绿色植被，使画面在粗粝的自然环境中保留了一定的层次感。多帧 caption 覆盖了该片段的主体、场景、光照和氛围，描述了自然光照、冷色调海面、清晰的岩石纹理，以及海浪运动带来的动态感。从镜头语言标注结果看，该 shot 被识别为 `extreme_wide` 景别、`high_angle` 高角度视角，构图方式为 `rule_of_thirds`，光照类型为 `natural`，整体色彩倾向为 `cool`，风格标签为 `cinematic`。相机运动模块将其判定为 `zoom_in`，说明画面存在较明显的推进或尺度变化；`motion_strength=0.8974` 表明该片段具有稳定的运动信号，可用于 T2V 训练中学习自然场景运动、航拍视角和海岸镜头语言。


## 参考文献

Castellano B (2012) PySceneDetect: Python and OpenCV-based Scene Cut/Transition Detection Program. Available at: https://www.brettcastellano.com/post/pyscenedetect.

Chen Z, Wang W, Tian H, Ye S, Gao Z, Cui E, Tong X, Hu J, Luo J, Ma S, others (2024) InternVL3: Exploring Advanced Training and Test-Time Scaling for Vision-Language Models. arXiv preprint arXiv:2504.10479.

Farnebäck G (2003) Two-Frame Motion Estimation Based on Polynomial Expansion. In: Proceedings of the 13th Scandinavian Conference on Image Analysis, pp 363-370.

Pexels (2014) Pexels: Free Stock Photos, Royalty Free Images & Videos. Available at: https://www.pexels.com.

Radford A, Kim J W, Hallacy C, Ramesh A, Goh G, Agarwal S, Sastry G, Askell A, Mishkin P, Clark J, others (2021) Learning Transferable Visual Models from Natural Language Supervision (CLIP). In: Proceedings of the 38th International Conference on Machine Learning, pp 8748-8763.

Schuhmann C, Beaumont R, Vencu R, Gordon C, Wightman R, Cherti M, Coombes T, Katta A, Mullis C, Wortsman M, others (2022) LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models. In: Advances in Neural Information Processing Systems 35:25278-25294.

Wang P, Bai S, Tan S, Wang S, Fan Z, Bai J, Chen K, Liu X, Wang J, Ge W, others (2024) Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv preprint arXiv:2409.12191.

