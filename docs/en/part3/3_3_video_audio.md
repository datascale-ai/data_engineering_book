## Chapter 8: Video and Audio Data Processing

### Chapter Summary

Video data is the modality with the largest volume, highest processing difficulty, and most complex information density in multimodal large model (LMM) training, referred to as the "deep water" of multimodal engineering. Unlike static images, video introduces the **Temporal Dimension**, meaning data is not merely pixel stacking but carries causal logic, physical laws, and motion patterns.

This chapter systematically breaks down how to transform continuous, unstructured video streams into discrete tokens the model can understand. We will start from the underlying **Shot Boundary Detection**, analyze content-based segmentation algorithms in depth; then analyze the "heart" of video generation—**Video Tokenizer**, comparing underlying principles of VQ-VAE and Google DeepMind's latest MagViT-v2; finally, we will demonstrate how to use **WhisperX** to achieve word-level and even phoneme-level precise alignment of audio and video, building spatio-temporally synchronized supervision signals for the model.

**Learning Objectives**:
* **Engineering Capability**: Master using PySceneDetect combined with ffmpeg keyframe metadata for efficient two-stage scene segmentation (Coarse-to-Fine) strategy.
* **Theoretical Depth**: Deeply understand the "Codebook Collapse" problem in Video Tokenization, and how MagViT-v2 completely solves this bottleneck through Lookup-Free Quantization (LFQ).
* **Data Pipeline**: Implement WhisperX-based Forced Alignment flow to solve precise subtitle alignment for multi-speaker, background noise acoustic environments.
* **Storage Optimization**: Understand storage sharding and efficient loading for massive video data.

**Scenario Introduction**:
> "Imagine you are training a world model like Sora. You have downloaded the 2-hour movie _Titanic_ as training data.
>
> If you simply split by every 10 seconds, you'll encounter severe 'semantic discontinuity': the first 5 seconds of a segment might be calm sea breeze on deck, the next 5 seconds suddenly jump to a noisy restaurant. This cross-scene 'hard cut' will confuse the model: 'How did the person teleport from outdoors to indoors in 0.1 seconds?' This not only wastes compute but teaches the model wrong physics.
>
> Furthermore, audio temporal precision is life. If your subtitles lag 2 seconds behind the picture, when Rose's mouth is moving on screen, the corresponding token is Jack's dialogue. The model will incorrectly associate 'Jack's voice features' with 'Rose's facial features.' In trillion-token training, such subtle misalignment can amplify into severe hallucinations."

---

### 8.1 Video Processing Pipeline: Scene Detection

Video is fundamentally not a continuous stream but a sequence of independent "shots" concatenated together. Each shot represents one camera turn-on and turn-off (or continuous camera movement). Training video generative models (Video Generative Models) requires each training sample (Training Clip) to be within the same shot, ensuring **Spatio-Temporal Continuity**.

#### 8.1.1 Micro-view of Video Structure: GOP and I-Frames

Before diving into segmentation algorithms, we need to understand the basics of video encoding.


* **I-Frame (Intra-coded picture)**: Keyframe. It is a complete image that can be decoded without depending on other frames. Usually also the starting point of scene transitions.
* **P-Frame (Predicted picture)**: Forward-predicted frame. Only stores the difference from the previous frame.
* **B-Frame (Bi-predictive picture)**: Bi-directional predicted frame. References both past and future frames for compression, highest compression ratio.

**GOP (Group of Pictures)**: The sequence between two I-frames. When a video player seeks, it typically "snaps" to the nearest I-frame because decoding must start there. Our segmentation strategy must leverage this to accelerate.

#### 8.1.2 Algorithm Selection and Strategy



![Figure 8-1: Two Video Scene Segmentation Strategies and HSV Histogram Difference](../../images/part3/图8_1_视频场景切分的两种策略与HSV直方图差异.png)
*Figure 8-1: Two Video Scene Segmentation Strategies and HSV Histogram Difference*

**PySceneDetect** is the industry-standard open-source tool. It provides multiple detectors, with core logic based on inter-frame difference analysis:

* **Strategy One: Threshold Detector (Hard Cut)**
    * **Principle**: Compute average difference (Delta) between adjacent frames in HSV color space or RGB brightness. When Delta > `threshold` (e.g., 30_0), mark as cut point.
    * **Applicable**: Most movies and user-generated content (UGC).
    * **Limitation**: Cannot detect gradual transitions.

* **Strategy Two: Adaptive Detector (Gradual Transitions / Fast Cut)**
    * **Principle**: No longer uses fixed threshold; maintains a sliding window. Compares ratio of "current frame" vs. "average frame difference within window."
    * **Applicable**: Fade in/out, dissolve, or action scenes with intense camera movement.

**Advanced Strategy: Two-Stage Cascade Splitting**
Running PySceneDetect on full decoded TB-scale video is very slow. We recommend the industrial "coarse then fine" approach:

1.  **Level-1 (Metadata Scan)**: Use `ffprobe` to quickly scan video stream metadata, extract all **I-Frame** timestamps. I-frames often appear at scene transitions (encoders tend to insert I-frames at abrupt changes). This step requires no frame decoding; speed is 100x+ over playback.
2.  **Level-2 (Content Analysis)**: Only run PySceneDetect's `ContentDetector` for precise frame-level localization within ±2 seconds of Level-1 identified potential cut points.

#### 8.1.3 Core Code: Scene Detection and Lossless Segmentation

The code below demonstrates the standard segmentation flow in production. Note the "stream copy" technique—key to avoiding storage explosion when processing massive video.

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_video_scenes(video_path, output_dir, threshold=27_0):
    """
    Detect scenes and cut video with ffmpeg losslessly
    Args:
        video_path: Input video path
        output_dir: Output directory
        threshold: Segmentation threshold (empirical: 27_0 works for most 1080p video)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info(f"Starting scene detection for: {video_path}")

    # 1. Scene detection
    # threshold=27_0: HSV space histogram difference threshold
    # min_scene_len=15: Ignore segments shorter than 0.5s (30fps).
    # Very short segments are usually flash, glitch, or segmentation noise—unsuitable for training data.
    scene_list = detect(
        video_path, 
        ContentDetector(threshold=threshold, min_scene_len=15)
    )
    
    # 2. Statistics and filtering
    # Add logic here: e.g., merge overly short adjacent scenes, or discard scenes under 3 seconds
    valid_scenes = []
    for scene in scene_list:
        start, end = scene
        duration = (end.get_frames() - start.get_frames()) / start.get_framerate()
        if duration >= 3_0: # Keep only segments >3s for training
            valid_scenes.append(scene)

    logging.info(f"Detected {len(scene_list)} scenes, kept {len(valid_scenes)} valid scenes.")
    
    # 3. Split video (Stream Copy)
    # Key: arg_override='-c:v copy -c:a copy'
    # This instructs ffmpeg to directly copy the binary stream without [decode -> pixels -> encode].
    # Benefit 1: Extremely fast (limited by disk I/O, not CPU).
    # Benefit 2: 100% lossless quality, no re-encoding artifacts.
    split_video_ffmpeg(
        video_path, 
        valid_scenes, 
        output_dir=output_dir, 
        show_progress=True,
        arg_override='-c:v copy -c:a copy' 
    )

# Pitfall: Data storage explosion disaster
# NEVER decode segmented video into image sequences (png/jpg) or numpy arrays for long-term storage!
# Do the math:
# 1 hour 1080p H.264 video ≈ 2GB
# Decoded: 3600s * 30fps * 1920 * 1080 * 3 bytes ≈ 670 GB
# Expansion factor > 300x.
# Always store in compressed format (mp4/mkv), only use GPU (NVDEC) for real-time decoding in training DataLoader __getitem__.
```

---

### 8.2 Video Tokenization: From Pixel Ocean to Discrete Islands

For Sora, Gen-2 and other Transformer-based diffusion models (DiT), modeling directly in pixel space is infeasible. A 4-second 1080p video contains about $3 \times 10^8$ pixels; computing the attention matrix would cause instant OOM.

Therefore, video must first be "compressed" into discrete tokens in latent space. This process is done by **Video Tokenizer**.

#### 8.2.1 Traditional Approach Pain: VQ-VAE and "Dead Codes"

**VQ-VAE (Vector Quantized Variational AutoEncoder)** is the foundation of early video generative models (e.g., VideoGPT).

* **Flow**:
    1.  **Encoder**: Split video into 3D patches (e.g., $16 \times 16 \times 16$ spatio-temporal blocks), compress to low-dimensional vectors $z_e(x)$.
    2.  **Quantization**: Maintain a Codebook with $K$ prototype vectors (Embeddings). For each $z_e(x)$, find the nearest vector $e_k$ in Codebook by Euclidean distance to replace it.
    3.  **Decoder**: Use $e_k$ to reconstruct video.

* **Fatal flaw: Codebook Collapse**
    Early in training, only a few codes (e.g., Code #5 and #100) are accidentally selected. Since only selected codes receive gradient updates, they become "better" and easier to be selected again. This forms a "rich get richer" Matthew effect.
    * **Consequence**: 90% of Codebook vectors become "dead codes," never used. This leads to extremely low effective vocabulary, generating blurry and detail-lacking video.
    * **Remediation**: Traditional methods require complex Reset strategies (e.g., k-means reset), training extremely unstable.

#### 8.2.2 SOTA Approach: MagViT-v2 and LFQ

Google DeepMind introduced **LFQ (Lookup-Free Quantization)** in MagViT-v2, fundamentally changing the game.



* **Core idea: No lookup, direct computation.**
    LFQ abandons the "find nearest neighbor" approach and directly generates tokens from the **sign** of latent variables.

* **Mathematical principle**:
    Assume Encoder output latent vector $z \in \mathbb{R}^D$ (e.g., $D=18$).
    LFQ binarizes each dimension:
    $$q_i = \begin{cases} 1 & \text{if } z_i > 0 \\ 0 & \text{if } z_i \le 0 \end{cases}$$
    
    Then combine the $D$ binary bits into an integer index:
    $$\text{Token ID} = \sum_{i=0}^{D-1} q_i \cdot 2^i$$

* **Why is LFQ revolutionary?**
    1.  **Infinite effective codebook**: If $D=18$, the natural codebook size is $2^{18} = 262,144$. All codes are combinations of $D$ independent dimensions; each dimension always participates in gradient updates. **Codebook utilization is constant at 100%.**
    2.  **Zero compute cost**: No expensive "full codebook distance computation"—only simple bit operations.
    3.  **Spatio-temporal compression**: MagViT-v2 combines **3D Causal CNN**, preserving temporal causality while compressing space (current token never leaks future information), critical for generative models.

#### 8.2.3 Architecture Comparison Table

| Feature | VQ-VAE (TATS/VideoGPT) | MagViT-v2 (LFQ) |
| :--- | :--- | :--- |
| **Quantization Mechanism** | Nearest Neighbor Search (lookup) | Sign Function (sign projection) |
| **Vocabulary Size (Vocab)** | Typically 1024 - 8192 (limited by VRAM and collapse) | $2^{18}$ (262k) or larger, easily extensible |
| **Codebook Utilization** | Low (prone to collapse, needs EMA etc.) | **100% (design avoids collapse)** |
| **Gradient Backprop** | Requires Straight-Through Estimator (STE) | Improved Entropy Penalty + STE |
| **Generation Quality** | Prone to blur, detail texture loss | Extremely clear, even better than original (denoising effect) |
| **Inference Speed** | Slower (especially with large codebook) | Extremely fast |
---

The evolution from VQ-VAE to MagViT-v2 is not simple parameter optimization but a paradigm shift in video discretization technology—from "search-based approximation" to "computation-based construction."

First, in computational complexity and scalability, traditional VQ-VAE has fundamental bottlenecks. Its quantization depends on nearest neighbor search, requiring computing Euclidean distance between feature vectors and all $K$ prototypes in the codebook—time complexity $O(K)$. This means expanding vocabulary to improve representation directly causes linear growth in inference latency. In contrast, MagViT-v2's LFQ (Lookup-Free Quantization) abandons lookup, using the sign function to project latent variables to binary strings. This process reduces compute complexity to constant $O(1)$, enabling the model to support $2^{18}$ or larger vocabulary without sacrificing inference speed, resolving the contradiction between large vocabulary and low latency.

Second, in codebook utilization and training stability, the two differ markedly. VQ-VAE has long suffered from "Codebook Collapse"—some encoding vectors never activate due to initialization or uneven gradient allocation, causing effective vocabulary to be far below design (often only 1024-8192). This forces researchers to introduce EMA (exponential moving average) or k-means reset and other complex engineering tricks. MagViT-v2's LFQ, based on independent dimension binarization combination, mathematically guarantees that codebook space is "combinatorially generated" rather than "discretely searched." As long as latent space dimensions stay active, the combined codes naturally cover the entire codebook space, achieving theoretical 100% utilization.

In summary, MagViT-v2's LFQ achieves unification of high compression, high fidelity, and low compute cost, completely solving traditional VQ-VAE's defects in detail texture loss and poor spatio-temporal consistency. For building Sora-scale massive video generative models, MagViT-v2 and derived Tokenizer architectures have become the industry's preferred choice.

### 8.3 Audio Alignment: WhisperX and Forced Alignment

Video is not only visual data—audio (Audio) provides natural, temporally dense text descriptions. Using audio, we can let the model learn multi-modal associations like "explosion sound corresponds to explosion light," "crying corresponds to tears."

However, ordinary ASR (e.g., raw Whisper) only gives "sentence-level" timestamps, typically 1-2 second error. This is completely insufficient for fine video training (e.g., lip-sync). We need **WhisperX**.


![Figure 8-2: Precision Comparison of Ordinary ASR (Segment-level) vs WhisperX (Word/Phoneme-level)](../../images/part3/图8_2_ASR与WhisperX的精度对比.png)
*Figure 8-2: Precision Comparison of Ordinary ASR (Segment-level) vs WhisperX (Word/Phoneme-level)*

#### 8.3.1 Why Forced Alignment?
* **ASR (OpenAI Whisper)**:
    * Output: `"Hello world"` -> `Timestamp: [0_0s -> 2_0s]`
    * Problem: Model only knows the sentence falls within these 2 seconds, not exactly when "world" starts.
* **Forced Alignment (WhisperX)**:
    * Principle: First transcribe to text, then use a pre-trained acoustic model (e.g., Wav2Vec2) to forcibly match **phonemes** in the text with audio waveform.
    * Output:
        * `"Hello"`: `[0_12s -> 0_58s]`
        * `"world"`: `[0_85s -> 1_45s]`
    * **Value**: You can build training pairs like: when video frame is at 0_85s, force model to focus on "world" Text Embedding. This is the foundation for fine multimodal alignment.

#### 8.3.2 Engineering Implementation: WhisperX Full Pipeline
WhisperX is a complex Pipeline combining VAD (voice activity detection), Whisper (transcription), Wav2Vec2 (alignment), and Pyannote (speaker diarization).

```python
import whisperx
import gc
import torch

def align_audio_transcript(audio_file, device="cuda", batch_size=16):
    """
    Use WhisperX for transcription and word-level forced alignment
    """
    # Step 1: Transcription
    # Use Large-v2 model for transcript accuracy
    # compute_type="float16" significantly speeds up, but requires Ampere+ GPU (A100/A10/3090/4090)
    print("1. Loading Whisper model...")
    model = whisperx.load_model(
        "large-v2", 
        device, 
        compute_type="float16" 
    )
    
    print("2. Transcribing...")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    
    # Critical: VRAM management
    # Whisper model is huge, and the next Alignment model is also VRAM-heavy.
    # Must explicitly delete model and trigger garbage collection, otherwise easily OOM (Out of Memory).
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Forced Alignment
    # Auto-loads corresponding language Wav2Vec2 model (e.g., wav2vec2-large-960h for English)
    print("3. Aligning...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], 
        device=device
    )
    
    # align() executes a Dynamic Programming-like algorithm
    # finding best matching path between text phoneme sequence and audio waveform features
    aligned_result = whisperx.align(
        result["segments"], 
        model_a, 
        metadata, 
        audio, 
        device, 
        return_char_alignments=False # Set True for character-level alignment (e.g., for karaoke subtitles)
    )

    # Result contains word_segments with precise start/end for each word
    # e.g.: [{'word': 'Hello', 'start': 0_1, 'end': 0_5, 'score': 0_98}, ...]
    return aligned_result

# Advanced tip:
# For speaker diarization (who said what), further call:
# diarize_model = whisperx.DiarizationPipeline(use_auth_token="YOUR_HF_TOKEN", device=device)
# diarize_segments = diarize_model(audio)
# whisperx.assign_word_speakers(diarize_segments, aligned_result)
```

#### 8.3.3 Production Environment Pitfalls

1.  **VAD Misjudgment and Background Music Interference**:
    * **Problem**: WhisperX heavily relies on VAD for segmenting silent segments. If video BGM is loud, VAD may treat entire segment as speech, or vice versa, drowning out speech.
    * **Solution**: Introduce **Demucs** or **Spleeter** for source separation.
    * **Flow**: `Raw Audio` -> `Demucs (Extract Vocal Track)` -> `WhisperX`. Feed only extracted pure vocal track to recognition for significantly higher accuracy.

2.  **Multi-speaker Overlap (Overlapping Speech)**:
    * **Problem**: Whisper weakly handles multiple people speaking simultaneously (Cocktail Party Problem), usually only transcribing the loudest person or generating confused text.
    * **Solution**: Enable `diarization=True`. Though this adds 30%-50% inference time, for TV drama, interview-class video data it's the only way to distinguish "who said what," avoiding model confusion of character identity.

3.  **Hallucination Timestamps**:
    * **Problem**: Whisper may produce "hallucinations" during long silence or pure music segments—repeating previous lyrics with wrong timestamp.
    * **Check**: In post-processing, check `word['score']` (confidence). If a consecutive string of words has confidence below 0_4, recommend discarding that segment's alignment.

---
