# Part 3: Multimodal Data Engineering

## Positioning

Part 3 covers image-text, documents, video, audio, and cross-modal alignment. It discusses multimodal sample structures, visual and acoustic cleaning, re-captioning, document understanding, temporal slicing, cross-modal alignment, and fusion training. This part extends the text pre-training pipeline from Part 2 into images, layouts, timelines, and heterogeneous modality spaces.

Multimodal data introduces four additional challenges. First is alignment: text, images, audio, and video do not naturally describe the same object, so semantic, spatial, or temporal anchors are needed. Second is representation: visual and acoustic signals must pass through patches, features, tokens, or placeholders before they can enter language-model training interfaces. Third is evaluation: image clarity, OCR reliability, audio-video synchronization, and cross-modal semantic consistency are hard to judge with cheap text metrics alone. Fourth is cost: decoding, re-captioning, OCR, ASR, vector scoring, and manual sampling all raise preprocessing and pre-training validation cost.

Chapters 8 through 11 start with image-text data engineering, then move to re-captioning and document understanding, then video and audio data engineering, and finally cross-modal alignment and fusion. The main line is this: single-modality cleaning does not prove cross-modal supervision is valid.

## Learning Objectives

- Distinguish the engineering differences among image-text pairs, interleaved image-text, document screenshots, audio/video clips, and cross-modal fusion samples.
- Master basic flows for image quality filtering, semantic alignment, re-captioning, OCR enhancement, and document structuring.
- Understand how video slicing, ASR, diarization, audio-video synchronization, and event labels form long-temporal samples.
- Design object-level, segment-level, and document-level cross-modal alignment data and identify common failure modes.
- Evaluate whether a multimodal data pipeline can enter training from the perspectives of cost, compliance, sampling, and traceability.

## Prerequisites

Readers should understand the methods in Part 2 for text pre-training sources, cleaning, deduplication, tokenization, serialization, and quality loops. Readers from computer vision, speech, or multimedia systems may focus on how image, audio, and video signals are converted into training-readable tokens, features, JSONL schemas, and metadata constraints.

## Chapter Logic

Chapter 8 establishes the asset view of static image-text samples, including image cleaning, CLIP/SigLIP filtering, AnyRes, and data mixture. Chapter 9 moves into fine-grained supervision production, including re-captioning, OCR, layout structure, and document understanding samples. Chapter 10 handles the time dimension through video slicing, audio transcription, acoustic separation, and temporal alignment. Chapter 11 integrates the previous chapters into cross-modal fusion samples, with emphasis on alignment layers, hard negatives, mixture ratios, and evaluation loops.

This part connects backward to the text pre-training pipeline in Part 2 and forward to instruction alignment and preference data in Part 4. In other words, Part 3 answers whether the model can see, hear, and align different modalities; Part 4 then asks how the model should act under human instructions and preferences.

## Part Contents

- [Chapter 8: Image-Text Pair Data Engineering](ch08_multimodal_image.md)
- [Chapter 9: Re-captioning and Document Understanding](ch09_recaptioning_ocr.md)
- [Chapter 10: Video and Audio Data Engineering](ch10_video_audio.md)
- [Chapter 11: Cross-modal Alignment and Fusion](ch11_cross_modal_alignment.md)

Suggested order: read Chapters 8 through 11 sequentially. Readers with VLM experience may first read Chapter 11 to understand cross-modal alignment objectives, then return to Chapters 8, 9, and 10 to locate where each alignment signal comes from.
