# Chapter 8: Image-Text Pair Data Engineering

## Abstract

This chapter discusses the fundamental engineering problems behind image-text pairs and interleaved image-text data, with a particular focus on why visual data cannot simply reuse the cleaning paradigm of pure text. It first explains the engineering challenges created by visual noise, semantic mismatch, resolution cost, and image representation. It then compares three sample paradigms: image-caption pairs, interleaved image-text documents, and document-grounded screenshots. The cleaning section begins with image decoding, resolution and aspect-ratio filtering, and NSFW, watermark, and privacy blocking, then introduces CLIP/SigLIP-based semantic matching and multi-granularity recaptioning. The second half of the chapter discusses AnyRes dynamic patching, aspect-ratio grouping, multimodal data mixing, and quality risks. An anonymized composite case illustrates why stock-photo watermark contamination and secondary cleaning are necessary. By the end, readers should be able to design a traceable, measurable, and cost-controlled image-text preprocessing pipeline.

## Keywords

Image-text pairs; interleaved image-text; CLIP Score; SigLIP; AnyRes; image cleaning; recaptioning; multimodal data

## Learning Objectives

- Explain the distinctive noise types, semantic alignment issues, resolution costs, and quality-evaluation challenges of image-text data.
- Distinguish image-caption pairs, interleaved image-text samples, and document-grounded samples.
- Design a multi-stage cleaning workflow for image decoding, size filtering, watermark and privacy blocking, and semantic matching.
- Explain the appropriate boundaries of CLIP/SigLIP filtering, recaptioning, and AnyRes dynamic patching.
- Identify how stock-photo contamination, image-text mismatch, and unbalanced data mixing affect model training.

## 8.1 Why Multimodal Data Is Harder Than Text

When an NLP data engineer first takes over a data-cleaning task for a vision-language model (VLM), the most immediate change is this: many deterministic rules that work well for pure text cover only a small fraction of problems in images.

### 8.1.1 The Hidden and Uncertain Nature of Visual Noise

In a pure-text data factory, "dirty data" or corrupted text can often be detected at low cost: write regular expressions, run MinHash, calculate perplexity (PPL), and sometimes filter without touching a GPU. In image data, however, noise often depends on semantics, spatial location, and visual context:

- **Concept-isolation noise**: a 4K landscape photo may be sharp, colorful, and visually beautiful, but an adversarial edit may place a semi-transparent NSFW watermark in one corner occupying only 15 x 15 pixels. Such an image creates high compliance risk for multimodal systems.
- **Spatial-distortion noise**: a cluttered street-market photo may be captioned as "a comb on the stall." The key comb may occupy only three pixels, making it hard for humans to see and likely to vanish after convolutional downsampling.
- **Frequency-domain compression noise**: a screenshot containing a detailed financial statement or medical ECG may have passed through repeated JPEG recompression. Severe ringing artifacts appear around high-frequency details. Humans may still infer the text, but OCR systems that rely on edge features are heavily damaged.

These forms of hidden visual noise cannot be removed through ordinary hash fingerprints or MD5 file comparisons. In multimodal data streams, teams usually need to deploy and maintain several pretrained visual classifiers, such as a ResNet-50-based binary watermark detector or an LAION Aesthetics-style scorer, for dense inference. Image cleaning itself therefore consumes meaningful GPU compute.

### 8.1.2 Semantic Absence and Cross-Modal Polysemy: WebTox

Multimodal alignment learning relies on a premise: the crawled `(Image, Text)` pair should describe the same thing. Yet massive quantities of web images and alt text are not naturally aligned.

Because web crawling and SEO practices have evolved in complex ways over more than a decade, crawler engineers frequently encounter high-risk pairs such as:

- **What the human eye sees**: a golden retriever happily running on grass.
- **The alt text extracted from HTML**: "2023 free shipping genuine premium discount full-site promotion buy one get one pet supplies."

If this commercially motivated inducement text is fed into a model without causal separation, back-propagation forces the model to establish false associations. It may learn unreasonable bindings such as "golden retriever fur texture equals free-shipping discount," a form of cross-modal semantic mismatch.

In multimodal data-engineering practice, researchers sometimes call this phenomenon **WebTox**, or web semantic toxic data. Experiments have shown that a large amount of this noise can make a hundred-billion-parameter model perform worse on visual-question-answering benchmarks than an unfine-tuned 1B baseline. CLIP Score-centered cross-modal semantic filtering emerged partly to mitigate this problem systematically.

### 8.1.3 The Cost Tradeoff Between Image Resolution and GPU Memory

In pure language modeling, the cost of processing either a report or a short poem mainly scales linearly with token length. In multimodal modeling, **image resolution significantly amplifies compute cost**.

Consider a mainstream visual encoder based on ViT (Vision Transformer) (Dosovitskiy et al. 2020). Suppose the patch size is fixed at $14 \times 14$ pixels:

- A $224 \times 224$ image is cut into $(224/14) \times (224/14) = 256$ patch tokens. The self-attention cost is roughly $256^2 = 65,536$ operations.
- If we raise the input resolution to $1008 \times 1008$ so that the model can read small digits on a scanned invoice, the visual-token length becomes $(1008/14) \times (1008/14) = 5184$.
- Since standard Transformer attention has quadratic complexity with sequence length, the attention cost in a single layer rises to $5184^2 \approx 26,873,856$ operations.

Increasing the image side length by only 4.5x can therefore increase **attention-layer compute** by nearly **410x**. This refers to the quadratic self-attention portion, not the full model FLOPs; FFN layers scale linearly with sequence length, so the actual total training-compute increase is still substantial but somewhat lower than 410x. One core task in image-text data engineering is therefore to balance local-detail preservation against training cost through dynamic cropping, dimensionality reduction, and multi-scale patching.

![Figure 8-1: Overview of image-text data engineering](../../images/part3/multimodal_data_panorama.png)

*Figure 8-1: Overview of multimodal image-text data engineering. The pipeline starts from DOM-tree crawling and PDF parsing, then moves through format parsing, watermark filtering, CLIP semantic alignment, interleaved-sequence assembly, and tokenized representation. Distributed computing and metadata form the foundation across the pipeline. Source: drawn for this book. Alt text: an overview of image-text data engineering showing DOM extraction, image download, format parsing, filtering, semantic alignment, recaptioning, and sequence assembly.*

---

## 8.2 Image-Text Sample Paradigms: From Pairs to Interleaving

Different training objectives require different data formats. Based on recent frontier architectures such as Flamingo (Alayrac et al. 2022), LLaVA (Liu et al. 2023), and GPT-4V, typical multimodal text data includes three major paradigms.

### 8.2.1 Image-Caption Pairs

This is the most basic and scalable paradigm.

- **Format**: exactly one image mapped to one independent descriptive text, such as `{ "image": "dog.jpg", "text": "A golden retriever playing fetch in the park." }`.
- **Representative open datasets**: LAION-5B (Schuhmann et al. 2022), COYO-700M.
- **Applicable stage**: mainly used during cold-start **contrastive pretraining**, such as training a CLIP precursor model or building a baseline visual-perception capability for a newly connected vision encoder.
- **Main limitation**: it is difficult to teach reasoning through image-caption pairs alone; they are better at building basic object recognition and cross-modal retrieval.

### 8.2.2 Interleaved Image-Text

To give a model the ability to reason across multiple images in complex contexts, the data engine must extract and restore the native interleaved layout from web pages.

- **Format**: similar to Wikipedia articles or long-form posts: a setup paragraph + `<img_1>` + development text + `<img_2>` + final summary. Image tokens are treated as special vocabulary items embedded inside a long text sequence.
- **Representative open datasets**: OBELICS (Laurencon et al. 2023), MMC4 (Zhu et al. 2023).
- **Applicable stage**: an important data form for modern **generative VLM pretraining**. It teaches the model to infer what later text or a later image should be given previous text and image 1.
- **Collection challenge and DOM parsing engineering**: interleaved data is much harder to build. A traditional text crawler skips `<img>` tags, but an interleaved-data crawler must parse a large and messy HTML DOM tree and compute **relative distances based on rendered coordinates**.

In many modern pages with complex CSS, the order in the document tree is not the order a user sees visually. Extracting only by HTML tag order can incorrectly bind a bottom-page disclaimer to a top-page illustration.

For this reason, engineering teams usually use a headless browser with a rendering engine, such as Playwright, to run JavaScript, generate a page snapshot, and extract elements using rules similar to the following.

Listing 8-1 shows simplified logic for extracting interleaved DOM nodes.

**Listing 8-1: Simplified DOM interleaved-node extraction**

```python
# Simplified pseudocode for extracting interleaved DOM nodes
text_nodes, img_nodes = get_rendered_nodes(page)
interleaved_sequence = []

for node in all_nodes_sorted_by_y_axis():
    if node.type == 'TEXT':
        if len(node.content.split()) > 5:  # discard short text such as navigation
            interleaved_sequence.append(node.content)
    elif node.type == 'IMAGE':
        if node.width > 200 and node.height > 200:
            # Convert a valid image into a placeholder token and store its URL in a side channel
            interleaved_sequence.append(f"<img_{node.id}>")
            save_to_image_db(node.url, node.id)
```

If the DOM structure is extracted out of order, the model learns the wrong image-text correspondence.

### 8.2.3 Long-Document Understanding and Screenshot Grounding

For real business use cases such as reading financial reports or invoices, natural-image training is not enough. High-resolution document data must be introduced.

- **Format**: the input is a rendered high-frequency, high-resolution document screenshot, such as an arXiv paper page or a dense Excel screenshot. The output is a structured JSON value sequence or bounding-box coordinates marked with `<box>`.
- **Applicable stage**: depends on high-resolution patching and OCR assistance. It is mainly used in SFT to teach precise value extraction and logical-structure understanding, such as formula and chart references in page layouts.
- **Coordinate normalization**: in grounding tasks, the model must output concrete pixel coordinates. Because training images have very different resolutions, the original absolute pixel coordinate `(X, Y)` is usually mapped into a discrete token bucket in `[0, 1000]`, such as `[<loc_255>, <loc_899>]`. This discretization turns continuous spatial coordinates into a vocabulary-like form that an LLM can process.

**Table 8-1: Image-text sample types, characteristics, and applicable tasks**

| Sample type | Data characteristics | Core acquisition method | Best-fit stage | Key capability gained |
| :--- | :--- | :--- | :--- | :--- |
| **Pure image-caption** | One-to-one text/image pairs, high noise | Web `<img alt>`, public-cloud OSS crawling | Alignment pretraining | Basic feature perception and cross-modal retrieval |
| **Interleaved image-text** | Many-to-many text/image mapping, long sequences | DOM rendering and parsing, PDF linearization | Main generative pretraining | Multi-step reasoning and few-shot contextual perception |
| **Long-document screenshot** | Very high resolution, text-dense | PDF rendering, automated headless-browser screenshots | Deep SFT / reinforcement learning | Layout understanding and form, paper, or invoice extraction |
| **Grounded caption** | Long text with bounding boxes `<box>` | Annotator box selection or rewriting by a closed-source large model | Advanced SFT / RAG alignment | Fine-grained spatial perception and hallucination resistance |

---

## 8.3 Cleaning, Filtering, and Semantic Alignment, Part I: Basic Cleaning

Raw data crawled from the web has highly uneven quality. It must pass through at least three filtering funnels at different levels. We call this stage the pre-cleaning phase; it involves heavy I/O, image decoding, and hardware-accelerated classifiers.

### 8.3.1 Image Decoding and GPU-Side Preprocessing

In text cleaning, `JSON.loads` or `open()` is almost free. For image archives at tens of TB or even PB scale, **decoding** can become the biggest throughput bottleneck in the entire training cluster. Web images may be old JPEGs, large PNGs, corrupted files, or WebP variants with malformed headers or embedded ICC profile errors.

If the pipeline uses CPU `Pillow` or `OpenCV-Python` to resize and normalize images:

- Even on an 8 x A100 node with PyTorch DataLoader `num_workers=64`,
- Dense CPU image resizing can saturate all physical cores,
- And interprocess communication must move large uncompressed RGB tensors into GPU memory, which can make PCIe bandwidth a bottleneck and produce GPU starvation, with MFU dropping below 20%.

**Engineering solution: an end-to-end NVIDIA DALI pipeline**

Large enterprise image-text processing arrays often force the pipeline onto **NVIDIA DALI (Data Loading Library)** (NVIDIA 2023), an accelerated GPU-memory-level workflow. The core idea is **push bits into the GPU as early as possible and decompress inside GPU memory**:

1. **CPU only moves bytes**: the CPU reads undecoded JPEG byte streams and does not decode them.
2. **NVJPEG hardware decoding**: after the bytes are sent through PCIe to the GPU, the GPU's dedicated JPEG decoder (NVJPEG) decompresses them directly in GPU memory.
3. **Fused operator transforms**: cropping, resizing, mean-variance normalization, and related operations are compiled into a CUDA graph and run directly on tensors.

With GPU-side decoding and fused preprocessing, latency for one 512 x 512 image can drop from roughly 8 ms on the CPU side to below 0.4 ms. This is an illustrative performance figure as of 2026-06; actual gains depend on image format, hardware, DALI version, and concurrency settings, and production systems should re-benchmark on target hardware.

### 8.3.2 Resolution and Aspect-Ratio Control

One of the fastest ways to improve the raw-cleaning stage is to define size-filtering rules:

- **Reject pixel islands**: discard images whose shorter side is below `224px` or whose total file size is below `20KB`. These are usually UI icons, such as like buttons or arrows, and provide little semantic signal for large models.
- **Identify extreme aspect ratios**: web long images, such as full-length e-commerce promotional posters, can be extreme, for example width 500 and height 9000. If such an image is forced into a $336 \times 336$ square for a normal ViT encoder, the content is severely compressed and shape features are lost. Pipelines therefore usually require the aspect ratio to stay within $[0.33, 3.0]$. Images outside that interval are flagged and routed to a targeted dynamic-slicing bypass; see Section 8.4.

### 8.3.3 Targeted Blocking of NSFW Content, Face Privacy, and Watermarks

Image-text engineering faces stricter ethics and compliance concerns than pure-text engineering: multimodal models should not learn private faces of ordinary people, sensitive content, or copyrighted watermark templates.

At this stage, the pipeline usually chains three or four small visual classifiers:

1. **NSFW classifier**: images whose probability exceeds a threshold, such as 0.4, are deleted or isolated for review.
2. **Watermark detector**: many web images come from stock-photo sites such as Getty Images or Shutterstock. If a model absorbs these images, it may frequently hallucinate watermark text in generated responses and create commercial risk. Samples with dense diagonal patterns or bright central watermarks must be filtered.
3. **Blur/aesthetic threshold**: an aesthetic predictor similar to the LAION team's AES model can remove heavily defocused, extremely dark, or color-noisy low-quality images.

**Multimodal sensitive-data filtering checklist, industrial version:**

- [ ] Has a text-side blocklist been integrated to filter or isolate violent, sexual, or otherwise sensitive text in `<alt>` tags?
- [ ] Has the visual NSFW classifier been supplemented with training for anime and illustration models to reduce misses?
- [ ] For portrait privacy, has a face-blurring algorithm been applied to high-resolution ordinary-person faces, excluding public figures where policy allows?
- [ ] Is the commercial-watermark blocking library updated weekly to prevent new image-host contamination?

After these three types of basic cleaning, a 10-billion-scale crawled library may shrink below 4 billion samples. The remaining images are visually cleaner, but they still have not proven semantic correspondence with text. CLIP Score and related cross-modal matching metrics are needed next.

---

## 8.4 Cleaning, Filtering, and Semantic Alignment, Part II: Deep Semantic Filtering

After low-level visual errors are removed, multimodal data engineering reaches its hardest stage: quantitatively measuring the match between an image and a sentence.

### 8.4.1 CLIP Score and SigLIP: The Quantitative Philosophy

Before CLIP (Contrastive Language-Image Pre-training) (Radford et al. 2021), image-text matching depended mainly on human rules or weakly supervised heuristics. CLIP uses large-scale contrastive learning to map image embeddings and text embeddings into the same high-dimensional vector space.

**1. Basic filtering and the legacy of InfoNCE Loss**

We usually run a stable pretrained CLIP, such as open-source `OpenCLIP ViT-L/14`, on both the image and its caption, then calculate the **cosine similarity**, also called CLIP Score.

- **High match (Score > 0.30)**: the image and text are highly consistent, for example the image is a cat and the text says "an orange cat sunbathing." These samples enter the golden dataset directly.
- **Low match (Score < 0.22)**: severe mismatch, for example the image is a cat and the text says "follow my account." These samples are usually discarded because they contribute reverse gradient noise.
- **Medium match (0.22 < Score < 0.30)**: a gray zone. Instead of discarding expensive collected data immediately, the pipeline triggers the recaptioning process described in the next section.

> **Note**: the thresholds above, 0.22 and 0.30, are based on `OpenCLIP ViT-L/14`. If `ViT-B/32` or `SigLIP` is used, the same data can have a significantly different score distribution. Thresholds must be recalibrated on the target data and should not be reused blindly.

**2. From CLIP to SigLIP: abandoning global Softmax**

In large enterprise data pipelines, traditional CLIP models are increasingly replaced by **SigLIP (Sigmoid Loss for Language Image Pre-Training)** (Zhai et al. 2023). Traditional CLIP computes a global Softmax probability over all image-text pairs inside the batch. This creates an engineering issue: with a very large distributed batch size, such as $32768$, the model must distinguish many fine-grained pair differences and may become overly sensitive to certain hard negatives, making inference-time CLIP Scores unstable.

SigLIP converts this global multi-class problem into **pairwise binary sigmoid prediction**. This gives SigLIP better tolerance for partial matches and complex-background image-text pairs, and a more stable score distribution. Engineering teams can set more consistent cutoffs, while still calibrating them on the target data.

Listing 8-2 shows a simplified SigLIP/CLIP image-text alignment filter.

**Listing 8-2: Simplified SigLIP/CLIP image-text alignment filter**

```python
# Pseudocode for SigLIP/CLIP image-text alignment filtering
import torch
from transformers import AutoProcessor, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
# Use SigLIP weights that do not require an extremely large batch size
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)

def filter_by_semantic_score(image, text_caption, threshold=0.25):
    inputs = processor(
        text=[text_caption],
        images=image,
        padding="max_length",
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract features and compute dot-product similarity
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)

        # Obtain logits scaled by the model's temperature factor
        logits_per_image = image_embeds @ text_embeds.T * model.logit_scale.exp()
        similarity = logits_per_image.item()

    return similarity >= threshold, similarity
```

### 8.4.2 Saving Valuable Images: Multi-Granularity Synthetic Recaptioning

When an image has high resolution, good composition, and rare entities, but its original web text is only a label such as "IMG_20230401.jpg," discarding it wastes a data asset. If compute allows, using expert VLMs such as LLaVA-1.5 (Liu et al. 2024), Qwen-VL-Max (Bai et al. 2023), or GPT-4V to regenerate descriptions is an important way to improve image-text training quality.

Modern large-model engineering usually applies a **multi-granularity recaptioning array** to this image batch so that both early cold-start alignment and later long-text generation are supported:

1. **Short caption extraction**
   - **Prompt**: "In one sentence only, identify the main subject in the center of the image and its core action. No more than 15 Chinese characters."
   - **Output**: "An Asian woman playing violin."
   - **Engineering value**: a short sentence with concentrated information and no complicated modifiers is well suited to early pretraining, where it establishes the basic attention binding between the vision encoder and text LLM. Feeding long descriptions too early can disperse alignment focus and increase hallucination.

2. **Detailed dense-caption rendering**
   - **Prompt**: "Objectively describe the composition, lighting, character features, clothing colors, background elements, and possible emotional atmosphere in about 150 Chinese characters."
   - **Output**: "At dusk in Times Square, the sky shows deep purple-orange clouds. In the center of the image, an Asian woman wearing distressed jeans and a white sweater plays a reddish-brown wooden violin with her eyes closed. The depth of field is shallow; behind her are blurred yellow taxis and glowing neon signs. The overall mood is melancholic and focused."
   - **Engineering value**: high-density information helps train detail recognition and more stable image description. Using such data in SFT improves observing, describing, and referring capabilities.

3. **Structured bounding boxes and OCR injection**
   - **Parallel-flow merge**: observing the image with a large model is still not accurate enough, especially when dense numbers appear. A side-car recaptioning workflow calls PaddleOCR at the same time. If the long description finds a billboard in the background, the merge script converts its coordinates into special tokens and inserts them into text: `... the background contains a billboard reading "<box_45_120_350_200> Broadway 5th Ave. </box>".` This converts visual signs into locatable strings and coordinate information inside the training sample.

![Figure 8-2: Image semantic alignment and filtering flow](../../images/part3/image_semantic_alignment_flow.png)

*Figure 8-2: Image semantic alignment and filtering flow. A CLIP- and heuristic-rule-based quantitative decision tree filters out low-match samples, sends medium-match but high-value images to the recaptioning pipeline, and finally stores zero-padded or dynamically sliced images in the training pool. Source: drawn for this book. Alt text: an image semantic alignment and filtering flow showing quality filtering, CLIP scoring, recaptioning, dynamic slicing, and training-pool ingestion.*

---

## 8.5 Sampling Mixes, Representations, and Training Adaptation

After cleaning, image-text samples still need slicing, packing, and ratio mixing before training. This step directly affects memory use, effective-token ratio, and capability distribution.

### 8.5.1 Image Token Occupancy and AnyRes Dynamic Patching

In common pure-text packing, 1,000 written characters may require only 300 tokens. In multimodal training, images consume many sequence positions. For example, a normal $336 \times 336$ image processed by ViT-L/14 occupies 576 token slots.

Early VLMs, including CLIP-era models, usually resized all input images to a fixed square size: a landscape photo and a vertical long document were both compressed to $224 \times 224$, causing severe aspect-ratio distortion. To solve this, modern data factories widely introduce **AnyRes**, or dynamic high-resolution preservation, in preprocessing.

![Figure 8-3: AnyRes dynamic multi-resolution patching](../../images/part3/anyres_dynamic_patching.png)

*Figure 8-3: AnyRes dynamic multi-resolution patching. The core idea is that the high-resolution panoramic input on the left is no longer forced into a square. Instead, it is divided by an adaptive grid into $1 \times 3$ native-resolution local patches and paired with a global thumbnail in the upper-right corner before entering the vision encoder, preserving both high-frequency local features and global semantics. Source: drawn for this book. Alt text: AnyRes dynamic multi-resolution patching showing a panorama divided into local patches and combined with a global thumbnail.*

**AnyRes principles and core strategies:**

1. **Zero-padding / letterboxing**: for images whose original aspect ratio should be preserved and whose resolution does not overflow, add black or mean-color borders around the image to form a square, allowing the model to learn undistorted geometry.
2. **Multi-patch splitting / grid cropping**: dynamically match a $336 \times 1008$ vertical image to a $1 \times 3$ grid and cut it into three $336 \times 336$ local sub-patches. To preserve global context, add a heavily downsampled **global context patch**. One original image is therefore input as four matrices of 576 tokens each, consuming 2,304 tokens in total.
3. **Positional embedding injection**: sub-patches cannot be passed to the model arbitrarily. During DataLoader assembly, each patch must receive relative two-dimensional position codes such as `[<row_1>, <col_1>]`, so that the model knows which patch is on the left or right.

If interleaved image-text data is not tightly controlled, GPU memory is consumed by image tokens and text-logic learning becomes inefficient. Pipelines therefore need **aspect-ratio grouping-based sequence packing**: place image-text pairs of similar shape into the same 4096-token sequence window, insert special boundary markers `<image>` and `</image>` between images and image patches, and use an attention mask to block cross-document contamination, reducing memory waste.

### 8.5.2 Tuning the Three-Way Data Mix

A balanced MLLM pretraining mix must allocate weights to different sources carefully. The following ratios are illustrative parameters as of 2026-06 and should be calibrated using model goals and ablation experiments:

1. **General natural images, roughly 50-60%**: provide basic world-object knowledge, such as cats and dogs, cars, landscapes, color calibration, and human expressions. This portion is usually based on strictly CLIP Score-filtered open datasets, such as a refined core subset of DataComp-1B (Gadre et al. 2023).
2. **Charts, plots, and mathematical or code diagrams, roughly 15-20%**: provide abstract mathematical reasoning capability. Without this portion, the model may misinterpret line charts, stock candlestick charts, or complex mind maps.
3. **High-density OCR document screenshots, roughly 20-30%**: scanned white papers, single-page PDFs, receipt and invoice images. This data is crucial for models that act as contract-review assistants or invoice helpers because it trains the rare "fine-grained text focus" capability that natural images almost never contain.

**Table 8-2: Image-cleaning strategies and cost comparison**

| Cleaning strategy | Compute cost | Core function and benefit | Residual risks and side effects |
| :--- | :--- | :--- | :--- |
| **Basic resolution cutoff** | Very low, I/O intensive | Remove meaningless color blocks and save roughly 30% storage in an illustrative setup | May wrongly remove historically meaningful documentary images that only survived in low resolution |
| **DALI hardware-accelerated decoding** | Low to medium, GPU intensive | Relieve DataLoader bottlenecks and improve decoding by an order of magnitude | High business integration cost; malformed JPEG files can trigger low-level library exceptions |
| **NSFW / watermark detection** | Medium, CNN forward pass | Enforce commercial compliance boundaries and prevent safety risk | Adversarial tiny watermarks are hard to eliminate; detectors need continuous evolution |
| **SigLIP/CLIP alignment** | High, dual-tower features | Directly reduce semantic mismatch and form the basis of cognitive quality | High-score regions may become semantically over-smoothed and may wrongly reject metaphorical or ironic images |
| **VLM synthetic recaptioning** | Very high, LLM generation | Add details to low-information original captions | Expensive and may introduce hallucinations or repetitive style from the previous model |

---

## 8.6 Anonymized Composite Case and Long-Term Commercial Guidance

The following case is an anonymized composite. Data scale and cost are used only to explain risk types. Real projects must evaluate GPU cost, sample size, and quality gains according to their specific hardware, data licenses, and evaluation criteria.

### 8.6.1 Reverse Learning Hijacked by Stock-Photo Libraries

In an early R&D stage, one team directly downloaded a subset of a cleaned open image-text dataset for alignment training. During staged blind interaction review, evaluators found a systematic problem: regardless of what landscape photo was shown, the model frequently ended with promotional text similar to "download high-definition watermark-free images from a certain stock site."

**Lessons and remediation**: this is a stock-photo contamination phenomenon. Even when a dataset has been filtered by a relatively high CLIP Score threshold, if OCR or feature classifiers were not used during initial collection to remove anti-theft watermarks and template promotional text, large image-hosting promotional templates can seep into the model's conditional probability distribution. For commercial large models, teams **must build negative hash lists for specific stock-photo sites and perform secondary cleaning on external data**.

### 8.6.2 Long-Term Maintenance of Multimodal Data Assets

Looking across the image-text data-engineering framework, the key to competitive image-text models is not only the vision encoder or training script. It is also the quality of data preparation before the model receives its first multimodal tensor token.

Only after DOM structure parsing, image-download verification, hardware decoding, aspect-ratio cropping, safety scanning, SigLIP quantitative evaluation, and high-resolution recaptioning can raw web images and weak text be converted into multimodal paired samples with bilingual descriptions, location anchors, and quality metadata.

This data asset is not just a JSON file on disk. It is a long-term engineering capability composed of source records, filtering rules, recaptioning models, sampling records, and evaluation feedback.

## Chapter Summary

As the opening chapter of Part III, this chapter systematically described four types of challenges that distinguish multimodal data from pure text, then introduced three major structural paradigms for image-text engineering. To reduce the throughput pressure caused by image archives, it discussed GPU-side decoding and preprocessing with DALI.

For complex semantic alignment, the chapter used a diagram to explain the combined workflow of CLIP Score filtering and VLM recaptioning (see Figure 8-2). Finally, through data-mix tuning and an anonymized composite case, it described the quality boundaries that enterprise-grade vision-language model training must maintain.

Although interleaved image-text data is an important modern multimodal training form, complex B-side industrial applications, such as financial-report parsing, complex invoice verification, and handwritten medical-form recognition, cannot be handled by natural-scene images alone. The next chapter enters **Chapter 9: Recaptioning and Document Understanding**, which discusses OCR, layout parsing, and long-document-understanding data engineering.

## References

Alayrac J B, Donahue J, Luc P, Miech A, Barr I, Hasson Y, Lenc K, Mensch A, Millican K, Reynolds M, others (2022) Flamingo: A Visual Language Model for Few-Shot Learning. Advances in Neural Information Processing Systems 35:23716-23736.

Bai J, Bai S, Yang S, Wang S, Tan S, Wang P, Lin J, Zhou C, Zhou J (2023) Qwen-VL: A Versatile Vision-Language Model's Understanding, Localization, Text Reading, and Beyond. arXiv preprint arXiv:2308.12966.

Dosovitskiy A, Beyer L, Kolesnikov A, Weissenborn D, Zhai X, Unterthiner T, Dehghani M, Minderer M, Heigold G, Gelly S, Uszkoreit J, Houlsby N (2020) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT). In: International Conference on Learning Representations 2021.

Gadre S Y, Ilharco G, Fang A, Hayase J, Smyrnis G, Nguyen T, Marten R, Wortsman M, Ghosh S, Zhang G, others (2023) DataComp: In Search of the Next Generation of Multimodal Datasets. Advances in Neural Information Processing Systems 36.

Laurencon H, Saulnier L, Tronchon L, Bekman S, Singh A, Lozhkov A, Wang T, Karamcheti S, Rush A M, Kiela D, Cord M, Wolf T (2023) OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents. Advances in Neural Information Processing Systems 36.

Liu H, Li C, Wu Q, Lee Y J (2023) Visual Instruction Tuning (LLaVA). Advances in Neural Information Processing Systems 36:34892-34916.

Liu H, Li C, Li Y, Lee Y J (2024) Improved Baselines with Visual Instruction Tuning (LLaVA-1.5). In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 26296-26306.

NVIDIA (2023) NVIDIA Data Loading Library (DALI). https://github.com/NVIDIA/DALI.

Radford A, Kim J W, Hallacy C, Ramesh A, Goh G, Agarwal S, Sastry G, Askell A, Mishkin P, Clark J, others (2021) Learning Transferable Visual Models From Natural Language Supervision (CLIP). In: Proceedings of the 38th International Conference on Machine Learning, pp 8748-8763.

Schuhmann C, Beaumont R, Vencu R, Gordon C, Wightman R, Cherti M, Coombes T, Katta A, Mullis C, Wortsman M, others (2022) LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models. Advances in Neural Information Processing Systems 35:25278-25294.

Zhai X, Mustafa B, Kolesnikov A, Beyer L (2023) Sigmoid Loss for Language Image Pre-Training (SigLIP). In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 11975-11986.

Zhu D, Chen J, Shen X, Li X, Elhoseiny M (2023) MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models. arXiv preprint arXiv:2304.10592.
