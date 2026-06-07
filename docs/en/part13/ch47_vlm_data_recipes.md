# Chapter 47: VLM Data Recipes from Pre-training to Visual Alignment

"With the same ViT visual encoder and the same LLM backbone, why did our faithful reproduction of the LLaVA-1.5 training pipeline score 10 to 15 points below Qwen2.5-VL and InternVL on difficult benchmarks such as MMMU and DocVQA?"

In spring 2025, a multimodal team at a leading AI lab ran into a painful recipe failure. Their setup looked sound: use a visual encoder comparable to Qwen2.5-VL's class, connect it to a similarly sized Chinese base LLM, and follow the classic LLaVA-1.5 (Liu et al. 2024b) two-stage recipe. Stage 1 used 558K LAION-CC-SBU image-text pairs for visual alignment pre-training. Stage 2 used LLaVA-Instruct-150K for full fine-tuning. Training took three weeks, the GPU bill reached six figures, and early internal tests looked decent: fluent dialogue, reasonable refusals, and acceptable instruction following.

Then the public MMMU (Yue et al. 2024) and DocVQA (Mathew et al. 2021) leaderboards delivered a shock. The model trailed Qwen2.5-VL-7B and InternVL3-8B by more than 12 points, even though its visual encoder was larger and the base LLM had been strengthened for Chinese.

After a week of investigation, the conclusion was clear: the architecture was not the main problem. The data recipe was. Four ignored data engineering dimensions created the gap.

* **Data quality.** Raw LAION alt-text had a median image-text relevance of only about 0.26 by CLIP cosine similarity, while GPT-4V-style recaptioned data used by InternVL was estimated around 0.61 [I]. Low-quality captions are like textbooks with wrong labels under the pictures.
* **Resolution strategy.** Fixed resizing to 336 x 336 erased dense text in financial PDFs, textbook diagrams, and documents.
* **Coverage of data types.** LLaVA-Instruct-150K lacks OCR-rich, ChartQA, and grounding instructions with bounding boxes, leaving the model weak at document understanding and precise localization.
* **Curriculum scheduling.** All data was fed uniformly across two stages. High-information-density data such as visual math reasoning was not upsampled late, and no annealed high-quality window was introduced at the end of pre-training.

This story is common. It reveals a central proposition of modern VLM engineering: the precision of the data recipe sets the ceiling of model intelligence. Architecture innovation has started to converge. The gap between leading labs and followers is now mostly in fine-grained, even harsh, multimodal data engineering.

> **Prerequisites and compliance boundary:** This chapter focuses on VLM-specific data recipes and curriculum differences. Chapters 8 through 11 already covered image-text collection, MinHash deduplication, OCR extraction, video and audio preprocessing, and cross-modal alignment. For image crawling, copyright, and provenance risk, see Chapter 4 and the compliance chapters. This chapter discusses recipes, not basic infrastructure.

![Figure 47-1: Multimodal data engineering panorama](../../images/part11/8_1_multimodal_data_panorama.png)

*Figure 47-1: Multimodal data engineering panorama.*

## 47.1 The Three-Stage VLM Data Pipeline

Qwen2.5-VL and InternVL3 reports show that modern VLM data recipes have standardized into a three-stage pipeline. Even the pre-training stage has evolved from simple concept binding into deeper visual feature structuring. Each stage has different quality, type-distribution, and scale requirements. Blindly mixing data from all stages is one of the main reasons recipes fail.

![Figure 47-2: Three-stage VLM data engineering pipeline](../../images/part11/32_1_vlm_three_stages_en.png)

*Figure 47-2: Three-stage VLM data engineering pipeline.*

**Stage 1: pre-training / feature alignment.** The goal is coarse alignment between visual concepts and text vocabulary. Scale usually ranges from hundreds of millions to billions of image-text pairs. Sources include CLIP-filtered LAION subsets (Schuhmann et al. 2022), DataComp-1B (Gadre et al. 2023), COYO-700M, and recent recaptioned datasets such as ShareGPT4V-1.2M and LLaVA-Recap-558K.

Three practices matter. First, freeze the LLM and train only the visual encoder and projector to prevent catastrophic forgetting under noisy image-text pairs. Second, apply CLIP-score filtering, often around 0.28 or higher, to remove low-relevance pairs; this can remove up to 70% of raw LAION-5B. Third, prefer recaptioning over raw alt-text. InternVL-style ablations show that replacing raw LAION alt-text with strong-VLM descriptions can improve visual alignment by several MMMU points [D] (Chen et al. 2023).

**Stage 2: multitask and high-resolution alignment.** This is the secret behind models that read invoices, reports, and complex paper figures. Scale is usually tens of millions of samples, but type diversity and format correctness matter much more. Key data types include high-resolution OCR screenshots with text coordinates, DocVQA, InfoVQA, TextVQA, grounding data with bounding boxes, interleaved web documents, ChartQA, PlotQA, and FigureQA.

The engineering challenge is resolution adaptation and token-length control. Raising OCR image resolution from 336 x 336 to 1344 x 1344 can increase vision tokens from about 256 to about 4096, forcing batch size down by roughly 16x. InternVL3 (Chen et al. 2024) uses dynamic resolution bucketing: images are clustered into predefined resolution buckets by aspect ratio and area, and each batch mixes only samples from the same bucket. This reduces padding waste and improves GPU utilization [D]. Models begin to unfreeze here, though many teams keep some LLM layers frozen to avoid language regression under extreme OCR distributions.

**Stage 3: supervised fine-tuning and alignment.** Data shrinks to the million or hundred-thousand scale. The goal is human-style interaction. Sources include visual CoT questions, visual math explanation data such as MathVista (Lu et al. 2023), GeoQA, MathV360K, GPT-4V synthetic dialogue distillation, multi-turn dialogue, and human preference feedback.

SFT has the highest quality requirement. Qwen2.5-VL reports that more than 30% of its SFT mixture is human-reviewed high-quality data [D], and low LLM-as-Judge scores are discarded. InternVL3's open SFT data, about 1.2M samples, reduces ordinary natural-scene pairs to below 10%, while OCR-rich, grounding, and chart data together exceed 60% [D]. High-quality data is scarce, so synthesis becomes central.

## 47.2 Cross-Model VLM Data Composition

Multimodal models have data barriers as high as text base models. Table 47-1 compares leading VLM recipes based on public reports and community inference as of April 2026. Tags follow the usual convention: [D] direct disclosure, [I] inference, [E] estimate.

**Table 47-1: Data composition of mainstream VLMs**

| Model family | Pre-training image-text scale | Pre-training cleaning | Interleaved document share | SFT multimodal instruction scale | Video scale | OCR / document specialization | High-resolution support |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-VL** | ~2B+ pairs [I] | In-house image filtering and rewriting | Very high, ~30% [I] | ~5M+ [E] | Very high, variable-length clips [D] | Strong, multilingual OCR | Native resolution |
| **InternVL 3** | ~1.2B pairs [D] | Full recaptioning with ShareGPT4V (Chen et al. 2023) | High, ~20% [I] | ~1.2M open [D] | Medium, keyframe extraction | Very strong, Chinese-English bilingual | Dynamic Hi-Res |
| **LLaVA-OneVision** | ~1B pairs [D] | Relies on high-quality open datasets | Medium, ~15% [E] | ~1M single / multi / video [D] | Medium, AnyRes-Video | Medium | AnyRes patching |
| **MiniCPM-V** | ~500M pairs [E] | Edge-focused extreme purification | High for layout-like data [I] | ~800K [E] | Weak, mainly image-text interaction | Strong, edge optimized | Adaptive slicing |

Several trends matter.

**Garbage in, garbage out is harsher in vision.** Qwen and InternVL no longer trust raw crawled text. They spend large amounts of compute recaptioning billions of images. LLaVA-OneVision reports that replacing raw alt-text with GPT-4V-like captions improves MMMU more than simply adding more data [D]. Data quality leverage is amplified in vision.

**Interleaved data bridges image-text reasoning.** Understanding one image and one sentence is basic; understanding logic across web pages where text and images alternate is advanced. Interleaved datasets such as MMC4 (Zhu et al. 2023) and OBELICS (Laurencon et al. 2023) strongly affect in-context learning and long-document reasoning. Qwen2.5-VL's heavier investment in interleaved web data [I] is likely one reason it performs well on complex multi-image reasoning.

**OCR and grounding have threshold effects.** Not all OCR data improves document understanding linearly. If OCR-rich data is too small a share of SFT, fine-grained reading drops sharply; beyond a threshold, gains flatten [E]. InternVL3's roughly 30% OCR-rich share [D] sits above that engineering safety threshold.

**Video data is now mandatory.** In the LLaVA-1.5 era, video data was optional. Qwen2.5-VL and InternVL3 make long-video understanding native, and video data scale now affects overall ranking. Qwen2.5-VL's Video-MME and MVBench strength comes substantially from large-scale variable-length video clips in pre-training [D], not only architecture.

**MiniCPM-V shows the edge-side refinement philosophy.** Under limited scale, it uses extreme quality control instead of scale expansion. Samples go through multi-model voting and human review [I]. With far less data, it performs strongly in edge deployment scenarios such as document screenshots and multilingual OCR. For vertical use cases, refined data can beat scale.

## 47.3 Native Resolution vs Dynamic Hi-Res

High-resolution images create token explosion. If every image is resized to 224 x 224, the model becomes near-sighted and cannot read dense invoices or formulas. Two preprocessing philosophies have emerged.

![Figure 47-3: Native vs dynamic resolution handling](../../images/part11/32_2_resolution_handling_en.png)

*Figure 47-3: Native resolution and dynamic resolution data pipelines.*

**Dynamic Hi-Res patching / AnyRes.** InternVL and LLaVA keep the visual encoder input fixed, such as 448 x 448. A 1000 x 2000 image is cropped into multiple patches without destroying aspect ratio, plus a low-resolution global thumbnail. The LLM sees a sequence such as `[global thumbnail] [patch 1] [patch 2] ... [patch N]`.

This is practical. It reuses strong open visual encoders such as CLIP-ViT without operator changes, is compatible with existing training stacks, and makes preprocessing simple. The cost is semantic discontinuity at patch boundaries. Large tables, continuous formulas, and horizontal PDFs can hallucinate across patches. InternVL3 reports higher error rates on cross-page table cases under dynamic patching [D].

**Native resolution / M-RoPE.** Qwen2-VL and Qwen2.5-VL allow images to enter at original resolution and aspect ratio. M-RoPE (Wang et al. 2024) extends position encoding into 2D image coordinates and 3D video time. The data loader feeds actual height and width token counts into attention computation without rigid padding.

This preserves global and local information and removes patch-boundary discontinuity, but it is much harder. Training data must be bucketed and packed by token count rather than image count to avoid extreme length variance and OOM. Qwen2.5-VL uses token-aware packing that keeps total vision tokens per batch in a target range, trading some throughput for stability [I].

**Table 47-2: Native resolution vs dynamic patching**

| Route | Representative models | Image preprocessing | Visual encoder changes | LLM token sequence | Tradeoff |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Native resolution** | Qwen2.5-VL | Preserve original image, dynamically expand by patch size | Remove fixed position embedding | 2D absolute coordinate mapping with M-RoPE | Highest accuracy, no boundary break / very hard engineering, memory fragmentation |
| **Dynamic patching** | InternVL3 / LLaVA | Preserve aspect ratio, dynamically crop | No modification needed | `<global> <patch1> <patch2>` linear sequence | Simple and compatible / patch seams and redundant computation |

For small and medium teams, dynamic patching is the better starting point. Overlap crops, such as 30% overlap, can reduce seam errors. If the goal is state-of-the-art DocVQA or OCRBench performance and the team has engineering capacity, native resolution is worth the cost.

## 47.4 Multimodal Instruction Synthesis

At SFT time, high-quality instruction data becomes the final ceiling. Human grounding labels and complex visual logic questions are expensive, so leading teams build synthetic data factories.

![Figure 47-4: Multimodal instruction synthesis pipeline](../../images/part11/32_3_instruction_synthesis_en.png)

*Figure 47-4: Multimodal instruction synthesis pipeline.*

Modern multimodal synthesis is more than asking GPT-4V to look at an image and write a question. It usually combines:

1. **Perception models.** Grounding DINO (Liu et al. 2023) extracts bounding boxes, OCR engines extract dense text, and depth models may extract 3D structure.
2. **Textual representation.** Perceived visual information is forced into structured text such as JSON or Markdown.
3. **LLM reasoning and recomposition.** Once images have been converted into precise text, a strong text LLM can generate complex reasoning questions such as computing tax rate from invoice fields and item names.
4. **Quality filtering and deduplication.** Self-consistency, LLM-as-Judge, and semantic deduplication remove hallucinated, illogical, or repetitive samples.

**Table 47-3: Multimodal instruction synthesis methods**

| Synthesis route | Core dependency | Typical use | Cost estimate | Noise and hallucination risk |
| :--- | :--- | :--- | :--- | :--- |
| **GPT-4V distillation** | Closed VLM API | Complex visual reasoning, long-document summaries | Very high, API quota dependent | Relatively low, but inherits teacher bias |
| **Self-distillation pipeline** | In-house perception model + strong open LLM | Fine-grained grounding, dense OCR | Medium, compute amortized | Higher, depends on OCR recall, can create phantom boxes |
| **Rule-template generation** | Structured database such as graph or table | Simple attribute QA, chart value lookup | Very low | Low, but instruction diversity is weak |

A recaptioning skeleton can look like:

```python
def recaption_batch(images, model, processor):
    results = []
    for image in images:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": RECAPTION_PROMPT}
            ]
        }]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        output_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        results.append(caption)
    return results
```

Several details matter. `temperature=0.7` helps avoid caption mode collapse; below 0.5, style becomes homogeneous. OCR-rich images should explicitly request all visible text in the prompt. Batches should mix natural images, documents, and charts to avoid distribution overfitting. After recaptioning, recompute image-caption relevance and discard low-scoring samples rather than recycling them.

## 47.5 Specialized Processing for Long Video, Documents, and OCR

As VLMs move into real-world use, static natural photographs become a smaller share of training data. Information-dense artificial images dominate. Three variants require special processing.

**Document-rich and OCR data.** To perform well on DocVQA and MMMU, teams collect financial reports with dense tables, medical-record screenshots, and academic posters. Processing goes beyond feeding images directly. High-resolution images first pass through specialized OCR or PDF parsers. Extracted text can be injected into the instruction context as a gold reference. Pipelines also use strong augmentations: color inversion, grayscale conversion, noise injection, masking, stains, or partial occlusion. This forces the visual encoder to actually read instead of letting the LLM guess from common sense. If document data falls below a threshold, fine-grained text reading collapses.

**Spatiotemporal redundancy in long video.** A ten-minute 30 fps surveillance video contains 18,000 frames. Converting all frames to tokens breaks current context windows. Early systems sampled a fixed 8 or 16 frames, which works for short videos but fails on long lectures or surveillance. Modern systems use content-aware dynamic frame rates. Static segments may keep one sparse key frame; fast motion may sample several frames per second and attach timestamps such as `<Time-Step=2.1s>`. This preprocessing makes long-video reasoning trainable under limited compute.

**Interleaved documents and web pages.** Complex knowledge often appears as text, diagram, more text, table, and captions. Data pipelines must serialize the DOM tree carefully and preserve physical order between text and images. Images should be treated as large tokens embedded in text. If cleaning removes ads but shifts a figure away from its paragraph, the model learns destructive misalignment. Labs invest heavily in interleaved cleaners for this reason.

## 47.6 Core Case Studies

### Case A: Reproducing InternVL3's Open SFT Data

InternVL is a major open reference because it releases model weights and a million-scale high-quality SFT dataset, InternVL-Chat-V1-5-SFT-1.2M. Its strength is extreme categorization and refinement.

The data is split into ChartQA, DocVQA, MathV360K, LLaVA-Instruct, ShareGPT4V, and other subsets. During packing, each batch maintains balanced type ratios so OCR data is not over-represented simply because it has longer token length.

One reproduction route is to download the CC-BY-4.0 InternVL-Chat-V1-5-SFT-1.2M dataset and use InternVL2-Training. On 8 A100 GPUs, the SFT stage for an InternVL3-8B-style run can be reproduced in roughly 36 hours under published-like settings: global batch size 512, learning rate 2e-5, and dynamic 448 x N resolution with at most 12 patches per image. The lesson is that 1.2M refined samples can produce broad visual dialogue ability when type coverage and information density are well balanced.

### Case B: Purifying LAION into LLaVA-Recap-558K

Raw LAION contains many irrelevant alt-texts, such as an image of a cat with text saying "click to buy cat food." Feeding this directly teaches nonsense. LLaVA-Recap-558K demonstrates a practical purification path.

1. Filter LAION-CC-SBU with CLIP score around 0.28, reducing roughly 3M samples to about 700K relevant pairs.
2. Use an aesthetic scorer such as LAION-Aesthetics-V2 to remove low-quality images, retaining about 558K.
3. Use LLaVA-1.5-13B with a system prompt requesting 150-250 word detailed descriptions to recaption all images.
4. Recompute CLIP score for new captions and discard low-relevance samples, roughly another 5%.

This costs about 558K times the 13B inference cost, on the order of 80 A100 GPU hours [E]. The resulting dataset is a strong VLM cold-start asset for small teams, improving median CLIP score over raw alt-text.

### Case C: Inferring Qwen2.5-VL's Long-Video Recipe

Qwen2.5-VL training data is not open, but reports and community experiments allow reasonable engineering inference. The report discloses variable timestamp video format, content-adaptive frame rates, and support for long clips through M-RoPE's time dimension [D].

A plausible pipeline is:

* collect videos from Creative Commons YouTube, Pexels, and WebVid-style sources, categorized as tutorials, documentaries, science, and events;
* split shots with PySceneDetect and filter extreme durations;
* compute frame-to-frame SSIM; sample static segments sparsely and dynamic segments more densely;
* feed keyframe sequences to a VLM to generate temporal captions with timestamps;
* validate temporal alignment by checking whether timestamp words match frame changes.

Community reproductions suggest that time-aligned captions improve long-video benchmark performance more than random frame sampling [E]. The bottleneck for long-video understanding is caption temporal quality, not raw frame count.

## 47.7 Pitfalls, Costs, and Boundaries

**Caption mode collapse and hallucination amplification.** A single teacher generating all captions can make the student always start with the same phrasing and inherit teacher biases. Use many prompt templates, higher decoding diversity such as temperature above 0.7 or top-p sampling, and reward-model filtering.

**High-resolution memory sink and padding penalty.** Native-resolution batches with irregular aspect ratios produce huge length variance. Padding wastes memory and backward compute. Use variable-length attention implementations such as Varlen FlashAttention (Dao et al. 2022), or pack by token count rather than sample count.

**Synthetic API cost.** Top closed VLM APIs can cost about $0.005 to $0.01 per medium-resolution image. A 1M high-quality instruction set may cost $5K to $10K in direct API fees. Industrial pipelines usually use cheaper open VLMs for first-pass generation and strong models for verification.

**Image compliance and copyright risk.** Images expose copyright and privacy issues more visibly than text. Crawling face images or watermarked commercial images can create serious legal risk. Production image crawling should follow provenance tagging, face blurring, and compliance audit procedures from the data compliance chapters.

**Type imbalance and capability seesaw.** If OCR-rich data exceeds the right range, natural-scene understanding can regress. A capability-aware sampler can run lightweight probes after each epoch and increase sampling for weaker categories in the next epoch.

**Hidden multilingual OCR bias.** Multilingual OCR datasets can be highly uneven in font coverage. Chinese may cover thousands of fonts while Arabic data covers far fewer. Real-world recognition can collapse on unseen scripts. Audit font coverage by language and use font synthesis to expand low-coverage languages.

**When not to chase the top recipe.** Vertical small-data scenarios, such as factory inspection, may perform better with a simple LLaVA-style recipe plus 300-500 high-quality domain labels and LoRA. Edge deployment should prioritize MiniCPM-V-style refined data. For product proof of concept, LLaVA-Recap-558K plus LLaVA-Instruct-150K can build a fast baseline before custom data engineering.

## 47.8 Summary

The rise of VLMs looks like a marriage between vision and language architectures, but in practice it is a data governance battle. This chapter started from a recipe failure and decomposed four VLM data dimensions.

The three-stage pipeline has very different scale, quality, and type requirements across pre-training, multitask alignment, and SFT. Cross-model comparisons show actionable rules: recaptioning beats raw alt-text, interleaved data controls reasoning depth, and edge-side refinement can beat scale for vertical scenarios. Native resolution and dynamic hi-res have no absolute winner; the right choice depends on resources. Synthetic data factories, OCR injection, dynamic video frame sampling, and interleaved cleaning are high-value modules that teams can reuse.

The case studies give three entry points: open reproduction through InternVL3, dirty-data purification through LAION recap, and recipe inference from Qwen2.5-VL long-video practice. The pitfalls remind us that the most complex recipe is not always the right recipe. Business scenario and team constraints come first.

After a VLM learns to understand the physical world and two-dimensional surfaces, it gains the foundation to intervene in the visual world. Chapter 48 turns the perspective around: when models generate pixels and videos rather than only observe them, the data recipe changes again.

> **Compliance boundary:** For image copyright and privacy, see Chapter 4 and the compliance chapters. For general multimodal preprocessing, see Chapters 8 through 11.

## References

Chen Z, Wu J, Wang W, Su W, Chen G, Xing S, Zhong M, Liu Q, Lu Y, Li B, others (2023) InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks (ShareGPT4V). arXiv preprint arXiv:2312.14238.

Chen Z, Wang W, Tian H, Ye S, Gao Z, Cui E, Tong X, Hu J, Luo J, Ma S, others (2024) InternVL3: Exploring Advanced Training and Test-Time Scaling for Vision-Language Models. arXiv preprint arXiv:2504.10479.

Dao T, Fu D Y, Ermon S, Rudra A, Re C (2022) FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. In: Advances in Neural Information Processing Systems 35:16344-16359.

Gadre S Y, Ilharco G, Fang A, Hayase J, Ilharco G, Marten T, Wortsman M, Goyal S, Guha E, Jain H, others (2023) DataComp: In Search of the Next Generation of Multimodal Datasets. In: Advances in Neural Information Processing Systems 36.

Laurencon A, Saulnier L, Tronchon L, Bekman S, Singh A, Lozhkov A, Wang T, Karamcheti S, Rush A M, Kiela D, others (2023) OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents. arXiv preprint arXiv:2306.16527.

Li B, Zhang Y, Guo D, Zhang R, Li F, Zhang J, Zhang Y, Zhu P, Zhang Z, Yang J, others (2024) LLaVA-OneVision: Easy Visual Task Transfer. arXiv preprint arXiv:2408.03326.

Liu H, Li C, Wu Q, Lee Y J (2024b) Visual Instruction Tuning (LLaVA-1.5). In: Advances in Neural Information Processing Systems 36.

Liu S, Zeng Z, Ren T, Li F, Zhang H, Yang J, Li C, Yang J, Su H, Zhu J, others (2023) Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. arXiv preprint arXiv:2303.05499.

Lu P, Bansal H, Xia T, Liu J, Li C, Hajishirzi H, Cheng H, Chang K W, Galley M, Gao J (2023) MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts. arXiv preprint arXiv:2310.02255.

Mathew M, Karatzas D, Jawahar C V (2021) DocVQA: A Dataset for VQA on Document Images. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp 2200-2209.

Radford A, Kim J W, Hallacy C, Ramesh A, Goh G, Agarwal S, Sastry G, Askell A, Mishkin P, Clark J, others (2021) Learning Transferable Visual Models from Natural Language Supervision (CLIP). In: Proceedings of the 38th International Conference on Machine Learning, pp 8748-8763.

Schuhmann C, Beaumont R, Vencu R, Gordon C, Wightman R, Cherti M, Coombes T, Katta A, Mullis C, Wortsman M, others (2022) LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models. In: Advances in Neural Information Processing Systems 35:25278-25294.

Wang P, Bai S, Tan S, Wang S, Fan Z, Bai J, Chen K, Liu X, Wang J, Ge W, others (2024) Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv preprint arXiv:2409.12191.

Yue X, Ni Y, Zhang K, Zheng T, Liu R, Zhang S, Stevens J, Jiang C, Zheng N, Sun T, others (2024) MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 9556-9567.

Zhu D, Chen J, Shen X, Li X, Elhoseiny M (2023) MiniGPT-4 / MMC4: An Open Large-Scale Dataset of Interleaved Image-Text Data. arXiv preprint arXiv:2306.04764.
