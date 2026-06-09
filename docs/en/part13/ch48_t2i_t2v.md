# Chapter 48: Data Engineering for Multimodal Generative Models: T2I and T2V Data Pipelines

## Abstract

The prompt-following capability of Text-to-Image (T2I) and Text-to-Video (T2V) generative models is largely determined by the supervisory language present in training data, rather than solely by the generative backbone architecture. This chapter adopts a "recipe" perspective to examine how generative models transform raw image and video assets into training-ready, controllable, and auditable data. The discussion begins by distilling three principal threads in generative data engineering: dense captioning as a progression from labels to executable descriptions, aesthetic and quality scoring as a router for training sets, and hard-constraint governance over copyright, safety, and privacy. The chapter then draws horizontal comparisons across image recaptioning approaches, video engineering approaches, and industry model approaches—using DALL·E 3, Stable Diffusion 3, FLUX, HunyuanVideo, Wan2.2, Open-Sora, and CogVideoX as representative systems—examining their respective trade-offs in data ingestion, filtering governance, and annotation strategy. The chapter further elaborates on specific stages in T2I and T2V data pipelines, including data ingestion sources such as LAION and DataComp, hierarchical filtering, optical flow motion and shot language selection, structured spatiotemporal caption augmentation, and training bucketing and routing by resolution and quality tier. The chapter emphasizes that generative model data pipelines have evolved from passive "data collection" into active "supervisory signal production" that actively shapes model capabilities.

## Keywords

Multimodal generative model data engineering; data recipes; open-source large models; training data; staged scheduling

## Learning Objectives

- Be able to distill the three principal threads of generative model data engineering: dense recaptioning, aesthetic and quality scoring for routing, and hard-constraint governance over copyright, safety, and privacy.
- Be able to compare the trade-offs in data ingestion, filtering governance, and annotation strategy across DALL·E 3, Stable Diffusion 3, FLUX, HunyuanVideo, Wan2.2, Open-Sora, and CogVideoX.
- Be able to design a T2I data pipeline covering data ingestion sources such as LAION and DataComp, hierarchical filtering, and dense recaption augmentation.
- Be able to design a T2V data pipeline covering optical flow motion and shot language selection, structured spatiotemporal caption augmentation, and training bucketing and routing by resolution and quality tier.
- Be able to explain why generative model data pipelines have evolved from passive data collection into active supervisory signal production.

## Preface: Same Architecture, Different Degrees of "Instruction Following"

In image generation and video generation, a frequently underestimated question is: from what kind of prompts is the model actually learning?

Early image-text data largely originated from web page alt-text, titles, surrounding text, and brief captions left by users at upload time. Such text was marginally adequate for multimodal understanding models. If an image contained a cat and the caption read "cat on sofa," the model could learn a coarse-grained correspondence between "cat" and "sofa." But once the objective shifts to generation, the problem immediately becomes more complex. Users do not simply request "generate a cat"; they demand "an orange tabby sprawled on a blue sofa near a window at dusk, side backlighting, shallow depth of field, a potted green plant to the right of the frame." If the training data never consistently describes color, position, lighting, composition, material, style, and subject relationships, the model will naturally struggle to follow these words precisely during generation.

The DALL·E 3 paper (Betker et al. 2023) brought this issue to the forefront. Its focus was not on building a more complex generative backbone, but on re-annotating training images with more descriptive captions. The core conclusion was direct: the more closely the training text resembles actual generation intent, the more readily a model learns prompt following. Sora's publicly released materials subsequently extended this approach, migrating the image-side recaptioning philosophy to the video domain: first train or employ a strong captioner to rewrite training videos into detailed descriptions; then at inference time, expand the user's short prompt into a longer description better suited for video model execution.

Stable Diffusion 3 (Esser et al. 2024) provides the clearest engineering evidence. The SD3 paper discloses that the team used CogVLM (Wang et al. 2023) to generate synthetic captions and adopted a mixed training strategy of 50% original captions and 50% synthetic captions. By changing only the caption distribution, the GenEval (Ghosh et al. 2023) overall score rose from 43.27 to 49.78; metrics closely related to prompt following—such as Color Attribution and Position—showed even larger gains, with some relative improvements exceeding 30%. This demonstrates that prompt-following capability is not entirely determined by model architecture: the supervisory language in training data plays an equally decisive role.

Video generation is even more dependent on this kind of "executable language." A video clip contains not only subjects and backgrounds but also actions, camera motion, shot size, pace, rhythm, lighting changes, and spatiotemporal relationships. If a caption reads only "a man walking on the street," the model learns nothing more than a vague theme. If the caption instead articulates "fixed camera, medium shot, man walking from screen left to screen right, background vehicles moving slowly, warm late-afternoon light, slight handheld shake," the model has an opportunity to learn more controllable motion and cinematic language.

This chapter addresses precisely this data chain: how T2I and T2V generative models transform raw images and video footage into training-ready, controllable, and auditable data. A boundary must be drawn with Chapter 10. Chapter 10 has already addressed general audio-visual sample engineering, including scene segmentation, shot boundary detection, keyframe selection, ASR, subtitle alignment, unified timelines, and clip-level/segment-level/frame-level sample organization. This chapter assumes that foundational processing has already been completed, and focuses on data stages unique to generative models: caption rewriting, aesthetic and quality scoring, safety and copyright filtering, video motion filtering, spatiotemporal caption augmentation, training bucketing, and data routing.

This chapter also does not expand on video generation model architectures, nor does it discuss the details of DiT, U-Net, Transformer, Flow Matching, or VAE; multimodal understanding tasks are addressed in Chapter 47 and the foundational chapters in Part III. The concern here is a different and more subtle form of engineering capability: how to transform raw media into the kind of generative supervision that a model can truly learn from.

---

## 48.1 Three Defining Characteristics of Generative Model Data

Generative model data still takes the form of "image/video + text" on the surface, but it differs markedly from data for understanding models. Understanding models must answer "what is in the image"; generative models must learn "how to transform a sentence into an image or video." The former emphasizes recognition; the latter emphasizes reconstruction, composition, and control. Generative data engineering therefore has at least three principal threads: caption granularity, aesthetic/quality scoring, and copyright and safety governance.

### 48.1.1 Caption Granularity: From Labels to Executable Descriptions

Text in traditional image-text data tends to be very brief. Web page alt-text, titles, product names, and social media descriptions offer only coarse-grained semantics. For contrastive learning models such as CLIP, short text is sufficient to support image-text matching; for generative models, short text yields impoverished supervisory signals.

Mainstream T2I/T2V pipelines increasingly emphasize dense captions. On the image side, captions typically need to cover subjects, attributes, quantities, spatial relationships, style, lighting, background, composition, and material. On the video side, action processes, motion direction, speed, shot type, shot size, camera motion, temporal changes, and atmosphere must additionally be incorporated. In many video generation projects, captions have expanded from a dozen words to dozens or even hundreds of words, and some systems adopt structured fields that are assembled into texts of varying length depending on the training stage.

HunyuanVideo (Kong et al. 2024) is one of the most detailed publicly disclosed examples. Rather than generating a single natural-language sentence per video clip, it organizes descriptions into structured fields including a short description, a dense description, background, style, shot type, lighting, and atmosphere, with additional annotations for camera motion categories. The benefit is clear: what the model encounters during training is not merely "what is in the frame," but also "how the shot is composed," "what the lighting looks like," and "how the action unfolds."

However, captions cannot grow without bound. Excessively long synthetic captions are prone to hallucination, inserting details into the description that do not exist in the image. Simultaneously, training text that is overly uniform and overly dense may cause model outputs to become rigid, sacrificing natural aesthetic quality and diversity. SD3's retention of 50% original captions is instructive: although original text is noisier, it covers a broader range of concepts; synthetic captions carry higher information density but are bounded by the captioner's knowledge and biases. Mixing the two is generally more stable than a wholesale replacement.

### 48.1.2 Aesthetic and Quality Scoring: The Router of the Training Set

Generative models are more dependent on visual appeal than understanding models. An understanding model can learn categories from low-resolution, color-shifted, or poorly composed images; a generative model that trains heavily on such samples will inherit these problems in its outputs.

Scoring systems such as LAION-Aesthetic (Schuhmann et al. 2022), PickScore (Kirstain et al. 2023), and HPSv2 (Wu et al. 2023) serve as data routers. LAION-Aesthetic functions more like a low-cost, large-scale sieve, rapidly carving out a higher-aesthetic subset from a massive image-text pool. PickScore and HPSv2 more closely approximate human preference evaluation, making them suitable for finer-grained ranking of generated outputs or training candidates. In practice, engineers rarely rely on a single score; instead, multiple signals are chained together: low-cost rule-based filtering first eliminates obviously low-quality samples, followed by stratification using aesthetic scores, preference scores, sharpness scores, OCR area, and subject completeness.

Video-side quality scoring is more complex. Beyond static aesthetics, one must assess blur, shake, compression artifacts, motion amplitude, subtitle occlusion, watermarks, logos, borders, and shot stability. The Open-Sora 2.0 (Open-Sora Team 2024) public pipeline includes aesthetic, motion, blur, OCR, and camera jitter filtering signals. HunyuanVideo likewise employs modules for technical quality, aesthetic quality, optical flow motion, OCR text regions, and watermark/logo/border detection, placing video samples into different training tiers.

The key insight here is that "higher scores are not always better." Generative models need coverage of the world's diversity. Aggressively pursuing high aesthetics will eliminate ordinary scenes, complex actions, long-tail objects, and multi-subject interaction samples. A more robust approach is tiered retention: delete low-quality samples, block high-risk samples, place boundary samples in review or low-weight training, and preserve rare but valuable samples separately.

### 48.1.3 Copyright, Safety, and Privacy: Hard Constraints for Generative Models

Generative models reproduce patterns from their training distribution, so data risks are amplified. Watermarks, logos, personal information, sensitive text, and copyrighted materials that appear frequently during training may be reproduced in similar forms during generation. These risks cannot be fully resolved through post-deployment moderation alone; intervention must begin at the training data stage.

Common governance signals include NSFW detection, watermark detection, OCR text ratio, logo detection, personal information identification, copyright source tagging, duplicate image clustering, user opt-out lists, and manual review queues. For video, additional handling is required for subtitle screen coverage, intro/outro sequences, channel identifiers, border templates, and derivative work watermarks. The public HunyuanVideo pipeline explicitly includes OCR filtering and watermark/logo/border detection; LAION-5B also carries NSFW, watermark, and similar metadata; OpenAI's system card emphasizes explicit content filtering, personal information reduction, and training exclusion mechanisms.

In engineering implementation, risk governance can be divided into three categories: safety issues trigger immediate blocking; copyright issues enter review or exclusion lists; aesthetic and quality issues are used for routing and weighting. Collapsing all three into a single aggregate score creates problems: samples with high aesthetics but high risk may be erroneously admitted, while low-aesthetic samples containing rare actions may be erroneously discarded. Audit metadata in generative data engineering must be preserved so that rollbacks, resampling, and accountability tracing remain possible.

---

## 48.2 Comparative Overview of Mainstream T2I/T2V Model Data Pipelines

When discussing the data pipelines of T2I and T2V models, the most common pitfall is simply listing each project's data sources, captioning tools, and filtering steps. While such a list appears information-rich, it yields little more than a horizontal inventory, leaving readers unable to discern the underlying engineering philosophy. Generative model data pipelines are better understood through three questions: first, where does raw data come from; second, what filtering and rewriting does the data undergo; and third, what training objective does all this processing ultimately serve. The differences among models typically lie not in any single module, but in the emphasis of the entire pipeline.

The T2I data chain is relatively mature, with core tensions centered on image-text alignment, caption granularity, and image quality stratification. Early image generation models relied heavily on web image-text pairs, with text typically drawn from alt-text, page titles, surrounding descriptions, or short labels left by uploaders. Such text has broad coverage but sparse descriptions; most samples state only "what is in the image" without specifying color, composition, material, lighting, spatial relationships, or style. As prompt following emerged as a critical capability, T2I data engineering placed caption rewriting at a more central position. The DALL·E 3 paper highlights the impact of highly descriptive captions on generation capability; Stable Diffusion 3 further demonstrates through ablation experiments that using CogVLM-synthesized captions mixed with original captions significantly improves model adherence to prompt instructions regarding color, position, and quantity. This illustrates that the T2I data pipeline has transitioned from "collecting image-text pairs" toward "producing supervisory text better suited for learning to generate."

DALL·E 3 and Stable Diffusion 3 represent the two most paradigmatic approaches on the image side. DALL·E 3 focuses on training a more powerful image description system that rewrites training images into more detailed text closely resembling user prompts, while at inference time employing a language model to expand user input. The underlying logic is clear: if the user writes briefly, the system first enriches the semantics; when training captions are more fine-grained, the model more readily learns fine-grained constraints. The challenge is that DALL·E 3's full data pool, filtering thresholds, and annotation strategies have not been made public, making it possible to learn the direction but difficult to fully reproduce. Stable Diffusion 3 is more valuable from an engineering standpoint because it discloses clearer experimental conclusions. Rather than replacing all original captions, SD3 mixes original and synthetic captions so that the model simultaneously retains the long-tail concepts present in web text while benefiting from more granular visual descriptions. This approach has greater practical value for the open-source community and engineering teams because it does not require constructing perfect captions all at once, but instead allows incremental improvement of existing image-text pools.

FLUX represents a different situation. Its model performance, community influence, and production capability are strong, but engineering details of its data pipeline are sparsely disclosed. Public materials focus primarily on model specifications, open-weight versions, generation quality, risk mitigation, and application capabilities, without systematically covering the raw training data pool, recaptioning methods, aesthetic filtering strategy, or copyright filtering details. When discussing FLUX in a book context, it is appropriate to position it as a model where "effects are strong but data chain disclosure is limited." It demonstrates that industry models have achieved high data quality, but community speculation should not be presented as established fact. For readers, FLUX's lessons are primarily outcome-oriented: high-quality T2I models invariably require large-scale filtering, aesthetic control, safety governance, and fine-grained textual supervision; the public record is insufficient to reconstruct the complete pipeline.

T2V model data pipelines are more complex than their T2I counterparts. Image data requires only a judgment on whether a single image is usable; video data additionally requires assessing whether a clip is stable, whether it contains meaningful motion, whether the action is continuous, whether the shot is sharp, and whether subtitles and watermarks obscure the frame. A clip with decent image quality but almost no motion provides limited value for learning video generation; conversely, a clip with strong motion but severe compression, erratic camera shake, and heavy subtitle coverage also degrades in training value. T2V data engineering therefore introduces a set of video-specific signals including motion score, optical flow intensity, camera jitter, blur, OCR area, cinematic shot language, action temporality, and camera motion type. Chapter 10 has handled general video segmentation, keyframes, ASR, and timelines; this section focuses on how the resulting clips, before entering generative model training, are filtered, described, stratified, and audited.

HunyuanVideo is currently the most appropriate core case study for T2V data pipelines in the publicly available literature. Its distinguishing feature is not simply that it used large-scale video data; more importantly, it has disclosed several key engineering stages of the video generation data chain at a level of detail amenable to analysis: how raw data enters the system, how clips are filtered for low-value samples, how video quality is stratified, how captions expand from a single sentence into structured descriptions, and how data ultimately serves different training stages. For readers wishing to reproduce a T2V data engineering setup, this is considerably more valuable than knowing the model parameter count alone.

Open-Sora and Open-Sora-Plan illustrate the community open-source approach. Open-Sora 2.0 emphasizes hierarchical filtering over large-scale video candidate pools. Basic conditions such as bpp, fps, duration, and aspect ratio are processed first, followed by filtering with signals including aesthetic, motion, blur, OCR, and jitter. On the caption side, low-resolution stages use lower-cost video description models, while high-resolution stages switch to stronger vision-language models and append motion scores to captions. This approach carries strong engineering sensibility: early training demands scale and coverage, while high-quality stages require cleaner and more accurate supervisory text. Open-Sora-Plan is better suited as a reference for resource-constrained teams. It begins with low copyright-risk video sources such as Mixkit, Pexels, and Pixabay, first building a smaller but more easily compliant data pool, then using models such as ShareGPT4V-Captioner-7B and LLaVA-1.6-34B to generate dense captions. Its significance lies in lowering the startup threshold for the T2V data chain, allowing research teams to avoid confronting complex commercial copyright data pools from the outset.

CogVideoX's public materials lean more toward components and methodology. It has not fully disclosed its training data pool, but CogVLM2-Caption, prompt rewriting tools, and video caption-related components are highly valuable to the community. What many teams actually lack is not a complete closed-source system, but caption tooling that can plug into their own pipeline. CogVideoX's contribution is the insight that video recaptioning can be modularized: the front end receives clips and keyframes produced by Chapter 10's pipeline, the middle stage calls a video captioner or multi-frame VLM, and the back end organizes outputs into short captions, long captions, action descriptions, and shot labels before feeding them into the training manifest.

Wan2.2 (Wan Team 2025) places greater emphasis on aesthetic direction for high-quality video generation. Public materials indicate that it expands image and video training data beyond Wan2.1, introducing finer-grained aesthetic data, cinematic style labels, and a prompt engineering system. The focus here is not simply on increasing data volume but on improving the visual refinement and creative qualities of the data. Video generation has progressed from "making images move" to "producing usable creative assets"; the practical viability of outputs is affected by lighting, color grading, cinematic feel, dynamic rhythm, and the stability of subjects and scenes. Wan2.2 is best positioned as a "cinematic data strategy" case: it demonstrates that T2V data engineering is moving beyond generic video sample cleaning toward stylized, high-aesthetic, and creatively organized data.

Comparing these models together reveals three relatively clear approaches. The first is the image recaptioning approach represented by DALL·E 3 and SD3, focused on making training text more granular and more closely aligned with user prompts. The second is the video engineering approach represented by HunyuanVideo and Open-Sora, focused on unifying motion, shot language, quality, OCR, watermarks, and structured captions into a single pipeline. The third is the industry model approach represented by FLUX and Wan2.2—publicly disclosed effects are strong, data chain details are limited, but one can observe trends toward high aesthetics, stringent safety constraints, and orientation toward creative use cases. For the purposes of this chapter, Table 48-1 serves primarily to consolidate these differences; what readers truly need to understand is this: generative model data pipelines have evolved from "data collection" to "supervisory signal production." Data no longer passively enters training sets; after filtering, rewriting, stratification, and routing, it actively shapes the generative capabilities of models.

**Table 48-1: Comparative Overview of Mainstream T2I/T2V Model Data Pipelines**

| Model / Project | Modality | Data Ingestion & Disclosure Level | Filtering & Governance Focus | Caption / Annotation Strategy | Implications for Data Engineering |
|---|---|---|---|---|---|
| DALL·E 3 | T2I | Data pool not fully disclosed; public materials emphasize high-quality image-text supervision | Safety filtering, personal information reduction, inference-side prompt rewriting | Highly descriptive captions; user prompts expanded at inference time | Demonstrates that highly descriptive captions are critical for prompt following |
| Stable Diffusion 3 | T2I | Large-scale image-text data; sources not enumerated individually | Paper focuses on model training and caption ablations; filtering chain less disclosed | CogVLM synthetic captions; mixed training of original and synthetic captions | Provides quantitative evidence that recaptioning improves generative prompt following |
| FLUX.1 / FLUX.1 Kontext | T2I / Image editing | Training data details sparsely disclosed | Model cards emphasize risk mitigation, usage boundaries, and open-weight versions | Foundational recaptioning details not systematically disclosed | Strong results, but data chain cannot be fully reproduced; suitable as an industry case study |
| HunyuanVideo | T2V + T2I pretraining | Internet-scale image and video data; pipeline disclosed in detail | Deduplication, concept resampling, DOVER, sharpness, optical flow, OCR, watermarks, logos, borders | Structured JSON captions including short description, dense description, background, style, shot type, lighting, atmosphere, and camera motion | Most suitable public case for detailed analysis of video generation data engineering |
| Wan2.2 | T2V / I2V / TI2V | Expanded image and video data over Wan2.1 | Emphasizes high-quality data, aesthetic data, and cinematic style labels | Introduces fine-grained aesthetics and creative prompt system | Illustrates the shift in T2V data from generic usability toward cinematic quality and creative controllability |
| Open-Sora 2.0 | T2V / I2V | Large-scale video candidate pool; community disclosure substantial | bpp, fps, duration, aspect ratio pre-filtering; multi-tier filtering with aesthetic, motion, blur, OCR, jitter | Low-resolution stages use LLaVA-Video; high-resolution stages use Qwen 2.5 Max; motion score appended to caption | Demonstrates a practical paradigm for open-source T2V hierarchical filtering and staged captioning |
| Open-Sora-Plan | T2V | CC0 video sources: Mixkit, Pexels, Pixabay | Begins with low copyright-risk assets to build a controllable small data pool | Dense captions generated by ShareGPT4V-Captioner-7B and LLaVA-1.6-34B | Suitable for resource-constrained teams building small-scale, reproducible T2V data chains |
| CogVideoX | T2V / I2V / V2V | Full training data pool not fully disclosed | Public contributions primarily at the component and toolchain level | CogVLM2-Caption for video-to-text; prompt rewriting tools provided | Caption components offer high reuse value and can be integrated into custom video pipelines |

The focus of this section is not to judge which pipeline is "best," but to understand the trade-offs each system makes in data engineering. T2I models rely more heavily on image-text relevance, aesthetic filtering, and caption rewriting; T2V models add motion quality, shot language, and spatiotemporal alignment on top of these; industry models typically retain more proprietary data details, while open-source projects decompose reproducible modules more explicitly. What truly guides engineering practice is not any project's specific configuration, but the trend these projects collectively reveal: as generative models advance toward higher controllability, higher aesthetics, and higher safety, the data pipeline increasingly becomes an integral part of model capability.

---

## 48.3 T2I Data Pipeline

The T2I data pipeline can be summarized as follows: a large-scale image-text candidate pool enters the system, undergoes basic cleaning and risk filtering, proceeds through caption rewriting and quality scoring, and is finally bucketed by resolution, aspect ratio, quality tier, and training stage.

### 48.3.1 Data Ingestion: LAION, DataComp, and Custom Collection

The significance of LAION-5B (Schuhmann et al. 2022) lies in its scale and openness. It provides billions of image-text pairs along with CLIP filtering scores, NSFW labels, watermark metadata, and other annotations. Many open-source image generation efforts begin with a LAION subset or a similar Common Crawl image-text pool.

DataComp (Gadre et al. 2023) serves a somewhat different purpose. It transforms data design itself into a benchmark: the model is fixed, and participants or researchers vary data sources and filtering strategies, then compare the resulting CLIP performance. This has direct implications for generative models. A large pool does not automatically yield good data; what truly impacts training outcomes is the filtering strategy, resampling strategy, and sample weighting.

Custom collection is typically used for vertical domains such as medical imaging, industrial imagery, product photographs, interior design, game assets, and remote sensing imagery. The advantage of custom-crawled data is semantic focus and stylistic consistency; the risks include more difficult handling of copyright, privacy, and duplicate samples. Accordingly, custom collection must always preserve source, license, download timestamp, hash, OCR detection results, and copyright risk scores as metadata.

### 48.3.2 Filtering: From "Whether Usable" to "How to Use"

The first layer of T2I filtering typically addresses hard problems: images that cannot be downloaded, corrupted formats, dimensions that are too small, duplicate images, language mismatches, excessively low image-text relevance, explicit content, obvious watermarks, and personal information exposure. Only the second layer addresses the concerns that are more specific to generative models: aesthetics, sharpness, composition, text region ratio, subject completeness, residual logos, style distribution, and concept coverage.

Aesthetic scores should not be used as binary pass/fail thresholds. A training set that retains only high-scoring images may become excessively "posterized": beautiful colors, stable compositions, centered subjects—but with ordinary everyday scenes, complex spatial relationships, crowded multi-person settings, and long-tail objects increasingly filtered out. A more robust strategy is tiered retention. Low-quality samples are deleted; high-quality samples enter the main training set; boundary samples are retained at low weight; rare-concept samples are separately labeled to prevent their elimination by aesthetic filtering.

### 48.3.3 Caption Rewriting: The Key Gain in the T2I Data Chain

The experience of both DALL·E 3 and SD3 points to the same conclusion: original captions are insufficient to support high-level prompt following. Generative models need to be exposed to training language that more closely resembles user generation intent.

T2I recaptioning commonly takes one of three forms:

1. Using a strong image captioner to generate detailed natural-language descriptions;
2. Expanding on the original caption while preserving its original concepts and supplementing attributes, relationships, and style;
3. Generating structured fields that are assembled into texts of varying length depending on the training stage.

SD3's 50/50 mixing strategy is particularly worth emulating. Replacing all original captions wholesale with VLM-generated captions may cause concept forgetting and stylistic uniformity; retaining some original captions preserves the rough diversity of the data distribution. In engineering practice, this can be extended to random-length mixing: the same image is sometimes paired with a short caption, sometimes with a dense caption, and sometimes with a long caption assembled from structured fields. This prevents the model from adapting exclusively to one text length.

**Table 48-2: Comparison of Caption Rewriting Approaches (DALL·E 3 / SD3 / CogVLM2 Distillation)**

| Approach | Training-side caption generator | Original caption retention strategy | Output form | Inference-side prompt rewriting | Advantages | Risks and costs |
|---|---|---|---|---|---|---|
| DALL·E 3 | Specially trained highly descriptive captioner | Not fully disclosed in public materials | High-density natural-language caption | Yes, user prompt expanded with GPT-4 | Marked improvement in prompt following; short prompts converted into executable descriptions | System is closed; captioner training details and data thresholds cannot be reproduced |
| Stable Diffusion 3 | CogVLM-generated synthetic captions | Explicitly uses 50% original + 50% synthetic captions | Mixed training with original and synthetic captions | Paper does not foreground inference-side rewriting | Clear ablation gains disclosed; high reproduction value | Synthetic captions inherit VLM biases and may introduce hallucinations |
| CogVLM2 distillation-based rewriting | CogVLM2 / CogVLM2-Caption generates dense captions, then distilled into controlled text via templates or LLM | Some original captions optionally retained; random-length mixing feasible | Dense captions, structured captions, random-length captions | Optional; often paired with long-prompt expansion | Open-source and reusable; suitable for vertical domains and community pipelines | Higher cost; excessively long captions may cause stylistic rigidity or semantic hallucinations |

### 48.3.4 Bucketing: Aligning Training Scheduling with Data Distribution

Image generation models typically require bucketing by resolution, aspect ratio, and quality tier. Bucketing is not merely an engineering acceleration technique; it affects the compositional distributions the model learns. Common buckets include aspect ratios such as 1:1, 3:4, 4:3, 9:16, and 16:9, and resolution tiers such as 512, 768, and 1024.

More sophisticated systems additionally route samples by data source, aesthetic score, caption type, safety tier, presence of text, presence of human subjects, and presence of complex multi-subject scenes. This allows training-time control of batch composition, preventing any category of samples from overwhelming others. For example, if text-rendering capability is weak, increase the sampling weight for images containing text at low risk; if spatial relationship modeling is poor, increase the weight for samples with prominent positional relationships; if hand detail on human subjects is poor, create a dedicated bucket for relevant samples.


![Figure 48-1: T2I Data Pipeline](../../images/part11/ch33_1.png)

*Figure 48-1: T2I Data Pipeline*

---

## 48.4 T2V Data Pipeline

Compared with T2I, T2V data engineering more closely resembles a true production line. Image data requires resolving two core questions: "can this image be learned from?" and "does this text correspond to this image?" Video data must additionally address shot boundaries, motion intensity, frame sampling methods, temporal descriptions, cinematic shot language, and training stratification. For this reason, T2V pipelines cannot simply replicate the T2I approach. The key challenge is not merely converting videos into captions, but transforming continuous footage into "spatiotemporal supervision that a model can learn from."

This section follows a clear main thread: **video source → shot segmentation (PySceneDetect) → motion detection → multi-frame sampled caption → spatiotemporal alignment → shot language annotation**. The underlying principles of shot segmentation, keyframe extraction, and unified timelines have already been covered in Chapter 10; this section does not repeat those general processing details, but instead emphasizes how generative models take over from these intermediate results and organize them into trainable T2V data. PySceneDetect's interface, detectors, and parameter specifications should be taken from the official documentation (PySceneDetect Contributors 2026).

From an engineering perspective, the pipeline's objectives are clear: first, eliminate clips with no learning value; second, convert the retained video clips into textual supervision that is sufficiently detailed and trustworthy; third, explicitly encode motion, shot language, and temporal structure—information unique to video—into the data. Only in this way will what a video generation model learns encompass not only "what is in the frame," but also "how the frame moves," "how the shot is composed," and "how the action unfolds."

### 48.4.1 Video Source: From Raw Video to Segmentable Assets

The starting point of a T2V data pipeline is typically not a collection of already-cleaned clips, but a raw video pool of diverse origins and highly variable quality. Public video platforms, CC0 stock footage sites, custom-crawled libraries, licensed film and television assets, and instructional or demonstration videos may all be candidate sources. The data value of different sources varies considerably: stock footage videos typically have stable image quality, clear shot language, and low copyright risk, making them suitable for foundational training sets; open-web videos offer broader coverage with richer actions and scenes, but suffer more severely from duplication, subtitles, watermarks, reuploaded derivative content, and copyright concerns; custom data is most targeted in domain but often limited in scale.

The first step in a T2V pipeline is therefore not immediate captioning, but organizing raw videos into intermediate assets that are segmentable, traceable, and auditable. At minimum, several metadata fields must be preserved: video source, duration, frame rate, resolution, aspect ratio, compression format, audio track status, basic OCR information, and source identifier. This ensures that any subsequent quality or copyright issue can be traced back to the original file, rather than being limited to anonymous clip-level information.

From a structural standpoint, Chapter 10 has already addressed the foundational methods for general video segmentation, so questions such as "why segment shots" and "how to define shot boundaries" are not revisited here. For this chapter, what matters more is the following: **T2V training does not consume raw long videos directly; it consumes shot-level clips.** This determines the core step that follows—shot segmentation.

### 48.4.2 Shot Segmentation: Using PySceneDetect to Decompose Long Videos into Training Units

Video generation training typically uses **single-shot clips** as the fundamental unit, rather than full-length videos. The reason is straightforward: a training sample that spans multiple shots will exhibit abrupt changes in visual style, viewpoint, shot size, and action logic, making accurate captioning very difficult and causing the model to learn confused spatiotemporal correspondences. Generative models are especially vulnerable to the scenario of "one sample containing multiple cinematic languages," as this weakens the stable mapping between prompt and output.

Accordingly, in T2V data pipelines, long videos are first subjected to shot segmentation upon entering the system. A common engineering choice is **PySceneDetect**. Although not designed specifically for generative models, it is well suited to serving as the front-end segmentation tool: it detects shot boundaries via signals such as content difference, luminance change, and edge variation, decomposing a video into relatively stable shot-level clips (PySceneDetect Contributors 2026). For this chapter, the significance of PySceneDetect lies not in the sophistication of its algorithm, but in the fact that it transforms "raw video" into "the minimal training unit consumable by a generative model."

One boundary condition must be stated clearly. Chapter 10 has already covered general shot segmentation and keyframe selection; this chapter does not revisit the algorithmic principles, but only underscores the trade-offs specific to generative use cases:

First, **cutting too coarsely is unacceptable.** If a single clip spans a dolly shot, a cut, a character transition, and a scene change, the resulting caption will be ambiguous.

Second, **cutting too finely is also unacceptable.** Clips that are too short lose the action process; the model can only observe a small segment of static or near-static frames.

Accordingly, a T2V data chain does not mechanically "segment and proceed." Post-segmentation filtering is still required. What actually enters training is the subset of clips with relatively clear shot boundaries, complete action processes, appropriate durations, and coherent content.

In many public video generation projects, shot-level clips are the prerequisite for all subsequent work. HunyuanVideo explicitly emphasizes single-shot clip organization; Open-Sora and Open-Sora-Plan, while differing in implementation details, also both default to clips as the primary training unit. In other words, **shot segmentation is not an optional preprocessing step; it is the entry gate of the T2V data pipeline.**

### 48.4.3 Motion Detection: Deciding Whether a Clip Is Worth Learning From

Once a video is segmented into shot-level clips, the next most important question becomes: does this clip contain motion worth learning from?

Image generation only requires adequate visual quality; video generation is different. A clip with almost no motion leads the model to learn something closer to continuous static frames, and it is likely to degrade into "an image that slightly jitters." Conversely, a clip with excessively strong or chaotic motion—rapid camera shake, severe compression, or frequently jumping perspectives—may cause the model to learn noise rather than meaningful temporal patterns. Truly valuable training video typically occupies a middle range: sufficient dynamic change without rendering action trajectories and subject relationships uninterpretable.

This is why **motion detection** is a uniquely necessary gate in T2V pipelines. Common approaches include optical flow estimation, inter-frame differencing, trajectory stability analysis, and camera jitter detection. HunyuanVideo uses optical flow to estimate motion velocity, combined with sharpness, OCR, and watermark signals for comprehensive filtering; Open-Sora 2.0 treats motion score as an explicit variable—using it not only to filter samples but also appending it directly to captions as a conditioning quantity during training.

From a training objective perspective, motion detection serves at least three functions.

First, it eliminates "static pseudo-videos." Many online videos are essentially slideshows with background music, PowerPoint-style screens, or footage where the subject barely moves. Such data provides limited benefit to video generation and tends to push the model toward static-frame-like outputs.

Second, it eliminates "chaotic motion." Rapid camera shake, extreme compression artifacts, and meaningless high-speed cuts all prevent the model from learning stable relationships between subject motion and background change.

Third, it provides a control dimension for subsequent captioning. That is, motion detection serves not only "deleting data" but also "writing data." When the system knows that a clip's motion amplitude is low, medium, or high, it can write corresponding expressions such as "slow movement," "fast running," or "violent shaking" into the caption, making training data more closely resemble the prompts users will actually input.

Motion detection in a T2V data chain is therefore far from a peripheral module; it effectively determines whether the model will learn to "move naturally."

### 48.4.4 Multi-Frame Sampled Captioning: Moving from Static Descriptions to Action Descriptions

After motion filtering, only the retained clips proceed to the captioning stage. The difference from T2I is pronounced here: image captioning needs only to process a single frame; a video captioner that still looks only at one frame cannot adequately describe action changes. A person standing still and a person rising and leaving look nearly identical in a single frame, yet from a video perspective the two are entirely different training samples.

T2V pipelines therefore typically do not "extract one frame and describe it," but instead employ **multi-frame sampling followed by captioning.** The system selects several representative frames from a clip and allows a VLM or video captioner to synthesize information from these frames into a more complete description. The resulting text can simultaneously cover subjects, actions, environment, and the process of change.

The value of multi-frame sampling is most apparent in three respects.
First, it allows captions to answer not only "what does this frame look like" but "what happened in this footage."
Second, it reduces the influence of single-frame contingencies. A particular frame may suffer severe occlusion or motion blur, but multi-frame aggregation allows the model to extract more stable semantics.
Third, it creates space for subsequent spatiotemporal alignment. Only when captions are grounded in multi-frame information can temporal order and action progression be explicitly represented.

In engineering practice, different projects use different captioners. Open-Sora-Plan uses ShareGPT4V-Captioner-7B and LLaVA-1.6-34B to generate dense captions; CogVideoX-related work has open-sourced the CogVLM2-Caption component for direct community reuse; HunyuanVideo goes further still—rather than targeting "a single sentence of output," it decomposes descriptions into structured fields. Regardless of the model used, the underlying principle is consistent: **the goal of captioning is not to title a video, but to translate the learnable content of a video into training supervision.**

### 48.4.5 Spatiotemporal Alignment: Making Captions Correspond to the Action Sequence

If multi-frame sampling addresses "seeing more comprehensively," then spatiotemporal alignment addresses "writing more accurately."

The problem with many video captions is not incorrect subject identification, but ambiguous action sequence. For example, "a person walks into a room, sits down, and picks up a cup"—without temporal information, the caption may be compressed to "a person sitting in a room holding a cup." This is barely adequate for understanding tasks, but is suboptimal for generative training because the model loses the sequential order of actions.

T2V data pipelines therefore typically include a layer of **spatiotemporal alignment** following multi-frame captioning. In its simplest form, the complete caption preserves the action sequence as closely as possible. More sophisticated approaches, applied to high-value samples, introduce explicit temporal information—for example, binding key frames to timestamps, or dividing a clip into sub-intervals and generating local descriptions for each. The timestamp mechanism in CogVLM2-Video is a canonical example of this signal: when the system can write "what happens at which second" into the supervisory text, the model's learning of action processes becomes more stable.

It is important to emphasize that spatiotemporal alignment does not mean this chapter re-examines long-video timeline design. That subject is handled in Chapter 10. This chapter addresses a narrower question: **when training a T2V model, how to make captions more faithfully correspond to the action sequence and spatial changes within a clip.** The more precise this alignment, the better the model learns fine-grained processes such as "the character first turns around, then steps forward, and finally leaves the frame," rather than outputting vaguely animated sequences.

**Table 48-3: Spatiotemporal Alignment Strategies for Video Captioning**

| Strategy | Method | Representative Projects | Advantages | Limitations | Applicable Position |
|---|---|---|---|---|---|
| Single-shot single caption | One overall description per single-shot clip | HunyuanVideo, Open-Sora | High throughput; suitable for large-scale pretraining | Intra-shot details and action order easily compressed | Large-scale pretraining main data |
| Multi-frame sampling + aggregated caption | Extract several keyframes; VLM synthesizes one description | Open-Sora-Plan, CogVideoX | More stable than single-frame; covers action changes | Temporal order may still be weakened | Medium-quality training set |
| Multi-frame alignment with timestamps | Key frames bound to timestamps; descriptions or QA generated | CogVLM2-Video | Action order explicit; benefits learning of long actions | High cost; complex organization | High-value samples, distillation sets, evaluation sets |
| Structured caption + field recombination | Generate short description, dense description, background, action fields, then assemble training text | HunyuanVideo | Easy to govern; amenable to sampling, statistics, and recombination | Annotation system is complex | High-quality pretraining and SFT |
| Motion score appended | Append motion intensity as a conditioning quantity to the caption | Open-Sora 2.0 | Training and control interfaces unified | Motion expression is coarse | Motion control enhancement |

### 48.4.6 Shot Language Annotation: Writing "How the Shot Is Taken" into Training Data

One of the most significant differences between video generation and image generation is that video inherently carries cinematic shot language. Users often request not only "a person running" but "a medium-shot tracking follow of a person running on a street," "a fixed-camera shot of waves crashing on shore," or "an aerial view of a vehicle passing through mountain roads." If training data contains no explicit shot information, the model treats these terms as ordinary text tokens and cannot execute them reliably.

Accordingly, the later stages of a T2V pipeline typically require **shot language annotation**. Rather than focusing only on subjects and actions, this stage captures shot size, camera position, angle, camera motion, and visual style. Common fields include: close-up/medium shot/wide shot, top-down/eye-level/low-angle, static camera, dolly-in, pull-out, pan, tilt, tracking shot, as well as style labels such as photorealistic, animated, cinematic, and documentary.

HunyuanVideo provides the most thorough disclosure in this area. It not only employs structured JSON captions but also incorporates a camera motion classifier, making "how the camera moves" a component of the training supervision. As a result, what the model learns is not merely "a person is walking," but "the camera slowly pushes in while capturing a person walking." This information is especially critical for user control of video generation, because in actual generation scenarios, the challenge in many prompts lies not with the subject but with the shot.

Shot language annotation also offers a very practical benefit: it differentiates training samples that contain the same action. "A girl turns and smiles" shot in close-up with a static camera, versus the same action shot with a handheld tracking follow, carries entirely different training implications. The former is closer to facial expression and portrait control; the latter simultaneously encompasses action and camera motion. Writing these differences into the data makes the model more controllable at generation time.

### 48.4.7 Data Stratification: Low Resolution Needs Volume; High Resolution Needs Purity

After shot segmentation, motion detection, multi-frame captioning, spatiotemporal alignment, and shot language annotation, data is not fed into training all at once. Video generation is typically trained in stages, and data must therefore be routed in corresponding tiers.

Low-resolution stages prioritize scale, aiming to learn world knowledge, basic actions, and rough temporal structure; high-resolution stages prioritize purity, emphasizing image quality, stability, shot consistency, and aesthetics; the final SFT or high-quality fine-tuning stage typically requires manually curated samples and more stringent caption quality control. HunyuanVideo discloses a similar philosophy: as resolution increases, filtering thresholds are progressively raised, and an additional high-quality fine-tuning set is constructed. Open-Sora 2.0 adopts a comparable strategy, using different captioners at different stages to balance throughput against accuracy.

This reveals that a T2V data pipeline is not a linear "clean then output" process, but a progressively converging training organization system. Video sources enter as a disordered large pool; shot segmentation transforms them into trainable units; motion detection selects clips with learning value; multi-frame captioning and spatiotemporal alignment produce supervisable samples; shot language annotation produces controllable samples; and finally samples are routed into different data buckets according to training stage. What ultimately determines model quality is typically not any single module in isolation, but whether these modules can operate in stable coordination.



![Figure 48-2: T2V Data Pipeline](../../images/part11/ch33_2.png)

*Figure 48-2: T2V Data Pipeline*


---

## 48.5 Comparative Dataset Disclosure: From SD3 to Open-Source Video Pipelines

### 48.5.1 Stable Diffusion 3: Caption Ablation as the Most Valuable Contribution

The most important contribution of the SD3 paper to this chapter is its public disclosure of recaptioning ablation results. It uses CogVLM to generate synthetic captions and adopts a 50/50 mixed training setup. Experiments show that caption rewriting measurably improves GenEval, especially on prompt-following sub-capabilities such as color, position, and counting.

This has significant implications for engineering practice. Many teams prioritize model architecture, training steps, and GPU memory requirements, while overlooking the quality of supervisory text. SD3's findings serve as a reminder: once an image generation model has entered the large-scale training phase, the caption distribution itself becomes an adjustable hyperparameter.

### 48.5.2 FLUX: Strong Results, Limited Data Chain Disclosure

FLUX was introduced by Black Forest Labs in 2024. Public materials focus primarily on model specifications, flow matching, open weights, the dev/schnell/pro model series, and guidance distillation. Models such as FLUX.1 dev and FLUX.1 schnell have had substantial community impact, with notable image quality, aesthetics, and prompt-following capability.

From a data engineering perspective, however, FLUX's foundational training data sources, filtering thresholds, recaptioning strategy, and bucketing details are sparsely disclosed. Subsequent model cards for FLUX Kontext, FLUX Krea, and similar models focus more on intended use, risk mitigation, and release strategy. In a book treatment, FLUX is best positioned as a case of "strong industry results with limited data transparency," and community speculation should not be presented as fact.

### 48.5.3 HunyuanVideo: Core Case Study for This Chapter

HunyuanVideo is appropriate as the core case study of this chapter not because it merely stated "we used large-scale video data," but because it has disclosed several critical stages of the video generation data chain to a degree that permits substantive analysis: how raw data enters the system, how clips are filtered for low-value samples, how video quality is stratified, how captions expand from single sentences into structured descriptions, and how data ultimately serves different training stages. For readers wishing to reproduce T2V data engineering, this is far more valuable than knowing the model parameter count alone.

Its data sources divide into image and video streams. The image side is used for unified image/video generation pretraining, starting from billions of image-text pairs; the video side comes from a large-scale internet video pool covering common visual domains including people, animals, plants, landscapes, vehicles, architecture, objects, and animation. The design logic is clear: image data provides large-scale visual concept coverage and static aesthetic distribution, while video data provides motion, shot language, and temporal change. If a video generation model relied solely on video data, the cost would be high and coverage might still be insufficient; combining high-quality image pretraining with video training allows the model to first learn rich spatial and visual knowledge before moving on to learning the dynamics of the temporal dimension.

In data organization, HunyuanVideo constructs multiple video training subsets with progressively increasing resolution, rather than mixing all videos into a single large pool for direct training. The video training sets described in the paper expand tier by tier from low to high resolution—for example, clips at the 256p level are used for early training, while higher-resolution data progressively enters later stages, up to 720p. Frame count also varies across stages: lower stages emphasize scale and coverage, while higher stages emphasize image quality, motion stability, and text alignment. This approach clarifies training objectives: early stages use large amounts of data to learn basic motion and world concepts; later stages use cleaner, higher-quality data to refine details, image quality, and cinematic shot expression.

Before formal filtering, raw videos are first organized into single-shot clips. The specific segmentation algorithm is not expanded here, as Chapter 10 has addressed general video segmentation. This chapter focuses on the processing that follows after a generative model takes over: whether a clip merits inclusion in the training set. HunyuanVideo selects sharp starting frames, using sharpness signals such as Laplacian operators to exclude blurry clip beginnings; it then employs an internal VideoCLIP model to extract video embeddings for duplicate sample detection and similar content aggregation. Large-scale public video pools contain abundant duplicate content—the same footage may appear repeatedly after cropping, transcoding, reuploading, or watermarking. Without deduplication, the model is pulled disproportionately by a small number of high-frequency assets, diminishing diversity. Embedding-based deduplication addresses the "too many duplicates" problem, while k-means concept clustering and resampling address the "too many common categories, too few long-tail categories" problem. The paper reports clustering centers on the order of tens of thousands, used to rebalance the distribution of video concepts.

The filtering stage is the most engineering-valuable component of HunyuanVideo's data pipeline. Rather than using a single aggregate score to determine whether a video is usable, it decomposes into multiple categories of quality signals. The first category is visual quality—sharpness, blur, compression noise, and overall appearance. The second is aesthetic quality; the paper uses video quality and aesthetic assessment methods such as DOVER to stratify video clips. The third is motion quality, where optical flow estimation is used to determine whether a clip contains meaningful motion, and whether that motion is excessively strong, chaotic, or nearly absent. The fourth is content interference, including OCR detection of large-area subtitles and text occlusion, and detection models similar to YOLOX to filter watermarks, logos, and borders. Video generation models are highly prone to learning these interference features; if training sets contain abundant platform watermarks, subtitle bars, channel bugs, and border templates, similar residuals may appear in generated outputs. HunyuanVideo therefore treats text, watermarks, logos, and borders as independent risk categories rather than folding them into a generic image quality score.

Filtered samples are not placed into a single unified training pool all at once; instead, they are distributed across different stages according to quality and resolution. Low-resolution stages can accommodate more samples, prioritizing coverage of world knowledge and basic actions; high-resolution stages raise filtering thresholds and retain sharper, more aesthetically pleasing, and more stable videos; finally, a high-quality fine-tuning dataset of approximately one million samples is constructed, emphasizing visual aesthetics and appealing motion through human annotation or human selection. This is critical: HunyuanVideo's data strategy is not "more is better," but "different stages consume data of different quality." This is also a common cost control approach in large-scale video generation training. Low-cost stages use quantity to establish capability ceilings; high-cost stages use high-purity data to refine output quality.

Caption generation is another core component. An ordinary video caption of "a person walking on the street" is far too sparse for training a T2V model. HunyuanVideo organizes captions as structured JSON rather than as a single indivisible block of natural language. Structured fields typically include short description, dense description, background, style, shot size, lighting, and atmosphere, plus additional camera motion labels. The short description summarizes subjects and actions; the dense description supplements object attributes, spatial relationships, and action processes; the background field specifies scene location; style records visual characteristics such as photorealistic, animated, and cinematic; shot type records shot sizes such as close-up, medium shot, and wide shot; and lighting and atmosphere encode lighting conditions and emotional tone into the training supervision. As a result, what the model learns is not only "what is in the video" but also "how this video was captured."

The camera motion classifier is also important. Video generation prompts frequently contain expressions such as "slow dolly-in," "fixed camera," "aerial top-down view," and "handheld tracking follow." If training data lacks stable camera motion labels, the model will struggle to map these terms to visual changes. HunyuanVideo's incorporation of camera motion into the annotation system is in essence supplying the T2V model with a complete cinematic language supervision system. For generative models, this type of information is more useful than ordinary classification labels, because what users ultimately control in video generation is most often precisely shot, rhythm, size, and atmosphere.

HunyuanVideo's data pipeline can be understood as a layered funnel: the outermost layer is a large-scale image and video pool; the first layer handles duplicates, low sharpness, format errors, and obviously low-quality samples; the second layer handles aesthetics, motion, blur, OCR, watermarks, logos, and borders; the third layer performs concept resampling to prevent common content from dominating; the fourth layer generates structured captions encoding action, background, style, shot size, lighting, atmosphere, and camera motion as trainable text; the final layer distributes samples into different training stages by resolution and quality. The value of this case study lies precisely here: it advances T2V data engineering from "video cleaning" to "training supervision production." Data is not merely filtered out; it is re-described, re-stratified, and re-routed, ultimately becoming the direct source from which the model learns video generation capability.

HunyuanVideo can therefore serve a more specific role in the dataset disclosure comparison of this chapter: it does not generically represent "public video models," but represents a relatively complete T2V data engineering paradigm. Compared with the image recaptioning experience of DALL·E 3 and SD3, it extends caption rewriting to the video domain. Compared with community pipelines such as Open-Sora, it provides a more industry-grade quality stratification and structured annotation philosophy. Compared with strong models with limited disclosure such as FLUX and Wan2.2, it is the most suitable primary case for this chapter's analysis of a data pipeline.

### 48.5.4 Wan2.2: Cinematic Data Strategy

Wan2.2 has been publicly available since July 2025, with an emphasis on cinematic quality in video generation, Mixture of Experts (MoE), T2V/I2V/TI2V capability, and larger-scale image and video training data. Public materials indicate that it expands image and video data beyond Wan2.1, introduces finer-grained aesthetic data and cinematic style labels, and enables the model to be better suited for creative scenarios in terms of lighting, color grading, cinematic texture, and visual style.

Wan2.2's data strategy is well suited as a case study for "aesthetic label-driven video generation." Unlike HunyuanVideo, it does not fully expand filtering and caption details, but it explicitly indicates that industry-grade video models are transitioning from "capable of generation" to "generating usable creative assets," and that aesthetic data and style labels will become important directions for future T2V data engineering.

### 48.5.5 Open-Sora / Open-Sora-Plan / CogVideoX: Community-Reproducible Approaches

Open-Sora 2.0 discloses a reasonably systematic data filtering philosophy: basic pre-filtering by bpp, fps, duration, and aspect ratio, followed by multi-stage filtering with aesthetic, motion, blur, OCR, and camera jitter signals. On the caption side, low-resolution stages use LLaVA-Video, while high-resolution stages switch to Qwen 2.5 Max, with motion score appended to captions.

Open-Sora-Plan places greater emphasis on low-risk startup. It builds datasets from CC0 video sources such as Mixkit, Pexels, and Pixabay, segments them into clips, and generates dense captions using ShareGPT4V-Captioner-7B and LLaVA-1.6-34B. For research teams without commercially licensed data, this approach is more readily deployable.

CogVideoX's contributions lie in open component release. CogVLM2-Caption and related prompt conversion tools allow the community to reuse video recaptioning capability. The full training data pool has not been completely disclosed, but the caption components themselves are highly valuable for constructing T2V data chains.

---

## 48.6 Case Studies

### Case A: LAION to SD3-Style Caption Rewriting

Assume an existing LAION subset or custom image-text pool, with the goal of transforming it into data better suited for T2I training. A viable workflow is as follows.

Begin with basic cleaning: download status verification, image integrity checks, hash deduplication, resolution filtering, language filtering, and initial NSFW and watermark screening. Then use CLIP score or a similar image-text matching score to remove clearly mismatched samples. Next proceed to aesthetic and preference scoring, partitioning samples into a main training set, a low-weight set, a review set, and a discard set.

The caption rewriting stage may use CogVLM2 or another strong VLM. Step one: have the VLM generate a dense caption covering subject, quantity, color, position, background, style, lighting, material, and composition. Step two: distill the dense caption into several length variants—short caption, medium caption, long caption, and structured fields. Step three: retain a portion of original captions to form a mixed supervisory signal similar to SD3.

Do not delete all original captions. Original text often contains web context, proper nouns, obscure entities, and natural user phrasing that, despite its noisiness, provides concept coverage value. Synthetic captions are cleaner and more detailed but are constrained by VLM biases. A more robust training manifest might be organized as follows: 50% original captions, 30% dense captions, 20% random-length distilled captions; for vertical domains, additionally include human-reviewed captions.

The core objective of this process is not "writing longer captions," but adjusting the supervisory language distribution. The model needs to be exposed to short prompts, long descriptions, naturally noisy text, and high-quality structured text simultaneously; only then will it not overfit to any single prompt style at inference time.

### Case B: Reproducing a HunyuanVideo-Style Video Data Pipeline

Reproducing HunyuanVideo's approach does not require building a general-purpose segmentation system from scratch. Chapter 10 has already addressed segmentation, timelines, and keyframes. This chapter begins from shot-level clips.

A minimum viable version can be designed as follows:

1. Ingest the clips, keyframes, timelines, and basic metadata produced by Chapter 10;
2. Compute quality signals for each clip: sharpness, blur, OCR area, logo/watermark score, motion intensity, jitter level, and basic aesthetic score;
3. Deduplicate using video embeddings to prevent the same footage from appearing repeatedly in the training set;
4. Perform concept clustering or topic bucketing on candidate clips to prevent common scenes from dominating;
5. Generate structured captions for retained samples, including short description, dense description, action description, background, shot size, lighting, style, and atmosphere;
6. Supplement high-quality samples with labels for camera movement, shot type, and motion level;
7. Organize training manifests by resolution and quality tier.

The emphasis of this pipeline is "routing." The low-resolution pretraining stage can accommodate more samples as long as risks are controlled; the high-resolution stage raises aesthetic, sharpness, and stability thresholds; the SFT stage uses manually curated or high-confidence caption samples. This preserves scale while ensuring purity in later training stages.

### Case C: Multi-Tier NSFW + Copyright + Watermark Filtering

NSFW, copyright, and watermark filtering cannot be collapsed into a single aggregate score. An aggregate score appears convenient but is dangerous in generative model data engineering: a visually appealing image is not thereby cleared for inclusion in a training set; a video with rich motion cannot offset watermark, logo, or privacy risks. A more robust approach is to decompose the filtering system into multiple decision layers, each responsible for a different class of problem, with independent audit fields preserved throughout.

The first layer is a hard safety block layer, handling high-risk samples such as NSFW content, risks involving minors, overt violence and gore, and personal privacy exposure. This layer is not appropriate for "down-weighted training," nor should subsequent models be expected to learn to self-censor around such content. Any sample matching a high-confidence rule should be immediately discarded; low-confidence samples enter a manual review queue. Generative models recombine visual patterns from training data; if safety risks are not eliminated during the training phase, subsequent reliance on inference-side moderation for remediation incurs substantially higher cost.

The second layer is a copyright and provenance governance layer, focused not on whether an image "looks good" but on whether it "can be used." This layer requires recording data source, download timestamp, license status, site terms of service, hash, similar-duplicate cluster membership, and opt-out list status. Samples of unclear provenance, suspected copyrighted assets, film or television screenshots, commercial photography, and brand promotional imagery should generally not enter the main training set directly, but should instead be routed to review, quarantine, or low-risk replacement processes. Once a generative model learns high-frequency copyrighted patterns, character likenesses, or specific commercial styles, generated outputs may exhibit close approximations of those materials.

The third layer is watermark, logo, and OCR interference filtering. Watermarks and logos are not always copyright issues, but they directly contaminate generated outputs. Platform bugs, subtitle bars, intro and outro sequences, border templates, and derivative work watermarks are especially common in video data; photography platform watermarks, marketplace identifiers, and promotional text are common in image data. If such samples are heavily represented in training, the model may spontaneously generate pseudo-watermarks in image corners or treat meaningless text as image texture. Handling can be differentiated by case: large-area watermarks are deleted directly; small corner watermarks may be cropped out; video clips with excessively high subtitle coverage should be filtered; samples with modest OCR text but valuable semantics may have their captions rewritten to explicitly note "frame contains text regions" and enter a low-weight bucket.

Only the fourth layer addresses quality and training routing—including aesthetic score, sharpness, motion intensity, subject completeness, composition quality, and concept rarity. The objective here is not a binary "keep or delete" decision, but determining where a sample is placed. High-aesthetic, high-consistency samples enter a high-quality training bucket; ordinary but clean samples enter a basic pretraining bucket; samples with mediocre image quality but rare concepts, complex actions, or special viewpoints may be retained at reduced weight; only low-quality samples with no special value are deleted.

The core principle of multi-tier filtering is therefore: safety risk takes precedence over quality score; copyright status takes precedence over aesthetic score; watermark and OCR interference require independent modeling; quality scoring is responsible only for training routing. The system's final output should not be merely a `keep/drop` flag, but should include safety label, copyright label, watermark label, OCR ratio, aesthetic score, motion score, review status, and routing result. Only then, when a model later exhibits safety, copyright, or visual contamination issues, can the data team trace the problem to specific sample clusters and perform rollbacks or resampling on the training manifest.



![Figure 48-3: Multi-Tier Aesthetic / Copyright / Safety Filtering Architecture](../../images/part11/ch33_3.png)

*Figure 48-3: Multi-Tier Aesthetic / Copyright / Safety Filtering Architecture*



---

## 48.7 Implementation Risks, Costs, and Boundaries

The first category of risk is caption hallucination. The more capable the captioner and the longer its output, the more likely it is to describe details that do not exist in the image—for example, interpreting a blurry background as architectural elements, characterizing ordinary clothing as a specific uniform, or representing a brief action as a complete event. The remedy is not simply shortening captions, but implementing sampling checks, confidence stratification, and random-length mixing. High-value samples can be manually reviewed; ordinary samples should at minimum retain a parallel original or short caption to reduce the influence of hallucinated supervision.

The second category of risk is video frame sampling bias. Sampling only the first, middle, and last frame may miss the key action; sampling uniformly may waste representation on static segments. Video captioning is best combined with motion intensity and key-change-point sampling. For high-value data, adding timestamps or local event descriptions allows the model to learn action sequences rather than only average visual appearance.

The third category of risk is aesthetic scoring drift. High-aesthetic samples tend to resemble professional photography, commercial posters, or post-processed imagery; yet the real world contains large amounts of valuable motion content that is not visually "beautiful." If filtering rules are too heavily skewed toward aesthetics, the model will lose samples depicting complex interactions, ordinary scenes, industrial environments, and long-tail objects. In practice, aesthetic scores should serve as routing signals rather than sole deletion criteria. Rare concepts, complex actions, and unusual viewpoints need protection mechanisms.

The fourth category of risk is deferred copyright and privacy handling. Addressing copyright risks after training is complete is extremely costly. The more robust approach is to preserve source, license, hash, OCR results, logo detection, watermark status, opt-out state, and review records at the time of data ingestion. This way, if a source subsequently needs to be removed, the system can roll back the training manifest rather than re-cleaning the entire corpus.

On the cost side, offline captioning for T2V is the most easily underestimated expense. Video decoding, multi-frame sampling, VLM inference, OCR, optical flow, deduplication, aesthetic scoring, and manual spot-checking all consume substantial GPU resources and storage bandwidth. Open-Sora-Plan's disclosed captioner inference speeds already demonstrate that video captioning is not a lightweight step. Engineering practice must use tiered approaches: deploy cheaper models for lower-value samples, route boundary samples to review, and reserve strong captioners and structured annotation only for high-value samples.

The scope boundaries of this chapter must also be respected. General video segmentation, ASR, speaker diarization, and subtitle timelines have been addressed in Chapter 10; multimodal understanding, video question answering, and event detection are handled by Chapter 47 and the foundational chapters in Part III. This chapter addresses only how training data for generative models is filtered, rewritten, routed, and audited. Maintaining this boundary avoids chapter redundancy and allows readers to see what is genuinely new in generative data engineering.

---

## Chapter Summary

Data engineering for T2I and T2V has evolved from the early "collect and clean" paradigm into a "collect–filter–rewrite–route–audit" workflow. DALL·E 3 and SD3 demonstrate that caption quality directly impacts prompt following; HunyuanVideo and Open-Sora demonstrate that video generation requires encoding motion, shot language, lighting, style, and spatiotemporal relationships explicitly into supervisory text; Wan2.2 reminds us that cinematic aesthetics and style labels are becoming new priorities in video generation data; and FLUX represents a different practical reality: model capability can be publicly demonstrated while the complete data chain remains undisclosed.

For engineering practice, what matters most is not memorizing specific fixed thresholds, but establishing a transferable set of data judgment criteria: whether raw media is safe, compliant, sharp, worth learning from, accurately described, and capable of entering the correct data bucket according to training stage. Only when these questions are systematically resolved does the ceiling imposed by model architecture have the opportunity to be fully realized.

This chapter also concludes Part XI. The preceding chapters addressed the collection, organization, understanding, and evaluation of multimodal data; this chapter shifts the perspective to the data pipelines of generative models. Generative AI data engineering is no longer merely "preparing more raw material"; it is evolving into a more refined capability for data orchestration—writing the world into a language that models can learn, keeping risks out of training sets, and transforming quality and style into controllable training variables.

## References

Betker J, Goh G, Jing L, Brooks T, Wang J, Li L, Ouyang L, Zhuang J, Lee J, Guo Y, others (2023) Improving Image Generation with Better Captions (DALL·E 3). OpenAI Technical Report.

PySceneDetect Contributors (2026) PySceneDetect Documentation. Available at: https://www.scenedetect.com/docs/latest/.

Esser P, Kulal S, Blattmann A, Entezari R, Müller J, Saini H, Levi Y, Lorenz D, Sauer A, Boesel F, others (2024) Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (Stable Diffusion 3). arXiv preprint arXiv:2403.03206.

Gadre S Y, Ilharco G, Fang A, Hayase J, Ilharco G, Marten T, Wortsman M, Goyal S, Guha E, Jain H, others (2023) DataComp: In Search of the Next Generation of Multimodal Datasets. In: Advances in Neural Information Processing Systems 36.

Ghosh S, Bhatt U, Bhattacharya R, Parmar P, Patel S, Islam M, Reddy K K, others (2023) GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment. In: Advances in Neural Information Processing Systems 36.

Kirstain Y, Polyak A, Singer U, Matiana S, Penna J, Levy O (2023) Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation (PickScore). In: Advances in Neural Information Processing Systems 36.

Kong X, Tian Y, Zhang J, Min R, Dai X, Deng X, Chen Q, Liu L, Ni M, others (2024) HunyuanVideo: A Systematic Framework for Large Video Generation Models. arXiv preprint arXiv:2412.03603.

Open-Sora Team (2024) Open-Sora: Democratizing Efficient Video Production for All. arXiv preprint arXiv:2412.20404.

Schuhmann C, Beaumont R, Vencu R, Gordon C, Wightman R, Cherti M, Coombes T, Katta A, Mullis C, Wortsman M, others (2022) LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models. In: Advances in Neural Information Processing Systems 35:25278-25294.

Wan Team (2025) Wan: Open and Advanced Large-Scale Video Generative Models. arXiv preprint arXiv:2503.20314.

Wang W, Lv Q, Yu W, Hong W, Qi J, Wang Y, Ji J, Yang Z, Zhao L, Song X, others (2023) CogVLM: Visual Expert for Pretrained Language Models. In: Advances in Neural Information Processing Systems 36.

Wu X, Sun K, Zhu F, Zhao R, Li H (2023) Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis (HPSv2). arXiv preprint arXiv:2306.09341.
