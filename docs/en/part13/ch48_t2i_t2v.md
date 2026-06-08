# Chapter 48: Multimodal Generative Model Data Engineering: T2I and T2V Pipelines

## Opening Note: Why Some Models Follow Prompts Better

In image and video generation, an underestimated question is: whose prompts is the model learning from?

Early image-text data mostly came from web alt-text, titles, nearby text, or short upload descriptions. This was often enough for multimodal understanding. If an image contains a cat and the caption says "cat on sofa," the model can learn a coarse relation between cat and sofa. Generation is harder. A user may ask for "an orange cat lying on a blue sofa by a window at dusk, side backlight, shallow depth of field, a green plant on the right." If training text rarely describes color, position, lighting, composition, material, style, and object relations, the model cannot reliably obey those words during generation.

The DALL-E 3 paper (Betker et al. 2023) put this issue at the center. Its focus was not merely a more complex image-generation backbone, but recaptioning training images with more descriptive captions. The key judgment was direct: the more training text resembles real generation requests, the easier prompt following becomes. Sora's public materials follow the same idea on the video side: use or train a strong captioner to rewrite training videos into detailed descriptions, and expand short user prompts at inference into descriptions better suited for the video model.

Stable Diffusion 3 (Esser et al. 2024) gives clearer engineering evidence. SD3 uses CogVLM (Wang et al. 2023) to generate synthetic captions and trains with a 50% original caption plus 50% synthetic caption mixture. Changing caption distribution alone raises GenEval (Ghosh et al. 2023) from 43.27 to 49.78, with stronger gains on prompt-following dimensions such as color attribution and position. Prompt following is therefore not determined only by architecture; supervision language in the training data is decisive.

Video generation depends even more on executable language. A video contains subject and background, but also motion, camera movement, shot type, speed, rhythm, lighting change, and temporal relation. A caption like "a man walking on the street" teaches a vague theme. A caption like "fixed camera, medium shot, a man walks from left to right, background cars move slowly, warm dusk light, slight handheld shake" can teach controllable motion and cinematography.

This chapter discusses how T2I and T2V generation systems turn raw images and videos into trainable, controllable, auditable data. Chapter 10 already covered general audio-video sample engineering: shot segmentation, keyframes, ASR, subtitle alignment, unified timelines, and clip / segment / frame-level organization. This chapter assumes those foundations and focuses on generation-specific steps: caption rewriting, aesthetic and quality scoring, safety and copyright filtering, video motion selection, spatiotemporal description, training bucketing, and data routing.

The chapter does not discuss model architecture such as DiT, U-Net, Transformer, Flow Matching, or VAE. It focuses on the quieter engineering capability: making raw media into generation supervision the model can actually learn.

## 48.1 Three Characteristics of Generative Model Data

Generation data still looks like image or video plus text, but it differs from understanding data. Understanding asks what is in the picture. Generation learns how a sentence becomes a picture or video. At least three themes matter: caption detail, aesthetic / quality scoring, and copyright / safety governance.

### 48.1.1 Caption Detail: From Tags to Executable Descriptions

Traditional image-text captions are short. Web alt-text, titles, product names, and social descriptions provide coarse semantics. They can support contrastive learning, but they are poor supervision for generation.

Modern T2I/T2V pipelines increasingly emphasize dense captions. Image captions should cover subject, attributes, count, spatial relations, style, lighting, background, composition, and material. Video captions add action process, movement direction, speed, shot type, camera motion, temporal change, and atmosphere. Many video projects now move from ten-word captions to dozens or hundreds of words, or to structured fields assembled into texts of different lengths by training stage.

HunyuanVideo (Kong et al. 2024) is a detailed public example. It organizes descriptions into fields such as short description, dense description, background, style, shot type, lighting, and atmosphere, and adds camera-motion labels. The model sees not only what appears, but how the shot is filmed and how it changes.

Captions should not become infinitely long. Long synthetic captions can hallucinate details and make generated outputs stiff or overly uniform. SD3's 50/50 mixture of original and synthetic captions is instructive: original captions are noisy but conceptually diverse; synthetic captions are dense but constrained by the captioner's biases. A mixture is often more stable than full replacement.

### 48.1.2 Aesthetic and Quality Scores as Routers

Generation models depend on "looking good" more than understanding models. An understanding model can learn categories from low-resolution, off-color, badly composed images. A generation model trained heavily on such samples will reproduce those flaws.

Scorers such as LAION-Aesthetic (Schuhmann et al. 2022), PickScore (Kirstain et al. 2023), and HPSv2 (Wu et al. 2023) act as data routers. LAION-Aesthetic is a low-cost large-scale sieve. PickScore and HPSv2 are closer to human preference. Pipelines usually chain multiple signals: cheap rules first, then aesthetic score, preference score, clarity, OCR area, subject completeness, and other routing signals.

Video quality scoring adds blur, jitter, compression noise, motion intensity, subtitle occlusion, watermark, logo, border, and camera stability. Open-Sora 2.0 (Open-Sora Team 2024) includes aesthetic, motion, blur, OCR, and camera-jitter filtering. HunyuanVideo uses technical quality, aesthetic quality, optical flow, OCR region, and watermark / logo / border detection to route samples into training tiers.

The goal is not "higher score is always better." Generation needs world diversity. Over-optimizing aesthetics can remove ordinary scenes, complex actions, rare objects, and multi-subject interactions. A better policy is: remove low-quality samples, block high-risk samples, send boundary samples to review or low-weight training, and preserve rare but useful samples separately.

### 48.1.3 Copyright, Safety, and Privacy as Hard Constraints

Generative models reproduce patterns from training distributions. Watermarks, logos, personal information, sensitive text, and copyrighted media can reappear in generated outputs. This cannot be solved only by final-stage content review; it must enter training-data governance.

Common governance signals include NSFW detection, watermark detection, OCR text ratio, logo detection, privacy-entity recognition, copyright source tags, duplicate image clustering, opt-out lists, and human review queues. Video adds subtitles, intros and outros, channel marks, border templates, and remix watermarks. HunyuanVideo explicitly includes OCR filtering and watermark / logo / border detection; LAION-5B includes NSFW and watermark metadata; OpenAI system cards emphasize explicit-content filtering, personal-information reduction, and training exclusions.

Engineering should separate three decisions. Safety problems hard-block. Copyright problems enter review or exclusion. Aesthetic and quality scores drive routing and weighting. Collapsing them into one total score is dangerous: a beautiful but risky sample may pass, while a low-aesthetic but rare-motion sample may be deleted. Audit metadata must be preserved for rollback, resampling, and accountability.

## 48.2 Comparing Mainstream T2I/T2V Data Pipelines

T2I and T2V pipelines should be read through three questions: where raw data comes from, how it is filtered and rewritten, and what training objective the processing serves. Differences usually lie in the full chain, not one module.

T2I data engineering has matured around image-text alignment, caption detail, and quality stratification. Early image generation relied on alt-text, titles, and nearby web text. These texts had broad coverage but sparse description. As prompt following became central, recaptioning moved to the core. DALL-E 3 emphasizes training an image description system and rewriting training images into richer text, while also expanding user prompts at inference. Its direction is clear but its full data pool and thresholds are not open. Stable Diffusion 3 is more reproducible because it discloses ablation evidence and uses a mixture of original and synthetic captions, preserving long-tail web concepts while adding visual detail.

FLUX represents a strong industrial model with limited data-chain disclosure. Public materials focus more on model cards, risk mitigation, open-weight variants, and output quality than on source pool, recaptioning, aesthetic filtering, or copyright filtering. It is useful as an industry case, but not a fully reproducible data recipe.

T2V pipelines are more complex. Videos must be stable, contain useful motion, maintain action continuity, avoid excessive blur and compression, and manage subtitles and watermarks. Video-specific signals include motion score, optical flow, camera jitter, blur, OCR area, shot language, action temporal order, and camera movement. Chapter 10 covered general video clipping; this chapter focuses on what happens before clips enter generative training.

HunyuanVideo is the clearest public core case. It uses large image and video pools and decomposes the pipeline into shot organization, deduplication, concept resampling, technical quality, aesthetic quality, blur detection, optical flow, OCR filtering, and watermark / logo / border detection. It generates structured captions rather than a single sentence, including short description, dense description, background, style, shot type, lighting, atmosphere, and camera motion. This writes cinematographic language into supervision.

Open-Sora and Open-Sora-Plan show community routes. Open-Sora 2.0 first filters bpp, fps, duration, and aspect ratio, then uses aesthetic, motion, blur, OCR, and jitter. It uses cheaper captioning at low resolution and stronger models at high resolution, appending motion score to captions. Open-Sora-Plan starts from low-copyright-risk sources such as Mixkit, Pexels, and Pixabay, then generates dense captions with ShareGPT4V-Captioner-7B and LLaVA-1.6-34B. CogVideoX discloses more components than full data, and its CogVLM2-Caption and prompt-rewriting tools are useful modules for custom pipelines.

Wan2.2 (Wan Team 2025) emphasizes cinematic quality. It expands image and video data on the basis of Wan2.1 and introduces finer aesthetic data, cinematic labels, and prompt systems. It reflects a shift from "make video move" to "make usable creative footage."

**Table 48-1: Mainstream T2I/T2V data pipelines**

| Model / project | Modality | Data entry and openness | Filtering and governance focus | Caption / annotation strategy | Data engineering lesson |
| --- | --- | --- | --- | --- | --- |
| DALL-E 3 | T2I | Data pool not fully disclosed; public materials emphasize high-quality image-text supervision | Safety filtering, personal-information reduction, inference-time prompt rewriting | Highly descriptive captions; user prompts expanded at inference | Descriptive captions are critical for prompt following |
| Stable Diffusion 3 | T2I | Large-scale image-text data, sources not itemized | Paper focuses on training and caption ablation; filtering chain less disclosed | CogVLM synthetic captions; original and synthetic captions mixed | Quantitative evidence that recaptioning improves prompt following |
| FLUX.1 / FLUX.1 Kontext | T2I / image editing | Training data details limited | Model cards emphasize risk mitigation, use boundaries, open-weight versions | Basic recaptioning details not systematically disclosed | Strong effect, but data chain is not fully reproducible |
| HunyuanVideo | T2V + T2I pre-training | Internet-scale image and video data; detailed pipeline disclosure | Deduplication, concept resampling, DOVER, clarity, optical flow, OCR, watermark, logo, border | Structured JSON captions with short / dense descriptions, background, style, shot, lighting, atmosphere, and camera motion | Best public case for video generation data engineering |
| Wan2.2 | T2V / I2V / TI2V | Expanded image and video data over Wan2.1 | High-quality data, aesthetic data, cinematic labels | Fine-grained aesthetic and creative prompt system | T2V data moves from general usability to cinematic controllability |
| Open-Sora 2.0 | T2V / I2V | Large candidate video pool, community disclosure | bpp, fps, duration, aspect ratio; aesthetic, motion, blur, OCR, jitter | LLaVA-Video at low resolution, Qwen 2.5 Max at high resolution; motion score in caption | Practical open-source staged filtering and captioning |
| Open-Sora-Plan | T2V | Mixkit, Pexels, Pixabay and other CC0 sources | Low-copyright-risk starting pool | ShareGPT4V-Captioner-7B and LLaVA-1.6-34B dense captions | Good for small reproducible T2V pipelines |
| CogVideoX | T2V / I2V / V2V | Full pool not fully open | More disclosed at component and tool-chain level | CogVLM2-Caption for video-to-text and prompt rewriting | Caption components are reusable in custom pipelines |

The point is not which pipeline is best. T2I depends more on image-text relevance, aesthetics, and recaptioning. T2V adds motion quality, shot language, and temporal alignment. Industrial models keep more details private, while open projects expose more reproducible modules.

## 48.3 T2I Data Pipeline

### 48.3.1 Data Entry: LAION, DataComp, and Custom Collection

A T2I data pool usually begins with public image-text pairs, licensed datasets, vertical-domain collections, or internal assets. The entry stage checks download success, image integrity, hash deduplication, resolution, aspect ratio, language, NSFW labels, and source rights. Image-text relevance should be scored before recaptioning; otherwise the captioner may describe images whose original pairing is already wrong, making provenance harder to debug.

### 48.3.2 Filtering: From "Usable" to "How Should It Be Used"

Filtering should route samples rather than simply keep or drop them. Low-resolution corrupt images are dropped. Clean ordinary images enter base pre-training. High-aesthetic and highly aligned samples enter high-quality stages. Rare concepts with moderate quality may be kept at low weight. Risky copyright or privacy samples enter review or exclusion. This routing mindset matters more than a single threshold.

### 48.3.3 Caption Rewriting: The Key Gain in T2I

Caption rewriting uses a strong VLM to produce dense descriptions, then often distills them into multiple lengths or structured fields. A robust recipe preserves some original captions because they contain web context, named entities, and long-tail natural phrasing. Synthetic captions add visual precision but can hallucinate. A practical training list may mix original captions, dense captions, and random-length distilled captions, with more human-reviewed captions in vertical domains.

| Route | Training-side captioner | Original caption strategy | Output form | Inference-time prompt rewriting | Advantages | Risks and costs |
| --- | --- | --- | --- | --- | --- | --- |
| DALL-E 3 | Dedicated descriptive captioner | Not fully disclosed | High-density natural-language caption | Yes | Strong prompt following, short prompts become executable | Closed system; captioner data and thresholds not reproducible |
| Stable Diffusion 3 | CogVLM synthetic captions | 50% original + 50% synthetic | Mixed raw and synthetic captions | Not the main paper focus | Clear ablation evidence and reproducible lesson | VLM bias and hallucination |
| CogVLM2 distillation | CogVLM2 / CogVLM2-Caption, then template or LLM distillation | Keep part of raw captions or random-length mix | Dense, structured, or random-length captions | Optional | Open and reusable, good for vertical domains | Higher cost; overlong captions can become stiff |

### 48.3.4 Bucketing: Match Training Schedule to Data Distribution

T2I data should be bucketed by resolution, aspect ratio, caption length, quality, and risk. Resolution buckets reduce padding waste. Caption-length buckets prevent short tags and long dense descriptions from destabilizing batches. Quality buckets support curriculum: broad data early, high-purity data late. Risk buckets keep audit and exclusion decisions traceable.

![Figure 48-1: T2I data pipeline](../../images/part11/ch33_1.png)

*Figure 48-1: T2I data pipeline.*

## 48.4 T2V Data Pipeline

T2V extends T2I with temporal structure. A video sample is not just a set of frames; it is a sequence of actions, camera movements, and visual states.

### 48.4.1 Video Sources: From Raw Video to Trainable Material

Sources include licensed stock video, CC0 or Creative Commons platforms, internal production footage, vertical-domain videos, and public web videos with usable terms. Source metadata is especially important because video often contains people, brands, subtitles, audio, and watermarks.

### 48.4.2 Shot Segmentation

Long videos should be split into single-shot clips before generative training. PySceneDetect-style tools can detect transitions, but training data needs further filtering: extremely short clips provide little motion, while very long clips are expensive and hard to caption. Chapter 10 covers general segmentation; here the focus is whether a clip is worth training.

### 48.4.3 Motion Detection

A T2V clip should contain useful motion. Optical flow, SSIM change, camera motion, and blur estimates can route clips. Near-static clips can still be useful for image-video consistency, but they should not dominate motion learning. Overly fast or chaotic clips may be filtered or sent to lower-weight buckets.

### 48.4.4 Multi-Frame Captioning

Single-frame captions lose temporal information. T2V captioning should sample key frames, summarize action changes, and include direction, speed, order, and camera motion. High-value samples can bind key frames to timestamps, creating captions that describe what happens when.

### 48.4.5 Spatiotemporal Alignment

Captions should correspond to the actual motion process. If a caption says the camera pushes in, frame statistics or visual evidence should support that. If a caption says an object enters from the left, sampled frames should show the sequence. Spatiotemporal alignment is a major difference between useful video supervision and generic video descriptions.

| Strategy | Method | Representative projects | Advantages | Weaknesses | Best use |
| --- | --- | --- | --- | --- | --- |
| Single-shot single caption | One clip, one overall description | HunyuanVideo, Open-Sora | High throughput, good for massive pre-training | Compresses in-shot detail and order | Main large-scale pre-training data |
| Multi-frame sampling + aggregate caption | Sample key frames and summarize with VLM | Open-Sora-Plan, CogVideoX | More stable than single-frame, covers action changes | Temporal order can still weaken | Medium-quality training set |
| Timestamped multi-frame alignment | Bind key frames to timestamps before description or QA | CogVLM2-Video-style pipelines | Clear action order, good for long action learning | Expensive and complex | High-value samples, distillation, evaluation |
| Structured caption + field recombination | Generate short description, dense description, background, motion fields | HunyuanVideo | Governable, easy to sample and recombine | Complex annotation system | High-quality pre-training and SFT |
| Motion score appended to caption | Add motion intensity as condition | Open-Sora 2.0 | Unified training and control interface | Coarse motion expression | Motion-control enhancement |

### 48.4.6 Shot-Language Labels

Generation users often request fixed camera, aerial view, handheld tracking, slow push-in, close-up, medium shot, and cinematic lighting. If training data lacks shot-language labels, the model cannot map these words to visual changes. Camera movement, shot type, lighting, and atmosphere should therefore enter captions or structured fields.

### 48.4.7 Data Stratification

Low-resolution stages need scale; high-resolution stages need purity. Early training can use more varied clips to learn world concepts and basic motion. Later stages should raise thresholds for clarity, aesthetics, motion stability, and caption alignment. SFT or high-quality tuning uses curated clips and human-reviewed captions.

![Figure 48-2: T2V data pipeline](../../images/part11/ch33_2.png)

*Figure 48-2: T2V data pipeline.*

## 48.5 Dataset Disclosure: From SD3 to Open Video Pipelines

### 48.5.1 Stable Diffusion 3: Caption Ablation Is the Key Evidence

SD3 is valuable because it shows quantitative gains from synthetic captions. Mixing original and CogVLM-generated captions improves prompt following without requiring a fully closed captioning system. For engineering teams, the lesson is to retrofit existing image-text pools gradually instead of waiting for perfect captions.

### 48.5.2 FLUX: Strong Output, Limited Data Disclosure

FLUX shows strong generation quality and practical influence, but its data chain is not fully disclosed. It should be discussed as an industry case: effective systems likely combine high-quality filtering, captioning, safety governance, and aesthetic control, but details should not be invented.

### 48.5.3 HunyuanVideo: Core Case

HunyuanVideo is the best public case for a complete T2V pipeline. It does not train on one undifferentiated video pool. Low-resolution clips support early training; higher-resolution data enters later; high-quality fine-tuning data is curated separately.

Raw videos are first organized into single-shot clips. The pipeline selects clear starting frames, uses clarity measures such as Laplacian signals, extracts video embeddings for deduplication, and uses clustering plus resampling to rebalance concepts. Duplicate video material is common because the same source may be cropped, transcoded, reuploaded, watermarked, and reused.

Filtering is multi-dimensional: visual clarity and compression, aesthetic quality such as DOVER, optical-flow motion, OCR text area, watermark, logo, and border detection. These signals should not be collapsed into one score. Watermarks and subtitles can directly pollute generated outputs, so they are independent risk items.

Filtered samples enter stages by quality and resolution. Low-resolution pre-training tolerates more data for coverage; high-resolution training uses cleaner and more beautiful clips; final fine-tuning uses curated high-quality clips. The strategy is not more is always better, but different stages consume different purity levels.

Structured captions are central. HunyuanVideo-style captions include short description, dense description, background, style, shot type, lighting, atmosphere, and camera motion. This teaches not only what is in the video, but how it is shot. The pipeline turns video cleaning into supervision production.

### 48.5.4 Wan2.2: Cinematic Data Strategy

Wan2.2 emphasizes cinematic video generation, MoE, T2V / I2V / TI2V capability, and larger image-video data. Public materials state that it expands data over Wan2.1 and adds finer aesthetic data and cinematic labels. It is a good case for aesthetic-label-driven video generation. T2V is moving from making motion possible to producing usable footage.

### 48.5.5 Open-Sora, Open-Sora-Plan, and CogVideoX

Open-Sora 2.0 discloses a systematic filtering approach: basic prefilters such as bpp, fps, duration, and aspect ratio, then aesthetic, motion, blur, OCR, and camera jitter. It uses lower-cost captioning for low-resolution stages and stronger models for high-resolution stages, and appends motion score to captions.

Open-Sora-Plan starts from lower-risk CC0 sources such as Mixkit, Pexels, and Pixabay, then uses ShareGPT4V-Captioner-7B and LLaVA-1.6-34B to generate dense captions. CogVideoX contributes reusable components such as CogVLM2-Caption and prompt conversion tools even though the full training pool is not open.

## 48.6 Case Studies

### Case A: Rewriting LAION into an SD3-Style Caption Mix

Given a LAION subset or internal image-text pool, start with basic cleaning: download status, image integrity, hash deduplication, resolution, language, NSFW, and watermark screening. Use CLIP score or similar relevance metrics to remove obvious mismatches. Then apply aesthetic and preference scoring to route samples into main training, low-weight, review, or discard sets.

For caption rewriting, use CogVLM2 or another strong VLM. Generate dense captions covering subject, count, color, position, background, style, lighting, material, and composition. Then distill them into short, medium, long, and structured variants. Keep part of the original captions to preserve web context, named entities, rare concepts, and natural user phrasing. A practical list might use 50% original captions, 30% dense captions, and 20% random-length distilled captions, with extra human-reviewed captions for vertical domains.

The goal is not simply longer captions. It is a supervision-language distribution that covers short prompts, long descriptions, noisy natural text, and high-quality structured text.

### Case B: Reproducing a HunyuanVideo-Style Pipeline

Start from shot-level clips produced by Chapter 10's pipeline. A minimal version:

1. input clips, keyframes, timelines, and metadata;
2. compute quality signals: clarity, blur, OCR area, logo / watermark score, motion intensity, jitter, aesthetic score;
3. deduplicate with video embeddings;
4. cluster or bucket concepts to avoid overrepresented scenes;
5. generate structured captions: short, dense, action, background, shot, lighting, style, atmosphere;
6. add camera movement, shot type, and motion-level tags for high-quality samples;
7. organize training manifests by resolution and quality tier.

The important idea is routing. Low-resolution pre-training can tolerate more samples; high-resolution stages raise aesthetic and stability thresholds; SFT uses human-curated or high-confidence caption data. This preserves scale while keeping late-stage purity.

### Case C: Multi-Level Filtering for NSFW, Copyright, and Watermarks

NSFW, copyright, and watermark filtering should not be reduced to one score. A beautiful image is not automatically usable, and a motion-rich video cannot offset privacy or copyright risk.

The first layer is hard safety blocking: NSFW, minor risk, explicit violence, and personal privacy leakage. High-confidence hits are discarded; low-confidence hits enter review. The second layer is copyright and source governance: source URL, download time, license, site terms, hash, duplicate cluster, and opt-out status. Unclear or likely copyrighted media enters review or exclusion. The third layer is watermark, logo, and OCR interference. Large watermarks are dropped; small corner marks may be cropped; subtitle-heavy clips are filtered; valuable samples with small text regions can be captioned and routed to low-weight buckets. The fourth layer is quality and training routing: aesthetics, clarity, motion, subject completeness, composition, and concept rarity.

The output should be more than `keep/drop`. It should include safety labels, copyright labels, watermark labels, OCR ratio, aesthetic score, motion score, review status, and routing decision.

![Figure 48-3: Multi-level filtering for aesthetics, copyright, and safety](../../images/part11/ch33_3.png)

*Figure 48-3: Multi-level filtering architecture for aesthetics, copyright, and safety.*

## 48.7 Pitfalls, Costs, and Boundaries

**Caption hallucination.** Stronger captioners and longer captions can invent details. Use sampling, confidence tiers, random-length mixing, and human review for high-value samples. Keep raw or short captions as a bypass to reduce hallucinated supervision.

**Video-frame sampling bias.** First-middle-last sampling can miss key actions; uniform sampling wastes frames on static segments. Captioning should combine motion intensity and key-change sampling. High-value data should include timestamps or local event descriptions.

**Aesthetic-score drift.** High aesthetic often means polished photos or commercial imagery, while many valuable real-world actions are not beautiful. Treat aesthetics as routing, not the only deletion criterion. Protect rare concepts, complex actions, and unusual views.

**Late copyright and privacy handling.** Dealing with copyright after training is extremely costly. Store source, license, hash, OCR, logo, watermark, opt-out status, and review records at ingestion so sources can be removed without cleaning the whole corpus again.

**T2V offline captioning cost.** Video decoding, multi-frame sampling, VLM inference, OCR, optical flow, deduplication, aesthetic scoring, and human sampling consume large GPU and storage bandwidth. Use cheaper models for low-value samples, review boundary samples, and reserve strong captioners for high-value clips.

This chapter's boundary is also important. General video slicing, ASR, speaker separation, and subtitle timelines belong to Chapter 10. Multimodal understanding, video QA, and event detection belong to the VLM chapters. This chapter only covers how generation training data is filtered, rewritten, routed, and audited.

## 48.8 Summary

T2I and T2V data engineering has evolved from collection and cleaning into collection, filtering, rewriting, routing, and auditing. DALL-E 3 and SD3 show that caption quality directly affects prompt following. HunyuanVideo and Open-Sora show that video generation needs motion, camera language, lighting, style, and spatiotemporal relations in supervision text. Wan2.2 shows that cinematic quality and aesthetic labels are becoming central. FLUX represents another reality: model capability can be public while the full data chain remains private.

For engineering practice, the important thing is not memorizing a threshold. It is building transferable data judgment: is the raw media safe, compliant, clear, useful, accurately described, and routed to the right training bucket? Only when those questions are solved can architecture reach its potential.

This chapter closes Part 13. The preceding chapters discussed multimodal data collection, organization, understanding, and evaluation; this chapter focused on generative data pipelines. Generative AI data engineering is not just preparing more media. It is a more precise orchestration ability: write the world in language the model can learn, keep risks out of the training set, and turn quality and style into controllable training variables.

## References

Betker J, Goh G, Jing L, Brooks T, Wang J, Li L, Ouyang L, Zhuang J, Lee J, Guo Y, others (2023) Improving Image Generation with Better Captions (DALL-E 3). OpenAI Technical Report.

Castellano B (2012) PySceneDetect: Python and OpenCV-based Scene Cut/Transition Detection Program. Available at: https://www.brettcastellano.com/post/pyscenedetect.

Esser P, Kulal S, Blattmann A, Entezari R, Muller J, Saini H, Levi Y, Lorenz D, Sauer A, Boesel F, others (2024) Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (Stable Diffusion 3). arXiv preprint arXiv:2403.03206.

Gadre S Y, Ilharco G, Fang A, Hayase J, Ilharco G, Marten T, Wortsman M, Goyal S, Guha E, Jain H, others (2023) DataComp: In Search of the Next Generation of Multimodal Datasets. In: Advances in Neural Information Processing Systems 36.

Ghosh S, Bhatt U, Bhattacharya R, Parmar P, Patel S, Islam M, Reddy K K, others (2023) GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment. In: Advances in Neural Information Processing Systems 36.

Kirstain Y, Polyak A, Singer U, Matiana S, Penna J, Levy O (2023) Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation (PickScore). In: Advances in Neural Information Processing Systems 36.

Kong X, Tian Y, Zhang J, Min R, Dai X, Deng X, Chen Q, Liu L, Ni M, others (2024) HunyuanVideo: A Systematic Framework for Large Video Generation Models. arXiv preprint arXiv:2412.03603.

Open-Sora Team (2024) Open-Sora: Democratizing Efficient Video Production for All. arXiv preprint arXiv:2412.20404.

Schuhmann C, Beaumont R, Vencu R, Gordon C, Wightman R, Cherti M, Coombes T, Katta A, Mullis C, Wortsman M, others (2022) LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models. In: Advances in Neural Information Processing Systems 35:25278-25294.

Wan Team (2025) Wan: Open and Advanced Large-Scale Video Generative Models. arXiv preprint arXiv:2503.20314.

Wang W, Lv Q, Yu W, Hong W, Qi J, Wang Y, Ji J, Yang Z, Zhao L, Song X, others (2023) CogVLM: Visual Expert for Pretrained Language Models. In: Advances in Neural Information Processing Systems 36.

Wu X, Sun K, Zhu F, Zhao R, Li H (2023) Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis (HPSv2). arXiv preprint arXiv:2306.09341.
