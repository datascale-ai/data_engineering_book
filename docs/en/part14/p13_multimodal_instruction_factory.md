# Project 13: Multimodal Instruction Factory

## Abstract

This project builds a reproducible data-engineering case around a "multimodal instruction factory." It focuses on business goals, data boundaries, architecture decisions, core implementation, acceptance metrics, and risk control. Installation commands and script details are condensed into an engineering-review perspective. The emphasis is on the relationship among sample schema, data flow, failure modes, and deliverables, so that readers can turn methods from earlier chapters into auditable and extensible project assets.

## Keywords

multimodal instruction factory; practical project; reproducible data engineering; data pipeline; acceptance metrics

## Project Goals and Reader Outcomes

This project uses the multimodal instruction factory as its core case. The goal is to build a multimodal instruction production chain covering images, text, OCR, charts, and dialogue tasks. After completing this chapter, readers should be able to identify the key data objects in this scenario, decompose the engineering pipeline, set acceptance metrics, and transfer the same approach to adjacent data-engineering tasks.

## Scenario Constraints and Data Boundaries

This project targets controlled assets and sample factories. It does not cover unauthorized media collection or fully automated safety review. These boundaries make the case reproducible and auditable. When data scale, data sources, permission scope, or deployment environment changes, sampling strategy, quality thresholds, runtime cost, and compliance requirements must be reassessed.

The boundary between this project and P03 must be explicit. P03 focuses on the classic LLaVA flow: image assets, OCR, bounding boxes, conversation templates, visual spot checks, and training packaging as the baseline chain. P13 focuses on modern multimodal-instruction factory capabilities: Qwen-VL-style generation, self-consistency quality calibration, LLM-as-Judge filtering, multilingual expansion, and unified packaging for multi-image and video references. P13 therefore does not repeat the proof of the LLaVA baseline pipeline; it shows how newer factory capabilities extend the data-factory skeleton already established in P03.

## Architecture Decisions

This project uses an architecture path of asset selection, task templates, caption/OCR signals, dialogue generation, quality scoring, and data packaging. The decision prioritizes input-output contracts, traceable versions, localizable exceptions, and reviewable results rather than compressing all logic into a one-off script.

## Sample Schema and Data Flow

The core data flow can be summarized as:

```text
visual assets -> metadata/OCR/caption -> instruction tasks -> multi-turn samples -> quality filtering -> multimodal training set
```

At minimum, the sample schema should retain fields such as `id`, `source`, `content_or_payload`, `metadata`, `quality_signals`, `split_or_stage`, and `audit_trace`. The exact fields are further refined by the data type, downstream task, and acceptance method used in this project.

## Core Implementation Fragments

The chapter keeps only implementation fragments that explain design trade-offs. Full scripts, long configurations, runtime logs, and large files should live in the companion repository or appendix notes. Code examples focus on input-output contracts, quality thresholds, exception handling, and acceptance interfaces.

## Experiment or Acceptance Metrics

Acceptance metrics include task coverage, image-text consistency, OCR usability, format pass rate, safety-filtering rate, and manual spot-check quality. If the project enters production, a course environment, or a public reproduction environment, it should also record version numbers, dependency environment, random seeds, sample spot-check results, and failure-sample review records.

| Acceptance dimension | Metric / evidence | Publication review rule |
| --- | --- | --- |
| Task coverage | Ratio of description, OCR, chart, grounding, and multi-turn QA tasks | Task types must correspond to data sources, model capability, and downstream training goals |
| Quality filtering | Image-text consistency, format pass rate, safety-filtering rate, self-consistency result, manual review quality | LLM-as-Judge results must retain scoring rules, spot-check calibration examples, and multi-sample consistency records |
| Multilingual expansion | Ratio of Chinese, English, and translated samples; cross-language terminology consistency; format-preservation rate | Multilingual samples must not be judged by quantity only; semantic consistency, visual reference, and proper-name translation require sampling review |
| Copyright safety | Image authorization, sensitive-content interception, redistribution boundary | Public examples should prefer authorized or owned assets; external images require separate registration |

*Table P13-1: Publication acceptance table for the multimodal instruction factory.*

## Cost, Risk, and Compliance Boundaries

Cost mainly comes from vision-language models, OCR, and manual review. Risk concentrates in image authorization, sensitive content, hallucinated descriptions, and task homogenization. When external data, personal information, copyrighted material, or third-party services are involved, retain source notes, permission status, masking strategy, call records, and manual-review records.

## Common Failure Modes

Common failures include input-distribution drift, missing schema fields, quality thresholds that are too loose or too strict, insufficient evaluation-sample coverage, unstable model calls, and results that cannot be traced. Troubleshooting should locate data boundaries and intermediate artifacts first, then inspect models, toolchains, and deployment environment.

## Reproducible Resource Notes

Reproduction materials should include data-source notes, a minimal sample set, configuration files, run commands, metric scripts, inspection reports, and output directories. The chapter keeps necessary fragments; full notebooks, long scripts, and large files should be maintained as companion resources.

## Background and Objectives

In VLM data engineering, the bottleneck is often not only the number of image-text pairs but also the construction of high-quality, diverse instruction data. In Project 3, the introductory LLaVA project, we showed how to generate simple descriptions and QA instructions from single images. For modern multimodal systems such as Qwen2.5-VL (Wang et al. 2024) and InternVL (Chen et al. 2024), that introductory data is no longer sufficient.

Industrial multimodal instruction synthesis must handle several challenges:

1. **Instruction diversity**: Beyond description, datasets need reasoning, fine-grained grounding, chart reading, and OCR tasks.
2. **Multi-source and multi-form input**: Data should support not only single images, but also interleaved images and video.
3. **Quality control**: Pure generation creates severe hallucinations, so multi-sample verification and LLM-as-Judge filtering (Zheng et al. 2023) are needed.

This project builds a complete multimodal instruction data factory. Starting from an image-only pool such as a LAION subset, it uses strong foundation models, including Qwen2.5-VL-7B and Qwen2.5-72B, to produce high-quality complex instructions in an automated and scalable way. After completing the project, readers can adapt the same production line to private image collections in domains such as medicine, law, and e-commerce.

## Architecture

The factory is divided into five components, shown in Figure 13-1.

![Multimodal Instruction Factory](../../images/part11/p13_mm_instruction_factory_arch_en.png)
*Figure 13-1 Qwen-VL-style multimodal instruction synthesis pipeline.*

1. **Seed selector**: Retrieves seed images from massive image pools, emphasizing OCR-rich images, charts, and realistic complex scenes.
2. **Instruction generator**: Defines six categories of complex instruction templates and calls Qwen2.5-VL through vLLM (Kwon et al. 2023) for high-throughput generation.
3. **Quality scorer and self-consistency**: Uses self-consistency (Wang et al. 2023) to validate reasoning tasks through repeated sampling.
4. **LLM-as-Judge filter**: Uses a strong text-only model such as Qwen2.5-72B-Instruct to score logic and detail, discarding samples below 4.0.
5. **Multilingual expander and packer**: Extends data through Chinese-English translation where needed and exports a unified format that supports image, multi-image, and video references.

Table P13-2 maps architecture components to code entry points and key artifacts. Unlike P03, P13 does not walk through LLaVA image-text preparation again. Its focus is how a modern multimodal instruction factory organizes seed selection, templates, generation, filtering, expansion, packaging, and acceptance into a reviewable chain.

| Stage | Code entry | Main input | Main output | Key review point |
| --- | --- | --- | --- | --- |
| Seed selection | `seed_selector.py` | LAION metadata or private visual-asset manifest | Seed list | Resolution, aspect ratio, original caption length, authorization status |
| Template management | `instruction_templates.py` | Task type | Prompt template | Task coverage, template repetition, prompt boundary |
| VLM generation | `generate_with_qwen_vl.py` | Seed list, templates, Qwen2.5-VL | Raw instruction records | Model version, sampling parameters, failed samples |
| LLM-as-Judge | `llm_judge.py` | Instruction and response | Scored records | Scoring rule, threshold, human calibration examples |
| Self-consistency | `self_consistency.py` | Multi-sample generations | Consistency score | Multi-sample agreement, reasoning-task stability |
| Multilingual expansion | `multilingual_expand.py` | High-quality English samples | Bilingual records | Terminology consistency, visual-reference preservation |
| Unified packaging | `pack_multi_image_video.py` | Scored records | `mm_sft_final.jsonl` | Qwen format, image/video paths, conversation fields |
| Unit tests | `tests/test_factory.py` | Template, judge, expansion, packaging functions | Test report | Basic contracts and example-output completeness |

*Table P13-2: Stage artifacts and code entry points for the multimodal instruction factory.*

The key function of Table P13-2 is to split "generation" out of a single model call. In real projects, VLM generation is only the middle of the pipeline. Before it, controlled seeds and task templates are required; after it, consistency checks, score filtering, multilingual review, and format packaging are required. If only the generation script is kept, the chapter becomes a demo. If stage artifacts and review fields are kept, the chapter has the engineering depth expected of a project chapter.

## Step-by-Step Implementation

### Step 1: Seed Selector

From an open LAION subset (Schuhmann et al. 2022), use metadata such as image width, height, original caption length, and tags to select promising seeds.

```python
# code/zh/project_13_mm_instruction_factory/seed_selector.py
from datasets import load_dataset


def select_seeds(dataset_name="laion/laion2B-en", num_samples=5000):
    print("Loading LAION metadata...")
    # In production, stream metadata first instead of downloading all images.
    ds = load_dataset(dataset_name, split="train", streaming=True)

    seeds = []
    for item in ds:
        try:
            w, h = item.get("WIDTH", 0), item.get("HEIGHT", 0)
            if w > 512 and h > 512 and 0.5 < (w / h) < 2.0:
                # Text longer than 10 words suggests richer visual context.
                if len(str(item.get("TEXT", "")).split()) > 10:
                    seeds.append({
                        "url": item["URL"],
                        "original_caption": item["TEXT"],
                    })
        except Exception:
            continue

        if len(seeds) >= num_samples:
            break

    print(f"Selected {len(seeds)} high-quality seed images.")
    return seeds


if __name__ == "__main__":
    select_seeds(num_samples=100)
```

### Step 2: Instruction Template Design

Unlike fixed-question LLaVA data, this pipeline needs diverse roles and task templates.

```python
# code/zh/project_13_mm_instruction_factory/instruction_templates.py
import random

TEMPLATES = {
    "detailed_description": [
        "Please provide a highly detailed, comprehensive description of this image, capturing every visible element, spatial relationship, and background context.",
        "Describe this image as if you are explaining it to someone who cannot see it, ensuring no detail is left out.",
    ],
    "complex_reasoning": [
        "Based on the visual evidence in the image, infer the sequence of events that likely led to this scene. Explain your reasoning step-by-step.",
        "What are the implicit relationships between the objects shown? Provide a logical deduction.",
    ],
    "ocr_reading": [
        "Extract all visible text in this image and format it into a structured markdown table or list.",
    ],
}


def get_random_prompt(task_type):
    return random.choice(TEMPLATES.get(task_type, TEMPLATES["detailed_description"]))
```

### Step 3: High-throughput Generation with vLLM

With vLLM's high concurrency, selected images and instruction templates can be sent to a base multimodal model at scale.

```python
# code/zh/project_13_mm_instruction_factory/generate_with_qwen_vl.py
from vllm import LLM, SamplingParams
from instruction_templates import get_random_prompt


def generate_instructions(seeds, model_path="Qwen/Qwen2.5-VL-7B-Instruct"):
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_num_seqs=16,
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)

    inputs = []
    for seed in seeds:
        task = "detailed_description"
        prompt = get_random_prompt(task)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image_url": {"url": seed["url"]}},
                {"type": "text", "text": prompt},
            ],
        }]

        # In production, use the transformers tokenizer to process messages.
        prompt_text = f"<|im_start|>user\n<|image_pad|>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": seed["url"]},
            "metadata": {"task": task, "url": seed["url"], "prompt": prompt},
        })

    print(f"Generating answers for {len(inputs)} seeds...")
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    results = []
    for output, req in zip(outputs, inputs):
        results.append({
            "url": req["metadata"]["url"],
            "task": req["metadata"]["task"],
            "instruction": req["metadata"]["prompt"],
            "response": output.outputs[0].text,
        })

    return results
```

### Step 4: LLM-as-Judge Quality Filtering

Generated responses often hallucinate. We introduce a strong judge model such as Qwen2.5-72B-Instruct. Because a text-only 72B model cannot directly inspect images, we use text-only evaluation: the judge scores the internal logic, completeness, and structure of the generated long response.

```python
# code/zh/project_13_mm_instruction_factory/llm_judge.py
def score_with_llm_judge(generated_data):
    """
    Demonstration logic. In a real pipeline this function calls a 72B judge model
    served by vLLM. Input is an instruction and response; output is a 1-5 score.
    """
    scored_data = []
    for item in generated_data:
        # Example production prompt:
        # Rate the quality of this response to the instruction. Score 1 to 5.
        word_count = len(item["response"].split())
        score = 4.5 if word_count > 50 else 3.0

        if score >= 4.0:
            item["judge_score"] = score
            scored_data.append(item)

    print(f"Filtered {len(generated_data)} down to {len(scored_data)} high-quality samples.")
    return scored_data
```

### Step 5: Unified Downstream Packaging

Whether the source is a single image, multiple images, or a video clip, the final output is written as JSONL in a community format such as ShareGPT or a model-specific format such as Qwen2.5-VL fine-tuning format.

```python
# code/zh/project_13_mm_instruction_factory/pack_multi_image_video.py
import json


def pack_to_qwen_format(scored_data, output_path="./data/mm_sft_final.jsonl"):
    formatted_dataset = []

    for item in scored_data:
        record = {
            "type": "image",
            "image": item["url"],
            "conversations": [
                {
                    "from": "user",
                    "value": f"<image>\n{item['instruction']}",
                },
                {
                    "from": "assistant",
                    "value": item["response"],
                },
            ],
        }
        formatted_dataset.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in formatted_dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(formatted_dataset)} samples to {output_path}")


if __name__ == "__main__":
    dummy_data = [{
        "url": "http://example.jpg",
        "instruction": "Describe",
        "response": "A cat.",
        "judge_score": 4.5,
    }]
    pack_to_qwen_format(dummy_data)
```

## Engineering Run Path and Minimal Reproduction

The P13 code directory is `code/zh/project_13_mm_instruction_factory`. Compared with P11 and P14, this project is more of a generative data factory. The minimal reproduction path is therefore not one fixed shell script, but a staged function chain: select seeds, generate instructions from templates, then run judge, self-consistency, multilingual expansion, and format packaging. In teaching environments, a small seed set and mock judge can first validate artifact contracts before replacing them with real Qwen2.5-VL and Qwen2.5-72B-Instruct services.

Listing P13-1 shows the minimal run order. A production implementation can wrap it in shell, Makefile, Airflow, or Ray, but the project chapter should make stage boundaries and artifact transfer explicit.

```python
from seed_selector import select_seeds
from generate_with_qwen_vl import generate_instructions
from llm_judge import score_with_llm_judge
from self_consistency import self_consistency_filter
from multilingual_expand import expand_multilingual
from pack_multi_image_video import pack_to_qwen_format

seeds = select_seeds(num_samples=100)
raw = generate_instructions(seeds)
consistent = self_consistency_filter(raw)
scored = score_with_llm_judge(consistent)
expanded = expand_multilingual(scored)
pack_to_qwen_format(expanded, "./data/mm_sft_final.jsonl")
```

This code describes the factory's minimal closed loop, but it is not yet a production script. Production runs need four additional controls. First, model calls must record model path, temperature, top-p, max tokens, and concurrency. Second, seeds must record source, authorization, and download status. Third, judge output must retain the scoring prompt, threshold, and human calibration set. Fourth, before packaging, the pipeline must check image links, conversation format, and sample deduplication.

| Category | Record item | Purpose |
| --- | --- | --- |
| Asset version | Image source, URL, authorization, download time | Proves sample traceability |
| Generation model | Qwen2.5-VL path, inference framework, sampling parameters | Explains output differences |
| Template version | Task type, template text, template hash | Controls task distribution |
| Judge version | Scoring model, scoring rubric, threshold | Reviews filtering results |
| Multilingual version | Translation model, terminology table, language ratio | Reviews cross-language consistency |
| Packaging version | Output format, field schema, target training framework | Ensures training scripts can read the data |
| Spot-check record | Human samples, failure samples, revision notes | Supports release gates |

*Table P13-3: Runtime records for the multimodal instruction factory.*

## Data Schema and Sample Contract

A minimal multimodal instruction record cannot contain only `image`, `instruction`, and `response`. The project chapter should emphasize that training formats can be narrow, but engineering intermediate states must be wider. Otherwise, once hallucination, format errors, or copyright problems appear, the data team cannot trace a sample back to its image, template, model call, or filtering step.

| Field | Example | Meaning |
| --- | --- | --- |
| `sample_id` | `p13_laion_000001` | Stable primary key across logs |
| `asset.url` | `https://...jpg` | Original image or video address |
| `asset.license` | `cc-by`, `internal` | Basis for release and redistribution judgment |
| `seed.original_caption` | Original alt text | Used to judge seed quality |
| `task.type` | `detailed_description`, `ocr_reading` | Controls task distribution |
| `prompt.template_id` | `ocr_v1_002` | Tracks template version |
| `generation.model` | `Qwen2.5-VL-7B-Instruct` | Tracks generation model |
| `generation.response` | Long response text | Candidate training sample |
| `quality.judge_score` | `4.5` | Basis for LLM-as-Judge filtering |
| `quality.consistency_score` | `1.0` | Multi-sample consistency |
| `language` | `en`, `zh` | Distinguishes multilingual samples |
| `audit_trace` | Batch, timestamp, script version | Supports review and takedown |

*Table P13-4: Intermediate-state sample schema for the multimodal instruction factory.*

The final `mm_sft_final.jsonl` file in Qwen format can be narrower than the intermediate state, but the intermediate state should not be discarded. Training files serve the training framework; audit files serve quality and release. They can be joined by `sample_id`.

## Quality Filtering: From Length Thresholds to Calibrated Rubrics

The demonstration `llm_judge.py` uses response length as a proxy: answers above a certain word count receive 4.5, while shorter answers receive 3.0. This is acceptable for teaching, but not for a real release gate. A real LLM-as-Judge setup should include at least four scoring dimensions: image-text consistency, answer completeness, task following, and safety/compliance.

| Scoring dimension | 5-point behavior | Low-score risk |
| --- | --- | --- |
| Image-text consistency | Describes only content supported by visual evidence | Hallucinates subjects, actions, or text |
| Task following | Strictly follows the template requirement, such as OCR table output | Off-task answer or invalid format |
| Detail completeness | Covers subjects, spatial relations, text, and background | Too short, generic, missing trainable information |
| Reasoning reliability | Reasoning steps are supported by visual evidence | Over-infers causality or intention |
| Safety and compliance | Avoids sensitive identity inference and improper content | Privacy, bias, or dangerous guidance |
| Language quality | Clear expression without severe repetition | Mechanical repetition, garbling, or abnormal language mixing |

*Table P13-5: LLM-as-Judge scoring rubric for multimodal instruction samples.*

Self-consistency complements judge blind spots. For complex reasoning questions, the model can generate multiple answers, then compare whether conclusions and key evidence agree. If different samples conflict on subjects, text, or spatial relationships, the record should not enter the training set even if one answer is long and fluent. The current `self_consistency.py` is a simplified interface and teaching implementation; real projects should plug in multi-sample generation and consistency metrics.

## Multilingual Expansion and Cross-Language Acceptance

Multilingual expansion is not simply copying an English instruction and adding an `instruction_zh` field. In multimodal tasks, cross-language errors often occur in visual references and proper-name translation. For example, "the sign on the left" may be translated as "the sign on the right," or brands, place names, and units may be localized incorrectly. P13 should treat Chinese and English as two sample sets that both require spot checks, not as a cheap way to double the count.

| Acceptance item | Check method | Common issue |
| --- | --- | --- |
| Reference consistency | Compare image against left/right, top/bottom, foreground/background | Direction words mistranslated |
| Terminology consistency | Check OCR, chart, bbox, caption against the glossary | Terminology changes across samples |
| Proper-name preservation | Check brands, places, people, and units | Over-translation or mistranslation |
| Format preservation | Check tables, lists, JSON, Markdown | Translation breaks structure |
| Safety boundary | Check whether sensitive content bypasses filtering in another language | English filtering works but Chinese filtering fails |

*Table P13-6: Multilingual expansion acceptance items.*

If the project targets Chinese-model training, do not only translate English samples into Chinese. Keep a portion of native Chinese templates and native Chinese judge prompts. Translated samples are useful for scale, but native Chinese samples better reflect real Chinese user questions.

## Test Coverage and Code Notes

`tests/test_factory.py` covers template existence, random prompt return type, judge filtering, Chinese expansion, and JSONL packaging. These tests prevent basic interface breakage, but they do not prove the factory is releasable. In particular, `generate_with_qwen_vl.py` is a teaching example. Before real vLLM or Qwen-VL integration, it needs input variables, exception handling, model-call result parsing, and failed-sample records. The chapter presents it to explain the generation-stage interface, not to claim production completeness.

| Test item | Covered | Still needed |
| --- | --- | --- |
| Template test | Three template types exist; prompt returns a string | Template repetition rate, task ratio |
| Judge test | Short responses filtered; long responses retained | Real judge rubric and human agreement |
| Multilingual test | Chinese expansion field generated | Semantic consistency and format preservation |
| Packaging test | JSONL file can be written | Conversation-field spot check |
| End-to-end mock | Test entry exists | Small-sample real model run |

*Table P13-7: Test coverage and acceptance gaps for the multimodal instruction factory.*

## Common Faults and Troubleshooting Paths

Seed-stage issues usually involve dead image links, missing image-size fields, or low-quality original captions. First count filtering reasons rather than only final seed count. If many images are removed by aspect-ratio filters, confirm field units and source schema.

Generation-stage issues include incorrect model input format, image-download failure, empty VLM output, and OOM or timeout from excessive concurrency. With vLLM, record concurrency, GPU memory, and failed requests. With APIs, record retry count, error codes, and billing units.

Filtering-stage issues often come from a judge that over-rewards long answers. Long answers are not necessarily high quality; in multimodal settings they may contain more hallucinations. Review high-score, low-score, and threshold-near samples separately, and periodically calibrate the judge with human labels.

Packaging-stage issues involve mismatch among image URLs, `<image>` markers, and the target training framework's conversation format. Randomly read JSONL lines and confirm that each line is valid JSON, `conversations[0].value` contains an image placeholder, assistant output is non-empty, and quality fields link back to intermediate records.

## Manual Spot Checks and Release Gates

Multimodal instruction factories often look good on automated metrics while failing under human reading. High-scoring samples may be fluent but unfaithful to the image; multilingual samples may be grammatical but wrong on direction, count, or OCR text. Manual spot checks are therefore mandatory before release.

| Review layer | Sample source | Review focus |
| --- | --- | --- |
| High-score samples | Highest judge-score batch | Whether the judge over-rewards long text |
| Boundary samples | Samples close to threshold | Whether the threshold is too strict or too loose |
| Low-score samples | Filtered samples | Whether valuable samples were wrongly removed |
| OCR samples | `ocr_reading` tasks | Text accuracy and format preservation |
| Reasoning samples | `complex_reasoning` tasks | Whether reasoning has visual evidence |
| Chinese samples | Multilingual expansion results | Terminology, direction, proper names |
| Multi-image / video samples | Packer extensions | Reference order and placeholders |

*Table P13-8: Manual review strata for the multimodal instruction factory.*

Manual review should use dual review plus arbitration. The first reviewer checks image-text consistency and task following. The second checks language quality and safety boundaries. Conflicts enter an arbitration pool, which is then used to revise judge prompts, templates, and thresholds. Manual review is not a one-time quality check; it is part of factory iteration.

Release gates should include at least four checks. First, sample sources must be traceable, and external images must not be represented only by naked URLs. Second, training files must be readable by the target framework, not merely valid JSON. Third, there must be an agreement report between judge and human review. Fourth, if multilingual samples are released, Chinese and English quality must be reported separately.

| Gate | Required evidence | Action on failure |
| --- | --- | --- |
| Source gate | URL, license, download status, deletion-request handling | Remove unauthorized or untraceable samples |
| Format gate | JSONL validation, small-sample training loader read | Fix packer or field schema |
| Quality gate | Judge distribution, human-review pass rate, failure types | Adjust templates, thresholds, or generation parameters |
| Multilingual gate | Separate Chinese and English review reports | Roll back low-quality translation batches |
| Safety gate | Sensitive content, privacy, identity-inference checks | Delete samples and update filtering rules |
| Version gate | Model version, template version, run batch | Freeze versions before release |

*Table P13-9: Release-gate checklist for the multimodal instruction factory.*

## Multi-Image and Video Extension Path

The presence of `pack_multi_image_video.py` indicates that this project targets more than single-image SFT. Modern VLM training increasingly depends on interleaved images, multi-image comparison, and short video clips. The core issue is not concatenating several `<image>` tags, but making the instruction clearly point to each visual input and making the answer explicitly express comparison, ordering, temporal change, or cross-image relation.

| Type | Input organization | Instruction focus | Common error |
| --- | --- | --- | --- |
| Single image | One `<image>` | Description, OCR, local reasoning | Hallucinated object or text |
| Multi-image comparison | `<image_1>`, `<image_2>` | Difference, similarity, ordering, change | Image order confused |
| Interleaved text-image | Text paragraphs with multiple images | Refer to images through context | Wrong image reference or missing context |
| Short video | Multiple frames or `<video>` | Action, temporal order, camera movement | Describes video as static image |
| Chart screenshot | Image plus OCR/table structure | Numeric reading, trend explanation | Fabricated value or axis |

*Table P13-10: Comparison of multimodal instruction types.*

For video, reuse P14's shot-level structure: `frame_paths`, `caption_en`, `shot_language`, and `camera_motion` can become video-instruction material for P13. A video QA sample can ask the model to explain how the subject moves or infer camera movement. P13 and P14 are therefore upstream and downstream: P13 is the instruction factory, while P14 is the video data pipeline.

## Deliverable Directory and Version Management

P13 deliverables should be separated into raw, scored, expanded, packed, and reports. This avoids mixing training files with audit files and makes stage-level rollback possible.

| Path | Content | Note |
| --- | --- | --- |
| `data/seeds.jsonl` | Seed asset list | URL, authorization, original caption, filtering reason |
| `data/generated_raw.jsonl` | Raw VLM generations | Not used directly for training; used for review |
| `data/scored.jsonl` | Judge-filtered records | Score, rubric, model version |
| `data/consistent.jsonl` | Self-consistency-filtered records | Multi-sample consistency evidence |
| `data/multilingual.jsonl` | Multilingual expansion samples | Language, glossary version, translation model |
| `data/mm_sft_final.jsonl` | Training input file | Targeting Qwen-VL or another training framework |
| `reports/task_distribution.json` | Task-distribution report | Checks task imbalance |
| `reports/human_review.md` | Manual review report | Core release-gate evidence |
| `reports/license_audit.md` | Copyright and source audit | Required for public release |

*Table P13-11: Deliverable directory for the multimodal instruction factory.*

For version management, hash templates and judge prompts. A model version can remain fixed while template text changes enough to shift sample distribution. A small judge-prompt change can also move pass rates. Release reports should include model version, template version, judge-prompt version, and data batch, not just "generated with Qwen2.5-VL."

## Data Dashboard and Continuous Iteration

After launch, the factory must continue observing sample distribution instead of generating once and sending data directly to training. The dashboard can start as JSONL statistics scripts; it does not need to be a complex platform. Each batch should answer: whether task types are balanced, whether judge pass rate is abnormal, whether multilingual ratio is stable, and which asset types dominate failure samples.

| Dashboard metric | Object | Purpose |
| --- | --- | --- |
| Seed pass rate | `seeds.jsonl` | Judge whether asset-selection thresholds are too strict |
| Task-type distribution | `task.type` | Prevent over-production of detailed-description samples |
| Average response length | Raw/scored samples | Detect templated short answers or verbose hallucination |
| Judge-score distribution | `quality.judge_score` | Observe model and rubric drift |
| Consistency distribution | `quality.consistency_score` | Detect unstable reasoning tasks |
| Chinese-English ratio | `language` | Control multilingual expansion scale |
| Format error rate | JSONL validation result | Detect packer or template problems |
| Manual-review pass rate | Review report | Judge release-gate readiness |
| Safety-interception rate | Safety filter | Monitor sensitive content and privacy risk |

*Table P13-12: Dashboard metrics for the multimodal instruction factory.*

Dashboards must be stored by batch. A sudden judge-pass-rate increase does not necessarily mean quality improved; the judge prompt may have become looser, templates may have become longer, or the model may have learned to produce verbose answers. If OCR-task pass rate is much lower than description-task pass rate, inspect OCR image quality and the scoring rubric separately instead of raising the global threshold.

## Sample Takedown and Copyright Response

Multimodal data triggers copyright, portrait-right, and privacy risks more easily than pure text. A public URL does not imply unlimited redistribution of images or generated results. P13 must keep a takedown path: when an image, author, or source collection must be deleted, the system should locate related instructions, translated samples, and final training files.

| Step | Operation | Affected artifact |
| --- | --- | --- |
| Register request | Record URL, author, source, request time, evidence | Ticket |
| Locate asset | Search seed by URL, hash, source, or sample ID | `seeds.jsonl` |
| Locate derived samples | Search raw, scored, multilingual, and packed records | All intermediate states |
| Delete training samples | Remove corresponding lines from `mm_sft_final.jsonl` | Training file |
| Recompute statistics | Update task distribution, language ratio, quality report | Reports |
| Publish note | Record deletion reason and new version | Release note |

*Table P13-13: Takedown path for multimodal instruction samples.*

The takedown mechanism requires stable `sample_id` in the intermediate state. If only the final Qwen conversation format is saved, it is hard to trace a training sample back to the original image and generation batch. P13 must therefore distinguish the narrow training table from the wide audit table.

## Domain Transfer: From General Images to Industry Assets

P13 can transfer to medical imaging, industrial inspection, e-commerce product images, legal-evidence screenshots, and educational charts. But templates and gates must be redesigned for each domain. Visual evidence and risk boundaries differ too much to reuse generic LAION templates directly.

| Domain | Asset type | Template adjustment | Risk control |
| --- | --- | --- | --- |
| Medical | Images, report screenshots | Describe abnormal regions; avoid diagnostic conclusions | Expert review, privacy masking |
| Industrial | Defect images, equipment photos | Describe defect location, morphology, severity | Internal confidentiality, misjudgment cost |
| E-commerce | Product images, detail-page screenshots | Attribute extraction, comparison, OCR reading | Brand authorization, exaggerated descriptions |
| Finance | Report screenshots, charts | Table reading, trend explanation, evidence citation | Numeric accuracy, investment-advice boundary |
| Education | Problem figures, board writing, textbook illustrations | Solving hints, chart understanding | Copyright, answer leakage |

*Table P13-14: Domain-transfer adjustments for the multimodal instruction factory.*

For domain transfer, build a small set of high-quality templates and expert-review samples before scaling. In high-risk domains, do not rely entirely on LLM-as-Judge. The judge can pre-filter, but release gates should be decided jointly by domain experts or rule systems.

## Relationship with P03 and P14

P13 sits between P03 and P14. P03 establishes the classic LLaVA image-text and conversation baseline. P13 adds Qwen-VL-style generation, judge, self-consistency, and multilingual expansion. P14 extends visual input from static images to video shots. Together they form a progression from single-image baseline to modern multimodal factory and then to video-generation data.

| Project | Core object | Key capability | Boundary not to confuse |
| --- | --- | --- | --- |
| P03 | LLaVA image-text pairs and conversation | Classic flow, OCR, bbox, visual spot checks | Does not emphasize newer Qwen-VL factory capability |
| P13 | Multimodal instruction samples | Templates, VLM generation, judge, multilingual packaging | Does not handle video cutting and T2V quality filtering |
| P14 | Video shot data | Shot segmentation, motion, aesthetics, caption, shot language | Does not handle large-scale instruction diversification |

*Table P13-15: Project boundaries among P03, P13, and P14.*

With this organization, readers can treat P03 as the baseline data structure, P13 as the instruction-generation factory, and P14 as the video-material and temporal-supervision source. A future Video-QA or Video-Instruct dataset can first use P14 to create video segments and shot fields, then use P13 templates, judge, and packaging to produce instruction samples.

## Results and Analysis

The example acceptance setting deploys Qwen2.5-VL-7B with vLLM on one node with four 4090 GPUs and calls a 72B model as judge through an API, producing a candidate batch of multimodal instruction samples. In formal reproduction, replace the example scale with actual task configuration, generation logs, and sample manifests.

- **Task distribution**: Detailed description (40%), complex reasoning (30%), OCR and tables (20%), and fine-grained grounding (10%). No single category exceeded 40%.
- **Quality distribution**: Samples passing LLM-as-Judge filtering should record mean, quantiles, and rejection reasons. An example acceptance report may show an average score such as **4.3 / 5.0**, but formal results must retain scoring details and judge version.

Formal acceptance should check four kinds of evidence. First, the data can be read by downstream training scripts. Second, the task-type distribution matches the planned ratio. Third, image, instruction, and answer are not obviously mismatched. Fourth, source license, model license, and redistribution rules for generated artifacts are registered. Only after these checks can generated data move from the candidate pool into the training set.

## Cost and Optimization

The industrial synthesis factory has the following cost profile:

- **Synthesis cost**: On private compute, a 7B VLM takes about 1-2 seconds to generate one long image-conditioned response. With commercial APIs, the cost is about $5-$10 per thousand high-quality samples.
- **Scalability**: vLLM tensor parallelism handles multimodal generation pressure well. When compute is limited, reduce `max_num_seqs` and lower the sampling temperature to prevent low-value divergence.

## Extensions

Compared with earlier LLaVA-style data pipelines that relied heavily on manual work or expensive GPT-4V distillation, the Qwen-VL plus LLM-as-Judge self-distillation pipeline substantially lowers fine-tuning cost.

Video clips can be inserted into the same pipeline by changing the packer: sampled frames can be represented as multiple `<image>` tags or one `<video>` field, enabling data synthesis for T2V or Video-QA models.

### Data Compliance and Open-source Licensing

When building and publishing instruction datasets, observe these constraints:

- **LAION seed images**: Original image links may be governed by CC-BY or other public licenses and should be used for research under the corresponding terms.
- **Qwen2.5-VL**: Model use and redistribution of generated content are governed by the model's open-source or commercial license.
- **Generated artifacts**: A dataset such as `dataforge-mm-instruction-50k` can be released under CC-BY-SA when the upstream licenses allow it.

## Chapter Summary

This chapter used the multimodal instruction factory as a project case to show how to build a multimodal instruction production chain covering image, text, OCR, chart, and dialogue tasks. Its main value is putting task definition, data boundaries, architecture decisions, sample schema, acceptance metrics, and reproduction resources into one chain, so the project is not merely a sequence of operations but a reviewable case study.

The boundary of the case must also remain explicit. It targets controlled assets and sample factories; it does not cover unauthorized media collection or fully automated safety review. In larger-scale, higher-risk, or more strictly regulated settings, data sources, permission status, human-review ratio, runtime cost, and rollback plans must be reassessed.

As part of Part 14, this chapter validates methods from earlier chapters at the project layer. Readers can combine this case with Part 13's data recipes, the platform-governance chapters, and the appendix checklists to form a closed loop from method understanding to engineering delivery.

## References

Bai S, Chen K, Liu X, Wang J, Ge W, Song S, Dang K, Wang P, Wang S, Tang J, others (2025) Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923.

Zhu J, Wang W, Chen Z, Liu Z, Ye S, Gu L, Duan Y, Tian H, Su W, Shao J, others (2025) InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models. arXiv preprint arXiv:2504.10479.

Kwon W, Li Z, Zhuang S, Sheng Y, Zheng L, Yu C H, Gonzalez J E, Zhang H, Stoica I (2023) Efficient Memory Management for Large Language Model Serving with PagedAttention. In: Proceedings of the 29th ACM Symposium on Operating Systems Principles, pp 611-626.

Schuhmann C, Beaumont R, Vencu R, Gordon C, Wightman R, Cherti M, Coombes T, Katta A, Mullis C, Wortsman M, others (2022) LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models. In: Advances in Neural Information Processing Systems 35:25278-25294.

Wang X, Wei J, Schuurmans D, Le Q, Chi E, Narang S, Chowdhery A, Zhou D (2023) Self-Consistency Improves Chain of Thought Reasoning in Language Models. In: International Conference on Learning Representations.

Zheng L, Chiang W L, Sheng Y, Zhuang S, Wu Z, Zhuang Y, Lin Z, Li Z, Li D, Xing E P, Zhang H, Gonzalez J E, Stoica I (2023) Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In: Advances in Neural Information Processing Systems 36.
