# Project 3: LLaVA Multimodal Instruction Data Factory

## Abstract
P03 focuses on building a multimodal instruction data factory for LLaVA-style training.

The project does not only pair images with captions.

It organizes visual assets, document images, chart images, OCR tasks, grounding coordinates, multi-image comparison, conversation templates, quality control, visualization checks, and training packaging into one data pipeline.

The project can be read through four lines.

- Multimodal asset planning: define image, document, chart, and multi-image sources.
- Supervision construction: create captioning, OCR, chart reading, grounding, and comparison tasks.
- QA and visual verification: check schema, semantics, bounding boxes, and low-quality samples.
- Training delivery: package LLaVA-style conversations into train, validation, smoke, and manifest artifacts.

The engineering chain is:

```text
visual assets -> multimodal seed schema -> task generation -> OCR/chart/grounding/multi-image samples -> LLaVA conversation format -> QA and visual replay -> training package
```

The core goal is to turn multimodal asset handling into a reviewable instruction-data factory.

## Keywords

LLaVA; multimodal instruction data; OCR; grounding; chart reading; multi-image comparison

## Project Goals and Reader Takeaways

This project uses a LLaVA-style multimodal instruction factory as the core case.

After completing the chapter, readers should understand how to structure multimodal assets, build task types, normalize coordinates, construct conversation records, run QA, and package multimodal training data.

## Scenario Constraints and Data Boundaries

The project uses controlled visual assets and generated instructions.

It does not claim to cover all vision-language tasks or all document understanding scenarios.

It does not replace large-scale multimodal pre-training.

The goal is a compact, auditable data factory.

## Architecture Decision

The project uses a three-layer architecture: asset processing, supervision construction, and QA/delivery.

This keeps image files, task records, coordinate metadata, QA results, and training records aligned.

## Sample Schema and Data Flow

The minimal sample should retain `sample_id`, `image_path`, `asset_type`, `task_type`, `question`, `answer`, `bbox`, `ocr_text`, `quality_signals`, `split`, and `audit_trace`.

```text
image asset -> seed record -> task record -> conversation sample -> QA result -> training record
```

## Core Implementation Fragments

The chapter keeps fragments for bbox normalization, multi-image payloads, and LLaVA conversation format.

Full generated samples and large image assets should live in companion resources.

## Experimental and Acceptance Metrics

Acceptance metrics include asset count, task distribution, bbox validity, OCR coverage, chart-task coverage, train/validation/smoke split consistency, QA pass rate, and visualization-check coverage.

The project report highlights the structural path from `87` assets to `267` training records.

## Cost, Risk, and Compliance Boundaries

Costs include asset preparation, image rendering, task generation, manual visual review, and multimodal model calls.

Risks include wrong image-text alignment, bad OCR targets, chart hallucination, bbox coordinate errors, overly easy samples, and hidden copyright issues in image sources.

## Common Failure Modes

Common failures include mismatched images and answers, OCR copied without task transformation, chart values read from the wrong axis, bboxes shifted by resize operations, multi-image order confusion, and low-quality samples that pass schema checks.

## Reproducible Resource Notes

Reproduction materials should include asset manifests, task configs, bbox conversion scripts, conversation templates, split manifests, QA reports, visualization artifacts, and check scripts.

## 1. Project Background: Why a Multimodal Instruction Data Factory Is Needed

General image-caption pairs are not enough for a capable vision-language assistant.

A useful multimodal model must describe images, read documents, understand charts, locate regions, compare multiple images, and answer questions with visual evidence.

Those abilities require different task structures.

P03 therefore treats multimodal instruction data as a factory rather than as a list of image captions.

The factory must preserve image assets, task intent, coordinate systems, conversation format, QA evidence, and training delivery.

## 2. Project Goals and Boundaries

### 2.1 Project Goals

The first goal is to build a structured multimodal asset pool.

The second goal is to generate representative tasks for captions, OCR, chart reading, grounding, and multi-image comparison.

The third goal is to package these tasks into LLaVA-style conversation records.

The fourth goal is to run QA, visualization checks, and training-interface validation.

### 2.2 Project Boundaries

The project does not cover full VLM pre-training.

It does not cover all object detection or segmentation tasks.

It does not guarantee production-grade document OCR.

It does not claim that all generated samples are high quality without review.

### 2.3 Role of Boundary Statements

Boundaries keep the case honest.

This project demonstrates a multimodal instruction data factory, not a complete multimodal model lifecycle.

## 3. Project Position: P03 in the Capability Chain

P03 sits after text-domain SFT and before larger multimodal recipes.

It shows how the same factory principles extend to images, documents, charts, and spatial grounding.

Its value is that it turns visual assets into structured supervision.

## 4. Overall Architecture: From Multimodal Assets to Training Assets

![Figure 1: LLaVA Multimodal Instruction Data Factory Overview](../../images/part10/10_3_fig01_llava_factory_overview.png)

### 4.1 Layer 1: Asset Processing

This layer prepares image assets, document renderings, chart images, and multi-image groups.

It keeps paths, metadata, dimensions, and asset types stable.

### 4.2 Layer 2: Supervision Construction

This layer creates caption, OCR, chart, grounding, and comparison tasks.

It converts assets into learnable instruction examples.

### 4.3 Layer 3: QA and Delivery

This layer checks schemas, visual grounding, semantic quality, low-quality cases, splits, manifests, and training records.

## 5. Engineering Prerequisites: Key Surfaces of a Multimodal Data Factory

### 5.1 Asset Planning and Sampling Strategy

The project must decide which visual domains and task types to cover.

Without planning, the dataset becomes an unbalanced image pile.

### 5.2 Data Processing and Interface Maintenance

Image paths, dimensions, IDs, bbox fields, and task labels must remain consistent.

### 5.3 Task Generation and Template Orchestration

Task templates should be explicit so captions, OCR, charts, grounding, and multi-image questions do not collapse into one generic QA format.

### 5.4 QA, Rollback, and Version Control

Visual data needs rollback because wrong image alignment can silently damage many samples.

### 5.5 Role of Responsibility Surfaces

These surfaces make multimodal data reviewable by data engineers, model engineers, annotators, and product reviewers.

![Figure 2: Multimodal Data Factory Responsibility Collaboration](../../images/part10/10_3_fig02_roles_and_responsibilities.png)

## 6. Asset-layer Design: Building the Multimodal Asset Pool

The asset pool should contain natural images, document images, chart images, and multi-image comparison groups.

### 6.1 Why Split Assets into Three Classes

Natural images, document screenshots, and charts require different questions and quality checks.

They should not be treated as one image category.

### 6.2 Engineering Meaning of Asset Balance

Balanced assets prevent the model from learning only easy captions.

They also expose failures in OCR, layout, and chart reasoning.

### 6.3 Why Derived Document and Chart Images Matter

Documents and charts connect the project to real business use cases.

They move multimodal instruction data beyond everyday image description.

![Figure 3: Multimodal Asset Layers](../../images/part10/10_3_fig03_asset_layers.png)

## 7. Data Schema: Structuring Multimodal Seeds

### 7.1 Why Schema Matters in Multimodal Scenarios

The model sees images, but the pipeline must track files, dimensions, prompts, answers, regions, and task types.

Schema is the shared contract.

### 7.2 A More Stable Minimal Schema

A stable schema should include ID, image path, image size, asset type, task type, prompt, answer, optional OCR text, optional bbox, split, and quality signals.

### 7.3 Engineering Value of Schema

Schema enables QA, visualization, training packaging, and failure attribution.

Without it, visual errors are hard to trace.

## 8. Image Sampling and Re-description

### 8.1 What Re-description Solves

Raw captions are often too shallow.

Re-description can convert an asset into detailed instruction-answer pairs.

### 8.2 Role of Template Generation

Templates keep task style consistent.

They also prevent all samples from becoming generic "describe the image" prompts.

### 8.3 Task-oriented Rewriting

Rewriting should target captioning, localization, document understanding, or comparison according to asset type.

## 9. Document Images and OCR Tasks

### 9.1 Position of OCR Tasks

OCR tasks teach the model to read visual text and answer questions grounded in that text.

### 9.2 Why OCR Results Cannot Be Pasted Directly into Training

Raw OCR text is not an instruction sample.

It must be transformed into questions, answers, and reviewable evidence.

### 9.3 Why Document Images Are Key to Real Business Use

Many enterprise multimodal tasks involve forms, receipts, reports, screenshots, or documents.

Document images are therefore a bridge to practical applications.

![Figure 4: Document-image Task Layers](../../images/part10/10_3_fig04_document_tasks.png)

## 10. Chart Image Tasks

### 10.1 Why Chart Tasks Need Their Own Class

Charts require axis reading, legend matching, trend comparison, and value extraction.

These are not ordinary caption tasks.

### 10.2 Chart Task Decomposition

The project should separate trend questions, value questions, comparison questions, and chart-summary tasks.

### 10.3 Why Chart Samples Suit Failure Attribution

Chart errors can often be traced to wrong axis, wrong series, wrong legend, or wrong trend.

That makes them useful for QA.

## 11. Region Localization and Coordinate Alignment

### 11.1 Why Input Coordinates and Training Coordinates Differ

Images may be resized, padded, cropped, or normalized before training.

Raw annotation coordinates may not match model input coordinates.

### 11.2 Why Clamp Is Needed

Clamping prevents invalid boxes from exceeding image boundaries after conversion.

### 11.3 Why Grounding Samples Should Not Be Generated Without Limit

Grounding quality depends on coordinate accuracy.

Large numbers of weak boxes can harm training.

### 11.4 Coordinate Alignment Implementation

```python
def normalize_bbox(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    return [x1 / width, y1 / height, x2 / width, y2 / height]
```

### 11.5 Real Engineering Meaning

Coordinate alignment is not a formatting detail.

It determines whether grounding supervision points to the right visual region.

![Figure 5: Bbox Coordinate Conversion and Normalization](../../images/part10/10_3_fig05_bbox_alignment.png)

## 12. Multi-image Interleaved Samples

### 12.1 Value of Multi-image Tasks

Multi-image tasks teach comparison, ordering, consistency checking, and cross-image reasoning.

### 12.2 Why Payload Construction Is Hard

The training payload must preserve image order and question references.

If the order changes, the answer can become wrong.

### 12.3 Multi-image Interleaving Implementation

```python
sample = {
    "images": ["image_a.png", "image_b.png"],
    "conversations": [
        {"from": "human", "value": "<image>\n<image>\nCompare the two charts."},
        {"from": "gpt", "value": "The first chart shows..., while the second chart shows..."},
    ],
}
```

### 12.4 Why Interleaved Sample Count Is Usually Limited

Multi-image samples are expensive to review.

They should be high-value and carefully checked.

## 13. LLaVA Conversation Template

### 13.1 What the Conversation Template Solves

The template maps multimodal tasks into the format expected by LLaVA-style training.

It standardizes roles, image tokens, questions, and answers.

### 13.2 Why Template Count Should Be Controlled

Too many templates create style fragmentation.

Controlled variety is better than uncontrolled prompt diversity.

### 13.3 LLaVA Format Example

```json
{
  "id": "sample_001",
  "image": "assets/doc_001.png",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is the invoice total?"},
    {"from": "gpt", "value": "The invoice total is $1,250."}
  ]
}
```

## 14. Quality Control: Structure of Multimodal QA

### 14.1 Structure Consistency Checks

Check image paths, required fields, role format, bbox validity, and split membership.

### 14.2 Semantic Quality Checks

Review whether the answer matches the image, whether OCR claims are correct, and whether chart conclusions are supported.

### 14.3 Visual Back-checks

Render bboxes and image references so reviewers can inspect alignment.

### 14.4 Why Maintain a Low-quality Sample Library

Low-quality samples teach the team where generation fails.

They support regression testing and prompt repair.

![Figure 6: Sample QA and Rollback Loop](../../images/part10/10_3_fig06_quality_loop.png)

## 15. Visual Verification: Bbox Reverse Rendering

### 15.1 What Reverse Rendering Solves

Reverse rendering shows whether a stored bbox points to the intended region.

### 15.2 Typical Errors

Typical errors include coordinate scaling mistakes, swapped axes, crop offsets, and boxes that include the wrong object.

### 15.3 Engineering Value

Visual verification turns hidden coordinate errors into visible artifacts.

## 16. Training Packaging

### 16.1 Train / Validation / Smoke Delivery

The project should export train, validation, and smoke splits.

Smoke records should cover image, document, chart, grounding, and multi-image formats.

### 16.2 Why the Manifest Matters

The manifest records counts, asset types, task distribution, file paths, and versions.

It lets training users verify what they received.

### 16.3 What Training Packaging Really Does

Packaging converts heterogeneous multimodal artifacts into one stable training interface.

## 17. Project Metrics

### 17.1 Why "87 Assets to 267 Training Records" Matters

The number shows that assets can produce multiple instruction records.

It indicates factory transformation rather than raw asset count.

### 17.2 What Asset-type Distribution Shows

Distribution across natural images, document images, charts, grounding samples, and multi-image tasks shows whether the factory is balanced.

### 17.3 Why 100% Pass Rate Should Not Be Over-read

A pass rate can mean schema consistency.

It does not prove semantic perfection.

### 17.4 Why 11/11 PASS Is Important

Project checks show that code, artifacts, and reports are aligned.

This is engineering completeness.

## 18. Cost Analysis

### 18.1 Why Manual Review Cost Must Be Seen Separately

Visual review is expensive and cannot be inferred from sample count.

### 18.2 Why Caption Cost Is Not Total Cost

The total cost includes asset preparation, OCR checks, bbox review, chart review, packaging, and failure replay.

### 18.3 What to Optimize First

Optimize asset quality, task template stability, and QA tools before simply increasing generation volume.

## 19. Failure Samples and Limitations

### 19.1 Most Obvious Current Limitations

The project is small, task types are limited, and semantic QA is still expensive.

### 19.2 Typical Failure Categories

Failures include image-answer mismatch, OCR misread, chart trend error, bbox misalignment, multi-image order confusion, and generic answers.

![Figure 7: Failure Attribution for Multimodal Samples](../../images/part10/10_3_fig07_failure_attribution.png)

### 19.3 Why Failure Attribution Must Be Fine-grained

Different error types require different fixes.

Prompt changes cannot fix all coordinate or asset problems.

## 20. Project Checks: Consistency Validation Loop

### 20.1 Why Project Checks Are Needed

Checks verify schema, files, splits, manifest counts, bbox validity, and report consistency.

### 20.2 What the Checks Cover

They should cover image existence, conversation schema, task labels, split disjointness, visualization artifacts, and training files.

### 20.3 Why This Shows Engineering Completeness

The data factory is complete only when its artifacts are checkable.

![Figure 8: Project Validation Loop](../../images/part10/10_3_fig08_validation_loop.png)

## 21. Relationship with Project 2

### 21.1 Shared Point 1: Both Emphasize the Seed Layer

Project 2 uses legal text seeds.

Project 3 uses visual asset seeds.

### 21.2 Shared Point 2: Both Emphasize Task Decomposition

Task categories make generation and QA reviewable.

### 21.3 Shared Point 3: Both Emphasize QA First

Review and checks must be designed before scale-up.

### 21.4 Shared Point 4: Both Emphasize Training Delivery

Both projects end with stable train/validation/smoke artifacts and manifests.

## 22. Future Extensions

### 22.1 From Single Images to Multi-page Documents

Future versions can include multi-page PDF reasoning and cross-page references.

### 22.2 From Static Charts to Complex Structural Diagrams

More complex charts and diagrams require stronger visual reasoning tasks.

### 22.3 From Multi-image Comparison to Agent Inputs

Multi-image payloads can become the basis for multimodal agent tasks.

### 22.4 From Controlled QA to Semi-automatic Review Panels

Review tools can render images, bboxes, answer evidence, and failure labels in one interface.

## 23. Main Deliverables

### 23.1 Intermediate Data Artifacts

- Asset manifest.
- Multimodal seed records.
- OCR task records.
- Chart task records.
- Grounding records.
- Multi-image records.

### 23.2 Training Data Artifacts

- LLaVA-style train split.
- Validation split.
- Smoke split.
- Training manifest.

### 23.3 Report and Check Artifacts

- QA report.
- Visualization check outputs.
- Metrics report.
- Project check report.

## 24. Closing

The value of P03 is not that the model can see images.

Its value is that the system can use images as structured training evidence.

Images, coordinates, document text, chart values, multi-image order, conversation format, and QA results all become data engineering objects.

## Special Topic: Spot Checks and Error Replay for Multimodal Annotation

### 1. Spot Checks Should Prioritize Whether Relationships Are Correct

Review should not only ask whether the answer sounds fluent.

It should ask whether the visual relationship is correct.

### 2. Error Replay Should Become a Fixed Asset

Repeated visual failures should enter a replay set.

The replay set helps future versions avoid regressions.

## Chapter Summary

This chapter used a LLaVA-style multimodal instruction data factory to show how visual assets become trainable instruction data.

The project connects asset planning, OCR tasks, chart tasks, grounding, multi-image comparison, conversation templates, QA, visualization checks, and training packaging.

Its boundary should remain clear: it is a structured data-engineering prototype, not a complete VLM training recipe.

## Release Review Notes

The first release review item is asset integrity.

Every image path should resolve.

Every image should have stable dimensions.

Every record should state asset type.

Natural images, document images, charts, grounding examples, and multi-image examples should be counted separately.

The second item is image-question alignment.

Reviewers should open representative samples and confirm that the question refers to the displayed image.

This catches the most damaging multimodal failure: a fluent answer attached to the wrong visual asset.

The third item is OCR transformation.

Raw OCR text should not be copied directly into training as if it were an answer.

It should be converted into a task that asks the model to read, extract, compare, or verify visual text.

The fourth item is chart reasoning.

Chart samples should be checked for axis, legend, series, unit, and trend.

The review should separate value-extraction errors from trend-description errors.

The fifth item is bbox validity.

Every bbox should be inside image bounds after normalization.

The release should include reverse-rendered examples.

A valid numeric box is not enough if it points to the wrong region.

The sixth item is multi-image ordering.

Multi-image records should preserve image order from prompt construction through training export.

If order is unstable, comparison answers become unreliable.

The seventh item is conversation schema.

LLaVA-style records should use consistent roles, image tokens, and answer fields.

The smoke set should contain at least one example for each major task type.

The eighth item is low-quality sample handling.

The project should keep a low-quality ledger.

The ledger should classify failures by image mismatch, OCR error, chart error, bbox error, prompt ambiguity, or generic answer.

The ninth item is train/validation/smoke split integrity.

Splits should avoid putting near-identical image-task pairs into both train and validation.

The tenth item is manifest consistency.

Counts by asset type, task type, split, and QA status should match actual files.

## Operating Notes

Daily operation should begin with asset health.

Missing image files should block the run.

Dimension changes should trigger bbox revalidation.

New document-rendering settings should trigger OCR and visual checks again.

New chart-rendering settings should trigger chart QA again.

If OCR errors rise, inspect source resolution and text extraction before changing the prompt.

If bbox errors rise, inspect coordinate conversion and image resizing.

If chart errors rise, inspect whether task labels distinguish value, trend, comparison, and summary.

If multi-image errors rise, inspect image order and prompt references.

If generic answers rise, inspect task templates and answer rubrics.

The project should keep visual replay cases.

Replay cases are especially important for multimodal data because failures are often invisible in JSON.

A replay viewer should show the image, prompt, answer, bbox if any, expected evidence, and failure tag.

Even a simple static HTML gallery can be valuable.

The factory should also keep versioned asset manifests.

When an image is replaced, regenerated, resized, or cropped, downstream bbox and QA artifacts may become stale.

Versioning makes these dependencies explicit.

## QA Rubric Notes

For caption samples, QA should ask whether the answer describes visible content rather than imagined context.

For OCR samples, QA should ask whether the answer reads the correct text and preserves important numbers.

For document samples, QA should ask whether layout and field association are correct.

For chart samples, QA should ask whether the answer uses the correct axis, legend, and unit.

For grounding samples, QA should ask whether the region actually contains the referenced object.

For multi-image samples, QA should ask whether the answer compares the intended images in the intended order.

For all samples, QA should ask whether the answer is useful for training or merely superficially correct.

This last question matters because multimodal data can pass schema checks while teaching very little.

## Scaling Notes

Scaling should start by broadening asset diversity, not by multiplying one easy task.

Document images should expand by layout type.

Chart images should expand by chart type and reasoning requirement.

Grounding samples should expand only when coordinate checks are reliable.

Multi-image samples should expand slowly because review cost is high.

The project should add semi-automatic QA panels before large expansion.

Those panels should make image review fast enough that quality does not collapse under volume.

The long-term direction is a multimodal data operations loop.

In that loop, failure replay, asset versioning, task templates, and QA reports evolve together.

## Final Acceptance Checklist

All image paths resolve.

Image dimensions are recorded.

Asset types are counted.

Task types are counted.

OCR tasks are reviewed.

Chart tasks are reviewed.

Grounding boxes are normalized.

Grounding boxes are reverse-rendered.

Multi-image order is stable.

Conversation roles are valid.

Image tokens are present.

Answers match visible evidence.

Low-quality samples are tagged.

Replay cases are retained.

Train and validation are disjoint.

Smoke set covers each task type.

Manifest counts match files.

Visualization artifacts are present.

QA report includes failure categories.

Prompt templates are versioned.

Asset manifest is versioned.

Review examples are preserved.

Chart units are checked.

Document text associations are checked.

Bbox coordinate conversions are tested.

Multi-image prompts are inspected manually.

Known limitations are stated.

Release notes include asset changes.

The dataset can be rebuilt from config.

Training users receive schema documentation.

OCR failure examples are retained.

Chart failure examples are retained.

Grounding failure examples are retained.

Multi-image failure examples are retained.

Asset replacement policy is documented.

Image resize policy is documented.

Coordinate convention is documented.

QA sampling ratio is documented.

Reviewer calibration examples are retained.

Future asset gaps are listed.

Training loader assumptions are documented.

Release blockers are listed.

Image-source status is recorded.

Document-rendering settings are recorded.

Chart-generation settings are recorded.

Task-template versions are retained.

Annotation assumptions are documented.

Review ownership is documented.

Split policy is documented.

Manifest hash policy is documented.

Smoke-test scope is documented.

Future QA tooling is listed.

Release rollback path is documented.

## References

1. Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual Instruction Tuning.
2. Liu, H., Li, C., Li, Y., et al. (2024). LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge.
3. Dai, W., Li, J., Li, D., et al. (2023). InstructBLIP: Towards General-purpose Vision-Language Models.
4. Mathew, M., Karatzas, D., & Jawahar, C. V. (2021). DocVQA.
5. Masry, A., Long, D. X., Tan, J. Q., et al. (2022). ChartQA.
