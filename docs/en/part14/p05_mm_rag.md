# Project 5: Multimodal RAG Enterprise Financial Report Assistant

## Abstract
P05 focuses on building a multimodal RAG assistant for enterprise financial reports.

The project is not simply "OCR plus question answering".

It organizes PDF page assets, visual retrieval, multi-page evidence recall, multimodal prompting, visual reasoning, evaluation, failure replay, and enterprise gates into one application-level data engineering pipeline.

The project can be read through four lines.

- Page asset construction: turn report PDFs into stable visual evidence pages.
- Vision-first retrieval: use visual document retrieval instead of relying only on OCR text.
- Multi-image reasoning: send evidence pages to a VLM with prompt constraints.
- Evaluation and operations: verify retrieval, answer evidence, failure modes, cost, and launch readiness.

The engineering chain is:

```text
financial-report PDF -> page images -> visual index -> Top-K page retrieval -> multi-image VLM prompt -> answer with evidence -> evaluation, replay, and gate checks
```

The core goal is to show that multimodal RAG depends on how the system uses visual evidence, not only on whether the model can see images.

## Keywords

Multimodal RAG; financial report; visual retrieval; ColPali; Byaldi; Qwen2.5-VL; evidence page

## Project Goals and Reader Takeaways

This project uses an enterprise financial report assistant as the core case.

After completing the chapter, readers should understand how to render report pages, build a visual index, retrieve multiple evidence pages, prompt a multimodal model, evaluate visual answers, and define enterprise launch gates.

## Scenario Constraints and Data Boundaries

The project focuses on high-value financial-report question answering.

It does not replace professional financial analysis, formal audit, or investor guidance.

It also does not claim that a small evaluation set proves enterprise-grade robustness.

The boundary is a reproducible multimodal RAG prototype with visible evidence and quality gates.

## Architecture Decision

The project adopts a vision-first path: render PDF pages, build visual embeddings, retrieve Top-K pages, and let a VLM reason over the retrieved page images.

This avoids losing table, chart, and layout information too early through OCR-only processing.

## Sample Schema and Data Flow

The minimal sample should retain `query_id`, `question`, `report_id`, `page_images`, `retrieved_pages`, `evidence_type`, `answer`, `citation_pages`, `quality_signals`, and `audit_trace`.

```text
PDF page -> visual asset -> index item -> retrieved evidence -> multimodal answer -> evaluation record
```

## Core Implementation Fragments

The chapter keeps fragments that explain indexing, retrieval, prompting, and multi-image evidence organization.

Full PDF files, model checkpoints, and long logs should be maintained as companion resources.

## Experimental and Acceptance Metrics

Acceptance metrics include retrieval hit rate, evidence-page precision, answer correctness, chart-trend correctness, table-column alignment, citation-page completeness, cost per query, and failure-replay coverage.

For enterprise use, the assistant must show evidence pages and uncertainty boundaries.

## Cost, Risk, and Compliance Boundaries

Costs include page rendering, visual embedding, index storage, VLM inference, manual evaluation, and failure replay.

Risks include wrong page retrieval, chart misreading, table misalignment, hallucinated financial conclusions, stale reports, and insufficient evidence display.

## Common Failure Modes

Common failures include table-of-contents pages being retrieved too often, chart trend misread, table column mismatch, topic drift across pages, low-resolution pages, and answers that sound right but cannot be traced to evidence.

## Reproducible Resource Notes

Reproduction materials should include PDF page assets, index configuration, query set, retrieval results, prompt templates, answer logs, evaluation labels, and replay cases.

## 1. Project Background: Why a Multimodal RAG Financial Report Assistant Is Needed

Financial reports are dense multimodal documents.

They contain text, tables, charts, footnotes, headings, page layouts, and cross-page references.

Traditional text RAG can work for many narrative sections, but it often loses chart layout, table alignment, and visual context.

An enterprise assistant that answers questions about revenue, margin, segment performance, or risk disclosures needs evidence that users can inspect.

P05 therefore builds a multimodal RAG pipeline where the page itself remains an evidence object.

The project demonstrates not just model vision, but system-level use of visual evidence.

## 2. Project Goals and Boundaries

### 2.1 Project Goals

The first goal is to build stable page assets from financial-report PDFs.

The second goal is to build a visual retrieval index.

The third goal is to retrieve multiple evidence pages for each question.

The fourth goal is to use a VLM to answer with page-grounded evidence and evaluate the result.

### 2.2 Project Boundaries

The project does not cover all financial documents.

It does not replace expert financial analysis.

It does not guarantee performance on every PDF layout.

It does not treat a small QA set as proof of production robustness.

### 2.3 Role of Boundary Statements

Boundaries keep the assistant from overclaiming.

Financial answers should be evidence-based, cautious, and reviewable.

## 3. Project Position: P05 in the Capability Chain

P05 sits between multimodal data engineering and application-level RAG.

It connects document assets, visual retrieval, multimodal reasoning, and enterprise QA.

Its value is showing how retrieval and generation should work when the evidence is visual pages, not only text chunks.

## 4. Overall Architecture: From Financial-report PDF to Multimodal Answer

![Figure 1: Multimodal RAG Financial Report Assistant Architecture](../../images/part10/10_5_fig01_overall_architecture.png)

### 4.1 Layer 1: Page Asset Layer

This layer renders PDF pages into images and records page numbers, dimensions, report IDs, and metadata.

It creates the evidence base.

### 4.2 Layer 2: Visual Retrieval Layer

This layer builds visual embeddings and retrieves Top-K relevant pages for a question.

It keeps layout and chart information visible.

### 4.3 Layer 3: Multi-image Reasoning Layer

This layer sends retrieved pages to a VLM and constrains the answer to evidence.

It is responsible for grounded synthesis, not open-ended speculation.

## 5. Data Flow and Core Idea: Vision-first Retrieval

### 5.1 Limits of OCR-first

OCR-first pipelines extract text before retrieval.

They can lose table columns, chart axes, page layout, and visual grouping.

### 5.2 Value of Vision-first

Vision-first retrieval keeps the page image as the retrieval object.

This is useful when charts and tables matter.

### 5.3 Why This Project Uses ViR plus VLM

Visual retrieval recalls pages with relevant visual evidence.

The VLM then reads and reasons over those pages.

Together they preserve both retrieval and visual understanding.

![Figure 2: Vision-first and OCR-first Route Comparison](../../images/part10/10_5_fig02_vision_vs_ocr.png)

## 6. Technology Choices: ColPali, Byaldi, and Qwen2.5-VL

### 6.1 Position of ColPali in Document Retrieval

ColPali-style visual document retrieval embeds page images for late-interaction retrieval.

It is well suited to visually rich documents.

### 6.2 Byaldi as an Indexing Framework

Byaldi provides a practical indexing and retrieval interface for ColPali-like retrieval.

It makes the prototype easier to reproduce.

### 6.3 Visual Model for Generation

Qwen2.5-VL or a similar VLM can consume retrieved page images and produce an answer.

The answer should remain tied to retrieved evidence.

### 6.4 Engineering Meaning of the Choices

The choices separate retrieval, evidence organization, and generation.

This makes failures easier to locate.

## 7. Page Asset Construction: Stable Page Evidence Base

### 7.1 What the Page Asset Layer Solves

It turns PDF pages into reusable evidence objects.

Each page has a file path, page number, report ID, dimensions, and optional metadata.

### 7.2 Why "Reviewable" Matters

Users and reviewers should be able to inspect the pages behind an answer.

This is especially important for financial reports.

### 7.3 Relation to Project Artifacts

The page asset manifest connects rendered images, retrieval index entries, answers, and evaluation labels.

![Figure 3: Page Assets and Page-number Mapping](../../images/part10/10_5_fig03_page_assets.png)

## 8. Index Construction: Organizing the Multimodal Index

### 8.1 Local Model Loading and Offline Mode

Enterprise prototypes often need local model loading or offline mode to control dependencies and data exposure.

### 8.2 Original Images Must Stay Bound to the Index

Index entries should point back to original page images.

Otherwise retrieved results cannot be inspected.

### 8.3 True Difficulty of Indexing

The difficulty is not only embedding pages.

It is preserving page identity, report identity, image quality, and index version.

### 8.4 Why Index Construction Affects the Ability Ceiling

If the right page is not retrievable, the VLM cannot answer correctly.

Retrieval quality sets the upper bound for answer quality.

![Figure 4: PDF Page Rendering and Visual Index Construction](../../images/part10/10_5_fig04_indexing_pipeline.png)

## 9. Retrieval Design: Top-K Multi-page Recall

### 9.1 Why Table-of-contents Pages Are Frequent False Recalls

TOC pages contain many keywords.

They can match many questions while lacking the actual evidence.

### 9.2 Value of Top-K

Top-K retrieval gives the system multiple chances to include the real evidence page.

It is especially important for cross-page financial questions.

### 9.3 Why Multi-page Recall Is Robustness Design

Financial answers often require comparing tables, charts, and notes across pages.

Single-page recall is brittle.

### 9.4 Further Filtering Logic

The system can filter TOC pages, rerank by page type, remove repeated section headers, and prefer pages with charts or tables when the question asks for them.

![Figure 5: Top-K Multi-page Recall and TOC Filtering](../../images/part10/10_5_fig05_topk_filtering.png)

## 10. Prompt Design: Anti-noise Constraints in Generation

### 10.1 Key Prompt Ideas

The prompt should tell the model to answer only from the provided pages, cite page numbers, state uncertainty, and avoid unsupported financial conclusions.

### 10.2 Why Multimodal Generation Is More Noise-sensitive

The model may attend to the wrong page, wrong chart region, or wrong table column.

Noise appears visually, not only textually.

### 10.3 A More Stable Prompt Skeleton

```text
Use only the provided report pages.
Identify the page or chart that supports the answer.
If the evidence is insufficient, say so.
Return the answer, evidence pages, and limitations.
```

### 10.4 Why Prompting Is Retrieval Post-processing

Prompting decides how retrieved evidence is presented to the model.

It is part of the retrieval pipeline, not only generation style.

## 11. Multi-image Context Organization

### 11.1 Basic Principles

Retrieved pages should be ordered by relevance and page number when useful.

The question should reference the page set clearly.

### 11.2 Why Order Matters

The model may treat earlier images as more important.

Wrong order can bias reasoning.

### 11.3 Why Output Style Should Be Limited

The assistant should produce concise conclusions, evidence pages, and limitations.

It should not produce unsupported financial advice.

![Figure 6: Multi-image Context Injection and Answer Constraints](../../images/part10/10_5_fig06_multi_image_prompting.png)

## 12. Step-by-step Practice: Minimal Reproducible Chain

### 12.1 Stage 1: Visual Index Construction

Render PDF pages, build the visual index, and write the page manifest.

```python
index = build_visual_index(page_images, model="colpali")
```

### 12.2 Stage 2: Multi-page Retrieval

For each question, retrieve Top-K page candidates and preserve scores.

```python
results = index.search(query, k=5)
```

### 12.3 Stage 3: Multi-image Reasoning

Send the question and retrieved page images to the VLM with evidence constraints.

```python
answer = vlm_answer(question=query, images=[r.image for r in results])
```

### 12.4 Stage 4: Result Return and Evidence Organization

Return answer, cited pages, retrieved pages, scores, and limitations in one record.

## 13. Real Run Records: Evidence and Logs

Run logs should include retrieved page IDs, scores, prompt version, model version, answer, and evaluation label.

### 13.1 Why Run Records Matter

Logs make failures replayable.

They also show whether retrieval or generation caused the error.

### 13.2 How Logs Help Prompt Adjustment

If the correct page is retrieved but the answer is wrong, prompt or generation needs work.

If the correct page is absent, retrieval needs work.

## 14. Evaluation and Verification

### 14.1 What Existing Metrics Show

Metrics may show strong results on a small set.

They should be read as prototype evidence.

### 14.2 Why These Metrics May Look Smooth

Small evaluation sets can be curated and less adversarial.

They may not expose all layout and retrieval failures.

### 14.3 Key Metrics in Multimodal Scenarios

Important metrics include retrieval hit rate, evidence-page precision, answer correctness, chart-trend correctness, table alignment, page citation correctness, and uncertainty handling.

### 14.4 Why Chart-trend Understanding Should Be Separate

Chart trend errors are common and high impact.

They deserve a separate metric rather than being hidden in answer accuracy.

![Figure 7: Retrieval and Answer Evaluation Framework](../../images/part10/10_5_fig07_eval_framework.png)

## 15. Metric Interpretation: Boundaries of Current Results

Strong early metrics are encouraging.

They do not prove enterprise readiness.

### 15.1 Why Small Evaluation Sets Can Be Too Smooth

Small sets may not contain TOC traps, noisy charts, multi-page conflicts, or low-resolution pages.

### 15.2 Why This Is Still a Good Signal

Smooth prototype results show that the core chain works.

They justify building harder replay and stress sets.

## 16. Failure Modes

### 16.1 TOC False Recall

The retrieval system may return table-of-contents pages because they contain many terms.

### 16.2 Correct Object, Wrong Trend

The model may find the right chart but misread increase, decrease, or comparison.

### 16.3 Table Column Alignment Error

Financial tables can be dense.

The model may read the wrong year or segment column.

### 16.4 Topic Drift Across Pages

Multi-page context can cause the answer to blend unrelated evidence.

### 16.5 Insufficient Page Clarity

Low-resolution pages reduce both retrieval and reasoning quality.

### 16.6 Why Failure Replay Matters

Failure replay turns mistakes into a stable evaluation asset.

## 17. Cost Analysis

### 17.1 Indexing Cost

Indexing cost includes page rendering, embedding, and storage.

### 17.2 Inference Cost

VLM inference cost depends on page count, resolution, prompt length, and model size.

### 17.3 Hidden Cost

Hidden costs include evaluation labeling, replay maintenance, PDF preprocessing, and business review.

### 17.4 Why Cost Must Be Measured Separately

Multimodal RAG can be expensive per query.

Cost must be considered before enterprise rollout.

## 18. Optimization Directions

### 18.1 Page Cropping and Tiling

Cropping or tiling can focus on tables and charts instead of full pages.

### 18.2 Patch-level Retrieval

Patch-level retrieval can improve evidence localization.

### 18.3 Retrieval Reranking

Reranking can remove TOC pages and prioritize evidence pages.

### 18.4 Multi-turn QA and Evidence Memory

Evidence memory can support follow-up questions without repeating retrieval from scratch.

### 18.5 Templated Answer Output

Templated outputs make answers easier to audit.

![Figure 8: Multimodal RAG Optimization Roadmap](../../images/part10/10_5_fig08_optimization_roadmap.png)

## 19. Engineering Deployment: Fit for High-value Low-frequency Scenarios

### 19.1 Suitable Scenarios

The project fits analyst support, report review, board-pack summarization, and high-value financial QA.

### 19.2 Less Suitable Scenarios

It is less suitable for low-value high-volume queries where text RAG is enough.

### 19.3 Why Start with High-value Scenarios

High-value scenarios justify multimodal cost and review.

They also demand evidence display.

## 20. Relationship with Traditional Text RAG

### 20.1 Text RAG Still Has Value

Narrative sections and extracted text remain useful.

Text RAG is cheaper and often faster.

### 20.2 What Multimodal RAG Fits Better

Multimodal RAG fits charts, tables, figures, and visually organized pages.

### 20.3 More Reasonable Long-term Shape

The best architecture is often hybrid.

Text RAG and visual RAG cooperate according to evidence type.

![Figure 9: Hybrid Architecture of Text RAG and Multimodal RAG](../../images/part10/10_5_fig09_hybrid_rag.png)

## 21. Quality Baseline for a Multimodal Financial Report Assistant

### 21.1 Retrieval Baseline

The correct evidence page should appear in Top-K.

### 21.2 Numeric Baseline

Numbers should be read from the correct table, chart, year, and segment.

### 21.3 Trend Baseline

Trend descriptions should match the visual evidence.

### 21.4 Evidence Baseline

Answers should cite pages or evidence types.

### 21.5 Cost Baseline

Cost per query should be tracked.

### 21.6 Why Baselines Beat One-off Demos

Baselines create repeatable release gates.

## 22. Deliverables and Reproduction Path

### 22.1 Main Deliverables

- Rendered page images.
- Page manifest.
- Visual index.
- Query set.
- Retrieval results.
- Multimodal answers.
- Evaluation labels.
- Replay set.
- Metrics report.

### 22.2 Why These Artifacts Matter

They connect source documents, evidence pages, model answers, and evaluation.

### 22.3 Reproduction Steps

Render pages, build the index, run retrieval, run VLM answers, evaluate outputs, and inspect replay failures.

## 23. Summary: The Key Is Not That the Model Can See Images, but That the System Can Use Images

Multimodal RAG is a system problem.

The model's visual ability matters, but page assets, retrieval, evidence organization, prompting, evaluation, and operations matter just as much.

P05 demonstrates how those pieces connect.

## Special Topic: Evaluation Set Design and Annotation Standards

### 1. What Question Types Should Be Covered

The evaluation set should include numeric lookup, trend reading, table comparison, chart interpretation, cross-page synthesis, and uncertainty cases.

### 2. Annotation Should Not Only Store Final Answers

Annotations should store evidence pages, answer rationale, acceptable variants, and known traps.

### 3. Evaluation Needs Regular and Stress Layers

The regular set checks normal operation.

The stress set checks TOC traps, low-resolution pages, ambiguous charts, and multi-page conflicts.

### 4. Failure Replay Needs Continuous Updates

Every release should add high-value failures to replay.

## Special Topic: Enterprise Launch Gates

### 1. Document Ingestion Gate

Not every PDF should enter the system directly.

Poorly rendered, stale, or unauthorized documents should be blocked.

### 2. Retrieval Gate

Evidence pages must meet retrieval quality thresholds before answer generation is trusted.

### 3. Answer Gate

Answers must be correct, cited, cautious, and reviewable.

### 4. Operations Gate

The system must support monitoring, replay, cost tracking, and issue ownership.

## Special Topic: Collaboration Workflow

### 1. Role Division

Data engineers manage page assets and indexes.

Model engineers manage retrieval and VLM calls.

Financial reviewers evaluate answer correctness.

Product owners define use cases and thresholds.

### 2. Daily Iteration

Failure cases should enter the next optimization cycle.

The team should separate retrieval failures from reasoning failures.

### 3. Business Integration

Start with high-value decisions before expanding scenarios.

## Special Topic: Evidence Presentation and Answer Display

### 1. Answers Should Show Conclusion, Evidence, and Limits

The assistant should not only provide a conclusion.

It should show supporting pages and uncertainty.

### 2. Page Numbers and Evidence Types Should Be Explicit

Financial users need to verify the source quickly.

Page numbers and chart/table references help.

### 3. Uncertain Answers Should Stay Restrained

If evidence is insufficient, the assistant should say so.

## Special Topic: From QA Assistant to Analysis Workbench

### 1. An Analysis Workbench Emphasizes Continuous Tasks

A workbench supports follow-up questions, saved evidence, comparisons, and analyst workflows.

### 2. Current Project Lays Key Foundations

Page assets, visual indexes, evidence logs, and replay sets are the foundation for a future workbench.

## Chapter Summary

This chapter used a multimodal RAG financial report assistant to show how page-level visual evidence can support enterprise QA.

The project connects PDF rendering, visual indexing, Top-K retrieval, multi-image prompting, evidence-grounded answers, evaluation, failure replay, cost analysis, and launch gates.

Its boundary should remain clear: it is a reproducible multimodal RAG prototype, not a certified financial analysis product.

## Release Review Notes

The first release review item is document eligibility.

Every report should have source status, date, version, and access permission recorded.

Outdated or unauthorized reports should not enter the index.

The second item is page rendering quality.

A reviewer should inspect sample pages at the resolution used for retrieval and generation.

If small numbers or chart labels are unreadable, the downstream model cannot be expected to recover them.

The third item is page manifest integrity.

Every page image should map to report ID, page number, file path, and render settings.

This mapping is the basis of evidence citation.

The fourth item is index versioning.

The visual index should record model version, page set, embedding configuration, and build time.

Answers from different index versions should not be mixed without labeling.

The fifth item is retrieval evaluation.

The correct evidence page should be checked for each evaluation question.

The review should distinguish between "page not retrieved" and "page retrieved but model misread it".

The sixth item is TOC and boilerplate filtering.

Table-of-contents pages, cover pages, and repeated section pages should be tracked as common false recalls.

The seventh item is prompt versioning.

Prompts should be versioned because wording changes can change evidence use.

The eighth item is answer evidence.

Every answer should return evidence pages or state that evidence is insufficient.

The ninth item is financial caution.

The assistant should avoid investment advice, audit conclusions, or unsupported causal claims.

The tenth item is replay coverage.

The replay set should contain retrieval failures, chart failures, table failures, and cross-page synthesis failures.

## Operating Notes

Daily operation should begin with document and index health.

If a report changes, page rendering and index building must be rerun.

If render settings change, previous visual embeddings may become stale.

If retrieval hit rate drops, inspect index version, query distribution, and page rendering quality.

If answer quality drops while retrieval remains strong, inspect prompt version and VLM behavior.

If cited pages are missing, inspect answer formatting and evidence extraction.

If cost rises, inspect Top-K size, image resolution, and model choice.

If TOC false recall rises, add page-type filters or rerankers.

If chart trend errors rise, add chart-specific evaluation cases.

If table alignment errors rise, add table-specific review prompts or page crops.

The assistant should be operated as an evidence system.

The answer alone is not the product.

The answer plus evidence plus limitations is the product.

This matters because financial users need to verify conclusions.

## Evaluation Notes

A minimal evaluation set should include simple lookup questions.

It should also include chart-trend questions.

It should include table comparison questions.

It should include multi-page synthesis questions.

It should include insufficient-evidence questions.

It should include questions where the TOC is a tempting false match.

Each question should store expected evidence pages.

It should also store acceptable answer variants.

For numerical answers, the annotation should store the expected value, unit, year, and segment.

For trend answers, the annotation should store direction, comparison baseline, and page evidence.

For synthesis answers, the annotation should store all required pages.

Evaluation should report retrieval and generation separately.

A single accuracy number hides too much.

Retrieval failure and reasoning failure require different fixes.

## Enterprise Deployment Notes

Enterprise rollout should begin with a narrow report type.

The team should start with high-value low-frequency questions.

Those questions justify visual retrieval cost and manual evaluation.

The product should show page thumbnails or links.

Users should not be forced to trust an answer without evidence.

The system should log question, retrieved pages, prompt version, model version, answer, and user feedback.

Feedback should enter a replay set.

High-risk answers should require review or a stronger caution template.

The assistant should also support abstention.

In financial settings, a careful "not enough evidence" is often better than a confident unsupported answer.

Long-term, the system can evolve into an analysis workbench.

That workbench would preserve evidence across turns, support comparisons, and allow analysts to annotate pages.

P05 already prepares the foundation because it treats page assets, retrieval logs, and replay cases as first-class artifacts.

## Final Acceptance Checklist

Report source status is recorded.

PDF render settings are recorded.

Page images are inspectable.

Page manifest maps IDs to page numbers.

Visual index version is recorded.

Retrieval model version is recorded.

Prompt version is recorded.

Top-K retrieval logs are stored.

Evidence pages are labeled.

TOC false recalls are tracked.

Chart questions are evaluated separately.

Table questions are evaluated separately.

Cross-page questions are evaluated separately.

Insufficient-evidence questions are included.

Answers cite page evidence.

Uncertainty wording is supported.

Cost per query is measured.

Replay cases are retained.

Retrieval and generation errors are separated.

Enterprise gates are documented.

Financial caution boundaries are stated.

Users can inspect cited pages.

Feedback logs are versioned.

The system can rebuild the index.

Known report limitations are listed.

Release notes include model and index changes.

Retrieval threshold policy is documented.

Page-citation policy is documented.

Abstention policy is documented.

Chart review examples are retained.

Table review examples are retained.

Evidence display behavior is documented.

User feedback routing is documented.

Future report gaps are listed.

Index rollback path is documented.

Prompt rollback path is documented.

Evidence-thumbnail behavior is documented.

High-risk query policy is documented.

Human-review trigger is documented.

Report refresh policy is documented.

Evaluation owner is documented.

Operational escalation path is documented.

## References

1. Faysse, M., Sibille, H., Wu, T., et al. (2024). ColPali: Efficient Document Retrieval with Vision Language Models.
2. Byaldi Authors. Byaldi documentation and examples for visual document retrieval.
3. Qwen Team. (2025). Qwen2.5-VL Technical Report.
4. Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
5. Chen, W., et al. (2022). Chart and table understanding datasets for document intelligence.
