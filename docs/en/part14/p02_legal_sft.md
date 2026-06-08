# Project 2: Vertical-domain Expert SFT for Law

## Abstract
P02 focuses on building a legal-domain SFT data factory.

The project is not about making a model sound legal.

It organizes legal source documents, layout-aware parsing, seed chunks, task taxonomy, controlled Self-Instruct expansion, explicit reasoning formats, preference pairs, QA records, risk-refusal data, training delivery, and validation into one data engineering chain.

The project can be read through four main lines.

- Knowledge boundary and seed construction: start from legal PDFs and preserve source metadata.
- Supervision construction: build legal QA, statute explanation, and case-analysis samples under constraints.
- Auxiliary supervision and risk control: add preference pairs, review records, and refusal samples.
- QA, packaging, and validation: make the dataset reviewable, trainable, and versionable.

The engineering chain is:

```text
legal PDFs -> layout-aware cleaning -> chunking and seed schema -> task taxonomy -> controlled Self-Instruct -> explicit legal reasoning -> preference pairs and review records -> risk refusal -> QA protocol -> training package and checks
```

The core goal is to turn legal data generation into a controlled factory with sources, boundaries, review paths, and training interfaces.

## Keywords

Legal SFT; domain data factory; legal QA; preference pairs; risk refusal; QA protocol

## Project Goals and Reader Takeaways

This project uses vertical-domain legal SFT as the core case.

After completing the chapter, readers should understand how to clean legal source documents, define legal seed schemas, design task taxonomies, expand supervision under constraints, add risk-refusal and preference data, and package the result for training.

## Scenario Constraints and Data Boundaries

The project does not provide legal advice.

It does not replace qualified lawyers.

It does not claim full jurisdiction coverage.

It uses legal texts and synthetic expansion to demonstrate a data factory, and all legal claims should remain tied to source scope and review policy.

## Architecture Decision

The project adopts a three-layer architecture: knowledge processing, supervision construction, and QA/delivery.

This architecture keeps source provenance, task intent, review records, risk boundaries, and training artifacts connected.

## Sample Schema and Data Flow

The minimal seed schema should retain `seed_id`, `source_doc`, `jurisdiction`, `domain`, `article_no`, `source_span`, `text`, `risk_tags`, and `audit_trace`.

```text
source document -> cleaned chunk -> seed record -> task sample -> reviewed sample -> training record
```

The generated sample should keep both the instruction and the source lineage.

## Core Implementation Fragments

The chapter keeps fragments that explain PDF cleaning, chunk schemas, task sampling, reasoning templates, and QA decisions.

Full generation prompts, long samples, and review logs should live in companion resources.

## Experimental and Acceptance Metrics

Acceptance metrics include seed count, generated sample count, task distribution, legal-domain distribution, preference-pair count, refusal-sample count, QA acceptance rate, 50-sample validation results, and consistency-check pass rate.

The project report highlights expansion from `2577` to `7737` samples as a structural signal rather than as a standalone quality claim.

## Cost, Risk, and Compliance Boundaries

Costs mainly come from legal source cleaning, generation, human review, preference construction, and validation.

Risks include hallucinated law, fabricated citations, uneven jurisdiction coverage, overconfident advice, excessive synthetic ratio, and high QA cost.

## Common Failure Modes

Common failures include broken PDF layout, lost article numbers, weak source spans, task imbalance, generated answers that exceed source scope, preference pairs without meaningful contrast, and refusal samples that are too artificial.

## Reproducible Resource Notes

Reproducible resources should include source-document notes, cleaned chunks, task configs, generation prompts, QA protocol, review logs, training splits, validation reports, and check scripts.

## 1. Project Background: Why a Legal SFT Data Factory Is Needed

Legal SFT is a strong example of domain data engineering because surface fluency is not enough.

A model that sounds legal can still be unsafe.

It may cite nonexistent provisions, overstate conclusions, ignore jurisdiction boundaries, or answer questions that require professional advice.

Therefore, the project starts from authoritative legal sources and builds a supervised factory around them.

The goal is not only to produce answers.

The goal is to preserve source, task type, legal reasoning structure, review status, preference signal, and risk boundary.

## 2. Project Goals and Boundaries

### 2.1 Project Goals

The first goal is to build cleaned legal seed chunks from source PDFs.

The second goal is to build three main task families: `legal_qa`, `statute_explanation`, and `case_analysis`.

The third goal is to construct preference pairs, review records, and risk-refusal samples.

The fourth goal is to package SFT, validation, smoke-test, and auxiliary data with reports and checks.

### 2.2 Project Boundaries

The project does not cover all legal domains or all jurisdictions.

It does not replace legal professionals.

It does not treat synthetic data as safe without review.

It does not claim that a lightweight validation set equals a full legal benchmark.

### 2.3 Role of Boundary Setting

Legal-domain projects require cautious boundaries.

The dataset should teach grounded legal assistance, not unauthorized legal advice.

Clear boundaries make the project reusable and safer.

## 3. Project Position: P02 in the Capability Chain

P02 sits at the intersection of domain SFT, source-grounded generation, and review operations.

It extends generic instruction tuning into a regulated domain.

Its core contribution is showing how generation becomes a factory when every sample has a source, task type, risk status, and review trail.

## 4. Overall Architecture: From Legal PDFs to Training Assets

![Figure 1: Legal-domain SFT Data Factory Overview](../../images/part10/10_2_fig01_legal_sft_factory_overview.png)

### 4.1 Layer 1: Knowledge Processing

This layer parses legal PDFs, removes layout artifacts, repairs broken text, chunks documents, and builds seed records.

It is the source layer of the factory.

### 4.2 Layer 2: Supervision Construction

This layer samples tasks, runs controlled Self-Instruct, uses explicit legal reasoning formats, and builds preference and refusal data.

It turns source chunks into supervised examples.

### 4.3 Layer 3: QA and Delivery

This layer reviews samples, records accept/rework/reject decisions, packages training splits, and writes reports and checks.

It makes the data usable by training teams.

## 5. Engineering Prerequisites: Key Surfaces of a Legal Data Factory

### 5.1 Domain Design and Knowledge Boundary

The factory must define legal domains, jurisdictions, document types, and source boundaries before generation.

### 5.2 Data Processing and Structured Orchestration

It must preserve IDs, chunks, article numbers, metadata, and source spans.

### 5.3 Generation Control and Task Orchestration

It must control task type, source grounding, reasoning format, and refusal rules.

### 5.4 QA and Acceptance Loop

It must support review decisions, error tags, rework, rejection, and final acceptance.

### 5.5 Role of Responsibility Surfaces

These surfaces prevent the project from becoming a prompt collection.

They define ownership for source, generation, review, and training delivery.

![Figure 2: Roles and Responsibilities in the Legal SFT Factory](../../images/part10/10_2_fig02_roles_and_responsibilities.png)

## 6. Seed Data: The Seed Layer as Supervision Starting Point

### 6.1 Statutory Text as the First Seed Batch

Legal source text provides the initial knowledge boundary.

The model should not invent law beyond this boundary.

### 6.2 Boundary of Legal Text

Source text alone is not a training dataset.

It must be cleaned, chunked, labeled, and tied to task formats.

## 7. PDF Parsing and Intelligent Cleaning

### 7.1 Limits of Plain Text Extraction

Legal PDFs contain headers, footers, embedded page numbers, broken lines, and article numbering.

Plain extraction often destroys structure.

### 7.2 Component Choices

The project should use PDF tools that support layout-aware extraction and page-span tracking.

### 7.3 Cropping Headers and Footers

```python
def crop_header_footer(page, top_ratio=0.08, bottom_ratio=0.08):
    width, height = page.rect.width, page.rect.height
    return page.get_textbox((
        0,
        height * top_ratio,
        width,
        height * (1 - bottom_ratio),
    ))
```

Cropping recurring page chrome helps prevent irrelevant text from entering seeds.

### 7.4 Removing Embedded Page Numbers

Embedded page numbers break legal clauses and confuse chunking.

They should be removed with targeted rules and reviewed examples.

### 7.5 Repairing Broken Text

Mechanical line breaks should be repaired while preserving legal numbering and paragraph boundaries.

### 7.6 Necessity of Fine-grained Cleaning Control

Legal cleaning must be auditable.

If a clause changes meaning during cleaning, the downstream sample becomes risky.

![Figure 3: Legal PDF Intelligent Cleaning Pipeline](../../images/part10/10_2_fig03_pdf_cleaning_pipeline.png)

![Figure 4: Embedded Page-number and Broken-word Repair Examples](../../images/part10/10_2_fig04_cleaning_examples.png)

## 8. Chunking and Schema: Structuring Legal Seeds

### 8.1 Chunking as a Required Step

Legal documents are long and hierarchical.

Chunking creates manageable source units while retaining legal context.

### 8.2 What Schema Solves

Schema records source document, jurisdiction, article, text span, legal domain, and risk tags.

It lets generated samples remain traceable.

### 8.3 Schema as the Foundation of the Seed Layer

Every generated sample should be able to trace back to a seed.

If seed metadata is weak, QA becomes expensive and unreliable.

![Figure 5: Legal Seed Sample Schema](../../images/part10/10_2_fig05_seed_schema.png)

## 9. Task System: Legal SFT Task Layers

### 9.1 Legal QA

`legal_qa` teaches source-grounded answers to factual or procedural questions.

It should preserve citation discipline.

### 9.2 Statute Explanation

`statute_explanation` teaches the model to explain provisions in clear language while preserving conditions and exceptions.

### 9.3 Case Analysis

`case_analysis` gives a short fact pattern and asks for issue, rule, analysis, conclusion, and caveat.

### 9.4 Quality-control Role of Task Decomposition

Task decomposition lets reviewers compare like with like.

It also prevents one task type from dominating the dataset.

![Figure 6: Legal Task Taxonomy](../../images/part10/10_2_fig06_task_taxonomy.png)

## 10. Task Distribution and Sample Structure

### 10.1 Task Balance and Source Distribution

The report should show task distribution and legal-domain coverage together.

Balanced tasks are not enough if source coverage is narrow.

### 10.2 Importance of Sample Structure

Each sample should retain instruction, answer, source, task type, reasoning format, risk flags, and review status.

![Figure 7: Task Distribution and Legal-domain Coverage](../../images/part10/10_2_fig07_task_vs_domain_distribution.png)

## 11. Self-Instruct: Necessity of Controlled Synthesis

### 11.1 Role of Synthetic Expansion

Synthetic expansion increases coverage and transforms source chunks into varied tasks.

### 11.2 Constraints of Legal Synthesis

The generator must not fabricate citations, exceed source scope, or give overconfident advice.

### 11.3 Weighted Roulette and Task Sampling

Weighted sampling can balance task classes and source domains.

It should be recorded in the manifest.

![Figure 8: Weighted Roulette Task Sampling](../../images/part10/10_2_fig08_weighted_task_sampling.png)

## 12. Explicit Chain of Thought: Expression Constraints for Legal Reasoning

### 12.1 Role of Explicit Reasoning

Explicit reasoning helps reviewers inspect how a conclusion was reached.

### 12.2 Boundaries of Legal CoT

Reasoning should be structured and cautious.

It should not encourage unauthorized advice or unsupported conclusions.

### 12.3 Engineering Value of CoT

A stable structure separates issue, rule, analysis, conclusion, and caveat.

It makes review faster.

![Figure 9: CoT Structure for Case-analysis Samples](../../images/part10/10_2_fig09_cot_structure.png)

## 13. Preference Pairs and Review Records

### 13.1 Role of Preference Pairs in Legal Scenarios

Preference pairs compare answers that may both be fluent but differ in correctness, citation quality, caution, or risk handling.

### 13.2 Current Preference Signal Construction

Preference data should record chosen, rejected, reason, source, and reviewer notes.

### 13.3 Role of Review Records

Review records are data assets.

They store decision, error tags, rework instruction, and reviewer metadata.

![Figure 10: Preference Pairs and Review Records](../../images/part10/10_2_fig10_preference_and_review.png)

## 14. Risk Refusal: Boundary-control Data

### 14.1 Relation Between Risk Refusal and System Prompt

Refusal should not live only in the system prompt.

It should appear in the dataset as realistic user requests and safe responses.

### 14.2 Role of Risk-refusal Samples

They teach the model to avoid fabricated law, unauthorized advice, missing jurisdiction facts, and unsafe certainty.

### 14.3 Current Risk-boundary Construction

The project adds risk-refusal samples as auxiliary supervision rather than afterthoughts.

![Figure 11: Risk-refusal Branching in Legal Scenarios](../../images/part10/10_2_fig11_risk_refusal_flow.png)

## 15. QA Protocol: Quality Gates for Legal Data

### 15.1 Review Dimensions

Review dimensions include correctness, source grounding, expression, format, risk handling, and citation discipline.

### 15.2 Accept, Rework, and Reject Rules

Accepted samples enter training.

Reworked samples need clear instructions.

Rejected samples should keep error tags for analysis.

### 15.3 Error Tagging

Common error tags include hallucinated citation, wrong jurisdiction, overconfident conclusion, missing caveat, weak reasoning, and format violation.

### 15.4 Necessity of QA Protocol

Legal SFT without QA protocol is not reliable.

The protocol turns review into an operating system.

![Figure 12: QA Review Loop](../../images/part10/10_2_fig12_qa_loop.png)

![Figure 13: QA Accept/Rework/Reject Decision Table](../../images/part10/10_2_fig13_qa_decision_table.png)

## 16. Vendor Collaboration and Human-machine Division

### 16.1 Layered Review

Machine checks handle formatting, duplicates, obvious citation issues, and source-span presence.

Human reviewers judge legal correctness and risk.

### 16.2 Risks in Vendor Collaboration

Vendor review can drift in standards, lose context, or optimize for speed rather than correctness.

Clear rubrics and audit samples are required.

### 16.3 Engineering Position of Collaboration

Collaboration design belongs in the data factory.

It is not a management appendix.

![Figure 14: Human-in-the-loop and Vendor-layered Review](../../images/part10/10_2_fig14_human_in_the_loop.png)

## 17. Training Packaging: From Supervised Samples to Training Interface

### 17.1 Packaging as an Independent Stage

Training packaging should not be treated as file copying.

It maps reviewed samples into trainable formats.

### 17.2 Main Training Artifacts

Artifacts include `train_sft.jsonl`, `val_sft.jsonl`, `smoke_sft.jsonl`, preference pairs, refusal samples, QA logs, and a training manifest.

### 17.3 Role of Smoke Test

The smoke set validates schema, loader behavior, and tokenization before full training.

![Figure 15: Training Packaging and Delivery Interface](../../images/part10/10_2_fig15_training_artifacts.png)

## 18. Result Overview

### 18.1 Sample Scale

The project expands from `2577` seed-level assets to `7737` training-facing samples.

This shows factory expansion capability, not automatic legal quality.

### 18.2 Task Distribution

The three main task classes remain visible, which means the task framework is stable.

### 18.3 Source Distribution

Uneven source distribution shows where coverage should be improved next.

### 18.4 Preference and Risk Data

Preference, QA, and risk-refusal records move the project from "can answer" toward "answers within boundaries."

### 18.5 Training and Delivery Artifacts

The final artifacts include SFT splits, auxiliary data, manifests, reports, and validation results.

![Figure 16: P02 Metric Dashboard](../../images/part10/10_2_fig16_metrics_dashboard.png)

## 19. Lightweight Downstream Validation

### 19.1 Validation Design

The project uses a lightweight sampled validation design rather than a heavy benchmark.

### 19.2 Validation Metrics

Metrics should cover answer correctness, citation behavior, refusal behavior, format adherence, and reviewer acceptance.

### 19.3 Validation Results

Results should be interpreted as regression evidence for the data factory, not as a full legal capability claim.

### 19.4 Difference from Heavy Benchmarks

Heavy benchmarks evaluate model ability.

Lightweight validation checks whether the data factory is producing stable task behavior.

### 19.5 Engineering Meaning

The 50-sample protocol gives the team a repeatable inspection path.

### 19.6 How to Read Such Experiments

Read them as safety and stability signals, not as final domain certification.

![Figure 17: 50-sample Validation Protocol](../../images/part10/10_2_fig17_eval_sampling_protocol.png)

## 20. Result Interpretation: Structural Signals of the Current Factory

### 20.1 Expansion from 2577 to 7737

This indicates the factory can expand source material into multiple supervision assets.

### 20.2 Balanced Main Tasks

Balanced task types indicate that the task framework is operational.

### 20.3 Uneven Legal-domain Distribution

The next focus should be filling coverage gaps, not only increasing volume.

### 20.4 Preference, QA, and Lightweight Validation Together

Together they show movement from answer generation toward stable legal assistance.

### 20.5 Visible Human-review Cost

Visible review cost means later versions must optimize review workflow.

### 20.6 What the Supplement Adds

The supplement adds stronger project evidence: task structure, auxiliary supervision, validation, and checks.

## 21. Quality Baseline for Legal SFT Data

### 21.1 Correctness Baseline

Answers must align with source text and avoid fabricated law.

### 21.2 Expression Baseline

Language should be clear, cautious, and not overconfident.

### 21.3 Format Baseline

Samples should follow required task format.

### 21.4 Risk Baseline

Out-of-scope or unsafe requests should be refused or redirected.

### 21.5 Baseline Versus Single Score

A baseline is more useful than one aggregate score because legal errors differ in severity.

## 22. Version Evolution

### 22.1 V1: Parse and Clean Statutes

The first version proves source processing.

### 22.2 V2: Add Three Main Task Types

The second version proves supervision construction.

### 22.3 V3: Add Preference Pairs and Review Records

The third version adds ranking and review signals.

### 22.4 V4: Add Risk Refusal and Launch Boundaries

The fourth version introduces explicit safety boundaries.

### 22.5 Record Value of Version Evolution

Version history tells future users what changed and why.

![Figure 18: P02 Version-evolution Roadmap](../../images/part10/10_2_fig18_version_timeline.png)

## 23. Cost Optimization

### 23.1 Cost Lessons

The main costs are source cleaning, generation, review, validation, and rework.

### 23.2 Automation Priorities

Automate formatting checks, source-span checks, duplicate detection, and obvious citation failures before automating legal judgment.

### 23.3 Necessity of Cost Analysis

Without cost analysis, the factory cannot scale responsibly.

## 24. Verification Loop: Consistency Checks

### 24.1 Role of Check Scripts

Check scripts verify that code, artifacts, metrics, and reports agree.

### 24.2 Current Validation State

The desired state is all command-level and artifact-level checks passing.

### 24.3 Engineering Role of Verification

Verification makes the data factory inspectable and reproducible.

![Figure 19: Code-artifact-report Consistency Validation](../../images/part10/10_2_fig19_validation_chain.png)

## 25. Limitations and Risks

### 25.1 Uneven Legal-domain Coverage

Some legal domains or jurisdictions may be underrepresented.

### 25.2 High Synthetic Ratio

Synthetic expansion must be balanced by review and source grounding.

### 25.3 Risk-refusal Samples Still Limited

More boundary cases are needed before launch.

### 25.4 High QA Cost

Legal review is expensive and must be planned as part of the system.

## 26. Cross-industry Transfer

### 26.1 Designs That Transfer Directly

Source-grounded seeds, task taxonomy, constrained synthesis, review records, refusal data, packaging, and checks transfer well.

### 26.2 Parts That Cannot Be Copied Directly

Legal risk categories, citation norms, and jurisdiction rules cannot be copied into other industries.

### 26.3 Transferable Method Chain

The transferable method is define sources, clean and chunk, build tasks, synthesize under constraints, review, add risk data, package, and validate.

![Figure 20: Cross-domain Transfer Method Chain](../../images/part10/10_2_fig20_cross_domain_transfer.png)

## 27. Main Deliverables

### 27.1 Seed and Processing Artifacts

- Cleaned legal source chunks.
- Seed schemas.
- Source manifests.

### 27.2 Main and Auxiliary Supervision Artifacts

- Legal QA samples.
- Statute explanation samples.
- Case-analysis samples.
- Preference pairs.
- QA review logs.
- Risk-refusal records.

### 27.3 Training Interface Artifacts

- Train split.
- Validation split.
- Smoke split.
- Training manifest.

### 27.4 Report and Verification Artifacts

- Metrics report.
- QA report.
- Validation report.
- Check report.

## 28. Summary: Organizing Generation into a Factory

Legal SFT data should not be a pile of generated prompts.

It should be a factory where every sample has source, task type, review path, risk boundary, and training interface.

That is the method value of P02.

## Special Topic: Release Gates for Legal SFT Data

### 1. Release Should Inspect Content Risk and Engineering Risk

Content risk includes hallucinated law, wrong jurisdiction, missing caveats, and unsafe advice.

Engineering risk includes missing source metadata, split leakage, schema inconsistency, and report mismatch.

### 2. Gate Value Is Making Caution a System Property

Caution should not depend only on one reviewer.

It should be encoded in task design, refusal data, QA rules, and release checks.

## Chapter Summary

This chapter used legal-domain SFT to show how a regulated vertical dataset can be organized as a data factory.

The project connects legal PDF cleaning, seed schemas, task taxonomy, controlled synthesis, reasoning templates, preference pairs, QA records, refusal data, training packaging, validation, and versioning.

Its boundary should remain clear: it is a data engineering case for legal-domain supervision, not legal advice and not a complete legal AI product.

## Release Review Notes

The first release review item is source authority.

Every seed should point to a legal source document.

The source should include jurisdiction, document type, and article or section reference when available.

The second item is layout cleaning.

Reviewers should inspect whether headers, footers, page numbers, and broken lines were removed without changing legal meaning.

This is especially important for article numbering.

The third item is chunk integrity.

Chunks should not cut a legal condition away from its exception.

If chunking splits related clauses, generated answers can become overconfident.

The fourth item is task distribution.

Legal QA, statute explanation, and case analysis should each have enough samples to justify the task taxonomy.

The fifth item is source distribution.

Task balance is not sufficient if the dataset comes from a narrow legal domain.

The report should show both task type and source coverage.

The sixth item is synthetic expansion ratio.

Synthetic samples should be tied to seed chunks.

They should also carry generation metadata and review status.

The seventh item is preference-pair quality.

Preference pairs should express meaningful differences.

Differences may include citation quality, caution, legal correctness, or refusal behavior.

Pairs where both answers are nearly identical should be removed.

The eighth item is refusal coverage.

Risk-refusal samples should cover missing facts, out-of-scope jurisdiction, unsafe advice, fabricated citation, and professional-service boundaries.

The ninth item is QA reproducibility.

Review decisions should include decision type, error tags, reviewer notes, and rework instructions.

The tenth item is training packaging.

The SFT split, preference split, refusal split, smoke set, and manifest should be checked together.

## Operating Notes

During daily operation, first inspect source coverage.

If one legal domain dominates, further generation may make imbalance worse.

Then inspect QA error tags.

If fabricated citations rise, tighten source-span checks before increasing generation.

If overconfident advice rises, add refusal and caveat samples.

If formatting errors rise, automate schema and template checks.

If reviewer disagreement rises, revise the rubric before continuing production.

Legal data factories should keep a release ledger.

The ledger should record source scope, task distribution, synthetic ratio, refusal coverage, review pass rate, validation results, and known risk boundaries.

It should also record which legal claims the data is not intended to support.

That negative scope is part of safety.

For downstream users, the most useful artifact is often the manifest.

The manifest tells them what the dataset contains, how it was reviewed, and where its boundaries are.

Without this manifest, a legal SFT dataset can be misused outside its intended scope.

## Transfer and Scaling Notes

The method can transfer to medicine, finance, insurance, and compliance.

However, transfer should start from authoritative sources in the new domain.

The task taxonomy must be rebuilt.

The refusal taxonomy must also be rebuilt.

Reviewers must understand the target domain.

The legal project is therefore a method template, not a content template.

Scaling should prioritize coverage and review efficiency before raw sample count.

Adding more samples without source balance can make the factory look larger while reducing trust.

## Final Acceptance Checklist

Source documents are listed.

Jurisdictions are recorded.

PDF cleaning examples are reviewed.

Article numbers are preserved.

Chunks retain source spans.

Seed schemas are complete.

Task taxonomy is explicit.

Task distribution is reported.

Source distribution is reported.

Synthetic ratio is visible.

Generated samples keep source links.

Preference pairs have contrast.

Risk refusals cover boundary cases.

QA decisions are logged.

Error tags are standardized.

Rework instructions are preserved.

Smoke split loads correctly.

Validation samples are reviewed.

Manifest matches artifacts.

Known legal boundaries are stated.

Release notes include review status.

Downstream users receive usage limits.

Citation policy is documented.

Refusal policy is documented.

Reviewer calibration examples are retained.

Synthetic prompt versions are retained.

Future coverage gaps are listed.

Legal-review sampling ratio is documented.

Risk examples are retained.

Citation examples are retained.

Domain expansion plan is documented.

## References

1. Wang, Y., Kordi, Y., Mishra, S., et al. (2023). Self-Instruct: Aligning Language Models with Self-Generated Instructions.
2. Wei, J., Wang, X., Schuurmans, D., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.
3. Bai, Y., Kadavath, S., Kundu, S., et al. (2022). Constitutional AI: Harmlessness from AI Feedback.
4. NIST. (2023). Artificial Intelligence Risk Management Framework.
5. OECD. (2024). AI, Data Governance, and Risk Management Guidance.
