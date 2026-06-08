# Project 1: Building a Distributed Mini-C4 Data Pipeline with Ray

## Abstract
P01 is the entry project for the practical part of the book.

It turns a small Common Crawl slice into a cleaned, deduplicated, language-aware, and training-ready corpus.

The point is not to maximize token count.

The point is to make the control surface of a pre-training corpus factory visible.

The project can be read through four lines.

- WARC parsing and main-text extraction: move from web records to usable text.
- Cleaning, deduplication, and language split: remove first-order noise and expose corpus structure.
- Quality filtering and experiment replay: move from "looks like text" to "suitable for training."
- Packaging, reporting, and checks: turn cleaned records into a training interface.

In engineering order, the chain is:

```text
Common Crawl WARC -> HTML parsing -> main-text extraction -> heuristic cleaning -> MinHash/LSH deduplication -> language split -> quality filtering -> train/validation/smoke split -> manifest and validation checks
```

The core goal is to show how a pre-training corpus pipeline becomes inspectable before it becomes large.

## Keywords

Mini-C4; Common Crawl; Ray; data cleaning; deduplication; pre-training corpus

## Project Goals and Reader Takeaways

This project uses Mini-C4 as a compact case for pre-training corpus construction.

After completing the chapter, readers should be able to identify the key gates of a web-corpus pipeline, explain why each gate exists, build a minimal reproducible corpus flow, and extend the method toward larger pre-training factories.

## Scenario Constraints and Data Boundaries

The project uses a bounded Common Crawl slice and a teaching-scale processing setup.

It does not represent a full production corpus factory, a complete copyright review system, or a multilingual quality platform.

The boundary is intentional.

Small scale lets the reader inspect every intermediate artifact and every filtering decision.

## Architecture Decision

The project adopts a staged architecture: parse first, clean second, deduplicate third, split by language, filter by quality, and package only after the corpus is stable.

This avoids the common mistake of exporting training files before knowing what the pipeline actually removed.

## Sample Schema and Data Flow

The minimal record should retain `id`, `url`, `warc_path`, `source`, `language`, `text`, `quality_signals`, `filter_reason`, `split`, and `audit_trace`.

```text
raw WARC record -> parsed text record -> cleaned record -> deduplicated record -> language branch -> quality-filtered record -> training record
```

These fields are not decorative.

They let later stages explain why a record was kept, rejected, or routed into a particular split.

## Core Implementation Fragments

The chapter keeps only implementation fragments that explain engineering decisions.

Long WARC readers, Ray job configuration, full logs, and generated reports should live in companion resources.

The main text focuses on contracts, thresholds, failure modes, and acceptance checks.

## Experimental and Acceptance Metrics

Acceptance metrics include extraction rate, cleaning rejection rate, duplicate removal ratio, language distribution, quality-filter pass rate, train/validation/smoke split consistency, and manifest/check pass rate.

For public reproduction, record input WARC identifiers, dependency versions, random seeds, thresholds, and sample-review results.

## Cost, Risk, and Compliance Boundaries

Costs mainly come from WARC download, HTML parsing, signature generation, quality scoring, and storage.

Risks include copyright uncertainty, language imbalance, over-filtering, under-filtering, evaluation contamination, and source metadata loss.

Any larger version must add stronger source governance and legal review.

## Common Failure Modes

Common failures include boilerplate dominating extracted text, aggressive cleaning removing valid documents, unstable deduplication caused by inconsistent normalization, language detectors misrouting mixed-language pages, and training splits leaking duplicate records.

Debug the intermediate artifacts before changing the model or the training recipe.

## Reproducible Resource Notes

Reproducible resources should include WARC identifiers, minimal records, configs, filtering thresholds, split manifests, reports, and validation scripts.

The project should be reproducible without requiring a full Common Crawl mirror.

## 1. Project Background: Engineering Position of Mini-C4

C4-style corpora look simple from the outside.

Start from web pages, remove noise, and export clean text.

In practice, every stage changes the eventual training distribution.

Main-text extraction decides whether the corpus contains content or navigation residue.

Cleaning decides whether obvious junk is removed without deleting useful edge cases.

Deduplication decides whether memorization pressure and source dominance are controlled.

Language split decides whether multilingual data is judged by appropriate gates.

Packaging decides whether the final data can actually be consumed by training code.

Mini-C4 keeps the scale small so each gate can be inspected.

That makes it a teaching prototype for pre-training data factories.

## 2. Project Goals and Boundaries

### 2.1 Project Goals

The first goal is to parse WARC records into clean text records with source metadata.

The second goal is to apply explainable cleaning and near-deduplication.

The third goal is to split by language and apply language-aware quality filters.

The fourth goal is to package the remaining corpus into training, validation, and smoke-test files with manifests and checks.

### 2.2 Project Boundaries

The project does not implement a commercial-scale crawler.

It does not claim complete copyright clearance.

It does not include a full toxicity, safety, or contamination filter.

It does not solve all multilingual quality scoring.

These topics are outside the minimal teaching case.

### 2.3 Role of Boundary Setting

The boundary prevents overclaiming.

A good engineering case states what it can do under stable assumptions.

It also shows which controls must be strengthened before scale-up.

## 3. Overall Architecture

![Figure 1: Mini-C4 Data Pipeline Overview](../../images/part10/10_1_fig01_mini_c4_pipeline_overview.png)

### 3.1 Process Overview

The pipeline starts from Common Crawl WARC files.

It extracts HTML response records, decodes pages, removes boilerplate, and emits normalized text records.

It then applies heuristic cleaning, near-deduplication, language split, quality filtering, and deterministic packaging.

Finally, it writes reports and validation results.

### 3.2 Three-stage Interpretation

The first stage moves from web records to text.

The second stage moves from text to corpus.

The third stage moves from corpus to training interface.

This framing is useful because the failure modes differ across stages.

Parsing errors are not the same as quality errors.

Training packaging errors are not the same as cleaning errors.

## 4. Data Acquisition: Engineering Choices Around Common Crawl

Common Crawl is open, large, and noisy.

It is a good starting point precisely because it exposes the same problems as real web corpora.

The project should start from a bounded WARC slice.

A bounded slice makes iteration cheap.

It also keeps the relationship between input, output, and report visible.

The pipeline should preserve WARC path, URL, MIME type, timestamp, and record position when possible.

These fields support provenance, debugging, source balancing, and later legal review.

## 5. WARC Parsing and Main-text Extraction

### 5.1 Main-text Extraction as the First Critical Gate

Raw HTML contains headers, menus, scripts, comments, cookie banners, ads, and duplicated navigation.

If this gate fails, every later stage receives the wrong object.

The extractor should emit text plus metadata, not only text.

### 5.2 Core Component Choices

![Figure 2: WARC-to-text Parsing Path](../../images/part10/10_1_fig02_warc_to_text.png)

The project should use a streaming WARC reader, robust HTML decoding, boilerplate removal, and whitespace normalization.

Each rejected record should still contribute to stage statistics.

### 5.3 Engineering Value of Streaming

Streaming prevents large files from becoming memory bottlenecks.

It also makes checkpointing and distributed processing easier.

Each record can be processed independently.

### 5.4 Core Implementation

```python
def parse_warc_record(record):
    if record.rec_type != "response":
        return None
    html = decode_response(record)
    text = extract_main_text(html)
    if not text or len(text.split()) < 50:
        return None
    return {
        "url": record.rec_headers.get_header("WARC-Target-URI"),
        "text": normalize_whitespace(text),
        "source": "common_crawl",
    }
```

The important part is the contract.

Every accepted record has normalized text and source fields.

Every rejection should have a reason.

### 5.5 Meaning of This Stage's Result

This stage answers whether the pipeline can obtain useful text from raw web records.

It does not yet answer whether the text is high quality.

That distinction keeps later metrics interpretable.

## 6. Heuristic Cleaning: First-pass Noise Removal

![Figure 3: Heuristic Cleaning Rules](../../images/part10/10_1_fig03_cleaning_rules.png)

### 6.1 Why Heuristic Cleaning Is Necessary

Web text contains obvious junk.

Removing that junk before expensive filtering reduces downstream cost and makes later errors easier to diagnose.

### 6.2 Main Cleaning Rules Used in This Project

Useful rules include minimum length, maximum length, average word length, symbol density, punctuation density, blacklisted boilerplate phrases, repeated-line checks, and repeated n-gram checks.

Examples include cookie notices, newsletter prompts, login prompts, and copyright boilerplate.

### 6.3 Characteristics of Heuristic Cleaning

Heuristics are not intelligent.

Their value is explainability.

When a document is removed, the pipeline can say which rule fired.

This is more useful than a single opaque quality score at the first gate.

### 6.4 Interpreting This Stage

Cleaning should reduce obvious noise without pretending to solve corpus quality.

If the rejection rate is too high, inspect examples before tightening thresholds.

## 7. Deduplication: Near-duplicate Processing in Web Corpora

![Figure 4: MinHash + LSH Deduplication](../../images/part10/10_1_fig04_dedup_minhash_lsh.png)

### 7.1 How Serious the Duplicate Problem Is

Web corpora contain reposts, mirrored pages, syndicated articles, template variants, and pages that differ only in ads or navigation.

If duplicates remain, training overweights repeated sources.

### 7.2 Why Pairwise Comparison Is Avoided

Pairwise comparison is not feasible beyond toy scale.

The project uses signatures and buckets to find likely duplicates efficiently.

### 7.3 Engineering Intuition of MinHash and LSH

MinHash approximates set similarity.

LSH places similar signatures in candidate buckets.

The pipeline then removes near duplicates without comparing every pair.

```python
def minhash_signature(text, num_perm=128):
    shingles = make_word_ngrams(text, n=5)
    sig = MinHash(num_perm=num_perm)
    for shingle in shingles:
        sig.update(shingle.encode("utf-8"))
    return sig
```

### 7.4 Why Ray Is Used

Signature computation is highly parallel.

Ray makes it natural to shard documents and compute signatures across workers.

The project uses Ray to show distributed data engineering rather than single-script processing.

### 7.5 Common Pitfalls

The common pitfalls are unstable normalization, thresholds that are too aggressive, and lost metadata for removed records.

Deduplication should produce both kept records and a duplicate report.

### 7.6 Interpreting This Stage

Deduplication is not only a compression step.

It changes the training distribution.

The report should explain which sources and languages were most affected.

## 8. Language Split: Why Language-aware Processing Is Necessary

![Figure 5: Language Split and Branch Processing](../../images/part10/10_1_fig05_language_split.png)

### 8.1 Different Languages Should Not Share One Quality Gate

English stopword ratios, word-length assumptions, and punctuation rules do not map cleanly to Chinese or mixed-language pages.

A single gate can unfairly reject valid documents.

### 8.2 Project Method

The project applies language detection after initial cleaning and deduplication.

Records are routed into language branches with language-specific thresholds and statistics.

### 8.3 Language Split as a Required Intermediate Layer

Language split supports later mixture planning, tokenizer diagnostics, and language-aware sampling.

It is a data-routing layer, not only a dashboard statistic.

## 9. Quality Filtering: From Text-like to Training-suitable

![Figure 6: Quality Filtering Decision Gate](../../images/part10/10_1_fig06_quality_filter.png)

### 9.1 Why Quality Filtering Is the Most Important Gate

After parsing and cleaning, many records still look like text but are not good training material.

They may be spam, thin pages, repeated fragments, low-information lists, or machine-generated noise.

### 9.2 English Quality Gate: KenLM Perplexity

A small language model can provide a rough signal for whether a document resembles normal language.

Perplexity should be treated as one feature, not as an absolute truth.

It works best when combined with length, repetition, and language confidence.

### 9.3 Main Interception Reasons

Common rejection reasons include too little text, high repetition, abnormal symbol density, poor language confidence, and high perplexity.

The pipeline should keep rejected examples for threshold review.

### 9.4 What Retention-rate Differences Mean

Different retention rates across Chinese and English may reflect detector bias, source composition, or threshold mismatch.

The correct response is inspection and branch-specific tuning, not blind threshold reuse.

## 10. Three Experiment Iterations: How the Pipeline Stabilizes

![Figure 7: Three-iteration Path](../../images/part10/10_1_fig07_three_iterations.png)

### 10.1 Experiment 1: Main-text Extraction Only

The first run verifies whether WARC parsing and extraction work at all.

It exposes decoding failures, empty pages, and boilerplate problems.

### 10.2 Experiment 2: Add Heuristic Cleaning and Deduplication

The second run asks whether obvious noise and near duplicates can be removed with explainable rules.

It produces the first meaningful retention funnel.

### 10.3 Experiment 3: Add Language Split and Quality Filtering

The third run separates language branches and adds quality gates.

It moves the output closer to training readiness.

### 10.4 Engineering Meaning of the Three Iterations

The iterations show that corpus engineering is not one pass.

Each stage creates evidence for the next threshold choice.

## 11. Training Data Packaging: From Cleaning Results to Training Interface

### 11.1 Cleaning Is Not Training Readiness

A cleaned corpus is not automatically trainable.

Training code needs stable splits, file formats, schemas, manifests, and smoke-test data.

### 11.2 Importance of Deterministic Splitting

Splits should be deterministic and ID-based.

The same input and config should produce the same train, validation, and smoke sets.

### 11.3 Role of Smoke Test

The smoke set is small but important.

It lets training code verify schema, tokenization, and loading before consuming the full corpus.

### 11.4 Engineering Value of the Manifest

The manifest records counts, paths, thresholds, versions, and hashes.

It is the bridge between the data pipeline and training users.

## 12. Data Evaluation: Judging Pipeline Value

![Figure 8: Data Retention Funnel](../../images/part10/10_1_fig08_funnel.png)

### 12.1 Data Retention Funnel

The funnel should show scanned records, parsed records, cleaned records, deduplicated records, language branches, quality-filtered records, and final splits.

### 12.2 What These Numbers Really Mean

The final corpus size is less informative than where data was lost.

If most loss happens at extraction, parsing is the bottleneck.

If most loss happens at quality filtering, thresholds or input quality deserve review.

### 12.3 Data Profile

The report should include language distribution, length distribution, source distribution, rejection reasons, duplicate clusters, and token estimates.

These profiles make the corpus understandable.

## 13. Cost Analysis: Resource Accounting and Bottlenecks

![Figure 9: Resource and Cost Breakdown](../../images/part10/10_1_fig09_cost_breakdown.png)

### 13.1 Storage Cost

Storage grows across raw WARC, parsed text, cleaned text, signatures, reports, and final splits.

A scalable system should plan where intermediate files are retained and where they can be compacted.

### 13.2 Compute Bottlenecks

The main compute bottlenecks are HTML parsing, signature generation, and quality scoring.

Ray helps with parallelism, but sharding and checkpointing still matter.

## 14. Verification Loop: Project Consistency Checks

![Figure 10: Project Validation Loop](../../images/part10/10_1_fig10_validation_loop.png)

### 14.1 Role of Project Checks

Project checks prove that the data, manifest, and report agree.

They also catch empty files, split overlap, missing metadata, and missing reports.

### 14.2 Check Results

The expected state is not merely that scripts ran.

The expected state is that the generated artifacts pass consistency checks.

### 14.3 Check Coverage

Checks should cover file existence, non-empty outputs, required fields, split disjointness, language statistics, manifest counts, and report references.

### 14.4 Engineering Meaning of Verification

Verification turns the project from a demonstration into a reproducible case.

Without checks, a corpus pipeline is too easy to misread.

## 15. Main Limitations and Risks

### 15.1 Low Retention Rate

Low retention can be a healthy result if it removes noise.

It is a problem if useful content is being removed silently.

### 15.2 Low Chinese Retention

Lower Chinese retention may indicate thresholds designed around English features.

This should trigger branch-specific review.

### 15.3 Limited Deduplication Scalability

Prototype deduplication can show the method but may not be enough for web-scale data.

Larger runs need stronger sharding, signature storage, and cluster-level controls.

## 16. Future Extensions

### 16.1 Deduplication Backend Upgrade

Move from prototype LSH storage to scalable signature stores and distributed candidate-pair management.

### 16.2 Chinese Quality Model Upgrade

Use language-specific quality models and review sets instead of applying English-centric thresholds.

### 16.3 Upstream Domain Filtering

Introduce source priors and domain-level policies before expensive processing.

### 16.4 Observability Enhancement

Add per-stage metrics, dashboards, sample viewers, and drift alerts for long-running pipelines.

## 17. Engineering Practice Summary: Method Value of Mini-C4

![Figure 11: Mini-C4 Engineering Methodology Summary](../../images/part10/10_1_fig11_methodology_summary.png)

Mini-C4 teaches a general principle: expand the control surface before expanding data volume.

A real pre-training corpus factory competes on continuous production ability.

That ability depends on source governance, parsing quality, cleaning explainability, deduplication stability, language-aware filters, packaging contracts, and validation loops.

## 18. Main Deliverables

### 18.1 Intermediate Data Artifacts

- `raw_manifest.jsonl`
- `parsed_text.jsonl`
- `cleaned_text.jsonl`
- `deduped_text.jsonl`
- `language_split/`
- `quality_filtered/`

### 18.2 Training Data Artifacts

- `train.jsonl`
- `validation.jsonl`
- `smoke.jsonl`
- `corpus_manifest.json`

### 18.3 Report and Check Artifacts

- `quality_report.md`
- `dedup_report.json`
- `pipeline_check.py`
- validation result files

## 19. Closing

The value of Mini-C4 is not that it creates a large corpus.

Its value is that it makes a pre-training data factory observable.

Once extraction, cleaning, deduplication, language split, quality gates, manifests, and checks are explicit, scale becomes an engineering extension rather than a leap of faith.

## Special Topic: Acceptance Baselines for the Mini-C4 Pipeline

### 1. Crawling and Parsing Baseline

The first acceptance baseline is that every accepted record has source metadata and normalized text.

Parsing failure reasons should be counted.

Records without enough text should be rejected explicitly.

### 2. Cleaning and Deduplication Baseline

Cleaning rules should have named rejection reasons.

Deduplication should preserve a kept record and a report of removed near duplicates.

Threshold changes should be reviewable.

### 3. Language Split and Quality Scoring Baseline

Language routing should be visible.

Quality thresholds should be language-aware.

Borderline examples should be kept for review.

## Special Topic: From Teaching Prototype to Large-scale Pre-training Factory

### 1. Expand Control Surfaces Before Data Volume

Scaling without controls only scales confusion.

A larger run should first strengthen source governance, observability, checkpointing, and validation.

### 2. Pre-training Corpus Factories Need Ingestion Governance

The most important decisions often happen before cleaning.

Source allowlists, blocklists, crawl-time metadata, and copyright status should enter the pipeline early.

### 3. The Final Competition Is Continuous Production Ability

A one-time corpus can be impressive.

A production corpus factory must update, audit, re-run, and compare versions continuously.

## Special Topic: Corpus Mixture and Training Mix Strategy

### 1. Preserve Mixture Information During Data Preparation

Language, domain, source, quality score, and filtering reason should survive into manifests.

Otherwise training teams cannot design mixture ratios later.

### 2. Training Mix Strategy Continues Quality Control

Sampling weights are another quality gate.

They should be based on profiles and evaluation results, not only record counts.

## Chapter Summary

This chapter used Mini-C4 to show how a web-corpus project becomes a reproducible data engineering pipeline.

The project connects WARC parsing, main-text extraction, cleaning, near-deduplication, language split, quality filtering, packaging, and validation.

Its boundary should stay clear: it is a teaching-scale prototype, not a complete production corpus platform.

Readers can combine this case with later chapters on data quality, DataOps, and open-source model recipes to build larger corpus factories.

## Release Review Notes

Before a Mini-C4 release, the team should review input scope first.

The review should list the WARC files, crawl dates, and source constraints.

It should also state whether any source category was intentionally excluded.

The second review item is extraction quality.

A reviewer should inspect accepted pages, rejected pages, and borderline pages.

This makes it possible to separate extraction failure from later quality filtering.

The third review item is heuristic cleaning.

Every cleaning rule should have a name.

Every rule should have example records that it removed.

If a rule removes many documents, the team should inspect whether it is acting as intended.

The fourth review item is deduplication stability.

Deduplication should not only report the final kept count.

It should also report representative duplicate clusters.

Those clusters tell the team whether MinHash and LSH are catching real near-duplicates or unrelated documents.

The fifth review item is language routing.

The team should inspect several records per language branch.

Mixed-language records should be tracked separately if they are common.

Language routing errors often explain strange quality-filter results.

The sixth review item is quality-score calibration.

Perplexity, repetition, and length rules should be read together.

A high perplexity score alone should not be treated as a universal rejection reason.

The seventh review item is split integrity.

Train, validation, and smoke sets should be disjoint by document ID and by deduplication cluster when possible.

This prevents near-duplicate leakage into validation.

The eighth review item is manifest accuracy.

The manifest should match the real files on disk.

Counts, hashes, thresholds, and versions should be checked before delivery.

The ninth review item is manual sampling.

A small release should still include human review examples.

Those examples help future maintainers understand why thresholds were chosen.

The tenth review item is rollback.

If a quality gate is found to be too strict or too loose, the team should know which intermediate artifact allows rerun.

Without rollback points, a corpus factory becomes hard to maintain.

## Operating Notes

Daily operation should begin with stage-level metrics rather than final token count.

If parsed records suddenly drop, inspect WARC decoding and HTML extraction.

If cleaned records suddenly drop, inspect rule-level rejection distribution.

If deduplication removes too much, inspect signature thresholds and text normalization.

If one language branch collapses, inspect detector confidence and branch thresholds.

If smoke data fails to load, inspect schema and split manifests before training code.

Large-scale versions should add checkpointing between stages.

They should also preserve rejected sample ledgers.

Rejected sample ledgers are useful because they make threshold tuning empirical.

They also support audits when users ask why a source disappeared.

The project should treat corpus construction as versioned production.

Each version should record source scope, thresholds, retention funnel, known defects, and planned fixes.

That version language turns Mini-C4 from a class exercise into a corpus factory pattern.

## Final Acceptance Checklist

Input WARC scope is recorded.

Source metadata is preserved.

Extraction failures are counted.

Accepted records are sample-reviewed.

Rejected records are sample-reviewed.

Cleaning rules have names.

Cleaning thresholds have examples.

Deduplication clusters are inspectable.

Language branches are counted.

Mixed-language pages are tracked.

Quality scores are decomposed.

Borderline samples are retained.

Train and validation are disjoint.

Smoke data loads successfully.

Manifest counts match files.

Reports mention known limitations.

Rollback points are documented.

Rejected ledgers are retained.

Version notes explain threshold changes.

The release can be reproduced from config.

Quality gates have owner notes.

Language thresholds have owner notes.

Dedup thresholds have owner notes.

Sampling ratios are documented.

Future scaling risks are listed.

Contamination-check status is recorded.

Source-review status is recorded.

Token-estimation method is documented.

## References

1. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.
2. Common Crawl Foundation. Common Crawl datasets and WARC documentation.
3. Broder, A. Z. (1997). On the resemblance and containment of documents.
4. Leskovec, J., Rajaraman, A., & Ullman, J. D. (2020). Mining of Massive Datasets.
5. Ray Project. Ray documentation for distributed data processing.
