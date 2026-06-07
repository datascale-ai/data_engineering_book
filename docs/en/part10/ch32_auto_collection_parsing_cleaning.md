# Chapter 32: Automated Collection, Parsing, and Cleaning Agents

## Chapter Abstract

The most tedious and time-consuming work in data engineering often happens at the moment data enters the system: how to collect it, how to handle unusual formats during parsing, who maintains cleaning rules, and who decides whether quality is acceptable. When data sources grow from dozens to hundreds, manual collection strategies and cleaning rules collapse. Rule-maintenance backlogs grow without bound, or quality gates become decorative.

The purpose of collection and cleaning agents is not to replace data engineers. It is to free them from repetitive rule adaptation so they can focus on exception handling, rule improvement, and architecture design.

This chapter starts from four common source types: web pages, PDFs, APIs, and code repositories. It explains how agents can detect source structure, handle parsing exceptions, generate cleaning rules, run quality filters, and trigger human review when uncertainty is high. The central question is how an agent should decide under uncertainty: skip, repair, or escalate after parsing failure; choose between conflicting cleaning rules; and set quality thresholds that do not destroy either data volume or model quality.

The chapter builds on Chapter 4's data acquisition, Chapter 5's cleaning and deduplication, Chapter 6's input pipelines, and Chapter 2's quality framework, upgrading traditional practice into agent-driven adaptive pipelines.

## 32.0 Learning Objectives

After reading this chapter, you should be able to:

- Design agent-driven collection architecture for heterogeneous multi-source data.
- Apply agentic collection strategies to web pages, PDFs, APIs, and code repositories.
- Build automatic repair and rule generation mechanisms based on parsing exceptions.
- Construct multi-layer quality filtering pipelines and decide when to auto-pass or trigger review.
- Evaluate the cost-benefit ratio and exception-handling efficiency of cleaning agents in different scenarios.

## Scenario: The Nightmare of Source Drift

An AI team needs to build a training dataset for a legal large model. Sources include a court judgment website, regulation PDFs, judicial APIs, and open-source legal knowledge repositories. The team initially assigns five engineers full-time to collection and cleaning. After three months:

**Web collection line.** The target website changes layout three times. Each change breaks more than 70 percent of crawler rules. Engineers spend about two days repairing rules each time while the pipeline sits idle.

**PDF parsing line.** More than 2,000 PDF files come from over 30 courts, each with different layouts. Engineers write special parsing rules for many layouts, but about 15 percent of files still suffer field misalignment.

**API collection line.** The judicial API changes field structure four times in two months. Each change breaks downstream cleaning and takes about one day to fix.

**Code repository line.** Open-source legal knowledge bases have different directory structures, file formats, and encodings. Even after manual adaptation, encoding errors remain common.

The problem is not that engineers are not working hard enough. The growth rate of rule maintenance has exceeded human processing capacity. As source count grows, maintenance cost is O(N) or even O(N^2), while human capacity is fixed. The team needs an agent system that can adapt to source-structure changes.

### Core Engineering Pain Points

1. **Source-structure drift.** Web structures, API schemas, and file formats change over time.
2. **Diverse parsing exceptions.** PDF layout variation, encoding chaos, and special characters cannot all be covered by one generic rule set.
3. **Cleaning-rule conflicts.** Different sources may define the same field differently, such as `yyyy-MM-dd` versus `dd/MM/yyyy`.
4. **Quality-filtering granularity.** Overly strict filters reduce data volume; loose filters fail training quality requirements.

## 32.1 Agent-Driven Collection Architecture

### 32.1.1 Unified Abstraction for Four Source Types

Collection from all sources can be abstracted into three layers: **connection layer -> extraction layer -> structuring layer**. The agent plays different roles at each layer.

```mermaid
flowchart LR
    W["Web pages"] --> C["Connection adapters<br/>connect, fetch, cache"]
    P["PDF files"] --> C
    A["APIs"] --> C
    R["Code repositories"] --> C
    C --> E["Extraction layer<br/>format detection, parsing, mapping"]
    E --> S["Structuring layer<br/>source_id, timestamp, content, metadata, quality_score"]
    S --> Q["Cleaning and quality agents"]
```

*Figure 32-1: Unified architecture for four-source collection agents*

Although web pages, PDFs, APIs, and repositories use different access methods, agent logic can be unified. The abstraction hides source differences behind adapters in the connection layer. Extraction and structuring share common logic.

**Connection-layer adapters.** Each source type has an adapter responsible for connection setup, authentication, data retrieval, and raw-data caching. The interface is uniform: `connect() -> fetch() -> cache_raw()`. Upper layers do not need to know whether the source is a web page, PDF, or API.

**Extraction-layer process.** Regardless of source, extraction follows the same sequence: raw data -> format identification -> content extraction -> structure mapping -> intermediate output. Differences appear in format identification and extraction: web pages need DOM parsing, PDFs need layout analysis, and API responses need JSON Schema validation.

**Structuring-layer output.** All sources should emit a unified schema with at least `source_id`, `extraction_timestamp`, `content_fields`, `metadata_fields`, and `quality_score`. This allows downstream cleaning and quality agents to handle all sources consistently.

### 32.1.2 Collection Task Generation and Failure Retry

The first responsibility of a collection agent is **automatic task generation**. In the traditional mode, engineers write collection configs manually for every source. In the agent mode, users describe data needs and the agent performs these steps:

1. **Source probing.** The agent lightly probes the source, identifies type, estimates volume, and detects access limits.
2. **Strategy generation.** It generates pagination, frequency control, concurrency, and retry strategy.
3. **Provenance capture.** Every collection records URL, collection time, HTTP headers, and file hash. This matters for compliance audit and copyright tracing.

*Table 32-1: Collection failure categories and retry strategies*

| Failure type | Detection | Retry strategy | Max retries | After limit |
| --- | --- | --- | --- | --- |
| Network timeout | HTTP status or exception | Exponential backoff: 1s, 2s, 4s, 8s | 5 | Mark temporarily unavailable; retry after 1 hour |
| Anti-crawling block | 403 or CAPTCHA page | Switch User-Agent or IP pool | 3 | Escalate to human |
| Structure change | Missing parsed fields > 50% | Trigger structure reprobe | 2 | Generate structure-change alert |
| Encoding error | Garbled-text detection | Detect encoding and transcode | 3 | Keep raw bytes and mark for review |
| Abnormal volume | Collected amount deviates > 3 sigma from history | Pause collection and report | 1 | Resume after human confirmation |

Retry strategy must be polite. Aggressive retry can worsen access blocks or create legal and operational risk.

### 32.1.3 Scheduling and Rate Limiting at Scale

When an agent manages hundreds of sources, scheduling is no longer simple cron execution. It becomes a constrained optimization problem: maximize throughput and freshness while respecting access limits.

*Table 32-2: Collection scheduling constraints*

| Constraint | Meaning | Example |
| --- | --- | --- |
| Frequency limit | Request-rate limit of target website or API | At most 10 requests per second |
| Concurrency limit | Maximum simultaneous collection tasks | At most 50 concurrent tasks |
| Time window | Accessible time window | Weekdays 08:00-20:00 |
| Volume estimate | Estimated volume and duration per run | Full collection 2 hours; incremental 10 minutes |
| Priority | Downstream importance | Training source > evaluation source > experimental source |

The scheduler should support three modes:

**Periodic collection.** For predictable updates, such as daily news or weekly reports. The agent learns update patterns and adjusts when real frequency deviates.

**Event-triggered collection.** For sources requiring fast response, such as API schema change notices or web-structure alerts. The agent subscribes to webhooks, RSS, or monitoring alerts and triggers incremental collection.

**Adaptive collection.** For unpredictable sources. The agent probes at low frequency, such as every six hours, and adjusts based on observed updates. Repeated no-update probes lower frequency; detected updates raise it.

### 32.1.4 Special Handling for Enterprise Documents and Databases

Two enterprise source classes require special strategy.

**Enterprise documents such as SharePoint, Confluence, and DingTalk Docs.** These platforms include versioning, permissions, and rich text. Agents must authenticate through platform APIs, check permissions, decide whether to collect latest or historical versions, parse tables, images, attachments, and embedded content, and follow enterprise classification and access-control policies.

**Direct database collection.** Many core enterprise datasets live in relational databases. Agents should generate extraction queries from schemas, support incremental collection through timestamp or version fields, limit production impact by using read replicas and off-peak windows, and handle cross-database joins when training data requires multiple tables.

## 32.2 Parsing Repair Agents

### 32.2.1 Detecting and Attributing Parsing Exceptions

The core ability of a parsing agent is not merely parsing successfully. It is knowing why parsing failed and what to do next.

Parsing exceptions fall into three types:

**Structural exceptions.** HTML DOM changes break selectors, PDF layout changes break region recognition, or API fields are added or removed.

**Encoding exceptions.** Declared encoding does not match actual encoding, files mix encodings, or special-character escaping fails.

**Semantic exceptions.** Parsed values are syntactically valid but semantically impossible, such as "month 13 day 45" or a negative amount in a business context where negative amounts cannot occur.

```mermaid
flowchart TD
    F["Parsing failure"] --> T{"Exception type"}
    T -->|Structural| R["Reprobe structure<br/>or generate extraction rule"]
    T -->|Encoding| E["Detect encoding<br/>transcode or preserve raw bytes"]
    T -->|Semantic| S["Run business rules<br/>mark semantic uncertainty"]
    R --> C{"Confidence high?"}
    E --> C
    S --> C
    C -->|Yes| V["Validate and write lineage"]
    C -->|No| H["Escalate to human review"]
```

*Figure 32-2: Parsing exception handling decision flow*

### 32.2.2 Parser Selection and Repair Rule Generation

After parsing failure, the agent should choose among several strategies rather than only repair or skip:

1. **Fallback parser chain.** PDFs may try PyMuPDF -> pdfplumber -> Tesseract OCR. HTML may try lxml -> html5lib -> BeautifulSoup.
2. **Automatic extraction rule generation.** When fixed selectors fail, the agent can generate rules from semantic features, such as "text that looks like a date" or "line containing the keyword case number."
3. **Partial repair.** If only some fields fail, the agent should preserve correctly parsed fields and repair only abnormal fields.

### 32.2.3 Parsing Multimodal Content

Modern sources often mix text, images, tables, screenshots, and rich layouts.

**Image-text mixed content.** For images in documents, the agent decides whether to keep references, run OCR, generate descriptions, or ignore images. The decision depends on downstream tasks. Multimodal model training may require preserving images; text-only training may need only extracted text or captions.

**Robust table parsing.** Tables often break parsers because of merged cells, nested headers, cross-page tables, and borderless layouts. A good strategy is multi-parser voting plus human fallback:

1. Use two or three parsers based on different signals: borders, whitespace alignment, and semantic rules.
2. Compare results. If row/column counts and headers match, accept with high confidence.
3. If results conflict, mark parsing as uncertain and submit for review.

**OCR conditions and quality control.** OCR has extra compute and latency cost. The agent should trigger OCR only when direct text extraction has character recognition below 85 percent, the document contains non-copyable text, and estimated OCR time fits the task budget.

OCR output also needs quality checks. Compare recognized text with known metadata such as title, author, and date. If recognition of these anchor fields is below threshold, mark the document as low-quality OCR and recommend human handling.

### 32.2.4 Systematic Handling of Multilingual and Encoding Issues

Data engineering agents face two language-related challenges.

**Language identification and routing.** The agent must identify document language during parsing to support later cleaning and labeling. This is not a simple Chinese/English decision. A single text may contain Chinese, English, code, and mathematical formulas.

**Encoding detection and repair.** Encoding errors are persistent. A file may declare one encoding while containing another, or mix multiple encodings. The agent should:

1. Read raw bytes and detect actual encoding using tools such as chardet or cChardet.
2. If detected encoding differs from declared encoding, decode with the detected encoding.
3. If some bytes cannot be decoded reliably, preserve raw bytes, mark the location and context as uncertain, and avoid silent replacement.
4. Record the entire process in Lineage: declared encoding, detected encoding, transcoding result, and undecodable-byte ratio.

### 32.2.5 Performance Optimization for Collection and Cleaning Agents

At hundreds of sources and TB-scale daily data, performance is necessary for survival.

**Incremental collection and change detection.** Full collection cost grows linearly with data volume. Agents should prefer incremental collection. APIs can use `Last-Modified` headers or `updated_at` fields; web pages can compare page hashes; PDFs can compare file hashes or version numbers.

**Parallelism and resource pools.** Agents should maximize parallelism within resource limits. Each source type can have its own pool so one source type does not starve others. When a pool exceeds 80 percent utilization, low-priority tasks should move to off-peak windows.

**Local cache and deduplication.** Recollecting identical content wastes compute and storage. Before collection, the agent should check content hashes in cache. Cache duration should follow freshness requirements: news may cache for hours, regulations for days.

## 32.3 Cleaning Rule Generation Agents

### 32.3.1 From Sampled Defects to Rule Candidates

In traditional cleaning-rule maintenance, humans write rules while data determines when rules are needed. Engineers often learn that a rule is missing only after downstream users report a problem. Agents can reverse this:

1. **Defect discovery.** Quality scans or downstream feedback identify failing fields and subsets.
2. **Pattern extraction.** The agent analyzes whether defects come from a source, a format pattern, or a specific transformation path.
3. **Rule candidate generation.** It proposes scope, transformation logic, and expected effect.
4. **Sandbox validation.** The rule runs against historical data in an isolated environment and produces a diff report.
5. **Human approval.** The data owner approves before production release.

### 32.3.2 Sandbox Validation and Diff Reports

Sandbox validation is the last defense before rule release. The sandbox should satisfy:

- **Data isolation.** Use a production snapshot; never affect production data.
- **Full validation.** Run against all matching data, not only samples, to catch edge cases.
- **Diff visualization.** Show before/after comparisons and highlight modified fields and modification volume.

*Table 32-3: Sandbox validation dimensions and pass conditions*

| Dimension | Check | Pass condition | If failed |
| --- | --- | --- | --- |
| Rule match scope | How many rows are affected | Affected rows within expected range, deviation < 20% | Adjust scope |
| Modification correctness | Whether modified values are legal | 100% format pass, > 95% semantic pass | Fix rule logic |
| Side-effect detection | Whether non-target fields changed | Non-target modification rate is 0% | Tighten rule conditions |
| Performance evaluation | Runtime on large data | Runtime < 150% of estimate | Optimize or batch execution |

## 32.4 Quality Filtering and Human Review Triggers

### 32.4.1 Routing by Quality Uncertainty

The hardest part of quality judgment is not deciding right or wrong. It is knowing how uncertain the agent is. A well-designed agent should request human help under uncertainty rather than forcing a possibly wrong decision.

*Table 32-4: Quality uncertainty routing*

| Quality dimension | High certainty: auto-pass | Medium certainty: mark and sample review | Low certainty: human review |
| --- | --- | --- | --- |
| Format correctness | Regex/constraints match 100% | Match rate 80%-99% | Match rate < 80% |
| Field completeness | Missing rate < 1% | Missing rate 1%-5% | Missing rate > 5% |
| Semantic reasonableness | All business rules pass | One or two rule warnings | Three or more warnings or blocking rules |
| Cross-source consistency | Same field values match across sources | Minor differences, such as formatting | Substantive differences |
| Distribution stability | Deviation from history < 2 sigma | Deviation 2-3 sigma | Deviation > 3 sigma |

### 32.4.2 Human Review Triggers

The following conditions must trigger human review:

1. **Low-confidence quality judgment.** Agent confidence is below threshold.
2. **High-risk field modification.** Amount, date, primary key, foreign key, and similar critical fields are affected.
3. **Rule conflict.** Two rules produce different cleaning suggestions for the same data.
4. **First-time source type.** The first batch from an unseen source type requires sampling review.
5. **Quality metric shock.** A batch's quality metric drops more than 20 percent versus the previous batch.
6. **Repeated repair failure.** The agent fails verification three times on the same batch.

### 32.4.3 Automated Quality Filtering Pipeline

Quality filtering should follow "wide entry, strict exit, tiered filtering." Each layer handles one quality dimension. Failed data is routed to the appropriate handling path rather than simply discarded.

```mermaid
flowchart LR
    I["Incoming structured data"] --> F["Format validation"]
    F --> C["Completeness validation"]
    C --> S["Semantic validation"]
    S --> X["Consistency validation"]
    X --> P["Pass to downstream"]
    F -->|fail| R1["Format repair queue"]
    C -->|warn/fail| R2["Missingness handling queue"]
    S -->|warn/fail| R3["Semantic review queue"]
    X -->|drift/conflict| R4["Cross-source review queue"]
```

*Figure 32-3: Tiered quality filtering pipeline*

Thresholds should be configurable by business need and data characteristics:

- **Format validation layer:** strict mode. Format mismatch rejects downstream entry.
- **Completeness validation layer:** flexible mode. Small missing rates are marked; above-threshold missingness alerts.
- **Semantic validation layer:** rule mode. Validate against predefined business rules.
- **Consistency validation layer:** comparison mode. Compare across sources, history, and upstream/downstream dependencies.

### 32.4.4 Priority and SLA Management for Human Review

*Table 32-5: Human review priority and SLA management*

| Priority | Trigger | SLA | Timeout action |
| --- | --- | --- | --- |
| P0 | Affects key downstream pipeline, such as model training export | Within 1 hour | Escalate and pause related pipelines |
| P1 | Low certainty plus high-risk field | Within 4 hours | Escalate |
| P2 | First-time source type | Within 24 hours | Mark as pending confirmation, allow with traceability |
| P3 | Minor quality metric fluctuation | Within 72 hours | Add to next quality review |

## 32.5 Case Review: Agentized Multi-Source Legal Data Collection and Cleaning

A legal AI team transforms its manual collection process into an agent-driven adaptive pipeline.

**Stage 1: collection agent launch.** The agent probes more than 50 source structures, generates collection strategies, and executes. In the first month, recovery time after web rule failure drops from two days to four hours. PDF parsing accuracy improves only from 85 percent to 90 percent.

**Stage 2: parsing repair agent.** The team configures a three-layer PDF parser fallback chain: PyMuPDF -> pdfplumber -> OCR, and trains layout recognition for 30 court formats. Parsing accuracy rises to 96 percent, but OCR fallback takes 20 times longer than normal parsing and creates batch timeouts.

**Stage 3: cleaning-rule generation and quality routing.** The agent generates over 40 cleaning-rule candidates from sampled defects; 35 pass sandbox validation and are released. Quality uncertainty routing reduces daily human review from over 500 documents to 80, focusing review on genuinely disputed cases.

### Key Metric Changes

| Metric | Before: manual | After: agent-assisted | Change |
| --- | --- | --- | --- |
| New source onboarding time | 3-5 days | 4-8 hours | -80% |
| Parsing accuracy | 85% | 96% | +11% |
| Rule maintenance staffing | 5 full-time engineers | 1.5 full-time engineers | -70% |
| Human review volume | 100% full review | 15% after routing | -85% |
| Data release cycle | 2 weeks | 3 days | -78% |

### Extended Case: Productionizing Multi-Source Collection at an AI Company

Another AI company collects training data from more than 200 sources, including news, papers, technical blogs, government open data, and social media. Its production agent faces three typical problems.

**Escalating anti-crawling defenses.** Two months after launch, about 30 percent of news sites update anti-crawling policies. The agent's automatic retry triggers stricter blocking, and some high-frequency retry IPs are blacklisted. Lesson: retry must include polite backoff. Aggressive retry can make the relationship worse.

**Hidden data drift.** An academic paper site redesigns its frontend. The visual page looks similar, but HTML changes completely. The agent monitors missing parsed fields, yet still extracts text. It accidentally mixes main content with sidebar recommendations, so missingness does not spike, but semantic quality silently falls.

**Cost runaway.** OCR calls triple in the third month because the agent sends every uncertain PDF to OCR. Cloud cost grows from $2,000 to $6,000 per month. High-cost operations need budgets and quotas.

Countermeasures:

- Maintain an anti-crawling friendliness score for each source and prefer official APIs or partnerships for difficult sources.
- Add content drift detection beyond missingness, monitoring semantic distribution changes in extracted text.
- Add operation cost quotas for OCR and model calls, escalating to human approval when quotas are exceeded.

### Case Lessons

1. **Collection agents must be polite.** Respect target access limits and avoid legal or IP-block risk.
2. **Quality monitoring cannot rely only on missingness.** Semantic drift is harder to detect and more damaging.
3. **Automation is not free.** Every operation has compute cost; budget controls belong in the design.
4. **Provenance is not only compliance.** Complete source records make debugging and trust-building possible.

## 32.6 Checklist: Collection and Cleaning Agent Deployment

- [ ] Does the system support a unified abstraction for web, PDF, API, and repository collection?
- [ ] Does retry strategy distinguish network, anti-crawling, structure, and encoding failures?
- [ ] Does each source type have a parser fallback chain?
- [ ] Does parsing exception detection cover structural, encoding, and semantic dimensions?
- [ ] Are cleaning rules sandbox-validated with diff reports before production release?
- [ ] Does sandbox validation check match scope, correctness, side effects, and performance?
- [ ] Is there a three-tier quality uncertainty model: auto-pass, mark, human review?
- [ ] Do low-certainty cases, high-risk fields, rule conflicts, and first-time source types trigger review?
- [ ] Is provenance recorded for every collection event: URL, time, and hash?
- [ ] Does abnormal quality movement automatically pause collection?

## 32.7 Chapter Links

- **Chapter 2:** data quality framework for filtering and uncertainty routing.
- **Chapter 4:** data collection foundation extended here with agent-driven adaptive collection.
- **Chapter 5:** cleaning and deduplication upgraded here into agent-generated cleaning rules.
- **Chapter 6:** input pipelines downstream of collection.
- **Chapter 31:** six-layer architecture applied to collection and cleaning.
- **Chapter 33:** labeling, synthesis, and evaluation agents consume cleaned outputs.

## 32.8 Further Reading: Engineering Practices for Collection and Cleaning Agents

### Collection Ethics and Compliance Boundaries

Agent-driven collection amplifies traditional compliance issues. If speed and coverage increase by 10x, previously small risks become large.

**Automatic robots.txt compliance.** Before web collection, the agent must check `robots.txt` and respect crawling rules. Exceptions require explicit approval records.

**Copyright and usage-right detection.** The agent should collect not only data but also usage conditions: Terms of Service, API terms, and repository licenses. Unclear permissions should be marked for legal review.

**Load control on target servers.** The agent should monitor response time and lower collection speed if its own traffic appears to harm service.

### Provenance and Reproducibility

Any training data batch should be regenerable. This requires:

1. **Collection configuration snapshot.** Save source URLs, parser rule versions, and cleaning rule versions for every run.
2. **Data fingerprint.** Compute content hashes for each batch.
3. **Version linkage.** Store data batches together with configuration snapshots and fingerprints so every batch can be traced to its production process.

### Versioning and Rollback for Cleaning Rules

Agent-generated cleaning rules need version management:

- Assign semantic versions to rules on release.
- Keep complete change records: editor, time, content, and reason.
- On rollback, roll back the rule and mark affected data batches for reprocessing.

## References

- Apache Nutch Documentation. https://nutch.apache.org/
- PyMuPDF (fitz) Documentation. https://pymupdf.readthedocs.io/
- Great Expectations: Data Quality Framework. https://greatexpectations.io/
- crawlee: Web scraping and browser automation. https://crawlee.dev/
