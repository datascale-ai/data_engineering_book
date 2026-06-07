# Appendix A: Tools and Frameworks Quick Reference

## A.1 Purpose of This Appendix

This appendix supports the engineering implementation layer of the book. It is not trying to list every tool you may have heard of. It answers a more practical question: when a team has decided to collect data, clean it, evaluate it, track experiments, release assets, and support teaching reproduction, how should it choose tools at different stages and understand the responsibility boundaries between them?

In real projects, tool selection usually fails not because one framework is wrong, but because problems from different layers are mixed together. A team that lacks version governance may try to solve it by adding another data-processing script. A team that lacks reviewable evaluation may keep spending budget on training. A team that lacks teaching images and course environments may assume that publishing a repository is the same as making the project reproducible.

This appendix therefore organizes tools by the data-engineering lifecycle rather than by popularity or vendor name.

Keep three principles in mind. First, **tools do not replace process**. If the process is unclear, changing tools usually only moves confusion into a new interface. Second, **interfaces and boundaries matter most in data engineering**: acquisition, cleaning, annotation, evaluation, release, and feedback must each have clear responsibilities. Third, **the closer a project is to open benchmarks or university collaboration, the more it should prioritize auditability, handoff, and teachable reproducibility**, not only one-off throughput or benchmark scores.

## A.2 Tool Overview by Lifecycle

Table A-1 gives a coarse but directly usable mapping. It is not the only correct answer, but it helps teams put the problem at the right layer first.

| Stage | Main Goal | Common Deliverables | Priority Tool Categories | Typical Risk |
| :-- | :-- | :-- | :-- | :-- |
| Data ingestion | Turn external sources into manageable inputs | Crawl outputs, raw files, ingestion logs | Crawling frameworks, connectors, object storage | Sources cannot be traced; collection criteria drift |
| Data cleaning | Denoise, deduplicate, standardize, decontaminate | Cleaning rules, abnormal sample pools, quality reports | Batch-processing frameworks, rule engines, quality validators | Cleaning is not reproducible; high-value samples are deleted |
| Annotation and augmentation | Create trainable supervision signals | Annotation tasks, QA workflows, augmentation artifacts | Annotation platforms, review systems, synthetic generation frameworks | Guideline drift; review chain breaks |
| Training preparation | Package data into model-consumable formats | JSONL/Parquet/Arrow, splits, indexes | Data-format libraries, tokenization and packing tools | Training and evaluation inputs diverge |
| Evaluation attribution | Build comparable experiments | Metric scripts, slice reports, attribution ledgers | Evaluation frameworks, experiment tracking, dashboards | Only averages are checked; evidence is not preserved |
| Open release | Create reusable assets | Cards, licenses, baseline bundles | Data cards, model cards, version-release tools | Versions cannot roll back; baselines become invalid |
| Teaching and reproduction | Let others reproduce work stably | Teaching images, experiment instructions, fixed versions | Containers, environment managers, course repositories | Environments drift mid-semester; scripts fail |

The table's most important role is not to decide the tool immediately. It is to confirm what kind of problem is being solved. Many debates become easier once they move from "which framework do I prefer?" to "what deliverable must this stage produce?"

## A.3 Data Ingestion, Storage, and Version Governance Tools

### A.3.1 Ingestion Tools Standardize Sources First

If data sources span webpages, document repositories, enterprise tables, object storage, and third-party open datasets, manual download plus ad hoc scripts will not remain sustainable. The ingestion layer should fix **where data comes from, when it comes in, and under what rules**.

| Category | Representative Tools / Frameworks | Scenarios | Strengths | Extra Concerns |
| :-- | :-- | :-- | :-- | :-- |
| Web crawling | Scrapy, Trafilatura | Public web text, news, knowledge bases | Mature ecosystem, easy customization | robots, copyright, update frequency |
| API/DB connectors | Airbyte, Fivetran-style connectors | SaaS, databases, internal business sources | Standardized ingestion, easier incrementality | Field-change management, permission minimization |
| Document import | Unstructured, Apache Tika | PDF, Office, scanned documents | Unified document entry | OCR errors, layout parsing bias |
| Object-storage access | S3/OSS/MinIO SDKs | Images, audio, video, large files | Suitable for lakehouse and offline processing | Lifecycle policies and cost control |

For large-model data engineering, the ingestion layer should retain two records. The **raw-entry record** explains source, retrieval time, authorization scope, and filters. The **engineering-entry record** explains format conversion, anonymization, slicing, and sampling before downstream use. Without these records, cleaning, evaluation, and compliance boundaries become hard to explain.

### A.3.2 Storage Is About Layering, Not Just Size

Many teams initially put everything into one object-storage bucket. This is convenient short term, but soon nobody can tell which layer is raw, which layer is cleaned, or which version downstream systems consumed. Storage must have at least basic layering.

| Layer | Recommended Carrier | Description |
| :-- | :-- | :-- |
| Raw | Object storage saved as-is | Preserve the original source; avoid overwriting |
| Staging | Parquet/JSONL/Arrow intermediate formats | For cleaning, sampling, and quality checks |
| Curated | Trainable/evaluable standard sets | Official versions for training and evaluation |
| Release | Release package, card, baseline bundle | For external use or course reproduction |

Object storage stores files but does not automatically provide version semantics. If the team expects long-term evolution, add versioning tools such as `DVC`, `lakeFS`, or a data-lake solution with snapshots. The value is practical: teams can answer which data version an experiment consumed, whether a public leaderboard maps back to a specific split, and whether a course environment is locked to the release-time version.

### A.3.3 Version-Governance Tools Are Team Memory Systems

`Git` is excellent for text configuration and scripts, but it is not designed to directly manage large data assets. Data engineering often uses `Git` as the control plane and a large-file version system as the data plane.

| Tool | Best For | Typical Use |
| :-- | :-- | :-- |
| Git | Code, configuration, schema, documentation | Process definitions, evaluation scripts, release notes |
| Git LFS | Medium-sized binaries | Small model files, sample data |
| DVC | Large-file and data-version references | Dataset versions, experiment-input binding |
| lakeFS | Branches and commits over object storage | Lakehouse-style data governance and collaboration |
| Delta Lake / Apache Iceberg | Large tabular data governance | Large-scale structured samples and metadata |

For cross-institution dataset construction, public evaluation, and teaching reproduction, a minimal combination is often enough: **Git for scripts and specifications, DVC or an equivalent for data versions, object storage for large files, and release pages for external documentation**.

## A.4 Cleaning, Validation, and Training Preparation Tools

### A.4.1 Choose Batch-Processing Frameworks by Data Form

Cleaning tools often fall into the habit of using one big framework for everything. Text, tables, document images, audio, and agent trajectories have very different processing needs. Split by data form.

| Data Form | Recommended Processing Mode | Common Tools |
| :-- | :-- | :-- |
| Large-scale line-level text | Batch map/filter/reduce | Spark, Ray Data, Beam |
| Documents and tables | Streaming extraction plus structural validation | Unstructured, Pandas, Arrow |
| Multimodal samples | Metadata batch processing plus file references | Ray, PyArrow, object-storage indexes |
| Audio and video | Offline transcoding and feature extraction | FFmpeg, torchaudio, decord |
| Agent trajectories | Structured event streams and replay | JSONL, Parquet, custom validators |

Spark is mature and stable for heavy batch processing and enterprise platforms. Ray Data is closer to Python, model inference, and multimodal processing. Beam is useful when unified batch/stream semantics matter. For many book projects, labs, and courses, the main bottleneck is not whether the system is distributed enough; it is unclear data contracts, unstable fields, and missing recovery paths for abnormal samples.

### A.4.2 Quality Validation Should Explain Failures

Quality validation is not about achieving zero errors. It is about classifying errors and creating write-back actions. Frameworks such as `Great Expectations` are useful for structured rules, while documents, multimodal samples, and reasoning data often require custom validators.

| Validation Layer | Question to Answer | Tool Form |
| :-- | :-- | :-- |
| Structure | Are JSON/table fields complete? | Schema validators, Pydantic |
| Statistics | Did distributions drift or outliers spike? | Profiling, dashboards |
| Semantics | Is the sample self-consistent and on task? | LLM judges, human spot checks |
| Task | Does it still satisfy training or evaluation protocols? | Special scripts, task validators |

For the specialized datasets in Part 12, validators should ideally output three objects: a failed-sample pool, failure-reason categories, and repair suggestions. Cleaning then becomes a process of translating problems into next actions.

### A.4.3 Deduplication, Decontamination, and Splitting Should Be Governed Separately

Do not combine deduplication, decontamination, and train/validation/test splitting into one opaque step. Use three steps:

1. Detect exact and near duplicates.
2. Check evaluation contamination and benchmark isolation.
3. Create official splits and freeze the version.

Text tasks commonly use `MinHash`, `SimHash`, and `n-gram overlap`. Document and image tasks must consider visual and layout-level near duplicates. Code and reasoning tasks must also watch template contamination, question-bank contamination, and evaluation-prompt leakage. A mature process lets future readers know whether a sample was excluded because of duplication, contamination, or split policy.

## A.5 Annotation, Experiment Tracking, and Evaluation Tools

### A.5.1 Annotation Platforms Should Be Judged by the QA Chain

The core of an annotation platform is not a polished UI. It is whether the platform supports task definition, review, arbitration, and write-back.

| Scenario | Key Capability | Common Platform Direction |
| :-- | :-- | :-- |
| Text classification/extraction | Rule-based annotation and QA sampling | Label Studio, Doccano |
| Preference/ranking | Pairwise comparison, arbitration, review | Custom platforms, questionnaire-style systems |
| Document/multimodal | Region annotation, box selection, OCR linkage | Label Studio, CVAT-style tools |
| Speech | Waveform playback, slicing, speaker and emotion tags | Speech-focused annotation platforms |

If a project will become an open benchmark or course experiment, preserve annotation version, guideline version, review conclusion, and disputed-sample list. Keeping only final labels makes it hard to reconstruct why boundary samples were defined in a certain way.

### A.5.2 Experiment Tracking Must Bind Data Versions

Tools such as `MLflow` and `Weights & Biases` are often misused by recording only model parameters and metrics while omitting data versions, slice results, and evaluation-script versions. Logs then look rich but cannot explain where improvement came from.

Track at least:

| Field | Description |
| :-- | :-- |
| dataset_version | Training or evaluation data version |
| split_version | Split-policy version |
| eval_script_version | Metric-script version |
| prompt_or_template_version | Prompt or template version |
| slice_report_uri | Location of the slice report |
| writeback_decision | Whether the data strategy was changed |

With these fields, experiment tracking moves from "what was run" to "why it was run, why it improved, and why it can be trusted."

### A.5.3 Evaluation Frameworks Need Slices and Evidence Preservation

Large-model evaluation tools are multiplying, but the engineering needs in Part 12 are broader than a single benchmark run. We need reproducible metrics, explainable slices, saved evidence, and traceable versions.

Evaluation frameworks should:

- Support multiple metrics in parallel, not only one total score.
- Export slice reports and error samples.
- Save evaluation inputs and outputs structurally.
- Support reruns and historical version comparisons.

Only then can evaluation results enter the release checks in Appendix B and cost budgets in Appendix C.

## A.6 Specialized Tools for Multimodal, RAG, and Agent Scenarios

### A.6.1 In Document and Multimodal Pipelines, Parsing and Judgment Are Different

In document understanding, table parsing, chart reasoning, and multimodal RAG, teams often collapse OCR, layout analysis, retrieval, and final QA into one model black box. A better toolchain separates four layers.

| Capability Layer | Role | Common Tool Direction |
| :-- | :-- | :-- |
| Parsing | Extract text, regions, and structure from raw files | OCR, document parsers, layout models |
| Storage | Store chunks, bounding boxes, page numbers, evidence metadata | Vector databases, object storage, structured tables |
| Retrieval | Recall candidate evidence bundles | BM25, vector search, hybrid retrieval |
| Judgment | Compose answers, refuse when needed, cite evidence | LLMs, rule validation, judges |

This separation lets the team identify whether the problem is failure to extract, failure to retrieve, or failure to use retrieved evidence correctly.

### A.6.2 Agent Data Toolchains Should Treat Trajectories as Assets

Agent tool-use data differs from ordinary QA data because intermediate states are themselves training assets. Function choice, arguments, observations, error recovery, and stopping conditions should not be treated as temporary logs.

Agent tooling should support:

- Saving complete event sequences.
- Replaying key steps.
- Binding observations to final answers.
- Extracting failed trajectories into specialized evaluation sets.

Without these capabilities, a team may get good final accuracy but still be unable to explain whether behavior is stable or convert the result into teaching experiments.

## A.7 Minimal Combinations That Can Be Implemented Directly

### A.7.1 Lab or Course Combination

- Code and specifications: `Git`
- Data versions: `DVC`
- Storage: object storage or shared network storage
- Cleaning and processing: `Python + Ray/Pandas`
- Annotation: `Label Studio` or `Doccano`
- Experiment tracking: `MLflow`
- Release: `Hugging Face Hub` or a project website

This is suitable for cross-institution specialized datasets, course reproduction, and medium-scale research projects. It is lightweight and relatively easy to hand off.

### A.7.2 Enterprise Data Platform Combination

- Workflow scheduling: `Airflow`
- Distributed processing: `Spark`
- Lakehouse governance: `Iceberg/Delta`
- Quality validation: `Great Expectations`
- Experiment tracking: `MLflow` or an internal platform
- Release: tiered internal/external repositories and audit dashboards

The goal is not fastest one-off development, but stable boundaries under multi-person collaboration.

### A.7.3 Multimodal and Agent-Heavy Combination

- Files and metadata: object storage plus structured tables
- Batch processing: `Ray Data`
- Document parsing: OCR / Unstructured / custom pipelines
- Retrieval and evidence: vector database plus rule indexes
- Trajectory records: JSONL/Parquet plus replay tools
- Evaluation attribution: specialized scripts plus experiment tracking

This fits the problem space of Chapters 38-41 because it naturally supports unified governance of documents, tables, charts, multimodal evidence, and agent tool trajectories.

## A.8 Ten Questions Worth Asking During Tool Selection

Tooling should be selected around problems, not the other way around.

1. Does the tool solve acquisition, governance, evaluation, or release?
2. Can it connect stably to the existing versioning system?
3. Can failed samples be exported and recovered separately?
4. Does it support fixed versions for teaching reproduction?
5. Are permission control, audit, and least exposure easy to implement?
6. Does it support structured metadata rather than only file piles?
7. Can it support multimodal or multi-turn trajectories, not only single text rows?
8. Is handoff cost acceptable when team members change?
9. Is there a risk of deep vendor lock-in or single-engineer lock-in?
10. If the team creates a public benchmark in a year, can the tool still support it?

If four or five of these cannot be answered, the team should improve process design before deploying the tool broadly.

### A.8.1 Common Tool-Selection Mistakes

| Mistake | Surface Reason | Actual Problem |
| :-- | :-- | :-- |
| Use one large platform for everything | A single platform seems simpler | Stage boundaries are hidden behind UI |
| Look only at throughput and benchmarks | Faster tools feel more advanced | Audit, handoff, teaching reproduction, and failed-sample recovery are ignored |
| Follow one engineer's past experience | It worked in a previous project | Organizational knowledge and fallback plans are missing |
| Deploy tools first and add process later | "Let's get it running first" | Interfaces become messy and rework grows |

These mistakes are dangerous because each contains a partial truth. A unified platform can reduce early integration cost; one experienced engineer can move quickly; throughput matters. But if these reasons replace lifecycle judgment, governance tends to collapse after a few versions.

### A.8.2 Recommended Mapping for the Later Parts and Appendices

The later parts of the book connect naturally to tool choices:

- Chapters 44-45: pre-training and post-training recipes need batch processing, version governance, experiment tracking, and data cards.
- Chapters 46-48: reasoning, multimodal, and generative scenarios need trajectory records, evaluation slices, storage layering, and inference services.
- Chapters 38-43: specialized datasets need fact checking, sample schemas, build pipelines, evaluation protocols, compliance audits, and reproducibility boundaries.
- Appendices A-C translate those capabilities into operational checklists and templates for project managers, teaching assistants, platform teams, and maintainers.

This reminds readers that appendices are not secondary extras. They translate engineering capabilities from the main text into operational language.

## A.9 Quick-Reference Fields to Maintain Long Term

To turn this appendix into a team asset, maintain a tool inventory table and update it quarterly.

| Field | Description |
| :-- | :-- |
| tool_name | Tool name |
| stage | Lifecycle stage |
| owner | Responsible person or group |
| current_version | Current version in use |
| replacement_plan | Replacement or upgrade plan |
| dependency_risk | External dependency risk |
| teaching_ready | Whether it can enter course images |
| public_release_ready | Whether it is suitable for open release |

This turns tool knowledge into organizational memory. Even after team members change, the team can still understand why a tool was chosen, where it can be replaced, and where it must not be changed casually.

## A.10 Summary

This appendix organizes common data-engineering tools and frameworks from a lifecycle perspective.

First, tool selection should start from deliverables and failure modes, not popularity.

Second, long-term tool combinations are usually not one large platform, but a bounded combination of version control, data storage, batch processing, quality validation, evaluation tracking, and release governance.

Third, for university collaboration, open benchmarks, and teaching reproduction, **handoff, reproducibility, and auditability** are often more important than one-off throughput.

## References

Gebru T, Morgenstern J, Vecchione B, Vaughan J W, Wallach H, Daumé III H, Crawford K (2021) Datasheets for Datasets. Communications of the ACM 64(12): 86-92.

Mitchell M, Wu S, Zaldivar A, Barnes P, Vasserman L, Hutchinson B, Spitzer E, Raji I D, Gebru T (2019) Model Cards for Model Reporting. In: Proceedings of the Conference on Fairness, Accountability, and Transparency, pp 220-229.

Pushkarna M, Zaldivar A, Kjartansson O, Cicconi P, Chen V, Efrat A, Zou Y, Mueller J, Taly A, Ehyaei A, Karkkainen K, Marathe A, Han X, Mittal A, Schuster T, Yarmand M, Sohn H, Dwarakanath N C, McCann B (2022) Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI. In: Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, pp 1776-1826.

DVC Documentation. Available at: https://dvc.org/doc

MLflow Documentation. Available at: https://mlflow.org/docs/latest/

Hugging Face Datasets Documentation. Available at: https://huggingface.co/docs/datasets
