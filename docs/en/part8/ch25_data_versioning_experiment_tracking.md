# Chapter 25: Data Version Management and Experiment Tracking

---

## Chapter Overview and Learning Objectives

"Last week's model is worse than the week before, but the data team says the data did not change." This sentence appears in LLM projects more often than most teams expect. The subtlety of data engineering is that data changes are often cumulative and gradual rather than single and obvious. Without systematic version management, a team cannot answer the basic question "what changed," let alone explain "why it changed."

This chapter explains how to connect data versions, sample changes, and experiment results into a complete traceability chain. Production ML experience shows that data dependencies, configuration drift, and missing experiment records are major sources of technical debt and must be made explicit through versioning and metadata management (Sculley et al. 2015; Polyzotis et al. 2017). The chapter covers version granularity, naming conventions, experiment tracking, result write-back, lineage visualization, governance rules, and a realistic failure-replay case.

After reading this chapter, readers should be able to design metadata tables, version names, and experiment cards that support reproducibility, root-cause analysis, and audit.

---

## Opening Scenario

An algorithm team finishes a large experiment and finds that the new model's math reasoning has dropped by about eight percentage points. The algorithm lead asks the data team to locate the cause.

First, the data owner checks the training-set versions. Both the current and previous datasets are stored in object storage, but the folders have no version tags, only upload times. Some files have already been overwritten or deleted by later tasks.

Second, the team tries to compare differences. Around 30,000 samples differ, but nobody can tell whether those records were added, modified, or deleted because file checksums were never recorded.

Third, the team traces sources. Some of the changed samples came from a new vendor annotation batch, some came from an adjusted cleaning rule, and some were manually added by an algorithm engineer for an experiment. These three types of change are not marked and cannot be separated.

The investigation lasts five days. The final root cause is a small change in a cleaning edge case four weeks earlier. It accidentally filtered out many high-quality samples containing mathematical formulas. The change had been described as a minor optimization and left no usable record.

The cost of missing version management is an information black hole between the moment a problem appears and the moment the team understands its cause. Without version management, iteration cannot be reviewed. Experiment-management systems and ML lifecycle platforms exist to connect models, data, parameters, and results (Vartak et al. 2016; Zaharia et al. 2018).

---

## 25.1 Why Iteration Cannot Be Reviewed Without Version Management

### 25.1.1 Why Data Changes Are Hardest to Trace

Many variables affect model performance: architecture, hyperparameters, training code, evaluation code, and data. The first four are usually versioned by Git, configuration files, or experiment tools. Data changes are harder for three reasons.

1. **Change granularity is inconsistent.** A code change can be traced to a line. A data change can be one sample, one shard, one source, or one cleaning rule. These granularities have very different model impact.

2. **Causal chains are complex.** Data changes can be passive: a vendor's labeling style drifts, a crawled site changes layout, or a dependency upgrade changes preprocessing behavior. Nobody explicitly "committed" the change.

3. **Impact is delayed.** A data change may become visible several versions later, after batching, mixing, training, and evaluation.

Data has both asset and process properties. As an asset, a dataset is a set of files used for training, evaluation, and reuse. As a process, it is the result of collection, cleaning, annotation, filtering, merging, and approval. Saving only final files loses process information. Saving only logs without binding them to a concrete data version cannot explain model changes. Version management unifies asset state and generation process.

LLM data makes this unification especially important. Pretraining corpora, SFT data, preference data, evaluation sets, and online feedback differ in update frequency, quality standard, and effect path. A small pretraining distribution shift may affect long-tail knowledge; SFT label changes can directly alter response style; preference-data changes can affect safety and refusal boundaries; evaluation contamination can inflate metrics.

Version management must answer four questions:

- What does this data version contain?
- What changed compared with the previous version?
- Why did those changes happen?
- Which experiments and models consumed the changed data?

Only when all four can be answered is a data change explainable.

### 25.1.2 From Folder Management to Lineage Management

Most data teams start with folder management: use dates in folder names, put new data in new folders, and inspect historical folders when something goes wrong. Dates tell when a folder was created, but they do not answer:

- What changed: which samples were added, deleted, or modified?
- Why it changed: what business or engineering decision caused the change?
- Who changed it: which role, tool, or job triggered it?
- What downstream impact it had: which experiments used it and what happened?

Lineage management upgrades folder management. It records not only the existence of a dataset, but also its history: which sources it came from, which processing steps transformed it, what each step changed, and which downstream tasks consumed it. A dataset becomes a living asset with a history, not an isolated directory.

Lineage should be represented as a graph. Raw sources, cleaned shards, annotation batches, datasets, experiments, models, and releases are nodes. Edges describe transformation, reference, or consumption. For example: "shard A was cleaned by script B to produce shard C," "dataset D references shard C," "experiment E used dataset D," and "model F was produced by experiment E."

Folder management records location, date, and informal names. Manifest management records dataset names, scale, owner, and description. Version management records snapshots, change logs, and rollback points. Lineage management records relationships among sources, processing, consumption, approval, and results. The later stages cost more, but they enable impact analysis, root-cause analysis, and governance rules.

### 25.1.3 Core Value of Version Management

Version management creates value in three direct scenarios.

**Reproducible experiments.** Six months later, an algorithm team may need to rerun a historical experiment. With version management, the experiment ID points to the dataset version, code commit, configuration, environment, and results.

**Traceable problems.** When a model problem appears, the team can follow the chain: model version -> experiment -> dataset version -> differences from previous version -> source shard or processing step -> change record.

**Auditable compliance.** If a regulator, customer, or internal audit asks which sources trained a model, version records can produce a report with source, transformation, authorization, and approval information.

A less obvious value is organizational learning. Without versions, lessons remain oral: "adding that kind of data seemed bad last time." With versions, experience becomes evidence: which dataset version, which samples, which experiment, which metric, and what conclusion.

Version management does not mean keeping every intermediate file forever. Mature systems distinguish permanent frozen versions, temporary process versions, and disposable artifacts. The goal is to retain enough information for reproduction, traceability, and audit without making storage and governance unmanageable.

---

## 25.2 Version Granularity and Naming Conventions

### 25.2.1 Five Levels of Version Granularity

Data version management is not a single-granularity problem. LLM teams should track five levels.

| Granularity | Main use | Key fields | Retention strategy |
|---|---|---|---|
| Sample | Compliance audit and annotation traceability | `sample_id`, `source`, `status` | Permanent |
| Shard | Quality analysis and processing traceability | `shard_id`, `script_version`, `quality_summary` | 12 months |
| Dataset | Experiment comparison and release | `dataset_id`, `version`, `shards`, `quality_report` | Permanent for frozen versions |
| Experiment | Result attribution and effect tracking | `experiment_id`, `dataset_version`, `results` | 18 months |
| Release | Deployment and compliance review | `release_version`, `model_checkpoint`, `approval` | Permanent |

*Table 25-1: Data version granularity and use cases*

The levels aggregate upward. Sample records answer where a single sample came from. Shard records answer how a batch was generated. Dataset records answer what combination was used for training. Experiment records answer how data affected model results. Release records answer what was delivered externally. Missing any layer creates a break in the traceability chain.

Teams do not need to automate every level on day one, but they must define ownership. Sample-level metadata usually comes from collection and annotation systems. Shard metadata comes from pipelines and annotation platforms. Dataset metadata comes from version-management tools. Experiment metadata comes from tracking systems. Release metadata comes from model registries and release workflows.

Granularity should be risk-based.

| Data type | Recommended minimum granularity | Reason | Simplifiable record |
|---|---|---|---|
| Formal training set | Shard plus dataset | Needs recipe and quality-change explanation | Temporary cleaning intermediates may be cleaned by policy |
| Key evaluation set | Sample plus dataset | Must prevent contamination and support item review | Irrelevant experiment logs need not be kept long-term |
| Preference data | Sample plus shard | Annotator, criterion, and disputed samples matter | Low-value draft samples may be archived |
| Online feedback | Sample | User request, permission, and deletion obligations | Aggregate statistics can be retained longer than raw content |
| Exploratory synthetic data | Dataset plus experiment | Used to test hypotheses with lower risk | Generation config may be enough for some samples |
| External release package | Release plus experiment | Must answer source and approval chain | Intermediate checkpoints can be compressed by policy |

*Table 25-2: Recommended version granularity by data type*

If the team can only say "this dataset has a problem" but cannot locate shard, source, or processing step, granularity is too coarse. If every tiny field change requires heavy approval and people bypass the system, governance is too fine.

### 25.2.2 Version Naming Conventions

Version names should be readable, sortable, and unique. They should be generated by systems whenever possible, with humans filling only the semantic fields that matter.

**Dataset versions use semantic versioning.**

Format: `MAJOR.MINOR.PATCH`

- `MAJOR`: major restructuring, such as changing core source, large relabeling, or incompatible format change.
- `MINOR`: new category or more than 10% additional samples.
- `PATCH`: repairs to existing samples or small updates.

Example: `dialogue-sft-zh_v2.3.1`.

**Shard versions use timestamp plus source tag.**

Format: `{source_tag}_{YYYYMMDD}_{sequence}_{hash}`

Example: `vendor_a_annotation_20240315_001_a3f7b2`.

**Experiment versions use project, date, and sequence.**

Format: `{project}_{YYYYMMDD}_{seq_num}`

Example: `edu-math_20240315_exp003`.

Names should avoid subjective terms such as `best_dataset`, `high_quality_v1`, or `new_cleaned`. Quality, fit, and experimental performance belong in metadata, not names. Names should identify; metadata should explain.

Names should also be separated from lifecycle state. Production mainline, experiment branches, sandbox data, frozen versions, and archives should not share a confused namespace. Metadata should include fields such as `environment` or `lifecycle_stage`, with values like `sandbox`, `candidate`, `frozen`, and `archived`.

Large organizations should maintain a shared abbreviation dictionary. A short tag such as `cs` can mean customer service, computer science, or content safety in different teams. Ambiguous tags eventually damage search, automation, and communication.

### 25.2.3 Branches, Snapshots, and Rollback Points

Version management needs three distinct operations.

**Branch.** A branch lets the team modify or extend data without touching the mainline. For example, a team can test whether adding 20% synthetic data improves performance. Branches support exploration and parallel recipes.

**Snapshot.** A snapshot is an immutable record of dataset state at a point in time. Quarterly freezes are snapshots. They support audit and stable reference versions.

**Rollback point.** A rollback point marks a known-good state before a high-risk change, such as replacing a cleaning rule, merging a vendor batch, changing labels, or deleting samples. It supports fast recovery.

Branches support exploration, snapshots support audit, and rollback points support recovery. Confusing them damages governance. A branch can change; a snapshot must not. A rollback point should be created before risk, not after failure.

Merge standards should consider quality, compliance, coverage, and maintenance cost, not only model metrics. A synthetic-data branch may improve a benchmark but still be unsuitable for the mainline if its source is unclear or its style is too narrow.

---

## 25.3 Experiment Tracking, Result Write-Back, and Audit Chains

### 25.3.1 Experiment Card Field Design

The core artifact of experiment tracking is the experiment card. It should contain enough information for someone to reproduce the experiment and understand the decision context months later.

**Basic information**

| Field | Type | Description |
|---|---|---|
| `experiment_id` | string | Unique experiment identifier |
| `experiment_name` | string | Human-readable name |
| `project` | string | Project name |
| `created_by` | string | Initiator |
| `created_at` | datetime | Creation time |
| `status` | enum | `pending`, `running`, `completed`, `failed`, `abandoned` |

**Data configuration**

| Field | Type | Description |
|---|---|---|
| `dataset_id` | string | Dataset ID |
| `dataset_version` | string | Dataset version |
| `data_splits` | object | Train/validation/test counts |
| `data_filters` | list | Extra filters used in this experiment |
| `data_mixing_weights` | object | Weights when multiple datasets are mixed |

**Model configuration**

| Field | Type | Description |
|---|---|---|
| `base_model` | string | Base model name and version |
| `training_framework` | string | Training framework, such as DeepSpeed or Megatron |
| `hyperparams` | object | Learning rate, batch size, and other full hyperparameters |
| `training_code_commit` | string | Git commit of training code |

**Evaluation results**

| Field | Type | Description |
|---|---|---|
| `eval_datasets` | list | Evaluation sets used |
| `metrics` | object | Metrics by evaluation set |
| `eval_code_commit` | string | Git commit of evaluation code |
| `eval_timestamp` | datetime | Evaluation completion time |

**Experiment notes**

| Field | Type | Description |
|---|---|---|
| `hypothesis` | string | What the experiment tries to verify |
| `motivation` | string | Business or data reason for the adjustment |
| `notes` | string | Observations and anomalies |
| `conclusion` | string | Final conclusion |
| `next_actions` | list | Follow-up actions |

*Table 25-3: Example experiment-card fields*

The `hypothesis` field is especially important. Many teams record what they did but not why. Six months later, results remain but the reason for running the experiment disappears. Requiring a hypothesis improves experiment design before training starts.

Cards should separate mandatory fields, conditionally mandatory fields, and optional fields. Mandatory fields guarantee reproduction and audit. Conditional fields apply when a data mixture changes, a new source appears, preference data is used, or compliance constraints apply. Optional fields capture human observations and review comments.

Failed experiments should record negative conditions: resource interruption, unsuitable evaluation set, preprocessing failure, missing logs, or uncertainty. This helps the team distinguish "the idea failed" from "the execution conditions were insufficient."

### 25.3.2 Result Write-Back and Bidirectional Links

An experiment card records the relationship from data to results. A complete system also needs the reverse relationship from results back to data. This is result write-back.

When evaluation results arrive, the system should not only write them to the experiment card. It should also write a summary into dataset metadata, creating a bidirectional link:

- From a dataset: which experiments used this version, and what effects did they produce?
- From an experiment: which dataset versions did it consume?

This allows a team to answer questions such as "how did this type of data recipe perform in earlier experiments?" instead of starting from zero each time.

Implementation options include:

- MLflow experiment APIs, recording dataset versions as parameters or artifacts.
- DVC commands such as `dvc params diff` and `dvc metrics diff`.
- A custom metadata service with a many-to-many table `(dataset_version, experiment_id)`.

Write-back should be automated. The experiment launch records dataset versions. Completion reads metrics and writes them back. Manual copying is where records disappear.

Write-back should include both metrics and explanation. Metrics record numerical outcomes; explanation records which sample types improved, which scenarios regressed, and which data changes likely caused the result. Without explanation, the team can compare experiments but cannot learn.

Stable IDs are essential. Dataset names can change, experiment names can repeat, and paths can migrate. Dataset IDs, version IDs, shard IDs, experiment IDs, and model IDs must remain stable across systems.

### 25.3.3 The Value and Capture of Failed Experiments

Failed experiments are often under-recorded because teams assume they have no value. That wastes knowledge. Failed experiments provide:

1. **Search-space elimination.** They show which directions are unlikely to work.
2. **Anomaly signals.** Loss spikes, extreme errors, and unexpected metric changes may reveal useful signals.
3. **Control baselines.** Later data recipes need comparison against historical failures.

A failed experiment should:

1. Fill `conclusion` and state why it is considered failed.
2. Fill `next_actions`.
3. Use `failed` or `abandoned` status rather than silently disappearing.
4. Enter a shared "known-bad direction" list if the direction is proven ineffective.

Different failures require different follow-up.

| Failure type | Typical symptom | Should rerun? | Information to capture |
|---|---|---|---|
| Execution failure | Training interruption, missing logs, insufficient resources | Usually yes | Resource configuration, cause, rerun condition |
| Data failure | Quality anomaly, distribution bias, labeling error | Rerun after repair | Affected data version, problem samples, repair strategy |
| Hypothesis failure | No metric gain or key scenario regresses | Usually not immediately | Hypothesis, comparison result, exclusion conclusion |
| Evaluation failure | Contaminated evaluation, wrong code, metric definition change | Rerun after fixing evaluation | Evaluation version, wrong definition, impact |
| Compliance failure | Insufficient authorization or sensitive data | Do not rerun until risk handled | Compliance conclusion, isolation, permission recovery |

*Table 25-4: Failed experiment types and capture requirements*

Recording failure also matters culturally. It signals that experiments are valuable when they reduce uncertainty, not only when they improve headline metrics.

### 25.3.4 Minimal Audit-Chain Information

An audit chain must answer the following questions.

| Question | Required information |
|---|---|
| What data trained this model? | Release -> experiment -> dataset version |
| Where did this data come from? | Dataset -> shard -> sample source |
| Who processed it and when? | Processing script version, operator, timestamp |
| What quality checks did it pass? | Quality record, evaluator, time |
| Was use compliant? | Compliance review, reviewer, conclusion |
| If one user's data must be deleted, what is affected? | Sample-source index -> shards -> datasets -> experiments |

*Table 25-5: Audit-chain information needs*

More records are not automatically better. The key information must exist and must be specific. "Data optimized," "checked," or "no issue" does not support audit. Records should say what changed, which metrics were checked, what compliance basis applied, and which restrictions remain.

Audit records need access control. Ordinary developers may see dataset versions, quality summaries, and experiment links. Data owners may see full change records. Compliance and security roles may see authorization files, access logs, and sensitive-field handling.

---

## 25.4 Lineage Visualization and Governance Rules

### 25.4.1 Representing Data Lineage Graphs

A data-lineage graph visualizes dependencies and transformations as a directed acyclic graph. In LLM data engineering, nodes usually include:

- **Source nodes:** raw data sources such as CommonCrawl 2024Q1 or vendor A annotation batch.
- **Processing nodes:** deduplication, language identification, quality review, de-identification.
- **Dataset nodes:** versioned datasets such as `dialogue-sft-zh_v2.3.1`.
- **Experiment nodes:** experiments using dataset versions.
- **Model nodes:** produced checkpoints or releases.

Edges mean "produced from," "filtered by," "annotated by," "evaluated with," "approved by," or "released as." Clear edge semantics make lineage queryable rather than merely decorative.

Lineage supports three query views:

**Forward tracing.** Starting from a source, find all downstream datasets, experiments, and models. This is impact analysis.

**Backward tracing.** Starting from a model or experiment, trace all upstream sources, processing steps, and quality approvals. This is root-cause analysis.

**Difference comparison.** Compare two dataset versions by lineage: sources changed, processing steps changed, and change scale. This is change audit.

Lineage graphs must be generated or updated by systems whenever possible. Hand-drawn diagrams are useful for communication, but they cannot be the sole audit source.

![Figure 25-1: Data lineage and experiment tracking graph](../../images/part8/图25_1_数据谱系与实验追踪图.png)

*Figure 25-1: Full lineage from sources to model release, with forward and backward tracing paths*

### 25.4.2 Change-Audit Workflow

Every dataset-version change should pass through a standardized audit workflow.

1. **Change request.** The proposer explains content, reason, and expected impact.
2. **Impact assessment.** The team identifies downstream datasets and experiments affected.
3. **Compliance review.** Legal reviews changes involving source or data-type changes.
4. **Technical review.** The data owner or senior data engineer reviews implementation.
5. **Execution.** The data engineer creates a rollback point and performs the change.
6. **Validation.** Quality checks compare actual results against expected impact.
7. **Record.** Metadata and lineage graph are updated.

| Step | Actor | Tool | Artifact |
|---|---|---|---|
| Change request | Proposer | Change form | Complete request |
| Impact assessment | Data engineer | Lineage query | Impact list |
| Compliance review | Legal / compliance | Compliance checklist | Pass, reject, or conditional pass |
| Technical review | Data owner | Code review | Approval notes |
| Execution | Data engineer | Processing scripts and version tool | New dataset version |
| Validation | Quality evaluator | Automated quality checks | Quality report |
| Record | Automation | Metadata service | Updated lineage and changelog |

*Table 25-6: Data-change audit workflow*

The point is to connect expected impact before change with observed impact after change. If the two differ greatly, the team should not blindly release the new version. It should reassess the change hypothesis.

Change records should also keep rejected alternatives. When a problematic batch appears, the team may delete, repair, down-weight, or isolate it. Recording why one option was chosen prevents the same debate months later.

### 25.4.3 Lineage Governance Rules

Lineage governance defines which data operations are free, which need approval, and which are prohibited.

**Freely allowed without approval**

- Create a data branch in a sandbox for local experimentation.
- View or export dataset statistical summaries.
- Add noncritical metadata such as tags or comments.

**Light approval**

- Merge a new shard into the mainline dataset.
- Fix a small number of sample labels or metadata fields.
- Modify a noncritical quality threshold.
- Publish a candidate version for internal experiments.

**Formal approval**

- Change core sources, such as replacing vendor or adding a new crawl site.
- Delete an existing dataset version.
- Share internal datasets with external partners.
- Modify a released dataset version, which should normally be prohibited.

**Always prohibited**

- Modify a frozen dataset without a version record.
- Use third-party copyrighted data without compliance review.
- Directly edit another person's submitted data without leaving a modification record.

Governance should follow lifecycle stage.

| Lifecycle stage | Allowed focus | Key constraint | Main evidence |
|---|---|---|---|
| Collection | Explore sources and build sample pools | Source, authorization, and sensitivity must be recorded | Source list, authorization, collection logs |
| Cleaning | Adjust rules and generate shards | Script version and filtering result must be traceable | Script, quality summary, diff report |
| Annotation | Distribute tasks, calibrate guide, collect results | Guide version and annotator identity must be recorded | Annotation log, guide version, consistency report |
| Training | Combine datasets and start experiments | Dataset versions must be stable and referenceable | Experiment card, data version, parameters |
| Evaluation | Compare models and analyze errors | Eval-set version and code must be fixed | Metrics, error samples, eval commit |
| Release | Freeze model and data evidence | Approval, quality, and compliance chains must be complete | Release package, approval record, audit report |
| Archive | Reduce storage while preserving evidence | Reproducibility and audit information must not be destroyed | Archive policy, index, restore note |

*Table 25-7: Lineage governance rules by lifecycle stage*

Governance rules should be embedded in tools. A frozen version should be read-only. A source without approval should not enter the mainline. A dataset that fails quality checks should not be released. Written policy alone is not enough in high-pressure projects.

Governance also includes storage cost. Frozen datasets and release packages are retained long-term. Candidate versions may be archived after the next stable version. Temporary intermediates can be deleted after their audit value expires. For multimodal data, hot/warm/cold storage tiers help control cost while keeping metadata online.

---

## 25.5 Case Study: Rapidly Replaying a Failed Experiment

### Background

A company trains a customer-service dialogue model. After the third major release, `v3.0.0`, online evaluation shows that accuracy on "refund process" questions drops from 82% to 67%, triggering customer complaints.

The algorithm team asks the data team: what changed in refund-process data during the past six weeks?

This is a data-change attribution problem. Customer-service data is time-sensitive: policies, flows, and scripts change. But "outdated" does not always mean "useless." Users still ask about historical orders and transition-period rules. If quality review treats all old process descriptions as invalid, it may delete valuable historical-context samples.

The company had recently built basic version management connecting model releases, training experiments, dataset versions, source shards, quality reviews, and annotation records. The data team therefore starts from the release package and experiment card rather than searching folders manually.

### Replay Process With Version Management

**Step 1: Locate the dataset version used by the model, 5 minutes.**

The release package points to training experiment `cs-dialog_20240401_exp012`. The experiment card points to dataset `cs-dialog-sft-zh_v2.8.0`. The team also verifies that training code, evaluation code, and base model did not change materially, narrowing attribution to data.

**Step 2: Compare current version with the previous stable version, 15 minutes.**

The lineage tool compares `v2.8.0` with `v2.6.0`, the last healthy version:

- `v2.6.0`: 182,000 samples.
- `v2.8.0`: 214,000 samples.
- Added: 32,156 samples from two new shards.
- Deleted: 1,823 samples filtered by quality checks.

The new shards mainly come from vendor B, while deleted samples concentrate in the quality-filtering step. The team therefore does not rollback the whole version immediately; it narrows the investigation.

**Step 3: Analyze refund-process label distribution, 20 minutes.**

The label distribution shows:

- `v2.6.0`: 6,847 refund-process samples.
- `v2.8.0`: 4,102 refund-process samples, down 40%.

The decrease is not uniform. Current-process samples remain stable, while old-process, historical-order, cross-system refund, and human-agent-intervention samples drop sharply. These are exactly the cases users complain about.

**Step 4: Trace why samples decreased, 30 minutes.**

The team traces the removed samples to shard `vendor_b_annotation_20240318_003`. In that shard, 3,201 refund-process samples were labeled "low quality" by reviewer `QA_Wang` on 2024-03-20 and then removed by the quality filter.

The audit record shows that the guide said "outdated processes should not enter formal training." The reviewer did follow the rule. The real problem is that the guide did not distinguish "invalid old process that should not be answered" from "historical process that users still need explained."

**Step 5: Confirm root cause and repair, 15 minutes.**

Root cause: the quality rule lacked business-time semantics. It treated historical refund processes as low-quality data, but those samples were valuable for explaining historical orders.

Repair plan:

1. Short term: recover removed refund-process samples from `v2.6.0`, add a "historical refund process" label, and re-ingest them.
2. Medium term: update the quality-review guide with retention rules for historical processes.
3. Long term: add an annotation dimension for whether a sample applies to a historical scenario.

The replay takes about 85 minutes, compared with the five-day no-version-management case described earlier.

| Replay stage | Without version management | With version management | Difference |
|---|---|---|---|
| Locate training data | Ask experiment owner and search folders | Use experiment card to find dataset version | Starts from records, not memory |
| Compare data differences | Manually compare directories | Generate version diff report | Differences become structured |
| Find problem samples | Write temporary scripts over historical files | Query by label and shard through lineage | Scope narrows to relevant subset |
| Confirm operation reason | Ask reviewers or search chat logs | Read review record and guide version | Cause becomes evidence |
| Repair | Rollback whole dataset or reclean | Restore affected samples and update rules | Repair becomes targeted |

*Table 25-8: Replay comparison with and without version management*

### Key Success Factors

The replay worked because four designs were present:

1. Experiment-to-data bidirectional links identified the dataset version behind `v3.0.0`.
2. Shard-level source tracking located the vendor batch.
3. Review records preserved who made the quality decision and why.
4. Business labels were versioned and queryable.

The case also shows that version management is organizational, not only technical. Algorithm engineers must fill experiment cards. Data engineers must maintain shard sources. Quality evaluators must record review decisions. Business experts must define labels. Repair requires multiple roles to agree.

The root cause was not simply "samples were wrongly deleted." It was "the quality rule did not express the business time dimension." Therefore the repair must update data standards, not only restore files.

In deployment, version management can follow a "record first, connect second, govern third" path. First, assign stable IDs and minimal fields to datasets, shards, experiments, and releases. Second, connect data versions to experiment results, quality reports, and approvals. Third, add approval, rollback, archive, and permission rules based on the evidence chain.

Version management does not guarantee correctness. A wrong data version can still be recorded perfectly. What versioning provides is explainability and traceability. It must work together with quality evaluation, experiment design, compliance review, and domain judgment.

---

## Chapter Summary

This chapter presented the core design of data version management and experiment tracking for LLM data engineering.

At the conceptual level, we analyzed why data changes are hard to trace: inconsistent granularity, complex causal chains, and delayed impact. We showed why teams must move from folder management to lineage management and identified three value scenarios: reproducible experiments, traceable problems, and auditable compliance.

At the versioning level, we defined five granularities: sample, shard, dataset, experiment, and release. We introduced semantic dataset versions, timestamped shard versions, experiment IDs, and the distinction among branches, snapshots, and rollback points.

At the experiment-tracking level, we designed a complete experiment card, emphasized the value of failed experiments, and explained result write-back as a bidirectional link between datasets and experiments.

At the governance level, we introduced lineage graphs, change-audit workflows, lifecycle-based governance rules, and storage-retention strategies.

Finally, a failed customer-service experiment showed how version management can reduce a root-cause replay from days to roughly 85 minutes.

![Figure 25-2: Full view of the version-management system](../../images/part8/图25_2_版本管理体系全景图.png)

*Figure 25-2: Data versioning and experiment tracking across five granularity levels and bidirectional links*

---

## Further Reading

**Tools.** DVC is a mature data-version-control tool deeply integrated with Git and large-file storage. MLflow tracks experiment parameters, metrics, artifacts, and comparison views. LakeFS provides Git-like branching, merging, and rollback for data lakes.

**Further study.** Zaharia et al.'s "Accelerating the Machine Learning Lifecycle with MLflow" (2018) explains lifecycle-management challenges and solutions. Google's "Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI" (2022) is a useful reference for standardized dataset documentation and experiment-card design.

---

## Next Chapter

Version management tells the team what changed and where it went. The next challenge is discovering problems early. Chapter 26 turns to data-platform observability: how to monitor data health, detect silent failures, route alerts, and build operating dashboards before data problems become model problems.

## References

Amershi S, Begel A, Bird C, DeLine R, Gall H, Kamar E, Nagappan N, Nushi B, Zimmermann T (2019) Software Engineering for Machine Learning: A Case Study. In: Proceedings of the 41st International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP), pp 291-300.

Armbrust M, Das T, Sun L, Yavuz B, Zhu S, Murthy M, Torres J, van Hovell H, Ionescu A, Luszczak A, Switakowski M, Szafranski M, Li X, Ueshin T, Mokhtar M, Boncz P, Ghodsi A, Paranjpye S, Senster P, Xin R, Zaharia M (2020) Delta Lake: High-Performance ACID Table Storage over Cloud Object Stores. Proceedings of the VLDB Endowment 13(12):3411-3424.

Baylor D, Breck E, Cheng H-T, Fiedel N, Foo C Y, Haque Z, Haykal S, Ispir M, Jain V, Koc L, Koo C Y, Lew L, Mewald C, Modi A N, Polyzotis N, Ramesh S, Roy S, Whang S E, Wicke M, Wilkiewicz J, Zhang X, Zinkevich M (2017) TFX: A TensorFlow-Based Production-Scale Machine Learning Platform. In: Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp 1387-1395.

Breck E, Cai S, Nielsen E, Salib M, Sculley D (2017) The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. In: IEEE International Conference on Big Data, pp 1123-1132.

Breck E, Polyzotis N, Roy S, Whang S E, Zinkevich M (2019) Data Validation for Machine Learning. In: Proceedings of Machine Learning and Systems 1, pp 334-347.

Buneman P, Khanna S, Tan W C (2001) Why and Where: A Characterization of Data Provenance. In: Proceedings of ICDT, pp 316-330.

DAMA International (2017) DAMA-DMBOK: Data Management Body of Knowledge, 2nd Edition. Technics Publications.

DVC Documentation (2024) Data Version Control Documentation. Available at: https://dvc.org/doc

Gebru T, Morgenstern J, Vecchione B, Vaughan J W, Wallach H, Daume III H, Crawford K (2021) Datasheets for Datasets. Communications of the ACM 64(12):86-92.

Kreuzberger D, Kuehl N, Hirschl S (2023) Machine Learning Operations (MLOps): Overview, Definition, and Architecture. IEEE Access 11:31866-31879.

Mitchell M, Wu S, Zaldivar A, Barnes P, Vasserman L, Hutchinson B, Spitzer E, Raji I D, Gebru T (2019) Model Cards for Model Reporting. In: Proceedings of the Conference on Fairness, Accountability, and Transparency, pp 220-229.

Moreau L, Missier P (2013) PROV-DM: The PROV Data Model. W3C Recommendation.

Peng R D (2011) Reproducible Research in Computational Science. Science 334(6060):1226-1227.

Polyzotis N, Roy S, Whang S E, Zinkevich M (2017) Data Management Challenges in Production Machine Learning. In: Proceedings of the 2017 ACM International Conference on Management of Data (SIGMOD), pp 1723-1726.

Sandve G K, Nekrutenko A, Taylor J, Hovig E (2013) Ten Simple Rules for Reproducible Computational Research. PLoS Computational Biology 9(10):e1003285.

Sculley D, Holt G, Golovin D, Davydov E, Phillips T, Ebner D, Chaudhary V, Young M, Crespo J-F, Dennison D (2015) Hidden Technical Debt in Machine Learning Systems. In: Advances in Neural Information Processing Systems 28, pp 2503-2511.

Simmhan Y L, Plale B, Gannon D (2005) A Survey of Data Provenance in e-Science. SIGMOD Record 34(3):31-36.

Stodden V, Leisch F, Peng R D (eds.) (2014) Implementing Reproducible Research. CRC Press.

Vartak M, Subramanyam H, Lee W-E, Viswanathan S, Husnoo S, Madden S, Zaharia M (2016) ModelDB: A System for Machine Learning Model Management. In: Proceedings of the Workshop on Human-In-the-Loop Data Analytics.

Zaharia M, Chen A, Davidson A, Ghodsi A, Hong S A, Konwinski A, Murching S, Nykodym T, Ogilvie P, Parkhe M, Xie F, Zumar C (2018) Accelerating the Machine Learning Lifecycle with MLflow. IEEE Data Engineering Bulletin 41(4):39-45.
