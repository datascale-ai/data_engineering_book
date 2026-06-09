# Chapter 25: Data Version Management and Experiment Tracking

## Abstract
"Our model performed worse last week than the week before, but the data team says nothing changed"—this complaint arises far more often in LLM projects than most teams expect. The insidious nature of data engineering lies in the fact that data changes tend to be cumulative and incremental rather than discrete and conspicuous. Without systematic version management, a team lacks the ability to answer even the most fundamental question—"what changed?"—let alone attribute causality to "why it changed."

This chapter is written for teams responsible for data versioning, experiment logging, and traceability. It systematically explains how to connect data versions, sample changes, and experiment outcomes to form a complete, end-to-end traceable chain. Experience from production machine learning systems shows that data dependencies, configuration drift, and missing experiment records are major sources of technical debt that must be explicitly governed through versioning and metadata management (Sculley et al. 2015; Polyzotis et al. 2017). The chapter unfolds across four dimensions: first, an analysis of why version management is a prerequisite for reviewable iteration; second, the establishment of a version granularity hierarchy and naming conventions; third, an engineering treatment of experiment tracking, result write-back, and audit trails; and finally, a discussion of lineage visualization and governance rule design.

Upon completing this chapter, readers should have mastered a complete metadata design scheme, version naming conventions, and an experiment card template, and should be able to understand—through a failed-experiment retrospective case study—how to locate the root cause of data problems in real engineering contexts. The fundamental requirement of reproducible research is that the relationships among data, code, parameters, environment, and results can be independently re-confirmed by others (Peng 2011; Sandve et al. 2013).

## Keywords

Data version management; experiment tracking; data lineage; metadata governance; reproducibility; audit trail

## Learning Objectives
- Understand the role of data version management in model iteration, issue retrospection, and compliance auditing.
- Distinguish among sample-level, shard-level, dataset-level, experiment-level, and release-package-level version granularities.
- Design the key fields for experiment cards, result write-back, and audit trails.
- Master the basic methods for data lineage graphs, change audit workflows, and the preservation of failed experiments.
- Evaluate the trade-offs among version retention policies, permission boundaries, and platform governance costs.

## Scenario Introduction

After a large-scale experiment, an algorithm team at a company discovered that the mathematical reasoning ability of their new model version had dropped significantly—roughly 8 percentage points below the previous version. The algorithm lead asked the data team to identify the cause, and the data lead began the investigation.

**Step 1: Examine the training set version.** The data lead found that training sets for both the current and the previous version were stored in object storage, but with no version tags—only upload timestamps. Portions of both directories had been overwritten or deleted by subsequent jobs.

**Step 2: Attempt to compare differences.** The data lead found approximately 30,000 sample-level differences between the two versions but could not determine whether these differences represented additions, modifications, or deletions—because MD5 checksums had not been recorded.

**Step 3: Attempt to trace data provenance.** Among those 30,000 differing samples, some came from a newly added outsourced annotation batch, some resulted from changes to data-cleaning logic that altered filtering outcomes, and some were samples manually added by algorithm engineers for experimental purposes. These three categories of changes had no distinguishing markers; they were all mixed together with no way to separate them.

The investigation lasted five full days. The root cause was ultimately identified: a boundary-condition modification in the data-cleaning logic had inadvertently filtered out a large number of high-quality samples containing mathematical formulas. This modification had been a "minor optimization" made four weeks earlier and left no change record whatsoever.

This case reveals the core cost of lacking version management: **there is an enormous information black hole between the moment a problem is discovered and the time spent investigating it.** Without version management, there is no reviewable iteration. The design of experiment management systems and machine learning lifecycle platforms exists precisely to address the difficulty of tracking the relationships among models, data, parameters, and results (Vartak et al. 2016; Zaharia et al. 2018).

---

## 25.1 Why Reviewable Iteration Is Impossible Without Version Management

### 25.1.1 Why Data Changes Are the Hardest to Track in Terms of Model Performance

In the complete pipeline of an LLM project, many variables affect model performance: model architecture, hyperparameters, training code, evaluation code, and data. Industrial ML practice research has also pointed out that changes in model performance often stem from the combined effects of data, features, training configuration, and serving environment rather than from any single, obvious code commit (Amershi et al. 2019). The first four categories of variables typically enjoy mature version management tooling—Git manages code, and configuration files record hyperparameters—but data changes remain the hardest to track, for three reasons:

1. **Inconsistent change granularity.** Code changes can be pinpointed to a line; data changes may involve a single sample, a batch, a category of sources, or a single cleaning rule. Changes at different granularities have vastly different effects on model performance, yet there is no uniform way to record them.

2. **Complex causal chains for changes.** Data changes are not always actively triggered; they sometimes occur passively. An outsourcing vendor's annotation style drifts, a crawled source site modifies its content, or a cleaning tool is upgraded—all of these can alter the final training data without anyone explicitly "committing" the change.

3. **Delayed impact of changes.** A single data change may not produce a visible effect on model performance until three versions later, because the changed data must go through batch processing, blending, and retraining before the effect becomes apparent. This delay makes root-cause analysis extremely difficult.

From an engineering management perspective, the fundamental reason data changes are hard to track is that data simultaneously possesses both "asset properties" and "process properties." As an asset, a dataset manifests as a collection of files that can be trained on, evaluated, and reused; as a process, a dataset is the result of a series of activities—collection, cleaning, annotation, filtering, merging, and approval. If a team saves only the final files, process information is lost; if only processing logs are saved without binding them to specific data versions, it is still impossible to explain changes in final model performance. The task of version management is precisely to unify asset state with generation history.

LLM data is especially in need of this unification. Pretraining corpora, SFT instruction data, preference data, evaluation sets, and online feedback data change at different frequencies, have different quality standards, and affect model performance through different pathways. Minor distributional shifts in pretraining corpora may not be apparent in overall metrics yet can affect long-tail knowledge. Changes in SFT data label definitions may directly alter the model's response style. Changes in preference data ranking criteria may affect safety, politeness, and refusal boundaries. Contamination of the evaluation set may cause model performance to be overestimated. Without version records, these effects are very difficult to disentangle.

Therefore, data version management is not simply the act of tagging each data delivery with a label; it requires enabling the team to answer four categories of questions: first, what does a given data version contain; second, what has changed relative to the previous version; third, why did those changes occur; and fourth, which experiments and models have consumed those changes. Only when all four questions can be answered do data changes become explainable.

### 25.1.2 Upgrading from "Folder Management" to "Lineage Management"

Most data teams start their version management journey with "folder management": naming folders by date, putting new data in new folders, and digging through historical folders when problems arise. The problem with this approach is not that nothing is recorded, but that the recorded information is insufficient. Dates tell you "when," but they cannot tell you:

- **What changed**: which samples were added, which were deleted, which were modified
- **Why it changed**: what was the business reason for this change, and who made the decision
- **Who made the change**: which role or tool triggered this change
- **Downstream impact of the change**: which experiments used this change and what results they produced

Lineage Management is a systematic upgrade from folder management. Data provenance research has long distinguished between "why a result appeared" and "where a result came from," describing data generation pipelines through the relationships among processes, entities, and activities (Buneman, Khanna and Tan 2001; Moreau and Missier 2013). It records not only the "state of existence" of data but also the "generation history" of data: which data sources a dataset was processed from, which processing steps it went through, what changes each step produced, and which downstream tasks ultimately consumed it. Lineage management transforms a dataset from an isolated file into a living asset with a complete historical record.

Folder management is common in the early stages because it is intuitive, low-cost, and well-suited to short-cycle exploration by small teams. But as a project enters a multi-team collaboration phase, folder naming quickly breaks down. Different members may use different naming conventions—`final`, `final_v2`, `cleaned_new`, `for_train_latest`—names that might be interpretable on the day they are created but lose their context within weeks. More critically, folders cannot express dependency relationships: a training set may be merged from multiple sources, some of which went through multiple rounds of filtering and annotation, and saving only the final directory cannot reconstruct those relationships.

Lineage management requires teams to treat data objects as nodes in a graph structure rather than as isolated directories in a filesystem. Raw data, cleaned shards, annotation batches, training sets, experiments, models, and release packages should all be connected through clearly defined edges. An edge represents some transformation, reference, or consumption relationship. For example: "Shard A was processed by cleaning script B to produce Shard C"; "Dataset D references Shard C"; "Experiment E used Dataset D"; "Model F was produced by Experiment E." This form of expression makes impact analysis and root-cause analysis possible.

From a capability-evolution perspective, folder management primarily records file locations, creation dates, and human-assigned names. It is suited to small-team exploration or temporary experiments but struggles to express differences, reasons, and downstream impacts. Manifest management further records dataset names, sizes, owners, and descriptions, and is appropriate for the early stages of multi-project sharing—but still lacks process history and experiment associations. Version management begins to record data states, change logs, snapshots, and rollback points, which suits stable training and evaluation phases; however, without lineage relationships, it remains difficult to conduct systematic impact analysis. Lineage management goes further to record the relationships among sources, processing, consumption, approvals, and results. It is suited to a platform-level DataOps stage; its construction cost is higher, but it supports automated querying, auditing, and governance rule enforcement.

### 25.1.3 The Core Value Proposition of Version Management

Building a version management system delivers its most direct value in three scenarios:

**Scenario 1: Experiment reproducibility.** The algorithm team wants to re-run a historical experiment six months later and needs to find the data version used at the time. Reproducible practice in computational science emphasizes that raw data, processing scripts, runtime environments, and results should all be included in the record rather than saving only the final output (Stodden, Leisch and Peng 2014). With version management, the corresponding dataset version number can be found directly via the experiment ID, and the training environment can be reconstructed.

**Scenario 2: Problem traceability.** After a model problem is discovered, the data lineage enables rapid localization: from "which model version has the problem" → "which batch of data did this model version use" → "what is different between this batch and the previous version" → "which processing step did the difference originate from" → "when did this step change, and why."

**Scenario 3: Compliance auditability.** When a regulatory authority requests the training data provenance for a given model, version management enables the generation of a complete audit report covering data sources, processing records, and authorization information.

Beyond these three scenarios, version management has an underappreciated additional value: it enables teams to accumulate knowledge over the long term. Without version management, experience tends to exist in verbal form—"adding a certain type of data last time seemed to hurt performance." With version management, teams can ground that experience in specific evidence: which data version, which samples, which experiment, what metrics, what conclusions. Experience transforms from a vague memory into a queryable record, providing a stable foundation for organizational learning.

Version management also reduces collaboration friction. When the algorithm team identifies model degradation, the data team need not re-explain all processing details from scratch—they can present a version-diff report showing the scope of changes. When the quality team detects anomalies, they can directly locate the affected shards. When the compliance team reviews data provenance, they can see source authorization and processing records. When the platform team manages storage and compute resources, they can make archival decisions based on version activity. In other words, version management provides a common factual basis for different roles.

It should be emphasized that the value of version management is not equivalent to saving all intermediate files. Unconstrained retention causes storage, retrieval, and governance costs to balloon. A mature versioning system should distinguish among frozen versions that must be retained permanently, process versions that require periodic retention, and temporary artifacts that can be cleaned up according to policy. The goal of version management is to retain information sufficient to support reproduction, retrospection, and auditing—not to retain every file forever.

---

## 25.2 Data Version Granularity and Naming Conventions

### 25.2.1 Five Levels of Version Granularity

Data version management is not a problem of a single granularity; it requires maintaining version information simultaneously at five levels.

![Figure 25-1: Overview of the Version Management System](../../images/part8/图25_1zh.png)

*Figure 25-1: Panoramic view of the data version management and experiment tracking system—five-level version granularity with bidirectional association architecture*

The Data Management Body of Knowledge typically treats data assets, metadata, lineage, quality, and lifecycle as shared governance objects; accordingly, version granularity should span multiple levels from individual samples to release packages (DAMA International 2017):

**Sample level (Sample)**: A single training sample. Version information includes sample ID, source URL/document ID, creation time, last modification time, and current status (active/deprecated/under_review). Sample-level versioning is primarily used for annotation quality traceability and compliance auditing.

**Shard level (Shard)**: A batch of logically related samples, typically the output of a single annotation task or the result of a single cleaning batch. Version information includes shard ID, number of samples contained, processing script version, processing time, and quality summary.

**Dataset level (Dataset)**: A complete dataset ready for training. Version information includes dataset ID, the list of component shards, version number (semantic versioning), the reason for creation, and a quality report. The dataset level is the most commonly used version granularity.

**Experiment level (Experiment)**: A specific training experiment. Version information includes experiment ID, dataset version used, model architecture, hyperparameter configuration, and evaluation results. Experiment-level versioning connects data to models.

**Release package level (Release)**: An externally released model version. Version information includes release version number, the corresponding model checkpoint, the dataset version used, the list of evaluation sets passed, and the release approval record.

| Version Granularity | Primary Use | Key Fields | Retention Policy |
|---|---|---|---|
| Sample level | Compliance auditing, annotation traceability | sample_id, source, status | Permanent retention |
| Shard level | Quality analysis, processing traceability | shard_id, script_version, quality_summary | Retain for 12 months |
| Dataset level | Experiment comparison, version release | dataset_id, version, shards, quality_report | Permanent retention (frozen versions) |
| Experiment level | Result attribution, performance tracking | experiment_id, dataset_version, results | Retain for 18 months |
| Release package level | Deployment management, compliance review | release_version, model_checkpoint, approval | Permanent retention |

*Table 25-1: Data version granularity and applicable scenarios*

These five levels are not parallel—they form a hierarchical aggregation. Sample-level records answer "where does this data item come from?"; shard-level records answer "how was this batch of data generated?"; dataset-level records answer "what combination was used in this training run?"; experiment-level records answer "how did data affect the model?"; release-package-level records answer "what was ultimately delivered externally?" A gap in any level creates a break in the retrospection chain.

In practice, teams need not automate management of all granularities from the start, but must clearly define the responsibility boundaries for each granularity. Sample-level information is typically produced by collection and annotation systems; shard-level information is maintained by data pipelines and annotation platforms; dataset-level information is maintained by version management tools; experiment-level information is maintained by experiment tracking systems; release-package-level information is maintained by model registration and release systems. If these systems share no unified IDs or association tables, data will remain only partially traceable.

Granularity design must also account for cost. Sample-level versioning is the most fine-grained but incurs the highest storage and indexing costs; dataset-level versioning is the most widely used but insufficient for explaining fine-grained quality issues; release-package-level versioning is best suited to compliance auditing but cannot directly answer questions about processing details. Teams should therefore adopt a strategy of "fine granularity for critical paths, moderate granularity for ordinary paths." For formal training sets, evaluation sets, high-sensitivity data, and online feedback data, sample-level or shard-level records should be retained where possible. For one-off exploratory data, only dataset-level and experiment-level records may be needed, but these must be marked as ineligible for formal release.

| Data Type | Recommended Minimum Granularity | Rationale | Records That May Be Simplified |
|---|---|---|---|
| Formal training set | Shard level + Dataset level | Need to explain data recipe and quality changes | Temporary cleaning intermediate files may be cleaned up according to policy |
| Critical evaluation set | Sample level + Dataset level | Need to prevent contamination and support item-by-item review | Irrelevant experiment logs need not be retained long-term |
| Preference data | Sample level + Shard level | Annotator identity, preference criteria, and disputed samples are critically important | Low-value draft samples may be archived |
| Online feedback data | Sample level | Involves user requests, permissions, and deletion requirements | Aggregated statistics may be retained long-term; raw content requires time-limited control |
| Exploratory synthetic data | Dataset level + Experiment level | Primarily for hypothesis validation; relatively low risk | Generation-process logs per sample need not be retained, but generation configuration must be retained |
| Externally released packages | Release package level + Experiment level | Need to account for model provenance and approval chain | Training intermediate checkpoints may be compressed according to policy |

*Table 25-2: Recommended version granularity by data type*

Overly fine granularity creates operational burden; overly coarse granularity weakens traceability. A practical heuristic is: when a problem occurs, can the team locate the scope of responsibility within a reasonable time? If the team can only identify that "some dataset has a problem" but cannot pinpoint the shard, source, or processing step, granularity is too coarse. If every minor field change requires complex approval workflows that cause the team to bypass the process, granularity and governance intensity are too fine. Version granularity should serve real investigation, reproduction, and auditing needs.

### 25.2.2 Version Naming Conventions

Version naming conventions should follow three principles: readability (human-interpretable), sortability (temporal order can be inferred), and uniqueness (no duplicates). The practice of data versioning tools such as DVC also demonstrates that clear version identifiers, metadata, and remote storage conventions are the foundation for teams to share experiments and roll back data states (DVC Documentation 2024).

**Dataset version: Semantic Versioning**

Use the `MAJOR.MINOR.PATCH` format with the following rules:

- `MAJOR`: A major data restructuring has occurred, such as replacing core data sources, large-scale re-annotation, or an incompatible data format change.
- `MINOR`: A new category of data has been added, or more than 10% of the sample count has been added.
- `PATCH`: Quality issues in existing samples have been fixed, or a small proportion of samples has been updated.

Example: `dialogue-sft-zh_v2.3.1`

- `dialogue-sft-zh`: Dataset name (task type–training stage–language)
- `v2.3.1`: Second major version, third feature version, first patch

**Shard version: Timestamp + Source identifier**

Format: `{source_tag}_{YYYYMMDD}_{sequence}_{hash}`

Example: `vendor_a_annotation_20240315_001_a3f7b2`

- `vendor_a_annotation`: Source identifier (annotation results from Vendor A)
- `20240315`: Processing date
- `001`: First batch of the day
- `a3f7b2`: First six characters of the content hash, for rapid integrity verification

**Experiment version: Project abbreviation + Date + Sequence number**

Format: `{project}_{YYYYMMDD}_{seq_num}`

Example: `edu-math_20240315_exp003`

- `edu-math`: Project abbreviation (educational mathematics sub-task)
- `20240315`: Experiment start date
- `exp003`: Third experiment of the day

The difficulty of naming conventions lies not in the rules themselves but in consistent long-term execution. Many teams have established naming rules but lack automatic validation and governance accountability, ultimately leading to multiple naming variants. To prevent this, version names should be generated by the system wherever possible, with humans responsible only for filling in required semantic fields. For example, a dataset name can be auto-generated from the combination of task type, training stage, language, and data domain; version numbers can be auto-incremented by the release workflow; content hashes can be computed by the system. The less manual input required, the more stable the naming becomes.

Naming should also avoid including information that changes frequently or carries overly subjective interpretations. Names such as `best_dataset`, `high_quality_v1`, and `new_cleaned` may be readable in the short term but become uninterpretable over time. "Best" relative to which experiment? "High quality" by whose standard? "New" relative to which old version? None of these questions can be answered from the name alone. A better approach is to record quality ratings, applicable scenarios, and experiment outcomes in metadata rather than cramming them into version names.

Version naming must also be distinguished from branching and environments. Production mainline, experimental branches, sandbox data, and archived versions should not share the same naming namespace. Otherwise, algorithm engineers may mistakenly use exploratory data to train official models, and quality evaluators may inadvertently include sandbox data in formal quality statistics. It is recommended to explicitly record an `environment` or `lifecycle_stage` field in metadata—such as `sandbox`, `candidate`, `frozen`, or `archived`. Names handle unique identification; lifecycle fields handle state expression.

Once established, naming conventions should be written into the data release workflow. Before a release, the system automatically checks whether the name conforms to the rules, whether the version number is incremented, whether the hash matches, and whether metadata is complete. For data that does not conform to conventions, the system may allow entry into the sandbox environment but should block entry into the formal training or release pipeline. This way, the convention no longer depends on manual reminders but becomes an entry condition for the version management system.

Naming conventions should also consider cross-team readability. Data engineers typically care about source and processing batches; algorithm engineers care about task, training stage, and experiment number; compliance teams care about authorization status and sensitivity classification; platform teams care about storage location and lifecycle status. A good version name cannot carry all of this information, but should provide a sufficiently stable entry point allowing different roles to query complete metadata further. Naming conventions and metadata management must therefore be designed together: names serve to locate objects; metadata serves to explain them.

In large enterprises, teams must also avoid different departments independently defining conflicting abbreviations. For example, `cs` might mean customer service in a customer-facing project, computer science in a computational research team, and content safety in a content moderation team. Project abbreviations, task abbreviations, and language codes should be entered into a shared vocabulary to prevent accumulated ambiguity over time. Managing this vocabulary may seem trivial, but it directly affects version retrieval, automation scripts, and cross-team communication.

### 25.2.3 Branches, Snapshots, and Rollback Points

Version management requires distinguishing three different types of operations. Modern lakehouse table formats also support data version retrospection and incremental change auditing through transaction logs, snapshots, and time-travel capabilities (Armbrust et al. 2020):

**Branch**: An experimental modification or extension of data that does not affect the mainline dataset. For example, to test "the effect of adding 20% synthetic data," one can create a branch from the current mainline dataset, add synthetic data on the branch, and decide after the experiment whether to merge it back to the mainline.

Branches are appropriate when: a data decision's effectiveness is uncertain, or multiple data recipes need to be maintained simultaneously for different algorithm experiments.

**Snapshot**: A precise record of the current state of a dataset at a specific point in time. Snapshots are read-only and cannot be modified after creation. Quarterly version freezes are a typical snapshot operation.

Snapshots are appropriate when: compliance auditing requires the preservation of historical states, or a stable reference version needs to be provided to external partners.

**Rollback point**: A marker created at a point in time when a version is known to be good, enabling rapid recovery to that state when a problem is discovered. Rollback points are typically set manually before major data changes.

Rollback points are appropriate when: backing up state safely before large-scale cleaning rule changes, and before merging outsourced annotation batches.

Branches, snapshots, and rollback points each serve different risk-control objectives. Branches support exploration, allowing teams to try new data recipes without affecting the mainline. Snapshots support auditing, ensuring that the state of data at a given point in time is immutable. Rollback points support recovery, helping teams quickly return to a known stable state when problems are discovered. Mixing these three carelessly leads to governance confusion. For example, if a branch is used as a snapshot, experimental data may continue to be modified, causing reproduction to fail. If a snapshot is treated as an ordinary branch and modified, audit credibility is undermined.

Branch management also requires clear merge criteria. Whether a data branch can be merged back into the mainline should not depend solely on whether model metrics improve; it should also consider quality, compliance, coverage, and long-term maintenance costs. A branch that incorporates large volumes of synthetic data and shows short-term metric improvements may not be suitable for the mainline if its data source is unexplainable, its sample style is uniform, or it diverges from real user scenarios. The version management system should require merge requests to include diff reports, quality reports, and experiment results—not just final metrics.

Snapshots must adhere to the read-only principle. If a problem is discovered in a frozen snapshot, the correct approach is not to modify it directly but to create a new version and record the reason for the fix. This may seem cumbersome but protects historical interpretability. An audit is concerned not only with "whether the current data is correct" but also with "what data state was available when that decision was made." If historical snapshots are silently modified, the organization loses the ability to explain and review past decisions.

Rollback points should be established before high-risk changes, not sought as an afterthought when problems arise. High-risk changes include adding a large-scale new data source, replacing cleaning rules, merging outsourced annotation batches, modifying the label schema, deleting samples, and changing data mixing weights. Creating a rollback point before each high-risk change significantly reduces the cost of trial and error, enabling teams to pursue necessary improvements while maintaining a safety boundary.

---

## 25.3 Experiment Tracking, Result Write-Back, and Audit Trails

### 25.3.1 Field Design for Experiment Cards

The core instrument of experiment tracking is the **Experiment Card**. Systems such as ModelDB and MLflow both treat experiment parameters, code versions, data versions, metrics, and artifacts as the core objects of unified experiment management (Vartak et al. 2016; Zaharia et al. 2018). A complete experiment card must record sufficient information so that any person can reproduce the experiment six months later and understand the decision context at the time.

The following is a standard experiment card field design:

**Basic Information**

| Field | Type | Description |
|---|---|---|
| experiment_id | string | Unique experiment identifier |
| experiment_name | string | Human-readable experiment name |
| project | string | Name of the parent project |
| created_by | string | Experiment initiator |
| created_at | datetime | Experiment creation time |
| status | enum | pending / running / completed / failed / abandoned |

**Data Configuration**

| Field | Type | Description |
|---|---|---|
| dataset_id | string | ID of the dataset used |
| dataset_version | string | Dataset version number |
| data_splits | object | Sample counts for train/val/test splits |
| data_filters | list | Additional filter conditions applied in this experiment (if any) |
| data_mixing_weights | object | Weight of each dataset when mixing multiple datasets |

**Model Configuration**

| Field | Type | Description |
|---|---|---|
| base_model | string | Base model name and version |
| training_framework | string | Training framework (e.g., DeepSpeed, Megatron) |
| hyperparams | object | Complete hyperparameter configuration (learning rate, batch size, etc.) |
| training_code_commit | string | Git commit hash of the training code |

**Runtime Environment Configuration**

| Field | Type | Description |
|---|---|---|
| runtime_env | object | Python, OS, training framework, and key runtime versions |
| container_image | string | Container image name, version, or digest, used to lock the runtime environment |
| dependency_lock | string | Path to the dependency lock file or package version snapshot (e.g., requirements.lock, conda-lock, poetry.lock) |
| hardware_profile | object | GPU/accelerator model, count, VRAM, CPU, and memory configuration |
| cuda_driver | string | Versions of CUDA, GPU driver, cuDNN/NCCL, and other critical low-level dependencies |
| random_seed | int | Random seed used for this experiment |
| determinism_flags | object | Settings related to deterministic training, such as deterministic operators, benchmark switches, and stochasticity controls |

**Evaluation Results**

| Field | Type | Description |
|---|---|---|
| eval_datasets | list | List of evaluation sets used |
| metrics | object | Metric results on each evaluation set (key-value pairs) |
| eval_code_commit | string | Git commit hash of the evaluation code |
| eval_timestamp | datetime | Evaluation completion time |

**Experiment Records**

| Field | Type | Description |
|---|---|---|
| hypothesis | string | Experiment hypothesis (what this experiment aims to validate) |
| motivation | string | Business motivation for the data or configuration adjustments |
| notes | string | Observations and anomaly records during the experiment |
| conclusion | string | Experiment conclusion |
| next_actions | list | Follow-up actions based on the results of this experiment |

*Table 25-3: Sample experiment card fields*

Special emphasis should be placed on the importance of the `hypothesis` field. Many teams record only "what was done" but not "why it was done." Six months later, the experiment results may still be available, but the rationale for running the experiment is nowhere to be found. Requiring the `hypothesis` field to be filled in forces the experiment initiator to explicitly state the experiment's purpose before it begins—and this alone can significantly improve the quality of experiment design.

Experiment card design should avoid two extremes. One extreme is too few fields, recording only experiment ID, model version, and metric results—leaving experiments without explanatory power. The other extreme is too many fields, requiring researchers to fill in large amounts of low-value information, which ultimately degrades record quality. A better approach is to distinguish among required fields, conditionally required fields, and optional fields. Required fields ensure reproducibility and auditability; conditionally required fields apply to specific types of experiments; optional fields capture supplementary observations and analysis.

Required fields typically include: experiment hypothesis, dataset version, training code version, base model version, key hyperparameters, evaluation set version, core metrics, and conclusion. Conditionally required fields depend on the nature of the experiment: if the experiment changes data mixing ratios, data mixing weights must be filled in; if the experiment uses a new data source, the compliance approval record must be filled in; if the experiment involves preference data, the annotation guide version and preference sampling strategy must be filled in. Through this layered design, experiment cards achieve both rigor and freedom from unnecessary overhead.

Experiment cards should also record negative conditions. Many experiments fail not because the hypothesis is wrong, but because training resources are insufficient, the evaluation set is inappropriate, data preprocessing has errors, or logs are missing. If only the final metrics are recorded, subsequent reviewers cannot determine the reason for failure. Recording anomalies, limitations, and uncertainties helps the team distinguish between "this direction is invalid" and "execution conditions were insufficient." This distinction matters greatly for long-term R&D, because erroneously ruling out a direction may incur a greater opportunity cost than repeating a failed experiment.

Specifically, reproducibility fields should be required, including dataset version, code commit, base model, key hyperparameters, and random seed, to ensure the experiment can be re-run. Interpretation fields should also be required, including `hypothesis`, `motivation`, and a data change summary, to explain why the experiment was conducted. Results fields—including metrics, evaluation set version, conclusion, and next actions—support experiment comparison and decision-making. Risk and process fields may be conditionally required: when an experiment involves a new data source, sensitive data, or compliance restrictions, approvals and usage boundaries should be recorded; when the training process is interrupted, logs are missing, or anomalous fluctuations occur, process limitations should be recorded to help assess result reliability later. Supplementary fields may remain optional, for manual observations, chart links, and review comments.

To improve experiment card quality, teams can establish pre-experiment and post-experiment checklists. The pre-experiment checklist confirms that the experiment hypothesis is clear, the dataset version is frozen or traceable, and the evaluation set version is specified. The post-experiment checklist confirms that metrics have been written in, conclusions are clear, failure reasons are recorded, and next actions are actionable. Neither checklist requires complex approval; both can be embedded in the experiment platform's submission interface. Their goal is not to impede experimentation but to make every experiment a piece of reusable knowledge.

### 25.3.2 Result Write-Back and Bidirectional Association

An experiment card records only the one-directional relationship from "data" to "results." A complete tracking system also needs the ability to trace back from "results" to "data"—this is the **Result Write-Back** mechanism.

Result write-back means: when the evaluation results of an experiment become available, rather than simply writing the results into the experiment card, the results are also "written back" to the metadata of the dataset, forming a bidirectional association. Production-grade ML platforms typically require training pipelines, metadata stores, validation components, and model registration systems to form a closed loop; otherwise, experiment results cannot stably feed back into data governance (Baylor et al. 2017; Kreuzberger, Kühl and Hirschl 2023):

- From the dataset perspective: one can query "which experiments used this dataset, and what results did they produce?"
- From the experiment perspective: one can query "which dataset versions did this experiment use?"

The value of this bidirectional association lies in enabling teams, when facing a new data decision, to quickly query "what were the results of previous experiments using this type of data configuration?" and to learn from existing experiments rather than starting from scratch every time.

Several technical implementation options exist for result write-back:

- Using the MLflow experiment tracking API to record dataset versions as artifact parameters of experiments
- Using DVC's `dvc params diff` and `dvc metrics diff` commands to compare experiment result differences across different data versions
- Building a custom metadata service that maintains a many-to-many association table for `(dataset_version, experiment_id)` pairs

Regardless of the technical approach chosen, result write-back should be automated and must not rely on manual entry. The dataset version should be automatically recorded when an experiment starts, and evaluation results should be automatically read and written in when the experiment completes. Reducing manual intervention reduces opportunities for omissions.

Result write-back should also distinguish between "metric write-back" and "interpretation write-back." Metric write-back records numerical results—accuracy, win rate, perplexity, safety evaluation pass rate, and so on. Interpretation write-back records why these results occurred—which sample types improved, which scenarios regressed, which data changes may have been responsible. Teams can compare experiments with metric write-back alone, but cannot learn from them. Adding interpretation write-back enables data teams to adjust collection, cleaning, and annotation strategies based on experiment results.

Interpretation write-back is especially suited to data-driven model optimization. When experiment results show an improvement in mathematical reasoning evaluation, the team needs to know whether the improvement came from adding more math problems, improving annotation guidelines, adjusting data mixing weights, or changing training configuration. If the experiment card can associate results with a summary of specific data changes, the data team can replicate effective strategies. Conversely, if all changes happen simultaneously without records, the team cannot determine which practices to keep even when better metrics are obtained.

Result write-back can also support data asset value assessment. A dataset that is used across many experiments and consistently delivers gains across multiple projects should be flagged as a high-value data asset, subject to stricter version freezing, quality maintenance, and permission management. Datasets that have consistently failed to improve performance, or are only effective in specific experiments, should be flagged as low-reuse or scenario-specific assets. This way, experiment tracking serves not only the algorithm team but also feeds back into data asset governance.

In technical implementation, bidirectional associations should use stable IDs wherever possible rather than relying on name matching. Dataset names may change, experiment names may repeat, and file paths may migrate—but unique IDs and version numbers should remain stable. A metadata service can maintain relationship tables among datasets, shards, experiments, models, and release packages, and provide query interfaces. This way, when a data source is determined to be at risk, the team can automatically list all affected experiments and models without manual searching.

### 25.3.3 The Value of Failed Experiments and Knowledge Preservation

A common misconception is that failed experiments do not need to be carefully recorded "because they have no value." Production-readiness assessment frameworks emphasize that failure cases, data anomalies, test coverage, and monitoring signals are all important evidence for reducing ML technical debt and should not be lost after an experiment concludes (Breck et al. 2017). This misconception causes enormous knowledge waste.

The value of failed experiments manifests in three ways:

1. **Eliminating a search space**: A failed experiment proves that "this path doesn't work," preventing the team from repeating similar mistakes in the same direction. If failed experiments are not recorded, a new algorithm engineer three months later may re-run the same experiment, wasting precious computational resources.

2. **Anomaly signals**: Failed experiments often contain valuable anomaly signals. Abnormal fluctuations in the loss curve, extreme errors on certain sample types, an unexpected rise in a metric on the evaluation set—these signals may be important findings even against the overall backdrop of "failure."

3. **Reference baselines**: When subsequent experiments try a new data recipe, historical failed experiments provide meaningful comparison groups. Without comparison groups, it is impossible to assess whether improvements are genuinely effective.

Knowledge preservation requirements for failed experiments:

1. The `conclusion` field must be filled in to explicitly record why the experiment is considered a failure.
2. The `next_actions` field must be filled in to record the follow-up actions based on the failure conclusion.
3. The `status` of a failed experiment must be marked as `failed` or `abandoned`; experiments must not silently disappear.
4. For "known failed directions," a shared "exclusion list" should be maintained, annotating which categories of experiments have been proven ineffective.

Failed experiment preservation also requires distinguishing among different failure types. Failures caused by resource interruptions, data quality issues, invalid hypotheses, and mismatched evaluation criteria each require completely different follow-up handling. If the team uniformly labels all these experiments as `failed`, subsequent reviewers still cannot understand their value. A more refined approach adds a failure classification field and requires supporting evidence.

For example, an experiment that terminates due to insufficient VRAM during training does not imply that the data recipe is invalid. An experiment that produces anomalous results due to evaluation set contamination does not imply a genuine improvement in model capability. An experiment that regresses because newly added data is mismatched to the task objective can clearly rule out that data direction. Failure classification enables the team to decide whether to re-run, fix the data, adjust evaluation, or add the direction to the exclusion list.

| Failure Type | Typical Manifestation | Should Re-Run? | Information to Preserve |
|---|---|---|---|
| Execution failure | Training interrupted, logs missing, insufficient resources | Usually should re-run | Resource configuration, failure cause, re-run conditions |
| Data failure | Quality anomaly, distribution skew, incorrect label definitions | Re-run after fixing data | Affected data version, problematic samples, fix strategy |
| Hypothesis failure | No metric improvement or regression on critical scenarios | Usually not immediately | Experiment hypothesis, comparison results, exclusion conclusion |
| Evaluation failure | Evaluation set contamination, evaluation code error, metric definition change | Re-run after fixing evaluation | Evaluation version, erroneous definition, scope of impact |
| Compliance failure | Insufficient data authorization, non-compliant use, unprocessed sensitive information | Should not re-run; risk must be addressed first | Compliance conclusion, isolation measures, permission revocation |

*Table 25-4: Failed experiment types and preservation requirements*

Failed experiments can also form an organization-level "anti-pattern library." For example, a certain type of synthetic data repeatedly causes the model's output style to become mechanical; a certain filtering rule repeatedly deletes historical business process samples in error; a certain evaluation set cannot distinguish genuine capability improvement from template memorization. If these lessons only exist in personal notes, they cannot prevent the team from repeating similar mistakes. An anti-pattern library need not be complex, but should include a problem description, trigger conditions, evidence, avoidance strategies, and applicable boundaries.

For R&D teams, carefully recording failures also has cultural significance. It sends the signal that an experiment's value does not come only from successful metrics but also from reducing uncertainty. Rewarding only successful experiments encourages teams to choose conservative approaches and even conceal unfavorable results; acknowledging the knowledge value of failed experiments encourages more honest experiment records and more systematic exploration. Experiment tracking from a DataOps perspective is precisely about institutionalizing this exploration process.

### 25.3.4 The Minimum Information Set for an Audit Trail

A complete audit trail must be able to answer the following core questions. Research on both data cards and model cards emphasizes that data sources, usage boundaries, evaluation conditions, and model behavior must be documented to support auditing and accountability tracing (Gebru et al. 2021; Mitchell et al. 2019):

| Question | Information Required |
|---|---|
| What data was used to train this model? | Release package → Experiment → Dataset version |
| Where did this batch of data come from? | Dataset → Shards → Sample sources |
| Who did what processing on this batch of data? | Processing script version + Operator + Timestamp |
| What quality checks did this batch of data pass? | Quality evaluation records + Evaluator + Time |
| Was the use of this data authorized through compliance review? | Data compliance review records + Reviewer + Review conclusion |
| If a user's data needs to be deleted, what is the scope of impact? | Sample-level source index → Associated shards → Associated datasets → Associated experiments |

*Table 25-5: Audit trail information requirements*

For auditing purposes, the principle is not "more records are better" but "critical information must be present." Every column of information in the table above constitutes the minimum necessary set for an audit; a gap in any single item creates a break in the audit chain.

The audit trail must also cover "deletion and correction" scenarios. For training samples that contain user data or restricted business data, teams may receive deletion requests, authorization revocations, or data correction requirements. In such cases, the audit system must be able to start from the sample-level index and identify which shards that sample entered, which datasets, which experiments, and which released models. Only then can the team determine whether the original data must be deleted, the dataset rebuilt, the model retrained, or only relevant samples excluded from future versions.

The audit trail must also distinguish between internal and external auditing. Internal auditing focuses more on problem localization and process improvement and may retain richer technical details. External auditing focuses more on source, authorization, use, approval, and risk handling, and requires structured reports. Data teams should avoid assembling audit materials ad hoc when an external audit arrives; instead, they should continuously accumulate evidence through routine version management. This way, when regulatory, client, or partner review requirements arise, the team can generate reports from the system rather than relying on last-minute manual compilation.

The minimum information set must also meet quality thresholds. Some fields may technically exist but contain content too vague to support an audit. For example, a change reason written as "data optimization," a quality conclusion written as "checked," and a compliance opinion written as "no issues" cannot genuinely explain any decision. Audit records should be as specific as possible: what was optimized, which metrics were checked, what is the compliance basis, and what are the limiting conditions. The key to an audit is not filling out a form but enabling subsequent reviewers to understand the judgment made at the time.

The audit trail should also have access controls. Not all audit information is appropriate for all teams to access—user-level source indexes, external vendor contract information, sensitive data authorization documents, and security incident records may all require stricter permissions. The version management system should support visibility controls at different granularities: ordinary R&D staff can view data versions, quality summaries, and experiment associations; Data Owners can view complete change records; compliance and security roles can view authorization records, access logs, and sensitive field processing records. This balances collaborative transparency with limiting the exposure of sensitive information.

---

## 25.4 Lineage Visualization and Governance Rules

### 25.4.1 Representations of Data Lineage Graphs

A data lineage graph is the visual representation of data provenance. A lineage graph is fundamentally a visualization of the relationships among data entities, processing activities, and responsible agents, consistent with the three core concepts of entities, activities, and agents in the W3C PROV data model (Moreau and Missier 2013). It displays the dependency and transformation relationships among data assets in the form of a directed acyclic graph (DAG).

In LLM data engineering, a typical data lineage graph contains the following node types:

- **Data source nodes**: Raw data sources, such as "Web crawl – CommonCrawl 2024Q1" and "Vendor A annotation batch 202403"
- **Processing nodes**: Data transformation steps, such as "deduplication filtering," "language identification," and "annotation quality review"
- **Dataset nodes**: A dataset at a particular version, such as "dialogue-sft-zh_v2.3.1"
- **Experiment nodes**: Experiments using a particular dataset, such as "edu-math_exp003"
- **Model nodes**: Models produced by training, such as "edu-math-7B-v1.2"

Directed edges between nodes express "this node produced that node"; edges can be annotated with transformation rules (e.g., the version of the cleaning script) and the transformation time.

Data lineage graphs support three typical query perspectives:

**Forward tracking**: Starting from a data source, trace all downstream outputs—which training sets did this crawled data ultimately enter, which experiments was it used in, and which models did it produce? Forward tracking supports impact analysis (if this data source has a problem, which artifacts are affected?).

**Reverse tracking**: Starting from a model or experiment, trace all upstream inputs—where did this model's training data come from, what processing did it undergo, and who conducted the quality evaluation? Reverse tracking supports root-cause analysis.

**Diff comparison**: Compare the lineage-level differences between two different versions of a dataset—which data sources changed, which processing steps changed, and how large is the difference? Diff comparison supports change auditing.

The value of a lineage graph lies not merely in "drawing it" but in being queryable and interpretable. Many teams have drawn data flow diagrams in documentation, but once such diagrams are decoupled from the actual system, they quickly become outdated. Lineage graphs with genuine governance value should be automatically or semi-automatically generated by data pipelines, version tools, experiment tracking systems, and release systems, and should be updated as data objects change. Manually maintained diagrams may be useful for training and communication but must not be the sole basis for auditing and retrospection.

Lineage graphs also need complexity control. LLM data pipelines often contain large numbers of data sources, processing steps, and experiment nodes; displaying all nodes at once makes the graph unreadable. Lineage visualization should therefore support filtering by perspective: during quality investigation, display data sources, processing steps, and quality reports; during experiment analysis, display datasets, experiments, and model metrics; during compliance auditing, display sources, authorizations, approvals, and usage scope. Different roles see not a single large graph but different views of the same underlying lineage data.

Edge semantics should also be carefully designed. Edges are not merely "connections"—they should also specify the type of relationship. For example, `derived_from` denotes derivation, `filtered_by` denotes filtering, `annotated_by` denotes annotation, `evaluated_with` denotes evaluation, `approved_by` denotes approval, and `released_as` denotes release. The more clearly edge semantics are defined, the more precise queries become. Otherwise, a lineage graph can only display paths without explaining the governance meaning of those paths.

![Figure 25-2: Data Lineage and Experiment Tracking Graph](../../images/part8/图25_2zh.png)

*Figure 25-2: Complete data lineage graph from data sources to model release, showing forward and reverse tracking paths*

### 25.4.2 Change Audit Workflow

Every change to a dataset version should go through a standardized audit workflow to ensure the change is authorized, recorded, and verified.

The standard change audit workflow is as follows:

1. **Change request**: The proposing party (typically a data engineer or annotation engineer) fills out a change request form, describing the change content, the reason for the change, and the expected impact.
2. **Impact assessment**: Assess which downstream datasets and experiments this change will affect.
3. **Compliance review**: If the change involves a change in data source or data type, review by a legal compliance specialist is required.
4. **Technical review**: The Data Owner or a senior data engineer reviews the technical implementation plan for the change.
5. **Change execution**: Execute the change after creating a rollback point.
6. **Change verification**: Run automated quality checks and compare against the expected impact described in the change request.
7. **Change recording**: Write the change log into the dataset's metadata and update the lineage graph.

| Step | Executor | Tool | Output |
|---|---|---|---|
| Change request | Requesting party | Change request form | Completed request form |
| Impact assessment | Data engineer | Lineage graph query tool | Impact scope list |
| Compliance review | Legal compliance | Compliance review checklist | Review conclusion (approved / rejected / conditionally approved) |
| Technical review | Data Owner | Code review | Approval comments |
| Change execution | Data engineer | Processing scripts + version tools | New dataset version |
| Change verification | Quality evaluator | Automated quality check tools | Quality report |
| Change recording | Automated | Metadata service | Updated lineage graph and change log |

*Table 25-6: Data change audit workflow*

The core of change auditing is connecting "pre-change judgment" with "post-change verification." The change request phase articulates expected impacts—for example, a new data source will improve task coverage, or a modified filtering rule will reduce duplication. The change verification phase must then check whether actual results match expectations. If the expected and actual results diverge significantly, the team should not simply release the new version but should re-evaluate the change assumption. This transforms change auditing from a mere approval workflow into a miniature experiment feedback loop.

Impact assessment is especially important. A seemingly local cleaning rule change may affect multiple downstream datasets; a label adjustment may make historical experiments and new experiments incompatible; a sample deletion may affect an already-frozen evaluation set. Impact assessment should use the lineage graph to automatically generate an initial scope, after which data engineers and the Data Owner assess the business implications. Relying entirely on human memory for impact assessment is a hallmark of an immature version management system.

Change records should also document "rejected alternatives." Many important decisions are not made with a single technical option but involve trade-offs among multiple options. For example, when a data quality problem is found in a batch, options may include deletion, repair, down-weighting, or isolation; when adding a new data source, options may include direct merging, routing to an experimental branch first, or using it only for evaluation. Recording why the current option was chosen and why others were rejected aids subsequent retrospection and prevents the team from re-debating the same question months later.

### 25.4.3 Lineage Governance Rules

Lineage governance rules are a set of "data behavioral guidelines" agreed upon by the data team, specifying which operations are permitted, which require approval, and which are prohibited. A survey on data provenance points out that lineage information only generates genuine engineering value when it is connected to querying, auditing, debugging, and reproduction needs (Simmhan, Plale and Gannon 2005).

The following is a reference set of governance rules:

**Freely permitted operations (no approval required)**

- Creating data branches in a sandbox environment for local experiments
- Viewing and exporting statistical summaries of datasets
- Adding or modifying non-critical metadata for datasets (e.g., tags, annotations)

**Operations requiring approval (lightweight approval by data engineer or Tech Lead)**

Any data version entering model training must pass automated quality checks. Automated data validation systems help teams detect data quality issues before training through statistical constraints, schema checks, and anomaly detection (Breck et al. 2019).

- Merging new data shards into the mainline dataset
- Modifying data cleaning rules
- Adding new data source categories

**Operations requiring formal approval (Data Owner + legal compliance approval)**

- Changing the core sources of a dataset (e.g., replacing the annotation vendor, adding new crawl sites)
- Deleting an existing dataset version
- Sharing an internal dataset with an external partner
- Modifying a released dataset version (prohibited in principle; exceptional cases require complete documentation)

**Permanently prohibited**

- Modifying a frozen dataset without a version record
- Using third-party copyrighted data without compliance review
- Directly modifying data submitted by others without leaving a modification record

Governance rules must also be tied to the data lifecycle. From collection, cleaning, annotation, training, and evaluation through release to archival, the operations permitted at each stage differ. The collection stage allows relatively flexible source exploration but requires recording authorization and provenance. The cleaning stage allows iterative rule adjustment but should retain processing script versions. The annotation stage allows guideline revisions but must record the effective date. The training stage requires stable dataset versions. After release, the emphasis shifts to freezing, auditing, and rollback. Applying a single uniform set of rules to all stages makes early exploration overly burdensome and makes the release stage insufficiently rigorous.

Governance rule enforcement should also be tiered. Low-risk operations can be completed through system prompts and automatic recording. Medium-risk operations require lightweight approval. High-risk operations require formal approval and auditing. Prohibited operations should be blocked directly by the system. Text-based policies alone cannot guarantee compliance because people under project pressure tend to bypass processes. Embedding rules into tools—such as frozen versions being read-only, failing quality checks blocking release, and unauthorized data sources being blocked from the mainline—is what genuinely reduces the probability of violations.

| Lifecycle Stage | Permitted Focus | Key Constraints | Primary Evidence |
|---|---|---|---|
| Collection | Explore data sources, build sample pools | Source, authorization, and sensitivity classification must be recorded | Source manifest, authorization records, collection logs |
| Cleaning | Adjust rules, generate shards | Script versions and filtering outcomes must be traceable | Processing scripts, quality summaries, diff reports |
| Annotation | Task distribution, guideline calibration, result collection | Guideline version and annotator information must be recorded | Annotation logs, guideline version, consistency reports |
| Training | Assemble datasets, launch experiments | Dataset version must be stable and citable | Experiment card, data version, parameter configuration |
| Evaluation | Compare model performance, analyze errors | Evaluation set version and evaluation code must be fixed | Metric records, error samples, evaluation commit |
| Release | Freeze model and data evidence | Approval, quality, and compliance chains must be complete | Release package, approval records, audit report |
| Archival | Reduce storage costs, retain necessary evidence | Information required for reproduction and auditing must not be destroyed | Archival policy, index records, recovery instructions |

*Table 25-7: Lineage governance rules by data lifecycle stage*

Lineage governance also requires periodic review. Common review items include: whether any data shards exist without source records; whether any official models exist without associated experiments; whether any frozen datasets have been modified; whether any experiments use expired evaluation sets; and whether any unclosed compliance exceptions exist. Through periodic review, teams can identify systematic gaps in the governance chain rather than waiting for incidents to occur before investigating.

Finally, governance rules should remain interpretable. Team members are more willing to follow rules not because there are many rules but because they understand the risks behind each rule. If the data lead can explain how a rule supports reproduction, rollback, or compliance, resistance to enforcement drops significantly. Conversely, if rules manifest only as additional approvals and forms, teams may regard them as a burden. Version management and experiment tracking governance must serve both engineering efficiency and risk control simultaneously.

Lineage governance should also be integrated with storage cost governance. As the number of versions grows, training data, shards, intermediate results, and model checkpoints can rapidly consume large amounts of storage. Without lifecycle policies, teams either retain everything indefinitely at uncontrolled cost or delete arbitrarily, causing reproduction failures. A reasonable approach is to define different retention policies for different objects: frozen datasets and release packages are retained long-term; candidate versions are archived after the next stable version is established; temporary intermediate results are cleaned up once confirmed to have no audit value. Cleanup actions themselves should also be recorded so future queries can explain why certain intermediate artifacts are no longer available.

For multimodal data with high storage costs, a tiered storage strategy may also be adopted. Hot data is kept in high-performance storage to support recent experiments and quality investigation; warm data is moved to lower-cost object storage with indexes and metadata retained; cold data is moved to archival storage, retrieved only for auditing or reproducing experiments. Regardless of strategy, metadata and version indexes must remain queryable online; otherwise, archival becomes another form of data loss.

---

## 25.5 Case Study: How to Rapidly Trace Back a Failed Experiment

### Case Background

A company is training a large conversational model for customer service. After the third major version iteration (`v3.0.0`) is released, online evaluation reveals that the model's accuracy on "refund process" questions has plummeted from 82% to 67%, triggering customer complaints.

The algorithm team immediately submits a retrospection request to the data team: **What changes occurred to "refund process"-related data in the past 6 weeks?**

On the surface, this question is about model performance degradation; in substance, it is a data change attribution test. Customer service data has pronounced business time-sensitivity: policies, procedures, scripts, and exception handling rules all change over time. For this type of data, "outdated" does not necessarily mean "without value," because online users may still inquire about historical orders, previous-version entitlements, or special processes during a migration period. If quality review judges sample validity only against the current process, it may erroneously delete data that retains historical explanatory value.

The company had completed foundational version management construction in the previous quarter and was now able to associate model release packages, training experiments, dataset versions, shard sources, quality reviews, and annotation records. The data team therefore did not start by manually searching through folders but instead began from the release package and experiment card, tracing back along the lineage chain. This starting point was deterministic rather than dependent on individual memory.

### Retrospection Process (With Version Management)

**Step 1: Locate the dataset version corresponding to the model version** (Time: 5 minutes)

Query the experiment tracking system for the training experiment ID corresponding to model `v3.0.0`: `cs-dialog_20240401_exp012`.

Query the experiment card to find the dataset version: `cs-dialog-sft-zh_v2.8.0`.

At this stage, the team also confirmed that the training code, evaluation code, and base model version had undergone no major changes. This action is critical because it first rules out the main confounding non-data factors. If training code and data change simultaneously, attribution becomes much more difficult; if evaluation code changes, the accuracy drop may merely reflect a change in evaluation definitions. The code commits, evaluation set versions, and hyperparameter records in the experiment card enable the team to quickly narrow the investigation scope.

**Step 2: Compare differences between the current version and the previous stable version** (Time: 15 minutes)

Using the lineage query tool, compare the differences between `v2.8.0` and `v2.6.0` (the most recent version with good performance):

- Total sample count: `v2.6.0` has 182,000 samples; `v2.8.0` has 214,000 samples.
- Samples added: 32,156 (from two new shards)
- Samples removed: 1,823 (removed by quality filtering)

The diff report also shows that the two new shards primarily come from a new round of annotation tasks from Vendor B, targeting supplementation of after-sales questions, while the deleted samples were concentrated in the quality filtering step rather than in the collection or annotation stage. This suggests the problem may not lie in the newly added data itself but in the quality review or filtering rules. The team therefore did not immediately roll back all of `v2.8.0` but continued to narrow the scope to avoid losing other valid improvements through a blunt rollback.

**Step 3: Analyze changes in "refund process" label data** (Time: 20 minutes)

Filter by business label to examine the distribution of "refund process" samples across both versions:

- `v2.6.0`: 6,847 refund process samples
- `v2.8.0`: 4,102 refund process samples (a decrease of 40%!)

Further analysis reveals that the decrease in refund process samples was not evenly distributed. Samples related to the current process remained largely stable; the largest decreases were in sub-categories such as legacy processes, historical orders, cross-system refunds, and manual customer service interventions. These sub-categories represent a small proportion of total samples yet correspond precisely to the scenarios most prone to error in online complaints. This finding shows that looking only at total sample volume or overall quality metrics is insufficient for identifying risk; distribution comparisons by business label and sub-scenario are essential.

**Step 4: Trace the source of the reduction** (Time: 30 minutes)

Query the reason for the reduction in "refund process" samples, locating shard `vendor_b_annotation_20240318_003`:

- In this shard, 3,201 "refund process" samples had been tagged as "low quality."
- Tracing to the specific quality review record: reviewer `QA_Engineer_Wang` conducted a batch review on 2024-03-20.

Querying `QA_Engineer_Wang`'s review records reveals: on March 20, he uniformly marked a batch of samples about the "legacy refund process" as low quality (reason: "process has been updated; description is outdated"), which were then removed by the quality filtering step.

The review records also preserve the justification at the time: the new customer service knowledge base had already replaced the refund process documentation, and the quality review guidelines stated that "outdated processes should not enter the formal training set." Judged at the individual-rule level, the reviewer followed the process; the real problem was that the guideline had not distinguished between "processes that have lapsed and should not be answered" and "processes that still need to be explained historically." Without review records, this type of issue is easily misidentified as individual operational error rather than incomplete quality standard design.

**Step 5: Confirm root cause and formulate a remediation plan** (Time: 15 minutes)

**Root cause**: Quality reviewers assessed data quality against current business processes, treating samples describing "legacy refund processes" as low-quality data. However, legacy process handling scenarios (refunds on historical orders) still exist online, and these samples have important value for helping the model understand "refund process evolution."

**Remediation plan**:

1. Short-term: Retrieve deleted refund process samples from `v2.6.0`, re-label them with a "historical refund process" tag, and re-ingest them.
2. Medium-term: Update quality review guidelines to explicitly define retention principles for "historical process data."
3. Long-term: Add an annotation dimension "applicable to historical scenarios?" to annotation tasks.

The entire retrospection took approximately 85 minutes from "problem report" to "root cause confirmed"—a sharp contrast to the earlier case of "5 days without version management."

Without version management, this investigation would likely have taken a very different path: the algorithm team first suspects training parameters and re-examines training logs; the data team then searches through recent weeks' data folders to determine which files were used for `v3.0.0`; the quality team needs to manually ask reviewers whether they processed refund samples; and the business team needs to recall when the refund process was updated. Every step depends on human memory, and different team members may give inconsistent timelines. Even if the cause is eventually identified, it is difficult to prove which operation caused the performance degradation.

With version management, the retrospection path flows from model to experiment, experiment to dataset, dataset to shard, shard to quality review, and review record to business rule. This is a structured evidence chain. It not only shortens investigation time but also changes the way the team discusses problems: discussions are no longer centered on "who might have changed the data" but on "where the evidence chain shows a rule produced an unintended effect."

| Retrospection Step | Typical Approach Without Version Management | Approach With Version Management | Difference |
|---|---|---|---|
| Locate training data | Ask experiment owner; search file folders | Find dataset version through experiment card | Starting point changes from memory to record |
| Compare data differences | Manually compare directories and file names | Automatically generate version diff report | Differences change from approximate to structured |
| Find problematic samples | Write ad-hoc scripts to scan historical files | Query lineage by label and shard | Scope narrows from full dataset to relevant subset |
| Confirm reason for change | Ask reviewers or search chat history | Check review records and rule versions | Reason changes from verbal explanation to evidence |
| Formulate remediation | Roll back the entire dataset or re-clean everything | Precisely restore affected samples and update rules | Fix changes from blunt to targeted |

*Table 25-8: Comparison of retrospection approaches with and without version management*

### Summary of Key Success Factors

The rapid retrospection succeeded because of the following critical version management design decisions. Reproducible computation research has repeatedly emphasized that only by binding data, code, environment, and results together in records can failed experiments become analyzable assets rather than one-time incidents (Peng 2011; Stodden, Leisch and Peng 2014):

1. **Bidirectional association between experiments and data**: Knowing which dataset version corresponds to `v3.0.0`.
2. **Shard-level source tracking**: Ability to locate the specific annotation batch.
3. **Complete preservation of review records**: Knowing who performed what review operations at what time.
4. **Version maintenance of business labels**: Ability to filter historical-version data by label.

The absence of any one of these would have broken the retrospection chain.

This case also demonstrates that version management is not merely a technical system—it is also a form of organizational collaboration. Experiment cards require the algorithm team to fill them in carefully; shard sources require the data engineering team to maintain in a standardized manner; quality review records require evaluators to document completely; business labels require product or business experts to participate in defining; and remediation plans require confirmation by multiple roles. Whenever any role treats record-keeping as extra overhead, the credibility of the entire chain is weakened.

From a governance perspective, the root cause in this case was not "incorrect sample deletion" but "quality rules failing to express the business time dimension." The remediation plan therefore cannot remain at the level of sample restoration; it must also update annotation and review guidelines: categorizing legacy processes, transitional processes, deprecated processes, and processes that still need to be explained historically; setting different handling strategies for each category; and adding an applicable time range or business status field to the data schema. This way, findings generated by version management can be translated into improvements to data standards.

From an experiment tracking perspective, a set of comparison experiments should also be run subsequently: a first group using the original `v2.8.0` data, a second group with historical refund samples restored, and a third group with restored samples and additional historical process labels. Through comparison experiments, the team can determine whether the accuracy recovery comes from increased sample volume or from the improved label dimension. If the data is only fixed without doing comparison experiments, the team resolves the immediate problem but cannot extract a more generalizable method.

Finally, this case also prompts teams to establish special governance rules for "business time-sensitive data." In customer service, policy, financial, educational, medical, and enterprise knowledge-base scenarios, whether data is outdated is not a simple yes/no question. Some old information must be deleted; some must be retained but with a time boundary attached; some can only be used for historical explanation and cannot guide current operations. Version management can help teams see these distinctions, but genuine governance still requires the joint participation of business semantics, quality rules, and experiment feedback.

When implementing in an enterprise, the version management system discussed in Chapter 25 can be advanced along a "record first, associate second, govern third" path. "Record first" means establishing stable IDs for datasets, shards, experiments, and release packages and recording the minimum required fields. "Associate second" means connecting data versions to experiment results, model versions, quality reports, and approval records. "Govern third" means gradually adding approval, rollback, archival, and access controls once an evidence chain is in place. This sequence matters. If full governance is pursued from the start without stable underlying object records, processes become hollow; if records exist without associations, data still fails to explain model performance.

The goal of the first phase should be to stop the team from using uninterpretable folder naming. Even without a complex platform, a unified version number, a dataset description sheet, and an experiment card can be established first. As long as every experiment can state which data version was used, which shards compose the dataset, and which sources the shards came from, the team has already crossed the threshold from verbal memory to structured records. This phase does not require much automation but does require strict enforcement of minimum fields.

The goal of the second phase is to integrate version records into actual workflows. When a dataset is released, a version description is automatically generated; when an experiment starts, the data version is automatically bound; when an experiment ends, metrics are automatically written back; and quality check results are automatically incorporated into dataset metadata. At this point, version management is no longer an additional document to fill in—it is the path through which teams complete their work. Only by entering the workflow will version records be stable, complete, and timely.

The goal of the third phase is to enable the versioning system to generate governance capability. Teams can use lineage relationships for impact analysis, use experiment write-back to assess data asset value, use audit trails to respond to compliance questions, and use rollback points to reduce the cost of high-risk changes. At this phase, version management is no longer merely an engineering support tool—it has become a foundation for organizational decision-making. Data Owners can use it to judge which datasets deserve long-term maintenance; algorithm leads can use it to determine which experiment conclusions are reliable; and compliance teams can use it to assess whether risk boundaries are clearly defined.

It is also worth noting that version management does not inherently guarantee correctness. An erroneous data version can be completely recorded; an experiment of insufficient quality can be accurately tracked. What a versioning system provides is interpretability and traceability, not automatic validation that all decisions are correct. Therefore, it must be used in conjunction with quality assessment, experiment design, compliance review, and business expert judgment. Only when all these mechanisms operate together can version management truly support reliable iteration.

In the long term, the maturity of data version management will directly influence the R&D velocity of large model teams. Without version management, every model fluctuation requires rebuilding the investigation from scratch. With basic version management, teams can reproduce experiments. With experiment tracking and result write-back, teams can compare data strategies. With lineage governance and audit trails, teams can share data assets across a larger organizational scope. The higher the maturity, the more teams can transform each experiment into organizational knowledge rather than letting experience dissipate with each project cycle.

The core of this chapter, therefore, is not to recommend any particular concrete tool but to emphasize an engineering principle: any data change that will affect model results, compliance boundaries, or organizational decisions should leave a queryable, interpretable, and verifiable record. Tools may differ; platform forms may differ; but this principle must never be absent.

---

## Chapter Summary

This chapter has systematically articulated the core design of version management and experiment tracking in LLM data engineering.

At the conceptual level, the chapter analyzed the three fundamental reasons why data changes are difficult to track (inconsistent granularity, complex causal chains, and delayed impact), and the necessity of upgrading from "folder management" to "lineage management." The core value of version management lies in three capabilities: experiment reproducibility, problem traceability, and compliance auditability.

At the version granularity level, the chapter defined a five-level version hierarchy spanning samples, shards, datasets, experiments, and release packages, as well as semantic version naming conventions and three types of version operations—branches, snapshots, and rollback points.

At the experiment tracking level, we designed complete experiment card fields (basic information, data configuration, model configuration, runtime environment configuration, evaluation results, and experiment records), emphasizing the knowledge preservation value of failed experiments and the importance of result write-back in forming bidirectional associations.

At the lineage governance level, the chapter introduced three query perspectives for data lineage graphs (forward tracking, reverse tracking, and diff comparison), as well as a standardized change audit workflow and tiered governance rules.

Finally, through a complete retrospective case study of a failed experiment, the chapter demonstrated the tangible value that a version management system delivers in real scenarios—compressing retrospection time from 5 days to 85 minutes.

---

## Further Reading

**Recommended Tools**

DVC (Data Version Control) is one of the most widely used data version control tools, deeply integrated with Git and supporting version management of large files and data lineage tracking. The "Data Registry" pattern in its documentation is a reference implementation for multi-project data sharing. MLflow is an open-source machine learning experiment tracking platform supporting unified recording of experiment parameters, metrics, and artifacts, as well as a visualization interface for version comparison. LakeFS is a version control tool for data lakes that provides Git-like branching, merging, and rollback operations, suited to version management of large-scale datasets.

**In-Depth Reading**

Zaharia et al.'s "Accelerating the Machine Learning Lifecycle with MLflow" (2018) is the original paper introducing MLflow and systematically presents the challenges and solutions for ML lifecycle management. Google's "Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI" (2022) provides best practices for dataset documentation and is an important reference for experiment card design.

---

## References

Amershi S, Begel A, Bird C, DeLine R, Gall H, Kamar E, Nagappan N, Nushi B, Zimmermann T (2019) Software Engineering for Machine Learning: A Case Study. In: Proceedings of the 41st International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP), pp 291-300.

Armbrust M, Ghodsi A, Xin R, Zaharia M (2020) Delta Lake: High-Performance ACID Table Storage over Cloud Object Stores. Proceedings of the VLDB Endowment 13(12):3411-3424.

Baylor D, Breck E, Cheng H-T, Fiedel N, Foo C Y, Haque Z, Haykal S, Ispir M, Jain V, Koc L, Koo C Y, Lew L, Mewald C, Modi A N, Polyzotis N, Ramesh S, Roy S, Whang S E, Wicke M, Wilkiewicz J, Zhang X, Zinkevich M (2017) TFX: A TensorFlow-Based Production-Scale Machine Learning Platform. In: Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp 1387-1395.

Breck E, Cai S, Nielsen E, Salib M, Sculley D (2017) The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. In: IEEE International Conference on Big Data, pp 1123-1132.

Breck E, Polyzotis N, Roy S, Whang S E, Zinkevich M (2019) Data Validation for Machine Learning. In: Proceedings of Machine Learning and Systems 1, pp 334-347.

Buneman P, Khanna S, Tan W-C (2001) Why and Where: A Characterization of Data Provenance. In: Proceedings of the 8th International Conference on Database Theory (ICDT), pp 316-330.

DAMA International (2017) DAMA-DMBOK: Data Management Body of Knowledge, 2nd Edition. Technics Publications.

DVC Documentation (2024) Data Version Control Documentation. Available at: https://dvc.org/doc

Gebru T, Morgenstern J, Vecchione B, Vaughan J W, Wallach H, Daumé III H, Crawford K (2021) Datasheets for Datasets. Communications of the ACM 64(12):86-92.

Kreuzberger D, Kühl N, Hirschl S (2023) Machine Learning Operations (MLOps): Overview, Definition, and Architecture. IEEE Access 11:31866-31879.

Mitchell M, Wu S, Zaldivar A, Barnes P, Vasserman L, Hutchinson B, Spitzer E, Raji I D, Gebru T (2019) Model Cards for Model Reporting. In: Proceedings of the Conference on Fairness, Accountability, and Transparency, pp 220-229.

Moreau L, Missier P (eds.) (2013) PROV-DM: The PROV Data Model. W3C Recommendation.

Peng R D (2011) Reproducible Research in Computational Science. Science 334(6060):1226-1227.

Polyzotis N, Roy S, Whang S E, Zinkevich M (2017) Data Management Challenges in Production Machine Learning. In: Proceedings of the 2017 ACM International Conference on Management of Data (SIGMOD), pp 1723-1726.

Sandve G K, Nekrutenko A, Taylor J, Hovig E (2013) Ten Simple Rules for Reproducible Computational Research. PLOS Computational Biology 9(10):e1003285.

Sculley D, Holt G, Golovin D, Davydov E, Phillips T, Ebner D, Chaudhary V, Young M, Crespo J-F, Dennison D (2015) Hidden Technical Debt in Machine Learning Systems. In: Advances in Neural Information Processing Systems 28, pp 2503-2511.

Simmhan Y L, Plale B, Gannon D (2005) A Survey of Data Provenance in e-Science. ACM SIGMOD Record 34(3):31-36.

Stodden V, Leisch F, Peng R D (eds.) (2014) Implementing Reproducible Research. CRC Press.

Vartak M, Subramanyam H, Lee W-E, Viswanathan S, Husnoo S, Madden S, Zaharia M (2016) ModelDB: A System for Machine Learning Model Management. In: Proceedings of the Workshop on Human-In-the-Loop Data Analytics (HILDA), Article 14.

Zaharia M, Chen A, Davidson A, Ghodsi A, Hong S A, Konwinski A, Murching S, Nykodym T, Ogilvie P, Parkhe M, Xie F (2018) Accelerating the Machine Learning Lifecycle with MLflow. IEEE Data Engineering Bulletin 41(4):39-45.
