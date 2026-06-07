# Chapter 27: Data Catalogs and Metadata Governance

## Introduction to Data Operations and Platform Development

As large-model applications deepen inside enterprises, the value of data becomes more visible. Yet many organizations still treat data as a pile of files. When data scientists need a dataset, they often locate sources by asking around, searching manually, or reading stale documents. This rough form of data management lowers reuse, increases compliance risk, hides quality problems, and encourages repeated construction.

In real enterprise environments, these problems worsen quickly as data scale grows. When multiple datasets exist across departments, collection windows, and application scenarios, but there is no unified catalog, version management, lineage tracking, or access control, data users cannot reliably answer basic questions: which dataset is latest, which one fits a business scenario, where it was produced, what processing it went through, what it may or may not be used for, who owns its quality, and when it will be retired.

This part focuses on how data teams move from reactive handling to proactive governance in the age of large-model applications. Earlier parts discussed how to construct training data and optimize application-level data. Here the question changes: in a continuously evolving, cross-department, multi-application data environment, how can an organization build a systematic data-asset governance system so that data is no longer scattered files, but a strategic asset that can be managed, controlled, and reused?

This chapter develops that question from three angles: data-asset definition, metadata governance, and lineage plus lifecycle management. The emphasis is organizational collaboration, process discipline, and long-term maintainability. The goal is to let every data user find the data they need, judge whether it is usable, use it safely, and receive timely notifications when it changes.

This shift is aligned with the broader movement toward data-centric AI. Over the past decade, industry investment in model architecture and compute has far exceeded investment in data governance, but production failures often originate in neglected data quality, documentation, and process issues (Sambasivan et al. 2021). The data-management field has also systematized these concerns: DAMA-DMBOK treats metadata management, data quality, data security, and data governance as mutually reinforcing functions (DAMA International 2017). This chapter asks how those principles can become engineering practice for large-model data teams.

## Chapter Guide

Enterprise data engineering often has a paradoxical condition: the organization has a great deal of data, but much of it is "visible yet unreachable." Data is scattered across databases, data lakes, object storage, and local files. Documentation lives in wikis, emails, and shared folders. There is no unified inventory of what exists, where it is, who can use it, and how good it is.

In this environment, reuse becomes expensive. A new project spends large amounts of time rediscovering, reprocessing, and revalidating data that may already exist. Worse, disorder becomes a source of compliance risk: sensitive data is used without clear permission records, unclear lineage blocks impact analysis, and version confusion makes audit trails unreliable.

Large-model workflows make the problem harder than traditional governance. The same data may be reused and transformed across pre-training, fine-tuning, RAG, and evaluation. A user-interaction log may train a preference model, populate a RAG knowledge base, and support system evaluation. Each use has different requirements for cleanliness, completeness, and freshness. Without a catalog, it is difficult to determine whether one dataset is suitable for one purpose.

Production machine-learning data lifecycle management is a known systems problem (Polyzotis et al. 2018). Enterprise data assets are always moving: new data arrives, old data expires, schemas evolve, and permissions change with the organization. Without a catalog and metadata layer, teams rely on memory and oral communication. That can barely work in a small team and fails at cross-department scale.

The core goal of this chapter is to build a framework that moves from data-asset definition to registration, governance, and application. It solves not only "where is the data?" but also "what can the data do?", "how good is it?", "who owns it?", and "who is allowed to use it?" With a centralized data catalog, normalized metadata model, complete lineage tracking, and strict permission governance, organizations can turn data from passive resources into active assets (Halevy et al. 2016).

## 27.1 Why Data Assets Are Not File Inventories

### 27.1.1 A Realistic Case of Enterprise Data Disorder

An AI team at a large internet company is building a user-preference fine-tuning dataset for a recommender system. The project lead asks the data team whether user click feedback exists. The data team says yes and points to a dataset path in the data lake. The project team spends two weeks importing, cleaning, and formatting it locally.

After training begins, model performance is far worse than expected. After repeated debugging, the team discovers the issue: the dataset came from a temporary analysis project three months earlier. Many records had been marked invalid because of a known logging bug, but that flag was never documented clearly. The dataset also used an older user taxonomy that was incompatible with the current system, causing widespread feature mismatch. Worse, it was supposed to be used only for one business line, but the team did not know that restriction and applied it elsewhere.

At the same time, another team building a knowledge-base RAG system needs high-quality text-pair data. They remember that another project collected a large volume of user feedback text. After weeks of searching, they find the data in the personal cloud drive of a retired employee. It has not been updated for two years, and there is no record of whether it may be used for model training or whether privacy compliance was checked.

These cases are common in organizations without data catalogs and metadata management. The root problem is not only data quality. It is that the visibility, trustworthiness, and usability of data cannot be guaranteed systematically.

In both cases, the data existed. What did not exist was knowledge about the data. Invalid flags, taxonomy versions, use restrictions, update times, training authorization, and privacy status may have existed in someone's memory or in a temporary document, but they were not attached to the dataset in a structured, searchable form. This separation between data and context is the breeding ground for data cascades: a small documentation gap upstream is amplified downstream into poor model performance, compliance exposure, and expensive rework (Sambasivan et al. 2021).

### 27.1.2 Definition and Dimensions of a Data Asset

In traditional file management, a "dataset" is often just files at a storage location. That view ignores context, quality, usage rules, and evolution.

In modern data governance, a **data asset** must include several dimensions:

1. **Identity and ownership**: a globally unique identifier and an owner responsible for quality and updates.
2. **Structure and schema**: fields, data types, value ranges, partitioning, and constraints.
3. **Lineage and transformation**: sources and the full transformation chain from source data to current form.
4. **Permissions and security**: who can read, modify, or delete the asset, especially for sensitive data.
5. **Quality and confidence**: quality metrics such as missingness, duplication, outliers, and distribution drift.
6. **Lifecycle**: creation, active use, maintenance, deprecation, archiving, and deletion states.

These dimensions follow a long line of data-quality research. Data quality is not simply accuracy; it includes completeness, timeliness, consistency, accessibility, and believability from the consumer's perspective (Wang and Strong 1996). The quality and confidence dimension translates "is this data good enough?" into measurable indicators.

Machine-learning practice has also made dataset documentation an engineering norm. Datasheets for Datasets asks every dataset to document motivation, composition, collection process, recommended uses, and known limitations (Gebru et al. 2021). Model Cards apply a similar idea to models (Mitchell et al. 2019). A data catalog systematizes these artifacts at organizational scale: a datasheet explains one dataset, while a catalog makes thousands of datasets discoverable, comparable, and governable.

### 27.1.3 From File Inventories to Data Catalogs

The shift is conceptual. A file inventory asks where data is. A data catalog asks what the data is, where it came from, what it can be used for, who owns it, and how trustworthy it is.

```yaml
asset_id: user_interaction_feedback_v3
owner: data-governance-team@company.com
schema:
  - user_id (string, required)
  - interaction_type (enum: like/dislike/share/collect)
  - has_invalid_flag (boolean)
quality_metrics: {completeness: 0.98, validity: 0.97, freshness: daily}
permissions: {read: [ux_team, ml_team], write: [data_governance_team]}
restrictions:
  - Do not use for cross-border data sharing.
  - Do not use for user-profiling models that may create discrimination risk.
end_of_life: 2025-12-31
```

The `restrictions` field is important because it states what cannot be done. In a traditional file inventory, such constraints often live only in memory. Once written into metadata, they become rules that can be checked and audited.

### 27.1.4 Section Summary

The difference between data assets and file inventories reflects the shift from passive to active governance. A data catalog lets users judge quickly whether a dataset is suitable for a project. In the age of large-model applications, building such a catalog is not optional infrastructure; it is a prerequisite for large-scale data use.

## 27.2 Dataset Registry and Metadata Model

### 27.2.1 Core Concepts of a Dataset Registry

If a data catalog is the map of data, a **dataset registry** is the passport system for data. Every data asset receives a unique identifier and a structured metadata description. The registry is not only a passive record; it actively supports governance, reuse, and compliance.

At scale, a dataset registry must solve three problems:

- **Discoverability**: users can quickly find datasets through search, browsing, and filtering (Fernandez et al. 2018).
- **Trustworthiness**: metadata and quality information are sufficient for users to judge usability.
- **Governability**: data managers can maintain assets, track usage, and govern version evolution.

Systems such as Google's Goods, Ground, and Data Tamer illustrate the direction. Goods automatically cataloged billions of internal datasets and inferred sources, owners, schemas, and dependencies (Halevy et al. 2016). Ground proposed a data context service that manages application, behavior, and change context around data (Hellerstein et al. 2017). Data Tamer showed how heterogeneous data sources can be curated and matched at scale with expert confirmation at key points (Stonebraker et al. 2013). Their shared lesson is that large-scale registries must rely heavily on automatic collection; manual registration alone decays quickly.

![Data asset registration and launch workflow](../../images/part9/ch27_fig01_zh.png)

*Figure 27-1: Data asset registration and launch workflow*

### 27.2.2 Complete Metadata Model

A production dataset registry usually includes the following metadata categories.

| Category | Representative Fields | Purpose |
| --- | --- | --- |
| **Identity** | asset_id, asset_name, description, asset_type | Unique identifier and basic description |
| **Ownership** | owner_id, steward_id, business_owner | Quality, maintenance, and business responsibility |
| **Structure** | schema, partitions, primary_key, row_count | Field definitions and physical structure |
| **Source and lineage** | source_systems, lineage, dependencies | Source tracing and dependency analysis |
| **Versioning** | version, changelog, schema_version | Version evolution |
| **Quality metrics** | quality_score, completeness, validity, timeliness, uniqueness | Quantified usability |
| **Usage records** | access_count, last_accessed, downstream_jobs, use_cases | Usage intensity and downstream dependency |
| **Access control** | access_level, read/write/delete_groups, compliance_tags | Access and compliance labels |
| **Risk and lifecycle** | risk_level, status, deprecation_date, end_of_life, retention_policy | Risk and lifecycle management |
| **License and compliance** | license, privacy_classification, pii_fields, data_residency | Legal and privacy requirements |

In real systems, each category expands into detailed fields. Quality may include completeness, validity, freshness, deduplication rate, and accuracy. Access control may separate read, write, and delete groups and attach compliance tags. Lifecycle records creation, deprecation, and planned deletion. This two-level category-field structure gives a clear top-level view while supporting detailed governance (Cai and Zhu 2015; Rahm and Bernstein 2001).

Once a metadata model is adopted, it becomes a contract between teams. If all assets use the same categories and fields, cross-team discovery, comparison, and integration become possible. If every team invents its own metadata format, the catalog becomes a collection of incompatible islands. The design principle should be: **minimal mandatory core, flexible optional extensions**; and **unified terminology with controlled values** for fields such as `access_level`, `status`, and `content_type`.

### 27.2.3 Layering and Inheritance in Metadata Models

Different data assets need different metadata extensions. Training datasets may need balance and feature-distribution fields. RAG knowledge bases may need retrieval quality and update-frequency fields. Evaluation sets may need golden-answer coverage and difficulty distribution.

A practical metadata model uses layers:

1. **Core metadata**: mandatory fields for all assets, including identity, ownership, structure, source, quality, and permissions.
2. **Type-specific metadata**: extensions for tables, files, streams, embeddings, and other asset types.
3. **Scenario-specific metadata**: fields for training data, evaluation data, RAG data, and similar uses.
4. **Custom metadata**: organization-specific fields.

Layering guarantees minimum usability while supporting flexibility. It also reduces the cost of merging and aligning models across departments (Noy and Musen 2000).

### 27.2.4 Automatic Metadata Collection and Maintenance

Complete metadata is important, but manual maintenance is expensive and error-prone. Production registries use automatic collection and periodic validation.

- **Structure** can be extracted by scanning database schemas, Parquet headers, JSON samples, and profiling results (Abedjan, Golab and Naumann 2015).
- **Lineage** can be inferred from processing DAGs.
- **Quality metrics** can be refreshed by scheduled quality checks.
- **Access records** can be collected from audit logs.

Metadata updates should follow a policy:

- **Automatic updates** for row count, modification time, and access statistics.
- **Passive updates** triggered when upstream sources change.
- **Human confirmation** for business meaning, usage scenarios, and risk.
- **Periodic validation** to detect drift between metadata and reality.

Systems such as Deequ let engineers define data-quality constraints declaratively and validate them at scale (Schelter et al. 2018). ML-focused validation frameworks can infer expected schemas and statistics and detect distribution drift in new batches before bad data reaches training (Breck et al. 2019). Connecting these tools to the registry turns `quality_metrics` from stale manual numbers into continuously refreshed signals.

### 27.2.5 Metadata Search and Discovery

The value of a registry depends on discovery. Good search usually includes:

- **Keyword search** across `asset_name`, `description`, and schema fields.
- **Filtered search** by asset type, owner, status, access level, and quality score.
- **Lineage search** for downstream assets depending on a source, or upstream assets required by an application.
- **Similar-data recommendation** based on schema and use-case similarity.
- **Quality-metric search**, such as "tables accessed more than 100 times in the last 30 days with completeness above 0.95."

Search quality directly affects reuse. If searching the catalog is slower than asking a colleague, users bypass it. Goods and Aurum show that the goal is to let users search data almost as naturally as searching the web, while using inferred schema similarity, usage relationships, and source lineage to build a navigable relationship network (Halevy et al. 2016; Fernandez et al. 2018).

### 27.2.6 Section Summary

A dataset registry uses structured metadata to turn files in storage into discoverable, understandable, trustworthy assets. The key is a metadata model that is both comprehensive and flexible, supported by automatic collection, validation, and strong search.

## 27.3 Lineage, Permissions, and Lifecycle

### 27.3.1 Complete Data Lineage Tracking

Data lineage is the soul of data-asset governance. It answers where data came from, what transformations it passed through, what it looks like now, and where it will flow. It supports troubleshooting, impact analysis, compliance audits, and data-quality diagnosis (Herschel, Diestelkämper and Ben Lahmar 2017).

Large-model workflows make lineage complex. One raw user-interaction log can enter multiple pipelines: preference-model training, knowledge-base construction, and system evaluation. Without clear lineage, it is hard to answer where a model's training data came from or what models are affected when an upstream field definition changes.

Lineage has three layers:

1. **System-level lineage**: the physical systems through which data flows, such as Kafka, stream processing, data lake, and offline warehouse.
2. **Logical transformation lineage**: how data is processed, such as raw click events, deduplication, aggregation, feature engineering, and vectorization.
3. **Semantic lineage**: how business meaning and risk change, such as when an anonymous user ID becomes linkable to profile attributes.

This corresponds to the classic idea of data provenance: why and where a result came from, which inputs and operations produced it, and how it should be interpreted (Buneman, Khanna and Tan 2001). Valuable lineage is not just a few upstream/downstream lines; it must support field-level impact analysis and compliance questioning.

```yaml
dataset: user_preference_features_v2
sources:
  - user_click_logs (Kafka topic)
transformations:
  - bot_filter: remove bot accounts (98% retained, data_quality_team)
  - dedup: deduplicate by (user, item, ts) (95% retained, data_quality_team)
  - feature_eng: aggregate feature vectors (depends on user_profile_table)
  - privacy_masking: hash user_id (GDPR 5.1.2, privacy_team)
downstream_assets:
  - preference_model_training
  - rag_knowledge_embeddings
  - model_evaluation_dataset
impact_analysis:
  - user_profile_table changes -> rerun feature_eng (severity: high)
  - quality failure in this dataset -> block dependent training jobs (severity: high)
```

The `impact_analysis` field makes lineage actionable. It records who is affected by upstream changes and what downstream systems are at risk if this asset fails.

![Data lineage graph](../../images/part9/ch27_fig02_zh.png)

*Figure 27-2: Data lineage graph*

### 27.3.2 Permission Governance and Access Control

Permission governance answers who can access what data, in what scenario, for what purpose, and how access is audited. It involves technical controls such as table-, row-, and column-level permissions, and organizational processes such as approval, periodic review, and anomaly alerts.

Permission layers include:

- **Dataset-level permissions** for the whole asset.
- **Table/field-level permissions** for fine-grained control.
- **Row-level permissions** based on conditions such as geography.
- **Contextual permissions** based on time, location, and purpose.

Role-Based Access Control (RBAC) reduces permission-maintenance complexity by assigning permissions to roles rather than individuals (Sandhu et al. 1996). In data-asset governance, RBAC often needs to be combined with attributes: sensitivity level, compliance tags, and access context. The same user data should have different visible granularity when used for preference-model training versus business analysis.

```yaml
dataset: user_interaction_feedback
permissions:
  ml_training_team:      full_raw_data
  rag_engineering_team:  deidentified_features
  business_analytics_team: aggregated_stats
  data_governance_team:  full_admin
```

A complete configuration would include access rationale, fields, row conditions, approval requirements, and audit sampling rates. The design minimizes sensitive exposure while preserving legitimate work.

Permissions must also be reviewed. Each asset should periodically check that:

1. Existing users still need access.
2. Missing users receive appropriate access.
3. Abnormal access patterns are investigated.
4. Permissions align with least privilege.

Permission governance also requires audit and anomaly detection. A query may be technically allowed but still suspicious, such as a normally daytime analyst account suddenly downloading row-level raw data at night. Audit logs are themselves data assets and should be protected and retained.

### 27.3.3 Lifecycle Management and State Transitions

Data assets are not permanent. They move from creation to active use, deprecation, archiving, and deletion. Each state needs explicit definition and management.

Lifecycle matters because exits and entrances both have engineering consequences. Stale data that never retires occupies storage, pollutes search results, and may be reused incorrectly. Sudden deletion or schema change can break downstream pipelines. Modeling lifecycle as a state machine makes dependency changes explicit and auditable (Sculley et al. 2015; Polyzotis et al. 2018).

| State | Characteristics | Typical Duration | Transition Condition | User Operation |
| --- | --- | --- | --- | --- |
| **CREATED** | Created, metadata complete, not yet online | 0-2 weeks | Pass quality and security review | Wait for activation |
| **ACTIVE** | In normal use, updated regularly, discoverable | Business-dependent | Need for retirement | Normal use |
| **DEPRECATED** | Planned retirement; new projects should not use it | 3-6 months | Downstreams migrated | Discouraged use |
| **ARCHIVED** | No active updates, retained for audit and history | Long term | Need for full deletion | Read-only |
| **DELETED** | Physically deleted if legally allowed | - | Permanent operation | No access |

![Data asset lifecycle state machine](../../images/part9/ch27_fig03_zh.png)

*Figure 27-3: Data asset lifecycle state machine*

Key transition activities include:

- **ACTIVE to DEPRECATED**: document the reason, migration path, user notifications, support plan, and migration deadline.
- **DEPRECATED to ARCHIVED**: confirm downstream migration, move to cold storage, keep read-only access for history and audit, and scan for unexpected new access.
- **ARCHIVED to DELETED**: confirm legal retention periods, obtain approvals, execute physical deletion, log deletion, and clean backups and snapshots.

Deletion is a compliance-sensitive operation in the age of GDPR and PIPL. A right-to-erasure request may require deleting original data, copies, backups, and derivatives. Without lineage, an organization cannot even know what downstream assets must be removed.

### 27.3.4 Section Summary

Lineage, permissions, and lifecycle are the three pillars of data-asset governance. Lineage explains flow and transformation. Permissions control access and reduce compliance and security risk. Lifecycle prevents assets from accumulating indefinitely or disappearing unexpectedly.

## 27.4 Catalog Cases and Templates

### 27.4.1 Example Enterprise Data Catalog

An e-commerce recommender system touches nearly all governance dimensions in this chapter: real-time streams, offline feature tables, training datasets, evaluation benchmarks, vector databases, RAG knowledge bases, and compliance logs. These assets belong to different teams, update at different rates, have different sensitivity levels, and depend on one another.

| Asset ID | Type | Quality Score | Status | Main Use |
| --- | --- | --- | --- | --- |
| `raw_user_click` | Stream | 0.91 | ACTIVE | Real-time recommendation, training source |
| `user_features` | Table | 0.96 | ACTIVE | SFT features, personalization |
| `product_emb_v3` | Vector DB | 0.94 | ACTIVE | RAG retrieval, similar-item recommendation |
| `pref_sft_v2` | Dataset | 0.98 | ACTIVE | Fine-tuning preference model |
| `rank_benchmark` | Dataset | 0.97 | ACTIVE | Model performance evaluation |
| `kb_chunks` | Table | 0.93 | ACTIVE | RAG knowledge source |
| `feedback_labeled` | Table | 0.94 | ACTIVE | SFT supervision and alignment |
| `click_v1_legacy` | Table | 0.78 | DEPRECATED | Historical analysis and migration reference |
| `audit_log` | Table | 0.96 | ACTIVE | Compliance checks and access tracing |

Even this small excerpt shows the catalog's value: heterogeneous asset types are visible together, quality is comparable, and deprecated assets are clearly marked.

### 27.4.2 Detailed Metadata Example for One Asset

```yaml
asset_id: user_preference_sft_v2
description: User preference scores and interaction feedback for preference-model training
owner_id: ml_training_team@company.com

storage: s3://.../user_preference_sft/v2/ (parquet, 45GB, 128M rows)
schema:
  - user_id:          string(hashed), unique user identifier
  - preference_score: float[0,1], higher means stronger preference

version: 2.1.0
lineage: raw_user_click -> bot_filter -> feature_engineering -> [preference ranking model, cold-start model]
quality_metrics: {overall: 0.98, completeness: 0.99, validity: 0.97}
known_issues: 0.3% out-of-range scores under repair; unstable scores for cold-start users

access_level: internal
pii_handling: user_id hashed
compliance: [GDPR, CCPA]
data_residency: US
status: ACTIVE
retention: 3y
expected_active_until: 2026-12-31
```

A mature data asset's metadata is a small specification. It records what the data is, whether it can be used, how it should be used, and how long it remains valid.

### 27.4.3 Overall Governance View Across Assets

Data-catalog managers need a dashboard view across hundreds or thousands of assets.

| Dimension | Representative Metric | Target | Alert Threshold |
| --- | --- | --- | --- |
| **Coverage** | Assets with metadata / lineage | >95% / >90% | <90% / <80% |
| **Quality** | Average quality score | >0.90 | <0.80 |
| **Compliance** | PII access violations | 0 | Any violation |
| **Lifecycle** | Deprecated assets not migrated; orphaned data ratio | 0 / <5% | >5% / >10% |
| **Usage** | Active asset ratio; reuse rate | >80% / >60% | <70% / <40% |

These metrics turn "how good is governance?" into measurable operational indicators. They must be collected continuously and automatically; otherwise, they become stale reporting numbers. Mature platforms embed metric collection into pipelines so the dashboard becomes an operational tool rather than a one-time slide.

![Role- and purpose-based permission matrix](../../images/part9/ch27_fig04_zh.png)

*Figure 27-4: Example permission-governance matrix*

### 27.4.4 Section Summary

The example shows how a data catalog is applied in practice. The key is a governance system that is broad enough to cover all assets, detailed enough to describe each asset, and observable enough to monitor governance health.

## 27.5 Governance Maturity, Roles, and Implementation Checklist

### 27.5.1 Maturity Evolution

Data governance is not a one-time project to build a catalog. It evolves in stages:

1. **Ad hoc**: no unified catalog; data is scattered across team tables, documents, and personal storage. Discovery depends on oral communication.
2. **Managed**: a centralized catalog appears; core assets register basic metadata such as owner, schema, and description. Metadata is largely manual.
3. **Governed**: metadata collection becomes automated; lineage, quality metrics, and permissions are standard. Registration flows, lifecycle state machines, and periodic permission reviews are institutionalized.
4. **Optimized**: governance integrates deeply into production and consumption. Quality checks run inside pipelines, lineage is extracted from job graphs, and access/compliance checks happen before data enters indexes.

This path reminds teams not to attempt everything at once. Cover high-value and high-risk core assets first, then expand. For many organizations, the biggest leap is from a manually maintained static catalog to an automatically collected living catalog.

### 27.5.2 Organizational Roles

Data governance is not only a technical system; it is a responsibility structure.

- **Data Owner**: often a business or team lead, ultimately responsible for quality, compliance, value, usage scope, and permission strategy.
- **Data Steward**: maintains metadata, monitors quality, and responds to issues. The steward connects business and technical context.
- **Governance Team**: defines global standards such as metadata schema, naming conventions, classification, compliance rules, and directory operation.
- **Data Consumer**: data scientists, algorithm engineers, analysts, and others who discover and use data; their feedback drives continuous improvement.

The earlier failure cases occur partly because ownership and stewardship are absent. If nobody owns whether a dataset can be used for training or what its restrictions are, that knowledge will not be recorded.

### 27.5.3 Data-Asset Governance Checklist

| Category | Key Question | Acceptance Standard |
| --- | --- | --- |
| Identity and ownership | Does the asset have a unique ID, owner, and steward? | Every asset maps to responsible people |
| Metadata completeness | Are structure, description, purpose, and restrictions registered? | Core fields complete; restrictions explicit |
| Lineage tracking | Are source, transformation, and downstream consumption recorded? | Can trace forward and backward |
| Quality monitoring | Are quality metrics measured automatically and continuously? | Metrics refreshed by pipelines |
| Permissions and compliance | Are role- and purpose-based permissions configured and PII tagged? | No unauthorized sensitive-data access |
| Lifecycle | Are state and transition conditions defined? | Deprecation, archive, and deletion have plans and approvals |
| Discoverability | Can users search, filter, and receive recommendations? | Target users can find data in reasonable time |
| Feedback loop | Can consumers report and close data issues? | Problems are reported, assigned, and repaired |

This checklist turns implicit expectations scattered across roles into one acceptance table. For high-risk data such as personal, financial, or medical data, stricter audits, sensitive-information detection, and access logging should be added.

### 27.5.4 Special Governance Challenges for Large Models

Large-model data governance faces challenges beyond traditional warehouses.

**Source and authorization tracking for training data.** Once data is used in model training, its influence is embedded into parameters and is hard to remove afterward. Lineage, authorization status, and compliance tags must be correct before data enters training. A dataset without training-authorization records can create serious compliance risk.

**Freshness and version governance for RAG knowledge sources.** RAG knowledge bases update continuously. Old documents must be retired or archived, or the model may answer with stale information. Lifecycle state machines support controlled knowledge evolution.

**Isolation and leakage protection for evaluation data.** If an evaluation set leaks into training data, results become invalid. Evaluation assets require strict permission isolation and lineage tracking.

These challenges are not only about data quality. They are about whether knowledge about the data is recorded and enforced. Without governance, risks cascade downstream into model-level failures (Sambasivan et al. 2021).

### 27.5.5 Section Summary

This section explained how data-asset governance lands in real organizations through maturity stages, role definitions, and implementation checklists. The goal is not merely to build a system. It is to ensure that knowledge and responsibility about data circulate reliably inside the organization.

## Chapter Summary

This chapter began with the definition of data assets and explained the difference between a data asset and a file inventory. A data asset is not only stored data; it includes identity, ownership, structure, lineage, permissions, quality, and lifecycle.

By building a dataset registry, enterprises can transform scattered data into assets that can be discovered and trusted. A structured metadata model covering identity, ownership, structure, lineage, versioning, quality, usage, permissions, lifecycle, and compliance provides the foundation for manageability and usability.

Lineage tracking lets organizations understand data flow and supports impact analysis and troubleshooting. Permission governance protects sensitive data. Lifecycle management prevents data assets from accumulating forever or disappearing unexpectedly.

In the age of large-model applications, this governance system is especially important. The same data may be used for pre-training, SFT, RAG knowledge sources, and evaluation, with different requirements in each scenario. Effective governance makes complex cross-application reuse possible without relying on oral communication and manual management.

Mature data organizations should build:

1. **Automatic metadata collection** from schemas, DAGs, audit logs, and quality-validation frameworks.
2. **Complete lineage tracking** from source systems to downstream applications.
3. **Flexible permission models** based on RBAC plus attributes and context.
4. **Health monitoring** that turns governance into measurable operational indicators.
5. **Usage feedback loops** that bring consumer reports and reuse needs back into governance.
6. **Clear roles and processes** for owners, stewards, governance teams, and consumers.

When these capabilities are in place, data stops being an isolated storage object and becomes a manageable, traceable, reusable enterprise asset. This is the shift from "having enough data" to "having effective data."

Ultimately, data catalogs and metadata governance serve a larger trend: AI competition is moving from model-centric to data-centric. As model architectures become more homogeneous and compute becomes purchasable, the real organizational gap increasingly lies in the quality, governability, and reusability of data assets. An organization that can answer what data it has, where it came from, how good it is, who can use it, and until when can continuously turn data into product capability while avoiding repeated discovery, compliance exposure, and hidden technical debt.

## References

Abedjan Z, Golab L, Naumann F (2015) Profiling relational data: a survey. The VLDB Journal 24(4):557-581.

Breck E, Polyzotis N, Roy S, Whang S E, Zinkevich M (2019) Data Validation for Machine Learning. In: Proceedings of the 2nd SysML Conference (MLSys).

Buneman P, Khanna S, Tan W-C (2001) Why and Where: A Characterization of Data Provenance. In: Proceedings of the 8th International Conference on Database Theory (ICDT), pp 316-330.

Cai L, Zhu Y (2015) The Challenges of Data Quality and Data Quality Metrics. Journal of Data and Information Quality 6(2-3):1-10.

DAMA International (2017) DAMA-DMBOK: Data Management Body of Knowledge, 2nd Edition. Technics Publications, Basking Ridge.

Fernandez R C, Abedjan Z, Koko F, Yuan G, Madden S, Stonebraker M (2018) Aurum: A Data Discovery System. In: 2018 IEEE 34th International Conference on Data Engineering (ICDE), pp 1001-1012.

Gebru T, Morgenstern J, Vecchione B, Vaughan J W, Wallach H, Daumé III H, Crawford K (2021) Datasheets for Datasets. Communications of the ACM 64(12):86-92.

Halevy A, Korn F, Noy N F, Olston C, Polyzotis N, Roy S, Whang S E (2016) Goods: Organizing Google's Datasets. In: Proceedings of the 2016 ACM SIGMOD International Conference on Management of Data, pp 795-806.

Hellerstein J M, Sreekanti V, Gonzalez J E, Dalton J, Dey A, Nag S, Ramachandran K, Arora S, Bhattacharyya A, Das S, Donsky M, Fierro G, She C, Steinbach C, Subramanian V, Sun E (2017) Ground: A Data Context Service. In: 8th Biennial Conference on Innovative Data Systems Research (CIDR).

Herschel M, Diestelkämper R, Ben Lahmar H (2017) A survey on provenance: What for? What form? What from? The VLDB Journal 26(6):881-906.

Mitchell M, Wu S, Zaldivar A, Barnes P, Vasserman L, Hutchinson B, Spitzer E, Raji I D, Gebru T (2019) Model Cards for Model Reporting. In: Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT*), pp 220-229.

Noy N F, Musen M A (2000) PROMPT: Algorithm and Tool for Automated Ontology Merging and Alignment. In: Proceedings of the 17th National Conference on Artificial Intelligence (AAAI), pp 450-455.

Polyzotis N, Roy S, Whang S E, Zinkevich M (2018) Data Lifecycle Challenges in Production Machine Learning: A Survey. ACM SIGMOD Record 47(2):17-28.

Rahm E, Bernstein P A (2001) A survey of approaches to automatic schema matching. The VLDB Journal 10(4):334-350.

Sambasivan N, Kapania S, Highfill H, Akrong D, Paritosh P, Aroyo L M (2021) "Everyone wants to do the model work, not the data work": Data Cascades in High-Stakes AI. In: Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, pp 1-15.

Sandhu R S, Coyne E J, Feinstein H L, Youman C E (1996) Role-Based Access Control Models. IEEE Computer 29(2):38-47.

Schelter S, Lange D, Schmidt P, Celikel M, Biessmann F, Grafberger A (2018) Automating Large-Scale Data Quality Verification. Proceedings of the VLDB Endowment 11(12):1781-1794.

Sculley D, Holt G, Golovin D, Davydov E, Phillips T, Ebner D, Chaudhary V, Young M, Crespo J-F, Dennison D (2015) Hidden Technical Debt in Machine Learning Systems. In: Advances in Neural Information Processing Systems 28, pp 2503-2511.

Stonebraker M, Bruckner D, Ilyas I F, Beskales G, Cherniack M, Zdonik S, Pagan A, Xu S (2013) Data Curation at Scale: The Data Tamer System. In: 6th Biennial Conference on Innovative Data Systems Research (CIDR).

Wang R Y, Strong D M (1996) Beyond Accuracy: What Data Quality Means to Data Consumers. Journal of Management Information Systems 12(4):5-33.
