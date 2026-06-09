# Chapter 27: Data Asset Catalog and Metadata Governance

## Abstract

In large language model data engineering, enterprise data frequently exists in a state of being "visible but unreachable": data is scattered across disparate storage systems, and critical facts about that data—provenance, definitions, usage restrictions, and responsible parties—reside only in the memories of a few individuals or in ad hoc documents. This situation gives rise to data cascade failures and compliance risks. This chapter systematically addresses these issues through data asset catalogs and metadata governance. It begins by defining the six dimensions that distinguish a data asset from a mere file inventory—identity and ownership, structural schema, lineage and transformation, permissions and security, quality and confidence, and lifecycle—drawing on the documentation conventions of datasheets for datasets. It then discusses the layered metadata model of a dataset registry, along with mechanisms for automated collection and search-based discovery. Subsequent sections analyze data lineage, role- and attribute-based access control, and lifecycle management modeled as a state machine. The chapter concludes by examining governance maturity progression, organizational role responsibilities, and an actionable implementation checklist, explaining how to translate this methodology into sustainable day-to-day operations within a real organization.

## Keywords

Data asset catalog; metadata governance; dataset registry; data lineage; permissions governance; lifecycle management

## Learning Objectives

- Distinguish data assets from file inventories, and characterize a data asset along six dimensions: identity and ownership, structural schema, lineage and transformation, permissions and security, quality and confidence, and lifecycle.
- Design a layered metadata model for a Dataset Registry, and explain automated collection and search-discovery mechanisms using datasheet conventions.
- Construct a governance framework encompassing data lineage, RBAC and ABAC access control, and state-machine-based lifecycle management.
- Assess an organization's data governance maturity, identify role responsibilities, and implement an actionable governance checklist.
- Explain how the decoupling of data from its context leads to data cascade failures and the associated engineering costs.

------

## Introduction: Data Assets, Data Products, and Data Contracts

As large language model applications deepen, the value of data within enterprises has grown increasingly prominent. Yet many organizations remain at a stage where data is treated as an accumulation of files—when a data scientist needs a dataset, they typically locate the data source through informal inquiries, manual searches, or outdated documentation. This rudimentary approach to data management not only results in low data utilization, but also introduces compliance risks, quality deficiencies, and redundant engineering efforts. In real enterprise environments, such problems worsen exponentially as data volume grows. When a system contains multiple copies of a dataset—possibly originating from different departments, collected at different times, and applied in different contexts—without unified catalog management, version control, lineage tracking, or access control, a cascade of problems emerges. Data consumers cannot accurately determine which dataset is the most current, which dataset applies to their use case, where the data was generated, what processing it has undergone, what purposes it may and may not serve, who is responsible for its quality, or when it will be retired.

This part focuses on how data teams in the large language model era can upgrade from a reactive posture to proactive governance. In earlier parts, we discussed how to construct training data and optimize application-level data. In this part, the central question shifts to: in a continuously evolving data environment that spans multiple departments and serves multiple applications, how can one establish a systematic data asset governance framework—one that transforms data from files scattered across the enterprise into strategic assets that are manageable, controllable, and reusable? This transformation carries profound organizational significance. The maturity of data governance directly determines the credibility of an organization's data-driven decisions, and equally determines the costs that technical teams incur from redundant efforts, compliance exposure, and efficiency losses.

This part will unfold along three directions—the definition of data assets, metadata governance, and lineage and lifecycle management—to systematically discuss how to build an enterprise-grade data catalog, registry, and governance framework. Unlike earlier parts, this section places greater emphasis on organizational collaboration, process standardization, and long-term maintainability. Its core objective is: to enable every data consumer within an organization to quickly find the data they need, accurately assess its usability, use it safely, and receive timely notifications when data changes.

It is worth emphasizing that this transformation aligns closely with the data-centric AI movement that has emerged in recent years. Over the past decade, the industry has invested far more in model architectures and computing power than in the governance of data itself. Yet a growing body of research points out that what truly constrains the reliability of AI systems in production is often the long-neglected problems of data quality, data documentation, and data process management (Sambasivan et al. 2021). When data lacks documentation, accountable owners, and quality records, downstream systems accumulate cascading, hard-to-trace data cascade failures whose costs typically materialize only after a system goes live. At the same time, the data management community has codified these practices into systematic knowledge frameworks; for example, the data management body of knowledge identifies metadata management, data quality, data security, and data governance as mutually reinforcing core functions (DAMA International 2017). It is against this backdrop that this part discusses how data teams in the large language model era can translate these principles into engineering practice.

The central question this part seeks to answer is: in the era of large language model data engineering, how can one build a governance framework that spans the full data lifecycle, elevating data from a passive resource to an active asset?

------

## Chapter Overview

In enterprise data engineering practice, a pervasive phenomenon exists: although organizations have accumulated vast amounts of data, that data is often in a state of being "visible but unreachable." Data is dispersed across different databases, data lakes, object storage systems, and local file systems; documentation is scattered across team wikis, emails, and shared folders; there is no unified inventory explaining what data exists, where it lives, who can use it, or what quality it offers. In such an environment, the cost of data reuse is extremely high. When a new project launches, teams must expend considerable effort rediscovering, reprocessing, and revalidating data that may already exist but has never been systematically managed. More seriously, this state of disarray also becomes a breeding ground for compliance risks. Sensitive data is used by multiple teams without explicit access records; unclear data lineage makes impact analysis impossible; version confusion prevents audits from tracing the true state of the data.

Unlike traditional data governance, the large language model era imposes new demands on data catalog and metadata management. Across the multiple stages of pre-training, fine-tuning, RAG, and evaluation, the same dataset may be reused multiple times, transformed multiple times, and updated through multiple versions. Systems must accurately track how data is used and what its quality status is across all these different stages. For example, a set of user interaction logs may be used simultaneously for training a preference model, for building a RAG knowledge base, and for system evaluation. Each use case imposes different requirements on data cleanliness, completeness, and freshness. Without a data catalog, it becomes very difficult to determine whether a given dataset meets the requirements of a specific purpose.

In fact, managing the data lifecycle in production-grade machine learning systems is itself a well-studied and challenging problem (Polyzotis et al. 2018). Unlike one-time datasets, enterprise data assets exist in a state of continuous flux and ongoing versioning: new data is constantly ingested, old data gradually expires, schemas evolve with the business, and permissions are adjusted as the organization changes. Without a catalog and metadata layer to record these changes, data teams can only rely on individual memory and informal communication to maintain systems—an approach that may barely function within a small team, but inevitably breaks down at cross-departmental, multi-application scale. The significance of data catalogs and metadata governance lies precisely in consolidating knowledge that is dispersed across individual memories and ad hoc documents into organization-level, queryable, auditable shared infrastructure.

Accordingly, the core objective of this chapter is to establish a complete framework spanning data asset definition, registration, governance, and application. This framework must address not only the discovery problem of "where is the data," but also a series of management questions: "what can the data be used for," "what is the data quality," "who is responsible for this data," and "who can use this data." By building a centralized data catalog, a standardized metadata model, complete lineage tracking, and rigorous permissions governance, an organization can transform data from a passive resource into an active asset (Halevy et al. 2016), thereby accelerating data application development while mitigating compliance and quality risks.

------

## 27.1 Why a Data Asset Is Not a File Inventory

### 27.1.1 Real-World Cases of Enterprise Data Chaos

An AI team at a large internet company was building a new user preference fine-tuning dataset for a recommendation system. The project lead first asked the data team whether user click feedback data was available. The data team replied "yes" and pointed to a dataset path in the data lake. The project team spent two weeks importing the data into their local environment, cleaning it, and formatting it. However, once model training began, they found that performance fell far short of expectations. After repeated debugging, they finally identified the root cause: the dataset came from a temporary analysis project three months earlier, in which a large number of records had been flagged as invalid due to a known bug in the log collection system—but this flag had never been clearly documented. Furthermore, the dataset used an older user classification taxonomy that was incompatible with the new version currently in use by the system, causing widespread feature mismatches. To make matters worse, the dataset was supposed to be used only for a specific line of business, but the team had no knowledge of this restriction, and inadvertently applied it to an unsuitable line of business.

Meanwhile, another team was building a knowledge-base RAG system and needed high-quality text pair data. They recalled that a company project had once collected a large volume of user feedback text. After weeks of searching and coordination, they eventually found this data on the personal cloud storage of a former employee—but the data had not been updated in two years, and much of its content was outdated. More importantly, the dataset had never been explicitly marked as permitted for model training, and contained no record of privacy compliance.

These two cases are not hypothetical. In organizations lacking data catalogs and metadata management, such situations are widespread. The root cause is not data quality per se, but the inability to systematically guarantee data visibility, trustworthiness, and usability.

Analyzing the root causes of these two cases reveals that their failures were not attributable to "data not existing," but to "knowledge about the data not existing." In the first case, critical facts such as the invalid flag, the taxonomy version, and the usage scope restriction had existed at some point in someone's memory or a temporary document, but were never attached to the data in a structured, searchable form. In the second case, the update timestamp, training authorization status, and privacy compliance state had similarly never been recorded. This state of "data decoupled from its context" is precisely the classic breeding ground for data cascade failures—an apparently minor documentation gap at an early stage is repeatedly amplified downstream, ultimately manifesting as poor model performance, compliance exposure, or the need to start over entirely (Sambasivan et al. 2021). In other words, what these teams lost was not the data itself, but the time, compute, and trust expended in re-acquiring "knowledge about the data."

### 27.1.2 Definition and Dimensions of a Data Asset

In the traditional file-management mindset, a "dataset" is simply a collection of files at some storage location. The problem with this view is that it completely ignores the context, quality, usage rules, and evolution of the data.

In modern data governance frameworks, a **data asset** must encompass the following key dimensions:

**1. Identity and Ownership Dimension** — A globally unique identifier, along with an owner accountable for data quality and updates.

**2. Structure and Schema Dimension** — Clearly defined fields, data types, value ranges, and partitioning schemes, which form the foundation for detecting quality issues.

**3. Lineage and Transformation Dimension** — Records of data provenance and the complete transformation chain from source data to its current form, essential for impact analysis.

**4. Permissions and Security Dimension** — Definitions of who may read, modify, and delete the data; particularly important for access control on sensitive data.

**5. Quality and Confidence Dimension** — Associated quality metrics (missing rate, duplication rate, anomalies, distribution shift, etc.), enabling users to quickly assess whether the data meets their requirements.

**6. Lifecycle Dimension** — From creation, active use, and maintenance through deprecation to deletion, each stage requires explicit status markers and management policies.

These six dimensions are not arbitrary; they are directly connected to long-standing research in data quality and data documentation. As early as the 1990s, researchers observed that data quality is a multidimensional concept that cannot simply be equated with "accuracy," but must be understood from the perspective of data consumers, encompassing multiple dimensions such as completeness, timeliness, consistency, accessibility, and credibility (Wang and Strong 1996). The "quality and confidence dimension" of a data asset is a direct continuation of this thinking in engineering practice: it requires decomposing the abstract question of "is the data usable" into a set of measurable, comparable concrete metrics, so that different users' understandings of "quality" can be aligned to a common standard.

In recent years, the machine learning community has further established "writing formal documentation for datasets" as an engineering norm. Datasheets for Datasets proposes that every dataset should, like a datasheet for an electronic component, be accompanied by documentation explaining its collection motivation, composition, collection process, intended uses, and known limitations (Gebru et al. 2021). The complementary Model Cards provides a similar documentation standard for models, requiring clear recording of a model's applicable scope, evaluation results, and potential biases (Mitchell et al. 2019). These documentation artifacts are entirely consistent in spirit with the "data asset dimensions" discussed in this chapter—both aim to transform critical context about data (or models) from personal knowledge into shareable, auditable structured records. A data asset catalog can be understood as the systematic, organizational-scale automation of such documentation conventions: a single dataset's datasheet answers "what is this one dataset," while the data asset catalog goes further to answer "how can tens of thousands of datasets be uniformly discovered, compared, and governed."

### 27.1.3 From File Inventory to Data Asset Catalog

This transition involves a fundamental shift in mindset. A file inventory is concerned with "where is the data," while a data asset catalog is concerned with "what is the data, where did it come from, what can it do, who is responsible for it, and what is its quality." Below is a concise partial example of structured metadata that binds the answers to these questions directly to the data:

```yaml
asset_id: user_interaction_feedback_v3      # globally unique identifier
owner: Data Governance Team (data-governance@company.com)
schema:                                      # fields, types, and constraints
  - name: user_id
    type: string
    required: true
  - name: interaction_type
    type: enum
    allowed_values:
      - like
      - dislike
      - share
      - collect
  - name: has_invalid_flag
    type: boolean
    description: whether the record is marked as invalid
quality_metrics:
  completeness: 0.98
  validity: 0.97
  freshness: daily
permissions:
  read:
    - ux_team
    - ml_team
  write:
    - data_governance_team
restrictions:                               # usage restrictions
  - Not to be used for cross-border data sharing (contains Chinese data, subject to PIPL)
  - Not to be used for user profiling (may lead to discrimination)
end_of_life: 2025-12-31                      # planned retirement date
```

The `restrictions` field is particularly noteworthy: it explicitly declares what the data "must not be used for." In a traditional file inventory, such constraints typically exist only in the memory of select individuals; once written into metadata, however, they become hard rules that can be enforced by systems and tracked in audits. It is precisely this contextual information that elevates a collection of files into a data asset that can be used with confidence.

### 27.1.4 Section Summary

The distinction between a data asset and a file inventory reflects the shift in data governance from reactive to proactive. A data asset catalog enables data consumers to quickly assess whether a given dataset is appropriate for their project. In the era of large language model applications, building a data asset catalog is not an optional activity—it is essential infrastructure for large-scale data operations.

---

## 27.2 Dataset Registry and the Metadata Model

### 27.2.1 Core Concepts of the Dataset Registry

If a data asset catalog is "the map of data," then a dataset registry is "the passport of data." Every data asset must obtain a unique identifier and a complete metadata description in the registry. The role of a dataset registry is not merely to passively record information, but to actively support data governance, data reuse, and compliance management.

In large-scale data engineering, a dataset registry must address three core requirements. First, it requires **discoverability**—the ability for users to quickly find the datasets they need through search, browsing, and filtering (Fernandez et al. 2018). Second, it requires **trustworthiness**—providing sufficient metadata and quality information so that users can accurately assess data usability. Third, it requires **governability**—furnishing data managers with management tools that enable them to systematically maintain data assets, track usage, and manage version evolution.

These three requirements are not merely theoretical; multiple influential systems in industry and academia have responded to them. Google's Goods system automatically builds a catalog for billions of datasets within the company through background crawling, inferring their provenance, owners, schemas, and interdependencies, enabling engineers to discover and understand data without prior registration (Halevy et al. 2016). Ground proposed the abstraction of a "data context service," arguing that systems must manage not only data itself but also the application context, behavioral context, and change context about that data (Hellerstein et al. 2017). In data integration and cleaning, Data Tamer demonstrated how to organize, match, and merge heterogeneous data sources at scale in a semi-automated manner, supplemented by expert confirmation at critical decision points (Stonebraker et al. 2013). The shared lesson from these systems is: at scale, the registry must rely as much as possible on automated collection rather than requiring manual entry by users—the latter almost invariably falls into disuse as scale grows, ultimately causing the catalog to drift out of sync with the actual state of the data.

A data asset typically passes through a standardized registration process before it can be discovered and put into use. As shown in Figure 27-1, the data creator first prepares metadata and submits a registration request; the system then performs a combination of automated and manual quality checks and security and permissions reviews; only upon passing these checks is the asset published to the data catalog and made searchable. This process institutionalizes "who may register data and what conditions must be met beforehand" as enforceable checkpoints, which are the prerequisite for the governability of the registry.

![Complete workflow from data asset creation to use](../../images/part9/ch27_fig01_zh.png)

*Figure 27-1: Data Asset Registration and Onboarding Workflow*



### 27.2.2 Complete Definition of the Metadata Model

A production-grade dataset registry typically organizes metadata fields into the following categories. The design of these fields must serve both discovery and usage while also supporting governance and compliance.

*Table 27-1: Core Metadata Field Categories for the Data Asset Registry*

| Field Category | Representative Fields | Purpose |
| --- | --- | --- |
| **Identity** | asset_id, asset_name, description, asset_type | Globally unique identifier and basic description |
| **Ownership** | owner_id, steward_id, business_owner | Clarify quality, maintenance, and business accountability |
| **Structure** | schema, partitions, primary_key, row_count | Field definitions and physical structure |
| **Provenance and Lineage** | source_systems, lineage, dependencies | Trace provenance and dependency relationships |
| **Version and Iteration** | version, changelog, schema_version | Record version evolution |
| **Quality Metrics** | quality_score, completeness, validity, timeliness, uniqueness | Quantify data usability |
| **Usage Records** | access_count, last_accessed, downstream_jobs, use_cases | Reflect usage frequency and downstream dependencies |
| **Access Control** | access_level, read/write/delete_groups, compliance_tags | Control access and compliance tagging |
| **Risk and Lifecycle** | risk_level, status, deprecation_date, end_of_life, retention_policy | Risk assessment and status management |
| **License and Compliance** | license, privacy_classification, pii_fields, data_residency | Satisfy legal and privacy requirements |

The table above lists representative fields by category; in actual production systems, each category is further expanded into multiple more granular fields. For example, quality metrics beyond an overall score are broken down into individual dimensions such as completeness rate, validity rate, timeliness, deduplication rate, and accuracy, allowing users to assess whether data meets the quality threshold for a specific purpose (Cai and Zhu 2015). Access control separately defines user groups with read, write, and delete permissions, accompanied by compliance tags. The lifecycle records key timestamps from creation through deprecation to planned deletion. This two-level "category–field" structure keeps the metadata model's top-level view clear while supporting fine-grained governance at each dimension, and simultaneously provides a unified vocabulary for field alignment and mapping between different data sources (Rahm and Bernstein 2001).

It is worth emphasizing that once the metadata model is established, it becomes a contract among teams within the organization. Only when all data assets are described using the same set of categories and fields does cross-team discovery, comparison, and integration become possible. Conversely, if each team invents its own metadata format, the catalog will degrade into a collection of non-interoperable islands. Therefore, metadata model design should follow two principles: first, **minimum core mandatory, extensions flexibly optional**—core fields (identity, ownership, structure, permissions) are mandatory for all assets, with the remainder extended on demand; second, **unified terminology, controlled vocabularies**—for critical fields such as `access_level`, `status`, and `content_type`, use controlled enumerations rather than free text, to prevent synonymous terms such as "internal," "confidential," and "restricted" from proliferating and causing filter failures. This schema governance is itself a microcosm of data governance at the metadata layer: it requires the organization to make a deliberate trade-off between "flexibility" and "consistency," and to institutionalize that trade-off through standards and tooling (DAMA International 2017).

### 27.2.3 Layering and Inheritance in the Metadata Model

In practice, different types of data assets may require different metadata extensions. For example, training datasets may need additional fields such as "sample balance" and "feature distribution"; RAG knowledge bases may need "retrieval effectiveness" and "update frequency"; evaluation sets may need "golden answer coverage rate" and "difficulty distribution."

To support this flexibility, metadata models typically adopt a layered design:

**Layer 1: Core Metadata** — The minimum set of fields required for all data assets, including identity, ownership, structure, provenance, quality, and permissions.

**Layer 2: Type-Specific Metadata** — Extensions based on the data asset type; for example, tabular data, file data, streaming data, and embedding vectors each have different extension fields.

**Layer 3: Use-Case Metadata** — Metadata specific to particular application scenarios; for example, training data, evaluation data, and RAG data each have their own specific fields.

**Layer 4: Custom Metadata** — Allows organizations to add custom fields based on their own governance requirements.

This layered design ensures minimum viability while supporting high flexibility. When multiple organizations or departments each maintain their own metadata extensions, the layering and inheritance mechanism also reduces the cost of merging and aligning different models (Noy and Musen 2000).

### 27.2.4 Automated Collection and Maintenance of Metadata

Complete metadata is undeniably important, but the cost of maintaining it manually is too high and prone to errors. Production-grade data registries typically adopt mechanisms for automated collection and periodic validation.

For **structural information**, automated extraction is possible by scanning database schemas, Parquet file headers, and JSON file samples; data profiling techniques can further automatically infer field types, value ranges, and potential constraints from samples (Abedjan, Golab and Naumann 2015). For **lineage information**, it can be automatically identified from the DAG of the data processing pipeline. For **quality metrics**, they can be automatically updated through periodic data quality check jobs. For **access records**, they can be automatically aggregated from audit logs.

The key is to establish a **metadata update strategy**:

- **Automatic updates**: Fields that can be automatically collected by the system—such as row counts, last-modified dates, and access statistics—are configured for periodic automated scanning.
- **Passive updates**: When an upstream data source changes (e.g., a new field is added or a constraint is modified), changes are automatically detected and update prompts are triggered.
- **Manual confirmation**: For fields requiring human judgment—such as business meaning, use cases, and risk level—a hybrid mode of "automated detection + manual confirmation" is adopted.
- **Periodic validation**: Metadata accuracy is periodically spot-checked, with automatic alerts when inconsistencies are detected.

For automated collection of quality metrics, there are mature engineering practices to draw upon. Systems such as Deequ allow engineers to declaratively define unit-test-style quality constraints for data (e.g., a given field's non-null rate should exceed a threshold, or a column's values should fall within a given set), and automatically validate these constraints at scale while continuously measuring quality metrics (Schelter et al. 2018). Data validation frameworks targeting machine learning scenarios go further, automatically inferring expected schemas and statistical characteristics from data and detecting deviations from historical distributions when new data batches arrive, thereby issuing alerts before low-quality data enters the training pipeline (Breck et al. 2019). Integrating this kind of quality validation with the metadata registry allows the `quality_metrics` fields to become not static numbers entered by hand and quickly outdated, but continuously refreshed, trustworthy real-time signals driven by the pipeline. This is especially critical for large language model data engineering: training and RAG pipelines routinely consume hundreds of millions of records, and any static, lagging quality description is unlikely to reflect the true state of the data.

### 27.2.5 Metadata Search and Discovery

The value of a dataset registry lies in whether it can help users quickly find the data they need. Good search capabilities typically include:

**Keyword search** — Supports searching within `asset_name`, `description`, and `schema` fields, and returns results sorted by relevance.

**Filtered search** — Supports multi-dimensional filtering by `asset_type`, `owner`, `status`, `access_level`, `quality_score`, and other dimensions.

**Lineage search** — Given a data source, quickly finds all downstream data assets that depend on it; or given an application, finds all upstream data assets it requires.

**Similar data recommendation** — Based on schema similarity and use-case similarity, recommends other potentially useful data assets, helping users discover potential reuse opportunities.

**Quality-metric search** — Users can filter by quality metrics, for example: "find tables that have been accessed more than 100 times in the past 30 days and have completeness > 0.95."

The convenience of search directly influences the willingness to reuse data. Many enterprises that have built data catalogs ultimately failed, often precisely because of poor search experiences that led users to continue asking around informally rather than using the catalog.

This point is particularly apparent in the design of systems such as Goods and Aurum: both take "enabling users to search for data the way they search the web" as a core goal, and weave isolated datasets into a navigable network of relationships by automatically inferring schema similarity, usage relationships, and upstream provenance (Halevy et al. 2016; Fernandez et al. 2018). Their lesson is that the success or failure of a data catalog often does not depend on how comprehensive the metadata model design is, but on whether users can find a sufficiently good candidate dataset within seconds. If a catalog search takes longer than simply asking a colleague, the catalog will be bypassed; with no one using it, it receives no maintenance; with no maintenance, its usability further declines—a vicious cycle. Search and discovery capability should therefore be regarded as the "storefront" of the data catalog, requiring sustained investment and continuous refinement rather than being considered complete once built.

### 27.2.6 Section Summary

Through a structured metadata model, the dataset registry transforms data from "files in storage" into "assets that can be discovered, understood, and trusted." The key is to build a metadata model that is both comprehensive and flexible, supports automated collection and validation, and provides strong search and discovery capabilities. Only when all these mechanisms are in place does the data catalog truly become an accelerator for organizational data reuse.

---

## 27.3 Lineage, Permissions, and Lifecycle

### 27.3.1 Complete Lineage Tracking

Data lineage is the soul of data asset governance. It answers a critical question: where did this data come from, what transformations has it undergone, what does it look like now, and where will it flow? Clear lineage relationships are essential across multiple scenarios: incident investigation, impact analysis, compliance auditing, and data quality diagnosis (Herschel, Diestelkämper and Ben Lahmar 2017).

In large language model applications, lineage tracking becomes particularly complex. A set of raw user interaction logs may simultaneously enter multiple processing chains: one for training a preference model, one for building a knowledge base, and one for system evaluation. Each chain contains multiple processing steps. Without clear lineage records, it is very difficult to answer questions such as "where did a model's training data come from?" and "if the definition of an upstream field changes, which models will be affected?"

**A complete framework for lineage tracking encompasses three levels:**

**Level 1: System-Level Lineage** — The flow of data between physical systems. For example: raw log storage in Kafka → stream processing system → data lake → offline analytics database. This level answers "what is the physical path of the data."

**Level 2: Logical Transformation Lineage** — The logical transformations applied to data during processing. For example: raw click events → deduplication → aggregation to user level → feature engineering → feature vectors. This level answers "how the data is processed."

**Level 3: Semantic Lineage** — How the business meaning of the data evolves. For example: the `user_id` field in raw click events represents an anonymous user ID → after being joined with a user profile, user characteristics can be inferred → privacy compliance must be considered when using this data for training. This level answers "what are the business risks of the data."

Data lineage has a more classical name in database research—data provenance. Addressing the two questions of "why does a result record have its current value" and "which inputs and operations produced it," researchers have developed different granularities of characterization such as why-provenance and where-provenance (Buneman, Khanna and Tan 2001), and over the following two decades established a systematic methodology covering databases, workflows, and scripted computation (Herschel, Diestelkämper and Ben Lahmar 2017). The three-level lineage discussed in this chapter can be seen as the engineering instantiation of these theories in enterprise data platforms: system-level lineage corresponds to "which systems did the data flow through," logical transformation lineage corresponds to classical why/how-provenance, and semantic lineage extends the scope of provenance from "how the data was computed" to "what the data means and what risks it carries." Understanding this connection helps avoid a common misconception—equating lineage with "recording a few upstream–downstream connections." Truly valuable lineage must be granular enough to support field-level impact analysis and compliance inquiries.

In practice, lineage can be fully described through structured records. The concise example below illustrates the core lineage information for a feature dataset—sources, key transformation steps, downstream consumers, and most importantly, impact analysis:

```yaml
dataset: user_preference_features_v2
sources:
  - name: user_click_logs
    type: Kafka topic
    description: raw user click events, full collection
transformations:                         # each step records the operation, output volume, and owner
  - step: bot_filter
    description: filter out bot accounts
    retained_ratio: 0.98
    owner: data_quality_team
  - step: dedup
    description: deduplicate by user, item, and timestamp
    retained_ratio: 0.95
    owner: data_quality_team
  - step: feature_eng
    description: aggregate to generate feature vectors
    depends_on:
      - user_profile_table
  - step: privacy_masking
    description: mask user_id -> hash
    policy: GDPR 5.1.2
    owner: privacy_team
downstream_assets:
  - preference_model_training              # SFT training data
  - rag_knowledge_embeddings               # RAG knowledge base
  - model_evaluation_dataset               # evaluation benchmark
impact_analysis:
  - condition: upstream user_profile_table changes
    action: feature_eng step must be re-run
    severity: high
  - condition: quality issue in this dataset
    action: interrupt training jobs depending on this version
    severity: high
```

The example above is highly condensed for brevity, but retains the essential skeleton of a lineage record. In production environments, each transformation step would also record more detailed processing logic, execution time, data volume ratios, and field changes. The **`impact_analysis` field** is particularly critical—it explicitly captures both "which downstream assets would be affected by an upstream change" and "which downstream systems would be impacted if this dataset has a problem," along with severity levels and mitigation actions. It is this field that transforms lineage from a static "data flow diagram" into a governance tool capable of driving incident investigation and change assessment.

As shown in Figure 27-2, a single piece of raw data may simultaneously enter multiple processing chains and flow through multiple layers of transformation before reaching downstream applications such as training, retrieval, and evaluation, forming a directed acyclic graph (DAG). Complete lineage records enable an organization to trace data destinations forward along this graph and locate problem sources in reverse.

The true power of lineage is most fully realized in impact analysis scenarios. Consider an upstream `user_profile_table` planning to modify the definition of a certain field. Without lineage, engineers can only guess from experience and memory "who will be affected," making it very easy to overlook dependencies. With complete lineage, the system can automatically enumerate along the DAG all downstream assets that indirectly depend on that field (such as the feature engineering step in this example and its outputs `preference_model_training` and `rag_knowledge_embeddings`), assess the scope of impact one by one, and notify the responsible parties. In reverse, when a model exhibits anomalies in production, the team can trace back along the lineage: did the raw data source have a problem, did some transformation step introduce a defect, or did the semantics of some upstream field drift without anyone's knowledge? Without lineage, incident investigation and change assessment degrade into searching for a needle in a haystack; with lineage, they become deterministic operations that can be executed by traversing a graph.

![Data Lineage Graph](../../images/part9/ch27_fig02_zh.png)

*Figure 27-2: Data Lineage Graph*



### 27.3.2 Permissions Governance and Access Control

Data permissions management must answer: who can access what data, in what context, for what reason, and how are access records audited. This involves not only technical implementation (e.g., dataset-level permissions, row-level permissions, column-level permissions) but also organizational processes (e.g., approval, periodic review, anomaly alerts).

**Multiple levels of the permissions model:**

**Dataset-level permissions** — Access rights for the entire data asset. For example, whether a given employee can access the "user interaction feedback dataset."

**Table/field-level permissions** — Fine-grained control for data assets containing multiple tables or multiple fields. For example, an employee may access the `user_id` field but not the `email` field.

**Row-level permissions** — Condition-based control at the row level. For example, employees in a given country can only access data from that country.

**Context-based permissions** — Based on the context of access (time, location, purpose). For example, access to the full dataset is permitted only in the context of "model training"; in all other contexts, only the de-identified version may be used.

The theoretical foundation for these permission levels is the widely adopted Role-Based Access Control (RBAC) model—which introduces "roles" as an intermediary layer between users and permissions, enabling permission management to be organized by responsibility rather than by individual, thereby substantially reducing the complexity of managing permissions in large organizations (Sandhu et al. 1996). In data asset governance, RBAC typically also needs to be combined with attribute-based control: beyond "which role does a user belong to," it is also necessary to consider the sensitivity level and compliance tags of the data itself, as well as the context in which access occurs (time, location, purpose), in order to support the field-level, row-level, and context-level controls described above. This is especially important for large language model applications—the same user data should be presented at completely different granularities depending on whether the purpose is "training a preference model" or "business analytics." This kind of "purpose-scoped access" requirement cannot be expressed by role alone and must be supplemented with attribute and context dimensions.

In large language model applications, a key permissions scenario is **differentiated data access**: the same dataset is accessed at different levels of granularity by different teams according to their respective purposes. For example:

```yaml
dataset: user_interaction_feedback
permissions:
  ml_training_team:
    access_level: full_raw_data
    reason: training requires complete data
    audit_sampling_rate: 1.0
  rag_engineering_team:
    access_level: deidentified_features
    pii_handling: user_id hashed
  business_analytics_team:
    access_level: aggregated_stats
    aggregation_rule: GROUP BY aggregations only
    min_group_size: 1000
  data_governance_team:
    access_level: full_admin
    includes_audit_logs: true
```

The example above lists only the access level for each team; a complete configuration would also include a justification for the grant, accessible fields, row-level conditions, approval requirements, and audit sampling rate for each entry. The design logic is: the training team needs full raw data for best results, but this comes with full-coverage access auditing; the RAG team uses only de-identified text and features; the business analytics team can only run aggregation queries to prevent reverse-engineering of individuals from aggregate results; the governance team has administrative access including audit logs. Through this purpose-scoped control at the field, row, and query levels, the organization minimizes exposure of sensitive information while meeting the data needs of all parties.

**Access approval and periodic review:**

Periodic review is an important component of permissions governance. Every data asset should have its permissions configuration reviewed periodically (e.g., quarterly) to ensure:

1. Personnel who have been granted permissions still require those permissions
2. No personnel who should have permissions have been missed
3. Non-compliant access patterns (e.g., bulk access at unusual times or from unusual locations) are identified and investigated
4. Permissions are aligned with the principle of least privilege

The other half of permissions governance is audit and anomaly detection. Configuring permissions alone is insufficient; it is also necessary to record "who accessed what data, when, and in what manner," and to remain alert to anomalous access patterns. For example, an analytics account that normally performs aggregation queries only during business hours, which suddenly downloads large volumes of row-level raw data late at night—even if technically "authorized" by permissions, such a pattern warrants an alert and human review. For sensitive data (especially assets containing PII), the audit sampling rate should be higher, ideally achieving complete traceability. These audit logs are themselves a data asset that must be properly protected, retained long-term, and made readily available for compliance reviews. By combining permissions configuration, access auditing, and anomaly alerts, permissions governance is elevated from a "static access control list" to a "dynamic security defense system."

### 27.3.3 Lifecycle Management and State Transitions

Data assets are not permanent. From creation, active use, and gradual deprecation to eventual deletion or archival, each stage requires a clear definition and management strategy.

The importance of lifecycle management stems from the fact that the "exit" of a data asset, like its "entry," produces engineering consequences. If outdated data is not retired in a timely manner, it will continue to consume storage, pollute search results, and potentially be misused by new projects. Conversely, if data is suddenly deleted or its schema is changed without the knowledge of downstream consumers, it will directly break pipelines that depend on it. Such problems are a significant source of technical debt in production-grade machine learning systems: unmanaged data dependencies accumulate in hidden ways, steadily increasing system maintenance costs and typically only coming to light when something breaks (Sculley et al. 2015). Modeling the lifecycle as a state machine with explicit transition conditions is essentially providing an explicit contract for data dependencies, ensuring that every state change undergoes evaluation, notification, and approval, thus transforming "implicit data debt" into a "visible, manageable, controlled process" (Polyzotis et al. 2018).

*Table 27-2: Lifecycle States of a Data Asset*

| State | Characteristics | Typical Duration | Transition Condition | User Action |
| --- | --- | --- | --- | --- |
| **CREATED** | Newly created; metadata complete but not yet live | 0–2 weeks | Passes quality check and security review | Awaiting activation |
| **ACTIVE** | In normal use; regularly updated; discoverable and usable | Depends on business needs | Whether deprecation is warranted | Free to use |
| **DEPRECATED** | Planned for retirement; new projects should not use it; existing usage gradually migrated | 3–6 months | All downstream consumers have migrated | Discouraged from use |
| **ARCHIVED** | No longer actively updated, but historical data retained for auditing and queries | Long-term | Whether complete deletion is required | Read-only access |
| **DELETED** | Data has been physically deleted (if not prohibited by law) | — | Permanent operation | No access |



As shown in Figure 27-3, the states above can be abstracted as a state machine: a data asset transitions sequentially through CREATED → ACTIVE → DEPRECATED → ARCHIVED → DELETED, with each transition triggered by explicit conditions, ensuring that every step of the lifecycle is auditable and traceable.

![Data Asset Lifecycle State Transition Diagram](../../images/part9/ch27_fig03_zh.png)

*Figure 27-3: Data Asset Lifecycle State Machine*



**Key activities during lifecycle transitions:**

**From ACTIVE to DEPRECATED** — Requires:

- A clear rationale (e.g., a newer version is superior, maintenance has been discontinued, the dataset has been superseded by a replacement)
- A clear migration path and migration guide
- Notification to all existing users, with migration support provided
- A migration deadline that is achievable but not distant

**From DEPRECATED to ARCHIVED** — Requires:

- Confirmation that all downstream consumers have completed migration
- Migration of active storage to cold storage to optimize costs
- Maintaining read-only access to support historical queries and auditing
- Periodic scanning to detect any unexpected new access

**From ARCHIVED to DELETED** — Requires:

- Confirmation that the legal retention period has been exceeded
- Obtaining necessary compliance and legal approvals
- Executing physical deletion and recording the deletion log
- Cleaning up all associated backups and snapshots

Against the backdrop of the intersection of large language models and privacy regulations, deletion is often not merely "cleaning up storage" but a controlled operation with strong compliance implications. Regulations such as the GDPR right to erasure and the deletion requirements of the Personal Information Protection Law (PIPL) may compel organizations to completely delete certain categories of personal data—including all copies, backups, and derived data—within specific timeframes. This imposes a sharp requirement on lifecycle management: deletion must be "demonstrably complete deletion." Without complete lineage records, an organization cannot even determine which downstream assets a dataset scheduled for deletion may have spawned, and thus cannot guarantee the thoroughness of the deletion. This again illustrates that lineage, permissions, and lifecycle are not isolated—only when they work together can an organization, when facing a regulatory inquiry, confidently demonstrate that "this data has been compliantly and traceably disposed of."

### 27.3.4 Section Summary

Lineage, permissions, and lifecycle are the three pillars of data asset governance. Lineage tracking enables an organization to understand how data flows and transforms, thereby supporting incident investigation, impact analysis, and compliance auditing. Permissions governance ensures that data is accessed by appropriate personnel in appropriate ways, mitigating compliance and security risks. Lifecycle management ensures that data assets do not accumulate indefinitely, while also preventing them from disappearing abruptly and breaking dependent systems. Together, these three dimensions constitute a mature, maintainable data asset ecosystem.

---

## 27.4 Asset Catalog Cases and Templates

### 27.4.1 A Real-World Enterprise Data Asset Catalog Example

To illustrate how data asset governance is applied in practice, this section uses the data asset catalog of an e-commerce recommendation system to present cases involving diverse data types and governance requirements. The recommendation system is chosen because it naturally touches virtually all governance dimensions discussed in this chapter: it simultaneously depends on real-time streaming data, offline feature tables, training datasets, evaluation benchmarks, RAG knowledge bases, and compliance audit logs. These assets belong to different teams, are updated at different frequencies, carry different sensitivity levels, and are interconnected by complex lineage dependencies. Table 27-3 excerpts representative assets from this catalog, spanning multiple types including streaming data, feature tables, vector stores, training sets, evaluation sets, and knowledge bases.

*Table 27-3: Data Asset Catalog Example for an E-Commerce Recommendation System*

| Asset ID | Type | Quality Score | Status | Primary Use |
| --- | --- | --- | --- | --- |
| `raw_user_click` | Stream | 0.91 | ACTIVE | Real-time recommendation, training data source |
| `user_features` | Table | 0.96 | ACTIVE | SFT features, personalized recommendation |
| `product_emb_v3` | Vector DB | 0.94 | ACTIVE | RAG retrieval, similarity recommendation |
| `pref_sft_v2` | Dataset | 0.98 | ACTIVE | Fine-tuning preference learning model |
| `rank_benchmark` | Dataset | 0.97 | ACTIVE | Model performance evaluation |
| `kb_chunks` | Table | 0.93 | ACTIVE | RAG system knowledge source |
| `feedback_labeled` | Table | 0.94 | ACTIVE | SFT supervision signal, alignment |
| `click_v1_legacy` | Table | 0.78 | DEPRECATED | Historical analysis, migration reference |
| `audit_log` | Table | 0.96 | ACTIVE | Compliance checking, access tracking |

In a real catalog, each asset would also include fields for owner, update frequency, and access volume, with the total number of assets often reaching tens or even hundreds. Even so, this excerpt already demonstrates the value of a data asset catalog: asset types are highly heterogeneous (streams, tables, vector stores, and datasets coexist); quality scores are immediately apparent (core training and evaluation data are noticeably higher than raw feedback and legacy logs); and status fields directly identify assets requiring governance (e.g., `click_v1_legacy` is marked as DEPRECATED, signaling that new projects should not depend on it). In other words, this catalog consolidates dispersed decision criteria into a single view, making "which data to use" a decision that can be resolved with a single lookup.

### 27.4.2 Detailed Metadata Example for a Single Asset

To illustrate how complete a production data asset's metadata can be, the following uses `user_preference_sft_v2` (User Preference Fine-Tuning Training Dataset v2) as a simplified example, retaining only representative fields in each module:

```yaml
# Identity and ownership
asset_id: user_preference_sft_v2
description: User preference ratings and interaction feedback on recommended products,
             used for training preference learning models
owner_id: ml_training_team@company.com   # additional steward / business_owner fields omitted

# Storage and structure (schema has 7 fields total; 2 shown here)
storage: s3://.../user_preference_sft/v2/ (parquet, 45GB, 128M rows)
schema:
  - name: user_id
    type: string
    pii_handling: hashed
    description: unique user identifier (hashed)
  - name: preference_score
    type: float
    range:
      min: 0
      max: 1
    description: preference score (higher values indicate stronger preference)

# Version, lineage, quality (all excerpted)
version: 2.1.0                            # see version_history
lineage:
  source: raw_user_click
  transformations:
    - bot filtering
    - feature engineering
  downstream_assets:
    - preference ranking model
    - cold-start model
quality_metrics:
  overall: 0.98
  completeness: 0.99
  validity: 0.97
known_issues:
  - description: 0.3% of scores are out of range
    status: fix in progress
  - description: preference scores for cold-start users are unstable

# Permissions, compliance, lifecycle
access_level: internal
pii_handling:
  user_id: hashed
compliance:
  - GDPR
  - CCPA
data_residency: US
status: ACTIVE
retention: 3y
expected_active_until: 2026-12-31
```

The structural definition documents each field's type and value range; version history records each change and its compatibility; quality metrics provide numerical values for each dimension alongside known issues; and the permissions, usage records, and lifecycle modules are each detailed down to the team, downstream application, and individual timestamps.

This example conveys a core principle: **the metadata of a mature data asset is itself a compact specification document**. It consolidates all the information needed to answer "what is this data, can it be used, how should it be used, and for how long" into a machine-readable, machine-verifiable structure, making the discovery, evaluation, and governance of data no longer dependent on informal oral transmission.

### 27.4.3 An Aggregate Governance View Across Multiple Assets

In a mature data organization, a data asset catalog may contain hundreds or even thousands of assets. Managers need an aggregate view of the health of the data asset portfolio, which is typically achieved through a dashboard. Table 27-4 lists key metrics commonly used in governance dashboards, organized along five dimensions—coverage, quality, compliance, lifecycle, and usage—with a target value and alert threshold specified for each metric.

*Table 27-4: Key Metrics for the Data Asset Governance Dashboard (Selected)*

| Dimension | Representative Metric | Target | Alert Threshold |
| --- | --- | --- | --- |
| **Coverage** | Percentage of assets with metadata / with lineage | >95% / >90% | <90% / <80% |
| **Quality** | Average quality score | >0.90 | <0.80 |
| **Compliance** | PII data access violation incidents | 0 | Any violation |
| **Lifecycle** | DEPRECATED assets not yet migrated / orphan data share | 0 / <5% | >5% / >10% |
| **Usage** | Active asset share / data reuse rate | >80% / >60% | <70% / <40% |

Each dimension in the table above can be further decomposed into additional metrics, such as the proportion of high-quality assets, permissions configuration completeness rate, and search hit rate. The significance of this metric framework is that it transforms the question "how well is data governance being done"—a question that previously could only be answered by intuition—into a set of quantifiable, threshold-alertable operational metrics. Once a metric crosses an alert threshold (e.g., the orphan data share exceeds 10%, or a PII access violation occurs), the system can proactively alert the governance team to intervene, turning data governance from a one-time project into sustainable day-to-day operations.

It is important to emphasize that these metrics are meaningful only when measured continuously and automatically. If quality scores, lineage coverage rates, and similar metrics depend on periodic manual aggregation, they will quickly become stale and misrepresentative, ultimately becoming "dashboard numbers that look good but no one believes." Mature governance platforms therefore embed metric collection into the data pipeline, using automated quality validation and metadata scanning to continuously refresh these metrics (Schelter et al. 2018), so that the dashboard genuinely becomes an actionable operational tool rather than a one-time reporting artifact. Seen from another angle, the governance dashboard and individual asset metadata are projections of the same mechanism at different scales: the former concerns itself with "the health distribution of the entire data asset portfolio," the latter concerns itself with "whether a specific dataset can be trusted," and both share the same continuously updated metadata foundation.

In cross-team governance practice, the health of the permissions dimension is particularly worth examining separately. As shown in Figure 27-4, a permissions matrix can visually present "which team holds what permissions on which assets," enabling at-a-glance identification of excessive permissions or permissions gaps.

![Permissions Management Matrix Based on Roles and Purposes](../../images/part9/ch27_fig04_zh.png)

*Figure 27-4: Permissions Management Matrix Example*



### 27.4.4 Section Summary

Through concrete cases, we have seen how data asset catalogs are applied in real organizations. The key is to build a governance framework that is both comprehensive (covering all data assets) and detailed (with complete metadata for each asset), while establishing a monitoring dashboard to maintain a real-time view of governance health.

---

## 27.5 Governance Maturity, Organizational Roles, and Implementation Checklist

The preceding sections have discussed data asset definitions, metadata models, lineage, permissions, lifecycle management, and catalog cases. In a real organization, however, these capabilities are rarely built all at once. Data governance is a path that must be pursued in stages, carried by clearly defined roles, and constrained by verifiable acceptance criteria. This section examines how to translate the foregoing methods into actual practice, from three perspectives: maturity progression, organizational roles, and an implementation checklist.

### 27.5.1 The Maturity Progression of Data Governance

A common misconception is to treat data governance as a "one-time project of building a catalog system." In reality, building governance capability more closely resembles a process of continuous evolution, which can be roughly divided into four stages.

**Stage 1: Ad Hoc.** No unified catalog exists; data is scattered across each team's databases, documents, and personal storage. Data discovery depends entirely on informal inquiries; metadata exists only in the memories of select individuals. The typical symptoms at this stage are precisely the two failure cases described in Section 27.1: data is "visible but unreachable," reuse costs are extremely high, and compliance risks are hidden.

**Stage 2: Managed.** The organization begins building a centralized data catalog, requiring core data assets to register basic metadata (owner, schema, description). Data can be searched, but metadata is largely maintained manually, coverage is limited, and lineage and quality information is often absent or lagging.

**Stage 3: Governed.** Metadata collection begins to be automated; lineage, quality metrics, and permissions configuration become standard. Registration processes, lifecycle state machines, and periodic permissions reviews are institutionalized. A governance dashboard is launched, enabling key metrics to be continuously monitored (Schelter et al. 2018).

**Stage 4: Optimized.** Governance is deeply integrated with data production and consumption processes: quality validation is embedded in pipelines, lineage is automatically extracted from job graphs, and permissions and compliance checks are automatically enforced before data enters the index. Data asset reuse rate and discovery rate become operational metrics that can be optimized; governance itself enters a virtuous cycle of "using data to govern data" (DAMA International 2017; Polyzotis et al. 2018).

The value of understanding this maturity progression is that it reminds teams not to attempt everything at once, but to prioritize high-value, high-risk core assets by business priority, then gradually expand breadth and depth. For most organizations, the leap from Stage 2 to Stage 3—upgrading from a "statically maintained catalog" to an "automatically collected living catalog"—is typically the step with the greatest payoff and the greatest challenge.

### 27.5.2 Organizational Structure and Roles in Governance

Data governance is not only a technical system but also a set of accountability assignments centered on data. Without clear role definitions, even the most sophisticated catalog system will gradually fall into disuse because "no one is responsible for the data." In mature data organizations, the following key roles typically exist (DAMA International 2017).

**Data Owner** — Typically a business or team lead who bears ultimate responsibility for the quality, compliance, and value of a given data asset, and who decides the data's scope of use and permissions policy.

**Data Steward** — Responsible for the day-to-day maintenance of a data asset, including metadata updates, quality monitoring, and issue response. The data steward is the hub connecting business and technology, ensuring that metadata is accurate, lineage is complete, and quality standards are met.

**Data Governance Team** — Formulates organization-wide governance standards (metadata standards, naming conventions, classification and grading, compliance requirements), operates the data catalog platform, and presides over permissions reviews and compliance audits.

**Data Consumer** — Includes data scientists, algorithm engineers, analysts, and others who discover and use data through the catalog. Consumer feedback (e.g., flagging data issues, submitting reuse requests) is important input for continuous governance improvement.

The collaboration among these roles is essentially the explicit assignment of "knowledge about data and accountability for data" to individuals. The failure cases in Section 27.1 occurred in part because of the absence of explicit owners and stewards—when no one is responsible for "can this dataset be used for training" or "what are its restrictions," this critical information naturally goes unrecorded and unmaintained.

### 27.5.3 Implementation Checklist for Data Asset Governance

To translate the foregoing methods into actionable acceptance criteria, a governance checklist can be established. Its purpose is not to replace specific implementations, but to help teams systematically verify that key steps are in place whenever a data asset goes live or a governance capability is built.

*Table 27-5: Data Asset Governance Implementation Checklist*

| Check Category | Key Question | Acceptance Criterion |
| --- | --- | --- |
| Identity and Ownership | Is there a unique identifier, a clear owner, and a clear steward? | Every asset can be traced to a responsible party |
| Metadata Completeness | Is structure, description, use case, and restrictions registered? | No missing core fields; restrictions explicitly declared |
| Lineage Tracking | Are source, transformations, and downstream consumers recorded? | Data destinations can be traced forward; problem sources can be located in reverse |
| Quality Monitoring | Are quality metrics measured automatically and continuously? | Metrics are refreshed by pipelines, not entered manually |
| Permissions and Compliance | Are permissions configured by role and purpose; is PII labeled? | Sensitive data has no unauthorized access; compliance tags are complete |
| Lifecycle | Are states and transition conditions defined? | Deprecation, archival, and deletion have plans and approvals |
| Discoverability | Can the asset be found through search, filtering, and recommendation? | Target users can find the data within a reasonable timeframe |
| Feedback Loop | Can consumer feedback be received and acted upon? | Data issues can be reported and resolved end-to-end |

The value of this checklist is directly aligned with the engineering checklist for RAG data pipelines (see Chapter 21): it consolidates implicit requirements scattered across different roles' minds onto a single acceptance sheet, enabling problems to surface earlier and accountability to be assigned more clearly. For high-risk scenarios (e.g., data involving personal privacy, financial information, or medical records), more stringent compliance auditing, sensitive information detection, and access traceability requirements should be layered on top.

### 27.5.4 Special Governance Challenges in Large Language Model Scenarios

This part has repeatedly emphasized that the large language model era imposes demands on data governance that go beyond those of traditional data warehouses. Three categories of challenges warrant particularly close attention.

**Provenance and authorization tracking for training data.** Once a dataset is used for model training, its influence becomes "embedded" in the model parameters and is difficult to remove after the fact. Therefore, the lineage, authorization status, and compliance annotations of training data must be accurately in place before data enters the training pipeline. A dataset without a clear training authorization record (such as the feedback data on a former employee's personal cloud storage described in Section 27.1), once used for training, may carry irreversible compliance risks. Attaching datasheet-style documentation to each training dataset has become an important component of responsible AI practice (Gebru et al. 2021).

**Timeliness and version governance for RAG knowledge sources.** In RAG systems, knowledge bases are continuously updated; outdated documents must be retired or archived in a timely manner, otherwise the model will generate incorrect answers based on stale knowledge. This requires knowledge assets to have clear version and lifecycle management—a direct echo of the knowledge update and version governance discussed in Chapter 23. The lifecycle state machine in the data asset catalog is the underlying mechanism supporting this kind of "controlled knowledge evolution."

**Isolation and leakage prevention for evaluation data.** If evaluation data leaks into training data, evaluation results are invalidated. Evaluation data assets therefore require strict permissions isolation and lineage tracking to ensure they are not inadvertently mixed into the training pipeline at any stage. Such risks are often subtle, and can only be detected during post-hoc audits through complete lineage records.

The common thread across these three categories of challenges is that none of them is a question of "whether the data itself is good," but rather a question of "whether knowledge about the data is accurately recorded and enforced." When data lacks governance, these risks are amplified layer by layer downstream in a "data cascade" pattern, ultimately manifesting as model-level failures that are difficult to diagnose (Sambasivan et al. 2021). This once again affirms the central position of this part: in the large language model era, data governance is not a supportive back-office function but foundational infrastructure that determines the reliability and compliance of AI systems.

### 27.5.5 Section Summary

This section has examined how data asset governance can be implemented in real organizations, from three perspectives: maturity progression, organizational roles, and an implementation checklist. Building governance capability is a staged, evolutionary process requiring a clear division of responsibilities among data owners, data stewards, the governance team, and data consumers, with verifiable checklists constraining each step. An organization that can clearly answer "what data do we have, where did it come from, what is its quality, who can use it, and for how long" can continuously transform data—a strategic asset—into product capability, while avoiding repeated costs from redundant discovery, compliance exposure, and hidden technical debt (Sambasivan et al. 2021; Polyzotis et al. 2018). In large language model scenarios, training data authorization tracking, RAG knowledge timeliness governance, and evaluation data isolation are governance challenges that warrant particular vigilance. Ultimately, the goal of governance is not to build a system, but to ensure that "knowledge about data and accountability for data" flow continuously and reliably throughout the organization.

---

## Chapter Summary

The essence of data asset catalogs and metadata governance is to consolidate "knowledge about data"—originally dispersed across individual memories and ad hoc documents—into organization-level, queryable, auditable shared infrastructure. This chapter first used the six dimensions of data assets to explain that what fundamentally distinguishes a data asset from a file inventory is the context, accountability, and constraints it carries. It then provided engineering methods for operationalizing these dimensions through the layered metadata model of the dataset registry, automated collection mechanisms, and search-discovery capabilities.

Lineage, permissions, and lifecycle constitute three mutually supporting pillars of data asset governance: lineage supports incident investigation, impact analysis, and compliance tracing, and must be granular to the field level; permissions achieve purpose-scoped differentiated access through a combination of roles and attributes, supplemented by auditing and anomaly detection; and lifecycle incorporates data onboarding, deprecation, and demonstrably complete deletion into a controlled process through a state machine with explicit transition conditions. Building governance capability is a staged, evolutionary process requiring a clear division of responsibilities among data owners, data stewards, the governance team, and data consumers, with verifiable checklists constraining each step. In large language model scenarios, training data authorization tracking, RAG knowledge source timeliness governance, and evaluation data isolation are governance challenges that warrant particular vigilance.

## References

Abedjan Z, Golab L, Naumann F (2015) Profiling relational data: a survey. The VLDB Journal 24(4):557–581.

Breck E, Polyzotis N, Roy S, Whang S E, Zinkevich M (2019) Data Validation for Machine Learning. In: Proceedings of the 2nd SysML Conference (MLSys).

Buneman P, Khanna S, Tan W-C (2001) Why and Where: A Characterization of Data Provenance. In: Proceedings of the 8th International Conference on Database Theory (ICDT), pp 316–330.

Cai L, Zhu Y (2015) The Challenges of Data Quality and Data Quality Metrics. Journal of Data and Information Quality 6(2-3):1–10.

DAMA International (2017) DAMA-DMBOK: Data Management Body of Knowledge, 2nd Edition. Technics Publications, Basking Ridge.

Fernandez R C, Abedjan Z, Koko F, Yuan G, Madden S, Stonebraker M (2018) Aurum: A Data Discovery System. In: 2018 IEEE 34th International Conference on Data Engineering (ICDE), pp 1001–1012.

Gebru T, Morgenstern J, Vecchione B, Vaughan J W, Wallach H, Daumé III H, Crawford K (2021) Datasheets for Datasets. Communications of the ACM 64(12):86–92.

Halevy A, Korn F, Noy N F, Olston C, Polyzotis N, Roy S, Whang S E (2016) Goods: Organizing Google's Datasets. In: Proceedings of the 2016 ACM SIGMOD International Conference on Management of Data, pp 795–806.

Hellerstein J M, Sreekanti V, Gonzalez J E, Dalton J, Dey A, Nag S, Ramachandran K, Arora S, Bhattacharyya A, Das S, Donsky M, Fierro G, She C, Steinbach C, Subramanian V, Sun E (2017) Ground: A Data Context Service. In: 8th Biennial Conference on Innovative Data Systems Research (CIDR).

Herschel M, Diestelkämper R, Ben Lahmar H (2017) A survey on provenance: What for? What form? What from? The VLDB Journal 26(6):881–906.

Mitchell M, Wu S, Zaldivar A, Barnes P, Vasserman L, Hutchinson B, Spitzer E, Raji I D, Gebru T (2019) Model Cards for Model Reporting. In: Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT*), pp 220–229.

Noy N F, Musen M A (2000) PROMPT: Algorithm and Tool for Automated Ontology Merging and Alignment. In: Proceedings of the 17th National Conference on Artificial Intelligence (AAAI), pp 450–455.

Polyzotis N, Roy S, Whang S E, Zinkevich M (2018) Data Lifecycle Challenges in Production Machine Learning: A Survey. ACM SIGMOD Record 47(2):17–28.

Rahm E, Bernstein P A (2001) A survey of approaches to automatic schema matching. The VLDB Journal 10(4):334–350.

Sambasivan N, Kapania S, Highfill H, Akrong D, Paritosh P, Aroyo L M (2021) "Everyone wants to do the model work, not the data work": Data Cascades in High-Stakes AI. In: Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, pp 1–15.

Sandhu R S, Coyne E J, Feinstein H L, Youman C E (1996) Role-Based Access Control Models. IEEE Computer 29(2):38–47.

Schelter S, Lange D, Schmidt P, Celikel M, Biessmann F, Grafberger A (2018) Automating Large-Scale Data Quality Verification. Proceedings of the VLDB Endowment 11(12):1781–1794.

Sculley D, Holt G, Golovin D, Davydov E, Phillips T, Ebner D, Chaudhary V, Young M, Crespo J-F, Dennison D (2015) Hidden Technical Debt in Machine Learning Systems. In: Advances in Neural Information Processing Systems 28, pp 2503–2511.

Stonebraker M, Bruckner D, Ilyas I F, Beskales G, Cherniack M, Zdonik S, Pagan A, Xu S (2013) Data Curation at Scale: The Data Tamer System. In: 6th Biennial Conference on Innovative Data Systems Research (CIDR).

Wang R Y, Strong D M (1996) Beyond Accuracy: What Data Quality Means to Data Consumers. Journal of Management Information Systems 12(4):5–33.
