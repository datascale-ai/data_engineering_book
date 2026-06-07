# Chapter 28: Data Productization and Data Contracts

## Chapter Guide

Chapter 27 explained how data catalogs and metadata governance turn organizational data from "visible but unreachable" into discoverable, understandable, and trustworthy assets. But discoverable and understandable data is not automatically dependable. A cataloged dataset may still have field meanings changed silently upstream, or may stop updating after one ingestion failure. When downstream training pipelines, RAG knowledge bases, or evaluation systems depend on it, small upstream changes can be amplified into model degradation, online incidents, or compliance risks (Sambasivan et al. 2021).

This chapter addresses that problem: how to upgrade data from static datasets into **data products** that consumers can rely on over time, and how to use explicit **data contracts** to define the rights and obligations of producers and consumers. Data productization is different from traditional one-time delivery. It treats data like a service-facing software product: structure, quality, freshness, privacy boundaries, and change behavior must be maintained continuously.

This shift is especially important in large-model data engineering. Training solidifies data characteristics into parameters. RAG systems continuously consume changing knowledge sources. Evaluation systems depend on stable data distributions so that results remain comparable. In these settings, unnoticed upstream changes can mean failed training, wrong online answers, or distorted evaluation conclusions.

The chapter proceeds in four parts. First, it explains the shift from datasets to data products. Second, it breaks down five core data-contract clauses: schema, quality, freshness, privacy, and compatibility. Third, it discusses change compatibility and consumer governance. Finally, it reviews a realistic incident in which an upstream field change caused both training and retrieval quality to decline, showing how contracts and rollback should have contained the damage.

## 28.1 From Datasets to Data Products

### 28.1.1 Limits of Dataset Thinking

In many organizations, early data delivery follows a simple dataset mindset: when a team needs data, the producer exports a file or opens a table, and delivery is considered complete. This can work for one-off analysis and prototypes. It breaks down when multiple teams and applications depend on the same data for a long time.

The first problem is that dataset thinking treats data as a static deliverable rather than a continuing service. A CSV export is disconnected from upstream as soon as it is delivered; downstream consumers cannot see schema evolution, definition changes, or quality fluctuations. The second problem is unclear responsibility. When data fails, consumers do not know whom to contact, and producers often do not feel responsible for downstream use. The third problem is uncontrolled change. Upstream producers may change fields, adjust meanings, or stop updates at any time. That freedom is convenient for producers and disastrous for consumers.

The root issue is that dataset thinking asks whether data has been delivered, not whether it can be used reliably over time. In production ML systems, unmanaged data dependencies accumulate as technical debt and raise maintenance cost until failures appear (Sculley et al. 2015). Solving this requires upgrading delivery from a file to a product.

### 28.1.2 Data Products Deliver Stable Capability, Not One-Time Files

The concept of a **data product** has become prominent with the rise of data mesh architectures (Dehghani 2022; Machado, Costa and Santos 2022). Its core idea is that data should be managed like a user-facing software product: it has clear consumers, promised service quality, a responsible owner, and predictable evolution.

Treating data as a product means delivering a stable capability that consumers can rely on: the structure will not change without reason, quality will not degrade silently, and updates will not stop unexpectedly. This predictability requires producers to take continuing responsibility, define service-level commitments, and follow communication and canary processes when change occurs.

The core of a data product is therefore not only the amount or richness of data. It is the system of **predictability** around the data. This matches a broader lesson from ML engineering: reliability in real environments often depends less on the model alone and more on the discipline around data (Paleyes, Urma and Lawrence 2022; Lwakatare et al. 2020).

### 28.1.3 Data Product Canvas: Inputs, Outputs, SLA, Owner, and Change Policy

To build a data product, the team must answer structured questions: what upstream inputs does it consume, what outputs does it provide, what service quality does it promise, who owns it, and how will it evolve? These questions can be summarized in a **Data Product Canvas**.

![Data product canvas](../../images/part9/ch28_fig01_zh.png)

*Figure 28-1: Data product canvas*

Several elements are central:

- **Inputs** describe upstream data sources and assets. They make upstream lineage explicit so that upstream changes can be assessed.
- **Outputs** define the external data interface: schema, granularity, field semantics, and access methods. Once published, the output interface becomes a consumer-facing commitment.
- **SLA** distinguishes a data product from an ordinary dataset. It can include update frequency and delay limits, availability, quality floors, and incident response time, following the spirit of service-level engineering (Beyer et al. 2016).
- **Owner** defines who is ultimately accountable for quality, stability, and evolution. A data product without an owner will decay.
- **Change Policy** defines what changes are allowed, how much notice is required, whether canarying is required, and how consumers are informed.

| Dimension | Dataset | Data Product |
| --- | --- | --- |
| Delivery form | One-time file or snapshot | Stable service that can be relied on |
| Responsible party | Often unclear | Explicit owner and steward |
| Service commitment | None | SLA for freshness, quality, and availability |
| Change management | Can change freely | Controlled by policy and contracts |
| Consumer relationship | Loose and invisible | Registered, traceable, notifiable |
| Evolution | Unpredictable | Versioned, canaried, backward-compatible |

### 28.1.4 Section Summary

Moving from datasets to data products is a shift from static files to stable capability. By defining inputs, outputs, SLA, owner, and change policy, a data product anchors predictability in process rather than memory. But a canvas only describes what a product should promise. To make promises executable, checkable, and accountable, a more formal machine-readable agreement is needed: the data contract.

## 28.2 Fields and Boundaries of a Data Contract

### 28.2.1 What Is a Data Contract?

If a data product canvas answers what a product promises, a **data contract** answers how those promises are precisely defined, machine-validated, and jointly followed. A data contract is a formal, versioned, executable agreement between data producers and consumers. It turns expectations that used to live in conversation and personal memory into structured, verifiable rules.

A data contract is not just a schema. A schema describes structure: fields and types. A data contract describes a broader set of commitments: structure, quality, freshness, privacy, and compatibility. It can be understood as an executable counterpart to dataset documentation practices such as Datasheets for Datasets (Gebru et al. 2021): documentation records composition, uses, and limits; contracts turn those commitments into pipeline-checkable artifacts.

Contracts are also bidirectional. They constrain what producers must provide and clarify what consumers can expect. They also state what consumers must not depend on. This makes responsibility clear when data fails: did the producer violate the contract, or did the consumer rely on behavior that was never promised?

### 28.2.2 Five Core Contract Clauses

| Contract Type | Question Answered | Example Clauses |
| --- | --- | --- |
| Schema | What does the data look like? | Field names, types, nullability, enum values, primary keys |
| Quality | How trustworthy is it? | Completeness >= 0.98, validity >= 0.97, deduplication rate, anomaly limits |
| Freshness | How current is it? | Updated before 08:00 daily, latency <= 2 hours, alert on missed update |
| Privacy | Who can use it, and how? | PII fields, anonymization requirements, access scope, compliance tags |
| Compatibility | How can it evolve without breaking downstream users? | Backward compatibility, notice period, deprecation window |

**Schema contracts** define field names, types, nullability, enum values, and keys. They must also define semantics. A field with the same type but a changed meaning is one of the most dangerous failure modes (Rahm and Bernstein 2001).

**Quality contracts** define quality floors through measurable indicators: completeness, validity, duplication, outlier rates, and distribution stability (Wang and Strong 1996; Redman 1998). They should be tied to consumer needs; one dataset may need different quality thresholds for aggregate analysis and model training (Strong, Lee and Wang 1997).

**Freshness contracts** define update frequency, latency limits, and missed-update alerts. In RAG systems, freshness directly affects answer correctness.

**Privacy contracts** define privacy and compliance boundaries: PII fields, anonymization, roles and purposes allowed to access the data, and relevant regulations.

**Compatibility contracts** define how the product evolves safely: which changes are backward-compatible, how much notice is required, and how long older versions remain available. Compatibility ideas mirror schema evolution in data-intensive systems (Kleppmann 2017).

### 28.2.3 Data Contract Template

![Data contract template with five clause groups](../../images/part9/ch28_fig02_zh.png)

*Figure 28-2: Data contract template*

```yaml
contract: user_interaction_feedback
version: 2.0.0
owner: data-platform-team@company.com
consumers: [preference_training, rag_feedback, analytics]

schema:
  - user_id:          string, required, hashed
  - interaction_type: enum{click,like,collect,share}, required
  - event_time:       timestamp(UTC, ms), required
quality:
  completeness: ">=0.98"
  validity:     ">=0.97"
  interaction_type_distribution: "click share 0.4-0.6; alert on abrupt drift"
freshness:
  update: "before 08:00 UTC daily"
  max_delay: "2h"
  on_miss: alert
privacy:
  pii_fields: [user_id]
  handling: hashed
  access: internal
compatibility:
  policy: backward_compatible
  notice_period: "14 days"
  deprecation: "old version retained for one version cycle"
```

The contract is **machine-readable, checkable, and versioned**. It can live in version control, evolve with the data product, be checked by CI before release, and trigger alerts or blocks when violated. It is no longer a passive document; it is an engineering artifact embedded in the production process.

### 28.2.4 Execution and Responsibility Boundaries

Contracts matter only if executed. Execution happens at multiple points:

- **Before release**, CI checks schema and quality clauses and blocks nonconforming data.
- **At runtime**, monitoring measures quality and freshness and alerts on violations.
- **During change**, compatibility checks classify the change and trigger notification or canary processes.

Contracts also define responsibility. Within the contract, producer failures are producer responsibility. If consumers depend on behavior outside the contract, such as assuming a nullable field is always non-null or an undocumented field will never disappear, they own that risk. Many production data incidents come from hidden consumer dependencies outside explicit contracts (Polyzotis et al. 2018; Shankar et al. 2022).

### 28.2.5 Section Summary

Data contracts turn product promises into precise, machine-readable, checkable agreements. Schema, quality, freshness, privacy, and compatibility cover the structural, quality, time, privacy, and evolution aspects of reliability. Static contracts alone are not enough, however, because data products inevitably change.

## 28.3 Change Compatibility and Consumer Governance

### 28.3.1 Change Is the Main Risk of Data Products

Data products evolve as business systems, upstream sources, structures, definitions, and distributions change. Change itself is healthy. The risk is **ungoverned change**: a producer changes data that consumers depend on without their knowledge.

The danger comes from a fundamental property of data products: downstream consumers are often invisible, numerous, and independent. A local field adjustment can affect a training pipeline, a RAG index, and several reports. Unlike software API breaks, data changes often do not fail loudly. The pipeline continues running, but the result becomes wrong. Silent failure is the core reason data incidents are hard to detect early (Sambasivan et al. 2021).

Change governance turns change from a silent-failure risk into a controlled, predictable, canaryable operation. It requires classifying changes, notifying and canarying risky changes, and making consumers visible for impact analysis.

### 28.3.2 Change Classification and Compatibility

| Change Type | Example | Compatibility | Handling |
| --- | --- | --- | --- |
| Backward-compatible | Add nullable fields, broaden value range, improve documentation | Safe | Notify; no canary required |
| Potentially breaking | Add enum values, adjust field semantics, distribution drift | Risky | Advance notice plus canary validation |
| Breaking | Delete/rename fields, change type or unit, tighten constraints | Dangerous | Version upgrade with old/new coexistence |

**Backward-compatible changes** do not break existing consumers. Old consumers can still read new data correctly (Kleppmann 2017).

**Breaking changes** inevitably break existing consumers, such as deleting a field, changing a type, or changing units. They must use version upgrades and coexistence windows.

The most dangerous category is **potentially breaking changes**. They may look compatible at the schema level but change semantics or distributions. Adding an enum value, changing a computation definition, or changing upstream sampling can break filters, features, and retrieval weights without type errors.

![Change compatibility decision tree](../../images/part9/ch28_fig03_zh.png)

*Figure 28-3: Change compatibility decision tree*

### 28.3.3 Advance Notice and Canary Validation

Risky changes require notice and canary validation before they affect all consumers.

1. **Field changes.** The platform should classify compatibility from the contract and notify registered consumers within the agreed notice period, such as 14 days. Breaking changes should release a new version beside the old version and migrate consumers gradually.
2. **Sample-distribution changes.** Drift is hidden because it changes statistics rather than structure. Quality-contract distribution monitoring should alert when key fields move beyond thresholds (Breck et al. 2019; Schelter et al. 2018).
3. **Index rebuilds.** In RAG, knowledge updates may rebuild vector indexes. A rebuild may change chunking, embedding models, and retrieval behavior. It should first run in a shadow or canary environment and compare recall, citation accuracy, and retrieval quality.
4. **Evaluation-set refreshes.** Evaluation sets are data products too. Refreshing them changes the baseline. New and old evaluation sets should coexist for regression comparison, and contamination into training data must be prevented.

The shared principle is to validate impact in a controlled scope before full rollout.

### 28.3.4 Consumer Governance and Impact Analysis

Advance notice and impact analysis require knowing who consumes the data product. Mature platforms require consumers to register before use, declaring scenario, depended fields, and SLA requirements. The consumer registry plus lineage graph forms the basis of **consumer impact analysis**.

When a change is proposed, impact analysis asks: which consumers are affected, how severe is the impact, and who must confirm the migration? The system can list consumers automatically and classify severity based on fields, scenarios, and SLA requirements. This turns guesswork into evidence-based evaluation.

### 28.3.5 Section Summary

Change is unavoidable. Ungoverned change is the main risk. Classifying changes as backward-compatible, potentially breaking, or breaking lets teams handle risk consistently. Notice and canary validation expose risk before it reaches downstream systems. Consumer registration and lineage make precise impact analysis possible.

## 28.4 Data Contract Failure Review

### 28.4.1 Incident Background

The best way to understand data contracts is to review what happens without them. This composite incident involves a data product named `user_interaction_feedback`, consumed by two downstream systems: a preference-model training pipeline (`preference_training`) and a relevance-feedback module for a RAG system (`rag_feedback`).

The key field is `interaction_type`, originally an enum with values `{click, like, collect, share}`. `click` is the largest and most important positive signal.

### 28.4.2 Timeline: How an Upstream Field Change Propagated

**Stage 1: an upstream "harmless" change.** The upstream logging team splits `click` into `click_card` and `click_detail` for finer behavior analysis. They see it as an enhancement: no field is deleted and the schema field list is unchanged. The change goes live without notification.

**Stage 2: silent downstream failure.** Both downstream systems filter on `interaction_type == "click"` to identify positive signals. After the change, the share of `click` drops near zero because it has been split into two new values. The pipelines do not error. Fields and types still exist. Training and RAG feedback jobs run normally, but positive samples have nearly disappeared.

**Stage 3: degradation is noticed.** A new preference model goes online and recommendation quality falls. At the same time, RAG answer relevance gets worse. Both teams first debug their own models and algorithms, trying hyperparameters, model rollback, and retrieval strategy changes. Days are spent without improvement because the shared data dependency is the cause (Sculley et al. 2015; Shankar et al. 2022).

**Stage 4: root cause located.** An engineer finally compares historical distributions and finds the cliff-like drop in `click`, tracing it to the upstream enum split. Nearly a week has passed.

### 28.4.3 Consumer Impact Analysis

| Consumer | Depended Field | Use | Impact | Consequence | Response |
| --- | --- | --- | --- | --- | --- |
| preference_training | interaction_type | Filter `== click` as positive samples | Severe | Positive samples nearly disappear; model quality drops | Block change; migrate first |
| rag_feedback | interaction_type | Weight chunks by click | Severe | Feedback weight fails; relevance drops | Block change; migrate first |
| analytics_daily | interaction_type | GROUP BY statistics by type | Medium | Reporting definition breaks | Notify and update definition |

This table turns vague concern into a row-level analysis: who is affected, how, how badly, and what to do.

![Consumer impact analysis](../../images/part9/ch28_fig04_zh.png)

*Figure 28-4: Consumer impact analysis*

### 28.4.4 How Contracts and Rollback Contain the Damage

With data contracts, the same change would follow a different path.

**First defense: schema contract interception.** `interaction_type` is declared as the closed enum `{click, like, collect, share}`. When upstream introduces `click_card` and `click_detail`, pre-release CI detects values outside the contract and blocks release. The change is classified as potentially breaking and must go through notice and canary validation.

**Second defense: quality-contract distribution monitoring.** Even if schema validation were bypassed, a distribution constraint such as "`click` share should be 0.4-0.6" would alert as soon as `click` drops sharply (Schelter et al. 2018). The issue would be found before training consumes the data.

**Third defense: versioning and rollback.** Consumers can pin to a validated data version instead of blindly following the latest data. If a new version fails, they roll back to the previous stable version. A compatibility shim can temporarily map `click_card` and `click_detail` back to `click` until downstream migration is complete.

Together, these defenses turn a multi-day incident into a pre-release block, canary, or quick rollback.

### 28.4.5 Lessons

First, **unchanged schema does not mean unchanged semantics**. The most dangerous changes often look compatible at the schema layer but drift semantically.

Second, **silent failure is more dangerous than explicit errors**. Contracts make hidden expectations visible and checkable.

Third, **versioning is the prerequisite for rollback**. Without versioning, there is no previous stable state to return to.

Fourth, **consumer visibility determines impact-analysis capability**. Registration and lineage make invisible dependencies explicit.

### 28.4.6 Section Summary

The `interaction_type` incident shows how an unmanaged enum split can silently damage both training and retrieval. Under contract protection, schema validation, distribution monitoring, and versioned rollback would intercept, alert, and contain the change. The value of data contracts is not to prevent change, but to make change transparent, controlled, and reversible.

## Chapter Summary

This chapter explained how to upgrade data from static datasets into data products and how data contracts define the rights and obligations of producers and consumers.

The shift from dataset to data product is a shift from delivering a file to providing stable capability. Data products define inputs, outputs, SLA, owner, and change policy so that predictability is anchored in process rather than personal memory.

Data contracts make those commitments precise, machine-readable, and checkable. Schema, quality, freshness, privacy, and compatibility clauses cover the five major dimensions of data reliability. The contract's bidirectional nature creates clear responsibility boundaries.

Change governance addresses the central tension of data-product evolution: how to keep evolving without harming downstream consumers. By classifying changes, notifying and canarying risky changes, and maintaining consumer visibility for impact analysis, teams can turn change from a silent-failure risk into a controlled operation.

The incident review shows that these mechanisms are not theoretical. A seemingly harmless enum split can cause days of training and retrieval degradation without contracts. With contracts, the same change is blocked, alerted, canaried, or rolled back.

Ultimately, data productization and data contracts pursue **predictability**. In large-model data engineering, where training, retrieval, and evaluation depend heavily on data stability, predictability is not a nice-to-have. It is a foundation of system reliability. If data catalogs solve whether data can be found and understood, data products and data contracts solve whether data can be depended on over time.

## References

Beyer B, Jones C, Petoff J, Murphy N R (2016) Site Reliability Engineering: How Google Runs Production Systems. O'Reilly Media, Sebastopol.

Breck E, Polyzotis N, Roy S, Whang S E, Zinkevich M (2019) Data Validation for Machine Learning. In: Proceedings of the 2nd SysML Conference (MLSys).

Dehghani Z (2022) Data Mesh: Delivering Data-Driven Value at Scale. O'Reilly Media, Sebastopol.

Gebru T, Morgenstern J, Vecchione B, Vaughan J W, Wallach H, Daumé III H, Crawford K (2021) Datasheets for Datasets. Communications of the ACM 64(12):86-92.

Kleppmann M (2017) Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems. O'Reilly Media, Sebastopol.

Lwakatare L E, Raj A, Crnkovic I, Bosch J, Olsson H H (2020) Large-scale machine learning systems in real-world industrial settings: A review of challenges and solutions. Information and Software Technology 127:106368.

Machado I A, Costa C, Santos M Y (2022) Data Mesh: Concepts and Principles of a Paradigm Shift in Data Architectures. Procedia Computer Science 196:263-271.

Paleyes A, Urma R-G, Lawrence N D (2022) Challenges in Deploying Machine Learning: A Survey of Case Studies. ACM Computing Surveys 55(6):1-29.

Polyzotis N, Roy S, Whang S E, Zinkevich M (2018) Data Lifecycle Challenges in Production Machine Learning: A Survey. ACM SIGMOD Record 47(2):17-28.

Rahm E, Bernstein P A (2001) A survey of approaches to automatic schema matching. The VLDB Journal 10(4):334-350.

Redman T C (1998) The Impact of Poor Data Quality on the Typical Enterprise. Communications of the ACM 41(2):79-82.

Sambasivan N, Kapania S, Highfill H, Akrong D, Paritosh P, Aroyo L M (2021) "Everyone wants to do the model work, not the data work": Data Cascades in High-Stakes AI. In: Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, pp 1-15.

Schelter S, Lange D, Schmidt P, Celikel M, Biessmann F, Grafberger A (2018) Automating Large-Scale Data Quality Verification. Proceedings of the VLDB Endowment 11(12):1781-1794.

Sculley D, Holt G, Golovin D, Davydov E, Phillips T, Ebner D, Chaudhary V, Young M, Crespo J-F, Dennison D (2015) Hidden Technical Debt in Machine Learning Systems. In: Advances in Neural Information Processing Systems 28, pp 2503-2511.

Shankar S, Garcia R, Hellerstein J M, Parameswaran A G (2022) Operationalizing Machine Learning: An Interview Study. arXiv preprint arXiv:2209.09125.

Strong D M, Lee Y W, Wang R Y (1997) Data Quality in Context. Communications of the ACM 40(5):103-110.

Wang R Y, Strong D M (1996) Beyond Accuracy: What Data Quality Means to Data Consumers. Journal of Management Information Systems 12(4):5-33.
