# Chapter 24: The DataOps Flywheel and Team Organization

---

## Chapter Overview and Learning Objectives

Scaling LLM data engineering is never only a tooling problem. It is also an organizational problem. Experience from DevOps and DataOps shows that stable delivery capability comes from the joint design of process, feedback, automation, and collaboration, not from replacing one tool with another (Kim et al. 2021; DataOps Manifesto 2024). When a team moves from "one person maintaining data scripts" to "multiple teams producing high-quality training data together," organizational structure, role boundaries, collaboration interfaces, and operating cadence become real bottlenecks.

This chapter explains how to build a scalable DataOps team and collaboration flywheel for LLM data engineering. It first shows why traditional data-team structures break under LLM workloads, then defines role boundaries, interface contracts, and RACI responsibilities. It then introduces the weekly operating rhythm of the DataOps flywheel, including meetings, SLAs, version freezes, cross-team asset sharing, risk escalation, and knowledge retention.

By the end of the chapter, readers will have a reusable organizational template: an LLM data-team role map, a RACI matrix, meeting cadence and deliverable tables, and practical principles for cross-team interfaces.

---

## Opening Scenario

An AI company is training a vertical model for education. The product team wants a new release in three weeks. The business team submits new annotation requests every day. The algorithm team needs clean training sets on schedule. The platform team maintains annotation tools and data pipelines. The legal team reviews whether data sources can be used.

Five teams, five rhythms. The product manager pushes progress in chat every day, but nobody can define what "data preparation is complete" means. The algorithm engineer receives a dataset version that does not match the annotation team's delivery version, and nobody knows who is responsible for aligning them. The platform team fixes a pipeline bug but does not notify annotation operations, so two days of annotation tasks become invalid. Legal identifies a copyright-risk batch, but the finding never reaches the filtering stage.

This is not an extreme case. It is normal once LLM data projects scale. The root cause is not that people are not working hard enough. The root cause is the absence of a shared organizational language: no clear role boundaries, no common delivery interface, and no predictable collaboration cadence.

The DataOps flywheel addresses precisely this missing language. It borrows the continuous-delivery idea of putting change into repeatable and verifiable pipelines, and it also absorbs SRE's emphasis on service levels, incident response, and postmortems (Humble and Farley 2010; Beyer et al. 2016).

---

## 24.1 Why LLM Data Teams Need a New Organizational Form

### 24.1.1 Structural Limits of Traditional Data Teams

Traditional data engineering teams inherited their shape from the data-warehouse and BI era. The core mission was to move business data into analytical systems accurately and reliably. Roles were clear: data engineers built ETL, analysts queried data, and BI developers produced reports. Data-quality problems were usually discovered by business users and then sent back to the warehouse team for repair.

LLM data engineering breaks this assumption. In LLM projects, data requirements come from rapidly changing sources: algorithm experiments change mixture ratios, product releases introduce new categories, RLHF requires preference annotation, and RAG scenarios require enterprise knowledge bases to keep changing. Data is no longer a passive object being transported. It becomes an experimental variable and a managed asset.

Three limits show up quickly.

First, responsibilities overlap and interfaces blur. LLM data projects involve data engineers, annotation operations, quality evaluators, algorithm engineers, product managers, and legal reviewers. Their outputs depend on one another, but their interfaces are rarely defined. What exactly must exist between "cleaning is complete" and "annotation may begin"? Who accepts the deliverable? Who traces quality defects?

Second, teams depend on single experts. Early teams often concentrate key decisions in a few people who "understand the whole pipeline." This works while the team is small, but it becomes a bottleneck as data volume doubles, annotation is split across vendors, and experiments run in parallel. It is also fragile: once that expert leaves, the pipeline may still run, but nobody can explain why.

Third, there is no continuous delivery rhythm. Many data teams deliver batches whenever tasks happen to finish. Algorithm teams cannot plan experiment windows, product teams cannot predict iteration cycles, and platform teams cannot plan capacity. Data work becomes push-driven: a batch appears only after somebody chases it.

These limits reinforce one another. Blurry interfaces increase reliance on experts. Expert reliance disrupts delivery rhythm. Unstable rhythm weakens organizational learning because the team is always reacting to the latest incident. In LLM projects, the result is high activity but low accumulation: repeated explanations, repeated confirmations, and repeated repairs.

| Structural limit | Typical symptom | Impact on LLM data projects | DataOps improvement |
|---|---|---|---|
| Overlapping responsibility | Multiple roles can modify data, but no final owner exists | Quality problems are hard to trace and repairs are duplicated | Establish a RACI matrix and data-owner mechanism |
| Blurry interfaces | Deliverables lack schema, acceptance criteria, and SLA | Downstream teams rework repeatedly; experiments slip | Build interface contracts and versioned data contracts |
| Single expert dependency | Key processes depend on a few senior people | The organization becomes fragile during leave, turnover, or parallel projects | Turn experience into templates, checklists, and postmortems |
| Missing cadence | Data batches are delivered ad hoc | Priorities change constantly; the team stays in firefighting mode | Establish weekly flywheel, monthly review, and version freeze |
| Invisible quality | Quality is sampled only after failure | Data defects become expensive once they enter training | Build quality metrics, sampling review, and automated validation |

*Table 24-1: Traditional data-team limits and DataOps improvements*

The organizational shift also changes how people understand responsibility. In traditional teams, many issues can be solved by asking the person who knows. In a DataOps organization, issues should be solved through institutional interfaces wherever possible. This does not reduce communication. It moves communication from ad hoc coordination to evidence-based collaboration through documents, metrics, and records.

### 24.1.2 New Collaboration Needs in LLM Projects

LLM data engineering introduces collaboration needs that traditional teams did not face.

**Need 1: tightly coupled data-model iteration.** In classic ML projects, data and algorithm teams can work relatively independently. In LLM projects, algorithm results directly reshape data demand. If evaluation shows weak math reasoning, the data team needs more math samples. If an RLHF experiment reveals an unpopular response style, annotation criteria must change. Data and algorithm teams must move in parallel, with fast feedback rather than serial handoff.

**Need 2: coordinated governance of heterogeneous sources.** LLM data comes from crawling, manual annotation, synthetic generation, third-party datasets, and user feedback. Each source has different quality standards, compliance requirements, and update frequencies. Without a shared governance frame, every source is managed in isolation.

**Need 3: continuous monitoring of annotation quality.** Annotation quality drifts over time. Annotator fatigue, unclear task boundaries, and vendor-specific interpretations can create systematic bias. A single pre-release inspection is not enough. Annotation quality must be operated continuously.

**Need 4: legal and security involvement from the beginning.** Copyright, privacy, and cross-border transfer risks cannot be reviewed only at release time. Compliance requirements affect collection, de-identification, retention, access, and usage boundaries. Legal and compliance roles must participate in the whole data-production lifecycle.

### 24.1.3 Design Principles for the New Organization

LLM data organizations should follow four principles.

**Interfaces before hierarchy.** Traditional organization design emphasizes reporting lines. DataOps design emphasizes delivery interfaces: who produces what, in what format, with what acceptance criteria. Clear interfaces matter more than strict hierarchy for cross-functional work.

**Asynchronous work with synchronous checkpoints.** Much data work can proceed asynchronously, such as annotation, cleaning, and sampling. Critical decisions require synchronous alignment, such as version freezes, quality-threshold changes, and escalation decisions.

**Knowledge capture and processization.** To reduce expert dependency, tacit knowledge must become explicit process. Every incident investigation, standard change, and cross-team conflict should leave reusable documentation.

**Small teams, large platforms.** Scaling data engineering should not simply mean adding people. Platform tools can multiply individual leverage: annotation management, quality evaluation, experiment tracking, lineage, permission control, and release workflows.

These principles mature in stages.

| Maturity stage | Organizational characteristics | Main risk | Priority capability | Avoid too early |
|---|---|---|---|---|
| Scripted collaboration | A few members rely on personal experience for collection, cleaning, and annotation | Single-point dependency; non-reproducible results | Basic version records, minimal quality checks, role confirmation | Heavy approval, excessive metrics, large platforms |
| Standardized collaboration | Multiple roles deliver through shared interfaces and regular cadence | Inconsistent execution; chaotic interface changes | RACI, schema, annotation guide, SLA | Complex orchestration before basic automation |
| Platformized collaboration | Tasks, annotation, quality, and versions enter a shared system | Platform detaches from real workflow | Workflow integration, automated checks, permission control, lineage | Feature accumulation unrelated to core flow |
| Measured collaboration | Metrics, postmortems, and experiment feedback drive improvement | Metric gaming; local optimization; ceremonial reviews | Value-flow metrics, quality trends, reuse value, learning loop | Metric contests detached from business goals |

*Table 24-2: DataOps maturity stages for LLM data teams*

Most companies should balance centralized and embedded models. Central teams standardize platforms, data rules, quality checks, and compliance boundaries. Embedded data roles stay close to algorithm and product needs. A practical structure is "central platform plus project embedding." The data owner becomes the key connector between data strategy, model goals, and organizational resources.

![Figure 24-1: Organizational evolution path for LLM data teams](../../images/part8/图24_1_LLM数据团队组织演进路径.png)

*Figure 24-1: Four-stage evolution from single-person scripts to platformized DataOps*

---

## 24.2 Roles, Interfaces, and RACI Design

### 24.2.1 Core Role Map for an LLM Data Team

A complete LLM data-engineering organization usually contains seven core roles. A small team may combine several roles in one person; a large company may split one role into subteams. The important point is not headcount, but accountability.

**Data Owner.** The overall owner of data assets. This role sets data strategy, approves dataset releases, resolves cross-team resource conflicts, and makes the final decision on whether a batch can enter training.

**Data Engineer.** Responsible for collection, cleaning, transformation, pipeline maintenance, and lineage. Data engineers own pipeline correctness, but they are not the only judges of content quality.

**Annotation Engineer / Annotation Operations.** Designs annotation tasks, manages annotators and vendors, controls task quality, and translates business needs into executable labeling instructions.

**Quality Evaluator.** Independently evaluates data quality across accuracy, consistency, coverage, duplicates, and rule compliance. This role should not be the same as the production role whenever possible.

**Algorithm Engineer.** The core consumer of data. Algorithm engineers must express data demand precisely and feed experiment results back to data teams as executable improvement tasks.

**Platform Engineer.** Builds and maintains annotation tools, pipeline platforms, storage, permissions, observability, and automation.

**Legal / Compliance Specialist.** Reviews sources, privacy, copyright, usage boundaries, retention, and access. This role should set upstream constraints, not only stamp approval at the end.

| Role | Core responsibility | Key input | Key output | Common failure mode |
|---|---|---|---|---|
| Data Owner | Strategy, priority, final release decision | Business goals, model roadmap, risk reports | Version decisions, resource allocation, conflict resolution | Approves without managing long-term quality and reuse |
| Data Engineer | Collection, cleaning, transformation, lineage | Sources, schema, collection authorization | Traceable batches, scripts, quality summaries | Focuses on pipeline success and ignores content explainability |
| Annotation Ops | Task design, distribution, annotator management | Annotation guide, sample pool, quality standards | Labeled data, logs, consistency reports | Standards are passed orally and vendors diverge |
| Quality Evaluator | Independent sampling, measurement, root-cause hints | Batches, rules, historical issues | Quality report, severity, repair suggestions | Only checks final results and misses process drift |
| Algorithm Engineer | Data requirements, experiment feedback, effect validation | Experiment results, error samples, evaluations | Data improvement requests, conclusions, release-risk feedback | Requests are too vague to execute |
| Platform Engineer | Tooling, permissions, stability, automation | Workflow needs, capacity plan, security requirements | Annotation platform, pipelines, monitoring | Tools do not match real workflows |
| Legal / Compliance | Source, privacy, copyright, use boundary | Collection plan, samples, business context | Risk level, constraints, approval decision | Joins too late, forcing completed data to be discarded |

*Table 24-3: Expanded role responsibilities in an LLM data team*

Two roles are especially easy to underdefine. Algorithm engineers are not only requesters; they are feedback providers who must convert experimental failure into specific data requests. Legal specialists are not only final approvers; they should turn risk constraints into data-admission rules.

### 24.2.2 Role Interface Contracts

Every pair of collaborating roles should define four elements: deliverable, format specification, acceptance criteria, and SLA.

| Element | Example: Data Engineer to Annotation Ops |
|---|---|
| Deliverable | Cleaned raw sample pool and files awaiting annotation |
| Format specification | JSONL with `id`, `content`, `source`, and `clean_timestamp` |
| Acceptance criteria | Duplicate rate < 1%; blank rate < 0.5%; abnormal character rate < 0.1% |
| Delivery SLA | Weekly batch delivered before Friday 18:00 |

*Table 24-4: Example interface from data engineering to annotation operations*

| Element | Example: Annotation Ops to Algorithm Engineer |
|---|---|
| Deliverable | Labeled training set with annotation metadata |
| Format specification | JSONL with `instruction`, `output`, `annotator_id`, `confidence`, and `revision_count` |
| Acceptance criteria | Inter-annotator agreement > 0.85; sampling acceptance rate > 95% |
| Delivery SLA | Completed five business days before iterative model release |

*Table 24-5: Example interface from annotation operations to algorithm engineering*

An interface contract should contain four layers:

| Contract layer | Questions answered | Recommended carrier | Validation method |
|---|---|---|---|
| Structural | Are fields present, typed correctly, and enumerations valid? | JSON Schema, Avro, table schema | Automated validation and CI |
| Semantic | What does each field mean, with which boundaries and edge cases? | Data dictionary, dataset card, annotation guide | Review and sample inspection |
| Operational | Who delivers, when, how accepted, and how feedback flows? | SLA, interface agreement, RACI | Weekly review and service metrics |
| Change | Why change, what is affected, how compatible, when deprecated? | Changelog, release notes, migration guide | Change review and downstream confirmation |

*Table 24-6: Four-layer structure of a data-role interface contract*

Versioned interfaces require compatibility windows. If a field meaning or format changes, downstream consumers need a migration period. For training data, compatibility is also an experimental concern: if an experiment uses data after a definition changed, the result may no longer be comparable to earlier runs.

### 24.2.3 RACI Responsibility Matrix

RACI stands for Responsible, Accountable, Consulted, and Informed. It is useful because it forces a team to distinguish execution from final accountability.

| Activity | Data Owner | Data Engineer | Annotation Ops | Quality Evaluator | Algorithm Engineer | Platform Engineer | Legal / Compliance |
|---|---|---|---|---|---|---|---|
| Data requirement definition | A | C | C | I | R | I | C |
| Data collection and cleaning | I | R/A | C | I | I | C | C |
| Annotation-task design | C | I | R/A | C | C | I | C |
| Annotation-quality review | I | I | C | R/A | C | I | I |
| Training-set release | A | C | C | C | R | I | C |
| Compliance review | A | I | I | I | I | I | R |
| Platform incident response | I | C | I | I | I | R/A | I |
| Data-version rollback | A | R | C | C | C | C | I |
| Cross-team sharing approval | A | C | I | I | C | I | C |
| Vendor management | I | I | R/A | C | I | I | I |

*Table 24-7: RACI matrix for an LLM data team*

Each row should have exactly one Accountable owner. If two people are "jointly accountable," nobody is truly accountable. RACI should also be reviewed after important version cycles because LLM data tasks change by phase: early projects emphasize collection and cleaning, middle phases emphasize annotation quality and experiment feedback, and late phases emphasize release, compliance, and online feedback.

| Scenario | How RACI is used | Management benefit | Avoid |
|---|---|---|---|
| Data-demand intake | Clarify requester, priority owner, and resource evaluator | Reduces queue-jumping and oral promises | Every request requiring top-level approval |
| Annotation-standard change | Clarify drafter, reviewer, and vendor notifier | Reduces drift in labeling criteria | Announcing changes only in meetings |
| Dataset release | Clarify release owner, quality confirmer, and notification list | Improves traceability | Filling release notes after the fact |
| Quality incident | Clarify repair owner, root-cause analyst, and final decision maker | Shortens diagnosis | Blaming the individual instead of the system |
| Compliance risk | Clarify risk identification, suspension, and restoration responsibility | Reduces legal and reputation risk | Compliance opinions disconnected from engineering action |

*Table 24-8: RACI usage in daily DataOps scenarios*

### 24.2.4 Escalation Paths and Exception Handling

No organizational design covers every case. DataOps needs predefined escalation paths.

**Emergency escalation.** If a decision must be made before the normal process can complete, such as a quality crisis before a key training window, the on-duty data owner may bypass routine approval and decide directly.

**Cross-RACI conflict.** If two teams disagree on who owns a quality defect, the data owner resolves the boundary instead of letting teams defend their own interpretations.

**Exception approval.** If the algorithm team needs unaccepted raw data for a temporary experiment, the exception must record reason, approver, usage limit, isolation method, and expiry date.

| Exception type | Use case | Required record | Approval | Expiry handling |
|---|---|---|---|---|
| Time exception | Experiment window is near and acceptance is unfinished | Batch, unfinished checks, risk note | Data owner or duty owner | Complete acceptance and decide whether to promote |
| Quality exception | Rough labels or exploratory data below formal quality | Gap, usage limit, isolation tag | Data owner and algorithm lead | Cannot enter formal release unless repaired |
| Format exception | Downstream needs a temporary non-standard format | Field difference, conversion script, compatibility window | Data engineering lead | Migrate or retire temporary format |
| Compliance exception | Source or authorization still under review | Source note, risk level, access list | Legal/compliance plus data owner | Reclaim access automatically at expiry |

*Table 24-9: DataOps exception types and handling requirements*

Every emergency escalation or exception should be reviewed in the next weekly or monthly retrospective. Exceptions without review erode governance. Exceptions with review reveal the gap between designed process and real workflow.

---

## 24.3 The DataOps Flywheel and Weekly Cadence

### 24.3.1 What the DataOps Flywheel Is

The DataOps flywheel is a model for continuous improvement in data teams. Its core idea is to use a fixed operating rhythm to connect demand, production, quality evaluation, and feedback into a self-reinforcing loop.

The LLM data flywheel is driven by four pools:

**Demand Pool.** Data requests from algorithm, product, and business teams, prioritized into executable tasks.

**Data Pool.** Current usable data assets, including cleaned raw data, labeled samples, and compliant datasets.

**Experiment Pool.** Algorithm experiments, including dataset versions, parameter configurations, evaluation results, and conclusions.

**Issue Pool.** Known problems, including quality defects, compliance risks, and pipeline incidents.

The loop works as follows: the demand pool generates tasks; tasks produce or update data; algorithm teams consume data and create experiment results; failures or findings enter the issue pool; repairs update the data pool and adjust demand priorities. Each weekly loop should produce clearer priorities, better data, and more reusable knowledge.

| Flywheel object | Key fields | Relationship to other objects | Management focus |
|---|---|---|---|
| Demand pool | ID, requester, priority, target metric, deadline | Links to data tasks, experiment plans, and business goals | Prevent vague demand; keep tasks executable |
| Data pool | Version, source, scale, quality summary, compliance status | Links to collection tasks, annotation batches, experiments | Make assets visible and reusable |
| Experiment pool | ID, data version, model version, evaluation result, conclusion | Links to demand goals, data versions, and error samples | Judge whether data investment improves the model |
| Issue pool | Severity, discovery source, root cause, owner, repair state | Links to data batches, interface contracts, postmortems | Prevent recurrence and improve process |

*Table 24-10: Four DataOps flywheel pools and their management fields*

Feedback must be short and precise. Each iteration should provide task-level feedback, quality-level feedback, and effect-level feedback. Otherwise the team may optimize delivery speed while losing sight of data value.

![Figure 24-2: Four-pool collaboration in the DataOps flywheel](../../images/part8/图24_2_DataOps飞轮四池协同示意图.png)

*Figure 24-2: The DataOps flywheel connecting demand, data, experiment, and issue pools*

### 24.3.2 Weekly Cadence

The flywheel needs fixed time anchors. For a medium LLM data team of 10-30 people, the following cadence works well.

| Rhythm | Time | Participants | Core output |
|---|---|---|---|
| Monday demand sync | Monday 09:30, 30 minutes | Data owner, algorithm representative, product manager | Weekly task list |
| Wednesday quality inspection | Wednesday, asynchronous | Quality evaluator, annotation operations | Inspection report |
| Friday delivery retrospective | Friday 16:00, 45 minutes | Full data team | Weekly report and next-week preview |
| Monthly review | Last Friday of month, 2 hours | Data team plus algorithm and product reps | Monthly report and next-month OKR draft |
| Quarterly version freeze | End of quarter | Data owner and algorithm reps | Baseline dataset version |

*Table 24-11: DataOps meeting cadence and deliverables*

Meetings are useful only when each node has clear input and output. Monday sync should start from an already curated demand pool, not from open-ended brainstorming. Wednesday inspection should start from automated checks and sampled examples, not from intuition. Friday review should start from versions, SLA records, and issue-pool changes, not from a vague progress round.

| Node | Recommended input | Recommended output | Success criterion | Common drift |
|---|---|---|---|---|
| Monday demand sync | Candidate demands, unfinished tasks, algorithm findings | Weekly commitment list and priorities | Demand is executable and ownership is clear | Becomes ad hoc demand collection |
| Wednesday inspection | Automated quality report and samples | Issue severity, repair suggestions, warnings | High-risk problems surface early | Only acceptance rate is discussed |
| Friday review | Versions, SLA records, issue changes | Next-week improvements and process changes | Unfinished work has a disposition | Only progress is reported |
| Monthly review | Quality trend, reuse, cost, incidents | Improvement plan and metric adjustments | Cross-team consensus forms | Too many metrics dilute focus |
| Quarterly freeze | Candidate version, quality report, compliance confirmation | Baseline version, lineage record, known limits | Version is reproducible and auditable | Freeze is only a file copy |

*Table 24-12: Inputs, outputs, and common drifts for DataOps operating nodes*

Mature teams distinguish decision meetings from learning meetings. Demand sync and version freeze are decision meetings. Quality inspection and postmortems are learning meetings. Mixing the two either makes decision meetings long or makes postmortems shallow.

### 24.3.3 SLA and Version Freeze

An SLA is an internal service promise from the data team to its consumers. It gives downstream teams predictable delivery while giving the data team a stable operating rhythm.

| Data type | Processing time | Quality target | Notes |
|---|---|---|---|
| Emergency data, P0 | Cleaning within 24 hours, annotation within 48 hours | Sampling acceptance > 90% | Requires data-owner approval |
| Regular iteration data, P1 | Cleaning within 3 business days, annotation within 5 business days | IAA > 0.85; acceptance > 95% | Scheduled weekly |
| Exploratory data, P2 | Cleaning within 5 business days; incomplete annotation allowed | Rough labels; acceptance > 80% | Internal experiments only |
| Historical-data repair | Based on impact, maximum 2 weeks | Same quality as original version | Repair log required |

*Table 24-13: Example SLA framework*

LLM data SLAs should cover time, quality, availability, traceability, and compliance. Time without quality encourages fast low-value delivery. Quality without time prevents downstream planning.

| SLA dimension | Core question | Example metric | Management value |
|---|---|---|---|
| Time | When will delivery or response happen? | P0 in 48 hours; regular batch in 5 business days | Helps algorithm and product teams plan experiments |
| Quality | Does the deliverable meet usage standards? | Acceptance rate, IAA, duplicate rate, anomaly rate | Keeps bad data out of training |
| Availability | Can downstream systems read and process it? | Schema validation pass rate, pipeline success, read failure | Reduces integration cost |
| Traceability | Can source and process be explained? | Lineage completeness, version-record completeness | Supports audit, rollback, and diagnosis |
| Compliance | Does it respect authorization and boundaries? | Authorization coverage, de-identification pass rate | Reduces legal and privacy risk |

*Table 24-14: Multi-dimensional DataOps SLA design*

Version freeze snapshots a dataset state at a scheduled point, such as month-end or quarter-end. Frozen versions are immutable. Later changes must create new versions. A useful frozen-version note should include the version goal, sources, scale, cleaning rules, annotation standard, quality summary, compliance conclusion, known defects, suitable use cases, and unsuitable use cases.

Version management can use three lifecycle states: candidate, frozen, and archived. Candidate versions allow repair. Frozen versions support formal experiments and reproducibility. Archived versions support long-term audit with stricter access.

Metrics should tell whether the flywheel is healthy:

| Metric layer | Examples | Data source | Interpretation | Misuse risk |
|---|---|---|---|---|
| Flow efficiency | Lead time, queue wait, SLA attainment | Demand pool and task system | Whether work flows smoothly | Chasing speed at the expense of quality |
| Quality stability | Acceptance rate, IAA, duplicate rate, rework | Quality reports and annotation platform | Whether data remains usable | Averages hide distribution problems |
| Collaboration reliability | Interface-change notice, version-record completeness | Interface docs and release system | Whether agreements are followed | Document completeness mistaken for understanding |
| Model effect | Error reduction, key-eval gain, data contribution | Experiment pool and evaluations | Whether data investment works | Attributing all model gain to data |
| Reuse value | Dataset reuse count, repeated-collection reduction | Asset catalog and access logs | Whether data becomes an organizational asset | Encouraging reuse without context boundaries |

*Table 24-15: Layered metrics for the DataOps flywheel*

Metrics need governance too: stable definitions, clear boundaries, distribution views, example-backed interpretation, and linked actions.

---

## 24.4 Cross-Team Collaboration and Risk Governance

### 24.4.1 Conflicts in Multi-Team Data Asset Sharing

When multiple LLM projects run in parallel, shared data assets become a coordination problem. Common conflicts include:

- **Ownership conflict.** One project spends months preparing a dataset, while another wants to reuse it without sharing maintenance cost.
- **Priority conflict.** Several projects compete for the same annotation, compute, or quality-review capacity.
- **Standard conflict.** A dataset suitable for one model may have label definitions or quality thresholds unsuitable for another.
- **Compliance conflict.** A dataset may be reusable technically but restricted legally by purpose, contract, or privacy constraints.

Data sharing should be managed through an asset catalog, usage boundaries, reuse approval, and contribution recognition. The team that produced a valuable dataset should receive visible credit through reuse count, saved cost, or model benefit. At the same time, not all data should be freely shared. Adversarial samples, sensitive feedback, and evaluation sets can be harmful if reused without boundaries.

### 24.4.2 Knowledge Retention and Postmortems

The strongest way to prevent knowledge loss is to require structured knowledge capture. Every data incident, quality investigation, and major annotation-standard change should produce a postmortem with:

- Event description: what happened and what was affected.
- Root-cause analysis: process issue, tool issue, or standard issue.
- Repair action: what fixed the problem.
- Prevention: how recurrence will be avoided.
- Process change: whether RACI, interface contracts, or SLA must change.

Postmortems should avoid two extremes: blame-oriented review and ceremonial review. Effective postmortems focus on systemic causes: why the current process allowed the error through, why monitoring did not catch it, and why prior similar issues were not institutionalized.

| Document type | Content | Maintainer | Update trigger | Main use |
|---|---|---|---|---|
| Incident postmortem | Timeline, impact, root cause, repair, prevention | Incident owner and quality evaluator | Incident | Prevent recurrence |
| Best practice | Effective data strategy, annotation method, quality rule | Responsible team | Monthly supplement | Spread successful practice |
| Standard template | Dataset card, annotation guide, quality report, release note | Data owner and platform engineer | Version cycle | Reduce collaboration cost |
| Decision record | Tradeoff, approval, exception reason, alternatives | Decision initiator and data owner | Decision | Preserve organizational memory |
| Onboarding guide | Role map, tools, FAQ, example flow | Team lead and senior members | Quarterly review | Shorten onboarding |

*Table 24-18: DataOps knowledge objects and maintenance mechanisms*

Knowledge artifacts must be searchable and linked. Postmortems should link to dataset versions, issue IDs, quality reports, and interface contracts. Dataset cards should link to collection tasks, annotation guides, and compliance approvals. Otherwise a knowledge base becomes a pile of documents rather than an engineering asset.

### 24.4.3 Risk Escalation and Decision Committee

The following risks should immediately escalate to the data-owner level:

- Data compliance risk, such as suspected copyright or privacy leakage.
- Large-scale training-set quality anomaly, such as more than 10% failed samples.
- Severe SLA violation on the critical path, such as a P0 request delayed by more than 48 hours.
- Cross-team conflict that cannot be resolved within the team.

For systemic risks affecting multiple projects, such as platform failure or policy change, a temporary decision committee should meet within 24 hours. The committee should focus on four questions: whether to suspend data use, whether to rollback or isolate versions, whether external teams or customers need communication, and whether the process must change.

| Risk level | Typical scenario | Response time | Decision body | Handling requirement |
|---|---|---|---|---|
| P0 | Privacy leak, major copyright risk, severe online data error | Immediate response; conclusion within 24 hours | Decision committee, legal, data owner | Suspend use, isolate data, formal incident report |
| P1 | Formal training-set quality anomaly, severe SLA breach | Respond within 4 hours; repair or mitigation within 48 hours | Data owner, quality lead, team leads | Define impact, repair plan, rollback strategy |
| P2 | Single-batch quality fluctuation, interface compatibility issue | Respond within one business day | Responsible team lead | Enter issue pool and weekly follow-up |
| P3 | Missing document, low-risk process deviation, noncritical metric anomaly | Next work cycle | Execution team | Fill document or add to monthly improvement |

*Table 24-19: DataOps risk levels and response mechanism*

Risk governance must also include permissions. LLM data spans public corpora, internal documents, user feedback, annotation results, generated data, and de-identified business data. Access should follow least privilege and should be limited by field, version, task, and time whenever possible.

| Sensitivity level | Example | Default access policy | Audit requirement |
|---|---|---|---|
| Public data | Open datasets, public webpages | Project members may request | Record version references and download behavior |
| Internal ordinary data | Knowledge base, product docs | Authorized related projects only | Record user, purpose, and expiry |
| Internal sensitive data | User feedback, business logs | Least privilege, de-identification first, time-bound | Field-level access and export logs |
| Restricted data | Privacy, copyright, or contract-risk data | Deny by default; exception approval | Full audit, isolated environment, expiry recovery |
| Derived data | Cleaned samples, labels, generated data | Reclassified by source and processing method | Source lineage and usage boundary |

*Table 24-20: LLM data sensitivity levels and access governance*

Vendor work should also be part of DataOps, not only procurement. Task distribution, guide training, calibration examples, sampling inspection, feedback, and permission recovery must all be recorded.

| Vendor-governance stage | Control point | Recommended evidence |
|---|---|---|
| Admission | NDA, compliance, and tool training completed | Training record, permission approval, test-task result |
| Task distribution | Uses formal guide and versioned examples | Task ID, guide version, sample-library reference |
| Process monitoring | Quality drift or response delay | Calibration results, rework rate, dispute ratio |
| Acceptance | Meets agreed quality | Sampling report, consistency report, issue list |
| Feedback | Standards corrected promptly | Q&A record, guide change, retraining record |
| Exit and recovery | Permissions closed and deliverables archived | Permission-recovery record, deletion confirmation, archive |

*Table 24-21: DataOps control points for outsourced annotation*

---

## 24.5 Case Study: From a Small Team to a Platformized Organization

### Background

Company E, an online education company, began building a K12 education model in early 2023. The initial team had five people: two algorithm engineers who also handled data engineering, and three part-time annotators from the curriculum team. The setup worked because the team was small and communication was direct.

In the second half of 2023, the project scaled. Two annotation vendors were added. Data volume grew from tens of thousands to millions of records. The algorithm team expanded to eight people. Two new subprojects started: oral evaluation and essay correction.

Problems emerged:

- Annotation tasks from three projects were mixed together, and vendors could not tell which standard applied to which batch.
- Algorithm engineers reported many duplicates in the latest training set, but nobody knew which cleaning step caused them.
- The oral-evaluation project built a separate cleaning tool incompatible with the main toolchain.
- A vendor annotation result was manually modified by an algorithm engineer and used as training data, breaking the quality-traceability chain.

The deeper cause was organizational. Tasks were still assigned through chat, annotation guides had inconsistent terminology, and algorithm engineers often copied intermediate files to meet experiment deadlines. Local efficiency created global opacity.

### Reorganization Process

Company E implemented DataOps reorganization in three stages.

**Stage 1, months 1-2: clarify roles and interfaces.**

The team ran a two-week role workshop and defined five roles: data owner, data engineer, annotation operations, quality evaluator, and algorithm liaison. It produced an initial RACI matrix and interface contracts for data engineering to annotation operations, and annotation operations to algorithm liaison.

The team also versioned the annotation guide. Every guide change had to include reason, affected fields, example samples, and effective time. Vendors could only follow released guide versions. Algorithm teams requesting label changes had to submit them through the demand pool.

| Object | Before | After |
|---|---|---|
| Data demand | Raised in chat without ID | All demands enter the demand pool with priority and acceptance criteria |
| Annotation guide | Multiple documents with unclear versions | Unified version management with reasons and examples |
| Data delivery | Files sent directly to algorithm engineers | Delivery through versioned release process |
| Quality review | Ad hoc sampling by experts | Quality evaluator issues a sampling report |
| Vendor communication | Multiple internal people answer separately | Annotation operations owns communication and FAQ |

*Table 24-22: Stage-1 role and interface clarification at Company E*

**Stage 2, months 3-4: build cadence and tool constraints.**

The company established Monday demand sync and Friday delivery review, unified data format as JSONL plus shared schema, deployed an internal annotation platform, and introduced DVC-based dataset versioning.

The key was not "adding a tool." The team first unified fields, task naming, and acceptance rules, then embedded task distribution, annotation return, sampling results, and version release into the platform. The system recorded the real data-production process instead of isolated operations.

**Stage 3, months 5-6: operate quality and capture knowledge.**

The team added automated checks for duplicates, formats, and consistency; created a monthly quality-trend report; and completed three major postmortems. Postmortems were split into incident analysis and process change. A postmortem closed only when the process change was implemented in a template, tool rule, or meeting mechanism.

| Stage | Main goal | Key actions | Organizational benefit |
|---|---|---|---|
| Months 1-2 | Clarify roles and interfaces | Workshops, RACI, contracts, guide versioning | Reduces unclear ownership and oral agreements |
| Months 3-4 | Establish rhythm and tool constraints | Weekly cadence, schema, annotation platform, versioning | Improves predictability and traceability |
| Months 5-6 | Build quality operations and learning | Automated checks, quality trends, postmortems, boundary examples | Improves quality stability and organizational learning |

*Table 24-23: Company E's DataOps reorganization stages and benefits*

### Results

Six months later, key metrics changed significantly.

| Metric | Before reorganization | After reorganization |
|---|---:|---:|
| Data-delivery delay rate | Around 40% estimated | < 10% |
| Annotation-batch acceptance rate | Around 82% | > 93% |
| Average time to locate data root cause | 3 days | 4 hours |
| Cross-project data reuse rate | Around 5% | > 30% |
| New-member onboarding time | Around 3 weeks | Around 1 week |

*Table 24-24: Core metrics before and after DataOps reorganization*

The deeper change was in working style. The team moved from firefighting to rhythmic execution. The data lead spent less time on cross-team coordination and more time on data strategy and quality standards. Algorithm feedback became more specific: experiments had to link data versions and error-sample analysis, so the data team could replenish samples by task type, knowledge point, or interaction scenario instead of blindly increasing volume.

| Lesson | Practice | Transferable insight |
|---|---|---|
| Govern interfaces before platforms | Unified schema, guide, and delivery rules before tools | Tools work when they carry real process |
| Record versions before automation | Every release gets a version and quality summary | Traceability is the base of later automation and audit |
| Triage first, optimize later | Solve P0/P1 problems and high-rework tasks first | DataOps should start from high-impact bottlenecks |
| Capture examples before training people | Boundary-example library aligns labeling | Examples transmit complex standards better than abstract rules |
| Build feedback before scaling data | Use experiment errors to guide data replenishment | Data growth must serve explicit model improvement |

*Table 24-25: Transferable lessons from Company E*

The case can be generalized into a lightweight checklist.

| Check dimension | Key question | Initial symptom | Target state |
|---|---|---|---|
| Role responsibility | Is there a final accountable owner for each key decision? | Many participants, no decision owner | Each key item has one A and clear R |
| Demand management | Do all data demands enter one pool? | Demands scattered across meetings and chat | ID, priority, acceptance criteria, deadline |
| Interface contract | Do upstream and downstream teams have stable delivery contracts? | Fields and meanings depend on oral explanation | Schema, semantic notes, and SLA documented |
| Annotation governance | Is the guide versioned and traceable? | Several versions coexist | Guide, examples, Q&A, and changes maintained together |
| Quality control | Can quality trends be observed continuously? | Sampling only before release | Automated checks plus human sampling and trend analysis |
| Version management | Can historical experiment data be reproduced? | Folder names and manual notes | Data versions, model experiments, and quality reports linked |
| Permission and compliance | Are permissions scoped by purpose and expiry? | Broad long-term project access | Least privilege, expiry recovery, and audit |
| Postmortem | Does every incident become process improvement? | Reviews stay as summary text | Closure requires process, tool, or template change |
| Data reuse | Are valuable assets discoverable? | Project-private storage | Discoverable, requestable, traceable, recoverable |
| Metric governance | Do metrics support learning rather than display? | Only quantity and progress | Flow, quality, collaboration, and effect metrics |

*Table 24-26: Lightweight checklist for DataOps transformation*

A twelve-week pilot is often more practical than a one-year platform plan.

| Week | Focus | Deliverable | Acceptance criterion |
|---|---|---|---|
| 1 | Interview core roles and map current flow | Problem list and data-flow sketch | Covers main production and consumption roles |
| 2 | Define data owner, RACI, and escalation | RACI draft and escalation path | Every key item has one accountable owner |
| 3 | Select one high-frequency chain as pilot | Pilot scope and risk boundary | Scope is small and impact is clear |
| 4 | Build demand pool and task template | Demand template and priority rule | New demands can be submitted and reviewed |
| 5 | Define interface contract and schema | Contract, schema, field notes | Upstream and downstream agree on delivery |
| 6 | Version guide and example library | Guide, boundary examples, FAQ | Vendors and internal staff use the same version |
| 7 | Add basic quality checks | Dedup, format, anomaly, sampling rules | A quality summary appears before release |
| 8 | Connect version release and experiment reference | Version note and experiment-reference rule | Experiments trace to data versions |
| 9 | Run weekly review and close first issues | Issue pool and postmortem records | High-priority issues have owner and deadline |
| 10 | Summarize quality, delivery, and collaboration metrics | Dashboard or monthly report | Metric definitions are clear |
| 11 | Evaluate pilot and revise process | Pilot report and process-change record | Benefits and residual risks are explicit |
| 12 | Decide rollout scope and next focus | Rollout plan and resource needs | Management and teams align |

*Table 24-28: Twelve-week DataOps pilot route*

Pilot selection matters. The pilot should not be a trivial side task, because success would not prove organizational value. It should also not be the riskiest core path. A good pilot has a reasonably complete data chain, involves at least two teams, and has visible measurable pain.

| Rollout scenario | Recommended governance strength | Process | Reason |
|---|---|---|---|
| Formal training set | High | Full demand, quality, version, compliance, release | Affects model capability and reproducibility |
| Key evaluation set | High | Strict freeze, access control, contamination prevention | Evaluation contamination damages model judgment |
| Exploratory data | Medium | Simplified demand, quality label, usage limit | Keeps experiment speed while preventing formal misuse |
| Public corpus cleaning | Medium | Source record, cleaning rule, quality summary | Lower risk but large scale requires traceability |
| Temporary analysis samples | Low | Time-bound access, basic record, expiry deletion | Short-term value does not justify heavy process |
| Sensitive user feedback | High | Least privilege, de-identification, audit, compliance | Privacy and boundary mistakes are costly |

*Table 24-29: Governance strength by data scenario*

The final self-test for DataOps is explainability. Can the team answer who approved the latest release and on what evidence? Which data version did a model experiment use? Which batches did an annotation-standard change affect? Which high-priority issue produced a prevention measure? If the answer is still "someone probably handled it" or "it is probably in a folder," DataOps is not yet part of the organization.

---

## Chapter Summary

This chapter presented the organizational design and DataOps flywheel for LLM data teams.

At the organizational level, we analyzed the three limits of traditional data teams in LLM projects: overlapping responsibility, single-expert dependency, and missing cadence. We then introduced four design principles: interfaces before hierarchy, asynchronous work with synchronous checkpoints, knowledge capture, and small teams with large platforms.

At the role level, we defined seven core roles: data owner, data engineer, annotation operations, quality evaluator, algorithm engineer, platform engineer, and legal/compliance specialist. We designed interface contracts, a RACI matrix, and exception-handling paths.

At the flywheel level, we introduced the four-pool model of demand, data, experiment, and issue pools; the weekly cadence of Monday demand sync, Wednesday quality inspection, Friday delivery review; and monthly reviews and quarterly freezes.

At the governance level, we discussed multi-team asset sharing, knowledge retention, postmortems, risk escalation, permission control, and vendor management.

Finally, the Company E case showed how a five-person team evolved into a platformized DataOps organization, reducing delivery delay, improving acceptance rate, shortening root-cause analysis, increasing reuse, and accelerating onboarding.

DataOps is not a toolset. It is an upgraded way of working. Its value is not that it runs perfectly on day one, but that fixed cadence and clear interfaces allow a team to improve quality and efficiency through repeated iterations.

![Figure 24-3: Full DataOps organization map](../../images/part8/图24_3_DataOps团队组织全景图.png)

*Figure 24-3: Integrated view of roles, interfaces, flywheel, and governance in an LLM DataOps team*

---

## Further Reading

**Methodology.** Kim et al.'s *The DevOps Handbook* is a classic reference for understanding DevOps culture and practice. Its three ways of flow, feedback, and continuous learning are directly useful for DataOps design. The DataOps Manifesto from DataKitchen summarizes widely cited principles for collaborative, repeatable, and feedback-driven data work.

**Open-source tools.** Apache Airflow is a mature workflow orchestrator for DAG-style data pipelines. Prefect offers a more flexible workflow system for dynamic tasks and error handling. Label Studio is a widely used open-source annotation platform for text, image, and audio data.

---

## Next Chapter

With organization and operating cadence in place, the next question is data version management. When datasets keep changing, how can the team know which data a model version used? How can a problem be traced back to its source? How can experiments remain reproducible?

Chapter 25 turns to the engineering implementation of data versioning and experiment tracking, from version-granularity design to experiment-card fields and traceable metadata.

## References

Amershi S, Begel A, Bird C, DeLine R, Gall H, Kamar E, Nagappan N, Nushi B, Zimmermann T (2019) Software Engineering for Machine Learning: A Case Study. In: Proceedings of the 41st International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP), pp 291-300.

Baylor D, Breck E, Cheng H-T, Fiedel N, Foo C Y, Haque Z, Haykal S, Ispir M, Jain V, Koc L, Koo C Y, Lew L, Mewald C, Modi A N, Polyzotis N, Ramesh S, Roy S, Whang S E, Wicke M, Wilkiewicz J, Zhang X, Zinkevich M (2017) TFX: A TensorFlow-Based Production-Scale Machine Learning Platform. In: Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp 1387-1395.

Beyer B, Jones C, Petoff J, Murphy N R (eds.) (2016) Site Reliability Engineering: How Google Runs Production Systems. O'Reilly Media.

Breck E, Cai S, Nielsen E, Salib M, Sculley D (2017) The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. In: IEEE International Conference on Big Data, pp 1123-1132.

Breck E, Polyzotis N, Roy S, Whang S E, Zinkevich M (2019) Data Validation for Machine Learning. In: Proceedings of Machine Learning and Systems 1, pp 334-347.

DAMA International (2017) DAMA-DMBOK: Data Management Body of Knowledge, 2nd Edition. Technics Publications.

DataOps Manifesto (2024) The DataOps Manifesto: 18 DataOps Principles. Available at: https://dataopsmanifesto.org/en/

Forsgren N, Humble J, Kim G (2018) Accelerate: The Science of Lean Software and DevOps. IT Revolution Press.

Gebru T, Morgenstern J, Vecchione B, Vaughan J W, Wallach H, Daume III H, Crawford K (2021) Datasheets for Datasets. Communications of the ACM 64(12):86-92.

Humble J, Farley D (2010) Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation. Addison-Wesley.

Humble J, Molesky J, O'Reilly B (2015) Lean Enterprise: How High Performance Organizations Innovate at Scale. O'Reilly Media.

Kim G, Humble J, Debois P, Willis J, Forsgren N (2021) The DevOps Handbook: How to Create World-Class Agility, Reliability, and Security in Technology Organizations, 2nd Edition. IT Revolution Press.

Kreuzberger D, Kuehl N, Hirschl S (2023) Machine Learning Operations (MLOps): Overview, Definition, and Architecture. IEEE Access 11:31866-31879.

Mitchell M, Wu S, Zaldivar A, Barnes P, Vasserman L, Hutchinson B, Spitzer E, Raji I D, Gebru T (2019) Model Cards for Model Reporting. In: Proceedings of the Conference on Fairness, Accountability, and Transparency, pp 220-229.

Project Management Institute (2021) A Guide to the Project Management Body of Knowledge (PMBOK Guide), 7th Edition. Project Management Institute.

Reis J, Housley M (2022) Fundamentals of Data Engineering. O'Reilly Media.

Sambasivan N, Kapania S, Highfill H, Akrong D, Paritosh P, Aroyo L M (2021) "Everyone wants to do the model work, not the data work": Data Cascades in High-Stakes AI. In: Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, pp 1-15.

Sculley D, Holt G, Golovin D, Davydov E, Phillips T, Ebner D, Chaudhary V, Young M, Crespo J-F, Dennison D (2015) Hidden Technical Debt in Machine Learning Systems. In: Advances in Neural Information Processing Systems 28, pp 2503-2511.

Skelton M, Pais M (2019) Team Topologies: Organizing Business and Technology Teams for Fast Flow. IT Revolution Press.
