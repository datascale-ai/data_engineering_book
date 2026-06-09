# Chapter 24: The DataOps Flywheel and Team Organization

---

## Abstract
Scaling LLM data engineering is not merely a matter of tools and processes—it is fundamentally an organizational challenge. The shared experience of DevOps and DataOps demonstrates that stable delivery capability emerges from the co-design of processes, feedback loops, automation, and organizational collaboration, not from replacing individual tools (Kim et al. 2021; DataOps Manifesto, accessed 2026). When a team evolves from "a single person maintaining data scripts" to "multiple teams collaborating to produce high-quality training data," organizational structure, role boundaries, collaboration interfaces, and operational cadence become real bottlenecks. This chapter is addressed to managers responsible for designing data team organization, processes, and coordination mechanisms. It systematically explains how to build a scalable DataOps team and collaboration flywheel for LLM data engineering.

The chapter unfolds across four dimensions. Research on high-performing technology organizations shows that delivery frequency, change lead time, recovery time, and failure rate are, at their core, reflections of team collaboration, feedback velocity, and the capacity for continuous improvement (Forsgren, Humble and Kim 2018). First, the chapter explains why traditional data team structures break down in LLM projects and reveals the design logic behind new organizational forms. Second, it establishes role divisions, interface protocols, and a RACI responsibility matrix so teams can be clear on "who does what, who decides, and who approves." Third, it introduces the operating mechanism and weekly cadence of the DataOps flywheel, including meeting systems, SLA settings, and version freezes. Finally, it discusses practical approaches to cross-team data asset sharing, risk governance, and knowledge preservation.

After reading this chapter, readers will have a directly applicable organizational template: a role map for LLM data teams, a RACI matrix, a meeting cadence and deliverables table, and design principles for cross-team interfaces.

---

## Keywords

DataOps; experiment tracking; data versioning; observability; organizational coordination

## Learning Objectives

- Identify the root causes of structural failures in traditional data team structures within LLM data projects.
- Design role divisions, interface protocols, and RACI responsibility matrices for LLM data teams.
- Build the weekly cadence of the DataOps flywheel, including meeting systems, SLAs, and version-freeze mechanisms.
- Design practical workflows for cross-team data asset sharing, risk governance, and knowledge preservation.
- Compare the trade-offs between interface-first and hierarchy-first organizational design principles in multi-team collaboration.

## Scenario Introduction

An AI company is training a domain-specific large model for educational applications. The product team wants a new iteration in three weeks, the business team has new data-annotation requirements every day, the algorithm team needs clean training sets on time, the platform team maintains annotation tools and data pipelines, and the legal team reviews data-source compliance.

Five teams. Five rhythms. The product manager urges progress in the group chat every day, yet nobody is clear on what "data preparation complete" means. The dataset version the algorithm engineers receive does not match the version the annotation engineers delivered, but nobody knows who is responsible for version alignment. The platform team fixed a data-pipeline bug but did not notify the annotation team, causing all annotation tasks to become invalid two days later. The legal review flagged a batch of data with copyright risk, but no process exists to pass that information back to the data-filtering stage.

This is not an extreme case—it is the normal state of LLM data projects after they scale up. The root of the problem is not that any individual lacks effort; it is the absence of a shared organizational language: no clear role boundaries, no unified delivery interfaces, no predictable collaboration cadence.

The DataOps flywheel exists precisely to fill this gap in organizational language. It draws on the continuous-delivery idea of "placing changes into a repeatable, verifiable pipeline," and absorbs the SRE emphasis on service levels, incident response, and post-mortem mechanisms (Humble and Farley 2010; Beyer et al. 2016).

---

## 24.1 Why LLM Data Teams Need a New Organizational Form

### 24.1.1 Structural Limitations of Traditional Data Teams

The design philosophy of traditional data engineering teams was derived from the division-of-labor model of the data-warehouse and BI era. Traditional data management bodies of knowledge emphasize functional divisions such as metadata, quality, security, and master data, but in machine learning systems the tight coupling between data and models places new pressure on this linear division of labor (DAMA International 2017; Amershi et al. 2019). In that era, the core task of data engineering was to "move business data stably and accurately into analytical systems," with clear role boundaries: data engineers handled ETL, analysts fetched data, and BI developers built reports. Data quality problems were typically discovered on the business side and then fed back to the data-warehouse side for remediation. This model worked in most scenarios because the needs of data producers (business systems) and consumers (analysts) were relatively stable.

LLM data engineering breaks that assumption. MLOps research typically treats model development, data management, training, deployment, monitoring, and feedback as a continuous lifecycle rather than isolated phases (Kreuzberger, Kühl and Hirschl 2023). Training-data requirements come from multiple rapidly changing sources: algorithm-team experiment results alter data-mix requirements; product-team version iterations introduce new data categories; the RLHF phase demands human-preference annotations; RAG scenarios require continuous updates to enterprise knowledge bases. Data is no longer a passive "object to be transported" but an active "experimental variable." This requires data teams to be not merely "pipeline maintainers" but "active stewards of data assets."

Under this new role, three limitations of traditional team structures surface rapidly. Technical debt in production machine learning systems typically originates not in model code itself but in data dependencies, implicit feedback loops, configuration drift, and unclear team boundaries (Sculley et al. 2015).

The first limitation is **overlapping responsibilities and ambiguous interfaces**. LLM data projects inherently involve multiple cross-functional roles: data engineers, annotation engineers, quality evaluators, algorithm engineers, product managers, and legal compliance specialists. The work outputs of these roles are highly interdependent, yet no standard interface definitions exist. Between "the data engineer completes cleaning" and "the annotation engineer begins annotation," what acceptance criteria should apply? Who confirms completion? If a quality problem arises, who is responsible for tracing it back? These questions do not arise in traditional data-warehouse projects, yet they arise daily in LLM data projects.

The second limitation is **single-expert dependency**. Many teams concentrate key decisions in the hands of a few "big-picture" individuals. This works when the team is small, but when the project scales—data volume doubles, annotation tasks are split among three outsourcing vendors, five experimental versions run simultaneously—the single expert becomes the bottleneck. More dangerously, the experience of single experts cannot be systematically preserved; once key personnel leave, the team falls into an "the data pipeline runs but nobody knows why" state of inexplicability.

The third limitation is **the absence of a continuous-delivery cadence**. Most data teams have no fixed release cadence; data delivery timing depends on task completion rather than a predefined release schedule. This prevents the algorithm team from planning experiment windows, the product team from predicting iteration cycles, and the platform team from basing capacity planning on any reliable forecast. Data engineering devolves into a "demand-push" mode of work: push once and get a batch, push again and get another batch—no internally driven delivery rhythm.

From a management perspective, these three limitations are not independent. Ambiguous interfaces amplify single-expert dependency, because only a few people know the implicit agreements between upstream and downstream; single-expert dependency in turn disrupts delivery cadence, because critical nodes must wait for particular individuals to make judgments; unstable delivery cadence further undermines organizational learning, because the team is always handling immediate incidents and rarely has the opportunity to distill experience into standard processes. In other words, the problem with traditional data teams is not simply "too few people" or "too few tools"—it is that the organizational system has not formed a repeatable, observable, governable operating mechanism.

This problem is especially pronounced in LLM data projects. Model training requires extensive data iteration, but each iteration's results may redirect the next round of data production. Without a stable organizational structure, a data team will constantly context-switch among algorithm experiments, annotation operations, quality reviews, and compliance approvals, ultimately creating a high-effort but low-accumulation work state. The team appears busy, yet large amounts of time are consumed by repeated explanations, repeated confirmations, and repeated remediation, with no accumulation of long-term capability.

Therefore, the first step in DataOps organizational design is not to immediately procure tools or build platforms, but to identify friction points in the existing organization. Friction points are usually hidden in daily work—for example: "Is this field mandatory?", "Can this sample batch be used for online evaluation?", "Does a change to the annotation standard by an outsourcing vendor require approval?", "Is it permissible for the algorithm team to access unvalidated data on an ad-hoc basis?" Each of these questions looks minor in isolation, but without unified rules they become hidden costs that continuously drain the team's attention. Table 24-1 shows the five structural limitations of traditional data teams, their typical manifestations, and corresponding DataOps improvement directions.

| Structural Limitation | Typical Manifestation | Impact on LLM Data Projects | DataOps Improvement Direction |
|---|---|---|---|
| Overlapping responsibilities | Multiple roles can modify data, but no single ultimate owner exists | Quality problems are difficult to trace after the fact; remediation actions are duplicated | Establish RACI matrices and a data-ownership mechanism |
| Ambiguous interfaces | Deliverables lack schema, acceptance criteria, and SLAs | Downstream rework is repeated; algorithm experiment scheduling is unstable | Establish role interface protocols and versioned contracts |
| Single-point dependency | Critical processes depend on the judgment of a small number of senior personnel | The organization becomes brittle when key personnel take leave, resign, or when projects run in parallel | Distill experience into templates, checklists, and post-mortem documents |
| Missing cadence | Data batches are delivered ad hoc in response to demand | Requirement priorities change frequently; the team is perpetually in reactive mode | Establish weekly flywheels, monthly reviews, and version freezes |
| Invisible quality | Spot checks are done only after a problem appears | Data problems that enter the training pipeline incur sharply escalating costs | Establish quality metrics, spot-check procedures, and automated validation |

*Table 24-1: Traditional Data Team Limitations and DataOps Improvement Directions*

It is also worth noting that adjusting organizational structure changes how team members understand responsibility. In traditional teams, many problems can be resolved by "asking someone who knows"; in a DataOps organization, problems are expected to be resolved through institutionalized interfaces wherever possible. This does not mean reducing communication—rather, it means shifting communication from ad hoc coordination toward collaboration grounded in evidence, documentation, and metrics. For LLM data teams, this transition can significantly reduce knowledge loss and allow newcomers, outsourcing teams, and cross-departmental collaborators to enter the same working context much faster.

### 24.1.2 New Collaboration Requirements in Large-Model Projects

LLM data engineering introduces several categories of new requirements that traditional teams have never encountered, requirements that fundamentally alter the logic of organizational design.

**Requirement 1: Tight-coupled iteration between data and models.** In traditional ML projects, data teams and algorithm teams can work relatively independently—the data team provides a stable dataset and the algorithm team runs experiments on it. In LLM projects, however, algorithm-experiment results directly feed back into data requirements: an evaluation revealing that the model performs poorly on "mathematical reasoning" requires the data team to immediately augment training samples in that area; an RLHF experiment showing that a certain response style is poorly received requires the annotation team to revise scoring criteria. This demands that data teams and algorithm teams work not "serially" but "in parallel," with a rapid requirement-response mechanism. Industrial ML practice research demonstrates that ML system development typically requires continuous collaboration among software engineering, data engineering, product, and research roles, rather than treating model delivery as the internal work of a single team (Amershi et al. 2019).

**Requirement 2: Coordinated governance of multi-source heterogeneous data.** LLM data comes from many sources: web crawling, human annotation, synthetic generation, third-party datasets, and user feedback. Each source has different quality standards, compliance requirements, and update frequencies. Without a unified governance framework, teams fall into a fragmented "everyone manages their own" state, unable to ensure that data entering the training corpus is globally consistent.

**Requirement 3: Continuous monitoring of annotation quality.** The quality of human annotation drifts over time: annotator fatigue affects consistency, unclear task boundaries lead to diverging standards, and differences in understanding among outsourcing vendors introduce systematic bias. This requires a continuously operating quality-monitoring system, not a "one-time spot check before going live."  Data-cascade research shows that early-stage data quality problems are repeatedly amplified in downstream models, products, and organizational processes, making it essential to treat data work as a first-class engineering object (Sambasivan et al. 2021).

**Requirement 4: Full-lifecycle involvement of compliance and security.** Copyright issues in data sources, privacy protection for user data, compliance requirements for cross-border data transfer—these legal risks cannot be addressed only before data release; they must be managed starting at the data collection and cleaning stages. This requires legal compliance roles to genuinely participate throughout the entire data engineering lifecycle, not merely to sign off at the end.

### 24.1.3 Design Principles for New Organizational Forms

Based on the foregoing analysis, the organizational design of LLM data teams should follow these core principles:

**Principle 1: Interfaces over hierarchy.** Traditional organizational design emphasizes reporting relationships. New data team design should focus more on the delivery interfaces between roles—who produces what, in what format, and what are the consumer's acceptance criteria. Clear interfaces do more to ensure smooth multi-team collaboration than strict reporting hierarchies. Team Topologies theory likewise emphasizes that organizational design should center on fast flow and clear team interaction patterns, not merely on static reporting relationships (Skelton and Pais 2019).

**Principle 2: Combine asynchronous cadence with synchronous checkpoints.** Much of data engineering work can be performed asynchronously (annotation tasks, data processing), but critical decisions require synchronous alignment (data-version freezes, quality-threshold adjustments). Organizational design should provide sufficient autonomy for asynchronous work while ensuring global alignment through fixed synchronous checkpoints (weekly meetings, monthly reviews, milestone reviews).

**Principle 3: Institutionalize knowledge.** Reducing dependence on single experts means converting tacit knowledge into explicit processes. Every data-problem investigation, every quality-standard adjustment, every cross-team conflict resolution should produce reusable process documentation—not simply reside in someone's memory.

**Principle 4: Small teams, large platforms.** Scaling data engineering should not be achieved simply by adding headcount, but by using platform tools to amplify individual productivity. Annotation management platforms, data quality assessment systems, experiment tracking tools—the return on these platform investments typically exceeds the return on equivalent headcount. Production-grade ML platform practice shows that unified pipeline, metadata, data-validation, and model-serving capabilities can substantially reduce duplicate engineering costs and improve delivery consistency (Baylor et al. 2017; Reis and Housley 2022).

These principles manifest differently at different organizational stages. Early-stage teams primarily need to resolve "who does what and how is it delivered"; mid-stage teams need to resolve "how to reliably reuse and reduce rework"; mature teams need to resolve "how to measure organizational capability and how to embed experience into platforms." Ignoring stage differences leads to two common mistakes: introducing overly heavy governance early, slowing exploration speed; or remaining dependent on verbal coordination in maturity, causing risk to spiral out of control during scale-up.

The new organizational form can therefore be understood as a progressive maturation process. Stage 1 is **scripted collaboration**, whose core objective is enabling a small number of people to get things done. Stage 2 is **standardized collaboration**, whose core objective is enabling different roles to deliver according to unified interfaces. Stage 3 is **platform-driven collaboration**, whose core objective is having processes, permissions, quality, and version management carried by systems. Stage 4 is **metrics-driven collaboration**, whose core objective is continuously optimizing organizational capability through metrics and post-mortems. The four stages are not simple replacements of one another but a progressive layering of capabilities. Table 24-2 shows the organizational characteristics, primary risks, and priority capability-building activities at each maturity stage.

| Maturity Stage | Organizational Characteristics | Primary Risks | Priority Capability Building | What to Avoid Investing in Too Early |
|---|---|---|---|---|
| Scripted collaboration | A small number of members rely on personal experience to complete collection, cleaning, and annotation | Single-point dependency; results are not reproducible | Basic version records, minimal quality checks, key-role confirmation | Complex approvals, excessive metrics, heavyweight platforms |
| Standardized collaboration | Multiple roles deliver according to interfaces; a fixed cadence begins to emerge | Inconsistent standard execution; chaotic interface changes | RACI, schemas, annotation guides, SLAs | Complex orchestration before large-scale automation |
| Platform-driven collaboration | Tasks, annotation, quality, and versioning enter a unified system | Platform diverges from real workflows; high tool-adoption cost | Embedded workflows, automated checks, access control, lineage tracking | Feature accumulation unrelated to core workflows |
| Metrics-driven collaboration | Continuous improvement driven by metrics, post-mortems, and experiment feedback | Metric gaming, local optimization, formalistic post-mortems | Value-stream measurement, quality trends, reuse benefits, organizational learning | Metric competitions disconnected from business objectives |

*Table 24-2: DataOps Maturity Stages for LLM Data Teams*

Organizational design must also handle the tension between centralization and embedding. A centralized data team benefits from unified standards, consolidated platform investments, and risk control, but may be distant from business contexts; embedded data staff are closer to algorithm and product needs but tend to form project silos. A practical approach is a hybrid "central platform + project embedding" model: the central team owns the general platform, data standards, quality rules, and compliance boundaries, while embedded personnel handle requirement clarification, contextual interpretation, and rapid feedback. This preserves standard consistency while preventing the data team from drifting away from specific model objectives.

In this hybrid model, the role of Data Owner is especially critical. The Data Owner is not merely an approver or a project manager, but the nexus connecting data strategy, model objectives, and organizational resources. Their work includes judging which data merits long-term preservation, which requirements should be treated as one-off experiments, which quality problems should be elevated to platform rules, and which shared assets should enter an enterprise-level catalog. An absent Data Owner causes the team to become reactive; an overly centralized Data Owner suppresses role autonomy. DataOps therefore needs to provide structural support for Data Owners through RACI matrices, regular meetings, and metric systems. Figure 24-1 illustrates the four-stage evolution path of LLM data teams from "single-person scripts" to "platform-driven DataOps."

![Figure 24-1: LLM Data Team Organizational Evolution Path](../../images/part8/图24_1zh.png)

*Figure 24-1: The four-stage evolution path of LLM data teams from "single-person scripts" to "platform-driven DataOps"*

---

## 24.2 Role Division, Interfaces, and RACI Design

### 24.2.1 Core Role Map of an LLM Data Team

A complete LLM data engineering team typically contains the following seven core role categories. In small teams, one person may fill multiple roles; in large teams, each role category may constitute a full sub-team. The key is not the number of personnel but the clarity of responsibility boundaries.

**Data Owner** is the ultimate responsible party for data assets, accountable for setting data strategy, approving dataset releases, and resolving cross-team data-resource disputes. The Data Owner is typically the technical lead or product lead of the data team, with the authority to make final decisions on "can this data batch enter the training corpus."

**Data Engineer** is responsible for data collection, cleaning, format conversion, and pipeline maintenance. This role is the core builder of the data pipeline, responsible for the technical implementation of data flows, but not primarily accountable for judgments about content quality.

**Annotation Engineer / Annotation Ops** is responsible for annotation task design, quality control, and annotator management. Annotation engineers need to understand both business requirements (what constitutes good data) and engineering constraints (how to complete annotation efficiently), making them a key role connecting business and platform.

**Quality Evaluator** is focused on objective assessment of data quality across dimensions such as consistency, accuracy, and coverage. Quality evaluators do not participate in data production; they perform independent reviews.

**Algorithm Engineer** is the primary consumer of data. Algorithm engineers need to clearly articulate data requirements (what type, what distribution, what format) and feed experiment results back to the data team (which categories of data are effective, which need augmentation).

**Platform Engineer** is responsible for building and maintaining annotation tools, data-pipeline platforms, and storage infrastructure. The platform engineer's deliverable is "system operational stability," not the data content itself.

**Legal / Compliance Specialist** is responsible for data-source compliance review, privacy-protection audits, and copyright-risk assessment. Legal compliance specialists cannot appear only at the end of a project; they must engage during the data-collection phase to establish compliance boundaries.

These seven role categories do not mean that all companies must establish seven separate positions. For startups or early-stage exploratory projects, one person filling multiple roles is common; for large enterprises, a single role may be split into multiple sub-teams. For example, Data Engineer may be further subdivided into Collection Engineer, Cleaning Engineer, Data Platform Engineer, and Data Reliability Engineer; Quality Evaluator may be further divided into automated-rule maintenance, manual spot-checking, model-evaluation interfacing, and data-bias analysis. The organizational design emphasis is not on achieving a complete set of job titles, but on ensuring that every category of responsibility has someone who performs it, someone who is accountable for it, and verifiable delivery interfaces to upstream and downstream.

In practice, the most commonly overlooked role is "Algorithm Engineer as data consumer." Many teams treat algorithm engineers solely as requirement-submitters, defaulting to the assumption that they need only state "we need more high-quality data." In a DataOps system, however, algorithm engineers also bear responsibility as feedback providers: they must translate experiment results into actionable data requirements—for example, explaining in which task types, which distribution ranges, which prompt patterns, or which user scenarios error samples are concentrated. Without this fine-grained feedback, the data team can only expand data based on intuition, making it impossible to close the loop between data improvements and model performance.

Similarly, the legal compliance role should not be understood as an end-of-pipeline approver. For LLM data projects, compliance judgments often directly affect data-collection scope, desensitization strategy, retention periods, and authorization approaches. If compliance requirements are discovered only after data production is complete, the team may need to discard data that has already been cleaned and annotated, incurring substantial waste. More mature teams therefore front-load compliance requirements as data-admission rules and incorporate them into collection-task templates, dataset specification documents, and quality-check checklists. In this way, compliance is no longer merely an approval action but becomes part of the data-production system. Table 24-3 summarizes the core responsibilities, key inputs and outputs, and common failure modes for each of the seven core roles.

| Role | Core Responsibilities | Key Inputs | Key Outputs | Common Failure Modes |
|---|---|---|---|---|
| Data Owner | Data strategy, prioritization, final release decisions | Business objectives, model roadmap, risk reports | Data-version decisions, resource allocation, conflict arbitration | Approves but does not steward; lacks long-term attention to quality and reuse |
| Data Engineer | Collection, cleaning, transformation, lineage maintenance | Data sources, schemas, collection authorization | Traceable data batches, processing scripts, quality summaries | Focuses only on pipeline validation; neglects content quality and downstream interpretability |
| Annotation Engineer / Annotation Ops | Annotation scheme, task dispatch, annotator management | Annotation guides, sample pool, quality standards | Annotated data, annotation logs, consistency reports | Standards communicated verbally, leading to inconsistent understanding among outsourcing vendors |
| Quality Evaluator | Independent spot-checking, quality measurement, issue attribution | Sample batches, evaluation rules, historical issues | Quality reports, issue severity grading, remediation recommendations | Performs only final acceptance checks; lacks ongoing observation of in-process quality |
| Algorithm Engineer | Data-requirement articulation, experiment feedback, performance validation | Model experiment results, error samples, evaluation sets | Data improvement recommendations, experiment conclusions, production-risk feedback | Requirements expressed too vaguely to be actionable by the data team |
| Platform Engineer | Toolchain, access control, stability, automation | Process requirements, capacity plans, security requirements | Annotation platform, data pipelines, monitoring and alerting | Tools disconnected from actual workflows; platform features difficult for the team to adopt |
| Legal / Compliance Specialist | Data sources, privacy, copyright, and usage boundaries | Collection plans, data samples, business scenarios | Compliance opinions, risk levels, usage restrictions | Engaged too late, causing already-produced data to be unusable |

*Table 24-3: Extended Description of LLM Data Team Role Responsibilities*

### 24.2.2 Role Interface Protocol Design

Clear interface protocols are the foundation of smooth multi-team collaboration. Between every pair of collaborating roles, the following four elements should be explicitly defined: **Deliverable**, **Format Specification**, **Acceptance Criteria**, and **Delivery SLA**.

Take the interface between Data Engineer and Annotation Engineer as an example. Table 24-4 shows its deliverable, format specification, acceptance criteria, and delivery SLA:

| Element | Content |
|---|---|
| Deliverable | Cleaned raw sample pool; list of files awaiting annotation |
| Format Specification | JSONL format; each line contains `id`, `content`, `source`, `clean_timestamp` fields |
| Acceptance Criteria | Duplication rate < 1%; blank rate < 0.5%; character-anomaly rate < 0.1% |
| Delivery SLA | Current-week batch delivered by 18:00 every Friday |

*Table 24-4: "Data Engineer – Annotation Engineer" Interface Example*

Take the interface between Annotation Engineer and Algorithm Engineer as an example. Table 24-5 shows its interface elements:

| Element | Content |
|---|---|
| Deliverable | Completed annotated training set, including annotation metadata |
| Format Specification | JSONL format; contains `instruction`, `output`, `annotator_id`, `confidence`, `revision_count` |
| Acceptance Criteria | Inter-annotator agreement (IAA) > 0.85; spot-check pass rate > 95% |
| Delivery SLA | Completed 5 business days before iteration-version release |

*Table 24-5: "Annotation Engineer – Algorithm Engineer" Interface Example*

Once an interface protocol is established, it should be written into the team's internal documentation and the corresponding schema files maintained in a version-control system. Any interface change must be communicated to downstream roles in advance, with a transition period specified.

The value of interface protocols lies in transforming "I assumed you would deliver it this way" into "we have jointly confirmed that deliverables must satisfy these conditions." When a data team is small, verbal agreements may seem more efficient, but as the team grows, verbal agreements rapidly degrade into inconsistent individual interpretations. In LLM data engineering in particular, data fields, annotation labels, filtering rules, and quality metrics may all change frequently; without versioned interface protocols, a small upstream change can render downstream experiment results inexplicable.

A mature interface protocol should contain at least three layers of information. The first is the **structural layer**: field names, data types, required-field constraints, enumeration values, and handling of missing values. The second is the **semantic layer**: the business meaning of each field, boundary conditions, and representative examples. The third is the **operational layer**: delivery frequency, acceptance method, anomaly-feedback path, and change-notification mechanism. Many teams write only the structural layer and neglect the semantic and operational layers, causing schemas to appear consistent while the actual data semantics have already drifted. For example, the field `source` may carry entirely different meanings across projects: some teams use it to denote the data-collection website, others the business system, and still others the annotation-task origin. Without semantic-boundary documentation in the interface protocol, downstream algorithm teams may draw erroneous conclusions during data-stratified evaluation. Interface protocols in DataOps should therefore be not merely technical schemas, but contractual documents oriented toward cross-role collaboration. Table 24-6 shows the content and verification methods for the structural, semantic, operational, and change layers of an interface protocol.

| Interface Layer | Questions to Address | Recommended Medium | Verification Method |
|---|---|---|---|
| Structural layer | Do fields exist? Are types correct? Are enumerations compliant? | JSON Schema, Avro Schema, database table definitions | Automated validation, CI checks |
| Semantic layer | Field meanings, business boundaries, edge-case examples, applicable scope | Data dictionary, dataset specification, annotation guide | Manual review, example audit |
| Operational layer | Who delivers, when, how is acceptance conducted, how is feedback handled | SLA document, interface protocol, RACI matrix | Weekly post-mortem, service-metric monitoring |
| Change layer | Reason for change, impact scope, compatibility strategy, deprecation timeline | Change log, version notes, migration guide | Change review, downstream confirmation |

*Table 24-6: Four-Layer Structure of Data Role Interface Protocols*

Interface protocols should also specify a "compatibility period." When field semantics or data formats change, downstream parties cannot simply be required to adapt immediately; instead, the protocol must clarify when the old version stops being supported, how the new version will be validated, and whether historical data requires backfilling. For training data, the compatibility period affects not only engineering pipelines but also experiment comparability. If a data batch enters the training corpus after a field-definition change, and the experiment record does not note this, subsequent changes in model performance may be misattributed to algorithmic improvement or regression.

From an organizational management perspective, maintenance responsibilities for interface protocols also need to be explicit. Typically, data engineers maintain the structural layer; annotation engineers or domain experts maintain the semantic layer; the Data Owner maintains the operational and change layers; and quality evaluators are responsible for checking whether protocols are actually followed. In this way, interface protocols are no longer static documents but become the baseline for the team's daily collaboration.

### 24.2.3 RACI Responsibility Matrix

The RACI model (Responsible, Accountable, Consulted, Informed) is a classic responsibility-assignment tool. Responsibility assignment matrices are commonly used in project management to clarify execution, accountability, consultation, and notification relationships, reducing responsibility vacuums in cross-role collaboration (Project Management Institute 2021). For LLM data projects, Table 24-7 provides a RACI matrix covering major decision items:

**R (Responsible)**: The actual executor—the person who performs the concrete work.
**A (Accountable)**: The ultimate accountable party—the person responsible for the outcome and holding decision authority. Only one A per item.
**C (Consulted)**: The party consulted—whose input must be sought before a decision is made.
**I (Informed)**: The party notified—who must be informed of the outcome after a decision is made.

| Item | Data Owner | Data Engineer | Annotation Engineer | Quality Evaluator | Algorithm Engineer | Platform Engineer | Legal / Compliance |
|---|---|---|---|---|---|---|---|
| Data requirement definition | A | C | C | I | R | I | C |
| Data collection and cleaning | I | R/A | C | I | I | C | C |
| Annotation task design | C | I | R/A | C | C | I | C |
| Annotation quality review | I | I | C | R/A | C | I | I |
| Training-set version release | A | C | C | C | R | I | C |
| Data compliance review | A | I | I | I | I | I | R |
| Platform incident response | I | C | I | I | I | R/A | I |
| Data version rollback | A | R | C | C | C | C | I |
| Cross-team data-sharing approval | A | C | I | I | C | I | C |
| Outsourcing vendor management | I | I | R/A | C | I | I | I |

*Table 24-7: RACI Responsibility Matrix for LLM Data Teams*

Several points require special attention when using this RACI matrix:

First, each row must have exactly one A. If two people are "jointly responsible" for an item, it effectively means nobody is responsible.

Second, when R and A reside in the same person (e.g., Annotation Engineer is R/A for annotation task design), that person both executes and is accountable; an external quality-review mechanism must be in place to provide checks and balances.

Third, C is not "casually asking someone"—it means formally soliciting input before a decision, giving the consulted party sufficient time to respond. If the consulted party cannot respond in time, that situation should be recorded, not simply skipped.

Two common deviations occur when RACI matrices are put into practice. The first is treating the RACI as an administrative division-of-labor table, filled out once at project kickoff and never updated. In reality, task types in LLM data projects change with project phases—early phases may focus on collection and cleaning, mid-phases shift toward annotation quality and experiment feedback, and later phases concentrate on version freezing, compliance auditing, and online feedback loops. The RACI matrix should therefore be reviewed at least once after each major version cycle to confirm that the current responsibility configuration still matches actual workflows.

The second deviation is setting A too high. Many organizations habitually place ultimate accountability with department heads; but if a head is too far from the concrete work, accountability becomes a formalistic approval. The correct approach is to keep A as close as possible to the decision context while ensuring the person has sufficient authority. For example, the A for annotation quality review can be the quality evaluation lead, not the data department director; the A for cross-project data-reuse approval should be the Data Owner, because such matters involve resource allocation and risk boundaries.

To give RACI real traction, teams can bind it to daily systems. For example, in the requirement-management tool, every data task must specify Responsible and Accountable; in the annotation platform, annotation-guide changes must record who was Consulted; on the data-version release page, all Informed parties should receive automatic notifications. In this way, RACI no longer depends on human memory but is embedded in workflows.

| Application Scenario | Specific Use of RACI | Management Benefit | Problems to Avoid |
|---|---|---|---|
| Data requirement initiation | Clarify who raises the requirement, who confirms priority, who assesses resources | Reduce requirement queue-jumping and verbal commitments | Avoid requiring all requirements to be approved by the highest-level accountable party |
| Annotation standard changes | Clarify who drafts, who reviews, who notifies outsourcing vendors | Reduce annotation criterion drift | Avoid announcing changes only verbally in meetings |
| Data-version release | Clarify the release owner, quality confirmation party, and notification targets | Improve version traceability | Avoid completing documentation only after release |
| Quality incident handling | Clarify the remediation executor, root-cause analyst, and final arbiter | Shorten localization time | Avoid attributing incidents simplistically to individual mistakes |
| Compliance-risk handling | Clarify responsibility for risk identification, suspension of use, and resumption of use | Reduce legal and reputational risk | Avoid compliance opinions being disconnected from engineering actions |

*Table 24-8: Applications of RACI in Everyday DataOps Scenarios*

### 24.2.4 Escalation Paths and Exception Handling

No organizational design can cover every situation. Clear escalation paths should be predefined for the following scenarios:

**Emergency escalation**: When a decision must be made within the normal process timeline but cannot proceed through normal channels (e.g., a data-quality crisis before a critical milestone), an emergency decision process should exist designating an on-call Data Owner with authority to bypass standard approvals and make immediate decisions.

**Cross-RACI boundary conflicts**: When two teams dispute ownership of a RACI item (e.g., "Is this data-quality problem the data engineer's responsibility or the annotation engineer's responsibility?"), the matter should be escalated to the Data Owner for final arbitration rather than allowing the two teams to persist in their respective positions.

**Exception approval process**: For situations that do not meet standard data criteria but have special requirements (e.g., the algorithm team urgently needs a batch of raw data that has not gone through the normal cleaning process for an experiment), an exception approval form should record the exception reason, the approver, and the isolation measures for the excepted data.

The core principle of exception-handling mechanisms is "allow exceptions, but prevent exceptions from becoming the norm." In LLM projects, exploratory experiments frequently require incomplete, unstable, or insufficiently validated data. Completely prohibiting such data from entering experimental workflows would deprive the team of rapid-exploration capability; allowing it unconditionally would compromise data-governance boundaries. Exception approvals should therefore make risks explicit: who may use this data, for what purposes only, whether it may enter the formal training corpus, and when it must be deleted or re-evaluated.

In practice, exceptions can be classified into three types. The first is **time exceptions**: to meet a critical experiment window, data may be used before the quality report is fully complete. The second is **quality exceptions**: certain exploratory data may fall below the formal training-set quality standard, but must be labeled as experimental data. The third is **compliance exceptions**: these require the greatest caution and should typically be allowed only in isolated environments for preliminary analysis, with mandatory involvement of the legal compliance specialist in approval. Different exception types should not carry the same approval intensity; otherwise, the team will bear excessive process costs on low-risk items while potentially approving high-risk items insufficiently.

| Exception Type | Applicable Scenarios | Required Records | Approval Requirements | Upon Expiry |
|---|---|---|---|---|
| Time exception | Critical experiment window approaching; full acceptance not yet complete | Batch used, outstanding check items, risk statement | Data Owner or on-call lead approval | Complete acceptance; confirm whether to promote to official status |
| Quality exception | Coarsely labeled data, weakly supervised data, or exploratory samples | Quality gaps, usage restrictions, isolation labels | Data Owner and algorithm lead jointly confirm | Non-conforming samples may not enter official versions |
| Format exception | Downstream temporarily needs a non-standard format for analysis | Field differences, conversion scripts, compatibility period | Data engineering lead approval | Complete format migration or retire the temporary format |
| Compliance exception | Data source or authorization boundary requires further determination | Source description, risk level, access list | Legal compliance specialist and Data Owner approval | Access permissions automatically revoked upon expiry |

*Table 24-9: DataOps Exception Approval Types and Handling Requirements*

Escalation paths should also include post-mortem requirements. Any item that triggers an emergency escalation or exception approval should be revisited at the next weekly post-mortem or monthly review: Was the exception reasonable? Did it expose a process-design flaw? Does the SLA or quality threshold need adjustment? Exceptions without post-mortems gradually erode institutional boundaries, while exceptions with post-mortems help the team identify gaps between actual workflows and institutional design.

---

## 24.3 The DataOps Flywheel and Weekly Cadence

### 24.3.1 What Is the DataOps Flywheel

The DataOps flywheel is a conceptual model describing the continuous improvement loop of a data team. The DataOps Manifesto—an online statement of principles—emphasizes customer collaboration, rapid feedback, repeatable processes, team collaboration, and the elimination of heroism; together, these principles sustain flywheel-style continuous improvement (DataOps Manifesto, accessed 2026). Its core idea is: through a fixed operational cadence, data requirements, data production, quality evaluation, and iterative feedback are chained into a self-reinforcing loop—each revolution of the loop more efficient and higher quality than the previous one.

![Figure 24-2: DataOps Team Organization Overview](../../images/part8/图24_2zh.png)

*Figure 24-2: LLM Data Team DataOps Overview—An Integrated View of Roles, Interfaces, Flywheel, and Governance*

In LLM data projects, the flywheel is driven by four core "pools":

**Demand Pool**: Aggregates data requirements from algorithm, product, business, and other stakeholders; sorts them by priority into an executable task list.

**Data Pool**: Currently available data assets, including cleaned raw data, completed annotated training samples, and compliance-approved datasets. The state of the Data Pool determines what data can be used in the current iteration.

**Experiment Pool**: Records of experiments currently running in the algorithm team, including the data versions used, parameter configurations, and evaluation results. The Experiment Pool is the ultimate source of feedback on data quality.

It is important to emphasize that the Experiment Pool must record not only data, models, metrics, and conclusions but also the runtime context needed to reproduce experiments. For LLM data experiments, this should include at minimum `runtime_env`, `container_image`, `dependency_lock`, `hardware_profile`, `cuda_driver`, `random_seed`, and `determinism_flags`. Here, `runtime_env` describes the Python, OS, and training framework versions; `container_image` locks the container image; `dependency_lock` records the dependency lock file or package-version snapshot; `hardware_profile` describes GPU/accelerator model, count, and VRAM; `cuda_driver` records the CUDA, driver, and communication-library versions; and `random_seed` together with `determinism_flags` describes randomness control and deterministic training settings. Without these fields, even when the data version and model parameters are complete, the team may be unable to reproduce experiment results weeks later.

**Issue Pool**: A list of discovered data problems, including quality issues, compliance risks, and pipeline failures. Issues in the Issue Pool must be systematically triaged and addressed, not patched ad hoc.

The operating logic of the flywheel is: the Demand Pool generates tasks → tasks drive data production, updating the Data Pool → the algorithm team draws data from the Data Pool to run experiments, whose results enter the Experiment Pool → problems discovered in experiment results enter the Issue Pool → once issues are resolved, the updated data re-enters the Data Pool while priority adjustments are made to the Demand Pool. This loop completes one revolution per week, and through continuous iteration the flywheel effect accumulates.

From a systems perspective, the four pools are not simple task lists but different analytical lenses on the information the team manages. The Demand Pool answers "why do this" and "what to do first"; the Data Pool answers "what assets are currently available"; the Experiment Pool answers "whether the data has genuinely improved the model"; the Issue Pool answers "where blockages and risks exist." If the four pools are not interconnected, the team will still fall into information silos. For example, the Issue Pool records that a certain sample type has low quality, but the Demand Pool does not adjust priority accordingly; the Experiment Pool finds that a certain data batch significantly improves model performance, but the Data Pool does not tag its high-value attribute. These disconnections weaken the flywheel's learning capacity.

Flywheel design should therefore emphasize traceability relationships between objects. A requirement should be traceable to its corresponding data batch; a data batch should be traceable to its corresponding experiment results; an experiment conclusion should be traceable to related issues and subsequent remediation actions. Only when these relationships are recorded can the team progress from "how many tasks did we complete this week" to "which data investments actually produced results." This is also what distinguishes DataOps from ordinary task management: DataOps focuses on value flow, not merely on workload.

| Flywheel Object | Key Fields | Relationships to Other Objects | Management Focus |
|---|---|---|---|
| Demand Pool | Requirement ID, requester, priority, target metrics, deadline | Linked to data tasks, experiment plans, and business objectives | Prevent requirement proliferation; ensure each requirement is actionable |
| Data Pool | Data version, source, scale, quality summary, compliance status | Linked to collection tasks, annotation batches, experiment records | Prevent data assets from becoming invisible and irreusable |
| Experiment Pool | Experiment ID, data version, model version, evaluation results, conclusions | Linked to requirement objectives, data versions, and error samples | Determine whether data investment produces model benefit |
| Issue Pool | Issue severity, discovery source, root cause, owner, remediation status | Linked to data batches, interface protocols, and post-mortem documents | Prevent recurring issues; drive process improvement |

*Table 24-10: Management Fields and Linkage Relationships of the Four DataOps Flywheel Pools*

Whether the flywheel can run continuously depends on whether feedback is sufficiently short and precise. If feedback takes too long, the data team cannot timely judge whether this week's work was effective; if feedback is too coarse, the team knows only that the model improved or degraded overall but has no idea how to adjust the data. Ideally, each iteration should produce feedback at three levels: task-level feedback indicating whether deliverables were completed on time; quality-level feedback indicating whether the data met standards; and outcome-level feedback indicating whether the data improved model performance. Only when all three types of feedback complement each other can the team avoid pursuing delivery speed at the expense of data value.

![Figure 24-3: DataOps Flywheel Four-Pool Coordination Diagram](../../images/part8/图24_3zh.png)

*Figure 24-3: DataOps Flywheel Operating Mechanism—The Coordinated Cycle of Demand Pool, Data Pool, Experiment Pool, and Issue Pool*

### 24.3.2 Weekly Cadence Design

For the flywheel to keep running, fixed time nodes are needed to drive it. Continuous delivery practice emphasizes fixed cadence, automated validation, and small-batch changes precisely to reduce delivery risk and improve feedback velocity (Humble and Farley 2010; Humble, Molesky and O'Reilly 2015). The following is a weekly cadence design suitable for a medium-sized LLM data team (10–30 people):

**Monday: Requirement Sync Meeting (30 minutes)**

- Participants: Data Owner, algorithm engineer representative, product manager
- Objective: Confirm the week's data-delivery targets, synchronize the latest data requirements from the algorithm side, and adjust Demand Pool priorities
- Deliverable: Current-week task list, clearly specifying the owner and deadline for each task

**Wednesday: Quality Inspection (Asynchronous)**

- Participants: Quality Evaluator, Annotation Engineers
- Objective: Spot-check annotation batches completed so far in the week; discovered problems are entered into the Issue Pool
- Deliverable: Quality inspection report (standardized template); Issue Pool update

**Friday: Delivery and Post-Mortem (45 minutes)**

- Participants: Entire data team
- Objective: Confirm the week's data delivery status; post-mortem on high-priority issues in the Issue Pool; preview next week's plan
- Deliverable: Weekly report (standardized template); preview of next week's task list

**Monthly: Milestone Review (2 hours)**

- Participants: Full data team + algorithm team representatives + product team representatives
- Objective: Review current-month data-quality trends; evaluate SLA achievement; adjust long-term data strategy
- Deliverable: Monthly data-quality report; draft OKRs for the following month

**Quarterly: Version Freeze and Post-Mortem**

- Objective: Freeze a stable version of the dataset as the quarter's "baseline dataset" for audit and retrospective purposes
- Deliverable: Quarterly dataset version specification; data lineage report

| Cadence | Time | Participants | Core Deliverable |
|---|---|---|---|
| Monday requirement sync | Every Monday 9:30, 30 min | Owner + Algorithm + Product | Current-week task list |
| Wednesday quality inspection | Every Wednesday (async) | Quality + Annotation | Inspection report |
| Friday delivery post-mortem | Every Friday 16:00, 45 min | Full data team | Weekly report + next week's plan |
| Monthly review | Last Friday of each month, 2 h | Full team + cross-team | Monthly report + next month OKRs |
| Quarterly version freeze | End of each quarter | Owner + Algorithm | Baseline dataset version |

*Table 24-11: DataOps Meeting Cadence and Deliverables*

The key to a meeting system lies not in the number of meetings but in ensuring that each synchronization node has clearly defined inputs and outputs. The input to Monday's requirement sync should be a Demand Pool that has already been preliminarily organized, not an open forum where everyone improvises ideas; the input to Wednesday's quality inspection should be automated check results and sampled examples, not ad hoc judgments by quality evaluators; the input to Friday's delivery post-mortem should be this week's data versions, Issue Pool changes, and SLA achievement records, not a simple report of "what was done." Without verifiable inputs, meetings degenerate into status updates; without clear outputs, they fail to advance the flywheel to the next revolution.

More mature teams further distinguish "decision meetings" from "learning meetings." Requirement syncs and version freezes are decision meetings, requiring clear prioritization, resource allocation, and responsibility assignment; quality inspections and incident post-mortems are learning meetings, requiring pattern identification, process updates, and knowledge preservation. The two types require different facilitation. Decision meetings emphasize clear boundaries and time control; learning meetings need adequate space to discuss root causes. Conflating the two types leads to decision meetings running too long, or post-mortems that produce only remediation conclusions without genuine learning.

| Node | Recommended Input | Recommended Output | Success Criteria | Common Deviation |
|---|---|---|---|---|
| Monday requirement sync | Demand Pool candidates, last week's incomplete tasks, latest algorithm-side conclusions | Current-week commitment list, priority ranking, resource adjustments | Are requirements actionable? Is responsibility clear? | Meeting becomes impromptu requirement collection |
| Wednesday quality inspection | Automated quality reports, spot-check samples, anomaly distributions | Issue severity grading, remediation recommendations, risk alerts | Are high-risk issues surfaced early? | Only checking pass rates without analyzing trends |
| Friday delivery post-mortem | Data versions, SLA records, Issue Pool changes | Next-week improvement items, process revisions, post-mortem themes | Are unresolved items clearly addressed? | Only reporting progress without discussing blockers |
| Monthly review | Quality trends, reuse rates, cost data, incident post-mortems | Monthly improvement plan, metric adjustments, resource recommendations | Is cross-team consensus formed? | Too many metrics; key focus is diluted |
| Quarterly version freeze | Candidate data versions, quality reports, compliance confirmation | Baseline version, lineage records, known limitations | Is the version reproducible and auditable? | Freeze is just a file copy without accompanying documentation |

*Table 24-12: Inputs, Outputs, and Common Deviations at DataOps Operational Nodes*

To prevent meeting overhead from becoming excessive, teams should also establish asynchronous mechanisms. Preliminary requirement evaluation, reading quality reports, and supplementing root-cause analysis can all be completed on a collaboration platform; meetings handle only matters requiring shared judgment or cross-role coordination. This maintains the synchronous cadence without consuming too much team time in meetings—especially important for distributed or geographically dispersed teams, where the time cost of synchronous meetings is often underestimated.

### 24.3.3 SLA Settings and Version-Freeze Mechanisms

An SLA (Service Level Agreement) is the data team's service commitment to its internal "customers" (algorithm and product teams). Reasonable SLA settings enable downstream teams to form reliable expectations about data delivery while giving the data team a clear work rhythm. Production machine learning system readiness assessments typically need to cover data, models, infrastructure, monitoring, and testing—not just offline metrics (Breck et al. 2017).

The following is a reference SLA framework:

| Data Type | Processing Deadline | Quality Target | Notes |
|---|---|---|---|
| Urgent-demand data (P0) | Cleaning within 24 h; annotation within 48 h | Spot-check pass rate > 90% | Requires Data Owner approval to trigger |
| Routine iteration data (P1) | Cleaning within 3 business days; annotation within 5 business days | IAA > 0.85; pass rate > 95% | Scheduled per weekly plan |
| Exploratory experimental data (P2) | Cleaning within 5 business days; partial annotation permitted | Coarse labeling; pass rate > 80% | Internal experiments only |
| Historical data remediation | Assessed by impact scope; maximum 2 weeks | Consistent with original version quality | Remediation log required |

*Table 24-13: SLA Framework Example*

The version-freeze mechanism refers to "snapshotting" the current state of the Data Pool at a predefined point in time (e.g., month-end or quarter-end) into an immutable baseline version. After freezing, subsequent data updates cannot modify the frozen version; changes can only occur in new versions. This mechanism ensures the reproducibility of algorithm experiments—it is always possible to trace back to "which data batch was used in that experiment." Experience with automated data-validation systems shows that continuously monitoring pattern drift, anomalous distributions, and training/serving skew helps teams detect data-pipeline problems earlier (Breck et al. 2019).

Version-freeze operating procedure:

1. Issue a freeze notice 3 days in advance to all data producers.
2. Within 24 hours before the freeze, Quality Evaluators complete the final spot check.
3. At the time of the freeze, generate a dataset snapshot, recording the version number, creation timestamp, quality summary, and data sources.
4. After the freeze, record known issues in this version in the Issue Pool for reference during the next version's remediation.

SLA should not be understood purely as a delivery deadline. For LLM data teams, SLA covers at least four dimensions: time, quality, availability, and traceability. The time dimension specifies when to deliver; the quality dimension specifies what standard the deliverable must meet; the availability dimension specifies whether the data can be stably read and used by downstream systems; the traceability dimension specifies whether data sources, processing steps, and approval records are complete. Specifying only time without quality encourages the team to deliver low-value data rapidly; specifying only quality without time prevents downstream teams from scheduling experiment windows.

SLA settings must also distinguish among different types of data tasks. Exploratory experimental data may allow lower quality thresholds, but must be clearly marked as ineligible for the formal training corpus; formal training data requires higher quality and compliance standards but may have longer delivery cycles; online-feedback data requires stricter privacy protection and access auditing. Placing all tasks under a single SLA causes high-risk tasks to receive insufficient review while excessively restricting low-risk exploratory tasks.

| SLA Dimension | Core Question | Example Metrics | Management Significance |
|---|---|---|---|
| Time | When is delivery or response completed? | P0 requirements delivered within 48 h; routine batches within 5 business days | Helps algorithm and product teams schedule experiment windows |
| Quality | Does the deliverable meet usage standards? | Spot-check pass rate, IAA, consistency, duplication rate, anomaly rate | Prevents low-quality data from entering the training pipeline |
| Availability | Can downstream reliably read and process the data? | Schema validation pass rate, pipeline success rate, read-failure rate | Reduces engineering integration costs |
| Traceability | Can sources and processing steps be explained? | Lineage completeness rate, version-record completeness rate, approval-record completeness rate | Supports auditing, rollback, and issue localization |
| Compliance | Are authorization and usage boundaries satisfied? | Authorization coverage rate, desensitization pass rate, access-audit coverage rate | Reduces legal, privacy, and reputational risk |

*Table 24-14: Multi-Dimensional SLA Design for DataOps*

The value of version freezing lies not merely in retaining a file snapshot but in preserving an interpretable organizational state. A proper dataset version description should include at minimum: version objective, data sources, sample size, cleaning rules, annotation standards, quality summary, compliance conclusions, known defects, applicable scenarios, and inapplicable scenarios. Without this documentation, a frozen version can only answer "which files were used at that time," not "why these files were suitable for use."

In practice, teams can adopt a three-tier management model: **candidate version → frozen version → archived version**. The candidate version is used for the current iteration and permits fixes and additions; the frozen version is used for formal experiments and reproducible evaluations and may not be directly modified; the archived version is used for long-term auditing and historical comparison, with stricter access permissions. Three-tier management balances iteration speed and stability, preventing teams from having to choose between frequent updates and strict reproducibility.

Version freezes should also be coupled with the Issue Pool. If a frozen version has known defects, they do not necessarily block the release, but they must be explicitly recorded with the impact scope described. For example, a version may have insufficient coverage of mathematical reasoning samples but may still serve as the baseline training corpus for a general-purpose dialogue model; a version may contain a small number of low-confidence annotation samples that have been isolated and labeled and will not participate in critical evaluations. Clearly documenting limitations is more aligned with engineering reality than pursuing a "defect-free version."

Beyond SLAs and version freezing, DataOps requires an appropriately sized metrics system. The goal of a metrics system is not to produce more reports but to help the team judge whether the flywheel is healthy. A common mistake is counting only output quantities—how many data rows were cleaned this week, how many annotation tasks were completed, how many issues were closed. Quantity metrics are necessary, but without quality, cycle-time, and outcome metrics, the team may pursue superficial throughput while ignoring whether the data genuinely improves model capability.

A reasonably comprehensive metrics system can be organized into four layers. The first layer is **flow efficiency**, focusing on the cycle time from requirement submission to delivery. The second layer is **quality stability**, focusing on whether data batches meet standards and whether issues recur. The third layer is **collaboration reliability**, focusing on whether interfaces, SLAs, and version records are followed. The fourth layer is **business and model outcomes**, focusing on whether data investment produces improvements in model metrics, user experience, or business results. Together, the four layers constitute the observation surface of the DataOps flywheel.

| Metric Layer | Example Metrics | Data Source | Interpretation | Risk of Misuse |
|---|---|---|---|---|
| Flow efficiency | Requirement delivery cycle, queue wait time, SLA achievement rate | Demand Pool, task system, version release records | Determine whether work is flowing smoothly | Pursuing speed only, sacrificing quality |
| Quality stability | Spot-check pass rate, IAA, duplication rate, anomaly rate, rework rate | Quality reports, annotation platform, automated check pipelines | Determine whether data is stable and usable | Looking only at averages, ignoring distributional variation |
| Collaboration reliability | Interface-change advance-notice rate, version-record completeness rate, issue closure time | Interface documents, Issue Pool, release system | Determine whether the team collaborates as agreed | Equating document completeness with genuine understanding |
| Model outcomes | Error-sample reduction rate, key evaluation improvement, data-gain contribution | Experiment Pool, evaluation system, error-analysis reports | Determine whether data investment is effective | Attributing all model improvements to data |
| Reuse value | Dataset reuse count, number of reusing projects, reduction in duplicate collection | Data asset catalog, access logs, project records | Determine whether data assets have accumulated as organizational capability | Encouraging boundless over-sharing |

*Table 24-15: Layered Metrics System for the DataOps Flywheel*

The metrics system should also distinguish leading indicators from lagging indicators. Leading indicators provide early warnings of risk—for example, growing Demand Pool backlog, interface changes not communicated in advance, rising quality anomaly rates. Lagging indicators reflect outcomes—for example, delayed deliveries, declining model performance, online incidents. Looking only at lagging indicators means the team can only respond after problems occur; looking only at leading indicators may generate excessive alerts and drain management attention. The appropriate approach is to use a small number of leading indicators to drive day-to-day management and use lagging indicators to verify whether improvements are effective.

When using metrics concretely, teams should guard against metric gaming. Any metric strongly tied to performance evaluation is susceptible to being optimized into a surface-level number. For example, if only annotation volume is evaluated, annotators may sacrifice quality of judgment; if only issue-closure time is evaluated, teams may split complex issues into smaller pieces or close them prematurely; if only reuse rate is evaluated, projects may reuse data that does not fit their scenarios. DataOps metrics should serve learning and improvement, not replace professional judgment.

A sound practice is to accompany each metric with explanatory documentation specifying its meaning, calculation methodology, applicable scope, and inapplicable scenarios. Quality evaluators can add an "Interpretation of Metrics" section to monthly reports explaining whether this month's changes were caused by process improvements, task structure changes, or data-source changes. In this way, metrics become not merely numbers but shared evidence for cross-team discussion.

| Metrics Governance Requirement | Specific Description | Example |
|---|---|---|
| Define the formula | The calculation method must be stable; changes must be recorded | Is the spot-check pass rate calculated by sample count or by batch count? |
| State boundaries | For which tasks is the metric applicable; for which is it not | IAA is more suitable for subjective annotation tasks; not suitable for pure format validation |
| Examine distributions | Look not only at the mean but also at variation across projects, sources, annotators, and question types | Overall pass rate is high, but one outsourcing vendor is consistently below average |
| Pair with examples | Important metric changes should be accompanied by representative samples | After the duplication rate falls, do semantic duplicates still exist? |
| Connect to actions | Each anomalous metric should correspond to an improvement action or observation plan | When rework rate rises, supplement boundary examples and provide training |

*Table 24-16: Basic Requirements for DataOps Metrics Governance*

---

## 24.4 Cross-Team Collaboration and Risk Governance

### 24.4.1 Conflicts and Coordination in Multi-Team Data Asset Sharing

When a company runs multiple LLM projects in parallel, data-asset sharing becomes a complex coordination problem. Different project teams may simultaneously compete for annotation resources, computing resources, and high-quality datasets. Common conflict scenarios include:

**Data-ownership conflicts**: Team A spent three months organizing a high-quality dialogue dataset; Team B wants to reuse it directly, but Team A worries that Team B's use will "taint" the dataset's reputation (e.g., by using it in scenarios that do not meet safety standards).

**Annotation-resource competition**: The company has only 20 annotators with specialized domain knowledge, but three projects simultaneously have urgent annotation requirements. How should annotation resources be allocated?

**Quality-standard conflicts**: Different projects have different quality standards (pre-training data prioritizes quantity; RLHF data prioritizes consistency). How can shared data infrastructure support differentiated quality requirements?

An effective mechanism for resolving these conflicts is to establish a **Data Asset Registry** that centrally registers all produced datasets, clearly specifying: the dataset's owning team, usage restrictions (which projects may use it, which may not), and the application process. Cross-project data sharing must go through the application process in the registry, not a direct "take it."  Data mesh and data-productization thinking both emphasize that cross-team sharing should be realized through clear ownership, contracts, and product interfaces—not through ad hoc copying (Dehghani 2022).

A data asset registry is not a simple file inventory but the governance entry point for internal data collaboration. Whether a dataset is suitable for sharing depends on multiple dimensions: Is data quality stable? Are source authorizations clear? Is field semantics interpretable? Is the update frequency predictable? Are usage restrictions clearly stated? Are there sensitive information or copyright risks? If the registry only records names and storage paths, consumers still cannot determine whether they can use the data, and providers cannot control usage boundaries.

Cross-team sharing should therefore follow a closed loop of "discoverable → applicable → trackable → revocable." Discoverable means consumers can find data assets by searching the registry. Applicable means users must explain intended use, scope, and duration before use. Trackable means the platform records access logs, download records, and version citations. Revocable means that when authorization expires or use-case changes occur, access permissions can be automatically revoked. Without all four elements, data sharing easily degrades into ad-hoc copying, creating new data silos and compliance risks.

| Governance Stage | Key Questions | Responsible Role | Recommended Mechanism |
|---|---|---|---|
| Data registration | Is the dataset formally included in the asset registry? | Data Owner, Data Engineer | Dataset specification, quality summary, lineage records |
| Usage application | Why does the consumer need the data, for how long, in what scenario? | Consumer team, Data Owner | Standard application form, use-case description, risk grading |
| Authorization approval | Does the request satisfy quality, compliance, and business boundaries? | Data Owner, Legal/Compliance Specialist | Tiered approvals, minimum-privilege principle, term controls |
| Usage monitoring | Is data being accessed and processed according to the approved use case? | Platform Engineer, Security/Compliance team | Access logs, anomaly alerts, download auditing |
| Reuse evaluation | Is data sharing producing business or model value? | Data Owner, Algorithm Engineer | Reuse rate, experiment benefits, cost-savings statistics |
| Permission revocation | Are access permissions promptly closed upon authorization expiry? | Platform Engineer, Data Owner | Automatic revocation upon expiry; exception renewal approvals |

*Table 24-17: Governance Closed Loop for Cross-Team Data Sharing*

At the organizational level, data sharing also raises incentive problems. Producing high-quality data requires investment in collection, cleaning, annotation, and validation, but teams that reuse this data often only see the result without bearing the production cost. If the organization does not recognize the provider's contribution, providers may lack motivation to share and may even prefer to withhold data to protect their own project interests. To avoid this, Data Owners can record in monthly or quarterly reviews the reuse count, reusing projects, cost savings, and model benefits of data assets, incorporating sharing contributions into team performance or platform-value assessments.

At the same time, data sharing should not be encouraged without limits. Some data, though sharable, is not suitable for broad reuse. For example, adversarial samples constructed specifically to address a model defect, if used indiscriminately in training by multiple teams, may pollute evaluations; some data containing user feedback may significantly increase privacy risk if desensitization and authorization are insufficient and the sharing scope is expanded. The goal of DataOps governance is not to let all data flow freely, but to let the right data flow within appropriate boundaries.

### 24.4.2 Knowledge Preservation and Post-Mortem Mechanisms

The most effective way to prevent organizational knowledge loss is to establish a mandatory knowledge-preservation process. Research on datasheets and model cards emphasizes writing data sources, usage boundaries, evaluation results, risks, and limitations into standardized documentation to reduce dependence on tacit knowledge held by individuals (Gebru et al. 2021; Mitchell et al. 2019). Every data incident, every quality-problem investigation, and every major revision to annotation standards should produce a structured post-mortem document containing the following elements:

- **Event description**: What happened; what was the scope of impact
- **Root-cause analysis**: Why it happened; was it a process problem, a tool problem, or a standards problem
- **Remediation actions**: What was done to resolve the problem
- **Preventive measures**: How to prevent the same type of problem from recurring
- **Process changes**: Does the RACI, interface protocol, or SLA need modification

The key to knowledge preservation lies not in the documents themselves but in the mechanism that ensures documents are read and updated. It is recommended to set aside 15 minutes at the monthly review to go over the month's post-mortem documents and confirm whether preventive measures have been implemented.

Post-mortem mechanisms should avoid two extremes. One extreme is "accountability-driven post-mortems," where the post-mortem becomes a search for a responsible party, ultimately causing team members to conceal information when incidents occur. The other extreme is "formalistic post-mortems," where each review produces similarly templated conclusions such as "strengthen communication," "increase attention," "optimize going forward"—without any concrete process changes. Effective post-mortems should focus on systemic causes: Why did the error pass through the current process? Why didn't monitoring detect it earlier? Why did the interface protocol fail to prevent misuse? Why did a similar problem appear before without being institutionally resolved?

For LLM data teams, knowledge-preservation objects include not only incidents but also success experiences. A data-augmentation strategy that significantly improved model performance, an annotation guide that reduced consistency disputes, a quality rule that detected a large number of anomalous samples in advance—all of these should enter the knowledge base. Recording only failures causes the knowledge base to become a problem archive, ignoring reusable positive practices. Mature teams categorize knowledge preservation into four types—incident post-mortems, best practices, standard templates, and decision records—with distinct maintenance responsibilities for each.

| Document Type | Recorded Content | Maintenance Responsibility | Update Frequency | Primary Use |
|---|---|---|---|---|
| Incident post-mortem | Incident timeline, impact scope, root cause, remediation, and preventive measures | Incident owner, Quality Evaluator | Event-triggered | Prevent recurrence of similar problems |
| Best practices | Effective data strategies, annotation methods, quality rules | Responsible team for the practice | Monthly supplementation | Disseminate successful experience |
| Standard templates | Dataset specification, annotation guide, quality report, release notes | Data Owner, Platform Engineer | Updated each version cycle | Reduce collaboration costs |
| Decision records | Key trade-offs, approval conclusions, exception reasons, alternatives considered | Decision initiator, Data Owner | Decision-triggered | Preserve organizational memory |
| Onboarding guide | Role descriptions, tool entry points, FAQs, example workflows | Team lead, senior members | Quarterly review | Shorten onboarding time |

*Table 24-18: DataOps Knowledge-Preservation Objects and Maintenance Mechanisms*

The knowledge base must also be searchable and linkable. Post-mortem documents should link to specific data versions, issue numbers, quality reports, and interface protocols; dataset specifications should link to collection tasks, annotation guides, and compliance approvals; decision records should link to subsequent execution results. Only when documents form relationships does the knowledge base become not a pile of materials but an engineering asset that supports issue localization and organizational learning.

In academic and engineering contexts, knowledge preservation can also be understood as reducing organizational technical debt. Technical debt exists not only in code and models but also in undocumented processes, unexplained decisions, and unverified assumptions. When a team cannot explain why a certain rule exists, or cannot determine why a certain historical version was released, the organization has accumulated technical debt at the process level. DataOps's post-mortem and documentation practices are the mechanism for continuously repaying this type of debt.

### 24.4.3 Risk Escalation and Decision Committees

For risks that cannot be resolved at the level of individual roles, clear escalation paths are required. The following types of risk should be immediately escalated to the Data Owner level:

- Data compliance risks (suspicion that a data batch has copyright issues or privacy-leakage risk)
- Large-scale training-set quality anomalies (spot-check finds more than 10% of samples non-conforming)
- Severe SLA violations on the critical path (P0 requirement undelivered for more than 48 hours)
- Cross-team conflicts that cannot be resolved internally

For systemic risks affecting multiple project teams (e.g., major platform failures, compliance-policy changes), a temporary decision committee should convene—including each project's Data Owner and the platform-team lead—to produce a remediation plan within 24 hours.

The design of risk-escalation mechanisms should follow a tiered principle. Low-risk issues can be handled within the execution team; medium-risk issues require Data Owner involvement; high-risk issues must enter cross-departmental decision-making. The basis for tiering includes not only problem severity but also impact scope, reversibility, compliance sensitivity, and whether external commitments are affected. For example, a small number of format anomalies in an experimental data batch with limited impact scope that is reversible can be treated as a routine issue; but if the same problem has already entered the online evaluation set and may affect product-quality judgments, it should be escalated to a high-priority risk.

| Risk Level | Typical Scenarios | Response Deadline | Decision Authority | Handling Requirements |
|---|---|---|---|---|
| P0 | Privacy breach, major copyright risk, critical online data seriously incorrect | Immediate response; remediation conclusion within 24 h | Temporary decision committee, Legal/Compliance, Data Owner | Suspend use, isolate data, produce formal incident report |
| P1 | Formal training-set quality anomaly, critical SLA severely violated | Response within 4 h; remediation or mitigation within 48 h | Data Owner, quality lead, relevant team leads | Identify impact scope, remediation plan, and rollback strategy |
| P2 | Single-batch quality fluctuation, interface compatibility issue | Response within 1 business day | Responsible team lead | Enter Issue Pool and track in weekly post-mortem |
| P3 | Missing documentation, low-risk process deviation, non-critical metric anomaly | Addressed in the next work cycle | Execution team | Supplement documentation or include in monthly improvement items |

*Table 24-19: DataOps Risk Tiering and Response Mechanisms*

The value of a decision committee lies not in broadening the participation list but in concentrating the authority to handle cross-team risks in the hands of those who can bear responsibility. The committee should avoid discussing every detail and instead focus on four questions: Is it necessary to suspend the use of related data? Is it necessary to roll back or isolate a version? Is it necessary to communicate with external teams or clients? Is it necessary to revise existing processes? For technical remediations that are already clearly defined, the committee only needs to confirm resources and priority, not substitute for the execution team's specific solution design.

Risk escalation should also be coupled with audit records. Every P0 or P1 risk incident should retain a timeline, participating personnel, key judgments, data versions, impact scope, and subsequent actions. Audit records are not for post-hoc accountability but to ensure the organization can explain its decisions. Especially when user data, copyrighted data, or high-impact business scenarios are involved, "why was this handled this way" is itself part of governance capability.

Cross-team risk governance must also incorporate access management. LLM data typically contains multiple sensitivity levels: public corpora, internal business documents, user feedback, human annotation results, model-generated samples, and desensitized business data. Different sensitivity levels correspond to different access-control requirements. If teams authorize only by project-member identity without distinguishing data type and intended use, overly broad permissions easily result. Broad permissions improve data-access convenience in the short term but erode audit capability over time and expand the blast radius of misuse.

Access control design should follow the principle of minimum necessary privilege. Algorithm engineers do not necessarily need access to raw user identifiers; annotation outsourcing vendors do not necessarily need to see the full business context; quality evaluators do not necessarily need to download the entire dataset. Platforms should support access restrictions by field, version, task, and time as much as possible. For highly sensitive data, access should have a default expiry; if extension is needed, the intended use must be re-stated and approval obtained.

| Data Sensitivity Level | Example | Default Access Policy | Audit Requirements |
|---|---|---|---|
| Public data | Open-source datasets, public web text | Project members may apply on demand | Record version citations and download behavior |
| Internal ordinary data | Internal knowledge base, product documentation, non-sensitive operational documents | Authorized related projects only | Record accessor, intended use, and term |
| Internal sensitive data | User feedback, business logs, human review records | Minimum privilege, desensitization preferred, term control | Record field-level access and export behavior |
| Restricted data | Data potentially involving privacy, copyright disputes, or contractual restrictions | Prohibited by default; approved by exception | Full audit, isolated environment, revocation upon expiry |
| Derived data | Cleaned samples, annotation results, model-generated data | Re-graded based on source and processing method | Record source lineage and usage boundaries |

*Table 24-20: LLM Data Sensitivity Levels and Access Governance*

Outsourcing collaboration is also an important risk-governance scenario. Many LLM data projects rely on external annotation vendors, crowdsourcing platforms, or domain experts to complete data production. Outsourcing expands capacity but introduces standard-transmission, access-control, quality-consistency, and confidentiality-management challenges. DataOps cannot treat outsourcing only as a procurement process; outsourcing vendors must be integrated into the data production chain: task dispatch, guide training, sample calibration, quality spot-checking, feedback, and access revocation all need to be clearly recorded.

The most common mistake in outsourcing management is specifying only delivery volume and acceptance rate in the contract without specifying in-process data. For example, annotator revision counts, low-confidence samples, disputed samples, calibration task results, and feedback response time are all important signals for judging outsourcing quality. If spot-checking is done only at final delivery, it is hard to detect standard-comprehension deviations in time. The more mature approach is to incorporate in-process metrics into the platform and observe inter-vendor differences during weekly quality inspections.

| Outsourcing Governance Stage | Key Control Points | Recommended Evidence |
|---|---|---|
| Onboarding | Have confidentiality, compliance, and tool-training been completed? | Training records, permission approvals, test-task results |
| Task dispatch | Are official annotation guides and versioned examples being used? | Task IDs, guide versions, example-library citations |
| In-process monitoring | Is there quality drift or response delay? | Calibration task results, rework rate, proportion of disputed samples |
| Delivery acceptance | Is the agreed quality standard met? | Spot-check report, consistency report, issue list |
| Problem feedback | Is standard-comprehension deviation corrected promptly? | Q&A records, guide changes, retraining records |
| Offboarding and revocation | Are permissions closed and deliverables archived? | Permission-revocation records, data-deletion confirmation, delivery archive |

*Table 24-21: DataOps Control Points for Outsourcing Annotation Collaboration*

Change management is equally fundamental to cross-team risk governance. In LLM data projects, changes can come from data-source adjustments, cleaning-rule modifications, annotation-label revisions, quality-threshold increases, compliance-policy changes, or platform-tool upgrades. Any change may affect historical comparability. Teams should distinguish compatible changes from breaking changes: compatible changes can take effect incrementally within the existing process; breaking changes require explicit migration plans, impact scope, and rollback strategies.

The goal of change management is not to prevent change but to make change interpretable. For algorithm teams, data-change records help explain experiment results; for quality teams, change records help determine whether metric fluctuations are reasonable; for compliance teams, change records demonstrate that the organization identified and addressed risks. Without change records, teams can only rely on memory to explain history—and memory is unreliable in complex projects.

---

## 24.5 Case Study: From Small Team to Platform-Driven Organization

### Case Background

An online education company (hereinafter "Company E") began building an educational large model for K–12 scenarios in early 2023. The initial team had only five people: two algorithm engineers who doubled as data engineers, and three part-time annotators (all internal instructional staff). This stage ran smoothly because the team was small, communication costs were low, and any problem could be resolved with a quick conversation.

In the second half of 2023, the project entered a scaling phase: two outsourcing annotation vendors were brought in, data volume expanded from tens of thousands to millions of records, the algorithm team grew to eight people, and two new sub-projects launched simultaneously (oral assessment and essay scoring). Problems erupted:

- The annotation tasks for three projects were mixed together; outsourcing vendors could not determine which project's standards applied to each batch
- Algorithm engineers reported that the latest training set contained large numbers of duplicate samples, but nobody knew which cleaning step had gone wrong
- The data engineer for the oral assessment project independently built a cleaning tool that was incompatible with the main project's toolchain
- Annotation results from an outsourcing vendor were directly modified by an algorithm engineer and used as training data, breaking the quality-traceability chain

Further investigation revealed deeper organizational causes. First, although the team had grown, its collaboration mode remained at the early small-team stage. Many tasks were assigned ad hoc through instant-messaging tools, without formal requirement IDs or delivery records. Second, the annotation guidelines used by outsourcing vendors and internal instructional staff were not fully consistent; some terms had different interpretations across projects. Third, to expedite experimental progress, the algorithm team frequently copied intermediate data files directly, bypassing the data owner and quality-evaluation stages. In the short term, these practices increased local efficiency; in the long term, they made data lineage, version relationships, and quality accountability progressively inexplicable.

Company E also found that data problems do not always surface at delivery time. Some samples appeared compliant during spot-checks but caused abnormal response styles for specific question types after entering model training; some duplicate samples had low proportions within individual projects but rapid duplication-rate increases after cross-project merging; some outsourcing annotation results met surface-level consistency criteria but had systematic divergences from internal instructional experts' preferences. This demonstrated that DataOps restructuring could not focus solely on task management but had to incorporate model-experiment feedback, cross-project reuse, and expert-knowledge calibration into organizational processes.

### Restructuring Process

Company E's data lead decided to implement a DataOps restructuring in three phases:

**Phase 1 (Months 1–2): Role Clarification and Interface Definition**

A two-week role workshop was organized with all core members participating. The following outcomes were produced:

- Five core roles were clearly identified: Data Owner (filled by the data lead), Data Engineers (2 people), Annotation Ops (1 person, dedicated to managing outsourcing vendors), Quality Evaluator (1 person seconded part-time from the instructional team), and Algorithm Liaison (the algorithm team's Tech Lead)
- Interface protocols between Data Engineer → Annotation Ops and Annotation Ops → Algorithm Liaison were drafted
- The initial version of the RACI matrix was completed

The role workshop was not simply a renaming of job titles. It required each member to list the tasks they actually performed, the upstream inputs they depended on, the outputs they delivered downstream, and the blockers they most frequently encountered. Through this process, the team discovered that many conflicts arose not from insufficient technical capability but from multiple implicit owners existing for the same item. For example, annotation guides were drafted by instructional staff, but outsourcing vendor questions were answered by Annotation Ops, and the algorithm side would sometimes modify label definitions ad hoc based on experiment results—with no one able to confirm which version of the guide was the official standard.

To resolve this, the team placed annotation guides under version control and stipulated that each guide change must include the reason for the change, affected fields, example samples, and effective date. Outsourcing vendors could only execute tasks according to the published version; if the algorithm team wished to adjust label definitions, they had to submit a change request through the Demand Pool. This practice increased process costs early on but significantly reduced subsequent rework.

| Clarification Object | Pre-Restructuring State | Post-Restructuring Rules |
|---|---|---|
| Data requirements | Raised ad hoc in instant messaging; no IDs | All requirements enter the Demand Pool with priorities and acceptance criteria |
| Annotation guides | Multiple documents coexisting; version unclear | Unified version control; changes must record reasons and examples |
| Data delivery | Files sent directly to algorithm engineers | Delivered through the data-version release process |
| Quality review | Relied on ad hoc spot-checks by instructional experts | Quality Evaluator issues reports according to spot-check plan |
| Outsourcing communication | Different internal members answering questions separately | Annotation Ops serves as unified point of contact and accumulates a FAQ |

*Table 24-22: Company E Phase-1 Role and Interface Clarification Results*

**Phase 2 (Months 3–4): Cadence Establishment and Tool Unification**

- The dual-meeting mechanism of Monday requirement sync and Friday delivery post-mortem was established
- Data format standards were unified across the three projects (JSONL + unified schema)
- An internal annotation management platform was deployed; all outsourcing vendor tasks were dispatched and collected through the platform
- A data version management system (based on DVC) was deployed; each dataset update generates a version number

The core of Phase 2 was not "rolling out tools" but enabling tools to carry the already-clarified processes. The team initially tried to go live with the annotation platform directly, but quickly discovered that without unified fields, unified task naming, and unified acceptance rules, the platform merely moved the chaos from offline to online. The team therefore first unified the data format and task templates, then connected task dispatch, annotation collection, spot-check results, and version release to the platform. The platform thus recorded not a series of scattered operations but the complete data-production process.

On the weekly cadence front, Company E did not place everything into meetings. The Monday meeting discussed only requirements that had been submitted to the Demand Pool with complete information; requirements with incomplete information were returned to the submitter. Friday post-mortems did not report all work item by item but focused on tasks not delivered per SLA, high-priority quality issues, and matters requiring cross-team decisions. This adjustment shifted meetings from "reporting work" to "resolving blockers," reducing repetitive communication.

After tool unification, the team also established three automated checkpoints. First, data must pass schema validation and deduplication checks before entering the annotation platform. Second, annotation completion must generate a batch-level quality summary. Third, data releases must associate a requirement ID, annotation batch, and quality report. If any checkpoint is missing, the data version cannot enter the formal training corpus. Through these checkpoints, processes that previously relied on manual reminders became system constraints.

**Phase 3 (Months 5–6): Quality System and Knowledge Preservation**

- An automated quality-check pipeline was established: every data delivery automatically runs deduplication, format checking, and consistency checking
- A monthly quality-trend report template was developed
- Three post-mortem documents for major issues were completed, and preventive measures were implemented as process changes

The difficulty of Phase 3 was transforming quality from an "acceptance action" into an "operational metric." The team no longer judged only whether a data batch was compliant at release time; instead, it continuously observed quality differences across projects, outsourcing vendors, question types, and individual annotators. The quality-trend report included batch pass rate, duplication rate, annotation consistency, rework rate, issue-closure time, and algorithm-feedback hit rate. Through layered analysis, the team found that the essay-scoring project had a significantly higher rework rate than the oral-assessment project, primarily not because of annotators' insufficient capability but because the essay-scoring criteria had an insufficient number of boundary examples.

In response to this finding, the quality evaluator and instructional experts jointly supplemented the boundary-example library and added training questions and calibration questions to the annotation platform. Each week, outsourcing vendors were required to complete a specified number of calibration questions; annotators whose consistency fell below the threshold required retraining. This mechanism moved quality management from after-the-fact spot-checking to the annotation process itself, reducing large-scale rework.

On knowledge preservation, Company E split post-mortem documents into two parts: "incident post-mortems" and "process changes." Incident post-mortems record what happened and why; process changes specify which templates, tool rules, or meeting mechanisms need modification. A post-mortem is considered closed only when the process change has been implemented in a system or document, preventing post-mortems from remaining at the level of text.

| Phase | Primary Objective | Key Actions | Organizational Benefit |
|---|---|---|---|
| Months 1–2 | Clarify roles and interfaces | Workshop, RACI, interface protocols, annotation guide versioning | Reduce ambiguous responsibilities and verbal agreement costs |
| Months 3–4 | Establish cadence and tool constraints | Weekly cadence, unified schema, annotation platform, version management | Improve delivery predictability and data traceability |
| Months 5–6 | Form quality operations and knowledge preservation | Automated checks, quality trends, post-mortem closed loops, boundary-example library | Improve quality stability and organizational learning capability |

*Table 24-23: Company E DataOps Restructuring Phases and Organizational Benefits*

### Results

Six months later, the main indicators for Company E's data team showed significant changes:

| Metric | Before Restructuring | After Restructuring |
|---|---|---|
| Data delivery delay rate | ~40% (estimated) | < 10% |
| Annotation batch pass rate | ~82% | > 93% |
| Data-issue root-cause localization time | Average 3 days | Average 4 hours |
| Cross-project data reuse rate | ~5% | > 30% |
| Onboarding time for new members | ~3 weeks | ~1 week |

*Table 24-24: Company E Core Metrics Before and After DataOps Restructuring*

The most important change was not merely numeric—it was the team's work state: from "reactively handling issues every day" to "advancing according to a cadence." The data lead's time spent handling cross-team coordination issues dropped from 60% to 20%, freeing more time for improving data strategy and quality standards.

From a project-outcome perspective, DataOps restructuring also improved the quality of feedback between data and models. Before restructuring, the algorithm team often only reported "this data batch didn't work well," and the data team found it difficult to determine whether to adjust collection, cleaning, or annotation. After restructuring, each experiment must associate a data version and error-sample analysis, and algorithm engineers must explain which question types, knowledge points, or interaction scenarios the problems concentrate in. The data team can accordingly supplement samples more precisely rather than blindly expanding data volume.

From a cost-structure perspective, the team did not significantly increase headcount, yet reduced repetitive work through processes and platforms. Previously, different projects would separately clean similar data; after restructuring, data from the same source first enters the shared data pool, then derives different annotation tasks based on project requirements. Previously, outsourcing vendor questions required multiple people to repeatedly explain; after restructuring, the FAQ and boundary-example library handled most standard-transmission work. Previously, locating data problems required searching multiple folders and chat records; after restructuring, issues can be rapidly tracked via requirement IDs, batch IDs, and version numbers.

From an organizational-culture perspective, DataOps restructuring gradually fostered the team's habit of discussing problems based on evidence. Quality disputes no longer rely primarily on individual judgment but return to spot-check samples, annotation guides, experiment results, and historical post-mortems. Requirement priority is no longer determined by who nags most urgently but by business impact, model benefit, resource cost, and risk level jointly. This cultural change is more important than the launch of any individual tool, because it determines whether the team can maintain collaboration quality as it continues to scale.

Company E's case also suggests that DataOps construction should not pursue a complete system all at once. For early-stage teams, the most important thing is first establishing role, interface, and version records; for rapidly expanding teams, the priority should be addressing weekly cadence, annotation quality, and data reuse; for platform-driven teams, the further need is building metrics systems, access auditing, and knowledge bases. The investment focus differs at each stage, but the shared objective is consistent: moving data work from individual-experience-driven to organizational-capability-driven.

| Experience Theme | Specific Practice | Transferable Insight |
|---|---|---|
| Govern interfaces before building platforms | First unify schema, annotation guides, and delivery rules; then go live with tools | Tools only produce governance value when they carry established processes |
| Record versions before pursuing automation | Generate version numbers and quality summaries for every data release | Traceability is the foundation for subsequent automation and auditing |
| Prioritize by severity before optimizing broadly | Address P0/P1 issues and high-rework tasks first | DataOps improvement should begin from high-impact blockers |
| Establish examples before training personnel | Use boundary-example libraries to align annotation understanding | For complex tasks, examples convey understanding more effectively than abstract rules |
| Establish feedback before expanding data | Supplement data based on experiment error analysis | Data-volume growth must serve clearly identified model-improvement objectives |

*Table 24-25: Transferable Lessons from Company E's Case*

Abstracting Company E's experience into an actionable checklist reveals that DataOps transformation covers five dimensions: organization, process, tools, metrics, and culture. The organizational dimension resolves who is responsible; the process dimension resolves how to collaborate; the tools dimension resolves how to institutionalize; the metrics dimension resolves how to determine whether improvements are effective; the cultural dimension resolves whether the team is willing to learn continuously based on facts. All five dimensions are indispensable. Organization and process without tools causes rules to remain in documents; tools and metrics without culture causes the team to treat systems as extra burdens; culture without interfaces and versions still makes collaboration difficult to scale.

For companies preparing to replicate similar practices, starting with a lightweight checklist is advisable. The value of a checklist lies not in achieving all requirements at once but in helping the team identify its currently weakest areas. For example, some teams already have good platform tools but lack a Data Owner and RACI; some teams already have fixed meetings but lack meeting inputs, outputs, and quality metrics; some teams already have extensive documentation but the documentation is not linked to versions, issues, and experiment records. Different weak points correspond to different improvement paths.

| Checklist Dimension | Key Question | Initial State Manifestation | Target State After Improvement |
|---|---|---|---|
| Role responsibility | Is there a clear ultimate accountable party for each type of data decision? | Multiple participants but no one willing to make the call | Each critical item has a unique A and an explicit R |
| Requirement management | Do all data requirements enter a unified Demand Pool? | Requirements scattered in meetings and chat records | Requirements have IDs, priorities, acceptance criteria, and deadlines |
| Interface protocols | Do upstream and downstream have stable delivery contracts? | Fields and definitions depend on verbal explanation | Schema, semantic descriptions, and SLAs are all documented |
| Annotation governance | Are annotation guides versioned and traceable? | Multiple versions coexist; outsourcing vendors have inconsistent understanding | Guides, examples, Q&As, and change records are maintained uniformly |
| Quality control | Can quality trends be continuously observed? | Ad hoc spot checks before release | Automated checks, manual spot checks, and trend analysis combined |
| Version management | Can data used in historical experiments be reproduced? | Folder naming and manual records | Data versions, model experiments, and quality reports are cross-linked |
| Access compliance | Is authorization by intended use and term? | Project members hold broad long-term permissions | Minimum privilege, automatic revocation upon expiry, access auditing |
| Incident post-mortems | Are incidents translated into process improvements? | Post-mortems remain at the textual summary level | Post-mortem closure depends on changes to processes, tools, or templates |
| Data reuse | Do high-value datasets enter the asset registry? | Privately held within projects | Discoverable, applicable, trackable, and revocable |
| Metrics governance | Do metrics serve learning rather than surface performance evaluation? | Only tracking quantities and progress | Simultaneously observing flow efficiency, quality, collaboration, and outcomes |

*Table 24-26: Lightweight DataOps Transformation Checklist*

This checklist can also serve as input for quarterly reviews. Teams can rate each item on a four-point scale—"not established," "partially established," "stably running," "continuously optimized"—and select two or three of the most critical gaps as improvement targets for the next quarter. This approach transforms DataOps construction from a one-time project into continuous capability building. Like model training, organizational capability also requires iteration: first establish a baseline, then observe feedback, then progressively optimize.

In Company E's case, the initial gaps concentrated in requirement management, interface protocols, and version management; once those issues were alleviated, new bottlenecks shifted to quality-trend analysis and cross-project reuse; further on, the team began to focus on access compliance, asset registries, and metrics governance. This demonstrates that the DataOps flywheel is not a static framework but a management mechanism that continuously expands as the organization matures. Each time the team solves one category of problem, the flywheel exposes higher-order problems, driving the organization into the next round of improvement.

It is worth emphasizing that DataOps transformation should not be described as a pure efficiency-improvement project. Efficiency improvement matters, of course, but if the team pursues only faster delivery, it may treat quality, compliance, and knowledge preservation as burdens. For LLM data teams, the true objective is increasing the **determinedness** of data work: requirements more determinate, delivery more determinate, quality more determinate, risk boundaries more determinate, historical decisions more interpretable. Increased determinedness indirectly improves efficiency, because the team no longer needs to repeatedly make up for the rework and coordination costs caused by indeterminacy.

In management communication, the benefits of DataOps can be broken down into three categories. The first is **short-term benefits**: for example, reduced delivery delay rates, shorter problem-localization time, fewer annotation rework instances. The second is **mid-term benefits**: for example, increased cross-project reuse, improved experiment reproducibility, faster onboarding for new members. The third is **long-term benefits**: for example, data-asset accumulation, platform-capability reuse, and organizational-risk control. Different levels of management focus on different benefits; the data lead needs to communicate the value of DataOps in appropriate terms for each audience, rather than emphasizing only engineering details.

| Benefit Type | Typical Metrics | Primary Beneficiaries | Communication Approach |
|---|---|---|---|
| Short-term efficiency benefits | Delivery cycle, delay rate, rework rate, issue-localization time | Data team, algorithm team | Explain how process improvements reduce waiting and repetitive work |
| Mid-term collaboration benefits | Version reproducibility rate, reuse rate, onboarding time | Project teams, platform team | Explain how standardization reduces cross-team collaboration costs |
| Long-term governance benefits | Audit completeness rate, compliance risk incidents, data-asset value | Management, compliance team | Explain how DataOps reduces systemic risk |
| Model-outcome benefits | Key evaluation improvement, error-sample reduction, data-gain contribution | Algorithm team, product team | Explain how data improvements serve model and business objectives |

*Table 24-27: Layered Communication of DataOps Transformation Benefits*

Finally, the case also reminds us that DataOps construction requires restraint. Not every process needs to be automated, not every decision needs approval, and not every metric needs a dashboard. The complexity of organizational governance must match the corresponding risk and scale. Early-stage teams can use simple templates and fixed meetings to establish basic order; mid-sized teams can introduce platforms and automated checks; large teams can then build more complete access control, auditing, asset registries, and metrics systems. Excessive governance harms exploration capability; insufficient governance makes scale-up unsustainable. The practical wisdom of DataOps lies precisely in establishing a dynamic balance among speed, quality, compliance, and learning.

In actual deployment, teams can also break the transformation plan into short cycles of approximately twelve weeks. Twelve weeks is not a fixed deadline but a sufficiently lightweight, verifiable organizational experiment window. The first three weeks focus on diagnosis and role confirmation; the middle six weeks focus on process and tool institutionalization; the final three weeks focus on metrics post-mortems and policy revision. Compared with planning a full-year platform blueprint at once, short-cycle transformation makes it easier for teams to see benefits and easier to adjust direction based on feedback.

| Week | Key Tasks | Deliverables | Acceptance Criteria |
|---|---|---|---|
| Week 1 | Interview core roles; map current data flows and main blockers | Current-state problem list, data-flow sketch | Covers major data-production and data-consumption roles |
| Week 2 | Confirm Data Owner, RACI, and critical decision items | RACI initial version, escalation-path draft | Each critical item has a unique accountable party |
| Week 3 | Select a high-frequency data chain as the pilot | Pilot scope, risk boundaries, pilot metrics | Pilot scope is small enough and impact is clear |
| Week 4 | Establish the Demand Pool and unified task templates | Requirement template, priority rules | New requirements can be submitted and reviewed using the template |
| Week 5 | Map interface protocols and data schemas | Interface protocols, schema files, field descriptions | Upstream and downstream reach agreement on delivery standards |
| Week 6 | Establish annotation guide versioning and example library | Annotation guide, boundary examples, FAQ | Outsourcing vendors and internal staff use the same version |
| Week 7 | Connect basic quality checks | Deduplication, format, anomaly-value, and spot-check rules | Quality summaries are automatically generated before data release |
| Week 8 | Establish version release and experiment linkage | Data version specification, experiment citation rules | Experiments can be traced to specific data versions |
| Week 9 | Run weekly post-mortem and close the first set of issues | Issue Pool, post-mortem records, improvement items | High-priority issues have owners and deadlines |
| Week 10 | Summarize quality, delivery, and collaboration metrics | Metrics dashboard or monthly report | Metric definitions are clear and changes can be explained |
| Week 11 | Assess pilot benefits and revise processes | Pilot evaluation report, process revision records | Improvement benefits and remaining risks are clearly stated |
| Week 12 | Decide on rollout scope and next-round priorities | Rollout plan, resource requirements, next-round objectives | Management and participating teams reach consensus |

*Table 24-28: Twelve-Week DataOps Pilot Transformation Roadmap*

The key to the twelve-week roadmap is selecting the right pilot. The pilot should not be the simplest edge-case task, because a successful edge case is difficult to demonstrate organizational value; nor should it be the highest-risk, most dependency-laden core chain, because early-stage mechanisms are not yet mature and the cost of failure is too high. The most suitable pilots typically satisfy three conditions: a relatively complete data chain, involvement of at least two teams, and sufficiently prominent and quantifiable problems. For example, an RLHF data iteration chain spanning algorithm and annotation teams is generally a more suitable DataOps pilot than a standalone offline cleaning script.

During the pilot, managers need to deliberately protect the transformation cadence. New processes create additional friction early on, and team members may perceive filling out requirement templates, recording version notes, and attending post-mortems as slowing things down. At this point, the data lead needs to explain the purpose of these actions and use concrete results to demonstrate their value. For example, the first time a problem is quickly localized via version records, the first time rework is reduced through an example library, the first time queue-jumping is prevented via the Demand Pool—all of these build the team's trust in the new mechanisms. Organizational change is not accomplished through policy documents but through repeatedly occurring positive evidence.

When the pilot moves to the rollout phase, the team should avoid mechanically copying the pilot template to all scenarios. Different data chains carry different risks, scales, and collaboration complexity; tiered governance should be applied. High-risk formal training data requires the complete process; exploratory data can use a simplified process; data involving user feedback requires strict access control and auditing, while public data can use lighter-weight authorization. Tiered governance reduces unnecessary process burdens and concentrates governance resources on genuinely high-risk, high-value scenarios.

| Rollout Scenario | Recommended Governance Intensity | Applicable Processes | Rationale |
|---|---|---|---|
| Formal training corpus | High | Complete requirement, quality, version, compliance, and release processes | Affects model capability and subsequent experiment reproducibility |
| Critical evaluation sets | High | Strict version freezing, access control, and contamination prevention | Contaminated evaluation sets affect model quality judgments |
| Exploratory experimental data | Medium | Simplified requirements, quality labeling, and usage restrictions | Need to preserve exploration speed while preventing accidental entry into formal pipelines |
| Public corpus cleaning | Medium | Source records, cleaning rules, and quality summaries | Lower risk but large scale; traceability required |
| Temporary analysis samples | Low | Term authorization, basic records, and deletion upon expiry | Value is short-term; heavy processes not warranted |
| Highly sensitive user feedback | High | Minimum privilege, desensitization, auditing, and compliance approval | Involves privacy and usage boundaries; misuse cost is high |

*Table 24-29: Tiered Governance Intensity for Different Data Scenarios*

Through this approach, the Company E case no longer represents merely the experience of a single organization but can be generalized into a universal implementation method: first diagnose organizational friction, then select a pilot chain; first clarify roles and interfaces, then connect tools; first build version and quality evidence, then discuss more complex platform solutions; first validate with a small number of scenarios, then roll out in tiers based on risk and value. This path is more consistent with the fundamental spirit of DataOps: through small-batch, verifiable, post-mortem-driven improvements, progressively forming stable data-engineering organizational capability.

To facilitate self-assessment after the transformation concludes, teams can also establish a concise set of verification questions. These do not replace formal metrics but help managers judge whether DataOps has truly entered daily operations rather than remaining in project documentation. If most questions cannot be answered clearly, the team is still in a formalistic construction phase; if the questions can be answered through system records, version descriptions, quality reports, and post-mortem documents, DataOps has begun to function as organizational infrastructure.

| Self-Assessment Question | Ideal Evidence |
|---|---|
| Who ultimately approved the most recent data release, and on what basis? | Release records, quality summaries, approval records |
| Which data versions did a certain model experiment use? | Experiment records, data version numbers, lineage information |
| Which batches were affected by the most recent annotation standard change? | Guide versions, change records, task IDs |
| Has a high-priority issue resulted in preventive measures? | Issue Pool, post-mortem documents, process changes |
| Which data assets are being reused by multiple projects? | Data asset registry, access logs, reuse records |
| Which permissions are expiring in the near future? | Permission system, authorization terms, revocation records |
| Were this month's quality fluctuations caused by process problems or data-structure changes? | Quality trend reports, sample analysis, change records |
| Can a new member understand the core processes within one week? | Onboarding guide, template examples, tool entry points |

*Table 24-30: Self-Assessment Questions for DataOps Operational Status*

These questions matter because they test the organization's **interpretability**. A team that can explain where data comes from, why it was used, how it was validated, who made decisions, and how problems were resolved already has the foundation to continue expanding. Conversely, if the team can only say "someone probably handled it" or "it's somewhere in a folder," even with fast short-term delivery speed, it will struggle to sustain long-term LLM data engineering.

DataOps's ultimate landing point is therefore not giving the team more institutions but ensuring that institutions genuinely reduce collaboration uncertainty. Good institutions should cause important problems to surface earlier, necessary decisions to happen faster, historical experience to be reused more easily, and high-risk operations to be harder to trigger inadvertently. When these effects persist, the team truly progresses from project-based collaboration to engineering-grade operations.

This is also the most fundamental insight this case offers to LLM data teams.

---

## Chapter Summary

This chapter has systematically presented the organizational design and DataOps flywheel mechanism for LLM data teams.

At the **organizational level**, we analyzed the three major limitations of traditional data teams in LLM projects (overlapping responsibilities, single-point dependency, and missing cadence), and the four design principles of new organizational forms (interfaces over hierarchy, combining asynchronous and synchronous modes, institutionalizing knowledge, and small teams with large platforms).

At the **role level**, we defined seven core role categories (Data Owner, Data Engineer, Annotation Engineer, Quality Evaluator, Algorithm Engineer, Platform Engineer, and Legal/Compliance Specialist), designed role interface protocols and a RACI responsibility matrix, and clarified escalation paths and exception-handling mechanisms.

At the **flywheel level**, we introduced the four-pool collaborative model—Demand Pool, Data Pool, Experiment Pool, and Issue Pool—along with the complete meeting system: weekly cadence (Monday requirement sync, Wednesday quality inspection, Friday delivery post-mortem), monthly reviews, and quarterly version freezes.

At the **governance level**, we discussed conflict-resolution mechanisms for multi-team data asset sharing, workflow design for knowledge preservation and post-mortems, and risk-escalation paths.

Finally, through the case of Company E evolving from a 5-person small team to a platform-driven DataOps organization, we illustrated the deployment process and quantitative outcomes of this methodology in real scenarios. The changes in the case also echo the perspective of lean enterprise practice: improvements in organizational capability come from value-stream optimization, rapid experimentation, feedback loops, and cross-functional collaboration—not from simply expanding team size (Humble, Molesky and O'Reilly 2015).

DataOps is not a set of tools but an upgrade to the way a team works. Its value lies not in operating perfectly from day one, but in enabling the team to continuously improve efficiency and quality through iterative cycles sustained by a fixed cadence and clear interfaces.

---

## Further Reading

**Methodology References**

*The DevOps Handbook* by Kim et al. is a classic for understanding DevOps culture and practices; its discussion of the "Three Ways" (flow, feedback, and continuous learning) offers important inspiration for DataOps design. The DataOps Manifesto, published by the DataKitchen team, is the most widely cited statement of DataOps values, covering 18 core principles.

**Recommended Open-Source Tools**

Apache Airflow is the most mature data-workflow orchestration tool, suitable for building pipeline DAGs. Prefect is a next-generation workflow orchestration tool that is more flexible than Airflow in handling dynamic tasks and error handling. Label Studio is a fully featured open-source annotation platform supporting text, images, audio, and other modalities.

---

## References

Amershi S, Begel A, Bird C, DeLine R, Gall H, Kamar E, Nagappan N, Nushi B, Zimmermann T (2019) Software Engineering for Machine Learning: A Case Study. In: Proceedings of the 41st International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP), pp 291–300.

Baylor D, Breck E, Cheng H-T, Fiedel N, Foo C Y, Haque Z, Haykal S, Ispir M, Jain V, Koc L, Koo C Y, Lew L, Mewald C, Modi A N, Polyzotis N, Ramesh S, Roy S, Whang S E, Wicke M, Wilkiewicz J, Zhang X, Zinkevich M (2017) TFX: A TensorFlow-Based Production-Scale Machine Learning Platform. In: Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp 1387–1395.

Beyer B, Jones C, Petoff J, Murphy N R (eds.) (2016) Site Reliability Engineering: How Google Runs Production Systems. O'Reilly Media.

Breck E, Cai S, Nielsen E, Salib M, Sculley D (2017) The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. In: IEEE International Conference on Big Data, pp 1123–1132.

Breck E, Polyzotis N, Roy S, Whang S E, Zinkevich M (2019) Data Validation for Machine Learning. In: Proceedings of Machine Learning and Systems 1, pp 334–347.

DAMA International (2017) DAMA-DMBOK: Data Management Body of Knowledge, 2nd Edition. Technics Publications.

DataOps Manifesto (accessed 2026) The DataOps Manifesto: 18 DataOps Principles. Online manifesto. Available at: https://dataopsmanifesto.org/en/

Dehghani Z (2022) Data Mesh: Delivering Data-Driven Value at Scale. O'Reilly Media.

Forsgren N, Humble J, Kim G (2018) Accelerate: The Science of Lean Software and DevOps. IT Revolution Press.

Gebru T, Morgenstern J, Vecchione B, Vaughan J W, Wallach H, Daumé III H, Crawford K (2021) Datasheets for Datasets. Communications of the ACM 64(12):86–92.

Humble J, Farley D (2010) Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation. Addison-Wesley.

Humble J, Molesky J, O'Reilly B (2015) Lean Enterprise: How High Performance Organizations Innovate at Scale. O'Reilly Media.

Kim G, Humble J, Debois P, Willis J, Forsgren N (2021) The DevOps Handbook: How to Create World-Class Agility, Reliability, and Security in Technology Organizations, 2nd Edition. IT Revolution Press.

Kreuzberger D, Kühl N, Hirschl S (2023) Machine Learning Operations (MLOps): Overview, Definition, and Architecture. IEEE Access 11:31866–31879.

Mitchell M, Wu S, Zaldivar A, Barnes P, Vasserman L, Hutchinson B, Spitzer E, Raji I D, Gebru T (2019) Model Cards for Model Reporting. In: Proceedings of the Conference on Fairness, Accountability, and Transparency, pp 220–229.

Project Management Institute (2021) A Guide to the Project Management Body of Knowledge (PMBOK Guide), 7th Edition. Project Management Institute.

Reis J, Housley M (2022) Fundamentals of Data Engineering. O'Reilly Media.

Sambasivan N, Kapania S, Highfill H, Akrong D, Paritosh P, Aroyo L M (2021) "Everyone wants to do the model work, not the data work": Data Cascades in High-Stakes AI. In: Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, pp 1–15.

Sculley D, Holt G, Golovin D, Davydov E, Phillips T, Ebner D, Chaudhary V, Young M, Crespo J-F, Dennison D (2015) Hidden Technical Debt in Machine Learning Systems. In: Advances in Neural Information Processing Systems 28, pp 2503–2511.

Skelton M, Pais M (2019) Team Topologies: Organizing Business and Technology Teams for Fast Flow. IT Revolution Press.
