# Chapter 26: Data Platform Observability

---

## Abstract
"Scheduled successfully" and "data is healthy" are two entirely different things. SRE practice emphasizes that system health cannot be measured solely by whether processes are running, but by whether services continuously meet user-perceptible reliability objectives (Beyer et al. 2016). Jobs on a data platform may run with all-green status while the quality of training data has quietly deteriorated — until the algorithm team notices anomalous model performance, the data team begins investigating, and discovers that problems have been accumulating for weeks.

This chapter is addressed to engineering teams responsible for platform stability, job monitoring, and quality alerting. It systematically describes how to establish an observability framework for an LLM data platform. Observability typically requires combining metrics, logs, traces, and contextual information to explain *why* a system is in its current state, rather than merely reporting that state (Sigelman et al. 2010; OpenTelemetry Authors 2024). The chapter proceeds along four lines: first, it analyzes why job success does not equal data health and identifies failure modes unique to LLM data platforms; second, it establishes a layered metric hierarchy that integrates task metrics, quality metrics, and business metrics into a unified observability framework; third, it covers alert strategy design, anomaly attribution workflows, and incident response mechanisms; and finally, it discusses capacity forecasting, cost alerting, and operational dashboard design.

Upon completing this chapter, readers will have a complete observability design for a data platform: a tiered monitoring-metrics reference table, an alert-severity-to-remediation-action table, a ready-to-reuse incident post-mortem template, and a full post-mortem walkthrough of a real platform incident.

## Keywords

Data platform observability; quality metrics; lineage tracking; alert attribution; incident post-mortem; cost alerting

## Learning Objectives

- Distinguish the three layers of success — schedule success, task success, and data correctness — and identify typical blind spots in each.
- Recognize silent-failure modes such as semantic drift, over-filtering, dependency version drift, and corpus contamination.
- Design a layered observability framework that integrates task metrics, quality metrics, and business metrics.
- Build alert tiering, anomaly attribution, and incident response workflows.
- Design capacity forecasting, cost alerting, and operational dashboards to support long-term platform operations.

---

## Scenario Introduction

The following is an anonymized composite case study; the figures are intended to illustrate the magnitude of observability problems and do not represent the actual statistics of any public platform. An LLM data platform at a certain company processes approximately two million raw records per day, producing roughly one hundred thousand training samples after cleaning, annotation, and format conversion. The platform's monitoring dashboard shows that over the past month all Airflow DAGs have maintained a success rate above 99.2%, average job durations have been stable, storage capacity is adequate, and everything appears to be in order.

At a monthly review meeting, however, the algorithm team reports that the past two weeks of training experiments have fallen noticeably short of expectations, with evaluation scores on Chinese long-text comprehension tasks dropping by approximately 6%. The data team begins investigating.

The findings are alarming. Three weeks earlier, a dependency library in the data-cleaning pipeline had silently upgraded its version. In the new version, the tokenizer changed how it handles truncation of long Chinese texts: documents exceeding 512 tokens are now truncated incorrectly rather than being split at semantic boundaries. Jobs continue to run successfully, output file sample counts look normal, and all "task metrics" are green — but the data content has quietly changed, and not a single alert was ever triggered.

This case reveals the central thesis of data platform observability: **"whether a job ran successfully" cannot substitute for "whether the data is healthy."** Silent errors and data-dependency issues in production machine learning systems regularly bypass traditional infrastructure monitoring and ultimately manifest as model quality degradation (Sculley et al. 2015; Polyzotis et al. 2017). These are problems at different layers, and they require monitoring systems at different layers.

---

## 26.1 Why Job Success Does Not Equal Data Health

### 26.1.1 Schedule Success, Task Success, and Data Correctness Are Not Equivalent

Understanding the distinction between these three layers is a prerequisite for building a correct observability framework.

**Schedule success** is the most fundamental layer: the scheduling system (e.g., Airflow) has successfully triggered a task, and the task has entered the run queue. Schedule success means only that "the task was started" — it does not mean the task did the right thing.

**Task success** is the second layer: the task process exits with code 0 and throws no uncaught exceptions. Task success means "the program ran to completion normally," but the result of that run may be incorrect — for example, it may have processed an empty file, a bug in a filter condition may have deleted all samples, or the output format may not conform to expectations.

**Data correctness** is the most critical layer: the output data is healthy in content and meets the quality standards expected by the business. Data validation research indicates that schema checks, statistical constraints, distribution-shift detection, and anomaly-value detection should be integrated into routine pipelines rather than reserved for post-incident manual investigation (Breck et al. 2019). Data correctness cannot be inferred from task status; it can only be verified by inspecting the data content itself.

| Layer | What Is Inspected | Typical Tools | Common Blind Spots |
|-------|-------------------|---------------|--------------------|
| Schedule success | Whether the task was started | Airflow/Dagster status | Task starts but immediately exits with an error |
| Task success | Process exit code | Monitoring system alerts | Task completes but output is empty or content is incorrect |
| Data correctness | Data content quality | Data quality testing frameworks | Content format is valid but semantics are wrong; distribution shift |

*Table 26-1: Three-Layer Definitions of Success and Typical Blind Spots*

These three layers have a clear containment relationship, but they cannot substitute for one another. Schedule success only indicates that the control plane has no obvious faults; task success only indicates that the execution plane has no explicit exceptions; data correctness is what actually establishes that results are trustworthy at the business-semantics layer. Many teams persistently misjudge platform health because they treat the green status of the control and execution planes as a proxy for the green status of the data layer. For a traditional batch reporting pipeline, such misjudgment may only mean a given metric is fixed one day late. For an LLM data platform, it can allow corrupt data to enter the training set and surface weeks later as model capability regression.

Data platform observability therefore begins by redefining "success." A data processing job is truly successful only if it satisfies at least four conditions: the job runs at the expected time; the processing completes without abnormal exits; the output data satisfies structural and statistical constraints; and the output data has not deviated from target in its semantics or business coverage. The first two conditions belong to the platform engineering domain; the latter two belong to data quality and business governance. Only by monitoring all four conditions simultaneously can a team avoid the silent risk of "all pipelines green, but the model is getting worse."

In practice, schedule success and task success are relatively easy to cover with automated monitoring, because they correspond to unambiguous machine signals: status codes, exit codes, elapsed time, retry counts, and resource utilization. Data correctness is harder, because it involves content, distribution, semantics, and business boundaries. A sample with all fields present, a reasonable length, and a correctly identified language does not necessarily qualify for training; an annotation batch that passes spot-checks does not necessarily mean it is free of systematic bias in critical business scenarios. Data health must be judged through multi-dimensional signals rather than a single metric.

Data correctness is also relative. The same batch of data may be high quality for one task and noise for another. For example, historical policy documents may be valuable for historical Q&A tasks but risky for current operational guidance tasks; user complaint text may be valuable for robustness training but can introduce privacy and safety risks if it is not desensitized and classified. Data health therefore cannot be defined solely as "absence of errors" — it must also be defined as "fit for use in a specified context." This is why data platform observability must be combined with data versioning, usage context, and data contracts.

### 26.1.2 Failure Modes Unique to LLM Data Platforms

Unlike traditional data warehouses, LLM data platforms face a class of distinctive "silent failure" faults — problems occur without any obvious error signal. Research on data cascades in high-stakes AI scenarios further confirms that data problems tend to be amplified progressively along organizational processes and model pipelines (Sambasivan et al. 2021):

**Semantic Drift**: Processing logic is unchanged, but the content of the data source has changed. For example, pages crawled by a web spider have turned into advertisement pages, or the annotation style of an outsourced vendor has imperceptibly shifted, causing the semantic distribution of the output data to deviate from expectations while structural metrics such as file format and sample count remain completely normal.

**Over-filtering**: An adjustment to a parameter in the cleaning logic (e.g., raising a quality-score threshold) causes large numbers of high-quality samples to be erroneously deleted. Training-set coverage drops, but the sample count remains within the "normal range" and does not trigger count-based alerts.

**Dependency Drift**: As illustrated in the case study above, an underlying dependency library used by the processing framework upgrades silently, altering certain data-processing behaviors, while the test cases do not cover that edge case.

**Annotation Drift**: In long-running annotation tasks, annotator fatigue or diverging interpretations cause annotation quality to decline slowly, but single-batch spot-checks may be unable to detect this trending change.

**Data Island**: One of several data sources suddenly stops updating (e.g., a third-party data-sharing arrangement expires), leaving the training data without coverage of a certain domain, while aggregate volume metrics do not change significantly.

What these failure modes have in common is: **surface-level system metrics are healthy, while the underlying data health has already deteriorated.** Production ML platforms must monitor data, models, infrastructure, and business feedback simultaneously in order to cover these cross-layer failures (Amershi et al. 2019; Kreuzberger, Kühl and Hirschl 2023). Traditional infrastructure monitoring cannot detect such issues; dedicated quality monitoring of data content is required.

LLM data platforms also exhibit several more insidious failure modes. The first is **corpus contamination**: data that should have been reserved for evaluation or validation sets is mistakenly merged into the training set, causing model evaluation scores to be artificially inflated. Contamination problems typically do not cause job failures or trigger quality metric anomalies; they are exposed only when evaluation results are inconsistent with real production behavior. The second is **class dilution**: after a large influx of general samples, the proportion of critical minority-class samples drops, causing model capability regression in minority-class scenarios, while overall metrics may remain stable. The third is **safety boundary drift**: changes to rejection rules, sensitive-information handling, or risk-content filtering alter the distribution of safety-related data, but ordinary quality metrics cannot capture this change.

The fourth is **metadata corruption**. The data content itself may be unchanged, but labels, provenance, timestamps, authorization status, or business categories contain errors, causing downstream systems to misuse the data. For example, a batch of data authorized only for internal evaluation may be marked as suitable for training; a batch of English data may be incorrectly assigned to the Chinese corpus; a batch of historical feedback data may be labeled as recent feedback. Metadata corruption does not necessarily affect training jobs in the short term, but it undermines data governance and experimental interpretability.

The fifth is **processing order drift**. In complex pipelines, the ordering of filtering, deduplication, desensitization, sharding, annotation, and sampling affects the final data distribution. After one step's position in the sequence is adjusted, jobs still succeed and output volume may appear normal, but the set of retained versus discarded samples has changed. This effect is especially pronounced when deduplication and sampling coexist: a change in ordering can significantly affect the retention rate of long-tail samples. An observability system must record the processing chain and key intermediate metrics in order to detect such issues.

These failure modes demonstrate that monitoring on an LLM data platform cannot be limited to jobs and services; it must also encompass data distribution, data semantics, metadata, processing rules, version dependencies, and downstream usage. In other words, data platform observability is a cross-layer capability: it must detect failures at the infrastructure layer as well as degradation at the data-asset layer.

### 26.1.3 Why Data Health Issues Are Typically Discovered Late

A hallmark of LLM data quality problems is "impact latency" — a problem may take weeks or even months to appear as degraded model performance after it occurs. This delay originates from two stages:

**Data-to-training latency**: After a data problem occurs, the affected data must accumulate to a sufficient volume before it is ingested into the training set; the training run itself takes time; and the trained model must go through evaluation before the performance anomaly is detected. In a regular iteration cycle, this latency is typically two to six weeks.

**Training-to-deployment latency**: After training is complete, the model typically undergoes multiple rounds of evaluation and canary rollout before full production deployment, adding another one to four weeks of delay.

The total impact latency can be as long as two to three months. This means that without real-time or near-real-time monitoring at the data layer, teams will always be responding reactively after the fact rather than issuing early warnings in time.

Delayed discovery is also related to organizational collaboration chains. The data team typically has first access to raw data, but model performance is observed by the algorithm team, production feedback is collected by the product or operations team, and compliance issues are the concern of the security and legal teams. If these signals are not aggregated on a unified platform, data health problems will be passed back and forth across organizational boundaries. A data source going offline may first manifest as a drop in training-set coverage, then as model regression in a certain capability, and finally as user complaints. The role of the observability system is to connect these scattered signals early.

Another reason is that teams lack baselines for "normal variation." Data metrics naturally fluctuate — daily ingestion volume, category proportions, quality scores, and annotation consistency are all influenced by business activity, vendor schedules, and sampling strategies. Without historical baselines, engineers cannot easily judge whether a fluctuation is anomalous. Excessive sensitivity causes alert fatigue; insufficient sensitivity means early signals are missed. Data health monitoring must therefore preserve both short-term windows and long-term trends, using trend changes to identify risk rather than relying solely on single-point thresholds.

For LLM data platforms, the most valuable monitoring is often not telling the team "it has already broken" but rather "it is in the process of breaking." Examples include: coverage of a certain sample category declining across three consecutive batches; annotation consistency for a particular vendor falling below the historical average for two consecutive weeks; the freshness of a data source gradually lagging behind business changes; and the filter ratio of a certain cleaning rule slowly increasing. If these signals are captured in time, the team can complete fixes before the data enters formal training.

---

## 26.2 Metrics, Logs, Traces, and Lineage in Combination

### 26.2.1 The Three-Layer Metric Hierarchy

The metric framework for an LLM data platform should be organized into three layers, each addressing a different set of questions. The SLI/SLO thinking from SRE practice can be applied to the data platform, but it requires extending service-request-level metrics to cover data content, quality, and business coverage (Beyer et al. 2016):

**Layer 1: Task Metrics**

Task metrics provide baseline monitoring of platform operational status, measuring "whether tasks complete on schedule." Key metrics include:

- Task success rate: number of successfully completed tasks in the past 24 hours / total triggered tasks
- Task duration: mean execution time and P95 percentile for each task type
- Queue backlog: trend in the number of tasks waiting to execute
- Retry rate: proportion of tasks that require a retry to succeed (high retry rates suggest instability in underlying resources or dependencies)
- Data throughput: volume of raw data processed per hour (record count and storage size)

Task metrics are the foundation of observability, but as noted above, all-green task metrics do not imply data health.

**Layer 2: Quality Metrics**

Quality metrics directly measure the health of data content, evaluating "whether the output data meets quality standards." Key metrics include:

- Blank rate: proportion of samples with empty or excessively short content (< 10 tokens)
- Duplication rate: proportion of samples that duplicate historical dataset records
- Format compliance rate: completeness and correctness of sample fields and format
- Language distribution: proportions of each language in a multilingual dataset (alerts triggered when proportions deviate from expected distribution)
- Annotation consistency: Inter-Annotator Agreement (IAA) score within a batch for tasks of the same type
- Mean quality score: mean and distribution of scores from an automated quality assessment model
- Category coverage rate: sample count and proportion for critical business categories in the training target

Quality metrics should be computed after every processing batch and maintained as time-series data to support trend analysis and anomaly detection. The ML Test Score framework also incorporates data testing, training/serving consistency, and monitoring coverage into production readiness assessments (Breck et al. 2017).

**Layer 3: Business Metrics**

Business metrics assess the overall health of data assets from a business-value perspective, measuring "whether the data supports business objectives." Key metrics include:

- Training-set domain coverage rate: whether the training set covers all target business scenarios
- Data freshness: proportion of recently generated data in the training set (to avoid knowledge cutoff issues)
- Annotation throughput: number of high-quality annotated samples completed per unit time (a core metric supporting the iteration cadence)
- Data request fulfillment rate: proportion of data requests from the algorithm team completed on time
- Compliant data ratio: proportion of total data that has passed compliance review

| Metric Layer | Typical Metrics | Update Frequency | Primary Audience |
|--------------|-----------------|------------------|------------------|
| Task metrics | Success rate, duration, throughput | Real-time / minute-level | Platform engineers, SRE |
| Quality metrics | Blank rate, duplication rate, consistency, coverage | Batch-level (hourly / daily) | Data engineers, quality assessors |
| Business metrics | Domain coverage, data freshness, fulfillment rate | Daily / weekly | Data owners, algorithm team, product team |

*Table 26-2: Tiered Monitoring Metrics*

The three-layer metric framework must avoid becoming siloed. Task metrics tell the team "whether the platform is running on schedule"; quality metrics tell the team "whether the data meets standards"; business metrics tell the team "whether the data supports objectives." Relying solely on task metrics causes teams to overlook silent quality issues; relying solely on quality metrics may mean not knowing whether a problem has already affected the business; relying solely on business metrics means investigation begins only after the problem has become apparent too late. All three layers must be correlated through a shared data version, a shared processing batch, and a shared time window to support complete judgments.

The metric framework should also distinguish between "health metrics" and "diagnostic metrics." Health metrics are used to determine whether an alert is needed — for example, duplication rate, blank rate, category coverage, data freshness, and annotation consistency. Diagnostic metrics are used to explain why an anomaly occurred — for example, source-proportion breakdowns, filter-reason distributions, processing-node latency, dependency version changes, and inter-vendor batch differences. Health metrics should be few and stable; diagnostic metrics can be richer and are used during incident investigation. Configuring all diagnostic metrics as alerts produces excessive noise; lacking diagnostic metrics makes it difficult to identify root causes after an alert fires.

Metrics must also have clear owners. Platform engineers own task success rates, queue backlogs, and resource utilization; data engineers own throughput, processing failure causes, and version outputs; quality assessors own duplication rates, blank rates, consistency, and coverage; data owners own business coverage, request fulfillment, and data asset reuse; compliance roles own authorization status, sensitive-data proportions, and access audits. Metrics without owners ultimately become unattended dashboard decorations.

Metric design should also preserve explanatory context. A single metric value is typically insufficient to judge whether it is anomalous. For example, a drop in data throughput could reflect a data-source outage or a seasonal business lull; a rise in duplication rate could indicate a crawler issue or a business event that caused users to submit many similar queries; a drop in annotation consistency could reflect declining annotator quality or an increase in task difficulty. Dashboards should display data provenance, version, business calendar, change logs, and quality rule versions alongside the metrics, to help readers interpret changes.

Finally, the metric framework must be periodically revised. As model objectives, data sources, and business scenarios evolve, existing metrics may no longer cover critical risks. Early-stage teams may monitor only format, volume, and deduplication; mature teams need to monitor semantic coverage, preference distribution, safety boundaries, evaluation contamination, and data asset reuse. A metric framework that never changes will gradually diverge from real risks. It is recommended to dedicate time at monthly quality reviews or quarterly data governance reviews to verify whether metrics remain effective.

### 26.2.2 Combining Logs, Traces, Audit Logs, and Lineage

The four core observability tools — Logs, Traces, Audit Logs, and Lineage — each have distinct roles on the data platform and must be used in combination. Log management, distributed tracing, and lineage context respectively answer the questions "what happened," "what path did a request take," and "where did the data come from" (NIST 2006; Sigelman et al. 2010; Hellerstein et al. 2017):

**Logs**: Record detailed runtime information during data processing, including the processing outcome of each record (passed / filtered / exception), the filter reason, and processing duration. System log research shows that logs are not only after-the-fact evidence but can also reveal large-scale system anomalies in advance through pattern mining (Oliner and Stearley 2007; Xu et al. 2009). Logs are the finest-grained records and are suited for troubleshooting, but not for indefinite full-volume retention (cost is prohibitive). A tiered logging approach is recommended:

- INFO level: batch-level summary (how many records were processed and filtered per batch)
- DEBUG level: per-record detailed processing trace (enabled only during troubleshooting)
- ERROR level: processing failures and exceptions (retained permanently)

**Traces**: Record the complete processing chain of a single record from pipeline entry to output, including each processing node traversed and the corresponding timestamps. Dapper's experience demonstrates that end-to-end distributed tracing is critical for locating cross-service performance bottlenecks and abnormal request paths (Sigelman et al. 2010). Unlike logs, Traces emphasize an end-to-end, cross-system perspective. On the data platform, a Trace can answer: "When did this sample enter, which processing steps did it pass through, and in which dataset version did it ultimately appear?"

**Audit Logs**: Record who performed what operation on data at what time, immutably, for compliance auditing. Audit logs must be retained permanently and must contain complete operator identity information. Typical auditable events include: dataset version creation/deletion, quality standard modification, compliance review approval/rejection, and data access requests.

**Lineage**: As described in Chapter 25, lineage records dependency relationships between data assets. The key distinction between lineage and logs/traces is that lineage describes the static "derivation relationship," while logs and traces describe the dynamic "processing process." The lineage graph tells you "which data sources were used to produce dataset v2.3"; a trace tells you "what processing steps this specific sample went through."

| Tool | Records | Timeliness | Primary Use |
|------|---------|------------|-------------|
| Logs | Processing-event details | Real-time | Fault investigation, filter-reason analysis |
| Traces | End-to-end path of a single record | Real-time | Data traceability, performance analysis |
| Audit logs | User operation events | Real-time, retained permanently | Compliance auditing, accountability tracing |
| Lineage | Data asset dependency relationships | Updated on each version change | Impact analysis, root-cause identification |

*Table 26-3: Characteristics of Typical Observability Information*

The way these four types of information are combined determines the ceiling of observability. Logs are best suited for explaining what happened within a single processing step; traces for explaining what path a single sample took; audit logs for explaining who performed what operation; lineage for explaining how data assets depend on one another. If a team retains only logs, the blast radius of an incident is still difficult to assess; if only lineage is available without logs, dependency relationships are known but specific processing details are not; without audit logs, questions about permissions, accountability, and compliance cannot be answered.

In a typical troubleshooting session, an engineer might first notice a coverage drop for a certain sample category from a metric alert, then use the lineage graph to identify the affected dataset and its upstream shards, then use traces to inspect the processing nodes traversed by anomalous samples, and finally use logs and audit logs to confirm that a change to a filter rule is the root cause. This workflow illustrates the complementary relationship of the four information types: metrics detect anomalies; lineage scopes the blast radius; traces connect sample paths; logs and audit logs explain specific actions.

To support this kind of combined querying, the platform must maintain consistent correlation keys. Batch ID, sample ID, dataset version, task run ID, code version, and operator ID should be consistent across logs, traces, audits, and lineage. If each system uses its own naming conventions, manual reconciliation is required during troubleshooting, significantly reducing efficiency. Unified IDs are the infrastructure of the observability system — seemingly mundane, but they determine whether cross-system querying is feasible at all.

The retention strategy for these four types of information should also differ. Full-volume DEBUG logs are expensive and should not be retained long-term; batch-level INFO logs and ERROR logs should be retained longer; traces may be sampled for ordinary samples, but should be retained in full for anomalous samples, critical datasets, and high-risk tasks; audit logs should be retained indefinitely and must be immutable; lineage information should be retained permanently alongside data versions. A uniform retention policy results in either wasted cost or insufficient evidence; tiered retention better matches the realities of a data platform.

Additionally, audit logs and observability logs must not be conflated. Observability logs serve engineering troubleshooting and may use sampling for cost and performance reasons; audit logs serve accountability and compliance and must be complete, immutable, and verifiable. Some teams write audit events to standard application logs to save cost, which makes it difficult to prove completeness and authenticity during a review. For critical events such as data access, permission changes, quality rule modifications, and version deletions, a dedicated audit channel should be established.

### 26.2.3 From Job Observability to Data Asset Observability

Traditional infrastructure monitoring (CPU, memory, disk, network) combined with job-status monitoring covers only two dimensions: "compute resource health" and "process health." An LLM data platform also requires coverage of a third dimension: **data asset health**.

The core of data asset health monitoring is establishing "dataset SLOs (Service Level Objectives)" — for each important dataset, defining its quality targets and monitoring rules. The SRE Workbook emphasizes that SLOs should serve actual user experience and error budget decisions; dataset SLOs should likewise serve the real risks of downstream training, evaluation, and business use (Beyer et al. 2018). For example:

```yaml
dataset_slo:
  dataset_id: cs-dialog-sft-zh
  dataset_version: v2.8.0
  contract_id: data-contract-cs-dialog-2024
  owner: data_platform_owner
  steward: cs_domain_data_steward
  severity: P1
  runbook_url: https://internal.example.com/runbooks/cs-dialog-slo
  escalation_policy:
    primary: data-platform-oncall
    secondary: data-owner-oncall
    notify_after: 30m
  slo:
    - metric: duplicate_rate
      threshold: 0.01       # duplication rate < 1%
      window: 7d            # 7-day rolling window
    - metric: blank_rate
      threshold: 0.005      # blank rate < 0.5%
    - metric: iaa_score
      threshold: 0.85       # annotation consistency > 0.85
    - metric: coverage_rate_by_category
      min_coverage:
        refund: 0.05        # "refund" category samples > 5%
        complaint: 0.08     # "complaint" category samples > 8%
  alert_channel: "#data-platform-alerts"
  on_violation: page_on_call
```

This SLO-driven data asset monitoring enables data quality issues to be discovered before training rather than being traced back after model performance has declined. It is important to note that a dataset SLO should not contain only metrics, thresholds, and alert channels; it should also include operational closure fields: `owner` identifies the ultimate responsible party; `steward` identifies the day-to-day maintainer; `runbook_url` points to the troubleshooting guide; `severity` determines the alert tier; `escalation_policy` defines the escalation path; `dataset_version` binds the currently controlled version; and `contract_id` links to the data contract or data product interface. Without these fields, an SLO can detect problems but cannot ensure they are correctly handed off, escalated, and resolved.

Dataset SLO design must be tailored to usage context. An SLO for a general pre-training corpus might emphasize language distribution, deduplication rate, toxic content ratio, and source diversity. An SLO for a customer service SFT dataset might emphasize business category coverage, annotation consistency, historical process retention, and sensitive information desensitization. An SLO for an evaluation set might emphasize contamination prevention, question stability, and version freeze. Applying the same set of metrics to every dataset creates the appearance of uniformity while failing to cover the critical risks specific to each.

Dataset SLOs should also embody error budget thinking. If a dataset repeatedly violates its SLO, the team should pause expanding its use and prioritize fixing quality and process issues; if a dataset consistently meets its SLO, the team may consider increasing the proportion of automated releases or expanding reuse. In this way, the SLO is not merely an alert threshold but a decision basis for data asset governance: stable assets earn greater trust; unstable assets require more scrutiny.

Moving from job observability to data asset observability also means extending the monitoring target from "a single run" to "a long-evolving data product." Once a job run is complete, the value of its task metrics rapidly diminishes; but datasets continue to be used for training, evaluation, reuse, and audit, and their quality status must be tracked over time. A dataset that is healthy today may not be healthy next week; a version that has been frozen does not mean its known limitations do not need to be documented. Data asset observability requires teams to maintain a continuously updated health record for each dataset.

A health record should at minimum include version history, quality trends, SLO compliance status, known issues, downstream usage, incident history, and responsible parties. For high-value data assets, it may also include usage returns, reuse counts, cost inputs, and model performance contributions. This way, data owners can see not only whether a dataset is "usable" but also whether it is "worth continuing to maintain."

---

## 26.3 Alert Strategy, Attribution, and Incident Response

### 26.3.1 Alert Design Principles

Two common pitfalls afflict alert design on data platforms: **too few alerts** (problems are discovered only after they occur) and **too many alerts** (alert fatigue renders even legitimate alerts ignored). Monitoring engineering practice emphasizes that alerts should correspond to actionable responses and should minimize the forwarding of non-actionable noise to on-call personnel (Turnbull 2014; Beyer et al. 2016). A good alert strategy requires striking a balance between sensitivity and precision.

The following five principles guide alert design for LLM data platforms:

**Principle 1: Every alert must be actionable.** When an alert fires, there must be a corresponding remediation action. If a recipient does not know what to do after receiving an alert, that alert should not exist (or should be converted into an informational notification rather than an alert).

**Principle 2: Alerts are tiered by impact.** Problems of different severity should receive different response levels rather than being handled with uniform urgency.

**Principle 3: Avoid alerting on isolated metrics.** Instantaneous fluctuations in a single metric may be noise; composite alerts (multiple metrics anomalous simultaneously) are more reliable. For example, "duplication rate rising" may be normal batch-to-batch variation, but "duplication rate rising + sample volume increasing + data freshness declining" appearing simultaneously strongly suggests a pipeline problem.

**Principle 4: Distinguish static thresholds from dynamic baselines.** Static thresholds (e.g., "alert if blank rate > 5%") are appropriate for metrics with well-defined quality standards. Dynamic baselines (e.g., "today's value is 3 standard deviations above the 30-day mean") are appropriate for metrics whose normal values change over time (e.g., daily processing volume may increase naturally with business growth).

**Principle 5: Alert consolidation.** If a single root cause triggers simultaneous anomalies in multiple metrics, these should be aggregated into one alert rather than generating ten separate notifications.

Alert design also requires a clear boundary between "notification" and "alert." A notification is used to keep stakeholders informed — for example, a low-risk batch has completed, a metric has fluctuated slightly, or a non-critical task is delayed. An alert implies that someone must take action. If all notifications feed into the alert channel, on-call personnel will gradually ignore alerts. A mature platform should classify information into five levels — dashboard display, asynchronous notification, work item, alert, and incident — and escalate progressively according to impact severity.

Dynamic baselines also require careful use. For data with strong seasonality or business cycles, static thresholds generate many false positives. But for compliance, security, and critical quality metrics, dynamic baselines may mask chronic degradation. For example, a low proportion of sensitive-information exposure should not have its threshold relaxed just because the historical mean is low. If coverage of a critical category continuously declines slowly, the dynamic baseline may follow it downward, preventing an alert from ever firing. Critical metrics should therefore be evaluated against absolute thresholds, relative changes, and long-term trends simultaneously.

Alerts should also carry sufficient context. An alert that merely says "duplicate_rate high" has little value. A useful alert should specify which dataset, which version, which batch, the current value, the historical baseline, the scope of impact, recent changes, recommended investigation entry points, and the responsible party. The alert itself is the first piece of material in the troubleshooting process; the more complete its information, the shorter the mean time to acknowledge and the mean time to locate the root cause.

Alert systems also require ongoing review. Each week or month, teams can track metrics such as alert count, false-positive rate, unacknowledged-alert proportion, mean time to acknowledge, mean time to recover, and repeat-alert proportion. A high false-positive rate indicates rules are too sensitive or lack context; a high unacknowledged proportion indicates problems with the notification channel or accountability mechanism; a high repeat-alert proportion indicates root causes have not been resolved. Alert hygiene is itself part of data platform operations.

### 26.3.2 Four-Tier Alert Framework

| Tier | Name | Example Trigger Conditions | Response Time | Notification Method | Remediation Action |
|------|------|---------------------------|---------------|---------------------|--------------------|
| P0 | Critical | Core pipeline fully down; training set accidentally deleted; large-scale compliance violation discovered | Within 15 minutes | Phone + SMS + instant message | Immediately page on-call engineer; initiate incident response procedure |
| P1 | High | Data throughput down > 50%; quality metric deviates significantly from baseline (> 3σ); critical-category data supply interrupted | Within 1 hour | Instant message + email | On-call engineer acknowledges within 1 hour and assesses impact scope |
| P2 | Medium | Data throughput down 20–50%; quality metric slightly off baseline (2–3σ); annotation task backlog exceeds threshold | Within 4 hours | Instant message | Address during that day's working hours |
| P3 | Low | Non-critical metric anomaly; trending warning (e.g., storage utilization rising for 7 consecutive days) | Within 24 hours | Email | Incorporate into that week's task plan |

*Table 26-4: Alert Tiers and Corresponding Remediation Actions*

P0 and P1 alerts must have a manual acknowledgment (ACK) mechanism. High-reliability systems typically use tiered alerting, escalation paths, and clearly assigned on-call responsibilities to shorten mean time to recovery and prevent incidents from escalating while no one is handling them (Beyer et al. 2018; Nygard 2018): the engineer who receives the alert must acknowledge it within the specified time ("acknowledged, handling now"), otherwise it is automatically escalated. P2 and P3 alerts do not require immediate acknowledgment but must be resolved and closed within their SLA windows.

The four-tier framework should also be aligned with the data lifecycle. Supply-interruption alerts at the ingestion stage, filter anomalies at the cleaning stage, consistency drops at the annotation stage, dataset SLO violations before training, and compliance risks after publication all represent data platform issues but with different impact paths. P0/P1 classification should not be based solely on the degree of metric deviation; it must also account for whether the issue affects formal training, critical evaluations, production releases, or compliance commitments. A low-proportion sensitive data exposure may be more severe than a high-proportion duplication issue in ordinary samples.

Alert escalation paths must be specified by role. Platform resource alerts are typically handled by platform engineers; data quality alerts by data engineers and quality assessors jointly; business coverage alerts require participation from data owners and algorithm representatives; compliance alerts must notify security or legal roles. Without role routing, alerts will be forwarded between teams, delaying resolution.

For P0 and P1, communication cadences should be established. The incident commander must provide regular status updates — for example, a sync every 30 minutes covering impact scope, current assessment, next steps, and estimated time to recovery. External communications should avoid premature conclusions but must promptly describe containment actions. Data incidents often affect multiple experiments and teams; without unified communication, the algorithm team may continue to use affected data, amplifying the damage.

### 26.3.3 Anomaly Attribution Decision Tree

When an alert fires, the first question an engineer faces is: where is the root cause? Research on tail latency and cross-layer dependencies in complex systems shows that local normality does not imply global normality, and that attribution must converge layer by layer — from infrastructure, to scheduling, to data sources, to processing logic, to data content (Dean and Barroso 2013; Kleppmann 2017). The following decision tree represents the most common attribution paths on an LLM data platform:

**Step 1: Determine whether this is a platform problem or a data problem**

- Check whether the underlying infrastructure (servers, storage, network) is normal
- Check whether the scheduling system has any task failures or delays
- If infrastructure and scheduling are both normal, the problem is at the data layer

**Step 2 (Data layer): Determine whether this is a source problem or a processing problem**

- Check whether the ingest volume for raw data is normal (is any data source offline?)
- Check whether the proportion from each data source has changed significantly
- If ingest volume and distribution are both normal, the problem is at the processing layer

**Step 3 (Processing layer): Determine whether this is a code problem or a configuration problem**

- Check whether there have been any recent code or dependency library version changes
- Check whether there have been any recent configuration parameter modifications (e.g., quality threshold, filter rules)
- If neither code nor configuration has changed, inspect the data content itself

**Step 4 (Data content): Determine whether this is distribution shift or an annotation quality problem**

- Check whether the quality score distribution across different data sources has changed significantly
- Check whether the IAA of recent annotation batches shows a declining trend
- Sample specific records and manually assess quality

The purpose of the attribution decision tree is not to replace engineering judgment but to provide a stable investigation sequence. During an incident, teams are prone to fixating on a recent change, a personal heuristic, or the most prominent metric and prematurely converging on a hypothesis. A structured decision tree helps engineers first rule out infrastructure and scheduling issues, then proceed to source, processing, and content layers, reducing unstructured investigation.

Attribution should also follow the principle of "contain first, then diagnose precisely." If an alert indicates that corrupt data is actively flowing into the formal training set, the team should pause the relevant pipeline or isolate the affected batches rather than waiting for a complete root-cause determination. Data incidents differ from ordinary service incidents: once corrupt data has been consumed by multiple experiments or versions, the subsequent cleanup cost escalates rapidly. Containment actions may include pausing writes, freezing a version, withdrawing a release candidate, notifying downstream teams to stop consuming, or rolling back to the last known-good dataset.

When investigating data content issues, automatic metrics and manual sampling should be used together. Automatic metrics excel at detecting statistical anomalies but may not assess semantic quality; manual sampling can identify semantic errors but has limited coverage. A good approach is first to use metrics to identify the anomalous sources, categories, batches, or vendors, and then apply stratified sampling within those localized scopes. This avoids blind sampling while compensating for the semantic blind spots of automated detection.

Attribution results should be written back into the issue backlog and knowledge base. Each incident investigation produces valuable troubleshooting paths — for example, a certain type of alert is typically caused by a specific data source going offline; a certain quality score drop is typically associated with a particular annotation task type; a certain dependency upgrade frequently affects specific language processing. Recording these observations in the troubleshooting handbook reduces the time to locate the next similar incident.

![Figure 26-1: Anomaly Attribution Decision Tree](../../images/part8/图26_1zh.png)

*Figure 26-1: LLM Data Platform Anomaly Attribution Decision Tree — Four-Level Diagnostic Path from Alert Trigger to Root-Cause Identification*

### 26.3.4 Tiered Incident Response and Runbooks

A data incident is defined as a situation in which data quality or platform state has deteriorated severely enough to affect the availability or reliability of training data. Incident response should not focus only on rapid repair; it should also document the timeline, impact scope, root cause, and preventive actions to form an organizational learning loop (Beyer et al. 2016; Nygard 2018). The following is a standardized incident response procedure:

**Incident trigger**: A P0 or P1 alert fires, and the problem has not been automatically recovered within 15 or 60 minutes respectively.

**Incident declaration**: The on-call engineer creates an incident ticket and fills in:

- Incident description (what was affected)
- Impact scope assessment (which datasets, which downstream jobs are affected)
- Incident Commander (IC)
- Notification scope (teams and individuals who need to be informed)

**Diagnosis and containment**:

1. Quickly diagnose the root cause following the attribution decision tree (target: root-cause identification within 30 minutes for P0)
2. Assess whether immediate containment is needed (e.g., pausing the affected pipeline to prevent continued production of corrupt data)
3. If a rapid fix is not feasible, assess whether rollback to the last healthy version is needed

**Recovery and verification**:

1. Execute the fix or rollback
2. Run automated quality checks to confirm the data has returned to a healthy state
3. Incident Commander declares the incident resolved

**Post-mortem**: Within 48 hours of incident resolution, complete an incident post-mortem report (format at the end of this chapter in the appendix).

Incident response requires clearly defined roles. The Incident Commander is responsible for overall coordination and decision cadence, not necessarily for personally investigating technical details. The technical lead is responsible for root-cause identification and the fix approach. The communications lead is responsible for status updates to affected teams. The scribe maintains the timeline, key judgments, and action items. In small teams, one person may hold multiple roles, but the responsibilities of each role must still be clear; otherwise, incidents risk having no decision-maker or having multiple people duplicating investigation.

Runbooks should be written as executable steps, not conceptual principles. For example, "check whether the data source is normal" should be elaborated to specify which dashboard to query, which metrics to examine, how to determine that a source is offline, and who to notify when it is offline. "Roll back to the last healthy version" should specify how to confirm which version is healthy, where the rollback command or procedure can be found, and how to verify the result after rollback. The more specific the runbook, the easier it is for on-call personnel to execute it under pressure.

Incident response should also distinguish between recovery and repair. Recovery means returning the system or data to an acceptable state as quickly as possible — for example, pausing the problematic pipeline, rolling back the data version, or reprocessing affected batches. Repair means eliminating the root cause — for example, adding tests, modifying rules, strengthening approvals, and improving monitoring. Many teams close an incident after recovery, leading to repeated occurrences of the same issue. Before formally closing an incident, it must be confirmed that each repair item has an owner, a deadline, and acceptance criteria.

For data incidents, the handling of affected data must also be specified. Whether the affected batches should be deleted, isolated, repaired, re-annotated, or downgraded for limited use must be recorded in the incident log. It is insufficient to fix the pipeline while leaving the already-produced data unaddressed; otherwise, corrupt data may be re-ingested in the future. Incident closure conditions should include: corrupt data has stopped propagating; affected versions have been flagged; repaired data has passed quality verification; and downstream teams have received explicit notification.

---

## 26.4 Capacity Forecasting, Cost Alerting, and Operational Dashboards

### 26.4.1 Three Dimensions of Capacity Forecasting

Capacity forecasting for an LLM data platform must cover three dimensions: processing volume, storage volume, and annotation volume. Designing data-intensive systems typically requires simultaneously considering throughput, latency, storage growth, fault recovery, and cost constraints, since these factors collectively determine platform scalability (Kleppmann 2017).

**Processing volume forecasting**:

Processing volume (the raw data volume to process per day) generally tracks business growth. A simple linear or exponential extrapolation based on historical trends is a starting point, but the following adjustment factors must be layered on:

- Seasonal factors: certain business data has a pronounced time-based pattern (e.g., e-commerce promotional events)
- Project-driven spikes: new project launches or pre-release data preparation for major versions create concentrated bursts of data demand
- Algorithmic evolution: new training paradigms (such as longer context windows or more granular RLHF pipelines) change data consumption patterns

**Storage volume forecasting**:

Storage growth is driven by three factors: newly produced data, the retention policy for historical data, and changes in data format (tokenized data is typically larger than raw data).

Key decision: retention periods for different data types. Raw crawled data: retain 12 months; processed shard data: retain 6 months; published dataset versions: retain permanently; historical experimental data: retain 18 months (see the version granularity table in Chapter 25).

**Annotation volume forecasting**:

Annotation volume forecasting must account for: the algorithm team's experiment roadmap (typically driven by the product roadmap); annotator throughput (annotated samples per person per day); and task complexity (time requirements can vary by up to 10× across different annotation task types).

The output of annotation volume forecasting determines team sizing plans and external vendor resource reservations; it should be completed at the start of each quarter and updated at monthly reviews.

Capacity forecasting should serve not only resource procurement but also data delivery commitments. Insufficient processing volume causes queuing in data cleaning and quality checks; insufficient storage forces teams to delete intermediate artifacts prematurely; insufficient annotation volume slows model iteration. All three capacity constraints ultimately translate into SLA risk for the data team. Capacity forecasting should therefore be coordinated with the demand backlog, version plans, and model training calendar, rather than being estimated by the platform team in isolation.

Processing volume forecasting should focus on peak demand, not just averages. LLM data projects frequently experience phase-specific spikes — intensive data supplementation before a major training run, emergency additions of a particular sample category following an evaluation gap finding, or historical data re-cleaning during a compliance remediation period. If the platform is designed only for average throughput, peak periods will produce task backlogs that in turn affect the training window. A better approach is to simultaneously estimate average processing volume, P95 processing volume, and emergency reprocessing capacity.

Storage forecasting must also consider version strategy. To support reproducibility and auditing, the data platform retains multiple historical versions, shards, and quality reports. More versions mean better traceability but higher storage costs. Platform teams should work with data owners to determine which data must be permanently retained, which can be archived, and which can be deleted after a specified period. Version management without a retention policy will push observability costs to an unsustainable level.

Annotation volume forecasting should incorporate quality factors. Low-quality annotation generates rework, and the actual annotation capacity consumed may far exceed the planned number. Complex tasks also require training, calibration, dispute resolution, and expert review — all of which must be counted as capacity. Simply computing "sample count / mean daily throughput per annotator" typically underestimates the true timeline. For high-value or high-risk tasks, it is preferable to reduce nominal throughput and preserve time for quality review.

Capacity forecasting results should appear in the operational dashboard and trigger advance warnings. For example, if the training plan for the next two weeks requires 3 million cleaned samples but the platform can deliver only 2.2 million at historical throughput, a capacity risk alert should fire in advance. If the backlog days for a certain annotation task type rise continuously, data owners should be notified in advance to adjust priorities or add outsourced resources. The goal of capacity observability is to make resource bottlenecks visible before they affect delivery.

### 26.4.2 Cost Monitoring and Alerting

The main cost categories of an LLM data platform are four. The cost of a cloud-based data system typically arises from a combination of compute, storage, networking, and platform services; cost observability must be designed together with capacity forecasting and retention policies (Nygard 2018):

| Cost Category | Primary Cost Drivers | Optimization Direction |
|---------------|----------------------|------------------------|
| Compute cost | CPU/GPU usage for data processing, format conversion, and quality assessment | Batch consolidation, off-peak scheduling, algorithmic optimization to reduce compute density |
| Storage cost | Storage of raw data, intermediate artifacts, and historical versions | Tiered storage (hot/warm/cold), automatic archival/deletion of expired data |
| Annotation cost | Unit price × annotation volume for outsourced annotation | Improve annotation task design quality to reduce rework; optimize the internal/external annotation ratio |
| Tool and platform cost | Subscription fees for annotation platforms, monitoring tools, and version management tools | ROI evaluation of build vs. buy |

*Table 26-5: Cost Driver and Optimization Direction Design*

Cost alerting should be configured at two levels:

**Absolute value alerting**: When any cost category exceeds 80% of the budget ceiling within a billing period, trigger a P2 alert; when it exceeds 100%, trigger a P1 alert.

**Growth rate alerting**: When the month-over-month growth rate of any cost category exceeds 30% (absent a clear business-growth driver), trigger an investigation request.

Beyond category-level monitoring, a production-grade data platform should establish showback/chargeback attribution — tracing costs to specific data assets, projects, teams, tenants, pipelines, and versions. Knowing only that "storage costs are rising" is insufficient for management decisions; teams also need to know which data asset, project, pipeline, or version strategy is driving the increase. Showback focuses on transparently displaying cost attribution to help teams understand resource consumption; chargeback goes further by charging costs against business or team budgets for resource governance and priority decisions.

Cost attribution records can use unified tags or billing dimensions — for example, `asset_id` identifying the data asset, `project_id` identifying the project, `team` identifying the responsible team, `pipeline_id` identifying the data pipeline generating the cost, `dataset_version` identifying the specific data version, and `cost_center` identifying the financial cost center. For multi-tenant platforms, `tenant_id`, `workspace_id`, or `business_domain` may also be added. These fields should be consistent across task scheduling, storage paths, metadata services, and cloud billing tags; otherwise, subsequent cost aggregation will produce large amounts of unattributable "shared cost."

A typical cost attribution event might contain: `asset_id=cs-dialog-sft-zh`, `project_id=customer-service-llm`, `team=data-platform`, `pipeline_id=cs-dialog-cleaning-dag`, `dataset_version=v2.8.0`, `cost_center=FIN-DATA-LLM`. With these fields, teams can answer questions such as "which data product costs the most," "which project generated the most reprocessing cost," "which version retention policy caused storage growth," and "which team needs to optimize pipeline efficiency."

Cost monitoring should avoid focusing only on the total bill. Rising total cost is not necessarily negative if it corresponds to more high-quality data, faster experiment iteration, or higher reuse value — it is a justified investment. Declining total cost is not necessarily positive if it results from reduced quality checks, premature deletion of historical versions, or insufficient annotation review — it may be introducing risk. Cost observability should present cost alongside output, quality, and risk.

A more informative cost metric is unit cost per effective data record — not simply total cost divided by sample count, but total cost divided by the volume of data that passes quality standards, enters a usable dataset, and is consumed by experiments. This prevents teams from diluting costs with low-quality, high-volume samples. For annotated data, teams may also compute unit cost per qualified annotation, per rework, and per high-value-category sample. The closer a cost metric is to the actual value stream, the better it supports management decisions.

Cost anomalies can also serve as incident signals. A sudden spike in compute cost may indicate task retries or an infinite loop; abnormal storage cost growth may indicate uncleaned intermediate artifacts or duplicate version saves; rising network costs may indicate incorrect cross-region data transfer configuration; rising annotation costs may indicate increased rework rates or unclear task designs. A cost dashboard that is correlated with task, quality, and version metrics can become part of anomaly attribution.

Cost governance must also guard against local optimization. If the platform team deletes intermediate data prematurely to reduce storage costs, it may undermine reproducibility and auditing. If the data team reduces review to cut annotation costs, it may degrade model quality. If the algorithm team reduces controlled experiments to save training costs, it may make data strategies difficult to validate. The observability dashboard should show all parties the trade-off costs of cost reductions, not just display cost decreases.

For budgeting, a layered approach is recommended: a baseline operating budget covering routine ingestion, cleaning, storage, and monitoring; a project incremental budget covering major training runs, dedicated annotation programs, and historical data reprocessing; and a risk reserve budget covering incident recovery, compliance remediation, and emergency capacity expansion. Layered budgets reduce ad hoc approval overhead and allow the data platform to remain elastic during peak periods.

### 26.4.3 Three-Dimensional Operational Dashboard Design

An LLM data platform's operational dashboards must serve audiences with different perspectives: platform engineers focus on stability; data owners focus on quality and efficiency; product and business teams focus on the business value of data assets. Research on data context services emphasizes that governance, lineage, quality, and usage information must be presented within the same context in order to support cross-role decisions (Hellerstein et al. 2017).

Three independent dashboard views are recommended:

**Dashboard 1: Platform Health View (for platform engineers / SRE)**

- Real-time status (green/yellow/red) for all data pipelines
- Task success rate trend over the past 24 hours
- Current queue backlog
- Storage and compute resource utilization
- Count of unresolved P0/P1 alerts

**Dashboard 2: Data Quality View (for data owners / quality assessors)**

- Trend over the past 7 days for key quality metrics (duplication rate, blank rate, annotation consistency)
- SLO compliance status for each dataset
- Summary of data volume and quality produced this week
- High-priority unresolved quality issues in the issue backlog

**Dashboard 3: Business Operations View (for product / algorithm teams)**

- Domain coverage heatmap for the training set
- Data request fulfillment rate (on-time delivery proportion)
- Current month's data iteration progress (against plan)
- Cost vs. output trend (cost efficiency metrics)

The design principle of the three-dimensional operational dashboard is "single source of data, multiple perspectives." The metrics presented to platform engineers, data owners, and business teams may differ, but the underlying data source should be consistent. Otherwise, the same problem appearing with different values in different dashboards erodes team trust. For example, if the Platform Health View shows a task success rate of 99% while the Data Quality View shows a key dataset SLO violation, without a shared batch and version reference teams will struggle to explain the discrepancy.

Operational dashboards should also support a drill-down path from overview to detail. High-level views display red/yellow/green status and trends; clicking on an anomalous metric should reveal the affected dataset, batch, source, recent changes, related alerts, and responsible parties. Dashboards without drill-down capability can only be used for reporting, not troubleshooting. Dashboards with only detail and no overview cannot support management decisions. A well-designed dashboard should simultaneously serve daily health checks, incident investigation, and monthly reviews.

Dashboard labels are also important. Metric names, calculation definitions, update timestamps, threshold meanings, and responsible parties should all be visible within the dashboard. Many monitoring dashboards fail not because they have too few charts but because no one knows how a given metric is actually calculated. For dashboards used across teams, metric definitions matter more than visual styling. When a metric definition changes, it should be marked in the dashboard to prevent misinterpretation of trends.

Operational dashboards should also offer multiple time scales. Minute-level views are used for incident response; hourly or daily views for batch quality observation; weekly views for demand fulfillment and annotation capacity management; monthly or quarterly views for cost, capacity, and data asset value assessment. A single time scale cannot meet the needs of all roles and can easily cause short-term fluctuations to be misread as long-term trends.

Finally, dashboards should not grow without bound. Each view should retain a small number of core metrics and clear action entry points. The Platform Health View is focused on whether on-call action is needed; the Data Quality View on whether data needs to be paused or repaired; the Business Operations View on whether model and product plans are affected. Additional diagnostic metrics can be placed on a dedicated diagnostic page rather than occupying the primary view. The goal of a monitoring dashboard is not to demonstrate how much data the team can collect but to help the team make correct decisions faster.

![Figure 26-2: Data Platform Observability Panorama](../../images/part8/图26_2zh.png)

*Figure 26-2: LLM Data Platform Observability Panorama — Architecture of the Three-Layer Metric Hierarchy and the Three-Dimensional Operational Dashboard*

---

## 26.5 Case Study: Post-Mortem of a Platform Incident

### 26.5.1 Incident Overview

**Time**: Tuesday, May 2024, 13:47

**Alert tier**: P1 (later escalated to P0)

**Incident description**: The data platform quality monitoring system raised an alert: the "medical and health" category coverage in the core training dataset `dialogue-sft-zh` had dropped sharply from a normal level of 8.2% to 1.3%, triggering a P1 alert. During the investigation it was discovered that the same problem affected all data batches over the preceding six days, and the incident was escalated to P0.

**Impact scope**: Among 6 days of incremental data (approximately 420,000 records), roughly 350,000 medical-and-health-category samples were lost; three experiments currently using these batches for training were affected.

At the time of the incident, all job statuses at the platform layer were normal. Crawler jobs ran on schedule; cleaning jobs had no failures; and data writes showed no anomalies. The initial trigger was the category coverage metric, not a job failure alert. This is highly characteristic: data incidents do not always present as system failures — they present as data asset quality degradation. Had the team looked only at scheduling status, this incident might have remained hidden until model performance declined on medical and health Q&A tasks.

The incident also exhibited clear latency. The code change occurred on May 15; the alert fired on May 21, spanning multiple batches. The coverage drop in any single batch had not reached the existing alert threshold, but the cumulative trend was unmistakable. After the incident was exposed, the team recognized that the existing monitoring attended only to point-in-time anomalies and lacked trend detection, as well as error-budget management for critical category coverage.

| Time | Event |
|------|-------|
| May 15, 09:00 | Data engineer Zhang made a "minor optimization" to the crawler filter rules: the keyword list in the filter rule was changed from an external JSON file to hardcoded values, with the goal of reducing configuration file dependencies |
| May 15, 09:15 | The change passed basic unit tests (the test samples did not include the medical and health category) |
| May 15, 10:00 | The change was deployed to the production environment; all subsequent batches were processed using the new logic |
| May 21, 13:47 | The quality monitoring system triggered a P1 alert: medical and health category coverage anomaly |
| May 21, 14:05 | On-call engineer Li received the alert and began investigating |
| May 21, 14:30 | Li used the lineage graph to trace the affected data batches, compared them against the code change history, and suspected a connection to the May 15 change |
| May 21, 14:45 | Zhang confirmed: during the hardcoding process, several key terms from the medical and health category keyword list had been omitted, causing more than 80% of that category's samples to be incorrectly filtered out |
| May 21, 14:50 | Incident escalated to P0; algorithm team notified to suspend use of the affected data batches |
| May 21, 16:30 | Fix deployed; reprocessing of the 6 days of affected data initiated |
| May 22, 08:00 | Data reprocessing complete; quality metrics returned to normal; incident resolved |

*Table 26-6: Case Timeline*

**Defect latency / total impact duration**: From the erroneous change deployment on May 15 at 10:00 to incident resolution on May 22 at 08:00, approximately 6 days and 22 hours. This measure reflects the actual window during which corrupt data had an impact and also reflects the inadequacy of monitoring detection capability.

**Time to recovery after alert**: From the P1 alert firing on May 21 at 13:47 to incident resolution on May 22 at 08:00, approximately 18 hours and 13 minutes. This measure reflects team efficiency in acknowledgment, diagnosis, repair, and reprocessing after the alert fired.

Both time measures are important. The first measures defect latency and total impact duration; the second measures recovery efficiency after the alert. A team with a mature observability framework should work to reduce both mean time to detect and mean time to recover. For data platforms, mean time to detect is often harder to optimize than mean time to recover, because data errors may accumulate slowly in statistical terms.

The timeline also shows that root-cause identification relied primarily on lineage and code change records. The on-call engineer did not manually inspect every batch but first used the lineage graph to confirm the upstream processing chain of the affected dataset, then compared it against the change log for the preceding week. This illustrates that an observability system must not only generate alerts; it must also support directed investigation. Without lineage and change records, the team would have had to manually compare large volumes of logs and output files, significantly extending the time to localize the issue.

### 26.5.2 Root Cause Analysis

**Immediate cause**: When the filter keyword list was moved from an external configuration file to hardcoded values during the code change, several keywords for the medical and health category were omitted.

**Systemic root causes**:

1. **Insufficient test coverage**: The test samples used for the code change did not include medical and health category data, so the tests could not detect the regression.

2. **Monitoring detection lag**: The quality monitoring system ran batch-level checks every six hours, and alert rules were based on per-batch deviation thresholds. The system was insufficiently sensitive to gradual change (the reduction per batch was not significant in isolation), and it took six days of accumulation before the alert fired.

3. **Insufficient change approval rigor**: This change was classified as a "minor optimization" and did not go through the full change approval process (no impact scope review, no rollback checkpoint established).

At a deeper level, the incident exposed three organizational assumptions. First, the team assumed that "replacing configuration with hardcoded values" constituted a low-risk engineering optimization, but in practice it changed the maintenance model for filter rules and the boundaries of quality risk. Second, the team assumed that passing unit tests was sufficient for release, but the test samples did not cover all critical business categories and could not represent the actual training data distribution. Third, the team assumed that batch-level thresholds were sufficient to detect coverage anomalies, while overlooking chronic decline and class dilution.

These assumptions were not the personal failure of any single engineer; they reflect process design deficiencies. A mature post-mortem avoids simplistically attributing an incident to "developer carelessness." The more valuable questions are: Why was a change that could affect critical category coverage classified as a minor change? Why did the test set not cover business categories? Why did monitoring fail to identify the continuous decline? Why did change approval not require an impact analysis? These questions point toward systemic improvement rather than individual blame.

Root cause analysis must also distinguish between the immediate cause, contributing factors, and defensive layers that were never triggered. The immediate cause is the keyword omission; contributing factors include insufficient test coverage, a lightweight approval process, and the absence of trend alerting; the untriggered defensive layers include a data quality smoke test, critical-category regression testing, and a change impact assessment. Enumerating the layers of defense helps the team decide which protective layer to add, rather than adding rules arbitrarily after an incident.

### 26.5.3 Remediation Measures

**Short-term (completed)**:

- Fix the keyword list; reprocess affected data
- Notify the algorithm team to update the dataset version used in affected experiments

**Medium-term (within 2 weeks)**:

- Add representative samples for the medical and health category to the test dataset, covering all business categories
- Add a "trend alert" rule to the quality monitoring system: if a category's coverage rate declines by more than 20% over three consecutive batches, trigger an alert

**Long-term (within 1 month)**:

- Establish a change-tiering policy for data processing: any change affecting filter rules must go through the full approval process
- Add a data quality smoke test to the CI/CD pipeline: after each code change, automatically run quality metric checks against a small reference dataset. Production-grade ML platforms typically treat data validation, model validation, and pipeline metadata as integral parts of the continuous delivery pipeline rather than post-release additions (Baylor et al. 2017; Breck et al. 2019)

The prioritization of remediation measures follows the sequence of contain first, recover next, then prevent. Short-term actions address the corrupt data already produced; medium-term actions improve detection capability; long-term actions change the change management process. This layering is important because a common pitfall in incident handling is fixing only the code without fixing the process, or defining a process without addressing the affected data. A data incident must simultaneously address "how to prevent recurrence" and "how to eliminate the impact that has already occurred."

When reprocessing the affected data, the team also retained an isolated copy of the original erroneous batches for post-mortem analysis and testing. The isolated copies cannot re-enter the training set but can serve as regression test samples to verify whether future filter rules would again incorrectly delete medical and health category data. Converting incident samples into test assets is an important method of continuous improvement for a data platform: every incident should increase the next round of detection capability rather than merely restoring the pre-incident state.

The team also required the three affected experiments to rebind to the repaired data version and record the incident's impact in their experiment cards. This prevents historical experiment conclusions from being misread. If an experiment used affected data without a flag, future algorithm teams might attribute model performance degradation to training strategy rather than insufficient data coverage. The impact records of data incidents must enter the experiment tracking system to support long-term post-mortems.

### 26.5.4 Incident Post-Mortem Template

The following is a standardized incident post-mortem template applicable to all P0/P1 data incidents. Effective post-mortems should focus on systemic improvement rather than individual blame — a principle consistently emphasized in SRE incident post-mortem practice and resilience engineering (Beyer et al. 2016; Beyer et al. 2018):

| Field | Content |
|-------|---------|
| Incident ID | INC-2024-0521-001 |
| Incident tier | P0 |
| Impact window | 2024-05-15 10:00 – 2024-05-22 08:00 (7 days total) |
| Incident Commander | Li (on-call engineer) |
| Participants | Zhang (data engineer), algorithm team representative |
| **Incident description** | Medical and health category samples were massively mis-filtered due to a filter rule change; 6 days of data affected |
| **Impact scope** | 420,000 incremental records; 3 in-progress training experiments |
| **Root cause** | Code change omitted keywords + insufficient test coverage + incomplete change approval process |
| **Response timeline** | See timeline above |
| **Remediation measures** | Short-/medium-/long-term measures as described above |
| **Preventive measures** | Update test dataset; add trend alerting; establish change-tiering policy |
| **Lessons learned** | "Minor changes" are not minor — any change affecting filter logic requires complete test coverage |

*Table 26-7: Incident Post-Mortem Report*

The post-mortem template should be consistent across all P0/P1 incidents, but the content must not be templated. In particular, the "root cause" and "preventive measures" fields must be written to an actionable level. For example, "strengthen testing" is not an acceptable preventive measure; "add 200 fixed regression samples covering all business categories to the filter-rule CI, with the quality lead reviewing the sample set monthly" is an actionable measure. Similarly, "improve monitoring sensitivity" is not an acceptable formulation; "add a trend alert for three consecutive batches with category coverage declining by more than 20%, routed to the data quality on-call" is a verifiable measure.

The post-mortem should also record which signals could have detected the incident earlier. In this case, the medical and health category coverage had been continuously below the 30-day mean since May 16, but no trend alert was configured; on May 17, the filter proportion for a certain medical-related keyword rose anomalously, but that metric was only displayed on the diagnostic dashboard and had not been incorporated into the alert rules. These "near-miss" signals are extremely valuable: they help teams improve alert strategies.

After the post-mortem concludes, action items should be incorporated into the issue backlog rather than remaining in the post-mortem document. Each action item requires an owner, a deadline, acceptance criteria, and an associated incident ID. At the next monthly operations review, it should be verified whether action items have been closed and whether closure has genuinely reduced risk. A post-mortem without action-item tracking is easily reduced to a one-time meeting; one with a tracking mechanism converts into platform capability.

### 26.5.5 Key Metric Improvements

Through the remediation of this incident, the team improved the following monitoring capabilities:

| Improvement Item | Before | After |
|-----------------|--------|-------|
| Detection latency for category coverage anomalies | ~6 days (in this incident, sufficient cumulative deviation was needed before the alert fired) | Target < 6 hours (target detection time after adding trend alerting) |
| Time to detect data quality regressions caused by code changes | Average 4 days | < 2 hours (CI smoke test) |
| Proportion of code changes with quality test coverage | ~35% | > 85% |

*Table 26-8: Metric Improvements*

These metric improvements do not mean that risk has been fully eliminated. The "< 6 hours" in the table is the target detection time established by the newly added trend alerting after remediation; it is not the actual detection time in this incident — in this incident, approximately 6 days elapsed from the erroneous change deployment to the alert. The intent of trend alerting is to advance detection of similar coverage anomalies from "days later" to "within hours," but if business category definitions change, manual review may still be required. CI smoke tests can detect regressions on representative samples but cannot cover all real data distributions. Improved change test coverage also cannot replace code review and impact analysis. Therefore, the post-remediation metrics should be understood as reinforced defensive layers, not as the complete elimination of this incident type.

The central lesson of the case study is that data platform observability must cover *change*. Many incidents do not arise from sudden system failure but from subtle changes in rules, dependencies, data sources, business categories, and usage patterns. An observability system that monitors only static state cannot detect these changes. A system that correlates metrics, logs, lineage, changes, and experiments can convert changes into interpretable signals.

From an implementation perspective, data platform observability should not be pursued as a complete system all at once. A prudent path is to first cover the high-value, high-risk data pipelines and then gradually extend coverage to all data assets. For early-stage teams, the primary objective is to record the task status, quality metrics, and version lineage of the core training set. For mid-size teams, the emphasis is on building alert tiering, SLOs, incident response, and dashboard drill-down capabilities. For platform-scale teams, further work is needed on audit logs, cost observability, capacity forecasting, cross-team operational views, and data asset health records.

Observability construction should also follow a risk-first principle. Not every dataset requires the same monitoring intensity. Formal training sets, critical evaluation sets, production feedback data, and compliance-sensitive data should have more stringent SLOs, alerting, and auditing. Exploratory experimental data may use lighter-weight monitoring but must be clearly marked as ineligible for the formal pipeline. Temporary analytical data needs only basic access and lifecycle records. Tiered monitoring prevents the platform team from being overwhelmed by low-value signals and concentrates resources on the objects that truly affect models and the business.

| Build Stage | Primary Objective | Key Capabilities | Acceptance Question |
|-------------|-------------------|------------------|---------------------|
| Foundation | Make platform problems visible | Task status, basic logs, quality summary, version records | Can you determine when the core dataset is produced and whether it passes basic quality checks? |
| Standard | Make data problems diagnosable | Three-layer metrics, Traces, lineage, tiered alerting, issue backlog | Can you trace from an alert to the affected batch, source, and processing step? |
| Operations | Make risks manageable | Dataset SLOs, incident response, post-mortem loop, operational dashboards | Can you detect trending quality degradation early and coordinate cross-team remediation? |
| Governance | Make assets auditable and manageable | Audit logs, access records, cost dashboards, capacity forecasting, health records | Can you explain the quality, cost, risk, and business value of data assets? |

*Table 26-9: Data Platform Observability Build Stages and Acceptance Questions*

Referring to the build stages table, if the Foundation stage is not yet complete, teams should not rush to build complex operational dashboards. If the Standard stage lacks lineage and Traces, troubleshooting will still be difficult after an alert fires. If the Operations stage lacks a post-mortem loop, incidents will recur. If the Governance stage lacks cost and audit views, platform value will be difficult to explain to management. Every stage should produce verifiable capabilities, not merely deploy new monitoring tools.

In practice, metric definition governance is most commonly underestimated. A metric travels through multiple systems from collection, computation, and aggregation to display. Unclear definitions create new ambiguities. For example, does "data throughput" count raw samples, cleaned samples, or samples that entered the training set? Is "annotation consistency" averaged per task, per annotator, or sample-weighted? Is "data freshness" measured by collection time, ingestion time, or business event time? When definitions are unclear, teams debate numbers rather than acting on problems. Therefore, every core metric should be accompanied by a metric specification sheet. At minimum, a specification sheet should include the metric definition, calculation formula, data source, update frequency, owner, threshold meaning, applicable scenarios, and known limitations. It need not be lengthy, but it must be queryable and version-controlled. When a metric definition changes, the reason and impact scope should be recorded to prevent trend charts from losing comparability without notice.

Another critical issue is the sampling strategy. To control costs, logs and traces often cannot be retained in full for the long term. But if sampling is poorly designed, critical anomalies will be missed. Ordinary low-risk tasks may be sampled proportionally, but high-risk datasets, the period during P0/P1 incidents, anomalous batches, and compliance-sensitive operations should use higher sampling rates or full retention. The sampling strategy should also be dynamically adjustable: when a metric enters a warning state, the system should automatically increase the retention granularity of related pipeline logs and traces to prepare evidence for subsequent investigation. Furthermore, this observability should be integrated into the data change process. Any change affecting data content, filter logic, sampling strategy, annotation guidelines, or quality thresholds should automatically be associated with a subsequent monitoring window. For a period after the change, the relevant metrics should enter a heightened observation state, and the dashboard should mark "recent change present." This way, when metric fluctuations occur, engineers can quickly determine whether they are related to the change. Monitoring without change context easily causes human-induced adjustments to be mistaken for natural variation, and can also cause genuine incidents to be explained away as normal change.

Observability construction also requires a clear division of team responsibilities. Platform engineers are responsible for collecting infrastructure metrics, maintaining log and trace pipelines, and ensuring the stability of dashboards and alerting systems. Data engineers define batch-level quality metrics, manage processing rule changes, and maintain lineage records. Quality assessors interpret quality metrics, design sampling rules, and maintain human evaluation benchmarks. Data owners determine dataset SLOs, alert priorities, and business impact assessments. The algorithm team provides feedback on model performance changes and error sample distributions. Security and compliance teams handle audit logs, access control, and sensitive data risks. Without clear responsibilities, the observability system produces "someone sees the metric, but no one is responsible for acting on it." These responsibilities should be written into day-to-day workflows, not improvised during incidents. Every core dataset should have a data owner; every key SLO should have a metric owner; every P0/P1 alert should have a default on-call role; and every operational dashboard should have a maintainer. Metrics without maintainers will see their thresholds gradually expire; dashboards without maintainers will gradually display inaccurate content; alerts without owners will generate alert fatigue. Observability itself requires governance.

For management, the value of observability investment is reflected not only in fewer incidents but also in higher decision quality. Capacity forecasting lets teams arrange resources in advance; cost monitoring helps teams understand input-output ratios; dataset SLOs tell teams which assets are stable and reliable; lineage and auditing let teams explain risk boundaries. Observability elevates the data platform from "a system that can run" to "a manageable asset portfolio."

Observability, however, has its limits. Monitoring systems can only observe pre-defined or inferrable signals; they cannot automatically understand all business semantics. Some issues still require joint judgment by domain experts, quality assessors, and algorithm engineers. For example, whether certain historical policy data should be retained, whether a particular type of user feedback represents genuine demand, or whether a safety rejection boundary is appropriate — these questions cannot be fully determined by metrics. A mature data platform should treat observability as decision evidence, not as a substitute for decisions.

Finally, teams should regularly conduct data incident drills. A drill can use a simulated scenario — for example, a critical data source going offline, evaluation set contamination, an erroneous filter deletion, annotation consistency degradation, or compliance data being inadvertently included in the training set. Through drills, teams can verify whether alerts fire, whether routing is correct, whether runbooks are executable, whether lineage queries work, and whether communication mechanisms are smooth. Incident response procedures that have never been drilled invariably expose deficiencies only during real incidents. After a drill, a post-mortem should be conducted just as for a real incident. The post-mortem should examine not only the technical system but also personnel response, document accessibility, whether permissions are adequate, dashboard clarity, and whether alert context is complete. Through continuous drills, data platform observability can progress from "having monitoring" to "being able to respond."

---

## Chapter Summary

This chapter systematically constructed an observability framework for LLM data platforms, covering the four dimensions of monitoring, alerting, attribution, and operations.

At the foundational understanding level, we clarified the distinction among schedule success, task success, and data correctness; analyzed the five categories of "silent failure" modes unique to LLM data platforms (semantic drift, over-filtering, dependency drift, annotation decay, and data islands); and examined the root causes of the delayed discovery of data quality problems.

At the metric framework level, we established a three-layer hierarchy of task metrics, quality metrics, and business metrics; discussed the respective roles and combined usage of the four observability tools — logs, traces, audit logs, and lineage; and described SLO-based data asset health monitoring design.

At the alerting and incident response level, we designed a four-tier alert framework (P0–P3) with corresponding remediation actions, provided a four-step decision tree for data platform anomaly attribution, and described a standardized incident response and post-mortem process.

At the operations management level, we discussed three-dimensional capacity forecasting methods for processing volume, storage volume, and annotation volume; cost monitoring classification and alert rules; and the design of three-dimensional operational dashboards serving different audiences.

Finally, through a complete post-mortem walkthrough of a real platform incident, we demonstrated how an observability framework can compress the time from problem occurrence to detection from six days to six hours.

Data platform observability is not a one-time construction effort; it is a continuously improving process driven by platform evolution and accumulated incident experience. Every incident is an opportunity to systematically enhance monitoring capability.

---

## Further Reading

**Classic References in Observability Engineering**

*Site Reliability Engineering* (the Google SRE Book), edited by Beyer et al., is the foundational work in the SRE field. Its chapters on SLO design and incident management are directly applicable to designing a data platform observability framework. Kleppmann's *Designing Data-Intensive Applications* offers a more foundational engineering thinking framework in its discussion of reliability and maintainability.

**Open-Source Data Quality Tools**

Great Expectations is currently the most mature data quality testing framework; it supports defining "data expectations" and automatically running checks within pipelines, integrating well with tools such as Airflow and dbt. Apache Griffin is a data quality tool designed specifically for big data scenarios and supports quality monitoring in both batch-processing and stream-processing modes. Evidently AI is an open-source library focused on ML data and model monitoring, providing ready-to-integrate components for data drift detection and model performance monitoring.

**Incident Management Tools**

PagerDuty is the most widely used incident response tool in the industry, supporting multi-tier alert routing, on-call scheduling, and incident ticket management. Opsgenie is Atlassian's incident management platform, deeply integrated with Jira and well suited for teams already using the Atlassian ecosystem.

---

## References

Amershi S, Begel A, Bird C, DeLine R, Gall H, Kamar E, Nagappan N, Nushi B, Zimmermann T (2019) Software Engineering for Machine Learning: A Case Study. In: Proceedings of the 41st International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP), pp 291-300.

Baylor D, Breck E, Cheng H-T, Fiedel N, Foo C Y, Haque Z, Haykal S, Ispir M, Jain V, Koc L, Koo C Y, Lew L, Mewald C, Modi A N, Polyzotis N, Ramesh S, Roy S, Whang S E, Wicke M, Wilkiewicz J, Zhang X, Zinkevich M (2017) TFX: A TensorFlow-Based Production-Scale Machine Learning Platform. In: Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp 1387-1395.

Beyer B, Jones C, Petoff J, Murphy N R (eds.) (2016) Site Reliability Engineering: How Google Runs Production Systems. O'Reilly Media.

Beyer B, Murphy N R, Rensin D K, Kawahara K, Thorne S (eds.) (2018) The Site Reliability Workbook: Practical Ways to Implement SRE. O'Reilly Media.

Breck E, Cai S, Nielsen E, Salib M, Sculley D (2017) The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. In: IEEE International Conference on Big Data, pp 1123-1132.

Breck E, Polyzotis N, Roy S, Whang S E, Zinkevich M (2019) Data Validation for Machine Learning. In: Proceedings of Machine Learning and Systems 1, pp 334-347.

Dean J, Barroso L A (2013) The Tail at Scale. Communications of the ACM 56(2):74-80.

Hellerstein J M, Sreekanti V, Gonzalez J E, Dalton J, Dey A, Nag S, Ramachandran K, Arora S, Bhattacharyya A, Das S, Donsky A, Fierro G, Kumar C, Mazzariol M, Narayanan S, Parameswaran A, Rahman T, Shah R, She C, Storey M, Turman C, Wu E (2017) Ground: A Data Context Service. In: Proceedings of CIDR.

Kleppmann M (2017) Designing Data-Intensive Applications. O'Reilly Media.

Kreuzberger D, Kühl N, Hirschl S (2023) Machine Learning Operations (MLOps): Overview, Definition, and Architecture. IEEE Access 11:31866-31879.

National Institute of Standards and Technology (2006) Guide to Computer Security Log Management. NIST Special Publication 800-92.

Nygard M T (2018) Release It!: Design and Deploy Production-Ready Software, 2nd Edition. Pragmatic Bookshelf.

Oliner A, Stearley J (2007) What Supercomputers Say: A Study of Five System Logs. In: Proceedings of the 37th Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN), pp 575-584.

OpenTelemetry Authors (2024) OpenTelemetry Specification. Available at: https://opentelemetry.io/docs/specs/

Polyzotis N, Roy S, Whang S E, Zinkevich M (2017) Data Management Challenges in Production Machine Learning. In: Proceedings of the 2017 ACM International Conference on Management of Data (SIGMOD), pp 1723-1726.

Sambasivan N, Kapania S, Highfill H, Akrong D, Paritosh P, Aroyo L M (2021) "Everyone wants to do the model work, not the data work": Data Cascades in High-Stakes AI. In: Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, pp 1-15.

Sculley D, Holt G, Golovin D, Davydov E, Phillips T, Ebner D, Chaudhary V, Young M, Crespo J-F, Dennison D (2015) Hidden Technical Debt in Machine Learning Systems. In: Advances in Neural Information Processing Systems 28, pp 2503-2511.

Sigelman B H, Barroso L A, Burrows M, Stephenson P, Moshchuk A, Osina D, Fikes J, Miller R (2010) Dapper, a Large-Scale Distributed Systems Tracing Infrastructure. Google Technical Report.

Turnbull J (2014) The Art of Monitoring. James Turnbull.

Xu W, Huang L, Fox A, Patterson D, Jordan M I (2009) Detecting Large-Scale System Problems by Mining Console Logs. In: Proceedings of the ACM SIGOPS 22nd Symposium on Operating Systems Principles (SOSP), pp 117-132.
