# Chapter 26: Data Platform Observability

---

## Chapter Overview and Learning Objectives

"The job succeeded" and "the data is healthy" are different statements. SRE practice teaches that system health cannot be judged only by whether a process is running; it must be judged by whether the service continues to meet reliability objectives that users can feel (Beyer et al. 2016). A data platform can show all-green jobs while training-data quality silently degrades. The algorithm team may notice model regressions weeks later, and only then does the data team discover that the problem has been accumulating for a long time.

This chapter explains how to build observability for LLM data platforms. Observability combines metrics, logs, traces, audit records, and context to explain why a system is in its current state, not merely report that state (Sigelman et al. 2010; OpenTelemetry Authors 2024). We cover why job success does not equal data health, layered metrics, logs and lineage, alert design, attribution, incident response, capacity planning, cost alerts, operating dashboards, and a platform-incident postmortem.

By the end of the chapter, readers should be able to design a practical observability system with monitoring layers, alert severity, runbook actions, postmortem templates, and dashboards for platform engineers, data owners, and business teams.

---

## Opening Scenario

An LLM data platform processes about two million raw records every day and produces around 100,000 training samples after cleaning, annotation, and format conversion. Its dashboard looks healthy: Airflow DAG success rate is above 99.2% for the past month, average job time is stable, and storage capacity is sufficient.

During a monthly review, the algorithm team reports that recent training experiments are weaker than expected. Chinese long-document understanding has dropped by around 6%. The data team investigates and finds the cause: three weeks earlier, a dependency in the cleaning pipeline upgraded silently. The new tokenizer changed how Chinese long texts were truncated. Texts longer than 512 tokens were cut incorrectly rather than split at semantic boundaries.

The jobs still succeeded. Output sample counts looked normal. All task metrics were green. But the data content had changed, and no alert was triggered.

This case reveals the core thesis of data-platform observability: **job success cannot be used as a proxy for data health**. Production ML systems often suffer from silent errors and data dependencies that bypass traditional infrastructure monitoring and finally appear as model-quality regressions (Sculley et al. 2015; Polyzotis et al. 2017).

---

## 26.1 Why Job Success Does Not Equal Data Health

### 26.1.1 Scheduling Success, Task Success, and Data Correctness Are Different

Correct observability begins by separating three layers.

**Scheduling success** means the scheduler, such as Airflow or Dagster, triggered the task and put it into the run queue. It only means the task started.

**Task success** means the process exited with code 0 and did not throw an uncaught exception. It means the program completed, but the output can still be wrong: an empty file, a filtering bug that removed everything, or a schema that downstream systems cannot interpret.

**Data correctness** means the produced data is content-healthy and meets quality standards for its intended use. This cannot be inferred from process state. It requires checks on data content, statistics, semantics, and business coverage.

| Layer | What is checked | Typical tool | Common miss |
|---|---|---|---|
| Scheduling success | Whether a task started | Airflow or Dagster status | Task starts and exits immediately with bad output |
| Task success | Process exit code | Monitoring and alerting | Task completes but output is empty or wrong |
| Data correctness | Content quality | Data-quality frameworks | Structurally valid data with semantic errors or drift |

*Table 26-1: Three definitions of success and their common blind spots*

The three layers are nested but not interchangeable. Scheduling success means the control plane works. Task success means the execution plane did not fail explicitly. Data correctness means the business-semantic result is trustworthy. Many teams overestimate platform health because control-plane and execution-plane signals are green.

For LLM data platforms, a task is truly successful only when four conditions hold:

1. It ran at the expected time.
2. It did not exit abnormally.
3. Its output satisfies structural and statistical constraints.
4. Its output has not drifted away from semantic and business expectations.

The first two conditions are platform-engineering concerns. The last two are data-quality and business-governance concerns. All four must be monitored together.

Data correctness is also relative. Historical policy text can be valuable for historical Q&A but risky for current operational guidance. User complaint text can be valuable for robustness training but risky if not de-identified and classified. Data health is not "no error." It is "fit for the declared use."

### 26.1.2 Failure Modes Specific to LLM Data Platforms

LLM data platforms face silent failures: the problem exists, but no obvious error signal appears.

**Semantic drift.** Processing logic stays the same, but source content changes. Crawled pages become ads, or a vendor's labeling style slowly shifts. File format and volume stay normal while meaning changes.

**Over-filtering.** A threshold or rule removes valuable samples. Total output may remain within a normal range, but coverage of a critical category drops.

**Dependency drift.** A library upgrade changes preprocessing behavior, as in the opening scenario. Tests did not cover the boundary case.

**Annotation drift.** Annotator fatigue or interpretation differences gradually lower consistency. A single batch may pass sampling while the trend deteriorates.

**Data island.** One source stops updating, such as a third-party feed ending, but other sources keep total volume stable.

Additional LLM-specific failures include evaluation contamination, category dilution, safety-boundary drift, metadata distortion, and processing-order drift. A sample can have complete fields and valid length but still be unsuitable for training. Metadata can mislabel restricted data as trainable. Changing the order of deduplication and sampling can alter long-tail retention while every job still succeeds.

These failures show that observability must cover data distributions, semantics, metadata, processing rules, dependency versions, and downstream use, not only jobs and services.

### 26.1.3 Why Data Health Problems Often Surface Late

LLM data problems often have delayed impact.

**Data-to-training delay.** A problem must accumulate, enter the training set, pass through training, and then appear in evaluation. This often takes two to six weeks.

**Training-to-deployment delay.** After training, models may go through evaluation, canary release, and full deployment, adding one to four weeks.

The total delay can reach two to three months. If monitoring does not catch problems at the data layer, teams remain in after-the-fact firefighting.

Late discovery is also organizational. Data teams see raw data first, algorithm teams see model effect, product teams see online feedback, and legal teams see compliance risk. If these signals do not meet in one platform, a data-source outage can first appear as coverage loss, then as model regression, and finally as user complaints.

Baselines matter. Daily collection count, category ratio, quality score, and annotation consistency naturally fluctuate. Without historical baselines, engineers cannot distinguish noise from risk. The most valuable monitoring often says "this is getting worse" rather than "this is already broken."

---

## 26.2 Combining Metrics, Logs, Traces, and Lineage

### 26.2.1 Three-Layer Metric System

LLM data metrics should be layered.

**Layer 1: task metrics.** These monitor whether the platform runs on time.

- Task success rate.
- Task duration and P95 duration.
- Queue backlog.
- Retry rate.
- Data throughput by records and bytes.

Task metrics are necessary but not sufficient.

**Layer 2: quality metrics.** These monitor whether output data satisfies quality standards.

- Blank or too-short sample rate.
- Duplicate rate.
- Format compliance rate.
- Language distribution.
- Annotation consistency, such as IAA.
- Average and distribution of quality scores.
- Coverage of critical categories.

Quality metrics should be computed for every batch and stored as time series.

**Layer 3: business metrics.** These monitor whether data supports business goals.

- Domain coverage of the training set.
- Data freshness.
- Annotation throughput of accepted samples.
- Data-demand response rate.
- Percentage of data that passed compliance review.

| Metric layer | Typical metrics | Update frequency | Main audience |
|---|---|---|---|
| Task metrics | Success rate, latency, throughput | Real time or minute-level | Platform engineers and SRE |
| Quality metrics | Blank rate, duplicate rate, consistency, coverage | Batch-level, hourly or daily | Data engineers and quality evaluators |
| Business metrics | Domain coverage, freshness, response rate | Daily or weekly | Data owner, algorithm team, product team |

*Table 26-2: Layered monitoring metrics*

The layers must be linked by the same data version, batch ID, and time window. If task metrics are green but dataset SLOs are violated, the dashboard should make the relationship explainable.

Teams should also separate **health metrics** from **diagnostic metrics**. Health metrics drive alerts. Diagnostic metrics explain alerts. Duplicates, blank rate, coverage, freshness, and consistency are often health metrics. Source ratio, filtering-reason distribution, dependency version, and vendor batch differences are often diagnostic metrics. Alerting on every diagnostic metric creates noise; lacking diagnostics makes incidents hard to locate.

Metric ownership must be explicit. Platform engineers own success rate, backlog, and resource use. Data engineers own throughput and processing failures. Quality evaluators own duplicates, blanks, consistency, and coverage. Data owners own business coverage and demand response. Compliance roles own authorization and access-audit metrics.

### 26.2.2 Logs, Trace, Audit Log, and Lineage

Four observability tools answer different questions.

**Logs** record processing events: how many records were processed, filtered, or failed; why records were rejected; and how long steps took. Logs are detailed and useful for debugging, but full DEBUG logs are expensive to keep. INFO can record batch summaries, DEBUG can be enabled during investigation, and ERROR should be retained longer.

**Trace** records the end-to-end path of a sample through the pipeline: when it entered, which steps processed it, and which dataset version it reached. Trace is the sample-level counterpart to job logs.

**Audit log** records who did what to data and when. It is immutable and compliance-oriented. Typical audit events include dataset creation or deletion, quality-standard change, approval decision, and access request.

**Lineage** records dependencies among data assets. Lineage describes static production relationships; logs and trace describe dynamic process events.

| Tool | Records | Timeliness | Main use |
|---|---|---|---|
| Log | Processing events | Real time | Debugging and filtering-reason analysis |
| Trace | End-to-end sample path | Real time | Sample traceability and performance analysis |
| Audit log | User operations | Real time, permanent | Compliance and accountability |
| Lineage | Data-asset dependencies | Updated on version change | Impact analysis and root cause |

*Table 26-3: Observable information types*

A typical diagnosis starts from an alert metric, uses lineage to identify affected datasets and upstream shards, uses trace to inspect abnormal samples, and uses logs plus audit records to confirm which rule or operation caused the change.

Unified keys are essential: batch ID, sample ID, dataset version, task-run ID, code version, and operator ID must align across logs, traces, audit logs, and lineage. Otherwise every incident becomes a manual join.

Retention policies differ. Full DEBUG logs should not be kept indefinitely. Traces can be sampled for normal data but retained fully for abnormal samples, critical datasets, and high-risk tasks. Audit logs must be immutable and long-term. Lineage should live as long as the data version.

### 26.2.3 From Job Observability to Data-Asset Observability

Traditional infrastructure monitoring covers compute resources and process health. LLM platforms need a third dimension: data-asset health.

A useful pattern is the **dataset SLO**.

```yaml
dataset_slo:
  dataset_id: cs-dialog-sft-zh
  slo:
    - metric: duplicate_rate
      threshold: 0.01
      window: 7d
    - metric: blank_rate
      threshold: 0.005
    - metric: iaa_score
      threshold: 0.85
    - metric: coverage_rate_by_category
      min_coverage:
        refund: 0.05
        complaint: 0.08
  alert_channel: "#data-platform-alerts"
  on_violation: page_on_call
```

Dataset SLOs must match use cases. A pretraining corpus may emphasize language distribution, deduplication, toxicity, and source diversity. A customer-service SFT dataset may emphasize business-category coverage, consistency, historical-flow retention, and sensitive-information removal. An evaluation set emphasizes contamination prevention, item stability, and version freeze.

Data-asset observability also means maintaining a health record for important datasets: version history, quality trends, SLO attainment, known issues, downstream use, incidents, and owners. For high-value assets, the record can include reuse count, cost, and model-effect contribution. The data owner can then judge not only whether a dataset is usable, but whether it is worth continued maintenance.

---

## 26.3 Alerting, Attribution, and Emergency Response

### 26.3.1 Alert Design Principles

Data-platform alerting has two common failure modes: too few alerts, so problems are discovered late; and too many alerts, so nobody trusts them. Good alerts are actionable and prioritized.

Five principles help:

1. **Every alert must have an action.** If the receiver cannot do anything, it should be a notification, not an alert.
2. **Severity follows impact.** Different risks need different response levels.
3. **Avoid isolated metrics.** Combined signals are more reliable than one noisy threshold.
4. **Use both static thresholds and dynamic baselines.** Some metrics have strict limits; others require trend detection.
5. **Deduplicate and group alerts.** One root cause should not create ten separate pages.

Alerts also need context: dataset, version, batch, current value, historical baseline, impact scope, recent changes, recommended investigation entry, and owner. An alert is the first incident document.

Teams should review alert quality regularly: alert count, false-positive rate, unacknowledged rate, mean time to acknowledge, mean time to recover, and repeated-alert ratio. Alerting itself is an operating system that needs maintenance.

### 26.3.2 Four-Level Alert System

| Level | Name | Example trigger | Response time | Notification | Action |
|---|---|---|---|---|---|
| P0 | Critical | Core pipeline outage, training set deleted, large compliance violation | Within 15 minutes | Phone, SMS, IM | Wake on-call engineer and start incident response |
| P1 | High | Throughput drops > 50%, quality metric deviates > 3 sigma, key category source outage | Within 1 hour | IM and email | On-call engineer acknowledges and scopes impact |
| P2 | Medium | Throughput drops 20-50%, quality deviates 2-3 sigma, annotation backlog exceeds threshold | Within 4 hours | IM | Handle during same workday |
| P3 | Low | Noncritical anomaly, trend warning such as storage rising for 7 days | Within 24 hours | Email | Add to weekly plan |

*Table 26-4: Alert levels and handling actions*

P0 and P1 alerts must require human acknowledgment. If the assigned engineer does not acknowledge within the window, escalation triggers automatically. P2 and P3 can be handled within the work SLA, but still need closure.

Alert routing should follow role ownership. Platform-resource alerts go to platform engineers. Data-quality alerts go to data engineers and quality evaluators. Business-coverage alerts involve the data owner and algorithm representative. Compliance alerts must notify security or legal roles.

For P0 and P1 incidents, the incident commander should update affected teams regularly with current impact, hypothesis, next action, and expected recovery time. Without one communication channel, algorithm teams may keep using affected data and expand the damage.

### 26.3.3 Anomaly Attribution Decision Tree

When an alert fires, the first question is where the root cause lives. A stable decision tree prevents ad hoc investigation.

**Step 1: Platform or data?**

- Check infrastructure: compute, storage, and network.
- Check scheduler: failed or delayed jobs.
- If infrastructure and scheduler are normal, move to the data layer.

**Step 2: Source or processing?**

- Check raw-data ingestion volume.
- Check source distribution.
- If volume and distribution are normal, inspect processing.

**Step 3: Code or configuration?**

- Check recent code and dependency changes.
- Check recent configuration changes, such as thresholds and filtering rules.
- If neither changed, inspect content.

**Step 4: Distribution drift or annotation quality?**

- Compare quality-score distribution by source.
- Check recent annotation-batch IAA trends.
- Sample concrete records for human review.

The tree is not a substitute for engineering judgment. It provides a stable order: first exclude infrastructure and scheduling, then source, processing, configuration, and content.

During incidents, follow "contain first, then diagnose precisely." If bad data is flowing into formal training, pause the pipeline or isolate the batch before waiting for full root cause. Data incidents spread through experiments and versions; late containment is expensive.

![Figure 26-1: Anomaly attribution decision tree](../../images/part8/图26_1_异常归因决策树.png)

*Figure 26-1: Four-level diagnosis path from alert trigger to root-cause localization*

### 26.3.4 Data-Incident Response and Runbook

A data incident is a serious data-quality or platform-health anomaly that affects the availability or reliability of training data. Response should restore service, preserve evidence, and improve the system.

Standard incident response:

**Trigger.** A P0 or P1 alert fires and does not self-recover within the defined window.

**Declare.** The on-call engineer creates an incident ticket with description, impact scope, incident commander, and notification list.

**Diagnose and contain.**

1. Use the attribution tree to find the likely root cause.
2. Decide whether to pause the pipeline, freeze the affected version, or stop downstream use.
3. If quick repair is impossible, evaluate rollback to the last healthy version.

**Recover and verify.**

1. Apply repair or rollback.
2. Run automated quality checks.
3. Incident commander declares recovery only after evidence confirms data health.

**Postmortem.** Complete a report within 48 hours.

Incident roles should be explicit: incident commander, technical lead, communication lead, and scribe. Small teams can combine roles, but the responsibilities should remain clear.

Runbooks must be executable. "Check whether the source is normal" is not enough. The runbook should say which dashboard to open, which metrics to read, how to judge outage, who to notify, how to rollback, and how to verify recovery.

Recovery and repair are different. Recovery returns data to an acceptable state. Repair removes the root cause. Incidents should not close until repair actions have owner, deadline, and acceptance criteria.

---

## 26.4 Capacity Forecasting, Cost Alerts, and Operating Dashboards

### 26.4.1 Three Dimensions of Capacity Forecasting

LLM data capacity planning needs to cover processing volume, storage volume, and annotation volume.

**Processing volume.** Daily raw-data processing usually follows business growth, but must be adjusted for seasonality, project-driven bursts, and algorithmic changes such as longer context or more complex RLHF workflows.

**Storage volume.** Storage grows because of new data, retention policy, and format changes. Tokenized data can be larger than raw data. Version management also increases storage because historical datasets, shards, quality reports, and checkpoints are retained.

**Annotation volume.** Annotation capacity depends on the experiment roadmap, annotator throughput, and task complexity. Some annotation tasks can be ten times slower than others. Training, calibration, dispute handling, and expert review must be included.

Capacity forecasting should serve delivery commitments, not only purchasing. Processing bottlenecks delay cleaning, storage limits force early deletion, and annotation shortage slows model iteration. All three become SLA risks.

Forecasts should include average, P95, and emergency reprocessing capacity. LLM data projects often have bursts before major releases, after evaluation failures, or during compliance reprocessing.

### 26.4.2 Cost Monitoring and Alerts

LLM data-platform costs fall into four categories.

| Cost category | Main driver | Optimization direction |
|---|---|---|
| Compute | CPU/GPU used for processing, conversion, and quality evaluation | Batch consolidation, off-peak scheduling, algorithm optimization |
| Storage | Raw data, intermediate artifacts, historical versions | Hot/warm/cold tiers, automatic archive or deletion |
| Annotation | Vendor price times volume | Better task design to reduce rework; balanced internal/external allocation |
| Tools and platform | Annotation platform, monitoring, versioning subscriptions | Build-vs-buy ROI |

*Table 26-5: Cost drivers and optimization directions*

Cost alerts should exist at two levels:

**Absolute alert.** When a cost category reaches 80% of budget in a billing cycle, trigger P2. At 100%, trigger P1.

**Growth-rate alert.** If month-over-month cost grows by more than 30% without a clear business driver, trigger investigation.

Cost observability should not only show total spend. A better metric is cost per effective sample: total cost divided by samples that pass quality, enter usable datasets, and are consumed by experiments. For annotation, useful metrics include cost per accepted label, cost of rework, and cost per high-value category sample.

Cost spikes can also signal incidents. Compute cost may rise because of retries or loops. Storage may rise because intermediates are not cleaned. Network cost may rise because of cross-region transfer. Annotation cost may rise because rework increased.

Cost governance must avoid local optimization. Reducing storage by deleting intermediate data too early can damage reproducibility. Reducing annotation review can lower immediate cost while harming model quality. Dashboards should show cost together with quality, risk, and value.

### 26.4.3 Three-Dimensional Operating Dashboards

Different audiences need different views, built from the same underlying data.

**Platform health view, for platform engineers and SRE.**

- Realtime pipeline status.
- Last-24-hour task success trend.
- Current queue backlog.
- Compute and storage utilization.
- Open P0/P1 alerts.

**Data quality view, for data owners and quality evaluators.**

- Seven-day trend for duplicates, blank rate, and annotation consistency.
- Dataset SLO status.
- This week's data volume and quality summary.
- Open high-priority quality issues.

**Business operations view, for product and algorithm teams.**

- Domain-coverage heatmap of training sets.
- Data-demand response rate.
- Data-iteration progress against plan.
- Cost versus output trend.

The design principle is "same source, different view." If platform health says task success is 99% while dataset quality view says a critical SLO is violated, the dashboard must let people drill into the common batch, version, and time window.

Dashboards need drill-down paths. High-level red/yellow/green status should open affected dataset, batch, source, recent change, related alert, and owner. Without drill-down, dashboards are decorative. Without overview, detailed panels cannot guide management.

Dashboards also need metric definitions: name, formula, data source, update time, threshold, owner, and known limits. Most dashboard failures are definition failures, not chart-style failures.

![Figure 26-2: Platform observability overview](../../images/part8/图26_2_平台可观测性全景图.png)

*Figure 26-2: LLM data-platform observability across metric layers and operating dashboards*

---

## 26.5 Case Study: Postmortem of a Platform Incident

### 26.5.1 Incident Overview

**Time:** Tuesday, May 2024, 13:47.

**Alert level:** P1, later upgraded to P0.

**Description:** The quality-monitoring system alerts that coverage of the "medical health" category in core dataset `dialogue-sft-zh` dropped from a normal 8.2% to 1.3%. Investigation finds that the same issue affected all data batches from the past six days.

**Impact:** Around 420,000 incremental samples were affected; medical-health samples decreased by about 350,000; three training experiments were using affected data.

At the platform layer, every job was normal. Crawlers ran on time, cleaning jobs did not fail, and data writes looked healthy. The issue was first detected by a category-coverage metric, not by a job-failure alert.

The incident also had latency. The code change happened on May 15. The alert fired on May 21. Each individual batch dropped slightly below normal but did not exceed the old threshold. The trend was obvious only across batches.

| Time | Event |
|---|---|
| May 15 09:00 | Data engineer Zhang "optimizes" crawler filtering by moving a keyword list from JSON config into code |
| May 15 09:15 | Basic unit tests pass; samples do not include medical-health category |
| May 15 10:00 | Change deploys to production |
| May 21 13:47 | P1 alert fires: medical-health coverage abnormal |
| May 21 14:05 | On-call engineer Li starts investigation |
| May 21 14:30 | Li uses lineage and code history to suspect the May 15 change |
| May 21 14:45 | Zhang confirms several medical-health keywords were omitted during hardcoding |
| May 21 14:50 | Incident upgraded to P0; algorithm team told to stop affected batches |
| May 21 16:30 | Fix deploys and six days of data are reprocessed |
| May 22 08:00 | Reprocessing completes; quality metrics recover; incident closes |

*Table 26-6: Incident timeline*

From alert to recovery, the team spent about 18 hours. From bad deployment to recovery, the affected window was almost seven days. Both numbers matter: the first measures response; the second measures detection.

### 26.5.2 Root-Cause Analysis

The direct cause was missing keywords in the new hardcoded filtering list. The system therefore filtered out most medical-health samples.

The contributing causes were:

- Test data did not cover every business category.
- Code review treated the change as a small refactor rather than a filtering-rule change.
- Monitoring relied on single-batch thresholds and missed multi-batch trends.
- The filtering-rule change was not connected to a strengthened observation window.

The root cause was not only one engineer's mistake. It was a change-management and test-coverage failure: a content-affecting change passed through the same process as a harmless implementation refactor.

### 26.5.3 Repair Measures

**Short term, within 24 hours**

- Restore the missing keywords.
- Reprocess the six affected days.
- Isolate the bad batches and mark them as invalid.
- Notify algorithm teams to update experiment dataset versions.

**Medium term, within two weeks**

- Add representative samples from all business categories to test data.
- Add trend alerting: if a category drops more than 20% for three consecutive batches, alert.

**Long term, within one month**

- Establish change classification: filtering-rule changes require full review.
- Add data-quality smoke tests in CI/CD for every content-affecting change.

The team also kept an isolated copy of the faulty batch as a regression-test asset. Incident samples should become future detection capability, not disappear after repair.

### 26.5.4 Incident Postmortem Template

The team standardizes the following template for P0/P1 data incidents.

| Field | Content |
|---|---|
| Incident ID | `INC-2024-0521-001` |
| Severity | P0 |
| Impact window | 2024-05-15 10:00 to 2024-05-22 08:00, seven days |
| Incident commander | Li, on-call engineer |
| Participants | Zhang, data engineer; algorithm representative |
| Incident description | Medical-health samples were heavily filtered because of a filtering-rule change |
| Impact scope | 420,000 incremental samples; three active experiments |
| Root cause | Missing keywords, insufficient test coverage, weak change approval |
| Timeline | See incident timeline |
| Repair actions | Short-, medium-, and long-term measures above |
| Prevention | Test dataset update, trend alerts, change classification |
| Lesson | "Small" changes are not small when they affect filtering logic |

*Table 26-7: Incident postmortem report*

Postmortem language must be executable. "Strengthen testing" is not enough. A good action says: "add 200 fixed regression samples covering all business categories to filtering-rule CI, and have the quality owner review the set monthly." Similarly, "improve monitoring" should become a specific alert rule, route, owner, and verification method.

Postmortems should also record signals that almost caught the incident. In this case, medical-health coverage had been below the 30-day average since May 16, and one keyword's filtering count rose on May 17, but neither signal was attached to an alert rule.

### 26.5.5 Key Metric Improvements

The remediation improved monitoring capability.

| Improvement | Before | After |
|---|---|---|
| Category-coverage anomaly detection delay | 6 days | < 6 hours |
| Data-quality regression detection after code change | Average 4 days | < 2 hours through CI smoke test |
| Code changes with quality-test coverage | Around 35% | > 85% |

*Table 26-8: Metric improvements after remediation*

The improvement does not eliminate all risk. Trend alerts can still require human interpretation. Smoke tests cover representative samples but not all real distribution. Higher test coverage does not replace code review and impact assessment. The metrics should be understood as strengthened defense layers, not proof that the incident type is impossible.

The core lesson is that observability must cover **change**. Many incidents come not from sudden system failure but from small shifts in rules, dependencies, sources, categories, or usage. If observability connects metrics, logs, lineage, changes, and experiments, change becomes explainable signal.

Implementation should proceed by risk. Formal training sets, key evaluation sets, online feedback, and sensitive data need stronger SLOs, alerts, and audit. Exploratory data can use lighter monitoring, but must be clearly prevented from entering formal paths.

| Stage | Main goal | Key capabilities | Acceptance question |
|---|---|---|---|
| Basic | Make platform problems visible | Task status, logs, quality summary, version record | Can we tell when core datasets are produced and whether they pass basic checks? |
| Standard | Make data problems diagnosable | Three-layer metrics, trace, lineage, alert severity, issue pool | Can an alert trace to affected batch, source, and processing step? |
| Operating | Make risk manageable | Dataset SLO, incident response, postmortem loop, dashboards | Can we detect quality drift early and coordinate response? |
| Governance | Make assets auditable and operable | Audit logs, permissions, cost panel, capacity forecast, health record | Can we explain quality, cost, risk, and value of a data asset? |

*Table 26-9: Data-platform observability maturity stages*

Metric definitions are often underestimated. "Throughput" might mean raw samples, cleaned samples, or training-ready samples. "Annotation consistency" might be averaged by task, annotator, or sample. "Freshness" might use collection time, ingestion time, or business-event time. Each core metric needs a short metric spec: definition, formula, source, update time, owner, threshold, applicable scenarios, and known limits.

Sampling strategy also matters. Normal low-risk tasks can be sampled, but high-risk datasets, P0/P1 incidents, abnormal batches, and compliance-sensitive operations should use higher or full retention. Sampling should increase automatically when a metric enters warning state.

Finally, data-incident drills are essential. Simulate source outage, evaluation contamination, filtering-rule deletion, annotation consistency decline, or restricted data entering training. Drills test whether alerts fire, routing works, runbooks are executable, lineage queries work, and communication is clear.

---

## Chapter Summary

This chapter built an observability framework for LLM data platforms across monitoring, alerting, attribution, and operations.

At the foundational level, we separated scheduling success, task success, and data correctness, and analyzed silent-failure modes such as semantic drift, over-filtering, dependency drift, annotation drift, and data islands.

At the metrics level, we defined task metrics, quality metrics, and business metrics; explained the roles of logs, trace, audit logs, and lineage; and introduced dataset SLOs for data-asset health.

At the alerting and incident level, we designed a four-level alert system, a four-step anomaly-attribution tree, and a standardized incident-response and postmortem process.

At the operating level, we discussed capacity planning across processing, storage, and annotation; cost monitoring and alerts; and dashboard views for platform engineers, data owners, and business teams.

Finally, the incident case showed how observability can reduce discovery time from days to hours and turn each incident into stronger monitoring capability.

Data-platform observability is not a one-time build. It evolves with platform maturity and incident experience. Every incident is an opportunity to make the next one easier to detect, contain, and explain.

---

## Further Reading

**Observability and SRE.** Google's *Site Reliability Engineering* is a foundational reference for SLO design and incident management. Martin Kleppmann's *Designing Data-Intensive Applications* provides deeper engineering context for reliable, maintainable data systems.

**Open-source data-quality tools.** Great Expectations supports data expectations and pipeline checks. Apache Griffin targets batch and streaming data-quality monitoring. Evidently AI provides practical drift detection and model-monitoring components.

**Incident-management tools.** PagerDuty provides alert routing, on-call scheduling, and incident workflows. Opsgenie integrates well with the Atlassian ecosystem and Jira-based operations.

---

## Next Chapter

Part 8 has now covered team organization, version management, and platform observability. The next part turns to a more sensitive foundation: privacy compliance and data security.

Chapter 27 starts from compliance frameworks and governance, building a full-lifecycle compliance system for LLM data engineering.

## References

Amershi S, Begel A, Bird C, DeLine R, Gall H, Kamar E, Nagappan N, Nushi B, Zimmermann T (2019) Software Engineering for Machine Learning: A Case Study. In: Proceedings of the 41st International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP), pp 291-300.

Baylor D, Breck E, Cheng H-T, Fiedel N, Foo C Y, Haque Z, Haykal S, Ispir M, Jain V, Koc L, Koo C Y, Lew L, Mewald C, Modi A N, Polyzotis N, Ramesh S, Roy S, Whang S E, Wicke M, Wilkiewicz J, Zhang X, Zinkevich M (2017) TFX: A TensorFlow-Based Production-Scale Machine Learning Platform. In: Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp 1387-1395.

Beyer B, Jones C, Petoff J, Murphy N R (eds.) (2016) Site Reliability Engineering: How Google Runs Production Systems. O'Reilly Media.

Beyer B, Murphy N R, Rensin D K, Kawahara K, Thorne B (eds.) (2018) The Site Reliability Workbook: Practical Ways to Implement SRE. O'Reilly Media.

Breck E, Cai S, Nielsen E, Salib M, Sculley D (2017) The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. In: IEEE International Conference on Big Data, pp 1123-1132.

Breck E, Polyzotis N, Roy S, Whang S E, Zinkevich M (2019) Data Validation for Machine Learning. In: Proceedings of Machine Learning and Systems 1, pp 334-347.

Dean J, Barroso L A (2013) The Tail at Scale. Communications of the ACM 56(2):74-80.

Hellerstein J M, Sreekanti V, Gonzalez J E, Dalton M, Dey A, Nag S, Ramachandran K, Arora S, Bhattacharyya A, Das S, Donsky A, Fierro G, Kumar C, Mazzariol M, Narayanan S, Parameswaran A, Rahman T, Shah R, She C, Storey M, Turman C, Wu E (2017) Ground: A Data Context Service. In: Proceedings of CIDR.

Kleppmann M (2017) Designing Data-Intensive Applications. O'Reilly Media.

Kreuzberger D, Kuehl N, Hirschl S (2023) Machine Learning Operations (MLOps): Overview, Definition, and Architecture. IEEE Access 11:31866-31879.

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
