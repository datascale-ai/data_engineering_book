# Project 9: Privacy-preserving Data Pipeline

## Abstract
P09 focuses on governance before sensitive data enters training, analytics, and sharing pipelines.

The chapter is not about a single masking trick.

It organizes control boundaries, sensitive-record handling, operational response, and acceptance checks into a complete privacy-preserving data pipeline.

The project can be read through four main lines.

- Control boundaries and privacy specifications: define compliance scope, classification policy, access boundaries, and technical options.
- Sensitive-record processing chain: perform PII detection, de-identification, isolation, and storage zoning.
- Operations and response loop: include alerts, audit, preflight, incident simulation, and postmortem in the main process.
- Evaluation and acceptance mechanism: verify consistency among code, artifacts, and reports through metrics, deliverables, and check scripts.

In engineering order, the chapter follows this chain.

```text
compliance-scope definition -> classification policy -> access boundary -> sensitive-record processing -> isolation and alerting -> operational preflight -> incident simulation -> metric evaluation -> project checks
```

The core goal is to raise privacy governance from local processing actions to a reproducible, reviewable, and acceptable engineering system.

## Keywords

Privacy preservation; PII detection; de-identification; compliance pipeline; audit response

## Project Goals and Reader Takeaways

This project uses a privacy-preserving data pipeline as the core case.

Its goal is to organize PII detection, classification, redaction, auditing, and checking into a privacy-preserving processing chain.

After completing this chapter, readers should be able to identify key data objects, decompose the engineering chain, set acceptance metrics, and transfer the case method to adjacent data engineering tasks.

## Scenario Constraints and Data Boundaries

The project validates a governance chain with rules and example data.

It does not replace legal advice, DPIA work, or a production-grade privacy platform.

These boundaries make the case reproducible and auditable.

When data scale, data source, permission scope, or deployment environment changes, sampling strategy, quality thresholds, runtime cost, and compliance requirements must be reassessed.

## Architecture Decision

The project adopts an architecture based on privacy specifications, PII detection, risk classification, differentiated redaction, audit logs, and project checks.

This path prioritizes input/output contracts, version traceability, exception localization, and reviewable outcomes.

It does not compress all logic into one disposable script run.

## Sample Schema and Data Flow

The core data flow is as follows.

```text
raw records -> PII detection -> data classification -> redaction/de-identification -> risk labels -> audit and check reports
```

The sample schema should at least retain fields such as `id`, `source`, `content_or_payload`, `metadata`, `quality_signals`, `split_or_stage`, and `audit_trace`.

Concrete fields are refined according to the project's data type, downstream use, and acceptance method.

## Core Implementation Fragments

The chapter keeps only implementation fragments that explain design tradeoffs.

Full scripts, long configurations, run logs, and large files should live in companion resources or appendix notes.

Code examples focus on input/output contracts, quality thresholds, exception handling, and acceptance interfaces.

## Experimental and Acceptance Metrics

Acceptance metrics include PII detection coverage, false-positive and false-negative samples, redacted-field coverage, audit-chain completeness, policy-hit rate, and check pass rate.

If the project enters production, course use, or public reproduction, it should also record version identifiers, dependency environments, random seeds, sample review results, and failure review logs.

*Table P09-1: Privacy Pipeline Publication Acceptance Table*

| Acceptance dimension | Metric / evidence | Publication review criterion |
| --- | --- | --- |
| Identification and processing | PII detection coverage, false-positive/false-negative samples, and redacted-field coverage | Each sensitive-field class should have rule origin, processing action, and review examples. |
| Audit loop | Policy-hit rate, audit-log completeness, and incident/postmortem records | Compliance claims must be traceable to data subject, processing basis, and operator. |
| Compliance boundary | Cross-border transfer, over-redaction, permission misconfiguration, and retention risk | A rule-based prototype must not be described as a complete legal-compliance conclusion. |

## Cost, Risk, and Compliance Boundaries

The main costs come from detection-rule maintenance, manual review, and compliance record keeping.

The main risks are missed redaction, over-redaction, cross-border transfer, and permission misconfiguration.

When external data, personal information, copyrighted content, or third-party services are involved, the project should keep source statements, permission status, redaction strategy, call records, and manual review records.

## Common Failure Modes

Common failures include input-distribution drift, missing schema fields, thresholds that are too loose or too strict, insufficient evaluation coverage, unstable model calls, and results that cannot be traced.

Troubleshooting should first locate data boundaries and intermediate artifacts, and only then inspect models, toolchains, and deployment environments.

## Reproducible Resource Notes

Reproduction materials should include data-source notes, minimal samples, configuration files, run commands, metric scripts, check reports, and artifact directories.

The main text keeps necessary fragments.

Complete notebooks, long scripts, and large files should be maintained separately.

## 1. Project Background: Why a Privacy-preserving Data Pipeline Is Needed

As training data, business logs, and analytical data platforms expand, more teams face the same problem.

Raw data naturally contains identity information, financial information, employee information, medical information, or other high-sensitivity attributes.

At the same time, business and algorithm teams want to move that data quickly into unified platforms for analytics, modeling, or sharing.

The risk is not only whether an email address has been deleted.

The earlier question is whether the whole control chain has been designed correctly.

Are raw records and redacted records stored in the same zone?

Which roles can access raw data, and which roles can only read de-identified results?

Can the system alert when someone bypasses the normal process and starts an export?

If an unauthorized access attempt occurs, does the system have isolation and review mechanisms?

How does the final project prove that these controls worked rather than only existing in a README?

P09 addresses exactly these questions.

According to the project report, its focus is not "perform redaction once".

It builds a privacy processing system that combines classification, permissions, redaction, isolation, audit, preflight, and incident review.

It serves the safety-control need before highly sensitive records enter training or analytical systems.

The project is representative because it shows governance-oriented data engineering rather than a single algorithm.

A mature privacy pipeline is not a regex script.

It is an operating model that defines responsibility boundaries, executes processing actions, and emits verification evidence.

## 2. Project Goals and Boundaries

### 2.1 Project Goals

This project focuses on four goals.

**Goal 1: Build an interpretable privacy specification layer.**

Scenario domain, compliance targets, risk goals, classification levels, access roles, and technical options are written down first.

The project therefore begins with control boundaries rather than text processing.

**Goal 2: Build a processing chain for sensitive records.**

Starting from raw records, the pipeline performs classification, PII detection, de-identification, isolation, and alerting.

The result is a reproducible data processing flow.

**Goal 3: Build an operations and incident-response loop.**

Through preflight, incident simulation, and postmortem, the pipeline shows not only how it runs under normal conditions but also how it responds when something goes wrong.

**Goal 4: Produce verifiable engineering deliverables.**

Final outputs include processed JSON/JSONL artifacts, metric files, the main report, test results, and project check reports.

This ensures that code, artifacts, and narrative remain consistent.

### 2.2 Project Boundaries

The project sets explicit boundaries to preserve reproducibility.

#### 1. Data-Scale Boundary

The current project uses a small sensitive-record sample set.

It contains `8` records.

The purpose is to demonstrate the method chain, not to prove large-scale throughput.

#### 2. Scenario Boundary

The project covers three representative domains: healthcare support, HR payroll, and financial KYC.

These are sufficient to show typical privacy-governance problems.

It does not yet extend to advertising attribution, multinational data flows, multi-tenant SaaS, or training-corpus supply chains.

#### 3. Technical-Implementation Boundary

The project includes differential privacy, TEE, FHE, and related options.

In this case, they mainly appear at the option and architecture-positioning layer.

This does not mean every capability has been deeply implemented.

#### 4. Governance-Capability Boundary

The project already includes classification, isolation, audit, preflight, and incident recovery.

It still has room to expand in cross-system permission linkage, complex export approval, continuous monitoring, and automated exception management.

### 2.3 Why Boundary Statements Matter

The clearer the boundary, the more credible the case.

The project does not need to become a myth that can do everything.

It needs to answer a more useful question.

Under limited samples, limited time, and limited implementation depth, how can a privacy project become a complete loop instead of stopping at conceptual explanation?

## 3. Project Position: P09 in the Capability Chain

In the full capability chain of large-model data engineering, P09 is not positioned at model training itself.

It sits at pre-training governance and safety control before data enters the system.

Many project chapters focus on constructing training data, designing supervision signals, evaluating preferences, optimizing reasoning, or connecting business systems.

P09 answers a different and often underestimated question.

When data itself carries privacy risk, how does the system decide who can see it, what can leave, what must be isolated, what must be logged, and how responsibility is assigned after an incident?

The chapter does not ask how to make the model stronger.

It asks how the data governance chain should be designed as an executable, verifiable, and explainable engineering process when sensitive data enters intelligent systems.

That is the engineering characteristic of P09 as a privacy-governance pipeline.

## 4. Overall Architecture: From Privacy Specification to Project Checks

![Figure 1: P09 Privacy-preserving Data Pipeline Overview](../../images/part10/10_9_fig01_privacy_pipeline_overview.png)

From an engineering perspective, P09 can be divided into three layers.

### 4.1 Layer 1: Policy and Boundary Definition

This layer answers what the system protects and under which rules.

It includes compliance scope, classification policy, access policy, and privacy-technology options.

### 4.2 Layer 2: Data Processing and Control Execution

This layer answers what happens after sensitive data enters the pipeline.

It includes raw-record construction, PII detection, sensitivity-level decisioning, identifier removal, masking and tokenization, restricted-record isolation, and alert/audit emission.

### 4.3 Layer 3: Validation and Operations Loop

This layer answers whether the system is reliable.

It includes preflight checks, incident simulation, postmortem reports, metric evaluation, and project check scripts.

The overall structure can be summarized as policy and scope definition, processing and control execution, and validation and delivery closure.

P09 does not produce instruction data.

It builds a chain from privacy specifications to control execution and governance evidence.

## 5. Engineering Prerequisites: Key Surfaces of a Privacy Pipeline

A privacy pipeline is not a linear scaling of one redaction script.

It is a governance chain composed of control objectives, processing rules, operational mechanisms, and acceptance standards.

### 5.1 Compliance Objectives and Policy Definition

This surface defines compliance goals, sensitivity levels, access boundaries, and violation conditions.

The project begins from control objectives rather than local processing techniques.

### 5.2 Data Processing and Artifact Orchestration

This surface handles pipeline orchestration, JSON/JSONL artifact generation, directory standardization, persisted processing logic, and evaluation-script integration.

It ensures that the processing chain can be reproduced and reviewed.

### 5.3 Security Operations and Response Loop

This surface handles alerts, audits, preflight, incident response, and review chains.

It ensures that controls exist not only in processing logic but also in daily operations.

### 5.4 Evaluation, Verification, and Acceptance

This surface checks whether code compiles, whether reports and metrics agree, whether artifacts are complete, whether redaction is thorough, and whether the overall status is acceptable.

### 5.5 Why These Surfaces Must Come First

The most common failure in privacy projects is often not a broken regex.

It is that key control surfaces were not fixed explicitly.

The policy definition is unclear.

The permission model lacks review.

The exception process has no owner.

Reports and artifacts cannot be reconciled.

When something goes wrong, there is no traceable path for localization.

This means a privacy pipeline is first a fully defined control chain.

It is not a collection of redaction actions.

![Figure 2: Key Engineering Surfaces for P09](../../images/part10/10_9_fig02_roles_and_responsibilities.png)

## 6. Privacy Specification Layer: Rules Before Processing

The first script in P09 is `src/build_privacy_specs.py`.

That fact is already meaningful.

The project does not read data first.

It first generates privacy specifications and policies.

The overall report also recommends this execution order: generate privacy specifications and policies first.

### 6.1 Compliance Scope File

In many projects, the reason a field must be protected is hidden in code comments.

P09 makes this explicit through `compliance_scope.json`.

This has three values.

It aligns project goals with compliance and risk language from the beginning.

It lets evaluation refer directly to scope instead of relying on oral understanding.

It gives the project a clear scope definition instead of an ad hoc script composition.

```python
def build_scope() -> dict:
    return {
        "pipeline_goal": "Build a reproducible privacy-preserving data processing pipeline for highly sensitive records.",
        "example_domains": ["healthcare_support", "payroll_hr", "financial_kyc"],
        "compliance_targets": [
            "least_privilege",
            "auditability",
            "de-identification before analytics",
            "incident response readiness",
        ],
        "risk_goals": [
            "prevent direct PII leakage to analytics consumers",
            "separate raw storage from redacted processing zones",
            "log suspicious access and export attempts",
        ],
    }
```

This structure shows that P09 starts from control objectives rather than a redaction action.

### 6.2 The Pivot Role of Classification Policy

Once classification policy is clear, later control actions have a basis.

The policy states which source types default to restricted, which field patterns require tokenization, which fields should be masked or removed, and whether a default level can still apply when no direct PII is detected.

`build_classification_policy()` organizes these rules as structured objects.

```python
def build_classification_policy() -> dict:
    return {
        "sensitivity_levels": [
            {"level": "restricted", "description": "direct PII, health identifiers, payroll details, bank data"},
            {"level": "confidential", "description": "internal case details and support notes"},
            {"level": "internal", "description": "aggregate metrics and sanitized analytics outputs"},
        ],
        "source_types": [
            {"source_type": "support_ticket", "default_level": "confidential"},
            {"source_type": "employee_payroll", "default_level": "restricted"},
            {"source_type": "kyc_form", "default_level": "restricted"},
            {"source_type": "analytics_export", "default_level": "internal"},
        ],
        "pii_rules": [
            {"pattern_name": "email", "action": "tokenize"},
            {"pattern_name": "phone", "action": "mask"},
            {"pattern_name": "ssn", "action": "remove"},
            {"pattern_name": "bank_account", "action": "tokenize"},
            {"pattern_name": "patient_id", "action": "tokenize"},
        ],
    }
```

### 6.3 Access Policy as a Prior Constraint

Many projects process data first and later add a sentence saying only administrators can access raw data.

P09 generates `access_policy.json` in the specification layer.

Permissions are therefore not a supplementary note.

They are prior constraints.

This is critical because the most expensive privacy error is often not an incomplete mask.

It is that someone who should never have seen the raw data saw it first.

![Figure 3: Four Privacy-Spec Artifact Groups](../../images/part10/10_9_fig03_specs_layer.png)

## 7. Raw Records and Scenario Construction: Governance Coverage with Small Samples

The project currently has only `8` raw records.

That number is small, but the records are not arbitrary.

The report shows that they span `3` domains, `4` data-source types, and `5` role models.

### 7.1 Structural Coverage from Small Samples

The goal is not statistical significance.

The goal is method demonstration.

If the sample covers patient IDs, emails, and phone numbers in healthcare support; SSNs, payroll notes, and payroll cycles in HR; emails, bank accounts, and review status in KYC; and lower-risk aggregate information in analytics export, it is enough to support a minimal reproducible privacy-control case.

### 7.2 How Samples Are Constructed

`build_raw_records()` in `run_privacy_pipeline.py` provides representative data directly.

```python
def build_raw_records() -> list[dict]:
    return [
        {
            "record_id": "rec_001",
            "source_type": "support_ticket",
            "domain": "healthcare_support",
            "owner_team": "care_ops",
            "payload": "Patient John Lee, patient id PT-483920, email john.lee@example.com, phone 415-555-2198 asked about claim denial.",
        },
        {
            "record_id": "rec_002",
            "source_type": "employee_payroll",
            "domain": "payroll_hr",
            "owner_team": "hr_ops",
            "payload": "Employee Marta Chen SSN 342-19-8842 salary adjustment note for payroll cycle 2026-04.",
        },
        {
            "record_id": "rec_003",
            "source_type": "kyc_form",
            "domain": "financial_kyc",
            "owner_team": "fin_ops",
            "payload": "KYC form for beta@corp.test with bank account 998877665544 and risk review pending.",
        },
    ]
```

This style avoids downloading external datasets while preserving the ability to understand the full pipeline.

It sacrifices some real-world complexity in exchange for stronger reproducibility.

### 7.3 Basis for Scenario Samples

Saying that a few samples were prepared is weak.

The important point is why these domains were chosen, which field patterns they cover, and which later controls they exercise.

![Figure 4: Raw Sensitive-Record Scenario Coverage](../../images/part10/10_9_fig04_raw_records_coverage.png)

## 8. PII Detection: Recognition Rules as the Processing Entry

Privacy processing begins with PII detection.

In P09, this step is rule-driven.

Email, phone, SSN, bank account, and patient ID each use independent regular expressions.

### 8.1 Rule-based Detection as the Starting Point

For this case, rule-based detection has three advantages.

It is interpretable because every hit can be traced to a rule.

It is controllable because false positives and misses can be localized to a specific pattern.

It is reproducible because copying the code gives the same result.

```python
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
BANK_RE = re.compile(r"\b\d{10,12}\b")
PATIENT_RE = re.compile(r"\bPT-\d{4,6}\b")


def detect_pii(text: str) -> list[dict]:
    detections = []
    for pattern_name, regex in [
        ("email", EMAIL_RE),
        ("phone", PHONE_RE),
        ("ssn", SSN_RE),
        ("bank_account", BANK_RE),
        ("patient_id", PATIENT_RE),
    ]:
        for match in regex.finditer(text):
            detections.append({"pattern_name": pattern_name, "match": match.group(0)})
    return detections
```

### 8.2 Detection Results as Data Assets

Many projects detect and replace directly without preserving detection structure.

P09 keeps `pii_detections` inside classification results.

This lets later evaluation count field distributions and lets check scripts verify that rules actually worked.

The project therefore moves from "redaction was performed" to "detection evidence was preserved".

### 8.3 What the Current Detection Distribution Shows

The report shows coverage across multiple field patterns: `email = 5`, `phone = 3`, `patient_id = 2`, and `bank_account = 2`.

Even in a small dataset, the project has minimum coverage across fields instead of processing only one identifier type.

![Figure 5: PII Detection Rules and Hit Distribution](../../images/part10/10_9_fig05_pii_detection_distribution.png)

## 9. Classification Logic: Combining Source Type and PII

Robust privacy classification rarely depends only on field content or only on source.

It combines both.

P09's `classify_record()` reflects this.

```python
def classify_record(record: dict, classification_policy: dict) -> dict:
    source_type_map = {
        item["source_type"]: item["default_level"]
        for item in classification_policy["source_types"]
    }
    detections = detect_pii(record["payload"])
    sensitivity = source_type_map.get(record["source_type"], "internal")
    if detections:
        sensitivity = "restricted"
    return {
        **record,
        "sensitivity_level": sensitivity,
        "pii_detections": detections,
        "requires_quarantine": sensitivity == "restricted",
    }
```

### 9.1 What This Logic Solves

It avoids two common mistakes.

If classification uses only source type, unusual content can be missed.

A lower-risk source can suddenly contain an email or account number, and following the default level would lose control.

If classification uses only regex hits, business semantics are ignored.

Payroll or KYC data may remain sensitive even when a direct pattern does not hit.

### 9.2 `requires_quarantine` as a Control Signal

Many projects write classification as only a label.

P09 also writes the control signal `requires_quarantine`.

Classification is therefore not for a pretty report.

It drives later system behavior.

A useful classification does not only tell the system what something is.

It tells the system what should happen next.

### 9.3 What the Current Results Show

The report shows that `7` of the `8` raw records are classified as restricted, and all `7` are isolated.

This matches the scenario choice.

Most sample records intentionally contain high-sensitivity features so that the governance chain is visible.

The goal is not to manufacture many low-risk samples.

![Figure 6: Classification Decision and Quarantine Trigger](../../images/part10/10_9_fig06_classification_and_quarantine.png)

## 10. Redaction and De-identification: Differentiated Strategies

The common simplification in redaction is replacing everything with `***`.

That can look safe, but it creates two problems.

It loses useful analytical structure.

It also fails to express the different control strength needed for different fields.

P09 uses three strategies in `redact_payload()`: tokenize, mask, and remove.

```python
def redact_payload(text: str, detections: list[dict]) -> str:
    redacted = text
    for detection in detections:
        match = detection["match"]
        if detection["pattern_name"] in {"email", "bank_account", "patient_id"}:
            replacement = hash_token(match)
        elif detection["pattern_name"] == "phone":
            replacement = "***-***-" + match[-4:]
        else:
            replacement = "[REMOVED_SSN]"
        redacted = redacted.replace(match, replacement)
    return redacted
```

### 10.1 Division of Labor Among Tokenize, Mask, and Remove

Tokenization fits emails, bank accounts, and patient IDs when the same-entity relationship should remain without exposing the original value.

Masking fits phone numbers when the last digits help operational review but the full value must not remain.

Removal fits SSNs because they are highly sensitive and do not need a reversible or linkable structure in this case.

### 10.2 Why `hash_token()` Matters

The helper script uses `sha256` to generate a stable token.

```python
def hash_token(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"tok_{digest[:12]}"
```

The same original value maps to the same token.

This avoids direct identifier exposure while still supporting weak linkage analysis.

### 10.3 Why Strategy Differences Are Necessary

Privacy engineering should not collapse every decision into a vague phrase such as "redaction".

An engineering-grade description distinguishes the control intent for each field type.

![Figure 7: De-identification Strategies by PII Type](../../images/part10/10_9_fig07_redaction_strategies.png)

## 11. Storage Zones and Isolation: Partitioning Results and Raw Data

One frequent and underestimated privacy risk is storage mixing.

Even after de-identification, if raw records and processed results remain in the same logical zone, much of the risk remains.

P09 uses `build_isolation_plan()` to define four zones: `raw_zone`, `quarantine_zone`, `redacted_zone`, and `audit_zone`.

```python
def build_isolation_plan() -> dict:
    return {
        "zones": [
            {"zone_name": "raw_zone", "store": "encrypted object storage", "access": ["privacy_admin"]},
            {"zone_name": "quarantine_zone", "store": "isolated secure bucket", "access": ["privacy_admin", "incident_responder"]},
            {"zone_name": "redacted_zone", "store": "analytics warehouse", "access": ["data_processor", "analyst"]},
            {"zone_name": "audit_zone", "store": "security log store", "access": ["auditor", "privacy_admin"]},
        ],
        "deid_flow": [
            "ingest raw restricted records",
            "classify and detect PII",
            "write restricted originals to raw_zone",
            "redact identifiers and emit sanitized records to redacted_zone",
            "quarantine flagged export attempts and emit audit alerts",
        ],
    }
```

### 11.1 Why the Zone Model Matters

The zone model binds who can see what to where data is stored.

Only then does the permission boundary move from an abstract statement to storage objects, workflow actions, and role sets.

### 11.2 Why `quarantine_zone` Is a Critical Design

Many projects have `raw_zone` and `redacted_zone` but no `quarantine_zone`.

When abnormal access or suspicious export appears, the system then lacks a middle state that neither continues processing nor discards evidence.

`quarantine_zone` pauses risk spread, gives incident responders investigation space, preserves evidence, and gives exception handling a clear landing point.

### 11.3 What Isolation Results Show

The report shows `7` restricted records and `7` isolated records.

Isolation logic is therefore consistent with classification logic.

It is not a separate and weaker afterthought.

![Figure 8: Storage Zones and Role Access Boundaries](../../images/part10/10_9_fig08_storage_zones.png)

## 12. Audit and Alerts: The Behavior Evidence Chain

A system that only processes data is not necessarily governable.

Sensitive moments often occur when someone tries to bypass a rule.

P09 builds alerts and audit as standalone artifacts so that the system leaves traceable evidence.

### 12.1 How Alerts Are Modeled

`build_alerts()` constructs two typical alerts.

```python
def build_alerts() -> list[dict]:
    return [
        {
            "alert_id": "alert_priv_001",
            "severity": "high",
            "actor": "analyst",
            "reason": "unauthorized raw zone access attempt",
            "status": "resolved",
        },
        {
            "alert_id": "alert_priv_002",
            "severity": "medium",
            "actor": "data_processor",
            "reason": "restricted export requested without approval",
            "status": "resolved",
        },
    ]
```

The two alerts are typical.

One is unauthorized access to the raw zone.

The other is an unapproved request to export restricted data.

They correspond to two of the most dangerous actions in privacy governance.

### 12.2 Why Audit Logs and Alerts Must Appear Together

Alerts tell the system that a risky action occurred.

Audit logs tell the system who did what and when.

The former supports real-time control.

The latter supports after-the-fact tracing.

Without either one, the governance chain is incomplete.

### 12.3 What the Current Metrics Show

The current project has `2` alerts, a `100%` alert resolution rate, and `5` audit events.

It is therefore no longer merely generating redacted files.

It has begun to express security-operations semantics.

![Figure 9: Alerts, Audit, and Incident Response](../../images/part10/10_9_fig09_alerts_and_audit.png)

## 13. Preflight: Checks Before Running

Many data projects have a main processing flow but no pre-run checks.

For a privacy pipeline, missing preflight creates a false impression.

Files may be produced while the prerequisites are not satisfied.

P09 runs preflight before incident simulation in `simulate_privacy_ops.py`.

That order is meaningful.

```python
preflight = {
    "checks": [
        {"name": "all records classified", "passed": len(classified) > 0 and all("sensitivity_level" in item for item in classified)},
        {"name": "restricted records isolated", "passed": all(item["requires_quarantine"] == (item["sensitivity_level"] == "restricted") for item in classified)},
        {"name": "alerts wired", "passed": len(alerts) >= 2},
        {"name": "role model present", "passed": len(access_policy["roles"]) >= 5},
        {"name": "privacy tech options documented", "passed": len(tech_options) >= 4},
    ]
}
```

### 13.1 Preflight Check Design

These checks cover the minimum preconditions for a working pipeline.

Records must be classified.

Restricted records must be isolated.

Alerts must not be empty shells.

The role model must exist.

Privacy-technology options must not be blank.

### 13.2 What the Current Result Shows

The report shows a preflight pass rate of `100%`.

Before operations simulation begins, the project satisfies the minimum prerequisites.

### 13.3 Independent Value of Preflight

Preflight reflects a mature engineering habit.

The project does not run everything and inspect afterward.

It confirms the minimum conditions first, then enters higher-risk processing and drill phases.

![Figure 10: Preflight Check Flow](../../images/part10/10_9_fig10_preflight_checks.png)

## 14. Incident Simulation and Postmortem: The Exception-Response Loop

Many cases only describe success paths.

Privacy governance cannot avoid failure scenarios because credibility often depends on how the system responds when someone crosses a boundary.

P09 writes the incident scenario as a structured record.

An analyst attempts to export restricted raw records without approval.

The fields `detection`, `containment`, `outcome`, and `response_minutes` are preserved explicitly.

```python
incident = {
    "incident_id": "privacy_inc_001",
    "scenario": "analyst attempted to export restricted raw records without approval",
    "detection": "access_alerts.jsonl high severity alert",
    "containment": [
        "quarantine the export request",
        "lock the analyst session",
        "require privacy_admin review",
    ],
    "outcome": "resolved with no confirmed external data leak",
    "response_minutes": 24,
}
```

The corresponding postmortem records root cause, what worked, and follow-up items.

### 14.1 Completeness of the Incident Design

The incident is not a generic statement that a security event occurred.

It decomposes the event chain into how it was detected, how it was contained, why it did not spread, and what should improve afterward.

### 14.2 What the Metrics Tell Us

The report shows `24` minutes of incident response time and `3` postmortem follow-up items.

Operations and review have become measurable results rather than a one-line future extension.

### 14.3 Independent Position of Incident and Postmortem

Including incident and postmortem makes one point clear.

A privacy pipeline is not a static ETL job.

It is a governance system with exception-response capability.

![Figure 11: Incident Response and Postmortem Loop](../../images/part10/10_9_fig11_incident_postmortem.png)

## 15. Evaluation Script: Structured Metric Generation

P09 evaluation is completed by `src/evaluate_privacy_pipeline.py`.

It does not hand-write a summary.

It reads artifacts from the processed directory, computes unified metrics, and generates `p9_metrics.json` and `p9_report.md`.

### 15.1 How Metrics Are Computed from Artifacts

The evaluation stage reads scope, classification, access, technical options, raw records, classified records, redacted records, quarantined records, alerts, audit logs, preflight, incident, and postmortem.

It then computes key results.

```python
metrics = {
    "domain_count": len(scope["example_domains"]),
    "compliance_target_count": len(scope["compliance_targets"]),
    "source_type_count": len(classification["source_types"]),
    "role_count": len(access["roles"]),
    "privacy_tech_count": len(tech_options),
    "raw_record_count": len(raw_records),
    "restricted_record_count": sum(item["sensitivity_level"] == "restricted" for item in classified),
    "quarantine_count": len(quarantined),
    "pii_detection_distribution": dict(Counter(
        detection["pattern_name"]
        for item in classified
        for detection in item["pii_detections"]
    )),
    "direct_pii_removed_rate": direct_pii_removed_rate,
    "alert_count": len(alerts),
    "resolved_alert_rate": round(sum(item["status"] == "resolved" for item in alerts) / max(1, len(alerts)), 4),
    "audit_event_count": len(audit_log),
    "preflight_pass_rate": round(preflight["passed_checks"] / max(1, preflight["total_checks"]), 4),
    "incident_response_minutes": incident["response_minutes"],
    "postmortem_follow_up_count": len(postmortem["follow_ups"]),
}
```

### 15.2 Why `has_direct_pii()` Is Critical

The evaluation script does not only count files.

It also runs regular expressions again against redacted results to check whether direct PII remains.

Evaluation is therefore not merely format checking.

It verifies the core governance objective.

### 15.3 Current Key Metrics

The report shows `3` domains, `4` compliance targets, `4` source types, and `5` roles.

Among `8` raw records, `7` are restricted and all `7` are quarantined.

Direct PII removal rate is `100%`.

Preflight pass rate is `100%`.

There are `2` alerts with `100%` resolution, plus `5` audit events.

These metrics point to one conclusion: P09 is about whether the governance chain is closed, not about data volume.

## 16. Code-to-Artifact Mapping: Scripts and Deliverables

A good chapter should not only list script names.

It should explain what each script does, what it outputs, and who consumes the output.

P09's flow is roughly as follows.

1. `build_privacy_specs.py` generates scope, classification, access, and technical options.
2. `run_privacy_pipeline.py` processes raw records and generates classification, redaction, isolation, audit, and alert artifacts.
3. `simulate_privacy_ops.py` generates preflight, incident, and postmortem artifacts.
4. `evaluate_privacy_pipeline.py` summarizes metrics and the main report.
5. `run_p9_checks.py` runs command-level and data-level checks.

### 16.1 Structural Value of Layered Mapping

The mapping makes the system easy to understand as a layered loop.

Define first, then process, then operate, then evaluate, then accept.

The logic is not hidden inside one large script.

### 16.2 Role of the Deliverable List

The report lists complete deliverables.

- `compliance_scope.json`
- `classification_policy.json`
- `access_policy.json`
- `privacy_tech_options.json`
- `raw_sensitive_records.jsonl`
- `classified_records.jsonl`
- `redacted_records.jsonl`
- `quarantine_records.jsonl`
- `audit_log.jsonl`
- `access_alerts.jsonl`
- `isolation_plan.json`
- `preflight_checklist.json`
- `incident_simulation.json`
- `postmortem_report.json`
- `p9_report.md`
- `p9_metrics.json`
- `p9_test_results.json`
- `p9_test_report.md`

This list matters because it defines completion as a reviewable file asset set, not as a notebook that ran once.

## 17. Check Script: Consistency Among Code, Artifacts, and Reports

Many projects end by saying that the run succeeded.

That is not enough to prove completion.

`run_p9_checks.py` addresses this problem.

### 17.1 What the Check Script Does

It has two classes of checks.

Command-level checks include `py_compile` and re-running `evaluate_privacy_pipeline.py`.

Data and artifact checks verify required files, the role and zone model, whether all records are classified, whether all restricted records are isolated, whether direct PII is removed from redacted outputs, and whether PII rules exist.

The report shows `13` total checks, all passing.

There are `2` command-level checks and `11` data/artifact checks.

The overall status is `PASS`.

### 17.2 Structural Position of the Verification Loop

Project 2 emphasized that code, artifacts, statistics, and reports must agree before a project has really run through.

P09 inherits the same engineering habit even though the task is different.

A data engineering case cannot prove itself only by description.

The check script must become the acceptance device.

### 17.3 Template Value of This Section

Many teams know that they should verify.

They do not know what verification should look like.

P09 gives a minimal template: compile, rerun evaluation, check files, check content, check metrics, and check consistency.

## 18. Metric Interpretation: Closure as the Core Signal

By scale, P09 is restrained.

It has only `8` raw records.

It is still worth a separate chapter because it shows strong closure.

### 18.1 What Closure Means

In this project, closure means that goals and boundaries exist, classification and controls exist, processing and isolation exist, alerts and audits exist, preflight and incident drills exist, metrics and check scripts exist, and reports plus test results exist.

### 18.2 Closure Versus Throughput

A small but complete project is often more educational than a large but vague one.

What transfers is structure, layering, field design, control logic, and verification pattern.

The one-time throughput number is less reusable.

### 18.3 Current Stage Judgment

P09 is closer to a privacy processing operating model than to a single redaction script.

It also acknowledges limited data representativeness, shallow implementation of advanced privacy technologies, and remaining room for cross-system governance.

The project is method-complete, scale-conscious, and explicit about limitations.

## 19. Comparison with Project 2: Supervision Assets and Control Assets

Using Project 2 as a reference for Project 9 is not a mechanical template.

It preserves the most valuable narrative structure.

### 19.1 Similarity: Both Emphasize Engineering Closure

Project 2 describes a legal-domain SFT data factory from seed knowledge, task design, QA, preference pairs, training delivery, and verification closure.

P09 is not an SFT project, but it follows the same discipline.

It defines boundaries first, then enters the main processing chain, and finally performs evaluation and acceptance.

### 19.2 Difference: One Produces Supervision Assets, the Other Produces Control Assets

Project 2's core deliverables are trainable data, preference pairs, and QA records.

P09's core deliverables are control policies, processing results, audit evidence, operations documents, and check reports.

Project 2 answers what the model should learn.

P09 answers how sensitive data can safely enter the system.

### 19.3 Value of the Comparison

The comparison shows a fuller capability map.

Industrial AI engineering is not only training and inference.

It also includes governance before data enters the system.

## 20. Future Extensions: Toward More Realistic Engineering Systems

Three directions are already visible: expand more high-risk scenarios, move advanced privacy technology from planning into implementation, and strengthen automation plus abnormal-access detection.

### 20.1 From Rule Detection to Multi-layer Recognition

Future versions can expand the regex layer into a dictionary and rule layer, a contextual classification layer, a NER/entity-normalization layer, and a combined risk-decision layer.

### 20.2 From Static Policies to Dynamic Governance

The current access policy is mostly static.

Future versions can add task-based temporary authorization, two-person approval, export rate limiting and thresholds, behavioral profiles, and anomaly detection.

### 20.3 From File-Level Delivery to Service-Level Control

Current deliverables are mainly JSON/JSONL and Markdown reports.

The next stage can turn policies into services, audits into real-time streams, and exceptions into workflows.

That would move P09 from a notebook/script project toward a service system.

### 20.4 From Incident Drills to Continuous Exercises

Incident simulation can evolve into regular tabletop exercises, automated fault injection, and red-team/blue-team audit drills.

This improves real resilience.

## 21. Minimal Reproducible Execution Chain

The minimal P09 run order is five steps.

```bash
python src/build_privacy_specs.py
python src/run_privacy_pipeline.py
python src/simulate_privacy_ops.py
python src/evaluate_privacy_pipeline.py
python src/run_p9_checks.py
```

These steps correspond to privacy specification, sensitive-record processing, operations and incident document generation, metric aggregation, and project acceptance.

The structure helps readers understand and reproduce the case in order.

### 21.1 Why the Run Chain Is Listed Separately

The run chain pulls the chapter back from narrative into action.

The most memorable part of a project chapter is often this step-by-step execution path.

![Figure 12: Minimal Reproducible Execution Chain](../../images/part10/10_9_fig12_execution_sequence.png)

## 22. Chapter Summary: System Capabilities Demonstrated by P09

On the surface, P09 looks like a small privacy-preserving project.

Structurally, it demonstrates a complete capability.

It defines governance boundaries before processing text.

It connects classification, de-identification, isolation, alerting, and audit into one control chain.

It includes preflight, incident, and postmortem in the main project flow.

It outputs metrics, reports, and checks instead of stopping at intermediate files.

It uses very small samples to make the organization of privacy engineering clear.

The value of this chapter is not proving that one technical point is advanced.

It shows how a pre-training data governance project can be designed as an explainable, reproducible, and acceptable engineering pipeline.

That is why it works as an engineering case.

## Special Topic: Viewing the Privacy Pipeline as an Operations Manual

The missing part in many privacy-preserving projects is not technical structure.

It is the operations-manual view.

Once the system runs daily, the team needs to know what to inspect first, where to block, and who handles exceptions.

Without this view, the project is just a group of scripts.

With it, the project begins to look like a sustainable governance system.

### 1. What Daily Operations Should Inspect First

For a pipeline like P09, daily operations should first inspect several health signals rather than only final metrics.

New records should all be classified.

Restricted and redacted zones should receive data according to policy.

Direct PII should not appear unexpectedly.

Recent runs should not show rising alerts or preflight failures.

Audit logs should be generated completely.

These signals help the team detect process drift before a real privacy issue occurs.

In privacy systems, once risk enters downstream training or applications, low-cost repair becomes difficult.

Early health signals are therefore especially important.

### 2. Exception Handling Must Have an Explicit Entry Point

One dangerous state in privacy governance is when exception requests are handled through informal conversation.

Without an explicit entry point, teams cannot distinguish reasonable exceptions from risky bypasses.

A more mature operations manual should state which cases may request temporary exceptions, who may request them, who approves them, whether the exception has time and use limits, and whether access is revoked plus audit records are completed afterward.

Writing these procedures protects the team.

Under pressure, a missing institutional entry point often shifts risk to individuals instead of the system.

### 3. Incident Handling Should Follow the Evidence Chain

When a privacy abnormality occurs, the team needs an evidence chain more than an immediate explanation.

P09 already preserves preflight, incident, and postmortem objects.

The next step is linking them into an operations sequence.

Freeze the related version or access path.

Confirm impact scope and data zones.

Extract audit logs, alert records, and processing records.

Then create the incident review and remediation actions.

For privacy projects, handling order matters more than wording.

Correct order usually keeps risk manageable.

Chaotic order can create secondary risk even when technical components exist.

## Special Topic: Metrics and Long-term Evolution of Privacy Governance

P09 is small, but it explains a long-term question well: how should privacy governance be measured?

Without stable metrics, teams easily interpret privacy work as "try not to have incidents".

A mature privacy pipeline should be continuously quantifiable, comparable, and improvable.

### 1. Metrics Should Not Only Count Detection Hits

Teams often first count how many sensitive items were detected.

That is useful, but not enough.

A high hit count can mean strong detection, or simply higher-risk input.

More valuable metrics include record classification coverage, restricted-data isolation correctness, residual direct-identifier rate in redacted data, preflight failure rate and cause distribution, and average time from incident detection to closure.

Together, these metrics better reflect whether the governance system is working.

### 2. Privacy Systems Should Measure Control Effectiveness

From a governance perspective, the core metric is whether controls are effective.

Control effectiveness means how many defined control points actually worked at critical moments.

High-risk records should be routed to restricted zones.

Direct PII should be removed from redacted versions.

Preflight should block unqualified jobs.

Alerts should trigger human attention.

Incident review should produce remediation.

If these controls exist only in documents rather than run results, governance remains fragile.

### 3. Long-term Evolution Moves from Rules to Combined Governance

P09 currently uses rules, policies, and operational processes as a minimal reproducible starting point.

As complexity grows, long-term evolution usually moves toward combined governance.

Rule detection provides fast and interpretable baseline protection.

Contextual classification identifies implicit risks that rules miss.

Permission and approval mechanisms constrain high-risk actions.

Audit and incident review provide after-the-fact traceability and institutional improvement.

Operational rhythm turns governance into continuous practice rather than a one-off project.

When these capabilities work together, the privacy pipeline grows beyond data masking.

It becomes a system for pre-training governance, pre-application governance, and organizational responsibility allocation.

P09's main value is that it already makes the skeleton of this growth path clear.

## Special Topic: Data Subject Requests and Cross-System Cleanup

Real privacy projects also face data subject requests, such as deletion, correction, restriction of processing, or access explanation.

Once data moves through multiple stages before and after training, a subject request is no longer a single action.

It becomes a cross-system cleanup chain.

### 1. The Hard Part of Subject Requests Is Propagation

For one database, deleting a record may be simple.

In a data engineering system, the same sensitive record may have entered raw ingestion, restricted processing, de-identified copies, evaluation samples, and audit or incident records.

The core difficulty is therefore not whether to delete.

It is knowing where the record propagated.

That is why P09 emphasizes zones, audit, and processing records.

They naturally provide handles for later deletion propagation and verification.

### 2. Cross-System Cleanup Requires Both Lineage and Responsibility

When a subject request appears, the team needs two capabilities at the same time.

Lineage helps locate which processing stages the record passed through and what derivatives were created.

Responsibility clarifies who freezes, who cleans, who reviews, and who confirms externally.

Lineage without responsibility shows where the problem is but does not ensure action.

Responsibility without lineage says action is required but cannot tell how deep the cleanup must go.

An executable cleanup chain must bind both.

## Chapter Summary

This chapter used the privacy-preserving data pipeline as a case to show how PII detection, classification, redaction, audit, and checks can be organized into a privacy-preserving processing chain.

Its main value is that task definition, data boundaries, architecture decisions, sample schema, metric acceptance, and reproducible resources are placed in one chain.

The project is not only a set of operating steps.

It becomes a reviewable case study.

The case boundary should be kept clearly.

It validates a governance chain with rules and example data.

It does not replace legal advice, DPIA, or a production privacy platform.

In larger-scale, higher-risk, or more compliance-constrained scenarios, teams should reassess data sources, permission status, manual review ratio, runtime cost, and failure rollback plans.

As part of Part 14, this chapter validates earlier methods at the project level.

Readers can combine it with the data recipes in Part 13, the platform-governance chapters, and the appendix checklists to form a loop from method understanding to engineering delivery.

## References

1. European Union. (2016). Regulation (EU) 2016/679: General Data Protection Regulation.
2. NIST. (2020). NIST Privacy Framework: A Tool for Improving Privacy through Enterprise Risk Management, Version 1.0.
3. Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. Foundations and Trends in Theoretical Computer Science.
4. Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., et al. (2021). Advances and Open Problems in Federated Learning. Foundations and Trends in Machine Learning.
5. OWASP Foundation. (2025). OWASP Top 10 for Large Language Model Applications.
