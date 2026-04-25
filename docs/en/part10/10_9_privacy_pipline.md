# Project 9: Privacy Protection Data Pipeline

## Overview of this chapter

P09 focuses on the governance process before sensitive data enters the training, analysis and sharing links. The focus of this chapter is not on single-point desensitization techniques, but on organizing control boundaries, sensitive record processing, operational response and acceptance mechanisms into a complete privacy protection data pipeline.

This chapter can be understood according to four main lines:

* Control boundaries and privacy specifications: clarify compliance scope, classification policies, access boundaries and technical options.
* Sensitive records processing chain: Complete PII detection, de-identification, isolation and storage partitioning.
* Operation and response closed loop: Incorporate alarms, audits, preflight, incident simulation and postmortem into the main process.
* Evaluation and acceptance mechanism: Verify the consistency of code, products and reports through indicators, deliverables and inspection scripts.

If read in engineering order, this chapter corresponds to a complete link:

**Compliance scope definition -> Classification policy -> Access boundary -> Sensitive record processing -> Isolation and alarm -> Operation and maintenance pre-inspection -> Accident simulation -> Indicator evaluation -> Project inspection**

The core goal of this structure is to upgrade privacy management from partial processing actions to an engineering system that is reproducible, reviewable, and acceptable.

---

## 1. Project background: The necessity of privacy protection data pipeline

As training data, business logs, and analytical data platforms continue to expand, more and more teams will encounter the same problem: raw data naturally contains identity information, financial information, employee information, medical information, or other highly sensitive attributes, but business departments and algorithm departments hope to send the data into a unified platform as soon as possible for analysis, modeling, or sharing.

At this time, the risk is not just "whether the mailbox was deleted", but whether the previous link is designed correctly. for example:

* Whether the original records and desensitized records are placed in the same area;
* Which roles have access to raw data, and which roles can only read de-identified results;
* Whether the system can alert when someone bypasses the regular process and initiates an export;
* If an unauthorized access attempt occurs, does the system have an isolation and recovery mechanism?
* How does the final project prove that these controls really work, instead of just staying in the README.

The core goal of P09 is to solve this type of problem. According to the overall project report, the focus of P09 is not to "do a desensitization", but to build a privacy processing system that organizes classification, permissions, desensitization, isolation, auditing, pre-inspection and accident review. It serves the security control needs of highly sensitive records before they enter training or analysis systems.

This type of project is very representative because it does not demonstrate a single-point algorithm, but a kind of **governed data engineering**:

> A truly mature privacy pipeline is not a regex script, but an operating model that can clarify responsibility boundaries, perform processing actions, and output verification evidence.

---

## 2. Project goals and boundaries

### 2.1 Project Goals

This project focuses on the following four goals.

**Goal 1: Build an explainable privacy specification layer. **
First, write down the scenario domain, compliance objectives, risk objectives, classification levels, access roles and technical options clearly, so that the project does not start with "processing text", but starts with "defining control boundaries".

**Goal 2: Establish a processing link for sensitive records. **
Starting from the original record, classification, PII detection, de-identification, isolation and alerting are completed to form a reproducible data processing process.

**Goal 3: Establish a closed loop of operation and maintenance and incident response. **
Through preflight, incident simulation and postmortem, the pipeline can not only "run in normal times", but also demonstrate "how to respond when something goes wrong".

**Goal 4: Form verifiable project delivery. **
The final output includes not only the processed JSON/JSONL products, but also indicator files, main reports, test results and project inspection reports to ensure that the code, products and narratives are consistent.

### 2.2 Project Boundaries

In order to maintain project reproducibility, this project explicitly sets several boundaries.

#### 1) Data scale boundary

The current project uses a small sample set of sensitive records, with a total of 8 records, focused on demonstrating method links rather than demonstrating large-scale throughput capabilities.

#### 2) Scene boundary

The project currently mainly covers three representative scene areas: medical support, HR salary and financial KYC. These scenarios are sufficient to represent typical issues in sensitive data governance, but have not been extended to more complex environments such as advertising attribution, cross-border data flows, multi-tenant SaaS, or training corpus supply chains.

#### 3) Technical implementation boundaries

The project includes descriptions of technical options such as differential privacy, TEE, and FHE, but they remain more at the "option level" and "architectural site level", which does not mean that these capabilities have been deeply engineered.

#### 4) Boundaries of governance capabilities

The project already has governance links such as classification, isolation, auditing, pre-inspection and accident recovery, but there is still significant room for expansion in terms of cross-system permission linkage, complex export approval, continuous monitoring and automated exception management.

### 2.3 The role of boundary description

The clearer the boundaries, the more credible the case. What is really needed is not a “can-do-anything” project myth, but a methodology that answers the following question:

> Under the premise of limited samples, limited time and limited implementation depth, how to make a privacy project a complete closed loop instead of just a concept description?

---

## 3. Project positioning: P09’s capability chain position

If the entire large model and data engineering capability chain is viewed as a system, then the position of P09 is not the training itself, but **pre-training governance** and **security control** before data enters the system.

Many project chapters focus on:

* How to construct training data;
* How to design supervision signals;
* How to make assessments and preferences;
* How to do inference optimization and business access.

And P09 answers another often underestimated question:

* When the data itself carries privacy risks, how does the system decide who can see it, what can be released, what must be isolated, what needs to be recorded, and how to hold people accountable if something goes wrong?

In other words, what this chapter is going to solve is not "how to train the model to be stronger", but:

> When sensitive data enters an intelligent system, how can the data governance link be designed into an executable, verifiable, and explainable engineering process?

This reflects the engineering characteristics of P09 as a privacy governance pipeline.

---

## 4. Overall architecture: processing pipeline from privacy specifications to project inspections

![Figure 1: P09 privacy protection data pipeline overall architecture](../../images/part10/10_9_fig01_privacy_pipeline_overview.png)

From an engineering perspective, the P09 can be broken down into three layers.

### 4.1 The first layer: strategy and boundary definition layer

This layer answers "What exactly does the system want to protect and what rules should it be protected by?" Mainly include:

* Compliance scope definition (compliance scope)
* Classification policy
* Access policy
* Privacy tech options

### 4.2 The second layer: data processing and control execution layer

This layer answers "what exactly happens after sensitive data comes in." Mainly include:

* Original record structure
*PII detection
* Sensitivity level determination
* Identifier removal, masking and tokenization
* restricted record isolation
* Alarm and audit log output

### 4.3 The third layer: verification and operation and maintenance closed-loop layer

This layer answers "Is the system really reliable?" Mainly include:

* preflight check
* Accident simulation
* postmortem report
* metrics evaluation
* Project check script

This structure can be summarized into three layers: the rule and scope definition layer, the processing and control execution layer, and the verification and delivery closed-loop layer. The focus of P09 is not on the production of command data, but on the complete link from privacy specification generation, control execution to governance evidence deposition.

---

## 5. Pre-engineering: What key aspects need to be clarified first for the privacy pipeline?

The privacy pipeline is not a linear amplification of a single desensitization script, but a governance chain composed of control objectives, processing rules, operating mechanisms, and acceptance criteria.

### 5.1 Compliance objectives and policy definition

This layer is responsible for clarifying compliance objectives, sensitivity levels, access boundaries and violation situations, so that projects start from control objectives rather than local processing techniques.

### 5.2 Data processing and product layout

This layer is responsible for pipeline orchestration, JSON/JSONL product generation, catalog standardization, processing logic placement and linkage with evaluation scripts to ensure that the processing chain can be reproduced and reviewed.

### 5.3 Security operation and response closed loop surface

This layer is responsible for alarm, audit, pre-check, incident response and review links to ensure that control measures exist not only in the processing logic, but also in the daily operation process.

### 5.4 Evaluation, verification and acceptance aspects

This layer is responsible for checking whether the code compiles, whether reports and indicators are consistent, whether the product is complete, whether desensitization is complete, and whether the overall status is within an acceptable range.

### 5.5 Pre-positioning of key aspects

The most common way privacy projects fail is often not that the rules or rules themselves are written incorrectly, but that key control surfaces are not explicitly fixed:

* The strategy is not clearly defined;
* The permission model lacks auditing;
* No one will undertake the exception process;
* Reports and products cannot be aligned;
* There is a lack of traceable positioning path after a problem occurs.

This means that the privacy pipeline is first and foremost a control chain that needs to be fully defined, rather than a splicing of several desensitization actions.

![Figure 2: P09 Privacy Pipeline Key Engineering Area Diagram](../../images/part10/10_9_fig02_roles_and_responsibilities.png)

---

## 6. Privacy specification layer: rule-first processing chain

The first script of P09 is `src/build_privacy_specs.py`. This fact speaks for itself: instead of reading the data, the project first generates privacy specifications and policies. The overall report also clearly gives the recommended execution sequence. The first step is to generate privacy specifications and policies.

### 6.1 Compliance Scope Document

In many projects, "why these fields should be protected" will be implicit in code comments. But P09 makes this explicit as `compliance_scope.json`. There are three values ​​in doing this:

* Align project goals with compliance and risk language from the start;
* Allow subsequent evaluations to reference the scope directly instead of relying on verbal understanding;
* Let the project present a clear scope definition from the beginning, rather than ad hoc scripted splicing.

The corresponding code is as follows:

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

This structure illustrates that the starting point of P09 is not the desensitization action, but the control goal.

### 6.2 The pivotal role of classification strategy

Once the classification strategy is clearly defined, many subsequent control actions will have a basis. for example:

* Which source types are restricted by default;
* Which field patterns require tokenize;
* Which fields are more suitable for mask or direct removal;
* Whether the default level can still be used when no explicit PII is identified.

`build_classification_policy()` organizes these rules into structured objects:

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

### 6.3 Pre-constraints of access policy

Many projects will first process the data and then temporarily write "Only administrators can access the original data." But P09 also generates `access_policy.json` at the specification layer, which means that permissions are not supplementary instructions, but a priori constraints.

This is very critical, because the most expensive mistake in privacy control is often not "the mask is not written completely", but "people who should not see the original data see it first."

![Figure 3: Relationship diagram of the four types of products in the privacy specification layer](../../images/part10/10_9_fig03_specs_layer.png)

---

## 7. Original records and scene construction: small sample governance coverage

The project currently has only 8 original records. The number isn't huge, but it's not just thrown together haphazardly. The overall report shows that these 8 records span 3 scene domains, 4 types of data sources, and correspond to 5 types of role models.

### 7.1 Structural coverage ability of small samples

Because the goal of this project is not to do statistical significance, but to do **method demonstration**. As long as the sample covers:

* Patient id, email, and phone number in medical support;
* SSN, salary notes, salary cycle in HR/payroll;
* Email, bank account and review status in KYC;
* Relatively low-risk aggregated information in analytics export;

Then it is enough to support a minimally reproducible privacy control case.

### 7.2 Sample construction method

`build_raw_records()` in `run_privacy_pipeline.py` gives representative data directly:

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

The advantage of this type of writing is that you can fully understand the pipeline logic without downloading an external data set first. It sacrifices a certain degree of real complexity in exchange for stronger reproducibility.

### 7.3 Construction basis of scene samples

If you only say "prepared some samples", the amount of information is actually very weak. What is more important is to write clearly: why these samples choose these fields, which field patterns are covered, and which subsequent control actions they serve.

![Figure 4: Original sensitive recording scene overlay](../../images/part10/10_9_fig04_raw_records_coverage.png)

---

## 8. PII detection: identification rules as processing entry

Privacy treatment really starts with PII detection. In P09, this step adopts a rule-driven approach: email, phone, SSN, bank account number, and patient ID are matched using independent regular rules.

### 8.1 Rules and Laws as a Starting Point

In this case, the rules approach has three obvious advantages:

* Interpretable: Each match knows what rule it was hit for;
* Controllable: accidental injuries and missed detections can be located to specific patterns;
* Reproducible: You can get the same result by copying the code.

The code is as follows:

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

### 8.2 Detection results as data assets

Many projects will directly replace the code after detection, without retaining the detection structure. But P09 puts `pii_detections` into the classification results, so that subsequent evaluation can count the field distribution, and the checking script can also verify whether the rules are really effective.

This upgraded the project from "desensitizing" to "leaving evidence for detection".

### 8.3 What does the current detection distribution indicate?

The overall report shows that PII detection covers a variety of field patterns, including email=5, phone=3, patient_id=2, and bank_account=2. This illustrates that even in a small-scale data set, the project already has minimal coverage of cross-field patterns, rather than only dealing with a single type of identifier.

![Figure 5: PII detection rules and hit distribution chart](../../images/part10/10_9_fig05_pii_detection_distribution.png)

---

## 9. Classification logic: joint determination of source type and PII

A truly robust privacy classification is often not about "looking only at field content" or "only looking at data sources", but a combination of the two. P09's `classify_record()` reflects this.

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

### 9.1 What problem does this logic solve?

It solves two common mistakes.

First, judging only by source type will miss the abnormal content. For example, if a clear email or account suddenly appears in a source that is supposed to be low risk, if the default level is still used, it will get out of control.

Second, only looking at regular hits will ignore business semantics. For example, certain payroll or KYC data should not be easily classified as low-sensitivity even if there is no direct hit in the current text.

### 9.2 `requires_quarantine` as control signal

Many projects only write the classification result as a label, but P09 further writes the control action signal `requires_quarantine`. This means that classification is not for reporting purposes but to drive subsequent system behavior.

This is very important in engineering because:

> A truly usable classification does not just tell you "what it is", but tells the system "how to deal with it next".

### 9.3 What do the current results indicate?

The overall report shows that 7 of the 8 original records were judged as restricted, and 7 of them entered quarantine. This matches the scenario selection of the project: most of the records in the sample set are inherently highly sensitive, and the purpose is to clearly display the governance links, rather than deliberately creating a large number of low-risk samples.

![Figure 6: Relationship diagram between classification judgment and isolation trigger](../../images/part10/10_9_fig06_classification_and_quarantine.png)

---

## 10. Desensitization and de-identification: differentiated de-identification strategy

When many teams do desensitization, the most common simplification is to "replace everything with ***". Although this approach seems safe, it will cause two problems:

* Loss of necessary analysis structures;
* Unable to reflect the different processing intensities that different fields should have.

P09 uses three strategies in `redact_payload()`: tokenize, mask and remove.

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

### 10.1 Division of labor among tokenize, mask and remove

* **tokenize** is suitable for fields such as email, bank account, and patient ID that need to retain "same entity consistency" but should not expose the original value;
* **mask** is suitable for fields such as phone numbers where retaining the last bit is helpful for operation and maintenance verification but cannot retain the full value;
* **remove** is suitable for fields such as SSN that are highly sensitive and do not need to retain the back-pointing structure.

### 10.2 What is the meaning of `hash_token()`

Use `sha256` in the auxiliary script to generate stable tokens:

```python
def hash_token(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"tok_{digest[:12]}"
```

The advantage of this is that the same original value will be mapped to the same token, which can avoid direct exposure of the original identifier and support subsequent weak association analysis.

### 10.3 The necessity of strategic differences

Because the most taboo thing in privacy engineering is to write all questions into a vague "desensitization treatment." To truly have engineering meaning, the control intentions of different fields must be distinguished.

![Figure 7: De-identification strategy diagram for different PII types](../../images/part10/10_9_fig07_redaction_strategies.png)

---

## 11. Storage partitioning and isolation: Partition control of results and original data

In privacy governance, a very frequent but often underestimated problem is: even if you do de-identification, if the original records and processing results are still mixed in the same logical area, many risks still exist.

P09 explicitly gives four types of zones through `build_isolation_plan()`: raw_zone, quartine_zone, redacted_zone and audit_zone.

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

### 11.1 Why the zone model is important

Because it binds "who can see what" and "where the data should be placed." Only in this way are permission boundaries not declared abstractly, but implemented together with storage objects, workflow actions, and role collections.

### 11.2 Why quarantine_zone is a key design

Many projects only have raw_zone and redacted_zone, but not quarantine_zone. The problem is: when abnormal access or suspicious export is discovered, the system lacks an intermediate state that "neither continues processing nor directly discards".

The meaning of quarantine_zone is:

* Pause risk spread;
* Leave room for investigation by incident responders;
* Maintain chain of evidence;
* Let the exception process have a clear landing point.

### 11.3 What do the isolation results indicate?

The overall report shows that there are currently 7 restricted records and 7 quarantine records. This shows that the logic of isolation is consistent with the logic of classification, rather than "categorization belongs to classification, and isolation is another matter."

![Figure 8: Storage partition and role access boundary diagram](../../images/part10/10_9_fig08_storage_zones.png)

---

## 12. Audit and Alarm: Behavior Evidence Chain

A system that only “processes data” does not mean a governable system. Because the really sensitive moments are often not dealt with on a daily basis, but when someone tries to circumvent the rules. P09 Building alarms and audits into separate products is precisely to allow the system to leave traceable evidence.

### 12.1 How alarms are modeled

`build_alerts()` constructs two typical alarms:

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

These two alarms are very typical: one is unauthorized access to the original area, and the other is an unauthorized request to export restricted data. They correspond to the two most dangerous types of actions in privacy governance.

### 12.2 Why audit logs and alarms must appear together

Alarms tell the system "risky actions have occurred", and audit logs tell the system "who did what and when". The former is more about real-time control, and the latter is more about after-the-fact tracking. Without any one of them, the governance chain will be incomplete.

### 12.3 What does the current indicator reflect?

The overall report shows that the current project has 2 alarms, an alarm resolution rate of 100%, and 5 audit events. This shows that the project is no longer "generating some desensitized files", but has begun to have security operation semantics.

![Figure 9: Alarm, audit and event response relationship diagram](../../images/part10/10_9_fig09_alerts_and_audit.png)

---

## 13. preflight: pre-run check

Many data projects only deal with the main process and have no pre-run checks. However, if the privacy pipeline lacks preflight, it is easy to create an illusion: it seems that all files can be output, but in fact the conditions are not met.

P09 In `simulate_privacy_ops.py`, do preflight first and then do incident simulation. This sequence is very engineering.

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

### 13.1 preflight check item design

Because they cover the "minimum prerequisite for the establishment of an assembly line":

* Classified;
* restricted is really isolated;
* Alarms are not empty shells;
* Character models are not missing;
* Privacy technology options are not blank.

### 13.2 What do the current results indicate?

The overall report shows that the preflight pass rate is 100%. This means that the project meets at least the minimum prerequisites before the operation and maintenance simulation can begin.

### 13.3 Independent value of preflight

Because this reflects a mature engineering habit:

> It is not "see after running", but "confirm that the minimum conditions are met first, and then enter the processing and drill stage of higher risks".

![Figure 10: preflight inspection flow chart](../../images/part10/10_9_fig10_preflight_checks.png)

---

## 14. incident simulation and postmortem: abnormal response closed loop

Many cases only talk about the path to success and not the path to failure. However, privacy governance cannot avoid failure scenarios, because what really determines whether the system is trustworthy is often "how the system responds when someone crosses the boundary."

P09 Write incident scenarios as structured records: The analyst attempted to export restricted raw records without approval, where detection, containment, outcome, and response_minutes were all explicitly retained.

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

The corresponding postmortem continues to record root cause, what_worked and follow_ups.

### 14.1 Integrity of event design

Because it does not generally write "a security incident occurred", but breaks the event chain into:

* How to be discovered;
* How to be contained;
* Why there is no spread;
* What improvements should be made in the future?

### 14.2 What indicators tell us

The overall report shows that the incident response took 24 minutes and the postmortem follow-up count was 3. This shows that the project has written operation and maintenance and review into measurable results, instead of just leaving a sentence of "future scalability".

### 14.3 Independent positions of incident and postmortem

By including incident and postmortem, it is easier to see one thing: the privacy pipeline is not a static ETL, but a governance system that includes exception response capabilities.

![Figure 11: Incident response and postmortem closed-loop diagram](../../images/part10/10_9_fig11_incident_postmortem.png)

---

## 15. Evaluation script: structured generation of indicators

Evaluation of P09 was completed by `src/evaluate_privacy_pipeline.py`. Instead of writing a summary manually, it directly reads various products in the processed directory, calculates unified indicators and generates `p9_metrics.json` and `p9_report.md`.

### 15.1 How are indicators calculated from products?

The evaluation phase will first read all products such as scope, classification, access, tech options, raw/classified/redacted/quarantined, alerts, audit, preflight, incident, postmortem, etc., and then calculate the key results.

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

### 15.2 Why `has_direct_pii()` is critical

The evaluation script does not simply count the number of files, but also uses regular expressions to check whether direct PII remains in the redacted results. This means that evaluation is not a “format check” but rather a verification of results against core governance objectives.

### 15.3 Current key indicators

According to the overall report, the key results of the current project include: 3 scenario domains, 4 compliance targets, 4 types of data sources, and 5 types of roles; 7 of the 8 original records are restricted and 7 are isolated; the direct PII removal rate is 100%; the preflight pass rate is 100%; 2 alarms and the resolution rate is 100%; and 5 audit events.

These indicators jointly point to a judgment: the focus of P09 is not on the amount of data, but on whether the governance chain is closed. The overall report also makes this clear.

---

## 16. Code-product mapping: correspondence between scripts and deliverables

A good chapter should not just list the name of the script, but also clearly state "what the script does, what it produces, and who consumes it." The overall process of P09 is roughly as follows:

1. `build_privacy_specs.py` Generate scope, classification, access and technical options;
2. `run_privacy_pipeline.py` processes original records and generates classification, desensitization, isolation, audit and alarm products;
3. `simulate_privacy_ops.py` generates preflight, incident, and postmortem;
4. `evaluate_privacy_pipeline.py` Summary indicators and main reports;
5. `run_p9_checks.py` performs command-level and data-level checks.

### 16.1 The structural value of hierarchical mapping

Because it makes it easy to understand the system as a closed loop built layer by layer: first define, then process, then operate and maintain, then evaluate, and then accept. It doesn't cram all the logic into one script, so it has a clear structure.

### 16.2 The role of deliverable list

The overall report lists the complete deliverables including:

* `compliance_scope.json`
* `classification_policy.json`
* `access_policy.json`
* `privacy_tech_options.json`
* `raw_sensitive_records.jsonl`
* `classified_records.jsonl`
* `redacted_records.jsonl`
* `quarantine_records.jsonl`
* `audit_log.jsonl`
* `access_alerts.jsonl`
* `isolation_plan.json`
* `preflight_checklist.json`
* `incident_simulation.json`
* `postmortem_report.json`
* `p9_report.md`
* `p9_metrics.json`
* `p9_test_results.json`
* `p9_test_report.md`.

The importance of this list is that it illustrates that the completion standard for P09 is not to run a notebook, but to form a set of reviewable file assets.

---

## 17. Check scripts: consistency of code, products and reports

Many projects only say "it ran successfully" at the end of writing. But this is not enough to prove that the project is actually completed. P09's `run_p9_checks.py` solves exactly this problem.

### 17.1 Check what the script does

It is divided into two categories of checks:

* **Command level check**: such as `py_compile` and re-execute `evaluate_privacy_pipeline.py`;
* **Data/product level checks**: such as whether required files exist, whether role and zone models exist, whether records are classified, whether restricted are isolated, whether direct PII is removed from redacted, whether PII rules exist, etc.

It can be seen from the overall report that the total inspection items are 13, all passed, and the overall status is PASS; including 2 command-level inspection items and 11 data/product-level inspection items.

### 17.2 Verify the structural position of the closed loop

10-2 It is particularly emphasized that "the code, products, statistics and reports are consistent with each other" before the project is truly successful. P09 Although the tasks are different, they inherit the same engineering habits:

> A data engineering case cannot be self-certified by description alone, but must make the check script an acceptor.

### 17.3 Template value of this section

Because what many teams really lack is not "knowing to do verification", but not knowing how to write the verification. P09 gives a minimal approach that is very suitable as a template:

* Can compile;
* Ability to rerun assessment;
* Have documents;
* Has content;
* There are indicators;
* Have consistency.

---

## 18. Indicator interpretation: closed-loop sense as the core signal

In terms of scale, P09 is very restrained: there are only 8 original records, which is far from a production-level data volume. It is still worth expanding on its own because it presents a strong sense of "closed loop".

### 18.1 What is the sense of closed loop?

In this project, the sense of closed loop is reflected in:

* Have goals and boundaries;
* With classification and control;
* With treatment and isolation;
* With alarms and audits;
* Have pre-inspection and accident drills;
* There are indicators and checking scripts;
* Reports and test results are available.

### 18.2 The difference between closed-loop sense and throughput

A small and complete project is often more valuable for learning than a large and vague project. Because what is reusable is often the structure, layering, field design, control logic and verification mode, rather than single throughput.

### 18.3 Judgment of current stage

P09 is more like a privacy processing operating model than a single desensitization script. At the same time, it also acknowledges that data representation is limited, advanced privacy technologies have not yet been implemented in depth, and cross-system governance can still be expanded. This shows that the project is at a stage where the method is complete, the scale is restrained, and the limitations are clear.

---

## 19. Comparison with 10-2: Supervise Assets and Control Assets

Writing 10-9 with reference to 10-2 is not to simply cover it up, but to retain its most valuable narrative skeleton.

### 19.1 Similarities: Both emphasize closed-loop engineering

10-2 talks about the SFT data factory in the legal field, starting from seed knowledge, task design, QA, preference pairs to training delivery and verification closed-loop layers. Although P09 is not an SFT project, it still follows:

* Define the boundary first;
* Then enter the main processing chain;
* Final evaluation and acceptance.

### 19.2 Difference: One core is to supervise assets, and the other core is to control assets

The core products of 10-2 are trainable data, preference pairs, and QA records;
The core products of P09 are control strategies, processing results, audit evidence, operation and maintenance documents and inspection reports.

That is to say:

* 10-2 Solve "what should the model learn";
* 10-9 Solve "How to safely enter sensitive data into the system".

### 19.3 The value of comparative analysis

Because it shows a more complete capability map: industry AI engineering is not just training and inference, but also includes governance capabilities before data enters the system.

---

## 20. Subsequent expansion: Towards a more realistic engineering system

Currently, three directions can be seen: expanding more high-risk scenarios, advancing advanced privacy technology from planning to implementation, and enhancing automation and abnormal access detection. These directions are further specified below.

### 20.1 Expand from rule detection to multi-layer recognition

In the future, the regex layer can be extended to:

* Dictionary and rules layer;
*Context classification layer;
* NER/entity normalization layer;
* Risk portfolio judgment layer.

### 20.2 Extending from static policy to dynamic governance

The current access policy is more static description. The next step can be to introduce:

* Task-based temporary authorization;
* Double approval;
* Export speed limits and thresholds;
* Behavior profiling and anomaly detection.

### 20.3 Extending from file-level delivery to service-level control

Deliverables now focus on JSON/JSONL and Markdown reports. In the next stage, policies can be service-oriented, auditing can be real-time, exception flow can be transformed, and P09 can be promoted from a notebook/script project to a service-oriented system.

### 20.4 Expand from drill-type incident to continuous drill

Incident simulation can be developed into regular tabletop exercises, automated fault injection, and red-blue adversarial auditing to increase the true resilience of the system.

---

## 21. Minimum reproducible running chain: execution sequence of the complete case

The minimum running sequence of P09 can be summarized as the following five steps:

```bash
python src/build_privacy_specs.py
python src/run_privacy_pipeline.py
python src/simulate_privacy_ops.py
python src/evaluate_privacy_pipeline.py
python src/run_p9_checks.py
```

These five steps correspond to:

1. Establish privacy specifications;
2. Handle sensitive records;
3. Generate operation and maintenance and accident documents;
4. Summarize evaluation indicators;
5. Complete project acceptance.

The value of this structure is that it not only makes it easier to understand the case, but also makes it easier to reproduce the case in order.

### 21.1 The role of a single column in the running chain

Because it repackages the entire chapter from "narration" to "action." What is most likely to leave an impression in a chapter is often this step-by-step execution chain.

![Figure 12: P09 minimum reproducible operation chain diagram](../../images/part10/10_9_fig12_execution_sequence.png)

---

## 22. Summary of this chapter: System capabilities demonstrated in P09

If you only look at the surface, P09 looks like a small privacy protection project; but from the perspective of engineering structure, it demonstrates a very complete capability:

* It can first define governance boundaries instead of dealing with text directly;
* It can connect classification, de-identification, isolation, alarm and audit into a control chain;
* It can incorporate preflight, incident and postmortem into the main project process;
* It can output metrics, reports and checks instead of staying in intermediate files;
* It can use a very small sample to make it clear "how privacy projects should be organized".

Therefore, the value of this chapter does not lie in proving how advanced a certain technical point is, but in showing:

> How to design a pre-training data governance project into an interpretable, reproducible, and acceptable engineering pipeline.

This is why it is suitable as an engineering case.

---

## Special Topic: The Operation Manual Perspective of the Privacy Pipeline

At the end of writing a privacy protection project, the most easily missing part is often not the technical structure, but the perspective of the operations manual. In other words, once the system enters daily operation, in what order should the team check, at what node should it be blocked, and who should handle exceptions. Without this part, the project becomes more like a set of scripts; with this part, it starts to resemble a sustainable running governance system.

### 1. What should you look at first in daily operation?

For assembly lines such as P09, the first thing to look at in daily operations is not necessarily the final indicator, but several "pre-health signals":

* Whether all newly entered records into the system have been graded;
* Whether the restricted and redacted areas are placed according to the strategy;
* Whether unrecognized direct PII occurs;
* Whether alarms increased or pre-check failed in the latest batch of runs;
* Whether the audit log is completely generated.

The point of these signals is that they help teams detect process drift early, before a real data breach or compliance issue occurs. For privacy systems, many risks are difficult to repair at low cost once they enter downstream training or applications, so front-end health signals are particularly important.

### 2. Exception handling must have a clear entrance

One of the most dangerous situations in privacy governance is when exception requests are always resolved verbally. Because once there is no explicit entry, it is difficult for the team to distinguish between "reasonable exceptions" and "risk bypasses." A more mature operation manual should at least make it clear:

* What circumstances allow application for temporary exceptions;
* Who can initiate an exception request;
* Who has the authority to approve;
* Whether the exception has a time range and purpose scope;
* Whether it is necessary to recover permissions and supplement audit records after the exception ends.

The significance of explicitly writing these processes into the operation manual is not only to standardize actions, but also to protect the team. Because in high-pressure scenarios, if there is no institutionalized entrance, many risky actions will eventually be borne by individuals rather than the system.

### 3. Accident handling should revolve around the evidence chain

Once a privacy-related exception occurs, what the team needs most is not "who can explain it first" but to quickly restore a chain of evidence. P09 has retained the objects preflight, incident and postmortem. The next step worth emphasizing is how to string them into the action sequence in the runbook:

* Freeze the relevant versions or access paths first;
* Reconfirm the scope of influence and data area;
* Then extract audit logs, alarm records and processing records;
* Finally form incident review and corrective actions.

For privacy projects, the order of disposition is more important than the wording. As long as the order is correct, the team can usually contain the risks within a manageable range; if the order is chaotic, even if the technical components are present, secondary risks can easily occur.

---

## Special Topic: Indicators and Long-term Evolution of Privacy Governance

The current scale of P09 is small, but it is already well suited to illustrate a longer-term issue: how privacy governance should be measured. Because without stable indicators, teams can easily understand privacy work as "trying to have as few incidents as possible"; in fact, a mature privacy pipeline should be able to be continuously quantified, compared and improved.

### 1. Indicators should not only look at the number of detection hits

When many teams talk about privacy indicators, the first thing that comes to mind is "how much sensitive information was identified this time." This certainly makes sense, but it is far from sufficient. Because the number of hits is high, it may mean that the system has strong recognition capabilities, or it may simply be that the input risk is higher. More valuable indicators often include:

* Record grading coverage;
* restricted data isolation accuracy rate;
* redacted data direct identifier residual rate;
* Pre-check failure rate and failure reason distribution;
*The average time from incident discovery to completion of disposal.

The combination of these indicators can more truly reflect whether the governance system is functioning.

### 2. The privacy system is more suitable for "control efficiency"

From a governance perspective, the more core indicator of P09 is actually not the amount of detection, but whether the control is effective. The so-called control efficiency can be understood as: how many control points defined by the system really play a role at critical moments.

For example:

* Whether high-risk records are actually routed to the restricted zone;
* Whether direct PII is actually removed in the redacted version;
* Whether pre-checking really blocks unqualified tasks;
* Whether the alarm actually triggered manual attention;
*Whether the incident review has actually resulted in subsequent rectification.

If these control points only "exist in documents" rather than "exist in operational results", then the governance system is still fragile.

### 3. Long-term evolution will move from rules to combination governance

P09 currently focuses on rules, strategies and operational processes, which are very suitable as the minimum reproducible starting point. As the complexity of the system increases, long-term evolution usually leads to "portfolio governance", that is, combining multiple capabilities to jointly bear risk control:

* Rule detection is responsible for fast, highly explainable bottom line protection;
* Contextual classification is responsible for identifying hidden risks that are difficult to cover by rules;
* Authorization and approval are responsible for restricting high-risk actions;
* Audit and incident review are responsible for post-event tracking and system improvement;
* Operational cadence is responsible for making governance an ongoing action rather than a one-time project.

When these capabilities begin to work together, the privacy pipeline is no longer just "coding data", but gradually grows into a complete system for pre-training governance, pre-application governance, and organizational responsibility allocation. P09 The most valuable thing at present is that it has clearly explained the skeleton of this growth path.

---

## Special topic: Data subject request and cross-system cleanup link

Another topic that is very common in privacy projects in real organizations but is easily missing in many technical cases is data subject requests. Examples include requests for deletion, rectification, restriction of processing or access to instructions. Once the system enters multiple stages before and after training, the subject request is no longer a single point action, but becomes a problem of clearing links across systems.

### 1. The real difficulty of subject request lies in propagation

For a single database, deleting a record may not be complicated; but for a data engineering system, the same sensitive record may have entered:

* Original access area;
* Restricted processing area;
* De-identified copy;
* Evaluation sample;
* Auditing and event logging.

This means that the core difficulty of the subject's request is not "whether to delete it", but "where has this record been spread to?" For this reason, it is necessary to emphasize partitioning, auditing and processing records in P09, because they naturally leave a starting point for subsequent deletion, propagation and verification.

### 2. Cross-system cleanup requires the coexistence of blood ties and responsibilities

Once a subject request occurs, the team needs at least two types of capabilities to exist at the same time:

* Bloodline ability to help locate and record which processing links it has gone through and which derivatives it has produced;
* Responsibility, clarify who is responsible for freezing, who is responsible for cleaning, who is responsible for review, and who is responsible for external confirmation.

If there is only blood relationship but no responsibility, the system will know where the problem is but no one will really deal with it; if there is only responsibility but no blood relationship, the team will know that it must be dealt with, but it will not know how deeply it should be dealt with. A truly executable cleanup link must tie the two together.

