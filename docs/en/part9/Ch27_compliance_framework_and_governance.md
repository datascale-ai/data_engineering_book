# Title 9: Privacy Compliance and Data Security

**Preliminary reading**
* **Core Goal**: Answer the three core questions of "whether the data can be used, how to use it, and whether the use is controllable."
* **Design Concept**: Adhere to the "left shift" of compliance and privacy protection, and put it into the system architecture and process design (Privacy by Design).
* **Implementation support**: Supported by governance templates, access control lists, and minimum governance links, system specifications, process control, automated auditing, and engineering implementation are connected to form a closed loop of compliance governance from demand to offline.

---

# Chapter 27: Data Compliance Framework and Governance

---

## Summary of this chapter

In the data-driven era, compliance is no longer a stamp stamped by the legal department before a project is launched, but an infrastructure constraint that determines whether the system can operate stably in the long term. Many systems have reached the online standards in terms of algorithm effect, business transformation, and grayscale performance, but were urgently stopped during the final review due to unclear data sources, blurred authorization boundaries, incomplete log traces, or exposure of sensitive information. The problem is not that the team "does not pay attention to compliance", but that compliance requirements have long been regarded as an approval appendix after the completion of research and development, rather than as constraints that must be included at the beginning of the system design.

This chapter revolves around a core proposition: **The cost and difficulty of repair of data compliance will rise rapidly as the project life cycle progresses**. Therefore, enterprises cannot regard compliance as a one-time legal review action, but must embed it into the entire process of requirement definition, data modeling, feature development, model training, online approval, audit traces, and offline destruction. In other words, truly effective compliance is not about “supplying documents after something goes wrong” but “moving the risk threshold forward before the system has formed inertia.”

This chapter will establish a complete framework from four levels. First, explain why compliance is a system constraint rather than an approval attachment, and explain the engineering implications of Privacy by Design. Second, establish data classification and grading, risk assessment matrix and responsibility chain for business systems, so that the team can clarify "what data is there", "who is responsible for the data", "under what scenarios can it be used, how can it be used, and who approves it". Third, automated interception mechanisms around RoPA, DPIA, audit traces, and CI/CD stages illustrate how compliance requirements can truly become part of the engineering process. Fourth, extract a minimum governance link to show how governance configuration, desensitization strategies, permission control, pre-check mechanism, event response and review closed loop fall into specific system products.

Unlike many contents that only stay at the level of regulatory provisions, this chapter pays more attention to "engineered governance." We not only explain what the regulations require, but also further discuss: how to mark data levels in the metadata center, how to restrict data entry into the analysis domain through policy files, how to use the pipeline to automatically block high-risk changes before going online, how to support audit backtracking through logs and lineage, and how to design specialized control measures for high-risk scenarios such as medical care, finance, minors, third-party processing, and cross-border transmission. Through these contents, readers will build a data compliance governance picture that is closer to the real enterprise environment.

---

## 27.0 Learning Objectives

After studying this chapter, readers should be able to:

* Understand why data compliance must be moved forward as a system constraint to the requirements, architecture and development stages, rather than just passive approval before going online.
* Master the design principles of data classification and grading, be able to distinguish low-sensitive, medium-sensitive and highly-sensitive data, and design differentiated processing strategies for different levels.
* Construct a data risk assessment matrix for real business, and combine data levels, usage purposes, processing actions and impact scope into executable risk judgment logic.
* Design a clear chain of responsibilities and clarify the boundaries of responsibilities of legal, business, platform, algorithm, data development, security and auditing roles in the governance process.
* Understand the role of RoPA, DPIA, audit traces, consent management, access approval, data retention and destruction mechanisms in the project life cycle.
* Learn to embed compliance requirements into the R&D pipeline to achieve automated inspection and blocking during code submission, configuration changes, data access, model training, and online approval stages.
* Be able to propose special governance strategies for high-risk scenarios such as medical care, finance, minors, third-party processing and cross-border transmission.
* Understand how compliance governance is implemented through configuration files, policy rules, log products and inspection scripts.

---

## Scene introduction

The user insight and recommendation system that the team spent three months developing is ready to be fully launched next week. The business side has high hopes for the 20% conversion rate improvement brought by this model, and the algorithm team is also confident in the results of offline evaluation and grayscale experiments. However, at the final review meeting on Friday, the head of legal and compliance pressed the pause button.

The head of compliance asked three questions in a row. First, the system uses the user's precise geographical location and browsing history in the past three months to determine whether the purpose of using these data is consistent with the user's authorization. Second, whether the test set in the data lake has been desensitized, and why can we still see clear text mobile phone numbers in some debugging logs? Third, if the user requests to log out of the account and exercise the right to delete, can the features and derived labels already formed in the training set be deleted simultaneously, or can only the original records in the main table be deleted.

There was silence in the conference room. The business side believes that it only puts forward the demand of "improving the recommendation effect"; the algorithm team believes that the data is provided by the platform side; the platform team believes that basic permissions have been given; the legal affairs department pointed out that there is no complete record of "what data is used by whom for what purpose" in the system. In the end, the project had to be postponed, and the team began to reorganize data sources, cleaning logic, desensitized links, and approval processes.

This case reveals a common but extremely costly problem: **compliance-behind**. After a system's table structure, feature processes, training pipelines, and log systems have been formed, supplementing authorization, reconstructing desensitization, and adding audit traces often means underlying transformation, historical data reruns, and cross-team process reconstruction. The cost incurred at this time is much higher than the cost of correcting the governance boundaries at the beginning of the system design.

To further illustrate the problem, let's look at three simplified failure cases.

### Case 1: Purpose Drift in Recommendation System

A content recommendation system initially only performed ranking optimization based on click behavior. Later, in order to improve advertising conversion, device identification, dwell time, transaction records and location data were introduced. The privacy policy in the early stages of the project only covered "improving service experience" and did not cover "precision marketing" and "automated pricing." The resulting system, while technically operational, had drifted in terms of legitimacy of purpose. The problem is not just "adding some more fields", but rather that when the purpose of use changes, the authorization and processing basis are not updated simultaneously.

### Case 2: Desensitization failure in test environment

A team enabled mobile phone number hashing and email masking in the production environment, but directly imported the production snapshot in the test environment to troubleshoot problems. Developers and testers have wider permissions to read library tables, and the logging system also prints complete request parameters by default. In the end, it is not the formal links that really expose sensitive information, but the test environment and operation and maintenance logs. This case shows that **system compliance will fail in the gray area if there is no technical enforcement**.

### Case 3: The model deletion request cannot be closed.

After receiving a user deletion request, a risk control model only deleted the user information in the source table, but did not synchronize the feature warehouse, training sample set, offline snapshot, model cache, and downstream portrait tags. Externally, it seems that the deletion has been completed, but internally, there are still historical copies remaining. This case shows that **data deletion is not just table-level deletion, but also a test of full-link deletion capabilities**.

### Core Engineering Pain Points behind the scene

From the above cases, we can summarize four types of pain points in real projects:

1. **The extremely high cost of delayed compliance**: Once problems are discovered late in the system, they are often accompanied by rework, delays, shutdowns, and potential penalties.
2. **Blurred boundaries of authority and responsibilities**: The business line wants data, the algorithm wants features, the platform side provides capabilities, and there are legal control provisions, but there is no clear closed loop of responsibility.
3. **Technology and system are out of touch**: The compliance document looks complete, but there is no technical grasp such as metadata annotation, desensitization engine, policy verification, and access audit.
4. **The life cycle is not closed**: From collection, processing, use, sharing, retention to deletion, there is no verifiable, traceable, and auditable closed loop.

---

## 27.1 Why compliance is a system constraint rather than an approval attachment

The traditional R&D process places compliance review at the end of the process, close to launch. When setting up a demand project, discuss the functional goals first, verify the effects first during development, and then ask the legal or compliance team to check before going online. This model may still work reluctantly in systems with simple functions and low data sensitivity, but it has become increasingly dangerous in today's data-driven systems.

Modern data governance emphasizes **Privacy by Design**. Its core is not to do an extra round of approvals, but to recognize that privacy and compliance themselves are part of the system architecture. How to layer the database, where to store the logs, whether the features are traceable, how to clean the training set, and whether the third-party API brings out sensitive fields. These are not problems that can be solved by "filling in an approval form before going online", but basic constraints that must be considered in the architecture stage.

### 27.1.1 The cost of post-compliance

The cost of post-compliance is mainly reflected in five aspects.

**First, the cost of architecture rework is high. **
If the system has extensively relied on a sensitive field as a core feature, and it is later discovered that the field is not legally authorized, it means that the table structure, feature engineering, model training, and downstream consumption all need to be adjusted.

**Second, it is difficult to recover historical data. **
When sensitive data has entered the training set, profiling system, cache layer, and report link, the cost of deleting, replacing, or retraining is much higher than the initial restricted usage range.

**Third, cross-team coordination costs increase exponentially. **
The closer a project is to being launched, the more participants there will be and the more complex the dependencies will be. Introducing compliance rectification at this time often requires simultaneous adjustments to business, algorithms, platforms, legal affairs, security, testing and operation and maintenance.

**Fourth, business window period is lost. **
Many projects are not "cannot be changed", but "too late to change". When compliance issues occur before going online, the most direct loss is missing business rhythm and market windows.

**Fifth, audit and penalty risks have increased sharply. **
Once the system is online and has a real impact on users, problems discovered are not just internal rectifications, but may escalate to external complaints, audit inspections or regulatory penalties.

In order to help readers form a more intuitive understanding of engineering, the cost of compliance rectification can be understood as a curve that rises rapidly as the life cycle progresses: correcting an unreasonable data field in the demand stage may just change a document; correcting in the training stage requires redoing data cleaning and feature development; correcting after going online may involve rollback, explanation, deletion, compensation and external communication.

### 27.1.2 Global Perspective: Compliance Baseline Differences

Compliance requirements are not entirely consistent for businesses across regions. China's PIPL emphasizes "inform-consent", separate consent for sensitive personal information, and control requirements for data export; Europe's GDPR emphasizes data minimization, the right to deletion, the right to portability, and the transparency of automated decision-making. When an enterprise runs the same business in different regions, it cannot assume that "one set of collection logic will prevail all over the world", but must have differentiated implementation capabilities at the technical layer.

This means that the system design cannot only focus on "whether the function can run smoothly", but also must focus on "how differences in regional rules are mapped to product, data and model links." For example, the same user profile may have different legal basis, retention period, user rights response mechanism and sharing restrictions in different jurisdictions. A truly mature governance system does not require developers to memorize all regulatory provisions, but to precipitate these differences into templates, policies, approval rules, and default behaviors.

### 27.1.3 Governance intersection: collaboration of data, models and business

Compliance does not exist in isolation. It is the intersection of data governance, model governance and business governance.

* **Data Governance** focuses on data quality, metadata, lineage, lifecycle and retention strategies.
* **Model Governance** focuses on feature sources, risk of bias, interpretability, training set boundaries, and purpose of model use.
* **Business Governance** focuses on business goals, user commitments, authorization basis, external disclosure and operating rules.

If these three are separated, common misalignments will occur in the system: the business requires new capabilities, the model quickly integrates new features, and the available interfaces are opened on the platform side, but no one makes a unified judgment on whether this data link is compliant. Only through a unified metadata management center, policy center and audit center can these requirements that were originally scattered among different teams be linked.

![Figure 27-1: Compliance shift left and governance collaborative architecture diagram](../../images/part9/图27_1_合规左移与治理协同架构图.png)
*Figure 27-1: Compliance shift left strategy - showing how compliance review moves from pre-launch to requirements analysis and architecture design stages*

### 27.1.4 Comparison between traditional process and left shift process

The following table illustrates the differences between the two processes in a more engineering way:

| Stage | Traditional model | Shift-left governance model |
| :--- | :--- | :--- |
| Requirements analysis | Focus on business functions and rarely define data boundaries | Clarify data types, purpose of use, authorization basis and output boundaries |
| Architecture design | Prioritize availability and performance | Synchronously define classification, desensitization, auditing, retention and deletion mechanisms |
| Data access | Align fields first, then add explanations | Complete registration, classification and legality verification before access |
| Feature development | Prioritize effects | Restrict highly sensitive fields from directly entering the training and analysis link |
| Testing and grayscale | Often verified with real data snapshots | The test environment is de-identified by default, and the logs are minimally exposed by default |
| Online review | Temporary legal review | Approval based on RoPA, DPIA, pre-inspection report and audit traces |
| Runtime | Mainly focus on faults and performance | Also focus on access exceptions, export risks, deletion requests and incident response |

### 27.1.5 “Security by Default” from the Perspective of Engineering Governance

From an architectural perspective, mature compliance governance does not rely on everyone’s awareness, but rather establishes “safe by default” system behavior:

* Newly accessed data sets have no permission by default and must apply explicitly.
* Sensitive fields are masked and displayed by default instead of "clear text first, then hidden if necessary".
*C2/C3 level data cannot directly enter external APIs or large model prompts by default.
* The test environment does not allow importing production plaintext snapshots by default.
* Data that has not been registered for use cannot enter the model training link by default.
* When RoPA, DPIA or approval records are missing, CI/CD will be blocked from going online by default.

The essence of this type of default behavior is to upgrade compliance from "human reminders" to "system guardrails."

---

## 27.2 Regulation mapping, risk classification and responsibility chain

If "compliance moves left" answers why compliance must move forward, then this section answers: **What exactly does the team need to manage after moving forward**? The answer can be summarized in three questions:

1. What data do we have?
2. In what scenarios can these data be used?
3. Who is responsible if something goes wrong?

### 27.2.1 Data classification and classification architecture

Not all data deserves governance with the same intensity. If all data is treated as the highest level of protection, the system will be overly rigid, and performance and iteration efficiency will be affected; but if all data is processed with uniform and loose standards, sensitive data will be exposed. Therefore, enterprises must establish a differentiated data classification and grading system.

This chapter uses a three-level classification as the basic framework:

| Security Levels | Definitions and Examples | Processing Requirements | Masking and Encryption Strategies |
| :--- | :--- | :--- | :--- |
| **L3 Highly Sensitive (C3)** | Sensitive personal information (biometrics, medical health, precise location), core business secrets (undisclosed financial reports) | User consent is required separately; entry into the analysis domain without desensitization is strictly prohibited; Legal veto | Storage-level strong encryption (AES-256); Full desensitization during display; Invisible when available (private computing) |
| **L2 Medium Sensitive (C2)** | General personal information (name, mobile phone number, device ID), internal business data | Included in privacy policy; restricted to authorized personnel and projects | Transmission encryption; Storage encryption or hash de-identification; Partial mask display |
| **L1 Low Sensitivity (C1)** | Public data, completely anonymized data, aggregated statistical results | General access control; can be used for extensive BI analysis and model training | No special requirements, on-demand storage |

The value of this grading system is not only to help the team label, but also to provide a unified basis for subsequent permission policies, desensitization rules, approval processes, log requirements, and retention periods.

### 27.2.2 Multi-dimensional classification from field level, table level to scene level

Many teams understand "grading" as giving an overall label to a table, but real systems are often more complex:

* C1, C2 and C3 fields may exist simultaneously in the same table.
* The same field has different risks in different scenarios.
* The same data may have different levels in raw, desensitized and aggregated states.

Therefore, mature grading requires at least three dimensions:

**Field-level classification**: For example, fields such as mobile phone number, email address, ID number, precise location, bank card number, and medical record number should have clear labels.  
**Table-level grading**: For example, data sets such as user profile, transaction flow, behavior logs, risk control labels, customer service recordings, etc. should have an overall grade baseline.  
**Scenario-level classification**: For example, "for model training", "for customer service retrieval", "for BI reporting" and "for external interfaces". These usage scenarios themselves will also affect the risk level.

Only by incorporating these three dimensions at the same time can the system avoid the confusion of "same table, different fields, and the same field, different uses".

### 27.2.3 Typical governance requirements for each level of data

#### 1. L1 low-sensitive data

L1 data is usually public data, anonymized data, or aggregated statistical results. For example, public macro indicators, regional-level conversion rates after de-identification, operational indicators without individual identification, etc.  
L1 data can usually be circulated in a wider range and used for scenarios such as BI analysis, model tuning, visual dashboards, and capacity planning. Its governance focuses on basic access controls, data quality assurance, and reasonable retention.

#### 2. Sensitive data in L2

L2 data usually includes general personal information or internal restricted data such as name, mobile phone number, email, device ID, employee number, internal business records, etc.  
This type of data cannot be exported at will, nor should it be explicitly exposed in logs, test environments, and external interfaces. Governance priorities include encryption, hash de-identification, partial mask display, least privilege access, and restricted use.

#### 3. L3 highly sensitive data

L3 data includes biometrics, medical health, precise location, financial accounts, undisclosed trade secrets, etc.  
Once this kind of data is leaked, the impact will be huge, so it usually requires separate consent, strict approval, independent storage, strong encryption and higher standards of auditing. By default, L3 data should not enter the general analysis domain directly, nor should it be consumed arbitrarily by downstream.

### 27.2.4 Risk Assessment Matrix

Simply knowing "what grade the data is" is not enough. The risk of a field is not static, but also depends on what it is used for and how it is processed.

For example:

* Using L3 data for automated decision-making is high risk.
* Using L2 data for internal analysis and not having a direct impact on individuals is a medium risk.
* Using L1 data for system stability monitoring is usually a low risk.

Therefore, risk assessment needs to consider the following dimensions simultaneously:

1. **Data Level**: C1/C2/C3
2. **Purpose of use**: service performance, risk control, recommendation, marketing, auditing, R&D testing, etc.
3. **Processing actions**: query, export, training, sharing, push, automated decision-making
4. **Influenced objects**: internal teams, partners, external users, cross-border recipients
5. **Result Impact**: Will it further affect user rights, pricing, portraits, recommendations, credit or account status?

To sum up, the above dimensions can be compressed into a judgment logic that is more convenient for project implementation:

```text
Risk level = Data sensitivity × Processing action intensity × Business impact scope
```

In engineering practice, this logic is often eventually encoded into a policy engine:
When the data level is higher, the processing actions are stronger, and the scope of impact is wider, more approvals, stricter desensitization, and higher levels of auditing are required.

### 27.2.5 Risk Matrix Example

| Data Class | Purpose of Use | Processing Actions | Risk Level | Default Controls |
| :--- | :---- | :------ | :--- | :------------------ |
| C1 | Stability Monitoring | Aggregation Queries | Low | General Access Control |
| C2 | Internal analysis | Profile analysis | Medium | Role authorization, partial desensitization, access traces |
| C2 | Model training | Batch extraction | Medium to high | Feature whitelist, training approval, result review |
| C3 | Automated decision-making | Training/scoring | High | DPIA, legal approval, strong audit, online blocking |
| C3 | Third-party sharing | Export/interface call | Extremely high | DPA, desensitization gateway, minimum field set, special evaluation |

![Figure 27-4: Risk matrix diagram composed of data classification, usage and processing actions](../../images/part9/图27_4_数据分级用途和处理动作构成的风险矩阵图.png)
*Figure 27-4: Risk matrix diagram composed of data classification, usage and processing actions*

### 27.2.6 Responsibility chain construction (RACI Matrix)

Without a clear chain of responsibilities, no matter how good the rules are, they will be distorted in execution. The governance system must clarify "who is responsible for making demands, who is responsible for judging legality, who is responsible for providing technical control, who is responsible for using process compliance, and who is responsible for audit review."

The following table shows a typical RACI design:

| Role | Key Responsibilities | RACI |
| :------ | :------------------- | :------------------- |
| Legal/Compliance | Interpret regulations, set red lines, and approve high-risk scenarios | Accountable |
| Business party | Describe the purpose of use, business necessity and user commitment | Responsible |
| Platform/Infrastructure | Provides classification, desensitization, permissions, auditing and lineage capabilities | Responsible |
| Algorithm/Data Development | Develop features, train models and consume data within the scope of authorization | Consulted / Informed |
| Security Team | Examine perimeter security, audit policies, abnormal access and export risks | Consulted |
| Audit/Internal Control | Regular review of traces, process execution and rectification closed loop | Informed / Consulted |

### 27.2.7 Common misunderstandings in the implementation of the responsibility chain

During actual implementation, teams often make three mistakes:

**Misunderstanding 1: Putting all compliance responsibilities on legal affairs. **
Legal affairs can explain boundaries, but they cannot replace the purpose of business clarification, nor can they replace the platform to achieve control.

**Misunderstanding 2: Push all technical control to the platform. **
The platform can provide capabilities, but if the business and algorithms do not declare their true uses, the platform will not know which usage behaviors are unreasonable.

**Misunderstanding 3: Thinking that “approval means compliance”. **
Approval is just a node. What really matters is whether the information before approval is complete, whether the behavior after approval is verifiable, and whether deviations can be continuously discovered during the runtime.

### 27.2.8 From institutional text to system label

Mature teams will map the chain of responsibilities to actual objects in the system:

* Business owner → project owner field
* Compliance approver → Approval node and work order flow
* Data owner → Data set metadata management
* Purpose of use → RoPA form fields
* Usage permissions → RBAC/ABAC policy
* Risk level → Pre-check rule parameters
* Access traces → Audit log events

This step is very critical, because only when the institutional language enters the system metadata, can compliance truly have the possibility of automated execution.

---

## 27.3 Project establishment, review and pre-launch inspection

If compliance governance wants to truly become a part of R&D, it cannot just stay at the principle level, but must become a clear process node. From project establishment to offline, at least six questions must be answered:

1. Is this processing activity registered?
2. Is there a lawful basis for this processing?
3. What levels of data are involved?
4. Has a risk assessment been conducted?
5. Is it capable of auditing and deleting?
6. Has the online threshold been reached?

### 27.3.1 Establish RoPA (Record of Processing Activity)

**RoPA (Record of Processing Activities)** is a general ledger of processing activities. It is not a form that is filled out temporarily for audit purposes, but a systematic registration of data use.

Each project should fill in at least the following information before applying for access to data, training models or open interfaces:

1. Data type and source (first party, third party, public data)
2. Purpose of use and legal basis
3. Involved systems, tables, fields and output objects
4. Data retention period (TTL) and destruction mechanism
5. Whether it contains sensitive data, involves third parties or cross-border transmission
6. Data owner, project owner, approvers and usage team

The value of RoPA is that without it, the team will never be able to tell "which data is used by whom for what reasons." Once a complaint, audit or deletion request occurs, the link cannot be located without a ledger.

#### RoPA minimal form design example

| Field | Description |
| :---------------- | :-------- |
| project_id | Project unique number |
| owner | project leader |
| system_name | system name |
| data_sources | Data sources and table names |
| purpose | purpose of use |
| legal_basis | Legal basis or authorization basis |
| data_level | data level |
| retention_days | retention period |
| third_party_share | Whether to share with a third party |
| cross_border | Whether cross-border is involved |
| deletion_path | Deletion path description |
| audit_required | Whether to force audit |
| approval_chain | Approval link |

#### Engineering requirements for RoPA

In a mature platform, RoPA should not be offline Excel, but should have the following characteristics:

* Submit through system forms and versioned storage
* Associated with project code warehouse, dataset metadata and approval work orders
* Support field-level and table-level automatic verification
* Linked with the CI/CD pipeline to block the online process when key information is missing
* Traceable historical changes to meet audit requirements

### 27.3.2 Perform a DPIA (Data Protection Impact Assessment)

When the project involves medium-to-highly sensitive data, new processing methods, automated decision-making, or high-impact scenarios, RoPA alone is not enough and requires further **DPIA (Data Protection Impact Assessment)**.

The core of DPIA is not to "write a long report", but to systematically answer the following questions:

* Is it really necessary to collect this data?
* Are these data outside the scope of the original authorization?
* Will the processing results have a significant impact on users?
* Is there a risk of discrimination, abuse, cross-border profiling or miscarriage of justice?
* If leakage, misuse or unauthorized access occurs, does the system have the ability to respond and stop losses?

#### Typical Assessment Steps for DPIA

**Step 1: Identify the processing activity. **
Clarify data sources, processing purposes, main participating systems and output results.

**Step 2: Identify risk points. **
Including data over-exploitation, purpose drift, third-party leakage, model bias, log exposure, deletion without closed loop, etc.

**Step 3: Assess necessity and proportionality. **
Determine whether the fields are minimally necessary and whether the processing method matches the business purpose.

**Step 4: Design control measures. **
For example, field deletion, default desensitization, strong auditing, access approval, model interpretability description, deleted link enhancement, etc.

**Step 5: Form an approval conclusion. **
Give a clear conclusion on whether it can go online, needs to go online after rectification, or is prohibited from going online.

#### DPIA Risk Score Example

| Dimensions | Rating description |
| :---- | :----------------------- |
| Data sensitivity | C1=1, C2=2, C3=3 |
| Usage Intensity | Query=1, Analysis=2, Training/Automated Decisions=3 |
| Spread scope | Internal single team = 1, multi-team sharing = 2, third party/cross-border = 3 |
| Equity impact | No direct impact = 1, indirect impact = 2, direct impact = 3 |

The higher the total score, the stricter the control requirements. For example, a total score of 8 or more can trigger legal approval and online blocking.

### 27.3.3 Minimum necessary principle and field slimming

Many governance failures are not due to insufficient controls but because too much unnecessary data is collected in the first place.
The “minimum necessary principle” requires the team to continuously ask:

* Is this field really necessary?
* Can a lower sensitivity field be used instead?
* Is it possible to aggregate first and then use it instead of using details directly?
* Is it possible to only retain for a short-term window, rather than long-term?

In engineering, the implementation of the minimum necessary principle includes:

* Field whitelist instead of opening the entire table
* The training feature set is isolated from the original details
* Time window clipping
* The original identifier is not retained by default
* Issued after aggregation, rather than releasing details

### 27.3.4 Consent management, authorization scope and purpose binding

Many data disputes are not about "the data itself is illegal" but "the use of the data exceeds the original notification". Therefore, the system needs to truly bind authorization and purpose.

This means:

* User authorized version should be traceable
* Each purpose should be mapped to a clear processing activity
*Need to be reassessed when use changes
* "Improved experience" cannot be applied to all business scenarios in a general way
* There must be a higher level of authorization and explanation mechanism for sensitive data, high-risk profiling, and automated decision-making

From a system perspective, consent management should at least retain: user ID, authorized version, authorization time, scope of use, withdrawal status, applicable product lines and validity period.

### 27.3.5 Audit preparation and compliance traces

If any system leaves no traces, it will ultimately be difficult to self-certify and implement it.
Therefore, authorization records, approval work orders, access logs, export records, policy hits, pre-check results, and deletion execution results should all have tamper-proof storage and traceability capabilities.

A mature audit system needs to record at least the following events:

* Who accessed what data at what time
* What roles are used for access and the basis for approval
* Which fields and how many records are involved in the access
* Whether exporting, downloading or sharing occurs
* Whether the desensitization, blocking or alarm rules are hit
* Whether there is a deletion, correction or query request for the specified user
* How long it took for the request to be completed

#### Audit log design example

| Field | Description |
| :--------------------- | :------------- |
| event_time | event time |
| actor | operating subject |
| role | role used |
| action | query, export, update, delete, share |
| dataset | dataset name |
| fields | Involved fields |
| record_count | Number of records |
| purpose | purpose of use |
| approval_id | Corresponding approval form |
| policy_result | Allow, allow after desensitization, block |
| trace_id | Link trace ID |

### 27.3.6 Pre-launch checks: Embedding compliance into CI/CD

The real watershed for many teams is not whether they have written the system, but whether they have turned the system into an automated threshold before going online.

A typical CI/CD compliance pre-check should include:

* Whether there is a valid RoPA
* Whether the required DPIA exists
* Whether data classification is completed
* Whether there are field-level desensitization rules
* Whether to configure access permissions and role boundaries
* Whether there is audit log output
* Whether to define the data retention period and destruction mechanism
* Whether to identify third-party sharing and cross-border transmission
* Whether to configure the deletion request processing path
* Whether it passes the test environment desensitization check

When any high-risk item is missing, the pipeline should block building, blocking deployment, or blocking training task execution.

![Figure 27-5: Compliance access control flow chart from data access to model training](../../images/part9/图27_5_从数据接入到模型训练的合规门禁流程图.png)
*Figure 27-5: Compliance access control flow chart from data access to model training*

### 27.3.7 Governance Pipeline: From Documentation Requirements to System Execution

When compliance requirements enter the engineering stage, governance objects cannot just stay in the system text, but need to be organized into executable pipelines. A minimal governance link typically includes the following steps:

1. Generate privacy specifications and policies
2. Execute privacy processing pipeline
3. Simulated operation and maintenance and event processing
4. Evaluate privacy pipelines
5. Run project check

This link reflects a key fact: **Compliance is not a single point of action, but an assembly line from policy generation, data processing, alarm response to assessment and verification**.

![Figure 27-3: Privacy specification and policy generation flow chart](../../images/part9/图27_3_P09隐私规格与策略生成流程图.png)
*Figure 27-3: Privacy specification and policy generation flow chart*

### 27.3.8 How to translate governance indicators into engineering language

The value of governance indicators does not lie in the sample size itself, but in whether it can show that the identification, processing, access control, alarm and inspection links have formed a closed loop. For example:

* There are 8 original records and 7 restricted records, indicating that the system has indeed identified and isolated most of the restricted data.
* The direct PII removal rate is 100%, indicating that the desensitization logic at least covers the direct identification in the sample.
* The preflight pass rate is 100%, indicating that a clear threshold has been established for the inspection link before going online.
* The alarm resolution rate is 100%, indicating that the alarm is not "seen but no one cares about it", but entered into a closed-loop process.
* There are 13 total inspection items and all passed, indicating that the current rules, products and inspection logic are self-consistent within the scope of the sample.

The significance of such indicators does not lie in the absolute value, but in that they transform compliance governance from an abstract concept into a system behavior that can be inspected, reviewed, and operated sustainably.

![Figure 27-2: DPIA and RoPA engineering approval flow](../../images/part9/图27_2_DPIA与RoPA工程化审批流.png)
*Figure 27-2: Data compliance life cycle - automated interception and audit process from business project establishment to offline*

### 27.3.9 Compliance online access control list (example)

Below is a pre-launch access control checklist that can be used directly for project review. In order to facilitate review and responsibility allocation, it is recommended to confirm item by item in four categories:

**1. Governance and Approval**

- ☐ RoPA registration completed and approved
- ☐ Necessary DPIA completed
- ☐ Completed online approval leaving traces

**2. Isolation of data and environment**

- ☐ Completed data classification and binding of field labels
- ☐ Training/analysis sets have been isolated from the original sensitive data
- ☐ The test environment does not have clear-text sensitive data snapshots

**3. Access control and audit traces**

- ☐ Direct identification is not printed in the log
- ☐ Configured role permissions and minimum access boundaries
- ☐ Audit logs and abnormal access alarms have been connected

**4. Life cycle and release control**

- ☐ Configured data retention period and destruction mechanism
- ☐ The full link processing path for deletion requests has been configured
- ☐ Identified third-party sharing and cross-border transfer risks
- ☐ Passed CI/CD Compliance Pre-Check

### 27.3.10 Runtime governance is not an add-on after launch

Many teams do a good job in pre-launch management, but quickly become lax after the go-live. In fact, the real risks often occur during runtime:

* After new members join the team, their permissions are not converged.
* When new requirements are added, the old approval is reused but its purpose has changed.
* The calling scope of third-party interfaces is gradually expanding
* Data export, report sharing and log troubleshooting have become new entrances to leaks
* Deletion requests, rectification requests and audit inquiries only appear intensively during the running period

Therefore, runtime governance should include:

* Periodic permission review
* Export behavior audit
* Assessment of new use changes
* Abnormal access detection
* Remove request SLA tracking
* Incident response and postmortem mechanism

![Figure 27-6: Audit log, alarm, event response and review closed-loop diagram](../../images/part9/图27_6_审计日志告警事件响应与复盘闭环图.png)
*Figure 27-6: Audit log, alarm, event response and review closed-loop diagram*

---

## 27.4 High-risk scenario governance

Not all scenarios require the same intensity of governance. Certain areas inherently carry higher sensitivities and more complex boundaries of responsibility, so specialized controls must be designed.

### 27.4.1 Healthcare and Finance: Basic Control of Strongly Regulated Industries

The common characteristics of medical and financial scenarios are: highly sensitive data, extremely high cost of misuse, strict regulatory requirements, and fragile user trust.

#### Medical scene

Medical data usually includes medical records, test results, health indicators, physiological characteristics, medication history, medical consultation records, etc. This data is not only highly sensitive personal information, but is often cross-linked with identity, family, insurance and financial information.

Therefore, medical scenarios should focus on controlling:

* Original health data and behavior log partition storage
* Strong encryption and fine-grained access control for sensitive fields
*Medical details do not enter the general analysis domain by default
* Must be de-identified and minimized before sharing externally
* Audit logs must cover access, export and sharing behaviors
* Deletion and correction requests need to be executed in conjunction with multiple systems

#### Financial scene

Common sensitive data in financial scenarios include bank card numbers, transaction flows, credit records, repayment behavior, equipment risk labels and risk control scores.
In this type of system, automated decisions often directly affect user credit, payment, pricing or account status. Therefore, in addition to data protection, we must also pay attention to model interpretability, fairness and misjudgment correction mechanisms.

Special controls for financial scenarios usually include:

* Highly sensitive account information is isolated from ordinary behavior logs
* Model input features can be tracked, explained, and deleted
* Manual review mechanism covers high-impact decisions
* Risk labels and original vouchers can be associated and traced back
* Key alarms for abnormal access, export and batch query

### 27.4.2 Minors Data Governance

The focus of minors' data governance is not just "one more check box", but to restructure the entire data processing method around protective principles.

Important things to consider:

* Independent guardian consent and withdrawal mechanism
* Stricter data minimization principles
* Commercial recommendations and in-depth profiling are prohibited or strictly restricted
*Shorter retention period
* Stronger default privacy protection
* A more understandable way of notification and explanation

In terms of engineering design, this can be achieved through “age stratification + special labels + special use restrictions”. For example, once an account is identified as a minor, certain recommendations, marketing, portraits and third-party sharing links should be turned off by default or enter a higher level of approval.

### 27.4.3 Externally sourced data and supply chain risks

Not all data used by enterprises is produced in-house. Many projects will bring in external vendors, partner datasets, or public data sources.
In this case, the biggest risk is often not the field itself, but unclear source legality, opaque authorization chain, and inconsistent usage commitments.

Governance requirements include:

* Verify data sources and collection basis
* Review whether the supplier has legal authorization and sub-authorization capabilities
* Sign DPA (Data Processing Agreement) and security clauses
* Clarify the division of responsibilities, leak notification and deletion coordination obligations
* Set independent labels and usage restrictions for external data

### 27.4.4 Delegated processing and third-party API risks

More and more enterprises are offloading some of their processing power to external cloud services, external models, or third-party APIs. The risk is that teams can easily mistake "calling a service" for just technical integration, while ignoring that it is essentially a data outbound or delegated processing.

Typical risks include:

* Directly bring plain text C2/C3 data into Prompt or request body
* The supplier uses the requested content for training or secondary processing
* Request logs are retained on third-party platforms for a long time
* The response result contains sensitive information that should not be returned
* The deployment location of third-party services conflicts with data localization requirements

Therefore, a border gateway must be established to automatically perform the following actions before requesting to go out of the domain:

* Field detection
* Plain text recognition
* Desensitization replacement
*Rule hit blocking
* Request to leave traces
* Approval of high-risk calls

![Figure 27-7: Third-party API/large model call border gateway diagram](../../images/part9/图27_7_第三方API与大模型调用边界网关图.png)
*Figure 27-7: Third-party API/large model call border gateway diagram*

### 27.4.5 Cross-border transmission governance

The core difficulty in cross-border transfers is that once a piece of data leaves the original jurisdiction, its subsequent processing, retention, sharing and auditing may become more complicated.
Therefore, cross-border governance should not only be solved at the contract level, but should also be done at the system level:

* Mark the cross-border flow path
* Control the minimum field set
* Prioritize the use of desensitized or anonymized results
* Clarify the recipient's role, purpose and retention period
* Keep special audit records for cross-border incidents
* Establish a stricter export approval process for highly sensitive data

### 27.4.6 New risks in the era of large models: Prompt compliance

In generative AI applications, new risk boundaries come from prompts, context splicing, and external knowledge invocation.
When many teams implement intelligent customer service, enhanced retrieval generation, summarization and insights, they will inadvertently bring plain text mobile phone numbers, ID numbers, medical record details, internal bills or complete transaction information into the model context.

Governance priorities include:

* Prompt input field whitelist
* Automatic desensitization of highly sensitive fields
* Context filtering for external large model calls
* Retention control of model logs and session records
* Audit and spot check reproducible output
* Establish higher-level access controls for knowledge bases containing personal information

### 27.4.7 Summary list of high-risk scenario governance

| Scenario | Main Risks | Core Control Measures |
| :---------- | :------------- | :------------------- |
| Medical | Health data leakage, purpose drift | Independent encryption zone, fine-grained permissions, strong auditing |
| Finance | Automated decision-making misjudgments, account information exposure | Feature traceability, manual review, export audit |
| Minors | Insufficient consent, excessive commercialization | Guardian mechanism, use restrictions, short retention period |
| Externally sourced data | Illegal sources, unclear responsibilities | Source verification, DPA, purpose binding |
| Third-party API | Clear text out of domain, log residue | Border gateway, desensitization, call traces |
| Cross-border transmission | Weak control after leaving the country | Minimum field set, special approval, special audit |
| Large model Prompt | Sensitive information enters context | Prompt filtering, field whitelist, session traces |

---

## 27.5 Cases and Governance Templates

The previous chapters explained the principles, methods and processes, and this section further provides a governance template that can be directly implemented in the project. The value of templates is that they turn abstract systems into maintainable, auditable, and automatically verifiable configuration objects.

### 27.5.1 Governance Toolbox: RoPA Statement Configuration (YAML Example)

On the platform side, each data application is required to submit a compliance configuration file similar to the following in the code warehouse before going online, which will be automatically pulled and verified by the CI/CD pipeline:

```yaml
# P09-User-Insight-Model RoPA Declaration
project_id: "P09-001"
project_name: "P09-User-Insight-Model"
owner: "algo_team_a"
biz_owner: "growth_recommendation_team"
legal_owner: "compliance_office"
data_usage_purpose: "User behavioral insight and recommendation"
legal_basis: "User Consent (v2.1 Terms of Service)"
processing_activity_type: "model_training_and_internal_analysis"
regions: ["CN"]
contains_sensitive_personal_info: true
requires_dpia: true
third_party_processing: false
cross_border_transfer: false
retention_days: 180
deletion_sla_days: 15

data_categories:
  - table: "dwd_user_behavior_log"
    level: "C1"
    fields:
      - "click_event"
      - "item_id"
      - "timestamp"
    purpose: "recommendation_feature_generation"
    retention_days: 180
    export_allowed: false

  - table: "dim_user_profile"
    level: "C2"
    fields:
      - "hashed_phone"
      - "age_band"
      - "province"
    purpose: "feature_enrichment"
    retention_days: 365
    anonymization_strategy: "K-Anonymity"
    export_allowed: false

  - table: "user_precise_location"
    level: "C3"
    fields:
      - "lng"
      - "lat"
      - "geo_hash_12"
    purpose: "high_risk_feature_candidate"
    retention_days: 30
    export_allowed: false
    legal_manual_approval_required: true

access_roles:
  - role: "algo_reader"
    datasets: ["dwd_user_behavior_log", "dim_user_profile"]
    action_scope: ["read_masked", "feature_compute"]
  - role: "platform_admin"
    datasets: ["*"]
    action_scope: ["policy_admin", "audit_read"]
  - role: "security_auditor"
    datasets: ["audit_log"]
    action_scope: ["read"]

controls:
  audit_log_required: true
  lineage_tracking_required: true
  pii_scan_required: true
  test_env_plaintext_forbidden: true
  prompt_plaintext_c2_c3_forbidden: true

pipeline_gate:
  block_if_missing_dpia: true
  block_if_unapproved_c3_usage: true
  block_if_retention_undefined: true
  block_if_deletion_path_missing: true
```

![Figure 27-8: Schematic diagram of full-link propagation and cleanup of user deletion requests](../../images/part9/图27_8_用户删除请求的全链路传播与清理示意图.png)
*Figure 27-8: Schematic diagram of full-link propagation and cleanup of user deletion requests*

### 27.5.2 Data classification strategy (JSON example)

```json
{
  "policy_name": "p09_classification_policy",
  "version": "1.0.0",
  "levels": {
    "C1": {
"description": "Public data or completely anonymized data",
      "default_controls": ["rbac_basic", "standard_logging"]
    },
    "C2": {
"description": "General personal information and internal business data",
      "default_controls": ["rbac_strict", "masked_display", "encrypted_storage", "audit_required"]
    },
    "C3": {
"description": "Sensitive personal information and core business secrets",
      "default_controls": ["legal_approval", "strong_encryption", "isolation_zone", "full_audit", "export_block"]
    }
  },
  "field_rules": [
    {
      "match": ["phone", "mobile", "email", "device_id"],
      "level": "C2",
      "masking": "partial_mask"
    },
    {
      "match": ["bank_account", "patient_id", "biometric", "precise_location"],
      "level": "C3",
      "masking": "full_mask"
    }
  ],
  "usage_constraints": [
    {
      "level": "C3",
      "forbid": ["external_prompt_plaintext", "test_env_plaintext", "open_export"]
    }
  ]
}
```

### 27.5.3 Access Control Policy (YAML Example)

```yaml
policy_id: "p09_access_policy"
version: "1.2.0"

roles:
  - name: "algo_reader"
    allowed_levels: ["C1", "C2"]
    restrictions:
      - "cannot_export_raw"
      - "cannot_access_c3"
      - "must_use_masked_view"

  - name: "risk_reviewer"
    allowed_levels: ["C1", "C2", "C3"]
    restrictions:
      - "approval_ticket_required"
      - "session_recording_required"

  - name: "security_auditor"
    allowed_levels: ["audit_only"]
    restrictions:
      - "read_only"

approval_rules:
  - if:
      action: "export"
      level: "C2"
    then:
      approvals_required: 2
      approvers: ["data_owner", "security_owner"]

  - if:
      action: "read"
      level: "C3"
    then:
      approvals_required: 2
      approvers: ["legal_owner", "platform_owner"]

  - if:
      action: "external_api_call"
      level_includes: ["C2", "C3"]
    then:
      gateway_scan_required: true
      plaintext_forbidden: true
```

### 27.5.4 DPIA template (Markdown form example)

```md
# DPIA Assessment Form

## 1. Basic information
- Project name:
- Project number:
- Business person in charge:
- Technical person in charge:
- Compliance Officer:
- Assessment date:

## 2. Description of processing activities
- Data sources involved:
- Data fields involved:
- Data level:
- Purpose of processing:
- Output object:
- Whether automated decision-making is involved:

## 3. Necessity and proportionality
- Whether the current field is minimally necessary:
- Whether there are low-sensitivity alternative fields:
- Is there a risk of over-collection:
- Is there a risk of usage drift:

## 4. Risk identification
- Risk of leakage:
-Risk of unauthorized access:
- Third-party sharing risks:
- Risk of model bias:
- Delete the risk of not closing the loop:
- Log exposure risks:

## 5. Control measures
- Field cropping:
- Default desensitization:
- Approval mechanism:
- Audit traces:
- Test environment isolation:
- External call gateway:

## 6. Evaluation conclusion
- [ ] available online
- [ ] Can go online after rectification
- [ ] Forbidden to go online

## 7. Rectification items and responsible persons
| Correction items | Responsible person | Deadline | Status |
| :--- | :--- | :--- | :--- |
|  |  |  |  |
```

### 27.5.5 Audit log structure (JSONL example)

```json
{"event_time":"2026-03-10T10:15:01Z","actor":"algo_user_a","role":"algo_reader","action":"query","dataset":"dim_user_profile_masked","fields":["hashed_phone","age_band"],"record_count":200,"purpose":"feature_validation","approval_id":"APR-1029","policy_result":"allow_masked","trace_id":"trace-001"}
{"event_time":"2026-03-10T10:22:48Z","actor":"platform_job_17","role":"pipeline_runner","action":"preflight_check","dataset":"p09_release_bundle","fields":[],"record_count":0,"purpose":"deployment_gate","approval_id":"N/A","policy_result":"pass","trace_id":"trace-002"}
{"event_time":"2026-03-10T10:29:11Z","actor":"external_gateway","role":"api_gateway","action":"external_api_call","dataset":"prompt_payload","fields":["masked_phone","case_summary"],"record_count":1,"purpose":"assisted_summary","approval_id":"APR-1081","policy_result":"allow_after_redaction","trace_id":"trace-003"}
{"event_time":"2026-03-10T10:33:56Z","actor":"ops_user_b","role":"ops_admin","action":"export","dataset":"user_precise_location","fields":["geo_hash_12"],"record_count":50,"purpose":"troubleshooting","approval_id":"APR-1099","policy_result":"blocked","trace_id":"trace-004"}
```

### 27.5.6 Preflight Checklist (JSON Example)

```json
{
  "project_id": "P09-001",
  "preflight_version": "1.0.0",
  "checks": [
    {"name": "ropa_exists", "status": "PASS"},
    {"name": "classification_policy_exists", "status": "PASS"},
    {"name": "access_policy_exists", "status": "PASS"},
    {"name": "pii_rules_loaded", "status": "PASS"},
    {"name": "restricted_records_quarantined", "status": "PASS"},
    {"name": "redacted_records_remove_direct_pii", "status": "PASS"},
    {"name": "audit_log_enabled", "status": "PASS"},
    {"name": "incident_simulation_exists", "status": "PASS"},
    {"name": "postmortem_template_exists", "status": "PASS"},
    {"name": "deletion_path_declared", "status": "PASS"},
    {"name": "external_prompt_plaintext_block", "status": "PASS"},
    {"name": "cross_border_flag_reviewed", "status": "PASS"},
    {"name": "release_gate", "status": "PASS"}
  ],
  "overall_status": "PASS"
}
```

### 27.5.7 Incident response and review template

```md
# Privacy Incident Postmortem

## 1. Event overview
- Event number:
- Discovery time:
- End time:
- Scope of influence:
- Event level:

## 2. Trigger reason
- Direct cause:
- root cause:
- Whether there is a missing process involved:
- Whether there is a permission configuration error involved:

## 3. Impact Analysis
- Datasets involved:
- Involved fields:
- Number of records involved:
- Whether it was leaked:
- Whether it affects user rights:

## 4. Disposal process
- Alarm timeline:
- Blocking action:
- Temporary mitigation measures:
- Permanent fixes:

## 5. Responsibility and Improvement
| Problem | Responsible Team | Improvement Measures | Completion Time |
| :--- | :--- | :--- | :--- |
|  |  |  |  |

## 6. Follow-up inspection
- [ ] Policy updated
- [ ] Permissions have been converged
- [ ] Audit rules have been added
- [ ] Documentation updated
- [ ] Related items have been checked simultaneously
```

### 27.5.8 Governance Deliverable Mapping (Sample)

In order to illustrate that the governance template is not a paper design, the following table takes a common product of a privacy governance pipeline as an example to show how the deliverables are mapped to the corresponding governance capabilities:

| Deliverables | Governance Implications |
| :---------------------------- | :-------- |
| `compliance_scope.json` | Define Compliance Scope |
| `classification_policy.json` | Define grading strategy |
| `access_policy.json` | Define access and permission boundaries |
| `privacy_tech_options.json` | Defining privacy technology options |
| `raw_sensitive_records.jsonl` | Original sensitive sample |
| `classified_records.jsonl` | Classification results |
| `redacted_records.jsonl` | Desensitization results |
| `quarantine_records.jsonl` | Quarantine results |
| `audit_log.jsonl` | Audit traces |
| `access_alerts.jsonl` | Abnormal access alarm |
| `isolation_plan.json` | Isolation policy |
| `preflight_checklist.json` | Access control check before going online |
| `incident_simulation.json` | Event Simulation |
| `postmortem_report.json` | Accident review |
| `p9_metrics.json` | Indicator summary |
| `p9_test_results.json` | Check results |
| `p9_test_report.md` | Test report |

### 27.5.9 From template to platform: Evolution path of governance system

Many organizations cannot build a complete platform at once at the beginning, so they can evolve at the following pace:

**Phase One: Template Governance**
First unify the RoPA template, DPIA template, grading standards and go-live checklist.

**Phase 2: Configuration governance**
Convert templates to YAML/JSON strategies and incorporate them into code repository and version management.

**Phase 3: Automated Governance**
Realize automatic inspection, automatic blocking and automatic alarm through CI/CD, data gateway, log system and audit platform.

**Phase 4: Platform Governance**
Unify metadata, permissions, policies, approvals, traces, and incident responses into a governance platform to achieve global visibility and cross-project reuse.

---

## 27.6 Minimum link for governance implementation

The governance framework, access control templates and deliverable mapping have been shown above. This section does not expand on the complete project details, but extracts a reusable minimum governance link to explain how compliance requirements are connected in series in the system.

### 27.6.1 Specification and scope definition

The starting point of the governance link is not a processing script, but a structured definition of scope, classification, access boundaries, and technical options. Only by clarifying the purpose, data level, role boundaries and available technologies can subsequent processing have an executable basis.

This step focuses on: **How ​​sensitive data is identified, restricted, rewritten, audited and verified before entering the training or analysis system**. It determines whether subsequent pipelines are enforcing rules or just remediating them after the fact.

### 27.6.2 Classification, desensitization and isolation

After the original sensitive records enter the system, they need to be classified first and then desensitized, isolated or blocked according to the policy. The most critical thing here is not to "rewrite a few fields", but to convert the data from the original visible state to a controlled and usable state, and retain the judgment basis required for subsequent audits.

At least four questions must be answered at this stage:

* How does the system identify direct PII?
* Which records are determined to be restricted?
* Which fields can be retained for analysis and which must be removed?
* Which data can only enter the isolation area and cannot enter the general processing area?

### 27.6.3 Audit, Alarm and Access Control

The governance link cannot be stopped for offline processing. The system also needs to record access, output alarms, perform pre-checks, and incorporate exceptions into the incident response and review process.

What this stage completes is "how the system interprets its own behavior":

* Audit log output
* Abnormal access alarm
* Preflight result verification
* Event simulation
*Accident review

Without these capabilities, it will be difficult for the governance system to support online approval, exception tracking, and post-auditing.

### 27.6.4 Indicators and Acceptance

The role of governance indicators is not to pursue a large sample, but to confirm whether the closed loop is established. The following indicators correspond to the four types of capabilities: identification, processing, release and auditing:

* 13 total inspection items
* All passed
* Covers command-level inspection and data/product-level inspection

If these indicators can be stably generated and passed continuously, it means that the governance system has moved from document requirements to operational inspection logic.

### 27.6.5 Division of labor with project-type chapters

The governance chapter focuses on the framework, templates, access control and acceptance logic, while the project chapter is more suitable for specific script organization, data directory, running interface and expansion path. This division of labor has two advantages:

* The governance chapter remains generic to facilitate migration to different business systems.
* The project chapter retains implementation details to facilitate independent development of data products and project evolution.

A project-based pipeline like P09 is more suitable to be fully developed in an independent project chapter; in this chapter, it is more suitable as a lightweight template after the governance template is implemented.

---

## 27.7 From compliance framework to organizational capabilities: implementation suggestions

The real difficulty of the governance system is not to write a beautiful framework, but to collaborate and execute it continuously across teams. Below is a set of suggestions for organizational implementation.

### 27.7.1 Don’t try to do all abilities at once

For most teams, the most realistic path is not to “platform it all in one step”, but to first seize a few key control points:

1. Unify the grading standards first
2. Reunify RoPA and DPIA templates
3. Then add the strategy file into the code repository
4. Then connect the key checks to CI/CD
5. Finally, gradually build a unified audit and metadata platform

### 27.7.2 Make “high value default items” first

If resources are limited, the first to go online should be the default rules that are prone to problems if they are missing, such as:

* Logs are prohibited from printing direct PII
* The test environment prohibits importing production clear text snapshots
*C3 data cannot be exported directly by default
* High-risk uses must be registered before use
* The desensitization gateway must be passed before calling the external model

These default rules immediately create guardrails that significantly reduce low-level risks.

### 27.7.3 Translate compliance language into R&D language

Communication breakdowns often occur between business, legal, platform and algorithm teams because they speak different languages:

* Legal affairs say "Legal basis, consent, proportionality"
* Business says "goals, conversions, efficiency"
* The platform says "Permissions, tables, interfaces, logs"
* The algorithm says "features, training, inference, effect"

What the governance system needs to do is to establish a translation layer.
For example, put "Legal Basis" into the RoPA field, "Minimum Necessity" into the field whitelist, "Deletion Rights" into the deletion link, "Audit Requirements" into the log schema, and "High Risk Processing" into the pre-check blocking rules.

### 27.7.4 Let audit and R&D share the same source of truth

In many organizations, one set of accounts is used for auditing, and another system is used for R&D. In the end, the two parties cannot match up.
It makes more sense to have audit, compliance, and R&D share the same structured source of truth: the same metadata, the same set of policy versions, the same set of approval records, and the same log link. Only in this way can we reduce the gap between "reporting one thing externally and doing another internally".

### 27.7.5 Indicator-based governance rather than slogan-based governance

Whether governance is effective ultimately depends on indicators. Consider establishing the following governance indicator system:

* Effective RoPA coverage
* DPIA completion rate
* Sensitive field annotation coverage
* Test environment desensitization coverage
* Audit log completeness rate
* On-time completion rate of deletion requests
* Abnormal export alarm closing rate
* Pre-check blocking hit rate
*Runtime permission convergence rate

Once indicators are established, governance changes from an “initiative” to a “managed object.”

---

## 27.8 Summary of this chapter

This chapter starts from a real and typical engineering dilemma and explains why data compliance cannot be used as an approval attachment before going online, but must be moved forward as a system architecture constraint. We have established a complete framework from cognition, system to engineering.

First, we discussed the high cost of compliance post-processing and argued that what Privacy by Design really means is incorporating authorization, classification, desensitization, auditing, deletion, and approval requirements into the requirements and architecture phases. Secondly, we established data classification and classification, risk matrix and responsibility chain, so that the system can clarify the governance boundaries between different data, different scenarios and different roles. Again, we illustrate how compliance requirements are embedded in the R&D pipeline around RoPA, DPIA, audit traces, and CI/CD pre-checking. Finally, we converged these requirements into a minimum governance link, linking policy generation, data processing, isolation auditing, alarm response, pre-flight inspection and accident review into a complete closed loop.

The core idea that this chapter wants to convey is: **Compliance is not to put the brakes on innovation, but to install guardrails on the system. **
A system without guardrails may run faster in the short term, but once it encounters real-world regulations, user rights, and auditing requirements, the cost will be far greater than the initial governance investment. On the contrary, by turning compliance into a set of system capabilities, the team will not only not lose efficiency, but will gain clearer data boundaries, a more stable collaboration process, and more sustainable business expansion capabilities.

---

## 27.9 Extended thinking

1. What is the biggest difference between prompt compliance and traditional data desensitization in generative AI applications?
2. Why is it said that the "right to delete" does not test the ability of a single table, but the ability of full-link blood and life cycle management?
3. If your organization is unable to build a complete governance platform in the short term, what default guardrails should be put in place first?
4. When business goals and compliance requirements conflict, how to find a compromise solution through the ideas of "minimum necessary" and "alternative fields"?
5. For cross-border, multi-regional, and multi-product line businesses, how can the governance system maintain unity while allowing differentiated implementation of rules?

