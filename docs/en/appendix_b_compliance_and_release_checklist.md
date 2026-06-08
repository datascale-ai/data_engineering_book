# Appendix B: Compliance and Release Checklist

## B.1 Purpose of This Appendix

This appendix focuses on the checkpoints that determine whether a dataset can continue moving downstream. It is not a collection of abstract compliance slogans. It asks concrete engineering questions: can a batch of data enter annotation, enter training, be released externally, support a course experiment, or connect to an online system?

In large-model data engineering, the most dangerous situation is often not that nobody has heard of a relevant regulation. It is that the team gradually treats compliance as an approval form to be completed at the end of the project. In reality, compliance and release checks must run through the data lifecycle. Unclear sources, ambiguous authorization, unstable anonymization, evaluation contamination, missing resource statements, and overexposed teaching environments are all much more expensive to fix late than early.

This appendix is therefore not legal advice. It is an engineering checklist framework that technical leads, project managers, course owners, and compliance contacts can use with the same vocabulary.

## B.2 Why Compliance Checks Must Shift Left

If compliance is checked only before release, teams usually encounter three expensive forms of rework. First, **source rework**: the data has already been collected and cleaned before the team discovers that the original authorization does not allow model training or redistribution. Second, **annotation rework**: annotation is complete before the team realizes that sensitive fields were not properly anonymized. Third, **release rework**: a benchmark is ready to publish before the team discovers unstable train/test boundaries or conflicts between external licenses and leaderboard rules.

A more stable approach is to split compliance into four gates:

1. Source and authorization checks before data ingestion.
2. Sensitivity and delegation-boundary checks before annotation and processing.
3. Data-use boundary checks before training and evaluation.
4. Public-exposure checks before external release or system launch.

Once these gates exist, many problems that would otherwise explode at the end of a project can be contained earlier.

## B.3 Pre-Ingestion Checklist

### B.3.1 Source and Authorization Are the First Gate

The first question before data ingestion is not whether the data is worth collecting, but whether it can be collected legally and contractually. At minimum, check the fields in Table B-1.

| Check Item | Question to Answer | Common Risk | Recommended Action |
| :-- | :-- | :-- | :-- |
| Source owner | Who provides the data? | Broken source chain or unclear resale path | Keep source notes and contact information |
| Usage license | Does it allow training, evaluation, and redistribution? | Allowed for papers but restricted for commercial or public use | Maintain license allowlists and graylists |
| robots / ToS | Does crawling conflict with site rules? | Rule violation and takedown risk | Preserve rule snapshots and timestamps |
| Jurisdiction boundary | Does the data involve cross-border or regulated industry data? | Unclear transfer boundary | Confirm with legal and security contacts |
| Update cycle | Does the source change over time? | The same version cannot be reproduced later | Freeze crawl windows and snapshots |

Source checks matter because every downstream engineering action amplifies source problems. The closer a project gets to model training and public release, the more expensive it becomes to change source strategy.

### B.3.2 Classify Personal and Sensitive Information Before Processing

If samples may contain names, contact information, ID numbers, medical records, location traces, information about minors, internal enterprise documents, or trade secrets, the team should not immediately debate which model to use for anonymization. It should first classify the information.

| Level | Typical Content | Default Strategy |
| :-- | :-- | :-- |
| L0 | Public, low-risk, explicitly authorized data | Enter the standard governance flow |
| L1 | Ordinary personal information or business fields | Minimize collection and anonymize before transfer |
| L2 | Sensitive personal information, medical, financial, educational, and similar data | Require special approval and isolated processing |
| L3 | Highly sensitive, classified, or strongly contract-restricted data | Generally exclude from general training pipelines |

The common mistake is treating anonymization as a universal pass. Often the issue is not whether a field has been masked, but whether the task itself depends on sensitive attributes. If it does, the team should first redesign the task boundary rather than assume one regex replacement can solve the problem.

## B.4 Annotation, Delegation, and Third-Party Collaboration

### B.4.1 Annotation Platforms Are Not Automatic Compliance Boundaries

Many teams upload samples to an external annotation platform and assume that the platform's permission system is sufficient. That is risky. At minimum, the annotation stage must confirm:

1. Whether the third party is explicitly authorized to access this category of data.
2. Whether annotation instructions disclose internal information beyond the task need.
3. Whether annotation results may flow back into other projects.
4. Whether logs, screenshots, previews, or caches create new exposure surfaces.

| Check Item | Question to Answer | Recommended Action |
| :-- | :-- | :-- |
| Delegation boundary | Who can see raw samples? | Use role layering and least privilege |
| Data anonymization | Has required processing been completed before annotation? | Isolate anonymized and raw versions |
| Annotation instructions | Do the guidelines disclose internal information? | Review instructions before release |
| Logs and caches | Does the platform retain previews or downloads? | Define cleanup rules and retention periods |
| Disputed samples | Can samples flow into public discussion spaces? | Isolate them and do not send raw text externally |

### B.4.2 University Collaboration Requires Clear Output Boundaries

In university collaboration, a common risk is an unclear boundary between project work and publishable research. Samples allowed for internal project use may not be allowed in paper appendices. Screenshots allowed in class demonstrations may not be packaged as an external dataset.

Define the output boundary at the start:

- Which results are for internal project use only.
- Which statistics can appear in papers while raw samples cannot be distributed.
- Which data can enter course images but cannot be downloaded.
- Which content can become a benchmark and which must remain an internal test set.

Without these boundaries, teams often reach a painful state where the engineering work is complete but the contract boundary does not permit release.

## B.5 Pre-Training and Pre-Evaluation Checklist

### B.5.1 A Legal Training Set Does Not Make the Evaluation Set Safe

Training and evaluation are often treated as one governance problem, but their risks differ. Training sets focus on source, authorization, sensitivity, and task fit. Evaluation sets must additionally address contamination, isolation, and comparison fairness.

| Check Item | Before Training | Before Evaluation |
| :-- | :-- | :-- |
| Source and license | Is training allowed? | Is public testing or leaderboard use allowed? |
| Data minimization | Are unnecessary fields retained? | Are standard answers or hints leaked? |
| Version freeze | Is the training version locked? | Are the test version and scripts locked? |
| Contamination check | Has it touched public test sets? | Has the test set been polluted by training corpora? |
| Resource statement | Can training resources be recorded? | Are submission resource conditions comparable? |

For open benchmarks, the evaluation set is not simply "good if it is high quality." External teams must trust that comparison conditions are stable and fair.

### B.5.2 LLM Judges and Automatic Anonymization Must Be Audited

As more workflows use large models for judging, summarization, classification, and anonymization, the **auxiliary model itself** must enter the compliance view. Check whether:

- The auxiliary model sends input samples to an external service.
- The service terms allow retention, training use, or human review of inputs.
- Judge conclusions determine official labels or launch decisions.
- Failed anonymization samples enter a human review channel.

From a governance perspective, calling an external API with sample content is a new data exposure event, even if the technical team thinks of it as an internal convenience.

## B.6 External Release and Public Benchmark Checklist

### B.6.1 Prepare Four Documents Before Release

A mature dataset or benchmark should have at least four document types before external release:

1. A data or benchmark card describing the task, sample structure, splits, and limits.
2. A license and usage note explaining what is allowed and prohibited.
3. A baseline bundle with a minimally reproducible baseline.
4. An update and dispute-handling mechanism describing versions, feedback, and withdrawal paths.

Publishing only samples and a paper link makes safe reuse difficult and dispute handling slow.

### B.6.2 Leaderboard Governance Matters More Than "Whether It Is Public"

Many benchmarks fail not because they are insufficiently public, but because they lack governance after publication.

| Check Item | Question to Answer |
| :-- | :-- |
| Submission method | What can be uploaded and what is prohibited? |
| Resource statement | Must model size, inference budget, and retrieval resources be reported? |
| Human review | What triggers manual review? |
| Removal rules | What happens after contamination, cheating, or unauthorized access is found? |
| Teaching isolation | Is the course version separate from the public leaderboard version? |

Clear rules prevent a public leaderboard from becoming a confusing score wall.

### B.6.3 Minimal Release Sign-Off Form

Every formal public release should create a minimal sign-off form.

| Field | Required Content |
| :-- | :-- |
| asset_name | Dataset, benchmark, or course image name |
| release_version | External release version |
| owner | Data owner and evaluation owner |
| source_check | Whether source and license review is complete |
| privacy_check | Whether anonymization and sensitive-information checks are complete |
| contamination_check | Whether train/test contamination checks are complete |
| baseline_ready | Whether the baseline bundle is complete |
| rollback_path | Withdrawal path after a problem is found |
| public_notice | Location of the release announcement or documentation |

The value of this form is that it records who approved the release and what evidence the decision relied on.

## B.7 Pre-Launch Checklist for Systems

### B.7.1 "Runs" and "Can Launch" Are Separated by at Least Five Control Layers

A model or data workflow running in an experiment environment is not enough for production. At least five control layers are needed.

| Control Layer | Key Question | Typical Checkpoints |
| :-- | :-- | :-- |
| Permissions | Who can access samples and results? | RBAC, least privilege, audit logs |
| Isolation | Are test and production environments separated? | Environment partitions, key management |
| Content | Can outputs violate rules or echo sensitive content? | Red-line samples, refusal policies |
| Rollback | Can the system be disabled quickly after an incident? | Version locking, canary switches |
| Records | Can the team review what happened? | Request logs, evaluation snapshots, incident ledger |

In large-model applications, many incidents occur not because the model suddenly becomes worse, but because data, prompts, retrieval sources, or evaluation definitions change without being caught by launch controls.

### B.7.2 Teaching Launch and Product Launch Have Different Thresholds

Course experiments often allow more explanatory process, more logging, and slower responses. Product environments usually cannot. Do not treat a course image as a production system, and do not expose a production audit environment directly to course practice.

Maintain separate versions for:

- Teaching images.
- Research reproduction experiments.
- Online service releases.

They may reuse data assets, but they should not share the same exposure surface and permission boundary.

### B.7.3 Special Checklist for Teaching Scenarios

University collaboration and course experiments are often misclassified as low risk because they involve fewer users and shorter cycles. In practice, teaching has its own risks: account sharing, raw samples pasted into reports, classroom recordings spreading, and mid-semester hot updates causing answer drift.

| Check Item | Question to Answer | Recommended Action |
| :-- | :-- | :-- |
| Version freeze | Will the version stay stable during the semester? | Lock the image before the course starts |
| Permission scope | Can students download raw data? | Disable raw-layer downloads by default |
| Sample display | Can screenshots leak sensitive content? | Use teaching-safe anonymized samples |
| Assignment submission | Can submissions contain raw data? | Add submission rules and automatic scanning |
| TA operations | Who handles questions and incidents? | Establish duty and escalation paths |

This checklist does not replace a product-launch checklist; it answers a different question: whether the system can be taught safely.

### B.7.4 Additional Checks for Cross-Border Flows, External APIs, and Third-Party Models

As workflows rely more on external foundation models, cloud OCR, cloud storage, and SaaS annotation platforms, teams must identify whether **samples leave the original control domain**.

Check whether:

1. The call chain sends raw text, images, or audio to external services.
2. Those services retain logs, caches, or training rights.
3. Cross-border transfer, third-party subprocessors, or multilevel subcontracting are involved.
4. The team has fallback plans if an external service is interrupted or its terms change.

Many post-launch issues are not about model quality. They come from hidden data exposure points in the dependency chain.

## B.8 Incident Response and Withdrawal Mechanisms

### B.8.1 Public Release Without Withdrawal Is High-Risk Release

Once a dataset, leaderboard, or course resource is public, assume that one day:

- Someone reports infringement or sensitive-information leakage.
- Train/test contamination is discovered.
- Baseline code contains a serious error.
- The leaderboard receives abnormal or suspicious submissions.
- A teaching image exposes an unauthorized access path.

Public assets need a withdrawal and repair process:

1. Intake: who receives reports.
2. Triage: whether immediate takedown is required.
3. Temporary action: hide downloads, freeze the leaderboard, or disable the image.
4. Investigation and review: confirm scope, cause, and remedies.
5. Public notice: publish revision notes and version changes.

### B.8.2 Minimal Incident Ledger

| Field | Description |
| :-- | :-- |
| incident_id | Incident identifier |
| reported_time | First report time |
| affected_asset | Dataset, leaderboard, or system involved |
| risk_level | Risk level |
| temporary_action | Temporary handling action |
| root_cause | Root cause |
| public_notice | Public notice link or note |
| preventive_action | Follow-up prevention measure |

Institutionalizing this ledger prevents the same type of problem from recurring across semesters, projects, and owners.

### B.8.3 High-Risk Red Flags

Use these red flags on a project board:

- Data sources can be traced only to personal cloud drives, chat files, or secondary repost links.
- All annotators can see all raw samples by default.
- Teaching experiments reuse production environments or production keys.
- Train/test splits are changed across versions without notice.
- A public leaderboard accepts uploads without resource statements or removal rules.
- External APIs see sample text, but nobody can explain their log-retention policy.
- After the project lead leaves, nobody can identify the withdrawal or repair entry point.

If two or three of these appear, the project should move into a special review instead of continuing as routine work.

## B.9 Role Division

Compliance and launch checks fail easily when "everyone knows they matter" but nobody owns them. At minimum, define four roles.

| Role | Main Responsibilities |
| :-- | :-- |
| Data owner | Source, authorization, versioning, takedown decisions |
| Evaluation owner | Test isolation, leaderboard definitions, baseline review |
| Security/compliance contact | Approval boundaries, sensitive-information handling, external report coordination |
| Teaching owner | Course images, semester versions, classroom-use boundaries |

These do not have to be four separate jobs, but the responsibilities cannot be absent.

### B.9.1 Questions for a Quarterly Compliance Review

For long-running projects, run a lightweight compliance review each quarter. It should answer:

1. Did any new data source this quarter change its license boundary?
2. Did the project add any external service call chain, and is it recorded?
3. Were there any sample withdrawals, anonymization patches, or leaderboard disputes?
4. Are teaching, research, and public versions still clearly isolated?
5. Which layer should be improved next quarter: source governance, permission governance, or withdrawal flow?

This turns compliance from a last-minute blocker into part of stable project evolution.

### B.9.2 Withdrawal Drills Should Happen Before Real Incidents

Many teams write withdrawal mechanisms but never test whether they can locate, freeze, announce, and rebuild affected versions within half a day. This is like writing a disaster-recovery plan and never running a drill.

Once per semester or quarter, run a small withdrawal drill:

1. Confirm that sample location can be traced from the release version back to the original source and intermediate versions.
2. Confirm that training sets, evaluation sets, teaching images, and public downloads can identify affected scope.
3. Confirm whether leaderboards, baselines, and course experiments need freezing or notices.
4. Confirm that external report handlers and internal owners can communicate without gaps.
5. Confirm that revised versions can be regenerated quickly while preserving historical notes.

The point is not only emergency response. It reveals whether a dataset truly has a governable asset structure.

## B.10 Summary

This appendix translates compliance and release concerns into a checklist that engineering teams can execute.

First, compliance is not a final add-on. It is a continuous set of gates in the data lifecycle.

Second, training, evaluation, public release, course reproduction, and product launch have different boundaries and must be checked separately.

Third, long-term risk is reduced not by one approval form but by source records, version freezes, incident withdrawal, and clear role ownership.

## References

Cybersecurity Law of the People's Republic of China. Available at: https://www.gov.cn/xinwen/2016-11/07/content_5129723.htm

Data Security Law of the People's Republic of China. Available at: https://www.gov.cn/xinwen/2021-06/11/content_5616919.htm

Personal Information Protection Law of the People's Republic of China. Available at: https://www.gov.cn/xinwen/2021-08/20/content_5632486.htm

National Institute of Standards and Technology (2023) AI Risk Management Framework (AI RMF 1.0). Available at: https://www.nist.gov/itl/ai-risk-management-framework

Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act). Available at: https://eur-lex.europa.eu/eli/reg/2024/1689/oj

Mitchell M, Wu S, Zaldivar A, Barnes P, Vasserman L, Hutchinson B, Spitzer E, Raji I D, Gebru T (2019) Model Cards for Model Reporting. In: Proceedings of the Conference on Fairness, Accountability, and Transparency, pp 220-229.
