# Appendix C: Cost Estimation and Resource Templates

## C.1 Purpose of This Appendix

The most underestimated part of a data-engineering project is often not technical complexity but cost structure. Many teams ask only how many GPUs are needed for training, while leaving unanswered questions such as: who collects and cleans samples, how many person-days annotation review requires, how much long-term evaluation and leaderboard maintenance costs, how much multimodal storage and egress traffic will cost, and whether teaching images need fixed resources for an entire semester.

This appendix does not provide a universal price list. It provides a method, tables, and review templates for cost estimation. If the method is stable, teams can quickly update estimates even when unit prices change. Without a method, knowing one moment's unit price is not enough for reliable budgeting.

## C.2 Why Costs Should Be Split by Lifecycle

Budget debates often lose focus because different roles look at different cost objects. Researchers may look only at training GPUs. Platform teams care about storage and feedback flows. Course owners care about images and teaching-assistant time. Management cares about quarterly investment and delivery cadence.

A more useful split for data-engineering projects is lifecycle-based:

| Cost Category | Core Question | Typical Unit |
| :-- | :-- | :-- |
| Data ingestion | What does it cost to obtain samples? | Person-days, API calls, crawl jobs |
| Cleaning and processing | What does it cost to turn samples into usable assets? | CPU/GPU hours, person-days |
| Annotation and review | How is supervision created? | Per sample, per hour, per QA round |
| Training and inference | What does model consumption of data cost? | GPU hours, tokens, throughput |
| Evaluation and release | What do comparison, leaderboard, and public maintenance cost? | Task batches, human review, person-days |
| Teaching and operations | What does long-term reproducibility cost? | Images, accounts, teaching-assistant time |

This split forces the team to recognize a key reality: the expensive part in the long term is often not the first training run, but governance and maintenance afterward.

## C.3 Overall Estimation Formula

A practical total-cost formula is:

\[
\text{Total Cost} =
C_{\text{ingest}} +
C_{\text{clean}} +
C_{\text{annotate}} +
C_{\text{train/infer}} +
C_{\text{evaluate}} +
C_{\text{release/teach}}
\]

For execution, each item can be split into fixed cost, variable cost, and rework-risk reserve:

\[
C_i = C_i^{fixed} + C_i^{variable} + C_i^{risk}
\]

Many budgets lose control not because the variable term is wrong, but because the risk term is missing. Re-annotation, anonymization rework, leaderboard review, sample withdrawal, and mid-semester image repair are not rare edge cases in real projects.

## C.4 Data Ingestion and Cleaning Cost Templates

### C.4.1 Ingestion Cost Is Not Just "How Much Was Collected"

The ingestion stage should estimate at least three costs: ingestion-script development, data-pull execution, and source/license verification.

| Item | Unit | Quantity | Unit Cost / Hours | Subtotal | Notes |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Ingestion script development | Person-days |  |  |  | Crawling, API access, parser adaptation |
| Data pull jobs | Batches |  |  |  | Includes retries |
| Source and license verification | Person-days |  |  |  | Partners, agreements, log organization |
| Storage landing | TB/month |  |  |  | Raw-layer storage |
| Data spot check | Person-days |  |  |  | First quality baseline |

If a project has many sources, source verification should be listed separately rather than hidden inside script development, because it often becomes a hidden critical path for launch.

### C.4.2 Cleaning Cost Often Underestimates Rework

Cleaning cost usually includes rule development, batch-processing resources, abnormal-sample review, and reruns.

| Cost Item | Estimation Method |
| :-- | :-- |
| Rule development | Number of rules x average design and test hours |
| Batch processing | Data volume x unit processing resource |
| Failed-sample review | Number of abnormal samples x review time per sample |
| Rerun cost | Main-pipeline resource per rerun x expected rerun count |

In document, multimodal, and speech projects, rerun cost is often ignored. Once OCR strategy, deduplication thresholds, or audio-slicing rules change, entire intermediate layers may have to be regenerated.

## C.5 Annotation, Review, and Synthetic Data Cost Templates

### C.5.1 Separate First-Pass Annotation from Review

Estimating annotation as one price per sample almost always underestimates the budget. Split annotation into:

- First-pass annotation.
- Review.
- Arbitration.
- Guideline iteration.
- Platform and management cost.

| Item | Unit | Quantity | Unit Cost / Hours | Subtotal | Notes |
| :-- | :-- | :-- | :-- | :-- | :-- |
| First-pass annotation | Item/page/minute |  |  |  | Depends on task granularity |
| Review | Item/page/minute |  |  |  | Estimate by sampling rate |
| Arbitration | Item/page/minute |  |  |  | Budget for high-dispute samples |
| Guideline iteration | Person-days |  |  |  | Guidelines and QA meetings |
| Platform maintenance | Month |  |  |  | Accounts, permissions, export |

For preference alignment, multimodal region annotation, or controllable speech-emotion tasks, arbitration and guideline iteration can become bigger bottlenecks than first-pass annotation.

### C.5.2 Synthetic Data Is Not Zero Labor Cost

When teams hear that large models can generate synthetic data, they often assume cost will drop sharply. In practice, the first generation pass may become cheaper, but verification and filtering may not.

| Cost Item | Description |
| :-- | :-- |
| Prompt/template design | Who defines the generation protocol |
| API or inference cost | Tokens or GPU cost for generation |
| Filtering and scoring | Automatic judges and rule filters |
| Human spot checks | Verification that samples are actually usable |
| Regeneration | Reprocessing failed samples |

The safest budget is not "the API bill for generating 100,000 samples." It is the full-chain cost of generation, filtering, sampling, and rework.

## C.6 Training and Inference Resource Templates

### C.6.1 Training Estimates Should Not Look Only at GPU Count

Training budgets are often simplified to "how many cards for how many days." The real cost also depends on effective throughput, probability of failed reruns, and number of tuning rounds.

| Item | Unit | Quantity | Unit Cost / Hours | Subtotal | Notes |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Training GPUs | GPU hours |  |  |  | Formal training |
| Tuning GPUs | GPU hours |  |  |  | Small-scale experiments |
| Evaluation inference | GPU hours / tokens |  |  |  | Includes slice evaluation |
| Data preparation CPU | Core hours |  |  |  | Tokenization, packing, validation |
| Failed-rerun reserve | Percentage |  |  |  | Keep as a separate line |

If a team does not reserve resources for failed reruns, the budget usually becomes inaccurate after the second experiment. For engineering projects, a 10-30% risk reserve is often more valuable than pretending the first estimate is exact.

### C.6.2 Split Inference Cost by Scenario

Inference cost should be split into at least three scenarios.

| Scenario | Characteristics | Estimation Focus |
| :-- | :-- | :-- |
| Offline batch evaluation | Batchable and queueable | Throughput, concurrency, result caching |
| Online service | Latency-sensitive | Peak concurrency, context length, fallback strategy |
| Teaching experiment | Time-window concentration | Semester images, account limits, retry cost |

Mixing these three makes it hard to explain why online budgets are high or why course weeks produce resource spikes.

## C.7 Evaluation, Leaderboard, and Public Maintenance Costs

### C.7.1 Long-Term Benchmark Cost Often Exceeds Initial Release Cost

A public benchmark costs more than its first release. Continuing costs include:

- Baseline reproduction and upgrades.
- Submission validation.
- Manual review of suspicious results.
- Dispute handling.
- Documentation and image maintenance.
- Teaching-version locking.

| Item | Cycle | Unit | Estimation Method |
| :-- | :-- | :-- | :-- |
| Baseline reruns | Quarterly / half-yearly | GPU hours | Number of baselines x cost per evaluation |
| Submission review | Monthly | Person-days | Submission count x review hours |
| Issue handling | Monthly | Person-days | Historical average issue volume |
| Documentation update | Per version release | Person-days | Card, FAQ, release notes |
| Teaching image maintenance | Semester | Person-days / instances | Pre-course freeze and in-semester patches |

Without this budget line, a public benchmark can be lively at launch and unreliable half a year later.

### C.7.2 Why Evaluation Cost Often Rises Late

Early in a project, evaluation may feel cheap because the team runs a few baselines internally. Once a dataset enters public release, courses, or multi-team collaboration, evaluation cost rises because:

- More slices are added, so one evaluation is no longer just one total score.
- More baselines must preserve historical comparability.
- Suspicious results require human review.
- Course experiments need a stable evaluation environment that cannot follow every mainline change.

Maintain two ledgers: internal R&D evaluation and public-maintenance evaluation. The first serves fast iteration; the second serves stable comparison.

## C.8 Storage, Network, and Archival Costs

### C.8.1 Multimodal Projects Must List Storage Separately

Text projects can sometimes survive rough disk estimates. Document, image, audio, video, and agent-trajectory projects cannot.

| Layer | Description |
| :-- | :-- |
| Raw files | Largest layer, but not always frequently accessed |
| Intermediate artifacts | OCR, features, transcoding, indexes |
| Curated versions | Official training and evaluation inputs |
| Release images | External release and course reproduction versions |
| Archive layer | Cold storage and long-term preservation |

Without this layering, teams often discover late that training was not the expensive part; permanently retaining every intermediate artifact was.

### C.8.2 Archival Strategy Determines Maintainability Over the Next Three Years

Archiving is not simply compressing old versions. It determines whether future review and reproduction remain possible.

A good archive budget answers:

1. Which versions must be kept long term.
2. Which versions keep only metadata and pointers.
3. Which course images must be frozen by semester.
4. Which public releases must be rollback-capable.

Projects without an archive strategy often save a little storage cost while losing reviewability.

### C.8.3 Network Egress and External Calls Should Not Be Hidden Under "Other"

In multimodal projects, RAG projects, and open leaderboards, network and external-call costs are often placed under a vague "other" category. This weakens budget judgment.

| Category | Common Source | Why It Must Be Separate |
| :-- | :-- | :-- |
| Egress traffic | Object-storage downloads, course image pulls | Peaks often align with course milestones |
| External APIs | Judges, OCR, translation, anonymization | Cost can scale rapidly with call volume |
| External synchronization | Public release, image replication, cross-region backup | Strongly tied to compliance and availability strategy |

Separating these costs lets the team ask whether growth is caused by real project scale, release style, course usage, or dependency choices.

## C.9 Three Reusable Resource Templates

### C.9.1 Small Research or Course Template

For labs, single courses, and prototypes.

| Module | Budgeting Logic |
| :-- | :-- |
| Data processing | Person-days first, resources second |
| Annotation | Small, high-quality samples; increase review ratio |
| Training | Estimate experiment rounds before formal runs |
| Evaluation | Emphasize slices and reviews rather than leaderboard expansion |
| Teaching | List images and teaching-assistant hours separately |

### C.9.2 Medium Team Project Template

For multi-member collaboration, quarterly delivery, and internal or external users.

| Module | Budgeting Logic |
| :-- | :-- |
| Data ingestion | Build fixed entry points and version strategy |
| Cleaning governance | Separate rerun cost and abnormal-review cost |
| Training and inference | Split formal training, tuning, and evaluation ledgers |
| Release maintenance | Treat benchmark and issue operations as long-term items |
| Compliance | Count approval, anonymization, and withdrawal flows as management cost |

### C.9.3 Open Benchmark or University Collaboration Template

For assets that require public release, course reproduction, and long-term maintenance.

| Module | Budgeting Logic |
| :-- | :-- |
| Data versioning | Keep at least raw, curated, and release layers |
| Documentation | Do not omit cards, FAQs, or release notes |
| Leaderboard | Budget for review, recheck, and removal mechanisms |
| Teaching | List frozen semester versions and experiment scripts |
| Handoff | Reserve cleanup time for each semester or version |

### C.9.4 Example Estimation Order for a University Dataset Release

For a university collaboration dataset moving from organization and evaluation to public release, estimate in this order:

1. Ingestion and license verification, because release depends on it.
2. Cleaning, anonymization, and version packaging, because stable release depends on them.
3. Baselines and evaluation scripts, because comparable release depends on them.
4. Teaching images, leaderboard operations, and issue handling, because long-term survival depends on them.

This order is opposite to many teams' intuition, but it matches the idea that a dataset is not merely pre-training raw material. It is an engineering asset that consumes maintenance resources over time.

### C.9.5 Budget Ledger Fields to Keep

Do not create a new spreadsheet from scratch every time. Maintain a continuous budget ledger.

| Field | Description |
| :-- | :-- |
| budget_cycle | Quarter, semester, or version cycle |
| asset_scope | Dataset, benchmark, or course involved |
| planned_cost | Budgeted value |
| actual_cost | Actual value |
| variance_reason | Reason for variance |
| reusable_asset_output | Long-term asset created in this cycle |
| one_off_cost | One-time cost |
| maintenance_cost | Future maintenance cost |

With these fields, a team can learn which investments become long-term assets and which expensive items are only reactive repairs.

## C.10 Quarterly Reviews Should Not Ask Only "Did We Overspend?"

The common mistake in budget reviews is checking only whether spending exceeded the plan. More useful questions are:

1. Did overspending come mainly from training, rework, annotation, or maintenance?
2. Which previously invisible cost must now enter the budget table?
3. Which expenses created sustainable assets, and which merely patched holes?
4. Were teaching, public benchmark, and production versions improperly mixed?
5. Which layer should be controlled next quarter, rather than which number?

The purpose of a budget is not to turn the project into a finance game. It is to explain which kind of data value the resources are buying.

### C.10.1 Three Charts Worth Keeping in Quarterly Reviews

For communicable budget reviews, keep three summary charts:

1. Lifecycle cost distribution: ingestion, cleaning, annotation, training, evaluation, and maintenance.
2. Budget-variance source: rework, call-volume growth, image maintenance, or human review.
3. Asset-creation comparison: which reusable versions, templates, or public assets were created by this quarter's spending.

These charts translate financial language back into engineering language.

## C.11 Summary

This appendix establishes cost-estimation methods and resource templates for data-engineering projects.

First, costs must be split by lifecycle; otherwise teams see only training cost and miss governance cost.

Second, the easiest items to underestimate are not unit prices, but rework, review, maintenance, and teaching reproduction.

Third, mature cost management is not only about saving money. It makes the relationship between resource investment and data-asset value explainable.

## References

Patterson D, Gonzalez J, Le Q, Liang C, Munguia L, Rothchild D, So D, Texier M, Dean J (2021) Carbon Emissions and Large Neural Network Training. arXiv preprint arXiv:2104.10350.

Narayanan D, Shoeybi M, Casper J, LeGresley P, Patwary M, Catanzaro B (2021) Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. In: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis.

Kwon W, Li Z, Zhuang S, Sheng Y, Zheng L, Yu C H, Gonzalez J E, Zhang H, Stoica I (2023) Efficient Memory Management for Large Language Model Serving with PagedAttention. In: Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, pp 611-626.

Kubernetes Documentation. Available at: https://kubernetes.io/docs/

vLLM Documentation. Available at: https://docs.vllm.ai/
