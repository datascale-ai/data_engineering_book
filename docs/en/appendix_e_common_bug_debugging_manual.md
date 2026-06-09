# Appendix E: Common Data-Engineering Bug Debugging Manual

## E.1 Purpose of This Appendix

This appendix addresses one of the most common and time-consuming problem types in data engineering: the pipeline "looks right," but the result is wrong. It is not a generic incident catalog. It is a debugging manual written for the context of this book, helping readers turn common problems from experience-driven troubleshooting into objects that can be located, classified, and regression-tested.

In large-model data engineering, bugs rarely live only in code. They can appear in data sources, splitting, deduplication, masking, parsing, annotation, evaluation, caching, permissions, write-back, or version references. More difficult, they often compound one another: "the metric did not change but the sample is wrong," "the sample is right but the version is wrong," or "the version is right but the release is wrong." Research on machine-learning technical debt and industrial case studies both show that data dependencies, configuration drift, test gaps, and cross-role collaboration issues are amplified during long-term maintenance (Sculley et al. 2015; Amershi et al. 2019).

This appendix therefore emphasizes **debugging order** rather than isolated repair tricks. Locate the layer first, classify the symptom second, and discuss fixes last.

## E.2 General Debugging Principles

1. Confirm whether the problem is reproducible.
2. Locate the failing layer before judging whether the issue propagated across layers.
3. Check the data version before the code version.
4. Find the smallest suspicious input before expanding the scope.
5. Preserve evidence before repairing.
6. Fix logic before tuning thresholds; fix structure before symptoms.
7. Verify that a regression test catches the problem before declaring it resolved.

Many teams jump directly into "change the code." This creates two side effects: original evidence disappears, and similar issues recur later. A safer method treats a bug as a complete loop: discovery, reproduction, localization, fix, verification, regression, and record.

## E.3 Common Bug Categories

Table E-1 gives a classification aligned with this book.

| Category | Common symptom | First checkpoint |
| :-- | :-- | :-- |
| Source issue | Original sample missing, authorization unclear, version drift | Source record, snapshot, license |
| Parsing issue | Missing field, table shift, OCR garbling | Parser, encoding, layout rule |
| Cleaning issue | False deletion, over-deduplication, over-masking | Rule version, threshold, sampled records |
| Split issue | Train/test leakage, evaluation contamination | Split file, random seed, index |
| Annotation issue | Inconsistent labels, guideline drift, rising disputes | Annotation guideline, arbitration record, sample distribution |
| Evaluation issue | Abnormal score, distorted slice, unstable comparison | Metric script, baseline, evaluation-set version |
| Agent issue | Wrong tool call, lost trajectory, permission overreach | Event log, tool call, authorization boundary |
| Compliance issue | Unclear export, purpose drift, log leakage | Jurisdiction, approval, audit trail, release strategy |

The purpose is not to label bugs for its own sake. It helps the team decide where to look first. A sudden evaluation drop is not necessarily model degradation; it may come from evaluation contamination or upstream false deletion. Earlier classification shortens debugging.

## E.4 Collection and Parsing Bugs

### E.4.1 Missing Content After Web Collection

Common causes:

- Selector is too narrow.
- Page structure changed.
- Anti-crawling response returned an alternative page.
- Encoding or rendering method changed.

First actions:

- Preserve the raw HTML.
- Compare the DOM before and after collection.
- Check whether the request was redirected to a CAPTCHA or warning page.
- Freeze a structural regression sample.

For long documents, announcements, papers, or knowledge-base pages, also check pagination and lazy loading. Many "missing text" problems are not collection failures; the page renders content only after scrolling. Save both screenshot and DOM so that evidence is not limited to one view.

### E.4.2 PDF or Document Parsing Misalignment

Common causes:

- Header and footer interference.
- Missing table lines.
- Wrong order between OCR and layout analysis.
- Confusion between page numbers and content pages.

First actions:

- Compare a single page manually.
- Run an A/B comparison with another parser.
- Preserve page screenshots and intermediate artifacts.

If parsing output contains many broken titles, merged tables, or reordered lists, do not rush into post-processing. First check whether the layout layer already corrupted reading order. Post-processing often beautifies a wrong ordering rather than fixing it.

## E.5 Cleaning and Splitting Bugs

### E.5.1 Sample Count Drops Sharply After Deduplication

Ask three questions first:

- Is the deduplication granularity too coarse?
- Were template samples mistaken for dirty samples?
- Were similar samples from different sources treated as duplicates?

Also distinguish literal duplication from semantic duplication. Literal duplicates fit rule-based deduplication. Semantic duplicates depend on task goals. Moderate repetition may be acceptable for training data; repetition in evaluation data often indicates contamination. They cannot use the same ruler.

### E.5.2 Training Performance Suddenly Gets Worse

Check first:

- Whether the new data version contains old bad samples.
- Whether the split leaked.
- Whether the evaluation set was contaminated.
- Whether a cleaning rule changed without a version bump.

Teams often suspect model architecture first, but data is frequently the real source of fluctuation. In incremental updates, even a small rule change can alter long-tail sample distribution. Record rule version and data version together.

## E.6 Annotation and Evaluation Bugs

### E.6.1 Annotation Consistency Declines

The usual cause is not that annotators suddenly became worse. More often:

- The guideline is unclear.
- Boundary samples increased.
- A new category was introduced without synchronizing criteria.

Recommended actions:

- Extract disputed samples for focused review.
- Add positive and negative examples for boundary cases.
- Add decision priorities for new categories.
- Define traceable fields for arbitration records.

### E.6.2 Evaluation Score Fluctuates Abnormally

Check first:

- Metric script version.
- Evaluation-set version.
- Prompt or template version.
- Whether new post-processing is enabled.

If the change appears only in a few slices, do not reject the model globally. First check whether the slice definition changed or sample distribution drifted. Many "overall regressions" are local defects amplified by aggregation.

### E.6.3 Evaluation Result Conflicts with Human Perception

This is common and easy to overlook. A model score may look higher while human experience is worse. Common reasons:

- The metric does not cover the behavior that matters.
- The evaluation set is too narrow.
- Post-processing makes wrong answers look more correct.
- Human review observes scenario-level experience while the metric sees local fields.

The fix is not "tune the model again." Reconfirm what is being optimized. If the goal is retrieval accuracy, metrics should focus on recall and ranking. If the goal is usability, add readability, completeness, and explainable failure.

## E.7 Agent and DataOps Bugs

### E.7.1 Agent Calls the Wrong Tool

Check first:

- Is the tool description specific enough?
- Was task context lost?
- Are permissions too broad?
- Did failure retry change state?

For agent systems, many bugs are not "cannot do it" but "did the wrong thing and continued." Record tool calls as structured events and distinguish plan, execution, retry, and rollback states. Otherwise debugging sees only final output, not the intermediate process.

### E.7.2 Problem Recurs After Rollback

Check first:

- Was only data rolled back, but not rules?
- Was only a table rolled back, but not cache?
- Was only a result rolled back, but not the reference chain?

This often creates the illusion of surface recovery while the root cause remains. Every rollback action should carry version, timestamp, owner, and impact scope, followed by a minimal validation set.

## E.8 Debug Record Template

Record these fields for every bug:

| Field | Content |
| :-- | :-- |
| Problem description | What happened |
| First discovery time | When it was found |
| Impact scope | Which data, tasks, or chapters are affected |
| Reproduction steps | How to reproduce stably |
| Suspicious versions | Related code, data, and configuration versions |
| Root-cause judgment | Current most likely cause |
| Fix action | What was changed |
| Regression result | Whether the fix has been verified |

If a problem returns after being fixed twice, the issue is often not a technical detail but incomplete recording. Debug records are valuable not only as archives, but as path dependency for future similar problems. For production or course-reproduction projects, frameworks such as the ML Test Score remind us that tests, monitoring, and rollback ability are part of system maturity (Breck et al. 2017).

## E.9 Three Common Cases

### E.9.1 Table Fields Shift

A batch of documents loses one column in every parsed table. The first reaction is "the parser is broken." Manual comparison shows that some source cells are merged and the new template moved the header. The real issue is inconsistency between template version and parsing rules. The fix is to include template version as an input condition and define separate rules for critical layouts.

### E.9.2 Split Leakage Inflates Evaluation

Train and test sets share near-duplicate samples, causing scores to rise. After repair, scores drop, but manual review confirms the result is more truthful. High scores do not always mean a better system; when data chains are long, contamination is often more common than real improvement.

### E.9.3 Agent Write-back Overwrites Old Results

An agent writes back automatically without checking versions, and new results overwrite old results. The system appears normal, but history has been rewritten. The fix is a version gate: every write-back must carry target version, source trajectory, and traceable evidence.

## E.10 Regression-Test Checklist

After fixing a bug, check at least:

1. Whether the minimal reproduction sample is normal.
2. Whether neighboring samples were harmed.
3. Whether upstream data can still trigger the same problem.
4. Whether downstream evaluation remains stable.
5. Whether logs, caches, and version records were updated together.
6. Whether rollback remains available.
7. Whether monitoring can detect a similar problem again.

If this checklist has not been executed, the bug is not truly fixed. Many "fixed" states mean only "nothing happened this time."

## E.11 Summary of This Appendix

The key to debugging is not speed, but stability. Preserve evidence first, narrow the scope second; identify the layer first, symptoms second; fix root cause first, then add regression. This turns a bug from a one-time repair into a lesson absorbed by the engineering process.

## E.12 A More Complete Troubleshooting Path

If a bug is viewed as a chain from discovery to recovery, the recommended order is:

1. Identify the anomaly: single point, batch issue, or long-term drift.
2. Freeze the scene: save input, output, logs, versions, and screenshots.
3. Build the minimal sample: reduce the issue to the smallest reproducible set.
4. Assign the layer: source, parsing, cleaning, split, annotation, evaluation, agent, or release.
5. Inspect recent changes: check what changed recently before guessing model degradation.
6. Verify hypotheses: disable suspicious factors one by one, not all at once.
7. Confirm root cause: it must explain the symptom, reproduce, and support regression.
8. Design regression: turn the issue into a future automated check.

This reduces patch-style troubleshooting. Surface fixes often hide deeper failures.

## E.13 Observation Points and Instrumentation

Debugging is hard not only because systems are complex, but because they lack observation points. For the data-engineering chains in this book, keep at least:

- Raw input snapshots.
- Intermediate structured results.
- Key thresholds and rule versions.
- Sampling and filtering logs.
- Tool calls and action logs.
- Version numbers before and after write-back.
- Exception context on failure.

Without these, bugs are guessed. Guessing once or twice is tolerable; guessing ten times trains the team into empiricism without evidence.

### E.13.1 More Logs Are Not Always Better

Too few logs are bad, but too many logs are also bad. Logs must be readable, filterable, and correlatable. Split key logs into three types:

1. Audit logs: who did what and when.
2. Runtime logs: where the program ran under what conditions.
3. Diagnostic logs: state, input, and intermediate values when exceptions occur.

Do not mix these into one large text block. Otherwise the team ends up with many logs and none that localizes the issue.

## E.14 Version, Cache, and Permission

Many bugs that look like code issues are actually version, cache, or permission issues.

### E.14.1 Version Issues

The most common symptom is "it worked yesterday and fails today." Causes include:

- Data snapshot changed.
- Configuration file changed.
- Dependency package changed.
- Template or prompt changed.
- Evaluation script changed.

Do not inspect only code commits. Inspect data versions and external dependencies. For long engineering chains, the version number itself is evidence.

### E.14.2 Cache Issues

Cache issues often look like "I changed it, but the result did not change." Usually:

- Old results remain in cache.
- Index was not rebuilt.
- Intermediate artifacts were not cleaned.
- Calls were routed to an old service.

The most effective fix is often not changing logic, but explicitly clearing cache, rebuilding indexes, forcing refresh, and rerunning the minimal sample.

### E.14.3 Permission Issues

Permission problems often disguise themselves as data absence or system failure:

- Data cannot be read because permission is insufficient.
- Write-back fails because the target table is not writable.
- Tool calls fail because the agent lacks execution permission.

During troubleshooting, confirm whether the permission boundary is working before assuming the service or interface is broken.

## E.15 Postmortem Template

After a bug is resolved, write a lightweight postmortem:

| Field | Content |
| :-- | :-- |
| Origin | How the problem arose |
| Trigger | Which change exposed it |
| Detection method | System, human, or monitoring |
| Temporary handling | How damage was contained first |
| Root cause | The final confirmed root cause |
| Long-term fix | What process or code changes follow |
| Preventive measure | How to detect earlier next time |

A postmortem is not a blame document. It converts experience into process. A problem is truly over only when later workflow can automatically block it.

## E.16 Common Troubleshooting Mistakes

### E.16.1 Looking Only at the Final Score

An abnormal final score may originate at the input side. Focusing only on the score misses splitting, contamination, and sample loss.

### E.16.2 Changing Only One Place

Some problems look like a single point but actually involve several links. Fixing one point and stopping often creates a system that works occasionally and fails often.

### E.16.3 No Fixed Regression Sample

Without fixed regression samples, there is no stable comparison baseline. Each fix runs on a different terrain, so it is hard to know whether the issue is fixed or merely not triggered.

### E.16.4 Treating "Has Not Reappeared" as "Resolved"

This is risky. Many issues are temporarily untriggered. A fix is reliable only after minimal samples, neighboring samples, upstream samples, and regression samples pass verification.

## E.17 Suggested Extensions

For team practice, extend this appendix with:

- Typical incident tickets collected by category.
- Standard troubleshooting scripts that automate common checks.
- A postmortem knowledge base that preserves root causes and fixes.

Then bugs are no longer "whoever hits it fixes it"; they become shared engineering knowledge.

## E.18 Quick Mapping from Symptom to Layer

When a bug appears, do not fix immediately. First judge which layer it resembles.

| Symptom | More likely layer |
| :-- | :-- |
| Data suddenly shrinks sharply | Source, parsing, or cleaning |
| Evaluation score suddenly jitters | Split, evaluation set, or script |
| Output looks correct but humans disagree | Annotation, evaluation definition, or post-processing |
| Agent often calls the wrong tool | Tool description, permission, or context |
| History is overwritten after write-back | Version, write-back policy, or lock mechanism |

This table does not replace investigation. It reduces first-round guessing cost.

## E.19 A More Complete Incident Postmortem Framework

Answer at least five questions:

1. What was the direct trigger?
2. Why was it not detected earlier?
3. Which control point failed?
4. Which link could have blocked it earlier?
5. What mechanism should be added to prevent recurrence?

Writing only "the cause was a wrong rule" has limited value. A useful postmortem reveals whether the gap was in definition, process, monitoring, or permission.

## E.20 Turning Troubleshooting into Standard Action

Mature teams do not treat every incident as improvisation. They turn troubleshooting into standard actions.

### E.20.1 Standard Troubleshooting Actions

- Fix the input.
- Reproduce the output.
- Check versions.
- Check logs.
- Locate the layer.
- Verify hypotheses.
- Create regression.

### E.20.2 Benefits of Standardization

Standardization is not formality. It lowers cognitive load. If everyone checks in the same order, team members can share judgment and hand over problems more easily. Site reliability engineering practice similarly emphasizes that incident response, monitoring, postmortems, and automated operations must become standard actions to reduce the long-term cost of repeated incidents (Google SRE 2016).

## E.21 Three Recurrence Patterns

### E.21.1 Rule Recurrence

One rule is fixed, but old logic remains elsewhere. The issue looks fixed while it has merely moved.

### E.21.2 Data Recurrence

Old or dirty samples flow into a new version again, causing periodic recurrence. This is common in incremental updates and multi-source aggregation.

### E.21.3 Process Recurrence

A control point never enters the formal process. Every time the owner, environment, or version changes, the issue returns. Process recurrence is harder to fix than technical recurrence because it requires institutions, not only code.

## E.22 Closing Note

The real value of a debugging manual is not telling you how to fix one bug. It helps the team form a stable way of judging problems. Debugging then no longer depends on one unusually skilled person; it becomes a reproducible engineering capability.

## E.23 Checklist of 20 Common Problem Types

The following 20 classes are not mutually exclusive, but they help locate issues quickly.

| Class | Check first |
| :-- | :-- |
| Collection missing | Entry page, redirect, anti-crawling, encoding |
| Parsing misalignment | Layout, OCR, table lines, page numbers |
| False deletion during cleaning | Rule version, threshold, sampled records |
| Over-deduplication | Granularity, fingerprint, semantic similarity |
| Split leakage | Split, random seed, index |
| Annotation drift | Guideline, disputed samples, arbitration record |
| Evaluation distortion | Script, baseline, evaluation-set version |
| Metric jitter | Version change, post-processing, cache |
| Agent overreach | Tool permission, context, approval |
| Write-back overwrite | Version number, lock mechanism, target table |
| Stale index | Rebuild process, refresh strategy, cache |
| Alignment error | Label criteria, field mapping, sorting |
| Feedback contamination | Feedback rules, training boundary, audit |
| Compliance omission | Jurisdiction, purpose, audit trail, approval |
| Resource exhaustion | Compute, disk, concurrency, queue |
| Version drift | Code, data, configuration, template |
| Interface change | Parameters, fields, dependencies, return value |
| Missing logs | Key instrumentation, exception context, trace ID |
| Permission failure | Read/write, tool, directory, secret |
| Rollback failure | Snapshot, dependency, lock, recovery path |

The point is not to memorize every class. It helps the team lock the first search area quickly.

## E.24 One Line from Collection to Compliance

In projects related to this book, bugs often propagate along a chain: collection breaks, parsing becomes unstable; parsing instability causes cleaning errors; cleaning errors bias splitting and evaluation; finally the issue appears during agent write-back or compliance approval.

Troubleshoot backward along this line:

1. Check whether the output is wrong.
2. Check whether the middle layer propagated the issue.
3. Return to the entry point and source.
4. Confirm whether an institutional control is missing.

The goal is to avoid circling around the surface symptom. The hard part is often not one function, but a missing constraint in the chain.

## E.25 Daily Maintenance Recommendations

To reduce recurrence, make three actions routine:

- Save a regression sample after each release.
- Record a version diff after each rule change.
- Add a postmortem note after each anomaly.

If these three actions continue, this manual becomes part of daily engineering rhythm rather than a document opened only after incidents.

## References

Sculley D, Holt G, Golovin D, Davydov E, Phillips T, Ebner D, Chaudhary V, Young M, Dennison D (2015) Hidden Technical Debt in Machine Learning Systems. In: Advances in Neural Information Processing Systems 28.

Breck E, Cai S, Nielsen E, Salib M, Sculley D (2017) The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. In: Proceedings of the IEEE International Conference on Big Data, pp 1123-1132.

Amershi S, Begel A, Bird C, Devanbu P, Gall H, Kamar E, Nagappan N, Nushi B, Zimmermann T (2019) Software Engineering for Machine Learning: A Case Study. In: Proceedings of the 41st International Conference on Software Engineering: Software Engineering in Practice, pp 291-300.

Google SRE (2016) Site Reliability Engineering: How Google Runs Production Systems. O'Reilly Media.
