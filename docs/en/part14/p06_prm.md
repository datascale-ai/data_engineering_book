# Project 6: CoT Reasoning Dataset Construction and PRM Training

## Abstract
P06 focuses on building a data factory for chain-of-thought reasoning and process reward model training.

The project does not only judge whether the final answer is correct.

It splits reasoning into steps, verifies each step where possible, assigns process labels, organizes reward buckets, and packages the result for PRM-style training.

The chapter can be read through four lines.

- Seed tasks and task specifications: define what the reasoning problem is before generating traces.
- Positive, negative, and repair trajectories: build process contrast instead of only keeping successful solutions.
- Step validation and labeling: turn reasoning into inspectable process units.
- Training packaging, noise control, and replay: make process supervision usable across iterations.

The engineering chain is:

```text
seed tasks -> task specs -> positive/negative/repair traces -> step segmentation -> automatic validation -> step labels -> reward buckets -> PRM training package -> evaluation and replay
```

The core goal is to make reasoning process signals trustworthy enough for training and review.

## Keywords

Chain-of-thought; process supervision; PRM; step labels; reward bucket; replay set

## Project Goals and Reader Takeaways

This project uses CoT reasoning dataset construction and PRM training as the core case.

After completing the chapter, readers should be able to define task specs, construct different trace types, split reasoning into steps, design step-level labels, interpret verifier results, and package PRM data for experiments.

## Scenario Constraints and Data Boundaries

The project uses a small and controlled task set, mainly around mathematics and code.

It does not train a frontier reasoning model.

It does not claim that automatic verification can replace human review.

It is a minimal process-supervision factory that makes the data objects and control points explicit.

## Architecture Decision

The project adopts an architecture based on task specs, trace construction, step segmentation, automatic validation, process labels, reward buckets, training delivery, and replay.

This keeps the relationship between task, trace, step, verifier, label, and training record visible.

## Sample Schema and Data Flow

The minimal step record should include `task_id`, `trace_id`, `trace_type`, `step_index`, `step_text`, `local_claim`, `verification_signal`, `label`, `reward_bucket`, `split`, and `audit_trace`.

```text
task -> trace -> step -> verifier output -> process label -> reward bucket -> PRM record
```

This schema keeps process information separate from final-answer correctness.

## Core Implementation Fragments

The chapter keeps fragments that explain task specs, trace types, step schemas, verifier outputs, and PRM training records.

Long generated traces and full verifier logs should be maintained in companion resources.

## Experimental and Acceptance Metrics

Acceptance metrics include seed-task coverage, trace distribution, step count, verifier pass rate, positive/negative/repair balance, reward-bucket distribution, smoke-test pass rate, and replay-set coverage.

For reproduction, record generation model, verifier version, prompt version, random seed, and manual review samples.

## Cost, Risk, and Compliance Boundaries

Costs mainly come from trace generation, verification, labeling, review, and replay maintenance.

Risks include incorrect step labels, ambiguous repair traces, over-trusting weak verifiers, synthetic-style bias, and using noisy process labels as if they were ground truth.

## Common Failure Modes

Common failures include positive traces that are too easy, negative traces that are unrealistic, repair traces with unclear error boundaries, verifier false positives, step segmentation that loses context, and reward buckets that imply false precision.

## Reproducible Resource Notes

Reproduction materials should include seed tasks, task specs, trace files, step files, verifier outputs, label mappings, reward-bucket definitions, training splits, metrics, and replay sets.

## 1. Project Background: Why a CoT and PRM Data Factory Is Needed

Outcome supervision tells a model whether the final answer is correct.

Process supervision asks a more detailed question: which reasoning steps are trustworthy, which are wrong, and which steps repair an earlier mistake?

This distinction matters for mathematics, code, tool use, and multi-step agents.

A final answer can be right for the wrong reason.

A final answer can be wrong even though many early steps were useful.

P06 therefore treats reasoning traces as data assets that must be segmented, verified, labeled, packaged, and replayed.

The project is small by design.

Its value is making the process-supervision structure visible.

## 2. Project Goals and Boundaries

### 2.1 Project Goals

The first goal is to define seed tasks and task specifications before trace generation.

The second goal is to generate positive, negative, and repair traces in parallel.

The third goal is to split traces into step-level records and attach verification signals.

The fourth goal is to package the resulting records into PRM-ready training artifacts with metrics and checks.

### 2.2 Project Boundaries

The project does not prove full reasoning-model improvement.

It does not solve all process-label ambiguity.

It does not replace human review for hard tasks.

It does not scale to every domain in the prototype stage.

### 2.3 Role of Boundary Setting

Process supervision is easy to overclaim.

Clear boundaries prevent the reader from treating verifier output as unquestionable truth.

The case demonstrates an engineering method, not a final PRM product.

## 3. Project Position: P06 in the Capability Chain

P06 sits between reasoning-data construction and reward-model training.

Earlier chapters discuss CoT, tool use, and data quality.

This project turns those ideas into a concrete process-supervision pipeline.

Its key contribution is to move from answer-level samples to step-level evidence.

![Figure 1: CoT and PRM Data Factory Overview](../../images/part10/10_6_fig01_prm_factory_overview.png)

## 4. Overall Architecture: From Seed Tasks to PRM Training Assets

### 4.1 Layer 1: Task and Trace Generation

This layer samples tasks, builds task specs, and generates positive, negative, and repair traces.

It defines the behavior distribution before validation begins.

### 4.2 Layer 2: Step Verification and Reward Assignment

This layer segments traces into steps, runs automatic checks, attaches validation signals, and assigns labels and reward buckets.

It is the core process-supervision layer.

### 4.3 Layer 3: Training Packaging and Delivery

This layer exports PRM train/validation/smoke splits, manifests, reports, and replay sets.

It gives training code a stable interface.

![Figure 2: Step-level Validation and Training Loop](../../images/part10/10_6_fig02_step_validation_loop.png)

## 5. Seed Tasks: The Task Layer as Supervision Starting Point

### 5.1 Why Seed Tasks Are Necessary

Seed tasks define the problem before the model starts reasoning.

Without them, traces are hard to compare, verify, and stratify.

### 5.2 Why the Project Starts with Math and Code

Math and code tasks often have verifiable outputs.

They provide a practical starting point for process supervision because numeric results and executable snippets can be checked.

### 5.3 Why Seed Tasks Are Not Final Training Samples

A seed task defines intent.

The final PRM sample is built only after traces, steps, verification, labels, and packaging are added.

## 6. Task Sampling and Specification Design

```python
task_spec = {
    "task_id": "math_000123",
    "domain": "math",
    "difficulty": "medium",
    "prompt": "A problem statement...",
    "reference_answer": "42",
    "verifier_type": "numeric",
    "target_trace_types": ["positive", "negative", "repair"],
}
```

### 6.1 What Specification Design Solves

The spec records domain, difficulty, expected verifier, reference answer, and trace targets.

It makes later traces comparable.

### 6.2 Why Sampling Cannot Mean Only Diversity

Sampling should control difficulty, domain, verifier type, and target trace type.

Raw diversity without evaluation structure is not enough.

### 6.3 Engineering Value of `task_spec`

The task spec becomes the anchor for trace generation, step labeling, validation, splitting, and reporting.

![Figure 3: Task Sampling and Spec Generation](../../images/part10/10_6_fig03_task_sampling.png)

## 7. Trace Generation: Building Positive, Negative, and Repair Together

Positive traces show correct reasoning.

Negative traces intentionally include plausible wrong paths.

Repair traces show the model identifying and correcting an error.

```python
trace = {
    "trace_id": "math_000123_repair_00",
    "task_id": "math_000123",
    "trace_type": "repair",
    "steps": [
        {"index": 0, "text": "Set up the equation..."},
        {"index": 1, "text": "Notice the earlier coefficient was wrong..."},
        {"index": 2, "text": "Recompute and get 42."},
    ],
}
```

### 7.1 Why Positive-only Traces Are Not Enough

A PRM needs contrast.

It needs to see good steps, bad steps, and recovery steps.

### 7.2 What Negative Traces Solve

Negative traces teach the reward model which plausible reasoning moves should receive low scores.

They are especially useful when errors are subtle.

### 7.3 Why Repair Traces Are More Important and More Dangerous

Repair traces teach recovery.

They are dangerous because the boundary between "mistake" and "repair" can be ambiguous.

### 7.4 What the Current Trace Structure Shows

The project treats trace type as a first-class field.

This makes process distribution visible in metrics and training packaging.

![Figure 4: Relationship Among Three Trace Types](../../images/part10/10_6_fig04_trace_types.png)

## 8. Step Segmentation and Schema: The Minimum Unit of Process Supervision

### 8.1 Why Step Is the Required Unit

Process supervision cannot operate on the whole answer only.

It needs local reasoning units.

The step is the minimum unit that can be reviewed, verified, and rewarded.

### 8.2 What the Schema Solves

![Figure 5: PRM Step Schema](../../images/part10/10_6_fig05_step_schema.png)

The schema connects each step to its task, trace, position, local claim, verifier result, label, and reward bucket.

This prevents labels from becoming detached from source context.

### 8.3 Why Step Schema Must Be Preserved Separately

Training records may compress context.

The step schema preserves auditability.

It lets reviewers trace a PRM sample back to the original reasoning trace.

## 9. Automatic Validation: Result Checking for Process Supervision

```python
verdict = {
    "step_id": "math_000123_repair_00_s02",
    "verifier_pass": True,
    "reason": "numeric_consistency",
    "parsed_value": 42,
    "expected_value": 42,
}
```

### 9.1 Why Validation Must Move into Data Production

If validation waits until training, noisy process labels have already entered the dataset.

Validation should be part of data creation.

### 9.2 Which Signals Automatic Validation Can Use

Signals include numeric equality, symbolic checks, code execution, final-answer comparison, format checks, and rule-based constraints.

### 9.3 Why Validation Is Not "the Stricter the Better"

Overly strict validation can reject useful repair steps or alternate solution paths.

The goal is reliable signal, not maximum rejection.

### 9.4 What Current Validation Results Mean

Validation results should be interpreted by trace type.

Negative and repair traces are expected to include failures.

![Figure 6: Step Validation and Result Comparison](../../images/part10/10_6_fig06_validation_pipeline.png)

## 10. Step Label Design: Making Process Fields Operable

### 10.1 Role of Step Labels

Step labels turn verification and review into training signals.

They tell the model which candidate step deserves reward.

### 10.2 Why Labels Cannot Be Only Binary

Binary correct/incorrect labels are too coarse.

Useful labels may include correct, incorrect, repaired, unsupported, format_error, and uncertain.

### 10.3 Why Process-only Supervision Signals Matter

Some steps matter even when the final answer is not yet visible.

Process-only signals let the reward model learn local reasoning quality.

![Figure 7: Step Labels and Process-only Signal](../../images/part10/10_6_fig07_step_labels.png)

## 11. Reward Bucket: Layered Scoring Mechanism

### 11.1 Engineering Meaning of Reward Buckets

Reward buckets group steps into coarse quality levels.

They are easier to review than fragile continuous scores.

### 11.2 Why Buckets Are More Stable Than Continuous Scores

Continuous scores can imply precision that the data does not support.

Buckets such as high, medium, low, and reject better match human and verifier uncertainty.

### 11.3 What the Current Bucket Structure Shows

The bucket design shows that the project values usable process ranking over false numerical precision.

## 12. PRM Data Packaging: Training Interface Layer

```python
prm_record = {
    "prompt": "Problem statement...",
    "partial_trace": "Step 1...\nStep 2...",
    "candidate_step": "Therefore the answer is 42.",
    "label": "correct",
    "reward_bucket": "high",
    "metadata": {
        "domain": "math",
        "trace_type": "repair",
        "verifier_reason": "numeric_consistency",
    },
}
```

### 12.1 Why the Packaging Layer Is Important

Training code should receive stable records.

It should not need to understand the internal generation pipeline.

### 12.2 Current Key Training Artifacts

Artifacts include PRM train, validation, and smoke splits; manifest files; verifier summaries; and replay records.

### 12.3 Why Smoke Test and Manifest Must Exist

The smoke set validates the training interface.

The manifest records counts, versions, label distributions, and file paths.

![Figure 8: PRM Data Packaging and Training Interface](../../images/part10/10_6_fig08_training_interface.png)

## 13. Data Scale and Structure: Signals That the Factory Has Formed

The important signal is not only total record count.

It is whether tasks, traces, steps, labels, buckets, verifier outputs, and splits are all present.

### 13.1 What These Numbers Mean

Counts by task domain, trace type, label, reward bucket, and split reveal whether the factory is balanced enough for early experiments.

### 13.2 Why Structure Matters More Than Total Volume

Large noisy process data can damage a PRM.

A smaller structured dataset is more useful for method validation.

## 14. Metric Interpretation: Meaning of Current Validation Results

### 14.1 Current Key Metrics

Key metrics include task count, trace count, step count, verifier pass rate, positive pass rate, negative failure rate, repair success rate, bucket distribution, and check pass rate.

### 14.2 Why a Pass Rate Below 100% Can Be Good

If the dataset contains negative and repair traces, some failures should exist.

A perfect pass rate may mean the data lacks contrast.

### 14.3 Why Positive 100% Pass Has Two Meanings

It may mean positives are clean.

It may also mean the task set is too easy or the verifier is too permissive.

![Figure 9: Validation Pass Rate by Trace Type](../../images/part10/10_6_fig09_validation_metrics.png)

## 15. Evaluation and Project Checks

Evaluation should check data consistency before downstream model performance.

```python
checks = [
    "all_steps_have_trace_id",
    "all_reward_buckets_valid",
    "no_empty_step_text",
    "verifier_outputs_present",
    "train_val_smoke_non_overlapping",
]
```

### 15.1 Why Evaluation Cannot Only Look at the Trained Model

If the data assets are inconsistent, model results are hard to interpret.

The project should validate process data first.

### 15.2 What Current Checks Cover

Checks should cover required files, step fields, label values, bucket values, verifier outputs, split overlap, and report consistency.

### 15.3 Why Check Scripts Belong in the Chapter

Check scripts make the project reproducible.

They show how acceptance is operationalized.

## 16. Noise Control: Governance of Negative and Repair Traces

### 16.1 Why Negative Examples Are Naturally Noisy

Negative traces must be plausible enough to teach contrast.

If they are too obviously wrong, they are less useful.

If they are too subtle, labels become uncertain.

### 16.2 Why Repair Traces Are More Ambiguous

Repair traces include both wrong and corrected reasoning.

The data factory must mark the transition point clearly.

### 16.3 Clear Signal Given by the Current Project

The project treats noise analysis as part of the main process.

It does not hide uncertain cases.

![Figure 10: Noise Sources in Negative and Repair Traces](../../images/part10/10_6_fig10_noise_sources.png)

## 17. Cost and Benefit: Priority of Structural Closure

### 17.1 Real Benefit of a Small Project

The early benefit is learning how to build process supervision assets.

It is not model benchmark improvement.

### 17.2 Why the Chapter Emphasizes Structural Benefit

Once the structure is correct, scale becomes safer.

If the structure is wrong, more data only adds more noise.

### 17.3 A More Realistic Engineering Judgment

The right next step is to improve labels, verifiers, and replay sets before expanding volume aggressively.

## 18. Main Deliverables

### 18.1 Intermediate Data Artifacts

- Seed task pool.
- Task specifications.
- Positive traces.
- Negative traces.
- Repair traces.
- Step-level records.
- Verifier outputs.

### 18.2 Training Data Artifacts

- PRM train split.
- PRM validation split.
- PRM smoke split.
- Training manifest.

### 18.3 Report and Check Artifacts

- Metrics report.
- Noise analysis.
- Check report.
- Replay set.

## 19. Limitations and Risks

### 19.1 Main Limitations

The task scope is small, verifier coverage is limited, and labels still require review.

Repair trajectories remain hard to judge.

### 19.2 Why Scaling First May Be Wrong

Scaling before label and verifier quality is stable can create a larger noisy dataset.

Process supervision is especially sensitive to label noise.

### 19.3 Why These Risk Judgments Must Remain

The chapter should not overstate PRM readiness.

Its credibility comes from naming the hard parts.

## 20. Future Extensions

### 20.1 Expand Task Range

Add symbolic math, competitive programming, tool-use reasoning, and multi-step planning tasks.

### 20.2 Refine Reward Definitions

Introduce finer but still reviewable reward categories for local correctness, recovery, efficiency, and unsupported claims.

### 20.3 Move Toward Complex Agent Tasks

Process supervision can extend to agents when each action, observation, and recovery decision becomes a step-like object.

## 21. Summary: Value of Trustworthy Process Signals

The value of P06 is not long reasoning text.

Its value is trustworthy process signal.

A PRM data factory records how steps are created, checked, labeled, packaged, and replayed.

That structure is the foundation for later reward-model experiments.

## Special Topic: Annotation Consistency and QA for PRM Data

### 1. Why Step-level Annotation Is More Ambiguous

Step quality often depends on surrounding context.

A step can be locally valid but globally unhelpful.

This makes annotation harder than final-answer checking.

### 2. Process Supervision Needs Layered QA

QA should inspect task specs, traces, step segmentation, verifier outputs, labels, and reward buckets.

One-level review is not enough.

### 3. QA Should Keep Useful Errors

Not every error should be deleted.

Some wrong or repaired steps are valuable training signals if they are labeled clearly.

## Special Topic: Strategy Choices When PRM Enters Training

### 1. PRM Data Does Not Need Immediate Standalone Training

Early PRM data can support reranking, critique models, verifier prompts, or mixed training before a separate PRM is trained.

### 2. Different Stages Need Different Uses

Prototype stages should emphasize diagnostics.

Scaling stages can emphasize reward-model training and policy optimization.

### 3. PRM Projects Need Their Own Version Language

Versions should describe task scope, trace mix, verifier changes, label policy, and replay-set changes.

## Special Topic: Value of Replay Sets in Process Supervision

### 1. What Problems Replay Sets Should Collect

Replay sets should collect ambiguous repairs, verifier disagreements, format failures, correct final answers with wrong intermediate steps, and hard negative traces.

### 2. Replay Sets Build Team Memory

A replay set prevents the team from rediscovering the same failures every release.

It becomes the memory of the process-supervision project.

## Chapter Summary

This chapter used a CoT and PRM data factory to show how process supervision can be organized as data engineering.

The case connects seed tasks, task specs, trace generation, step segmentation, automatic validation, process labels, reward buckets, training packaging, noise control, and replay.

Its boundary should remain clear: it is a structured prototype, not a complete reasoning-model training system.

## Release Review Notes

The first release review item is seed-task clarity.

Each task should have a clear prompt, domain, difficulty, expected answer, and verifier type.

Ambiguous tasks should be repaired before trace generation.

The second item is trace-type balance.

Positive, negative, and repair traces should each be visible.

A dataset with only positive traces cannot teach a PRM enough contrast.

The third item is step segmentation.

Steps should be neither too large nor too small.

A step that contains several reasoning moves is hard to label.

A step that is only a fragment may lose meaning.

The fourth item is verifier coverage.

Every verifier output should state what it checked and what it did not check.

Numeric, symbolic, code, and format verifiers have different blind spots.

The fifth item is label policy.

The release should define each label clearly.

Labels such as correct, incorrect, repair, unsupported, and uncertain should not overlap.

The sixth item is reward-bucket policy.

Reward buckets should be coarse enough to be reliable.

They should not pretend to provide exact continuous reward.

The seventh item is negative trace realism.

Negative traces should be plausible.

Trivial wrong steps teach little.

The eighth item is repair trace clarity.

Repair traces should show where the error was noticed and what changed.

Without this boundary, repair labels become noisy.

The ninth item is split integrity.

Train, validation, and smoke sets should be disjoint by task ID when possible.

The tenth item is replay coverage.

The replay set should include hard negatives, ambiguous repairs, verifier disagreements, and formatting failures.

## Operating Notes

Daily operation should inspect verifier pass rates by trace type.

If positive pass rate drops, inspect generation quality and verifier configuration.

If negative pass rate is too high, inspect whether negative traces are actually wrong.

If repair pass rate is confusing, inspect the boundary between wrong step and corrected step.

If many labels become uncertain, revise task specs or verifier coverage.

If reward-bucket distribution collapses, the reward policy may be too coarse or too strict.

If smoke tests fail, inspect schema and split files before changing generation.

The project should keep verifier disagreement examples.

Disagreements are not merely failures.

They reveal where process supervision is hard.

Those examples should be reviewed before scaling.

The project should also keep a label-change ledger.

When label definitions change, previous data may need migration or version separation.

PRM data is especially sensitive to label-policy drift.

## QA Notes

QA should inspect task specs first.

Bad task specs create bad traces.

QA should then inspect trace construction.

Positive traces should be clean.

Negative traces should be instructive.

Repair traces should be interpretable.

QA should inspect step boundaries.

The step is the unit of reward, so boundary errors become reward errors.

QA should inspect verifier outputs.

A verifier pass should not be treated as a universal correctness guarantee.

QA should inspect reward buckets.

Buckets should reflect process usefulness, not only final-answer correctness.

Finally, QA should inspect replay cases.

Replay cases protect the project from repeating old mistakes.

## Scaling Notes

Scaling PRM data should start with verifier and label stability.

Expanding volume before labels are stable can increase noise.

New domains should be added one at a time.

Each new domain needs task specs, verifier choices, label examples, and replay cases.

Agent tasks can be added later by treating actions, observations, and recovery decisions as process steps.

That extension should not happen until the math/code prototype has stable process semantics.

The long-term goal is not only a larger dataset.

The long-term goal is a versioned process-supervision system.

That system should explain what changed in task mix, trace mix, verifier policy, label policy, reward buckets, and replay cases.

## Final Acceptance Checklist

Seed tasks are unambiguous.

Task specs include verifier type.

Positive traces are present.

Negative traces are present.

Repair traces are present.

Step segmentation is reviewed.

Verifier outputs are stored.

Verifier limits are documented.

Labels are defined.

Reward buckets are defined.

Uncertain cases are retained.

Replay cases are retained.

Train and validation are disjoint.

Smoke data loads correctly.

Manifest matches artifacts.

Noise analysis is present.

Label-policy changes are versioned.

Verifier changes are versioned.

Known ambiguity is stated.

Scaling plan prioritizes label stability.

Reviewer calibration examples are retained.

Future verifier gaps are listed.

Replay promotion policy is documented.

## References

1. Lightman, H., Kosaraju, V., Burda, Y., et al. (2023). Let's Verify Step by Step.
2. Cobbe, K., Kosaraju, V., Bavarian, M., et al. (2021). Training Verifiers to Solve Math Word Problems.
3. Wei, J., Wang, X., Schuurmans, D., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.
4. Yao, S., Yu, D., Zhao, J., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models.
5. OpenAI. (2023). Process supervision research notes.
