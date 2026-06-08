# Project 4: Synthetic Math and Code Textbook Factory

## Abstract
P04 focuses on building a synthetic textbook factory for mathematics and code.

The project is not merely a batch generation script.

It organizes seed problems, curriculum planning, Evol-Instruct expansion, program-of-thought solutions, sandbox validation, textbook packaging, and training delivery into one data pipeline.

The project can be read through four main lines.

- Seed and curriculum planning: define which topics, chapters, and difficulty levels should exist.
- Evolution and PoT generation: turn seed questions into richer problems and executable solution paths.
- Verification and packaging: validate generated code, organize chapters, and produce teaching assets.
- Training interface and operations: export stable training data, manifests, review notes, and release checks.

In engineering order, the project follows this chain.

```text
seed problems -> chapter plan -> Evol-Instruct expansion -> PoT solution generation -> sandbox validation -> textbook packaging -> train/validation/smoke split -> manifest and checks
```

The core goal is to turn synthetic reasoning data from isolated samples into a reviewable textbook-style asset.

## Keywords

Synthetic textbook; Evol-Instruct; Program of Thought; sandbox validation; math data; code data

## Project Goals and Reader Takeaways

This project uses a synthetic math and code textbook factory as the core case.

After completing the chapter, readers should understand how to preserve seed constraints, expand tasks safely, validate executable reasoning, package curriculum artifacts, and connect textbook data to training systems.

## Scenario Constraints and Data Boundaries

The project uses controlled seed problems and generated examples.

It does not claim to replace professional textbook authors, cover all STEM subjects, or prove large-scale educational effectiveness.

The boundary makes the case reproducible.

When the subject scope, learner level, generation model, or review standard changes, the chapter plan, verification policy, and release gates must be reassessed.

## Architecture Decision

The project adopts a layered path: seed selection, curriculum planning, problem evolution, PoT generation, sandbox verification, textbook packaging, and training delivery.

This path keeps teaching structure and executable correctness visible.

It avoids treating synthetic textbook data as an unstructured pile of generated answers.

## Sample Schema and Data Flow

The minimal record should keep `id`, `subject`, `topic`, `chapter`, `difficulty`, `seed_question`, `evolved_question`, `solution_code`, `execution_result`, `quality_signals`, `split`, and `audit_trace`.

```text
seed question -> evolved problem -> PoT solution -> sandbox result -> textbook unit -> training record
```

The schema must preserve both learning intent and verification evidence.

## Core Implementation Fragments

The chapter keeps fragments that explain task evolution, PoT generation, sandbox validation, and packaging contracts.

Long prompts, full generated textbooks, logs, and notebooks should be maintained as companion resources.

## Experimental and Acceptance Metrics

Acceptance metrics include seed coverage, chapter-plan completeness, executable solution pass rate, problem difficulty distribution, topic distribution, packaging completeness, smoke-test pass rate, and review issue closure.

For public reproduction, record model version, prompt version, sandbox configuration, random seed, and manual review samples.

## Cost, Risk, and Compliance Boundaries

Costs mainly come from generation calls, code execution, manual review, and textbook editing.

Risks include hallucinated solutions, brittle generated code, shallow curriculum coverage, unsafe code, excessive synthetic style uniformity, and unreviewed copyright contamination from seed material.

## Common Failure Modes

Common failures include easy problems evolving into unsolvable ones, PoT code passing for the wrong reason, timeout-heavy validation, chapters without teaching progression, smoke tests that do not reflect real training needs, and release notes that do not match artifacts.

## Reproducible Resource Notes

Reproducible resources should include seed files, prompt templates, chapter plans, generation configs, sandbox scripts, packaging outputs, manifests, and release checks.

## 1. Project Background: Why a Synthetic Math and Code Textbook Factory Is Needed

![Figure 1: Project Positioning for the Synthetic Math and Code Textbook Factory](../../images/part10/10_4_fig01_project_positioning.png)

Reasoning data is often collected as independent Q&A pairs.

That is not enough for a textbook-style dataset.

A textbook needs progression, topic balance, difficulty control, examples, solutions, review notes, and release discipline.

Math and code are especially suitable for this project because many answers can be verified through execution or exact checking.

The project therefore treats generated content as a curriculum asset rather than as raw model output.

Its central question is how to turn synthetic reasoning samples into something that can be taught, reviewed, packaged, and trained on.

## 2. Project Goals and Boundaries

![Figure 2: P04 Project Goals and Scope](../../images/part10/10_4_fig02_goals_and_scope.png)

### 2.1 Project Goals

The first goal is to build a controlled seed layer.

The second goal is to evolve seed problems into richer math and code tasks.

The third goal is to generate executable program-of-thought solutions.

The fourth goal is to validate generated solutions through a sandbox and package them into textbook and training artifacts.

### 2.2 Project Boundaries

The project does not cover every STEM domain.

It does not guarantee that all generated content is pedagogically perfect.

It does not replace human editorial review.

It does not claim that a 100% sandbox pass rate proves deep reasoning quality.

### 2.3 Role of Boundary Setting

Clear boundaries make the case credible.

The prototype shows a method for organizing synthetic textbook production.

It does not claim to be a finished publishing platform.

## 3. Project Position: P04 in the Capability Chain

P04 sits between synthetic data generation and training asset construction.

Earlier chapters discuss synthetic data, reasoning data, and quality control.

This project turns those ideas into a concrete data factory.

Its value is not simply that a model can write more problems.

Its value is that generated problems become curriculum-aware, executable, reviewable, and deliverable.

## 4. Overall Architecture: From Seed Problems to Textbook Volumes

![Figure 3: P04 Overall Pipeline](../../images/part10/10_4_fig03_pipeline_overview.png)

### 4.1 Layer 1: Seed and Chapter Planning

This layer defines subjects, topics, chapters, seed problems, and target difficulty.

It prevents generation from drifting into random problem lists.

### 4.2 Layer 2: Evolution Generation and PoT Construction

This layer expands problems and produces programmatic solution paths.

It is the core generation layer.

It must keep solvability and learning intent visible.

### 4.3 Layer 3: Verification, Packaging, and Delivery

This layer runs sandbox checks, packages textbook units, writes curriculum maps and teacher guides, and exports training data.

It turns generated samples into releaseable assets.

## 5. Engineering Prerequisites: Key Surfaces of a Textbook Factory

![Figure 4: Role Collaboration in the Textbook Factory](../../images/part10/10_4_fig04_roles_and_responsibilities.png)

### 5.1 Curriculum Planning and Chapter Design

Curriculum planning defines the learning sequence.

It decides what should be learned before what, and which examples support which concept.

### 5.2 Data Processing and Interface Maintenance

This surface handles schemas, IDs, file paths, splits, manifests, and generated artifact consistency.

### 5.3 Generation Orchestration and Task Expansion

This surface controls prompts, model calls, retries, evolution types, and difficulty movement.

### 5.4 Verification, Rollback, and Quality Control

This surface validates code, reviews problem quality, tracks failures, and decides which samples should be repaired or rejected.

### 5.5 Role of These Responsibility Surfaces

Without these surfaces, a textbook factory becomes a prompt loop.

With them, it becomes a production system with ownership and review gates.

## 6. Seed Layer: Why Problem Seeds Matter

![Figure 5: Mapping Seed Problems to Chapter Plans](../../images/part10/10_4_fig05_seed_to_plan.png)

### 6.1 Why Keep the Question and Discard Old Answers

The seed question captures topic and intent.

Old answers may be incomplete, stylistically inconsistent, or incompatible with the desired solution format.

Keeping the question while regenerating the solution gives the pipeline control.

### 6.2 What the Seed Layer Must Control

The seed layer controls subject, topic, prerequisite, difficulty, expected solution type, and acceptable verification method.

It is the root of curriculum structure.

## 7. Evol-Instruct: Problem Evolution Mechanism

![Figure 6: Evol-Instruct Evolution Path](../../images/part10/10_4_fig06_evol_path.png)

### 7.1 From Simple Questions to Complex Application Problems

Evolution should add context, constraints, and reasoning depth.

It should not simply make wording longer.

### 7.2 Why Scenarioization Matters

Scenarioized problems connect abstract concepts to application settings.

They help the generated textbook feel like a learning resource rather than a list of puzzle fragments.

### 7.3 Why Increasing Difficulty Must Preserve Solvability

Difficulty growth is valuable only if the problem remains well-posed.

The factory must reject evolved tasks that become ambiguous or unsolvable.

## 8. PoT Choice: Programmatic Reasoning Path

![Figure 7: CoT and PoT Comparison](../../images/part10/10_4_fig07_cot_vs_pot.png)

### 8.1 Value and Limits of CoT

Chain-of-thought explains reasoning in natural language.

It is reviewable, but it can look plausible while containing hidden arithmetic or logic errors.

### 8.2 Engineering Advantages of PoT

Program of Thought makes part of the reasoning executable.

This allows the pipeline to run code, capture output, and reject failures.

### 8.3 Why PoT Fits Math and Code Textbooks

Math and code tasks often have verifiable intermediate structure.

Executable solutions give the factory a stronger acceptance signal than prose alone.

## 9. Generation Chain: From Prompt to Code Solution

![Figure 8: Detailed Generation Chain](../../images/part10/10_4_fig08_generation_chain.png)

### 9.1 Step 1: Problem Evolution

The generator receives a seed and produces a richer problem aligned with subject, topic, and difficulty.

### 9.2 Step 2: PoT Generation

The generator then writes a solution path with executable code or a code-like computation segment.

### 9.3 Engineering Details in API Orchestration

The orchestration should record prompt version, model version, retry count, output status, and error messages.

These fields are necessary for later debugging.

### 9.4 Why Failure Retry Matters

Generation failures are expected.

Retries should be bounded and logged.

Unbounded retries can hide quality problems and inflate cost.

## 10. Sandbox Validation: The Core Gate of Generate-then-Verify

![Figure 9: Sandbox Validation Execution Path](../../images/part10/10_4_fig09_sandbox_validation.png)

### 10.1 What Happens Without Validation

Without validation, synthetic textbook data easily becomes fluent hallucination.

The answer may sound like a lesson while being numerically wrong or logically broken.

### 10.2 Why Code Extraction Comes First

The validator needs a clean executable segment.

If code extraction is unstable, execution results cannot be trusted.

### 10.3 Why Execution Timeout Is Required

Generated code may loop, hang, or consume excessive resources.

Timeouts protect the pipeline and make failure modes explicit.

### 10.4 Why This Gate Separates Textbook Data from Hallucinated Text

Sandbox validation gives the factory a concrete evidence layer.

It does not prove pedagogy, but it blocks many false solutions.

## 11. Textbook Packaging: Organizing Course Assets

![Figure 10: Textbook Packaging Artifacts](../../images/part10/10_4_fig10_packaging_outputs.png)

### 11.1 A Textbook Is Not a Pile of Samples

A textbook needs chapters, learning objectives, examples, exercises, solution notes, and review status.

Packaging gives generated samples this structure.

### 11.2 Current Textbook Deliverables

The project should include textbook units, chapter plans, curriculum maps, teacher-guide notes, generated exercise files, and validation reports.

### 11.3 Why the Curriculum Map Matters

The curriculum map explains how topics, difficulty, and examples connect.

It lets reviewers inspect teaching progression.

### 11.4 Why the Teacher Guide Is Not Cosmetic

The teacher guide records intent, expected pitfalls, and review notes.

It is part of the quality system.

## 12. Training Packaging: Moving Textbook Data into Training

![Figure 11: Training Packaging Interface](../../images/part10/10_4_fig11_training_interface.png)

### 12.1 What Training Needs

Training needs stable JSONL records, consistent fields, train/validation/smoke splits, and a manifest.

It should not need to interpret textbook internals.

### 12.2 Current Training Assets

Training assets include instruction-style records, solution records, validation metadata, split files, and a training manifest.

### 12.3 Why Smoke Test Must Be Separate

The smoke set protects the training interface.

It lets loaders and tokenizers validate the shape of the data before full training.

## 13. Metrics and Results: Structural Signals

### 13.1 Current Key Metrics

Key metrics include number of seed problems, generated examples, executable pass rate, topic distribution, difficulty distribution, textbook-unit count, and smoke-test pass rate.

### 13.2 Why a 100% Pass Rate Needs Careful Reading

A 100% pass rate can mean the validator worked.

It can also mean that the sample set is too easy or the checks are too narrow.

The result should be interpreted with sample difficulty and review coverage.

### 13.3 What Topic Distribution Shows

Topic distribution tells whether the textbook covers the intended learning map.

If a few topics dominate, the generator is not respecting curriculum balance.

### 13.4 What Difficulty Distribution Shows

Difficulty distribution tells whether the textbook has progression.

A good factory should not generate only easy or only hard tasks.

## 14. Engineering Effects: Core Problems Solved by P04

### 14.1 It Reduces the Credibility Problem of Free-form CoT

Executable reasoning gives the pipeline a stronger check than prose-only reasoning.

### 14.2 It Reduces the Organization Problem of Textbook Content

Chapter plans and curriculum maps turn scattered samples into a learning asset.

### 14.3 It Reduces the Gap Between Textbook Data and Training Intake

Training packaging gives downstream users a stable interface.

### 14.4 It Makes the Project Acceptable

Reports, manifests, sandbox results, and checks give reviewers concrete evidence.

## 15. Cost and Optimization: Bottlenecks Before Scaling

### 15.1 Current Cost Characteristics

The main costs are generation, retries, sandbox execution, review, and editing.

### 15.2 Where the Real Bottleneck Lies

The bottleneck is usually not generation speed alone.

It is the combined cost of verification and review.

### 15.3 How to Scale

Scale by improving prompt stability, filtering failures earlier, caching validated examples, and expanding review tools.

### 15.4 Why Optimization Is Not Only About Lower Price

The goal is better usable output per review hour.

Cheaper low-quality generation is not real optimization.

## 16. Key Deliverables

- Seed problem file.
- Chapter plan.
- Curriculum map.
- Evolved problem set.
- PoT solution file.
- Sandbox execution report.
- Textbook units.
- Teacher guide.
- Training JSONL files.
- Validation manifest.
- Release check report.

## 17. Chapter Summary: Method Value of P04

P04 shows that synthetic data quality depends on organization, not only generation.

The project keeps seed intent, curriculum structure, executable verification, textbook packaging, and training delivery in one chain.

That makes the factory reviewable.

The method can extend beyond math and code, but every new subject needs its own validation and editorial standards.

## Special Topic: Acceptance Criteria for a Textbook Factory

### 1. Sample-level Acceptance: Is Each Record Worth Keeping?

Each record should have a clear problem, an executable or reviewable solution, topic labels, difficulty labels, and validation status.

Records without learning value should not survive because they happen to be syntactically valid.

### 2. Volume-level Acceptance: Does the Textbook Have Teaching Structure?

The volume should show chapter progression, topic balance, and consistent style.

It should not feel like a random concatenation of generated tasks.

### 3. Training-level Acceptance: Can the Asset Enter Training Stably?

Training files should have stable schema, disjoint splits, manifests, and smoke-test records.

The training interface is part of the deliverable.

## Special Topic: Operating Mechanism of the Textbook Factory

### 1. Role Division

Curriculum owners define the learning map.

Data engineers maintain schemas and pipelines.

Model engineers manage generation and retry logic.

Reviewers inspect correctness and teaching value.

### 2. Version Rhythm

Textbooks need release cycles.

Each release should state added topics, changed prompts, validation status, and known limitations.

### 3. Failure Rework

Bad samples should become improvement clues.

The factory should classify failures by ambiguity, wrong solution, bad code, poor pedagogy, or schema issue.

### 4. Manual Spot Checks

Human review remains necessary because execution does not prove teaching quality.

Reviewers should inspect both correctness and usefulness.

## Special Topic: From Project Prototype to Subject Platform

### 1. From Two Subjects to STEM Textbook Families

The same architecture can expand from math and code to physics, chemistry, statistics, and engineering.

Each subject needs its own validation tools.

### 2. From Text-only Textbooks to Multimodal Textbooks

Future versions can add diagrams, tables, plots, and interactive examples.

This requires image assets and multimodal QA.

### 3. From Offline Textbook Production to Learning-feedback Reflux

Learner errors and teacher edits can become new seeds.

This turns the factory into a continuous improvement loop.

## Special Topic: Pre-release Checklist for Textbook Versions

### 1. Check Three Kinds of Consistency First

Check schema consistency, chapter-map consistency, and manifest-to-file consistency.

### 2. Check Teaching Usability, Not Only Engineering Usability

A file can load correctly and still be a poor lesson.

Review should inspect examples, explanations, and exercise progression.

### 3. Confirm Issue Ledger Status

Before release, unresolved issues should be closed, explicitly inherited, or marked as known limitations.

## Special Topic: Editorial Review in Textbook Factories

### 1. Editorial Review Asks Whether the Content Is Worth Learning

Correctness is necessary but not sufficient.

The sample should also teach a useful concept.

### 2. Review Mechanisms Stabilize Style

Style consistency comes from templates, guides, examples, and repeated review.

It should not be left to model output alone.

## Chapter Summary

This chapter used the synthetic math and code textbook factory to show how generated reasoning data can become a curriculum asset.

The case connects seed problems, task evolution, PoT generation, sandbox validation, textbook packaging, training delivery, and release checks.

Its boundary should remain clear: it is a reproducible prototype for textbook-style data engineering, not a finished educational platform.

## Release Review Notes

The first release review item is seed quality.

Seed problems should have clear topics, expected difficulty, and valid source status.

Seeds with ambiguous wording should be fixed before evolution.

The second item is chapter alignment.

Every generated problem should belong to a chapter or curriculum unit.

If many records cannot be placed, the curriculum map is too weak.

The third item is evolution validity.

Evol-Instruct should make problems richer without breaking solvability.

Reviewers should inspect whether context, constraints, and difficulty are meaningful.

The fourth item is PoT correctness.

Executable code should match the problem statement.

Passing code is not enough if it solves a different problem.

The fifth item is sandbox security.

The sandbox should block dangerous imports, long execution, network access, and filesystem writes unless explicitly allowed.

The sixth item is timeout behavior.

Timeouts should be recorded as validation failures.

They should not be silently retried until success.

The seventh item is textbook packaging.

Textbook units should show learning objectives, problem statements, solutions, and review status.

The eighth item is teacher-guide usefulness.

The guide should explain why the problem exists and what misconception it targets.

The ninth item is training package integrity.

Train, validation, smoke, and manifest files should agree with textbook artifacts.

The tenth item is issue ledger status.

Known bad samples should be closed, repaired, rejected, or explicitly inherited as known limitations.

## Operating Notes

Daily operation should inspect generation failure rate, sandbox pass rate, retry counts, and review issue types.

If evolution failures rise, inspect seed clarity and prompt constraints.

If PoT failures rise, inspect code extraction and execution environment.

If sandbox timeouts rise, inspect task difficulty and generated code style.

If textbook review failures rise, inspect curriculum alignment rather than only the generator.

If training smoke tests fail, inspect schema and manifest before editing generation prompts.

The factory should keep a failure ledger.

Each failure should record whether it came from seed ambiguity, evolution drift, code error, sandbox timeout, packaging mismatch, or teaching-quality issue.

The ledger turns bad samples into process improvements.

Textbook releases should use version notes.

Version notes should record topics added, topics removed, prompt changes, verifier changes, pass rates, review results, and known issues.

This gives educators and model teams a shared language for dataset changes.

## Editorial Notes

A generated problem can be correct and still not be worth teaching.

Editorial review should ask whether the problem teaches a concept, demonstrates a method, and fits the chapter sequence.

It should also ask whether the solution is readable.

Readable solutions are important because training data teaches style as well as correctness.

The project should maintain style examples.

Style examples help future generation remain consistent across releases.

For larger subject coverage, each subject needs its own editorial rubric.

Mathematics, code, physics, and statistics do not share the same validation evidence.

The factory architecture transfers, but the subject rubric must be rebuilt.

## Final Acceptance Checklist

Seeds have topics.

Seeds have difficulty labels.

Chapter plans are complete.

Evolution prompts are versioned.

Generated problems remain solvable.

PoT code is extracted cleanly.

Sandbox limits are configured.

Timeouts are recorded.

Execution results are stored.

Teacher-guide notes are present.

Curriculum map matches chapters.

Textbook units have review status.

Training splits are disjoint.

Smoke set covers key formats.

Manifest matches artifacts.

Issue ledger is reviewed.

Known weak topics are listed.

Release notes include prompt changes.

Editorial samples are preserved.

The next version has clear improvement targets.

## References

1. Xu, C., Sun, Q., Zheng, K., et al. (2023). WizardLM: Empowering Large Language Models to Follow Complex Instructions.
2. Chen, W., Ma, X., Wang, X., & Cohen, W. W. (2022). Program of Thoughts Prompting.
3. Cobbe, K., Kosaraju, V., Bavarian, M., et al. (2021). Training Verifiers to Solve Math Word Problems.
4. OpenAI. (2023). Process supervision and reasoning model research notes.
5. NIST. (2023). Artificial Intelligence Risk Management Framework.
