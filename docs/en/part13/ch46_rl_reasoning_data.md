# Chapter 46: Reasoning Models and RL Data Engineering: The R1 / QwQ Paradigm

## Chapter Overview and Learning Objectives

Chapter 45 discussed SFT, preference alignment, reward models, and RLVR data interfaces in post-training. This chapter moves one step further and focuses on one of the most important changes in the open community since 2025: reasoning models no longer rely only on human-written long chain-of-thought samples. They increasingly use reinforcement learning and verifiable rewards to expand their own reasoning-trajectory space.

In traditional post-training, the central data engineering question is often how to construct better answers. In reasoning models such as R1, QwQ, and Kimi-1.5, the question becomes how to let a model explore, fail, verify, filter, and then turn successful trajectories into the next round of supervised data. Data engineering no longer manages only instruction and response. It manages task pools, sampled trajectories, verifiers, reward signals, failure reasons, rejection-sampling results, and second-round SFT data.

The chapter has four threads:

* The shift from human-written CoT to RL-generated Long-CoT.
* The R1-style data flywheel: cold-start SFT, large-scale RL, rejection sampling, and second-round SFT.
* Reward and verifier engineering: rule-based reward, model-based reward, verifier pools, and process signals.
* Reproduction paths: OpenThoughts, Sky-T1, and low-cost reasoning data factories that small teams can run.

The engineering chain is:

**task pool -> cold-start SFT -> multi-sample rollout -> rule verification -> RL update -> rejection sampling -> second-round SFT -> evaluation and feedback**

The core goal is to decompose reasoning ability from a model behavior into a set of data engineering objects that can be produced, verified, and reviewed.

## 46.1 Why Reasoning Data Is No Longer Just Written CoT

In early instruction tuning, teams often understood reasoning ability as giving the model more answers with steps: detailed math solutions, step-by-step code analysis, or logic derivations. Such data can teach a model to answer as if it is reasoning, but it has a natural limit: the model is imitating prewritten trajectories rather than exploring error space.

This limit appears quickly on complex tasks. A model can write fluent "first, next, therefore" reasoning while none of the steps are verified. It can perform well on familiar training-set patterns but break when math conditions, boundary inputs, or code tests change slightly. Human-written CoT is also expensive and cannot cover large numbers of failure, repair, and boundary trajectories.

R1-Zero-style experiments give data engineers a different insight: on verifiable tasks, a model does not necessarily need to see many human CoT examples before useful reasoning behavior emerges. If the task can be programmatically verified, sampling and reward signals can gradually help the model discover better reasoning paths. Final answers to math problems, unit tests for code, schema checks for structured outputs, and return status from tool calls can all become training signals.

This does not make SFT unimportant. The difference between R1 and R1-Zero shows the opposite: pure RL can stimulate reasoning behavior, but it may produce poor readability, messy format, language mixing, and unstable answers. Cold-start SFT provides base format, readable reasoning style, and output boundaries. The key is not choosing only SFT or only RL, but placing SFT and RL in one data flywheel.

Traditional CoT data and RL reasoning data differ in three ways.

First, the supervision unit changes. Traditional CoT usually supervises a complete answer. RL reasoning data can supervise one rollout, a candidate group, a final answer, a verifier result, or even an intermediate step.

Second, quality judgment changes. Traditional CoT depends on humans or strong models judging whether something is well written. RL reasoning data prioritizes verifiable signals: is it correct, does it pass tests, does it satisfy the format?

Third, the lifecycle changes. Traditional CoT is usually a static dataset. RL reasoning data is generated cyclically. As the current model improves, sampled trajectories become richer and rejection sampling can leave more new supervision data.

This chapter is therefore not about RL algorithms themselves. It is about the data engineering questions under an RL paradigm: where tasks come from, how verifiers are written, how rollout trajectories are stored, which traces enter second-round SFT, which failures enter the hard-case pool, and how reward hacking is prevented.

The object managed by data engineers also changes. Previously, a sample was mainly bounded by prompt and answer. Now a sample may correspond to a task family, sampling configuration, several candidate trajectories, multiple verifier outputs, a human-audit conclusion, and a downstream training destination. Reasoning data is not a longer answer; it is a training asset with a production process. It must explain why the model generated the trajectory, why the system decided to keep it, and which behavior should change after it enters training.

This matters for project planning. Downloading a Long-CoT dataset and converting it into SFT format gives a static fine-tuning run. Connecting task pool, sampling, verification, filtering, and feedback gives a reasoning data flywheel. Static fine-tuning can improve one model at one point in time. A flywheel lets each round's successes and failures become next-round data: success traces stabilize ability, failure traces expand hard cases, format failures repair protocols, and verifier loopholes update rules.

Good first-round RL reasoning tasks usually have four properties. They can be generated or collected at scale, such as math problem sets, code repair tasks, SQL queries, table calculations, rule configurations, and tool-call chains. Answers can be verified by some mechanism, not necessarily fully automatic but clear enough to produce pass/fail. Difficulty has a gradient. And the task distribution matches the target application rather than only benchmark score chasing.

Tasks that are poor first-round RL targets should be identified early. Open-ended strategy analysis, medical advice, legal explanation, emotional support, and creative writing often lack a unique answer and can carry high error cost. These tasks may later use model-based reward, human preference, or expert audit, but they should not be the first training targets before verifiers mature. For small teams, starting with verifiable tasks and then expanding to semi-open tasks is usually more stable than covering everything at once.

## 46.2 The Four Stages of the R1-Style Data Flywheel

The R1-style reasoning flywheel can be split into four stages: cold-start SFT, large-scale RL, rejection sampling, and second-round SFT. These stages form a repeatable closed loop rather than a one-time linear pipeline.

![Figure 46-1: Four stages of an R1-style reasoning data flywheel](../../images/part11/31_1_r1_reasoning_flywheel.png)

*Figure 46-1: Data feedback among cold-start SFT, large-scale RL, rejection sampling, and second-round SFT.*

### 46.2.1 Stage 1: Cold-Start SFT

Cold-start SFT does not aim to create the strongest reasoning model. It gives the model readable, stable, parseable reasoning-output format. This stage usually needs a small set of high-quality Long-CoT samples covering math, code, logic tasks, format following, and necessary general QA.

Cold-start data should satisfy four conditions. The reasoning process should be readable, showing key steps, condition references, and conclusion closure. The format should be parseable, commonly by placing reasoning inside `<think>` and the final answer inside `<answer>` or a fixed field. At least some tasks should be verifiable, otherwise RLVR cannot connect later. Language and style should be consistent: a Chinese-facing model should not mix English templates throughout its reasoning, and an English code-explanation model should not retain Chinese prompt artifacts.

The scale need not be large. For teaching or small-team projects, a few thousand to tens of thousands of high-quality Long-CoT examples often beat hundreds of thousands of low-quality CoT samples. Community projects such as Sky-T1 show that small, carefully constructed reasoning data can significantly improve math and code performance in 32B-class open models [D].

The common mistake is making cold-start samples too perfect. Real post-RL traces contain probing, checks, condition review, and corrections. If human cold-start samples show only linear derivations, the model learns an overly neat explanation style. That style reads well on easy problems but may lack self-checking on complex ones. Cold-start data can therefore keep moderate intermediate checks such as confirming boundary conditions or testing a small example.

Cold-start data must also control answer leakage. Synthetic data often knows the standard answer first and then backfills reasoning, causing steps and conclusion to become too tightly coupled. The model may learn to guess by pattern rather than reason. Safer records include fields such as `answer_source`, `verified_by`, `trace_quality`, and `leakage_risk`. If samples come from teacher distillation, record teacher model, sampling temperature, and filters.

After cold-start SFT, do not inspect only benchmark scores. Check whether the model stably emits the target format, places final answers in parseable positions, avoids over-explaining simple tasks, keeps the intended language, and avoids dangerous actions in code tasks. Only when these engineering conditions hold can later sampling and verifiers connect reliably.

### 46.2.2 Stage 2: Large-Scale RL

In the RL stage, the model no longer imitates given answers. It performs multi-sample rollouts on a task pool and updates policy using reward signals. The key data object is no longer one SFT sample, but a trajectory group:

```json
{
  "task_id": "math_000123",
  "prompt": "...",
  "samples": [
    {
      "sample_id": "s0",
      "reasoning": "...",
      "answer": "42",
      "verifier_pass": true,
      "reward": 1.0
    }
  ],
  "verifier": "sympy_v1",
  "model_version": "policy_step_0200"
}
```

This structure shows that RL data engineering must record task, sampling, answer, verifier, reward, and model version. If only final weights are saved, the team cannot reconstruct why the model improved or where reward hacking came from.

The first RL task pools usually focus on math, code, and structured output because verifiers can be written. Open writing, emotional support, legal advice, and similar tasks may involve reasoning, but they cannot be judged by one rule and are poor first RLVR targets.

Task difficulty distribution determines learning signal. If tasks are too easy, most rollouts pass and rewards lack discrimination. If tasks are too hard, almost every rollout fails and the model receives little positive feedback. A practical pool has four tiers: basic tasks for format stability, medium tasks for main learning signal, hard tasks for upper-bound expansion, and held-out tasks for evaluation only. After each round, sampling ratios can be adjusted by pass rate.

Sampling configuration is part of the recipe. Temperature, top_p, max_tokens, stop tokens, candidate count, and random seed all change trajectory distribution. Low temperature gives stability but weak exploration. High temperature increases diversity but also format failures and invalid reasoning. Multi-sample rollout is necessary because one sample shows only one path, while a group reveals the model's uncertainty. If 1 of 16 candidates is correct, the task still carries learning value. If all 16 are correct, it is better as stability evaluation. If all 16 are unparseable, repair format or lower difficulty first.

Training logs and data logs must align. Loss, mean reward, and benchmark score are not enough. Record pass rate by task type, format failure rate, average trajectory length, repeated-fragment ratio, language-mixing ratio, and verifier-error ratio. When a model becomes better at reasoning but more verbose, the team can tell whether the cause is reward design, sampling limits, or second-round SFT mixture.

### 46.2.3 Stage 3: Rejection Sampling

Rejection sampling connects the flywheel. The model generates multiple candidates for the same task, and the system keeps high-quality traces using verifiers, voting, format checks, and safety filters. DeepSeek-R1 reports large-scale reasoning data built by rejection sampling and mixes in non-reasoning data to preserve general ability [D].

Rejection sampling is not simply choosing the highest-reward sample. A reliable filter checks at least five signals: final answer correctness, obvious contradictions in reasoning, parseable output format, language mixing or repetitive lengthening, and safety, copyright, or privacy triggers.

If only final answer is checked, the system may keep traces that guessed the answer with a wrong process. If only RM score is used, reward-model bias may be amplified. Rejection sampling should therefore combine hard filtering and soft ranking. Hard filters remove wrong answers, unparseable format, safety violations, excessive repetition, and length-limit violations. Soft ranking selects among valid samples for clear reasoning, moderate length, stable language, and complete condition references. This avoids poisoning second-round SFT with very long but barely relevant correct answers.

Homogeneity is another risk. Keeping only the top trace per problem may bias the training set toward one expression template. A better approach keeps a limited number of diverse successful traces while deduplicating near-equivalent text. Math may keep algebraic, geometric, or enumerative solutions. Code may keep different implementation strategies, as long as readability and complexity are reasonable.

Failure traces are valuable. Fully wrong samples should not enter second-round SFT, but they should enter error analysis. Near-miss samples can enter hard-case pools. Format-wrong but conceptually useful samples can enter format-repair data. Samples revealing verifier loopholes should trigger verifier updates. Mature flywheels do not delete failures; they turn failures into engineering tasks.

### 46.2.4 Stage 4: Second-Round SFT

High-quality traces retained by rejection sampling can be repackaged as SFT data. Second-round SFT stabilizes behaviors discovered during RL so the model can produce similar traces without extensive search.

Mixture design is crucial. If second-round SFT uses only reasoning traces, the model may treat every question as a long reasoning problem. If it focuses only on math and code, it may lose chat, factual QA, and format-following ability. DeepSeek-R1's mixture of reasoning and non-reasoning SFT data is important because it enhances reasoning while preserving assistant behavior.

Training records should keep source tags such as:

* `source=rl_rejection_math`
* `source=rl_rejection_code`
* `source=non_reasoning_sft`
* `source=safety_alignment`
* `source=format_following`

This allows later diagnosis of overthinking, overlong answers, or general ability regression. A practical mixture has an ability-enhancement layer from RL success traces and an ability-preservation layer from general dialogue, factual QA, short answers, format following, and refusals. Product assistants need this balance: meeting-note organization should not trigger hundreds of words of reasoning, while SQL generation may need explicit assumptions and constraints.

Evaluation after second-round SFT should split capability and behavior. Capability evaluation checks math, code, logic, long context, and structured output. Behavior evaluation checks answer length, format stability, language consistency, safety boundaries, and ordinary conversation. Many reasoning-model product issues are not hard-task failures; they are overthinking on easy tasks.

Second-round SFT should also produce an independent manifest: shard source, filtering rules, sampling date, policy model, verifier version, and deduplication fingerprint. When a model version later behaves strangely, the team can locate whether the cause is a rejection-sampling batch, a verifier version, or a non-reasoning mixture ratio.

## 46.3 Key Technical Choices

The core of R1-style data engineering is reward signal and verifier design. Without reliable reward, RL amplifies existing bias. Without traceable data structures, rejection sampling cannot be reviewed.

![Figure 46-2: Reward signals and verifier structure for reasoning data](../../images/part11/31_2_reward_verifier_architecture.png)

*Figure 46-2: Relationship among rule-based reward, model-based reward, and human audit.*

### 46.3.1 Rule-Based Reward and Model-Based Reward

Rule-based reward is produced by programs. Math answers can be compared, code can run unit tests, JSON output can be checked against a schema, and SQL can be executed and compared against result tables. Its advantages are stability, low cost, and reproducibility. Its limitation is coverage.

Model-based reward is produced by a reward model or LLM-as-Judge. It can cover open QA, explanation quality, style, and safety boundaries, but is more vulnerable to preference drift, length bias, and prompt sensitivity.

The two reward types should be layered, not substituted. For answer-verifiable tasks, use rule-based reward first. For open tasks, use model-based reward with human sampling and audit sets. For high-risk domains, do not rely only on model judgment.

Reward should not be stored only as a float. A useful record stores `reward_score`, `reward_source`, `pass_flag`, `failure_reason`, and `audit_notes`. A math answer may get 1.0 through symbolic comparison, code may get 1.0 after all tests pass, format error may get 0.0, and a correct but overlong answer may get 0.8. The score trains the algorithm; the reason field repairs the data.

Rule-based reward is reproducible, but it can also create loopholes. If a verifier only checks the final answer, the model can ignore process. If tests are incomplete, code can overfit them. If a JSON schema only checks field presence, meaningless values can pass. Verifiers need unit tests, regression tests, and adversarial cases like production code.

Model-based reward is useful for human preferences, but should be constrained. If a judge model controls all rewards, the training system inherits its tastes, such as longer explanations, polite phrasing, or confident tone. For reasoning tasks, a judge is better used to supplement verifiable answers by checking self-consistency, missing conditions, and obvious hallucination. Store judge prompts so future score changes can be compared.

| Reward type | Suitable tasks | Advantages | Risks | Data fields to record |
| --- | --- | --- | --- | --- |
| Rule-based reward | Math, code, structured output | Stable, reproducible, cheap | Limited coverage, rule loopholes | verifier version, test cases, failure reason |
| Model-based reward | Open QA, style, safety | Broad coverage | Preference drift, length bias, judge contamination | judge version, scoring prompt, rationale |
| Human audit | High-risk and disputed samples | Reliable judgment | High cost, limited scale | annotator agreement, review rounds |

For most teams, the safer order is rule-based reward first, small-scale model-based reward second, and human audit for high-risk samples. Starting with judge-based reward for every task may run quickly, but later it becomes hard to explain learned preferences.

### 46.3.2 Verifier Pools

A verifier pool is infrastructure, not a single script. It is a set of verifiers that can be versioned, tested, and rolled back.

Math verifiers usually include answer extraction, unit normalization, symbolic comparison, and tolerance checks. Equivalent answers such as `1/2`, `0.5`, and `50%` should not be compared only as strings. Symbolic tools such as `sympy` (Meurer et al. 2017) are safer.

Code verifiers include sandbox execution, timeouts, memory limits, unit tests, and safety blocking. Code reward should record failure type: compile error, runtime error, timeout, wrong answer, or format error.

Format verifiers check JSON, XML, tool-call parameters, and `<think>/<answer>` tags. Many reasoning-model failures are not wrong answers but unparseable answers.

A maintainable verifier pool needs four interfaces. `extract` extracts final answers, code blocks, JSON fields, or tool parameters. `normalize` handles units, whitespace, case, math expressions, and dependency arrangement. `check` executes verification logic. `explain` returns failure reasons for logs and hard-case analysis.

For math, `extract` handles forms such as "the answer is 1/2" or "therefore x = 0.5"; `normalize` converts fractions, decimals, percentages, and units; `check` distinguishes exact equality, numerical approximation, and symbolic equivalence; `explain` returns `parse_error`, `not_equivalent`, `unit_mismatch`, or `multiple_answers`. Without these reasons, pass-rate drops are hard to debug.

Code verification is harder. It must isolate the filesystem, block network, limit CPU and memory, set timeouts, and record dependency versions. Test cases are data assets. Public benchmark tests may be insufficient for business logic, so hidden tests, boundary tests, and safety tests are needed. If a model passes public tests but fails online, the verifier often missed real constraints.

Long-CoT trace quality can also be analyzed internally. Three useful structural tags are reflection, verification, and backtracking. Reflection revisits assumptions. Verification checks intermediate or final results. Backtracking retreats from a wrong path. These should not be forced into fixed templates, but they are useful labels when analyzing traces.

![Figure 46-3: Long-CoT trace patterns](../../images/part11/31_3_long_cot_trace_patterns.png)

*Figure 46-3: Long-CoT sample profile showing reflection, verification, and backtracking patterns.*

### 46.3.3 Chinese-English Mixing in Long-CoT

Language mixing is common in Long-CoT data. Frequent switching between Chinese and English can destabilize final format. Practices around QwQ and Kimi-1.5 suggest purifying mixed reasoning data.

Cold-start prompts should require language consistency: Chinese tasks reason in Chinese inside `<think>`, English tasks stay English. During RL or rejection sampling, language-mixing penalties can reduce the priority of traces that switch languages frequently. In second-round SFT, prioritize language-stable and answer-parseable traces.

Language consistency is not meant to block cross-lingual knowledge. It improves controllability. A model can use cross-lingual knowledge, but training data should keep user-facing expression stable.

The policy should depend on task type. Chinese math explanations should use natural Chinese reasoning. Code tasks may keep English API names and error messages. Paper QA can preserve original terms while explaining in Chinese. Filters can use lightweight language detectors to estimate language ratios per trace. For Chinese tasks, excessive English reasoning can be marked `language_mixing`; for code tasks, only natural-language spans should be counted, not code keywords.

### 46.3.4 Trajectory Storage and Version Control

Reasoning data depends on metadata more than ordinary SFT. A minimal record should include:

| Field | Meaning |
| --- | --- |
| `task_id` | Original task ID |
| `source` | Data source, such as GSM8K (Cobbe et al. 2021), MATH (Hendrycks et al. 2021), HumanEval (Chen et al. 2021), or proprietary task pool |
| `policy_model` | Model version that generated the trace |
| `sampling_config` | temperature, top_p, max_tokens, seed |
| `reasoning_trace` | Reasoning process |
| `final_answer` | Final answer |
| `verifier_name` | Verifier name |
| `verifier_version` | Verifier version |
| `reward` | Reward value |
| `failure_reason` | Failure reason |
| `selected_for_sft` | Whether the trace enters second-round SFT |

Without these fields, a team cannot tell whether performance came from the model, sampling, verifier, or filter.

Storage should distinguish raw trajectories from training views. Raw trajectories preserve model output, sampling configuration, logs, and verifier results. Training views are cleaned SFT or RL inputs for training. If only the training view is saved, debugging information is lost. If raw trajectories are trained directly, errors, repeats, and sensitive content can enter the model.

A useful version structure has three layers: task version, such as `task_pool_math_v3`; rollout version, such as `rollout_qwen32b_step1200_temp0.8_v1`; and training-set version, such as `sft_rejection_mix_2025_02_01`. This supports many rollout rounds from one task pool, multiple filtering strategies from one rollout, and multiple training experiments from one filtered set.

Deduplication is part of storage. Reasoning data has task duplicates and trace-template duplicates. Task duplicates can leak evaluation data. Template duplicates teach fixed phrasing. Deduplicate tasks with text, answer, and semantic embeddings; deduplicate traces with n-gram or paragraph-level similarity. This is especially important when mixing open and proprietary data because the same benchmark problem may appear in rewritten form.

## 46.4 Dataset and Model Disclosure Comparison

With DeepSeek-R1, QwQ-32B, Kimi-1.5, and similar models, traditional SFT instruction datasets increasingly give way to RL and Long-CoT trajectory datasets. Public reports and dataset cards disclose different levels of detail, so this section marks direct information as [D], public-context inference as [I], and teaching estimates as [E].

### DeepSeek-R1: Cold Start, RL, and Rejection Sampling

The DeepSeek-R1 report (Guo et al. 2025) describes two paths: R1-Zero and R1 [D]. R1-Zero shows that large-scale RL can elicit reasoning behavior without traditional SFT cold start, but readability and stability suffer. R1 adds a small amount of cold-start Long-CoT data before RL, then uses RL, rejection sampling, and second-round SFT to produce a more stable model.

The route has three key data engineering points. Cold-start data emphasizes readability rather than large scale. Rejection sampling produces large reasoning trajectories; the disclosed 600K reasoning data and 200K non-reasoning data show that R1 maintains general assistant behavior while improving math and code [D]. Non-reasoning data is important because pure math-code RL can regress ordinary dialogue and factual QA.

For reproduction, the most valuable lesson is responsibility separation. Cold start makes the model speak clearly. RL makes it explore. Rejection sampling keeps good traces. Second-round SFT stabilizes behavior. Trying to solve all four with one big SFT dataset usually reaches a ceiling: the model imitates long reasoning format, but does not receive explicit feedback when wrong, nor does it create more useful traces.

### QwQ-32B: Open Weights and RL Post-training

The QwQ-32B (Qwen Team 2025) card states that training includes pre-training and post-training, with SFT and RL in post-training [D]. Compared with DeepSeek-R1, full data ratios are less disclosed, so this chapter does not invent them.

QwQ-style models show that reasoning traces often include waiting, checking, reflection, and backtracking. These may come from model sampling, training data, and RL objectives together [I]. The data engineering point is not to hard-code words such as "wait," but to preserve complete traces of trial, verification, and correction.

Open weights do not equal open data recipes. A model can be evaluated or used as a teacher, policy, or baseline, but its undisclosed data ratios should not be reported as facts. A common project use is to let QwQ generate Long-CoT candidates and then filter them with local verifiers. This is not reproducing QwQ's recipe; it is using a strong reasoning model as a candidate generator under local task constraints.

### Kimi-1.5: Long Context and RL Scaling

Kimi k1.5 (Kimi Team 2025) emphasizes long-context scaling and improved policy optimization, and reports an RL framework that does not depend on more complex MCTS, value functions, or PRMs [D]. This route reminds us that reasoning data is not only math and code; it can come from long-context tasks.

Long-context reasoning is about evidence management. The model must locate evidence in long documents, multi-part material, or multi-turn context and convert evidence into a reasoning chain. Data records need citation positions, evidence snippets, answer rationale, and failure reasons. If only final answers are stored, the system cannot tell evidence-based reasoning from plausible guessing.

This expands reasoning data from "question-answer" to "context-evidence-reasoning-answer." Long-context verifiers are more complex. Document QA must check whether citations support conclusions. Multi-turn tasks must check whether earlier constraints were preserved. Tool-augmented tasks must check tool-call order and intermediate results. Layered evaluation is useful: format, evidence support, final answer, then concision and safety.

### OpenThoughts and Sky-T1: Community Reproduction Paths

OpenThoughts-114K (Gunther et al. 2025) is an important open reasoning dataset. Its Hugging Face card shows an Apache-2.0 license and parquet data [D]. Its value is downloadable, inspectable Long-CoT samples that researchers can use for reproduction and recipe study.

Sky-T1 (NovaSky-Berkeley 2025) shows a low-cost route. Public materials state that Sky-T1-32B-Preview is based on Qwen2.5-32B-Instruct and trained with a smaller high-quality reasoning dataset; the team released model, data, and code [D]. It shows that reasoning improvement is not only a product of large-scale RL. Structured Long-CoT SFT can produce strong gains in some settings.

These projects turn reasoning data from an internal-company artifact into something the community can inspect and reproduce. They are good starting points: download open data, inspect fields and licenses, sample trace quality, and connect verifiable tasks to local verifiers. But open data is not business data. Before use, check target language fit, task coverage, and evaluation leakage risk.

**Table 46-1: Reasoning model and dataset disclosure comparison**

| Model / dataset | Core driving stage | Reasoning trace source | Open / downloadable | Distinctive strategy | Tag |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-R1 | Cold start + RL + rejection sampling + SFT | In-house multi-sample rollout and rule verification | Model open, training data not fully open | 600K reasoning + 200K non-reasoning data | [D] |
| QwQ-32B | SFT + RL | Not fully disclosed | Weights open | Mid-size reasoning model, RL post-training emphasized | [D/I] |
| Kimi k1.5 | Long-context RL | Not fully disclosed | Not released as full dataset | Long-context scaling and policy optimization | [D] |
| OpenThoughts-114K | SFT / open reproduction | Community synthesis and curation | Dataset open | 114K-class Long-CoT data | [D] |
| Sky-T1 | Small-scale Long-CoT SFT | QwQ distillation and curation | Model, data, and code open | Low-cost reproduction of reasoning ability | [D] |

**Table 46-2: Long-CoT data feature comparison**

| Dimension | DeepSeek-R1 | QwQ-32B | Kimi k1.5 | OpenThoughts / Sky-T1 |
| --- | --- | --- | --- | --- |
| Main tasks | Math, code, general alignment | Math, code, reasoning | Long-context, multimodal, reasoning | Math, code, science, logic |
| Trace source | RL rollout and rejection sampling | SFT + RL post-training | Long-context RL | Synthesis and distillation |
| Downloadable | Not fully open | Weights open, data not fully open | Not fully open | Downloadable |
| Verification signal | Mainly rule verification | Limited disclosure | RL reward and evaluation | Many rule-verifiable tasks |
| Reproduction value | Understand industrial flywheel | Understand open-weight reasoning model | Understand long-context RL | Suitable for small-team experiments |

Disclosure is uneven. Industrial models often disclose stages, results, and partial scale, but not full data. Community projects more often disclose data and scripts, but scale and coverage are limited. Reproduction should avoid two extremes: giving up because industrial data is not open, or assuming downloadable community data reaches industrial performance. A practical route is to learn the industrial stage design, use community data for a minimal loop, and gradually replace generic task pools with proprietary tasks.

## 46.5 Case Studies

This section decomposes three components: OpenThoughts-114K, a rule-based reward verifier pool, and rejection sampling in practice. They correspond to data source, reward signal, and data feedback.

### Case A: OpenThoughts-114K

OpenThoughts-114K shows the shape of open reasoning data. It is no longer a simple `{"instruction": "...", "response": "..."}` dataset. Samples are organized around problem, reasoning, answer, and metadata.

A reasoning sample answers three questions: what is the task, how did the model reason, and can the final answer be verified? Math answers can use symbolic or reference comparison. Code can run in a sandbox. Logic tasks can use constraint checks or human sampling. The value of OpenThoughts is not only size; it shows that reasoning data stores intermediate process, not only answers.

For industrial reproduction, the useful starting point is to find verifiable tasks: SQL generation, table calculation, rule configuration, API call chains, code repair, and structured extraction. If a verifier can be written, a small RLVR or rejection-sampling pipeline can be built.

Ingesting open reasoning data should use four checks. Verify license, inspect fields, sample quality across task types, and check contamination against planned evaluation sets. After ingestion, do not train on everything directly. Build a `curated` subset for complete, language-stable, verifiable, well-sourced samples. Put valuable but incomplete samples into `needs_repair`, and exclude unverifiable or evaluation-overlap samples.

| Subset | Entry condition | Use |
| --- | --- | --- |
| `curated_long_cot` | Complete reasoning, verifiable answer, clear license | Cold-start SFT |
| `verifiable_tasks` | Standard answer or tests available | RLVR / rejection sampling |
| `needs_repair` | Valuable problem but incomplete fields | Data repair |
| `excluded_eval_overlap` | Highly similar to evaluation set | Forbidden for training |
| `audit_samples` | Suspicious high-score or near-correct low-score cases | Human sampling |

### Case B: Building a Rule-Based Reward Verifier Pool

A rule-based reward pool has three layers. The task layer stores question, reference answer, task type, difficulty, source, and license. Math tasks need answer form; code tasks need function signature, tests, and runtime environment.

The verification layer extracts final answers, normalizes units, simplifies symbols, compares with tolerance, creates sandboxes, runs unit tests, and validates formats such as JSON, XML, tool parameters, or `<answer>` tags.

The log layer records verifier version, input, output, error type, and runtime. A verifier without logs cannot be safely used in training because model regressions cannot be attributed to model, task, or verifier.

Basic math verification:

```python
def verify_math(predicted, reference):
    pred_expr = normalize_and_parse(predicted)
    ref_expr = normalize_and_parse(reference)
    if pred_expr is None or ref_expr is None:
        return {"pass": False, "reason": "parse_error"}
    return {
        "pass": symbolic_equal(pred_expr, ref_expr),
        "reason": "ok"
    }
```

Code verification needs stronger safety:

```python
def verify_code(code, tests, timeout=5):
    result = run_in_sandbox(code, tests, timeout=timeout)
    return {
        "pass": result.all_tests_passed,
        "reason": result.failure_type,
        "runtime_ms": result.runtime_ms
    }
```

Real systems must handle malicious code, infinite loops, dependencies, floating-point tolerance, multiple correct answers, and insufficient test coverage.

Before a verifier pool goes online, test the verifier itself. Math tests should cover integers, fractions, decimals, percentages, intervals, multiple answers, unit conversion, and approximate values. Code tests should cover correct code, syntax errors, runtime errors, timeouts, memory limits, missing dependencies, and malicious operations. Format tests should cover missing fields, wrong field types, extra fields, nested errors, and unparseable output.

Use a unified failure enum such as `parse_error`, `wrong_answer`, `test_failed`, `timeout`, `format_error`, `unsafe_code`, `judge_disagree`, and `verifier_error`. `verifier_error` must be separated from model error; if the verifier crashes, the model should not be marked wrong.

Verifier quality also needs evaluation. Maintain a golden verifier set with human-confirmed correct and incorrect answers. Run regression tests after every verifier update. If a new version fails known-correct answers, parsing or normalization became too strict. If it passes known-wrong answers, rules became too loose.

Cost also matters. Symbolic simplification can be slow, code sandboxes consume CPU and memory, and LLM judges cost inference. A practical pipeline uses cheap rules first, then sends disputed samples to more expensive validators.

### Case C: Rejection Sampling in Practice

Rejection sampling filters high-quality traces from many candidates generated by the model. A minimal process has five steps: sample 4 to 16 candidates per task, parse reasoning and final answer, call verifiers, run secondary filters for format, language, repetition, and safety, and package retained samples as second-round SFT data.

Failure samples should be retained. They are not garbage; they support error analysis, PRM data, and hard-case construction.

| Type | Meaning | Later use |
| --- | --- | --- |
| `pass_good_trace` | Correct answer, clear reasoning | Second-round SFT |
| `pass_bad_trace` | Correct answer, messy process | Audit or PRM data |
| `fail_near_miss` | Nearly correct, local error | Hard-case pool |
| `fail_invalid` | Format error or unparseable output | Format-repair data |

The most common mistake is keeping only successful answers and deleting all failures. That removes error analysis from the flywheel. Success traces enhance ability, failures diagnose weaknesses, and format failures repair protocol.

Retention ratio should depend on difficulty. Easy tasks with high pass rates may keep one concise trace. Medium tasks can keep two or three diverse solutions. Hard tasks may keep near-miss failures even without success. Also define a "not trainable but analyzable" region: privacy, copyright, unsafe code, or verifier-loophole traces should not become positive training data, but they are valuable audit cases.

Rejection sampling should output two artifacts. `sft_selected.jsonl` contains trainable high-quality traces. `rollout_audit.parquet` contains all candidates, scores, failure reasons, and filter decisions. Saving only the former saves space short-term but destroys long-term interpretability.

A complete selection decision can look like:

```json
{
  "sample_id": "math_000123_s07",
  "verifier_pass": true,
  "hard_filter_pass": true,
  "quality_score": 0.86,
  "selected_for_sft": true,
  "selection_reason": "correct_answer;clear_trace;language_stable",
  "excluded_reason": null
}
```

Explicit decision records allow later threshold changes without resampling every task.

## 46.6 Pitfalls

**Reward hacking.** If the verifier has a loophole, the model will exploit it. A math extractor that reads only the final number can be fooled by appending the right number after wrong reasoning. Check answer, process, format, and abnormal patterns together.

**Length explosion.** RL may favor longer reasoning chains. Longer is not better; it increases cost and can reduce readability. Add length limits, redundancy checks, and concise positive traces.

**Language mixing.** Chinese-English mixing hurts user experience and parsers. Unify language requirements in cold start and filter frequent switching during rejection sampling.

**Reasoning that looks plausible but is unverifiable.** Many Long-CoT samples look like reasoning but no step can be checked. They may fit SFT, but not RLVR. Grade task pools by verifiability before RL.

**Overreliance on one benchmark.** If all tasks come from a few math or code benchmarks, the model learns those task forms. Use multiple sources, difficulty levels, and expression styles, and keep evaluation isolation.

**Second-round SFT overthinking.** If all second-round data is long reasoning, the model may output long `<think>` traces for simple questions. Mix in non-reasoning dialogue, short answers, and format-following data.

**No verifier versioning.** The same sample can receive different rewards under different verifier versions. Record versions and run regression tests on each update.

**Training-evaluation leakage.** Reasoning data often comes from public problem sets, community synthesis, and distillation, while evaluations come from similar sources. Deduplicate questions, answers, key constraints, and rewritten variants before training.

**Treating teacher output as truth.** Strong teacher models still produce wrong reasoning, overlong explanations, answer leakage, and hallucinated citations. Teacher data needs verifier checks or human sampling.

**Ignoring negative samples.** Positive-only curation prevents PRM training and failure-boundary analysis. Negative samples may not enter SFT, but they should be saved by error type.

**Uncontrolled sampling cost.** Multi-sample rollout multiplies token cost. Control by difficulty-based candidate counts, early stopping on repeated failures, truncating overlong traces, and cleaning low-value tasks.

**Reward not aligned with product goals.** Better math benchmark scores do not necessarily mean better product behavior. Each reward should map to correctness, executability, readability, safety, and cost.

These problems show that reasoning data engineering is not finished by attaching an RL algorithm. The contracts among task, verifier, sampling, and training data must be maintained continuously. If any definition changes, downstream data must be reinterpreted.

## 46.7 Costs, Risks, and Boundaries

RL reasoning flywheels cost mainly in multi-sample rollout, verifier execution, and second-round training. Sampling cost grows linearly with candidate count; code verification consumes sandbox resources; second-round SFT requires repackaging and training.

Small teams should not start with full RL. A more realistic path is:

* use open Long-CoT data for cold-start SFT;
* build a small verifier pool;
* run rejection sampling to generate 10K to 30K high-quality traces;
* validate gains with LoRA or short-step SFT.

The goal is to establish the data production loop before scaling model and task size.

| Stage | Cost source | Small-team downgrade |
| --- | --- | --- |
| Cold-start SFT | Data cleaning, training | Use open Long-CoT subset |
| Multi-sample rollout | vLLM inference tokens | Reduce candidates from 16 to 4 |
| Verifier | Sandbox, symbolic computation | Start with math, then code |
| Rejection sampling | Filtering and packaging | Keep only `pass_good_trace` |
| Second-round SFT | Training GPU | LoRA or short-step smoke test |

Boundaries must be clear. The R1-style flywheel fits math, code, structured output, tool calling, and some long-context tasks. It does not fit all open chat tasks. For safety, medical, financial, and legal tasks, rule verification covers only part of factuality or format and cannot replace expert review.

Cost evaluation should include more than training GPU. Much spending happens before and after training: building task pools, generating candidates, running verifiers, storing trajectories, auditing, and repeated evaluation. Multi-sample rollout can consume most resources before the team discovers verifier quality is insufficient. Use staged budgets: first 100 to 500 tasks to validate structure and verifiers, then 3K to 10K tasks to produce trainable rejection-sampling data, and only then larger RL or second-round SFT. Each stage should have stop conditions, such as repairing prompts when format pass rate is too low or repairing verifiers when verifier-error rate is high.

Risk boundaries include safety, copyright and license, privacy, and evaluation risk. Open reasoning data is not always commercial or redistributable. Proprietary logs must be anonymized before entering task pools, and generated reasoning traces need access control. Contamination can make a model look strong offline but fail online.

The flywheel works best for tasks that are verifiable, iterable, and analyzable. SQL can be executed and compared. Code repair can run tests. Table reasoning can check values. Tool calling can validate parameters and return status. Value judgment, strategic consulting, and professional advice need more complex evaluation and should usually begin as evaluation or human-audit objects rather than main RL training tasks.

A minimal viable flywheel can be implemented as follows: choose a narrow domain such as short math, SQL generation, or structured extraction; prepare several hundred cold-start samples; implement a verifier; run 4-way sampling with the current model; save traces and verification logs; select first-version SFT data; train a small model or LoRA; and compare with an isolated evaluation set. Scaling only makes sense after this loop works.

## 46.8 Summary

This chapter decomposed the reasoning-model paradigm represented by R1, QwQ, and Kimi-1.5 from a data engineering perspective. Unlike traditional SFT, a reasoning data flywheel is not just longer CoT collection. It organizes task pools, rollout trajectories, verifiers, rewards, rejection sampling, and second-round SFT into a closed loop.

Three conclusions matter most.

First, cold-start SFT solves readability and format stability; RL solves exploration; rejection sampling distills high-quality traces; and second-round SFT stabilizes behavior.

Second, rule-based reward is the preferred entry point for reasoning data engineering. Whenever tasks can be programmatically verified, reasoning training can move from "humans judge whether it is good" to "the system verifies whether it is correct."

Third, OpenThoughts and Sky-T1 show that small teams can prototype a working reasoning data flywheel with high-quality Long-CoT data, lightweight verifiers, and rejection sampling.

The minimal project artifact is not a "reasoning model." It is a set of data assets: task pool, verifier pool, rollout trace store, rejection-sampling training set, failure library, and evaluation report. The model is one consumer of those assets. If the assets keep updating, the team can switch base models, algorithms, and deployment budgets.

Chapter 47 turns to multimodal understanding data engineering. Unlike text reasoning, multimodal models also need images, pages, video frames, OCR, spatial positions, and multi-image relations. The next chapter discusses how these visual inputs enter pre-training, multitask alignment, and multimodal SFT recipes.

## References

Guo D, Yang D, Zhang H, Song J, Wang P, Zhu Q, Xu R, Zhang R, Ma S, Bi X, others (2025) DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2501.12948.

Team Kimi, Du A, Gao B, Xing B, Jiang C, Chen C, Li C, Xiao C, Du C, Liao C, others (2025) Kimi k1.5: Scaling Reinforcement Learning with LLMs. arXiv preprint arXiv:2501.12599.

Touvron H, Martin L, Stone K, Albert P, Almahairi A, Babaei Y, Bashlykov N, Batra S, Bhargava P, Bhosale S, others (2023) Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv preprint arXiv:2307.09288.

Cobbe K, Kosaraju V, Bavarian M, Chen M, Jun H, Kaiser L, Plappert M, Tworek J, Hilton J, Nakano R, others (2021) Training Verifiers to Solve Math Word Problems. arXiv preprint arXiv:2110.14168.

Chen M, Tworek J, Jun H, Yuan Q, Pinto H P O, Kaplan J, Edwards H, Burda Y, Joseph N, Brockman G, others (2021) Evaluating Large Language Models Trained on Code. arXiv preprint arXiv:2107.03374.

Zhou C, Liu P, Xu P, Iyer S, Sun J, Mao Y, Ma X, Efrat A, Yu P, Yu L, Zhang S, Ghosh G, Lewis M, Zettlemoyer L, Levy O (2023) LIMA: Less Is More for Alignment. Advances in Neural Information Processing Systems, 36, 55006-55021.

Zelikman E, Wu Y, Mu J, Goodman N (2022) STaR: Bootstrapping Reasoning with Reasoning. Advances in Neural Information Processing Systems, 35, 15476-15488.

Madaan A, Tandon N, Gupta P, Hallinan S, Gao L, Wiegreffe S, Alon U, Dziri N, Prabhumoye S, Yang Y, Gupta S, Majumder B P, Hermann K, Welleck S, Yazdanbakhsh A, Clark P (2023) Self-Refine: Iterative Refinement with Self-Feedback. Advances in Neural Information Processing Systems, 36, 46534-46594.

Lightman H, Kosaraju V, Burda Y, Edwards H, Baker B, Lee T, Leike J, Schulman J, Sutskever I, Cobbe K (2024) Let's Verify Step by Step. International Conference on Learning Representations.

Zheng L, Chiang W-L, Sheng Y, Zhuang S, Wu Z, Zhuang Y, Lin Z, Li Z, Li D, Xing E, Zhang H, Gonzalez J, Stoica I (2023) Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. Advances in Neural Information Processing Systems, 36, 46595-46623.

Gao L, Schulman J, Hilton J (2023) Scaling Laws for Reward Model Overoptimization. Proceedings of the 40th International Conference on Machine Learning, pp 10835-10866.

Hosseini A, Yuan X, Malkin N, Courville A, Sordoni A, Agarwal R, others (2024) V-STaR: Training Verifiers for Self-Taught Reasoners. arXiv preprint arXiv:2402.06457.

Shi F, Suzgun M, Freitag M, Wang X, Srivats S, Vosoughi S, Chung H W, Tay Y, Ruder S, Zhou D, others (2022) Language Models Are Multilingual Chain-of-Thought Reasoners. arXiv preprint arXiv:2210.03057.

Jaech A, Kalai A, Lerer A, Richardson A, El-Kishky A, Low A, Helyar A, Madry A, Beutel A, Carney A, others (2024) OpenAI o1 System Card. arXiv preprint arXiv:2412.16720.

Ott S, Hebenstreit K, Lievin V, others (2023) ThoughtSource: A Central Hub for Large Language Model Reasoning Data. Scientific Data, 10(1), 528.

Hsieh C-Y, Li C-L, Yeh C-K, Nakhost H, Fujii Y, Ratner A, Krishna R, Lee C-Y, Pfister T (2023) Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes. Findings of the Association for Computational Linguistics: ACL 2023, pp 8003-8017.

Patil S G, Zhang T, Wang X, Gonzalez J E (2024) Gorilla: Large Language Model Connected with Massive APIs. Advances in Neural Information Processing Systems, 38.
