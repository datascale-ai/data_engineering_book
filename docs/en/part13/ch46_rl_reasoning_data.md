# Chapter 46: Reasoning Models and RL Data Engineering: The R1/QwQ Paradigm

## Abstract
The previous chapter discussed SFT, preference alignment, reward models, and RLVR data interfaces in the post-training stage. This chapter moves further forward, focusing on the most significant shift in the open-source community since 2025: reasoning models no longer rely solely on manually written long chain-of-thought samples. Instead, they actively expand their reasoning trajectory space through Reinforcement Learning (RL) and verifiable rewards.

In the traditional post-training pipeline, the central question in data engineering is typically "how to construct better answers." In reasoning models of the R1/QwQ/Kimi-1.5 generation, the question transforms into "how to let the model explore, trial-and-error, and self-filter on verifiable tasks, then distill successful trajectories into supervised data for the next round." This means data engineering no longer only manages instructions and responses—it must manage task pools, sampled trajectories, verifiers, reward signals, failure reasons, rejection-sampling outputs, and second-round SFT data.

This chapter can be understood along four main threads:

* **The data paradigm shift in reasoning models**: from human-written CoT to RL-generated Long-CoT.
* **The R1-style data flywheel**: cold-start SFT, large-scale RL, rejection sampling, and second-round SFT.
* **Reward and verifier engineering**: rule-based rewards, model-based rewards, verifier pools, and process-level signals.
* **Open-source reproduction paths**: OpenThoughts, Sky-T1, and low-cost reasoning data factories operable by small teams.

Reading in engineering order, this chapter traces a complete pipeline:

**Task pool construction → Cold-start SFT → Multi-path sampling → Rule-based verification → RL update → Rejection sampling → Second-round SFT → Evaluation and feedback**

The core objective underlying this structure is to decompose "reasoning capability" from a model property into a set of data engineering objects that can be produced, verified, and audited.

---

## Keywords

Reasoning models; data recipes; open-source large language models; training data; staged scheduling

## Learning Objectives

- Explain the data paradigm shift from human-written CoT to RL-generated Long-CoT in reasoning models.
- Construct the four-stage R1-style data flywheel: cold-start SFT, large-scale RL, rejection sampling, and second-round SFT.
- Design reward and verifier engineering components including rule-based rewards, model-based rewards, verifier pools, and process reward signals.
- Compare dataset exposure across R1, QwQ, Kimi-1.5, and related paradigms, and reproduce low-cost reasoning data factory paths such as OpenThoughts and Sky-T1.
- Identify common failure patterns in reasoning data engineering and evaluate their costs, risks, and applicability boundaries.

## 46.1 The RL Paradigm Problem Setting: Why Reasoning Data Is No Longer Just Writing CoT

In the early instruction fine-tuning era, teams often understood reasoning capability as "giving the model more step-by-step answers." For example, writing out detailed solutions for math problems, step-by-step analyses for coding problems, and inference chains for logic problems. Such data can indeed teach a model to "respond as if reasoning," but it has a fundamental limitation: the model is merely imitating pre-written trajectories and has not genuinely learned to explore the error space.

This limitation quickly manifests on complex tasks. A model can fluently write "first, second, therefore," but each step may not have been verified; it can also perform well on common problem types in the training set, but when faced with slightly varied mathematical conditions, boundary inputs, or code test cases, the reasoning chain breaks. More importantly, manually writing CoT is expensive and cannot cover problem types at scale. A team can write 10,000 or 100,000 high-quality CoT examples, but it is very difficult to manually produce enough failure trajectories, correction trajectories, and boundary-case trajectories.

The insight that R1-Zero experiments brought to data engineering is this: on verifiable tasks, models do not necessarily need to see large quantities of human-written CoT before producing usable reasoning behavior. As long as a task can be programmatically verified, the model can gradually discover more effective reasoning paths through sampling and reward signals. The final answer to a math problem, unit tests for a code problem, schema validation for structured output, and the return status of a tool call can all serve as part of the training signal.

This does not mean SFT is unimportant. On the contrary, the difference between R1 and R1-Zero precisely illustrates that pure RL can stimulate reasoning behavior but may also produce outputs with poor readability, disordered formatting, language mixing, and unstable responses; cold-start SFT can provide the model with basic formatting, readable reasoning styles, and output boundaries. The key to reasoning data engineering is not choosing "only SFT" or "only RL," but placing both SFT and RL within the same data flywheel.

From a data perspective, traditional CoT data and RL reasoning data differ in three ways.

**First, the unit of supervision is different.** The supervision unit for traditional CoT is typically a complete answer; for RL reasoning data, it can be a single sample, a set of candidates, a final answer, a verifier result, or even an intermediate step.

**Second, quality judgment differs.** Traditional CoT relies on human or strong-model judgment of "whether it is well-written"; RL reasoning data prioritizes verifiable signals to judge "whether it is correct, whether it passes tests, whether it meets the format."

**Third, the data lifecycle differs.** Traditional CoT data is typically a static dataset; RL reasoning data is generated in a loop. The stronger the model's current capability, the richer the sampled trajectories, and the more new supervised data rejection sampling can retain.

Therefore, this chapter does not discuss the RL algorithm itself, but rather the data engineering problems under the RL paradigm: where tasks come from, how verifiers are written, how sampled trajectories are stored, which trajectories enter second-round SFT, which failed trajectories enter the hard-case pool, and how to prevent the model from gaming reward signals.

In this paradigm, the objects that data engineers face have also changed. In the past, the main boundaries of a sample were the prompt and the answer; now, a single sample may correspond to a task family, a set of sampling parameters, several candidate trajectories, multiple verifier outputs, one human audit conclusion, and a downstream training destination. In other words, reasoning data is not "a longer answer" but a training asset with an associated production process. It must be able to explain three things: why the model generated this trajectory, why the system judged this trajectory worth retaining, and what category of behavior this trajectory is intended to change in the model once it enters training.

This point is especially important for project planning. If a team only downloads a Long-CoT dataset and converts it into SFT format, the result is a one-time static fine-tuning pass; if a team can connect task pool, sampling, verification, filtering, and feedback into a loop, the result is a reasoning data flywheel. Static fine-tuning can improve model performance at a single point in time, but it cannot continuously absorb new solutions that the model discovers on its own. The data flywheel allows the team to convert every round of successes and failures into data for the next round: successful trajectories are used to stabilize capabilities, failed trajectories to expand hard cases, format failures to repair protocols, and verifier bugs to update rules.

From a business deployment perspective, tasks suitable for entering the RL reasoning flywheel typically share four characteristics. First, problems can be generated at scale or reliably collected, such as math problem banks, code repair tasks, SQL queries, table calculations, rule configurations, and tool call chains. Second, answers can be verified by some mechanism—not necessarily fully automatic, but at least capable of producing a clear pass or fail signal. Third, task difficulty has gradients, including both problems the model can easily solve and problems requiring multi-step reasoning. Fourth, the task distribution is relevant to the target application; training on problem types disconnected from real usage purely for benchmark scores should be avoided.

Tasks unsuitable for direct entry into the first round of RL also need to be identified in advance. For example, open-ended strategic analysis, medical advice, legal interpretation, emotional support, and creative writing typically have no single correct answer, and the cost of errors may be high. Such tasks can incorporate model-based rewards, human preferences, or expert audits in later stages, but they should not be among the first batch of training targets when verifiers have not yet matured. For small teams, it is generally more stable to first establish a closed loop on verifiable tasks and gradually expand to semi-open tasks, rather than covering all scenarios from the start.

---

## 46.2 The Four Stages of the R1-Style Data Flywheel

The R1-style reasoning data flywheel can be broken into four stages: cold-start SFT, large-scale RL, rejection sampling, and second-round SFT. These four stages are not a linear one-time process but a closed loop that can be run repeatedly.

![Figure 46-1: The four stages of the R1-style reasoning data flywheel](../../images/part11/31_1_r1_reasoning_flywheel.png)
*Figure 46-1: The data feedback relationships among cold-start SFT, large-scale RL, rejection sampling, and second-round SFT.*

### 46.2.1 Stage One: Cold-Start SFT

The goal of cold-start SFT is not to train the model into a high-performance reasoning model, but to equip the model with readable, stable, and parseable reasoning output format. This stage typically requires a small number of high-quality Long-CoT samples covering math, code, logic problems, format adherence, and necessary general Q&A.

Cold-start data must satisfy four conditions.

**First, the reasoning process must be readable.** Samples must not contain only the final answer, nor should they consist of chaotic internal drafts; they must be able to exhibit key steps, condition references, and conclusion consolidation.

**Second, the format must be parseable.** A common approach is to place the reasoning process inside `<think>` tags and the final answer inside `<answer>` or a fixed field, so that downstream verifiers can reliably extract the answer.

**Third, tasks must be verifiable.** At least some cold-start tasks should have unambiguous answers or test scripts; otherwise it is difficult to connect them to RLVR later.

**Fourth, language and style must be consistent.** If the model needs to serve Chinese-speaking users, frequent switching to English templates within the reasoning chain should be avoided; if the model needs to produce English code explanations, Chinese prompt residues that break formatting must also be avoided.

The scale of cold-start SFT need not be large. For educational or small-team projects, a few thousand to tens of thousands of high-quality Long-CoT examples are often more valuable than hundreds of thousands of low-quality CoT examples. Community projects such as Sky-T1 also demonstrate that small-scale, carefully constructed reasoning data can significantly improve the mathematical and coding performance of 32B-class open-source models [D].

The most common pitfall in the cold-start phase is writing samples that are too "perfect." Reasoning trajectories produced after real RL training typically include probing, checking, revisiting conditions, and correction, whereas manually written cold-start samples that present only linear derivation teach the model an overly tidy explanation style. While such a style has good readability on simple problems, it may lack self-checking capability on complex ones. Therefore, cold-start data may retain moderate intermediate checks, such as "we need to verify the boundary condition here," "this expression holds only when the denominator is nonzero," or "let us first validate with a small example." These expressions are not intended to create verbosity but to provide the model with extensible reasoning behavior templates for subsequent RL.

Cold-start data must also control "answer leakage." In many synthetic datasets, the generator already knows the gold answer and reverse-engineers the reasoning process, leading to a tight coupling between steps and conclusion. Models trained on such samples may guess answers directly by pattern matching without actually reasoning. A more reliable approach is to retain check fields between the problem, the gold answer, and the reasoning process, such as `answer_source`, `verified_by`, `trace_quality`, and `leakage_risk`. If samples come from strong-model distillation, the teacher model, sampling temperature, and filtering rules should also be recorded to prevent data bias from becoming untraceable later.

After cold-start SFT, evaluation should not stop at benchmark scores. More important checkpoints include: whether the model stably outputs the target format, whether it places the final answer in a parseable position, whether it avoids excessive expansion on simple questions, whether it maintains the target language for Chinese-language tasks, and whether it avoids executing dangerous operations in coding tasks. Only when these engineering conditions are met can downstream sampling and verifiers reliably connect.

### 46.2.2 Stage Two: Large-Scale RL

In the RL stage, the model no longer merely imitates given answers; instead it performs multi-path sampling over a task pool and updates its policy based on reward signals. The key data object here is no longer a single SFT sample but a group of trajectories:

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

This structure shows that RL data engineering must record the task, samples, answers, verifier, rewards, and model version. If only the final trained model weights are saved, it is impossible to diagnose why the model improved or to localize reward hacking.

The task pool for large-scale RL typically prioritizes math, code, and structured output, because these tasks allow for verifiers. Open-ended writing, emotional support, legal consultation, and similar tasks, while possessing reasoning components, are difficult to judge as right or wrong with a single rule and are therefore not suitable as the first batch of RLVR tasks.

The difficulty distribution of the task pool directly affects the learning signal in RL. If problems are too easy, most samples will pass verification, leaving rewards with insufficient discrimination; if problems are too hard, almost all samples will fail, and the model will rarely receive positive feedback. In engineering practice, tasks can be divided into four tiers: foundational problems for stabilizing format, intermediate problems for generating the primary learning signal, hard problems for extending the upper bound, and held-out problems for evaluation only. After each training round, the team can adjust sampling ratios based on pass rates—downsampling problems with excessively high pass rates and retaining those with remaining exploration room.

Sampling configuration is also part of the data recipe. Temperature, top_p, max_tokens, stop tokens, number of candidates, and random seeds all affect the trajectory distribution. Lower temperatures yield stable answers but insufficient exploration; higher temperatures increase diversity but also increase format failures and invalid reasoning. R1-style flywheels typically require multi-path sampling, because a single sample can only observe one path for a given task, while multi-path sampling reveals the model's uncertainty on the same task. For a given problem, if only 1 out of 16 candidates is correct, the task still has learning value; if all 16 are correct, it is better suited for stability evaluation; if all 16 cannot be parsed, format or difficulty issues should be addressed first.

RL training logs must be aligned with data logs. Recording only loss, mean reward, and benchmark scores is insufficient; also required are per-task-category pass rates, format failure rates, mean trajectory lengths, repetition fragment ratios, language-mixing ratios, and verifier error ratios. This way, when the model suddenly becomes "better at reasoning but more verbose," the team can determine whether the issue stems from reward design, sampling limits, or second-round SFT data mixing.

### 46.2.3 Stage Three: Rejection Sampling

Rejection sampling is the key connector in the R1-style flywheel. The model generates multiple candidate responses to the same task, and the system retains high-quality trajectories through verifiers, voting, format checks, and safety filtering. The DeepSeek-R1 report (Guo et al. 2025) discloses the practice of constructing large-scale reasoning data through rejection sampling and mixing in non-reasoning data to preserve general capabilities [D].

Rejection sampling is not equivalent to simply selecting the sample with the highest reward. A reliable filter must examine at least five types of signals:

* Whether the final answer is correct;
* Whether the reasoning process contains obvious contradictions;
* Whether the output format is parseable;
* Whether language mixing or repetitive padding is present;
* Whether safety, copyright, or privacy filters are triggered.

If only the final answer is examined, the system may retain trajectories where "the answer was guessed correctly but the reasoning is wrong"; if only the RM score is considered, the system may amplify reward model biases. Therefore, rejection sampling is best conducted with multi-signal filtering.

Multi-signal filtering can be divided into hard filtering and soft ranking. Hard filtering removes unacceptable samples, such as those with wrong answers, unparseable formats, safety-rule violations, obvious repetition, or exceeded length limits. Soft ranking selects from among multiple qualified samples the trajectories most suitable for SFT, for example prioritizing candidates with clear reasoning steps, moderate length, consistent language, and complete condition references. This avoids an extreme case: the model generates a very long answer that is ultimately correct but whose middle contains large amounts of irrelevant content; if filtered only by passing verification, such samples would pollute second-round SFT.

Rejection sampling must also avoid sample homogenization. If only the highest-scoring trajectory is retained per problem, the training set may become overly biased toward a particular expression template. A better approach is to retain a limited number of diverse successful trajectories per task while deduplicating near-duplicate texts. Math problems can retain different solution approaches such as algebraic, geometric, and enumerative methods; coding problems can retain different implementation strategies, provided readability and complexity do not become excessive. Diversity is not about making answers flashy but about equipping the model with stronger transfer capability across similar tasks.

Failed trajectories also have value at the rejection sampling stage. Completely incorrect samples should not enter second-round SFT but can be fed into an error analysis table; near-correct samples can enter the hard-case pool; format-incorrect but conceptually sound samples can enter the format repair set; samples that were mis-judged due to verifier bugs should trigger verifier updates. A mature data flywheel does not simply delete failed samples but converts them into engineering tasks for the next round.

### 46.2.4 Stage Four: Second-Round SFT

The high-quality trajectories retained from rejection sampling can be repackaged into SFT data for second-round SFT. The role of second-round SFT is to stabilize the high-quality behaviors that RL incidentally discovered, so that the model can more easily generate similar trajectories without performing a sampling search.

Second-round SFT requires careful attention to data mixing. If only reasoning trajectories are used, the model may degrade into "producing lengthy reasoning for all questions"; if only math and code are emphasized, the model may lose general capabilities such as casual conversation, factual Q&A, and format adherence. Therefore, the design choice in the DeepSeek-R1 pipeline of supplementing with non-reasoning SFT data is important: it allows the model to reinforce reasoning capability while retaining assistant behavior.

In engineering practice, second-round SFT data should retain source labels:

* `source=rl_rejection_math`
* `source=rl_rejection_code`
* `source=non_reasoning_sft`
* `source=safety_alignment`
* `source=format_following`

If over-thinking, excessively long responses, or degraded general capability appear after training, these labels allow the team to trace back to data mixing ratios.

The data mix for second-round SFT can be stratified into "capability preservation" and "capability enhancement." Enhancement data comes from RL success trajectories, targeting improved math, code, tool-calling, or long-context reasoning; preservation data comes from general conversation, factual Q&A, short responses, format adherence, and safety refusals, targeting prevention of the model treating all problems as competition problems. For product-facing assistants, this distinction is especially important. When a user asks "how should I organize today's meeting notes," the model should not first output hundreds of words of reasoning; when a user requests a SQL query, the model should display relevant constraints and assumptions where necessary.

Evaluation after second-round SFT should also be divided into two categories. The first is capability evaluation, examining whether math, code, logic, long-context, and structured output have improved. The second is behavioral evaluation, examining whether response length, format stability, language consistency, safety boundaries, and ordinary conversational experience have degraded. Many engineering problems with reasoning models are not "inability to solve hard problems" but "overthinking simple questions." Focusing only on hard-problem benchmarks risks missing this category of product experience degradation.

On the data versioning side, second-round SFT should generate an independent manifest recording the source, filtering rules, sampling date, policy model, verifier version, and deduplication fingerprint for each training shard. This way, when an abnormal model version is discovered later, the team can quickly determine whether the cause is a certain batch of rejection-sampled data, a specific verifier version, or the proportion of certain non-reasoning data. For long-term projects, this manifest is more important than any individual data file.

---

## 46.3 Key Technologies and Design Decisions

The core of R1-paradigm data engineering lies in reward signal and verifier design. Without reliable rewards, RL merely amplifies the model's existing biases; without a traceable data structure, rejection sampling cannot be audited.

![Figure 46-2: Reward signal and verifier architecture for reasoning data](../../images/part11/31_2_reward_verifier_architecture.png)
*Figure 46-2: The relationship among rule-based rewards, model-based rewards, and human audits.*

### 46.3.1 Rule-Based Rewards and Model-Based Rewards

**Rule-based rewards** refer to rewards generated by programmatic rules. Math problems can compare final answers, coding problems can run unit tests, JSON output can be validated against a schema, and SQL can be executed and result tables compared. Their advantage is stability, low cost, and reproducibility; their limitation is restricted coverage.

**Model-based rewards** refer to rewards produced by reward models or LLM-as-Judge. They can cover open-ended Q&A, explanation quality, style, and safety boundaries, offering broader applicability, but they are also more susceptible to bias, length bias, and prompt sensitivity.

These two reward types should not replace each other but should be used in layers. For tasks with verifiable answers, rule-based rewards take priority; for open-ended tasks, model-based rewards can be used but should be paired with human spot-checks and audit sets; for high-risk domains, reliance on model review alone is insufficient.

In training systems, rewards should ideally not be stored as a single floating-point number. A single score facilitates algorithmic consumption but complicates engineering debugging. A more practical structure is to simultaneously save `reward_score`, `reward_source`, `pass_flag`, `failure_reason`, and `audit_notes`. For example, a math problem that passes symbolic comparison receives 1.0, a code problem that passes all tests receives 1.0, a format error receives 0.0, and a correct but excessively long answer might receive 0.8. Scores participate in training, but the reason fields determine how data is repaired afterward.

The greatest advantage of rule-based rewards is reproducibility, but they also incentivize the model to find rule loopholes. If the verifier only reads the final answer, the model may ignore the process; if test cases have insufficient coverage, the model may write code that overfits the tests; if the JSON schema only checks for field existence, the model may fill in meaningless content. Therefore, rule-based rewards are not "reliable as soon as written" but need unit testing, regression testing, and anomalous-sample testing, just like production code.

The advantage of model-based rewards is coverage of more complex human preferences, but they must be constrained to appropriate positions. Allowing a judge model to directly decide all rewards risks the training system inheriting the judge's tastes—for example, preferring longer explanations, more polite phrasing, or more confident tone. For reasoning tasks, a judge is better suited as a supplementary evaluator: checking whether the reasoning process is self-consistent, whether conditions were missed, or whether obvious hallucinations are present—not substituting for verifiable answers. Judge prompts must also be saved in data records; otherwise, when evaluation standards change, different data batches cannot be compared.

| Reward Type | Applicable Tasks | Advantages | Risks | Key Data Records |
| --- | --- | --- | --- | --- |
| Rule-based reward | Math, code, structured output | Stable, reproducible, low cost | Limited coverage; rules may have loopholes | Verifier version, test cases, failure reason |
| Model-based reward | Open Q&A, style, safety | Broad coverage | Preference drift, length bias, judge contamination | Judge version, scoring prompt, rationale |
| Human audit | High-risk samples, disputed samples | Reliable judgment | High cost, limited scale | Annotator consistency, review rounds |

For most teams, a prudent order is to first implement rule-based rewards, then introduce small-scale model-based rewards, and finally perform human audits on high-risk samples. This sequence captures automation benefits early while keeping uncertainty within manageable bounds. Starting with model-based review covering all tasks makes the system appear to validate quickly, but subsequently it becomes very difficult to explain why the model learned a particular preference.

### 46.3.2 Verifier Pools

The verifier pool is the infrastructure of the reasoning data flywheel. It is not a single script but a collection of versioned, testable, and rollback-capable verifiers.

**Math verifiers** typically include answer extraction, unit normalization, symbolic comparison, and tolerance judgment. For equivalent answers such as `\frac{1}{2}`, `0.5`, and `50%`, string comparison alone is insufficient. A more robust approach is to use symbolic tools such as `sympy` (Meurer et al. 2017) for expression simplification.

**Code verifiers** typically include sandboxed execution, timeout control, memory limits, unit tests, and security interception. For code tasks, rewards must not only check whether execution succeeds but also record the failure type, such as compilation error, runtime error, timeout, wrong answer, or format error.

**Format verifiers** check JSON, XML, tool call parameters, and `<think>/<answer>` tags. Many reasoning model failures are not wrong answers but answers the system cannot parse. Format rewards can reduce such engineering failures.

A maintainable verifier pool typically requires four types of interfaces. The first, `extract`, is responsible for extracting the final answer, code block, JSON field, or tool parameters from model output. The second, `normalize`, handles unit conversion, whitespace cleanup, case normalization, mathematical expression standardization, and code dependency resolution. The third, `check`, executes the actual verification logic. The fourth, `explain`, outputs failure reasons for use in logs, audits, and hard-case analysis.

Taking a math task as an example: `extract` must handle expressions such as "the answer is 1/2," "therefore x=0.5," and "the result is 50%"; `normalize` must convert fractions, decimals, percentages, and unit-bearing answers into comparable forms; `check` must distinguish between exact equality, numerical approximation, and symbolic equivalence; `explain` must return `parse_error`, `not_equivalent`, `unit_mismatch`, or `multiple_answers`. Without these subdivided reasons, the team can only observe a drop in pass rates without knowing whether to fix the problem, the parser, or the model.

Code task verifiers are more complex. They must isolate the filesystem, restrict network access, control CPU and memory, set execution timeouts, and record dependency versions. From a data engineering perspective, test cases themselves are data assets. Tests from public benchmarks may be insufficient to cover business logic; proprietary code tasks additionally require hidden tests, boundary tests, and security tests. When a model performs well on public tests but fails in production, the common cause is not an RL algorithm issue but insufficient coverage of real constraints by the verifier.

Long-CoT trajectory quality can also be observed from internal structure. Three common segment types are Reflection, Verification, and Backtrack: Reflection is used to re-examine assumptions, Verification checks intermediate conclusions or the final answer, and Backtrack retreats and changes the solution approach when a path is found to be incorrect. These three patterns should not be mechanically encoded as fixed templates but should serve as structural labels when analyzing reasoning trajectories.

![Figure 46-3: Long-CoT data sample cross-section](../../images/part11/31_3_long_cot_trace_patterns.png)
*Figure 46-3: A cross-section of a Long-CoT data sample illustrating three reasoning trajectory patterns: Reflection, Verification, and Backtrack.*

### 46.3.3 Chinese-English Mixed Reasoning Strategy in Long-CoT

When processing Long-CoT data, the open-source community frequently encounters the language mixing problem. When models frequently alternate between Chinese and English during the reasoning process, the format of the final output may become unstable. In practices related to QwQ and Kimi-1.5, addressing this problem primarily depends on language purification strategies applied to mixed reasoning data.

First, when constructing Long-CoT seeds in the cold-start phase, language consistency requirements should be specified at the prompt level. For example, Chinese tasks should require that internal derivations inside `<think>` use Chinese, and English tasks should require English throughout. Second, during the RL or rejection sampling phase, a format penalty for language mixing can be introduced: if the model frequently switches languages within continuous reasoning segments, the priority of that trajectory is lowered. Finally, in second-round SFT, trajectories with stable language and parseable answers should be preferred.

Language consistency is not intended to restrict the model's cross-lingual knowledge but to improve engineering controllability. Reasoning models may leverage cross-lingual knowledge, but the final training data should maintain stable user-facing expression.

Chinese-English mixing also affects evaluation. For many Chinese-language problems, if the model is permitted to use extensive English in `<think>`, the final answer may still be correct, but the user experience will degrade; conversely, in certain code and mathematical notation contexts, forcing Chinese for variable names or technical terms also reduces readability. Therefore, language policy should not simply be set to "Chinese only" or "English only" but defined by task type. For example, Chinese math explanations require natural Chinese reasoning; coding tasks permit English API names and error messages; paper Q&A can retain original-language terminology while providing explanations in Chinese.

For data filtering, a lightweight language detector can calculate the language ratio per trajectory segment and apply a threshold based on task language. For Chinese tasks, if the proportion of English in the main reasoning body is too high, the sample can be flagged as `language_mixing`; for code tasks, only natural-language segments need to be checked, excluding code keywords from the English proportion. This detail may seem minor but significantly affects rejection sampling quality.

### 46.3.4 Trajectory Storage and Version Control

Reasoning data relies more heavily on metadata than ordinary SFT data. The recommended minimum fields are:

| Field | Meaning |
| --- | --- |
| `task_id` | Original problem ID |
| `source` | Data source, e.g., GSM8K (Cobbe et al. 2021), MATH (Hendrycks et al. 2021), HumanEval (Chen et al. 2021), proprietary problem bank |
| `policy_model` | Model version that generated the trajectory |
| `sampling_config` | temperature, top_p, max_tokens, seed |
| `reasoning_trace` | Reasoning process |
| `final_answer` | Final answer |
| `verifier_name` | Verifier name |
| `verifier_version` | Verifier version |
| `reward` | Reward value |
| `failure_reason` | Failure reason |
| `selected_for_sft` | Whether included in second-round SFT |

Without these fields, the team will find it very difficult to determine whether a performance gain came from the model, sampling, verifier, or data filtering.

Trajectory storage must also distinguish between "raw trajectories" and "training views." Raw trajectories should be as complete as possible, retaining model outputs, sampling configurations, logs, and verification results; training views are cleaned versions used as SFT or RL inputs for actual training. The two must not be conflated. If only training views are saved, much debugging information is lost; if raw trajectories are used directly for training, errors, duplicates, and sensitive content may enter the model.

A recommended three-tier versioning scheme is as follows. The first tier is the task version, e.g., `task_pool_math_v3`, recording problems and gold answers. The second tier is the trajectory version, e.g., `rollout_qwen32b_step1200_temp0.8_v1`, recording the sampling results of a specific model under a specific configuration. The third tier is the training set version, e.g., `sft_rejection_mix_2025_02_01`, recording the data that ultimately enters training. This structure supports one task pool mapping to multiple sampling rounds, one set of sampling results mapping to multiple filtering strategies, and one filtering strategy mapping to multiple training experiments.

Deduplication is also part of trajectory storage. Two common types of duplication arise in reasoning data: problem duplication and trajectory-template duplication. Problem duplication causes evaluation leakage; trajectory-template duplication causes the model to learn fixed phrasings. The former can be addressed through problem text, gold-answer, and semantic-vector deduplication; the latter through n-gram or paragraph-level similarity detection on reasoning steps. For projects mixing open-source and proprietary data, deduplication is especially important, since the same benchmark problem may appear in different paraphrased forms.

---

## 46.4 Dataset Exposure Comparison

With the emergence of reasoning models such as DeepSeek-R1, QwQ-32B, and Kimi-1.5, traditional SFT instruction datasets have gradually given way to data formats centered on RL and Long-CoT trajectories. The information disclosed in public technical reports and dataset cards is not entirely consistent; therefore, this section marks clearly confirmed sources as `[D]`, reasonable inferences based on public descriptions as `[I]`, and pedagogical estimates as `[E]`.

### DeepSeek-R1: Cold Start, RL, and Rejection Sampling

The DeepSeek-R1 report (Guo et al. 2025) discloses two paths, R1-Zero and R1 [D]. R1-Zero demonstrates that large-scale RL can stimulate reasoning behavior even without a traditional SFT cold start, but issues with output readability and stability remain. R1 adds a small amount of cold-start Long-CoT data before RL, then forms a more stable model through RL, rejection sampling, and second-round SFT.

The DeepSeek-R1 pipeline has three key data engineering points.

**First, cold-start data emphasizes readability.** It is not a large-scale accumulation but rather a means to provide the model with a standard reasoning format and response style.

**Second, rejection sampling generates large-scale reasoning trajectories.** The 600K reasoning data samples and 200K non-reasoning data samples disclosed in the report indicate that R1 not only reinforces math and coding capabilities but also maintains general assistant behavior [D].

**Third, mixing in non-reasoning data is critical.** If the model undergoes RL only on math and coding tasks, it may degrade on ordinary conversation, factual Q&A, and format adherence.

From a reproduction perspective, the most instructive aspect of DeepSeek-R1 is not some hidden ratio but the division of responsibilities across stages. The cold-start stage ensures the model "communicates clearly," the RL stage ensures the model "tries more approaches," rejection sampling ensures the data "retains the good ones," and second-round SFT ensures behavior "reproduces reliably." These four steps each solve a distinct problem. If a team attempts to solve all problems with a single large-scale SFT dataset, they typically hit a ceiling: the model learns to mimic long reasoning formats but receives no explicit feedback for wrong answers and does not automatically produce more useful trajectories.

The data scale disclosed by DeepSeek-R1 also highlights a ratio consideration: enhancing reasoning capability cannot be decoupled from general assistant capability. 600K reasoning data samples can reinforce math, code, and complex problem-solving; 200K non-reasoning data samples are used to maintain conversational, writing, factual Q&A, and safety behaviors [D]. This principle can be transferred to small-team projects. Even when producing only 20,000 rejection-sampled trajectories, a certain proportion of short answers, format adherence, and refusal samples should be mixed in, to prevent the model from outputting long chains for every problem.

### QwQ-32B: Open-Weight Reasoning Model and RL Post-Training

The public card for QwQ-32B (Qwen Team 2025) indicates that its training includes pre-training and post-training, with post-training comprising SFT and RL [D]. Compared to DeepSeek-R1, the complete data recipe for QwQ is less extensively disclosed; therefore, this chapter does not speculate on specific data proportions.

From a data format perspective, the key insight offered by QwQ-class models is that reasoning trajectories frequently exhibit patterns such as pausing, checking, reflecting, and backtracking. These patterns may arise from the model's own sampling or from the joint effect of training data and RL objectives [I]. For data engineers, the key is not to hard-code words like "Wait" into samples but to retain complete trajectories showing the model attempting, verifying, and correcting.

Another insight that QwQ offers for open-source reproduction is that open model weights do not equate to a fully disclosed data recipe. Users can directly evaluate the model's capabilities and distill its outputs, but they cannot use this to reconstruct the complete training process. When citing such models in a project, practitioners should treat them as a usable teacher, policy, or baseline, rather than presenting undisclosed training data proportions as established facts.

When using QwQ-class models in a project, the common path is to have them generate candidate Long-CoT outputs, then filter them with proprietary verifiers. The resulting data is not "copying the QwQ recipe" but "leveraging a strong reasoning model to generate candidates, then filtering with local task constraints." This approach is suitable for the cold-start phase but cannot replace the subsequent closed loop. The reason is straightforward: the capability boundaries and biases of the teacher model will enter the student model; only by connecting a proprietary task pool and verifier will the data gradually align with the target scenario.

### Kimi-1.5: Long-Context and RL Scaling

The Kimi k1.5 (Kimi Team 2025) report emphasizes long-context scaling and improved policy optimization methods, and states that its RL framework does not rely on more complex MCTS, value functions, or PRMs [D]. This approach reminds us that reasoning data does not come only from math and code but can also come from long-context tasks.

The difficulty in long-context reasoning lies in evidence management. The model must locate evidence within long documents, multi-section materials, or multi-turn contexts, then transform that evidence into a reasoning chain. On the data side, citation positions, evidence fragments, answer bases, and failure reasons must be recorded. If only the final answer is recorded, it is impossible to determine whether the model reasoned from evidence or made a linguistic prior guess.

The Kimi k1.5 pipeline extends reasoning data engineering from "problem–answer" to "context–evidence–reasoning–answer." Long-context task verifiers are also more complex. For document Q&A, the verifier must check not only whether the answer is correct but also whether the citation comes from the provided material; for multi-turn tasks, the system must judge whether the model remembered earlier constraints; for tool-augmented tasks, the order of tool calls and intermediate results must also be checked. Without evidence fields, long-context RL can easily reward responses that "seem plausible but lack grounding."

Such tasks can adopt layered evaluation. The first layer checks format, e.g., whether the answer includes citation numbers. The second layer checks evidence, e.g., whether the cited passage genuinely supports the conclusion. The third layer checks the final answer, e.g., whether values, entities, or judgments are correct. The fourth layer checks conciseness and safety. The first two layers are what distinguish long-context tasks from ordinary math problems. They require the model not only to pursue the final answer but also to learn to bind answers to traceable evidence.

### OpenThoughts and Sky-T1: Community Reproduction Paths

OpenThoughts-114K (Guha et al. 2025) is one of the significant reasoning datasets in the open-source community. The Hugging Face dataset card indicates it is released under the Apache-2.0 license and provides data in Parquet format [D]. Its value lies in providing downloadable, inspectable, and training-ready Long-CoT samples, enabling researchers to reproduce experiments and study reasoning data recipes.

Sky-T1 (NovaSky-Berkeley 2025) demonstrates another low-cost path. Public documentation shows that Sky-T1-32B-Preview is based on Qwen2.5-32B-Instruct and was trained using a small-scale, high-quality reasoning dataset; the team simultaneously released the model, data, and training code [D]. This demonstrates that improvements in reasoning capability do not necessarily require large-scale RL; in some scenarios, well-structured Long-CoT SFT can also yield meaningful gains.

The shared value of OpenThoughts and Sky-T1 is that they transform reasoning data from "producible only by large organizations internally" into objects that the community can inspect and use for reproducible experiments. For readers of this book, these projects serve as suitable experimental starting points: download the open data, inspect field schemas and licenses, read a sample of trajectories across task types, and then connect verifiable tasks to a local verifier. After completing this workflow, the team can move from "using a dataset" to "maintaining a data production pipeline."

However, open-source data cannot be equated directly with business data. The problem types, languages, difficulty levels, and answer styles of datasets like OpenThoughts have their own distributions, and Sky-T1's training objectives may not align with enterprise scenarios. Before using them in a project, three checks are required: whether the target language is consistent, whether the target task types are covered, and whether evaluation set contamination risk is present. Only after passing these checks is open-source data suitable as cold-start data or control-experiment data.

**Table 46-1: Comparison of Dataset Exposure Across Major Reasoning Models**

| Model/Dataset | Core Driving Stages | Reasoning Trajectory Source | Open-Source/Downloadable | Distinctive Strategy | Annotation |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-R1 | Cold start + RL + Rejection sampling + SFT | Proprietary multi-path sampling and rule-based verification | Model open; training data not fully open | 600K reasoning data + 200K non-reasoning data | [D] |
| QwQ-32B | SFT + RL | Not fully disclosed by developers | Model weights open | Medium-scale reasoning model emphasizing RL post-training | [D/I] |
| Kimi k1.5 | Long-context RL | Not fully disclosed by developers | Not released as a complete dataset | Long-context scaling and policy optimization | [D] |
| OpenThoughts-114K | SFT / open-source reproduction | Community synthesis and curation | Dataset open | 114K-scale Long-CoT data | [D] |
| Sky-T1 | Small-scale Long-CoT SFT | QwQ distillation and curation | Model, data, and code open | Low-cost reproduction of reasoning capability | [D] |

**Table 46-2: Comparison of Long-CoT Data Characteristics**

| Dimension | DeepSeek-R1 | QwQ-32B | Kimi k1.5 | OpenThoughts / Sky-T1 |
| --- | --- | --- | --- | --- |
| Primary tasks | Math, code, general alignment | Math, code, reasoning | Long-context, multimodal, reasoning | Math, code, science, logic |
| Trajectory source | RL sampling and rejection sampling | SFT + RL post-training | Long-context RL | Synthesis and distillation |
| Downloadable | Not fully open | Model open; data not fully open | Not fully open | Downloadable |
| Verification signal | Primarily rule-based verification | Limited disclosure | RL rewards and evaluation | Many rule-verifiable tasks |
| Reproduction value | Understanding the industrial flywheel | Understanding open-weight reasoning models | Understanding long-context RL | Suitable for small-team experiments |

From Tables 46-1 and 46-2, it is evident that the degree of data disclosure among reasoning models is uneven. Industrial models typically disclose stages, results, and partial scale information but do not release complete data; community projects more readily release data and scripts but have limited scale, coverage, and training resources. Two extremes should be avoided in reproduction: assuming that nothing can be learned from industrial models because their complete data is unavailable, and assuming that community data alone can directly match industrial model performance because it is downloadable. A more realistic approach is to study the stage design of industrial pipelines, use community data to complete a minimal closed loop, and then gradually replace the general problem bank with proprietary tasks.

This also explains why this chapter emphasizes data engineering objects rather than merely comparing model scores. Model scores vary with benchmarks, evaluation templates, and decoding parameters; data objects determine whether the team can continuously improve. As long as a task pool, verifier pool, trajectory log, and feedback mechanism are established, models can be swapped from 7B to 32B or transitioned from SFT to RL; without these objects, even the strongest base model can only be fine-tuned once.

---

## 46.5 Case Studies

This section dissects three components: OpenThoughts-114K, the rule-based reward verification pool, and rejection sampling in practice. They correspond respectively to data sourcing, reward signals, and data feedback.

### Case A: Anatomy of the OpenThoughts-114K Dataset

OpenThoughts-114K offers a window into the structure of open-source reasoning data. It is no longer a simple `{"instruction": "...", "response": "..."}` format but organizes samples around problems, reasoning, answers, and metadata.

A reasoning sample typically needs to answer three questions:

* What is the problem;
* How the model reasons;
* Whether the final answer can be verified.

For math problems, the final answer can be verified by symbolic computation or gold-answer comparison. For coding problems, execution in a sandbox is possible. For logic problems, constraint checking or human spot-checks can serve as judgment. The value of OpenThoughts is not only its volume but also what it shows researchers about how reasoning data should be organized: saving not just the answer but also the intermediate reasoning process.

For industrial reproduction, the instructive takeaway from OpenThoughts is to start with verifiable tasks. Do not attempt to build a full RL reasoning flywheel across all business scenarios from the start; instead, first identify tasks where "answers can be programmatically verified," such as SQL generation, table computation, rule configuration, API call chains, code repair, and structured extraction. As long as verifiers can be written for these tasks, a small-scale RLVR or rejection sampling data pipeline can be established.

When ingesting data such as OpenThoughts, a four-step check can be applied. The first step is a license check, confirming whether the data permits research, commercial use, or redistribution. The second step is a field check, confirming whether each sample contains a problem, reasoning, answer, and source. The third step is a quality spot-check, randomly reading samples across different task types and recording the proportion of wrong answers, invalid reasoning, language mixing, and format anomalies. The fourth step is a contamination check, comparing problems against the planned evaluation sets for similarity to avoid overlap between training and evaluation sets.

After ingestion, the data should not all directly enter training. A more reliable approach is to establish a `curated` subset retaining only samples with complete fields, stable language, verifiable answers, and clear task sources. Samples with incomplete fields but valuable problems can enter `needs_repair`; samples with unverifiable answers or potential evaluation-set contamination should enter `excluded`. This stratification adds upfront workload but reduces training anomalies later.

An actionable data catalog is as follows:

| Subset | Entry Condition | Use |
| --- | --- | --- |
| `curated_long_cot` | Complete reasoning, verifiable answer, clear license | Cold-start SFT |
| `verifiable_tasks` | Has gold answer or test cases | RLVR / rejection sampling |
| `needs_repair` | Valuable problem but incomplete fields | Data repair |
| `excluded_eval_overlap` | High similarity to evaluation set | Training prohibited |
| `audit_samples` | High score but suspicious, or low score but near-correct | Human spot-check |

### Case B: Building a Rule-Based Reward Verification Pool

A rule-based reward verification pool can be decomposed into three layers.

**The first layer is the task layer.** The task pool should contain problems, gold answers, task type, difficulty, source, and license. For math tasks, the answer expression format should also be recorded; for code tasks, function signatures, test cases, and runtime environment should also be recorded.

**The second layer is the verification layer.** The math verifier is responsible for extracting the final answer, performing unit normalization, symbolic simplification, and tolerance comparison. The code verifier is responsible for creating sandboxes, running unit tests, limiting execution time, and recording errors. The format verifier checks JSON, XML, tool call parameters, or `<answer>` tags.

**The third layer is the log layer.** Every verification must record the verifier version, input, output, error type, and elapsed time. Verifiers without logs are difficult to integrate into a training system, because if the model behaves abnormally, the team cannot determine whether the problem lies with the model, the problem, or the verifier.

The basic workflow for a math verification pool is as follows:

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

The code verification pool requires stricter security boundaries:

```python
def verify_code(code, tests, timeout=5):
    result = run_in_sandbox(code, tests, timeout=timeout)
    return {
        "pass": result.all_tests_passed,
        "reason": result.failure_type,
        "runtime_ms": result.runtime_ms
    }
```

These two examples are structural illustrations only. Real systems must handle malicious code, infinite loops, environment dependencies, floating-point errors, multiple valid answers, and insufficient test coverage.

Before deploying a verification pool, a set of samples specifically designed to test the verifiers should be prepared, rather than immediately allowing model-sampled results to enter training. The math verifier must cover at minimum: integers, fractions, decimals, percentages, intervals, multiple solutions, unit conversions, and approximate values. The code verifier must cover at minimum: correct code, syntax errors, runtime errors, timeouts, memory overflow, missing dependencies, and malicious operations. The format verifier must cover at minimum: missing fields, wrong field types, extra fields, nested structure errors, and unparseable output.

The verification pool also needs to define a unified set of failure reason enumerations. Without a unified enumeration, different scripts may output `wrong`, `failed`, `not pass`, and `bad answer`, making subsequent statistics chaotic. It is recommended to keep failure reasons within a fixed set, such as `parse_error`, `wrong_answer`, `test_failed`, `timeout`, `format_error`, `unsafe_code`, `judge_disagree`, and `verifier_error`. `verifier_error` must be distinguished from model errors: if the verifier itself crashes, it must not be counted as the model answering incorrectly.

Verifier pool quality must also be evaluated. A simple approach is to maintain a golden validation set containing human-confirmed correct and incorrect answers; after every verifier update, a regression run is performed. If the new version causes large numbers of previously correct answers to fail, the parser or normalization has become too strict; if it allows large numbers of previously wrong answers to pass, the rules have become too loose or a loophole has appeared. Training systems may only use verifier versions that pass regression.

In production environments, verifiers must also consider cost. Symbolic simplification of math expressions can be time-consuming for complex expressions; code sandboxes consume CPU and memory; LLM judges incur additional inference costs. The data pipeline can apply cheap rules in a first-pass filter and then forward a small number of disputed samples to more expensive verifiers. This controls cost while preserving sufficient quality signal.

### Case C: Rejection Sampling in Practice

The goal of rejection sampling is to select high-quality trajectories from a large number of model-generated candidates. A minimal workflow can be divided into five steps.

**Step one**: sample multiple candidates for the same task—for example, 4 to 16 candidates per problem, depending on model cost and task difficulty.
**Step two**: parse the reasoning process and final answer of each candidate.
**Step three**: call the verifier to check the final answer.
**Step four**: apply secondary filtering to candidates that pass verification, removing those with disordered formatting, language mixing, excessive repetition, or safety risks.
**Step five**: package the retained samples into second-round SFT data.

Rejection sampling should retain failed samples. Failed samples are not garbage; they help the team analyze common model errors and can serve as future PRM or hard-case data. It is recommended to classify samples into four categories:

| Type | Meaning | Downstream Use |
| --- | --- | --- |
| pass_good_trace | Correct answer, clear reasoning | Enter second-round SFT |
| pass_bad_trace | Correct answer, chaotic process | Enter audit or PRM data |
| fail_near_miss | Nearly correct, local error | Hard-case pool |
| fail_invalid | Format error or unparseable | Format repair data |

The most common error in rejection sampling is retaining only successful answers while deleting all failed trajectories. This deprives the data flywheel of error analysis capability. The better approach: successful trajectories are used for enhancement, failed trajectories for diagnosis, and format failures for repairing output protocols.

The retention ratio in rejection sampling should be linked to task difficulty. For simple problems with already high pass rates, retaining only the most concise trajectory per problem avoids redundancy; for intermediate problems, retaining 2 to 3 trajectories with different solution approaches improves generalization; for hard problems, even without successful trajectories, several near-correct failed samples should be retained for error analysis. This produces a dataset that is neither overwhelmed by simple problems nor missing information from hard ones.

During filtering, a "not trainable but analyzable" zone should also be established. Trajectories containing private information, copyright risk, or unsafe code should not enter training even if the answer is correct; however, they can serve as test samples for safety filters. Similarly, if a trajectory shows the model exploiting a verifier loophole, this trajectory must not be treated as a positive sample but should enter `reward_hacking_cases` for use in fixing the verifier and writing regression tests.

Rejection sampling can output two data files. The first is `sft_selected.jsonl`, containing only trainable high-quality trajectories; the second is `rollout_audit.parquet`, containing all candidates, scores, failure reasons, and filtering decisions. The former is consumed by the training system; the latter is used by the data engineering and evaluation systems for auditing. Many projects save only the former, which saves short-term storage but sacrifices long-term explainability.

A complete filtering decision can be written with the following fields:

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

This explicit decision record allows the team to reuse old trajectories when adjusting thresholds later. For example, the first pass selects only samples with `quality_score >= 0.85`; if more data is needed in a subsequent pass, samples scoring 0.75 to 0.85 can be re-selected without resampling all tasks.

---

## 46.6 Common Failure Patterns

**First, reward hacking.**
If the verifier has loopholes, the model will learn to exploit them. For example, if a math answer extractor only reads the last number, the model may append the correct number at the end while the preceding reasoning is entirely wrong. The solution is to simultaneously check the answer, the process, the format, and anomalous patterns.

**Second, length explosion.**
RL may cause the model to favor generating longer reasoning chains. Length is not equivalent to quality; excessively long reasoning increases inference cost and may reduce readability. Training data should include length limits, redundancy detection, and concise trajectories.

**Third, language mixing.**
Chinese-English mixing degrades user-facing output experience and affects parsers. The cold-start phase should standardize language requirements, and the rejection sampling phase should filter trajectories with frequent language switching.

**Fourth, reasoning that appears plausible but is unverifiable.**
Many Long-CoT samples are written to look like reasoning, but no step can be programmatically checked. Such samples are suitable for SFT but not for RLVR. Before entering the RL stage, the task pool must be stratified by "degree of verifiability."

**Fifth, over-reliance on a single benchmark.**
If all tasks come from a small set of math or code benchmarks, the model may only learn specific problem types. The task pool should cover different difficulty levels, different sources, and different problem formulations, and a held-out evaluation zone should be maintained.

**Sixth, second-round SFT causing overthinking.**
If second-round SFT consists entirely of long reasoning trajectories, the model may output lengthy `<think>` blocks even for simple questions. The solution is to mix in non-reasoning conversations, short answers, and format-adherence data.

**Seventh, verifiers lacking version management.**
The same batch of samples may receive different rewards under different verifier versions. Without recording versions, training results cannot be audited. Every verifier update should be accompanied by a regression test run.

**Eighth, training and evaluation set leakage.**
Reasoning data often comes from public problem banks, community synthetic data, and model distillation, and evaluation sets often come from the same sources. Without similarity-based deduplication before training, model scores may be inflated. Leakage is not limited to identical problems; it may also manifest as paraphrased problem statements, identical numerical structures, or variants of the same code problem. The solution is multi-level deduplication of problems, answers, and key constraints, along with maintaining a held-out evaluation set.

**Ninth, treating teacher model outputs as ground truth.**
When using a strong model to generate Long-CoT, it is tempting to assume the outputs are inherently high-quality. In reality, teacher models can also produce incorrect reasoning, overly long explanations, answer leakage, and hallucinated citations. All teacher-generated data should undergo verifier checks or human spot-checks; quality control must not be skipped on the grounds that the teacher is strong.

**Tenth, neglecting the organization of negative samples.**
Many teams organize only positive samples, making it impossible to later train a process reward model or analyze model failure boundaries. Negative samples need not enter SFT but should be saved by error type. A structured negative sample library helps the team identify task-type weaknesses, verifier loopholes, and sampling configuration issues.

**Eleventh, out-of-control sampling costs.**
Multi-path sampling rapidly amplifies token costs. Sampling 32 candidates per problem, each generating thousands of tokens, can produce very high costs even from a small problem bank. Control measures include tiered sampling by difficulty, reducing the number of candidates for easy problems, early stopping for problems that repeatedly fail, truncating excessively long trajectories, and periodically removing low-value tasks.

**Twelfth, reward objectives misaligned with product objectives.**
A model performing better on math benchmarks does not mean it is more useful in a product. If the product requires concise, citable, and executable responses but training rewards only consider the final answer, the model may learn verbose output. Every category of reward should map to a product objective: correctness, executability, readability, safety, and cost.

These issues collectively demonstrate that reasoning data engineering does not end with connecting an RL algorithm to a model; it requires continuous maintenance of the contract among tasks, verifiers, sampling, and training data. Any time the definition of one component changes, all subsequent data must be reinterpreted.

---

## 46.7 Cost, Risk, and Applicability Boundaries

The cost of an RL reasoning data flywheel primarily comes from three sources: multi-path sampling, verifier execution, and second-round training. Sampling cost grows linearly with the number of candidates; code verification also consumes sandbox resources; second-round SFT requires repackaging and training data.

For small teams, starting with full RL training is not recommended. A more realistic path is:

* First, use open-source Long-CoT data for cold-start SFT;
* Then, build a small-scale verifier pool;
* Next, perform rejection sampling to generate 10,000 to 30,000 high-quality trajectories;
* Finally, validate the gains using LoRA or short-step SFT.

The focus of this path is to first validate the data production closed loop, then consider scaling the model and task volume.

| Stage | Cost Source | Small-Team Downgrade Option |
| --- | --- | --- |
| Cold-start SFT | Data cleaning, training | Use an open-source Long-CoT subset |
| Multi-path sampling | vLLM inference tokens | Reduce per-problem candidates from 16 to 4 |
| Verifier | Sandbox, symbolic computation | Start with math, then add code |
| Rejection sampling | Filtering and packaging | Retain only pass_good_trace |
| Second-round SFT | Training GPU | LoRA or short-step smoke test |

Applicability boundaries also need to be specified clearly. The R1-style flywheel is best suited for math, code, structured output, tool calling, and some long-context tasks; it is not suitable for all open-ended conversational tasks. For safety, medical, financial, and legal tasks, rule-based verification can only cover a portion of facts or format requirements and cannot replace expert review.

When estimating cost, do not only estimate training GPU hours. The primary expense in a reasoning data flywheel often occurs before and after training: constructing the task pool, generating candidates, running verifiers, storing trajectories, human audits, and repeated evaluation. Especially in the multi-path sampling stage, token costs grow rapidly with the number of candidates, maximum length, and problem scale. Without budget controls, the team may spend most of its resources in the first sampling round, only to find that verifier quality is insufficient and the data is unusable.

Therefore, small teams can adopt a staged budget approach. Stage one covers only 100 to 500 tasks, with the goal of validating whether the data structure and verifiers are functional. Stage two expands to 3,000 to 10,000 tasks, with the goal of producing trainable rejection-sampled data. Stage three then considers larger-scale RL or second-round SFT. Each stage should have a stopping condition—for example, if the format pass rate falls below a threshold, fix the prompt first; if the verifier error rate is too high, fix the verifier first, rather than continuing to scale sampling.

Risk boundaries fall into four main categories. The first is safety risk: the model may generate harmful content in code, tool calls, or professional advice. The second is copyright and license risk: not all open-source reasoning data is suitable for commercial use or redistribution. The third is privacy risk: proprietary business logs must be desensitized before entering the task pool, and model-generated reasoning traces must also be subject to desensitization and access controls. The fourth is evaluation risk: training set contamination makes the model appear stronger but the improvement cannot be reproduced in production.

Regarding applicability boundaries, the R1-style flywheel is best suited for tasks that are "verifiable, iterable, and whose errors can be analyzed." SQL generation can execute queries and compare result tables, code repair can run tests, table reasoning can verify values, and tool calling can check parameters and return status. Once verifiers are established for these tasks, a stable closed loop forms. Conversely, value judgment, strategic consulting, and professional advisory tasks require more complex evaluation frameworks and should typically serve first as evaluation or human-audit objects rather than as primary RL training tasks.

From a project management perspective, a minimum viable reasoning data flywheel can be implemented as follows: first select a narrow task domain, such as short math problems, SQL generation, or structured extraction; prepare a few hundred cold-start samples; implement a verifier; run 4-path sampling with the current model; retain trajectories and verification logs; filter a first-version SFT dataset; train a small model or LoRA; and finally compare gains against a held-out evaluation set. Only after the closed loop is validated does scaling make sense.

---

## Chapter Summary

This chapter decomposed the reasoning model paradigm represented by R1/QwQ/Kimi-1.5 from a data engineering perspective. Unlike traditional SFT, the core of the reasoning data flywheel is not collecting longer CoT but organizing the closed loop of task pool, sampled trajectories, verifiers, rewards, rejection sampling, and second-round SFT.

Three key conclusions emerge from this chapter.

**First**, cold-start SFT addresses readability and format stability, RL addresses exploration, rejection sampling addresses high-quality trajectory distillation, and second-round SFT addresses behavior stabilization.

**Second**, rule-based rewards are the priority entry point for reasoning data engineering. As long as tasks can be programmatically verified, reasoning capability training can advance from "human judgment of quality" to "system verification of correctness."

**Third**, the open-source community's OpenThoughts and Sky-T1 demonstrate that small teams can also build a functional reasoning data flywheel prototype through high-quality Long-CoT data, lightweight verifiers, and rejection sampling.

Mapping this chapter to a project implementation, the minimum deliverable is not a "reasoning model" but a set of data assets: a task pool, a verifier pool, a trajectory archive, a rejection-sampled training set, a failure sample library, and an evaluation report. The model is merely one consumption result of these assets. As long as these assets are continuously updated, the team can swap between different base models, different training algorithms, and different deployment budgets.

The next chapter turns to data engineering for multimodal understanding models. Unlike text-based reasoning, multimodal models must also handle images, pages, video frames, OCR, spatial positions, and multi-image relationships. Chapter 47 will discuss how these visual inputs enter pre-training, multi-task alignment, and multimodal SFT data recipes.

## References

Guo D, Yang D, Zhang H, Song J, Wang P, Zhu Q, Xu R, Zhang R, Ma S, Bi X, others (2025) DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2501.12948.

Team Kimi, Du A, Gao B, Xing B, Jiang C, Chen C, Li C, Xiao C, Du C, Liao C, others (2025) Kimi k1.5: Scaling Reinforcement Learning with LLMs. arXiv preprint arXiv:2501.12599.

Touvron H, Martin L, Stone K, Albert P, Almahairi A, Babaei Y, Bashlykov N, Batra S, Bhargava P, Bhosale S, others (2023) Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv preprint arXiv:2307.09288.

Cobbe K, Kosaraju V, Bavarian M, Chen M, Jun H, Kaiser L, Plappert M, Tworek J, Hilton J, Nakano R, others (2021) Training Verifiers to Solve Math Word Problems. arXiv preprint arXiv:2110.14168.

Chen M, Tworek J, Jun H, Yuan Q, Pinto H P O, Kaplan J, Edwards H, Burda Y, Joseph N, Brockman G, others (2021) Evaluating Large Language Models Trained on Code. arXiv preprint arXiv:2107.03374.

Guha E, Marten R, Keh S, Raoof N, Smyrnis G, Bansal H, Nezhurina M, Mercat J, Vu T, Sprague Z, others (2025) OpenThoughts: Data Recipes for Reasoning Models. arXiv preprint arXiv:2506.04178.

Zhou C, Liu P, Xu P, Iyer S, Sun J, Mao Y, Ma X, Efrat A, Yu P, Yu L, Zhang S, Ghosh G, Lewis M, Zettlemoyer L, Levy O (2023) LIMA: Less Is More for Alignment. Advances in Neural Information Processing Systems, 36, 55006–55021.

Zelikman E, Wu Y, Mu J, Goodman N (2022) STaR: Bootstrapping Reasoning with Reasoning. Advances in Neural Information Processing Systems, 35, 15476–15488.

Madaan A, Tandon N, Gupta P, Hallinan S, Gao L, Wiegreffe S, Alon U, Dziri N, Prabhumoye S, Yang Y, Gupta S, Majumder B P, Hermann K, Welleck S, Yazdanbakhsh A, Clark P (2023) Self-Refine: Iterative Refinement with Self-Feedback. Advances in Neural Information Processing Systems, 36, 46534–46594.

Lightman H, Kosaraju V, Burda Y, Edwards H, Baker B, Lee T, Leike J, Schulman J, Sutskever I, Cobbe K (2024) Let's Verify Step by Step. International Conference on Learning Representations.

Zheng L, Chiang W-L, Sheng Y, Zhuang S, Wu Z, Zhuang Y, Lin Z, Li Z, Li D, Xing E, Zhang H, Gonzalez J, Stoica I (2023) Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. Advances in Neural Information Processing Systems, 36, 46595–46623.

Gao L, Schulman J, Hilton J (2023) Scaling Laws for Reward Model Overoptimization. Proceedings of the 40th International Conference on Machine Learning, pp 10835–10866.

Hosseini A, Yuan X, Malkin N, Courville A, Sordoni A, Agarwal R (2024) V-STaR: Training Verifiers for Self-Taught Reasoners. arXiv preprint arXiv:2402.06457.

Shi F, Suzgun M, Freitag M, Wang X, Srivats S, Vosoughi S, Chung H W, Tay Y, Ruder S, Zhou D, others (2022) Language Models Are Multilingual Chain-of-Thought Reasoners. arXiv preprint arXiv:2210.03057.

Jaech A, Kalai A, Lerer A, Richardson A, El-Kishky A, Low A, Helyar A, Madry A, Beutel A, Carney A, others (2024) OpenAI o1 System Card. arXiv preprint arXiv:2412.16720.

Ott S, Hebenstreit K, Liévin V, others (2023) ThoughtSource: A Central Hub for Large Language Model Reasoning Data. Scientific Data, 10(1), 528.

Hsieh C-Y, Li C-L, Yeh C-K, Nakhost H, Fujii Y, Ratner A, Krishna R, Lee C-Y, Pfister T (2023) Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes. Findings of the Association for Computational Linguistics: ACL 2023, pp 8003–8017.

Patil S G, Zhang T, Wang X, Gonzalez J E (2024) Gorilla: Large Language Model Connected with Massive APIs. Advances in Neural Information Processing Systems, 38.
