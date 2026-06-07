# Chapter 45: LLM Post-training Data Engineering in Practice: SFT and Preference Alignment

## 45.0 Opening Scenario: Why Alpaca Does Not Produce a GPT-4-Style Assistant

From 2023 to early 2024, the open-source community explored instruction tuning intensively. Consider a common team scenario: an application team wants to build a vertical-domain assistant on top of an open base model. It collects about 200K Alpaca, Dolly, and Self-Instruct-style instruction samples and runs supervised fine-tuning (SFT).

After training, surface metrics look normal. Loss decreases steadily, and the model changes from a text completer into something willing to answer questions. But internal red-team tests and real business gray release reveal several gaps.

1. **Weak complex instruction following.** When a prompt contains more than three constraints, such as outputting JSON, staying under 100 words, and using three paragraphs, the model often drops one constraint.
2. **Unstable refusal boundaries.** For misleading, harmful, or clearly unsupported questions such as dangerous-goods instructions or fake medical diagnosis, the model may answer unreliably instead of identifying risk and refusing.
3. **Loose long-answer structure.** It struggles to write layered, logically progressive long responses, and multi-turn consistency is unstable.
4. **Weak tool-use awareness.** When an answer requires an external API, the model tends to fabricate rather than output a standard tool-call JSON.

No amount of cleaning of the 200K samples makes the model feel close to Llama-3.3-Instruct or Qwen2.5-Instruct. The lesson is that post-training is not simply adding more instruction samples. It is a staged process of data organization and behavior calibration.

SFT establishes behavior templates: it teaches the model what format to use, but does not by itself teach which acceptable answer is better. Preference alignment is an independent data production and review mechanism, not a small supplement after SFT. Industrial online optimization must put user feedback, rejection sampling, reward models, and contamination checks into one data chain. The value of open models comes not only from weights, but from post-training recipe information disclosed in reports, dataset cards, and training flows.

## 45.1 The Three Layers of Post-training Data: SFT, Preference Alignment, and Online Optimization

To make a base model behave reliably in interaction, post-training data engineering usually builds a layered pipeline. The layers differ in data shape, optimization target, and engineering difficulty.

**Layer 1: SFT data.** SFT is fundamentally about formatting. It turns a base model from an unconscious probability predictor into a rule-following assistant. SFT data defines answer formats, task boundaries, assistant tone, basic safety behavior, and multi-turn alternation. In modern practice, SFT data commonly ranges from \(10^5\) to \(10^6\) samples. Quality, including diversity, correctness, and strict formatting, matters more than quantity. One million low-quality samples are often worse than one hundred thousand carefully edited ones.

**Layer 2: preference alignment data.** If SFT teaches how to answer, preference alignment teaches which of two valid answers is better. It shapes the model's reward surface. Preference data can support several training paradigms: reward-model training for RLHF, or direct preference methods such as DPO, IPO, KTO, GRPO, and RLVR. Scale varies widely from \(10^5\) to \(10^7\) preference pairs, depending on whether construction uses large automatic generation, multi-sample candidate groups, and online feedback.

**Layer 3: online continuous optimization data.** Releasing the model is not the end of post-training; it is the beginning. Likes, dislikes, refusal logs, hard cases, human-reviewed red-line samples, A/B results, and safety incidents enter the continuous optimization chain through streaming or batch processing. Without this layer, the model remains at a one-time static release and eventually falls behind changing user distributions.

SFT cannot replace preference data because SFT uses maximum likelihood estimation. During training, the model is encouraged to imitate target tokens, but it does not see alternative responses and their relative quality. For a complex question, several answers may be valid but differ in usefulness, safety, factuality, or clarity. SFT struggles to teach generation-time selection among them. Preference data introduces chosen / rejected contrast and suppresses syntactically correct but logically or value-wise inferior regions of the response space. Post-training data engineering must therefore manage both the behavior shape produced by SFT and the feedback loop produced by preferences and online optimization.

![Figure 45-1: Three-stage LLM post-training pipeline](../../images/part11/30_1_posttrain_three_stage_pipeline.png)

*Figure 45-1: Data flow among SFT, preference alignment, and online continuous optimization.*

From an implementation perspective, the three layers imply three different asset-management styles. SFT data is a behavior-template library. It should be stable, clean, broad, and simple, usually built around fields such as `messages`, `instruction`, `input`, `output`, `source`, `license`, and `quality_score`. Once the schema is stable, SFT data can enter different training frameworks and be compared across versions.

Preference data is a behavior-discrimination library. It should not store only chosen and rejected texts. It should also store candidate generator, sampling temperature, candidate count, annotator or judge model, rationale, conflict-review result, and labels for safety, factuality, code, math, or tool use. Without this metadata, a later DPO regression is hard to diagnose. Many teams first store only two answers and a `label`, then realize during preference drift that they lack the context to explain the drift.

Online optimization data is an event-log library. It comes from real users, red teams, evaluation platforms, customer feedback, and production monitoring, so it naturally has time, scenario, version, and permission attributes. It must be tied to model version, prompt version, system policy, user authorization, anonymization state, and evaluation-set isolation. Otherwise, valuable real feedback may be unusable for privacy, contamination, or traceability reasons. Mature post-training data engineering is measured not only by sample generation, but by whether each sample can explain where it came from, why it is trustworthy, which stage it can enter, and how to roll it back.

A robust post-training repository should maintain at least four manifests. A source manifest records dataset name, license, collection time, filters, and owner. A sample-processing manifest records deduplication, filtering, rewriting, translation, scoring, and human review. A training-consumption manifest records which model, training stage, and experiment configuration used a data version. An evaluation-isolation manifest records which benchmarks, red-team sets, and acceptance sets must never enter training. These manifests look tedious, but they decide whether a post-training pipeline is an operational system rather than a one-off experiment.

## 45.2 Comparing Open Post-training Transparency

Before building an internal post-training pipeline, we should compare public routes taken by mainstream open models. This section uses Tulu-3, Llama-3, Qwen2.5, and Nemotron-4 as four representative routes. The goal is not to rank leaderboard scores, but to build an engineering method for reading public information and judging its practical value.

Data tags follow the same convention:

* **[D]**: explicitly disclosed in a report, paper, or dataset card.
* **[I]**: reasonably inferred from disclosed process, split ratios, or context.
* **[E]**: estimated for teaching or engineering discussion; not an official number.

**Table 45-1: Post-training transparency and scale of mainstream open models**

| Model / project | Post-training stages | Openness | SFT scale | Preference / reward data scale | Key data sources | Reproduction value |
| --- | --- | --- | --- | --- | --- | --- |
| **Tulu-3** | SFT / DPO / RLVR | High | 939K [D] | DPO mixture scale should be checked item by item [D] | SFT-Mix, DPO mix, RLVR verifier | Fully open recipe baseline, suitable for reproduction and transfer |
| **Llama-3** | SFT / reward model / RLHF | Medium | Not fully open; process and partial statistics disclosed [D/I] | Multi-round preference annotation and RM/DPO data, total scale must be checked against reports [D/I] | Human annotation, multi-round RM, rejection sampling | Understand industrial multi-round iteration and heavy human involvement |
| **Qwen2.5** | SFT / preference / large-scale synthetic data | Medium | Partially disclosed; exact totals depend on reports [D/I] | Partially disclosed, scale unknown [D/I] | Instruction synthesis, multilingual and multitask data; Magpie as seed-free synthesis comparison | Observe Chinese, multilingual post-training and large-scale synthesis |
| **Nemotron-4** | Reward model / HelpSteer2 | High | Not the focus here | HelpSteer2 at roughly 10K prompt / pair scale; verify by dataset card [D] | Attribute-level preference labels, Daring-Anteater SFT data | Important template for reward-model data design |

Tulu-3 is one of the best reproduction baselines in this chapter. It releases weights, data mixtures, training code, and evaluation methods, making it easier to convert a paper recipe into an inspectable engineering process. Llama-3 (Dubey et al. 2024) represents a capital-intensive industrial path. It discloses mechanisms such as multi-round post-training, preference annotation, reward-model retraining, and rejection sampling, but many data details are not fully open. Qwen2.5 is valuable for Chinese, multilingual, multitask, and synthetic-data routes; Magpie (Xu et al. 2024) should be discussed as a comparable seed-free synthesis method, not claimed as an official Qwen recipe without direct evidence. Nemotron-4 and HelpSteer2 (Wang et al. 2024b) are important because they annotate preference along dimensions such as helpfulness, correctness, coherence, complexity, and verbosity.

## 45.3 Three SFT Synthesis Routes: Self-Instruct, Evol-Instruct, and Magpie

Once SFT is understood as the first layer of post-training, the next question is how to obtain high-quality instructions. Manual writing is expensive and cannot cover the long tail. Instruction synthesis is therefore mainstream. This section compares three routes.

### 45.3.1 Self-Instruct: Expanding from Seed Tasks

Self-Instruct (Wang et al. 2023) relies on a small set of human-written seed tasks, often several hundred examples. A strong model generalizes from those seeds to generate new `instruction`, `input`, and `output` triples. Its advantage is fast coverage expansion in early projects, especially for teaching a base model how to start a dialogue. Its risk is template dependence: generated tasks can become homogeneous, and difficulty often concentrates in common ranges rather than edge cases.

### 45.3.2 Evol-Instruct: Increasing Complexity with Evolution Rules

WizardLM (Xu et al. 2024) introduced Evol-Instruct to address shallow task difficulty. Simple instructions are made harder through evolution rules: add constraints, deepen reasoning, introduce multiple branches, require multi-step answers, and so on. This generates strong complex instruction-following data and forces the model to learn deeper obedience to logic, not surface response patterns. The risk is that higher difficulty is not the same as higher quality. Repeated evolution can cause intent drift, contradictory constraints, or meaningless complexity. Difficulty calibration and answer verification are the key QC tasks.

### 45.3.3 Magpie: Seed-Free Instruction and Response Generation

Magpie weakens the dependence on human seeds. It uses the conversational prior of an already aligned instruct model, such as Llama-3-Instruct, to generate user instructions and corresponding answers. A very short pre-query prompt, or even an `[INST]` marker, can trigger natural user-like questions. This reduces human intervention and generates a distribution closer to real long-tail user queries. When discussing open post-training, Magpie is useful as a comparison for large-scale synthetic instruction diversity; it should not be tied to a model's official recipe without evidence. Its main risk is amplification of model bias, hallucination, and unsafe content, so distribution and safety filters are mandatory.

**Table 45-2: Engineering comparison of three SFT synthesis routes**

| Route | Seed dependence | Generation method | Suitable tasks | Main risks | QC focus | Representative material | Role in this chapter |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Self-Instruct** | Medium | Seed-inspired expansion | Broad general instruction coverage | Templates and homogeneity | ROUGE deduplication, diversity evaluation, answerability checks | Self-Instruct paper | Basic synthesis route |
| **Evol-Instruct** | Medium | Rule-based complexity evolution | Complex instruction following, multi-step reasoning | Logical conflict, intent drift | Difficulty calibration, code-level answer validation | WizardLM project | Complexity route |
| **Magpie** | Low | Prior-driven seed-free self-generation | Large-scale natural dialogue instructions | Bias amplification, hallucination, unsafe content | Distribution-diversity filtering, strict safety filtering | Magpie paper | New route for open post-training |

![Figure 45-2: Self-Instruct, Evol-Instruct, and Magpie pipelines](../../images/part11/30_2_sft_synthesis_pipelines.png)

*Figure 45-2: Data entry points, generation methods, and QC positions for three SFT synthesis routes.*

No synthesis route should feed generated data directly into training. A safer pipeline uses four gates. The format gate checks roles, missing fields, JSON or ChatML parsing, truncation, and mojibake. The semantic gate checks whether the instruction is answerable, whether the answer covers the core question, and whether there is mismatch or skipped reasoning. The distribution gate checks whether task type, language, length, domain, difficulty, and safety category are over-concentrated. The leakage gate checks near-duplicates against evaluation sets, benchmark prompts, public answers, and internal holdout sets.

These gates work best as automatic filtering plus human sampling. Automatic checks handle format errors, duplicates, length outliers, low-quality templates, sensitive terms, and obvious safety issues. Human sampling judges naturalness, usefulness, and whether complex tasks preserve intent. In Evol-Instruct, scripts often cannot tell whether a harder-looking task is still the same task. Without human sampling or a strong verifier, the model learns many complex but invalid patterns.

SFT data also needs stratified mixing. At minimum, split it into general QA, knowledge explanation, complex instruction following, code and tools, math and reasoning, and safety / refusal. Each slice should record count, average length, source, filtering rate, and sampling pass rate. For reproducing open recipes, the most valuable artifact is not a copied ratio, but a change log of mixture ratios. When code ability, refusal behavior, or Chinese expression changes, the team can trace which slice moved.

Good SFT data is not always suitable for every round. The first SFT round should use clear, stable, broad samples to establish base assistant behavior. Later incremental SFT should add hard cases, domain tasks, tool use, and safety-boundary repairs. Pouring all data into one run can drown high-value boundary samples. A better design is curriculum-style SFT versions such as `sft_base_mix`, `sft_complex_mix`, `sft_safety_patch`, and `sft_domain_patch`, each with its own manifest and evaluation report.

## 45.4 Preference Data Engineering: From RLHF to DPO, GRPO, and RLVR

After SFT data is obtained, preference data determines the model's final personality. Its shape changes with the training paradigm.

### 45.4.1 RLHF: Preference Pairs and Reward Models

Reinforcement learning from human feedback (RLHF) (Ouyang et al. 2022) is the classic alignment route. Its core data shape is multi-candidate ranking or preference pairs.

* **Data shape:** prompt + chosen + rejected, or relative ranking scores for \(N\) candidates under the same prompt.
* **Engineering focus:** generate discriminative candidate answers efficiently; maintain annotator consistency on value judgments; ensure reward-model data covers real production prompt distributions.

### 45.4.2 DPO: Data Requirements for Direct Preference Optimization

Direct Preference Optimization (DPO) (Rafailov et al. 2023) rewrites the RL objective as a binary cross-entropy loss and avoids explicitly training a reward model.

* **Data shape:** strict prompt + chosen + rejected triples.
* **Engineering focus:** preference-pair quality. DPO is sensitive to the signal between chosen and rejected. If rejected samples are too poor, such as garbled text, DPO learns little. If the contrast is mostly surface style, such as a harsh but correct rejected answer versus a gentle but factuality-wrong chosen answer, DPO can mislead the model.

### 45.4.3 GRPO: In-Group Relative Comparison

GRPO (Shao et al. 2024) is often used for long reasoning. It does not rely on a global absolute reward baseline; it emphasizes relative quality within the same prompt's candidate group.

* **Data shape:** prompt + candidate group + relative reward signals. Candidate groups often contain 4 to 8 answers, depending on sampling cost and task difficulty.
* **Engineering focus:** preserve group structure, sampling parameters such as temperature, verifier signals, and group metadata. The pipeline becomes efficient batched multi-sample generation.

### 45.4.4 RLVR: Verifiable Rewards as a New Interface

Reinforcement learning with verifiable rewards (RLVR) moves preference from subjective "humans prefer this" toward objective "the result is computable and verifiable."

* **Data shape:** task + answer + verifier signal.
* **Suitable tasks:** math, code generation, format following, and tool calling. Math can compare final answers, code can run unit tests, structured output can use JSON/XML schemas or regex checks, and tool calls can check API status.
* **Engineering difficulty:** build verifiers with broad coverage and few loopholes. This chapter explains the data shape; Chapter 46 develops the R1-style reasoning flywheel.

**Table 45-3: Data requirements of preference paradigms**

| Paradigm | Core data shape | Reward source | Suitable tasks | Data engineering difficulty | Interface to Chapter 46 |
| --- | --- | --- | --- | --- | --- |
| **RLHF** | prompt + multiple candidates + human / AI preference | Human annotation / trained RM | General assistant behavior, value alignment | High annotation cost, cross-annotator consistency | Industrial background for multi-round iteration |
| **DPO** | prompt + chosen + rejected | Offline preference pairs | Chat QA, refusal boundaries, style tuning | Negative-sample quality, overconfidence control | Precursor to second-round SFT or reasoning alignment |
| **GRPO** | prompt + candidate group + relative reward | In-group advantage | Complex reasoning, code generation, diverse sampling | Sampling efficiency, group metadata storage | Core interface for R1-style training |
| **RLVR** | task + answer + verifier signal | Hard rules / unit tests / math verifier | Math, code pass rate, strict structured output | Verifier robustness and coverage | Main object of the Chapter 46 reasoning flywheel |

The first engineering principle is to preserve candidate groups rather than only win-loss results. Multiple candidates under one prompt reveal the current policy's answer space. If only chosen and rejected are saved, the team loses whether all candidates were poor, only one was obviously wrong, or several answers were acceptable in different styles. This information determines whether to train an RM, run DPO, or return to SFT templates.

The second principle is to record preference reasons. Chosen may win because it is factual, clear, safer, or simply longer and more pleasing to a judge model. Without a rationale field, "true preference" cannot be distinguished from scoring bias. Preference data should include metadata such as `preference_reason`, `error_type`, `safety_label`, `factuality_label`, and `style_label`. Human-reviewed data should also include `annotator_id_hash`, `review_round`, and `disagreement`.

The third principle is to split preference data into training, calibration, and audit sets. The training set optimizes the model. The calibration set monitors RM or DPO overfitting. The audit set detects preference bias and should remain stable. It can include samples such as short-correct versus long-vague answers, safety refusal versus over-refusal, and factually correct but plain answers versus polished but wrong answers. Running this audit set after each training round catches reward hacking and sycophancy early.

The fourth principle is to manage verifiable tasks separately. Math, code, structured output, and tool calls can use verifiers instead of relying only on subjective preference. For these tasks, store `verifier_name`, `verifier_version`, `test_case_hash`, `pass_rate`, and `failure_reason`. These fields support both RLVR and rejection sampling. They are the reusable interface for the reasoning data flywheel in Chapter 46.

## 45.5 Case A: Tulu-3 as an Open Post-training Reference

Tulu-3 is the main reproducible reference in this chapter. It covers SFT, preference optimization, and verifiable rewards, and it exposes data, training code, and evaluation methods at a relatively high level.

### 45.5.1 Why Tulu-3 Is a Useful Baseline

Some "open" projects release only final weights; others release partial fine-tuning datasets. Tulu-3's advantage is that it discusses SFT-Mix, DPO Mix, and RLVR verification logic in one public chain. It provides an end-to-end baseline for understanding what data each stage needs and how one stage feeds the next.

### 45.5.2 SFT-Mix: Behavior Templates

Tulu-3 (Lambert et al. 2025) does not chase unbounded SFT scale. Its SFT-Mix has about 939K samples [D], according to public training-data disclosures; use should match the corresponding dataset card or paper table. The mixture reflects manual control: base conversation, multitask following, multi-turn interaction, API tool use, code generation, math reasoning, and core safety tasks are mixed according to stage goals.

The lesson is not to chase row count. Blindly adding millions of simple code-completion samples can make the model forget how to chat naturally. Tulu-3 shows that downsampling and balancing high-quality sources can quickly solidify broad assistant behavior in a limited number of fine-tuning steps.

### 45.5.3 DPO: Preference Polishing

After SFT establishes base behavior, Tulu-3 introduces a DPO mix for preference polishing. The exact scale and composition should be checked against papers and dataset cards to avoid mixing statistics from different stages or splits. DPO data should include external datasets and typical behavior failures after SFT. Chosen and rejected should not be random. A dangerous pattern is selecting an answer that is more polite but less factual as chosen. Uncalibrated LLM-as-a-Judge systems can introduce length bias and sycophancy; high-quality DPO pairs need clear factual or safety differences.

### 45.5.4 RLVR: Verifiable Task Layer

Tulu-3 finally introduces RLVR. For tasks with objective correctness, human labeling is expensive and error-prone, so Tulu-3 uses rule-based verifiers. RLVR is not suitable for every task. It mainly applies where a clear terminal state can be extracted: final numeric answers in math, AST parsing and unit-test execution in code, and similar checks. This section sets up the interface; Chapter 46 explains how rule-based rewards support an R1-style reasoning data flywheel.

![Figure 45-3: Tulu-3 three-stage post-training flow](../../images/part11/30_3_tulu3_posttrain_flow.png)

*Figure 45-3: Data flow from SFT-Mix to DPO mix and RLVR in Tulu-3.*

Migrating Tulu-3 to an internal project can be broken into four steps. First, classify public recipe sources into directly reusable, structurally useful, and domain-replacement-needed. Second, preserve the SFT, DPO, RLVR stage order while adjusting task ratios for domain risk. Third, write each stage's inputs, outputs, filters, and evaluation sets into a manifest. Fourth, rerun contamination and safety-boundary checks after migration because a public recipe that passes general evaluation is not automatically safe for an industry scenario.

More concretely, migration should start from control points, not from a list of datasets to download. Control point one is SFT-Mix source registration: why each source enters the mixture, what ability it supports, whether its license permits use, and whether it conflicts with evaluation. Control point two is DPO preference boundaries: do chosen/rejected differences stably reflect factuality, safety, concision, and instruction following rather than length or tone? Control point three is RLVR verifier versioning: rules, unit tests, answer extractors, and parsers need versions. Control point four is post-stage evaluation: after each stage, evaluate general ability, reasoning, code, safety, and domain tasks separately.

A small-team reproduction template can run as follows. Week one: dissect the public recipe and align fields, without training yet. Week two: build a local SFT-Mix and run a small-model or short-step smoke test to validate format, loss curve, and basic evaluation. Week three: add DPO data, focusing on chosen/rejected quality rather than pair count. Week four: introduce a small verifiable-task set and validate verifier, answer extraction, and failure logs. Only after these stabilize should data scale and training steps grow.

Tulu-3 also reminds us that an open recipe does not remove the need for evaluation isolation. Public datasets are reused repeatedly and may contain benchmark styles, answer templates, or common task forms. When reusing public SFT or DPO data, build a private holdout evaluation set and run near-duplicate checks before training. For math and code, check both prompt and answer; checking only prompt can still leak canonical solutions.

## 45.6 Case B: Reading Llama-3 Multi-Round RLHF

Llama-3 (Dubey et al. 2024) represents a high-investment industrial path. Unlike Tulu-3, which emphasizes reproducible public recipes, the Llama-3 report emphasizes multi-round RLHF iteration. The key is not one dataset, but the flow among preference collection, reward-model updates, rejection sampling, and failure-sample feedback.

**Why multi-round iteration and RM retraining?** After initial SFT, RLHF with an initial reward model quickly moves the policy toward high-RM-score regions. But the RM has out-of-distribution blind spots. As the policy improves, its outputs move away from the first RM's training distribution. Without RM updates, the model can learn reward hacking. Llama-3-style iterations generate new responses on hard prompts with the current model, send them to annotation, and add the new data to RM training. Exact scale per round should be reported only when disclosed; otherwise use `[I]` or state that it is not disclosed.

**The bridge role of rejection sampling fine-tuning (RSFT).** Llama-3 uses RSFT as a bridge between RLHF and SFT. For each prompt, the system samples multiple outputs, scores or filters them with the latest RM or preference mechanism, and turns high-quality answers into pseudo-label data. This data flows into the next SFT or post-training round and improves the base output distribution. Candidate count, sampling strategy, and retention ratio should not be asserted without sources.

**Feedback mechanism.** Each iteration captures failures and boundary samples from red-team and evaluation. High-scoring samples reinforce positive behavior; failures can be turned into rejected samples for DPO or RM training. This creates an online optimization loop. Tulu-3 shows a reproducible public method asset; Llama-3 shows the operational value of continuous industrial iteration.

| Iteration step | Data input | Processing action | Data output | Writing note |
| --- | --- | --- | --- | --- |
| Initial sampling after SFT | Hard prompts, online failures | Multi-candidate generation | Candidate responses | Mark sampling scale as `[I]` when unsourced |
| Preference collection | Candidate answers | Human annotation or RM-assisted filtering | chosen/rejected or ranking data | Annotation scale must be checked against reports |
| RM update | New and historical preference data | Retrain or update RM | New RM checkpoint | Describe mechanism, do not invent parameters |
| Rejection sampling fine-tuning | Candidates + RM scores | Select high-quality answers | Pseudo-label SFT data | Candidate count requires source support |
| Failure feedback | Red-team, evaluation, online logs | Classify, deduplicate, risk-label | hard cases / rejected pool | Privacy and contamination boundaries must be stated |

The most useful engineering lesson is that every post-training round redefines the data distribution. After the first SFT round, failures often involve basic instruction following and safety boundaries. After several preference rounds, failures shift toward long-context consistency, multi-constraint tasks, factual details, tool use, and subtle safety edges. Training later rounds on the first-round distribution yields diminishing returns. Multi-round iteration is not repeated training; it is data distribution moving with model capability.

RSFT is important because it finds high-quality trajectories in the current model's candidate space and stabilizes behaviors that the model can already produce occasionally. It is closer to the model's frontier than hand-written SFT and easier to control than pure RL. But its risk is clear: an RM with length, format, or over-refusal bias will amplify that bias through rejection sampling. RSFT must be paired with RM audit sets and human sampling; the highest-scoring sample is not automatically truth.

Small teams cannot reproduce Llama-3's annotation scale, but they can reuse the rhythm. A simplified loop might use 10K to 20K high-quality SFT samples, sample multiple candidates for 1K to 3K hard prompts, filter with a light RM, rule verifier, or human group, convert selected answers into RSFT data, then add new failures to the next hard-prompt pool. This is far smaller than an industrial pipeline but already contains the essential loop: model generation, review, data feedback, retraining.

Privacy and compliance are essential. Online failures are not automatically trainable samples. User prompts may contain personal information, trade secrets, medical or financial sensitivity, or usage contexts that prohibit training. Real RLHF or RSFT systems must anonymize, check usage authorization, classify data, and maintain audit logs before any sample enters training.

## 45.7 Reward Hacking, Data Contamination, and Process Reward Data

Post-training faces systemic risks. Without controls, more compute and annotation can produce negative returns.

### 45.7.1 Data-Side Symptoms of Reward Hacking

Reward hacking is one of the most common and underestimated risks in preference alignment. When optimization is governed by an imperfect RM, the model finds the easiest way to score high rather than genuinely solve the user's problem.

Common symptoms include length bias, where long answers score better and become bloated; sycophancy, where the model flatters the user's implied stance; safety overgeneralization, where ordinary educational or fictional requests are refused; and fake reasoning, where steps look organized but have no verifiable logic.

Data-side defenses include constructing long but useless answers as rejected samples against short and precise chosen answers, introducing fact-check APIs, unit tests, and math verifiers as hard constraints, retaining failed trajectories rather than only final successes, and auditing RM training distribution regularly.

### 45.7.2 Data Contamination in Post-training

Chapter 5 covered benchmark contamination generally. Post-training has additional hidden contamination modes.

1. **SFT includes evaluation-set answers.** Synthetic instruction construction may accidentally feed test-set ground truth into training.
2. **Preference training solidifies benchmark style.** Annotators or RMs may score benchmark-template answers highly, turning evaluation style into a reward preference.
3. **Implicit rejection-sampling contamination.** Using external evaluation pass rates to filter training samples leaks test metrics.
4. **Feedback-loop contamination.** Online systems may directly feed users' benchmark prompts back into daily training logs.

| Contamination type | Location | Typical symptom | Check | Handling |
| --- | --- | --- | --- | --- |
| SFT contamination | Instruction synthesis, data mixing | Model is unusually familiar with benchmark questions | n-gram / embedding near-duplicate checks | Remove contaminated samples and rebuild split |
| Preference contamination | RM / DPO data | Model prefers benchmark-style expressions | Check both prompt and answer | Downweight or isolate related preference pairs |
| Rejection-sampling contamination | Candidate filtering | Evaluation score is used to select training samples | Inspect filtering signal source | Forbid evaluation sets in training selection |
| Online-feedback contamination | User-log feedback | Test prompts enter training logs | Source tags and blacklist filtering | Build benchmark quarantine |

Contamination often results from multiple ordinary-looking steps. A public math benchmark may first be used for SFT, then the model generates candidates from it, then an RM learns preferences over benchmark-style answers, and finally rejection sampling filters on the same problem set. Each step looks normal in isolation; together, the evaluation set has influenced content, preference direction, and selection strategy.

A practical control is a benchmark quarantine. Official benchmarks, red-team acceptance sets, holdout sets, and critical A/B samples enter an isolation list. Before data enters SFT, DPO, RM, RSFT, or RLVR, the pipeline checks near-duplicates on both prompt and answer. Code and math need structural checks too: function names, tests, comments, reference solutions, problem numbers, variable relations, and final answers. String deduplication alone is not enough.

Judge contamination is another risk. Many teams use strong models as LLM-as-a-Judge and then build preference data from judge outputs. If the judge has seen benchmarks or carries preference biases, those biases enter the new model. The answer is not to ban judges, but to record judge version, prompt, rationale, and known bias, and to calibrate with human audit sets. Important data should use multiple review sources: rule verifier, LLM judge, and human sampling.

### 45.7.3 Process Reward Data and the Chapter 46 Interface

To address answers that are correct but reasoning processes unreliable, industry uses process reward models (PRMs). This chapter does not implement PRM, but its data value is clear: reasoning models need rewards not only on final answers, but also on intermediate reasoning steps, verifier states, and complete rejection-sampling trajectories. This leads directly to Chapter 46's reasoning data flywheel.

The core unit of process reward data is a step, not a full response. A usable sample should store the problem, full trajectory, step segmentation, local judgment per step, final answer, final verification result, and error type. If only the entire CoT is stored, the PRM cannot learn where the trajectory first diverges. Math tasks may label algebra error, missing condition, or final numeric error. Code tasks may label correct idea but wrong boundary condition, complexity failure, or uncovered tests.

Process reward data must distinguish "process correct but result wrong" from "result correct but process wrong." The former may indicate final calculation or extraction failure. The latter may indicate guessing or pattern memorization. Systems that look only at final answer reward the second type incorrectly. PRM, RLVR, and rejection sampling are valuable together because they preserve both process and result signals.

## 45.8 Pitfalls, Costs, and Boundaries

Post-training involves route choice, cost, organizational capability, and risk control.

Common pitfalls include SFT-only training that produces professional-looking format but unstable behavior under jailbreaks; chasing instruction count and amplifying template bias; weak DPO signal when chosen and rejected are both poor or too similar; narrow RM coverage that creates online reward loopholes; using Magpie-like seed-free generation without diversity, difficulty, factuality, and safety filters; applying RLVR to tasks without standard answers; and transferring open recipes into medical or financial domains without rebuilding safety boundaries.

Engineering costs are easy to underestimate. Human preference annotation needs domain-capable annotators and consistency training. Synthetic data inference consumes GPU time, especially for Evol-Instruct and multi-candidate sampling. Alignment adds multi-candidate sampling and ongoing RM / verifier maintenance. Data audit and contamination detection run through the whole pipeline and are central to trustworthiness.

| Cost item | Main source | Often underestimated | Degraded strategy |
| --- | --- | --- | --- |
| Human preference annotation | chosen/rejected, ranking, attribute scoring | Annotator consistency training and review | Start with a small high-consistency set |
| Synthetic-data inference | Self-Instruct, Evol-Instruct, Magpie | Low effective yield after filtering | Reduce candidate count, cover core tasks first |
| Multi-candidate sampling | RSFT, RM selection, GRPO groups | Token cost grows linearly with candidate count | Sample more on hard prompts, downsample ordinary prompts |
| RM / verifier maintenance | RM retraining, rule repair, unit-test expansion | Rule loopholes can pollute training | Version verifiers and run regression tests |
| Contamination detection | Benchmark quarantine, near-duplicate checks | Answer-side contamination is subtler than prompt-side | Deduplicate both prompt and answer |

Project cost control can be staged. Stage one allows only SFT and validates schema, training scripts, and basic evaluation. Stage two introduces a small preference-pair set to verify that DPO or RM changes a clear behavior, such as reducing verbosity or repairing refusal boundaries. Stage three introduces multi-candidate sampling and RSFT to see whether occasional high-quality answers can be stabilized. Stage four considers RLVR or PRM after the team has verifiers, error types, and evaluation sets. Staged investment is safer than full RLHF from the beginning.

Cost should be measured by effective sample yield, not only GPU hours. If a synthesis pipeline generates one million samples but only fifty thousand survive format, quality, safety, contamination, and human sampling, cost should be calculated on the fifty thousand. Likewise, if rejection sampling generates 32 candidates and keeps one, the visible training set is small but the discarded 31 candidates already consumed inference cost. Reports should include raw generated volume, retained volume, human pass rate, training-consumed volume, and final evaluation gain.

Boundaries must be explicit. General chat assistants can rely more on preference data and human review. Math, code, and structured tasks benefit from verifiers. Domain assistants require expert review and compliance audit. Open-ended tasks without standard answers should not be forced into RLVR. Sensitive medical and financial tasks should not rely only on LLM-as-a-Judge. The goal is not to force all tasks into one paradigm, but to choose the right supervision signal for each task.

## 45.9 Summary: From Behavior Alignment to Reasoning Data Flywheels

This chapter decomposed core data recipes in post-training and highlighted several principles.

1. Post-training data engineering is not polishing individual samples. It is system coordination across SFT, preference optimization, and online feedback.
2. The main value of open recipes is not copying their files, but studying disclosed data shapes, stage order, cleaning mechanisms, and iterative feedback loops.
3. SFT, DPO, RLHF, and RLVR form a bridge to the next generation of RL reasoning paradigms discussed in Chapter 46.

When reward signals expand from subjective human preference to rule- or program-verifiable signals, post-training enters the reasoning data flywheel. In that flywheel, cold-start SFT provides base behavior, multi-sample generation creates candidate trajectories, rule-based reward provides verification, rejection sampling extracts high-quality traces, and the traces feed back into second-round SFT. Chapter 46 develops this data engineering loop in detail.

## References

Wang Y, Kordi Y, Mishra S, Liu A, Smith N A, Khashabi D, Hajishirzi H (2023) Self-Instruct: Aligning Language Models with Self-Generated Instructions. Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, pp 13484-13508.

Ouyang L, Wu J, Jiang X, Almeida D, Wainwright C, Mishkin P, Zhang C, Agarwal S, Slama K, Ray A, Schulman J, Hilton J, Kelton F, Miller L, Simens M, Askell A, Welinder P, Christiano P F, Leike J, Lowe R (2022) Training Language Models to Follow Instructions with Human Feedback. Advances in Neural Information Processing Systems, 35, 27730-27744.

Rafailov R, Sharma A, Mitchell E, Manning C D, Ermon S, Finn C (2023) Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. Advances in Neural Information Processing Systems, 36, 53728-53741.

Ethayarajh K, Xu W, Muennighoff N, Jurafsky D, Kiela D (2024) Model Alignment as Prospect Theoretic Optimization. Proceedings of the 41st International Conference on Machine Learning, pp 12634-12651.

Gheshlaghi Azar M, Guo Z D, Piot B, Munos R, Rowland M, Valko M, Calandriello D (2024) A General Theoretical Paradigm to Understand Learning from Human Preferences. Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, pp 4447-4455.

Grattafiori A, Dubey A, Jauhri A, Pandey A, Kadian A, Al-Dahle A, Letman A, Mathur A, Schelten A, Vaughan A, others (2024) The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.

Lambert N, Morrison J, Pyatkin V, Huang S, Ivison H, Brahman F, Miranda L J V, Liu A, Dziri N, Lyu X, Gu Y, Malik S, Graf V, Hwang J D, Yang J, Le Bras R, Tafjord O, Wilhelm C, Soldaini L, Smith N A, Wang Y, Dasigi P, Hajishirzi H (2025) Tulu 3: Pushing Frontiers in Open Language Model Post-Training. Second Conference on Language Modeling.

Yang A, Li A, Yang B, Zhang B, Hui B, Zheng B, Yu B, Gao C, Huang C, Lv C, others (2025) Qwen3 Technical Report. arXiv preprint arXiv:2505.09388.

Wang Z, Dong Y, Delalleau O, Zeng J, Shen G, Egert D, Zhang J J, Sreedhar M N, Kuchaiev O (2024) HelpSteer 2: Open-Source Dataset for Training Top-Performing Reward Models. Advances in Neural Information Processing Systems, 37, 1474-1501.

Xu C, Sun Q, Zheng K, Geng X, Zhao P, Feng J, Tao C, Lin Q, Jiang D (2024) WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions. International Conference on Learning Representations.

Xu Z, Jiang F, Niu L, Deng Y, Poovendran R, Choi Y, Lin B Y (2025) Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing. International Conference on Learning Representations.

Liu A, Feng B, Xue B, Wang B, Wu B, Lu C, Zhao C, Deng C, Zhang C, Ruan C, others (2024a) DeepSeek-V3 Technical Report. arXiv preprint arXiv:2412.19437.

Liu C Y, Zeng L, Liu J, Yan R, He J, Wang C, Yan S, Liu Y, Zhou Y (2024b) Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs. arXiv preprint arXiv:2410.18451.

Singhal P, Goyal T, Xu J, Durrett G (2024) A Long Way to Go: Investigating Length Correlations in RLHF. First Conference on Language Modeling.

Zhou K, Zhu Y, Chen Z, Chen W, Zhao W X, Chen X, Lin Y, Wen J-R, Han J (2023) Don't Make Your LLM an Evaluation Benchmark Cheater. arXiv preprint arXiv:2311.01964.

Lightman H, Kosaraju V, Burda Y, Edwards H, Baker B, Lee T, Leike J, Schulman J, Sutskever I, Cobbe K (2024) Let's Verify Step by Step. International Conference on Learning Representations.
