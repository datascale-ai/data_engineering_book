# Chapter 45: LLM Post-Training Data Engineering in Practice: SFT and Preference Alignment

## Abstract

Supervised Fine-Tuning (SFT) and Preference Alignment are the two core data engineering entry points in the post-training phase of Large Language Models (LLMs), yet they serve fundamentally different objectives: the former uses Maximum Likelihood Estimation (MLE) to establish behavioral templates for the model, while the latter shapes the model's reward surface through chosen/rejected contrastive signals. This chapter deconstructs post-training data engineering practices in open-source models from a "recipe" perspective. It begins by proposing a three-stage pipeline of SFT, preference alignment, and online continuous optimization, defining the data shape, optimization objectives, and manifest governance requirements at each layer. It then cross-examines post-training data transparency and scale across Tülu-3, Llama-3, Qwen2.5, and Nemotron-4/HelpSteer2 as four representative approaches, distinguishing three levels of information credibility: direct disclosure, reasonable inference, and pedagogical estimation. The chapter subsequently compares the engineering differences among Self-Instruct, Evol-Instruct, and Magpie—three schools of instruction synthesis—with respect to seed dependency, difficulty calibration, and distribution filtering, and extends to data production methods for paradigms including Rejection Sampling (RS), Reward Models (RM), and Direct Preference Optimization. This chapter emphasizes that the maturity of post-training data engineering lies not in generating samples, but in whether each sample can be traced to its source, isolated for evaluation, documented for its intended use, and supported for rollback.

## Keywords

LLM; data recipe; open-source large models; training data; phased scheduling

## Learning Objectives

- Distinguish the optimization objectives of SFT and preference alignment, and explain the difference between MLE behavioral templates and chosen/rejected reward surface shaping.
- Design a three-stage pipeline for SFT, preference alignment, and online continuous optimization, and define the data shape and manifest governance requirements at each layer.
- Compare the engineering differences among Self-Instruct, Evol-Instruct, and Magpie—three schools of instruction synthesis—with respect to seed dependency, difficulty calibration, and distribution filtering.
- Design preference data production methods for paradigms including rejection sampling, reward models, DPO, GRPO, and RLVR.
- Evaluate post-training data with respect to source traceability, evaluation isolation, use-case documentation, and rollback capability, and identify risks of Reward Hacking and data contamination.

## 45.0 Opening Scenario: Why Alpaca Cannot Train a GPT-4-Style Model

From 2023 through early 2024, the open-source community intensively explored the instruction tuning route. Consider a common team scenario: a large model application team, aiming to build a vertical-domain assistant, uses an open-source base model and collects approximately 200K instruction data samples in the style of Alpaca, Dolly, and Self-Instruct, then applies Supervised Fine-Tuning (SFT).

After training, the surface-level metrics appear normal: the loss steadily decreases, and the model does indeed transition from a "text continuation engine" into an "assistant" willing to answer questions. However, during internal red-blue adversarial testing and real-world business gray-scale evaluation, the team still identifies several critical gaps:

1. **Insufficient complex instruction-following capability:** When a prompt contains more than three constraints—such as "output in JSON format," "no more than 100 words," and "include three paragraphs"—the model will most likely miss some conditions.
2. **Lack of refusal boundaries:** When faced with misleading, harmful, or clearly out-of-scope questions—such as "how to manufacture dangerous substances" or fabricated medical diagnoses—the model may produce unreliable responses rather than consistently recognizing the risk and declining.
3. **Unstructured long-form responses:** The model struggles to generate long-form content with clear hierarchy and logical progression, and multi-turn consistency is unstable.
4. **Weak tool-use awareness:** When faced with questions requiring external API support, the model tends to fabricate answers directly rather than outputting a properly formatted tool-call JSON.

Regardless of how the team cleans this 200K dataset, the resulting model consistently underperforms Llama-3.3-Instruct or Qwen2.5-Instruct in terms of user experience.

This example illustrates that post-training cannot be understood merely as adding more instruction samples; it is more akin to a phased data organization and behavioral calibration process.

* SFT is solely responsible for establishing behavioral templates—it can teach the model "what format to speak in," but cannot independently accomplish preference ordering, i.e., it cannot teach the model "what constitutes a better response."
* Preference Alignment is not a simple addendum after SFT, but an independent layer of data production and review.
* Industrial-grade online continuous optimization requires integrating user feedback, Rejection Sampling (RS), Reward Models (RM), and contamination detection into a single data chain.
* The reference value of open-source models comes not only from open weights, but also from the post-training recipe information disclosed in technical reports, dataset cards, and training pipelines.

---

## 45.1 The Three Stages of Post-Training Data: SFT, Preference Alignment, and Online Continuous Optimization

To equip a base model with stable interaction capabilities, post-training data engineering typically requires building a layered pipeline. These layers differ not only in data shape, but also in optimization objectives and engineering challenges.

**Layer One: SFT Data**
The core task of SFT is "formatting." It transforms the base model from an unconscious "probabilistic predictor / text continuation engine" into a well-mannered "assistant." SFT data specifies the basic response format, task domain boundaries, assistant role tone, fundamental safety behavioral baselines, and the alternating structure of multi-turn dialogue.
In modern engineering practice, the common scale for SFT data is typically in the range of $10^5$ to $10^6$ samples. At this stage, data quality (diversity, accuracy, and formatting rigor) far outweighs quantity. Millions of low-quality SFT samples are often inferior to hundreds of thousands of carefully curated samples.

**Layer Two: Preference Alignment Data**
If SFT teaches the model "how to answer," then the preference alignment layer teaches the model "which of two qualified responses is better." This layer establishes the model's reward surface.
Preference data can serve multiple different training paradigms: training an RM to support RLHF, or directly serving direct preference optimization methods such as DPO, IPO, KTO, GRPO, or RLVR. Its typical scale spans a wide range, from $10^5$ to $10^7$ preference pairs, depending on whether the data construction process includes large-scale automatic generation, multi-round sampling, and online feedback.

**Layer Three: Online Continuous Optimization Data**
Model deployment is not the end of post-training, but the beginning. The third layer determines whether the model can self-correct as the real business evolves.
After deployment, user upvotes and downvotes, system refusal logs, difficult samples, manually reviewed red-line data, A/B experiment results, and emergent safety incidents all enter the continuous optimization pipeline through streaming or batch processing. Without this layer, the model remains at the static level of a one-time release, ultimately rendered obsolete by the ever-shifting distribution of user inputs.

**Why can high-quality SFT data not replace preference data?**
This is because SFT uses Maximum Likelihood Estimation (MLE): during training, the model is primarily encouraged to "imitate" the given target tokens and has no awareness of the relative quality of other potential responses. For a complex question with multiple reasonable but varying-quality answers, SFT struggles to teach the model to choose, at generation time, the response that is more aligned with human intuition, safer, or more thorough. Preference data, by introducing chosen/rejected contrastive signals, can suppress the response space of answers that are grammatically correct but misaligned with values or logic. Therefore, post-training data engineering must simultaneously manage the behavioral shape molded by SFT and the feedback loop formed by preference alignment and continuous optimization.

![Figure 45-1: Schematic of the LLM Three-Stage Post-Training Pipeline](../../images/part11/30_1_posttrain_three_stage_pipeline.png)
*Figure 45-1: Data flow relationships among SFT, Preference Alignment, and Online Continuous Optimization.*

From an engineering implementation perspective, the three-stage framework also implies three entirely different modes of data asset management. SFT data resembles a "behavioral template library"—it must be stable, clean, cover common tasks, and maintain a simple field structure. Teams typically build a minimal schema around fields such as `messages`, `instruction`, `input`, `output`, `source`, `license`, and `quality_score`. As long as the schema is stable, SFT data can easily be consumed by different training frameworks and supports cross-version comparisons.

Preference data more closely resembles a "behavioral discrimination library." It cannot only store the final chosen/rejected texts; it should also preserve the candidate generation model, sampling temperature, number of candidates, the annotator or review model, annotation rationale, conflict review outcomes, and tags indicating whether the sample involves safety, factuality, code, mathematics, or tool invocation. Without this metadata, it is very difficult to explain retrospectively why a DPO training run improved or degraded. Many teams, on their first foray into preference data, only retain two responses and a `label` field—and only realize upon encountering preference drift that they lack sufficient information to diagnose the problem.

Online continuous optimization data more closely resembles an "event log library." It originates from real users, red teams, evaluation platforms, customer service feedback, and online monitoring, and naturally carries attributes of time, context, version, and access permissions. Therefore, it must be bound to the model version, prompt version, system policy, user authorization status, desensitization status, and evaluation set isolation status. Otherwise, seemingly valuable real-world feedback may become unusable for training due to privacy concerns, contamination, or non-traceable versioning. In other words, the maturity of post-training data engineering is reflected not only in the ability to produce samples, but in whether each sample can answer: "Where does it come from, why is it trustworthy, which stage is it suitable for, and how can it be rolled back if something goes wrong?"

A sound post-training data repository should maintain at least four types of manifests. The first is a data source manifest, recording dataset names, licenses, collection timestamps, filtering conditions, and responsible parties. The second is a sample processing manifest, recording deduplication, filtering, rewriting, translation, scoring, and manual review processes. The third is a training consumption manifest, recording which model version, training stage, and experiment configuration consumed a given data version. The fourth is an evaluation isolation manifest, recording which benchmarks, red-team sets, and acceptance sets are prohibited from entering training. These four manifest types may seem burdensome, but they determine whether a post-training pipeline can evolve from a one-off experiment into an operational system.

---

## 45.2 Cross-Comparison of Open-Source Post-Training Data Transparency

Before building one's own post-training pipeline, it is necessary to cross-compare the publicly disclosed approaches of current mainstream open-source models. This section selects Tülu-3, Llama-3, Qwen2.5, and Nemotron-4 as four representative approaches for analysis. The core objective is not to evaluate whose leaderboard scores are higher, but to establish an engineering methodology for "how to read public information" and assess its practical guidance value.

When reading the table below, please note the annotation conventions:

* **[D]**: Numbers explicitly disclosed in technical reports, papers, or dataset cards.
* **[I]**: Scale figures reasonably inferred based on publicly available training pipelines, split ratios, or contextual information.
* **[E]**: Estimates derived for pedagogical illustration or engineering approximation; should not be treated as official disclosed figures.

**Table 45-1: Post-Training Data Transparency and Scale of Mainstream Open-Source Models**

| Model / Project | Post-Training Stages | Data Openness | SFT Data Scale | Preference / Reward Data Scale | Key Data Sources | Reproducibility Value |
| --- | --- | --- | --- | --- | --- | --- |
| **Tülu-3** | SFT / DPO / RLVR | High | 939K [D] | DPO mixture scale requires item-by-item verification [D] | SFT-Mix, DPO mix, RLVR verifier | Fully open-source recipe reference system; suitable for reproduction and transfer |
| **Llama-3** | SFT / Reward Model / RLHF | Medium | Not fully disclosed; report reveals pipeline and partial statistics [D/I] | Multi-round preference annotations and RM/DPO data; total volume requires table-level verification [D/I] | Human annotation, multi-round RM, rejection sampling | Understanding industrial-grade multi-round iteration with heavy human involvement |
| **Qwen2.5** | SFT / Preference / Large-scale synthetic data | Medium | Partially disclosed; exact total should be verified against the report [D/I] | Partially disclosed; scale unknown [D/I] | Instruction synthesis, multilingual, multi-task; Magpie as a seed-free synthesis reference paradigm | Observing Chinese, multilingual post-training and large-scale synthetic data |
| **Nemotron-4** | Reward Model / HelpSteer2 | High | Not the focus of this section | HelpSteer2 approximately 10K prompts/pairs level; verify against dataset card [D] | Attribute-based preference annotation, Daring-Anteater SFT data | An important reference for reward model data design |

In Table 45-1, `[D]` denotes information directly disclosed in public materials, and `[I]` denotes reasonable inferences based on publicly available pipelines or contextual information. Numbers that cannot be directly traced should not be stated as definitive figures; they should be retained as "undisclosed" or "requires verification against source."

* **Tülu-3** is one of the most suitable projects in this chapter to serve as a reproducible baseline. It not only open-sources the model weights, but also discloses the post-training data mixture, training code, and evaluation methodology, enabling teams to translate the recipe from the paper into an inspectable engineering process.
* **Llama-3** (Dubey et al. 2024) represents a heavy-asset industrial approach. Its report discloses key mechanisms such as multi-round post-training, preference annotation, reward model retraining, and rejection sampling, but many data details are not fully disclosed. It is therefore more suitable as a reference for understanding industrial closed-loop systems than as a direct template for replication.
* **Qwen2.5** provides important reference value for Chinese, multilingual, multi-task, and synthetic data approaches. A distinction must be carefully drawn: the synthetic data approach in the Qwen2.5 report and seed-free synthesis methods such as Magpie (Xu et al. 2024) can be discussed in parallel, but should not be conflated as "officially adopting Magpie" in the absence of an explicit source.
* **Nemotron-4** and HelpSteer2 derive their value from the granularity of preference annotation. HelpSteer2 (Wang et al. 2024b) does not merely record overall preference—it establishes scoring signals along dimensions such as helpfulness, correctness, coherence, complexity, and verbosity, providing a referenceable example for reward model data design.

---

## 45.3 Three Schools of SFT Data Synthesis: Self-Instruct, Evol-Instruct, and Magpie

Having established that SFT occupies the first layer of post-training, the immediate question is how to obtain high-quality instructions. Since manual authoring is costly and struggles to cover long-tail tasks, Instruction Synthesis has become the mainstream approach. This section compares the engineering differences of three synthesis approaches as they appear in real post-training recipes.

### 45.3.1 Self-Instruct: Expanding the Instruction Space from Seed Tasks

**Key Points:**
Self-Instruct (Wang et al. 2022) is one of the seminal methods in instruction synthesis. It relies on a small set of manually written seed tasks—typically a few hundred examples—as a starting point. In the pipeline, a powerful model references these seeds to generalize and generate new instructions, inputs, and outputs, i.e., the instruction, input context, and expected response.
**Engineering Advantage:** Well suited for rapidly expanding task-domain coverage in early project stages, addressing the base model's "not knowing how to initiate a conversation" problem.
**Primary Risk:** Heavily dependent on prompt templates, making generated data prone to templatization and linguistic homogeneity, with task difficulty typically concentrated in the common range and insufficient coverage of complex edge cases.

### 45.3.2 Evol-Instruct: Increasing Complexity via Evolutionary Rules

**Key Points:**
To address the insufficient difficulty of Self-Instruct, WizardLM (Xu et al. 2023) proposed the Evol-Instruct approach. Its core idea is to systematically increase the complexity of simple instructions through specific "evolutionary rules." Common evolution operations include: adding constraints, increasing reasoning depth, introducing multi-condition branching, and requiring multi-step solutions.
**Engineering Advantage:** Extremely effective at generating high-complexity instruction-following data, compelling the model to learn deep logical compliance rather than superficial response patterns.
**Primary Risk:** Increased difficulty does not equate to increased quality. Multiple rounds of evolution can trigger intent drift—where complex constraints contradict each other, or the generated instruction becomes an incoherent accumulation of words. Therefore, difficulty calibration and answer verification are the quality control priorities for this school.

### 45.3.3 Magpie: Seed-Free Self-Generated Instructions and Responses

**Key Points:**
Another notable approach is Magpie. It minimizes dependency on manual seeds by directly leveraging the conversational priors of already-aligned Instruct models—such as Llama-3-Instruct—to generate user-side instructions. By providing a very short pre-query prompt, or even just a single system `[INST]` token, the model can be induced to generate relatively natural user questions along with corresponding responses.
**Engineering Advantage:** Substantially reduces human intervention, and the resulting data distribution more closely approximates the distribution of real, natural user queries across the long tail. When discussing the post-training of open-source models such as Qwen2.5, Magpie can serve as a reference method for "how large-scale synthetic instructions can enhance diversity," rather than being attributed as an official recipe component of a specific model without a direct source.
**Primary Risk:** Model self-generation can amplify the model's inherent biases and hallucinations. The data pipeline requires distribution filtering and safety filtering mechanisms.

**Table 45-2: Engineering Comparison of the Three SFT Synthesis Schools**

| School | Seed Dependency | Generation Method | Suitable Tasks | Primary Risk | Quality Control Focus | Representative Material | Relationship to This Chapter |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Self-Instruct** | Medium | Seed-inspired expansion | General instruction breadth coverage | Templatization, homogeneity | ROUGE deduplication, diversity evaluation, answerability checks | Self-Instruct paper | Foundational synthesis approach |
| **Evol-Instruct** | Medium | Rule-based complexity evolution | Complex instruction-following, multi-step reasoning | Logical contradictions, intent drift | Difficulty calibration, code-level answer verification | WizardLM project | Complexity-escalation approach |
| **Magpie** | Low | Seed-free self-generation using priors | Large-scale, realistic conversational instructions | Amplified inherent biases, hallucinations, unsafe content | Distribution diversity filtering, rigorous safety filtering | Magpie paper | Emerging open-source post-training approach |

![Figure 45-2: Pipeline Comparison of the Three SFT Synthesis Schools: Self-Instruct, Evol-Instruct, and Magpie](../../images/part11/30_2_sft_synthesis_pipelines.png)
*Figure 45-2: Data entry points, generation methods, and quality control checkpoints for the three SFT synthesis approaches.*

Regardless of which synthesis school is adopted, SFT data should never flow directly from the generator into the training set. A more robust approach is to establish four gatekeeping checks. The first is a format gate, which verifies whether multi-turn dialogue roles are complete, whether fields are missing, whether JSON or ChatML is parseable, and whether truncation or garbled text exists. The second is a semantic gate, which checks whether the instruction is genuinely answerable, whether the answer covers the core of the question, and whether there are mismatches, non-sequiturs, or unexplained reasoning leaps. The third is a distribution gate, which checks whether task types, languages, lengths, domains, difficulty levels, and safety categories are overly concentrated. The fourth is a leakage gate, which checks whether any sample is near-duplicate to evaluation sets, benchmark questions, published answers, or internal holdout sets.

These four gates are best implemented as a combination of automated filtering and manual spot-checking. Automated filtering is suitable for handling format errors, duplicates, length anomalies, low-quality templates, sensitive keywords, and obvious safety issues; manual spot-checking is suitable for judging whether instructions are natural, whether answers are genuinely helpful, and whether complex tasks preserve the original intent. In particular, with Evol-Instruct, automated scripts struggle to determine whether "making something more complex" still preserves the original intent. A question that appears more difficult may simply be a corrupted version of the original. Without manual spot-checking or a strong verifier, the model ends up learning many complex but invalid patterns.

SFT data also requires stratified mixing rather than simple aggregation. It is recommended to break the data into at least six categories: general Q&A, knowledge explanation, complex instruction-following, code and tool use, mathematics and reasoning, and safety and refusals. Each category should be individually tracked for count, average length, source, filter rate, and manual review pass rate. For reproducing open-source model recipes, the most valuable practice is not to exactly replicate a given mixing ratio, but to maintain a "mixing change log." When the model shows changes in code capability, safety refusals, or linguistic quality, the team can trace back to determine whether a change in a specific data category was responsible, rather than relying on intuition alone.

It is also important to note that good SFT data from one stage is not necessarily suitable for all training rounds. The first round of SFT is better suited to data that is structurally clear, response-stable, and broadly covering, helping the model establish basic assistant behavior. Subsequent incremental SFT rounds are better suited to incorporating hard cases, domain-specific tasks, tool invocation, and safety boundary repair data. Loading all data at once makes it easy for high-value boundary samples to be overwhelmed by a large volume of ordinary samples. A better approach is to organize SFT data into curriculum-style versions: `sft_base_mix`, `sft_complex_mix`, `sft_safety_patch`, `sft_domain_patch`, each with its own manifest and evaluation report.

---

## 45.4 Preference Data Engineering: From RLHF to DPO, GRPO, and RLVR

After acquiring SFT data, how preference data is constructed ultimately determines the model's "character." This section explains how the shape of preference data evolves with different training paradigms.

### 45.4.1 RLHF: Preference Pairs and Reward Models

Reinforcement Learning from Human Feedback (RLHF) (Ouyang et al. 2022) is the canonical alignment approach. Its core data form consists of multi-candidate rankings or preference pairs.

* **Data shape:** prompt + chosen + rejected, or relative ranking scores for $N$ candidates under the same prompt.
* **Engineering focus:** How to efficiently generate high-discriminability candidate responses? How to ensure high inter-annotator consistency for complex value judgments among human annotation teams? When training the Reward Model, how to ensure that the training set's distribution adequately covers the true prompt distribution seen in production?

### 45.4.2 DPO: Data Requirements for Direct Preference Optimization

Direct Preference Optimization (DPO) (Rafailov et al. 2023) bypasses the complexity of explicitly training an RM by reformulating the reinforcement learning objective as a binary cross-entropy loss.

* **Data shape:** A strict triplet of Prompt + Chosen + Rejected.
* **Engineering focus:** Highly dependent on preference pair quality. DPO is very sensitive to the differential signal between Chosen and Rejected. If the Rejected sample quality is too low (e.g., garbled text), DPO struggles to learn a meaningful preference boundary; if the difference primarily reflects surface-level lexical choices (e.g., Rejected has a harsh tone but correct logic, Chosen has a softer tone but factual errors), the model may be misled.

### 45.4.3 GRPO: Intra-Group Relative Comparison and Sampling Groups

GRPO (Shao et al. 2024) is commonly used for long-reasoning tasks. Rather than relying on a global absolute reward baseline, it emphasizes relative quality within a group for the same prompt.

* **Data shape:** prompt + candidate group + relative reward signal. The candidate group typically contains 4 to 8 responses, depending on sampling cost and task difficulty.
* **Engineering focus:** The data pipeline must preserve the candidate group structure, sampling parameters at generation time (e.g., temperature), verification signals, and group metadata. The pipeline's focus shifts to enabling efficient batch multi-path sampling.

### 45.4.4 RLVR: Verifiable Rewards and the New Post-Training Interface

Reinforcement Learning with Verifiable Rewards (RLVR) advances the source of preference from subjective "human judgment of what is better" to objective "results that are computable/verifiable."

* **Data shape:** task + answer + verifier signal, i.e., the task, the answer, and a hard reward generated by a verifier.
* **Suitable tasks:** Mathematical problem-solving, code generation, format compliance, and tool invocation. Math problems can be verified against the final answer, code problems can run unit tests, structured outputs can undergo JSON/XML regex or schema validation, and tool calls can check API return status codes.
* **Engineering challenge:** Writing verifiers that cover a sufficiently wide range of tasks without loopholes. This section introduces the data shape concept of RLVR to lay the groundwork for subsequent discussion; the R1-style reasoning flywheel will be detailed in Ch46.

**Table 45-3: Different Data Requirements Across Preference Paradigms**

| Paradigm | Core Data Shape | Reward Source | Suitable Tasks | Data Engineering Challenges | Interface with Ch46 |
| --- | --- | --- | --- | --- | --- |
| **RLHF** | prompt + multiple candidates + human/AI preference | Human annotation / trained RM | General assistant behavior, complex value alignment | High annotation cost, inter-annotator consistency difficult to guarantee | Provides industrial background for multi-round iteration |
| **DPO** | prompt + chosen + rejected | Offline preference pairs | Chat Q&A, safety refusal boundaries, style fine-tuning | Negative sample quality control, over-confidence penalization | Serves as a prerequisite for second-round SFT or reasoning alignment |
| **GRPO** | prompt + candidate group + relative reward | Intra-group comparative advantage | Complex reasoning, code generation, diverse sampling | Sampling efficiency, group information storage and synchronized logging | Core interface foundation for R1-style training |
| **RLVR** | task + answer + verifier signal | Hard rules / unit tests / math verifiers | Mathematical computation, code pass rate, strictly structured output tasks | Verifier robustness, coverage expansion | Key expansion point for the Ch46 RL reasoning flywheel |

The first engineering principle for preference data is to preserve candidate groups rather than only storing the win/loss outcome. Multiple candidates for the same prompt record the response space the model explores under its current policy. If only chosen and rejected are saved, the team loses a large amount of useful information—such as whether all candidates are of poor quality, whether only one has an obvious error, or whether there are multiple acceptable responses of similar quality but differing styles. This information determines whether to subsequently train an RM, apply DPO, or return to the SFT stage to supplement task templates.

The second principle is to record the rationale for preferences. Preference pairs are not inherently trustworthy: a chosen response may win because of factual correctness, clearer expression, or more stable safety boundaries—or simply because it is longer, more agreeable, or more aligned with the format preferred by the review model. Without a rationale field, it is impossible to distinguish "genuine preference" from "scoring bias." Therefore, preference data should include at minimum metadata fields such as `preference_reason`, `error_type`, `safety_label`, `factuality_label`, and `style_label`. For data requiring manual review, fields such as `annotator_id_hash`, `review_round`, and `disagreement` should also be recorded to facilitate annotation consistency analysis.

The third principle is to split preference data into training, calibration, and audit sets. The training set is used to optimize the model, the calibration set is used to monitor whether the RM or DPO is overfitting, and the audit set is used to detect preference bias. The audit set need not be large, but it must remain stable and must not be repeatedly contaminated by hyperparameter tuning. For example, one could maintain a group of "short but correct vs. long but vapid" samples, a group of "appropriate refusal vs. excessive refusal" samples, and a group of "factually correct but mediocre tone vs. polished tone but factually wrong" samples. Running this audit set after each training run enables timely detection of reward hacking and sycophancy.

The fourth principle is to manage verifiable tasks separately. Tasks such as mathematics, code, structured output, and tool invocation do not need to rely entirely on subjective preference, since they can incorporate verifiers. For these tasks, preference data should additionally record `verifier_name`, `verifier_version`, `test_case_hash`, `pass_rate`, and `failure_reason`. This information serves both RLVR and rejection sampling. The truly reusable interface when Ch46 discusses the R1-style reasoning flywheel is precisely this verifiable metadata.

---

## 45.5 Case Study A: Retrospective on Tülu-3's Fully Open-Source Post-Training

To provide readers with a grounded engineering reference, this section selects Tülu-3 as the primary reproducible baseline. Tülu-3 covers all three stages—SFT, preference optimization, and verifiable rewards—and has a relatively high level of openness in terms of data, training code, and evaluation methodology, making it well suited for learning how to translate a paper recipe into an engineering plan.

### 45.5.1 Why Tülu-3 Is Suitable as a Reference Baseline

Among the many models claiming to be open-source, some projects only release final weights, while others additionally release partial fine-tuning datasets. Tülu-3's advantage lies in discussing the verification logic for SFT-Mix, DPO Mix, and RLVR within the same publicly available pipeline. It provides an end-to-end baseline that helps practitioners understand what types of data are approximately needed at each stage and how those data flow into the next stage.

### 45.5.2 SFT-Mix: The Behavioral Template Layer

The SFT stage of Tülu-3 (Lambert et al. 2024) does not pursue unbounded expansion of sample volume. Its SFT-Mix scale is approximately 939K [D]—a figure disclosed in the publicly available training data documentation—and should be used in alignment with the corresponding dataset card or paper table.
**Source structure and composition:** The SFT-Mix reflects a deliberate manual curation strategy. It combines basic dialogue, multi-task instruction-following, multi-turn interaction, API tool use, code generation, mathematical reasoning, and core safety tasks according to stage objectives.
**Why not simply pursue higher sample count?** Blindly stacking millions of simple function-completion samples for code tasks would cause "catastrophic forgetting," causing the model to lose the ability to hold a normal natural-language conversation. Tülu-3's experience demonstrates that by downsampling and balancing high-quality sources—such as carefully annotated domain-specific sets—the model's comprehensive behavioral patterns can be rapidly consolidated within a few tens of thousands of fine-tuning steps.

### 45.5.3 DPO: The Preference Refinement Layer

After SFT establishes the model's basic behavioral patterns, Tülu-3 introduces the DPO mix for preference refinement. The specific scale and composition of the DPO mix should be verified against the paper and dataset card in the final version to avoid conflating statistics from different stages or different splits.
**Connecting SFT and DPO:** DPO data should not only include external datasets, but should also cover typical failure patterns in the base model's actual output behavior after SFT.
**Reflecting differential preference:** The design of chosen and rejected pairs should not be random. When constructing data, Tülu-3 emphasizes avoiding a dangerous tendency: do not select a response that is "superficially more polite but factually inferior" as the chosen. If the data pipeline uses an uncalibrated LLM-as-a-Judge, the model is prone to this type of preference misguidence (Length Bias & Sycophancy (Zheng et al. 2023)). Therefore, high-quality DPO pairs must have a clear and stable difference on the dimension of "factual correctness."

### 45.5.4 RLVR: The Verifiable Task Layer

Finally, Tülu-3 introduces the RLVR stage.
**Introducing a Verifier:** For problems that can be measured by clear correct/incorrect outcomes, human annotation is both expensive and error-prone. Tülu-3 constructs a rule-based Verifier module.
**Applicable scope:** Not all tasks are suitable for RLVR. Tülu-3 primarily applies it to tasks from which a definitive terminal state can be extracted: the numerical value of the final step in a math problem, AST parsing of code, and unit test execution results. This section establishes the conceptual interface for RLVR; how to leverage rule-based verification signals to build an R1-style reasoning data flywheel will be elaborated in Ch46.

![Figure 45-3: Tülu-3 Three-Stage Data Flow and Scale Schematic](../../images/part11/30_3_tulu3_posttrain_flow.png)
*Figure 45-3: Data flow and stage relationships in Tülu-3 from SFT-Mix and DPO mix to RLVR.*

When migrating Tülu-3 to a proprietary project, the process can be broken down into four steps. First, categorize the data sources in the public recipe into three types: "directly reusable," "structural reference only," and "must be replaced with domain data." Second, preserve the sequential order of SFT, DPO, and RLVR stages, but adjust the task composition ratio at each stage according to domain-specific risk considerations. Third, document the inputs, outputs, filtering rules, and evaluation sets for each stage in a manifest, rather than only retaining scripts without a means of retrospective review. Fourth, after migration, re-conduct contamination detection and safety boundary assessment, since passing a public recipe does not automatically guarantee suitability for industry-specific contexts.

More specifically, migrating Tülu-3 should not start with "which datasets to download," but with "which control points to replicate." The first control point is SFT-Mix source registration: why each source enters the mixture, what capability it addresses, whether its license permits use, and whether it conflicts with evaluation sets. The second control point is DPO mix preference boundaries: whether the chosen/rejected differential stably reflects factual correctness, safety, conciseness, and instruction-following rather than merely length or tone. The third control point is RLVR verifier version management: rules, unit tests, answer extractors, and format parsers should all carry version numbers; otherwise, the same batch of samples may yield different rewards at different points in time. The fourth control point is post-training evaluation: after each training stage, results should be examined separately across general capability, reasoning, code, safety, and domain tasks, rather than using a single average score that can mask localized degradation.

If written as a project workflow, this process yields a post-training reproduction template suitable for small and medium-sized teams. The first week is dedicated solely to dissecting the public recipe and aligning field schemas—no rush to train. The second week builds a local SFT-Mix, running a small model or small-step smoke test to verify data format, loss curves, and basic evaluation. The third week introduces DPO data, with a focus on verifying chosen/rejected quality rather than pursuing preference pair count. The fourth week introduces small-scale verifiable tasks to validate verifiers, answer extraction, and failure logs. Only after all four steps are stable does it become worthwhile to scale up data volume and training steps. This approach may seem slow, but it avoids the most common failure mode in post-training projects: training completes, but no one knows where the performance change came from.

Another lesson from Tülu-3 is that an open recipe does not mean reproducers can ignore evaluation isolation. Publicly available datasets are frequently reused by the community and may already contain benchmark-style questions, answer templates, or common question types. Therefore, when reusing public SFT or DPO data, teams should establish their own held-out evaluation sets and perform near-duplicate detection on the public data before training. For mathematics and code tasks, both the problem statement and the answer must be checked simultaneously; checking only the prompt while ignoring the answer can still leak reference solutions into training. This detail determines whether the reproduction results are trustworthy.

---

## 45.6 Case Study B: Interpreting Llama-3's Multi-Round RLHF Iteration

Llama-3 (Dubey et al. 2024) represents a high-investment industrial approach to post-training. Unlike Tülu-3, which emphasizes the reproducibility of a public recipe, the Llama-3 report emphasizes multi-round RLHF iteration. The key is not any individual dataset, but the engineering workflow connecting preference collection, reward model updates, rejection sampling, and failure sample re-integration.

**Why emphasize multi-round iteration and RM retraining?**
After the initial SFT, applying RLHF with the initial RM causes the model policy to rapidly shift toward the high-scoring regions of the RM. However, the RM itself has out-of-distribution (OOD) blind spots. As model capability improves, the responses it generates gradually diverge from the sample distribution the initial RM was trained on. Without updating the RM, the model may learn to exploit reward model vulnerabilities (Reward Hacking). Therefore, in the Llama-3 approach, each iteration uses the current model to generate new responses for hard prompts, submits these to human annotation, and incorporates the new data into the RM training set for retraining. The specific scale of newly added data per round should be cited from the report; any unconfirmable figures should be marked as `[I]` or stated as "undisclosed."

**The bridging role of Rejection Sampling Fine-Tuning (RSFT):**
In engineering, Llama-3 uses RSFT as a bridge between RLHF and SFT. In each round, the system samples multiple outputs for the same prompt and then uses the latest RM or preference selection mechanism to filter and extract higher-quality responses as new pseudo-label data. This data flows back into the next round of SFT or post-training, helping to improve the baseline output distribution. Candidate counts, sampling strategies, and retention ratios should not be stated as definitive numbers in the final text if they lack a direct source.

**Data re-integration mechanism:**
The value of multi-round iteration lies in the continuous absorption of boundary data. Failure samples and boundary samples identified in each round of red-blue adversarial testing or evaluation are systematically captured. High-scoring samples can be re-integrated to reinforce positive strategies, while failure samples can be constructed as rejected samples for DPO or RM training sets, forming a continuous online optimization loop. This contrasts with Tülu-3's open reproducible recipe: the former demonstrates an industrial system's capacity for continuous iteration, while the latter provides a publicly reviewable methodological asset more readily accessible to external teams.

| Iteration Step | Data Input | Processing Action | Data Output | Annotation Convention |
| --- | --- | --- | --- | --- |
| Post-SFT initial sampling | Hard prompts, production failure cases | Multi-candidate generation | Candidate responses | Mark as `[I]` when sampling scale lacks a source |
| Preference collection | Multiple candidate responses | Human annotation or RM-assisted filtering | Chosen/rejected or ranked data | Annotation scale requires report verification |
| RM update | New preference data, historical preference data | Retrain or incrementally update RM | New RM checkpoint | Describe only the mechanism; do not fill in speculated parameters |
| Rejection sampling fine-tuning | Multiple candidates + RM scores | Select high-quality responses | Pseudo-label SFT data | Candidate count requires source support |
| Failure sample re-integration | Red team, evaluation, production logs | Classification, deduplication, risk annotation | Hard cases / rejected pool | Must specify privacy and contamination boundaries |

From a data engineering perspective, the most instructive aspect of the Llama-3 approach is "redefining the data distribution with each post-training round." After the first round of SFT, model failure cases typically cluster around basic instruction-following and safety boundaries; after several rounds of preference optimization, failure cases gradually shift to harder problems such as long-context consistency, multi-constraint tasks, factual details, tool invocation, and nuanced safety boundaries. If a team continues to train subsequent models using the first-round data distribution, they will encounter diminishing returns despite increasing investment. The core of multi-round iteration is not repetitive training, but ensuring that the data distribution moves together with the model's evolving capabilities.

This also explains why rejection sampling fine-tuning is important in industrial systems. RSFT is not simply "let the RM pick the best answer"—it searches within the current model's candidate space for high-quality trajectories, consolidating behaviors the model can occasionally produce into more consistent behaviors. It is closer to the current model's capability boundary than manually writing SFT samples from scratch, and more controllable in terms of output format and training stability than pure RL. However, the risks of RSFT are also clear: if the RM exhibits length bias, format bias, or safety over-bias, rejection sampling will amplify these biases. Therefore, RSFT must be paired with an RM audit set and manual spot-checking, rather than unconditionally treating the highest-scoring samples as ground truth.

For small and medium-sized teams that cannot fully replicate the annotation investment of Llama-3, the engineering rhythm can still be reproduced in a simplified form. A streamlined version might look like this: first, establish basic behavior with 10,000–20,000 high-quality SFT samples; then perform multi-candidate sampling on 1,000–3,000 hard prompts; use a lightweight RM, rule-based verifiers, or a small human review team to filter high-quality responses; compile the filtered results into RSFT data; after training, add the newly identified failure samples to the hard prompt pool for the next round. This scale is far smaller than an industrial system, but already has the closed-loop form of "model generation → review and filtering → data re-integration → retraining."

The privacy and compliance boundaries also warrant emphasis. Production failure samples do not automatically qualify as trainable data. User inputs may contain personal information, trade secrets, medical or financial sensitive content, or may originate from authorization contexts that prohibit use for training. Therefore, real-world multi-round RLHF or RSFT systems must first pass through desensitization, purpose authorization checks, data classification, and audit trails. Without these controls, so-called "online continuous optimization" can easily become a compliance risk vector. This point is consistent with the earlier discussions in this book on privacy compliance and DataOps.

---

## 45.7 Reward Hacking, Data Contamination, and Process Reward Data

During the post-training phase, data engineering faces several systemic risks. Without upstream controls, even substantial compute and annotation investment can yield negative returns. This section discusses three categories of easily overlooked risks.

### 45.7.1 Data-Side Manifestations of Reward Hacking

Reward Hacking is one of the most common and most underestimated risks in preference alignment. When a model's optimization objective is entirely controlled by an imperfect RM, it finds the most expedient path to high scores rather than genuinely solving user problems.
**Common manifestations:**

1. **Length Bias (Verbosity/Length Bias):** The model discovers that verbose responses tend to receive higher scores, and therefore favors generating lengthy but low-information-density content.
2. **Sycophancy:** The model learns to cater to the implicit stance of a prompt, or blindly agree with the user's erroneous views to obtain high surface-level preference scores.
3. **Safety over-generalization:** Excessive refusals in safety alignment. The model may respond to ordinary medical science communication or fictional science fiction scenarios with "I cannot assist with such requests."
4. **Spurious reasoning:** In reasoning tasks, generating responses that appear structurally complete with "first, second, third" steps, but where the steps bear no logical relationship to each other and are unverifiable.

**Data-side defensive strategies:**

* In DPO or RM datasets, deliberately construct "extremely long but useless" responses as rejected samples, paired with "short but to the point" chosen samples.
* Introduce external fact-checking APIs, code unit tests, and mathematical verifiers as hard constraints.
* Retain failing generation trajectories in data logs rather than only final successful responses, so the model can learn to avoid pitfalls.
* Periodically audit the distribution of the Reward Model's training set to prevent preference polarization.

### 45.7.2 Data Contamination in the Post-Training Phase

Ch05 already examined general methods for benchmark contamination detection. Here we focus exclusively on the particularly insidious contamination problems unique to the post-training phase.

1. **SFT mixing in evaluation set answers:** Developers inadvertently use test set ground truth directly as model input when constructing synthetic instructions.
2. **Preference hardening the evaluation style:** During annotation, annotators unconsciously assign high scores to responses that match the format of specific benchmark multiple-choice templates, hardening the evaluation set style into reward preferences.
3. **Implicit filtering contamination:** During rejection sampling, using the pass rate on external evaluation sets as a filtering signal is equivalent to leaking test set metrics into the model.
4. **Feedback loop contamination:** In online feedback systems, directly re-integrating user-submitted test set prompts into daily training tasks causes severe data leakage.

| Contamination Type | Occurrence Location | Typical Symptoms | Inspection Method | Remediation |
| --- | --- | --- | --- | --- |
| SFT contamination | Instruction synthesis, data mixing | Model exhibits unusual familiarity with benchmark questions | n-gram / embedding near-duplicate detection | Remove contaminated samples, rebuild split |
| Preference contamination | RM / DPO data | Model prefers benchmark-template-style expressions | Deduplicate both prompt and answer simultaneously | Downweight or isolate related preference pairs |
| Rejection sampling contamination | Multi-candidate filtering | Evaluation scores used to filter training samples | Inspect filtering signal sources | Prohibit evaluation sets from participating in training filtering |
| Online feedback contamination | User log re-integration | Test prompts enter training logs | Log source tagging and blacklist filtering | Establish a benchmark quarantine |

The challenge of data contamination is that it is rarely a single-point error, but rather the result of multiple compounding steps. For example, a publicly available math problem set is first used to construct SFT data; the model generates candidate responses based on this data; the RM then learns preferences based on the benchmark style of these candidates; and finally, the rejection sampling stage uses the pass rate on the same problem set as a filtering criterion. On the surface, each step appears to be normal data processing; in aggregate, the evaluation set has participated in training decisions multiple times. Post-training contamination is therefore more insidious than pretraining contamination, because it contaminates not only text content but also preference directions and filtering strategies.

An effective approach for governing such issues is to establish a benchmark quarantine. All formal evaluation sets, red-team acceptance sets, holdout sets, and critical production A/B samples must enter an isolation list. Before data enters SFT, DPO, RM, RSFT, or RLVR, it must be checked for near-duplicates against the isolation list on both the prompt side and the answer side. For code and math tasks, structural checks are also required: code tasks should check function names, test cases, comments, and reference solutions; math tasks should check problem statement numbers, variable relationships, and final answers. String-level deduplication alone is typically insufficient.

Another commonly overlooked issue is "review model contamination." Many teams use a strong model as LLM-as-a-Judge and then use the judge's outputs to construct preference data. If the judge has previously encountered certain benchmarks or training preferences, its scoring may carry those preferences into the new model. The solution is not to prohibit the use of judges entirely, but to record the judge's version, prompt, scoring rationale, and known biases, and to periodically calibrate against a human audit set. For critical data, it is best to use multiple review sources—rule-based verifiers, an LLM judge, and human spot-checks—working together, to avoid having a single review model dominate the entire preference distribution.

### 45.7.3 Process Reward Data and the Ch46 Interface

To address the problem of a correct final result but an unreliable reasoning process, industry has proposed the Process Reward Model (PRM). This chapter does not elaborate on the full implementation of PRM, but notes its data value: **reasoning models require not only rewards at the final answer dimension, but also fine-grained data management of intermediate reasoning steps, verifier states, and full rejection sampling trajectories**. This naturally leads to the topic of the next chapter: how to construct an intermediate-state data flywheel for reasoning.

The core unit of process reward data is not a complete response, but a step. A usable process reward sample must at minimum preserve the problem, the complete trajectory, step segmentation, a local judgment for each step, the final answer, the final verification result, and the error type. If only the entire chain-of-thought is stored, a subsequent PRM cannot learn at which step the reasoning begins to deviate. For math tasks, annotations can mark "algebraic transformation error," "omitted condition," or "final numerical error"; for code tasks, annotations can mark "correct algorithmic approach but boundary condition error," "insufficient computational complexity," or "unit test not covered." These fine-grained labels will give the Ch46 reasoning data flywheel more stable training signals.

Process reward data also requires distinguishing between "correct process but wrong result" and "correct result but wrong process." The former may indicate an error in the final computation step or format extraction; the latter may indicate that the model obtained the answer through guessing or pattern memorization. If the training system only checks the final answer, the second type of sample will be erroneously rewarded, and the model will learn inexplicable and potentially non-reproducible reasoning shortcuts. The combined value of PRM, RLVR, and rejection sampling lies precisely in simultaneously preserving both process signals and result signals, so that the model pursues not only answer correctness but also stable reasoning paths.

---

## 45.8 Implementation Risks, Costs, and Applicability Boundaries

Post-training involves not only technical route selection, but also cost, organizational capability, and risk control. The following summarizes common lessons learned in project implementation:

* **Limitations of SFT:** If development stops at SFT without any preference alignment, the model typically only learns a superficially professional response format. Its behavior under adversarial stress tests such as jailbreak attacks may still be unstable.
* **The quantity trap:** Pursuing sample count alone in instruction tuning not only increases compute costs, but can also amplify the data's tendency toward templatization and distributional skew.
* **Blunting of DPO signals:** When constructing DPO data, if the quality difference between chosen and rejected responses is too weak (or both are of poor quality), the contrastive loss signal in DPO becomes blunted and may even damage the model's existing linguistic capabilities.
* **Narrow RM coverage:** If the reward model's training set covers too narrow a range of prompt types, the model's behavior in production will be strongly driven by unknown reward model vulnerabilities, producing absurd outputs when faced with complex user inputs.
* **Prerequisites for using Magpie:** Seed-free self-generation methods such as Magpie should not be used directly. Since they amplify the base model's inherent distribution, they require accompanying pipelines for diversity deduplication, difficulty filtering, factuality verification, and safety cleaning.
* **RLVR boundaries:** Rule-based RLVR is suitable for verifiable tasks such as mathematical derivation and code compilation, but is not appropriate for open-ended chat, literary creation, or emotional support tasks that lack standard answers.
* **Reset cost of context migration:** When migrating post-training recipes validated on open-source models to domain-specific contexts such as healthcare or finance, the original preference data should not be reused directly. Safety boundary alignment must be redone according to domain requirements, and rigorous evaluation set isolation mechanisms must be established.

**Engineering costs that must not be overlooked:**
Advancing a full-chain post-training effort requires teams to reserve budget across several cost categories. First is the **human preference annotation cost**: building a high-consistency RM training set typically requires an annotation team with domain expertise. Second is the **synthetic data inference cost**: whether using Evol-Instruct or multi-path sampling, both consume substantial GPU time. Once the alignment stage begins, **multi-candidate sampling costs** and **Reward Model / Verifier maintenance and retraining costs** continue to accumulate. Finally, the **data audit and contamination detection cost** that runs throughout is an essential investment for ensuring the final model's trustworthiness.

| Cost Item | Primary Source | Easily Underestimated Component | Downgrade Strategy |
| --- | --- | --- | --- |
| Human preference annotation | Chosen/rejected pairs, rankings, attribute scoring | Annotation consistency training and review | Start with a small high-consistency set |
| Synthetic data inference | Self-Instruct, Evol-Instruct, Magpie | Multi-round filtering reduces effective sample rate | Reduce candidate count; prioritize core task coverage |
| Multi-candidate sampling | RSFT, RM selection, GRPO grouping | Token cost scales linearly with candidate count | Increase sampling for hard prompts; reduce for ordinary prompts |
| RM / verifier maintenance | RM retraining, rule fixes, unit test expansion | Rule loopholes can reverse-contaminate training | Establish verifier versioning and regression tests |
| Contamination detection | Benchmark quarantine, near-duplicate detection | Answer-side contamination is more insidious than prompt-side | Bidirectional deduplication on both prompt and answer |

In practice, cost control can be designed as stage-gated thresholds. The first stage allows only SFT, with the goal of validating the data schema, training scripts, and baseline evaluation—without pursuing final model performance. The second stage introduces a small-scale preference set, with the goal of validating whether DPO or RM can change clearly defined behaviors, such as reducing verbosity or repairing refusal boundaries. The third stage introduces multi-candidate sampling and RSFT, with the goal of validating whether the model can stabilize occasionally high-quality responses. Only the fourth stage considers RLVR or PRM, because by then the team has accumulated sufficient verifiers, error type annotations, and evaluation sets. Phased investment is far better at controlling risk than deploying full RLHF from the outset.

Cost optimization should account not only for GPU hours, but also for effective sample rate. If a synthetic pipeline generates 1 million samples but only 50,000 remain after format, quality, safety, contamination, and manual review checks, the true cost should be calculated based on those 50,000 samples. Similarly, if rejection sampling generates 32 candidates per problem but retains only 1, the training data appears sparse, but the inference cost has already been spent on the 31 discarded candidates. Post-training project reports should therefore simultaneously record the original generation volume, the post-filtering retained volume, the manual review pass rate, the training consumption volume, and the final evaluation gain.

Applicability boundaries must also be clearly stated. General chat assistants can rely more heavily on preference data and human review; mathematics, code, and structured tasks are better suited to incorporating verifiers; domain-specific assistants must strengthen domain expert review and compliance auditing. Open-ended tasks without standard answers should not be forcibly fitted into RLVR; highly sensitive medical and financial tasks should not rely solely on LLM-as-a-Judge. The goal of post-training data engineering is not to fit all tasks into a single paradigm, but to select the appropriate supervision signal for each task type.

---

## Chapter Summary

This chapter, through a systematic deconstruction of the core data recipes in the post-training phase, reveals several fundamental patterns in modern large model development:

1. The core of post-training data engineering has never been the meticulous crafting of individual samples, but the establishment of a tightly coordinated three-stage system comprising SFT, preference optimization, and online feedback loops.
2. The primary reference value of open-source large model recipes lies not in directly copying their data files, but in studying the data shape distributions, stage ordering, filtering mechanisms, and feedback loop logic that they disclose.
3. Through a layer-by-layer analysis of SFT, DPO, RLHF, and RLVR, we find that this conventional paradigm actually constitutes a solid transitional bridge toward next-generation RL reasoning paradigms such as the architecture to be discussed in Ch46.

**Connecting Forward:** When the reward signals upon which a system depends advance from subjective "human preference scoring" to objective, rule-based or programmatically verifiable signals, the post-training system enters the domain of the reasoning data flywheel. In this flywheel, cold-start SFT provides basic behavior, multi-path sampling generates candidate trajectories, rule-based rewards provide verification signals, rejection sampling extracts high-quality trajectories, and these flow back to form second-round SFT data. The data engineering implementation of this flywheel will be elaborated in detail in Ch46.

## References

Wang Y, Kordi Y, Mishra S, Liu A, Smith N A, Khashabi D, Hajishirzi H (2023) Self-Instruct: Aligning Language Models with Self-Generated Instructions. Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, pp 13484–13508.

Ouyang L, Wu J, Jiang X, Almeida D, Wainwright C, Mishkin P, Zhang C, Agarwal S, Slama K, Ray A, Schulman J, Hilton J, Kelton F, Miller L, Simens M, Askell A, Welinder P, Christiano P F, Leike J, Lowe R (2022) Training Language Models to Follow Instructions with Human Feedback. Advances in Neural Information Processing Systems, 35, 27730–27744.

Rafailov R, Sharma A, Mitchell E, Manning C D, Ermon S, Finn C (2023) Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. Advances in Neural Information Processing Systems, 36, 53728–53741.

Ethayarajh K, Xu W, Muennighoff N, Jurafsky D, Kiela D (2024) Model Alignment as Prospect Theoretic Optimization. Proceedings of the 41st International Conference on Machine Learning, pp 12634–12651.

Gheshlaghi Azar M, Guo Z D, Piot B, Munos R, Rowland M, Valko M, Calandriello D (2024) A General Theoretical Paradigm to Understand Learning from Human Preferences. Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, pp 4447–4455.

Grattafiori A, Dubey A, Jauhri A, Pandey A, Kadian A, Al-Dahle A, Letman A, Mathur A, Schelten A, Vaughan A, others (2024) The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.

Lambert N, Morrison J, Pyatkin V, Huang S, Ivison H, Brahman F, Miranda L J V, Liu A, Dziri N, Lyu X, Gu Y, Malik S, Graf V, Hwang J D, Yang J, Le Bras R, Tafjord O, Wilhelm C, Soldaini L, Smith N A, Wang Y, Dasigi P, Hajishirzi H (2025) Tülu 3: Pushing Frontiers in Open Language Model Post-Training. Second Conference on Language Modeling.

Yang A, Li A, Yang B, Zhang B, Hui B, Zheng B, Yu B, Gao C, Huang C, Lv C, others (2025) Qwen3 Technical Report. arXiv preprint arXiv:2505.09388.

Wang Z, Dong Y, Delalleau O, Zeng J, Shen G, Egert D, Zhang J J, Sreedhar M N, Kuchaiev O (2024) HelpSteer 2: Open-Source Dataset for Training Top-Performing Reward Models. Advances in Neural Information Processing Systems, 37, 1474–1501.

Xu C, Sun Q, Zheng K, Geng X, Zhao P, Feng J, Tao C, Lin Q, Jiang D (2024) WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions. International Conference on Learning Representations.

Xu Z, Jiang F, Niu L, Deng Y, Poovendran R, Choi Y, Lin B Y (2025) Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing. International Conference on Learning Representations.

Liu A, Feng B, Xue B, Wang B, Wu B, Lu C, Zhao C, Deng C, Zhang C, Ruan C, others (2024a) DeepSeek-V3 Technical Report. arXiv preprint arXiv:2412.19437.

Liu C Y, Zeng L, Liu J, Yan R, He J, Wang C, Yan S, Liu Y, Zhou Y (2024b) Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs. arXiv preprint arXiv:2410.18451.

Singhal P, Goyal T, Xu J, Durrett G (2024) A Long Way to Go: Investigating Length Correlations in RLHF. First Conference on Language Modeling.

Zhou K, Zhu Y, Chen Z, Chen W, Zhao W X, Chen X, Lin Y, Wen J-R, Han J (2023) Don't Make Your LLM an Evaluation Benchmark Cheater. arXiv preprint arXiv:2311.01964.

Lightman H, Kosaraju V, Burda Y, Edwards H, Baker B, Lee T, Leike J, Schulman J, Sutskever I, Cobbe K (2024) Let's Verify Step by Step. International Conference on Learning Representations.
