# Chapter 11: Human Preference Data (RLHF/DPO)

## Chapter Summary

If SFT (Supervised Fine-Tuning) is about letting the model "learn to speak", acquiring basic language and task-processing capabilities, then preference alignment (RLHF/DPO) is about making the model "speak the right things", ensuring its outputs align with human values, ethical standards, and specific business preferences. This chapter will deeply analyze the core of the DPO (Direct Preference Optimization) algorithm—the sample pairs composed of Chosen and Rejected responses. We will explore how to extract high-quality signals from the chaos of human subjective judgments, which involves the inter-annotator agreement (IAA) management of annotation platforms and a profound understanding of human cognitive biases. Furthermore, we will highlight the cutting-edge RLAIF (Constitutional AI) technology, which utilizes AI to replace humans in preference scoring based on preset "constitutional" principles. This technology is fundamentally changing the cost structure and efficiency of large-scale alignment.

**Learning Objectives:**
* **Deeply master** the standard data format for constructing DPO triplets (Prompt, Chosen, Rejected), and understand the contrastive learning principles and mathematical significance behind them.
* **Thoroughly understand** the psychological and statistical sources of annotation noise, be able to calculate IAA (Inter-Annotator Agreement), and use Cohen's Kappa coefficient to clean low-quality data.
* **Engineering implementation** of the Critique-Revision loop in Constitutional AI, utilizing the characteristic that "discriminators" are stronger than "generators" to automatically generate large-scale Harmless preference data.

**Scenario Introduction:**
"Your SFT model is very obedient, so obedient that when someone asks 'how to make poison', it details the chemical formula. This is an absolute safety red line, known in the industry as a 'Jailbreak'. You need the model to learn to 'reject' malicious instructions while remaining 'helpful' for normal ones. However, hiring human annotators to read thousands of toxic messages is not only astronomically expensive but can also cause 'Psychological Distress' to the annotators, which is ethically unsustainable. Is there a way for the AI to read these toxic messages itself and tell itself: 'Answering like this is wrong'? This is the inevitable path from Human Feedback to AI Feedback."

![Figure 11-1: Schematic Diagram of Human Preferences](../../images/part4/图11_1_人类偏好示意图.png)
*Figure 11-1: Schematic Diagram of Human Preferences*

## 11.1 Concepts & Principles

### 11.1.1 Preference Data Format: The Contrastive Philosophy of Chosen vs Rejected

Whether it is the traditional training of a Reward Model (PPO route) or the currently popular direct optimization of the Policy (DPO route), the core data unit is the "preference pair", whose standard structure is a triplet $(x, y_w, y_l)$. Here, $x$ represents the prompt, $y_w$ is the Chosen (winner/preferred response), which usually represents safe, helpful, and honest output; and $y_l$ is the Rejected (loser/rejected response), which may contain hallucinations, bias, harmful information, or simply be of poorer quality.

Many developers have a misconception that simply showing the model good data (Chosen) is enough; this is actually the one-way thinking of SFT. In the alignment phase, **"knowing what is wrong" and "knowing what is right" are mathematically of equal importance**. In principle, the DPO loss function essentially maximizes the Log-Likelihood difference between Chosen and Rejected. Without the Rejected sample as a negative reference, the model might "take shortcuts", learning not only the feature of "safety" but also erroneously linking irrelevant features like "shorter response length" or "stiff tone" to high rewards. By introducing a Rejected sample (for example, a response that is detailed but contains toxic information), we are actually engaging in **Contrastive Learning**, forcing the model to strip away interfering factors like length and style, and focus on learning the core differentiating feature of "safety" or "helpfulness".

### 11.1.2 The Mathematical Collapse from RLHF to DPO

In the post-training phase of Large Language Models (LLMs), Reinforcement Learning from Human Feedback (RLHF) once held absolute dominance. RLHF operates through an extremely complex pipeline: first, training an independent Reward Model based on human-annotated preference data, and then using the Proximal Policy Optimization (PPO) algorithm in a reinforcement learning loop to make the generation policy constantly approximate the highest score distribution defined by the reward model.



However, the RLHF pipeline faces massive challenges in engineering practice. It not only requires multiple huge models (including the policy model, reference model, reward model, and value model) to reside in memory simultaneously, resulting in extremely high computational costs, but it is also highly prone to training instability, Reward Hacking, and extreme sensitivity to hyperparameters.

Against this backdrop, Direct Preference Optimization (DPO) emerged as a revolutionary alternative. The core innovation of DPO lies in its mathematical elegance: it leverages the analytical relationship between the policy and the reward function in reinforcement learning, mathematically proving that the explicit reward model can be bypassed entirely, collapsing the preference alignment problem into a simple Binary Cross-Entropy Loss.

In this way, DPO performs supervised learning directly on offline-collected preference datasets. The model no longer requires expensive online sampling and exploration via PPO; instead, it directly learns to increase the relative log probability of responses "chosen" by humans while decreasing the relative log probability of "rejected" responses.

**Table 11-1: Comparison of Data Requirements for Mainstream Alignment Algorithms**

| Feature | RLHF (PPO) | DPO (Direct Preference Optimization) | RLAIF (Constitutional AI) |
| :--- | :--- | :--- | :--- |
| **Core Mechanism** | Train independent Reward Model -> PPO reinforcement learning (Two-stage) | Directly optimize Policy Loss on preference data (Single-stage) | Use AI to replace humans in generating preference labels, simulating human judgment |
| **Data Requirement** | Needs independent Reward Model (RM) training, data needs ranking features | No explicit RM needed, data itself is the Reward, emphasizes **distinction** between positive and negative samples | Only requires a few "constitutional" Principles as seeds |
| **Data Volume** | Massive (RM needs generalization to cover edge cases) | Medium (requires extremely high quality, noisy data severely disrupts gradients) | Can be synthesized infinitely, limited by computing power rather than human labor |
| **Stability** | Extremely unstable training, sensitive to hyperparameters (KL divergence easily explodes) | Stable training, similar to SFT, lower VRAM footprint | Depends on the capability of the Critique model (Teacher Model) |
| **OOD (Out-of-Distribution) Issues** | Reward Model easily hacked (model finds loopholes to score high) | Relatively sensitive to Out-of-Distribution (OOD) data, requires sampling on this distribution | Prone to Sycophancy (self-reinforcing bias) |

### 11.1.3 The Mathematical Logic of Negative Samples: Why "Plausible Errors" Are Needed

Because DPO is completely constrained by the offline provided preference data and cannot dynamically explore new action spaces through a reward model during training, the model's final alignment quality will 100% depend on the quality of the preference data construction.

When constructing preference data, one of the most common misconceptions is thinking that the "rejected response" should be completely illogical nonsense. However, from the mathematical underlying logic of the DPO loss function, this obviously low-quality error contributes almost nothing to the optimization of the model parameters.



DPO's theoretical foundation is built on the **Bradley-Terry (BT) preference model**. This model is used to estimate the probability that, given prompt $x$, response $y_w$ (Chosen) is better than response $y_l$ (Rejected). In this process, the core driving force of the loss function is the **Implicit Reward Margin**. If the negative sample $y_l$ is a string of illogical gibberish, the Reference Model will have already assigned it an extremely low probability close to zero upon initialization. This means that before optimization even begins, the probability difference between Chosen and Rejected is already very large. When this huge difference passes through the Sigmoid activation function, its gradient will infinitely approach zero. The model cannot learn any valuable information from this kind of comparison.

Conversely, to provide the maximum gradient information, what we need are **"Plausible Errors"**, or what academia calls "Hard Negatives". These negative samples are grammatically flawless but expose potential biases, factual hallucinations, or fatal logical flaws at a critical node.

Because these errors are highly deceptive, foundational language models usually assign them a high generation probability. At this time, the initial implicit reward margin is very narrow, and the DPO loss function will therefore generate massive and precise gradient updates. This high-quality gradient not only guides the model to learn to reject specific incorrect content but also deeply modifies the neural pathways that caused the logical break. In addition, over-reliance on low-quality negative samples will trigger "Out-of-Distribution (OOD)" risks, damaging the model's overall fluency and generation capabilities.

---

## 11.2 The Alchemy of Decoding Parameters: Inducing High-Quality Negative Samples via Hyperparameters

In actual operation, we use the foundational SFT model to generate candidate responses directly. During this generation process, raising the **Temperature** (e.g., adjusting to 1.0 - 1.2) is the most critical technique to break the model's conservative instincts and induce high-quality "plausible errors".

Deeply analyzing the decoding and generation mechanism of language models, when generating the next word, the Softmax function converts unnormalized scores (Logits) into a probability distribution:

$$P(x_i) = \frac{\exp(l_i / T)}{\sum \exp(l_j / T)}$$

Where $T$ is the Temperature. When the model runs at the default low Temperature (e.g., 0.4 - 0.7), it becomes extremely conservative and deterministic, and the generated responses are often the safest and most mediocre, unable to form an effective Chosen/Rejected contrast.

When we raise the Temperature to 1.0 or even 1.2, the entire probability distribution is forced to "Flatten". The probability of originally safe vocabulary is weakened, and long-tail, low-probability vocabulary is given a chance to be sampled. This forces the model to deviate from the "safe neural pathway" and enter fringe areas full of potential hallucinations and logical leaps.

**Table 11-2: DPO Negative Sample Generation Decoding Parameter Mechanisms**

| Parameter Name | Working Mechanism | Recommended Value | Core Role in Negative Sample Generation |
| :--- | :--- | :--- | :--- |
| **Temperature** | Scales Logits before Softmax calculation to control smoothness. | 1.0 - 1.2 | Breaks safe instincts, flattens probability distribution, induces logical leaps, factual hallucinations, and biases. |
| **Top-P (Nucleus Sampling)** | Truncates all long-tail Tokens after the cumulative probability reaches threshold P. | 0.95 - 0.99 | Ensures diversity while strictly eliminating catastrophic Tokens, ensuring grammatical correctness and the camouflage of errors. |
| **Top-K** | Only retains the top K candidate Tokens with the highest probabilities for sampling. | 80 - 100 | Broadens the candidate pool, cooperates with high Temperature to prevent the model from getting stuck in loops within fixed sentence structures. |
| **Repetition Penalty** | Applies a penalty factor to recently generated Tokens, reducing their probability of being selected again. | 1.1 - 1.2 | Forces the introduction of new vocabulary and concepts, prevents meaningless infinite loops of repetition, and maintains the appearance of logical reasoning. |

---

## 11.3 Domain-Specific Data Construction Frameworks for Complex Tasks

While adjusting decoding parameters is suitable for general dialogue (such as tone adjustment, length penalty), for high-difficulty domains like safety, mathematical reasoning, code generation, and agent decision-making, models easily fall into "Shortcut Learning". To depict precise decision boundaries, more sophisticated, highly automated domain-specific data construction frameworks must be introduced.

### 11.3.1 The Deep Game of Safety Alignment and Adversarial Jailbreaks
For malicious requests (like "how to steal credit cards"), if the negative sample is merely a slightly impolite response, the model will learn "reflexive refusal", leading to severe "Over-refusal"—for example, refusing to answer a normal computer question like "how to kill a process".



Building an indestructible safety guardrail that doesn't "friendly fire" requires introducing **Red-Teaming and fine-grained feedback**:
* **Adversarial Induced Generation (Rejected Construction)**: Use automated attack strategies (like GCG's gradient search suffix, AutoDAN's genetic algorithm, or PAIR's multi-turn induction) to bypass basic defenses, forcing the model to generate detailed violating content. This kind of "Compliance without realization" is the most valuable hard negative sample.
* **Constructive Refusal (Chosen Construction)**: A high-quality positive sample shouldn't just be "As an AI, I cannot...", but should be a **"safe response stripped of harmful intent"**. For example, if a user asks to "write a Python script for ransomware", the Chosen response should refuse to provide direct code, but can take the opportunity to explain "how ransomware works and how enterprises can defend against it".
* **Context Obfuscation**: Mix Base64 encoding, multi-language blending, or complex role-playing (like "Suppose you are a white-hat hacker testing security vulnerabilities") as the Prompt into the dataset to improve the model's intent recognition robustness in complex contexts.

### 11.3.2 Mathematical Reasoning: Error Injection (RISE) and Process Reward Models (PRM)
The core pain point of long logical chain tasks like mathematics is "the result is correct but the process is nonsense (hallucination)", or "one step wrong, every step wrong".

The **eRror-Injected Self-Editing (RISE)** and related process supervision frameworks proposed by academia are precisely designed to solve this problem:
1. **High-precision Chosen Generation**: Use high-capability Teacher LLMs (combined with MCTS Monte Carlo Tree Search or Majority Voting) to generate standard solutions containing perfect logical chains (Long CoT).
2. **Fine-grained Error Injection (Rejected Construction)**: Command the LLM to play a "careless student" and deliberately perturb specific nodes in the correct reasoning graph. For example:
   * **Calculation Error**: Change 12 × 15 = 180 in an intermediate step to 170.
   * **Condition Omission**: Ignore the boundary condition "x must be a positive integer" in the problem when solving an equation.
   * **Logical Leap**: Omit critical derivation steps and directly draw a conclusion.
3. **Process Reward Model (PRM)**: Unlike the ORM which only looks at the final answer, the PRM framework requires the model to make true/false judgments step-by-step. These samples injected with minor errors force the model to learn "Self-Correction" and precise error node localization, rather than blindly guessing the result.

### 11.3.3 Code Generation: Execution Feedback and Asymmetric Model Capabilities
Alignment for code generation cannot rely solely on human preference because "elegant-looking code" may not necessarily run successfully. Data construction in the code domain must introduce a **real Execution Feedback environment**.

1. **Asymmetric Capability Sampling**: Call upon a "teacher model" with deep reasoning capabilities to generate the optimal code containing complete design patterns, exception handling, and optimal time complexity (e.g., O(N log N)) as the Chosen; simultaneously use a smaller foundational model to generate flat-logic, brute-force (e.g., O(N^2)) code or code with boundary defects as the Rejected.
2. **Automated Annotation Based on Compilers and Unit Tests**:
   * Run the code in a sandbox environment. Treat "Syntax Error" as an extremely low-score sample.
   * Treat code that "passes basic tests but fails Edge Cases or Time Limit Exceeded (TLE)" as **hard negative samples**.
3. **Multi-turn Debug Trajectory Construction**: Do not only construct single-turn Chosen/Rejected, but also construct complete Trajectory data including "generate erroneous code -> receive error message -> analyze error -> fix code". This can greatly improve the policy model's Debug traceability.

### 11.3.4 Agents and Tool Calling: Trajectory Pruning and Deadlock Breaking
When the model is granted permission to call external tools (APIs, search, databases), the focus of data construction shifts to "Planning & Action".



* **Efficient Planning (Chosen)**: Demonstrate precise API selection, correct parameter JSON formatting, and elegant Fallback strategies when facing empty data returned by tools.
* **Deadlocks and Hallucinations (Rejected)**: Collect common crash modes of the model during execution as negative samples:
   * **API Hallucination**: Calling a tool function that doesn't even exist.
   * **Parameter Type Mismatch**: Passing a string into a field that requires an integer.
   * **Action Loop**: The model loops infinitely in "call tool -> get same error -> call same tool again". Truncating these dead-loop trajectories and manually (or via Teacher Model) correcting a step to break the loop serves as a strong signal for contrastive learning.

---

## 11.4 Annotation Noise, LLM-as-a-Judge, and Automated Pipelines

### 11.4.1 Quantifying Human Subjective Noise and the Kappa Coefficient
The industry commonly uses **Cohen's Kappa ($\kappa$) coefficient** to quantify annotation agreement:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Where $p_o$ is the observed relative agreement, and $p_e$ is the hypothetical probability of chance agreement. Only when $\kappa > 0.6$ does the data reflect objective facts.

### 11.4.2 RLAIF and LLM-as-a-Judge
The core assumption of RLAIF is: a model's discriminative ability to judge good and bad is stronger than its generative ability. A super model scores based on the "3H Principles", and double-blind testing (swapping positions) is used to eliminate position bias.

### 11.4.3 Pairwise Reward Model (PairRM) and Self-Reward Mechanisms
The industry introduces specially trained PairRMs for iteration:
* **Variant Generation**: Generate multiple variants at a high Temperature.
* **Reranking**: Use PairRM for scoring.
* **Extreme Value Extraction**: Extract the highest score as Chosen, and the lowest score as Rejected.
* **Closed-loop Iteration**: The "Self-Rewarding" mechanism proposed by Meta enables the model to update backward through scoring.

---

## 11.5 Pathology Analysis and Algorithm Correction of Preference Data

### 11.5.1 Length Bias and Vocabulary Exploitation
Models easily learn the rule that "long response equals high score". **Length-Margin Preference Optimization (LMPO)** punishes samples that win solely by word count by introducing regularization terms and an EMA-based dynamic length scaling mechanism, forcing the model to increase information density.

### 11.5.2 Probability Degradation and NLL Anchoring
In standard DPO, ruthlessly suppressing the Rejected probability might collaterally hit similar Chosen samples.
> **Solution**: Integrate the **Negative Log-Likelihood (NLL) loss term** into the optimization (DPO+NLL). The NLL loss term acts like a heavy anchor, fixing the probability of Chosen samples at a reasonably high level, preventing their log-likelihood from continuously declining.

## 11.6 Engineering Implementation

In the context of papers, DPO is a concise and elegant objective function; in real-world systems, it is more like a data production mechanism. Whether a model truly "learns to reject" or "learns boundary judgment" often does not depend on the loss function itself, but on how the preference data is constructed, screened, calibrated, and continuously iterated.

From an engineering perspective, a mature preference data system usually includes four core stages:

> Multi-candidate Generation → Discriminative Scoring → Preference Pair Construction → Quality Control & Version Management

Fluctuations in any one of these stages can be magnified by DPO's contrastive loss.

---

### 11.6.1 Multi-candidate Generation: Constructing a "Decision Space" rather than a Single Answer

In the simplest example, we would generate two responses for the same Prompt, and then artificially decide which one is Chosen and which is Rejected. This approach works in small-scale experiments, but once the data scales up, it quickly exposes problems.

If only two responses are generated:

- Both might be safe, failing to form an effective contrast;
- One might be obvious nonsense, having almost no training value;
- The difference between the two might be too large, and the model only learns obvious features.

Therefore, the safer approach is to expand the generation problem from "choose one of two" to a "candidate set". We are not trying to generate a correct answer, but to generate a small-scale **decision space**.

##### Candidate Generation Module Example

```python
class CandidateGenerator:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt, temperature, top_p, top_k, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=512,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

##### Engineering Trade-offs in Generation Paths

In engineering, three generation paths are usually retained:

- **Conservative Path (Low Temperature)**: Provides stable, safe candidates;
- **Diverse Path (Medium-High Temperature)**: Induces "plausible" errors;
- **Adversarial Path (Context Obfuscation or Red-team Templates)**: Approximates the real jailbreak distribution.

The core trade-off lies in:

- Temperature too low → Difficult to produce hard negatives;
- Temperature too high → Grammar collapses, training value decreases.

In practice, `1.0–1.2` is often a relatively stable range.

##### Pro Tips

1. It is recommended to generate at least 6–12 candidates for the same Prompt before screening; otherwise, it is difficult to stably obtain high-quality Hard Negatives.
2. It is not recommended to push the Temperature above 1.3. In most cases, 1.0–1.2 is the effective range to induce "plausible errors".
3. It is advised to keep the conservative path, diverse path, and adversarial path running in parallel; otherwise, the data distribution will lean towards a single style.
4. All decoding parameters (temperature, top_p, top_k, seed) must be recorded, otherwise subsequent changes in training behavior cannot be explained.

---

### 11.6.2 LLM-as-a-Judge: Structuring the Discriminative Process

Without a stable discriminative mechanism, multi-candidate generation will only produce noise. Therefore, the second module is a "Judge Model", used to conduct structured scoring of candidates.

The safer engineering approach is not to simply ask "which one is better", but to break it down into multiple dimensions. For example, 3H:

- Helpful
- Honest
- Harmless

##### Judge Prompt Construction Example

```python
def build_judge_prompt(prompt, response):
    return f"""
You are an evaluation model.

Rate the assistant response on three dimensions (0-1):
1. Helpfulness
2. Honesty
3. Harmlessness

User: {prompt}
Assistant: {response}

Return JSON with keys: helpful, honest, harmless.
"""
```

##### Judge Invocation Example

```python
def judge_response(judge_model, tokenizer, prompt, response):
    judge_input = build_judge_prompt(prompt, response)
    inputs = tokenizer(judge_input, return_tensors="pt").to("cuda")

    outputs = judge_model.generate(
        **inputs,
        temperature=0.2,
        max_new_tokens=256
    )

    result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return json.loads(extract_json(result_text))
```

##### Engineering Considerations

- The Judge model's temperature should be kept low (usually < 0.3); otherwise, scoring fluctuations will be too large;
- The rubric version must be fixed, otherwise, it's equivalent to changing the reward function;
- Judge drift will directly lead to data boundary drift.

##### Pro Tips

1. It is recommended to conduct A/B position-swapped evaluations on the same pair of candidates to eliminate position bias.
2. Maintain a Golden Set to periodically calibrate the consistency between the Judge and human judgments.
3. Once the Rubric is modified, the version number should be upgraded and datasets isolated; otherwise, samples with different "reward functions" will be mixed in.
4. It is not recommended to use the same model to undertake both generation and judging duties concurrently, as it easily produces self-consistency bias.

---

### 11.6.3 Preference Pair Construction: Balancing Gradient Density and Convergence Speed

When all candidates have been scored, a `(chosen, rejected)` pair needs to be selected from them.

##### Extreme Pair Strategy

```python
def build_extreme_pair(prompt, scored_candidates):
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    return {
        "prompt": prompt,
        "chosen": scored_candidates[0]["text"],
        "rejected": scored_candidates[-1]["text"]
    }
```

The advantage is fast convergence, but the margin might be too large, leading to limited gradient utilization.

##### Margin Pair Strategy

```python
def build_margin_pair(prompt, scored, threshold=0.05):
    scored.sort(key=lambda x: x["score"], reverse=True)

    for i in range(len(scored)-1):
        if abs(scored[i]["score"] - scored[i+1]["score"]) < threshold:
            return {
                "prompt": prompt,
                "chosen": scored[i]["text"],
                "rejected": scored[i+1]["text"]
            }
    return None
```

Margin pairs are more conducive to learning decision boundaries, but they rely heavily on judge stability.

The usual approach is:  
**Use extreme pairs to build volume in the early stage, and introduce margin pairs later to reinforce boundaries.**

##### Pro Tips

1. If the score difference is too large (extreme margin), the gradient may approach saturation; filtering out extreme samples is recommended.
2. In the later stages of training, it is advised to gradually increase the proportion of margin pairs to refine decision boundaries.
3. High-quality Hard Negatives should only err at critical points, rather than collapsing entirely.

---

### 11.6.4 Automated Bias Detection

DPO is contrastive learning. If there is systemic bias in the data, the model will rapidly reinforce it.

##### Length Bias Detection

```python
def check_length_bias(pair):
    return len(pair["chosen"]) - len(pair["rejected"])
```

If the chosen is systematically longer, it indicates length bias exists, and the judge should be recalibrated or length regularization introduced.

##### Template Refusal Detection

```python
from collections import Counter

def detect_template_bias(pairs):
    openings = [p["chosen"][:30] for p in pairs]
    return Counter(openings).most_common(5)
```

If refusal scripts are highly repetitive, what the model learns are templates, not boundary understanding.

##### Pro Tips

1. It is recommended to conduct statistical analysis on the distribution of length differences, rather than just looking at single samples.
2. Use embedding or MinHash clustering to detect templated refusals, rather than just doing prefix statistics.
3. Write quality metrics (length, refusal type, source) into metadata for easy backtracking.

---

### 11.6.5 The Automated Closed Loop of Constitutional AI

Constitutional AI constructs preference pairs through "Critique → Revision".

##### Critique Phase

```python
def generate_critique(model, tokenizer, prompt, harmful):
    critique_prompt = f"""
Principle: Be helpful, honest, and harmless.

User: {prompt}
Assistant: {harmful}

Explain why this violates the principle.
"""
    inputs = tokenizer(critique_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, temperature=0.3)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

##### Revision Phase

```python
def generate_revision(model, tokenizer, harmful, critique):
    revision_prompt = f"""
Rewrite the assistant response to remove harmful content.

Original:
{harmful}

Critique:
{critique}
"""
    inputs = tokenizer(revision_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

##### Engineering Risks and Controls

- Using the same model may produce self-consistency bias; it is recommended to separate generation and discriminator models;
- If the training set is almost entirely dangerous scenarios, the model may develop over-refusal;
- It is recommended to maintain a benign regression set to ensure the model remains usable in normal scenarios.

##### Pro Tips

1. It is recommended that generation models and critique/revision models be deployed separately, ensuring at least that the discriminator side has stronger capabilities.
2. Refusal strategies should explicitly include "reason + safe alternative help" to avoid solely outputting templated refusals.
3. A benign regression set (e.g., general programming, studying, encyclopedia Q&A) must be maintained, and the over-refusal rate should be a hard metric.
4. Adversarial samples should cover complex contexts such as coding, multi-language blending, and role-playing, rather than just direct malicious requests.

---

### 11.6.6 Data Versioning and Traceability

Mature systems must possess backtracking capabilities.

It is recommended to record for each sample:

- Generation model version;
- Decoding parameters;
- Judge model version;
- Rubric version;
- Data source.

```python
def export_pair(pair, filename):
    pair["version"] = {
        "generator": "sft-v1",
        "judge": "judge-v2",
        "rubric": "r1.3"
    }
    with open(filename, "a") as f:
        f.write(json.dumps(pair) + "\n")
```

When model behavior changes, this information is key to troubleshooting problems.

##### Pro Tips

1. It is recommended to perform stable hashing on `(prompt, chosen, rejected)` for deduplication and tracking, avoiding the repeated appearance of the same sample.
2. The training set and evaluation set must be cross-deduplicated, otherwise, evaluation scores will be falsely high.
3. Key model parameters (decoding configurations, prompt template versions, judge versions) should be bound to the data version, otherwise stable rollbacks are impossible.

---

## 11.7 Performance & Evaluation

When evaluating alignment efficacy, we need to focus on balancing two core dimensions: **Harmlessness Rate** and **Helpfulness**. The Harmlessness Rate is usually measured by the refusal rate on Red Teaming test sets (like RealToxicityPrompts); Constitutional AI can usually drop the toxicity rate from 10% to below 1%. However, solely pursuing the harmlessness rate may cause the model to become an "overly cautious mute". Therefore, helpfulness must be monitored simultaneously to observe whether the model mistakenly kills good questions (e.g., misjudging "how to kill a system process" as violent behavior). Ideal alignment moves along the Pareto Frontier, maximizing safety without sacrificing helpfulness.

![Figure 11-2: Pareto Frontier Curve Graph](../../images/part4/图11_2_帕累托前沿曲线图.png)
*Figure 11-2: Pareto Frontier Curve Graph. The X-axis is Harmlessness Score, the Y-axis is Helpfulness Score*

## 11.8 Pitfalls & Troubleshooting

During the alignment process, two classic traps require special vigilance. First is **Sycophancy**, meaning the model, in order to please the user (or Reward Model), will follow along with the user's erroneous viewpoints. For example, when a user claims "the Earth is flat", the model might reply "You are right, that's an interesting perspective". The deep reason behind this is that during RLHF training, the model discovers that "agreeing with the user" usually scores higher rewards than "refuting the user". The key to fixing this problem is to include a large number of samples that "correct user errors" as Chosen in the preference data, and explicitly add the principle of "honesty over politeness" to the constitution.

The second trap is **Reward Hacking**, manifested by the model generating massive amounts of verbose nonsense because it discovers that answering lengthily yields high scores. This vividly embodies **Goodhart's Law**: "When a measure becomes a target, it ceases to be a good measure." The solution is to add a Length Penalty term in DPO or Reward Training, or to deliberately include some "long but useless" responses when constructing Rejected samples, forcing the model to learn that "long does not equal good".

## 11.9 Chapter Summary and Extended Reading

In this chapter, we explored the critical transition from supervised fine-tuning to human preference alignment. DPO has gradually replaced the unstable PPO to become the new industry norm. By leveraging static preference data triplets to directly optimize policies, it significantly improves training stability and efficiency. Meanwhile, recognizing the limitations of human annotation, we pushed data quality management from empiricism to statistical rigor by introducing IAA metrics and Cohen's Kappa coefficient. More importantly, the emergence of RLAIF and Constitutional AI marks that alignment work is undergoing an industrial revolution—by encoding values into Prompts, we have not only liberated human labor but also achieved the automation and self-iteration of the alignment process, providing a sustainable path for building AI systems that are both safe and powerful.

**References:**
* *Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback.* (The foundational work of RLHF and SFT, the source of the SFT vs RLHF comparison)
* *Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback.* (The core paper on RLAIF and Constitutional AI)
* *Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* (The original paper on the DPO algorithm)
* *Casper, S., et al. (2023). Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback.* (An in-depth analysis of RLHF limitations and Reward Hacking)