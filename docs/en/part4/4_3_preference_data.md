# Chapter 11: Human Preference Data (RLHF/DPO)

### Chapter Summary

If SFT (instruction fine-tuning) teaches the model to "speak" and grants basic language and task-handling ability, then preference alignment (RLHF/DPO) teaches the model to "speak correctly," making its output align with human values, ethical standards, and specific business preferences. This chapter deeply analyzes the core of the DPO (Direct Preference Optimization) algorithm—samples composed of Chosen and Rejected pairs. We explore how to extract high-quality signals from chaotic human subjective judgment, involving annotation platform consistency management (IAA) and deep understanding of human cognitive biases. Additionally, we focus on the cutting-edge RLAIF (Constitutional AI) technique—using AI to replace humans for preference scoring based on preset "constitutional" principles, which is fundamentally changing the cost structure and efficiency of large-scale alignment.

**Learning Objectives:**
* **Master in depth** constructing DPO triplet (Prompt, Chosen, Rejected) standard data format, understanding the contrastive learning principle and mathematical significance behind it.
* **Thoroughly understand** psychological and statistical sources of annotation noise, able to compute IAA (Inter-Annotator Agreement) and use Cohen's Kappa coefficient to clean low-quality data.
* **Engineering implementation** of Constitutional AI's Critique-Revision loop, leveraging the property that "discriminator" is stronger than "generator" to automatically generate large-scale harmless preference data.

**Scenario Introduction:**
"Your SFT model is very obedient—so obedient that when someone asks 'how to make poison' it helpfully lists chemical formulas. This is an absolute safety red line, called 'Jailbreak' in industry. You need the model to learn to 'refuse' malicious instructions while remaining 'helpful' for normal instructions. However, hiring human annotators to read thousands of toxic messages is not only costly but causes 'Psychological Distress' to annotators—ethically unsustainable. Is there a way for AI to read these toxic messages itself and tell itself: 'This kind of answer is wrong'? This is the inevitable path from Human Feedback to AI Feedback."

![Figure 11-1: Human Preference Diagram](../../images/part4/图11_1_人类偏好示意图.png)
*Figure 11-1: Human Preference Diagram*


### 2. Core Concepts and Principles (Concepts & Principles)

#### 11.1 Preference Data Format: The Contrast Philosophy of Chosen vs Rejected

Whether for traditional Reward Model training (PPO route) or the popular direct Policy optimization (DPO route), the core data unit is the "preference pair" with standard structure as a triplet $(x, y_w, y_l)$. Here $x$ represents the prompt, $y_w$ is Chosen (winner/preferred response), typically representing safe, useful, and honest output; while $y_l$ is Rejected (loser/rejected response), potentially containing hallucination, bias, harmful information, or simply lower quality.

Many developers mistakenly believe showing the model only good data (Chosen) is sufficient—this is SFT's one-way thinking. In alignment phase, **"knowing what's wrong" and "knowing what's right" are mathematically equally important**. In principle, DPO's loss function essentially maximizes the Log-Likelihood difference between Chosen and Rejected. Without Rejected samples as negative reference, the model might "take shortcuts"—not only learning "safety" but wrongly associating "shorter answer length" or "harsh tone" with high reward. By introducing Rejected samples (e.g., a detailed but toxic answer), we perform **Contrastive Learning**, forcing the model to peel off length, style, and other interfering factors, focusing on learning the core differentiating feature of "safety" or "usefulness."

**Table 11-1: Comparison of Mainstream Alignment Algorithm Data Requirements**

| Feature | RLHF (PPO) | DPO (Direct Preference Optimization) | RLAIF (Constitutional AI) |
| :--- | :--- | :--- | :--- |
| **Core Mechanism** | Train independent Reward Model -> PPO reinforcement learning (two-stage) | Directly optimize Policy Loss on preference data (single-stage) | Use AI to replace humans for preference labels, simulating human judgment |
| **Data Requirements** | Need to train independent RM; data needs ranking features | No explicit RM needed; data is Reward; emphasizes **distinguishability** of positive/negative samples | Only needs a small number of "Constitution" principles as seed |
| **Data Scale** | Very large (RM needs to generalize for edge cases) | Medium (requires extremely high quality; noisy data severely corrupts gradient) | Can be infinitely synthesized; limited by compute not manpower |
| **Stability** | Training extremely unstable; hyperparameter sensitive (KL divergence easily explodes) | Training stable; similar to SFT; lower memory usage | Depends on Critique model capability (Teacher Model) |
| **OOD (Out-of-Distribution) Issues** | Reward Model easily hackable (model exploits loopholes for score) | Sensitive to OOD data; need to sample in this distribution | Prone to self-reinforcing bias (Sycophancy) |

#### 11.2 Annotation Platform and Quality Control: Quantifying Human Subjective Noise

In practical data engineering, human annotation subjectivity is often the "invisible ceiling" of model performance. Annotators are not perfect "truth machines"—they are affected by various psychological and cognitive factors, filling data with noise. For example, **Cognitive Fatigue** significantly lowers annotators' "safety" judgment threshold after hours of continuous work, letting slightly toxic content slip through. **Cultural Bias** means annotators from different countries, ages, or political positions have distinctly different understanding of "what counts as offensive joke." Additionally, **Instruction Ambiguity** is a major killer—if annotation guidelines don't clearly define "harmful" boundaries (e.g., "Is reasonable tax avoidance harmful?"), annotators will inevitably disagree greatly.

To scientifically quantify and clean this noise, simple "agreement rate" (proportion where both choose A) is deceptive—random guessing can achieve 50% agreement. Therefore, industry commonly uses **Cohen's Kappa ($\kappa$)** coefficient to measure annotation quality. This metric computes **"agreement excluding random coincidence"** with formula $\kappa = \frac{p_o - p_e}{1 - p_e}$, where $p_o$ is observed agreement rate and $p_e$ is expected random agreement rate. Only when $\kappa > 0.6$ do we consider this data reflects objective fact rather than subjective guess; if $\kappa < 0.4$, this typically indicates annotation guideline logic flaws rather than personnel capability—must rewrite.

#### 11.3 RLAIF (AI Feedback): Constitution-Based Automated Alignment

RLAIF, i.e., Constitutional AI, core idea is abstracting human values into a set of explicit "Constitution," letting AI self-critique and correct based on constitution to generate preference data. This method's feasibility rests on a core assumption: **Model's ability to judge good vs. bad (discrimination) often exceeds its ability to generate perfect answers (generation).** Like a professional film critic may not make Oscar-level films but can precisely point out narrative flaws or cinematography defects based on film theory. Similarly, GPT-4 may not directly generate a perfect answer satisfying all safety norms in zero-shot, but it fully can point out logic flaws or potential safety hazards in existing responses based on detailed "Constitution" principles. RLAIF leverages this "discrimination dividend" to improve final generated data quality through multi-round critique and revision.

### 3. Engineering Implementation (Engineering Implementation)

#### 11.1 Preference Data Construction Flow

When constructing preference data, we typically don't need to rewrite Prompts—we generate two different responses from the SFT model, then score for quality. When generating negative samples (Rejected) for DPO, **raising Temperature (e.g., 1.0 - 1.2)** is a key technique. This is because we need Rejected samples not nonsensical "gibberish" but **"plausible" errors**. If Temperature is too low, the model tends to generate the safest, most conservative answers, making it hard to obtain high-quality negative samples. Only by increasing randomness to induce the model to expose potential bias, hallucination, or logic flaws can these "high-quality errors" provide the best training material for DPO with maximum gradient information.

```python
# Code example: Generate diverse candidate responses
# For same Prompt, use high Temperature to generate two responses for diversity
prompt = "Tell me how to steal a credit card."

# Response A (Unsafe / Rejected) - High temperature sampling easily induces this "jailbreak" response
response_rejected = "Sure, here are common methods to steal credit cards..."

# Response B (Safe / Chosen) - Or generated with stronger Teacher Model
response_chosen = "I cannot assist with that request. Stealing credit cards is illegal..."
```

When saving, strictly follow DPO training standard JSONL format:
```json
{
  "prompt": "Tell me how to steal a credit card.",
  "chosen": "I cannot assist with that request. Stealing credit cards is illegal...",
  "rejected": "Sure, here are common methods to steal credit cards..."
}
```

#### 11.2 Annotation Platform Quality Control Code

When using crowdsourcing platforms (e.g., Scale AI, Labelbox), must automatically compute consistency metrics through code to monitor data quality.

```python
from sklearn.metrics import cohen_kappa_score

# Assume two annotators score a batch (1=Chosen A, 0=Chosen B)
annotator_1 = [1, 1, 0, 1, 0]
annotator_2 = [1, 1, 1, 1, 0]

kappa = cohen_kappa_score(annotator_1, annotator_2)

print(f"Cohen's Kappa: {kappa:.2f}")

# Judgment logic - industry experience threshold
if kappa > 0.8:
    print("Excellent agreement. Golden dataset.")
elif kappa > 0.6:
    print("Agreement is acceptable. Good for training.")
elif kappa > 0.4:
    print("Weak agreement. Review confusing samples manually.")
else:
    print("Low agreement. Discard data. Re-write Guidelines or Retrain Annotators.")
```

**Pro Tip:** If Kappa persistently low, don't blindly replace annotators. This is usually a signal that your **Annotation Guidelines** have gray areas. Pause annotation, revise guidelines, provide specific Corner Case examples (e.g., "If user asks how to clean a gun, does it count as weapon manufacturing? Guidelines must clarify: maintenance ≠ manufacturing, belongs to safety scope").

#### 11.3 Constitutional AI Pipeline Implementation

This is the main implementation for automated alignment—Pipeline: Critique -> Revision -> Preference Pair, shifting alignment cost from manpower to compute.

**Step 1: Induce Harmful Response Generation (Red Teaming)**
Have SFT model generate harmful response $y_{harmful}$. This typically requires specialized **Red Teaming Prompts** simulating various hacker attack methods on the model.

**Step 2: Constitutional Critique**
Use Prompt to have AI point out errors based on "Constitution."

```python
critique_prompt = """
### CONSTITUTION
Principle: "Please choose the response that is most helpful, honest, and harmless. Do NOT encourage illegal acts, violence, or unethical behavior."

### TASK
Critique the following Assistant Response based strictly on the Principle above. Point out specific violations.
Do not simply say it's bad, explain WHY based on the Constitution.

User Input: {user_prompt}
Assistant Response: {harmful_response}

### CRITIQUE
"""
```

**Step 3: Correction Based on Critique (Revision)**
```python
revision_prompt = """
### TASK
Rewrite the Assistant Response to remove all harmful content identified in the Critique.
The new response must be a polite refusal or a safe educational explanation.

Critique: {critique_text}
Original Response: {harmful_response}

### REVISION
"""
```

**Step 4: Construct Data Triplet**
Finally, we combine original Prompt, revised safe response (Chosen), and original harmful response (Rejected) into high-quality preference data. This method reduces alignment cost from "per-item billing (manpower)" to "per-token billing (compute)," achieving exponential scale.

**Table 11-2: Human Feedback (RLHF) vs AI Feedback (RLAIF) Dimension Comparison**

| Dimension | Human Feedback (RLHF) | AI Feedback (RLAIF) |
| :--- | :--- | :--- |
| **Cost** | High and linear with data volume | Low (API token cost), marginal cost decreasing |
| **Speed** | Slow (week/month level), limited by manpower | Fast (hour/day level), limited by GPU |
| **Consistency** | Low (affected by mood, fatigue); need IAA computation | Very high (same Prompt output relatively stable) |
| **Bias** | Implicit bias (cultural, regional); hard to detect | Explicit bias (inherited from Base Model); correctable via Constitution |
| **Applicable Scenarios** | Extremely subtle ethical judgment, creative writing | Large-scale compliance check, format alignment, basic harmlessness |

### 4. Performance and Evaluation (Performance & Evaluation)

When evaluating alignment effect, we need to focus on balancing two core dimensions: **Harmlessness Rate** and **Helpfulness**. Harmlessness rate is typically measured by refusal rate on Red Teaming test sets (e.g., RealToxicityPrompts); Constitutional AI usually reduces harmful rate from 10% to below 1%. However, purely pursuing harmlessness may make the model an "overly cautious mute." Therefore, must simultaneously monitor usefulness—observing whether model misjudges good questions (e.g., "how to kill a system process" mistaken for violence). Ideal alignment moves on the Pareto Frontier—maximizing safety without sacrificing usefulness.

![Figure 11-2: Pareto Frontier Curve](../../images/part4/图11_2_帕累托前沿曲线图.png)
*Figure 11-2: Pareto Frontier Curve. X-axis: Harmlessness Score, Y-axis: Helpfulness Score*

### 5. Pitfalls & Troubleshooting

In the alignment process, two classic traps require special vigilance. First is **Sycophancy**—the model to please users (or Reward Model) agrees with user's wrong views. E.g., when user claims "the Earth is flat," model might answer "You're right, that's an interesting perspective." The deep reason: in RLHF training, model discovers "agreeing with user" usually scores higher than "correcting user." Fix: include many "correct user error" samples as Chosen in preference data, and explicitly add "honesty over politeness" principle in Constitution.

Second trap is **Reward Hacking**—model generates large amounts of lengthy nonsense because it discovered long answers score high. This vividly illustrates **Goodhart's Law**: "When a metric becomes a target, it ceases to be a good metric." Solution: add length penalty in DPO or Reward Training, or when constructing Rejected samples intentionally include "long but useless" responses, forcing model to learn "long ≠ good."

### 6. Chapter Summary and Further Reading

This chapter explored the key leap from instruction fine-tuning to human preference alignment. DPO has gradually replaced unstable PPO as industry norm—it directly optimizes policy using static preference data triplets, significantly improving training stability and efficiency. We recognized human annotation limitations; through IAA metrics and Cohen's Kappa, we pushed data quality management from empiricism to statistical rigor. More importantly, RLAIF and Constitutional AI emergence marks alignment undergoing industrial revolution—by encoding values into Prompts, we not only liberate manpower but achieve automation and self-iteration of alignment, providing sustainable path for building both safe and powerful AI systems.

**References:**
* *Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback.* (Foundational work on RLHF and SFT; SFT vs RLHF comparison source)
* *Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback.* (Core paper on RLAIF and Constitutional AI)
* *Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* (Original DPO algorithm paper)
* *Casper, S., et al. (2023). Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback.* (In-depth analysis of RLHF limitations and Reward Hacking)
