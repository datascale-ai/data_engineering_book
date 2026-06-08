# Chapter 33: Labeling, Synthesis, and Evaluation Agents

## Chapter Abstract

Labeling and evaluation are two of the most human-intensive stages in LLM data engineering. Labeling requires many human judgments. Evaluation requires carefully designed test cases and consistency calibration. As model iteration accelerates and data needs diversify, fully manual labeling and evaluation quickly become bottlenecks.

Labeling, synthesis, and evaluation agents are not meant to replace annotators and evaluators. They amplify human capacity in three ways: reducing cognitive load for each annotation, expanding long-tail coverage through synthetic data, and accelerating feedback loops through automated evaluation.

This chapter starts with labeling-assistance agents, covering task explanation, example recommendation, gray-zone judgment, and conflict arbitration. It then moves to synthetic data agents, covering seed expansion, prompt generation, difficulty control, and verifier calls. Next it discusses evaluation and red-team agents that generate challenge sets and adversarial samples. Finally, it explains consistency calibration: how to avoid the self-congratulatory loop where agents evaluate data produced by agents. The chapter builds on Chapters 12-17, upgrading SFT, preference data, annotation, synthetic data, and collapse governance into agent-driven workflows.

## 33.0 Learning Objectives

After reading this chapter, you should be able to:

- Design labeling-assistance agents for task explanation, example recommendation, gray-zone handling, and annotator calibration.
- Build seed expansion, prompt generation, and difficulty-control pipelines for synthetic data agents.
- Explain how evaluation agents generate challenge sets, failure cases, and red-team samples.
- Calibrate agent judges against human judges and avoid automated evaluation self-delusion.
- Evaluate the return on investment of labeling and evaluation agents in real workflows.

## Scenario: Annotation Avalanche and Distorted Evaluation

A conversational model team labels 5,000 new samples every week and maintains a 2,000-question evaluation set. The team has 15 people: 10 annotators, three evaluation designers, and two data engineers. Three problems appear at the same time.

**Annotation avalanche.** The model moves from v2.1 to v2.3 and adds code generation and long-document summarization. Weekly annotation demand jumps from 5,000 to 12,000 samples. Annotators work overtime, but quality falls. Fatigue leads to more arbitrary decisions in gray-zone cases, and inter-annotator agreement (IAA) drops from 0.88 to 0.72.

**Evaluation distortion.** The evaluation set was built three months ago. Many questions have effectively been memorized by the model. Offline accuracy reaches 92 percent, but online user satisfaction is only 3.8/5. The evaluation set has lost discriminative power.

**Synthetic data runaway.** To compensate for labeling capacity, the team uses large amounts of synthetic data. Difficulty is uncontrolled: samples are either too easy or unreasonable. Training gains diminish and early collapse signals appear.

### Core Engineering Pain Points

1. **Structural gap between labeling capacity and demand.** Every new model capability dimension expands labeling demand superlinearly.
2. **Evaluation-set aging.** Static evaluation sets fail quickly under rapid model iteration.
3. **Missing synthetic-data quality loop.** Difficulty, diversity, and realism need systematic control.
4. **Human-agent judgment mismatch.** Agent-assisted labeling and evaluation must stay aligned with human judgment.

## 33.1 Labeling-Assistance Agents

### 33.1.1 Four Dimensions of Labeling Assistance

The labeling-assistance agent is an intelligent co-pilot for annotators. It does not replace judgment; it reduces cognitive load and improves consistency.

```mermaid
flowchart TD
    A["Labeling-assistance agent"] --> T["Task explanation"]
    A --> E["Example recommendation"]
    A --> G["Gray-zone support"]
    A --> C["Conflict arbitration suggestions"]
    T --> L["Annotator"]
    E --> L
    G --> L
    C --> R["Reviewer"]
```

*Figure 33-1: Four dimensions of labeling-assistance agents*

**Task explanation.** When annotation guidelines change or new dimensions are added, the agent parses updates, generates examples, and highlights common mistakes. For complex tasks, such as multi-turn dialogue quality scoring, it decomposes standards into checklists.

**Example recommendation.** For gray-zone cases, the agent retrieves similar historical annotations and shows labels and rationales, helping annotators make more consistent decisions.

**Gray-zone support.** The agent identifies likely controversial samples in advance, such as semantically ambiguous cases, boundary conditions, or conflicting standards. It flags them and recommends which guideline priorities to consult.

**Conflict arbitration suggestions.** When annotators disagree, the agent analyzes whether the conflict comes from different guideline interpretations or genuine sample ambiguity, then generates an arbitration recommendation for reviewers.

### 33.1.2 Annotator Calibration and Consistency Monitoring

Inter-annotator agreement is a core quality metric. Agents can monitor each annotator's label distribution and deviation from baselines.

*Table 33-1: Annotator consistency monitoring metrics*

| Metric | Calculation | Alert threshold | Suggested action |
| --- | --- | --- | --- |
| Individual-group consistency | Cohen's Kappa / Fleiss' Kappa | < 0.7 | Notify annotator and push calibration samples |
| Label distribution shift | KL divergence, individual versus group | > 0.3 | Check for systematic bias |
| Gray-zone annotation time | Time per sample | < 3s or > 120s | Too fast may be careless; too slow may indicate confusion |
| Revision rate | Proportion changed after review rejection | > 15% | Retrain or adjust assignment |

### 33.1.3 Annotation Economics: Cost, Quality, and Speed

Annotation quality cannot be pursued without cost constraints. The key principle is: **perfect labels have infinite cost; engineering needs a Pareto balance among cost, quality, and speed.**

*Table 33-2: Annotation cost structure comparison*

| Cost item | Manual mode | Agent-assisted mode | Savings source |
| --- | --- | --- | --- |
| Annotator hourly wage | $15-25/hr | $15-25/hr | No change |
| Time per sample | 3-5 min simple / 10-20 min complex | 1-2 min / 4-8 min | Agent provides context and examples |
| Gray-zone arbitration time | 5-10 min per case | 2-3 min per case | Agent pre-analysis and recommendation |
| Annotator training cycle | 2-4 weeks | 1-2 weeks | Real-time guideline explanation and calibration |
| Review staffing | 1 reviewer : 5 annotators | 1 reviewer : 10 annotators | Agent prefilters problematic labels |
| Rework rate | 15-25% | 5-10% | Real-time consistency checks |

In many teams, the investment in labeling-assistance agents pays back within two to three months, mainly through 30-50 percent efficiency gains and 50-70 percent rework reduction.

When deadlines are tight, agents can support data-driven tradeoffs:

- If deadline pressure dominates, allow an "uncertain" label for gray-zone samples and preserve quality on simple samples.
- If quality dominates, increase calibration frequency and arbitration coverage.
- If annotator fatigue appears, insert breaks or switch task types.

### 33.1.4 Intelligent Assignment of Labeling Tasks

Agents can assign the right samples to the right annotators.

*Table 33-3: Annotator-profile dimensions for assignment*

| Annotator profile dimension | Collection method | Assignment logic |
| --- | --- | --- |
| Domain proficiency | Historical accuracy by domain | Prefer samples in strong domains |
| Gray-zone ability | Consistency on gray-zone samples | Route ambiguous samples to strong annotators |
| Annotation speed | Historical time per sample | Match urgency |
| Fatigue state | Continuous work duration plus recent accuracy trend | Route easy samples or recommend rest |
| Language ability | Declared languages | Match sample language |

Each batch should also include calibration samples with known answers. If an annotator scores below 85 percent on calibration, the agent should recommend retraining or reassignment.

## 33.2 Synthetic Data Agents

### 33.2.1 From Seed Expansion to Quality Reports

The core loop is a controlled generate-verify-filter pipeline.

```mermaid
flowchart LR
    S["High-quality seed data"] --> P["Seed expansion plan"]
    P --> G["Prompt generation"]
    G --> D["Synthetic samples"]
    D --> V["Multi-layer validation"]
    V -->|pass| B["Synthetic data bank"]
    V -->|fail| F["Reject or regenerate"]
    B --> M["Distribution and collapse monitoring"]
    M --> P
```

*Figure 33-2: Synthetic data agent closed-loop pipeline*

**Seed expansion.** Diversity depends on seed diversity. The agent extracts semantic features, intent types, and difficulty levels from high-quality labeled samples, then identifies under-covered types and sparse difficulty bands.

**Prompt generation.** The agent generates diverse prompts from seeds and expansion strategy. Prompt generation must control:

1. **Semantic diversity.** Cover different topics, scenarios, and phrasings rather than only high-frequency patterns.
2. **Difficulty.** Generate a gradient from simple extraction to multi-step reasoning using Bloom-like levels or custom rubrics.
3. **Constraint reasonableness.** Constraints must be logically satisfiable. "Summarize the full plot of a long novel in five characters" is not a useful constraint.

**Verifier calls.** Synthetic data must pass multiple checks before entering storage.

*Table 33-4: Multi-layer validation for synthetic data*

| Layer | Check | Tool or method |
| --- | --- | --- |
| Format | Conforms to schema | Structured validation |
| Consistency | Prompt-response pair is logically consistent | Agent judge plus rules |
| Difficulty | Difficulty matches target | Inferred from model answer accuracy |
| Safety | Harmful or policy-violating content | Safety classifier |
| Diversity | Semantic similarity to existing data | Embedding similarity deduplication |

### 33.2.2 Collapse Risk Control

The greatest risk of synthetic data is model collapse: as a model trains heavily on its own generated data, the distribution narrows and diversity disappears. Agents can control this risk through:

1. **Diversity budget.** Set semantic diversity budgets for each synthetic batch, such as maximum cosine similarity to existing training data.
2. **Real-data anchoring.** Keep synthetic data below a safe training proportion, often below 50 percent, and preserve human-labeled data as an anchor.
3. **Distribution monitoring.** Monitor synthetic-data embedding distributions and alert when concentration increases.

### 33.2.2.1 Early Indicators of Collapse

Collapse has a latency period before it becomes obvious. Monitoring should track:

**Distribution shift.** Compute Wasserstein distance or MMD between synthetic and real embedding distributions. If the distance expands for three consecutive weeks, warn even before a hard threshold is crossed.

**Diversity decay curve.** Track semantic diversity through eigenvalue distribution of the embedding cosine-similarity matrix. A sustained decline is a strong collapse signal.

**Tail coverage.** Check whether synthetic data covers low-frequency but valid real-data types. Collapse usually starts in the tail.

**Human sampling.** Every two weeks, manually review 50-100 synthetic samples. Automated metrics can miss content degradation that looks statistically normal.

### 33.2.2.2 Intellectual Property and Compliance

Synthetic data is generated by a model, but its legal and IP risks are not simple.

**Derivative-work risk.** If synthetic data is based on copyrighted seed data, it may be treated as derivative in some contexts. Engineering should be conservative: if seed copyright is unclear, mark the synthetic batch as "copyright pending" and restrict use.

**Personal information leakage.** The model may memorize and reproduce PII. Agents should scan for names, phone numbers, addresses, ID numbers, and other personal identifiers.

**Toxic content generation.** Synthetic agents can generate harmful content under certain prompts. Safety validation must apply to synthetic data; do not assume the generator is safe.

### 33.2.3 Difficulty Control for Synthetic Data

Difficulty control prevents synthetic data from collapsing into trivial or impossible samples.

*Table 33-5: Difficulty-control dimensions for synthetic data*

| Dimension | Evaluation method | L1: basic | L3: medium | L5: difficult |
| --- | --- | --- | --- | --- |
| Reasoning steps | Minimum steps needed to answer | 1-2 | 3-4 | 5+ |
| Knowledge breadth | Independent knowledge points required | One domain | 2-3 domains | 3+ domains |
| Constraint complexity | Number and type of constraints | None or one simple constraint | 2-3 constraints | 4+ constraints or contradiction detection |
| Distractor information | Irrelevant information to ignore | None | One distractor | Multiple distractors plus active filtering |
| Format requirement | Output format complexity | Free text | Structured output | Strict format plus content constraints |

The agent should set a target distribution, such as L1:L3:L5 = 40%:40%:20%, and adjust generation dynamically to keep the distribution balanced.

## 33.3 Evaluation and Red-Team Agents

### 33.3.1 Automatic Challenge Set Generation

The first ability of an evaluation agent is generating discriminative test samples. A useful challenge set separates model capability levels. If all models score full marks or zero, the set has no value.

Generation strategies:

1. **Failure-sample analysis.** Extract patterns from historical errors and generate targeted test cases.
2. **Difficulty gradients.** Build progressively harder samples for the same capability dimension.
3. **Adversarial samples.** Create plausible-looking traps, such as contradictory context or questions requiring implicit-assumption detection.

### 33.3.2 Red-Team Sample Generation

A red-team agent actively searches for safety weaknesses and capability boundaries:

- **Jailbreak prompt variants.** Rephrase known jailbreak patterns, switch languages, or wrap them in role-play.
- **Boundary stress tests.** Use extreme inputs such as very long text, special-character sequences, or nested formats.
- **Knowledge blind-spot probing.** Generate samples targeting under-covered time ranges, regions, or domains.

### 33.3.2.1 Attack Strategy Categories

Red-team attacks can be grouped into six dimensions:

1. **Jailbreak.** Tests whether the model can be induced to bypass restrictions through role-play, hypothetical settings, or encoded requests.
2. **Prompt injection.** Tests whether the model distinguishes user instructions from data content.
3. **Knowledge-boundary probing.** Systematically explores coverage gaps across domains, regions, and time.
4. **Reasoning traps.** Constructs logically plausible questions with hidden errors, circular definitions, or false presuppositions.
5. **Social bias probing.** Builds contrastive samples across gender, race, age, region, and other axes.
6. **Multilingual robustness.** Asks the same question in different languages, dialects, and mixed-language forms.

The red-team agent's output should be a continuously updated **security defect database**, not just a report. Each defect includes attack sample, model response, risk rating, and recommended mitigation. The database feeds safety training and input-filter rules.

### 33.3.2.2 Frequency and Triggers for Red-Team Testing

Red-team testing belongs in continuous integration.

| Trigger | Scope | Depth |
| --- | --- | --- |
| Every model release | Full red-team set | Deep |
| Weekly routine | Newly added red-team samples | Medium |
| New data source | Security tests for that source's domain | Medium |
| Security incident | Incident-specific tests | Deep |
| Agent rule change | Tests within affected rule scope | Shallow |

When a new vulnerability type appears, the agent should generate variants and expand the test set. This keeps red-team evaluation fresh as attack methods evolve.

### 33.3.3 Evaluation Slice Generation

Evaluation agents should slice results along multiple dimensions to locate weaknesses.

*Table 33-6: Evaluation slicing dimensions*

| Slice dimension | Examples | Use |
| --- | --- | --- |
| Capability | Reasoning, knowledge, generation, safety | Locate capability gaps |
| Difficulty | L1-L5 | Understand capability ceiling |
| Domain/topic | Medical, legal, financial, code | Assess domain coverage |
| Input length | Short < 512 tokens, medium, long > 4096 tokens | Evaluate long-context ability |
| Language | Chinese, English, mixed Chinese-English | Evaluate multilingual ability |

### 33.3.4 Evaluation Set Health Monitoring

Evaluation sets need continuous health monitoring.

*Table 33-7: Evaluation set health metrics*

| Metric | Calculation | Healthy threshold | If unhealthy |
| --- | --- | --- | --- |
| Discriminative power | Score variance across models | Variance > 0.05 | Add harder questions if variance is too small |
| Ceiling effect | Best-model score | < 95% | Add harder questions |
| Floor effect | Random baseline score | > 10% | Check solvability if near zero |
| Freshness | Share of questions added in last month | > 20% | Trigger update |
| Coverage | Coverage across capability x difficulty | > 80% | Add missing dimensions |

## 33.4 Consistency Calibration: Avoiding Automated Evaluation Self-Delusion

### 33.4.1 Calibrating Agent Judges Against Human Judges

When agents participate in both generation and evaluation, self-evaluation risk appears. Calibration is the defense:

1. **Double-blind calibration.** Periodically sample agent-evaluated items and have humans score them independently. Compute consistency.
2. **Disagreement analysis.** Analyze samples where agent and human scores diverge. Identify systematic bias, such as overrating a sample type.
3. **Calibration set maintenance.** Maintain a high-quality human-labeled calibration set. Before a new evaluation task, the agent judge must show acceptable agreement on that set.

### 33.4.2 Human-Agent Consistency Matrix

*Table 33-8: Human-agent evaluation consistency matrix*

| Evaluation dimension | Agent-only scoring accuracy | Human scoring accuracy | Agent-human consistency | Suggested use |
| --- | --- | --- | --- | --- |
| Factual accuracy | 82% | 95% | 0.78 | Agent prescreen plus human review |
| Response relevance | 88% | 93% | 0.85 | Agent primary plus human sampling |
| Logical consistency | 75% | 90% | 0.72 | Agent suggests, human decides |
| Safety judgment | 91% | 97% | 0.89 | Agent automatic plus human review of high risk |
| Style/tone | 70% | 85% | 0.65 | Human-led |

Agents can take more automatic responsibility in rule-like dimensions such as safety and relevance. Human judgment should dominate subjective dimensions such as style and tone.

## 33.5 Case Review: Agentizing a Dialogue Model Labeling and Evaluation System

A conversational model team agentizes labeling and evaluation in three stages.

**Stage 1: labeling-assistance agent.** The agent provides task explanations and gray-zone example recommendations. IAA recovers from 0.72 to 0.86 in two weeks. Gray-zone consistency improves from 0.55 to 0.78.

**Stage 2: synthetic data agent.** From 500 seed samples, the agent expands to 15,000 synthetic samples across eight capability dimensions and five difficulty levels. Verifiers filter out 12 percent of invalid samples. Training experiments show the optimal synthetic ratio is 35 percent; above that, marginal gains decline.

**Stage 3: evaluation and red-team agents.** The agent generates 200 new evaluation questions each week and replaces stale items. The red-team agent finds vulnerabilities in code injection and role-play jailbreaks, motivating targeted safety training.

### Key Lessons

1. **Labeling assistance depends on gray-zone detection accuracy.** Marking simple samples as gray-zone increases unnecessary cognitive load.
2. **Synthetic data needs verifiers.** Unvalidated synthetic data introduces noise and can reduce performance.
3. **Evaluation sets need an update rhythm.** Replacing 20-30 percent of items monthly helps preserve discriminative power.

### Extended Case: Synthetic Data Collapse Warning and Intervention

In the fourth month, the team's monitoring system raises a collapse warning. Synthetic-data embeddings are becoming concentrated and drifting away from human-labeled data.

Early signals:

- Semantic diversity index drops for three consecutive weeks, from 0.78 to 0.62.
- Prompt patterns are oversampled: prompts beginning with "please explain" reach 35 percent of synthetic data, versus 12 percent in real data.
- Difficulty distribution polarizes: L1 and L5 increase while L3 drops from 40 percent to 25 percent.

Interventions:

1. Reduce synthetic-data share in training from 40 percent to 25 percent and increase human-labeled data weight.
2. Add diversity constraints: no prompt pattern may exceed 20 percent of one synthetic batch.
3. Add reverse validation: train a temporary model on synthetic data and test on real data. If real-data performance drops by more than 2 percent, pause that batch.

After four weeks, the diversity index recovers to 0.75 and real-world model performance stabilizes. The lesson is clear: a synthetic data agent must be a monitor, not only a generator.

## 33.6 Checklist: Labeling, Synthesis, and Evaluation Agent Deployment

- [ ] Does the labeling-assistance agent cover task explanation, example recommendation, gray-zone judgment, and conflict arbitration?
- [ ] Is there real-time IAA monitoring and calibration?
- [ ] Does synthetic seed expansion preserve semantic diversity?
- [ ] Does prompt generation control difficulty gradient and constraint reasonableness?
- [ ] Does synthetic data pass format, consistency, difficulty, safety, and diversity validation before storage?
- [ ] Are diversity budgets and real-data anchoring ratios defined?
- [ ] Can the evaluation agent generate challenge sets and red-team samples automatically?
- [ ] Is there periodic double-blind calibration between agent judges and human judges?
- [ ] Are evaluation results sliced by capability, difficulty, domain, and input length?
- [ ] Is the monthly evaluation-set refresh ratio defined, preferably 20-30 percent?

## 33.7 Chapter Links

- **Chapters 12-13:** SFT data and preference alignment consume labeling and synthetic data.
- **Chapters 14-15:** annotation-system design and synthetic data foundations are agentized here.
- **Chapters 16-17:** synthetic data generation and collapse governance become operational mechanisms in this chapter.
- **Chapter 31:** architecture and task boundaries guide the design of labeling and evaluation agents.
- **Chapter 32:** collection and cleaning outputs become inputs for labeling and synthesis.
- **Chapter 35:** security, permissions, and human-AI collaboration connect to red-team agent governance.

## 33.8 Further Reading: Deeper Engineering Issues

### Making Annotation Standards Operational

Guidelines are often written by experts in natural language: "judge whether the answer is friendly" or "evaluate whether reasoning is coherent." Humans already need training to apply such standards consistently; agents need even more structure.

For an agent, "is the answer friendly" must become checkable criteria:

- Does it use offensive or belittling language?
- Does it provide alternatives when refusing?
- Is the tone neutral and objective?
- Does it show empathy when the user is confused?

The key ability is not abstract understanding. It is converting vague standards into checks that humans and agents can apply consistently.

### The Realism Paradox of Synthetic Data

Synthetic data has a paradox. Its value comes from imitating real-data distribution. If it imitates perfectly, it adds no new information. If it adds new information, that information must be validated.

The engineering answer is **realism anchoring**:

- Every synthetic item should trace to at least one real seed's semantic source.
- The "novel" part of synthetic data should not contradict real-data distribution.
- If synthetic distribution diverges significantly from real distribution, the agent should pause the batch and mark it for review.

### Defending Against Goodhart's Law in Evaluation

Goodhart's Law says that when a metric becomes a target, it stops being a good metric. If a model knows the evaluation set, the score no longer reflects real capability.

Defenses:

1. Keep part of the evaluation set hidden for final assessment.
2. Continuously update evaluation questions with new knowledge backgrounds and reasoning patterns.
3. Use multiple capability slices rather than a single score.
4. Use online feedback, such as likes, complaints, and corrections, to supplement offline evaluation.

## References

- Bowman, S. R., et al. "Measuring Agreement on Disagreeable Annotations." ACL 2023.
- Perez, E., et al. "Red Teaming Language Models with Language Models." EMNLP 2022.
- Zheng, L., et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS 2023.
- Shumailov, I., et al. "The Curse of Recursion: Training on Generated Data Makes Models Forget." 2023.
