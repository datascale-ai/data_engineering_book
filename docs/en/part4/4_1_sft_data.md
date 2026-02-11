# Chapter 9: Instruction Fine-Tuning Data (SFT Data) —— Building the Model's "Code of Conduct"

## Chapter Summary

This chapter delves into a critical phase in the lifecycle of Large Language Models (LLMs)—the key transition from "general pre-training" to "specific instruction following." We move beyond simple data collection to explore how to build high-quality SFT (Supervised Fine-Tuning) datasets through highly engineered Prompt systems and automated pipelines (Self-Instruct, Evol-Instruct). Beyond technical implementation, we also analyze the theoretical foundations: Why can a small amount of high-quality data unlock enormous model capabilities? We focus on addressing insufficient data diversity, inadequate instruction complexity, and missing reasoning ability, ultimately building an intelligent agent that understands both knowledge and rules.

### Learning Objectives
* **Master iterative System Prompt engineering**: Write System Prompts that control model output format, style, and depth; understand how role definition affects data distribution.
* **Deep understanding of automated data generation pipelines**: Reproduce and improve Self-Instruct and Evol-Instruct algorithms, build domain instruction sets from scratch, understand the "teacher-student" distillation logic behind them.
* **Learn to construct Chain-of-Thought (CoT) data**: Enhance model logic through explicit reasoning steps, breaking the "black box" mapping of Transformers.
* **Design data filtering and deduplication mechanisms**: Master advanced cleaning strategies from ROUGE deduplication to semantic vector clustering, ensuring quality and diversity of synthetic data.

> Scenario Introduction:
"Imagine your team has just finished pre-training a 70B parameter base model, consuming millions of dollars in compute and reading nearly all text on the internet. At this point, it's like a knowledgeable but introverted librarian, with a mind full of Shakespeare's plays, Python code, and quantum mechanics formulas. Yet when you excitedly type 'Please help me create a weight loss plan' at the demo, the model mechanically continues with '...is a good goal, usually including diet and exercise,' or even starts writing 'definition of weight loss plan,' then generates a pile of Wikipedia-style nonsense.

Why does this happen? Because the base model's training objective is 'predict the next token'—it doesn't understand the interaction pattern of 'instruction' and 'response.' To turn this 'scholar' into a personal assistant that understands your intent, you need to feed it thousands of high-quality Q&A pairs, teaching it how to speak and how to solve problems step by step. But manually writing 100,000 instructions is expensive and slow, and human imagination is often limited to specific patterns. How can we automatically produce high-quality training data that is both complex and diverse without relying on large-scale human annotation? This is the core engineering challenge this chapter addresses."

---

## 9.1. Core Concepts and Principles (Concepts & Principles)

In the SFT phase, the industry has reached consensus: **data quality matters far more than quantity**. The data we need is not merely "input-output" pairs, but samples covering various task types, complexity levels, and reasoning patterns.

### 9.1.1 Why Quality Matters More Than Quantity? — The Surface Form Hypothesis

Many beginners mistakenly believe SFT is for "learning new knowledge." However, based on findings from classic studies like LIMA (Less Is More for Alignment), the core role of SFT is **not** to inject knowledge, but to **align format**.

**The Surface Form Hypothesis** posits that models have already acquired the vast majority of world knowledge and logical ability during pre-training. SFT merely teaches the model a specific "interaction format" or "style" to extract the latent capabilities from pre-training. In other words, if pre-training is like having the model read an entire library, SFT only teaches it how to answer in a tone humans prefer—not the content of the books.

This explains why model performance drops sharply once data contains errors, noise, or logical gaps—because the model is learning "how to index knowledge." If the index is wrong, no amount of knowledge can be correctly retrieved. Therefore, a few thousand high-quality, high-diversity samples often train better models than millions of low-quality, homogeneous samples.

### 9.1.2 Engineering Perspective on Prompt Engineering

In data synthesis, Prompts are no longer simple dialogue inputs but the source code for generating data. We treat Prompts as programmable modules, controlling synthesized data distribution through iterative optimization. An excellent Prompt system typically includes:

* **System Prompt**: Defines the "persona" and "boundaries" of the data generator. This is not just assigning an identity—it activates the model's latent domain-specific vocabulary distribution through role-playing. For example, playing "strict lawyer" vs. "enthusiastic salesperson" generates distinctly different sentence structures.
* **Few-Shot Examples**: Anchor output format and style through In-Context Learning. These examples serve as "denoising," telling the model "this is the standard answer I want."
* **Negative Constraints**: Explicitly prohibit the model from generating certain data patterns. In LLM generation, models tend to be lazy or use common clichés; negative constraints are key to breaking this statistical inertia (e.g., "Do not use 'Once upon a time there was a mountain' as story opening").

### 9.1.3 Automated Construction Methodology

To break through the bottleneck of human data, the industry has evolved two core strategies. Understanding their differences is crucial for building balanced datasets.

* **Self-Instruct**: Focuses on **breadth**. Uses strong models (e.g., GPT-4) to generate many new tasks from few seed tasks. Its core assumption: the model has seen enough task types; we only need to induce them through prompts.
* **Evol-Instruct**: Focuses on **depth**. Rewrites simple instructions into complex ones through specific evolution operators (e.g., "add constraints," "deepen reasoning"). This directly addresses Self-Instruct's tendency to generate simple, short instructions, forcing the model to climb in logical complexity and constraint satisfaction.

![Figure 9-1: Self-Instruct vs. Evol-Instruct Comparison](../../images/part4/图9_1_自我指令和进化指令对比.png)
*Figure 9-1: Self-Instruct vs. Evol-Instruct Comparison*

**Table 9-1: Comparison of Mainstream Instruction Data Construction Strategies**

| Feature | Manual Annotation | Self-Instruct | Evol-Instruct |
| :--- | :--- | :--- | :--- |
| **Core Goal** | Extremely high precision, domain-specific knowledge | Increase task diversity | Increase task complexity |
| **Cost** | Very high ($1-$10/item) | Low ($0.01/item) | Medium ($0.03/item, requires multi-round calls) |
| **Input Source** | Domain experts | Seed task pool | Existing simple instructions |
| **Operation Logic** | Expert authorship and review | "Generate a new task different from existing tasks" | "Rewrite this task to be harder, e.g., add constraints" |
| **Typical Operators/Methods** | Cleaning, review, crowdsourcing | ROUGE deduplication, noun/verb filtering | Deepening evolution, Breadth evolution |
| **Use Cases** | Core business logic, RLHF golden datasets | General task coverage expansion, cold start | Enhancing code, math, and logical reasoning |
| **Potential Risks** | Scale difficult; quality fluctuates due to fatigue | Prone to homogeneous, simple instructions | May generate overly complex or unsolvable "hallucinated instructions" |

### 9.1.4 Chain-of-Thought (CoT) Data: Breaking the Reasoning Black Box

CoT's core lies in breaking the "input → output" black-box mapping, forcing the model to make implicit reasoning explicit.

From a cognitive science perspective, humans perform a series of intermediate computations mentally when solving complex problems (e.g., math). Although Transformer models are powerful, without CoT training they tend to guess answers directly—like asking students to write the answer without showing their work, which is highly error-prone. CoT data activates the Transformer's intermediate computation layers, allowing it to allocate more compute to difficult problems by extending the generated token sequence (More compute time = More tokens generated).

### 9.1.5 Data Format Standardization (Data Formatting Standards)

Before engineering implementation, we must understand how data is "fed" to the model. This is not just a JSON parsing issue but relates to how the model understands conversation history. The industry mainly adopts **ChatML** (Chat Markup Language) format, which clearly distinguishes System, User, and Assistant boundaries and prevents prompt injection attacks.

Note that during training, we typically only compute Loss on tokens in the `assistant` response, while masking `system` and `user` parts. This is because we want the model to learn "how to answer," not "how to ask."

```json
// ChatML format example
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum entanglement."},
    {"role": "assistant", "content": "Quantum entanglement is a phenomenon..."}
  ]
}
```

---

## 9.2. Engineering Implementation (Engineering Implementation)

This section guides building a complete data synthesis pipeline, involving toolchain selection and pipeline stability design.

### Environment/Dependencies
* `langchain` / `langsmith`: For managing Prompt templates, LLM call chains, and tracking debugging.
* `rouge_score`: For computing text similarity and deduplication.
* `numpy / scikit-learn`: For vectorized deduplication (advanced), computing semantic distance via Embedding.
* `openai` or `vllm`: For calling teacher models. vLLM is suitable for locally deployed high-throughput open-source teacher models (e.g., Mixtral-8x7B).
* `chromadb` / `faiss`: Vector databases for large-scale deduplication and retrieval.

### 9.2.1 Prompt Engineering for Data Production

In synthetic data engineering, Prompts must be highly robust. We adopt iterative thinking to refine the System Prompt, like developing software versions.

#### Task Objective: Construct a batch of instruction data for training a "Financial Analysis Assistant."

**Step 1: Iteratively Write System Prompt**
* **V1: Too simple, leading to data homogenization**
    * **Defect analysis**: Model-generated instructions tend to be very short and concentrated on basic concept explanations (e.g., "What is a stock?"). This reflects LLM's default tendency to generate "high-probability, low-complexity" text.
    ```python
    # V1 Prompt - Poor performance
    system_prompt_v1 = """
    You are a financial expert. Please generate 5 questions and answers about finance.
    """
    ```

* **V2: Add structured requirements**
    * **Improvement**: Introduced JSON format requirements for easier parsing. Added "role" setting to guide style.
    * **Defect analysis**: Although format is correct, content still lacks depth, no reasoning process. Model may only generate textbook-style definitions, not practical scenarios.
    ```python
    # V2 Prompt - Structural improvement
    system_prompt_v2 = """
    You are a Senior Financial Analyst with 20 years of experience.
    Generate 5 pairs of instruction-response data focused on corporate finance.
    Format the output as a JSON list.
    Each item should have: 'instruction', 'input' (optional), 'output'.
    """
    ```

* **V3: Final production-ready version**
    * **Improvement**: Introduced Few-Shot, Negative Constraints, and Complexity Requirements. This is the standard paradigm used in industry.
    * **In-depth analysis**: Through explicit "Anti-Patterns," we cut off the model's path to generating "nonsense." Exemplars are not just format references but anchors for thinking depth.

    ```python
    # V3 Prompt - High robustness production version
    system_prompt_v3 = """
    ### ROLE
    You are a Chief Market Strategist at a top-tier investment bank. Your goal is to train a junior analyst model. You demand precision, depth, and actionable insights.

    ### OBJECTIVE
    Generate 5 high-quality, complex instruction-following examples related to market analysis, risk management, or quantitative trading.

    ### CONSTRAINTS
    1. **Complexity**: Do NOT ask simple definitional questions (e.g., "What is a bond?"). Instead, ask for scenario analysis, portfolio adjustments, or impact assessments.
    2. **Format**: Strictly output a valid JSON list.
    3. **Reasoning**: The 'output' must demonstrate step-by-step analytical reasoning before giving the conclusion.
    4. **Anti-Patterns**:
       - Avoid generic advice like "Consult a financial advisor."
       - Avoid short, one-sentence responses.
       - Avoid vague statements; use numbers and specific financial instruments where possible.

    ### OUTPUT FORMAT
    [
      {
        "instruction": "...",
        "input_context": "..." (can be null),
        "output": "..."
      }
    ]

    ### EXEMPLAR (One-Shot)
    [
      {
        "instruction": "Given a portfolio heavily weighted in tech stocks (60%), analyze the impact of a sudden 50bps rate hike by the Fed.",
        "input_context": null,
        "output": "First, we identify the correlation... Tech stocks are long-duration assets... Discounted Cash Flow (DCF) models would show... Therefore, the portfolio would likely suffer significant drawdown. I recommend hedging via..."
      }
    ]
    """
    ```
    **Pro Tip:** In V3, we made "Anti-Patterns" explicit. This is key to preventing the model from "slacking." LLMs tend to generate safe, mediocre answers (e.g., "please consult a professional"), which are low-value noise in training data and must be explicitly prohibited.

### 9.2.2 Automated Construction Methods: Self-Instruct and Evol-Instruct

We implement a simplified pipeline based on Evol-Instruct. The core is how to "evolve" simple instructions into complex ones through Prompts, with verification mechanisms introduced in the process.

#### Core Code Breakdown: Evol-Instruct Pipeline

**Step 1: Define Evolution Operators (Evolution Prompts)**

The essence of Evol-Instruct lies in this set of Prompt templates. We need to define different evolution directions: depth (add constraints, deepen reasoning) and breadth (mutation). The following code shows how to build Prompts for "add constraints" and "deepen reasoning." The design of these Prompts directly determines the upper limit of data quality.

```python
class EvolutionPrompts:
    @staticmethod
    def get_deepening_prompt(instruction):
        """
        Depth evolution: Increase logical reasoning depth.
        By requiring 'explicitly ask for multiple-step reasoning', force the model from intuitive to analytical answers.
        """
        return f"""
        I want you to act as a Prompt Rewriter.
        Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle.
        But the rewritten prompt must be reasonable and must be understood and responded by humans.
        
        # Given Prompt #:
        {instruction}
        
        # Method #:
        If #Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly ask for multiple-step reasoning.
        
        # Rewritten Prompt #:
        """

    @staticmethod
    def get_constraints_prompt(instruction):
        """
        Depth evolution: Add specific constraints.
        Limit word increase (10-20 words) to prevent instructions from becoming verbose without substance.
        """
        return f"""
        I want you to act as a Prompt Rewriter.
        ... [header omitted for brevity]...
        
        # Given Prompt #:
        {instruction}
        
        # Method #:
        Please add one more constraint/requirement into #Given Prompt#.
        You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
        
        # Rewritten Prompt #:
        """

    @staticmethod
    def get_breadth_prompt(instruction):
        """
        Breadth evolution: Generate entirely new instructions on different topics based on existing ones.
        Prevents data distribution collapse into narrow domains.
        """
        return f"""
        I want you to act as a Prompt Creator.
        Please generate a brand new prompt that has the same difficulty level as #Given Prompt# but covers a completely different topic or domain.
        
        # Given Prompt #:
        {instruction}
        
        # New Prompt #:
        """
```

**Step 2: Execute Evolution Loop and Exception Handling**

```python
import random

# Assume we have an LLM call interface
def call_llm(prompt):
    # Call GPT-4 or other strong model
    # In production, add retry mechanisms for network jitter
    pass

def evolve_instruction(base_instruction, depth=1):
    current_instruction = base_instruction
    
    for i in range(depth):
        # Randomly select an evolution strategy
        # Strategy probability can be adjusted; e.g., more Breadth early, more Deepening later
        strategy = random.choice(['deepening', 'constraints', 'breadth'])
        
        if strategy == 'deepening':
            prompt = EvolutionPrompts.get_deepening_prompt(current_instruction)
        elif strategy == 'constraints':
            prompt = EvolutionPrompts.get_constraints_prompt(current_instruction)
        else:
            prompt = EvolutionPrompts.get_breadth_prompt(current_instruction)
            
        # Get evolved instruction
        evolved_candidate = call_llm(prompt)
        
        # Quality check (simple): Prevent evolution failure
        # Often models output "Sorry, I can't do that" or simply repeat the original
        if "sorry" in evolved_candidate.lower() or len(evolved_candidate) < 10:
            print(f"Evolution failed at step {i}, keeping previous instruction.")
            break
            
        # Advanced check: Simple heuristic to detect simple repetition
        if evolved_candidate.strip() == current_instruction.strip():
             print(f"Evolution stagnant at step {i}.")
             break

        current_instruction = evolved_candidate
        
    return current_instruction

# Example run
seed = "Write a Python script to calculate Fibonacci numbers."
complex_instruction = evolve_instruction(seed, depth=3)
# Expected result: "Write a Python script to calculate the nth Fibonacci number using dynamic programming, optimize for memory usage, and handle negative input values."
```

**Step 3: Performance Optimization Tips**
* **Batch Processing:** Don't call the API one by one. Construct a Prompt list with 20 instructions, let the model return 20 evolved results at once. This significantly reduces token cost and network latency (High Throughput).
* **Failure Filter:** Evolution often fails (e.g., model starts repeating). Implement a filter: if evolved instruction length shortened or contains typical refusal phrases ("As an AI..."), discard the sample.
* **Diversity Control:** In batch generation, explicitly require "Generate diverse topics" in the System Prompt to avoid all instructions in one batch being about "Python programming."

### 9.2.3 Chain-of-Thought (CoT) Data: Constructing Step-by-Step Reasoning Samples

The core value of SFT data lies in teaching the model "how to think." Ordinary Q&A pairs (Q: 1+1? A: 2) only teach the result; CoT teaches the process.

#### CoT Data Construction Prompt Template

We don't just add "Let's think step by step"—we require the model to output in a specific format (e.g., `<thinking>` tags). This formatted data is highly useful in subsequent training; we can choose to have the model output the thinking process or hide it during inference.

```python
# System Prompt for CoT Generation
cot_system_prompt = """
You are a reasoning expert. When answering the user's question, you MUST follow this format:

1. Enclose your thought process in <thinking> tags.
2. Break down the problem into logical steps.
3. Check for edge cases or potential errors in your logic (Self-Correction).
4. Output the final answer in <answer> tags.

Do NOT output the answer directly without the thinking process.
"""

# V3 Prompt Iteration for Math CoT
user_prompt_cot = """
Question: A farmer has chickens and cows. There are 20 heads and 50 legs. How many of each are there?

Please provide a Step-by-Step solution.
"""
```

**Example output (as training data):**
```xml
<thinking>
Step 1: Define variables. Let C be chickens, K be cows.
Step 2: Set up equations based on the problem statement.
   Equation 1 (Heads): C + K = 20
   Equation 2 (Legs): 2C + 4K = 50
Step 3: Solve for C from Equation 1. 
   C = 20 - K.
Step 4: Substitute C into Equation 2.
   2(20 - K) + 4K = 50
   40 - 2K + 4K = 50
   2K = 10 
   K = 5.
Step 5: Solve for C using the value of K. 
   C = 20 - 5 = 15.
Step 6: Verify the solution. 
   Heads: 15 + 5 = 20. Correct.
   Legs: 15*2 + 5*4 = 30 + 20 = 50. Correct.
</thinking>
<answer>
Chickens: 15, Cows: 5
</answer>
```
**Pro Tip:** This data strategy with `<thinking>` tags is called "Internal Monologue" training. During SFT, keep these tags so the model learns to output the thinking process. In production, use parsing code to capture `<thinking>` content and only show `<answer>` to users, or implement a "Thinking..." UI animation showing partial reasoning steps for explainability.

---

## 9.2.4. Performance and Evaluation (Performance & Evaluation)

Data generation is only the first step; how to evaluate generated data quality is equally critical. We cannot wait until model training completes (potentially days and thousands of dollars) to discover poor data quality.

### Evaluation Metrics
* **Instruction Following Rate:** Usually an automated test. Use GPT-4 as judge to determine whether model-generated responses strictly satisfy all constraints in the input (e.g., "word limit," "include specific keywords," "JSON format").
* **Complexity Distribution:** Use NLP tools (e.g., SpaCy) to analyze verb diversity, syntactic tree depth, and average length of generated instructions. Plot distribution histograms to ensure Evol-Instruct truly increased difficulty, not just verbosity.
* **Diversity:** Compute ROUGE-L or use Embedding Cosine Similarity. If average similarity between samples in the dataset is too high, "mode collapse" has occurred—data lacks diversity.

### Benchmarks

In academia and industry, there are recognized benchmarks for testing post-SFT model capability:
* **WizardLM paper data:** Models trained on data evolved through 4 rounds of Evol-Instruct typically show 10%-20%+ improvement on GSM8K (math) and HumanEval (code) compared to models using only raw data.
* **MT-Bench:** A multi-turn dialogue evaluation set specifically testing instruction following, reasoning, and multi-turn dialogue ability, typically scored by GPT-4.
* **Cost reference:** Generating 52K Self-Instruct data using `gpt-3.5-turbo` costs approximately $500-$1000 (depending on Prompt length and rounds). This is highly cost-effective compared to hundreds of thousands of dollars for manual annotation.

---

## 9.2.5. Pitfalls & Troubleshooting

In practice, data synthesis is full of pitfalls. Here are common failure modes and solutions.

* **Pitfall 1: Mode Collapse**
    * **Symptom:** Model-generated instructions are monotonous—e.g., 1000 generated samples are all "Please write an article about X" or "Please write a Python function."
    * **Cause:** Seed tasks too homogeneous, or System Prompt Temperature set too low, causing the model to fall into local optima.
    * **Fix:** Increase Seed Task diversity (cover 100+ domains: cooking, law, programming, literature); raise Temperature (0.7 → 1.0); explicitly require "Generate a task from a domain different from previous examples" in System Prompt.

* **Pitfall 2: Hallucinated Constraints**
    * **Symptom:** Model learns "must output JSON" from training data, causing it to force JSON output even for casual chat ("Hello"), or output `<thinking>` tags when not requested.
    * **Cause:** Severely skewed training data distribution—100% complex instructions, lacking simple general dialogue data.
    * **Fix:** Data mixing. Mix 10%-20% general dialogue data (e.g., ShareGPT or chit-chat) into Evol-Instruct data to prevent overfitting to specific formats. This is called "General Capability Replay."

* **Pitfall 3: Evolution Failure (Degradation)**
    * **Symptom:** Evolved instructions become absurd, logically contradictory ("Write a 1000-word article without vowels") or extremely verbose.
    * **Fix:** Implement "length penalty" or "complexity truncation." If evolved instructions are complex but GPT-4 cannot answer (or answer quality is poor), the sample is invalid (Bad Case). Introduce "teacher model scoring" for GPT-4 to evaluate evolved instruction feasibility.

* **Pitfall 4: Catastrophic Forgetting**
    * **Symptom:** After SFT, the model learns to follow instructions but seems "dumber," forgetting some world knowledge from pre-training.
    * **Cause:** SFT dataset modifies model weight distribution, over-focusing on specific task forms and squeezing general knowledge storage.
    * **Fix:** Lower learning rate, reduce epochs (SFT typically only needs 2-3 epochs). Add small amounts of pre-training data (Pre-training Replay) to SFT data to maintain parameter distribution stability.

---

## 9.2.6. Chapter Summary and Further Reading

We treat Prompts as the source code of data—they must be managed with rigorous version control and iterative testing like software engineering code. In this framework, Self-Instruct solves the "from zero to one" cold-start challenge, while Evol-Instruct conquers the "from easy to hard" complexity climb. Their organic combination constitutes the golden paradigm for building high-performance datasets. Meanwhile, Chain-of-Thought (CoT) data is far from a simple problem-solving trick—by making reasoning explicit, it effectively allocates computational resources to critical reasoning steps, fundamentally enhancing the model's ability to handle complex logic. Ultimately, the core barrier in data synthesis is not generation capability but the art of filtering and cleaning—between the ease of mass generation and the difficulty of precise screening, only the ability to extract gold from sand truly constitutes core competitiveness.

### References and Further Reading
* *Wang, Y., et al. (2022). Self-Instruct: Aligning Language Models with Self-Generated Instructions.* (Foundational work on automated instruction generation)
* *Xu, C., et al. (2023). WizardLM: Empowering Large Language Models to Follow Complex Instructions.* (Detailed introduction to Evol-Instruct evolution operators)
* *Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* (Foundational work on CoT)
* *Zhou, C., et al. (2023). LIMA: Less Is More for Alignment.* (Establishes theoretical basis that SFT mainly learns format rather than knowledge; "quality > quantity")
