# Chapter 10: Synthetic Data —— From "Data Mining" to "Data Farming"

## Chapter Summary

As the large model competition intensifies, high-quality natural data on the public internet faces depletion. We have almost "read" the entire internet, but the model's intellectual ceiling is far from reached. At this point, synthetic data is no longer optional—it is the new engine for model capability leap. This chapter deeply analyzes Microsoft's Phi series "textbook-level" data synthesis methods and explores how to transform from passive "data collectors" to active "data creators."

We go beyond generating text to build rigorous logical verification loops through Program of Thought (PoT) using code execution, and how to use GPT-4o and other multimodal models to synthesize complex image-text instruction data. We demonstrate how to move from simple "imitating humans" to "surpassing humans," constructing training sets that are purer and more educational than the real world through algorithmic means.

### Learning Objectives
* Build a "Textbook Quality" classifier to filter high-value samples from massive web data.
* Implement PoT (Program of Thought) data generation pipeline, using Python interpreter to verify correctness of math/code data.
* Master LLaVA/GPT-4o-based multimodal instruction synthesis to construct image reasoning Q&A pairs.

### Scenario Introduction: When Data Becomes the Bottleneck
>"You're training a small model (1.3B) specialized for Python programming. To make it as powerful as possible, you wrote crawlers to scrape all open-source code from GitHub. Yet when testing after training, you despairingly discover the model learned to write buggy code, even learning to write 'TODO: Fix this later' or 'This code is trash, do not use' inside functions.
Simply adding more data is no longer effective—the more garbage you feed the model, the more chaotic its output. Then Microsoft's Phi-1 paper hits you like a wake-up call: 'Textbooks Are All You Need.' What you need is textbook-like code with clear logic, perfect comments, and progressive teaching—not 'spaghetti code' only God and the original author can understand. But where do you find hundreds of billions of tokens of perfect textbooks? Since we can't find them, we must learn to 'create' this data out of thin air. How to build a tireless 'virtual professor' to batch-produce these perfect textbooks? This is the core engineering challenge this chapter addresses."

---

## 10.1 Core Concepts and Principles (Concepts & Principles)

The core challenge of synthetic data lies in **quality control** and **verification loops**. Because model-generated text often contains hallucinations or errors without checking. Training models on erroneous data leads to "Model Autophagy" or "Model Collapse"—where model output variance gradually vanishes and content becomes extremely homogeneous and detached from reality. The three methods introduced in this chapter address quality issues for text, code/math, and multimodal data respectively.

### 10.1.1 Why Does Synthetic Data Quality Matter Far More Than Quantity?

In early deep learning, we believed "data volume equals justice"—that with enough data, models could learn everything. But in the synthetic data era, this dogma has been overturned.

**Signal-to-Noise Ratio Theory:**
Model training is essentially an information compression process. High-quality data (e.g., textbooks) has extremely high information density and rigorous logical chains—the model needs few samples to capture underlying patterns. Low-quality data (e.g., forum spam, chit-chat) is full of noise and logical gaps. If training sets mix large amounts of low-quality synthetic data, the model will be forced to fit this noise to reduce Loss, causing "logic circuits" to short-circuit.

Learning physics: reading a classic like *Feynman Lectures* (high-quality synthetic data) surpasses watching 10,000 fragmented physics popularization videos (low-quality data). Phi-1's success proved: 6B tokens of textbook-level data can outperform 1000B tokens of web-crawled data in training effect. In synthetic data, **verification cost** has become the new currency.

### 10.1.2 Textbook-Level Data (Textbooks Are All You Need)

Microsoft Phi-1's core idea: rather than training with 1TB of garbage data, use 6B tokens of high-quality data. Its core lies in building a "filter" and an "amplifier."

First, we don't completely abandon web data—we train a classifier (Quality Classifier) to identify content with "educational value." This isn't just checking grammar but whether content is logically self-consistent and contains definitions and reasoning. Second, we use powerful generative models (e.g., GPT-4) as "amplifiers," synthesizing self-contained knowledge fragments with similar style but entirely new content based on these high-quality snippets.

![Figure 10-1: Phi-1 Process Diagram](../../images/part4/图10_1_Phi-1流程示意图.png)
*Figure 10-1: Phi-1 Process Diagram*

### 10.1.3 Code and Math Synthesis: PoT (Program of Thought)

LLMs are essentially probabilistic models—they don't have true logic inference chips. Therefore, when LLMs perform arithmetic (e.g., 234 * 567) or complex logical derivation, they easily hallucinate. PoT (Program of Thought) thinks: since LLMs aren't good at calculation but good at translation, have them "translate" math problems into code, then let the Python interpreter compute the result.

This is the only domain in synthetic data achieving **100% accuracy verification**. We put generated code into a Python sandbox for execution. If it errors, discard; if it runs successfully, the execution result is Ground Truth. This "execution equals verification" mechanism completely solves the synthetic data verifiability problem, enabling low-cost generation of infinite math and logical reasoning data.

**Table 10-1: Synthetic Data Verification Strategy Comparison**

| Data Type | Generator | Core Challenge | Verifier | Verification Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **General Text** | GPT-4 / Gemini | Hallucination | LLM (Judge) / Reward Model | Depends on strong model scoring; low consistency; prone to "judge bias" |
| **Math/Logic** | GPT-4 + PoT Prompt | Calculation errors | Python Interpreter | **Execution consistency**: Code run result matches expected answer; logic absolutely correct |
| **Code** | DeepSeek Coder / GPT-4 | Syntax errors, logic bugs | Unit Tests / Compiler | **Unit tests**: Via assert or successful compilation, ensure functionality |
| **Multimodal** | GPT-4o / LLaVA | Visual hallucination (fabrication) | CLIP Score / Grounding DINO | Check if generated description matches image Embedding; prevent fabricating non-existent objects |

### 10.1.4 Multimodal Instruction Synthesis: Bridging Perception

For image data, traditional annotation costs are extremely high and descriptions are brief. LLaVA proposed a brilliant "blind men and elephant" strategy. We use existing detection models to convert images into symbolic text descriptions (Caption + Bounding Boxes), then feed this pure text to strong text-only models (e.g., GPT-4). GPT-4 can't see images, but it can "imagine" image content through this metadata and generate complex reasoning dialogue based on it. This not only solves multimodal data scarcity but greatly elevates instruction complexity and logic.

---

## 10.2 Engineering Implementation (Engineering Implementation)

This section delves into code-level implementation, building a complete data synthesis pipeline. We focus on ensuring data diversity and preventing generated "textbooks" from being monotonous.

### 10.2.1 Textbook-Level Data: Classifier and Synthesis Pipeline

**Step 1: Train Quality Classifier**
We need to train a lightweight model (e.g., Random Forest or BERT-Tiny) to select "textbooks" from massive data. The key here is annotation data source. Typically we need expert or GPT-4 careful annotation of thousands of samples as seed.

**Code Implementation: Feature Engineering and Training**
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# 1. Prepare annotation data
# First annotate a small number of samples (e.g., 1000) with GPT-4 as "gold standard"
# Prompt: "Determine if this text is of educational value for a student..."
# Label 1: High Quality (Textbook-like), Label 0: Low Quality (Noise)
# This step is critical; annotation quality directly determines classifier ceiling
data = [
    {"text": "Python lists are mutable sequences...", "label": 1},
    {"text": "Hey guys check out my cat photo...", "label": 0},
    # ... more data
] 
df = pd.DataFrame(data)

# 2. Build classifier pipeline
# Phi-1 paper uses pretrained model Embedding; here simplified to TF-IDF for demonstration
# In production, recommend DeBERTa-v3-small or similar lightweight Transformer for Embedding
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1))
])

# 3. Train
X = df['text']
y = df['label']
pipeline.fit(X, y)

# 4. Predict (Filtering Phase)
# Score massive web data; retain only high-scoring data
web_snippet = "Standard library documentation for Python..."
score = pipeline.predict_proba([web_snippet])[0][1]

if score > 0.8:
    print("Keep this data for training: High Educational Value")
else:
    print("Discard: Low Signal-to-Noise Ratio")
```

**Step 2: Synthetic Textbook Fragment Generation**
With classifier-filtered seed data, we need to "expand" it. Prompt design here requires extreme skill.

Prompt iteration: Synthetic Python tutorial
* **V1 Prompt:** "Write a tutorial about Python lists."
    * **Result:** Flat narrative, lacking depth, like mediocre blog post.
* **V3 Prompt (Phi-style):** Introduced specific pedagogy, requiring definitions, comparisons, complexity analysis, and pitfall warnings.

```python
# V3 Prompt - Textbook-style synthesis
synthetic_textbook_prompt = """
### ROLE
You are a professor of Computer Science writing a definitive textbook on Python.

### OBJECTIVE
Write a comprehensive, self-contained chapter section on the topic: "List Comprehensions vs. Map/Filter".

### REQUIREMENTS
1. **Tone**: Educational, clear, precise, and rigorous. Avoid conversational filler.
2. **Structure**:
   - Start with a conceptual definition explaining *why* this feature exists.
   - Provide a "Before and After" code example (Loop vs. Comprehension).
   - Explain the *computational complexity* (Big O) implications.
   - Include a "Common Pitfall" section (e.g., readability vs. brevity).
3. **Diversity**: Use realistic variable names (e.g., `inventory_items`, `sensor_readings`), NOT generic ones like `x`, `y`, `foo`. 
   (This is crucial to prevent the model from overfitting to toy examples).

### OUTPUT
[Markdown Content]
"""
```

### 10.2.2 Back-Translation: Deriving Questions from Answers

Besides generating from scratch, another efficient method is "back-translation." In code, we often easily find high-quality code snippets (e.g., high-star library functions on GitHub) but lack corresponding natural language instructions.

We can use LLM summarization: input complex code, ask the model: "Please write as detailed as possible various user requirement instructions such that this code exactly solves the problem." This quickly generates massive (Instruction, Output) pairs with guaranteed high-quality human code. This method is especially suitable for enhancing model understanding of complex code logic.

### 10.2.3 Code and Math Synthesis: PoT (Program of Thought)

This is the strongest means to ensure synthetic data correctness. By forcing the model to generate code, we convert fuzzy natural language reasoning into precise program logic.

**Core Code Breakdown: Generation and Verification Loop**
```python
import subprocess
import tempfile
import os

# 1. PoT Generation Prompt
# Require model to write solution steps as Python function solver()
pot_prompt = """
Question: Janet has 3 times as many eggs as Bob. Bob has 5 eggs. How many eggs do they have in total?

Instruction:
Write a Python function named `solver()` that returns the answer.
Do not output the number directly. Write the code to calculate it.
Include comments explaining the logic.
"""

# Assume LLM returns the following code string
generated_code = """
def solver():
    # Bob has 5 eggs
    bob_eggs = 5
    # Janet has 3 times as many as Bob
    janet_eggs = 3 * bob_eggs
    # Total eggs
    total = janet_eggs + bob_eggs
    return total
"""

# 2. Code Execution Sandbox
# WARNING: Directly executing generated code is extremely dangerous; must run in sandbox
def execute_generated_code(code_str):
    try:
        # In production use Docker, gVisor, or nsjail for isolation
        local_scope = {}
        
        # Limit execution time to prevent infinite loops
        # Here uses simplified exec for demo; production needs resource module for CPU/memory limits
        exec(code_str, {}, local_scope)
        
        if 'solver' in local_scope:
            result = local_scope['solver']()
            return result, "Success"
        else:
            return None, "No solver function found"
    except Exception as e:
        return None, f"Execution Error: {str(e)}"

# 3. Verification and Data Saving
result, status = execute_generated_code(generated_code)

if status == "Success":
    print(f"Verified Answer: {result}")
    # Data saving strategy:
    # Strategy A (PoT): Save Instruction -> Code. Train model to write code for solving.
    # Strategy B (CoT): Save Instruction -> "Let's calculate... [Reasoning]... The answer is {result}".
    # Strategy B uses code as intermediate step to generate pure text reasoning data.
    save_to_dataset(pot_prompt, generated_code, result)
else:
    print("Discard bad data: Code failed to execute")
```
**Pro Tip:** Generated data can train not only PoT but also ordinary CoT models. Method: use successfully executed code as "intermediate step," execution result as "final answer," reversely construct `<thinking>...code...</thinking><answer>...result...</answer>` format. This "borrowing chickens to lay eggs" method can significantly improve pure text model arithmetic accuracy.

### 10.2.4 Multimodal Instruction Synthesis: LLaVA Pipeline

Using text-only models to synthesize multimodal data is LLaVA's innovation. This method's core lies in **symbolizing** visual information—since text-only models (e.g., GPT-4) can't see images, we translate images into "code" they can read.

![Figure 10-2: LLaVA Data Synthesis Process Diagram](../../images/part4/图10_2_LLaVA数据合成流程示意图.png)
*Figure 10-2: LLaVA Data Synthesis Process Diagram*

#### 1. Engineering Pipeline: From Pixels to Symbols

Before Prompt design, we need toolchain to "deconstruct" images into structured data (Metadata) readable by text models:

1.  **Global Semantics (Captioning)**
    * **Tool**: CLIP or BLIP for one-sentence description.
    * **Role**: Provide overall context.
    * **Output example**: `"A young girl riding a horse on a beach at sunset."`

2.  **Local Details (Object Detection)**
    * **Tool**: Grounding DINO to extract objects and coordinates (Bounding Box).
    * **Role**: Provide spatial anchoring entities.
    * **Output example**: `{'girl': [100, 200, 300, 400], ...}`

3.  **Data Synthesis**
    * **Action**: Fill above information into Prompt, call GPT-4 to generate dialogue.

#### 2. Prompt Engineering: Design and Considerations

With structured data, Prompt design becomes the key to data quality. Below is LLaVA-style Prompt template and architectural considerations:

```python
# System Prompt for Multimodal Data Generation
multimodal_gen_prompt = """
### CONTEXT
You are an AI visual assistant. You cannot see the image directly, but I will provide its metadata.
Your task is to generate a conversation between a Human and Yourself about this image.

### IMAGE METADATA
# [Data injection point]: Fill pipeline-extracted data here
- **Caption**: "{caption}"
- **Objects**: {object_list_with_boxes}

### INSTRUCTIONS
1. **Conversation Style**: Generate a multi-turn Q&A (User asking, Assistant answering).
2. **Reasoning**: The Human should ask complex questions (e.g., "What suggests this is a safe environment?"). You answer based on the visual evidence.
3. **Spatial Awareness**: Use the bounding box info to describe relative positions if asked (e.g., "The ocean is in the background...").
4. **Visual Consistency**: Do NOT hallucinate objects not listed in the metadata.
"""
```

**Architectural considerations behind Prompt design:**

* **Why provide both Caption and Objects? (Complementarity)**
    Objects alone (girl, horse, ocean) are discrete, lacking action and atmosphere; Caption alone (girl riding horse) lacks specific location. Combined, GPT-4 can construct complete mental "scene graph."

* **Why emphasize "Spatial Awareness"? (Spatial alignment)**
    Text models inherently lack spatial sense. By forcing them to process `[x1, y1, x2, y2]` coordinate data, we're forcing the text model to learn "visual alignment"—understanding pixel regions corresponding to "left," "lower right," etc.

* **Why add "Visual Consistency" constraint? (Hallucination suppression)**
    Text models' biggest flaw is easy "brain-filling." E.g., seeing "beach" they might fabricate "seagulls flying." Must explicitly prohibit generating objects not in Metadata to ensure high signal-to-noise ratio.

* **Why generate "Complex Reasoning"? (Data dimension upgrade)**
    Simple "What's this? A horse" has no training value. We need to leverage GPT-4's intelligence to artificially create "requires thinking" samples (e.g., causal inference, sentiment analysis) through synthesis, so small models (Student) can distill large model reasoning through learning.

---

### 10.2.5 Advanced Strategies for Multimodal Instruction Data Synthesis

Although "symbolic reasoning" based on text-only models (like early LLaVA v1) pioneered multimodal instruction synthesis, its main defect is "lossy compression of visual information"—text models cannot directly perceive pixel-level visual features; relying only on metadata for reasoning easily causes hallucination.

To break this bottleneck, industry and academia have evolved three more mainstream and efficient synthesis strategies: **Visual Strong Model Distillation**, **Mixture-of-Experts Pipeline**, and **Evolution Instruction Generation**.

#### 1. Visual Strong Model Distillation

This is currently (as of 2024-2025) the most mainstream method for building high-performance open-source multimodal models (e.g., LLaVA-NeXT, ShareGPT4V), often considered SOTA (State-of-the-Art).

**Core Idea**
This method abandons "using text models to guess visual content" and adopts the "Teacher-Student" distillation paradigm. Using closed-source top multimodal models (e.g., GPT-4o, Gemini 1.5 Pro) as "teacher models," directly process raw image signals to generate high-quality, high-density detailed descriptions (Dense Caption) and complex reasoning Q&A pairs for open-source models (student models) to learn.

**Advantage Analysis**
* **Eliminate modality gap**: Teacher model directly "sees" image pixels, capturing lighting, texture, micro-expressions that text metadata cannot convey.
* **Suppress hallucination**: Descriptions based on real visual input greatly reduce factual error probability.

**Implementation Flow**
The core lies in building a Prompt that can induce teacher model to output exhaustive information. Below is standard distillation flow pseudocode:

```python
def generate_dense_instruction(image_path, api_client):
    """
    Use SOTA MLLM to generate high-density multimodal instruction data
    """
    
    # System Prompt key: Require extremely detailed capture and logical association
    distillation_prompt = """
    You are an expert visual analyst. Analyze the provided image with extreme detail.
    
    Tasks:
    1. Dense Captioning: Provide a comprehensive description of every corner of the image, covering colors, textures, lighting, and background details.
    2. Object Relationships: Analyze the interactions between objects (e.g., causality, spatial relations).
    3. OCR Extraction: Transcribe any visible text verbatim.
    4. Q&A Generation: Based on the visual details above, create a logical reasoning question that cannot be answered without looking at the image.
    """

    # Key difference: Input contains real Image Tensor, not merely Bounding Box
    response = api_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": distillation_prompt},
            {"role": "user", "content": [{"type": "image_url", "url": image_path}]}
        ]
    )
    
    return parse_response(response)
```

---

#### 2. Domain Specialization: Mixture-of-Experts Pipeline

For vertical domains where general models struggle—Document AI, complex chart analysis, or autonomous driving data—general visual distillation often lacks precision. Adopting "Mixture-of-Experts" strategy is best practice.

**Core Logic**
This method doesn't rely on single model's end-to-end capability but assembles multiple specialized small models (Experts) as perception frontend, converting unstructured images into fine structured data for LLM integration.

Logic flow:
$$\text{Image} \xrightarrow{\text{Experts}} [\text{OCR} + \text{Layout} + \text{Detection}] \xrightarrow{\text{Aggregation}} \text{Structured Context} \xrightarrow{\text{LLM}} \text{Instruction}$$

**Application Scenarios**
Typical applications include financial invoice processing, medical imaging reports (DICOM), etc.

1.  **OCR Expert (e.g., PaddleOCR)**: Extract all text and precise coordinates $(x_1, y_1, x_2, y_2)$ from image.
2.  **Layout Expert (e.g., LayoutLM)**: Parse document topology, identify table rows/columns, paragraph hierarchy, and title relationships.
3.  **Synthesis (LLM)**: Fill structured data into Prompt template.
    * *Example Prompt:* "This is an invoice's structured data, invoice number at (100, 200), total amount 500.00. Please generate a multi-turn Q&A about 'financial audit verification' based on this."

![Figure 10-3: Mixture-of-Experts Pipeline Diagram](../../images/part4/图10_3_多专家混合流水线示意图.png)
*Figure 10-3: Mixture-of-Experts Pipeline Diagram*

---

#### 3. Evolution Instruction Generation (Visual Evol-Instruct)

Inspired by WizardLM in text domain, Visual Evol-Instruct aims to solve training data "homogenization" and "oversimplification." When base dataset only contains simple recognition tasks (e.g., "What's in the image?"), models cannot learn higher-order reasoning. This method forces "dimensional upgrade" of existing data through Prompt Engineering.

**Core Logic**
$$\text{Simple VQA} \xrightarrow{\text{Complexity Constraints}} \text{Complex Reasoning VQA}$$

By applying specific evolution instructions to LLM, data complexity can be elevated in these dimensions:

* **Reasoning Deepening**:
    * *Original*: "What is this person holding?"
    * *Evolved*: "Based on the object's use and the person's clothing, infer this person's profession and what activities they might perform next."
* **Counterfactual Reasoning**:
    * *Original*: "The car in the image is red."
    * *Evolved*: "If the red sports car in the image were replaced with an old bicycle, how would the scene's atmosphere change? Does this fit the modern architecture style in the background?"
* **Comparative Analysis**:
    * Input two similar images, require model to analyze subtle differences (lighting changes, object displacement), training model's fine-grained observation.

Through combined use of these three strategies, we can build high-quality multimodal instruction datasets rich in visual detail and deep logical reasoning, laying solid foundation for training LLaVA, MiniGPT-4, and similar models.

## 10.3. Performance and Evaluation (Performance & Evaluation)

After training on synthetic data, evaluation is especially important—we need to confirm whether the model truly "learned" or merely "memorized" patterns in synthetic data.

### Evaluation Metrics
* **Pass@1 (Code):** For PoT synthetic data, we test Pass@1 on HumanEval. Phi-1 achieved 50%+ Pass@1 with only 6B data, surpassing many models trained on 100x more data. This proves data quality's overwhelming advantage.
* **Hallucination Rate:** Compare synthetic multimodal answers with original image CLIP similarity to detect whether non-existent objects were generated. We can build negative sample sets specifically inducing model to answer non-existent objects, checking if model refuses.
* **Decontamination:** A dirty but necessary step. Check whether synthetic data inadvertently contains test set (e.g., HumanEval) questions. Through N-gram overlap detection, ensure model generalizes rather than cheats.

### Benchmarks
* **PoT vs CoT:** On math (e.g., GSM8K), PoT typically outperforms pure text CoT by 5-10%. Reason: PoT outsources calculation to CPU (best at computation) while GPU focuses on logic translation—optimal compute allocation.

---

## 10.4. Pitfalls & Troubleshooting

Synthetic data is beautiful but full of traps. A slight slip and the model falls into "self-congratulation" loops.

* **Pitfall 1: Self-Confirmation Bias**
    * **Symptom:** Model-generated code runs but logic is wrong (e.g., 2+2=5, and model-generated test cases are also wrong, coincidentally passing the wrong function).
    * **Fix:** Must introduce external, deterministic Solver or human-reviewed Unit Test library. Never fully rely on model-generated test cases to verify model-generated code—it's like letting the criminal judge themselves.

* **Pitfall 2: Lack of Visual Grounding**
    * **Symptom:** In multimodal synthetic data, model discusses details not in metadata (fabrication). E.g., Metadata only has "dog," but model describes "dog collar color" when the dog in the image has no collar.
    * **Fix:** Add strict instruction in Prompt: "Only strictly rely on the provided metadata. Do not invent details." Also use CLIP Score to filter out generated text with too low similarity to original image.

* **Pitfall 3: The Homogenization Trap**
    * **Symptom:** If all data is GPT-4 generated, your model becomes a "low-end GPT-4," losing diversity. All answer tones and sentence structures are strikingly consistent.
    * **Fix:** **Entropy Injection**. Randomly inject different Personas in Prompt (e.g., "grumpy programmer," "patient kindergarten teacher"), or require different programming styles (recursive vs. iterative), forcing data distribution expansion.

---

## 10.5. Chapter Summary and Further Reading

The "Textbooks Are All You Need" paradigm established the core principle that data quality (educational value) takes priority over quantity. Synthetic data technology grants us the ability to precisely control data signal-to-noise ratio. Under this framework, Program of Thought (PoT) guarantees data rigor by converting reasoning into executable code and using compiler determinism for verification. Meanwhile, Symbolic-to-Synthetic method uses metadata (e.g., Bounding Box) to guide text models to generate multimodal content, achieving effective conversion from unimodal to multimodal data. This evolution marks data engineering's transition from passive "mining" to active "production": through seed prompt construction, instruction complexity (Evolution), quality filtering, and final synthesis, systematically building high-quality datasets via standard industrial process.

### References
* *Gunasekar, S., et al. (2023). Textbooks Are All You Need (Phi-1).*
* *Chen, W., et al. (2022). Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks.*
* *Liu, H., et al. (2023). Visual Instruction Tuning (LLaVA).*
* *Shumailov, I., et al. (2023). The Curse of Recursion: Training on Generated Data Makes Models Forget.* (Important research on model collapse)
