# Project Two: Vertical Domain Expert SFT (Legal)

> **Scenario**: Build industry expert fine-tuning data from unstructured PDF documents.
> **Core Technology**: Self-Instruct for instruction construction, CoT reasoning enhancement, data diversity balance.
> **Output**: `domain_expert.jsonl` instruction fine-tuning dataset.

### 1. Project Background (Project Brief)

- **Task Definition:** Extract knowledge from unstructured legal regulation PDF documents, use LLM Self-Instruct to build vertical domain instruction fine-tuning dataset with "Chain-of-Thought (CoT)" capability.
- **Input and Output:**
  - **Input:** Raw PDF documents (e.g., Civil Code, Criminal Law, with header, footer, watermark interference).
  - **Output:** `domain_expert.jsonl`, containing Instruction (user instructions) and Output (expert response with thinking process).
- **Challenge Analysis:**
  1. **PDF noise cleaning**: Legal documents frequently have citation markers (e.g., `[1]`), line-break-severed Chinese words (e.g., `法 律`), and page numbers embedded in text (e.g., `- 195 -`) that are extremely difficult to clean.
  2. **Data homogeneity**: Simple "legal provision explanation" is insufficient for expert model training; need to construct diverse tasks like case analysis and document drafting.
  3. **Missing reasoning ability**: Ordinary QA pairs lack logical derivation; need to force model to generate CoT (Chain of Thought).

  ### 2. Architecture Design (Architecture Design)

**Data Pipeline Diagram:**

![Figure 2: Building Vertical Domain Expert SFT](../../images/part6/图2_构建垂直领域专家SFT数据流水线图.png)


- **Technology Stack:**
  - **PDF Parsing (pdfplumber)**: Compared to PyPDF2, pdfplumber provides more precise Bounding Box control, convenient for removing headers/footers (code sets 5% top/bottom cutoff).
  - **Cleaning Engine (Regex)**: "Glue code" for Chinese word-break and citation markers—key to improving data quality.
  - **Generation Model (DeepSeek-V3)**: Leverages strong logical reasoning and low-cost API for Self-Instruct data synthesis.
  - **Orchestration Logic (Python)**: Uses weighted roulette wheel algorithm for task type diversity balance.

### 3. Step-by-Step Implementation

#### Phase 1: Data Acquisition and Intelligent Cleaning (The Dirty Work)

PDF extraction's biggest pain is format chaos. The `clean_text_smart` function in `data_processing.py` is the core for this. We focus on solving "Chinese false space" and "embedded page number" issues.

**Key Code Logic:**

```python
def clean_text_smart(text):
    """
    Cleaning core logic: Fix format damage from PDF parsing
    """
    # 1. Remove citation references (e.g., [1], [1-3])
    text = re.sub(r'\[\s*\d+(?:[-–,]\d+)*\s*\]', '', text)

    # 2. Remove page numbers embedded in text (e.g., "- 195 -")
    # Use lookahead assertion to prevent accidental deletion of body numbering
    text = re.sub(r'(?:^|\s|\n)[-—–－]\s*\d+\s*[-—–－](?=\s|\n|$)', ' ', text)

    # 3. Fix Chinese word-break (core fix)
    # Scenario: PDF "法 律 规 定" gets recognized with spaces, need to merge
    pattern_broken_zh = r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])'
    # Execute twice for consecutive breaks
    text = re.sub(pattern_broken_zh, r'\1\2', text)
    text = re.sub(pattern_broken_zh, r'\1\2', text) 
    
    return text.strip()
```

#### Phase 2: Diversified Instruction Synthesis (Diversity & CoT)

To avoid model becoming a law-memorizing bookworm, we designed **Task Pool** and **probability sampling** in `generate_instructions.py`, forcing the model to generate three different task types.

**Diversity Balance Strategy:**

```python
# Task weight configuration (data distribution control)
TASK_POOL = [
    # Task A: Complex case analysis (reasoning focus) - weight 60%
    ("case_analysis", PROMPT_CASE_ANALYSIS, 0.6),
    # Task B: Legal document drafting (generation focus) - weight 20%
    ("doc_drafting", PROMPT_DOCUMENT_DRAFTING, 0.2),
    # Task C: Legal concept discrimination (knowledge focus) - weight 20%
    ("concept_explain", PROMPT_CONCEPT_EXPLAIN, 0.2)
]

# Roulette wheel selection logic
rand = random.random()
cumulative_prob = 0
for name, tpl, prob in TASK_POOL:
    cumulative_prob += prob
    if rand <= cumulative_prob:
        # Hit task type, use corresponding Prompt
        selected_prompt_tpl = tpl
        break
```

#### Phase 3: Formatting and CoT Enhancement

In the Prompt, we explicitly require model to return JSON with "thinking process." In post-processing, we make implicit chain-of-thought explicit, concatenating into final training target format.

**CoT Formatting Logic:**

```python
# Parse model JSON response, force chain-of-thought format construction
if isinstance(raw_output, dict):
    thought = raw_output.get("思考过程") or raw_output.get("analysis")
    answer = raw_output.get("法律建议") or raw_output.get("conclusion")
    
    # Explicitly write thinking process into Output, training model to "think before speak"
    formatted_output = f"#### 🧠 Thinking Process\n{thought}\n\n#### 📝 Expert Advice\n{answer}"
```

### 4. Results Showcase (Showcase)

After processing, raw dull legal provisions transform into Q&A pairs with deep reasoning. Below is a single data sample from `domain_expert.jsonl`:

**Source (Raw Legal Provision Fragment):**
> ...Article 150: Where one party uses fraud to cause the other party to perform civil juristic acts against their true intention, the defrauded party has the right to request the people's court or arbitration institution to revoke...

**Generated SFT Data (JSONL Entry):**

| Field | Content Example |
| :--- | :--- |
| **Instruction** | I was deceived when signing the contract—the other party said the antique was authentic but it was fake. What should I do now? Can I just refuse to honor it? |
| **Output** | **#### 🧠 Thinking Process**<br>1. **Analyze legal relationship**: User entered contract under other party's fraud (performed civil juristic act).<br>2. **Applicable provision**: Per Civil Code Article 150, this is "voidable" civil juristic act, not directly "void" or "refuse to honor."<br>3. **Action path**: Defrauded party must exercise right of revocation, and must do so through court or arbitration institution.<br><br>**#### 📝 Expert Advice**<br>Based on your description, you entered the contract against your true intention—this is a **voidable civil juristic act**. You cannot directly "refuse to honor" or unilaterally repudiate the contract. We recommend:<br>1. **Exercise right of revocation**: You may file a lawsuit with the people's court or apply to an arbitration institution to revoke the contract.<br>2. ... |

**Data Distribution Analysis:**
- **Case analysis (60%)**: Enhance model's complex logic handling.
- **Document drafting (20%)**: Enhance model's professional writing.
- **Concept explanation (20%)**: Consolidate foundational domain knowledge.

### 5. Cost and Optimization (Cost & Optimization)

- **Resource consumption:**
  - **API cost**: Using DeepSeek-V3, generating 1000 high-quality CoT samples costs ~$0.5 - $1.0 (longer input/output tokens).
  - **Time cost**: Single-threaded ~2 seconds per sample.
- **Scaling considerations:**
  - **Concurrency acceleration**: Current code is single-threaded (`time.sleep`); production should use `asyncio` + `Semaphore` for concurrent requests; 10-20x efficiency gain possible.
  - **Quality control**: Currently relies only on Prompt constraints; recommend adding "Reward Model scoring" or "rule filter" step to remove overly short or JSON parse-failed samples.
