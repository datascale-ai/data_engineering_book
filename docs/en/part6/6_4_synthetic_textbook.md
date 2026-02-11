# Project Four: Synthetic Math/Code Textbook

> **Scenario**: Enhance small model logic reasoning ability.
>
> **Core Technology**: Evol-Instruct evolution strategy, Python code execution sandbox (Sandbox) verification, PoT (Program of Thought) data formatting.
>
> **Output**: High-quality synthetic reasoning dataset after verification.

### 1. Project Background (Project Brief)

*   **Task Definition:** Build a high-quality "Program of Thought" (PoT) dataset. We use LLMs (DeepSeek-V3) to "evolve" simple math problems into complex word problems, generate corresponding Python code solutions, and verify answer correctness through code execution sandbox.
*   **Input and Output:**
    *   **Input:** Base math datasets (e.g., GSM8K, MBPP) raw JSONL files.
    *   **Output:** Cleaned JSONL dataset containing `question` (evolved problem), `thought_process` (code solution reasoning), `execution_output` (execution result).
*   **Challenge Analysis:** This project's biggest difficulty is **"hallucination elimination."** LLM-generated code often looks correct but cannot run (syntax errors or logic bugs). We need to build automated "Sandbox" to filter out non-executable samples, ensuring "textbook" rigor.

### 2. Architecture Design (Architecture Design)

### Data Pipeline Diagram
![Figure 5: Synthetic Math/Code Textbook](../../images/part6/图5_合成数学代码教科书数据流水线图.png)

### Technology Stack

*   **Data Source:** `HuggingFace Datasets` (GSM8K/MBPP).
*   **Generation Engine:** `DeepSeek-V3` (via SiliconFlow API) —— cost-effective code generation model.
*   **Orchestration Logic:** Python scripts (Evol-Instruct strategy).
*   **Verification Environment:** Python `subprocess` (local sandbox) —— *production recommend Docker or MicroVM.*

### 3. Step-by-Step Implementation

### Phase 1: Seed Data Acquisition (Seed Preparation)

Everything starts with high-quality seeds. We don't need massive data—just representative logic cores.

**Key Actions:**
1.  Download GSM8K (math) and MBPP (code) data.
2.  Random sample as "evolution" foundation.

**Glue Code (Data Sampler):**
*Code from `download_data.py` and `sampler.py`*

```python
# Core logic: Extract seeds from massive data, keep only Question field
# Original Answer discarded because we let model regenerate code-based solution
sampled = random.sample(data, SAMPLE_SIZE)
for entry in sampled:
    seed_entry = {
        "id": random.randint(1000, 9999), 
        "seed_question": entry['question'], # Keep only question
        "original_answer": entry['answer']  # For reference only
    }
```

### Phase 2: Evol-Instruct and PoT Generation (Evolution & Generation)

This is the project core. We can't just do simple "Q&A pairs"—we need the model to think like a human expert.

**Flow Logic:**
1.  **Evol (Evolution):** Rewrite simple problems (e.g., "1+1=?") into complex scenarios (e.g., "Xiaoming has 1 apple, affected by inflation..."), adding constraints.
2.  **PoT (Code solution):** Force model to write Python code to solve, not directly output text answer.

**Core Prompts (Prompt Engineering):**
*Code from `evol.py`*

```python
def get_evol_prompt(seed_question):
    return f"""
    You are a professional math competition problem composer. Please rewrite the following basic math problem into a more complex, logically rigorous one.
    【Original】: {seed_question}
    【Rewrite Requirements】:
    1. Add constraints: Introduce more variables or limitations.
    2. Add reasoning depth: Don't give numbers directly; have logical relationships between numbers.
    3. Scenario-ize: Put abstract numbers into concrete physical or business scenarios.
    ...
    """

def get_pot_prompt(evolved_question):
    return f"""
    Please write Python code to solve the following math problem.
    ...
    1. Write a function named `solve()`.
    2. Clearly write reasoning steps in code comments.
    3. `solve()` must return the final numerical answer.
    ...
    """
```

### Phase 3: Sandbox Verification

Generated data has large amounts of "dead" samples (Syntax Error, Timeout, Loop). Must verify through execution.

**Sandbox Logic:**
1.  Use regex to extract code blocks from Markdown.
2.  Start subprocess (`subprocess`) to execute code.
3.  **Critical:** Set `timeout` to prevent infinite loops from blocking pipeline.

**Verification Script:**
*Code from `sandbox.py`*

```python
def execute_code(code, timeout=5):
    """
    Execute Python code and get output.
    WARNING: This function should only be called in strongly isolated sandbox (minimal privilege container/micro-VM, no network, restricted filesystem).
    To prevent accidentally executing arbitrary code in host environment, will raise exception if sandbox not explicitly declared.
    Can explicitly allow by setting environment variable EXECUTE_CODE_SANDBOXED=1 inside sandbox container.
    """
    # Basic protection: Prohibit executing arbitrary code in undeclared sandbox
    if os.environ.get("EXECUTE_CODE_SANDBOXED") != "1":
        raise RuntimeError(
            "execute_code can only be used in controlled sandbox environment; "
            "please set environment variable EXECUTE_CODE_SANDBOXED=1 in isolated container/micro-VM before calling."
        )
    try:
        # Use subprocess to start independent process
        result = subprocess.run(
            ['python3', '-c', code],
            capture_output=True,  # Capture stdout
            text=True,
            timeout=timeout,      # Must set timeout!
            check=False,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, f"Error: {result.stderr.strip()}"
```

### 4. Results Showcase (Showcase)

After sandbox cleaning, we get `verified_textbook.jsonl`. This is textbook-grade synthetic data.

**Data Sample Comparison:**

| Phase | Content Example |
| :--- | :--- |
| **Original Seed** | Jenny has 5 apples, ate 2, how many left? |
| **Evol Evolution** | Jenny runs a fruit store with 5 crates of apples (12 per crate). Monday she sold 40% of inventory, and 2 single items spoiled from improper storage. Please calculate the exact number of remaining sellable apples. |
| **PoT Solution** | `def solve(): total = 5 * 12; sold = total * 0.4; ... return remaining` |
| **Execution Result** | `34` (verified, saved to dataset) |

**Verification Statistics:**
Typically, post-Evol code one-shot pass rate (Pass@1) is **60%-80%**. The 20% error data filtered by sandbox are exactly the ones that would pollute model training—**removing them significantly improves SFT model logic consistency.**

### 5. Cost and Optimization (Cost & Optimization)

*   **Resource consumption:**
    *   **API cost:** Each valid sample consumes ~2 LLM calls (evolution + solution). Using cost-effective models like DeepSeek-V3, generating 1k high-quality textbook samples can be controlled under $5.
    *   **Time cost:** Local Python single-threaded is slow; verifying 1k code samples takes ~5-10 minutes.

*   **Security Warning (Critical):**
    *   This project uses `subprocess` for local execution. When processing unknown or untrusted model-generated code, **extremely high risk** exists (e.g., `os.system('rm -rf /')`).
    *   **Production transformation plan:** Must migrate `sandbox.py` execution environment to **Docker container** or **AWS Firecracker** micro-VM, with network access disabled.

*   **Scaling considerations:**
    *   If data scales to millions, single-machine script cannot support. Need to introduce `RabbitMQ` or `Kafka` for task distribution, building distributed "generate-verify" cluster.
