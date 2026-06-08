# Project 12: R1 Reasoning Flywheel

## Abstract

This project builds a reproducible data-engineering case around an R1 reasoning flywheel. It frames the work through business goals, data boundaries, architecture decisions, implementation slices, acceptance metrics, and risk control, while keeping installation commands and script details subordinate to engineering review. The chapter emphasizes the relationship among sample schema, data flow, failure modes, and deliverables so readers can turn reasoning-data methods into auditable and extensible project assets.

## Keywords

R1; project practice; reproducible data engineering; reasoning data pipeline; acceptance metrics

## Project Goals and Reader Takeaways

The goal is to build a reasoning-data loop that connects Long-CoT cold start, rejection sampling, and SFT feedback. After completing the chapter, readers should be able to identify the key objects in a reasoning flywheel, split the engineering chain into runnable stages, set acceptance metrics, and transfer the same approach to SQL generation, code repair, structured extraction, tool use, or enterprise problem banks.

## Scenario Constraints and Data Boundaries

The project focuses on the reasoning-data engineering chain. It does not cover a complete reinforcement-learning platform or large-scale online training. These boundaries make the case reproducible and auditable. If task scale, data source, permission scope, model backend, or deployment environment changes, sampling strategy, verifier thresholds, runtime cost, and compliance requirements must be reviewed again.

## Architecture Decision

The architecture follows cold-start samples, long-chain reasoning, candidate sampling, verifier filtering, rejection sampling, and SFT feedback. This path prioritizes input/output contracts, version traceability, anomaly localization, and reviewable results instead of compressing all logic into one long script.

## Sample Schema and Data Flow

The core flow can be summarized as:

```text
reasoning seeds -> Long-CoT generation -> multi-candidate sampling -> verification/scoring -> rejection sampling -> feedback SFT dataset
```

At minimum, samples should retain `id`, `source`, `content_or_payload`, `metadata`, `quality_signals`, `split_or_stage`, and `audit_trace`. For this project, the most important additional fields are prompt identity, candidate index, parsed answer, verifier result, reward score, and feedback destination.

## Core Implementation Slice

The chapter keeps code fragments that expose design choices: data normalization, candidate sampling, verifier interfaces, rejection-sampling logic, merge manifests, and evaluation hooks. Full scripts, long configurations, logs, and large artifacts should remain in the companion project directory.

## Experimental or Acceptance Metrics

Acceptance metrics include reasoning correctness, candidate retention rate, verifier coverage, long-chain length distribution, feedback-sample quality, format pass rate, and cost per accepted sample. For production, course, or public reproduction use, reports should also record model versions, backend mode, random seed, sample inspection notes, and failed-candidate review.

| Acceptance dimension | Metric or evidence | Publication review focus |
| --- | --- | --- |
| Candidate generation | Number of sampled candidates, long-chain length distribution, and task-source coverage | Explain the difference among mock, vLLM, and real model sampling |
| Filtering and feedback | Verifier coverage, candidate retention rate, feedback-sample quality, and format pass rate | Every feedback sample should trace back to its original task, candidate trace, and verifier result |
| Risk control | Self-reinforced errors, verifier bias, and overly long trace noise | Do not equate rejection-sampling pass rate directly with reasoning capability improvement |

## Cost, Risk, and Compliance Boundaries

Cost mainly comes from long-chain generation, multi-candidate sampling, and verification. Risks concentrate on self-reinforced errors, verifier bias, noisy long traces, and overfitting to automatically checkable tasks. When external datasets or third-party model services are involved, source status, license boundaries, call logs, and human review notes should be preserved.

## Common Failure Modes

Common failures include prompt pools that are too narrow, schema fields missing from candidate traces, verifier thresholds that accept brittle answers, model-call instability, feedback data that cannot be traced to original tasks, and evaluation slices that do not cover the target reasoning behavior. Troubleshooting should first inspect data boundaries and intermediate artifacts before changing the model or training code.

## Reproducible Resource Notes

Reproduction materials should include data-source notes, minimum samples, environment files, run commands, verifier scripts, evaluation reports, training manifests, and artifact directories. The chapter keeps essential snippets; full notebooks, long scripts, and large files should be maintained separately.

## R1-style Reasoning Data Flywheel: From Long-CoT Cold Start to Rejection-sampling Feedback

### Background and Objectives

Reasoning models such as R1 and QwQ (Guo et al. 2025; Qwen Team 2025) teach us more than "models can output longer chains of thought." They show that reasoning capability can be decomposed into a runnable data engineering pipeline. Traditional SFT projects usually revolve around a fixed set of `instruction-response` samples, and the data itself changes little after training. An R1-style data flywheel is different: for the same task, the model generates multiple candidate traces; the system uses verifiers to decide which traces are correct, which are format-stable, and which are worth feeding back; the selected successful traces are then reorganized into the next round of supervised fine-tuning data. Data becomes part of the model capability iteration loop instead of a static asset prepared before training.

For small and medium teams, fully reproducing large-scale RL training is often unrealistic. Online sampling at scale, reward modeling, repeated policy updates, and long-context reasoning all consume sustained GPU resources and require complex training-system governance. A more practical path is to build the data flywheel first: cold-start data can be generated, multi-sample tracing can run, verifiers can score automatically, rejection sampling can select high-quality traces, second-round SFT data can be merged, and evaluation scripts can compare before and after training. Once this chain is stable, it becomes reusable when larger models, more complex rewards, or domain tasks are introduced.

This project is based on `code/zh/project_12_r1_flywheel` and implements a minimal runnable R1-style reasoning data flywheel. It does not attempt to reproduce the full RL process of R1-Zero, nor does it treat benchmark score as the only target. Instead, it emphasizes a data production process that is runnable, replaceable, and auditable. The default base model is `Qwen2.5-7B-Instruct` (Hui et al. 2024), and the main data sources are `OpenThoughts` (Gunther et al. 2025), `GSM8K` (Cobbe et al. 2021), `MATH-500` (Hendrycks et al. 2021), and `HumanEval` (Chen et al. 2021).

The overall chain is:

```text
OpenThoughts / GSM8K / MATH-500 / HumanEval
  -> cold-start SFT data
  -> vLLM sampled reasoning traces
  -> math / code / format verifier
  -> rejection sampling
  -> merged SFT data
  -> LoRA training and evaluation
```

After completing this project, readers should understand three things. First, the core engineering objects in an R1-style system are not just model weights, but task pools, sampled traces, verifiers, rejection-sampling outputs, and training manifests. Second, a reasoning data flywheel can start with rule rewards and supervised feedback before full RL is introduced. Third, whenever a target task can support an automatic verifier, the same structure can migrate to SQL generation, code repair, structured extraction, tool use, or enterprise problem banks.

## 2. Architecture: A Six-component R1 Reasoning Data Flywheel

This project contains six components: cold-start extraction, multi-sample reasoning, verifier pool, rejection sampling, second-round SFT merge, and training/evaluation. Components exchange files and shared schemas instead of being tightly coupled in one long script. This lets each layer be rerun independently: sampling failures only require resampling, verifier updates only require rejection sampling, and mixture changes only require data merging.

![Figure P12-1: R1-style self-reasoning flywheel architecture](../../images/part11/p12_r1_reasoning_flywheel_architecture.png)
*Figure P12-1 Closed-loop structure from cold-start extraction, multi-sample tracing, verifier pool, and rejection sampling to second-round SFT merge, training, and evaluation.*

The first component is **cold-start extraction**, implemented by `cold_start_data.py`. It extracts SFT-ready samples from existing data sources and normalizes them into `messages` format. Math samples are organized with `Reasoning:` and `Final Answer:`. Code samples are organized with `Reasoning:` and a fenced Python code block. Cold-start data does not need to produce the strongest model; it gives the model a basic output structure, style, and parseable format.

The second component is **multi-sample reasoning**, implemented by `sample_traces.py`. It generates multiple candidate traces for the same prompt and records `prompt_id`, `sample_idx`, `raw_trace`, `parsed_answer`, and `generation_params`. The project supports three backends: mock, local vLLM Python API, and external OpenAI-compatible API. In production, inference services and data processing scripts can be deployed separately to reduce coupling among CUDA, torch, and vLLM dependencies.

The third component is the **verifier pool**, implemented by `verifier_pool.py`. It provides rule-based math, code, and format verifiers. The math verifier extracts answers, parses numbers, and compares values; the code verifier extracts Python code blocks and runs tests; the format verifier checks for structures such as `Reasoning:`, `Final Answer:`, and `Code:`. These verifiers are not complex reward models, but they are stable, cheap, and interpretable.

The fourth component is **rejection sampling**, implemented by `rejection_sampling.py`. It reads sampled traces, groups them by prompt, calls the verifiers, sorts candidates by `verifier_pass`, `reward_score`, and `sample_idx`, and keeps the better candidates for each problem. Selected traces are repacked as SFT samples and written to `rejection_selected_10k_30k.jsonl`. This is the core of the flywheel: the model proposes candidates, the system selects automatically, and successful traces flow back into training.

The fifth component is **second-round SFT merge**, implemented by `merge_sft_data.py`. It merges cold-start data and rejection-sampled data into `merged_sft_data.jsonl` and writes `training_manifest.json`. Cold-start and feedback samples both enter SFT, but they mean different things: the former provides initial format and reasoning style, while the latter captures successful traces discovered by the current model.

The sixth component is **training and evaluation**. `train_lora.py` provides a minimal LoRA demonstration, and `eval_gsm8k_math.py` evaluates GSM8K / MATH. In this project, training and evaluation act as validation interfaces for the data flywheel rather than as a full experiment platform. Their main purpose is to test whether merged data can enter training and whether a unified script can compare the base model and LoRA adapter.

Main artifacts:

| Stage | Default Artifact | Meaning |
| --- | --- | --- |
| Cold-start extraction | `data/processed/cold_start_5k.jsonl` | First-round SFT samples |
| Cold-start statistics | `data/processed/cold_start_summary.json` | Source, domain, and count statistics |
| Multi-sample tracing | `data/sampled_traces/*.jsonl` | Candidate reasoning traces |
| Verifier output | `data/verified_candidates/*.jsonl` | Verification results for each candidate |
| Rejection sampling | `data/processed/rejection_selected_10k_30k.jsonl` | High-score traces for feedback |
| Data merge | `data/training/merged_sft_data.jsonl` | Second-round SFT input |
| Training manifest | `data/training/training_manifest.json` | Composition of training data |
| Evaluation result | `data/reports/eval_results_gsm8k_math.json` | GSM8K / MATH comparison |

---

## 3. Step-by-Step Implementation: From Cold Start to Feedback-ready SFT Data

### Step 1: Prepare the Environment and Task Data

The project is best run in an isolated conda environment:

```bash
cd code/zh/project_12_r1_flywheel
conda env create -f environment.yml
conda activate p12-r1-flywheel
```

Before real sampling, run tests to confirm that the mock pipeline and core modules are intact:

```bash
pytest -q
```

`tests/test_pipeline.py` covers cold-start extraction, math/code/format verifiers, mock sampling, rejection sampling, SFT merge, mock LoRA training, and mock evaluation. Passing tests only means the engineering skeleton is intact; it does not prove real vLLM sampling or large-scale training has completed. If these tests fail, do not start long-running jobs.

Input data can be understood by task type. `OpenThoughts` provides Long-CoT cold-start samples, `GSM8K` and `MATH-500` provide math problems, and `HumanEval` provides code tasks. In real deployments, the same data can come from public datasets or internal problem banks. As long as samples can be normalized to the shared schema and include reference answers or tests, they can enter the flywheel.

---

### Step 2: Extract Cold-start SFT Data

The cold-start stage normalizes different sources into one SFT message format:

```bash
python cold_start_data.py \
  --max-openthoughts 5000 \
  --max-math 100 \
  --max-gsm8k 100 \
  --max-code 100
```

Default outputs:

```text
data/processed/cold_start_5k.jsonl
data/processed/cold_start_summary.json
```

Each sample contains `record_id`, `source_dataset`, `domain`, `prompt`, `reference_reasoning`, `reference_answer`, and `messages`. `messages` is what training consumes:

```python
record = {
    "record_id": "math_gsm8k_000001",
    "source_dataset": "gsm8k",
    "domain": "math",
    "prompt": "A math problem...",
    "reference_reasoning": "Step-by-step solution...",
    "reference_answer": "42",
    "messages": [
        {"role": "system", "content": "You are a careful reasoning assistant."},
        {"role": "user", "content": "A math problem..."},
        {"role": "assistant", "content": "Reasoning: ...\nFinal Answer: 42"},
    ],
}
```

For code tasks, the assistant content becomes:

````text
Reasoning: explain the implementation idea
Code:
```python
def solve(...):
    ...
```
````

One subtle detail matters. In `HumanEval`, `canonical_solution` often contains only the function body. If it is used directly, the model may learn incomplete code snippets. `pipeline_utils.py` implements `render_humaneval_solution(...)`, which combines the prompt function signature with `canonical_solution` to form a complete function definition. This makes `reference_answer` more suitable for training and verification.

After generating cold-start data, inspect `cold_start_summary.json`. Check total count, `domain_distribution`, and several sampled `messages` records. If this stage has empty answers or malformed formats, downstream sampling and rejection sampling will amplify the contamination.

---

### Step 3: Generate Multi-sample Reasoning Traces with Mock or vLLM

Sampling takes prompts from `cold_start_5k.jsonl` and produces multiple candidate traces for each prompt. Use the mock backend first:

```bash
python sample_traces.py \
  --input data/processed/cold_start_5k.jsonl \
  --output-dir data/sampled_traces \
  --num-examples 20 \
  --num-samples 2 \
  --backend mock \
  --force-mock
```

The mock backend is not for evaluating model ability. It only checks that data can flow downstream. For real sampling, start vLLM and call it through an OpenAI-compatible API:

```bash
bash scripts/serve_qwen_vllm.sh
```

Then run:

```bash
python sample_traces.py \
  --input data/processed/cold_start_5k.jsonl \
  --output-dir data/sampled_traces \
  --num-examples 100 \
  --num-samples 4 \
  --backend openai \
  --parallel-prompts 4
```

Core fields in a sampling record:

```python
sample = {
    "prompt_id": "math_gsm8k_000001",
    "sample_idx": 0,
    "source_dataset": "gsm8k",
    "domain": "math",
    "generation_params": {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 768,
    },
    "raw_trace": "Reasoning: ...\nFinal Answer: 42",
    "parsed_answer": "42",
    "finish_reason": "stop",
    "token_count": 512,
}
```

The most important field is the full `raw_trace`, not just the final answer. It supports three downstream uses: verification, feedback training, and error analysis. If memory or throughput is limited, reduce `num_samples`, `parallel_prompts`, and `max_tokens`. These changes affect scale but should not alter the data schema.

---

### Step 4: Build Math, Code, and Format Verifiers

The verifier determines which candidates become feedback data. `verifier_pool.py` provides three verifier types.

The math verifier extracts final answers, prioritizing patterns such as `\boxed{}` and `Final Answer:`, then tries numeric parsing. If both prediction and reference can be parsed numerically, it compares with tolerance; otherwise it falls back to normalized string comparison. It returns `verifier_pass`, `reward_score`, `parsed_answer`, and `reason`.

The code verifier extracts fenced Python code blocks or snippets containing `def`, then runs them with tests. Passing tests receive high scores. Missing code blocks, timeout, exceptions, and failed tests are all recorded with specific reasons. The current implementation is suitable for HumanEval-style tasks; it is not a complete code evaluation platform, but it is enough for a rejection-sampling prototype.

The format verifier checks output stability. Math tasks should contain `Reasoning:` and a parseable final answer; code tasks should contain `Reasoning:` and a Python code block. Many candidate traces are not completely wrong but are too unstable to parse. The format verifier marks these cases early.

Example verifier output:

```python
verdict = {
    "verifier_type": "math",
    "verifier_pass": True,
    "format_pass": True,
    "reward_score": 1.0,
    "parsed_answer": "42",
    "verification_reason": "exact_numeric_match",
    "verification_details": {
        "expected": "42",
    },
}
```

This structure is more useful than a single score. If the rejection-sampling pass rate suddenly drops, `verification_reason` can show whether the issue is format drift, answer parsing, code execution, or task difficulty. For a data flywheel, interpretable failure reasons are as valuable as successful samples.

---

### Step 5: Run Rejection Sampling and Create Feedback Samples

Rejection sampling reads candidates from `data/sampled_traces`, calls verifiers, and selects candidates by prompt:

```bash
python rejection_sampling.py \
  --cold-start data/processed/cold_start_5k.jsonl \
  --sample-dir data/sampled_traces \
  --selected-per-prompt 2 \
  --min-reward 0.8
```

Selection priority:

1. `verifier_pass`
2. `reward_score`
3. `sample_idx`

By default, at most `selected_per_prompt` traces are kept per prompt. Selected records are written to:

```text
data/processed/rejection_selected_10k_30k.jsonl
```

Verifier results for each prompt are written to:

```text
data/verified_candidates/*.jsonl
```

Selected traces are reorganized into SFT format:

```python
selected = {
    "record_id": "math_gsm8k_000001_rs0",
    "source_dataset": "gsm8k",
    "domain": "math",
    "prompt": "A math problem...",
    "messages": [
        {"role": "system", "content": "You are a careful reasoning assistant."},
        {"role": "user", "content": "A math problem..."},
        {"role": "assistant", "content": "Reasoning: ...\nFinal Answer: 42"},
    ],
    "verifier_pass": True,
    "reward_score": 1.0,
}
```

Avoid a common misunderstanding: rejection sampling is not simply "delete all failures." Current training data uses only successful traces, but failed traces are still kept in `verified_candidates`. They support error analysis, verifier repair, hard-case pool construction, and future process reward modeling.

If the target is `10K+` feedback samples, candidate scale must be estimated from pass rate. For example, with four samples per problem, 25% pass rate, and at most one selected sample per prompt, selected sample count can be much lower than candidate count. When scaling, do not only loosen `min_reward`; inspect pass rate, format failure rate, and task difficulty distribution.

---

### Step 6: Merge Second-round SFT Data, Train LoRA, and Evaluate

After rejection sampling, merge cold-start and feedback data:

```bash
python merge_sft_data.py
```

Default outputs:

```text
data/training/merged_sft_data.jsonl
data/training/training_manifest.json
```

The merge deduplicates by prompt and assistant content and records source distribution. Cold-start and feedback samples should keep different `source_stage` values because they serve different roles. Cold-start data provides stable format and basic reasoning style; feedback data comes from model sampling and verifier selection, representing successful traces explored by the current policy.

Demonstration LoRA training:

```bash
python train_lora.py \
  --dataset data/training/merged_sft_data.jsonl \
  --output-dir data/training/lora_ckpt \
  --max-train-samples 1024 \
  --epochs 2
```

Evaluation:

```bash
python eval_gsm8k_math.py \
  --model-path <base-model-path> \
  --adapter-path data/training/lora_ckpt \
  --max-examples 100 \
  --tasks gsm8k,math \
  --backend openai
```

Default evaluation output:

```text
data/reports/eval_results_gsm8k_math.json
```

LoRA and evaluation scripts primarily validate the data loop. They do not guarantee stable score gains after one training run. Results depend on sample scale, sampling quality, verifier strictness, training mixture, learning rate, and evaluation isolation. A working engineering loop and model improvement are related but not identical.

---

## Results and Analysis

The final project output is not a single score table, but a set of auditable data assets. Minimal acceptance should include:

| Artifact | Checkpoint |
| --- | --- |
| `cold_start_5k.jsonl` | Complete fields; `messages` is directly usable for SFT |
| `cold_start_summary.json` | Source and domain distribution are visible |
| `sampled_traces/*.jsonl` | Each prompt has multiple candidate traces |
| `verified_candidates/*.jsonl` | Each candidate has verifier results and failure reasons |
| `rejection_selected_10k_30k.jsonl` | High-score traces are repacked as SFT samples |
| `merged_sft_data.jsonl` | Cold-start and feedback data are merged |
| `training_manifest.json` | Merge scale and domain distribution are recorded |
| `eval_results_gsm8k_math.json` | Base and LoRA results can be compared |

Engineering acceptance has three layers. First, chain acceptance: `pytest -q` passes, and mock mode completes cold start, sampling, verification, rejection sampling, merge, training, and evaluation. Second, real sampling acceptance: vLLM can be called by `sample_traces.py`, results enter `sampled_traces`, and verifiers can process them. Third, effect acceptance: after LoRA training, GSM8K / MATH show stable improvement over the base model. This project prioritizes the first two layers; the third requires larger data and more tuning.

Main cost comes from multi-sample inference and training. If resources are limited:

| Bottleneck | Downgrade Strategy |
| --- | --- |
| Insufficient GPU memory | Lower `max_model_len`, `max_num_seqs`, or prompt concurrency |
| Slow sampling | Reduce `num_samples` from 16 to 4 |
| Low verifier pass rate | Start with math tasks and defer code tasks |
| Too few feedback samples | Increase candidate sampling rather than blindly loosening the verifier |
| Slow LoRA training | Start with `--max-train-samples 1024` for smoke training |
| Slow evaluation | Start with `--max-examples 100`, then expand |

## References

Chen M, Tworek J, Jun H, Yuan Q, Pinto H P O, Kaplan J, Edwards H, Burda Y, Joseph N, Brockman G, others (2021) Evaluating Large Language Models Trained on Code. arXiv preprint arXiv:2107.03374.

Cobbe K, Kosaraju V, Bavarian M, Chen M, Jun H, Kaiser L, Plappert M, Tworek J, Hilton J, Nakano R, Hesse C, Schulman J (2021) Training Verifiers to Solve Math Word Problems. arXiv preprint arXiv:2110.14168.

Guo D, Yang D, Zhang H, Song J, Zhang R, Xu R, Zhu Q, Ma S, Wang P, Bi X, others (2025) DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2501.12948.

Gunther F, Bhatt U, Gupta D, Mukherjee S, others (2025) Open-Thoughts: Exploring Quality, Quantity, Diversity and Creativity in Reasoning Data. arXiv preprint arXiv:2506.04178.

Hendrycks D, Burns C, Kadavath S, Arora A, Basart S, Tang E, Song D, Steinhardt J (2021) Measuring Mathematical Problem Solving with the MATH Dataset. In: Advances in Neural Information Processing Systems 34:24262-24273.

Hui B, Yang J, Cui Z, Yang J, Liu D, Zhang L, Liu B, Yu B, Lu K, Chi K, others (2024) Qwen2.5: A Party of Foundation Models. arXiv preprint arXiv:2412.15115.

Qwen Team (2025) QwQ-32B: Embracing the Power of Reinforcement Learning for Reasoning Models. Qwen Blog.
