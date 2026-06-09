# Project 12: A Pedagogical R1 Reasoning Data Flywheel

## Abstract

This project constructs a reproducible data engineering case study around the "Pedagogical R1 Reasoning Data Flywheel," focusing on business objectives, data boundaries, architectural decisions, core implementation, acceptance metrics, and risk controls. The chapter distills installation commands and script details into an engineering retrospective perspective, highlighting the relationships among sample schemas, data flows, failure modes, and deliverables, thereby helping readers translate the methodologies presented earlier into auditable, scalable project assets.

## Keywords

R1; project practice; reproducible data engineering; data pipeline; acceptance metrics

## Project Objectives and Reader Takeaways

This project uses the "Pedagogical R1 Reasoning Data Flywheel" as its core case study. The objective is to build a closed-loop reasoning data pipeline covering Long-CoT cold start, rejection sampling, and recirculated SFT. Upon completing this chapter, readers should be able to identify the key data objects in this scenario, decompose the engineering chain, define acceptance metrics, and transfer the case methodology to comparable data engineering tasks.

## Scenario Constraints and Data Boundaries

The focus is on the reasoning data engineering pipeline; it does not cover a complete reinforcement learning platform or large-scale online training, nor does it claim to reproduce the full training process of DeepSeek-R1. These boundaries make the case reproducible and auditable. When data scale, data sources, permission scope, or deployment environment change, sampling strategies, quality thresholds, operating costs, and compliance requirements must be reassessed.

## Architectural Decisions

This project adopts an architectural path of "cold-start samples → long-chain reasoning → candidate sampling → verification and filtering → rejection sampling → SFT recirculation." This decision prioritizes input/output contract clarity, version traceability, exception localizability, and result verifiability over compressing all logic into a single one-off script run.

## Sample Schema / Data Flow

The core data flow can be summarized as:

Listing P12-1 provides a process or path example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```text
Reasoning seeds -> Long-CoT generation -> Multi-candidate sampling -> Verification/scoring -> Rejection sampling -> Recirculated SFT dataset
```

This excerpt translates the above process into an inspectable, structured representation.

The sample schema should retain at minimum the fields `id`, `source`, `content_or_payload`, `metadata`, `quality_signals`, `split_or_stage`, and `audit_trace`; specific fields are further refined by the data types, downstream tasks, and acceptance methods of this project.

## Core Implementation Excerpts

The main text retains only the key implementation excerpts that illuminate design trade-offs. Complete scripts, long configuration files, run logs, and large files should be placed in the companion repository or appendix; code presentation focuses on input/output contracts, quality thresholds, exception handling, and acceptance interfaces.

## Experiment or Acceptance Metrics

Acceptance metrics include reasoning accuracy, candidate retention rate, verification coverage, long-chain length distribution, recirculated sample quality, and cost per sample. If the project enters production, a course, or a public reproducibility experiment environment, the version number, dependency environment, random seeds, sample spot-check results, and failed-sample post-mortem records should also be logged.

| Acceptance Dimension | Metric / Evidence | Publication Review Criterion |
| --- | --- | --- |
| Candidate generation | Number of multi-path samples, long-chain length distribution, and task-source coverage | Describe the differences between mock, vLLM, and real model sampling |
| Filtering and recirculation | Verifier coverage, candidate retention rate, recirculated sample quality, and format pass rate | Each recirculated sample should be traceable to its original task, candidate trace, and verification result |
| Risk control | Post-mortem of self-reinforcing errors, verifier bias, and overly long trace noise | Do not equate a rejection-sampling pass with an improvement in reasoning capability |

*Table P12-1: Publication Acceptance Table for the Pedagogical R1 Reasoning Data Flywheel*

## Cost, Risk, and Compliance Boundaries

Costs arise primarily from long-chain generation, multi-candidate sampling, and verification. Risks concentrate on self-reinforcing errors, verifier bias, and overly long trace noise. When external data, personal information, copyrighted content, or third-party services are involved, source descriptions, permission status, desensitization strategies, call logs, and manual review records should be retained.

## Common Failure Modes

Common failures include input distribution drift, missing schema fields, quality thresholds that are too loose or too tight, insufficient evaluation sample coverage, unstable model calls, and non-traceable results. When troubleshooting, prioritize locating data boundaries and intermediate artifacts before examining models, toolchains, and deployment environments.

## Reproducibility Resource Description

Reproducibility materials should include data source descriptions, minimal samples, configuration files, run commands, metric scripts, inspection reports, and artifact directories. The main text retains necessary excerpts; complete notebooks, long scripts, and large files are maintained separately as companion resources.

## R1-Style Reasoning Data Flywheel — From Long-CoT Cold Start to Rejection-Sampling Recirculation

### Background and Objectives

The insight brought by reasoning models such as R1 / QwQ (Guo et al. 2025; Qwen Team 2025) is not merely that "models can output longer chains of thought," but that reasoning capability can be decomposed into a runnable data engineering pipeline. Traditional SFT projects typically revolve around a fixed batch of `instruction-response` samples, and the data itself rarely changes after training. The R1-style data flywheel is different: the model generates multiple candidate traces for the same task, the system uses a verifier to determine which traces are correct, which are format-stable, and which are worth recirculating, and then reorganizes the selected successful samples into supervised fine-tuning data for the next round. In this way, data is no longer a static asset prepared before training but becomes part of the model's capability iteration.

For small and medium-sized teams, fully reproducing large-scale RL training is often impractical. Large-scale online sampling, reward modeling, multi-round policy updates, and long-context inference all require continuous GPU resources and complex training-framework governance. A more actionable approach is to first build the "data flywheel" itself: cold-start data can be generated, multi-path sampling can run, the verifier can score automatically, rejection sampling can select high-quality traces, second-round SFT data can be merged, and evaluation scripts can compare changes before and after training. As long as this pipeline is stable, subsequent integration of larger models, more complex rewards, or migration to industry tasks all have a reusable engineering foundation.

This project is built around the code in `code/zh/project_12_r1_flywheel`. The goal is to implement a pedagogical, small-scale, minimal-runnable version of an R1-style reasoning data flywheel. It does not aim to reproduce the complete reinforcement learning process of R1-Zero in one shot, nor is it equivalent to reproducing the full training system of DeepSeek-R1. This chapter retains only the data engineering stages — cold start, candidate sampling, verification and filtering, rejection sampling, and SFT recirculation — and positions benchmark scores in the role of closed-loop validation rather than treating them as the sole objective. The project uses `Qwen2.5-7B-Instruct` (Hui et al. 2024) as the base model by default, and `OpenThoughts` (Guha et al. 2025), `GSM8K` (Cobbe et al. 2021), `MATH-500` (Hendrycks et al. 2021), and `HumanEval` (Chen et al. 2021) as the primary data sources, constructing a closed loop from cold-start SFT to rejection-sampling recirculation.

The entire pipeline can be summarized as:

Listing P12-2 provides a process or path example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```text
OpenThoughts / GSM8K / MATH-500 / HumanEval
  -> cold-start SFT data
  -> vLLM sampled reasoning traces
  -> math / code / format verifier
  -> rejection sampling
  -> merged SFT data
  -> LoRA training and evaluation
```

This excerpt translates the above process into an inspectable, structured representation.

Upon completing this project, readers should understand three things. First, the key engineering objects in an R1-style system are not individual model weights but the task pool, sampled traces, verifier, rejection-sampling results, and training data manifest. Second, the reasoning data flywheel can be validated first using rule-based rewards and supervised recirculation, without entering full RL from the outset. Third, as long as the target task can construct an automatic verifier, this structure can be transferred to SQL generation, code repair, structured extraction, tool calls, or an enterprise-internal question bank.

## 2. Architecture Design: A Six-Component Pedagogical R1 Reasoning Data Flywheel

This project's architecture can be decomposed into six components: cold-start data extraction, multi-path reasoning sampling, verifier pool, rejection sampling, second-round SFT data merging, and training and evaluation. The six components communicate via files and a unified schema, rather than being tightly coupled in a single long script. The benefit of this approach is that each layer can be re-run independently: if sampling fails, only the sampling step needs to be re-run; if the verifier is updated, only rejection sampling needs to be re-run; if training proportions change, only the data merge needs to be redone.

![Figure P12-1](../../images/part11/p12_r1_reasoning_flywheel_architecture.png)
*Figure P12-1: The closed-loop structure from cold-start data extraction, multi-path reasoning sampling, verifier pool, and rejection sampling to second-round SFT merging and training evaluation.*

The first component is **cold-start data extraction**. The corresponding script is `cold_start_data.py`. It extracts samples suitable for SFT from existing data sources and normalizes them into the `messages` format. Math samples are organized with `Reasoning:` and `Final Answer:` sections; code samples are organized with `Reasoning:` and a fenced Python code block. The purpose of cold-start data is not to directly train the highest-performance model, but to give the model a basic reasoning output structure, language style, and parseable format.

The second component is **multi-path reasoning sampling**. The corresponding script is `sample_traces.py`. It generates multiple candidate reasoning traces for the same prompt and records `prompt_id`, `sample_idx`, `raw_trace`, `parsed_answer`, and `generation_params`. The project supports three backends simultaneously: mock, local vLLM Python API, and an external OpenAI-compatible API. In a production environment, the inference service and data processing scripts can be deployed separately to reduce conflicts among CUDA, torch, and vLLM dependencies.

The third component is the **verifier pool**. The corresponding script is `verifier_pool.py`. It implements three categories of rule-based verifiers: math, code, and format. The math verifier extracts the final answer, parses the numeric value, and compares it; the code verifier extracts fenced Python code blocks and runs test cases; the format verifier checks whether structures such as `Reasoning:`, `Final Answer:`, and `Code:` are present. The current verifiers are rule-based rewards rather than complex reward models, but they have the advantages of being stable, inexpensive, and interpretable.

The fourth component is **rejection sampling**. The corresponding script is `rejection_sampling.py`. It reads sampled traces, groups them by prompt, calls the verifier, and then sorts by `verifier_pass`, `reward_score`, and `sample_idx` to retain the best candidates for each question. Retained traces are repackaged into SFT samples and written to `rejection_selected_10k_30k.jsonl`. This step is the core of the data flywheel: the model generates candidates, the system automatically filters them, and successful traces are recirculated for training.

The fifth component is **second-round SFT data merging**. The corresponding script is `merge_sft_data.py`. It merges cold-start data and rejection-sampled data into `merged_sft_data.jsonl` and generates `training_manifest.json`. Although both cold-start samples and recirculated samples enter SFT, they carry different meanings: the former provides the initial format and reasoning style, while the latter preserves successful traces that the model itself explored.

The sixth component is **training and evaluation**. `train_lora.py` provides a minimal LoRA demonstration training, and `eval_gsm8k_math.py` provides a GSM8K/MATH evaluation entry point. In this project, training and evaluation serve as validation interfaces for the data flywheel rather than being written as a complete experiment platform. Their primary function is to verify: can the merged data enter training, and after training, can a unified evaluation script compare the base model against the LoRA adapter?

The main artifacts are as follows:

| Stage | Default Artifact | Description |
| --- | --- | --- |
| Cold-start extraction | `data/processed/cold_start_5k.jsonl` | First-round SFT samples |
| Cold-start statistics | `data/processed/cold_start_summary.json` | Source, domain, and count statistics |
| Multi-path sampling | `data/sampled_traces/*.jsonl` | Model candidate reasoning traces |
| Verifier output | `data/verified_candidates/*.jsonl` | Verification results for each candidate |
| Rejection sampling | `data/processed/rejection_selected_10k_30k.jsonl` | High-scoring traces eligible for recirculation |
| Data merging | `data/training/merged_sft_data.jsonl` | Second-round SFT input |
| Training record | `data/training/training_manifest.json` | Training data composition |
| Evaluation results | `data/reports/eval_results_gsm8k_math.json` | GSM8K/MATH comparison results |

*Table P12-2: Stage and Description Reference Table*

---

## 3. Step-by-Step Implementation: From Cold-Start Samples to a Recirculable SFT Dataset

### Step 1: Prepare the Environment and Task Data

The project recommends running in an isolated conda environment. After entering the code directory, create the environment:

Listing P12-3 provides a command-line run example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```bash
cd code/zh/project_12_r1_flywheel
conda env create -f environment.yml
conda activate p12-r1-flywheel
```

This excerpt translates the above process into an inspectable, structured representation.

Before formal sampling, run the tests first to confirm that the mock pipeline and base modules are intact:

Listing P12-4 provides a command-line run example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```bash
pytest -q
```

This excerpt translates the above process into an inspectable, structured representation.

The current `tests/test_pipeline.py` covers cold-start extraction, math/code/format verifiers, mock sampling, rejection sampling, SFT merging, mock LoRA training, and mock evaluation. A passing test only indicates that the engineering skeleton is intact; it does not mean that real vLLM sampling or large-scale training has been completed. However, if the tests do not pass, one should not proceed directly to long-running tasks.

Input data can be understood by task type: `OpenThoughts` provides Long-CoT cold-start samples, `GSM8K` and `MATH-500` provide math problems, and `HumanEval` provides coding tasks. In real deployments, data can come from public datasets or be replaced with an enterprise-internal question bank; as long as samples can be organized into a unified schema and provide reference answers or test cases, they can enter this flywheel.

---

### Step 2: Extract Cold-Start SFT Data

The goal of the cold-start stage is to organize data from different sources into a unified SFT message format. The command is as follows:

Listing P12-5 provides a command-line run example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```bash
python cold_start_data.py \
  --max-openthoughts 5000 \
  --max-math 100 \
  --max-gsm8k 100 \
  --max-code 100
```

This excerpt translates the above process into an inspectable, structured representation.

The default output files are:

Listing P12-6 provides a process or path example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```text
data/processed/cold_start_5k.jsonl
data/processed/cold_start_summary.json
```

This excerpt translates the above process into an inspectable, structured representation.

Each cold-start sample contains `record_id`, `source_dataset`, `domain`, `prompt`, `reference_reasoning`, `reference_answer`, and `messages`. The `messages` field is what training actually consumes; its structure is approximately as follows:

Listing P12-7 provides a Python implementation excerpt illustrating the input/output relationships, structural constraints, or execution modes in this section.
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
        {"role": "assistant", "content": "Reasoning: ...\nFinal Answer: 42"}
    ],
}
```

This excerpt translates the above process into an inspectable, structured representation.

For coding tasks, the `assistant` content becomes:

Listing P12-8 provides a process or path example illustrating the input/output relationships, structural constraints, or execution modes in this section.
````text
Reasoning: explain the implementation idea
Code:
```python
def solve(...):
    ...
```
````

This excerpt translates the above process into an inspectable, structured representation.

There is an easily overlooked detail here. The `canonical_solution` in `HumanEval` often contains only the function body; if used directly for training, the model may learn incomplete code fragments. The project implements `render_humaneval_solution(...)` in `pipeline_utils.py`, which concatenates the function signature from the prompt with the `canonical_solution` to form a complete function definition, making `reference_answer` more suitable for training and verification.

After cold-start data is generated, `cold_start_summary.json` should be inspected first. The three key items to check are: whether the total number of samples meets expectations, whether `domain_distribution` is overly skewed toward a single task, and whether spot-checking the `messages` field reveals empty responses or format corruption. If data quality at this step is insufficient, subsequent sampling and rejection sampling will suffer amplified contamination.

---

### Step 3: Generate Multi-Path Reasoning Traces Using Mock or vLLM

The input to the sampling stage is the prompts from `cold_start_5k.jsonl`; the output is multiple candidate reasoning traces per prompt. To validate the pipeline first, the mock backend can be used:

Listing P12-9 provides a command-line run example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```bash
python sample_traces.py \
  --input data/processed/cold_start_5k.jsonl \
  --output-dir data/sampled_traces \
  --num-examples 20 \
  --num-samples 2 \
  --backend mock \
  --force-mock
```

This excerpt translates the above process into an inspectable, structured representation.

The mock backend is not used for evaluating real model capability; it is used only to check whether the data pipeline can continue flowing downstream. For real sampling, the vLLM service can be started and called via the OpenAI-compatible API:

Listing P12-10 provides a command-line run example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```bash
bash scripts/serve_qwen_vllm.sh
```

This excerpt translates the above process into an inspectable, structured representation.

Then run the sampling script:

Listing P12-11 provides a command-line run example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```bash
python sample_traces.py \
  --input data/processed/cold_start_5k.jsonl \
  --output-dir data/sampled_traces \
  --num-examples 100 \
  --num-samples 4 \
  --backend openai \
  --parallel-prompts 4
```

This excerpt translates the above process into an inspectable, structured representation.

The core fields of a sampling record are as follows:

Listing P12-12 provides a Python implementation excerpt illustrating the input/output relationships, structural constraints, or execution modes in this section.
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

This excerpt translates the above process into an inspectable, structured representation.

The most important aspect of the sampling stage is retaining the complete `raw_trace` rather than keeping only the final answer. The complete trace has three downstream uses: first, it is passed to the verifier to determine correctness; second, high-quality candidates are recirculated for training; third, it serves as an error analysis sample to help identify whether the model failed due to format errors, answer errors, or reasoning-path errors.

If VRAM or throughput is insufficient, `num_samples`, `parallel_prompts`, and `max_tokens` can be reduced. These reductions affect only scale and should not alter the data format.

---

### Step 4: Build Math, Code, and Format Verifiers

The verifier is the core of this project because it determines which candidates can become recirculated data. The verifiers in the current `verifier_pool.py` fall into three categories.

The math verifier first extracts the final answer from the model output, prioritizing patterns such as `\boxed{}` and `Final Answer:`, then attempts to parse the numeric value. If both the predicted value and the reference answer can be parsed as numbers, a tolerance-based comparison is performed; otherwise, a normalized string comparison is used. It returns `verifier_pass`, `reward_score`, `parsed_answer`, and `reason`.

The code verifier extracts fenced Python code blocks or code snippets containing `def`, then runs them together with test cases. A passing test receives a high score; missing a code block, execution timeout, thrown exceptions, or test failures all have their specific reasons recorded. The current implementation is suited to HumanEval-style tasks and is not equivalent to a full code evaluation platform, but it is sufficient to support a rejection-sampling prototype.

The format verifier checks whether the output structure is stable. A math response should contain at least `Reasoning:` and a parseable final answer; a code response should contain at least `Reasoning:` and a Python code block. Many candidate traces are not entirely wrong in content but fail answer extraction due to format instability. The format verifier's role is to flag such samples early.

A verifier output can be understood as:

Listing P12-13 provides a Python implementation excerpt illustrating the input/output relationships, structural constraints, or execution modes in this section.
```python
verdict = {
    "verifier_type": "math",
    "verifier_pass": True,
    "format_pass": True,
    "reward_score": 1.0,
    "parsed_answer": "42",
    "verification_reason": "exact_numeric_match",
    "verification_details": {
        "expected": "42"
    },
}
```

This excerpt translates the above process into an inspectable, structured representation.

This structure is more valuable than a single score. If the rejection-sampling pass rate drops suddenly later on, the `verification_reason` field allows one to diagnose whether the problem originates from format drift, answer parsing, code execution, or the task itself being too difficult. For a data flywheel, interpretable failure reasons are just as important as successful samples.

---

### Step 5: Rejection Sampling and Generating Recirculated Samples

Rejection sampling reads candidates from `data/sampled_traces`, calls the verifier, and then groups by prompt to filter. The command is as follows:

Listing P12-14 provides a command-line run example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```bash
python rejection_sampling.py \
  --cold-start data/processed/cold_start_5k.jsonl \
  --sample-dir data/sampled_traces \
  --selected-per-prompt 2 \
  --min-reward 0.8
```

This excerpt translates the above process into an inspectable, structured representation.

The filtering priority is:

1. `verifier_pass`
2. `reward_score`
3. `sample_idx`

By default, at most `selected_per_prompt` traces are retained per prompt. Selected records are written to:

Listing P12-15 provides a process or path example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```text
data/processed/rejection_selected_10k_30k.jsonl
```

This excerpt translates the above process into an inspectable, structured representation.

Meanwhile, the verification results for each prompt are written to:

Listing P12-16 provides a process or path example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```text
data/verified_candidates/*.jsonl
```

This excerpt translates the above process into an inspectable, structured representation.

The output samples from rejection sampling are reorganized into SFT format:

Listing P12-17 provides a Python implementation excerpt illustrating the input/output relationships, structural constraints, or execution modes in this section.
```python
selected = {
    "record_id": "math_gsm8k_000001_rs0",
    "source_dataset": "gsm8k",
    "domain": "math",
    "prompt": "A math problem...",
    "messages": [
        {"role": "system", "content": "You are a careful reasoning assistant."},
        {"role": "user", "content": "A math problem..."},
        {"role": "assistant", "content": "Reasoning: ...\nFinal Answer: 42"}
    ],
    "verifier_pass": True,
    "reward_score": 1.0,
}
```

This excerpt translates the above process into an inspectable, structured representation.

One misconception must be avoided here: rejection sampling is not simply "deleting all failed samples." The current training data uses only successful traces, but failed traces are still saved in `verified_candidates`. They can be used to analyze the model's common errors, fix verifier bugs, build a hard-case pool, or train a reward model in subsequent stages.

If the goal is to generate `10K+` recirculated samples, the candidate volume must be back-calculated from the pass rate. For example, when sampling 4 traces per question with a 25% pass rate and retaining at most 1 trace per question, the final number of selected samples will be noticeably lower than the total number of candidates. When scaling up, one should not simply relax `min_reward`; the pass rate, format failure rate, and task difficulty distribution should also be examined.

---

### Step 6: Merge Second-Round SFT Data, Train LoRA, and Evaluate

After rejection sampling is complete, merge the cold-start data with the recirculated data:

Listing P12-18 provides a command-line run example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```bash
python merge_sft_data.py
```

This excerpt translates the above process into an inspectable, structured representation.

Default output:

Listing P12-19 provides a process or path example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```text
data/training/merged_sft_data.jsonl
data/training/training_manifest.json
```

This excerpt translates the above process into an inspectable, structured representation.

During merging, deduplication is performed on prompt and assistant content, and the source distribution is recorded. Cold-start data and recirculated data should retain different `source_stage` values in training, because the two serve different roles. Cold-start samples primarily provide a stable format and baseline reasoning style, while recirculated samples come from model sampling and verifier filtering, representing successful traces already explored by the current policy.

The LoRA demonstration training command is:

Listing P12-20 provides a command-line run example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```bash
python train_lora.py \
  --dataset data/training/merged_sft_data.jsonl \
  --output-dir data/training/lora_ckpt \
  --max-train-samples 1024 \
  --epochs 2
```

This excerpt translates the above process into an inspectable, structured representation.

The evaluation command is:

Listing P12-21 provides a command-line run example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```bash
python eval_gsm8k_math.py \
  --model-path <base-model-path> \
  --adapter-path data/training/lora_ckpt \
  --max-examples 100 \
  --tasks gsm8k,math \
  --backend openai
```

This excerpt translates the above process into an inspectable, structured representation.

Evaluation results are written by default to:

Listing P12-22 provides a process or path example illustrating the input/output relationships, structural constraints, or execution modes in this section.
```text
data/reports/eval_results_gsm8k_math.json
```

This excerpt translates the above process into an inspectable, structured representation.

It should be emphasized that LoRA and the evaluation script in this project are primarily used to validate the data closed loop, not to guarantee stable metric gains from a single training run. Final gains depend heavily on sample scale, sampling quality, verifier strictness, training data proportions, learning rate, and evaluation set isolation. Engineering closed-loop validation and model performance improvement are two related but non-equivalent outcomes.

---

## Results Presentation and Analysis

The final output of this project is not a single score table but a set of reviewable data assets. The minimum acceptable results should include:

| Artifact | Checkpoint |
| --- | --- |
| `cold_start_5k.jsonl` | Fields are complete; `messages` can be used directly for SFT |
| `cold_start_summary.json` | Source and domain distribution is visible |
| `sampled_traces/*.jsonl` | The same prompt has multiple candidate traces |
| `verified_candidates/*.jsonl` | Each candidate has a verifier result and failure reason |
| `rejection_selected_10k_30k.jsonl` | High-scoring traces are repackaged as SFT samples |
| `merged_sft_data.jsonl` | Cold-start and recirculated data are merged |
| `training_manifest.json` | Merge scale and domain distribution are recorded |
| `eval_results_gsm8k_math.json` | Base and LoRA evaluation results can be compared |

*Table P12-3: Artifact and Checkpoint Reference Table*

From an engineering perspective, acceptance can be divided into three tiers. The first tier is pipeline acceptance: `pytest -q` passes, and mock mode can complete cold start, sampling, verification, rejection sampling, merging, training, and evaluation. The second tier is real-sampling acceptance: the vLLM service can be called by `sample_traces.py`, sampling results enter `sampled_traces`, and can be processed by the verifier. The third tier is performance acceptance: after LoRA training, there is a stable gain over the base model on GSM8K/MATH. The current project prioritizes the first two tiers; the third tier requires larger-scale data and multiple rounds of hyperparameter tuning.

In terms of cost, the primary expenses come from multi-path sampling and training. If resources are constrained, the following fallback strategies can be applied:

| Resource Bottleneck | Fallback Strategy |
| --- | --- |
| Insufficient VRAM | Reduce `max_model_len`, `max_num_seqs`, or concurrent prompts |
| Sampling too slow | Reduce `num_samples` from 16 to 4 |
| Low verifier pass rate | Run only math tasks first, defer code tasks |
| Insufficient recirculated samples | Expand candidate sampling rather than blindly relaxing the verifier |
| Slow LoRA training | First run a smoke train with `--max-train-samples 1024` |
| Long evaluation time | Start with `--max-examples 100`, then expand the evaluation scale |

*Table P12-4: Resource Bottleneck and Fallback Strategy Reference Table*

## Chapter Summary

This chapter uses the "Pedagogical R1 Reasoning Data Flywheel" as a case study to demonstrate the engineering organization of a closed-loop reasoning data pipeline covering Long-CoT cold start, rejection sampling, and recirculated SFT. The primary value of this case lies in placing task definition, data boundaries, architectural decisions, sample schemas, metric acceptance, and reproducibility resources in a single chain, transforming the project from a sequence of operational steps into a reviewable case study.

The boundaries of this case must also be clearly preserved. The focus is on a pedagogical, small-scale reasoning data engineering pipeline; it does not cover a complete reinforcement learning platform, large-scale online training, or the full reproduction of DeepSeek-R1 training. In scenarios with larger scale, higher risk, or stricter compliance requirements, data sources, permission status, manual review rates, operating costs, and failure rollback plans should be reassessed.

As part of Part 14, this chapter corresponds to the project-level validation of the methods presented in earlier parts. Readers can use this case alongside the data recipes of Part 13, the platform governance chapters in the earlier sections, and the checklists in the appendix to form a closed loop from methodological understanding to engineering delivery.

## References

Chen M, Tworek J, Jun H, Yuan Q, Pinto H P O, Kaplan J, Edwards H, Burda Y, Joseph N, Brockman G, others (2021) Evaluating Large Language Models Trained on Code (HumanEval). arXiv preprint arXiv:2107.03374.

Cobbe K, Kosaraju V, Bavarian M, Chen M, Jun H, Kaiser L, Plappert M, Tworek J, Hilton J, Nakano R, Hesse C, Schulman J (2021) Training Verifiers to Solve Math Word Problems (GSM8K). arXiv preprint arXiv:2110.14168.

Guo D, Yang D, Zhang H, Song J, Zhang R, Xu R, Zhu Q, Ma S, Wang P, Bi X, others (2025) DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2501.12948.

Guha E, Marten R, Keh S, Raoof N, Smyrnis G, Bansal H, Nezhurina M, Mercat J, Vu T, Sprague Z, others (2025) OpenThoughts: Data Recipes for Reasoning Models. arXiv preprint arXiv:2506.04178.

Hendrycks D, Burns C, Kadavath S, Arora A, Basart S, Tang E, Song D, Steinhardt J (2021) Measuring Mathematical Problem Solving with the MATH Dataset. In: Advances in Neural Information Processing Systems 34:24262-24273.

Hui B, Yang J, Cui Z, Yang J, Liu D, Zhang L, Liu B, Yu B, Lu K, Chi K, others (2024) Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115.

Qwen Team (2025) QwQ-32B: Embracing the Power of Reinforcement Learning for Reasoning Models. Qwen Blog.
