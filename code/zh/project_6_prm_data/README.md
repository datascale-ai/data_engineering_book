# P6 CoT and PRM Data Factory

This project builds a small, reproducible CoT reasoning dataset and PRM training dataset.

## Scope

The current implementation covers:

1. Project goals and task setting
   - Math and code reasoning tasks.
   - Final-answer validation and step-level process supervision.
2. Reasoning trace generation
   - CoT, scratchpad-style steps, positive/negative/repair traces.
   - Step labels for correct, incorrect, and repaired reasoning.
3. Automatic validation and scoring
   - Rule checks for math answers.
   - Execution and unit tests for code traces.
   - Reward bucket assignment.
4. PRM data organization
   - Step-level JSONL slices.
   - Train/val/smoke splits.
5. Experimental results and review
   - Outcome-only versus process supervision comparison.
   - Quality issues and reward-bucket diagnostics.
6. Extension directions
   - Can extend to science reasoning, table reasoning, and agent planning.

## Environment

Dedicated conda environment:

```bash
conda activate p6-prm
```

Environment files:

- `environment.yml`
- `environment.lock.yml`

Recommended creation commands:

```bash
conda env create -f environment.yml
conda env update -n p6-prm -f environment.lock.yml --prune
```

## Recommended Run Order

```bash
python src/sampler.py
python src/generate_traces.py
python src/validate_and_score.py
python src/prepare_prm_data.py
python src/evaluate_prm.py
python src/run_p6_checks.py
```

## Main Outputs

- `data/processed/seed_pool.jsonl`
- `data/processed/task_spec.json`
- `data/processed/cot_traces.jsonl`
- `data/processed/trace_summary.json`
- `data/processed/validated_traces.jsonl`
- `data/processed/step_rewards.jsonl`
- `data/processed/validation_summary.json`
- `data/training/prm_step_dataset.jsonl`
- `data/training/train.jsonl`
- `data/training/val.jsonl`
- `data/training/smoke_test.jsonl`
- `data/training/training_manifest.json`
- `data/reports/p6_report.md`
- `data/reports/p6_metrics.json`
- `data/reports/p6_test_results.json`
- `data/reports/p6_test_report.md`


## Repository Smoke Test

From the repository root, run this project's smoke check through the unified runner:

```bash
python scripts/run_all_project_smoke_tests.py --project P6
```

Expected output: a `P6: PASS` or `P6: FAIL` line plus a report in `smoke_reports/` with the failing command and stderr when a check cannot complete.
