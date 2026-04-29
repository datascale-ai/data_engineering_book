# P7 Agent Tool-Use Data Factory

This project builds a small, reproducible agent tool-use dataset with tool schema definitions, successful trajectories, recovery trajectories, multi-turn memory cases, and safety refusal samples.

## Scope

1. Scenario definition and tool boundaries
   - Search, customer DB, calendar, Python execution, and memory tools.
   - Explicit high-risk boundaries for sensitive data, destructive code, and unauthorized booking.
2. Tool schema and trajectory templates
   - Tool descriptions, parameters, returns, and error codes.
   - Single-step, multi-step, recovery, memory, and safety templates.
3. Success and recovery trajectories
   - Successful tool calls, argument fixes, fallback retries, and structured observations.
4. Multi-turn state and memory
   - Memory write/read trajectories across multiple user turns.
   - State-aware recovery after memory misses.
5. Evaluation and safety governance
   - Tool-call success rate, recovery success rate, unsafe block rate, and unauthorized-call checks.
6. Extension directions
   - Expand from tool use to broader agent workflow and cross-session orchestration.

## Environment

Dedicated conda environment:

```bash
conda activate p7-tooluse
```

Environment files:

- `environment.yml`
- `environment.lock.yml`

Recommended creation commands:

```bash
conda env create -f environment.yml
conda env update -n p7-tooluse -f environment.lock.yml --prune
```

## Recommended Run Order

```bash
python src/build_tooling.py
python src/generate_trajectories.py
python src/simulate_tool_env.py
python src/prepare_agent_dataset.py
python src/evaluate_tooluse.py
python src/run_p7_checks.py
```

## Main Outputs

- `data/processed/tool_schemas.json`
- `data/processed/trajectory_templates.json`
- `data/processed/task_specs.json`
- `data/processed/raw_trajectories.jsonl`
- `data/processed/executed_trajectories.jsonl`
- `data/processed/tool_execution_log.jsonl`
- `data/processed/execution_summary.json`
- `data/training/agent_tooluse_dataset.jsonl`
- `data/training/train.jsonl`
- `data/training/val.jsonl`
- `data/training/smoke_test.jsonl`
- `data/training/training_manifest.json`
- `data/reports/p7_report.md`
- `data/reports/p7_metrics.json`
- `data/reports/p7_test_results.json`
- `data/reports/p7_test_report.md`


## Repository Smoke Test

From the repository root, run this project's smoke check through the unified runner:

```bash
python scripts/run_all_project_smoke_tests.py --project P7
```

Expected output: a `P7: PASS` or `P7: FAIL` line plus a report in `smoke_reports/` with the failing command and stderr when a check cannot complete.
