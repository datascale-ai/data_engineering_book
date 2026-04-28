# P2 Legal SFT Factory

This project builds a small, reproducible legal-domain SFT data factory.

## Scope

The current implementation focuses on Chinese legal texts and covers:

1. Scenario and goals
   - Legal QA, statute explanation, and case analysis.
   - Professional, stable, and compliant legal-answer style.
2. Seed data and instruction system
   - PDF parsing from core law documents.
   - Seed catalog creation and task taxonomy export.
3. Synthetic expansion and distillation
   - Offline teacher-style synthesis from law articles.
   - Judge-style filtering plus rejected-sample logging.
4. QA and preference enhancement
   - Review records, preference pairs, and high-risk refusal samples.
5. Lightweight downstream validation
   - Sampled paired audit over accepted vs. rejected legal responses.
6. Evaluation and cost accounting
   - Coverage, style consistency, risk handling, and review-cost estimates.
7. Extension directions
   - Can be extended to finance, medical, or tax verticals.

## Environment

Dedicated conda environment:

```bash
conda activate p2-legal-sft
```

Environment files:

- `environment.yml`
- `environment.lock.yml`

Recommended creation commands:

```bash
conda env create -f environment.yml
conda env update -n p2-legal-sft -f environment.lock.yml --prune
```

Use `environment.yml` for a lightweight reproducible setup, and `environment.lock.yml` when you want to mirror the exact validated environment used for the saved reports and checks.

## Recommended Run Order

```bash
python src/data_processing.py
python src/generate_instructions.py
python src/build_preference_data.py
python src/prepare_training_data.py
python src/downstream_validation.py
python src/evaluate_factory.py
python src/run_p2_checks.py
```

## Main Outputs

- `data/processed/raw_chunks.jsonl`
- `data/processed/legal_seed_dataset.jsonl`
- `data/processed/instruction_taxonomy.json`
- `data/processed/domain_expert_sft.jsonl`
- `data/processed/synthetic_candidates_rejected.jsonl`
- `data/processed/legal_preference_pairs.jsonl`
- `data/processed/legal_qa_review.jsonl`
- `data/processed/legal_risk_refusal_sft.jsonl`
- `data/processed/legal_risk_register.jsonl`
- `data/training/final_sft_dataset.jsonl`
- `data/training/train.jsonl`
- `data/training/val.jsonl`
- `data/training/smoke_test.jsonl`
- `data/training/training_manifest.json`
- `data/reports/p2_downstream_validation.json`
- `data/reports/p2_downstream_validation.md`
- `data/reports/p2_report.md`
- `data/reports/p2_metrics.json`
- `data/reports/p2_test_results.json`
- `data/reports/p2_test_report.md`


## Repository Smoke Test

From the repository root, run this project's smoke check through the unified runner:

```bash
python scripts/run_all_project_smoke_tests.py --project P2
```

Expected output: a `P2: PASS` or `P2: FAIL` line plus a report in `smoke_reports/` with the failing command and stderr when a check cannot complete.
