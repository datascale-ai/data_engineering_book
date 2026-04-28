# P4 Synthetic Math and Code Textbook

This project builds a small, reproducible synthetic textbook factory for math and Python code education.

## Scope

The current implementation covers:

1. Project goals and boundaries
   - Math explanation and code explanation chapter generation.
   - Audience, difficulty levels, and chapter output format definitions.
2. Seed pool and chapter templates
   - Curated seeds from GSM8K and MBPP.
   - Chapter-level templates with concept note, example, exercise, and verification sections.
3. Generation, verification, and correction
   - Offline chapter synthesis.
   - Program execution and unit-test verification for both math and code tracks.
4. Quality risk control
   - Checks for missing sections, short chapters, duplicate style, and execution failures.
   - Manual review sample export.
5. Evaluation and cost accounting
   - Coverage, verification pass rate, difficulty distribution, and cost estimates.
6. Extension directions
   - Can expand to problem banks, lecture notes, tutoring data, and curriculum sequencing.

## Environment

Dedicated conda environment:

```bash
conda activate p4-synth
```

Environment files:

- `environment.yml`
- `environment.lock.yml`

Recommended creation commands:

```bash
conda env create -f environment.yml
conda env update -n p4-synth -f environment.lock.yml --prune
```

## Recommended Run Order

```bash
python src/sampler.py
python src/evol.py
python src/sandbox.py
python src/package_textbook.py
python src/quality_control.py
python src/prepare_training_data.py
python src/evaluate_factory.py
python src/run_p4_checks.py
```

## Main Outputs

- `data/processed/seed_pool.jsonl`
- `data/processed/chapter_plan.json`
- `data/processed/synthetic_textbook_chapters.jsonl`
- `data/processed/verified_textbook.jsonl`
- `data/processed/verification_failures.jsonl`
- `data/processed/execution_results.jsonl`
- `data/processed/quality_audit.jsonl`
- `data/processed/low_quality_flags.jsonl`
- `data/processed/manual_review_samples.jsonl`
- `data/processed/curriculum_map.json`
- `data/processed/textbook_catalog.json`
- `data/processed/editorial_style_guide.md`
- `data/training/final_textbook_dataset.jsonl`
- `data/training/train.jsonl`
- `data/training/val.jsonl`
- `data/training/smoke_test.jsonl`
- `data/training/training_manifest.json`
- `books/foundations_of_quantitative_reasoning.md`
- `books/python_problem_solving_workbook.md`
- `books/teacher_guide.md`
- `data/reports/p4_report.md`
- `data/reports/p4_metrics.json`
- `data/reports/p4_test_results.json`
- `data/reports/p4_test_report.md`


## Repository Smoke Test

From the repository root, run this project's smoke check through the unified runner:

```bash
python scripts/run_all_project_smoke_tests.py --project P4
```

Expected output: a `P4: PASS` or `P4: FAIL` line plus a report in `smoke_reports/` with the failing command and stderr when a check cannot complete.
