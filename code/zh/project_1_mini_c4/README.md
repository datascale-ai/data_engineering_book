# P1 Mini-C4

This project builds a small, reproducible Mini-C4 style preprocessing pipeline on top of a Common Crawl shard.

## Environment

The project now has a reusable conda environment:

```bash
conda activate p1-mini-c4
```

Environment files:

- `environment.yml`: concise, project-facing dependency spec.
- `environment.lock.yml`: full export from the created `p1-mini-c4` environment for stricter reproduction.

The latest validation run was executed inside `p1-mini-c4`, not `base`.

## Deliverables

1. Project goals and scope
   - One WARC shard for a mini-scale dataset build.
   - Single-node CPU preprocessing with local models.
   - English/Chinese split plus final training-ready JSONL export.
2. Data retrieval and parsing
   - Download Common Crawl metadata and one WARC file.
   - Parse HTML responses and extract main text into JSONL.
3. Cleaning, deduplication, and quality filtering
   - Rule-based cleaning.
   - MinHash + LSH deduplication.
   - FastText language split.
   - Language-aware quality filtering with extra spam/menu/adult rejection rules.
4. Serialization and training preparation
   - Deterministic train/val split.
   - Serialized JSONL export with estimated token counts.
   - Train/val shard packaging.
   - Small smoke-test subset for downstream training checks.
5. Evaluation and cost analysis
   - Automatic metrics JSON and Markdown report.
   - Retention rates, language distribution, top domains, storage footprint, and optimization notes.
6. Extension directions
   - Multi-shard scaling, stronger domain filtering, Chinese LM scoring, and downstream tokenizer/pretraining reuse.

## Recommended Run Order

```bash
python src/1_download_data.py
python src/2_process_warc.py
python src/3_clean_data.py
python src/4_deduplicate.py
python src/5_split_lang.py
python src/6_quality_filter.py
python src/7_prepare_training_data.py
python src/9_training_smoke_test.py
python src/8_evaluate_dataset.py
python src/10_run_p1_checks.py
```

## Key Outputs

- `data/processed/extracted_data.jsonl`
- `data/processed/clean_data.jsonl`
- `data/processed/deduplicated_data.jsonl`
- `data/processed/data_en.jsonl`
- `data/processed/data_zh.jsonl`
- `data/processed/final_data_en.jsonl`
- `data/processed/final_data_zh.jsonl`
- `data/processed/final_data.jsonl`
- `data/training/serialized_dataset.jsonl`
- `data/training/train.jsonl`
- `data/training/val.jsonl`
- `data/training/smoke_test.jsonl`
- `data/training/training_manifest.json`
- `data/reports/p1_metrics.json`
- `data/reports/p1_report.md`
- `data/reports/p1_test_results.json`
- `data/reports/p1_test_report.md`

## Notes

- The current mini setup is intentionally small and suitable for local experimentation.
- `src/6_quality_filter.py` now rejects many mixed-language, navigation-heavy, repetitive, and adult/spam pages that previously leaked into the final corpus.
- `src/7_prepare_training_data.py` provides the training-facing packaging that was previously missing from P1.


## Repository Smoke Test

From the repository root, run this project's smoke check through the unified runner:

```bash
python scripts/run_all_project_smoke_tests.py --project P1
```

Expected output: a `P1: PASS` or `P1: FAIL` line plus a report in `smoke_reports/` with the failing command and stderr when a check cannot complete.
