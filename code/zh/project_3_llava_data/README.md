# P3 LLaVA Multimodal Instruction Factory

This project builds a small, reproducible multimodal instruction data factory for a LLaVA-style model.

## Scope

The current implementation covers:

1. Project goals and boundaries
   - Image description, region grounding, document-style reading, and multi-image comparison.
   - Small-sample reproducibility with local COCO images and annotations.
2. Image collection and re-description
   - General images from local COCO subsets.
   - Derived document-style cards and chart-style cards for OCR and chart-reading tasks.
3. Instruction construction and alignment
   - Scene description, counting QA, chart QA, OCR summary, and bbox grounding.
   - Multi-image interleaved comparisons.
4. Quality control and spot checks
   - Rule-based validation, low-quality filtering, and bbox visualization outputs.
5. Result presentation and engineering learnings
   - Training split, smoke test, report, metrics, and test artifacts.
6. Extension directions
   - Can be extended to real OCR documents, charts, long-context interleaving, and video QA.

## Environment

Dedicated conda environment:

```bash
conda activate p3-llava
```

Environment files:

- `environment.yml`
- `environment.lock.yml`

Recommended creation commands:

```bash
conda env create -f environment.yml
conda env update -n p3-llava -f environment.lock.yml --prune
```

## Recommended Run Order

```bash
python src/collect_multimodal_assets.py
python src/generate_llava_data.py
python src/alignment.py
python src/interleaved.py
python src/visualize_bbox.py
python src/quality_control.py
python src/prepare_training_data.py
python src/evaluate_factory.py
python src/run_p3_checks.py
```

## Main Outputs

- `data/processed/asset_manifest.jsonl`
- `data/processed/asset_collection_summary.json`
- `data/processed/llava_instruct.jsonl`
- `data/processed/llava_alignment.jsonl`
- `data/processed/llava_interleaved.jsonl`
- `data/processed/quality_audit.jsonl`
- `data/processed/low_quality_flags.jsonl`
- `data/processed/manual_review_samples.jsonl`
- `data/processed/qa_visual_audit.jsonl`
- `data/training/final_llava_dataset.jsonl`
- `data/training/train.jsonl`
- `data/training/val.jsonl`
- `data/training/smoke_test.jsonl`
- `data/training/training_manifest.json`
- `data/reports/p3_report.md`
- `data/reports/p3_metrics.json`
- `data/reports/p3_test_results.json`
- `data/reports/p3_test_report.md`


## Repository Smoke Test

From the repository root, run this project's smoke check through the unified runner:

```bash
python scripts/run_all_project_smoke_tests.py --project P3
```

Expected output: a `P3: PASS` or `P3: FAIL` line plus a report in `smoke_reports/` with the failing command and stderr when a check cannot complete.
