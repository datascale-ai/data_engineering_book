# P5 Multimodal RAG Financial Report Assistant

This project builds a small, reproducible offline RAG assistant for a Chinese annual report PDF.

## Scope

The current implementation covers:

1. Scenario and goals
   - Financial report QA, chart/table-oriented hinting, and page localization.
   - Accuracy, citation clarity, and response traceability as the main objectives.
2. Document ingestion and multimodal parsing
   - PDF text extraction, page rendering, and page/block indexing.
   - Heuristic detection of paragraph, table-like, chart-like, and footnote blocks.
3. Retrieval and answering chain
   - Page-level recall, block-level reranking, and evidence assembly.
   - Offline answer synthesis with page citations and page-image references.
4. Evaluation and error repair
   - Retrieval hit rate, citation accuracy, and answer keyword checks.
   - Reference query set for regression testing.
5. Cost, latency, and deployment
   - No external API cost in the default pipeline.
   - Fully local parsing and retrieval.
6. Extension directions
   - Can be extended to prospectuses, contracts, and policy documents.

## Environment

Dedicated conda environment:

```bash
conda activate p5-rag
```

Environment files:

- `environment.yml`
- `environment.lock.yml`

System tools used by the default pipeline:

- `pdftotext`
- `pdftoppm`

Recommended creation commands:

```bash
conda env create -f environment.yml
conda env update -n p5-rag -f environment.lock.yml --prune
```

## Recommended Run Order

```bash
python src/index.py
python src/evaluate_rag.py
python src/run_p5_checks.py
```

## Main Outputs

- `data/processed/page_units.jsonl`
- `data/processed/block_units.jsonl`
- `data/processed/rag_index.json`
- `data/page_images/`
- `data/eval/reference_questions.jsonl`
- `data/eval/evaluation_results.jsonl`
- `data/eval/failure_replay.jsonl`
- `data/reports/p5_report.md`
- `data/reports/p5_metrics.json`
- `data/reports/p5_test_results.json`
- `data/reports/p5_test_report.md`


## Repository Smoke Test

From the repository root, run this project's smoke check through the unified runner:

```bash
python scripts/run_all_project_smoke_tests.py --project P5
```

Expected output: a `P5: PASS` or `P5: FAIL` line plus a report in `smoke_reports/` with the failing command and stderr when a check cannot complete.
