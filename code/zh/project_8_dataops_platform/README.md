# P8 Enterprise DataOps Platform

This project builds a small, reproducible enterprise DataOps platform prototype covering scope definition, architecture, version and experiment lineage, observability, and governance.

## Scope

1. Platform goals and scope
   - Unified target for ingestion, processing, evaluation, release, monitoring, and audit.
   - Explicit tenant, project, role, and permission model.
2. Core architecture
   - Scheduler, metadata, storage, and service layers.
   - Platform APIs, task queues, and UI panels.
3. Version and experiment lineage
   - Dataset versions, experiment runs, lineage graph, and rollback records.
4. Observability and operations
   - Metrics, alerts, audit logs, SLA tracking, and incident reviews.
5. Organization and governance
   - Team interfaces, standard workflows, and exception handling.
6. Extension directions
   - Shared multi-BU platform, stronger policy automation, and disaster recovery.

## Environment

Dedicated conda environment:

```bash
conda activate p8-dataops
```

Environment files:

- `environment.yml`
- `environment.lock.yml`
- `environment.preview.yml`

Recommended creation commands:

```bash
conda env create -f environment.yml
conda env update -n p8-dataops -f environment.lock.yml --prune
conda env create -f environment.preview.yml
```

## Recommended Run Order

```bash
python src/build_platform_specs.py
python src/simulate_platform_ops.py
python src/evaluate_platform.py
python src/render_p8_chapter.py
python src/run_p8_checks.py
```

## Main Outputs

- `data/processed/platform_scope.json`
- `data/processed/architecture_spec.json`
- `data/processed/api_catalog.json`
- `data/processed/task_queues.json`
- `data/processed/governance_policy.json`
- `data/processed/operating_model.json`
- `data/processed/dataset_versions.jsonl`
- `data/processed/experiment_runs.jsonl`
- `data/processed/lineage_graph.json`
- `data/processed/rollback_events.jsonl`
- `data/processed/alerts.jsonl`
- `data/processed/audit_log.jsonl`
- `data/processed/incident_reviews.jsonl`
- `data/processed/sla_report.json`
- `data/console/ui_panels.json`
- `data/reports/p8_report.md`
- `data/reports/p8_chapter_preview.pdf`
- `data/reports/p8_preview_stats.json`
- `data/reports/p8_metrics.json`
- `data/reports/p8_test_results.json`
- `data/reports/p8_test_report.md`


## Repository Smoke Test

From the repository root, run this project's smoke check through the unified runner:

```bash
python scripts/run_all_project_smoke_tests.py --project P8
```

Expected output: a `P8: PASS` or `P8: FAIL` line plus a report in `smoke_reports/` with the failing command and stderr when a check cannot complete.
