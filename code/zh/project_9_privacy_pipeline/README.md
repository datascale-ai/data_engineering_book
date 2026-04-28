# P9 Privacy-Preserving Data Pipeline

This project builds a small, reproducible privacy-preserving data processing pipeline covering compliance scope, data classification, access boundaries, de-identification, isolation, privacy technology options, preflight checks, and incident review.

## Scope

1. Scenario and compliance goals
   - Sensitive examples from healthcare support, payroll HR, and financial KYC.
   - Clear auditability, least-privilege, and privacy risk goals.
2. Data classification and access design
   - Sensitivity levels, source-type classification, and role boundaries.
   - Access controls across raw, quarantine, redacted, and audit zones.
3. De-identification, isolation, and secure processing
   - PII detection, masking, tokenization, and quarantine handling.
   - Audit log and suspicious access alerts.
4. Privacy technology integration
   - Differential privacy, TEE, FHE, and tokenization integration notes.
5. Launch checks and postmortem
   - Preflight checklist, incident simulation, and follow-up actions.
6. Extension directions
   - Extend to cross-system privacy orchestration and stronger policy controls.

## Environment

Dedicated conda environment:

```bash
conda activate p9-privacy
```

Environment files:

- `environment.yml`
- `environment.lock.yml`

Recommended creation commands:

```bash
conda env create -f environment.yml
conda env update -n p9-privacy -f environment.lock.yml --prune
```

## Recommended Run Order

```bash
python src/build_privacy_specs.py
python src/run_privacy_pipeline.py
python src/simulate_privacy_ops.py
python src/evaluate_privacy_pipeline.py
python src/run_p9_checks.py
```

## Main Outputs

- `data/processed/compliance_scope.json`
- `data/processed/classification_policy.json`
- `data/processed/access_policy.json`
- `data/processed/privacy_tech_options.json`
- `data/processed/raw_sensitive_records.jsonl`
- `data/processed/classified_records.jsonl`
- `data/processed/redacted_records.jsonl`
- `data/processed/quarantine_records.jsonl`
- `data/processed/audit_log.jsonl`
- `data/processed/access_alerts.jsonl`
- `data/processed/isolation_plan.json`
- `data/processed/preflight_checklist.json`
- `data/processed/incident_simulation.json`
- `data/processed/postmortem_report.json`
- `data/reports/p9_report.md`
- `data/reports/p9_metrics.json`
- `data/reports/p9_test_results.json`
- `data/reports/p9_test_report.md`


## Repository Smoke Test

From the repository root, run this project's smoke check through the unified runner:

```bash
python scripts/run_all_project_smoke_tests.py --project P9
```

Expected output: a `P9: PASS` or `P9: FAIL` line plus a report in `smoke_reports/` with the failing command and stderr when a check cannot complete.
