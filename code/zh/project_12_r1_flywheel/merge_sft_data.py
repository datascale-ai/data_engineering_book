from __future__ import annotations

from collections import Counter
from pathlib import Path

from pipeline_utils import (
    PROCESSED_DIR,
    TRAINING_DIR,
    dump_run_report,
    ensure_standard_dirs,
    load_jsonl,
    make_arg_parser,
    utc_ts,
    write_json,
    write_jsonl,
)

COLD_START_FILE = PROCESSED_DIR / "cold_start_5k.jsonl"
REJECTION_FILE = PROCESSED_DIR / "rejection_selected_10k_30k.jsonl"
MERGED_FILE = TRAINING_DIR / "merged_sft_data.jsonl"
MANIFEST_FILE = TRAINING_DIR / "training_manifest.json"


def merge_sft_data(cold_start_path: Path, rejection_path: Path, output_path: Path) -> dict:
    ensure_standard_dirs()
    cold_rows = load_jsonl(cold_start_path)
    rejection_rows = load_jsonl(rejection_path)

    merged = []
    seen = set()
    for source_stage, rows in [("cold_start", cold_rows), ("rejection_sampling", rejection_rows)]:
        for row in rows:
            key = (row["prompt"], row["messages"][-1]["content"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "record_id": row["record_id"],
                    "source_stage": source_stage,
                    "source_dataset": row["source_dataset"],
                    "domain": row["domain"],
                    "messages": row["messages"],
                }
            )

    write_jsonl(merged, output_path)
    summary = {
        "created_at": utc_ts(),
        "merged_records": len(merged),
        "cold_start_records": len(cold_rows),
        "rejection_records": len(rejection_rows),
        "domain_distribution": dict(Counter(row["domain"] for row in merged)),
    }
    write_json(summary, MANIFEST_FILE)
    dump_run_report(
        TRAINING_DIR / "merge_report.md",
        "Merged SFT Data Report",
        [("Summary", summary)],
    )
    return summary


def main() -> None:
    parser = make_arg_parser("Merge cold-start and rejection-sampled data.")
    parser.add_argument("--cold-start", type=str, default=str(COLD_START_FILE))
    parser.add_argument("--rejection", type=str, default=str(REJECTION_FILE))
    parser.add_argument("--output", type=str, default=str(MERGED_FILE))
    args = parser.parse_args()
    print(merge_sft_data(Path(args.cold_start), Path(args.rejection), Path(args.output)))


if __name__ == "__main__":
    main()
