from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline_utils import (
    LOGS_DIR,
    PROCESSED_DIR,
    SAMPLED_DIR,
    VERIFIED_DIR,
    dump_run_report,
    ensure_standard_dirs,
    load_json,
    load_jsonl,
    make_arg_parser,
    setup_logging,
    utc_ts,
    write_json,
    write_jsonl,
)
from verifier_pool import verify_candidate

COLD_START_FILE = PROCESSED_DIR / "cold_start_5k.jsonl"
REJECTION_FILE = PROCESSED_DIR / "rejection_selected_10k_30k.jsonl"
SUMMARY_FILE = PROCESSED_DIR / "rejection_sampling_summary.json"


def _index_examples(path: Path) -> dict[str, dict[str, Any]]:
    return {row["record_id"]: row for row in load_jsonl(path)}


def _load_all_trace_rows(sample_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(sample_dir.glob("*.jsonl")):
        rows.extend(load_jsonl(path))
    return rows


def run_rejection_sampling(
    cold_start_path: Path,
    sample_dir: Path,
    selected_per_prompt: int,
    min_reward: float,
    output_path: Path,
) -> dict[str, Any]:
    ensure_standard_dirs()
    logger = setup_logging("rejection_sampling", LOGS_DIR / "rejection_sampling.log")
    example_index = _index_examples(cold_start_path)
    trace_rows = _load_all_trace_rows(sample_dir)

    grouped_verified: dict[str, list[dict[str, Any]]] = defaultdict(list)
    verified_rows: list[dict[str, Any]] = []
    for row in trace_rows:
        example = example_index[row["prompt_id"]]
        try:
            verdict = verify_candidate(example, row["raw_trace"])
        except Exception as exc:
            logger.warning("verification failed for %s sample %s: %s", row["prompt_id"], row["sample_idx"], exc)
            verdict = {
                "verifier_type": example["domain"],
                "verifier_pass": False,
                "format_pass": False,
                "reward_score": 0.0,
                "parsed_answer": None,
                "verification_reason": f"verification_exception:{type(exc).__name__}",
                "verification_details": {"error": str(exc)},
            }
        enriched = {**row, **verdict}
        verified_rows.append(enriched)
        grouped_verified[row["prompt_id"]].append(enriched)

    for prompt_id, rows in grouped_verified.items():
        shard = VERIFIED_DIR / f"{prompt_id}.jsonl"
        write_jsonl(rows, shard)

    selected: list[dict[str, Any]] = []
    pass_counter = Counter()
    for prompt_id, rows in grouped_verified.items():
        rows = sorted(
            rows,
            key=lambda item: (
                item["verifier_pass"],
                item["reward_score"],
                -item["sample_idx"],
            ),
            reverse=True,
        )
        kept = 0
        for row in rows:
            if row["reward_score"] < min_reward:
                continue
            if kept >= selected_per_prompt:
                break
            selected.append(
                {
                    "record_id": f"{prompt_id}_sample_{row['sample_idx']}",
                    "source_stage": "rejection_sampling",
                    "prompt_id": prompt_id,
                    "source_dataset": row["source_dataset"],
                    "domain": row["domain"],
                    "prompt": row["prompt"],
                    "raw_trace": row["raw_trace"],
                    "parsed_answer": row["parsed_answer"],
                    "reward_score": row["reward_score"],
                    "verifier_type": row["verifier_type"],
                    "verifier_pass": row["verifier_pass"],
                    "selected": True,
                    "messages": [
                        {"role": "system", "content": "You are a careful reasoning assistant."},
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["raw_trace"]},
                    ],
                }
            )
            kept += 1
            pass_counter[row["domain"]] += 1

    write_jsonl(selected, output_path)
    verified_total = len(verified_rows)
    verified_pass = sum(1 for row in verified_rows if row["verifier_pass"])
    summary = {
        "created_at": utc_ts(),
        "verified_total": verified_total,
        "verified_pass": verified_pass,
        "selected_total": len(selected),
        "selected_per_prompt": selected_per_prompt,
        "min_reward": min_reward,
        "selection_by_domain": dict(pass_counter),
        "pass_rate": round(verified_pass / verified_total, 4) if verified_total else 0.0,
    }
    write_json(summary, SUMMARY_FILE)
    dump_run_report(
        PROCESSED_DIR / "rejection_sampling_report.md",
        "Rejection Sampling Report",
        [("Summary", summary)],
    )
    logger.info("selected %s traces from %s verified candidates", len(selected), verified_total)
    return summary


def main() -> None:
    parser = make_arg_parser("Run rule-based verification and rejection sampling.")
    parser.add_argument("--cold-start", type=str, default=str(COLD_START_FILE))
    parser.add_argument("--sample-dir", type=str, default=str(SAMPLED_DIR))
    parser.add_argument("--selected-per-prompt", type=int, default=2)
    parser.add_argument("--min-reward", type=float, default=0.8)
    parser.add_argument("--output", type=str, default=str(REJECTION_FILE))
    args = parser.parse_args()
    summary = run_rejection_sampling(
        cold_start_path=Path(args.cold_start),
        sample_dir=Path(args.sample_dir),
        selected_per_prompt=args.selected_per_prompt,
        min_reward=args.min_reward,
        output_path=Path(args.output),
    )
    print(summary)


if __name__ == "__main__":
    main()
