from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
from typing import Any

from pipeline_utils import (
    DEFAULT_GSM8K,
    DEFAULT_HUMANEVAL,
    DEFAULT_MATH,
    DEFAULT_OPEN_THOUGHTS,
    LOCAL_GSM8K_FILE,
    LOCAL_HUMANEVAL_GLOB,
    LOCAL_MATH500_FILE,
    LOCAL_MBPP_FILE,
    LOCAL_OPEN_THOUGHTS_DATA_GLOB,
    LOGS_DIR,
    PROCESSED_DIR,
    deterministic_hash,
    dump_run_report,
    ensure_standard_dirs,
    extract_boxed_answer,
    import_available,
    load_jsonl,
    make_arg_parser,
    maybe_load_dataset,
    maybe_load_local_parquet_dataset,
    normalize_text,
    render_humaneval_solution,
    sample_records,
    setup_logging,
    split_reasoning_steps,
    utc_ts,
    write_json,
    write_jsonl,
)

COLD_START_FILE = PROCESSED_DIR / "cold_start_5k.jsonl"
COLD_START_SUMMARY = PROCESSED_DIR / "cold_start_summary.json"

FALLBACK_GSM8K_ROWS = [
    {
        "question": "Lina has 12 apples. She gives 5 to a friend and buys 9 more. How many apples does she have now?",
        "answer": "Lina starts with 12 apples. After giving away 5, she has 7. She buys 9 more, so 7 + 9 = 16.\n#### 16",
    },
    {
        "question": "A notebook costs 4 dollars. How much do 7 notebooks cost?",
        "answer": "Each notebook costs 4 dollars. For 7 notebooks, compute 4 * 7 = 28.\n#### 28",
    },
    {
        "question": "A train travels 60 miles per hour for 3 hours. How far does it travel?",
        "answer": "Distance equals speed times time. The train travels 60 * 3 = 180 miles.\n#### 180",
    },
    {
        "question": "Mira reads 15 pages on Monday and twice as many on Tuesday. How many pages does she read in total?",
        "answer": "Tuesday she reads 2 * 15 = 30 pages. In total she reads 15 + 30 = 45 pages.\n#### 45",
    },
    {
        "question": "There are 9 boxes with 6 pencils in each box. How many pencils are there?",
        "answer": "There are 9 groups of 6 pencils. Multiplying gives 9 * 6 = 54.\n#### 54",
    },
    {
        "question": "A jar has 30 candies. If 40 percent are red, how many red candies are there?",
        "answer": "Forty percent of 30 is 0.4 * 30 = 12.\n#### 12",
    },
    {
        "question": "Sam saves 8 dollars each week for 5 weeks, then spends 6 dollars. How much remains?",
        "answer": "Sam saves 8 * 5 = 40 dollars. After spending 6 dollars, 40 - 6 = 34 dollars remain.\n#### 34",
    },
    {
        "question": "A rectangle has length 10 and width 3. What is its area?",
        "answer": "Area is length times width. So the area is 10 * 3 = 30.\n#### 30",
    },
]

FALLBACK_MBPP_ROWS = [
    {
        "text": "Write a function add(a, b) that returns the sum of two numbers.",
        "code": "def add(a, b):\n    return a + b",
        "test_list": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
    },
    {
        "text": "Write a function square(x) that returns x squared.",
        "code": "def square(x):\n    return x * x",
        "test_list": ["assert square(4) == 16", "assert square(-3) == 9"],
    },
    {
        "text": "Write a function is_even(n) that returns True when n is even.",
        "code": "def is_even(n):\n    return n % 2 == 0",
        "test_list": ["assert is_even(4) is True", "assert is_even(5) is False"],
    },
    {
        "text": "Write a function max_of_two(a, b) that returns the larger value.",
        "code": "def max_of_two(a, b):\n    return a if a >= b else b",
        "test_list": ["assert max_of_two(3, 5) == 5", "assert max_of_two(8, 2) == 8"],
    },
]


def _as_messages(prompt: str, reasoning: str, answer: str, domain: str) -> list[dict[str, str]]:
    assistant = (
        f"Reasoning: {reasoning.strip()}\nFinal Answer: {answer.strip()}"
        if domain != "code"
        else f"Reasoning: {reasoning.strip()}\nCode:\n```python\n{answer.strip()}\n```"
    )
    return [
        {"role": "system", "content": "You are a careful reasoning assistant."},
        {"role": "user", "content": prompt.strip()},
        {"role": "assistant", "content": assistant},
    ]


def _open_thoughts_records(limit: int, seed: int, logger) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    split = None
    source_name = DEFAULT_OPEN_THOUGHTS
    if import_available("datasets"):
        try:
            if Path(LOCAL_OPEN_THOUGHTS_DATA_GLOB.replace("*.parquet", "")).exists():
                split = maybe_load_local_parquet_dataset(LOCAL_OPEN_THOUGHTS_DATA_GLOB, split="train")
                source_name = "local_openthoughts_parquet"
                logger.info("loading OpenThoughts from local parquet files")
        except Exception as exc:
            logger.warning("failed to load local OpenThoughts parquet: %s", exc)
        if split is None:
            try:
                split = maybe_load_dataset(DEFAULT_OPEN_THOUGHTS, split="train", streaming=True)
                logger.info("falling back to remote OpenThoughts streaming")
            except Exception as exc:  # pragma: no cover
                logger.warning("failed to stream OpenThoughts: %s", exc)
                return []
    else:
        logger.info("datasets unavailable, skipping OpenThoughts extraction.")
        return []

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(split):
        prompt = ""
        cot = ""
        conversations = row.get("conversations") or []
        if conversations:
            for turn in conversations:
                speaker = (turn.get("from") or turn.get("role") or "").lower()
                content = turn.get("value") or turn.get("content") or ""
                if not prompt and speaker in {"user", "human"}:
                    prompt = content
                elif not cot and speaker in {"assistant", "gpt"}:
                    cot = content
        prompt = prompt or row.get("problem") or row.get("question") or row.get("prompt") or ""
        cot = cot or row.get("reasoning") or row.get("chain_of_thought") or row.get("response") or ""
        answer = row.get("answer") or extract_boxed_answer(cot or "") or ""
        if not prompt or not cot or not answer:
            continue
        if len(split_reasoning_steps(cot)) < 3:
            continue
        record = {
            "record_id": f"ot_{idx}",
            "source_dataset": source_name,
            "domain": "math",
            "prompt": normalize_text(prompt),
            "reference_reasoning": cot.strip(),
            "reference_answer": normalize_text(str(answer)),
        }
        record["messages"] = _as_messages(
            record["prompt"],
            record["reference_reasoning"],
            record["reference_answer"],
            record["domain"],
        )
        records.append(record)
        if len(records) >= limit:
            break
    return sample_records(records, limit, seed)


def _math_records(limit: int, logger) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    if LOCAL_MATH500_FILE.exists():
        logger.info("loading MATH-500 from local jsonl")
        split = load_jsonl(LOCAL_MATH500_FILE)
        source_name = "local_math500_jsonl"
    elif import_available("datasets"):
        try:
            split = maybe_load_dataset(DEFAULT_MATH, split="train", streaming=True)
            source_name = DEFAULT_MATH
            logger.info("falling back to remote competition_math streaming")
        except Exception as exc:  # pragma: no cover
            logger.warning("failed to stream competition_math: %s", exc)
            return []
    else:
        return []

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(split):
        prompt = row.get("problem") or ""
        solution = row.get("solution") or ""
        answer = extract_boxed_answer(solution) or solution
        if not prompt or not solution or not answer:
            continue
        if len(split_reasoning_steps(solution)) < 2:
            continue
        record = {
            "record_id": f"math_{idx}",
            "source_dataset": source_name,
            "domain": "math",
            "prompt": normalize_text(prompt),
            "reference_reasoning": solution.strip(),
            "reference_answer": normalize_text(answer),
        }
        record["messages"] = _as_messages(
            record["prompt"],
            record["reference_reasoning"],
            record["reference_answer"],
            record["domain"],
        )
        records.append(record)
        if len(records) >= limit:
            break
    return records


def _local_gsm8k_records(limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    rows = load_jsonl(LOCAL_GSM8K_FILE) if LOCAL_GSM8K_FILE.exists() else FALLBACK_GSM8K_ROWS
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        answer = extract_boxed_answer(row.get("answer", "")) or row.get("answer", "")
        reasoning = row.get("answer", "").replace("####", "\n####")
        if not answer or len(split_reasoning_steps(reasoning)) < 2:
            continue
        record = {
            "record_id": f"gsm8k_{idx}",
            "source_dataset": DEFAULT_GSM8K,
            "domain": "math",
            "prompt": normalize_text(row["question"]),
            "reference_reasoning": reasoning.strip(),
            "reference_answer": normalize_text(answer),
        }
        record["messages"] = _as_messages(
            record["prompt"],
            record["reference_reasoning"],
            record["reference_answer"],
            record["domain"],
        )
        records.append(record)
        if len(records) >= limit:
            break
    return records


def _local_mbpp_records(limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    rows = load_jsonl(LOCAL_MBPP_FILE) if LOCAL_MBPP_FILE.exists() else FALLBACK_MBPP_ROWS
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prompt = normalize_text(row.get("text", ""))
        code = row.get("code", "").strip()
        tests = row.get("test_list", [])
        if not prompt or not code or len(tests) < 2:
            continue
        reasoning = "Identify the required function signature, implement the behavior, and satisfy the unit tests."
        record = {
            "record_id": f"mbpp_{idx}",
            "source_dataset": "mbpp_local_as_humaneval_proxy",
            "domain": "code",
            "prompt": prompt,
            "reference_reasoning": reasoning,
            "reference_answer": code,
            "tests": tests,
        }
        record["messages"] = _as_messages(
            record["prompt"],
            record["reference_reasoning"],
            record["reference_answer"],
            record["domain"],
        )
        records.append(record)
        if len(records) >= limit:
            break
    return records


def _humaneval_records(limit: int, logger) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    split = None
    source_name = DEFAULT_HUMANEVAL
    if import_available("datasets"):
        try:
            if Path(LOCAL_HUMANEVAL_GLOB.replace("*.parquet", "")).exists():
                split = maybe_load_local_parquet_dataset(LOCAL_HUMANEVAL_GLOB, split="train")
                source_name = "local_humaneval_parquet"
                logger.info("loading HumanEval from local parquet files")
        except Exception as exc:
            logger.warning("failed to load local HumanEval parquet: %s", exc)
        if split is None:
            try:
                split = maybe_load_dataset(DEFAULT_HUMANEVAL, split="test", streaming=True)
                logger.info("falling back to remote HumanEval streaming")
            except Exception as exc:  # pragma: no cover
                logger.warning("failed to stream HumanEval: %s", exc)
                return []
    else:
        return []

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(split):
        prompt = row.get("prompt") or ""
        canonical_solution = row.get("canonical_solution") or ""
        test_code = row.get("test") or ""
        entry_point = row.get("entry_point") or ""
        if not prompt or not canonical_solution or not test_code:
            continue
        full_solution = render_humaneval_solution(prompt, canonical_solution)
        if not full_solution.lstrip().startswith("def "):
            logger.warning("skipping HumanEval sample %s because function signature reconstruction failed", idx)
            continue
        reasoning = "Read the function specification, implement the function, and satisfy the provided tests."
        tests = [test_code]
        if entry_point:
            tests.append(f"check({entry_point})")
        record = {
            "record_id": f"humaneval_{idx}",
            "source_dataset": source_name,
            "domain": "code",
            "prompt": prompt.rstrip(),
            "reference_reasoning": reasoning,
            "reference_answer": full_solution,
            "tests": tests,
            "test_setup_code": "",
        }
        record["messages"] = _as_messages(
            record["prompt"],
            record["reference_reasoning"],
            record["reference_answer"],
            record["domain"],
        )
        records.append(record)
        if len(records) >= limit:
            break
    return records


def build_cold_start_data(
    max_openthoughts: int,
    max_math: int,
    max_gsm8k: int,
    max_code: int,
    seed: int,
    output_path: Path = COLD_START_FILE,
) -> dict[str, Any]:
    ensure_standard_dirs()
    logger = setup_logging("cold_start_data", LOGS_DIR / "cold_start_data.log")
    logger.info("building cold-start data")

    records: list[dict[str, Any]] = []
    records.extend(_open_thoughts_records(max_openthoughts, seed, logger))
    records.extend(_math_records(max_math, logger))
    records.extend(_local_gsm8k_records(max_gsm8k))
    humaneval_records = _humaneval_records(max_code, logger)
    if humaneval_records:
        records.extend(humaneval_records)
    else:
        records.extend(_local_mbpp_records(max_code))

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        key = deterministic_hash(record["prompt"] + "\n" + record["reference_answer"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    random.Random(seed).shuffle(deduped)

    write_jsonl(deduped, output_path)
    summary = {
        "created_at": utc_ts(),
        "output_path": str(output_path),
        "record_count": len(deduped),
        "domain_distribution": dict(Counter(row["domain"] for row in deduped)),
        "source_distribution": dict(Counter(row["source_dataset"] for row in deduped)),
        "local_openthoughts_available": Path(LOCAL_OPEN_THOUGHTS_DATA_GLOB.replace("*.parquet", "")).exists(),
        "local_math500_available": LOCAL_MATH500_FILE.exists(),
        "local_humaneval_available": Path(LOCAL_HUMANEVAL_GLOB.replace("*.parquet", "")).exists(),
        "humaneval_strategy": "Prefer local HumanEval parquet, otherwise datasets fallback, otherwise local MBPP fallback.",
    }
    write_json(summary, COLD_START_SUMMARY)
    dump_run_report(
        PROCESSED_DIR / "cold_start_report.md",
        "Cold Start Extraction Report",
        [
            ("Summary", summary),
            ("Notes", "This project now prefers local OpenThoughts, MATH-500, and HumanEval files under data/ before any remote fallback."),
        ],
    )
    logger.info("cold-start records: %s", len(deduped))
    return summary


def main() -> None:
    parser = make_arg_parser("Extract cold-start SFT data.")
    parser.add_argument("--max-openthoughts", type=int, default=0)
    parser.add_argument("--max-math", type=int, default=0)
    parser.add_argument("--max-gsm8k", type=int, default=128)
    parser.add_argument("--max-code", type=int, default=32)
    parser.add_argument("--output", type=str, default=str(COLD_START_FILE))
    args = parser.parse_args()
    summary = build_cold_start_data(
        max_openthoughts=args.max_openthoughts,
        max_math=args.max_math,
        max_gsm8k=args.max_gsm8k,
        max_code=args.max_code,
        seed=args.seed,
        output_path=Path(args.output),
    )
    print(summary)


if __name__ == "__main__":
    main()
