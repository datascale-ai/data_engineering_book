from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from mock_generators import generate_mock_trace
from pipeline_utils import (
    LOGS_DIR,
    PROCESSED_DIR,
    SAMPLED_DIR,
    RuntimeConfig,
    build_answer_prompt,
    default_runtime_config,
    ensure_standard_dirs,
    estimated_tokens,
    load_jsonl,
    make_arg_parser,
    post_json,
    safe_slug,
    setup_logging,
    utc_ts,
    write_json,
    write_jsonl,
)

COLD_START_FILE = PROCESSED_DIR / "cold_start_5k.jsonl"
TRACE_SUMMARY_FILE = PROCESSED_DIR / "sample_trace_summary.json"


def _build_vllm_sampling_params(temperature: float, top_p: float, max_tokens: int):
    from vllm import SamplingParams  # pragma: no cover

    return SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["<|im_end|>"],
    )


def _init_vllm(runtime: RuntimeConfig):
    from vllm import LLM  # pragma: no cover

    return LLM(
        model=runtime.model_path,
        trust_remote_code=True,
        tensor_parallel_size=runtime.tensor_parallel_size,
        gpu_memory_utilization=runtime.gpu_memory_utilization,
        max_model_len=runtime.max_model_len,
        max_num_seqs=runtime.max_num_seqs,
    )


def _sample_via_openai_api(
    example: dict[str, Any],
    num_samples: int,
    runtime: RuntimeConfig,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    model_name = runtime.served_model_name or runtime.model_path
    prompt = build_answer_prompt(example["prompt"], example["domain"]) + "\n\nQuestion:\n" + example["prompt"]
    headers = {"Authorization": f"Bearer {runtime.api_key}"}
    rows: list[dict[str, Any]] = []
    response = post_json(
        runtime.api_base.rstrip("/") + "/chat/completions",
        payload={
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a careful reasoning assistant."},
                {"role": "user", "content": prompt},
            ],
            "n": num_samples,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
        headers=headers,
    )
    for choice in response["choices"]:
        text = choice["message"]["content"]
        rows.append(
            {
                "trace_text": text,
                "parsed_answer": text,
                "finish_reason": choice.get("finish_reason", "stop"),
                "token_count": estimated_tokens(text),
            }
        )
    return rows


def generate_samples_for_example(
    example: dict[str, Any],
    num_samples: int,
    runtime: RuntimeConfig,
    temperature: float,
    top_p: float,
    max_tokens: int,
    logger,
    llm=None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    backend = runtime.resolved_backend()
    if backend == "vllm" and llm is not None:
        prompt = build_answer_prompt(example["prompt"], example["domain"])
        sampling_params = _build_vllm_sampling_params(temperature, top_p, max_tokens)
        outputs = llm.generate([prompt] * num_samples, sampling_params)  # pragma: no cover
        generations = [item.outputs[0].text for item in outputs]
        parsed = [{"trace_text": text, "parsed_answer": text, "finish_reason": "stop", "token_count": estimated_tokens(text)} for text in generations]
    elif backend == "openai":
        parsed = _sample_via_openai_api(
            example=example,
            num_samples=num_samples,
            runtime=runtime,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    else:
        parsed = [generate_mock_trace(example, sample_idx=idx, seed=runtime.seed) for idx in range(num_samples)]

    for idx, item in enumerate(parsed):
        rows.append(
            {
                "created_at": utc_ts(),
                "prompt_id": example["record_id"],
                "source_dataset": example["source_dataset"],
                "domain": example["domain"],
                "sample_idx": idx,
                "generation_params": {
                    "backend": backend,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "seed": runtime.seed,
                },
                "prompt": example["prompt"],
                "reference_answer": example["reference_answer"],
                "raw_trace": item["trace_text"],
                "parsed_answer": item["parsed_answer"],
                "finish_reason": item["finish_reason"],
                "token_count": item["token_count"],
            }
        )
    logger.info("sampled %s traces for %s", len(rows), example["record_id"])
    return rows


def run_sampling(
    input_path: Path,
    output_dir: Path,
    num_examples: int,
    num_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    offset: int = 0,
    parallel_prompts: int = 1,
    skip_existing: bool = False,
    runtime: RuntimeConfig | None = None,
) -> dict[str, Any]:
    ensure_standard_dirs()
    runtime = runtime or default_runtime_config()
    logger = setup_logging("sample_traces", LOGS_DIR / "sample_traces.log")
    rows = load_jsonl(input_path)[offset : offset + num_examples]

    llm = None
    if runtime.resolved_backend() == "vllm":
        try:  # pragma: no cover
            llm = _init_vllm(runtime)
        except Exception as exc:
            logger.warning("vLLM init failed, falling back to mock backend: %s", exc)
            runtime.force_mock = True

    output_dir.mkdir(parents=True, exist_ok=True)
    def process_one(example: dict[str, Any]) -> tuple[str, int]:
        samples = generate_samples_for_example(
            example=example,
            num_samples=num_samples,
            runtime=runtime,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logger=logger,
            llm=llm,
        )
        shard = output_dir / f"{safe_slug(example['record_id'])}.jsonl"
        if skip_existing and shard.exists():
            logger.info("skipping existing shard for %s", example["record_id"])
            return example["record_id"], 0
        write_jsonl(samples, shard)
        return example["record_id"], len(samples)

    total = 0
    if runtime.resolved_backend() == "openai" and parallel_prompts > 1:
        with ThreadPoolExecutor(max_workers=parallel_prompts) as executor:
            futures = {executor.submit(process_one, example): example["record_id"] for example in rows}
            for future in as_completed(futures):
                _, sample_count = future.result()
                total += sample_count
    else:
        for example in rows:
            _, sample_count = process_one(example)
            total += sample_count

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "num_examples": len(rows),
        "offset": offset,
        "num_samples_per_prompt": num_samples,
        "total_samples": total,
        "backend": runtime.resolved_backend(),
        "model_path": runtime.model_path,
        "parallel_prompts": parallel_prompts,
        "skip_existing": skip_existing,
    }
    write_json(summary, TRACE_SUMMARY_FILE)
    return summary


def main() -> None:
    parser = make_arg_parser("Sample reasoning traces with vLLM or mock backend.")
    parser.add_argument("--input", type=str, default=str(COLD_START_FILE))
    parser.add_argument("--output-dir", type=str, default=str(SAMPLED_DIR))
    parser.add_argument("--num-examples", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--parallel-prompts", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--backend", type=str, default="auto")
    parser.add_argument("--force-mock", action="store_true")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--api-base", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--served-model-name", type=str, default="")
    args = parser.parse_args()

    runtime = default_runtime_config()
    runtime.backend = args.backend
    runtime.force_mock = args.force_mock
    if args.model_path:
        runtime.model_path = args.model_path
    if args.api_base:
        runtime.api_base = args.api_base
    if args.api_key:
        runtime.api_key = args.api_key
    if args.served_model_name:
        runtime.served_model_name = args.served_model_name

    summary = run_sampling(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        num_examples=args.num_examples,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        offset=args.offset,
        parallel_prompts=args.parallel_prompts,
        skip_existing=args.skip_existing,
        runtime=runtime,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
