from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from pipeline_utils import (
    LOCAL_GSM8K_FILE,
    LOCAL_MATH500_FILE,
    LOGS_DIR,
    REPORTS_DIR,
    RuntimeConfig,
    build_answer_prompt,
    default_runtime_config,
    ensure_standard_dirs,
    extract_boxed_answer,
    import_available,
    load_jsonl,
    make_arg_parser,
    parse_numeric_value,
    render_qwen_chat,
    setup_logging,
    utc_ts,
    write_json,
)
from sample_traces import generate_samples_for_example

EVAL_RESULT_FILE = REPORTS_DIR / "eval_results_gsm8k_math.json"


def _simple_accuracy(preds: list[str], golds: list[str]) -> float:
    correct = _count_correct(preds, golds)
    return round(correct / len(golds), 4) if golds else 0.0


def _count_correct(preds: list[str], golds: list[str]) -> int:
    correct = 0
    for pred, gold in zip(preds, golds):
        pred_num = parse_numeric_value(pred)
        gold_num = parse_numeric_value(gold)
        if pred_num is not None and gold_num is not None and abs(pred_num - gold_num) < 1e-6:
            correct += 1
        elif pred.strip().lower() == gold.strip().lower():
            correct += 1
    return correct


def _load_gsm8k_examples(max_examples: int) -> list[dict[str, Any]]:
    rows = load_jsonl(LOCAL_GSM8K_FILE)[:max_examples]
    return [
        {
            "record_id": f"eval_gsm8k_{idx}",
            "source_dataset": "gsm8k",
            "domain": "math",
            "prompt": row["question"],
            "reference_answer": extract_boxed_answer(row["answer"]) or row["answer"],
        }
        for idx, row in enumerate(rows)
    ]


def _load_math_examples(max_examples: int) -> list[dict[str, Any]]:
    rows = load_jsonl(LOCAL_MATH500_FILE)[:max_examples]
    return [
        {
            "record_id": f"eval_math_{idx}",
            "source_dataset": "math500",
            "domain": "math",
            "prompt": row["problem"],
            "reference_answer": row["answer"],
        }
        for idx, row in enumerate(rows)
    ]


def _predict_answers(
    examples: list[dict[str, Any]],
    runtime: RuntimeConfig,
    logger,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> list[str]:
    preds: list[str] = []
    for example in examples:
        sampled = generate_samples_for_example(
            example=example,
            num_samples=1,
            runtime=runtime,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logger=logger,
        )[0]["parsed_answer"]
        preds.append(extract_boxed_answer(sampled) or sampled)
    return preds


def _load_local_model_and_tokenizer(model_path: str, adapter_path: str = ""):
    import torch  # pragma: no cover
    from transformers import AutoModelForCausalLM, AutoTokenizer  # pragma: no cover

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if adapter_path:
        from peft import PeftModel  # pragma: no cover

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _predict_answers_local_with_model(
    examples: list[dict[str, Any]],
    model,
    tokenizer,
    logger,
    max_tokens: int = 256,
) -> list[str]:
    import torch  # pragma: no cover

    preds: list[str] = []
    for example in examples:
        user_prompt = build_answer_prompt(example["prompt"], example["domain"]) + "\n\nQuestion:\n" + example["prompt"]
        prompt = render_qwen_chat(
            tokenizer,
            [
                {"role": "system", "content": "You are a careful reasoning assistant."},
                {"role": "user", "content": user_prompt},
            ],
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        preds.append(extract_boxed_answer(text) or text)
        logger.info("evaluated %s via local model", example["record_id"])
    return preds


def _release_local_model(model) -> None:
    import torch  # pragma: no cover

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _select_shard_rows(rows: list[dict[str, Any]], shard_id: int, num_shards: int) -> list[dict[str, Any]]:
    if num_shards <= 1:
        return rows
    return [row for idx, row in enumerate(rows) if idx % num_shards == shard_id]


def _aggregate_shard_results(shard_paths: list[Path], output_path: Path, model_path: str, adapter_path: str) -> dict[str, Any]:
    shard_results = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    gsm8k_total = sum(item["counts"]["gsm8k_total"] for item in shard_results)
    gsm8k_base_correct = sum(item["counts"]["gsm8k_base_correct"] for item in shard_results)
    gsm8k_lora_correct = sum(item["counts"]["gsm8k_lora_correct"] for item in shard_results)
    math_total = sum(item["counts"]["math_total"] for item in shard_results)
    math_base_correct = sum(item["counts"]["math_base_correct"] for item in shard_results)
    math_lora_correct = sum(item["counts"]["math_lora_correct"] for item in shard_results)
    result = {
        "created_at": utc_ts(),
        "model_path": model_path,
        "adapter_path": adapter_path,
        "backend": "local-multigpu",
        "num_shards": len(shard_results),
        "shard_paths": [str(path) for path in shard_paths],
        "counts": {
            "gsm8k_total": gsm8k_total,
            "gsm8k_base_correct": gsm8k_base_correct,
            "gsm8k_lora_correct": gsm8k_lora_correct,
            "math_total": math_total,
            "math_base_correct": math_base_correct,
            "math_lora_correct": math_lora_correct,
        },
        "metrics": {
            "gsm8k_base_accuracy": round(gsm8k_base_correct / gsm8k_total, 4) if gsm8k_total else 0.0,
            "gsm8k_lora_accuracy": round(gsm8k_lora_correct / gsm8k_total, 4) if gsm8k_total else 0.0,
            "gsm8k_delta": round((gsm8k_lora_correct / gsm8k_total) - (gsm8k_base_correct / gsm8k_total), 4) if gsm8k_total else 0.0,
            "math_base_accuracy": round(math_base_correct / math_total, 4) if math_total else 0.0,
            "math_lora_accuracy": round(math_lora_correct / math_total, 4) if math_total else 0.0,
            "math_delta": round((math_lora_correct / math_total) - (math_base_correct / math_total), 4) if math_total else 0.0,
        },
    }
    write_json(result, output_path)
    return result


def run_eval_multigpu(
    model_path: str,
    adapter_path: str,
    max_examples: int,
    devices: list[str],
    tasks: list[str],
    backend: str,
    force_mock: bool,
    output_path: Path,
) -> dict[str, Any]:
    ensure_standard_dirs()
    shard_dir = REPORTS_DIR / f"eval_shards_{utc_ts().replace(':', '').replace('-', '')}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [shard_dir / f"shard_{idx}.json" for idx in range(len(devices))]
    procs: list[tuple[int, str, subprocess.Popen[str], Path]] = []
    child_backend = "local" if backend == "auto" else backend

    for shard_id, (device, shard_path) in enumerate(zip(devices, shard_paths)):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model-path",
            model_path,
            "--adapter-path",
            adapter_path,
            "--max-examples",
            str(max_examples),
            "--backend",
            child_backend,
            "--tasks",
            ",".join(tasks),
            "--shard-id",
            str(shard_id),
            "--num-shards",
            str(len(devices)),
            "--output-file",
            str(shard_path),
        ]
        if force_mock:
            cmd.append("--force-mock")
        proc = subprocess.Popen(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append((shard_id, device, proc, shard_path))

    failures: list[str] = []
    for shard_id, device, proc, shard_path in procs:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            failures.append(
                f"shard={shard_id} device={device} returncode={proc.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        elif not shard_path.exists():
            failures.append(f"shard={shard_id} device={device} completed without output file {shard_path}")
    if failures:
        raise RuntimeError("Multi-GPU evaluation failed:\n\n" + "\n\n".join(failures))

    return _aggregate_shard_results(shard_paths, output_path, model_path, adapter_path)


def run_eval(
    model_path: str,
    adapter_path: str,
    max_examples: int,
    tasks: list[str] | None = None,
    backend: str = "openai",
    force_mock: bool = False,
    shard_id: int = 0,
    num_shards: int = 1,
    output_path: Path = EVAL_RESULT_FILE,
) -> dict[str, Any]:
    ensure_standard_dirs()
    logger = setup_logging("eval_gsm8k_math", LOGS_DIR / "eval_gsm8k_math.log")

    base_runtime = default_runtime_config()
    base_runtime.model_path = model_path
    base_runtime.backend = backend
    base_runtime.force_mock = force_mock

    tuned_runtime = default_runtime_config()
    tuned_runtime.model_path = adapter_path if adapter_path and Path(adapter_path).exists() else model_path
    tuned_runtime.backend = backend
    tuned_runtime.force_mock = force_mock

    tasks = tasks or ["gsm8k", "math"]
    gsm8k_examples = _select_shard_rows(_load_gsm8k_examples(max_examples), shard_id, num_shards) if "gsm8k" in tasks else []
    math_examples = _select_shard_rows(_load_math_examples(max_examples), shard_id, num_shards) if "math" in tasks else []

    gsm8k_golds = [row["reference_answer"] for row in gsm8k_examples]
    math_golds = [row["reference_answer"] for row in math_examples]

    use_local_eval = backend == "local" or (backend == "auto" and import_available("torch") and adapter_path)
    if use_local_eval:
        base_model, tokenizer = _load_local_model_and_tokenizer(model_path, "")
        gsm8k_base_preds = _predict_answers_local_with_model(gsm8k_examples, base_model, tokenizer, logger)
        math_base_preds = _predict_answers_local_with_model(math_examples, base_model, tokenizer, logger)
        _release_local_model(base_model)

        tuned_model, tokenizer = _load_local_model_and_tokenizer(model_path, adapter_path)
        gsm8k_tuned_preds = _predict_answers_local_with_model(gsm8k_examples, tuned_model, tokenizer, logger)
        math_tuned_preds = _predict_answers_local_with_model(math_examples, tuned_model, tokenizer, logger)
        _release_local_model(tuned_model)
    else:
        gsm8k_base_preds = _predict_answers(gsm8k_examples, base_runtime, logger)
        gsm8k_tuned_preds = _predict_answers(gsm8k_examples, tuned_runtime, logger)
        math_base_preds = _predict_answers(math_examples, base_runtime, logger)
        math_tuned_preds = _predict_answers(math_examples, tuned_runtime, logger)

    gsm8k_base_correct = _count_correct(gsm8k_base_preds, gsm8k_golds)
    gsm8k_lora_correct = _count_correct(gsm8k_tuned_preds, gsm8k_golds)
    math_base_correct = _count_correct(math_base_preds, math_golds)
    math_lora_correct = _count_correct(math_tuned_preds, math_golds)
    result = {
        "created_at": utc_ts(),
        "model_path": model_path,
        "adapter_path": adapter_path,
        "backend": "local" if use_local_eval else base_runtime.resolved_backend(),
        "max_examples": max_examples,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "counts": {
            "gsm8k_total": len(gsm8k_golds),
            "gsm8k_base_correct": gsm8k_base_correct,
            "gsm8k_lora_correct": gsm8k_lora_correct,
            "math_total": len(math_golds),
            "math_base_correct": math_base_correct,
            "math_lora_correct": math_lora_correct,
        },
        "metrics": {
            "gsm8k_base_accuracy": round(gsm8k_base_correct / len(gsm8k_golds), 4) if gsm8k_golds else 0.0,
            "gsm8k_lora_accuracy": round(gsm8k_lora_correct / len(gsm8k_golds), 4) if gsm8k_golds else 0.0,
            "gsm8k_delta": round((gsm8k_lora_correct / len(gsm8k_golds)) - (gsm8k_base_correct / len(gsm8k_golds)), 4) if gsm8k_golds else 0.0,
            "math_base_accuracy": round(math_base_correct / len(math_golds), 4) if math_golds else 0.0,
            "math_lora_accuracy": round(math_lora_correct / len(math_golds), 4) if math_golds else 0.0,
            "math_delta": round((math_lora_correct / len(math_golds)) - (math_base_correct / len(math_golds)), 4) if math_golds else 0.0,
        },
    }
    write_json(result, output_path)
    return result


def main() -> None:
    parser = make_arg_parser("Evaluate GSM8K/MATH prompts with base vs LoRA model.")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--adapter-path", type=str, default="")
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--tasks", type=str, default="gsm8k,math")
    parser.add_argument("--backend", type=str, default="openai")
    parser.add_argument("--output-file", type=str, default=str(EVAL_RESULT_FILE))
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--multi-gpu-devices", type=str, default="")
    parser.add_argument("--force-mock", action="store_true")
    args = parser.parse_args()
    output_path = Path(args.output_file)
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    devices = [item.strip() for item in args.multi_gpu_devices.split(",") if item.strip()]
    if devices:
        result = run_eval_multigpu(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            max_examples=args.max_examples,
            devices=devices,
            tasks=tasks,
            backend=args.backend,
            force_mock=args.force_mock,
            output_path=output_path,
        )
    else:
        result = run_eval(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            max_examples=args.max_examples,
            tasks=tasks,
            backend=args.backend,
            force_mock=args.force_mock,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            output_path=output_path,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
