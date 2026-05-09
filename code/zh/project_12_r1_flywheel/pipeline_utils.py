from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"
REPORTS_DIR = DATA_DIR / "reports"
SAMPLED_DIR = DATA_DIR / "sampled_traces"
VERIFIED_DIR = DATA_DIR / "verified_candidates"
LOGS_DIR = ROOT_DIR / "logs"
TESTS_DIR = ROOT_DIR / "tests"

PROJECT4_DIR = ROOT_DIR.parent.parent / "project_4_synth" / "data"
LOCAL_GSM8K_FILE = PROJECT4_DIR / "gsm8k_train.jsonl"
LOCAL_MBPP_FILE = PROJECT4_DIR / "mbpp_train.jsonl"
LOCAL_OPEN_THOUGHTS_DIR = DATA_DIR / "OpenThoughts"
LOCAL_OPEN_THOUGHTS_DATA_GLOB = str(LOCAL_OPEN_THOUGHTS_DIR / "data" / "*.parquet")
LOCAL_MATH500_FILE = DATA_DIR / "MATH-500" / "test.jsonl"
LOCAL_HUMANEVAL_GLOB = str(DATA_DIR / "HumanEval" / "openai_humaneval" / "*.parquet")

DEFAULT_MODEL_PATH = Path("/data/xuxin/Qwen/Qwen2.5-7B-Instruct")
DEFAULT_OPEN_THOUGHTS = "open-thoughts/OpenThoughts-114k"
DEFAULT_MATH = "hendrycks/competition_math"
DEFAULT_GSM8K = "gsm8k"
DEFAULT_HUMANEVAL = "openai_humaneval"

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


@dataclasses.dataclass
class RuntimeConfig:
    model_path: str
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.82
    tensor_parallel_size: int = 1
    max_num_seqs: int = 4
    backend: str = "auto"
    seed: int = 42
    force_mock: bool = False
    api_base: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    served_model_name: str = ""

    def resolved_backend(self) -> str:
        if self.force_mock:
            return "mock"
        if self.backend != "auto":
            return self.backend
        if os.environ.get("R1_VLLM_API_BASE"):
            return "openai"
        return "vllm" if import_available("vllm") else "mock"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_standard_dirs() -> None:
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        TRAINING_DIR,
        REPORTS_DIR,
        SAMPLED_DIR,
        VERIFIED_DIR,
        LOGS_DIR,
        TESTS_DIR,
    ]:
        ensure_dir(path)


def setup_logging(name: str, log_file: Path | None = None) -> logging.Logger:
    ensure_standard_dirs()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        ensure_dir(log_file.parent)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def write_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def safe_slug(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:max_len] or "item"


def deterministic_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def deterministic_bucket(text: str, buckets: int = 100) -> int:
    return int(deterministic_hash(text)[:8], 16) % buckets


def estimated_tokens(text: str) -> int:
    return max(1, math.ceil(len(normalize_text(text)) / 4))


def sample_records(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(records):
        return list(records)
    rng = random.Random(seed)
    items = list(records)
    rng.shuffle(items)
    return items[:limit]


def first_number(text: str) -> str | None:
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", text or "")
    return match.group(0) if match else None


def parse_numeric_value(text: str) -> float | None:
    number = first_number(text)
    if number is None:
        return None
    try:
        return float(number.replace(",", ""))
    except ValueError:
        return None


def extract_boxed_answer(text: str) -> str | None:
    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"####\s*(.+)$",
        r"final answer\s*[:：]\s*(.+)$",
        r"answer\s*[:：]\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return normalize_text(match.group(1))
    return None


def split_reasoning_steps(text: str) -> list[str]:
    parts = re.split(r"(?:\n+|(?<=\.)\s+(?=[A-Z0-9]))", text)
    steps = [normalize_text(part) for part in parts if normalize_text(part)]
    return steps


def import_available(module_name: str) -> bool:
    with contextlib.suppress(Exception):
        __import__(module_name)
        return True
    return False


def try_import(module_name: str):
    __import__(module_name)
    return sys.modules[module_name]


def choose_device_map() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return "auto"
    gpu_count = len([token for token in visible.split(",") if token.strip()])
    return "auto" if gpu_count != 1 else "cuda:0"


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


def default_runtime_config() -> RuntimeConfig:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    tensor_parallel_size = len([token for token in visible.split(",") if token.strip()]) if visible else 1
    tensor_parallel_size = max(1, tensor_parallel_size)
    return RuntimeConfig(
        model_path=os.environ.get("QWEN_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
        max_model_len=env_int("R1_MAX_MODEL_LEN", 4096),
        gpu_memory_utilization=env_float("R1_GPU_MEMORY_UTILIZATION", 0.82),
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=env_int("R1_MAX_NUM_SEQS", 4),
        backend=os.environ.get("R1_INFER_BACKEND", "auto"),
        seed=env_int("R1_SEED", 42),
        force_mock=os.environ.get("R1_FORCE_MOCK", "0") == "1",
        api_base=os.environ.get("R1_VLLM_API_BASE", "http://127.0.0.1:8000/v1"),
        api_key=os.environ.get("R1_VLLM_API_KEY", "EMPTY"),
        served_model_name=os.environ.get("R1_SERVED_MODEL_NAME", ""),
    )


def read_local_jsonl_candidates(paths: list[Path], limit: int | None = None) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            merged.extend(load_jsonl(path))
    return merged[:limit] if limit is not None else merged


def maybe_load_dataset(dataset_name: str, split: str, subset: str | None = None, streaming: bool = False):
    datasets = try_import("datasets")
    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if subset:
        return datasets.load_dataset(dataset_name, subset, **kwargs)
    return datasets.load_dataset(dataset_name, **kwargs)


def maybe_load_local_parquet_dataset(data_files: str, split: str = "train"):
    datasets = try_import("datasets")
    return datasets.load_dataset("parquet", data_files=data_files, split=split)


def render_humaneval_solution(prompt: str, canonical_solution: str) -> str:
    prompt = (prompt or "").rstrip()
    canonical_solution = (canonical_solution or "").rstrip()
    if not prompt or not canonical_solution:
        return ""

    prompt_lines = prompt.splitlines()
    signature_lines: list[str] = []
    collecting = False
    for line in prompt_lines:
        if line.lstrip().startswith("def "):
            collecting = True
        if collecting:
            signature_lines.append(line.rstrip())
            if line.rstrip().endswith(":"):
                break
    if not signature_lines:
        return canonical_solution

    signature = "\n".join(signature_lines).rstrip()
    body = canonical_solution
    if body.lstrip().startswith("def "):
        return body
    if body and not body.startswith((" ", "\t")):
        body = "\n".join(("    " + line if line else line) for line in body.splitlines())
    return f"{signature}\n{body}".rstrip()


def render_chat_messages(messages: list[dict[str, str]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "").upper()
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def render_qwen_chat(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def run_python(code: str, timeout: int = 5) -> tuple[bool, str, str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        script = Path(tmp_dir) / "snippet.py"
        script.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(script)],
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return proc.returncode == 0, proc.stdout.strip(), proc.stderr.strip()


def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_answer_prompt(question: str, domain: str) -> str:
    if domain == "code":
        return (
            "You are solving a coding task. Think step by step, then provide Python code.\n"
            "Return sections exactly as:\n"
            "Reasoning: ...\n"
            "Code:\n```python\n...\n```\n"
        )
    return (
        "You are solving a reasoning task. Think step by step and end with a clear final answer.\n"
        "Return sections exactly as:\n"
        "Reasoning: ...\n"
        "Final Answer: ...\n"
    )


def dump_run_report(path: Path, title: str, sections: list[tuple[str, Any]]) -> None:
    ensure_dir(path.parent)
    lines = [f"# {title}", ""]
    for heading, payload in sections:
        lines.append(f"## {heading}")
        lines.append("")
        if isinstance(payload, str):
            lines.append(payload)
        else:
            lines.append("```json")
            lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
            lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: int = 300) -> dict[str, Any]:
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def make_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", type=str, default="")
    return parser
