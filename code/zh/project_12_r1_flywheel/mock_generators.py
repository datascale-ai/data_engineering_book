from __future__ import annotations

import random
from typing import Any

from pipeline_utils import extract_boxed_answer, normalize_text, parse_numeric_value


def _math_trace(question: str, answer: str, rng: random.Random, variant: int) -> tuple[str, str]:
    numeric = parse_numeric_value(answer)
    if numeric is None:
        final = answer
    elif variant % 4 == 0:
        final = str(int(numeric) if numeric.is_integer() else numeric)
    elif variant % 4 == 1:
        final = str(int(numeric + 1) if float(numeric).is_integer() else round(numeric + 1, 2))
    elif variant % 4 == 2:
        final = str(int(numeric) if float(numeric).is_integer() else numeric)
    else:
        final = str(int(numeric - 1) if float(numeric).is_integer() else round(numeric - 1, 2))

    reasoning = [
        f"Read the problem carefully: {normalize_text(question)[:180]}",
        "Identify the quantities and the arithmetic relationship.",
        "Carry out the calculation step by step and sanity check the magnitude.",
    ]
    if variant % 4 == 1:
        reasoning.append("A noisy arithmetic slip is introduced in this sample.")
    elif variant % 4 == 3:
        reasoning.append("This sample explores an alternative path before concluding.")
    trace = "Reasoning: " + " ".join(reasoning) + f"\nFinal Answer: {final}"
    return trace, final


def _code_trace(question: str, canonical_code: str, rng: random.Random, variant: int) -> tuple[str, str]:
    code = canonical_code
    if variant % 4 == 1:
        code = canonical_code.replace("return", "return None  # incorrect", 1)
    elif variant % 4 == 3:
        code = canonical_code + "\n# Alternative explanation path\n"

    reasoning = [
        f"Understand the specification: {normalize_text(question)[:160]}",
        "Implement the required function directly.",
        "Ensure the return value matches the expected tests.",
    ]
    trace = "Reasoning: " + " ".join(reasoning) + f"\nCode:\n```python\n{code.rstrip()}\n```"
    return trace, code


def generate_mock_trace(example: dict[str, Any], sample_idx: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed + sample_idx)
    domain = example["domain"]
    if domain == "code":
        trace, parsed = _code_trace(example["prompt"], example["reference_answer"], rng, sample_idx)
    else:
        answer = extract_boxed_answer(example["reference_answer"]) or example["reference_answer"]
        trace, parsed = _math_trace(example["prompt"], answer, rng, sample_idx)
    return {
        "trace_text": trace,
        "parsed_answer": parsed,
        "finish_reason": "stop",
        "token_count": max(32, len(trace) // 4),
    }
