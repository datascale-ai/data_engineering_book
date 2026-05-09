from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from pipeline_utils import (
    LOGS_DIR,
    extract_boxed_answer,
    parse_numeric_value,
    run_python,
    setup_logging,
)

LOGGER = setup_logging("verifier_pool", LOGS_DIR / "verifier_pool.log")


@dataclass
class VerificationResult:
    verifier_type: str
    verifier_pass: bool
    reward_score: float
    parsed_answer: str | None
    reason: str
    details: dict[str, Any]


def _extract_code_block(text: str) -> str | None:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if "def " in text:
        return text.strip()
    return None


def math_verifier(prediction: str, expected: str) -> VerificationResult:
    parsed_prediction = extract_boxed_answer(prediction) or prediction.strip()
    pred_value = parse_numeric_value(parsed_prediction)
    gold_value = parse_numeric_value(expected)
    passed = False
    reward = 0.0
    reason = "non_numeric"
    if pred_value is not None and gold_value is not None:
        passed = abs(pred_value - gold_value) < 1e-6
        reward = 1.0 if passed else max(0.0, 1.0 - min(1.0, abs(pred_value - gold_value) / max(1.0, abs(gold_value))))
        reason = "exact_numeric_match" if passed else "numeric_mismatch"
    else:
        norm_pred = parsed_prediction.strip().lower()
        norm_gold = expected.strip().lower()
        passed = norm_pred == norm_gold
        reward = 1.0 if passed else 0.0
        reason = "string_match" if passed else "string_mismatch"
    return VerificationResult(
        verifier_type="math",
        verifier_pass=passed,
        reward_score=round(reward, 4),
        parsed_answer=parsed_prediction,
        reason=reason,
        details={"expected": expected},
    )


def format_verifier(prediction: str, domain: str) -> VerificationResult:
    if domain == "code":
        has_reasoning = "Reasoning:" in prediction
        has_code = _extract_code_block(prediction) is not None
        passed = has_reasoning and has_code
        score = 1.0 if passed else 0.0
        reason = "valid_code_sections" if passed else "missing_reasoning_or_code"
        parsed = _extract_code_block(prediction)
    else:
        has_reasoning = "Reasoning:" in prediction
        parsed = extract_boxed_answer(prediction)
        passed = has_reasoning and parsed is not None
        score = 1.0 if passed else 0.0
        reason = "valid_reasoning_answer_sections" if passed else "missing_reasoning_or_answer"
    return VerificationResult(
        verifier_type="format",
        verifier_pass=passed,
        reward_score=score,
        parsed_answer=parsed,
        reason=reason,
        details={"domain": domain},
    )


def code_verifier(prediction: str, tests: list[str], test_setup_code: str = "") -> VerificationResult:
    code = _extract_code_block(prediction)
    if not code:
        return VerificationResult(
            verifier_type="code",
            verifier_pass=False,
            reward_score=0.0,
            parsed_answer=None,
            reason="missing_code_block",
            details={},
        )
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return VerificationResult(
            verifier_type="code",
            verifier_pass=False,
            reward_score=0.0,
            parsed_answer=code,
            reason=f"syntax_error:{exc.msg}",
            details={},
        )
    body = [test_setup_code.strip(), code.strip(), *tests]
    ok, stdout, stderr = run_python("\n\n".join(part for part in body if part.strip()), timeout=8)
    passed = ok
    score = 1.0 if ok else 0.0
    return VerificationResult(
        verifier_type="code",
        verifier_pass=passed,
        reward_score=score,
        parsed_answer=code,
        reason="tests_passed" if ok else "tests_failed",
        details={"stdout": stdout, "stderr": stderr},
    )


def verify_candidate(example: dict[str, Any], prediction: str) -> dict[str, Any]:
    domain = example["domain"]
    format_result = format_verifier(prediction, domain)
    if domain == "code":
        task_result = code_verifier(
            prediction,
            tests=example.get("tests", []),
            test_setup_code=example.get("test_setup_code", ""),
        )
    else:
        task_result = math_verifier(prediction, example["reference_answer"])

    reward = round(0.2 * format_result.reward_score + 0.8 * task_result.reward_score, 4)
    passed = format_result.verifier_pass and task_result.verifier_pass
    return {
        "verifier_type": task_result.verifier_type,
        "verifier_pass": passed,
        "format_pass": format_result.verifier_pass,
        "reward_score": reward,
        "parsed_answer": task_result.parsed_answer or format_result.parsed_answer,
        "verification_reason": task_result.reason,
        "verification_details": task_result.details,
    }
