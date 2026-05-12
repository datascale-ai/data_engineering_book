from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cold_start_data import build_cold_start_data
from eval_gsm8k_math import run_eval
from merge_sft_data import merge_sft_data
from rejection_sampling import run_rejection_sampling
from sample_traces import run_sampling
from train_lora import run_lora_training
from verifier_pool import code_verifier, format_verifier, math_verifier, verify_candidate


def test_cold_start_builds_records(tmp_path):
    output = tmp_path / "cold_start.jsonl"
    summary = build_cold_start_data(
        max_openthoughts=0,
        max_math=0,
        max_gsm8k=8,
        max_code=4,
        seed=123,
        output_path=output,
    )
    assert output.exists()
    assert summary["record_count"] >= 8
    assert "math" in summary["domain_distribution"]


def test_math_verifier_exact_match():
    result = math_verifier("Reasoning: compute.\nFinal Answer: 42", "42")
    assert result.verifier_pass is True
    assert result.reward_score == 1.0


def test_format_verifier_math():
    result = format_verifier("Reasoning: hello\nFinal Answer: 9", "math")
    assert result.verifier_pass is True


def test_code_verifier_executes():
    prediction = "Reasoning: implement.\nCode:\n```python\ndef add(a, b):\n    return a + b\n```"
    tests = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]
    result = code_verifier(prediction, tests)
    assert result.verifier_pass is True


def test_sample_and_reject_pipeline(tmp_path):
    cold = tmp_path / "cold.jsonl"
    sample_dir = tmp_path / "samples"
    reject = tmp_path / "reject.jsonl"
    build_cold_start_data(0, 0, 6, 2, seed=7, output_path=cold)
    sample_summary = run_sampling(
        input_path=cold,
        output_dir=sample_dir,
        num_examples=4,
        num_samples=3,
        temperature=0.6,
        top_p=0.9,
        max_tokens=256,
    )
    reject_summary = run_rejection_sampling(
        cold_start_path=cold,
        sample_dir=sample_dir,
        selected_per_prompt=2,
        min_reward=0.5,
        output_path=reject,
    )
    assert sample_summary["total_samples"] == 12
    assert reject.exists()
    assert reject_summary["verified_total"] == 12


def test_verify_candidate_math_path():
    example = {
        "domain": "math",
        "reference_answer": "18",
    }
    verdict = verify_candidate(example, "Reasoning: done\nFinal Answer: 18")
    assert verdict["verifier_pass"] is True


def test_merge_sft_data(tmp_path):
    cold = tmp_path / "cold.jsonl"
    sample_dir = tmp_path / "samples"
    reject = tmp_path / "reject.jsonl"
    merged = tmp_path / "merged.jsonl"
    build_cold_start_data(0, 0, 6, 2, seed=13, output_path=cold)
    run_sampling(cold, sample_dir, 3, 2, 0.6, 0.9, 256)
    run_rejection_sampling(cold, sample_dir, 1, 0.5, reject)
    summary = merge_sft_data(cold, reject, merged)
    assert merged.exists()
    assert summary["merged_records"] >= summary["cold_start_records"]


def test_mock_lora_training(tmp_path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"record_id":"1","messages":[{"role":"system","content":"s"},{"role":"user","content":"u"},{"role":"assistant","content":"a"}]}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "lora"
    result = run_lora_training(
        dataset_path=dataset,
        output_dir=output_dir,
        model_path="dummy",
        max_train_samples=1,
        epochs=1,
        learning_rate=2e-4,
        batch_size=1,
        force_mock=True,
    )
    assert result["mode"] == "mock"
    assert (output_dir / "adapter_config.mock.json").exists()


def test_mock_eval(tmp_path):
    result = run_eval(
        model_path="dummy",
        adapter_path="",
        max_examples=4,
        backend="mock",
        force_mock=True,
    )
    assert "metrics" in result
    assert "gsm8k_base_accuracy" in result["metrics"]
    assert "math_lora_accuracy" in result["metrics"]
