#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = [
    ("P1", ROOT / "code/zh/project_1_mini_c4", "src/run_p1_checks.py"),
    ("P2", ROOT / "code/zh/project_2_sft_data", "src/run_p2_checks.py"),
    ("P3", ROOT / "code/zh/project_3_llava_data", "src/run_p3_checks.py"),
    ("P4", ROOT / "code/zh/project_4_synth", "src/run_p4_checks.py"),
    ("P5", ROOT / "code/zh/project_5_rag", "src/run_p5_checks.py"),
    ("P6", ROOT / "code/zh/project_6_prm_data", "src/run_p6_checks.py"),
    ("P7", ROOT / "code/zh/project_7_agent_tooluse", "src/run_p7_checks.py"),
    ("P8", ROOT / "code/zh/project_8_dataops_platform", "src/run_p8_checks.py"),
    ("P9", ROOT / "code/zh/project_9_privacy_pipeline", "src/run_p9_checks.py"),
    ("P10", ROOT / "code/zh/project_10_llm_flywheel", "src/run_p10_checks.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all project smoke checks and write a summary report.")
    parser.add_argument("--project", action="append", help="Project id to run, such as P1. Can be repeated.")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per project in seconds.")
    parser.add_argument("--full", action="store_true", help="Run each project's artifact-level check script.")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "smoke_reports")
    return parser.parse_args()


def run_project(project_id: str, project_root: Path, script: str, timeout: int, full: bool) -> dict:
    script_path = project_root / script
    src_files = sorted((project_root / "src").glob("*.py"))
    missing = [
        path
        for path in [
            project_root / "README.md",
            project_root / "environment.yml",
            script_path,
        ]
        if not path.exists()
    ]
    if missing:
        return {
            "project": project_id,
            "script": str(script_path.relative_to(ROOT)),
            "passed": False,
            "returncode": None,
            "stdout": "",
            "stderr": "Missing required project files: "
            + ", ".join(str(path.relative_to(ROOT)) for path in missing),
        }
    if not src_files:
        return {
            "project": project_id,
            "script": str(script_path.relative_to(ROOT)),
            "passed": False,
            "returncode": None,
            "stdout": "",
            "stderr": "No Python files found under src/.",
        }

    if not full:
        completed = subprocess.run(
            [sys.executable, "-m", "py_compile", *[str(path) for path in src_files]],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "project": project_id,
            "script": "py_compile src/*.py",
            "passed": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }

    if not script_path.exists():
        return {
            "project": project_id,
            "script": str(script_path.relative_to(ROOT)),
            "passed": False,
            "returncode": None,
            "stdout": "",
            "stderr": "Smoke check script is missing.",
        }

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "project": project_id,
            "script": str(script_path.relative_to(ROOT)),
            "passed": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "project": project_id,
            "script": str(script_path.relative_to(ROOT)),
            "passed": False,
            "returncode": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": f"Timed out after {timeout} seconds.",
        }


def render_markdown(results: list[dict], timestamp: str) -> str:
    passed = sum(1 for item in results if item["passed"])
    lines = [
        "# Project Smoke Test Report",
        "",
        f"- Timestamp: {timestamp}",
        f"- Overall status: {'PASS' if passed == len(results) else 'FAIL'}",
        f"- Passed projects: {passed}/{len(results)}",
        "",
        "| Project | Status | Script | Return code |",
        "| --- | --- | --- | ---: |",
    ]
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(f"| {result['project']} | {status} | `{result['script']}` | {result['returncode']} |")

    lines.extend(["", "## Failure Details", ""])
    for result in results:
        if result["passed"]:
            continue
        lines.append(f"### {result['project']}")
        if result["stderr"]:
            lines.append("")
            lines.append("```text")
            lines.append(result["stderr"][-2000:])
            lines.append("```")
        if result["stdout"]:
            lines.append("")
            lines.append("```text")
            lines.append(result["stdout"][-2000:])
            lines.append("```")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    selected = {item.upper() for item in args.project} if args.project else None
    projects = [item for item in PROJECTS if selected is None or item[0] in selected]
    timestamp = datetime.now(timezone.utc).isoformat()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    results = [run_project(project_id, root, script, args.timeout, args.full) for project_id, root, script in projects]

    payload = {
        "timestamp_utc": timestamp,
        "overall_passed": all(item["passed"] for item in results),
        "results": results,
    }
    (args.report_dir / "project_smoke_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.report_dir / "project_smoke_report.md").write_text(render_markdown(results, timestamp), encoding="utf-8")

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{result['project']}: {status} ({result['script']})")

    return 0 if payload["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
