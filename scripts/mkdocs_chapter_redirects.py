"""mkdocs hook: write <meta refresh> stub pages at old chapter URLs.

Phase 4.1 of the optimization plan renamed Part 4-9 chapter files from
`X_Y_slug.md` (and `ChXX_slug.md` in part9) to `chXX_slug.md`. To keep
external links working, this hook generates HTML stub files at the old
URLs that redirect to the new URLs.

Both the default-locale URL (e.g. /part4/4_1_SFT/) and the explicit-locale
URL (e.g. /zh/part4/4_1_SFT/) get a stub.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


REDIRECTS: dict[str, str] = {
    "part4/4_1_SFT": "part4/ch12_sft",
    "part4/4_2_preference": "part4/ch13_preference",
    "part4/4_3_qa": "part4/ch14_qa",
    "part5/5_1_data_synthesis": "part5/ch15_data_synthesis",
    "part5/5_2_distillation": "part5/ch16_distillation",
    "part5/5_3_quality": "part5/ch17_quality",
    "part6/6_1_COT": "part6/ch18_cot",
    "part6/6_2_tool": "part6/ch19_tool",
    "part6/6_3_agent": "part6/ch20_agent",
    "part7/7_1_rag_pipeline": "part7/ch21_rag_pipeline",
    "part7/7_2_multimodal_rag_visual_retrieval": "part7/ch22_multimodal_rag_visual_retrieval",
    "part7/7_3_online_feedback_knowledge_update": "part7/ch23_online_feedback_knowledge_update",
    "part8/8_1_dataops_flywheel_team": "part8/ch24_dataops_flywheel_team",
    "part8/8_2_data_versioning_experiment_tracking": "part8/ch25_data_versioning_experiment_tracking",
    "part8/8_3_data_platform_observability": "part8/ch26_data_platform_observability",
    "part9/Ch27_compliance_framework_and_governance": "part9/ch27_compliance_framework_and_governance",
    "part9/Ch28_federated_learning_and_privacy_preserving_technologies": "part9/ch28_federated_learning_and_privacy_preserving_technologies",
}


_STUB = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Redirecting…</title>
<link rel="canonical" href="{target}/">
<meta name="robots" content="noindex">
<meta http-equiv="refresh" content="0; url={target}/">
<script>location.replace("{target}/" + location.hash);</script>
</head>
<body>
<p>Redirecting to <a href="{target}/">{target}/</a>…</p>
</body>
</html>
"""


def on_post_build(config: Any, **_: Any) -> None:
    site_dir = Path(config["site_dir"])
    for old, new in REDIRECTS.items():
        # Default-locale URL: /<old>/index.html → /<new>/
        _write_stub(site_dir / old / "index.html", target=f"/{new}")
        # Explicit-locale URL: /zh/<old>/index.html → /zh/<new>/
        _write_stub(site_dir / "zh" / old / "index.html", target=f"/zh/{new}")


def _write_stub(path: Path, target: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_STUB.format(target=target), encoding="utf-8")
