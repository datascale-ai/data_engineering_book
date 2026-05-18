# Project 13: Multimodal Instruction Factory

A pipeline for synthesizing high-quality, complex multimodal instructions (Qwen-VL style).

## Scripts
- `seed_selector.py`: Filters high-quality seeds from LAION.
- `instruction_templates.py`: Templates for different instruction tasks.
- `generate_with_qwen_vl.py`: vLLM generation script using Qwen2.5-VL.
- `llm_judge.py`: Quality filtering using Qwen2.5-72B-Instruct.
- `self_consistency.py`: Self-consistency sampling verification.
- `multilingual_expand.py`: Expands data multilingually.
- `pack_multi_image_video.py`: Packs into target JSONL format.
