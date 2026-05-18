# Project 11: Mini-DeepSeek Data Pipeline

This repository contains the code to replicate the data engineering pipeline for a Mini-DeepSeek style model training (1B tokens subset).

## Components
- `mix_sampler.py`: Multi-source dataset sampler based on specific ratios.
- `cross_source_dedup.py`: MinHash LSH based deduplication across different sources.
- `train_tokenizer.py`: Trains a BPE tokenizer with 150K vocabulary size.
- `pack_shuffle.py`: Packs variable-length tokens into 4096-length shards and shuffles globally.
- `run_pipeline.sh`: End-to-end execution script.

## Usage
Simply run the shell script:
```bash
./run_pipeline.sh
```

## Dataset Licenses
- FineWeb-Edu: CC0
- The Stack v2: SPDX Whitelist
- OpenWebMath: ODC-By
- arXiv: ArXiv specific licenses
