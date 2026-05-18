#!/bin/bash
set -e

mkdir -p ./data

echo "Step 1: Multi-source sampling..."
python mix_sampler.py

echo "Step 2: Cross-source minhash deduplication..."
python cross_source_dedup.py

echo "Step 3: Training tokenizer..."
python train_tokenizer.py

echo "Step 4: Pack and shuffle data..."
python pack_shuffle.py

echo "Pipeline completed successfully! Shards saved to ./data/mixed_1b_final_packed"
