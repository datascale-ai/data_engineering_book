# P12 R1 Reasoning Flywheel

最小可运行版 R1 风格推理数据飞轮工程，聚焦以下模块：

- 冷启动 SFT 数据抽取
- rule-based verifier 池
- vLLM 多次采样
- 拒绝采样
- 二轮 SFT 数据合并
- LoRA 演示训练脚本
- 评估脚本与测试

项目重点是“数据飞轮工程可复现”，不是完整 RL 复现，也不把 benchmark 结果作为当前提交重点。

## 目录

```text
project_12_r1_flywheel/
├── cold_start_data.py
├── verifier_pool.py
├── sample_traces.py
├── rejection_sampling.py
├── merge_sft_data.py
├── train_lora.py
├── eval_gsm8k_math.py
├── pipeline_utils.py
├── mock_generators.py
├── scripts/
├── tests/
├── data_schema.md
├── environment.yml
├── environment.lock.yml
└── README.md
```

## 环境

建议使用独立 conda 环境：

```bash
cd /home/xuxin/book/11_p12_r1_reasoning_flywheel/project_12_r1_flywheel
conda env create -f environment.yml
conda activate p12-r1-flywheel
```

如果需要锁定版本，可参考 `environment.lock.yml`。

## 数据与运行模式

项目默认优先读取本地数据源：

- `data/OpenThoughts`
- `data/MATH-500`
- `data/HumanEval`
- `project_4_synth/data/gsm8k_train.jsonl`
- `project_4_synth/data/mbpp_train.jsonl`

推理服务推荐使用外部 `vllm serve`，项目通过 OpenAI 兼容 API 调用真实采样服务。

## 推荐运行顺序

### 1. 冷启动数据

```bash
python cold_start_data.py \
  --max-openthoughts 5000 \
  --max-math 100 \
  --max-gsm8k 100 \
  --max-code 100
```

### 2. 启动 vLLM 服务

```bash
bash scripts/serve_qwen_vllm.sh
```

常用环境变量：

```bash
export GPU_ID=4
export PORT=18011
export MODEL_PATH=/data/xuxin/Qwen/Qwen2.5-7B-Instruct
export SERVED_MODEL_NAME=Qwen2.5-7B-Instruct
```

### 3. 真实采样

```bash
export R1_INFER_BACKEND=openai
export R1_VLLM_API_BASE=http://127.0.0.1:18011/v1
export R1_VLLM_API_KEY=EMPTY
export R1_SERVED_MODEL_NAME=Qwen2.5-7B-Instruct
export QWEN_MODEL_PATH=/data/xuxin/Qwen/Qwen2.5-7B-Instruct

python sample_traces.py \
  --input data/processed/cold_start_5k.jsonl \
  --output-dir data/sampled_traces \
  --num-examples 100 \
  --num-samples 4 \
  --backend openai
```

### 4. 拒绝采样

```bash
python rejection_sampling.py \
  --cold-start data/processed/cold_start_5k.jsonl \
  --sample-dir data/sampled_traces \
  --selected-per-prompt 2 \
  --min-reward 0.8
```

### 5. 合并 SFT 数据

```bash
python merge_sft_data.py
```

### 6. LoRA 演示训练

```bash
python train_lora.py \
  --dataset data/training/merged_sft_data.jsonl \
  --output-dir data/training/lora_ckpt
```

### 7. 评估脚本

```bash
python eval_gsm8k_math.py \
  --model-path /data/xuxin/Qwen/Qwen2.5-7B-Instruct \
  --adapter-path data/training/lora_ckpt \
  --max-examples 10 \
  --tasks gsm8k
```

## 测试

```bash
pytest -q
```

当前测试覆盖：

- 冷启动抽样
- verifier
- 多次采样
- 拒绝采样
- 合并逻辑
- 端到端 smoke test

## 说明

- 本提交保留完整工程代码、环境文件、测试和数据格式说明
- 运行产物、缓存和大体积中间文件不建议直接进入 PR
- 如需复现实验，可按本 README 中的顺序重新生成数据与日志
