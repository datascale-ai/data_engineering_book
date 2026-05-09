from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pipeline_utils import (
    DEFAULT_MODEL_PATH,
    LOGS_DIR,
    TRAINING_DIR,
    dump_run_report,
    ensure_dir,
    ensure_standard_dirs,
    import_available,
    load_jsonl,
    make_arg_parser,
    render_qwen_chat,
    setup_logging,
    write_json,
)

MERGED_FILE = TRAINING_DIR / "merged_sft_data.jsonl"
LORA_DIR = TRAINING_DIR / "lora_ckpt"
TRAINING_RESULT = TRAINING_DIR / "lora_training_result.json"
def run_lora_training(
    dataset_path: Path,
    output_dir: Path,
    model_path: str,
    max_train_samples: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    force_mock: bool = False,
) -> dict[str, Any]:
    ensure_standard_dirs()
    logger = setup_logging("train_lora", LOGS_DIR / "train_lora.log")
    rows = load_jsonl(dataset_path)[:max_train_samples]
    ensure_dir(output_dir)

    if force_mock or not (import_available("transformers") and import_available("peft")):
        logger.info("using mock LoRA training path")
        marker = {
            "mode": "mock",
            "message": "Dependencies unavailable or mock requested. This checkpoint is a placeholder for the minimal runnable pipeline.",
            "train_samples": len(rows),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }
        write_json(marker, output_dir / "adapter_config.mock.json")
        write_json(marker, TRAINING_RESULT)
        return marker

    # Minimal trainable demonstration without TRL to keep dependencies small.
    import torch  # pragma: no cover
    from peft import LoraConfig, get_peft_model  # pragma: no cover
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments  # pragma: no cover

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if world_size > 1 and local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        model_kwargs["device_map"] = {"": local_rank}
    else:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    encoded_rows = []
    for row in rows:
        messages = row["messages"]
        full_text = render_qwen_chat(tokenizer, messages, add_generation_prompt=False)
        prompt_text = render_qwen_chat(tokenizer, messages[:-1], add_generation_prompt=True)
        full_ids = tokenizer(full_text, truncation=True, max_length=1024)["input_ids"]
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=1024)["input_ids"]
        labels = full_ids.copy()
        assistant_start = min(len(prompt_ids), len(labels))
        for idx in range(assistant_start):
            labels[idx] = -100
        encoded_rows.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )

    class TinyDataset(torch.utils.data.Dataset):  # pragma: no cover
        def __len__(self):
            return len(encoded_rows)

        def __getitem__(self, idx):
            return {key: torch.tensor(value) for key, value in encoded_rows[idx].items()}

    def data_collator(features):  # pragma: no cover
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        return batch

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        logging_steps=1,
        save_steps=20,
        save_total_limit=1,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=TinyDataset(), data_collator=data_collator)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    result = {
        "mode": "real",
        "train_samples": len(rows),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "output_dir": str(output_dir),
    }
    write_json(result, TRAINING_RESULT)
    dump_run_report(TRAINING_DIR / "lora_training_report.md", "LoRA Training Report", [("Summary", result)])
    return result


def main() -> None:
    parser = make_arg_parser("Run minimal LoRA SFT training.")
    parser.add_argument("--dataset", type=str, default=str(MERGED_FILE))
    parser.add_argument("--output-dir", type=str, default=str(LORA_DIR))
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--force-mock", action="store_true")
    args = parser.parse_args()
    result = run_lora_training(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        model_path=args.model_path,
        max_train_samples=args.max_train_samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        force_mock=args.force_mock,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
