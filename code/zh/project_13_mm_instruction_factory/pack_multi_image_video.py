import json

def pack_to_qwen_format(scored_data, output_path="./data/mm_sft_final.jsonl"):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    formatted_dataset = []
    for item in scored_data:
        record = {
            "type": "image",
            "image": item["url"],
            "conversations": [
                {
                    "from": "user",
                    "value": f"<image>\n{item['instruction']}"
                },
                {
                    "from": "assistant",
                    "value": item["response"]
                }
            ]
        }
        formatted_dataset.append(record)
        
    with open(output_path, "w", encoding="utf-8") as f:
        for record in formatted_dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(formatted_dataset)} samples to {output_path}")

if __name__ == "__main__":
    dummy_data = [{"url": "http://example.jpg", "instruction": "Describe", "response": "A highly detailed complex description extending past 50 words to pass the simulated quality check threshold in the LLM judge.", "judge_score": 4.5}]
    pack_to_qwen_format(dummy_data)
