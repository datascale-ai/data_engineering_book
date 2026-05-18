# 模拟生成逻辑
from instruction_templates import get_random_prompt

def generate_instructions(seeds, model_path="Qwen/Qwen2.5-VL-7B-Instruct"):
    print(f"Initializing vLLM engine with {model_path}...")
    results = []
    for seed in seeds:
        task = "detailed_description"
        prompt = get_random_prompt(task)
        results.append({
            "url": seed["url"],
            "task": task,
            "instruction": prompt,
            "response": f"Generated complex reasoning for image at {seed['url']}. Detailed elements identified. The overall structure is sound and coherent, providing logical deductions and deep insights."
        })
    print(f"Generated answers for {len(inputs)} seeds...")
    return results
