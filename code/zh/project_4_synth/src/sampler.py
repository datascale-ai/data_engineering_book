import json
import random
import os

# 配置
INPUT_FILE = "../data/gsm8k_train.jsonl"  # 输入文件
OUTPUT_FILE = "../data/seed_samples.jsonl" # 输出的测试样本
SAMPLE_SIZE = 10  # 先取10条来测试

def sample_data():
    data = []
    
    # 1. 读取原始数据
    print(f"正在读取 {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # 2. 随机打乱并抽取
    print(f"原始数据共 {len(data)} 条，正在抽取 {SAMPLE_SIZE} 条...")
    sampled = random.sample(data, SAMPLE_SIZE)
    
    # 3. 保存采样数据
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in sampled:
            # GSM8K 的原始字段是 'question' 和 'answer'
            # 我们只需要 question 作为种子
            seed_entry = {
                "id": random.randint(1000, 9999), # 给个ID方便追踪
                "seed_question": entry['question'],
                "original_answer": entry['answer']
            }
            f.write(json.dumps(seed_entry, ensure_ascii=False) + '\n')
            
    print(f"✅ 采样完成！样本已保存至 {OUTPUT_FILE}")
    
    # 打印一条看看样子
    print("\n--- 样本示例 ---")
    print(f"问题: {sampled[0]['question']}")

if __name__ == "__main__":
    sample_data()