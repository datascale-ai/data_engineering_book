import json
import random

# 文件路径（请根据你的实际路径调整）
file_path = '../data/processed/raw_chunks.jsonl'

def view_random_samples(path, n=5):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # 读取所有行
            lines = f.readlines()
            
            if not lines:
                print("文件是空的！")
                return
            
            # 随机采样
            samples = random.sample(lines, min(n, len(lines)))
            
            print(f"--- 随机抽检 {len(samples)} 条数据 ---")
            for i, line in enumerate(samples):
                data = json.loads(line)
                print(f"\n[样本 {i+1}]")
                print(f"来源: {data.get('source')}")
                print(f"类型: {data.get('type')}")
                print(f"内容摘要: {data.get('content')[:150]}...") # 仅展示前150字
                print("-" * 30)
                
    except FileNotFoundError:
        print(f"错误：找不到文件 {path}。请确保处理脚本已成功运行。")

if __name__ == "__main__":
    view_random_samples(file_path)