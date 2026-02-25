import json
import os
import time
import requests
from tqdm import tqdm

# --- 配置中心 ---
API_KEY = "sk-lrdpxzsnhsbckhjrzekbrtccomruhcwzyrlwbroqwojtwtsw" 
BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3"

# 文件路径
INPUT_FILE = "../data/seed_samples.jsonl"
OUTPUT_FILE = "../data/evolved_samples.jsonl"

# --- 核心 Prompts ---
def get_evol_prompt(seed_question):
    return f"""
    你是一个专业的数学竞赛命题专家。请将下面这个基础数学问题重写为一个更复杂、逻辑更严密的问题。
    【原题】: {seed_question}
    【重写要求】:
    1. 增加约束条件：引入更多变量或限制。
    2. 增加推理深度：不要直接给出数字，让数字之间存在逻辑关联。
    3. 场景化：将抽象的数字放入具体的物理或商业场景中。
    4. 保持可解性：确保问题依然有明确的数学解。
    5. **只输出新问题的内容**，不要包含“好的”等废话。
    """

def get_pot_prompt(evolved_question):
    return f"""
    请编写一段 Python 代码来解决以下数学问题。
    【问题】: {evolved_question}
    【要求】:
    1. 编写一个名为 `solve()` 的函数。
    2. 在代码注释中清晰地写出推理步骤。
    3. `solve()` 函数必须返回最终的数值答案。
    4. 代码必须是完整可执行的。
    5. 使用 Markdown 代码块格式输出，即：
       ```python
       def solve():
           # ...
           return result
       print(solve())
       ```
    """

# --- API 调用函数 (优化版) ---
def call_siliconflow(prompt, model=MODEL_NAME, max_retries=3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 4096, # 增加 token 上限，防止代码写一半截断
        "stream": False
    }

    for attempt in range(max_retries):
        try:
            # 打印调试信息，让你知道它在工作
            # print(f"  [DEBUG] 发送请求中 (尝试 {attempt+1}/{max_retries})...") 
            
            # 关键修改：timeout 延长到 180 秒
            response = requests.post(BASE_URL, json=payload, headers=headers, timeout=180)
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                if not content:
                    print("  [WARN] API 返回内容为空")
                    continue
                return content
            else:
                print(f"  [API Error] {response.status_code}: {response.text[:100]}...")
                time.sleep(5) # 出错后多等一会
                
        except requests.exceptions.Timeout:
            print(f"  [Timeout] 请求超时 (超过180秒)，正在重试...")
        except Exception as e:
            print(f"  [Connection Error] {e}")
            time.sleep(5)
            
    return None

# --- 主流程 ---
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 未找到种子文件: {os.path.abspath(INPUT_FILE)}")
        return

    seeds = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                seeds.append(json.loads(line))
    
    print(f"🚀 开始进化流程，共 {len(seeds)} 条种子数据...")
    print(f"💡 提示：生成代码可能较慢（每条约30-60秒），请耐心等待。")
    
    results = []
    
    # 使用 tqdm，但设置 mininterval 让它不要刷屏太快
    pbar = tqdm(seeds, mininterval=1.0)
    
    for entry in pbar:
        # 1. 字段适配
        q_text = entry.get('seed_question') or entry.get('base_question') or entry.get('question')
        entry_id = entry.get('id') or entry.get('idx') or "unknown"
        
        if not q_text:
            continue

        pbar.set_description(f"Processing ID {entry_id} (Evol Step)")
        
        # 2. 进化阶段
        evolved_q = call_siliconflow(get_evol_prompt(q_text))
        if not evolved_q:
            continue
            
        pbar.set_description(f"Processing ID {entry_id} (Code Step)")
        
        # 3. 解题阶段
        pot_solution = call_siliconflow(get_pot_prompt(evolved_q))
        if not pot_solution:
            continue
            
        # 4. 保存
        new_entry = {
            "original_id": entry_id,
            "original_question": q_text,
            "evolved_question": evolved_q,
            "pot_solution": pot_solution,
            "model_used": MODEL_NAME
        }
        results.append(new_entry)
        
        # 实时保存（防止程序中途崩溃全白跑）
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
             f.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
        
        # 稍微停顿
        time.sleep(0.5)

    print(f"\n✅ 任务完成！结果已追加保存至: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()