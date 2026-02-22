import json
import os
import re
import subprocess
import signal
from tqdm import tqdm

# --- 路径配置 ---
INPUT_FILE = "../data/evolved_samples.jsonl"
OUTPUT_FILE = "../data/verified_textbook.jsonl" # 最终成品

# --- 工具函数 ---

def extract_python_code(text):
    """从 Markdown 文本中提取 Python 代码块"""
    # 匹配 ```python ... ``` 或者 只是 ``` ... ```
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    
    # 如果没写 python 标签，尝试匹配通用代码块
    pattern_generic = r"```\s*(.*?)\s*```"
    match_generic = re.search(pattern_generic, text, re.DOTALL)
    if match_generic:
        return match_generic.group(1)
    
    return None

def execute_code(code, timeout=5):
    """
    执行 Python 代码并获取输出。
    注意：这是直接在本地执行，生产环境建议用 Docker 隔离。
    """
    try:
        # 使用 subprocess 启动一个新的 Python 进程执行代码
        result = subprocess.run(
            ['python3', '-c', code],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, f"Error: {result.stderr.strip()}"
            
    except subprocess.TimeoutExpired:
        return False, "Error: Execution Timed Out"
    except Exception as e:
        return False, f"Error: {str(e)}"

# --- 主流程 ---

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {os.path.abspath(INPUT_FILE)}")
        return

    print(f"🚀 开始沙箱验证...")
    print(f"📄 读取: {INPUT_FILE}")
    
    valid_count = 0
    total_count = 0
    
    # 准备写入最终文件
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        
        lines = fin.readlines()
        total_count = len(lines)
        
        for line in tqdm(lines, desc="Verifying Code"):
            try:
                entry = json.loads(line)
            except:
                continue
                
            # 1. 提取代码
            raw_response = entry.get('pot_solution', '')
            code = extract_python_code(raw_response)
            
            if not code:
                # 没提取到代码，跳过
                continue
            
            # 2. 执行代码
            is_success, output = execute_code(code)
            
            # 3. 如果成功，保存进最终数据集
            if is_success and output:
                # 构建最终训练数据格式
                # 这种格式包含了问题、思考过程(代码)、和执行结果(答案)
                final_entry = {
                    "question": entry['evolved_question'],
                    "thought_process": raw_response, # 包含注释和代码的完整回答
                    "executable_code": code,
                    "execution_output": output,
                    "source": "synthetic_math_v1"
                }
                fout.write(json.dumps(final_entry, ensure_ascii=False) + '\n')
                valid_count += 1
    
    print(f"\n✅ 验证结束！")
    print(f"📊 统计: 总数 {total_count} -> 合格 {valid_count}")
    print(f"通过率: {valid_count/total_count:.1%}")
    print(f"💾 最终数据集已保存至: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()