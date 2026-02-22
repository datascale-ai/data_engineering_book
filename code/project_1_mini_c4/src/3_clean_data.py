import json
import os
from tqdm import tqdm
import re

# ================= 配置部分 =================
# 1. 自动定位路径 (保持和你之前的习惯一致)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# 输入文件 (上一步的输出)
INPUT_FILE = os.path.join(DATA_DIR, "extracted_data.jsonl")
# 输出文件 (清洗后的文件)
OUTPUT_FILE = os.path.join(DATA_DIR, "clean_data.jsonl")

# ================= 核心清洗规则 =================
def is_high_quality(text):
    """
    判断文本是否“干净”。返回 True 表示保留，False 表示丢弃。
    """
    
    # --- 规则 1: 长度过滤 ---
    # 太短通常是导航、版权声明；太长可能是日志堆栈
    if len(text) < 100 or len(text) > 2_000_000:
        return False
        
    # --- 规则 2: 平均词长 (Gopher/C4 经典规则) ---
    # 正常的英语/中文文本，平均词长通常在一定范围内。
    # 如果全是 "a b c d" (过短) 或 "supercalifragilistic..." (过长/代码)，则是垃圾
    words = text.split()
    if len(words) == 0:
        return False
    mean_word_len = sum(len(w) for w in words) / len(words)
    
    # 英文通常 3-10 是正常范围；中文分词后略有不同，但作为粗筛
    # 如果平均每个词超过 15 个字符，通常是 URL 列表或乱码代码
    if mean_word_len > 15:
        return False

    # --- 规则 3: 符号密度 (Symbol Ratio) ---
    # 如果一句话里包含大量的 '{', '}', '<', '>', '#', 通常是代码或 JSON 数据
    # 我们检查常用代码符号的占比
    code_symbols = {'{', '}', '[', ']', '<', '>', '\\'}
    symbol_count = sum(1 for char in text if char in code_symbols)
    
    # 如果代码符号占比超过 10%，视为垃圾
    if symbol_count / len(text) > 0.1:
        return False

    # --- 规则 4: 关键词黑名单 (Blocklist) ---
    # 常见的系统报错、Cookie 提示、Lorem Ipsum
    bad_phrases = [
        "lorem ipsum", 
        "javascript is disabled", 
        "enable cookies",
        "403 forbidden",
        "404 not found",
        "access denied",
        "rights reserved" # 只有版权声明的短文本
    ]
    
    # 转换为小写进行快速检查
    text_lower = text.lower()
    for phrase in bad_phrases:
        if phrase in text_lower:
            # 如果文本很短且包含脏词，直接删掉
            # 如果文本很长但包含脏词，可能是正文里引用了，这里为了严格，简单粗暴处理：
            # 策略：如果文本短于 500 字且包含脏词 -> 删
            if len(text) < 500:
                return False
    
    return True

# ================= 主流程 =================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    print(f"🚀 开始清洗: {INPUT_FILE}")
    print(f"💾 输出位置: {OUTPUT_FILE}")

    stats = {
        "total": 0,
        "kept": 0,
        "dropped": 0
    }

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="Cleaning Data"):
            stats["total"] += 1
            try:
                item = json.loads(line)
                text = item.get("text", "")
                
                # 调用清洗函数
                if is_high_quality(text):
                    f_out.write(line) # 原样写入，或者只写入 text
                    stats["kept"] += 1
                else:
                    stats["dropped"] += 1
                    
            except json.JSONDecodeError:
                continue

    print("\n✅ 清洗完成！")
    print(f"📊 总数: {stats['total']}")
    print(f"🗑️ 丢弃: {stats['dropped']} ({(stats['dropped']/stats['total'])*100:.2f}%)")
    print(f"💎 保留: {stats['kept']}")

if __name__ == "__main__":
    main()