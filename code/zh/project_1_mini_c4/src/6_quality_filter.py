import os
import json
import kenlm
from tqdm import tqdm

# ================= 配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

INPUT_FILE = os.path.join(DATA_DIR, "data_en.jsonl") 
OUTPUT_FILE = os.path.join(DATA_DIR, "final_data.jsonl")
MODEL_PATH = os.path.join(MODEL_DIR, "en.arpa.bin")

# 根据调试结果：
# -5.3 ~ -5.9 是正常句子
# -6.4 是导航菜单垃圾
# 所以我们选 -6.0 作为分界线
PERPLEXITY_THRESHOLD = -6.0

# =======================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到 KenLM 模型: {MODEL_PATH}")
        return

    print(f"🚀 加载 KenLM 模型: {MODEL_PATH} ...")
    model = kenlm.Model(MODEL_PATH)
    print("✅ 模型加载完毕！")

    stats = {
        "total": 0,
        "kept": 0,
        "dropped": 0
    }

    print(f"🔄 开始质量过滤 (阈值: {PERPLEXITY_THRESHOLD})...")
    
    # 用于调试：只打印前几条的得分
    debug_count = 0 
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="KenLM Filtering"):
            stats["total"] += 1
            try:
                item = json.loads(line)
                text = item.get("text", "")
                
                words = text.split()
                num_words = len(words)
                
                # 过滤极短文本
                if num_words < 3:
                    stats["dropped"] += 1
                    continue

                # --- 核心计算 ---
                log_score = model.score(text)
                normalized_score = log_score / num_words
                
                # if debug_count < 5:
                #     status = "✅ 保留" if normalized_score > PERPLEXITY_THRESHOLD else "❌ 丢弃"
                #     print(f"\n[调试] ID: {debug_count+1}")
                #     print(f"  得分: {normalized_score:.4f}")
                #     print(f"  状态: {status}")
                #     print(f"  文本: {text[:60]}...") # 只打印前60个字符
                #     debug_count += 1

                # --- 判定 ---
                if normalized_score > PERPLEXITY_THRESHOLD:
                    item["perplexity_score"] = normalized_score
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                    stats["kept"] += 1
                else:
                    stats["dropped"] += 1
                    
            except Exception as e:
                continue

    print("\n🎉 全部流程结束！")
    print(f"📊 最终统计:")
    print(f"   输入总数: {stats['total']}")
    print(f"   🗑️ 丢弃 (低质量): {stats['dropped']} ({(stats['dropped']/stats['total'])*100:.2f}%)")
    print(f"   💎 保留 (高质量): {stats['kept']}")
    print(f"💾 最终文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()