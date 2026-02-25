import ray
import json
import os
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import time

# ================= 配置 =================
# 自动设置路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

INPUT_FILE = os.path.join(DATA_DIR, "clean_data.jsonl")  # 上一步清洗完的文件
OUTPUT_FILE = os.path.join(DATA_DIR, "deduplicated_data.jsonl")

# MinHash 参数 (C4 标准参数: num_perm=128)
NUM_PERM = 128 
THRESHOLD = 0.8  # 相似度阈值，超过 0.8 视为重复

# ================= Ray Actor =================
# 我们初始化 Ray，利用单机所有 CPU 核心
ray.init(ignore_reinit_error=True)

def get_minhash(text, num_perm=128):
    """
    计算一段文本的 MinHash 签名
    """
    m = MinHash(num_perm=num_perm)
    # 使用简单的 shingle (按单词分)
    words = text.split()
    for w in words:
        m.update(w.encode('utf8'))
    return m

@ray.remote
def process_batch(lines, batch_id):
    """
    Ray Worker: 处理一批数据，计算 MinHash
    返回: List of (url, minhash_obj, text_content)
    """
    results = []
    for line in lines:
        try:
            item = json.loads(line)
            url = item['url']
            text = item['text']
            
            # 计算签名
            minhash = get_minhash(text, NUM_PERM)
            results.append((url, minhash, text))
        except Exception:
            continue
    return results

# ================= 主流程 =================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return

    print("🚀 第一阶段: 并行计算 MinHash 签名...")
    
    # 1. 读取所有数据并分批 (Batching)
    # 为了避免内存爆炸，我们按块读取，但为了演示简单，这里假设内存够大
    # 实际上应该用 Ray Dataset 或者流式读取，这里用简易版分批
    batch_size = 1000
    all_lines = []
    
    # 读取文件 (如果文件有几十G，不要这样读，要用流式)
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    total_records = len(all_lines)
    print(f"📚 总记录数: {total_records}")

    # 将数据切分成小块，分发给 Ray
    batches = [all_lines[i:i + batch_size] for i in range(0, total_records, batch_size)]
    
    # 提交任务给 Ray (非阻塞)
    futures = [process_batch.remote(batch, i) for i, batch in enumerate(batches)]
    
    # 获取结果 (阻塞等待所有 CPU 算完)
    print("⏳ 等待 CPU 计算中 ")
    processed_batches = ray.get(futures)
    
    # 展平结果
    # results 结构: [(url, minhash, text), (url, minhash, text), ...]
    results = [item for batch in processed_batches for item in batch]

    
    print("\n🚀 第二阶段: 构建 LSH 索引并去重...")
    # 这一步通常难以并行化，必须在主进程构建全局索引
    # 就像查字典一样，必须有一本完整的字典
    
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    
    unique_records = []
    duplicate_count = 0
    
    # 开始遍历并查重
    for url, minhash, text in tqdm(results, desc="LSH Deduplication"):
        # 查询 LSH 桶里是否有相似的
        # query 返回的是已经存在于桶里的 key (这里我们用 url 当 key)
        duplicates = lsh.query(minhash)
        
        if len(duplicates) > 0:
            # 发现重复！
            duplicate_count += 1
            # 策略：简单的丢弃当前这条，保留桶里那条
            # 你也可以根据时间戳保留最新的，这里从简
        else:
            # 没有重复，插入桶中
            lsh.insert(url, minhash)
            unique_records.append({"url": url, "text": text})

    print(f"\n✅ 去重完成！")
    print(f"🗑️ 发现重复: {duplicate_count}")
    print(f"💎 剩余有效: {len(unique_records)}")
    
    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in unique_records:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    ray.shutdown()

if __name__ == "__main__":
    main()