import sys
import json
import gzip
from warcio.archiveiterator import ArchiveIterator
import trafilatura
from tqdm import tqdm
import os

# ================= 配置部分 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 获取项目根目�?(�?src 的上一�?
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 3. 拼接数据目录的绝对路�?
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# 4. 定义具体文件路径
WARC_FILENAME = "CC-MAIN-2023-50-segment-1700679099281.0-1700679117904.warc.gz"
INPUT_FILE = os.path.join(RAW_DIR, WARC_FILENAME)
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "extracted_data.jsonl")

# 限制处理的记录数 (用于测试，设�?None 则处理所�?
LIMIT_RECORDS = 10000 
# ===========================================

def extract_text_from_warc(warc_path, output_path, limit=None):
    """
    读取 WARC 文件，提取正文，并保存为 JSONL
    """
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"🚀 开始处�? {warc_path}")
    print(f"💾 输出结果: {output_path}")

    counter = 0
    success_count = 0
    
    # 打开输出文件
    with open(output_path, 'w', encoding='utf-8') as out_f:
        # 打开输入 WARC 文件
        with open(warc_path, 'rb') as stream:
            # ArchiveIterator 会自动处�?gzip 解压
            for record in tqdm(ArchiveIterator(stream), desc="Processing Records"):
                
                # 我们只关�?HTTP 响应 (response)，忽略请�?(request) 和元数据
                if record.rec_type == 'response':
                    
                    # 1. 检查是否是 HTML 内容
                    content_type = record.http_headers.get_header('Content-Type')
                    if not content_type or 'text/html' not in content_type:
                        continue
                    
                    # 2. 读取原始字节�?
                    try:
                        content = record.content_stream().read()
                    except Exception:
                        continue
                        
                    # 3. 使用 Trafilatura 提取正文
                    # include_comments=False: 去除网友评论 (根据需求调�?
                    # include_tables=False: 去除表格
                    # no_fallback=True: 如果快速模式失败，不尝试备用解析器 (为了速度)
                    text = trafilatura.extract(
                        content, 
                        include_comments=False, 
                        include_tables=False, 
                        no_fallback=False
                    )
                    
                    # 4. 如果提取到了文本，则保存
                    if text and len(text.strip()) > 0:
                        # 获取 URL
                        url = record.rec_headers.get_header('WARC-Target-URI')
                        
                        # 构建数据对象
                        data = {
                            "url": url,
                            "text": text,
                            # 你可以在这里添加更多元数据，�?timestamp
                        }
                        
                        # 写入一�?JSON
                        out_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                        success_count += 1
                
                counter += 1
                if limit and counter >= limit:
                    break
    
    print(f"\n�?处理完成�?)
    print(f"📊 扫描记录�? {counter}")
    print(f"📄 成功提取�? {success_count}")

def main():
    # 自动查找目录下第一�?warc.gz 文件 (方便你不用手动改文件�?
    input_path = INPUT_FILE
    if not os.path.exists(input_path):
        raw_dir = os.path.dirname(INPUT_FILE)
        files = [f for f in os.listdir(raw_dir) if f.endswith('.warc.gz')]
        if files:
            input_path = os.path.join(raw_dir, files[0])
            print(f"自动发现文件: {input_path}")
        else:
            print(f"�?错误: 找不到输入文�?{INPUT_FILE}，且目录下没有其�?warc.gz 文件")
            return

    extract_text_from_warc(input_path, OUTPUT_FILE, LIMIT_RECORDS)

if __name__ == "__main__":
    main()
