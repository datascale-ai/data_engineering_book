import pdfplumber
import os
import re
import json
from tqdm import tqdm

# --- 1. 配置路径 ---
RAW_DATA_DIR = '../data/raw' 
PROCESSED_DATA_DIR = '../data/processed'

# 确保输出目录存在
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- 2. 核心组件：智能清洗函数 (增强修复版) ---
def clean_text_smart(text):
    """
    针对 PDF 提取的文本进行深度清洗
    解决：页码混入、引用标号残留、中文被空格切断等问题
    """
    if not text:
        return ""

    # A. 去除参考文献引用标号
    # 匹配 [1]、[1,2]、[1-3]、［12］ 这种格式
    text = re.sub(r'\[\s*\d+(?:[-–,]\d+)*\s*\]', '', text)
    text = re.sub(r'［\s*\d+(?:[-–,]\d+)*\s*］', '', text)

    # B. 【新增】去除嵌在文本中间的页码 (针对 - 195 - 问题)
    # 解释：
    # (?:^|\s|\\n)  : 前面必须是开头、空白或换行
    # [-—–－]       : 匹配各种破折号（包括全角 －，半角 -，Em dash — 等）
    # \s*\d+\s* : 中间是数字，允许有空格
    # [-—–－]       : 后面必须跟着闭合的破折号
    # (?=\s|\\n|$)  : 后面必须是空白、换行或结尾 (避免误删 "Item-1-A" 这种编号)
    text = re.sub(r'(?:^|\s|\\n)[-—–－]\s*\d+\s*[-—–－](?=\s|\\n|$)', ' ', text)

    # C. 去除孤立的行级页码
    # 策略：如果一行只有数字和横杠，删掉。
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        s_line = line.strip()
        # 如果这一行全是数字和各种符号（破折号、空格），判定为页码行
        if re.fullmatch(r'[-—–－\s\d]+', s_line):
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # D. 修复中文断词 (核心修复)
    # 逻辑：查找 [中文] [空格] [中文] 的模式，把空格去掉
    # 执行两次以处理连续的断词 (如 "A B C" -> "AB C" -> "ABC")
    pattern_broken_zh = r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])'
    text = re.sub(pattern_broken_zh, r'\1\2', text)
    text = re.sub(pattern_broken_zh, r'\1\2', text) 

    # E. 规范化空白字符
    # 这里的策略是保留换行符 \n 以维持段落结构，但把行内的多个空格合并为一个
    # 也就是：把 "word1   word2" 变成 "word1 word2"，但不把 "\n" 变成空格
    # 如果想完全压扁成一行，可以用 re.sub(r'\s+', ' ', text)
    text = re.sub(r'[ \t\r\f]+', ' ', text) # 仅合并水平空白，保留 \n
    
    return text.strip()

# --- 3. 策略 A：法律文档处理 ---
def process_legal_doc(file_path):
    filename = os.path.basename(file_path)
    print(f"⚖️ [法律] 正在解析: {filename}")
    full_text = ""
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in tqdm(pdf.pages, desc="读取页面", leave=False):
                # 裁剪页眉页脚 (上下各切除 5%)
                width, height = page.width, page.height
                bbox = (0, height * 0.05, width, height * 0.95)
                try:
                    page_crop = page.crop(bbox=bbox)
                    text = page_crop.extract_text()
                    if text:
                        full_text += "\n" + text
                except Exception:
                    continue
    except Exception as e:
        print(f"❌ 无法打开文件 {filename}: {e}")
        return []

    # 全文清洗
    full_text = clean_text_smart(full_text)

    # 正则匹配：第[数字/中文]条
    # 优化正则：支持 "第一条" 和 "第1条"
    pattern = r"(第[0-9零一二三四五六七八九十百千]+条[\s\S]*?)(?=第[0-9零一二三四五六七八九十百千]+条|$)"
    matches = re.findall(pattern, full_text)
    
    chunks = []
    for match in matches:
        # 再次清洗单个片段多余的空白
        cleaned = re.sub(r'\s+', ' ', match).strip()
        if len(cleaned) > 15: # 稍微提高阈值，过滤杂音
            chunks.append({
                "source": filename,
                "type": "legal_article",
                "content": cleaned
            })
            
    print(f"   => 提取到 {len(chunks)} 个法条片段")
    return chunks

# --- 4. 策略 B：医疗指南处理 (滑动窗口) ---
def process_medical_doc(file_path, chunk_size=500, overlap=100):
    filename = os.path.basename(file_path)
    print(f"⚕️ [医疗] 正在解析: {filename}")
    full_text = ""
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in tqdm(pdf.pages, desc="读取页面", leave=False):
                width, height = page.width, page.height
                bbox = (0, height * 0.05, width, height * 0.95)
                try:
                    page_crop = page.crop(bbox=bbox)
                    text = page_crop.extract_text()
                    if text:
                        # 拼接时加上换行，防止跨页粘连
                        full_text += text + "\n"
                except Exception:
                    continue
    except Exception as e:
        print(f"❌ 无法打开文件 {filename}: {e}")
        return []

    # 全文清洗 (在切片前做，修复断词效果最好)
    full_text = clean_text_smart(full_text)
    # 将多余换行压扁，方便滑动窗口计算字数
    full_text = re.sub(r'\s+', ' ', full_text)

    # --- 滑动窗口切分 ---
    chunks = []
    total_len = len(full_text)
    start = 0
    
    pbar = tqdm(total=total_len, desc="切分文本", unit="char", leave=False)
    
    while start < total_len:
        end = min(start + chunk_size, total_len)
        chunk_text = full_text[start:end]
        
        # 智能截断：优先在句号处截断
        last_period = chunk_text.rfind('。')
        
        # 只有截断后长度依然合理（> overlap）才截断
        if last_period != -1 and (start + last_period) < total_len:
            current_length = last_period + 1
            if current_length > overlap:
                end = start + current_length
                chunk_text = full_text[start:end]
            
        if len(chunk_text) > 50: 
            chunks.append({
                "source": filename,
                "type": "medical_guide",
                "content": chunk_text
            })
        
        # 步长计算 (防止死循环)
        step = len(chunk_text) - overlap
        if step <= 0:
            step = 1 # 强制前进
            
        start += step
        pbar.update(step)

    pbar.close()
    print(f"   => 切分为 {len(chunks)} 个文本块")
    return chunks

# --- 5. 主程序 ---
def main():
    if not os.path.exists(RAW_DATA_DIR):
        print(f"错误：找不到源数据目录 {RAW_DATA_DIR}")
        return

    files = [f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith('.pdf')]
    if not files:
        print("目录中没有找到 PDF 文件。")
        return
        
    all_chunks = []
    print(f"🚀 开始清洗处理，共 {len(files)} 个文件...")
    
    for filename in files:
        file_path = os.path.join(RAW_DATA_DIR, filename)
        
        # 简单分类逻辑
        if "法" in filename and "指南" not in filename and "规范" not in filename:
            chunks = process_legal_doc(file_path)
        else:
            chunks = process_medical_doc(file_path)
            
        all_chunks.extend(chunks)

    # 保存
    output_file = os.path.join(PROCESSED_DATA_DIR, 'raw_chunks.jsonl')
    print(f"\n💾 正在保存至 {output_file} ...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
    print(f"✅ 处理完成！总计生成 {len(all_chunks)} 条清洗后的数据。")

if __name__ == "__main__":
    main()