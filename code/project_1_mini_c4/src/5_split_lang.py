import json
import fasttext
import re
import os


current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_dir, '..', 'models', 'lid.176.ftz')

model = fasttext.load_model(model_path)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

INPUT_FILE = os.path.join(DATA_DIR, "deduplicated_data.jsonl")  
OUTPUT_DIR = os.path.join(current_dir, '..', 'data', 'processed')
# 定义输出文件句柄
files = {
    'en': open(os.path.join(OUTPUT_DIR, 'data_en.jsonl'), 'w', encoding='utf-8'),
    'zh': open(os.path.join(OUTPUT_DIR, 'data_zh.jsonl'), 'w', encoding='utf-8'), # 包含简繁
    'others': open(os.path.join(OUTPUT_DIR, 'data_others.jsonl'), 'w', encoding='utf-8')
}

print("开始处理数据...")
count = 0

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            text = data.get('text', '').replace('\n', ' ')
            
            if not text:
                continue

            # 使用 FastText 预测语言
            # k=1 表示取可能性最大的那个
            predictions = model.predict(text, k=1) 
            lang_label = predictions[0][0] # 例如 __label__en
            lang = lang_label.replace('__label__', '')
            
            # 简单的分类逻辑
            if lang == 'en':
                files['en'].write(json.dumps(data, ensure_ascii=False) + '\n')
            elif lang == 'zh':
                files['zh'].write(json.dumps(data, ensure_ascii=False) + '\n')
            else:
                # 俄语和其他语言去这里
                data['detected_lang'] = lang # 顺便标记一下是什么语言
                files['others'].write(json.dumps(data, ensure_ascii=False) + '\n')
                
            count += 1
            if count % 1000 == 0:
                print(f"已处理 {count} 行...")
                
        except Exception as e:
            print(f"Skipping line due to error: {e}")

# 关闭文件
for f in files.values():
    f.close()

print("处理完成！生成了 data_en.jsonl, data_zh.jsonl 和 data_others.jsonl")