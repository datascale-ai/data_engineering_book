import os
import json
import base64
import random
import glob
from openai import OpenAI
from tqdm import tqdm

# --- 配置区 ---
API_KEY = "sk-lrdpxzsnhsbckhjrzekbrtccomruhcwzyrlwbroqwojtwtsw" # 你的 Key
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen3-VL-32B-Thinking"

# 动态路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(CURRENT_DIR, "../data/images/")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "../data/llava_interleaved.json")
# --------------

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def generate_comparison(img1_path, img2_path):
    fname1 = os.path.basename(img1_path)
    fname2 = os.path.basename(img2_path)
    
    # 构造 Prompt：要求多图对比
    prompt = "Here are two images. Please briefly compare them. What are the common objects or significant differences in the scene?"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img1_path)}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img2_path)}"}}
                    ]
                }
            ],
            max_tokens=200
        )
        content = response.choices[0].message.content
        
        return {
            "id": f"compare_{random.randint(1000,9999)}",
            "image": [fname1, fname2], # 多图列表
            "conversations": [
                {"from": "human", "value": "Image 1: <image>\nImage 2: <image>\nCompare these two images."},
                {"from": "qwen", "value": content}
            ]
        }
    except Exception as e:
        print(f"API Error: {e}")
        return None

def main():
    images = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    if len(images) < 2:
        print("图片不够，无法对比。")
        return

    results = []
    print(">>> 正在生成多图交错数据 (生成 10 条示例)...")
    
    for _ in tqdm(range(10)): # 生成 10 条即可满足项目展示
        pair = random.sample(images, 2)
        entry = generate_comparison(pair[0], pair[1])
        if entry: results.append(entry)
        
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"完成！已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()