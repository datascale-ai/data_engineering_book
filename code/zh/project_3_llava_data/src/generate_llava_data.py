import os
import json
import base64
import glob
import re
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
# 硅基流动 API 配置
API_KEY = "sk-lrdpxzsnhsbckhjrzekbrtccomruhcwzyrlwbroqwojtwtsw"
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"

# 数据路径配置
IMAGE_DIR = "/home/xuxin123/book/project_3_llava_data/data/images/"
OUTPUT_FILE = "/home/xuxin123/book/project_3_llava_data/data/llava_instruct.json"
# ===========================================

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def encode_image(image_path):
    """将本地图片读取并转换为 Base64 编码"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"读取图片失败 {image_path}: {e}")
        return None

def clean_json_string(content):
    """
    清洗模型返回的字符串，去除 Markdown 代码块标记，提取纯 JSON
    """
    try:
        # 移除 ```json 和 ``` 标记
        content = content.strip()
        if content.startswith("```"):
            # 使用正则提取代码块内的内容
            match = re.search(r"```(?:json)?\s*(.*)\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)
        return content
    except Exception as e:
        return content

def generate_llava_entry(image_path):
    """
    调用 Qwen2.5-VL 生成符合 LLaVA 格式的图文指令数据
    """
    base64_img = encode_image(image_path)
    if not base64_img:
        return None

    filename = os.path.basename(image_path)

    # 核心 Prompt：加强了对纯文本 JSON 的要求
    system_prompt = """
    你是一个构建多模态数据集的专家。请分析传入的图片，并生成符合 LLaVA 训练格式的数据。
    要求：
    1. 识别图片中的主要物体，并提供归一化坐标 [ymin, xmin, ymax, xmax] (范围 0-1000)。
    2. 生成两轮对话：
       - 第一轮：用户询问图片内容，助手详细描述（包含 Bbox）。
       - 第二轮：用户根据图片细节提问（如推理、计数、关系），助手回答。
    3. 重要：直接返回 JSON 字符串，不要使用 Markdown 代码块，不要包含任何解释性文字。
    """

    user_prompt = f"""
    请为图片 "{filename}" 生成数据。
    返回格式示例：
    {{
        "conversations": [
            {{ "from": "human", "value": "<image>\\n请详细描述这张图片。" }},
            {{ "from": "qwen", "value": "图片中有一只猫 [200, 300, 500, 600] 趴在沙发上..." }},
            {{ "from": "human", "value": "猫在看哪里？" }},
            {{ "from": "qwen", "value": "它正盯着窗外的鸟..." }}
        ]
    }}
    """

    try:
        # --- 修改点：移除了 response_format 参数 ---
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                        }
                    ]
                }
            ],
            temperature=0.7, 
        )
        
        # 获取内容并清洗
        raw_content = response.choices[0].message.content
        cleaned_content = clean_json_string(raw_content)
        
        # 尝试解析 JSON
        try:
            data = json.loads(cleaned_content)
        except json.JSONDecodeError:
            print(f"警告: {filename} 解析 JSON 失败，模型返回了非标准格式。")
            return None
        
        # 构造最终的 LLaVA 数据条目
        entry = {
            "id": filename,
            "image": filename, 
            "conversations": data.get("conversations", [])
        }
        
        if not entry["conversations"]:
            return None
            
        return entry

    except Exception as e:
        print(f"API 调用错误 ({filename}): {e}")
        return None

def main():
    # 1. 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 2. 扫描所有图片
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    if not image_files:
        print(f"错误: 在 {IMAGE_DIR} 下没有找到图片！请检查路径。")
        return

    print(f"找到 {len(image_files)} 张图片，开始使用 {MODEL_NAME} 生成数据...")
    
    final_dataset = []
    
    # 3. 循环处理
    for img_path in tqdm(image_files, desc="Processing"):
        entry = generate_llava_entry(img_path)
        if entry:
            final_dataset.append(entry)
            
            # 实时保存
            if len(final_dataset) % 5 == 0:
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(final_dataset, f, ensure_ascii=False, indent=2)

    # 4. 最终保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n========================================")
    print(f"处理完成！")
    print(f"成功生成: {len(final_dataset)} 条数据")
    print(f"保存路径: {OUTPUT_FILE}")
    print(f"========================================")

if __name__ == "__main__":
    main()