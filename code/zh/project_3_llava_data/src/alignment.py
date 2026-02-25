import os
import json
import glob
from tqdm import tqdm

# --- 配置路径 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(CURRENT_DIR, "../data/images/")
ANNOTATION_FILE = os.path.join(CURRENT_DIR, "../data/annotations/instances_val2017.json")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "../data/llava_instruct.json")
# ---------------

def load_coco_annotations():
    print("正在加载 COCO 标注文件 (可能需要几秒钟)...")
    with open(ANNOTATION_FILE, 'r') as f:
        coco = json.load(f)
    
    # 构建快速查找字典: image_id -> [annotations]
    # 和 category_id -> name
    img_to_anns = {}
    cat_to_name = {cat['id']: cat['name'] for cat in coco['categories']}
    
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        
        # 转换 bbox: COCO [x,y,w,h] -> LLaVA [ymin, xmin, ymax, xmax] (0-1000)
        # 注意：这里我们暂时存原始数据，处理图片时再归一化
        img_to_anns[img_id].append({
            "bbox": ann['bbox'], # [x,y,w,h]
            "label": cat_to_name.get(ann['category_id'], "object")
        })
    
    return img_to_anns

def convert_bbox(bbox, width, height):
    # COCO: x_topleft, y_topleft, w, h
    x, y, w, h = bbox
    
    # 转换为 LLaVA: [ymin, xmin, ymax, xmax] 并归一化到 0-1000
    xmin = int((x / width) * 1000)
    ymin = int((y / height) * 1000)
    xmax = int((x + w) / width * 1000)
    ymax = int((y + h) / height * 1000)
    
    return [
        max(0, min(1000, ymin)),
        max(0, min(1000, xmin)),
        max(0, min(1000, ymax)),
        max(0, min(1000, xmax))
    ]

def main():
    import cv2 # 用于获取图片尺寸
    
    if not os.path.exists(ANNOTATION_FILE):
        print(f"错误：找不到标注文件 {ANNOTATION_FILE}")
        return

    img_to_anns = load_coco_annotations()
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    
    dataset = []
    print(f"正在处理 {len(image_paths)} 张图片并进行数据对齐...")

    for img_path in tqdm(image_paths):
        fname = os.path.basename(img_path)
        
        # 1. 从文件名提取 image_id (例如 000000001296.jpg -> 1296)
        try:
            image_id = int(fname.split('.')[0])
        except ValueError:
            continue # 不是 COCO 格式的文件名，跳过
            
        # 2. 获取该图的标注
        anns = img_to_anns.get(image_id, [])
        if not anns:
            continue # 这张图没有标注
            
        # 3. 读取图片获取宽高
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape
        
        # 4. 构建对话数据
        # 策略：挑选一个物体，构建 Detection 问答
        # 这是最标准的 LLaVA Alignment Data 格式
        
        # 随机选最多 3 个物体来构建问答
        selected_anns = anns[:3] 
        
        for ann in selected_anns:
            label = ann['label']
            llava_box = convert_bbox(ann['bbox'], w, h)
            
            # 构造一条指令数据
            entry = {
                "id": f"{image_id}_{label}",
                "image": fname,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Where is the {label} in the image? <image>"
                    },
                    {
                        "from": "qwen",
                        "value": f"The {label} is located at {llava_box}."
                    }
                ]
            }
            dataset.append(entry)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"成功！已生成 {len(dataset)} 条精准对齐的指令数据。")
    print(f"文件位置: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()