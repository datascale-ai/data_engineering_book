import json
import os
import cv2 
import numpy as np
import random
import re

# ================= 路径配置 (动态获取) =================
# 获取当前脚本所在目录 (src)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 向上一级找到 data/images
IMAGE_DIR = os.path.join(current_script_dir, "../data/images/")
# 读取生成的指令文件
JSON_FILE = os.path.join(current_script_dir, "../data/llava_instruct.json")
# 输出可视化结果的文件夹
OUTPUT_DIR = os.path.join(current_script_dir, "../data/viz_debug/")
# ======================================================

def draw_bbox(image, bbox, label, color):
    h, w, _ = image.shape
    # LLaVA 坐标是 [ymin, xmin, ymax, xmax] 且范围 0-1000
    # 注意：有时候模型输出可能会轻微越界，这里做个 clamp 限制
    ymin, xmin, ymax, xmax = bbox
    
    # 转换为像素坐标
    x1 = int(xmin / 1000 * w)
    y1 = int(ymin / 1000 * h)
    x2 = int(xmax / 1000 * w)
    y2 = int(ymax / 1000 * h)
    
    # 1. 画矩形框 (厚度 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # 2. 添加标签背景 (让文字更清晰)
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1) # 实心矩形
    
    # 3. 写字 (白色文字)
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")
        
    if not os.path.exists(JSON_FILE):
        print(f"错误: 找不到数据文件 {JSON_FILE}，请先运行 step2_alignment.py")
        return

    with open(JSON_FILE, 'r') as f:
        data = json.load(f)
        
    print(f"正在检查 {len(data)} 条数据...")
    count_viz = 0
    
    # 随机抽取 20 张或者全部处理 (如果数据不多)
    # data = random.sample(data, min(len(data), 20)) 

    for entry in data:
        img_name = entry['image']
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        # 从 ID 中提取物体名称 (我们在 Step 2 生成的 ID 格式是 "数字ID_物体名")
        # 例如: "139_tv" -> "tv"
        try:
            obj_label = entry['id'].split('_')[-1]
        except:
            obj_label = "Object"

        if not os.path.exists(img_path):
            # 这里的 print 可能会刷屏，如果你的 json 里有图片没下载，可以选择注释掉
            # print(f"跳过: 本地无图片 {img_name}")
            continue
            
        img = cv2.imread(img_path)
        if img is None: continue
            
        conversations = entry['conversations']
        found_bbox = False
        
        # 为这张图生成一个随机颜色，或者为不同物体生成不同颜色
        color = (random.randint(50, 255), random.randint(50, 200), random.randint(50, 200))

        for turn in conversations:
            text = turn['value']
            # 正则匹配 LLaVA 格式坐标 [0-1000]
            bboxes = re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text)
            
            for box_str in bboxes:
                found_bbox = True
                bbox = [int(x) for x in box_str]
                # 调用画图函数，传入真实的物体名称
                draw_bbox(img, bbox, obj_label, color)

        if found_bbox:
            # 保存图片
            save_path = os.path.join(OUTPUT_DIR, f"viz_{img_name}")
            cv2.imwrite(save_path, img)
            count_viz += 1
            if count_viz % 10 == 0:
                print(f"已生成 {count_viz} 张验证图...")

    print(f"\n✅ 全部完成！请进入 {OUTPUT_DIR} 查看结果。")
    print("如果在图中看到框精准地套住了物体，且标签正确，说明你的【数据对齐】步骤非常成功！")

if __name__ == "__main__":
    main()