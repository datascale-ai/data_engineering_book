# 项目三：构建 LLaVA 多模态指令集

> **适用范围**：多模态大模型（LMM）开发、数据工程、视觉指令微调（Visual Instruction Tuning）

### 1. 项目背景 (Project Brief)

- **任务定义：**
  构建一个高质量的视觉指令微调数据集，支持单图问答（Visual QA）、物体定位（Grounding）以及多图上下文推理（Interleaved Image-Text），用于训练像 LLaVA 或 Qwen-VL 这样的多模态模型。

- **输入与输出：**
  - **Input:** 
    - 原始图片库 (`.jpg` / `.png`)
    - 结构化标注数据（如 COCO 格式的 `instances.json`，包含 Bbox 坐标）
  - **Output:** 
    - 符合 LLaVA 训练标准的 JSON 文件（包含 `image`, `conversations` 字段）。
    - 经过坐标归一化和格式对齐的 Grounding 数据。

- **难点分析：**
  1.  **坐标系对齐（Coordinate Alignment）：** 原始检测数据的坐标通常是像素绝对值（x, y, w, h），而 LLaVA 模型要求归一化到 `[0-1000]` 区间且顺序为 `[ymin, xmin, ymax, xmax]`，一旦算错，模型将出现严重的"幻觉"。
  2.  **多图逻辑构建：** 传统的 Image-Caption 数据是一图一文，构建"多图交错"对话需要构造合理的对比性 Prompt，诱导模型理解图像间的关联。

### 2. 架构设计 (Architecture Design)

- **数据流水线图：**
![图3：构建LLaVA多模态](../images/chapter6/图3_构建LLaVA多模态指令集数据流水线图.png)



- **技术栈清单：**
  - **OpenAI Compatible API (SiliconFlow/Qwen):** 用于生成高质量的图文描述和多图对比逻辑，利用大模型的 Reasoning 能力构造对话。
  - **Python & OpenCV:** 核心胶水语言。OpenCV 必不可少，用于读取图像真实尺寸（H, W）以进行坐标归一化，并用于可视化的"画框验证"。
  - **JSON:** LLaVA 标准数据交换格式。

### 3. Step-by-Step 实战 (Implementation)

#### 阶段一：多图交错数据生成 (Interleaved Data Generation)
为了让模型学会"对比"两张图片，我们利用 API 动态输入多张图像并请求对比。

**关键逻辑：** 利用 VLM API 构造多图输入的 Prompt。

```python
# 摘自 interleaved.py
def generate_comparison(img1_path, img2_path):
    # 构造 Prompt：要求多图对比
    prompt = "Here are two images. Please briefly compare them..."
    
    # 构建多图 Payload
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"...{img1_path}..."}}, # 图1
                {"type": "image_url", "image_url": {"url": f"...{img2_path}..."}}  # 图2
            ]
        }
    ]
    # ... 发送请求并解析结果 ...
```

#### 阶段二：核心处理——Bounding Box 对齐 (Alignment)
这是本项目最核心的数学部分。COCO 数据集使用 `[x_topleft, y_topleft, width, height]`，而 LLaVA 需要 `[ymin, xmin, ymax, xmax]` 且数值需归一化为 0-1000 的整数。

**关键函数：** 坐标归一化转换

```python
# 摘自 alignment.py
def convert_bbox(bbox, width, height):
    # COCO 原始输入: x, y, w, h
    x, y, w, h = bbox
    
    # 转换为 LLaVA 格式: [ymin, xmin, ymax, xmax] 并归一化到 0-1000
    # 必须使用 max/min 截断，防止浮点误差导致越界
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
```

#### 阶段三：格式化与验证 (Verification)
数据生成后，绝不能直接送入训练。必须通过**可视化反向验证**。如果我们在图片上画出的框是歪的，训练出来的模型一定是废的。

**验证逻辑：** 解析生成的 JSON，将 `[0-1000]` 坐标还原回像素坐标并绘图。

```python
# 摘自 visualize_bbox.py
def draw_bbox(image, bbox, label, color):
    h, w, _ = image.shape
    ymin, xmin, ymax, xmax = bbox # 读取 LLaVA 格式
    
    # 还原为像素坐标用于画图
    x1 = int(xmin / 1000 * w)
    y1 = int(ymin / 1000 * h)
    x2 = int(xmax / 1000 * w)
    y2 = int(ymax / 1000 * h)
    
    # OpenCV 画框
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    # ...
```

### 4. 效果展示 (Showcase)

**1. 数据结构示例：**
最终生成的 `llava_instruct.json` 呈现如下标准结构，可以直接被 Training Pipeline 读取：

```json
{
  "id": "1296_laptop",
  "image": "000000001296.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "Where is the laptop in the image? <image>"
    },
    {
      "from": "qwen",
      "value": "The laptop is located at [350, 201, 680, 505]."
    }
  ]
}
```

**2. 可视化验证报告：**
运行 `visualize_bbox.py` 后，在 `viz_debug` 目录下生成的验证图。如果框精准地套住了物体（如下图所示），说明数据流水线逻辑正确。

 **效果图生成 **

![图4：效果图生成](../images/chapter6/图4_viz_000000001490.jpg)


### 5. 成本与优化 (Cost & Optimization)

- **资源消耗：**
  - **API 成本：** `interleaved.py` 依赖外部 LLM API。生成 10,000 条多图对比数据，按照 $0_5/1M Tokens 计算，成本约为 $20-$30。
  - **计算耗时：** `alignment.py` 是纯 CPU 计算，处理 COCO 验证集（5k 张图）仅需数秒。

- **扩展性思考：**
  - **并发处理：** 当处理百万级图像（如 Objects365）时，单线程读取图片获取 `(h, w)` 会成为瓶颈。可以引入 `multiprocessing` 库，开启 16 个进程并行读取和转换。
  - **负样本挖掘：** 当前代码只生成了"物体在哪里"的正样本。为了增强模型鲁棒性，需要扩展代码生成"图片里有大象吗？-> No"这类负样本数据。
