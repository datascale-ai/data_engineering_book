# Project Three: Building LLaVA Multimodal Instruction Dataset

> **Scope**: Multimodal LLM (LMM) development, data engineering, visual instruction tuning (Visual Instruction Tuning)

#### 1. Project Background (Project Brief)

- **Task Definition:**
  Build a high-quality visual instruction fine-tuning dataset supporting single-image QA (Visual QA), object localization (Grounding), and multi-image context reasoning (Interleaved Image-Text), for training multimodal models like LLaVA or Qwen-VL.

- **Input and Output:**
  - **Input:** 
    - Raw image library (`.jpg` / `.png`)
    - Structured annotation data (e.g., COCO-format `instances.json` with Bbox coordinates)
  - **Output:** 
    - JSON files conforming to LLaVA training standard (with `image`, `conversations` fields).
    - Grounding data with coordinate normalization and format alignment.

- **Challenge Analysis:**
  1.  **Coordinate alignment (Coordinate Alignment):** Raw detection data coordinates are typically pixel absolutes (x, y, w, h), while LLaVA requires normalization to `[0-1000]` range with order `[ymin, xmin, ymax, xmax]`—once wrong, model suffers severe "hallucination."
  2.  **Multi-image logic construction:** Traditional Image-Caption data is one image one text; building "multi-image interleaved" dialogue requires constructing reasonable comparative Prompts to induce model to understand inter-image relationships.

#### 2. Architecture Design (Architecture Design)

- **Data Pipeline Diagram:**
![Figure 3: Building LLaVA Multimodal Instruction Dataset](../../images/part6/图3_构建LLaVA多模态指令集数据流水线图.png)



- **Technology Stack:**
  - **OpenAI Compatible API (SiliconFlow/Qwen):** For generating high-quality image-text descriptions and multi-image comparison logic; leverages LLM reasoning for dialogue construction.
  - **Python & OpenCV:** Core glue language. OpenCV essential for reading image dimensions (H, W) for coordinate normalization and "draw-box verification" visualization.
  - **JSON:** LLaVA standard data exchange format.

#### 3. Step-by-Step Implementation

##### Phase 1: Multi-Image Interleaved Data Generation

To teach the model to "compare" two images, we use API to dynamically input multiple images and request comparison.

**Key Logic:** Use VLM API to construct multi-image input Prompt.

```python
# From interleaved.py
def generate_comparison(img1_path, img2_path):
    # Construct Prompt: Require multi-image comparison
    prompt = "Here are two images. Please briefly compare them..."
    
    # Build multi-image payload
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"...{img1_path}..."}}, # Image 1
                {"type": "image_url", "image_url": {"url": f"...{img2_path}..."}}  # Image 2
            ]
        }
    ]
    # ... send request and parse result ...
```

##### Phase 2: Core Processing—Bounding Box Alignment

This is the project's core math. COCO uses `[x_topleft, y_topleft, width, height]` while LLaVA needs `[ymin, xmin, ymax, xmax]` with values normalized to 0-1000 integers.

**Key Function:** Coordinate normalization conversion

```python
# From alignment.py
def convert_bbox(bbox, width, height):
    # COCO raw input: x, y, w, h
    x, y, w, h = bbox
    
    # Convert to LLaVA format: [ymin, xmin, ymax, xmax] normalized to 0-1000
    # Must use max/min for clipping to prevent float error overflow
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

##### Phase 3: Formatting and Verification

Data generation must not go directly to training. Must pass **visualization reverse verification**. If our drawn boxes are wrong, the trained model will be useless.

**Verification Logic:** Parse generated JSON, restore `[0-1000]` coordinates to pixel coordinates and draw.

```python
# From visualize_bbox.py
def draw_bbox(image, bbox, label, color):
    h, w, _ = image.shape
    ymin, xmin, ymax, xmax = bbox # Read LLaVA format
    
    # Restore to pixel coordinates for drawing
    x1 = int(xmin / 1000 * w)
    y1 = int(ymin / 1000 * h)
    x2 = int(xmax / 1000 * w)
    y2 = int(ymax / 1000 * h)
    
    # OpenCV draw box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    # ...
```

#### 4. Results Showcase (Showcase)

**1. Data Structure Example:**
Final generated `llava_instruct.json` has this standard structure, directly readable by Training Pipeline:

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

**2. Visualization Verification Report:**
After running `visualize_bbox.py`, verification images generated in `viz_debug` directory. If boxes precisely frame objects (as shown below), data pipeline logic is correct.

 **Effect Image Generation:**

![Figure 4: Effect Image](../../images/part6/图4_viz_000000001490.jpg)


#### 5. Cost and Optimization (Cost & Optimization)

- **Resource consumption:**
  - **API cost:** `interleaved.py` depends on external LLM API. Generating 10,000 multi-image comparison samples at $0.5/1M Tokens costs ~$20-$30.
  - **Compute time:** `alignment.py` is pure CPU; processing COCO validation set (5k images) takes seconds.

- **Scaling considerations:**
  - **Concurrent processing:** When processing millions of images (e.g., Objects365), single-threaded image reading for `(h, w)` becomes bottleneck. Can introduce `multiprocessing` for 16 processes parallel read and convert.
  - **Negative sample mining:** Current code only generates "where is object" positive samples. For model robustness, extend code to generate negative samples like "Is there an elephant in the image? -> No."
