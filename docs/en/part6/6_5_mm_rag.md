# Project Five: Multimodal RAG Enterprise Financial Report Assistant

> **Scope**: Capstone Projects - Solving complex document (charts, tables) retrieval challenges

### 1. Project Background (Project Brief)

- **Task Definition:** Build an RAG system that can "understand" complex charts and data tables in enterprise annual reports, achieving in-depth Q&A on financial reports through visual retrieval (Visual Retrieval) and multimodal LLM (VLM).
- **Input and Output:**
  - **Input:** PDF-format enterprise annual financial reports (containing mixed-layout text, cross-page tables, trend line charts, pie charts, etc.).
  - **Output:** Natural language analysis answers based on chart data trends and specific values.
- **Challenge Analysis:** 
  1. **Structure loss**: Traditional RAG uses OCR to text, easily losing table row-column correspondence, and completely unable to handle trend charts without text captions.
  2. **Semantic fragmentation**: Reports often have "see figure below" references; text and charts separated cause retrieval truncation.
  3. **Retrieval noise**: Table of contents pages often contain keywords, easily causing false recall, crowding context window.

### 2. Architecture Design (Architecture Design)

This project's core concept is **"ViR (Vision in Retrieval) + VLM (Vision Language Model)."** We no longer force PDF to text; instead use **ColPali** to treat each PDF page as an image for visual encoding, directly retrieve visual features, then feed retrieved images to multimodal LLM for interpretation.

### Data Pipeline Diagram

![Figure 6: Multimodal RAG Enterprise Financial Report Assistant](../../images/part6/图6_多模态RAG企业财报助手数据流水线图.png)


### Technology Stack

| Component | Tool/Model | Selection Rationale |
| :--- | :--- | :--- |
| **Visual Retrieval Model** | **ColPali (v1.2)** | Current SOTA document retrieval model; based on PaliGemma; understands page layout, font size, chart visual features; no OCR needed. |
| **Index Framework** | **Byaldi** | ColPali lightweight wrapper; simplifies multimodal model tensor storage and retrieval flow. |
| **Multimodal LLM** | **Qwen2.5-VL-72B** | Alibaba Tongyi Qianwen latest vision model; excels at chart understanding (ChartQA) and document parsing (DocVQA). |


### 3. Step-by-Step Implementation

### Phase 1: Visual Index Building (Visual Indexing)

Unlike traditional RAG's `Chunking -> Embedding`, here we do `Page -> Screenshot -> Visual Embedding`.

**Key Code Logic (`index.py`):**

```python
from byaldi import RAGMultiModalModel

# 1. Load local ColPali model (solves HuggingFace connection issues)
MODEL_PATH = "/path/to/models/colpali-v1.2-merged"
INDEX_NAME = "finance_report_2024"

def build_index():
    # 2. Initialize model (supports load_in_4bit for memory reduction)
    RAG = RAGMultiModalModel.from_pretrained(MODEL_PATH, verbose=1)
    
    # 3. Build index
    # Principle: Byaldi converts PDF to images, computes visual vectors and stores
    RAG.index(
        input_path="annual_report_2024.pdf",
        index_name=INDEX_NAME,
        store_collection_with_index=True, # Must store original image references
        overwrite=True
    )
```

**Practical Notes:**
*   **Debug:** First run encountered OOM (out of memory).
*   **Solution:** ColPali full version needs ~10GB+ memory. When insufficient, add `load_in_4bit=True` to `from_pretrained`.

### Phase 2: Multi-Page Visual Retrieval (Multi-Page Retrieval)

A typical pitfall in financial report Q&A: **keyword "operating results" also appears on table of contents.** If only retrieving Top-1, might only get table of contents, causing model to fail answering. Therefore, strategy needs to retrieve Top-K (recommend 3-5 pages) and filter.

**Key Code Logic (`rag_chat.py` - Retrieval Part):**

```python
# Load index
RAG = RAGMultiModalModel.from_index(INDEX_NAME)

# Increase retrieval pages to prevent only hitting table of contents
RETRIEVAL_K = 4 

results = RAG.search(user_query, k=RETRIEVAL_K)

# Results contain: page_num (page number), base64 (image data), score (relevance)
```

### Phase 3: Multi-Image Context Generation (Multi-Image Generation)

We feed all K retrieved images as context to VLM, leveraging model's long window and multi-image capability for comprehensive analysis.

**Key Code Logic (`rag_chat.py` - Generation Part):**

```python
# Build multimodal payload
content_payload = []

# 1. System Prompt: Define role and anti-interference instructions
content_payload.append({
    "type": "text", 
    "text": f"You are a professional CFO assistant... may contain table of contents, please ignore table of contents, directly answer based on pages with specific data..."
})

# 2. Inject multiple images
for res in results:
    content_payload.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{res.base64}", 
            "detail": "high" # Enable high-res mode for small text recognition
        }
    })

# 3. Send to Qwen2.5-VL-72B
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-72B-Instruct",
    messages=[{"role": "user", "content": content_payload}],
    temperature=0.1 # Maintain factuality
)
```

### 4. Results Showcase (Showcase)

Below is a real execution record of the system on Huawei 2024 financial report. The system successfully skipped table of contents, comprehensively integrating information from pages 49, 91, 130, and 8 for quantitative analysis of R&D investment and intangible asset changes.

```text
>>> Please ask: How are the operating results?

🔍 Retrieving Top-4 relevant pages...
📄 Hit pages: [49] [91] [130] [8]
🚀 Sending to LLM for comprehensive analysis...

🤖 Financial Report Assistant Answer:
----------------------------------------
Based on the provided financial report screenshots, operating results can be analyzed from the following aspects:

### 1. **R&D and Innovation Investment**
- **R&D Expenses**: In 2024, R&D expenses were RMB 179.7 billion, accounting for 20.8% of annual revenue. Cumulative R&D investment over the past decade exceeded RMB 1,249 billion. This indicates continued substantial investment in R&D and innovation,committed to technology breakthrough and product upgrade.
- **R&D Personnel**: As of December 31, 2024, R&D employees numbered approximately 113,000, accounting for 54.1% of total employees, showing company emphasis on R&D team and sustained innovation investment.

### 2. **Intangible Assets**
- **Goodwill and Intangible Assets**: Screenshots show detailed data on goodwill, software, patents and royalties, trademark rights, and other intangible assets. From 2023 to 2024, goodwill increased from 4,424 to 4,496 million yuan... (abbreviated) ...indicating continued growth in intangible assets.

### 3. **Corporate Social Responsibility and Sustainable Development**
- **Tech Classroom Project**: Huawei officially launched "Tech Classroom" project... (abbreviated)
- **Mobile Digital Classroom Project**: Huawei partnered in Kenya for DigiTruck project...

### 4. **Chairman's Message**
- **Strategic Focus and Continued Innovation**: Chairman noted in message that 2024 was a challenging year; Huawei employees remained confident and proactive, achieving business development goals with overall operations meeting expectations...

In summary, Huawei's 2024 operating results were outstanding, with significant achievements in R&D and innovation, intangible assets, corporate social responsibility and sustainable development.
----------------------------------------
```

### 5. Cost and Optimization (Cost & Optimization)

- **Resource consumption:**
  - **Index cost:** ColPali processing is slow (~0.5s/page); indexing 200-page report takes 2-3 minutes.
  - **Inference cost:** Multimodal token consumption is huge. One 1024x1024 image ≈ 1000-1500 tokens. Each Top-4 retrieval means Input Token at least 5000+. Using SiliconFlow API for Qwen2.5-VL-72B, single Q&A cost ~0.05-0.1 RMB.

- **Optimization ideas:**
  1. **Precision optimization:** For ultra-high-resolution financial tables, can "crop" PDF pages before indexing—split one large image into 4 smaller images for separate indexing, improving local retrieval clarity.
  2. **Image cropping:** ColPali can locate relevant regions (Patch-level retrieval); future can crop only relevant "chart regions" from page to feed LLM, significantly reducing token consumption.
  3. **Caching mechanism:** For fixed high-frequency questions like "revenue amount" or "net profit amount," cache VLM parsing results to avoid repeated visual reasoning.
