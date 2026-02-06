# 项目五：多模态 RAG 企业财报助手

> **适用范围**：Capstone Projects - 解决复杂文档（图表、表格）检索难题

### 1. 项目背景 (Project Brief)

- **任务定义：** 构建一个能够"看懂"企业年报中复杂图表与数据表格的 RAG 系统，通过视觉检索（Visual Retrieval）和多模态大模型（VLM）实现对财报的深度问答。
- **输入与输出：**
  - **Input:** PDF 格式的企业年度财务报告（包含混合排版的文本、跨页表格、趋势折线图、饼图等）。
  - **Output:** 基于图表数据趋势和具体数值的自然语言分析回答。
- **难点分析：** 
  1. **结构丢失**：传统 RAG 使用 OCR 转文字，容易丢失表格的行列对应关系，且完全无法处理不带文字说明的趋势图。
  2. **语义断层**：财报中常出现"见下图"的指代，文本与图表分离导致检索截断。
  3. **检索噪音**：目录页（Table of Contents）常包含关键词，容易误召回，挤占上下文窗口。

### 2. 架构设计 (Architecture Design)

本项目的核心理念是 **"ViR (Vision in Retrieval) + VLM (Vision Language Model)"**。我们不再将 PDF 强行转为文本，而是利用 **ColPali** 将每一页 PDF 视为一张图片进行视觉编码，直接检索视觉特征，最后将命中的图片原图喂给多模态大模型进行解读。

### 数据流水线图

![图6：多模态RAG企业财报助手](../images/chapter6/图6_多模态RAG企业财报助手数据流水线图.png)


### 技术栈清单

| 组件 | 工具/模型 | 选择理由 |
| :--- | :--- | :--- |
| **视觉检索模型** | **ColPali (v1_2)** | 当前 SOTA 的文档检索模型，基于 PaliGemma，能理解页面布局、字体大小和图表视觉特征，无需 OCR。 |
| **索引框架** | **Byaldi** | ColPali 的轻量级封装库，简化了多模态模型的张量存储和检索流程。 |
| **多模态大模型** | **Qwen2_5-VL-72B** | 阿里通义千问最新视觉模型，在图表理解（ChartQA）和文档解析（DocVQA）任务上表现极佳。 |


### 3. Step-by-Step 实战 (Implementation)

### 阶段一：视觉索引构建 (Visual Indexing)

不同于传统 RAG 的 `Chunking -> Embedding`，这里我们进行的是 `Page -> Screenshot -> Visual Embedding`。

**关键代码逻辑 (`index.py`)：**

```python
from byaldi import RAGMultiModalModel

# 1. 加载本地 ColPali 模型 (解决 HuggingFace 连接问题)
MODEL_PATH = "/path/to/models/colpali-v1_2-merged"
INDEX_NAME = "finance_report_2024"

def build_index():
    # 2. 初始化模型 (支持 load_in_4bit 降低显存需求)
    RAG = RAGMultiModalModel.from_pretrained(MODEL_PATH, verbose=1)
    
    # 3. 建立索引
    # 原理：Byaldi 会将 PDF 转为图片，计算视觉向量并存储
    RAG.index(
        input_path="annual_report_2024.pdf",
        index_name=INDEX_NAME,
        store_collection_with_index=True, # 必须存储原图引用
        overwrite=True
    )
```

**实战复盘：**
*   **Debug:** 首次运行时遇到 OOM（显存溢出）。
*   **Solution:** ColPali 完整版需要约 10GB+ 显存。显存不足时，可在 `from_pretrained` 中添加 `load_in_4bit=True` 参数。

### 阶段二：多路视觉检索 (Multi-Page Retrieval)

财报问答的一个典型坑点是：**关键词"经营结果"在目录页也会出现**。如果只检索 Top-1，很可能只拿到目录，导致模型无法回答。因此，策略上需要检索 Top-K (建议 3-5 页) 并过滤。

**关键代码逻辑 (`rag_chat.py` - Retrieval Part)：**

```python
# 加载索引
RAG = RAGMultiModalModel.from_index(INDEX_NAME)

# 增加检索页数，防止只命中目录页
RETRIEVAL_K = 4 

results = RAG.search(user_query, k=RETRIEVAL_K)

# 结果包含：page_num (页码), base64 (图片数据), score (相关性)
```

### 阶段三：多图上下文生成 (Multi-Image Generation)

我们将检索到的 K 张图片全部作为上下文喂给 VLM，利用模型的长窗口和多图处理能力进行综合分析。

**关键代码逻辑 (`rag_chat.py` - Generation Part)：**

```python
# 构建多模态 Payload
content_payload = []

# 1. System Prompt: 明确角色与抗干扰指令
content_payload.append({
    "type": "text", 
    "text": f"你是一个专业的CFO助手...其中可能包含目录页，请忽略目录，直接根据包含具体数据的页面回答问题..."
})

# 2. 注入多张图片
for res in results:
    content_payload.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{res.base64}", 
            "detail": "high" # 开启高清模式以识别小字
        }
    })

# 3. 发送给 Qwen2_5-VL-72B
response = client.chat.completions.create(
    model="Qwen/Qwen2_5-VL-72B-Instruct",
    messages=[{"role": "user", "content": content_payload}],
    temperature=0_1 # 保持事实性
)
```

### 4. 效果展示 (Showcase)

以下是系统针对华为 2024 财报的真实运行记录。可以看到系统成功跳过了目录页，综合了第 49、91、130、8 页的信息，对研发投入、无形资产变化进行了量化分析。

```text
>>> 请提问: 经营结果如何？

🔍 正在检索 Top-4 个相关页面...
📄 命中页码: [49] [91] [130] [8]
🚀 正在发送给大模型进行综合分析...

🤖 财报助手回答:
----------------------------------------
从提供的财报截图来看，经营结果可以从以下几个方面进行分析：

### 1. **研发与创新投入**
- **研发费用**：2024年，研发费用支出为人民币1,797亿元，占全年收入的20_8%。近十年累计投入的研发费用超过人民币12,490亿元。这表明公司在研发和创新方面持续投入大量资源，致力于技术的突破和产品的升级。
- **研发人员**：截至2024年12月31日，研发员工约11_3万名，占总员工数量的54_1%，显示出公司对研发团队的重视和对技术创新的持续投入。

### 2. **无形资产**
- **商誉及无形资产**：截图显示了商誉、软件、专利权及特许权使用费、商标使用权及其他无形资产的详细数据。从2023年到2024年，商誉从4,424百万元增加到4,496百万元...（中略）...这表明公司在无形资产方面持续增长。

### 3. **社会责任与可持续发展**
- **科技小学堂项目**：华为正式发起"科技小学堂"项目...（中略）
- **移动数字课堂项目**：华为携手伙伴在肯尼亚发起DigiTruck项目...

### 4. **董事长致辞**
- **战略聚焦与持续创新**：董事长在致辞中提到，2024年是充满挑战的一年，华为全体员工坚定信心、积极奋进，实现业务发展目标，整体经营达到预期...

综上所述，华为在2024年的经营结果表现出色，公司在研发与创新、无形资产、社会责任与可持续发展等方面均取得了显著成就。
----------------------------------------
```

### 5. 成本与优化 (Cost & Optimization)

- **资源消耗：**
  - **索引成本**：ColPali 处理速度较慢（约 0_5s/页），一份 200 页的财报索引需 2-3 分钟。
  - **推理成本**：多模态 Token 消耗巨大。一张 1024x1024 的图片约为 1000-1500 tokens。每次 Top-4 检索意味着 Input Token 至少 5000+。使用 SiliconFlow API 调用 Qwen2_5-VL-72B，单次问答成本约 0_05-0_1 元人民币。

- **优化思路：**
  1. **精度优化**： 对于超大分辨率的财务大表，可以在索引前对 PDF 页面进行"切片（Cropping）"处理，将一张大图切成 4 张小图分别建立索引，提高局部检索的清晰度。
  2. **图片裁剪**：ColPali 能够定位相关区域（Patch-level retrieval），未来可只将页面中相关的"图表区域"裁剪出来喂给大模型，大幅降低 Token 消耗。
  3. **缓存机制**：对于"营收多少"、"净利润多少"等高频固定问题，将 VLM 的解析结果缓存，避免重复进行视觉推理。
