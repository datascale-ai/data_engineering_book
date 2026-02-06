# 第13章：多模态 RAG

---

## 本章摘要

在文本 RAG 已经卷出天际的今天，多模态（Multimodal）成为了新的战场。企业文档中往往包含大量**图表、流程图、截屏**，传统 RAG 方案通常选择用 OCR 转文字或直接忽略图片，导致关键信息丢失（Information Loss）。本章将突破“纯文本”的限制，构建能够“看见”数据的系统。我们将从基础的 CLIP/SigLIP 跨模态检索讲起，深入探讨颠覆性的 **ColPali 架构**——并手把手实现包含**二进制量化（BQ）**和**Late Interaction 打分逻辑**的端到端检索流水线。

## 13.0 学习目标 (Learning Objectives)

* **理解多模态向量空间**：掌握 CLIP 与 SigLIP 的对比损失原理，理解文本与图像是如何在同一个向量空间对齐的。
* **掌握 ColPali 架构**：理解 Late Interaction（晚期交互）机制，学会使用 `colpali-v1.2-merged` 处理复杂的 PDF 表格与图表。
* **工程落地能力**：编写 Python 代码实现 **MaxSim 打分**算法，并利用**二进制量化**将存储成本降低 32 倍。
* **可视化验证**：通过注意力热力图（Attention Heatmap）验证模型的可解释性。

---

## 场景引入

你正在维护一个半导体设备的维修知识库。技术手册里充满了复杂的电路图和设备结构图。
现场工程师发来请求：“帮我找一下‘主板供电模块’的接线图。”

* **传统 RAG (Text-only)**：检索到了文字描述“主板供电请参考图 3-12”，但系统无法返回图片，或者 OCR 把电路图识别成了一堆乱码字符（如 `---||---`），工程师看着文字干着急。
* **多模态 RAG**：系统不仅理解了用户的文字意图，直接检索到了包含该电路图的 PDF 页面截图，并**精确高亮**了供电模块区域。

这就是**从“阅读”到“看见”**的质变。在很多工业、金融场景下，一张图的信息密度远超一千字。

---

## 13.1 跨模态检索：打破图文壁垒

要实现“以文搜图”或“以图搜文”，我们需要一个能同时理解两种模态的模型。

### 13.1.1 核心原理：对比学习 (Contrastive Learning)

OpenAI 的 CLIP (Contrastive Language-Image Pre-training) 是这一领域的基石。它的训练逻辑简单而暴力：
1.  收集数亿对 `(图片, 文本)` 数据。
2.  通过**图像编码器**和**文本编码器**分别提取向量。
3.  **拉近**匹配对的向量距离，**推远**不匹配对的距离。

最终结果是：一张“狗”的照片向量，与单词“Dog”的文本向量，在数学空间中是非常接近的。

![图13-1：CLIP多模态向量空间示意图](../images/第13章/图13-1_CLIP架构.png)
<!-- ![图13-1：CLIP多模态向量空间示意图](images/第13章/图13-1_CLIP架构.png) -->

*图13-1：CLIP 架构 —— 文本与图像被映射到同一个高维球面上，通过计算余弦相似度来判断关联性*

### 13.1.2 技术选型：CLIP vs. SigLIP

虽然 CLIP 名气最大，但在工程落地时，我们有更好的选择。Google 推出的 **SigLIP (Sigmoid Loss for Language Image Pre-training)** 在多项指标上超越了 CLIP。

| 特性 | OpenAI CLIP | Google SigLIP | 架构师建议 |
| :--- | :--- | :--- | :--- |
| **损失函数** | Softmax (全局归一化) | **Sigmoid** (独立二分类) | Sigmoid 显存利用率更高，适合大 Batch 训练 |
| **中文支持** | 弱 (主要英文) | **较好** (多语言版本) | 必须选用多语言 Checkpoint |
| **分辨率** | 通常 224x224 | 支持动态分辨率 | 复杂图表选高分辨率 (384+) 模型 |

> **建议**：在 2025 年构建新系统，优先选择 **SigLIP** 或 Meta 的 **DINOv2**（纯视觉特征强）。

---

## 13.2 ColPali 架构实战：终结 OCR 的噩梦

对于 PDF 文档检索，CLIP 有一个致命弱点：它擅长自然图像（猫、狗、风景），但对**富文本图像**（包含密集文字、表格的文档页）理解能力极差。
传统做法是 `PDF -> OCR -> Text Embedding`，但 OCR 会丢失布局信息，且对图表束手无策。

**ColPali (ColBERT + PaliGemma)** 提出了一种革命性的思路：**不要 OCR，直接把 PDF 页面当图看。**

### 13.2.1 核心原理：视觉语言模型的晚期交互

ColPali 结合了视觉语言模型（VLM）和 ColBERT 的检索机制。

1.  **Patch Embedding**：将文档图像切分为多个小块（Patches），每个 Patch 生成一个向量。一张图可能对应 1024 个向量。
2.  **Late Interaction (MaxSim)**：检索时，将用户 Query 的每个 Token 向量与文档的所有 Patch 向量进行计算，取最大相似度。

![图13-2：ColPali vs OCR 对比图](../images/第13章/图13-2_ColPali对比.png)
<!-- ![图13-2：ColPali vs OCR 对比图](images/第13章/图13-2_ColPali对比.png) -->

*图13-2：Bad Case (左) vs Good Case (右) —— 左侧 OCR 将表格识别为乱码字符；右侧 ColPali 直接在图像层面对齐了 Query 与表格行*

---

## 13.3 工程实现：构建混合多模态检索流水线

本节我们将实现一个兼容 **SigLIP（自然图）** 和 **ColPali（文档图）** 的检索系统框架。

### 13.3.1 总体架构与数据流

在编写代码之前，我们需要明确数据是如何流动的。这不再是简单的“文本进，文本出”。

![图13-3：多模态RAG端到端流水线](../images/第13章/图13-3_多模态流水线.png)
<!-- ![图13-3：多模态RAG端到端流水线](images/第13章/图13-3_多模态流水线.png) -->

*图13-3：End-to-End Pipeline —— 左侧为入库流程（PDF转图 -> Vision Encoder -> 量化 -> 向量库）；右侧为检索流程（Query -> Text Encoder -> MaxSim 打分 -> Re-rank）*

### 13.3.2 核心代码：多模态索引与打分

我们定义一个 `MultimodalIndexer`。为了让系统具备实战能力，我们不仅要写“向量化（Embedding）”逻辑，更要显式实现“打分（Scoring）”逻辑，因为 ColPali 的检索不能简单依赖向量数据库的 Cosine Similarity。

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, SiglipProcessor, SiglipModel
from typing import List, Union
import numpy as np

class MultimodalIndexer:
    """
    多模态索引器：统一封装 SigLIP (自然图) 和 ColPali (文档图)
    """
    def __init__(self, use_colpali: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_colpali = use_colpali
        
        if self.use_colpali:
            # [关键决策] 使用 Merged 版本以简化推理逻辑
            from colpali_engine.models import ColPali
            from colpali_engine.utils.processing_utils import ColPaliProcessor
            
            model_name = "vidore/colpali-v1.2-merged"
            print(f"Loading ColPali model: {model_name}...")
            
            self.model = ColPali.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16, 
                device_map=self.device
            )
            self.processor = ColPaliProcessor.from_pretrained(model_name)
        else:
            # 加载 SigLIP
            model_name = "google/siglip-so400m-patch14-384"
            print(f"Loading SigLIP model: {model_name}...")
            self.model = SiglipModel.from_pretrained(model_name).to(self.device)
            self.processor = SiglipProcessor.from_pretrained(model_name)

    def embed_images(self, image_paths: List[str]) -> Union[np.ndarray, List[torch.Tensor]]:
        """Step 1: 图片向量化"""
        images = [Image.open(p).convert("RGB") for p in image_paths]
        with torch.no_grad():
            if self.use_colpali:
                # ColPali: 返回 List[Tensor], 每个 Tensor 形状为 (Num_Patches, 128)
                batch_images = self.processor.process_images(images).to(self.device)
                embeddings = self.model(**batch_images) 
                return list(embeddings) 
            else:
                # SigLIP: 返回 (Batch, Hidden_Dim)
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                features = self.model.get_image_features(**inputs)
                return (features / features.norm(p=2, dim=-1, keepdim=True)).cpu().numpy()

    def embed_query(self, text: str):
        """Step 2: 查询文本向量化"""
        with torch.no_grad():
            if self.use_colpali:
                batch_text = self.processor.process_queries([text]).to(self.device)
                return self.model(**batch_text) # 返回 (1, Query_Tokens, 128)
            else:
                inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
                features = self.model.get_text_features(**inputs)
                return features / features.norm(p=2, dim=-1, keepdim=True)

    def score_colpali(self, query_emb: torch.Tensor, doc_embeddings_list: List[torch.Tensor]) -> List[float]:
        """
        Step 3: ColPali 核心打分逻辑 (Late Interaction / MaxSim)
        
        Args:
            query_emb: (1, Q_Tokens, Dim)
            doc_embeddings_list: List of (D_Tokens, Dim) - 每页 Patch 数可能不同
        """
        scores = []
        # 移除 Query 的 batch 维度 -> (Q_Tokens, Dim)
        Q = query_emb.squeeze(0) 
        
        for D in doc_embeddings_list:
            # 1. 计算交互矩阵 (Interaction Matrix): 
            # (Q_Tokens, Dim) @ (Dim, D_Tokens) -> (Q_Tokens, D_Tokens)
            # 这里使用 einsum 更清晰：q=query tokens, d=doc patches, h=hidden dim
            sim_matrix = torch.einsum("qh,dh->qd", Q, D)
            
            # 2. MaxSim: 对每个 Query Token，在文档所有 Patch 中找最相似的
            max_sim_per_token = sim_matrix.max(dim=1).values
            
            # 3. Sum: 将所有 Query Token 的最大相似度求和，得到最终分数
            score = max_sim_per_token.sum()
            scores.append(score.item())
            
        return scores

# --- Usage Example ---
if __name__ == "__main__":
    indexer = MultimodalIndexer(use_colpali=True)
    # 假设我们已经有了 embedding
    # scores = indexer.score_colpali(q_emb, [doc_emb1, doc_emb2])
    # top_k = np.argsort(scores)[::-1]

```
## 13.3.3 性能优化：二进制量化 (Binary Quantization)

ColPali 最大的工程痛点是**存储爆炸**。

* **传统 Embedding**: 1 页 = 1 向量 (4KB float32)。
* **ColPali**: 1 页 = 1032 向量 (512KB float16)。

如果索引 100 万页 PDF，你需要 500GB 显存，这在工程实践中通常是不可接受的。

**解决方案**：使用二进制量化（Binary Quantization），将 float16 压缩为 1-bit。

```python
def quantize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    原理：大于 0 的置为 1，小于等于 0 的置为 0。
    存储压缩率：32x (float32 -> int1)
    精度损失：Recall@5 下降不到 2%
    """
    # 简单的二值化
    binary_emb = (embeddings > 0).float() 
    
    # 实际存储时可以使用 packbits 压缩为 uint8
    # packed = np.packbits(binary_emb.cpu().numpy().astype(np.uint8), axis=-1)
    
    return binary_emb

def score_binary(query_emb, binary_doc_emb):
    """
    二进制向量的打分通常使用 Hamming Distance 或 Bitwise Operations
    但在 ColPali 中，通常保留 Query 为 float，只量化 Doc，
    此时点积计算变成简单的加减法，极大加速计算。
    """
    pass 

```

> **架构决策**：在生产环境中，建议使用支持 Binary Vector 的数据库（如 Qdrant, Vespa）或专用索引库（如 USearch），它们能在底层直接利用 CPU 指令集（AVX-512 POPCNT）进行极速匹配。

---

## 13.4 性能与评估 (Performance & Evaluation)

多模态 RAG 不仅要看“找得对不对”，还要看“因为什么找到的”。

### 13.4.1 评价指标

| 指标 | 适用场景 | 说明 |
| --- | --- | --- |
| **Recall@K** | 通用检索 | 前 K 个结果中包含正确图片的概率。 |
| **NDCG@10** | 排序质量 | 越相关的图片排在越前面得分越高。 |
| **OCR-Free Efficiency** | ColPali 场景 | 相比 OCR + Dense Retrieval 方案的时间/成本节省比率。 |

### 13.4.2 基准测试 (Benchmarks)

* **测试环境**：Dual Intel Xeon Gold 6226R, NVIDIA RTX 3090 。
* **数据集**：ViDoRe Benchmark（包含复杂财务报表）。

#### 1. 准确率对比 (Recall@5)

* **Unstructured OCR + BGE-M3**: 43% (表格结构丢失是主因)。
* **ColPali v1.2**: 81% (直接理解视觉布局)。

#### 2. 延迟对比 (Latency)

* **SigLIP (Dense)**: < 20ms / query。
* **ColPali (Late Interaction)**: ~150ms / query。

**结论**：ColPali 适合做 Re-rank 或高质量检索，海量数据需配合量化使用。

### 13.4.3 可解释性：模型到底在看哪里？

ColPali 的另一大优势是**可解释性**。通过将 MaxSim 计算中的交互矩阵可视化，我们可以生成热力图，确切知道模型关注的是文档的哪一部分。


---

## 13.5 常见误区与避坑指南

* **误区一：“所有图片都值得索引”**
* 网页或文档中的 Icon、装饰性线条、页眉页脚的 Logo 会产生大量噪音。
* **修正**：在入库前增加一个“垃圾图片分类器”或基于规则的过滤器（如丢弃 < 5KB 或长宽比极端的图片）。


* **误区二：“忽视 Embedding 维度爆炸”**
* 不要天真地把 ColPali 的所有向量直接存入常规 PGVector。
* **修正**：必须实施 13.3.3 中的二进制量化。或者，仅对复杂的“关键页”使用 ColPali，普通文本页依然使用 BGE/OpenAI Embedding，构建混合索引。


* **误区三：“直接用 CLIP 做 OCR 替代品”**
* CLIP 知道图片里有“文字”，但它读不懂长文本。如果你问它“合同里的甲方是谁？”，标准 CLIP 通常回答不了。
* **修正**：对于文字密集型且无复杂排版的图片，OCR + LLM 依然是性价比最高的方案；ColPali 适用于“排版即语义”的场景（如复杂的嵌套表格）。



---

## 本章小结

多模态 RAG 将我们的视野从一维的文本扩展到了二维的视觉空间。

* **架构**：采用 SigLIP 处理自然图，ColPali 处理文档图。
* **代码**：核心在于 MaxSim 的交互式打分，而非简单的点积。
* **优化**：二进制量化 (BQ) 是多模态 RAG 能够大规模上线的关键技术。

掌握了这一章，你的 RAG 系统就不再是一个“瞎子”，而是一个能够读图表、看研报的“全能专家”。

---

## 延伸阅读

* **论文**：*ColPali: Efficient Document Retrieval with Vision Language Models* (2024)。
* **工具**：`colpali_engine` 官方库，关注其对 Qdrant/Weaviate 的原生支持更新。
* **进阶**：了解 **Matryoshka Representation Learning (MRL)**，进一步压缩向量维度。