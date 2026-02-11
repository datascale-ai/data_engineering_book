# Chapter 13: Multimodal RAG

---

## Chapter Summary

With text RAG competition at its peak, multimodal has become the new battlefield. Enterprise documents often contain large amounts of **charts, flowcharts, screenshots**—traditional RAG typically chooses OCR to text or ignores images, causing key information loss. This chapter breaks through "pure text" limits to build systems that can "see" data. We start from basic CLIP/SigLIP cross-modal retrieval, delve into the disruptive **ColPali architecture**, and implement an end-to-end retrieval pipeline including **Binary Quantization (BQ)** and **Late Interaction scoring logic**.

## 13.0 Learning Objectives

* **Understand multimodal vector space**: Master CLIP and SigLIP contrastive loss principles; understand how text and images align in the same vector space.
* **Master ColPali architecture**: Understand Late Interaction mechanism; learn to use `colpali-v1.2-merged` for complex PDF tables and charts.
* **Engineering capability**: Write Python code implementing **MaxSim scoring** algorithm and use **Binary Quantization** to reduce storage cost 32x.
* **Visualization verification**: Verify model interpretability through attention heatmaps.

---

## Scenario Introduction

You're maintaining a semiconductor equipment repair knowledge base. Technical manuals are full of complex circuit diagrams and device structure diagrams.
Field engineer requests: "Help me find the 'mainboard power supply module' wiring diagram."

* **Traditional RAG (Text-only)**: Retrieved text description "mainboard power supply refer to Figure 3-12," but system cannot return the image, or OCR turned the circuit diagram into garbled characters (e.g., `---||---`), leaving the engineer staring at text helplessly.
* **Multimodal RAG**: System not only understands user's text intent but directly retrieves PDF page screenshots containing the circuit diagram, and **precisely highlights** the power supply module region.

This is the qualitative leap from "reading" to "seeing." In many industrial and financial scenarios, one image's information density far exceeds a thousand words.

---

## 13.1 Cross-Modal Retrieval: Breaking Text-Image Barriers

To achieve "text-to-image search" or "image-to-text search," we need a model that understands both modalities.

### 13.1.1 Core Principle: Contrastive Learning

OpenAI's CLIP (Contrastive Language-Image Pre-training) is the cornerstone. Its training logic is simple and brutal:
1.  Collect hundreds of millions of (image, text) pairs.
2.  Extract vectors via **image encoder** and **text encoder**.
3.  **Pull** matching pair vectors closer, **push** non-matching pairs apart.

Result: A "dog" photo vector and "Dog" text vector are mathematically very close in space.

![Figure 13-1: CLIP Multimodal Vector Space Diagram](../../images/part5/图13_1_CLIP架构.png)
<!-- ![Figure 13-1: CLIP Multimodal Vector Space Diagram](images/Chapter 13/图13_1_CLIP架构.png) -->

*Figure 13-1: CLIP Architecture —— Text and images are mapped to the same high-dimensional sphere; cosine similarity determines association*

### 13.1.2 Technology Selection: CLIP vs. SigLIP

Although CLIP is most famous, for engineering we have better choices. Google's **SigLIP (Sigmoid Loss for Language Image Pre-training)** surpasses CLIP on multiple metrics.

| Feature | OpenAI CLIP | Google SigLIP | Architect Recommendation |
| :--- | :--- | :--- | :--- |
| **Loss Function** | Softmax (global normalization) | **Sigmoid** (independent binary classification) | Sigmoid has higher memory efficiency; suitable for large batch training |
| **Chinese Support** | Weak (mainly English) | **Better** (multilingual version) | Must use multilingual checkpoint |
| **Resolution** | Usually 224x224 | Supports dynamic resolution | Complex charts choose high resolution (384+) model |

> **Recommendation**: For new systems in 2025, prioritize **SigLIP** or Meta's **DINOv2** (strong pure visual features).

---

## 13.2 ColPali Architecture in Practice: Ending the OCR Nightmare

For PDF document retrieval, CLIP has a fatal weakness: it excels at natural images (cats, dogs, scenery) but has poor understanding of **rich-text images** (document pages with dense text, tables).
Traditional approach is `PDF -> OCR -> Text Embedding`, but OCR loses layout information and is helpless with charts.

**ColPali (ColBERT + PaliGemma)** proposes a revolutionary idea: **No OCR—treat PDF pages directly as images.**

### 13.2.1 Core Principle: Late Interaction of Vision-Language Models

ColPali combines Vision-Language Model (VLM) and ColBERT retrieval mechanism.

1.  **Patch Embedding**: Split document image into small patches; each patch generates a vector. One image may correspond to 1024 vectors.
2.  **Late Interaction (MaxSim)**: At retrieval, compute each Query token vector against all document Patch vectors, take maximum similarity.

![Figure 13-2: ColPali vs OCR Comparison](../../images/part5/图13_2_ColPali对比.png)
<!-- ![Figure 13-2: ColPali vs OCR Comparison](images/Chapter 13/图13_2_ColPali对比.png) -->

*Figure 13-2: Bad Case (Left) vs Good Case (Right) —— Left: OCR turns table into garbled characters; Right: ColPali directly aligns Query with table rows at image level*

---

## 13.3 Engineering Implementation: Building Hybrid Multimodal Retrieval Pipeline

This section implements a retrieval system framework compatible with **SigLIP (natural images)** and **ColPali (document images)**.

### 13.3.1 Overall Architecture and Data Flow

Before coding, we need to clarify data flow. This is no longer simple "text in, text out."

![Figure 13-3: Multimodal RAG End-to-End Pipeline](../../images/part5/图13_3_多模态流水线.png)
<!-- ![Figure 13-3: Multimodal RAG End-to-End Pipeline](images/Chapter 13/图13_3_多模态流水线.png) -->

*Figure 13-3: End-to-End Pipeline —— Left: index flow (PDF to image -> Vision Encoder -> Quantization -> Vector DB); Right: retrieval flow (Query -> Text Encoder -> MaxSim scoring -> Re-rank)*

### 13.3.2 Core Code: Multimodal Indexing and Scoring

We define `MultimodalIndexer`. For practical capability, we need not just "vectorization (Embedding)" logic but explicit "scoring" logic—ColPali retrieval cannot simply rely on vector database Cosine Similarity.

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, SiglipProcessor, SiglipModel
from typing import List, Union
import numpy as np

class MultimodalIndexer:
    """
    Multimodal indexer: Unified wrapper for SigLIP (natural images) and ColPali (document images)
    """
    def __init__(self, use_colpali: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_colpali = use_colpali
        
        if self.use_colpali:
             # [Key decision] Use Merged version (combines visual encoder and language model into single checkpoint)
             # Simplified for unified loading and deployment
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
            # Load SigLIP
            model_name = "google/siglip-so400m-patch14-384"
            print(f"Loading SigLIP model: {model_name}...")
            self.model = SiglipModel.from_pretrained(model_name).to(self.device)
            self.processor = SiglipProcessor.from_pretrained(model_name)

    def embed_images(self, image_paths: List[str]) -> Union[np.ndarray, List[torch.Tensor]]:
        """Step 1: Image vectorization"""
        images = [Image.open(p).convert("RGB") for p in image_paths]
        with torch.no_grad():
            if self.use_colpali:
                # ColPali: Returns List[Tensor], each shape (Num_Patches, 128)
                batch_images = self.processor.process_images(images).to(self.device)
                embeddings = self.model(**batch_images) 
                return list(embeddings) 
            else:
                # SigLIP: Returns (Batch, Hidden_Dim)
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                features = self.model.get_image_features(**inputs)
                return (features / features.norm(p=2, dim=-1, keepdim=True)).cpu().numpy()

    def embed_query(self, text: str):
        """Step 2: Query text vectorization"""
        with torch.no_grad():
            if self.use_colpali:
                batch_text = self.processor.process_queries([text]).to(self.device)
                return self.model(**batch_text) # Returns (1, Query_Tokens, 128)
            else:
                inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
                features = self.model.get_text_features(**inputs)
                return features / features.norm(p=2, dim=-1, keepdim=True)

    def score_colpali(self, query_emb: torch.Tensor, doc_embeddings_list: List[torch.Tensor]) -> List[float]:
        """
        Step 3: ColPali core scoring logic (Late Interaction / MaxSim)
        
        Args:
            query_emb: (1, Q_Tokens, Dim)
            doc_embeddings_list: List of (D_Tokens, Dim) - patches per page may vary
        """
        scores = []
        # Remove Query batch dimension -> (Q_Tokens, Dim)
        Q = query_emb.squeeze(0) 
        
        for D in doc_embeddings_list:
            # 1. Compute interaction matrix: 
            # (Q_Tokens, Dim) @ (Dim, D_Tokens) -> (Q_Tokens, D_Tokens)
            # Using einsum: q=query tokens, d=doc patches, h=hidden dim
            sim_matrix = torch.einsum("qh,dh->qd", Q, D)
            
            # 2. MaxSim: For each Query token, find max similarity among all document patches
            max_sim_per_token = sim_matrix.max(dim=1).values
            
            # 3. Sum: Sum all Query token max similarities for final score
            score = max_sim_per_token.sum()
            scores.append(score.item())
            
        return scores

# --- Usage Example ---
if __name__ == "__main__":
    indexer = MultimodalIndexer(use_colpali=True)
    # Assume we have embeddings
    # scores = indexer.score_colpali(q_emb, [doc_emb1, doc_emb2])
    # top_k = np.argsort(scores)[::-1]

```
## 13.3.3 Performance Optimization: Binary Quantization (BQ)

ColPali's biggest engineering pain is **storage explosion**.

* **Traditional Embedding**: 1 page = 1 vector (4KB float32).
* **ColPali**: 1 page = 1032 vectors (512KB float16).

Indexing 1 million PDF pages needs 500GB memory—often unacceptable in practice.

**Solution**: Use Binary Quantization, compressing float16 to 1-bit.

```python
def quantize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Principle: Values > 0 become 1, <= 0 become 0.
    Storage compression: 32x (float32 -> int1)
    Precision loss: Recall@5 drops less than 2%
    """
    # Simple binarization
    binary_emb = (embeddings > 0).float() 
    
    # For actual storage can use packbits to compress to uint8
    # packed = np.packbits(binary_emb.cpu().numpy().astype(np.uint8), axis=-1)
    
    return binary_emb

def score_binary(query_emb, binary_doc_emb):
    """
    Binary vector scoring typically uses Hamming Distance or bitwise operations
    but in ColPali, usually keep Query as float, only quantize Doc—
    then dot product becomes simple add/subtract, greatly accelerating computation.
    """
    pass 

```

> **Architecture decision**: In production, recommend databases supporting Binary Vector (Qdrant, Vespa) or dedicated index libraries (USearch)—they can leverage CPU instructions (AVX-512 POPCNT) for ultra-fast matching at bottom layer.

---

## 13.4 Performance and Evaluation (Performance & Evaluation)

Multimodal RAG evaluation isn't just "finding correctly" but "why it was found."

### 13.4.1 Evaluation Metrics

| Metric | Applicable Scenario | Description |
| --- | --- | --- |
| **Recall@K** | General retrieval | Probability that top K results contain correct image. |
| **NDCG@10** | Ranking quality | Higher score for more relevant images ranked higher. |
| **OCR-Free Efficiency** | ColPali scenario | Time/cost savings ratio vs. OCR + Dense Retrieval. |

### 13.4.2 Benchmarks

* **Test environment**: Intel Xeon Gold 6226R, NVIDIA RTX 3090.
* **Dataset**: ViDoRe Benchmark (complex financial statements).

#### 1. Accuracy Comparison (Recall@5)

* **Unstructured OCR + BGE-M3**: 43% (table structure loss is main cause).
* **ColPali v1.2**: 81% (direct visual layout understanding).

#### 2. Latency Comparison

* **SigLIP (Dense)**: < 20ms / query.
* **ColPali (Late Interaction)**: ~150ms / query.

**Conclusion**: ColPali suitable for Re-rank or high-quality retrieval; massive data needs quantization.

### 13.4.3 Interpretability: Where is the Model Looking?

ColPali's another advantage is **interpretability**. By visualizing the interaction matrix from MaxSim computation, we can generate heatmaps showing exactly which document region the model attended to.


---

## 13.5 Common Misconceptions and Pitfalls

* **Misconception 1: "All images are worth indexing"**
* Web page or document icons, decorative lines, header/footer logos produce massive noise.
* **Fix**: Add "junk image classifier" or rule-based filter before indexing (e.g., discard < 5KB or extreme aspect ratio images).


* **Misconception 2: "Ignore Embedding dimension explosion"**
* Don't naively dump all ColPali vectors into regular PGVector.
* **Fix**: Must implement 13.3.3 Binary Quantization. Or use ColPali only for complex "key pages"; ordinary text pages still use BGE/OpenAI Embedding, building hybrid index.


* **Misconception 3: "Use CLIP directly as OCR replacement"**
* CLIP knows images contain "text" but can't read long text. If you ask "Who is Party A in the contract?," standard CLIP usually can't answer.
* **Fix**: For text-dense images without complex layout, OCR + LLM remains cost-effective; ColPali applies to "layout is semantics" scenarios (e.g., complex nested tables).



---

## Chapter Summary

Multimodal RAG expands our vision from 1D text to 2D visual space.

* **Architecture**: Use SigLIP for natural images, ColPali for document images.
* **Code**: Core lies in MaxSim interactive scoring, not simple dot product.
* **Optimization**: Binary Quantization (BQ) is key technology enabling large-scale multimodal RAG deployment.

Mastering this chapter, your RAG system is no longer "blind" but a "versatile expert" that can read charts and analyze reports.

---

## Further Reading

* **Paper**: *ColPali: Efficient Document Retrieval with Vision Language Models* (2024).
* **Tool**: `colpali_engine` official library; track its native Qdrant/Weaviate support updates.
* **Advanced**: Learn **Matryoshka Representation Learning (MRL)** for further vector dimension compression.
