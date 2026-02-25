# 第 13 章: マルチモーダル RAG

---

## 章の概要

テキスト RAG 競争が最高潮に達する中、マルチモーダルが新たな戦場となっています。企業ドキュメントには、大量の**チャート、フローチャート、スクリーンショット**が含まれることがよくあります。従来の RAG では通常、テキストへの OCR を選択するか、画像を無視するため、重要な情報が失われます。この章では、「純粋なテキスト」の制限を打ち破って、データを「見る」ことができるシステムを構築します。基本的な CLIP/SigLIP クロスモーダル取得から開始し、破壊的な **ColPali アーキテクチャ**を掘り下げ、**バイナリ量子化 (BQ)** や **Late Interaction スコアリング ロジック**を含むエンドツーエンドの取得パイプラインを実装します。

## 13.0 学習目標

* **マルチモーダル ベクトル空間を理解する**: CLIP と SigLIP の対照的な損失原理をマスターします。テキストと画像が同じベクトル空間内でどのように配置されるかを理解します。
* **マスター ColPali アーキテクチャ**: Late Interaction メカニズムを理解します。複雑な PDF の表やグラフに `colpali-v1.2-merged` を使用する方法を学びます。
* **エンジニアリング機能**: **MaxSim スコアリング** アルゴリズムを実装する Python コードを作成し、**バイナリ量子化**を使用してストレージ コストを 32 分の 1 に削減します。
* **視覚化検証**: アテンション ヒートマップを通じてモデルの解釈可能性を検証します。

---

## シナリオの紹介

あなたは半導体装置の修理知識ベースを維持しています。技術マニュアルには複雑な回路図やデバイス構造図が満載です。
フィールド エンジニアは「『メインボード電源モジュール』の配線図を見つけるのを手伝ってください」と要求しています。

* **従来の RAG (テキストのみ)**: 「メインボードの電源は図 3-12 を参照してください」というテキスト説明が取得されましたが、システムが画像を返せないか、OCR によって回路図が文字化け (例: `---||---`) になり、エンジニアはテキストを無力に見つめることになります。
* **マルチモーダル RAG**: システムはユーザーのテキストの意図を理解するだけでなく、回路図を含む PDF ページのスクリーンショットを直接取得し、電源モジュール領域を **正確にハイライト**します。

これは「読む」から「見る」への質的飛躍です。多くの産業および金融シナリオでは、1 つの画像の情報密度は 1,000 ワードをはるかに超えています。

---

## 13.1 クロスモーダル検索: テキストと画像の障壁を打ち破る

「テキストから画像への検索」または「画像からテキストへの検索」を実現するには、両方のモダリティを理解するモデルが必要です。

### 13.1.1 基本原則: 対照学習

OpenAI の CLIP (Contrastive Language-Image Pre-training) が基礎となります。そのトレーニング ロジックはシンプルかつ残酷です。
1. 数億の（画像、テキスト）ペアを収集します。
2. **画像エンコーダー**と**テキストエンコーダー**を介してベクトルを抽出します。
3. 一致するペアのベクトルを **引き寄せ**、一致しないペアを **押し**ます。

結果: 「犬」の写真ベクトルと「犬」のテキスト ベクトルは、数学的に空間的に非常に近くなります。

![図 13-1: CLIP マルチモーダル ベクトル空間図](../../images/part5/图13_1_CLIP架构.png)
<!-- ![図 13-1: CLIP マルチモーダル ベクトル空間図](images/Chapter 13/图13_1_CLIP架构.png) -->

*図 13-1: CLIP アーキテクチャ —— テキストと画像は同じ高次元の球体にマッピングされます。コサイン類似性が関連性を決定します*

### 13.1.2 テクノロジーの選択: CLIP と SigLIP

CLIP が最も有名ですが、エンジニアリングに関しては、より良い選択肢があります。 Google の **SigLIP (言語画像事前トレーニング用シグモイド損失)** は、複数の指標で CLIP を上回っています。

|特集 | OpenAIクリップ | Google SigLIP |建築家の推薦 |
| :--- | :--- | :--- | :--- |
| **損失関数** | Softmax (グローバル正規化) | **シグモイド** (独立した二項分類) | Sigmoid はメモリ効率が高くなります。大規模なバッチトレーニングに適しています |
| **中国語サポート** |苦手（主に英語） | **より良い** (多言語バージョン) |多言語チェックポイントを使用する必要があります |
| **解決策** |通常は 224x224 |動的解像度をサポート |複雑なチャートには高解像度 (384+) モデルを選択 |

> **推奨事項**: 2025 年の新しいシステムの場合は、**SigLIP** または Meta の **DINOv2** (強力な純粋なビジュアル機能) を優先してください。

---

## 13.2 ColPali アーキテクチャの実践: OCR の悪夢に終止符を打つ

PDF ドキュメントの検索に関しては、CLIP には致命的な弱点があります。CLIP は自然画像 (猫、犬、風景) には優れていますが、**リッチテキスト画像** (テキストが密集したドキュメント ページ、表) についてはあまり理解できません。
従来のアプローチは `PDF -> OCR -> Text Embedding` ですが、OCR ではレイアウト情報が失われ、グラフでは役に立ちません。

**ColPali (ColBERT + PaliGemma)** は、**OCR なし - PDF ページを直接画像として扱う**という革新的なアイデアを提案しています。

### 13.2.1 基本原則: 視覚言語モデルの後期相互作用

ColPali は、Vision-Language Model (VLM) と ColBERT 検索メカニズムを組み合わせています。

1. **パッチのEmbedding**: ドキュメント画像を小さなパッチに分割します。各パッチはベクトルを生成します。 1 つの画像は 1024 個のベクトルに対応します。
2. **遅延インタラクション (MaxSim)**: 取得時に、すべてのドキュメント パッチ ベクトルに対して各クエリ トークン ベクトルを計算し、最大の類似性を取得します。

![図 13-2: ColPali と OCR の比較](../../images/part5/图13_2_ColPali对比.png)
<!-- ![図 13-2: ColPali と OCR の比較](images/Chapter 13/图13_2_ColPali对比.png) -->

*図 13-2: 悪いケース (左) と良いケース (右) —— 左: OCR によりテーブルが文字化けしてしまいます。右: ColPali はクエリを画像レベルでテーブル行と直接位置合わせします*

---

## 13.3 エンジニアリングの実装: ハイブリッド マルチモーダル検索パイプラインの構築

このセクションでは、**SigLIP (自然画像)** および **ColPali (文書画像)** と互換性のある検索システム フレームワークを実装します。

### 13.3.1 全体的なアーキテクチャとデータ フロー

コーディングする前に、データの流れを明確にする必要があります。これはもはや単純な「テキスト入力、テキスト出力」ではありません。

![図 13-3: マルチモーダル RAG エンドツーエンド パイプライン](../../images/part5/图13_3_多模态流水线.png)
<!-- ![図 13-3: マルチモーダル RAG エンドツーエンド パイプライン](images/Chapter 13/图13_3_多模态流水线.png) -->

*図 13-3: エンドツーエンドのパイプライン —— 左: インデックス フロー (PDF から画像 -> ビジョン エンコーダー -> 量子化 -> ベクトル DB)。右: 取得フロー (クエリ -> テキスト エンコーダー -> MaxSim スコアリング -> 再ランク)*

### 13.3.2 コアコード: マルチモーダルなインデックス作成とスコアリング

`MultimodalIndexer` を定義します。実用的な機能を実現するには、「ベクトル化 (Embedding)」ロジックだけでなく、明示的な「スコアリング」ロジックも必要です。ColPali 検索は、単純にベクトル データベースのコサイン類似性に依存することはできません。

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
## 13.3.3 パフォーマンスの最適化: バイナリ量子化 (BQ)

ColPali のエンジニアリング上の最大の問題は **ストレージの爆発**です。

* **従来のEmbedding**: 1 ページ = 1 ベクトル (4KB float32)。
* **ColPali**: 1 ページ = 1032 ベクター (512KB float16)。

100 万 PDF ページのインデックス作成には 500 GB のメモリが必要ですが、実際には受け入れられないことがよくあります。

**解決策**: バイナリ量子化を使用し、float16 を 1 ビットに圧縮します。

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

> **アーキテクチャに関する決定**: 運用環境では、Binary Vector (Qdrant、Vespa) または専用のインデックス ライブラリ (USearch) をサポートするデータベースを推奨します。これらのデータベースは、最下層での超高速マッチングのために CPU 命令 (AVX-512 POPCNT) を活用できます。

---

## 13.4 性能と評価 (性能と評価)

マルチモーダル RAG の評価では、単に「正しく見つかった」だけではなく、「なぜ見つかったのか」も評価されます。

### 13.4.1 評価指標

|メトリック |該当するシナリオ |説明 |
| --- | --- | --- |
| **リコール@K** |一般的な検索 |上位 K 個の結果に正しい画像が含まれる確率。 |
| **NDCG@10** |ランキング品質 |関連性の高い画像のスコアが高いほど、上位にランクされます。 |
| **OCR 不要の効率** | ColPali シナリオ |時間/コストの節約率と OCR + 高密度取得の比較。 |

### 13.4.2 ベンチマーク

* **テスト環境**: Intel Xeon Gold 6226R、NVIDIA RTX 3090。
* **データセット**: ViDoRe ベンチマーク (複雑な財務諸表)。

#### 1. 精度の比較 (再現率@5)

* **非構造化 OCR + BGE-M3**: 43% (テーブル構造の損失が主な原因)。
* **ColPali v1.2**: 81% (直接視覚的にレイアウトを理解)。

#### 2. レイテンシの比較

* **SigLIP (密)**: < 20ms/クエリ。
* **ColPali (遅延インタラクション)**: クエリあたり最大 150 ミリ秒。

**結論**: ColPali は再ランク付けまたは高品質の検索に適しています。大量のデータには量子化が必要です。

### 13.4.3 解釈可能性: モデルはどこを見ているのか?

ColPali のもう 1 つの利点は **解釈可能性** です。 MaxSim 計算からの相互作用行列を視覚化することで、モデルがどのドキュメント領域に注目したかを正確に示すヒートマップを生成できます。


---

## 13.5 よくある誤解と落とし穴

* **誤解 1: 「すべての画像にはインデックスを作成する価値がある」**
* Web ページやドキュメントのアイコン、装飾線、ヘッダー/フッターのロゴは大きなノイズを発生します。
* **修正**: インデックス作成前に「ジャンク画像分類子」またはルールベースのフィルターを追加します (例: 5KB 未満または極端なアスペクト比の画像を破棄します)。


* **誤解 2: 「Embedding次元の爆発を無視する」**
* すべての ColPali ベクトルを通常の PGVector に単純にダンプしないでください。
* **修正**: 13.3.3 バイナリ量子化を実装する必要があります。または、複雑な「キー ページ」にのみ ColPali を使用します。通常のテキスト ページは引き続き BGE/OpenAI Embeddingを使用し、ハイブリッド インデックスを構築します。


* **誤解 3: 「OCR の代替として CLIP を直接使用する」**
* CLIP は画像に「テキスト」が含まれていることを認識しますが、長いテキストを読み取ることはできません。 「契約書の当事者 A は誰ですか?」と尋ねられても、標準の CLIP では通常は答えることができません。
* **修正**: 複雑なレイアウトを持たないテキスト密度の高い画像の場合、OCR + LLM は依然として費用対効果が高くなります。 ColPali は、「レイアウトはセマンティクス」シナリオ (複雑なネストされたテーブルなど) に適用されます。



---

## 章の概要

マルチモーダル RAG は、私たちの視野を 1D テキストから 2D 視覚空間に拡張します。

* **アーキテクチャ**: 自然画像には SigLIP を使用し、ドキュメント画像には ColPali を使用します。
* **コード**: コアは単純なドット積ではなく、MaxSim インタラクティブ スコアリングにあります。
* **最適化**: バイナリ量子化 (BQ) は、大規模なマルチモーダル RAG 導入を可能にする主要なテクノロジです。

この章をマスターすると、RAG システムはもはや「盲目」ではなく、チャートを読み取ってレポートを分析できる「多用途のエキスパート」になります。

---

## さらに読む

* **論文**: *ColPali: ビジョン言語モデルを使用した効率的な文書検索* (2024)。
* **ツール**: `colpali_engine` 公式ライブラリ;ネイティブの Qdrant/Weaviate サポートの更新を追跡します。
* **上級**: さらなるベクトル次元圧縮については、**マトリョーシカ表現学習 (MRL)** を学習してください。
