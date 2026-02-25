# プロジェクト 1: 「Mini-C4」事前トレーニング データセットの構築

### 1. プロジェクトの背景 (プロジェクトの概要)

* **タスク定義:** 小型 C4 (Colossal Clean Crawled Corpus) データセット パイプラインを構築します。私たちの目標は、乱雑な生の Web データ (共通クロール) を、大規模モデルの事前トレーニングに直接使用できる低ノイズで重複排除された高品質の純粋なテキスト データに変換することです。
* **入力と出力:**
    * **入力:** Common Crawl からの生の WARC アーカイブ (HTTP ヘッダー、HTML ソース、文字化けしたテキストなどを含む)。
    * **出力:** クリーン テキストと品質スコアを含む、分類および採点された JSONL ファイル (例: `data_en.jsonl`、`final_data.jsonl`)。
* **課題分析:**
    * **信号対雑音比が非常に低い:** 生の Web コンテンツの 90% 以上は、ナビゲーション バー、広告、JavaScript コード、および無意味なプレースホルダーです。
    * **計算集約型:** 大規模なコーパスにおけるペアワイズ比較の重複排除は、非常にリソースを消費します。
    * **品質の定量化:** 文章が「高品質の人間の言語」であるか「機械が生成したゴミ」であるかを機械に自動的に判断させるにはどうすればよいでしょうか?

### 2. アーキテクチャ設計 (アーキテクチャ設計)

非構造化 Web データを処理するために、次のじょうご型の処理アーキテクチャを設計しました。

**データ パイプライン図:**

![図 1: Mini-C4 事前トレーニング データセット パイプライン](../../images/part6/图1_构建Mini_C4预训练集数据流水线图.png)
<!-- ![図 1: Mini-C4 事前トレーニング データセット パイプライン](images/Practical Projects/图1_构建Mini_C4预训练集数据流水线图.png) -->

**テクノロジースタック:**

* **解析レイヤー: Trafilatura**
    * *意思決定の根拠:* 従来の BeautifulSoup と比較して、Trafilatura は Web 記事の抽出に最適化されており、ナビゲーション、フッター、ボイラープレート テキストを自動的に削除し、より高い抽出効率と精度を実現します。
* **コンピューティングレイヤー: Ray**
    * *意思決定の根拠:* Python のネイティブ マルチプロセッシングはビッグ データと格闘します。 Ray は非常にシンプルな分散プリミティブを提供しており、数行のコードでマルチコア CPU やクラスター全体で MinHash 計算を並列化できます。
* **品質レイヤー: KenLM**
    * *意思決定の根拠:* 軽量の N-gram 言語モデル ライブラリ。 GPT-3 論文と CCNet 論文は両方とも、テキストの自然さを測定するための中心的な指標として KenLM の複雑度を使用しています。

### 3. 段階的な実装

#### フェーズ 1: 乱雑な HTML からメイン コンテンツを抽出する (抽出とクリーニング)

未加工の WARC ファイルには、テキスト以外のノイズが大量に含まれています。まず圧縮アーカイブのストリーミング読み取りに `warcio` を使用し、次に `trafilatura` を使用してコア コンテンツを抽出します。その後、初期フィルタリングにヒューリスティック ルールを適用します。

**コアコード: 解析とヒューリスティッククリーニング**

```python
import trafilatura
from warcio.archiveiterator import ArchiveIterator

# 1. Extraction logic (from 2_process_warc.py)
def extract_text(content_stream):
    text = trafilatura.extract(
        content_stream, 
        include_comments=False, 
        include_tables=False
    )
    return text

# 2. Heuristic cleaning rules (from 3_clean_data.py)
def is_high_quality(text):
    # Rule A: Length and average word length filter
    words = text.split()
    if not words:
        # Empty or whitespace-only text, treat as low quality
        return False
    mean_word_len = sum(len(w) for w in words) / len(words)
    if mean_word_len > 15: # Overly long words usually garbage or code
        return False
        
    # Rule B: Symbol density (Symbol Ratio)
    code_symbols = {'{', '}', '[', ']', '<', '>', '\\'}
    symbol_count = sum(1 for char in text if char in code_symbols)
    if len(text) > 0 and (symbol_count / len(text) > 0.1): # Too many code symbols
        return False
        
    # Rule C: Blacklist keywords
    bad_phrases = ["lorem ipsum", "enable cookies", "403 forbidden"]
    if any(p in text.lower() for p in bad_phrases):
        return False
        
    return True
```

#### フェーズ 2: 分散 MinHash 重複排除

インターネットには、繰り返されるコンテンツ (再投稿、ミラー) が大量にあります。 MinHash の並列計算には Ray を使用し、LSH (局所性敏感ハッシュ) を組み合わせて $O(N^2)$ の複雑さを $O(N)$ に軽減します。

**コア コード: レイ並列署名の計算**

```python
import ray
from datasketch import MinHash

# Initialize Ray to use all CPU cores
ray.init()

@ray.remote
def process_batch(lines, num_perm=128):
    """Ray Worker: Parallel MinHash fingerprint computation for a batch"""
    results = []
    for line in lines:
        item = json.loads(line)
        m = MinHash(num_perm=num_perm)
        # Shingling: Update hash by words
        for w in item['text'].split():
            m.update(w.encode('utf8'))
        results.append((item['url'], m, item['text']))
    return results

# Main flow: Map-Reduce style
# Map: Distribute compute tasks
futures = [process_batch.remote(batch) for batch in batches]
# Reduce: Collect results and build LSH index
results = ray.get(futures)
# ...continue with MinHashLSH index building...
```

#### フェーズ 3: 言語の識別と複雑さのフィルタリング (品質フィルタリング)

クリーンアップされたデータには、さまざまな品質の複数の言語が混在しています。最初に言語ルーティングに FastText を使用し、次に複雑さのために KenLM を使用します。混乱が少ないということは、より流暢で、より「人間らしい」文章を意味します。

**コア コード: KenLM スコアリング**

```python
import kenlm
import fasttext

# 1. Language routing (from 5_split_lang.py)
lid_model = fasttext.load_model('lid.176.ftz')
def predict_lang(text):
    # k=1 takes highest probability language
    predictions = lid_model.predict(text, k=1)
    return predictions[0][0].replace('__label__', '')

# 2. Perplexity filtering (from 6_quality_filter.py)
kenlm_model = kenlm.Model('en.arpa.bin')
PERPLEXITY_THRESHOLD = -6.0  # Experience threshold: below this usually low-quality text

def filter_by_perplexity(text):
    words = text.split()
    if not words:
        # Empty text as low quality; avoid division by zero
        return False, -10.0
    # Compute normalized score (Log Score / Length)
    log_score = kenlm_model.score(text)
    normalized_score = log_score / len(words)
    
    if normalized_score > PERPLEXITY_THRESHOLD:
        return True, normalized_score
    return False, normalized_score
```

### 4. 結果ショーケース (ショーケース)

このパイプラインの後、データの状況は根本的に変わりました。

**ケース 1: ナビゲーション バーのノイズ (除去)**
> *生:* "ホーム | 私たちについて | お問い合わせ | Cookie を有効にする | Copyright 2023..."
> *結果:* **[破棄]** (短いテキストとキーワードのブラックリスト ルールがトリガーされました)

**ケース 2: コード スニペット (削除)**
> *Raw:* "function(x) { return x > 0 ? true : false; } var a = [1,2,3];"
> *結果:* **[破棄]** (トリガーされたシンボル密度 > 10% ルール)

**ケース 3: 高品質の記事 (保持およびスコアリング)**
> *生:* 「ジェームズ・ウェッブ宇宙望遠鏡は、創造の柱の新しい画像を捉えました...」
> *結果:* **[保持]**
> *KenLM スコア:* -4.82 (しきい値 -6.0 より良い)

**データ統計:**
シングルクロールサンプルテストの場合:
* **生の記録:** 10,000
* **抽出された有効なテキスト:** ~4,500 (HTML 解析損失)
* **クリーニング後の残り:** ~2,800 (ヒューリスティック フィルター損失)
* **重複排除後の残り:** ~2,100 (~25% 重複率)
* **最終的な高品質セット:** ~1,800 (KenLM フィルター)

### 5. コストと最適化 (コストと最適化)

* **リソース消費量:**
    * **コンピューティング:** このプロジェクト コードは、単一マシンの 16 コア CPU、64G RAM、1GB WARC の処理に約 5 ～ 8 分かかります。
    * **ボトルネック:** `MinHashLSH` インデックスの構築は現在シングルスレッド (`4_deduplicate.py` 内) であり、完全にメモリに依存しています。

* **スケーリングに関する考慮事項:**
    データが TB レベル (実際の C4 など) にスケールされる場合は、現在のアーキテクチャをアップグレードする必要があります。
    1. **LSH ストレージ:** インメモリ `MinHashLSH` は使用できません。ハッシュバケットには Redis または Cassandra が必要です。
    2. **並列戦略:** Ray タスクを「単一マシン マルチコア」から「マルチマシン クラスター」に拡張します。
    3. **I/O 最適化:** データ読み取りは、ストリーミングカラム型処理に PyArrow を使用して、ローカル ファイルシステムから S3 に移行する必要があります。
