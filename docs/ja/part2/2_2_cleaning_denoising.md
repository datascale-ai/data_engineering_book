# 第 4 章: クリーニングとノイズ除去

---

## 章の概要

インターネットから得られる生データは未処理の鉱石のようなもので、本当に価値のある「濃縮物」はほんの一部しか占めていない可能性があります。この章では、トレーニング前データ クリーニングの 3 つのコア テクノロジ、つまり低品質のドキュメントを削除するヒューリスティック フィルタリング ルール、重複コンテンツを排除する大規模重複排除、およびユーザー情報を保護するプライバシー データ クリーニングについて詳しく説明します。これらのテクニックを習得すると、読者は産業グレードのデータ クリーニング パイプラインを構築して、生の Web データを高品質の事前トレーニング コーパスに変換できるようになります。

---

## シナリオの紹介

前の章の取り組みの結果、チームは Common Crawl から 50TB の中国語 Web テキストを抽出することに成功しました。しかし、データをランダムにサンプリングして検査すると、さまざまな厄介な問題が見つかります。多くのページには、実質的なコンテンツのないナビゲーション テキストが数語しか含まれていません。一部のページは JavaScript コードまたは CSS スタイルの残骸でいっぱいです。特定の Web サイトのコンテンツが何百回も繰り返しクロールされました。また、ユーザーの電子メールや電話番号などの機密情報が大量に含まれています。

さらに悪いことに、トレーニング エンジニアは、前回クリーンアップされていないデータでトレーニングされたモデルには深刻な「オウム」問題があったと告げました。モデルは同じ文章を繰り返し出力し、特定の Web サイトのコンテンツを完全に暗唱することさえありました。これは明らかにデータの重複が原因です。

これらの問題を体系的に解決するにはどうすればよいでしょうか?この章では完全な答えを提供します。

---

## 4.1 ヒューリスティックフィルタリングルール

ヒューリスティック フィルタリングは、データ クリーニングにおける防御の最前線です。一連の定量化可能なルールを使用して、明らかに低品質のドキュメントを迅速に除外します。これらのルールは単純に見えるかもしれませんが、実際にはほとんどのノイズの多いデータをフィルタリングして除去できるため、コスト効率の高いクリーニング アプローチとなります。

![図 4-1: データ クリーニング パイプライン](../../images/part2/图4_1_数据清洗流水线.png)

*図 4-1: データ クリーニング パイプライン アーキテクチャ — 生データからクリーン コーパスまでの 8 段階の処理フロー*

### 4.1.1 言語の検出

言語検出は、多言語データ処理における基本的なステップです。中国語モデルをトレーニングするには、まず Common Crawl の膨大なデータから中国語のコンテンツをフィルタリングする必要があります。これには正確な言語検出機能が必要です。

**FastText Language Detector** は、現在最も一般的に使用されているツールです。 Facebook AI Research によって開発されたその事前トレーニング済みモデルは、非常に高速かつ高精度で 176 言語をサポートします。 FastText には 2 つの事前トレーニング済みモデルが用意されています。 `lid.176.bin` は精度が高い完全バージョンですが、サイズは大きくなります (~126MB)。 `lid.176.ftz` は、サイズが小さい (~917KB) ものの、精度がわずかに低い圧縮バージョンです。大規模なデータ処理の場合は、製品版を推奨します。

```python
import fasttext

# Load language detection model
lang_model = fasttext.load_model('lid.176.bin')

def detect_language(text: str, min_confidence: float = 0_8) -> tuple:
    """
    Detect text language
    
    Args:
        text: Text to be detected
        min_confidence: Minimum confidence threshold
    
    Returns:
        (language code, confidence) or (None, 0) if confidence insufficient
    """
    # Preprocessing: remove newlines, take first 1000 characters
    text = text.replace('\n', ' ')[:1000]
    
    # Predict
    predictions = lang_model.predict(text, k=1)
    lang = predictions[0][0].replace('__label__', '')
    confidence = predictions[1][0]
    
    if confidence >= min_confidence:
        return lang, confidence
    return None, confidence

def filter_by_language(documents: list, target_lang: str = 'zh') -> list:
    """Filter documents by specified language"""
    results = []
    for doc in documents:
        lang, conf = detect_language(doc['text'])
        if lang == target_lang:
            doc['detected_lang'] = lang
            doc['lang_confidence'] = conf
            results.append(doc)
    return results
```

言語検出では、実際にはいくつかの特殊なケースが発生します。混合言語のドキュメント (中国語と英語の技術ブログなど) は誤って分類される可能性があります。短いテキストは検出精度が低くなります。 50 文字未満のテキストの言語フィルタリングをスキップすることをお勧めします。コード スニペットはさまざまな言語として識別される場合があり、コンテンツ タイプの判断が必要になります。

### 4.1.2 テキスト品質スコアリング

言語検出は、ドキュメントがターゲット言語であることを確認するだけで、コンテンツの品質を判断することはできません。文法的に正しいスパム広告と高品質の技術記事は、同じ言語検出スコアを受け取る場合があります。これには、より洗練された品質評価メカニズムが必要です。

**パープレキシティ フィルタリング** は、言語モデルに基づいた品質評価方法です。パープレキシティは、テキストにおける言語モデルの「驚き」を測定します。テキストがモデルのトレーニング データの分布と類似している場合、パープレキシティは低くなります。テキストに多くのノイズ、意味不明な表現、または不自然な表現が含まれている場合、困惑度は高くなります。

KenLM は、複雑さを計算するために最も一般的に使用されるツールです。 N-gram 言語モデルに基づいており、非常に高速で、大規模なデータ処理に適しています。

```python
import kenlm

class PerplexityFilter:
    def __init__(self, model_path: str, max_perplexity: float = 500):
        """
        Initialize perplexity filter
        
        Args:
            model_path: KenLM model path (.arpa or .bin)
            max_perplexity: Perplexity threshold; documents exceeding this value will be filtered
        """
        self.model = kenlm.Model(model_path)
        self.max_perplexity = max_perplexity
    
    def compute_perplexity(self, text: str) -> float:
        """Compute text perplexity"""
        # KenLM returns log10 probability
        log_prob = self.model.score(text, bos=True, eos=True)
        # Convert to perplexity
        num_words = len(text.split()) + 1  # +1 for EOS
        perplexity = 10 ** (-log_prob / num_words)
        return perplexity
    
    def filter(self, documents: list) -> list:
        """Filter high-perplexity documents"""
        results = []
        for doc in documents:
            ppl = self.compute_perplexity(doc['text'])
            if ppl <= self.max_perplexity:
                doc['perplexity'] = ppl
                results.append(doc)
        return results
```

パープレキシティのしきい値設定は、特定のデータに基づいて調整する必要があります。一般に、高品質のニュースや百科事典のテキストのパープレキシティは 100 ～ 200、通常の Web コンテンツのパープレキシティは 200 ～ 500、低品質のコンテンツ (意味不明、機械翻訳など) は通常 500 を超えます。最初に小さなサンプルでパープレキシティの分布を分析し、次に適切なしきい値を決定することをお勧めします。

### 4.1.3 ヒューリスティックルールセット

言語検出と複雑性フィルタリングに加えて、明らかに低品質のコンテンツを迅速に削除できる、シンプルだが効果的なヒューリスティック ルールのセットがあります。これらのルールは、大量のデータからの観察と経験に基づいて設計されています。

**長さのフィルタリング**は最も基本的なルールです。短すぎるドキュメント (数単語のみのナビゲーション テキストなど) にはトレーニング価値がないため、直接削除する必要があります。ドキュメントが長すぎる場合は、切り詰めたり分割したりする必要がある場合があります。一般的なしきい値: 最小長は 200 文字または 50 単語、最大長は 100,000 文字です。

**特殊文字の比率** により、ノイズの多いコンテンツを識別できます。文書内の英数字以外の文字の割合が高すぎる場合は、コードの残骸、意味不明な文字、またはフォーマット エラーが発生している可能性があります。同様に、桁の比率が高すぎる場合は、ログ ファイルまたはデータ テーブルを示している可能性があります。

**重複行比率** により、テンプレート化された低品質のページを検出できます。ドキュメントに同一の行が多数ある場合 (ナビゲーション バーが複数の場所で繰り返されている場合など)、コンテンツの品質は低くなります。

**語彙の多様性**は、文書の情報の豊富さを測定します。 10 個の異なる単語のみを使用する文書は、500 個の異なる単語を使用する文書よりも明らかに価値が低くなります。一般的な指標は、合計単語に対する一意の単語の比率であるタイプ トークン比 (TTR) です。

以下は包括的なヒューリスティック フィルターの実装です。

```python
import re
from collections import Counter

class HeuristicFilter:
    def __init__(self, config: dict = None):
        """
        Initialize heuristic filter
        
        Default config suitable for Chinese pre-training data
        """
        self.config = config or {
            'min_length': 200,           # Minimum character count
            'max_length': 100000,        # Maximum character count
            'min_words': 50,             # Minimum word count
            'max_special_ratio': 0_3,    # Maximum special character ratio
            'max_digit_ratio': 0_3,      # Maximum digit ratio
            'max_duplicate_line_ratio': 0_3,  # Maximum duplicate line ratio
            'min_avg_word_length': 2,    # Minimum average word length
            'max_avg_word_length': 20,   # Maximum average word length
            'min_unique_word_ratio': 0_1 # Minimum vocabulary diversity
        }
    
    def check_length(self, text: str) -> bool:
        """Check document length"""
        length = len(text)
        return self.config['min_length'] <= length <= self.config['max_length']
    
    def check_special_chars(self, text: str) -> bool:
        """Check special character ratio"""
        if len(text) == 0:
            return False
        special = len(re.findall(r'[^\w\s]', text, re.UNICODE))
        ratio = special / len(text)
        return ratio <= self.config['max_special_ratio']
    
    def check_digit_ratio(self, text: str) -> bool:
        """Check digit ratio"""
        if len(text) == 0:
            return False
        digits = len(re.findall(r'\d', text))
        ratio = digits / len(text)
        return ratio <= self.config['max_digit_ratio']
    
    def check_duplicate_lines(self, text: str) -> bool:
        """Check duplicate line ratio"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) == 0:
            return False
        unique_lines = len(set(lines))
        duplicate_ratio = 1 - (unique_lines / len(lines))
        return duplicate_ratio <= self.config['max_duplicate_line_ratio']
    
    def check_vocabulary_diversity(self, text: str) -> bool:
        """Check vocabulary diversity"""
        words = text.split()
        if len(words) < self.config['min_words']:
            return False
        unique_ratio = len(set(words)) / len(words)
        return unique_ratio >= self.config['min_unique_word_ratio']
    
    def filter(self, text: str) -> tuple:
        """
        Apply all filtering rules
        
        Returns:
            (passed or not, failure reason or None)
        """
        checks = [
            (self.check_length, 'length'),
            (self.check_special_chars, 'special_chars'),
            (self.check_digit_ratio, 'digit_ratio'),
            (self.check_duplicate_lines, 'duplicate_lines'),
            (self.check_vocabulary_diversity, 'vocabulary_diversity')
        ]
        
        for check_func, name in checks:
            if not check_func(text):
                return False, name
        
        return True, None
```

### 4.1.4 品質階層化戦略

実際には、単にデータを「保持」または「破棄」として二項に分類するのは大雑把すぎることがよくあります。より洗練されたアプローチは、データを品質ごとに階層化し、異なる品質層に異なるサンプリング重みを割り当てることです。

一般的な階層化戦略: データを高品質、中品質、低品質の層に分割します。高品質のデータ (例: 権威あるサイトからのもの、すべてのヒューリスティック チェックに合格したもの、複雑さが低いもの) は、より高いサンプリング重みを受け取ります。中品質のデータは通常の重みを受け取ります。低品質だが許容できるデータには、より低い重みが与えられます。この戦略により、データの多様性を確保しながら、高品質のデータがトレーニングでより大きな役割を果たすことができます。

RefinedWeb の論文では、データを 5 つの層に分割し、それぞれに異なるフィルタリングしきい値を設定して、層別化戦略を詳細に文書化しています。この洗練された品質管理は、高品質の事前トレーニング データセットを構築するための鍵となります。

![図 4-2: 品質フィルタリング ファネル](../../images/part2/图4_2_质量过滤漏斗.png)

*図 4-2: データ品質フィルタリング ファネル — 100% 生データから最終的な 4% のクリーン コーパスまでの階層化フィルタリング プロセス*

---

## 4.2 大規模な重複排除

データの重複はトレーニング前のデータの敵です。 Common Crawl では、同じ記事が複数の Web サイトに転載されたり、同じ Web ページが異なる月に繰り返しクロールされたりして、大量の重複コンテンツが発生する可能性があります。調査によると、重複排除されていないデータによりモデルが繰り返されるコンテンツに過剰適合し、モデルの品質に重大な影響を与える「オウム」現象が発生することがわかっています。

重複排除は 2 つのレベルに分けることができます。正確な重複排除では同一のドキュメントが削除されます。ファジー重複排除は、非常に類似しているが同一ではない文書 (再版時にわずかに変更された記事など) を削除します。 TB スケールのデータでは、どちらのタイプでも効率的なアルゴリズムと分散実装が必要です。

### 4.2.1 正確な重複排除: ハッシュ方式

正確な重複排除の中心的な考え方は、各ドキュメントのフィンガープリントを計算することです。同じフィンガープリントを持つ文書は重複とみなされます。最も単純な方法は、MD5 や SHA256 などのハッシュ関数を使用します。

```python
import hashlib

def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def exact_dedup(documents: list) -> list:
    """Exact deduplication: keep first document for each hash value"""
    seen_hashes = set()
    results = []
    
    for doc in documents:
        doc_hash = compute_hash(doc['text'])
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            doc['hash'] = doc_hash
            results.append(doc)
    
    return results
```

分散シナリオの場合、Spark または Ray を並列重複排除に使用できます。

```python
import ray

@ray.remote
def compute_hashes_batch(documents: list) -> list:
    """Batch compute hashes"""
    return [(compute_hash(doc['text']), doc) for doc in documents]

def distributed_exact_dedup(documents_path: str, output_path: str):
    """Distributed exact deduplication"""
    ds = ray.data.read_parquet(documents_path)
    
    # Compute hashes
    ds = ds.map(lambda doc: {**doc, 'hash': compute_hash(doc['text'])})
    
    # Group by hash, keep first per group
    ds = ds.groupby('hash').map_groups(lambda group: group.head(1))
    
    # Save results
    ds.write_parquet(output_path)
```

正確な重複排除は効率的ですが、処理できるのは同一のドキュメントのみです。わずかに異なる重複コンテンツ (例: 同じニュースが異なるヘッダー/フッターで異なるサイトに再投稿される) の場合、正確な重複排除は無力です。

### 4.2.2 ファジー重複排除: MinHash LSH

ファジー重複排除の目的は、「非常に類似しているが同一ではない」文書を識別することです。これは計算的に複雑な問題です。単純に 2 つのドキュメントを比較するには O(n²) 時間の計算量が必要ですが、数十億のドキュメントではまったく実行不可能です。

MinHash LSH (Locality-Sensitive Hashing) は、この問題を解決するためのコア アルゴリズムです。基本的な考え方: 最初にドキュメントを N-gram セットに変換し、次に MinHash を使用してセットを固定長の署名に圧縮し、最後に LSH を使用して同様の署名を同じバケットにクラスタリングします。同じバケット内のドキュメントのペアのみを詳細に比較する必要があるため、計算が大幅に削減されます。

MinHash LSH を理解するには、次の 3 つの手順が必要です。

**ステップ 1: n グラム分解。** ドキュメントを n グラム (連続する n 文字または単語) のセットとして扱います。たとえば、「大モデルデータ」の 3 グラムのセットは、{"大モデル", "モデル数", "型データデータ"} となります。文書全体ではなく N グラムを使用すると、局所的な類似性がより適切に捕捉されます。

**ステップ 2: MinHash 署名。** MinHash は、セットを固定長の署名に圧縮するための技術です。 2 つのセット間の Jaccard 類似性は、MinHash 署名の一致度によって近似できます。署名が長いほど正確な推定値が得られますが、ストレージと計算のコストが高くなります。

**ステップ 3: LSH バケット化。** MinHash 署名を複数のバンドに分割し、それぞれに複数のハッシュ値を含めます。 2 つのドキュメントの任意のバンドのすべての位置に同一のハッシュ値がある場合、それらは同じバケットに配置されます。バンドの数とバンド サイズを調整することで、類似性のしきい値と再現率を制御します。

以下は完全な MinHash LSH 実装です。

![図 4-3: MinHash LSH アルゴリズム](../../images/part2/图4_3_MinHash_LSH算法.png)

*図 4-3: MinHash LSH アルゴリズムの 3 つのステップ — N グラム分解、MinHash 署名計算、LSH バケット化、複雑さを O(n²) から O(n) に削減*

```python
import hashlib
import struct
from typing import Set, List, Tuple
import numpy as np

class MinHashLSH:
    def __init__(self, 
                 num_hashes: int = 128,
                 num_bands: int = 16,
                 ngram_size: int = 5,
                 threshold: float = 0_8):
        """
        Initialize MinHash LSH
        
        Args:
            num_hashes: MinHash signature length
            num_bands: Number of LSH bands
            ngram_size: n-gram size
            threshold: Similarity threshold (reference value, actual threshold determined by band params)
        """
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.ngram_size = ngram_size
        
        # Generate random parameters for hash functions
        self.hash_params = [
            (np.random.randint(1, 2**31), np.random.randint(0, 2**31))
            for _ in range(num_hashes)
        ]
        
        # LSH buckets
        self.buckets = [{} for _ in range(num_bands)]
    
    def get_ngrams(self, text: str) -> Set[str]:
        """Extract n-gram set"""
        text = text.lower().replace(' ', '')
        ngrams = set()
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.add(text[i:i + self.ngram_size])
        return ngrams
    
    def compute_minhash(self, ngrams: Set[str]) -> np.ndarray:
        """Compute MinHash signature"""
        signature = np.full(self.num_hashes, np.inf)
        
        for ngram in ngrams:
            # Compute base hash value for ngram
            h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
            
            # Use multiple hash functions
            for i, (a, b) in enumerate(self.hash_params):
                hash_val = (a * h + b) % (2**31 - 1)
                if hash_val < signature[i]:
                    signature[i] = hash_val
        
        return signature.astype(np.uint32)
    
    def get_bands(self, signature: np.ndarray) -> List[str]:
        """Split signature into bands"""
        bands = []
        for i in range(self.num_bands):
            start = i * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]
            band_hash = hashlib.md5(band.tobytes()).hexdigest()
            bands.append(band_hash)
        return bands
    
    def insert(self, doc_id: str, text: str):
        """Insert document into LSH index"""
        ngrams = self.get_ngrams(text)
        if len(ngrams) == 0:
            return
        
        signature = self.compute_minhash(ngrams)
        bands = self.get_bands(signature)
        
        for band_idx, band_hash in enumerate(bands):
            if band_hash not in self.buckets[band_idx]:
                self.buckets[band_idx][band_hash] = []
            self.buckets[band_idx][band_hash].append(doc_id)
    
    def find_candidates(self, text: str) -> Set[str]:
        """Find candidate similar documents"""
        ngrams = self.get_ngrams(text)
        if len(ngrams) == 0:
            return set()
        
        signature = self.compute_minhash(ngrams)
        bands = self.get_bands(signature)
        
        candidates = set()
        for band_idx, band_hash in enumerate(bands):
            if band_hash in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][band_hash])
        
        return candidates

def jaccard_similarity(set1: Set, set2: Set) -> float:
    """Compute Jaccard similarity"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0
```

### 4.2.3 分散型重複排除の実践

TB 規模のデータで MinHash LSH を実行するには、慎重に設計された分散戦略が必要です。一般的なフローには次のものが含まれます。

**フェーズ 1: 署名の計算。** すべてのドキュメントを並行して走査し、それぞれの MinHash 署名を計算します。このフェーズは完全に並列化可能であり、分散コンピューティング リソースを十分に活用できます。

**フェーズ 2: バンドのグループ化。** 各ドキュメントをバンド値でグループ化します。同じバンド値を持つドキュメントは、後続の比較のために同じパーティションに割り当てられます。

**フェーズ 3: グループ内重複排除。** 各パーティション内で、候補の重複ペアに対して詳細な類似度計算を実行して、真の重複関係を決定します。

**フェーズ 4: 推移閉包。** ドキュメント A が B を複製し、B が C を複製する場合、A、B、C はすべて 1 つの重複グループとみなされます。重複した関係の推移閉包を計算する必要があります。

**フェーズ 5: 保持するドキュメントを選択します。** 各重複グループ内で、保持する代表 (通常は最高品質または最長) を 1 つ選択し、残りを削除します。

```python
import ray

def distributed_fuzzy_dedup(input_path: str, output_path: str, 
                            threshold: float = 0_8):
    """
    Distributed fuzzy deduplication pipeline
    """
    # Read data
    ds = ray.data.read_parquet(input_path)
    
    # Phase 1: Compute MinHash signatures
    def compute_signature(doc):
        lsh = MinHashLSH()
        ngrams = lsh.get_ngrams(doc['text'])
        signature = lsh.compute_minhash(ngrams)
        bands = lsh.get_bands(signature)
        return {**doc, 'signature': signature.tolist(), 'bands': bands}
    
    ds = ds.map(compute_signature)
    
    # Phase 2: Group by band value, find candidate pairs
    # (Simplified here, actual implementation needs more complex grouping logic)
    
    # Phase 3&4: Intra-group exact comparison, build duplicate relationship graph
    # ...
    
    # Phase 5: Select retained documents
    # ...
    
    # Save results
    ds.write_parquet(output_path)
```

実際のエンジニアリングでは、既存のツールを使用することをお勧めします。 **text-dedup** は、Spark および Ray 分散実装を備えた MinHash LSH、SimHash、Suffix Array などのさまざまなアルゴリズムを実装するオープンソースのテキスト重複排除ライブラリです。 **Dolma** の重複排除モジュールも高品質のリファレンス実装です。

### 4.2.4 ドキュメント内の重複排除

ドキュメントレベルの重複排除に加えて、ドキュメント内の重複コンテンツも処理する必要があります。一般的なケースとしては、Web ページ上で繰り返し表示されるナビゲーション バー、ヘッダー、フッターが挙げられます。 JavaScript レンダリングの問題によるコンテンツの重複。特定の CMS システムによって生成される、テンプレート化された重複した段落。

ドキュメント内の重複排除戦略は比較的単純です。ドキュメントを段落または固定長のチャンクに分割し、各チャンクのハッシュを計算し、重複するチャンクを削除します。

```python
def remove_duplicate_paragraphs(text: str, min_length: int = 50) -> str:
    """Remove duplicate paragraphs within document"""
    paragraphs = text.split('\n\n')
    seen_hashes = set()
    unique_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if len(para) < min_length:
            unique_paragraphs.append(para)
            continue
        
        para_hash = hashlib.md5(para.encode()).hexdigest()
        if para_hash not in seen_hashes:
            seen_hashes.add(para_hash)
            unique_paragraphs.append(para)
    
    return '\n\n'.join(unique_paragraphs)

def remove_duplicate_ngrams(text: str, n: int = 10, threshold: int = 3) -> str:
    """Remove high-frequency duplicate n-grams within document"""
    words = text.split()
    ngram_counts = Counter()
    
    # Compute n-gram frequencies
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        ngram_counts[ngram] += 1
    
    # Mark positions to remove
    remove_positions = set()
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        if ngram_counts[ngram] >= threshold:
            # Keep first occurrence, remove subsequent duplicates
            for j in range(i + n, len(words) - n + 1):
                if tuple(words[j:j + n]) == ngram:
                    for k in range(j, min(j + n, len(words))):
                        remove_positions.add(k)
    
    # Rebuild text
    result_words = [w for i, w in enumerate(words) if i not in remove_positions]
    return ' '.join(result_words)
```

---

## 4.3 プライバシー データのクリーニング (PII の削除)

トレーニング前のデータには、電子メール アドレス、電話番号、ID 番号、銀行カード番号、自宅の住所などの個人を特定できる情報 (PII) が必然的に含まれます。今日のデータ コンプライアンス要件 (GDPR、CCPA、個人情報保護法など) がますます厳しくなっているため、PII のクリーニングは道義的責任だけでなく、法的義務でもあります。

### 4.3.1 PII の種類とリスク

PII は、直接識別子と準識別子に分類できます。直接識別子は、名前、ID 番号、社会保障番号、電話番号、電子メール アドレスなど、単独で個人を識別できます。準識別子だけでは個人を特定することは困難ですが、生年月日、郵便番号、職業、雇用主などを組み合わせると個人特定につながる可能性があります。

PII をトレーニング前データに保持すると、複数のリスクが伴います。 1 つ目はプライバシー漏洩のリスクです。モデルはトレーニング データ内の機密情報を「記憶」し、推論中に悪意を持って抽出される可能性があります。 2 つ目はコンプライアンスのリスクです。データ保護規制に違反すると、巨額の罰金が科せられる可能性があります。最後に評判リスクです。モデルが他人の個人情報を出力すると、企業のイメージに重大なダメージを与えます。

![図 4-4: PII の種類とリスク](../../images/part2/图4_4_PII类型与风险.png)

*図 4-4: PII の種類とリスク レベル — 直接識別子 (高リスク) と準識別子 (中リスク) の分類*

### 4.3.2 Microsoft Presidio

Presidio は、Microsoft のオープンソース PII 検出および匿名化ツールキットであり、複数の言語と PII タイプをサポートしています。これは、2 つのコア コンポーネントを備えたモジュラー設計を使用しています。アナライザーはテキスト内の PII エンティティの識別を担当し、アノニマイザーは識別された PII の処理 (置換、マスキング、削除など) を担当します。

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Initialize engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def analyze_pii(text: str, language: str = 'en') -> list:
    """
    Identify PII in text
    
    Returns:
        List of PII entities with type, position, and confidence
    """
    results = analyzer.analyze(
        text=text,
        language=language,
        entities=[
            'EMAIL_ADDRESS', 'PHONE_NUMBER', 'CREDIT_CARD',
            'IP_ADDRESS', 'PERSON', 'LOCATION', 'DATE_TIME'
        ]
    )
    return results

def anonymize_pii(text: str, language: str = 'en') -> str:
    """
    Anonymize PII in text
    
    Replace identified PII with placeholders
    """
    # First identify
    analyzer_results = analyzer.analyze(text=text, language=language)
    
    # Define anonymization strategy
    operators = {
        'EMAIL_ADDRESS': OperatorConfig('replace', {'new_value': '<EMAIL>'}),
        'PHONE_NUMBER': OperatorConfig('replace', {'new_value': '<PHONE>'}),
        'CREDIT_CARD': OperatorConfig('replace', {'new_value': '<CREDIT_CARD>'}),
        'IP_ADDRESS': OperatorConfig('replace', {'new_value': '<IP>'}),
        'PERSON': OperatorConfig('replace', {'new_value': '<PERSON>'}),
        'LOCATION': OperatorConfig('replace', {'new_value': '<LOCATION>'}),
        'DATE_TIME': OperatorConfig('keep', {})  # Dates/times can usually be kept
    }
    
    # Anonymize
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators=operators
    )
    
    return anonymized.text
```

### 4.3.3 中国の PII の処理

Presidio では中国語のサポートが比較的限定されています。中国語の事前トレーニング データの場合は、通常、ルールベースのマッチングを正規表現で補う必要があります。

```python
import re

class ChinesePIIFilter:
    """Chinese PII filter"""
    
    patterns = {
        'phone': [
            r'1[3-9]\d{9}',  # Mobile number
            r'0\d{2,3}-?\d{7,8}',  # Landline
        ],
        'id_card': [
            r'\d{17}[\dXx]',  # 18-digit ID
            r'\d{15}',  # 15-digit ID
        ],
        'email': [
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        ],
        'bank_card': [
            r'\d{16,19}',  # Bank card number (needs context judgment)
        ],
        'ip_address': [
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
        ],
        'qq': [
            r'[Qq][Qq][:：]?\s*\d{5,11}',
            r'[Qq][:：]?\s*\d{5,11}',
        ],
        'wechat': [
            r'[Vv][Xx][:：]?\s*[a-zA-Z0-9_-]{6,20}',
            r'微信[:：]?\s*[a-zA-Z0-9_-]{6,20}',
        ],
    }
    
    def __init__(self):
        self.compiled_patterns = {}
        for pii_type, patterns in self.patterns.items():
            self.compiled_patterns[pii_type] = [
                re.compile(p) for p in patterns
            ]
    
    def find_pii(self, text: str) -> list:
        """Find all PII"""
        findings = []
        for pii_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    findings.append({
                        'type': pii_type,
                        'value': match.group(),
                        'start': match.start(),
                        'end': match.end()
                    })
        return findings
    
    def anonymize(self, text: str) -> str:
        """Anonymize PII"""
        findings = self.find_pii(text)
        
        # Process in reverse order by position to avoid affecting subsequent positions
        findings.sort(key=lambda x: x['start'], reverse=True)
        
        for finding in findings:
            placeholder = f"<{finding['type'].upper()}>"
            text = text[:finding['start']] + placeholder + text[finding['end']:]
        
        return text
```

### 4.3.4 PII 処理戦略のトレードオフ

PII 処理は、精度と再現率の間のトレードオフに直面します。過度に積極的なフィルタリングは、通常のコンテンツを誤って削除する可能性があります（たとえば、通常の番号シーケンスを電話番号として誤認するなど）。一方、過度に保守的なフィルタリングは、真の機密情報を見逃す可能性があります。

実際には、層別戦略が推奨されます。リスクの高い PII (ID 番号、銀行カード番号など) の場合は、より厳密な一致ルールを使用します。見逃すよりは過剰にフィルターする方が良いでしょう。中リスクの PII (電話番号、電子メールなど) の場合は、中程度のしきい値を使用して精度と再現率のバランスをとります。リスクの低い情報 (日付、場所など) については、特定のシナリオに基づいて処理するかどうかを決定します。

もう 1 つの重要な決定は、代替戦略です。一般的な選択肢としては、次のものが挙げられます。完全な削除。単純ですが、文章の流暢性が損なわれる可能性があります。プレースホルダー置換 (`<EMAIL>` など) が修正され、意味情報は保持されますが、不自然なパターンが生じる可能性があります。ランダム生成置換 (例: 実際の電子メールをランダムな電子メールに置き換える)。元の配布に最も近いですが、実装が複雑です。ほとんどの事前トレーニング データセットは、精度と複雑さのバランスをとるためにプレースホルダー置換を使用します。

---

## 4.4 完全なクリーニング パイプライン

上記で紹介したコンポーネントを接続して、完全なデータ クリーニング パイプラインを構築します。

### 4.4.1 パイプラインのアーキテクチャ

工業用グレードの洗浄パイプラインには通常、次のステージが含まれており、順番に実行されます。

**フェーズ 1: 形式の標準化。** さまざまなソースからのデータを統一形式に変換し、エンコードの問題を処理し、必要なメタデータを抽出します。

**フェーズ 2: 言語フィルタリング** 言語検出に FastText を使用し、ドキュメントをターゲット言語で保持します。混合言語の文書の場合は、主言語ごとに分類します。

**フェーズ 3: ヒューリスティック フィルタリング** 長さ、特殊文字、重複行などに関するヒューリスティック ルールを適用して、明らかに低品質のコンテンツを迅速にフィルタリングします。

**フェーズ 4: ドキュメント内の重複排除。** ドキュメント内の重複した段落と N グラムを削除します。

**フェーズ 5: PII のクリーニング** 機密の個人情報を特定し、匿名化します。

**フェーズ 6: 品質スコアリング。** 複雑さなどの品質指標を計算して、その後の品質階層化の基礎を提供します。

**フェーズ 7: ドキュメント間の重複排除。** MinHash LSH を使用して大規模なファジー重複排除を行い、類似性の高いドキュメントを削除します。

**フェーズ 8: 品質の階層化とサンプリング。** 品質スコアによってデータを階層化し、各層のサンプリングの重みを決定します。

```python
import ray
from dataclasses import dataclass
from typing import Optional

@dataclass
class CleaningConfig:
    """Cleaning configuration"""
    target_language: str = 'zh'
    min_length: int = 200
    max_length: int = 100000
    max_perplexity: float = 500
    dedup_threshold: float = 0_8
    anonymize_pii: bool = True

class DataCleaningPipeline:
    def __init__(self, config: CleaningConfig):
        self.config = config
        self.lang_filter = LanguageFilter(config.target_language)
        self.heuristic_filter = HeuristicFilter()
        self.perplexity_filter = PerplexityFilter(max_ppl=config.max_perplexity)
        self.pii_filter = ChinesePIIFilter() if config.target_language == 'zh' else None
        self.deduplicator = MinHashLSH(threshold=config.dedup_threshold)
    
    def process_document(self, doc: dict) -> Optional[dict]:
        """Process single document"""
        text = doc.get('text', '')
        
        # Phase 2: Language filtering
        lang, conf = self.lang_filter.detect(text)
        if lang != self.config.target_language:
            return None
        
        # Phase 3: Heuristic filtering
        passed, reason = self.heuristic_filter.filter(text)
        if not passed:
            return None
        
        # Phase 4: Intra-document deduplication
        text = remove_duplicate_paragraphs(text)
        
        # Phase 5: PII cleaning
        if self.config.anonymize_pii and self.pii_filter:
            text = self.pii_filter.anonymize(text)
        
        # Phase 6: Quality scoring
        perplexity = self.perplexity_filter.compute_perplexity(text)
        if perplexity > self.config.max_perplexity:
            return None
        
        return {
            **doc,
            'text': text,
            'language': lang,
            'lang_confidence': conf,
            'perplexity': perplexity
        }
    
    def run(self, input_path: str, output_path: str):
        """Run complete pipeline"""
        # Read data
        ds = ray.data.read_parquet(input_path)
        
        # Phases 1-6: Single document processing
        ds = ds.map(self.process_document)
        ds = ds.filter(lambda x: x is not None)
        
        # Phase 7: Inter-document deduplication
        ds = self.deduplicator.deduplicate(ds)
        
        # Save results
        ds.write_parquet(output_path)
```

### 4.4.2 品質のモニタリングと反復

パイプラインのクリーニングは 1 回限りのタスクではなく、継続的な監視と反復的な最適化が必要なプロセスです。次の監視メカニズムが推奨されます。

**フィルタ レートのモニタリング**: 各ステージの統計フィルタ レート。ステージが突然大量のデータをフィルタリングして除外した場合は、しきい値の設定が不適切であるか、データ分布が変更されていることを示している可能性があります。

**サンプル検査**: フィルタリング ルールの精度を評価するための洗浄結果の定期的な手動検査。誤って削除された良好なサンプルと見逃された不良サンプルの両方に注意が必要です。

**下流のフィードバック**: トレーニング後のモデルの評価結果が最終的な品質検証となります。モデルのパフォーマンスが低い場合は、データに問題があるかどうかを遡って分析する必要があります。

---

## 4.5 章の概要

この章では、トレーニング前データ クリーニングのコア テクノロジーを体系的に紹介しました。

ヒューリスティック フィルタリング: 言語検出では、FastText を使用してターゲット言語ドキュメントを迅速にフィルタリングします。パープレキシティ フィルタリングは KenLM を使用してテキストの品質を評価します。ヒューリスティック ルール セットは、長さ、特殊文字、重複行、語彙の多様性などの複数の側面をカバーします。品質階層化戦略はデータをさまざまな層に分割し、その後のサンプリングの基礎を提供します。

大規模な重複排除では、正確な重複排除ではハッシュ方式を使用して同一のドキュメントを迅速に削除します。ファジー重複排除では、MinHash LSH アルゴリズムを使用して、類似性の高いコンテンツを識別します。 TB 規模のデータには分散実装が必要です。ドキュメント内の重複排除は、段落および N グラム レベルの重複を処理します。

プライバシー クリーニング: PII 検出では、Presidio またはカスタム正規表現ルールを使用できます。匿名化戦略では、精度と情報保持の間のトレードオフが必要です。中国の PII の処理には、特別に設計されたルール セットが必要です。

完全なクリーニング パイプラインは各コンポーネントを接続し、形式の標準化、言語フィルタリング、ヒューリスティック フィルタリング、ドキュメント内重複排除、PII クリーニング、品質スコアリング、ドキュメント間重複排除、品質階層化の順序で実行します。継続的な品質監視と反復的な最適化がデータ品質を確保する鍵となります。

![図 4-5: 章の知識構造](../../images/part2/图4_5_本章知识结构.png)

*図 4-5: 第 4 章 知識構造 — 3 つの主要テーマ: ヒューリスティック フィルタリング、大規模な重複排除、PII クリーニング*

---

## さらに読む

データ クリーニングに関する詳細な内容については、次のリソースを参照する価値があります。

RefinedWeb の論文には、Common Crawl から高品質の事前トレーニング セットを構築するための完全なクリーニング フローが文書化されています。 Dolma データセットの技術レポートでは、Allen AI のクリーニング戦略とツールが紹介されています。 text-dedup オープンソース ライブラリ (github.com/ChenghaoMou/text-dedup) は、さまざまな重複排除アルゴリズムの実装を提供します。 Microsoft Presidio ドキュメント (microsoft.github.io/presidio) は、PII 処理に関する信頼できるリファレンスです。 CCNet の論文では、Facebook の共通クロール データ処理方法、特にパープレキシティ フィルタリングの詳細が紹介されています。

---

## 次の章のプレビュー

次の章「トークン化とシリアル化」では、トレーニング前のデータ準備における最後の重要なステップ、つまり、クリーンなテキストをモデルが理解できるトークン シーケンスに変換する方法について説明します。 BPE、WordPiece、Unigram などのトークン化アルゴリズムの原理と選択について学びます。特定の分野の語彙を拡張する方法。データミキシングとカリキュラム学習のサンプリング戦略。

次の章に入るときに、次の質問を検討してください。コードに特化したモデルをトレーニングしている場合、標準の GPT-2 トークナイザーはどのような問題に遭遇するでしょうか?
