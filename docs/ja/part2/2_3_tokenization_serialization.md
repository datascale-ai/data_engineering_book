# 第 5 章: トークン化とシリアル化

---

## 章の概要

トークン化は、生のテキストとニューラル ネットワークを接続する架け橋です。クリーニング後の高品質のコーパスは、トレーニングのために Transformers にフィードする前に、モデルが理解できる数値シーケンスに変換する必要があります。この章では、3 つの主流アルゴリズムである BPE、WordPiece、Unigram を含むトークナイザーの仕組みについて詳しく説明します。特定のドメインの語彙を構築および拡張する方法を紹介します。最後に、トレーニング中のさまざまなデータ タイプの表示順序と比率を決定するデータ混合とカリキュラム学習戦略について説明します。

---

## シナリオの紹介

あなたのチームは、コードに特化した大規模なモデルをトレーニングしています。標準の GPT-2 トークナイザーで予備実験を行った後、奇妙な現象を発見しました。モデルは、生成されたコードのインデントでエラーを頻繁に起こし、4 つのスペースを複数の異なるトークンに分割し、インデントの不一致を引き起こします。さらに悪いことに、`def` や `return` などの一部の一般的なプログラミング キーワードは複数のサブワードに分割されており、モデルがその意味を理解するために追加のコンテキストを使用する必要があります。

分析後、問題はトークナイザーにあることがわかります。 GPT-2 トークナイザーは Web テキストでトレーニングされており、コードの特殊な構造 (空白、キャメルケース、特殊記号など) をうまく処理できません。コードタスク用に特化したトークナイザーを設計することは、モデルのパフォーマンスを向上させるための重要なステップとなっています。

この例は、トークナイザーは決して無視できる「前処理の詳細」ではなく、モデルの機能に大きな影響を与えることを示しています。

---

## 5.1 トークナイザーの原則

トークナイザーの中心的なタスクは、連続したテキスト文字列を個別のトークン シーケンスにセグメント化し、各トークンを整数 ID にマップすることです。この一見単純なタスクには、実際には複雑なアルゴリズム設計とエンジニアリングのトレードオフが含まれます。

### 5.1.1 なぜサブワードトークン化なのか?

深層学習の初期の頃、自然言語処理では通常、単語レベルまたは文字レベルのトークン化が使用されていました。単語レベルのトークン化では、完全な各単語がトークンとして扱われます。利点はセマンティクスが明確であることですが、欠点は膨大な語彙 (考えられるすべての単語をカバーする必要がある) と語彙外 (OOV) 単語を処理できないことです。文字レベルのトークン化では、各文字がトークンとして扱われます。利点は、OOV の問題がない小さな語彙であることですが、欠点は、シーケンスが長すぎるため、モデルが長距離の依存関係を把握することが困難になることです。

サブワードのトークン化は妥協です。テキストを単語より小さく、文字より大きい単位に分割します。高頻度の単語はそのまま残ります。低頻度の単語は、より小さなサブワード単位に分割されます。たとえば、「不幸」は「不」+「幸福」+「らしさ」に分割される可能性があります。このアプローチでは、語彙サイズを制御し、特定の意味情報を保持しながら、サブワードの組み合わせを通じて目に見えない語彙を処理できます。

![図 5-1: トークン化の粒度の比較](../../images/part2/图5_1_分词粒度对比.png)

*図 5-1: トークン化の粒度の比較 — 単語レベル、文字レベル、およびサブワード レベル間のトレードオフ*

現在、ほとんどすべての主流の大規模言語モデルはサブワード トークン化を使用しています。 GPT シリーズは BPE を使用し、BERT は WordPiece を使用し、T5 と LLaMA は SentencePiece (BPE と Unigram をサポート) を使用します。これらのアルゴリズムの原理を理解することは、トークナイザーのカスタマイズと最適化の基礎となります。

### 5.1.2 BPE: バイトペアのエンコーディング

BPE (バイト ペア エンコーディング) はもともとデータ圧縮アルゴリズムでしたが、後に Sennrich らによってニューラル機械翻訳に導入されました。 2015 年に最も広く使用されるサブワード トークン化アルゴリズムになりました。

BPE の核となる考え方は非常に直感的です。つまり、文字レベルから開始し、目標の語彙サイズに達するまで、最も頻繁に発生する隣接するトークンのペアを繰り返しマージします。具体的な手順:

1. すべてのトレーニング テキストを文字シーケンスに分割し、各文字を初期トークンとして使用します。
2. すべての隣接トークンペアの頻度をカウントします。
3. 最も頻度の高いトークンのペアを新しいトークンにマージします。
4. 語彙が目標サイズに達するまでステップ 2 ～ 3 を繰り返します。

以下は、簡略化された BPE トレーニングの実装です。

```python
from collections import Counter, defaultdict

def train_bpe(corpus: list, vocab_size: int) -> dict:
    """
    Train BPE tokenizer
    
    Args:
        corpus: Training corpus list
        vocab_size: Target vocabulary size
    
    Returns:
        Merge rules dictionary
    """
    # Initialize: split each word into characters, add end-of-word marker
    word_freqs = Counter()
    for text in corpus:
        for word in text.split():
            # Add end-of-word marker </w> to distinguish same character in middle vs end of word
            word_freqs[' '.join(list(word)) + ' </w>'] += 1
    
    merges = {}
    vocab = set()
    
    # Initial vocabulary is all characters
    for word in word_freqs:
        for char in word.split():
            vocab.add(char)
    
    while len(vocab) < vocab_size:
        # Count adjacent token pair frequencies
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            tokens = word.split()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq
        
        if not pair_freqs:
            break
        
        # Find highest-frequency pair
        best_pair = max(pair_freqs, key=pair_freqs.get)
        
        # Merge this pair
        new_token = best_pair[0] + best_pair[1]
        merges[best_pair] = new_token
        vocab.add(new_token)
        
        # Update word frequency table
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            new_word = word.replace(
                best_pair[0] + ' ' + best_pair[1], 
                new_token
            )
            new_word_freqs[new_word] = freq
        word_freqs = new_word_freqs
    
    return merges

def apply_bpe(text: str, merges: dict) -> list:
    """Apply BPE tokenization"""
    tokens = list(text) + ['</w>']
    
    while True:
        # Find mergeable pairs
        pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        merge_pair = None
        for pair in pairs:
            if pair in merges:
                merge_pair = pair
                break
        
        if merge_pair is None:
            break
        
        # Perform merge
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == merge_pair:
                new_tokens.append(merges[merge_pair])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    
    return tokens
```

BPE の重要なバリアントは、GPT-2 によって導入されたバイトレベル BPE です。従来の BPE は文字レベルで動作し、Unicode エンコードの問題を処理する必要があります。バイト レベルの BPE はバイト レベルで直接動作し、各バイトを印刷可能な文字にマッピングするため、エンコードの問題を回避し、あらゆる言語をネイティブにサポートします。 GPT シリーズがあらゆる言語のテキストを処理できるのはこのためです。

### 5.1.3 WordPiece: BERT の選択

WordPiece は、Google が BERT 用に開発したトークン化アルゴリズムで、BPE とよく似ていますが、マージ ペアの選択基準が主に異なります。

BPE は、最も頻繁に発生するペアをマージ用に選択します。 WordPiece は、トレーニング データの可能性を最大化するペアを選択します。具体的には、候補ペア (A、B) について、WordPiece はトレーニング データ上のマージされた語彙の言語モデル確率ゲインを計算し、マージ用に最大のゲインを持つペアを選択します。

実際には、これは WordPiece が「共起確率が独立した出現確率の積よりもはるかに高い」ペアをマージする傾向があることを意味します。この基準により、WordPiece は、頻度は低いが意味のあるパターンに対してより敏感になります。

WordPiece のもう 1 つの特徴は、`##` プレフィックスを使用して単語の先頭以外のサブワードを識別することです。たとえば、「playing」は ["play", "##ing"] としてトークン化される場合があります。この表現により、元の単語内のサブワードの位置が明確に区別され、モデルが語彙構造を理解するのに役立ちます。

```python
# WordPiece tokenization example (using HuggingFace tokenizers)
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize WordPiece tokenizer
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Train
trainer = WordPieceTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Use
output = tokenizer.encode("unhappiness")
print(output.tokens)  # ['un', '##happi', '##ness']
```

### 5.1.4 Unigram: トークン化に関する確率論的な観点

Unigramトークン化は2018年に工藤氏によって提案され、BPE/WordPieceとは全く異なるアプローチを採用しています。 BPE と WordPiece はボトムアップ方式であり、小さな単位から開始して、徐々に大きな単位に統合します。 Unigram はトップダウンで、考えられるすべてのサブワードを含む大きな語彙から開始し、徐々に目標サイズまで枝刈りしていきます。

Unigram はトークン化を確率の問題としてモデル化します。各トークンの語彙 V と確率 P(t) が与えられると、テキストのトークン化の結果は、総確率を最大化するセグメンテーションになります。

$$P(x_1, x_2, ..., x_n) = \prod_{i=1}^{n} P(x_i)$$

トレーニング プロセスでは EM アルゴリズムが使用されます。E ステップは、現在の語彙の下で各トークンの予想出現数を計算します。 M ステップはトークンの確率を更新します。次に、目標語彙サイズに達するまで、削除による総尤度への影響が最小限に抑えられるトークンを削除します。

Unigram の独自の利点は、複数のトークン化結果の確率モデリングを自然にサポートしていることです。特定のテキストに対して、複数の有効な分割方法が存在する可能性があります。 Unigram はそれぞれに確率を割り当てることができます。これは、特定のアプリケーション シナリオ (音声認識における複数の仮説処理など) で非常に役立ちます。

### 5.1.5 3 つのアルゴリズムの比較

3 つの主流のサブワード トークン化アルゴリズムには、それぞれ特徴があります。選択には、特定のシナリオに基づいたトレードオフが必要です。

|アルゴリズム |コアアイデア |利点 |デメリット |代表的なアプリケーション |
|------|----------|------|------|----------|
| BPE |ボトムアップ、周波数駆動のマージ |シンプルで直感的な、迅速なトレーニング |貪欲な戦略は最適ではないかもしれない | GPTシリーズ、LLaMA |
|ワードピース |ボトムアップ、可能性主導のマージ |低周波の意味のあるパターンに敏感 |計算の複雑さの増加 | BERT、蒸留BERT |
|ユニグラム |トップダウンの確率モデリング |理論的にはエレガントで、複数のセグメンテーションをサポートします。ゆっくりとしたトレーニング | T5、mT5、アルバート |

![図 5-2: トークン化アルゴリズムの比較](../../images/part2/图5_2_分词算法对比.png)

*図 5-2: BPE、WordPiece、および Unigram トークン化アルゴリズムの比較*

実際のエンジニアリングでは、SentencePiece が最も一般的に使用されるトークン化ツールキットです。 BPE アルゴリズムと Unigram アルゴリズムの両方をサポートし、言語に依存しない前処理 (空間ベースのトークン化に依存しない) を提供し、主流の深層学習フレームワークとシームレスに統合します。

```python
import sentencepiece as spm

# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='my_tokenizer',
    vocab_size=32000,
    model_type='bpe',  # or 'unigram'
    character_coverage=0_9995,
    num_threads=16
)

# Load and use
sp = spm.SentencePieceProcessor(model_file='my_tokenizer.model')
tokens = sp.encode('Hello, world!', out_type=str)
print(tokens)  # ['▁Hello', ',', '▁world', '!']
ids = sp.encode('Hello, world!')
print(ids)  # [1234, 56, 789, 10]
```

---

## 5.2 語彙の設計と拡張

語彙はトークナイザーの中核コンポーネントです。語彙のサイズ、カバレッジ、および構造は、モデルのパフォーマンスと効率に直接影響します。

### 5.2.1 語彙サイズのトレードオフ

語彙のサイズは、トークナイザーの設計において最も重要なハイパーパラメーターの 1 つです。語彙が大きいほど、完全な単位として保持されるトークンが多くなり、シーケンスは短くなりますが、Embedding行列が大きくなり、パラメータが増加します。語彙が少ないほど、より多くの単語がサブワードに分割され、シーケンスが長くなりますが、モデル パラメーターは少なくなります。

主流の大規模モデルの語彙サイズは通常、32K から 128K の範囲です。 GPT-2 は 50,257、LLaMA は 32,000、GPT-4 は約 100,000 を使用すると報告されています。語彙サイズを選択するときは、次の点を考慮してください。

**計算効率**: 語彙が増えるほど、Embedding層と出力層のパラメーターも多くなります。語彙サイズ V の d 次元モデルの場合、Embedding行列には V × d パラメーターが含まれます。 V が 32K から 128K に増加すると、これは 4 倍に増加します。

**シーケンスの長さ**: 語彙が大きいほど、平均して各トークンがカバーする文字数が多くなり、同じテキストが分割されるトークンの数は少なくなります。 Transformer の計算の複雑さはシーケンスの長さが 2 次であるため、これは長いドキュメントの場合に特に重要です。

**希少語の処理**: 語彙が増えるほど、より多くの希少語を完全なトークンとして保持できるようになり、UNK や過剰セグメンテーションの問題が軽減されます。しかし、これはまた、まれなトークンではトレーニング サンプルが少なくなり、Embedding品質の低下につながる可能性があることも意味します。

```python
# Analyze impact of different vocabulary sizes on sequence length
def analyze_vocab_size_impact(text: str, vocab_sizes: list) -> dict:
    """Analyze impact of vocabulary size on tokenization results"""
    import sentencepiece as spm
    
    results = {}
    for vocab_size in vocab_sizes:
        # Train tokenizers with different vocabulary sizes
        spm.SentencePieceTrainer.train(
            input='corpus.txt',
            model_prefix=f'tokenizer_{vocab_size}',
            vocab_size=vocab_size,
            model_type='bpe'
        )
        
        sp = spm.SentencePieceProcessor(model_file=f'tokenizer_{vocab_size}.model')
        tokens = sp.encode(text)
        
        results[vocab_size] = {
            'num_tokens': len(tokens),
            'chars_per_token': len(text) / len(tokens),
            'compression_ratio': len(text.encode('utf-8')) / (len(tokens) * 2)
        }
    
    return results
```

### 5.2.2 多言語語彙の設計

多言語モデルをトレーニングする場合、語彙設計はさらなる課題に直面します。それは、限られた語彙空間内でさまざまな言語のカバレッジのバランスをどう取るかということです。

よくある問題は「語彙の呪い」です。トークナイザーが多言語コーパスで直接トレーニングされた場合、高リソース言語 (英語など) が語彙スペースのほとんどを占めることになりますが、低リソース言語のカバー範囲は著しく不十分です。これにより、低リソース言語のテキストが過剰にセグメント化され、シーケンスの長さが増大し、モデルのパフォーマンスが低下します。

これに対処するための一般的な戦略は次のとおりです。

**コーパスのバランス**: トークナイザーをトレーニングする前に、さまざまな言語からコーパスをオーバーサンプリングまたはアンダーサンプリングして、言語間で重みのバランスを高めます。

**温度サンプリング**: 第 3 章で説明した多言語データ バランシング戦略と同様に、温度パラメーターを使用して、さまざまな言語のサンプリング確率を制御します。

**言語固有の文字の範囲**: 頻度が低い場合でも、各ターゲット言語の基本的な文字セットが語彙に含まれていることを確認します。 SentencePiece は、これを制御する `character_coverage` パラメーターを提供します。

```python
# Multilingual tokenizer training example
import sentencepiece as spm

# Use character coverage to ensure multilingual support
spm.SentencePieceTrainer.train(
    input='multilingual_corpus.txt',
    model_prefix='multilingual_tokenizer',
    vocab_size=64000,
    model_type='unigram',
    character_coverage=0_9999,  # High coverage ensures rare characters included
    input_sentence_size=10000000,
    shuffle_input_sentence=True,
    # Special handling for CJK characters
    byte_fallback=True  # Fall back to byte level for unknown characters
)
```

### 5.2.3 ドメイン固有の語彙拡張

事前トレーニングされたモデルを特定のドメイン (医療、法律、コードなど) に適用すると、ドメイン用語が過度にセグメント化されるという問題によく遭遇します。これはシーケンスの長さが増加するだけでなく、モデルの専門的な概念の理解にも影響を与える可能性があります。

語彙の拡張は効果的な解決策です。基本的な考え方: 元の語彙を保持しながら、新しいドメイン固有のトークンを追加します。

```python
from transformers import AutoTokenizer

def extend_tokenizer(base_tokenizer_name: str, 
                     domain_terms: list,
                     output_dir: str) -> None:
    """
    Extend pre-trained tokenizer vocabulary
    
    Args:
        base_tokenizer_name: Base tokenizer name
        domain_terms: List of domain-specific terms
        output_dir: Output directory
    """
    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
    
    print(f"Original vocabulary size: {len(tokenizer)}")
    
    # Filter already existing tokens
    new_tokens = []
    for term in domain_terms:
        if term not in tokenizer.get_vocab():
            new_tokens.append(term)
    
    # Add new tokens
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added} new tokens")
    print(f"New vocabulary size: {len(tokenizer)}")
    
    # Save extended tokenizer
    tokenizer.save_pretrained(output_dir)
    
    return tokenizer

# Example: Extend vocabulary for medical domain
medical_terms = [
    '冠状动脉',
    '心肌梗死',
    '动脉粥样硬化',
    'COVID-19',
    'mRNA疫苗',
    '计算机断层扫描',
    # ... more terms
]

tokenizer = extend_tokenizer(
    'meta-llama/Llama-2-7b',
    medical_terms,
    './medical_tokenizer'
)
```

語彙拡張後は、それに応じてモデルのEmbedding行列を拡張する必要があります。新しいトークンのEmbeddingは通常、ランダムな値または既存の関連トークンの平均に初期化され、意味のある表現を得るために継続的な事前トレーニングを通じて学習されます。

```python
from transformers import AutoModelForCausalLM

def resize_model_embeddings(model_name: str, 
                            tokenizer,
                            output_dir: str) -> None:
    """Resize model embedding layer to match extended vocabulary"""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Resize embedding layer
    model.resize_token_embeddings(len(tokenizer))
    
    # Optional: Initialize new embeddings with mean of similar tokens
    # This usually gives better results than random initialization
    
    model.save_pretrained(output_dir)
```

### 5.2.4 語彙設計のベストプラクティス

業界の経験に基づいて、語彙設計のベスト プラクティスをいくつか示します。

**十分な特殊トークンの位置を予約**: 将来の特殊トークン (新しい制御シンボル、ドメイン マーカーなど) のためにいくつかのトークン ID を予約します。多くのトークナイザーは 100 ～ 1000 のポジションを予約します。

**数値とコード記号の適切な分割を確保する**: 数値は多くのタスクで重要ですが、標準のトークナイザーでは数値の処理が不十分なことがよくあります。 1 桁を独立したトークンとして保持するか、特別な数値エンコード戦略を使用することを検討してください。

**エッジ ケースをテストする**: 語彙を完成させる前に、非常に長い単語、特殊文字、混合言語テキスト、コード スニペットなど、さまざまなエッジ ケースをテストします。トークン化の結果が期待どおりであることを確認します。

**語彙に関する決定を文書化する**: 語彙サイズ、トレーニング コーパス、特別なトークン リストなどを記録し、その後のモデルの反復とトラブルシューティングを容易にします。

---

## 5.3 データ混合とカリキュラム学習

トークナイザーを決定したら、次の重要な質問は、トレーニング データをどのように整理して提示するかということです。さまざまなソースや品質からのデータをどのような割合で混合する必要がありますか?トレーニング中のデータの順序は重要ですか?

### 5.3.1 データ混合戦略

第 3 章で説明したように、高品質の事前トレーニング データセットは通常、Web、書籍、コード、論文、対話などの複数のソースを混合します。各ソースはデータ量と品質が異なります。単に元の比率で混合するだけでは最適ではないことがよくあります。

**静的混合**は最も単純な戦略です。トレーニングを開始する前に各ソースの混合比率を決定し、データをシャッフルしてから、順番にトレーニングします。この方法は実装が簡単ですが、柔軟性に欠けます。

```python
# Static data mixing example
import random

def static_mix(data_sources: dict, target_size: int) -> list:
    """
    Statically mix multiple data sources
    
    Args:
        data_sources: {source_name: (data_list, weight)}
        target_size: Target dataset size
    
    Returns:
        Mixed data list
    """
    mixed_data = []
    
    # Compute sample count for each source
    total_weight = sum(w for _, w in data_sources.values())
    
    for source_name, (data, weight) in data_sources.items():
        num_samples = int(target_size * weight / total_weight)
        
        # If insufficient data, repeat sampling
        if len(data) < num_samples:
            sampled = random.choices(data, k=num_samples)
        else:
            sampled = random.sample(data, num_samples)
        
        mixed_data.extend(sampled)
    
    random.shuffle(mixed_data)
    return mixed_data

# Usage example
data_sources = {
    'web': (web_data, 0_6),
    'books': (book_data, 0_15),
    'code': (code_data, 0_1),
    'papers': (paper_data, 0_1),
    'wikipedia': (wiki_data, 0_05)
}

mixed = static_mix(data_sources, target_size=1000000)
```

![図 5-3: データ混合戦略](../../images/part2/图5_3_数据混合策略.png)

*図 5-3: 静的ミキシング戦略と動的ミキシング戦略の比較*

**ダイナミックミキシング**により、トレーニング中に混合比率を調整できます。一部の研究では、トレーニング段階ごとに最適なデータ比率が異なる可能性があることが示唆されています。たとえば、より多様なデータを使用して早期にトレーニングすると、モデルが広範な言語理解を確立するのに役立ちます。高品質データの割合を増やして後でトレーニングすると、きめ細かい機能が向上します。

```python
class DynamicDataMixer:
    """Dynamic data mixer"""
    
    def __init__(self, data_sources: dict, schedule: list):
        """
        Initialize dynamic mixer
        
        Args:
            data_sources: Data source dictionary
            schedule: [(step_threshold, weights_dict), ...]
                     Use different mixing weights at different training steps
        """
        self.data_sources = data_sources
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.current_step = 0
    
    def get_weights(self) -> dict:
        """Get weights for current step"""
        for step_threshold, weights in reversed(self.schedule):
            if self.current_step >= step_threshold:
                return weights
        return self.schedule[0][1]
    
    def sample_batch(self, batch_size: int) -> list:
        """Sample one batch"""
        weights = self.get_weights()
        batch = []
        
        for source_name, weight in weights.items():
            num_samples = int(batch_size * weight)
            data = self.data_sources[source_name]
            batch.extend(random.choices(data, k=num_samples))
        
        random.shuffle(batch)
        self.current_step += 1
        return batch[:batch_size]

# Usage example: Emphasize diversity early, quality later
schedule = [
    (0, {'web': 0_5, 'books': 0_2, 'code': 0_15, 'papers': 0_1, 'wiki': 0_05}),
    (100000, {'web': 0_4, 'books': 0_25, 'code': 0_15, 'papers': 0_15, 'wiki': 0_05}),
    (500000, {'web': 0_3, 'books': 0_3, 'code': 0_2, 'papers': 0_15, 'wiki': 0_05}),
]

mixer = DynamicDataMixer(data_sources, schedule)
```

### 5.3.2 カリキュラム学習

カリキュラム学習は、人間の学習にヒントを得たトレーニング戦略です。中心となるアイデアは、モデルに最初に「簡単な」サンプルを学習させ、次に「難しい」サンプルに徐々に移行させることです。この戦略は、収束を加速し、最終的なパフォーマンスを向上させることが複数の研究で証明されています。

トレーニング前のシナリオでは、「簡単」と「難しい」は複数の方法で定義できます。

**長さベース**: 通常、長いテキストよりも短いテキストの方が学習しやすいです。カリキュラムは短いシーケンスから始めて、徐々に長さを増やしていくことができます。

**難度ベース**: 難度の低いテキスト (言語モデルがより「慣れている」テキスト) は、「簡単な」サンプルとみなすことができます。小さな事前トレーニング済みモデルを使用してサンプルの難易度を評価し、メイン モデルのサンプルを難易度順に並べることができます。

**ノイズ レベル ベース**: 最初は高品質でノイズの少ないテキストが使用され、次に徐々に低品質だが固有の情報が含まれる可能性のあるテキストが導入されます。

```python
import numpy as np

class CurriculumScheduler:
    """Curriculum learning scheduler"""
    
    def __init__(self, 
                 data: list, 
                 difficulty_scores: list,
                 total_steps: int,
                 strategy: str = 'linear'):
        """
        Initialize curriculum scheduler
        
        Args:
            data: Data list
            difficulty_scores: Difficulty score for each sample (higher = harder)
            total_steps: Total training steps
            strategy: Curriculum strategy ('linear', 'sqrt', 'exp')
        """
        self.data = np.array(data)
        self.difficulty_scores = np.array(difficulty_scores)
        self.total_steps = total_steps
        self.strategy = strategy
        
        # Sort by difficulty
        sorted_indices = np.argsort(self.difficulty_scores)
        self.sorted_data = self.data[sorted_indices]
        self.sorted_scores = self.difficulty_scores[sorted_indices]
    
    def get_curriculum_fraction(self, current_step: int) -> float:
        """
        Compute data fraction to use at current step
        
        Return value in [0, 1], indicating proportion of easiest data to use
        """
        progress = current_step / self.total_steps
        
        if self.strategy == 'linear':
            return progress
        elif self.strategy == 'sqrt':
            return np.sqrt(progress)
        elif self.strategy == 'exp':
            return 1 - np.exp(-3 * progress)
        else:
            return progress
    
    def sample_batch(self, current_step: int, batch_size: int) -> list:
        """Sample batch according to current progress"""
        fraction = self.get_curriculum_fraction(current_step)
        
        # Determine available data range
        available_size = max(int(len(self.sorted_data) * fraction), batch_size)
        available_data = self.sorted_data[:available_size]
        
        # Random sample from available range
        indices = np.random.choice(len(available_data), size=batch_size, replace=True)
        return available_data[indices].tolist()
```

![図 5-4: カリキュラム学習の図](../../images/part2/图5_4_课程学习示意图.png)

*図 5-4: カリキュラムの学習原則 — 簡単なサンプルから難しいサンプルへの段階的な移行*

### 5.3.3 データのサンプリングとバッチ構築

実際のトレーニングでは、データの編成方法が効率と有効性の両方に影響します。エンジニアリング上の重要な考慮事項をいくつか示します。

**パック戦略**: コンピューティング リソースを最大限に活用するには、通常、複数の短いシーケンスが固定長シーケンスにパックされます。これにより、パディングによる計算の無駄が削減されます。重要な問題は、梱包後に注意マスクをどのように扱うかです。異なる書類が相互に影響を与えるべきではありません。

```python
def pack_sequences(sequences: list, max_length: int, eos_token_id: int) -> list:
    """
    Pack multiple short sequences to fixed length
    
    Args:
        sequences: List of token id sequences
        max_length: Target sequence length
        eos_token_id: End-of-sequence token ID
    
    Returns:
        List of packed sequences, each of length max_length
    """
    packed = []
    current_pack = []
    current_length = 0
    
    for seq in sequences:
        seq_with_eos = seq + [eos_token_id]
        
        if current_length + len(seq_with_eos) <= max_length:
            current_pack.extend(seq_with_eos)
            current_length += len(seq_with_eos)
        else:
            # Current pack full, start new one
            if current_pack:
                # Pad to max_length
                current_pack.extend([eos_token_id] * (max_length - current_length))
                packed.append(current_pack)
            
            current_pack = seq_with_eos
            current_length = len(seq_with_eos)
    
    # Handle last pack
    if current_pack:
        current_pack.extend([eos_token_id] * (max_length - current_length))
        packed.append(current_pack)
    
    return packed
```

**ドキュメント境界の処理**: シーケンスをパッキングするときは、生成中にモデルがドキュメント境界を越えて注意を実行しないようにするために、「ドキュメント境界マスク」を作成する必要があります。

**データ読み込み効率**: TB 規模のデータセットの場合、データ読み込み自体がボトルネックになる可能性があります。一般的な最適化方法には、前処理されたデータをバイナリ形式で保存する (numpy memmap など)、マルチプロセスの並列ロード、次のバッチのプリフェッチが含まれます。

### 5.3.4 シリアル化とストレージ形式

トークン化後、トレーニング中に高速に読み取るために、トークン シーケンスを効率的な形式で保存する必要があります。

**一般的なストレージ形式**には次のものが含まれます。

**NumPy memmap**: トークン ID を numpy 配列として保存し、メモリ マッピング経由でアクセスします。利点はシンプルかつ直接的であり、ランダム アクセスをサポートします。欠点は、圧縮をサポートしていないこと、ストレージが大きいことです。

```python
import numpy as np

def save_as_memmap(token_ids: list, output_path: str):
    """Save token ID list as memmap format"""
    arr = np.array(token_ids, dtype=np.uint16)  # Assume vocabulary < 65536
    fp = np.memmap(output_path, dtype='uint16', mode='w+', shape=arr.shape)
    fp[:] = arr[:]
    fp.flush()
    
def load_memmap(path: str, shape: tuple):
    """Load memmap format token IDs"""
    return np.memmap(path, dtype='uint16', mode='r', shape=shape)
```

**Arrow/Parquet**: Apache Arrow 形式を使用し、圧縮と効率的な列指向アクセスをサポートします。 HuggingFace データセットは内部でこの形式を使用します。

**カスタム バイナリ形式**: 一部の大規模プロジェクトでは、特定のアクセス パターンに最適化されたカスタム バイナリ形式が使用されます。たとえば、GPT-NeoX で使用されるバイナリ パッキング形式。

```python
# Use HuggingFace Datasets for tokenized data
from datasets import Dataset

def tokenize_and_save(raw_data: list, tokenizer, output_dir: str):
    """Tokenize and save as Datasets format"""
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=2048,
            return_attention_mask=False
        )
    
    # Create Dataset
    ds = Dataset.from_dict({'text': raw_data})
    
    # Tokenize
    tokenized_ds = ds.map(
        tokenize_function,
        batched=True,
        num_proc=16,
        remove_columns=['text']
    )
    
    # Save
    tokenized_ds.save_to_disk(output_dir)
```

---

## 5.4 完全なデータ準備パイプライン

上で説明したステップを接続して、生のテキストからトレーニングの準備ができたデータまでの完全なパイプラインを構築します。

```python
from dataclasses import dataclass
from typing import Optional
import sentencepiece as spm

@dataclass
class DataPrepConfig:
    """Data preparation configuration"""
    # Tokenizer config
    tokenizer_path: str
    max_seq_length: int = 2048
    
    # Data mixing config
    mix_weights: dict = None  # {source: weight}
    
    # Curriculum learning config
    use_curriculum: bool = False
    curriculum_strategy: str = 'linear'
    
    # Output config
    pack_sequences: bool = True
    output_format: str = 'arrow'  # 'arrow', 'memmap', 'jsonl'

class DataPreparationPipeline:
    """Data preparation pipeline"""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
        self.tokenizer = spm.SentencePieceProcessor(model_file=config.tokenizer_path)
    
    def tokenize_document(self, text: str) -> list:
        """Tokenize single document"""
        return self.tokenizer.encode(text)
    
    def process_source(self, source_path: str, source_name: str) -> list:
        """Process single data source"""
        documents = self.load_documents(source_path)
        
        tokenized = []
        for doc in documents:
            tokens = self.tokenize_document(doc['text'])
            if len(tokens) > 10:  # Filter too-short documents
                tokenized.append({
                    'input_ids': tokens,
                    'source': source_name,
                    'length': len(tokens)
                })
        
        return tokenized
    
    def mix_sources(self, sources: dict) -> list:
        """Mix multiple data sources"""
        mixed = []
        weights = self.config.mix_weights or {s: 1_0 for s in sources}
        total_weight = sum(weights.values())
        
        # Determine sample count per source
        total_samples = sum(len(data) for data in sources.values())
        
        for source_name, data in sources.items():
            weight = weights.get(source_name, 1_0) / total_weight
            num_samples = int(total_samples * weight)
            
            if len(data) >= num_samples:
                sampled = random.sample(data, num_samples)
            else:
                sampled = random.choices(data, k=num_samples)
            
            mixed.extend(sampled)
        
        random.shuffle(mixed)
        return mixed
    
    def pack_and_save(self, data: list, output_path: str):
        """Pack and save data"""
        if self.config.pack_sequences:
            sequences = [d['input_ids'] for d in data]
            packed = pack_sequences(
                sequences, 
                self.config.max_seq_length,
                self.tokenizer.eos_id()
            )
        else:
            packed = [d['input_ids'] for d in data]
        
        # Select output format based on config
        if self.config.output_format == 'arrow':
            self.save_as_arrow(packed, output_path)
        elif self.config.output_format == 'memmap':
            self.save_as_memmap(packed, output_path)
        else:
            self.save_as_jsonl(packed, output_path)
    
    def run(self, source_paths: dict, output_path: str):
        """Run complete pipeline"""
        # 1. Process each data source
        sources = {}
        for source_name, path in source_paths.items():
            print(f"Processing {source_name}...")
            sources[source_name] = self.process_source(path, source_name)
        
        # 2. Mix data
        print("Mixing data sources...")
        mixed = self.mix_sources(sources)
        
        # 3. Optional: Apply curriculum ordering
        if self.config.use_curriculum:
            print("Applying curriculum ordering...")
            mixed = self.apply_curriculum(mixed)
        
        # 4. Pack and save
        print("Packing and saving...")
        self.pack_and_save(mixed, output_path)
        
        print(f"Done! Saved {len(mixed)} samples to {output_path}")
```

![図 5-5: 完全なデータ準備パイプライン](../../images/part2/图5_5_数据准备完整流水线.png)

*図 5-5: 生のテキストからトレーニング可能なデータまでの完全なパイプライン*

---

## 5.5 章の概要

この章では、トークン化とデータのシリアル化のコアテクノロジーを体系的に紹介しました。

トークナイザーの原則: サブワードのトークン化は、現在の大規模モデルの主流の選択肢であり、語彙サイズとシーケンスの長さのバランスが取れています。 BPE は、シンプルかつ効率的な周波数駆動のボトムアップ マージ戦略を使用します。 WordPiece は、低頻度の意味のあるパターンに対してより敏感な尤度主導の結合基準を使用します。 Unigram は、理論的にはよりエレガントなトップダウン確率モデリングを使用します。 SentencePiece は最も一般的に使用されるツールキットで、複数のアルゴリズムと言語に依存しない処理をサポートしています。

語彙の設計: 語彙のサイズには、計算効率、シーケンスの長さ、および稀な単語の処理の間のトレードオフが必要です。主流モデルは通常 32K ～ 128K を使用します。多言語語彙の設計では、言語間でカバレッジのバランスをとり、「語彙の呪い」を回避する必要があります。ドメイン固有の語彙拡張により、専門用語の処理が向上しますが、それに応じてモデルEmbedding層を拡張する必要があります。

データ混合の場合: 静的混合はシンプルかつ直接的です。ダイナミックミキシングにより、トレーニング中に比率を調整できます。カリキュラム学習戦略は簡単なサンプルから始まり、徐々に難しいサンプルに移行することで、収束を加速し、パフォーマンスを向上させることができます。大規模なトレーニングには、データのパッキングと効率的なストレージ形式が重要です。

![図 5-6: 章の知識構造](../../images/part2/图5_6_本章知识结构.png)

*図 5-6: 第 5 章 知識構造 — 3 つのテーマ: トークン化アルゴリズム、語彙設計、データ構成*

---

## さらに読む

トークン化とデータのシリアル化に関する詳細な内容については、次のリソースを参照する価値があります。

SentencePiece の論文 (Kudo と Richardson、2018 年) では、言語に依存しないサブワードのトークン化が導入されています。 BPE の論文 (Sennrich et al.、2015) は、BPE を NLP に導入した先駆的な研究です。 Unigram の論文 (Kudo、2018) は、サブワードのトークン化に関する確率論的な観点を提供しています。 HuggingFace Tokenizers ライブラリのドキュメント (huggingface.co/docs/tokenizers) は、信頼できる実用的なリファレンスです。カリキュラム学習については、Bengio らの調査論文が包括的な理論的枠組みを提供しています。

---

## 次の章のプレビュー

これで、テキスト事前学習データエンジニアリングに関するすべてのコンテンツが完了しました。次の章「画像とテキストのペアの処理」では、マルチモーダル データ エンジニアリングの分野に入ります。 LAION-5B スタイルの画像とテキストのペア データを処理する方法、同時実行性の高い画像ダウンロードに img2dataset を使用する方法、およびマルチモーダル データ クリーニング パイプラインを構築する方法を学びます。

次の章に入るときに、次の質問について考えてみましょう。画像の「品質」はどのように定義されるべきですか?解像度と明瞭さ以外に、他にどのような点を考慮する必要がありますか?
