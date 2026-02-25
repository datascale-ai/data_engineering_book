# 第10章: 合成データ —— 「データマイニング」から「データファーミング」へ

## 章の概要

大規模モデルの競争が激化するにつれ、公共のインターネット上の高品質な自然データが枯渇に直面しています。私たちはインターネット全体をほぼ「読んで」いますが、モデルの知的限界には遠く及ばないのです。現時点では、合成データはもはやオプションではありません。合成データは、モデルの機能を飛躍的に向上させるための新しいエンジンです。この章では、Microsoft の Phi シリーズの「教科書レベル」のデータ合成手法を深く分析し、受動的な「データ コレクター」から能動的な「データ作成者」に変革する方法を探ります。

テキストの生成にとどまらず、コード実行を使用した思考プログラム (PoT) を通じて厳密な論理検証ループを構築し、GPT-4o やその他のマルチモーダル モデルを使用して複雑な画像とテキストの命令データを合成する方法を説明します。単純な「人間を模倣する」から「人間を超える」に移行する方法を示し、アルゴリズム的手段を通じて現実世界よりも純粋で教育的なトレーニング セットを構築します。

### 学習目標
* 大量の Web データから価値の高いサンプルをフィルタリングする「教科書品質」分類器を構築します。
* Python インタプリタを使用して数学/コード データの正確性を検証する、PoT (プログラム オブ ソート) データ生成パイプラインを実装します。
* LLaVA/GPT-4o ベースのマルチモーダル命令合成をマスターして、イメージ推論 Q&A ペアを構築します。

### シナリオの紹介: データがボトルネックになるとき
>「あなたは、Python プログラミングに特化した小さなモデル (1.3B) をトレーニングしています。それをできるだけ強力にするために、GitHub からすべてのオープンソース コードをスクレイピングするクローラーを作成しました。しかし、トレーニング後にテストすると、モデルがバグのあるコードを書くことを学習し、関数内で「TODO: これは後で修正します」または「このコードはゴミです。使用しないでください」と書くことさえ学習していることに絶望的に気づきました。
単にデータを追加するだけでは効果がなくなり、モデルに与えるゴミの量が増えるほど、その出力はより無秩序になります。そんなとき、Microsoft の Phi-1 論文が目覚ましのようにあなたを襲います。「必要なのは教科書だけだ」。必要なのは、明確なロジック、完璧なコメント、進歩的な教えを備えた教科書のようなコードであり、神と原作者だけが理解できる「スパゲッティ コード」ではありません。しかし、何千億もの完璧な教科書はどこで見つかるのでしょうか?それらを見つけることができないため、私たちはこのデータを何もないところから「作成」する方法を学ばなければなりません。これらの完璧な教科書をバッチ生産するための、たゆまぬ「バーチャル教授」を構築するにはどうすればよいでしょうか?これが、この章で取り上げるエンジニアリング上の主要な課題です。」

---

## 10.1 中心となる概念と原則 (概念と原則)

合成データの中心的な課題は、**品質管理**と**検証ループ**にあります。モデルが生成したテキストには、確認せずに幻覚やエラーが含まれることがよくあるためです。誤ったデータに基づいてモデルをトレーニングすると、「モデルのオートファジー」または「モデルの崩壊」が発生します。つまり、モデルの出力の差異が徐々になくなり、内容が非常に均質になり、現実から切り離されてしまいます。この章で紹介する 3 つの方法は、それぞれテキスト、コード/数学、マルチモーダル データの品質の問題に対処します。

### 10.1.1 合成データの量よりも品質がはるかに重要なのはなぜですか?

初期のディープラーニングでは、「データ量は正義」、つまり十分なデータがあればモデルはすべてを学習できると信じられていました。しかし、合成データの時代では、この定説は覆されました。

**信号対雑音比の理論:**
モデルのトレーニングは本質的には情報圧縮プロセスです。高品質のデータ (教科書など) は、非常に高い情報密度と厳密な論理チェーンを持っています。モデルは、基礎となるパターンを捕捉するために少数のサンプルを必要とします。低品質のデータ (フォーラムのスパム、おしゃべりなど) には、ノイズと論理的なギャップがたくさんあります。トレーニング セットに大量の低品質の合成データが混在している場合、モデルは損失を削減するためにこのノイズを強制的に適合させ、「論理回路」が短絡する原因となります。

物理学を学ぶ: *ファインマン講義* のような古典を読むこと (高品質の合成データ) は、10,000 本の断片的な物理普及ビデオ (低品質データ) を見ることよりも優れています。 Phi-1 の成功は、教科書レベルのデータの 60 億トークンが、トレーニング効果において Web クロールされたデータの 1000 億トークンを上回る可能性があることを証明しました。合成データでは、**検証コスト** が新しい通貨になりました。

### 10.1.2 教科書レベルのデータ (必要なのは教科書だけです)

Microsoft Phi-1 の中心的なアイデア: 1TB のゴミデータでトレーニングするのではなく、6B トークンの高品質データを使用します。その核心は「フィルター」と「アンプ」の構築にあります。

まず、Web データを完全に放棄するわけではありません。「教育的価値」のあるコンテンツを識別するための分類子 (品質分類子) をトレーニングします。これは単に文法をチェックするだけではなく、コンテンツが論理的に一貫性があり、定義と推論が含まれているかどうかをチェックします。第 2 に、強力な生成モデル (GPT-4 など) を「増幅器」として使用し、これらの高品質のスニペットに基づいて、スタイルは似ているがまったく新しいコンテンツを備えた自己完結型の知識の断片を合成します。

![図 10-1: Phi-1 プロセス図](../../images/part4/图10_1_Phi-1流程示意图.png)
*図 10-1: Phi-1 プロセス図*

### 10.1.3 コードと数学の合成: PoT (思考プログラム)

LLM は本質的に確率モデルであり、真の論理推論チップを備えていません。したがって、LLM が算術 (234 * 567 など) または複雑な論理導出を実行すると、幻覚が現れやすくなります。 PoT (思考プログラム) は次のように考えます。LLM は計算は得意ではありませんが、翻訳は得意であるため、数学の問題をコードに「翻訳」させ、Python インタプリタに結果を計算させます。

これは、**100% の精度検証**を達成した合成データの唯一のドメインです。生成されたコードを Python サンドボックスに入れて実行します。エラーの場合は破棄します。正常に実行された場合、実行結果は Ground Truth になります。この「実行イコール検証」メカニズムは、合成データの検証可能性の問題を完全に解決し、無限の数学および論理推論データを低コストで生成できるようにします。

**表 10-1: 合成データ検証戦略の比較**

|データ型 |ジェネレーター |コアチャレンジ |検証者 |検証メカニズム |
| :--- | :--- | :--- | :--- | :--- |
| **一般的なテキスト** | GPT-4 / ジェミニ |幻覚 | LLM (審査員) / 報酬モデル |強力なモデルのスコアリングに依存します。一貫性が低い。 「ジャッジバイアス」になりやすい |
| **数学/論理** | GPT-4 + PoT プロンプト |計算エラー | Python インタプリタ | **実行の一貫性**: コードの実行結果は予想される答えと一致します。論理は絶対に正しい |
| **コード** | DeepSeek コーダー / GPT-4 |構文エラー、ロジックのバグ |単体テスト/コンパイラ | **単体テスト**: アサートまたはコンパイルの成功により、機能を確認します。
| **マルチモーダル** | GPT-4o / LLaVA |幻視（捏造） | CLIP スコア / グラウンディング DINO |生成された説明が画像のEmbeddingと一致するかどうかを確認します。存在しないオブジェクトの捏造を防ぐ |

### 10.1.4 マルチモーダル命令合成: ブリッジング認識

画像データの場合、従来のアノテーションのコストは非常に高く、説明は簡潔です。 LLaVAは見事な「盲人と象」戦略を提案した。既存の検出モデルを使用して画像を記号的なテキスト説明 (キャプション + 境界ボックス) に変換し、この純粋なテキストを強力なテキスト専用モデル (GPT-4 など) に供給します。 GPT-4 は画像を見ることはできませんが、このメタデータを通じて画像コンテンツを「想像」し、それに基づいて複雑な推論ダイアログを生成できます。これにより、マルチモーダルなデータ不足が解決されるだけでなく、命令の複雑さとロジックが大幅に向上します。

---

## 10.2 エンジニアリングの実装 (エンジニアリングの実装)

このセクションでは、コードレベルの実装について詳しく説明し、完全なデータ合成パイプラインを構築します。データの多様性を確保し、生成される「教科書」が単調にならないよう注力しています。

### 10.2.1 教科書レベルのデータ: 分類器と合成パイプライン

**ステップ 1: 品質分類子をトレーニングする**
大量のデータから「教科書」を選択するには、軽量モデル (Random Forest や BERT-Tiny など) をトレーニングする必要があります。ここで重要なのは、アノテーション データ ソースです。通常、シードとして数千のサンプルに対する専門家または GPT-4 の慎重なアノテーションが必要です。

**コードの実装: 機能エンジニアリングとトレーニング**
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# 1. Prepare annotation data
# First annotate a small number of samples (e.g., 1000) with GPT-4 as "gold standard"
# Prompt: "Determine if this text is of educational value for a student..."
# Label 1: High Quality (Textbook-like), Label 0: Low Quality (Noise)
# This step is critical; annotation quality directly determines classifier ceiling
data = [
    {"text": "Python lists are mutable sequences...", "label": 1},
    {"text": "Hey guys check out my cat photo...", "label": 0},
    # ... more data
] 
df = pd.DataFrame(data)

# 2. Build classifier pipeline
# Phi-1 paper uses pretrained model Embedding; here simplified to TF-IDF for demonstration
# In production, recommend DeBERTa-v3-small or similar lightweight Transformer for Embedding
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1))
])

# 3. Train
X = df['text']
y = df['label']
pipeline.fit(X, y)

# 4. Predict (Filtering Phase)
# Score massive web data; retain only high-scoring data
web_snippet = "Standard library documentation for Python..."
score = pipeline.predict_proba([web_snippet])[0][1]

if score > 0.8:
    print("Keep this data for training: High Educational Value")
else:
    print("Discard: Low Signal-to-Noise Ratio")
```

**ステップ 2: 合成教科書フラグメントの生成**
分類器でフィルタリングされたシード データの場合は、それを「拡張」する必要があります。ここでの迅速な設計には高度なスキルが必要です。

プロンプト反復: 合成 Python チュートリアル
* **V1 プロンプト:** 「Python リストに関するチュートリアルを作成します。」
    * **結果:** 平凡なブログ投稿のような、平坦な物語で深みがありません。
* **V3 プロンプト (ファイ スタイル):** 定義、比較、複雑さの分析、落とし穴の警告を必要とする特定の教育法が導入されました。

```python
# V3 Prompt - Textbook-style synthesis
synthetic_textbook_prompt = """
### ROLE
You are a professor of Computer Science writing a definitive textbook on Python.

### OBJECTIVE
Write a comprehensive, self-contained chapter section on the topic: "List Comprehensions vs. Map/Filter".

### REQUIREMENTS
1. **Tone**: Educational, clear, precise, and rigorous. Avoid conversational filler.
2. **Structure**:
   - Start with a conceptual definition explaining *why* this feature exists.
   - Provide a "Before and After" code example (Loop vs. Comprehension).
   - Explain the *computational complexity* (Big O) implications.
   - Include a "Common Pitfall" section (e.g., readability vs. brevity).
3. **Diversity**: Use realistic variable names (e.g., `inventory_items`, `sensor_readings`), NOT generic ones like `x`, `y`, `foo`. 
   (This is crucial to prevent the model from overfitting to toy examples).

### OUTPUT
[Markdown Content]
"""
```

### 10.2.2 逆翻訳: 回答から質問を導き出す

スクラッチから生成する以外に、もう 1 つの効率的な方法は「逆変換」です。コードでは、高品質のコード スニペット (GitHub のハイスター ライブラリ関数など) は簡単に見つかりますが、対応する自然言語命令が不足していることがよくあります。

LLM 要約を使用できます。複雑なコードを入力し、モデルに「このコードが問題を正確に解決できるように、さまざまなユーザー要件の指示をできるだけ詳細に書いてください。」と尋ねます。これにより、高品質が保証されたヒューマン コードを含む大規模な (命令、出力) ペアが迅速に生成されます。この方法は、複雑なコード ロジックのモデルの理解を強化するのに特に適しています。

### 10.2.3 コードと数学の合成: PoT (思考プログラム)

これは、合成データの正確性を保証する最も強力な手段です。モデルにコードを生成させることで、あいまいな自然言語推論を正確なプログラム ロジックに変換します。

**コアコードの内訳: 生成と検証ループ**
```python
import subprocess
import tempfile
import os

# 1. PoT Generation Prompt
# Require model to write solution steps as Python function solver()
pot_prompt = """
Question: Janet has 3 times as many eggs as Bob. Bob has 5 eggs. How many eggs do they have in total?

Instruction:
Write a Python function named `solver()` that returns the answer.
Do not output the number directly. Write the code to calculate it.
Include comments explaining the logic.
"""

# Assume LLM returns the following code string
generated_code = """
def solver():
    # Bob has 5 eggs
    bob_eggs = 5
    # Janet has 3 times as many as Bob
    janet_eggs = 3 * bob_eggs
    # Total eggs
    total = janet_eggs + bob_eggs
    return total
"""

# 2. Code Execution Sandbox
# WARNING: Directly executing generated code is extremely dangerous; must run in sandbox
def execute_generated_code(code_str):
    try:
        # In production use Docker, gVisor, or nsjail for isolation
        local_scope = {}
        
        # Limit execution time to prevent infinite loops
        # Here uses simplified exec for demo; production needs resource module for CPU/memory limits
        exec(code_str, {}, local_scope)
        
        if 'solver' in local_scope:
            result = local_scope['solver']()
            return result, "Success"
        else:
            return None, "No solver function found"
    except Exception as e:
        return None, f"Execution Error: {str(e)}"

# 3. Verification and Data Saving
result, status = execute_generated_code(generated_code)

if status == "Success":
    print(f"Verified Answer: {result}")
    # Data saving strategy:
    # Strategy A (PoT): Save Instruction -> Code. Train model to write code for solving.
    # Strategy B (CoT): Save Instruction -> "Let's calculate... [Reasoning]... The answer is {result}".
    # Strategy B uses code as intermediate step to generate pure text reasoning data.
    save_to_dataset(pot_prompt, generated_code, result)
else:
    print("Discard bad data: Code failed to execute")
```
**プロのヒント:** 生成されたデータは、PoT だけでなく通常の CoT モデルもトレーニングできます。方法: 実行に成功したコードを「中間ステップ」、実行結果を「最終回答」として使用し、`<thinking>...code...</thinking><answer>...result...</answer>` 形式を逆に構築します。この「鶏を借りて卵を産む」方法により、純粋なテキスト モデルの演算精度が大幅に向上します。

### 10.2.4 マルチモーダル命令合成: LLaVA パイプライン

テキストのみのモデルを使用してマルチモーダル データを合成することは、LLaVA の革新です。このメソッドの核心は、視覚情報を**記号化**することにあります。テキストのみのモデル (GPT-4 など) は画像を認識できないため、画像を読み取り可能な「コード」に変換します。

![図 10-2: LLaVA データ合成プロセス図](../../images/part4/图10_2_LLaVA数据合成流程示意图.png)
*図 10-2: LLaVA データ合成プロセス図*

#### 1. エンジニアリング パイプライン: ピクセルからシンボルまで

プロンプト設計の前に、画像をテキスト モデルで読み取り可能な構造化データ (メタデータ) に「分解」するためのツールチェーンが必要です。

1. **グローバル セマンティクス (キャプション)**
    * **ツール**: 一文の説明用の CLIP または BLIP。
    * **役割**: 全体的なコンテキストを提供します。
    * **出力例**: `"A young girl riding a horse on a beach at sunset."`

2. **局所的な詳細 (物体検出)**
    * **ツール**: DINO を接地してオブジェクトと座標を抽出します (バウンディング ボックス)。
    * **役割**: 空間アンカー エンティティを提供します。
    * **出力例**: `{'girl': [100, 200, 300, 400], ...}`

3. **データ合成**
    * **アクション**: 上記の情報をプロンプトに入力し、GPT-4 を呼び出してダイアログを生成します。

#### 2. プロンプトエンジニアリング: 設計と考慮事項

構造化データでは、迅速な設計がデータ品質の鍵となります。以下は、LLaVA スタイルのプロンプト テンプレートとアーキテクチャ上の考慮事項です。

```python
# System Prompt for Multimodal Data Generation
multimodal_gen_prompt = """
### CONTEXT
You are an AI visual assistant. You cannot see the image directly, but I will provide its metadata.
Your task is to generate a conversation between a Human and Yourself about this image.

### IMAGE METADATA
# [Data injection point]: Fill pipeline-extracted data here
- **Caption**: "{caption}"
- **Objects**: {object_list_with_boxes}

### INSTRUCTIONS
1. **Conversation Style**: Generate a multi-turn Q&A (User asking, Assistant answering).
2. **Reasoning**: The Human should ask complex questions (e.g., "What suggests this is a safe environment?"). You answer based on the visual evidence.
3. **Spatial Awareness**: Use the bounding box info to describe relative positions if asked (e.g., "The ocean is in the background...").
4. **Visual Consistency**: Do NOT hallucinate objects not listed in the metadata.
"""
```

**プロンプト設計の背後にあるアーキテクチャ上の考慮事項:**

* **キャプションとオブジェクトの両方を提供するのはなぜですか? (補完性)**
    オブジェクト (女の子、馬、海) だけでは個別であり、動きや雰囲気が欠けています。キャプション（馬に乗った女の子）だけでは特定の場所がありません。 GPT-4 を組み合わせると、完全な精神的な「シーン グラフ」を構築できます。

* **なぜ「空間認識」を重視するのでしょうか? (空間調整)**
    テキスト モデルには本質的に空間感覚が欠けています。 `[x1, y1, x2, y2]` 座標データの処理を強制することで、テキスト モデルに「視覚的な配置」、つまり「左」、「右下」などに対応するピクセル領域を理解することを強制的に学習させます。

* **「視覚的な一貫性」制約を追加するのはなぜですか? (幻覚抑制)**
    テキスト モデルの最大の欠点は、簡単に「脳を埋める」ことです。たとえば、「ビーチ」を見て「カモメが飛んでいる」と捏造するかもしれません。高い信号対雑音比を確保するには、メタデータにないオブジェクトの生成を明示的に禁止する必要があります。

* **なぜ「複雑な推論」が生成されるのでしょうか? (データ次元のアップグレード)**
    単なる「これは何だ？馬」では育成価値がありません。 GPT-4 のインテリジェンスを活用して、合成を通じて「思考が必要な」サンプル (因果推論、感情分析など) を人工的に作成する必要があります。これにより、小さなモデル (学生) が学習を通じて大規模なモデル推論を抽出できるようになります。

---

### 10.2.5 マルチモーダル命令データ合成の高度な戦略

テキストのみのモデル (初期の LLaVA v1 など) に基づく「記号推論」はマルチモーダル命令合成の先駆けとなりましたが、その主な欠点は「視覚情報の非可逆圧縮」です。テキスト モデルはピクセル レベルの視覚特徴を直接認識できません。メタデータだけに頼って推論すると、幻覚が起こりやすくなります。

このボトルネックを打開するために、産業界と学界は、**ビジュアルストロングモデル蒸留**、**専門家の混合パイプライン**、および**進化命令生成**という3つのより主流で効率的な合成戦略を進化させてきました。

#### 1. 視覚的な強力なモデルの抽出

これは現在 (2024 年から 2025 年の時点で) 高性能のオープンソース マルチモーダル モデル (LLaVA-NeXT、ShareGPT4V など) を構築するための最も主流の方法であり、SOTA (最先端) とみなされることがよくあります。

**核となるアイデア**
この方法では、「テキスト モデルを使用して視覚的なコンテンツを推測する」ことを放棄し、「教師と生徒」の蒸留パラダイムを採用します。クローズドソースのトップマルチモーダルモデル (GPT-4o、Gemini 1.5 Pro など) を「教師モデル」として使用し、生の画像信号を直接処理して、高品質で高密度の詳細な説明 (Dense Caption) と、学習用のオープンソース モデル (学生モデル) の複雑な推論 Q&A ペアを生成します。

**利点の分析**
* **モダリティのギャップを排除**: 教師モデルは画像ピクセルを直接「見て」、テキストのメタデータでは伝えられない照明、テクスチャ、微細な表現をキャプチャします。
* **幻覚の抑制**: 実際の視覚入力に基づいた説明により、事実誤認の可能性が大幅に減少します。

**導入の流れ**
中心となるのは、教師モデルに徹底的な情報を出力させるプロンプトを構築することにあります。以下は標準的な蒸留フローの擬似コードです。

```python
def generate_dense_instruction(image_path, api_client):
    """
    Use SOTA MLLM to generate high-density multimodal instruction data
    """
    
    # System Prompt key: Require extremely detailed capture and logical association
    distillation_prompt = """
    You are an expert visual analyst. Analyze the provided image with extreme detail.
    
    Tasks:
    1. Dense Captioning: Provide a comprehensive description of every corner of the image, covering colors, textures, lighting, and background details.
    2. Object Relationships: Analyze the interactions between objects (e.g., causality, spatial relations).
    3. OCR Extraction: Transcribe any visible text verbatim.
    4. Q&A Generation: Based on the visual details above, create a logical reasoning question that cannot be answered without looking at the image.
    """

    # Key difference: Input contains real Image Tensor, not merely Bounding Box
    response = api_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": distillation_prompt},
            {"role": "user", "content": [{"type": "image_url", "url": image_path}]}
        ]
    )
    
    return parse_response(response)
```

---

#### 2. ドメインの専門化: 専門家の混合パイプライン

ドキュメント AI、複雑なチャート分析、自動運転データなど、一般的なモデルが困難な垂直領域では、一般的な視覚的抽出では精度が欠けていることがよくあります。 「専門家の混合」戦略を採用することがベスト プラクティスです。

**コアロジック**
この方法は、単一モデルのエンドツーエンド機能に依存せず、認識フロントエンドとして複数の特殊な小型モデル (エキスパート) を組み立て、非構造化画像を LLM 統合用の微細構造化データに変換します。

ロジックフロー:
$$\text{画像} \xrightarrow{\text{エキスパート}} [\text{OCR} + \text{レイアウト} + \text{検出}] \xrightarrow{\text{集約}} \text{構造化コンテキスト} \xrightarrow{\text{LLM}} \text{命令}$$

**アプリケーション シナリオ**
一般的なアプリケーションには、財務請求書処理、医療画像レポート (DICOM) などが含まれます。

1. **OCR エキスパート (例: PaddleOCR)**: 画像からすべてのテキストと正確な座標 $(x_1, y_1, x_2, y_2)$ を抽出します。
2. **レイアウト エキスパート (例: LayoutLM)**: ドキュメント トポロジを解析し、テーブルの行/列、段落階層、およびタイトルの関係を識別します。
3. **合成 (LLM)**: 構造化データをプロンプト テンプレートに入力します。
    * *プロンプトの例:* 「これは請求書の構造化データ、請求書番号 (100, 200)、合計金額 500.00 です。これに基づいて、「財務監査の検証」に関する複数ターンの Q&A を生成してください。」

![図 10-3: 専門家の混合パイプライン図](../../images/part4/图10_3_多专家混合流水线示意图.png)
*図 10-3: 専門家の混合パイプライン図*

---

#### 3. Evolution 命令の生成 (Visual Evol-Instruct)

テキスト ドメインの WizardLM からインスピレーションを得た Visual Evol-Instruct は、トレーニング データの「均質化」と「過度の単純化」を解決することを目的としています。基本データセットに単純な認識タスク (例: 「画像の中に何があるか?」) のみが含まれている場合、モデルは高次の推論を学習できません。この方法では、プロンプト エンジニアリングを通じて既存データの「次元アップグレード」を強制します。

**コアロジック**
$$\text{単純な VQA} \xrightarrow{\text{複雑性の制約}} \text{複雑な推論 VQA}$$

LLM に特定の進化命令を適用することにより、次の点でデータの複雑性が高まる可能性があります。

* **推論の深化**:
    * *原文*: 「この人は何を持っていますか?」
    * *進化*: 「物の用途とその人の服装に基づいて、この人の職業と次にどのような活動を行うかを推測します。」
* **反事実的推論**:
    * *原文*: 「画像の車は赤いです。」
    * *進化*: 「画像内の赤いスポーツカーが古い自転車に置き換えられたら、シーンの雰囲気はどう変わりますか? これは背景の現代建築スタイルに適合しますか?」
* **比較分析**:
    * 2 つの類似した画像を入力し、微妙な違い (照明の変化、オブジェクトの変位) を分析するモデルを必要とし、モデルの詳細な観察をトレーニングします。

これら 3 つの戦略を組み合わせて使用​​することで、視覚的な詳細と深い論理的推論に富んだ高品質のマルチモーダル命令データセットを構築し、LLaVA、MiniGPT-4、および同様のモデルをトレーニングするための強固な基盤を築くことができます。

##10.3。実績と評価 (実績と評価)

合成データでのトレーニング後の評価は特に重要です。モデルが合成データのパターンを本当に「学習した」のか、それとも単に「記憶した」だけなのかを確認する必要があります。

### 評価指標
* **Pass@1 (コード):** PoT 合成データの場合、HumanEval で Pass@1 をテストします。 Phi-1 は、わずか 6B データで 50%+ Pass@1 を達成し、100 倍以上のデータでトレーニングされた多くのモデルを上回りました。これはデータ品質の圧倒的な優位性を証明しています。
* **幻覚率:** 合成マルチモーダル応答と元の画像の CLIP 類似性を比較して、存在しないオブジェクトが生成されたかどうかを検出します。特にモデルが存在しないオブジェクトに応答するように誘導し、モデルが拒否したかどうかを確認する負のサンプル セットを構築できます。
* **除染:** 汚いですが必要なステップです。合成データにテスト セット (HumanEval など) の質問が誤って含まれていないかどうかを確認します。 N グラムのオーバーラップ検出を通じて、モデルがチートではなく一般化されていることを確認します。

### ベンチマーク
* **PoT と CoT:** 数学 (GSM8K など) では、通常、PoT は純粋なテキスト CoT よりも 5 ～ 10% 優れています。理由: PoT は計算を CPU (計算能力に優れたもの) にアウトソーシングし、GPU はロジック変換、つまり最適な計算割り当てに重点を置きます。

---

##10.4。落とし穴とトラブルシューティング

合成データは美しいですが、罠もたくさんあります。ほんの少し滑ると、モデルは「自己満足」のループに陥ります。

* **落とし穴 1: 自己確認バイアス**
    * **症状:** モデルで生成されたコードは実行されますが、ロジックが間違っています (例: 2+2=5、モデルで生成されたテスト ケースも間違っており、偶然間違った関数を渡しています)。
    * **修正:** 外部の決定論的ソルバーまたは人間がレビューした単体テスト ライブラリを導入する必要があります。モデル生成されたコードを検証するために、モデル生成されたテスト ケースに完全に依存しないでください。これは、犯罪者自身に判断を委ねるようなものです。

* **落とし穴 2: 視覚的な根拠の欠如**
    * **症状:** マルチモーダル合成データでは、モデルがメタデータにない詳細について議論します (捏造)。たとえば、メタデータには「犬」しかありませんが、画像内の犬に首輪がない場合、モデルには「犬の首輪の色」が記述されます。
    * **修正:** プロンプトに厳密な指示を追加:「提供されたメタデータのみに厳密に依存します。詳細を発明しないでください。」また、CLIP スコアを使用して、元の画像との類似性が低すぎる生成されたテキストを除外します。

* **落とし穴 3: 均質化の罠**
    * **症状:** すべてのデータが GPT-4 で生成される場合、モデルは「ローエンド GPT-4」になり、多様性が失われます。すべての答えの口調と文章の構造は驚くほど一貫しています。
    * **修正:** **エントロピー注入**。プロンプトに異なるペルソナ (例: 「気難しいプログラマー」、「忍耐強い幼稚園の先生」) をランダムに挿入するか、異なるプログラミング スタイル (再帰的対反復的) を要求して、データ分布の拡張を強制します。

---

##10.5。各章の概要と詳細情報

「必要なのは教科書だけ」というパラダイムは、データの質 (教育的価値) が量よりも優先されるという中心原則を確立しました。合成データ技術により、データの信号対雑音比を正確に制御できるようになります。このフレームワークの下で、Program of Thought (PoT) は、推論を実行可能コードに変換し、検証にコンパイラの決定論を使用することにより、データの厳密性を保証します。一方、シンボリックから合成への方法では、メタデータ (バウンディング ボックスなど) を使用してテキスト モデルがマルチモーダル コンテンツを生成するようにガイドし、ユニモーダル データからマルチモーダル データへの効果的な変換を実現します。この進化は、データ エンジニアリングの受動的な「マイニング」から能動的な「生産」への移行を示しています。シード プロンプト構築、命令の複雑さ (進化)、高品質のフィルタリング、最終合成を通じて、標準的な工業プロセスを通じて高品質のデータセットを体系的に構築します。

### 参考文献
* *Gunasekar、S.、他。 （2023年）。必要なのは教科書だけです (Phi-1)。*
* *Chen、W.、他。 （2022年）。思考プログラムのプロンプト: 数値推論タスクの推論から計算を解きほぐします。*
* *Liu、H.、他。 （2023年）。ビジュアルインストラクションチューニング (LLaVA)*
* *Shumailov、I.、他。 （2023年）。 The Curse of Recursion: Training on Generated Data Makes Models Forget.* (モデルの崩壊に関する重要な研究)
