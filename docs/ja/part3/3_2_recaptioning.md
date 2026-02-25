## 第 7 章: データの再キャプション

### 章の概要

インターネット上のオリジナルの Alt-text (代替テキスト) は、基本的に、検索エンジン最適化 (SEO) のために Web 開発者によって設計された補助コンテンツです。その中心的な目標は、検索結果における Web ページのランキングを向上させることであり、画像自体のビジュアル コンテンツを正確かつ包括的に記述することではありません。これにより、大量の生の代替テキストが生成され、ビジュアル言語モデル (VLM) トレーニングの「ビジュアル テキストの正確な位置合わせ」という中心要件を満たすことができなくなります。この章では、主流のビジュアル言語モデル (VLM) を活用して、大規模な画像データを自動的に再キャプションするための効率的でスケーラブルな「合成キャプション ファクトリ」を構築する方法を体系的に紹介します。説明の粒度 (概要から詳細まで) を正確に制御するプロンプト エンジニアリングの重要な役割を詳しく調査し、さまざまな下流タスクで区別される説明精度の要件を解決し、VLM のリッチ テキスト画像 (ドキュメント、ポスター、グラフなど) 内のテキストの認識が弱いことに対処する補足として光学式文字認識 (OCR) テクノロジを導入して、複雑な画像のモデルの理解をさらに強化します。

**学習目標**:
* 代替テキストの「3 つの罪」（無関係、短すぎる、視覚的な省略）の本質的な原因と、そのような低品質の説明が視覚言語モデルと生成視覚モデルのトレーニングに引き起こす具体的な害（例：モデルの幻覚、視覚とテキストの整合性の失敗、一般化の不全）を深く理解します。
* vLLM (効率的な大規模モデル推論エンジン) を使用して、LLaVA や CogVLM などの主流の VLM を展開する完全なワークフローをマスターし、高スループット推論の核となる原理を理解し、迅速な大規模画像リキャプションを実現します。
* さまざまな下流タスク (CLIP スタイルのデュアルタワー モデルの事前トレーニング、Sora スタイルの生成モデル トレーニングなど) に基づいて階層化されたプロンプト戦略を設計でき、タスクの要件に正確に一致する簡単または詳細な画像説明を柔軟に生成できます。
* コア OCR アプリケーション メソッドをマスターし、OCR 認識結果と VLM プロンプトの動的な融合を実装し、ドキュメントおよびポスター クラスのリッチ テキスト画像の低い記述品質を解決し、そのような画像の記述精度と豊かさを大幅に向上させます。

**シナリオの紹介**:
> 「ソラのようなモデルをトレーニングしていると想像してください。エッフェル塔を背景に夕日の中を走るゴールデンレトリバーの画像をモデルに与えます。ただし、生データのラベルは「IMG_20240501.jpg」または「最高のドッグフード 50% オフ」です。このようなデータでは、モデルは「ゴールデン レトリバー」と「エッフェル塔」の視覚的な対応関係を学習することはなく、ましてや「夕日の照明」を理解することはできません。犬と塔をテキストに正確に書き込むには、AI にアノテーターとして機能させる「データ再キャプション」が必要です。」

### 7.1 代替テキストの制限: 生の Web 記述が使用できないのはなぜですか?

視覚言語モデルと生成視覚モデルのトレーニングでは、データ品質がモデルの上限を直接決定します。Web ページの生の代替テキストは、まさに低品質の視覚テキスト データの主なソースの 1 つです。 DeepMind (「弱教師画像テキスト データを使用した言語画像の事前トレーニングのスケーリング」) および OpenAI (「人間のフィードバックによる指示に従う言語モデルのトレーニング」) の内部調査レポートによると、インターネットからクロールされた生の代替テキストをトレーニング データとして直接使用すると、モデルのパフォーマンスが時期尚早に「上限」に達する (つまり、データ量の増加に関係なく、モデルの視覚的理解とテキスト生成の精度がそれ以上改善できない) か、さらには低下する可能性があります。核心的な問題は「3 つの罪」として要約できます。

* **非常にノイズが多い**: 大量の代替テキストには、ファイル名 (例: 「IMG_20240501.jpg」)、日付、無関係な SEO キーワードの積み重ね (例: 「安い靴を買う、ナイキ、アディダス」、「近くの最高のコーヒーショップ」) のみが含まれています。このような記述は画像のビジュアル内容とは全く関係ありません。これらをトレーニングに使用すると、モデルが視覚とテキストの対応を確立できないだけでなく、モデルの言語能力が損なわれ、モデルが無関係で冗長なテキスト、さらには重度の幻覚を生成する原因になります。
* **視覚的な省略**: 代替テキストのデザイン意図は主に Web ページの機能をサポートすることであり、視覚的なコンテンツを説明することではありません。多くの場合、画像の機能 (例: 「購入ボタンをクリック」、「詳細を表示」) または商業的属性 (例: 「赤の XL サイズ」、「期間限定割引」) のみを説明し、画像の視覚的な詳細 (例: オブジェクトの形状、色、テクスチャ、空間関係、照明効果) を完全に無視します。たとえば、「胸に白いヴィンテージのロゴが付いた赤い純綿の T シャツが木製のテーブルの上に平らに置かれている」という画像には、単なる「赤い T シャツのプロモーション」という代替テキストが含まれている可能性があります。このような説明では、モデルに視覚的な特徴を学習させることはできません。
* **短すぎる**: Common Crawl (世界最大の Web クロール データセット) の統計によると、代替テキストの 50% 以上が 5 単語未満であり、30% は 3 単語未満です。このような非常に短い説明では、複雑な視覚的ロジック、空間関係、および詳細情報を伝えることはできません。たとえば、複数のオブジェクト、シーン、および相互作用関係を含む「草の上に横たわるゴールデンレトリバー、赤いボールの上に前足、野の花でいっぱいの丘の中腹の背景」を説明することはできません。

**再キャプションの価値**: データ再キャプションの中心的な価値は、AI を使用して高品質の「ビジュアルテキストの正確な位置合わせ」説明を自動的に生成し、低品質の生の代替テキストを置き換え、モデルのパフォーマンスの上限を打ち破ることです。これは、トップの業界調査によって確認されています。OpenAI は、DALL-E 3 論文 (「DALL・E 3: アライメントを改善した自己回帰画像生成のスケーリング」) で、トレーニングに最大 95% の合成長文テキスト (合成キャプション、つまり VLM によって生成された再キャプション テキスト) を使用することが、OpenAI の指示追従能力と視覚復元精度が Stable Diffusion XL (SDXL) をはるかに上回る主な理由の 1 つであると明示しています。合成長文テキストは、画像の視覚的な詳細、論理的関係、シーンの雰囲気を正確にキャプチャすることができ、モデルが「見ているものを説明する」ことを真に学習できるため、その後の生成、認識、理解の能力が向上します。

### 7.2 合成キャプション ファクトリー: VLM を使用してデータを再生成する

大規模な画像の再キャプションを実現するために、手動のアノテーションのみに依存すると、非常にコストがかかるだけでなく（画像ごとに数分、データセットは数百万または数十億の画像になることがよくあります）、一貫性のないアノテーション標準と低効率という問題もあります。したがって、VLM 主導の「合成キャプション ファクトリー」を構築する必要があります。生の画像を入力として、高品質で標準化されたテキストの説明を出力として受け取り、自動化とバッチ処理を通じてデータの再キャプションを完了し、データ価値の「再生」を実現します。

この「ファクトリー」の中核となるロジックは、生の画像を最適化された VLM にフィードし、慎重に設計されたプロンプトによって記述の粒度とスタイルを制御し、次に効率的な推論エンジンによって処理スループットを向上させ、最終的に下流タスクの要件を満たす高品質な記述を出力することです。全体のフローは、「モデルの選択とアーキテクチャ設計」、「迅速な戦略の最適化」、および「エンジニアリングの導入」という 3 つのコア リンクに分割できます。

#### 7.2.1 モデルの選択とアーキテクチャ

VLM アーキテクチャは、記述の品質、速度、適用可能なシナリオを直接決定します。現在主流の VLM アーキテクチャは主に 3 つのタイプに分類されます。以下の表に、各アーキテクチャの代表的なモデル、メリット/デメリットの比較、推奨シナリオを示します。選択は、下流タスクの要件 (記述精度、処理速度、データ型など) に基づいて柔軟に行うことができます。

|モデルアーキテクチャ |代表機種 |利点 |デメリット |推奨されるシナリオ |
| :--- | :--- | :--- | :--- | :--- |
| **Q-Former 接続** | BLIP-2、InstructBLIP |パラメータ数が少ない (通常は数十億で、大規模な言語モデルをはるかに下回る)、推論が速い (1 つの画像の推論に数十ミリ秒かかる場合がある)、トレーニングと展開のコストが低い、テキストの幻覚が発生しにくい (説明が画像にぴったりと一致する) |説明の長さが短く、細部のキャプチャが平均的で、「繰り返しの説明」が発生しやすい (中心となるオブジェクトがほとんどなく、詳細が拡張されていない)、複雑なシーンの理解が限られている |大量の画像の迅速な初期スクリーニング (貴重なデータをフィルタリングするための数十億の画像の大まかな要約など)、または短い代替テキストの生成 (説明の長さに厳密な制限があるシナリオの場合) |
| **MLP プロジェクション + LLM** | LLaVA-1_6 / ネクスト |非常に詳細な説明、微妙な画像の詳細 (照明、テクスチャ、オブジェクトの相互作用など) をキャプチャし、強力な指示に従います (「シーンの順序で説明する」、「コア オブジェクトをハイライトする」などのプロンプト要件に正確に対応します)、マルチターン ダイアログをサポートします (マルチターン プロンプトを通じて説明の品質を最適化できます)。大量のロジック計算 (LLaMA 2 7B/13B のような 7B 以上のパラメータ LLM が必要)、比較的遅い推論、プロンプト制約なし、冗長な説明になりがち |高品質の長い形式の高密度キャプションを生成するためのメイン モデル (例: Sora スタイルの生成モデル、SD3 画像生成モデル、正確で詳細なビジュアルテキスト配置データを必要とするシナリオのトレーニング) |
| **ビジョンファーストのアーキテクチャ** | CogVLM、Qwen-VL |高い視覚解像度 (HD 画像入力をサポート、一部のモデルは 4K をサポート)、きめ細かいオブジェクト認識に優れ、特にリッチテキスト画像 (ドキュメント、チャート、UI スクリーンショット) 内のテキストおよび小さなウィジェット (ボタン、入力フィールド) の精度が高く、テキストと視覚要素の関連付けを理解します。 VRAM 使用率が高い (7B モデルの展開には少なくとも 24GB VRAM が必要)、非標準アーキテクチャ (モデルごとに展開アプローチが異なる)、展開がやや面倒、推論速度が中程度 |特にドキュメント、チャート、UI スクリーンショット、ポスター、およびその他のリッチテキスト データ (ドキュメント画像、UI インターフェイス、または画像内の正確なテキスト認識を必要とするシナリオを生成するトレーニング モデルなど) 向け |

補足: 3 つのアーキテクチャの主な違いは、「ビジュアル モジュールが言語モジュールにどのように接続されるか」にあります。Q-Former アーキテクチャは、専用の Q-Former モジュールを使用して、軽量言語モデルに入力する前に視覚特徴を言語が理解できるベクトルに変換します。 MLP プロジェクション アーキテクチャは、多層パーセプトロン (MLP) を使用して視覚的特徴を言語モデルEmbedding空間に投影し、大規模な言語モデルと深く統合します。ビジョンファーストアーキテクチャは、ビジュアルモジュールの解像度と認識能力を強化し、言語モジュールの冗長計算を弱め、「ビジョンファースト」を優先します。

#### 7.2.2 即時戦略: 粒度の制御

プロンプト エンジニアリングは、「合成キャプション ファクトリー」の「コア コントローラー」です。同じ VLM が、異なるプロンプト ガイダンスの下で、完全に異なるデータ分布 (説明の長さ、詳細の豊富さ、スタイル) を生成します。したがって、特定の下流タスク要件に基づいて階層化されたプロンプト戦略を設計し、生成された説明がタスクのニーズに完全に一致するように説明の粒度を正確に制御する必要があります。

基本原則: プロンプト設計では、「タスクの指示」、「説明の範囲」、および「粒度の要件」を明確に指定し、曖昧な表現を避ける必要があります (たとえば、「このイメージを説明する」だけを使用すると、モデルの出力が不安定になります)。一方、「制約」（「単語数は 20 語以下」、「主要なオブジェクトと背景を強調表示」など）を追加すると、出力品質をさらに最適化できます。




![図 7-1: 簡潔なプロンプト戦略と詳細なプロンプト戦略](../../images/part3/图7_1_简略与详细的Prompt策略.png)
*図 7-1: 迅速な戦略の概要と詳細*


図 7-1 は、2 つのコア プロンプト ストラテジーの出力の違いを直感的に比較しています。短いプロンプトは、コア オブジェクトとシーンのみを含む簡潔な説明を生成します。詳細プロンプトは、オブジェクトの形状、照明、色、空間関係などを含む豊富な詳細情報を含む説明を生成します。2 つの戦略は、それぞれ異なる下流タスクに適応します。

以下に、最も一般的に使用される 2 つの階層化プロンプト戦略を示します。実際のニーズに基づいて柔軟に調整するか、これに基づいて中粒度のプロンプト戦略を設計します。

**戦略 1: 簡単な説明 (簡単なキャプション)**
* **プロンプト**: 「この画像を一文で簡潔に説明してください。」
  補足的な最適化プロンプト (安定性のため): 「この画像を 1 文で簡潔に説明し、主要な被写体と重要な背景のみに焦点を当て、冗長な詳細は含まないでください。」
* **目的**: CLIP スタイルのデュアルタワー モデル (ビジョン テキスト デュアルタワー アーキテクチャ) のコンテキスト長制限に適応します。このようなモデルでは通常、テキスト入力が 77 トークン以下に制限されます。過度に長い説明は切り捨てられるため、正常な学習が妨げられます。厳密な記述長要件があるシナリオ (画像の検索、迅速な注釈など) にも適しています。
* **期待される出力**: 「エッフェル塔近くの芝生の上を走るゴールデンレトリバー」。
  出力特性: 長さは 10 ～ 20 ワードに制御され、中心となるオブジェクト (ゴールデン レトリバー) のみ、キー アクション (走る)、中心となる背景 (エッフェル塔、草)、余分な詳細はなく、簡潔かつ明確です。

**戦略 2: 詳細な説明 (詳細なキャプション)**
* **プロンプト**: 「この画像を詳細に説明してください。主要な主題から始めて、背景、照明、色、芸術的なスタイルについて説明してください。オブジェクト間の具体的な相互作用についても言及してください。」
  補足的な最適化プロンプト (より詳細なキャプチャのため): 「この画像を詳細に説明してください。まず、主要な被写体の外観 (形状、色、テクスチャ)、次に背景シーン、照明効果 (明るさ、色温度)、カラー マッチング、および芸術的スタイルについて説明します。最後に、オブジェクト間の相互作用と画像全体の雰囲気について言及します。」
* **目的**: GenAI モデル トレーニング (Sora、SD3、Ideogram など) に適応します。このようなモデルでは、高精度で命令に準拠した画像を生成するために、画像の詳細な特徴、論理的関係、シーンの雰囲気を学習するための詳細な記述が必要です。正確なビジュアルとテキストの位置合わせが必要なシナリオ (ビジュアル QA、画像編集など) にも適しています。
* **期待される出力**: 「緑の芝生の上を楽しそうに走るふわふわのゴールデンレトリバーのダイナミックな広角ショット。犬の毛皮は夕日の暖かい金色の光に照らされ、薄茶色の毛束が日光の中で輝いています。走るとき耳は後ろに下がり、尻尾は高く上がって幸せな気分を示しています。ぼやけた背景の中で、エッフェル塔の象徴的な鉄の格子構造が紫色のグラデーションの空にそびえ立っています。芝生には小さなシロツメクサの花が点在しており、犬にソフトな焦点を当て、主要な被写体を強調するぼかした背景により、画像全体の雰囲気は暖かく活気に満ちています。」
  出力特性: 長さは通常 50 ～ 200 ワードで、核となるオブジェクトの詳細、背景シーン、照明、色、芸術的スタイル、オブジェクトのインタラクション、全体的な雰囲気をカバーし、詳細が豊富で、ビジュアル テキストの位置合わせの精度が高くなります。

補足: 上記の 2 つの戦略に加えて、説明の関連性をさらに高めるために、たとえば、電子商取引画像 (「製品の外観、色、サイズ、質感、配置に焦点を当て、電子商取引のプロモーションに適したこの商品画像を詳細に説明します」) や文書画像 (「テキストの内容、レイアウト、フォント スタイル、テキストの色を含めて、この文書画像を詳細に説明します」) については、「タスク指向プロンプト」を設計できます。

#### 7.2.3 エンジニアリング実装: vLLM を使用した高スループット推論サービスの構築

大規模なデータの再キャプション (10 億規模の画像データセットの処理など) の場合、通常の HuggingFace `generate()` でははるかに不十分です。推論が遅く、スループットが低く、GPU リソースを効率的に利用できません。 1 つの GPU で処理できる画像は 1 日に数千枚までです。大規模な処理には、かなりの時間とハードウェアのコストがかかります。したがって、専用の大規模モデル推論エンジンである vLLM が必要です。vLLM は、VLM 推論のスループットを 3 ～ 5 倍向上させながら、GPU VRAM の使用量を削減し、効率的で大規模な画像の再キャプションを実現できる 2 つのコア最適化手法である PagedAttendance と Continuous Batching をサポートします。

vLLM は、カリフォルニア大学バークレー校の研究チームによって開発された高効率の大規模モデル推論エンジンです。主な利点は「高スループット、低遅延、高い GPU 使用率」で、HuggingFace と互換性のある API インターフェイスと最小限の移行コストを備えた LLaVA、CogVLM、およびその他の主流 VLM の導入に最適です。

以下は、モデルの初期化、プロンプト テンプレートの設計、バッチ処理、出力抽出を含む、高スループットの画像再キャプションを目的とした vLLM を使用して LLaVA-1_5-7b-hf を展開するためのコア コードです。主要なパラメーターの解釈と最適化のヒントを含む完全なワークフローです。

```python
from vllm import LLM, SamplingParams
from PIL import Image
import os
from tqdm import tqdm  # For displaying batch processing progress

# Initialize vLLM inference engine
# tensor_parallel_size=4: Use 4 GPUs for tensor parallelism, for large models (7B/13B) deployment, adjust based on GPU count (1, 2, 4, 8)
# Note: Tensor parallelism requires multiple same-model GPUs with NVLink support for faster data transfer
# trust_remote_code=True: Allow loading LLaVA custom code (e.g., vision-language fusion module), as LLaVA architecture is non-standard HuggingFace
# model: Model name from HuggingFace Hub (e.g., llava-hf/llava-1_5-7b-hf, llava-hf/llava-1_5-13b-hf)
# gpu_memory_utilization=0_9: Set GPU VRAM utilization to 90%, balance throughput and stability, avoid OOM
llm = LLM(
    model="llava-hf/llava-1_5-7b-hf",
    tensor_parallel_size=4,
    trust_remote_code=True,
    gpu_memory_utilization=0_9
)

# Define Prompt template (LLaVA requires specific dialogue format, otherwise affects instruction following)
# Tip: Adding "Analyze the image" in Prompt often works better than "Describe the image"
# because "Analyze" guides model to observe image details more carefully, reducing perfunctory descriptions
# Here using detailed description template, can replace with brief description template as needed
prompt_template = "USER: <image>\nAnalyze this image and describe it in extreme detail. Start with the main subject, then describe the background, lighting, colors, and artistic style. Mention any specific interactions between objects. ASSISTANT:"

# Configure sampling parameters for description quality and stability
# temperature=0_2: Lower randomness (0-1 range), lower temperature = more stable, image-fitting descriptions, fewer hallucinations
# For more diverse descriptions, adjust to 0_5-0_7; if too high (>0_8), likely to produce image-unrelated hallucinations
# max_tokens=256: Limit output length, prevent overly verbose descriptions
# top_p=0_95: Nucleus sampling, keep only tokens with cumulative probability to 95%, further reduce hallucination risk
sampling_params = SamplingParams(
    temperature=0_2,
    max_tokens=256,
    top_p=0_95
)

def load_image_batch(image_dir, batch_size=32):
    """
    Batch load images for efficient processing
    image_dir: Image folder path, all images in this folder
    batch_size: Images per batch, adjust based on GPU VRAM (16, 32, 64), larger VRAM = larger batch_size
    return: Batch image list (PIL.Image format) and corresponding image path list
    """
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('jpg', 'png', 'jpeg'))]
    image_batches = []
    path_batches = []
    
    # Load images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        for path in batch_paths:
            try:
                # Load image and convert to RGB (avoid model errors from grayscale/transparent images)
                img = Image.open(path).convert('RGB')
                batch_images.append(img)
            except Exception as e:
                print(f"Failed to load image {path}: {e}")
                continue
        if batch_images:  # Skip empty batches
            image_batches.append(batch_images)
            path_batches.append(batch_paths)
    return image_batches, path_batches

def process_batch(image_batch):
    """
    Process a batch of images, generate corresponding recaption text
    image_batch: List[PIL.Image], batch image list
    return: List[str], recaption text list for each image
    """
    # Generate corresponding Prompt for each image
    prompts = [prompt_template for _ in range(len(image_batch))]
    
    # vLLM supports direct multi_modal_data input, no manual image format conversion needed
    # This step is non-blocking; vLLM internally does Continuous Batching for efficient GPU utilization
    # When one batch completes partially, immediately load next batch partial data to avoid GPU idle
    outputs = llm.generate(
        prompts, 
        sampling_params, 
        multi_modal_data={"image": image_batch}
    )
    
    # Extract generated description text, remove Prompt part, keep only model response
    captions = []
    for output in outputs:
        # Extract content after ASSISTANT: as model-generated description
        caption = output.outputs[0].text.strip().replace("ASSISTANT:", "").strip()
        captions.append(caption)
    return captions

def save_captions(image_paths, captions, save_path):
    """
    Save recaption text corresponding to image paths for subsequent use (e.g., model training)
    image_paths: Image path list
    captions: Recaption text list
    save_path: Save file path (txt format)
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        for path, cap in zip(image_paths, captions):
            # Format: image_path\trecaption_text, for easy reading and parsing
            f.write(f"{path}\t{cap}\n")

# Main function: batch process images and generate recaptions
if __name__ == "__main__":
    image_dir = "path/to/your/image/directory"  # Replace with your image folder path
    save_path = "recaption_results.txt"        # Recaption results save path
    batch_size = 32                            # Images per batch, adjust based on GPU VRAM
    
    # Load image batches
    image_batches, path_batches = load_image_batch(image_dir, batch_size)
    
    # Batch process and save results
    with open(save_path, 'w', encoding='utf-8') as f:
        for img_batch, path_batch in tqdm(zip(image_batches, path_batches), total=len(image_batches)):
            captions = process_batch(img_batch)
            # Write current batch results
            for path, cap in zip(path_batch, captions):
                f.write(f"{path}\t{cap}\n")
    print(f"Recaptioning complete, results saved to {save_path}")

```
補足的なエンジニアリング最適化のヒント:

* **画像の前処理**: バッチ読み込み時に、画像サイズの違いによるモデル推論速度の変動や VRAM の不安定性を避けるために、画像のサイズを均一に変更します (例: 224×224 または 448×448)。画像を正規化して記述精度を向上させます。
* **エラー処理**: バッチ処理の中断を避けるために、画像の読み込み失敗とモデル推論の失敗に対する例外処理を追加します。失敗したイメージの場合は、パスを記録し、個別に処理します。
* **ハードウェアの最適化**: 導入には NVIDIA A100、A800 を推奨します。 VRAM 少なくとも 24GB (7B モデル);非常に大規模な場合は、GPU クラスターと vLLM 分散推論を使用してスループットを高めます。
* **プロンプト キャッシュ**: 同じタイプの画像 (バッチ電子商取引ポスターなど) の場合、プロンプト テンプレートをキャッシュして、繰り返し生成を回避し、処理速度を向上させます。

### 7.3 OCR 拡張: 画像からのテキストの抽出と融合

通常の VLM はある程度の視覚的理解を持ち、画像内のオブジェクト、シーン、単純なテキストを認識できますが、高密度テキスト画像 (文書、ポスター、グラフ、PDF スクリーンショット) では 2 つの主要な問題に直面します。1 つはテキスト認識精度が低く、誤認識や省略が起こりやすい (特に芸術的なフォント、ぼやけたテキスト)。 2 つ目は、テキストを視覚要素に効果的に関連付けることができないため、説明でテキストの意味や役割が省略されてしまうことです。

たとえば、「サマー セール 50% オフ」という大きなテキストを含む電子商取引ポスターは、通常の VLM からは「赤いプロモーション ポスター」のみを取得し、テキストは完全に無視されます。文字が認識されていても「サマーセール30％オフ」などのエラーが発生する場合があります。しかし、テキストはそのような画像を要約するために非常に重要であり、中心的な意味と目的を直接決定します。

ベスト プラクティスは、VLM の「外部頭脳」として専用の OCR エンジン (PaddleOCR、Tesseract など) を導入することです。 OCR を使用して画像テキストを正確に抽出し、VLM プロンプトと動的に融合して、VLM がテキストを結合してより正確でリッチな説明を生成できるようにします。これにより、ドキュメントおよびポスタークラスのリッチテキスト画像の再現品質が大幅に向上します。

OCR (光学文字認識) テクノロジーの中核は、画像内の印刷テキストや手書きテキストを編集可能なテキストに変換することです。そのテキスト認識精度は、特に高密度テキストや複雑なフォントのシナリオにおいて、通常の VLM をはるかに上回ります。現在、最も広く使用されている産業用、オープンソース、無料の高精度 OCR エンジンは、PaddleOCR (Baidu PaddlePaddle オープンソース OCR) です。多言語、マルチフォント、ぼやけたテキスト認識、高速推論、シンプルな導入、GPU アクセラレーションをサポートしており、VLM との組み合わせに非常に適しています。




![図 7-2: OCR 拡張パイプライン](../../images/part3/图7_2_OCR增强流水线.png)
*図 7-2: OCR 拡張パイプライン*


**チャート コアの解釈**: 図 7-2 は、完全な OCR 拡張 VLM 再キャプション フローを示しています。コアは「OCRテキスト抽出→コンテキスト構築→プロンプトフュージョン→VLM説明生成」で、VLMのテキスト認識の弱点をOCRで補い、「視覚的な詳細+テキスト情報」の二重精度の説明を実現します。

#### 7.3.1 OCR 拡張パイプライン

OCR 拡張コアは、単純な連結ではなく、OCR で抽出されたテキストと VLM プロンプトを有機的に融合させます。パイプライン全体には 3 つの主要なステップがあり、それぞれに明確な最適化の方向性があり、テキストが効果的に要約の品質を向上させることができます。

1. **検出と認識**: PaddleOCR を使用して生画像を処理します。最初にすべてのテキスト領域 (テキスト位置の境界ボックス) を検出し、次に各領域を認識して、認識されたテキストと信頼度 (0 ～ 1、高い = より正確) を出力します。主な目標: 「正確なテキスト抽出、誤った認識のフィルタリング」 - 信頼性の低い結果をフィルタリングして、誤解を招く VLM を回避します。
2. **コンテキストの構築**: すべての有効な OCR テキスト (信頼性の低いフィルタリング後) を実際の画像の位置 (上から下、左から右、列順に複数列) で連結して、人間が読めるテキスト コンテキストを構築します。必要に応じて、VLM が階層と役割を理解できるようにテキスト (タイトル、本文、ボタンのテキストなど) を分類します。たとえば、ポスター テキスト「サマー セール」（タイトル）、「50% オフ」（サブタイトル）、「6 月 1 日 - 6 月 10 日」（時間）は、タイトル/本文ラベルを使用して「サマー セール、50% オフ、6 月 1 日 - 6 月 10 日」に連結されます。
3. **プロンプト フュージョン**: 構築されたテキスト コンテキストを VLM プロンプトに自然に統合し、VLM に「画像にはこれらのテキストが含まれています。テキストとビジュアル要素の組み合わせを記述してください」と明示的に指示し、VLM がテキストとビジュアル (位置、色、フォント スタイル、テキストの意味とシーン) を関連付けるように導きます。重要なのは、「自然な融合、冗長性のない」ことです。プロンプトの最後にテキストを不自然に追加して、VLM が視覚的な詳細を無視することを避けます。

**補足事項**: パイプラインの最適化は「信頼性フィルタリング」と「プロンプト フュージョン」に重点を置いています。信頼性の低いテキストをフィルタリングしないと、間違ったテキストが VLM の誤解を招きます。プロンプトの融合がぎこちない場合、VLM はテキストとビジュアルを分離し、真の強化を実現できません。

#### 7.3.2 コアコード: OCR 結果の挿入

以下は、PaddleOCR を使用して画像テキストを抽出し、VLM プロンプトに動的に融合するためのコア コードです。 OCR で強化された大規模な画像再キャプションのためのセクション 7.2.3 vLLM バッチ処理とシームレスに統合できます。このコードには、テキスト抽出、信頼性フィルタリング、コンテキスト構築、プロンプト融合のための完全なロジックに加えて、主要なパラメータの解釈と最適化のヒントが含まれています。

```python
from paddleocr import PaddleOCR
import os
from PIL import Image

# Initialize OCR engine (recommend GPU for speed; set use_gpu=False if no GPU)
# use_angle_cls=True: Enable text direction detection, supports tilted text (e.g., tilted poster, rotated documents) to avoid recognition errors
# lang='en': English recognition; for Chinese set lang='ch'; supports Chinese-English mixed (lang='ch_en')
# det_model_dir, rec_model_dir: Can specify OCR detection/recognition model paths; auto-download if not specified
# gpu_mem=500: GPU VRAM limit (MB), adjust based on GPU
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=True,
    gpu_mem=500)

def generate_ocr_enhanced_prompt(image_path, base_prompt="Describe this image in detail."):
    """
    Generate OCR-enhanced VLM Prompt, integrating OCR-extracted text into Prompt
    image_path: Raw image path
    base_prompt: Base Prompt (e.g., brief/detailed description) as Prompt body
    return: Complete OCR-enhanced Prompt; if no valid text, return base Prompt
    """
    # Step 1: Run OCR, extract image text and confidence
    # result is nested list: [[[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], [text, confidence]], ...]
    # [x1,y1]~[x4,y4] are Bounding Box coordinates (top-left, top-right, bottom-right, bottom-left)
    # text is recognized content, confidence is recognition confidence
    result = ocr.ocr(image_path, cls=True)
    
    # Handle OCR result: if no text or empty, fall back to base Prompt
    if not result or not result[0]:
        return f"USER: <image>\n{base_prompt}\nASSISTANT:"
    
    # Step 2: Extract valid text (filter low confidence), build text context
    detected_texts = []
    for line in result[0]:
        text = line[1][0]  # Recognized text
        confidence = line[1][1]  # Recognition confidence
        # Filter results below 0_8 confidence (threshold adjustable, 0_7-0_9 based on image text clarity)
        # Also filter empty text and meaningless garbage (e.g., symbols only, spaces)
        if confidence > 0_8 and text.strip() and len(text.strip()) > 1:
            detected_texts.append(text.strip())
    
    # Build text context: concatenate by recognition order, comma-separated, human-readable
    ocr_context = ", ".join(detected_texts)
    
    # Step 3: Dynamically fuse OCR result with base Prompt, generate enhanced Prompt
    # Key technique: Tell model "I have detected these texts..." so model knows this is image text
    # and guide model to associate text with visuals (position, color, font, text meaning vs. scene)
    if len(ocr_context) > 10:  # Only enhance when text long enough (>10 chars) to avoid redundancy
        enhanced_prompt = (
            f"USER: <image>\n"
            f"I have detected these text segments in the image: '{ocr_context}'. "
            f"Using this text as a reference, describe the image in detail, "
            f"paying attention to how the text relates to the visual elements (such as the position, color, and font style of the text, "
            f"and the connection between the text content and the image scene). {base_prompt}\n"
            f"ASSISTANT:"
        )
        return enhanced_prompt
    else:
        # If text too short (1-2 words), don't enhance, avoid redundancy, fall back to base Prompt
        return f"USER: <image>\n{base_prompt}\nASSISTANT:"

# Test code: verify OCR-enhanced Prompt generation
if __name__ == "__main__":
    # Test image path (replace with your rich-text image, e.g., poster, document screenshot)
    test_image_path = "path/to/your/test/poster.jpg"
    # Base Prompt (detailed description template)
    base_prompt = "Describe this image in extreme detail. Start with the main subject, then describe the background, lighting, colors, and artistic style."
    # Generate enhanced Prompt
    enhanced_prompt = generate_ocr_enhanced_prompt(test_image_path, base_prompt)
    print("OCR-enhanced Prompt:")
    print(enhanced_prompt)
```

**補足的な最適化のヒント:**

* **信頼しきい値の調整**: クリアテキスト画像 (HD ドキュメント、正式なポスター) の場合は、0_8 ～ 0_9 に調整して、いくつかのエラーをフィルタリングします。ぼやけた複雑なフォントの画像 (古いポスター、手書き) の場合は、有効なテキストの欠落を避けるために 0_7 ～ 0_8 に調整します。
* **テキスト コンテキストの最適化**: 複数列の階層テキスト (タイトル、本文、脚注) の場合、境界ボックス座標を使用して分類および連結します (例: 「タイトル: サマー セール; 本文: 50% オフ、6 月 1 日から 6 月 10 日まで; 脚注: 最終解釈権は留保」)。VLM テキスト階層をより明確に理解できるようにします。
* **プロンプト フュージョンの最適化**: 画像タイプごとにフュージョンの文言を調整します。ドキュメント画像には「テキストのレイアウトとテキストとドキュメント構造の関係の説明」が追加され、ポスター画像には「テキストのフォント スタイルとプロモーション シーンにおけるテキストの役割の説明」が追加され、よりターゲットを絞った説明が可能になります。
* **マルチ OCR エンジン フュージョン**: 非常に高い精度が必要な場合は、PaddleOCR と Tesseract の両方を使用し、結果の共通部分を取得してテキスト認識精度を高めます。

**実用的な利点**: OCR 強化により、リッチテキスト画像の再現品質が大幅に向上します。一般的な比較:

* **電子商取引ポスター上の通常の VLM (OCR なし)**: 「白い背景に赤いプロモーション ポスター。曖昧なテキストと下部にボタンが付いています。」
* **OCR 拡張 VLM**: 「白い背景の赤いプロモーション ポスター。ポスターの上部中央に大きな白い太字で「SUMMER SALE 50% OFF」というテキストがあり、右下にある小さな青いボタンで「今すぐ購入」というテキストが特徴です。「SUMMER SALE」というテキストは装飾的なフォントであり、黄色のシャドウ効果が目立ちます。全体のレイアウトはシンプルで目を引くもので、プロモーション情報を強調することに重点を置いています。背景は次のとおりです。無地の白なので、赤いポスターと白い文字がより目立ちます。」

この違いは、正確なテキストを生成するモデル (表意文字、SD3、ドキュメント生成モデル) をトレーニングする場合に重要です。正確なテキストを含むキャプションにより、モデルは「テキストの視覚的表現」と「テキストとシーンの関連付け」を学習し、より準拠した画像を生成します。視覚的な QA と画像検索の場合、OCR で強化された説明により、タスクの精度が向上し、画像の核となる意味のモデルの理解も向上します。
