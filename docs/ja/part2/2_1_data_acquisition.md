## 第 6 章: 画像とテキストのペアの処理

### 章の概要

次世代の基盤モデルを構築する際、データ エンジニアリングの焦点は、単純なテキストのクリーニングから、物理世界からの多次元信号のキャプチャ、位置合わせ、および再構築へと移りました。言語モデル データ エンジニアリングが「ノイズ除去」に関するものであれば、マルチモーダル データ エンジニアリングは「相関」と「調整」に関するものです。 GPT-4V、Gemini、Sora の出現により、私たちは単一モダリティ データではもはや世界を理解したいというモデルの欲求を満たすことができないことに気づきました。

この章では、10 億規模のマルチモーダル データセットを構築するための完全なエンジニアリング パイプラインの詳細な分析を提供します。これは、画像をダウンロードするためのいくつかのスクリプトを記述するだけではなく、ネットワーク プロトコル、分散ストレージ、異種コンピューティング、美的評価を含む包括的なキャンペーンです。データ パラダイムの基礎となるロジックを調査し、分散コンピューティング フレームワークを使用して大規模な画像の高同時取得の課題を解決する方法を分析し、GPU ハードウェア アクセラレーションを活用して画像前処理の I/O ボトルネックを突破します。さらに、意味論的および美的基準に基づいて自動クリーニング ループを構築し、モデルに供給されるデータの関連性と安全性の両方を保証します。

**学習目標**:
* LAION-5B (画像とテキストのペア) および OBELICS (インターリーブ ドキュメント) パラダイムのトレーニングの利点とエンジニアリングの課題を深く理解し、ハイブリッド データ戦略の設計方法を習得します。
* PySpark と Ray Data に基づいて分散ダウンローダーを作成し、DNS ボトルネックとロングテール レイテンシーを処理し、10,000 画像/秒以上のスループットを達成できるようになります。
* NVIDIA DALI パイプライン設計をマスターし、CPU デコードのボトルネックを解決し、GPU Direct コンセプトを使用してデータ読み込みを最適化します。
* CLIP セマンティック フィルタリング、審美的スコアリング、安全性検出を含む多段階のクリーニング ファネルを構築し、さまざまなビジネス シナリオに合わせたしきい値調整戦略をマスターします。

**シナリオの紹介**:
> 「次のシナリオを想像してみてください。あなたのクローラ チームは、Common Crawl から 20 億の生の URL を抽出し、数千の Parquet ファイルに保存しました。あなたのタスクは、このデータを 2 週間以内に GPT-4V の事前トレーニングに適した高品質のデータセットに変換することです。単一マシン上で従来の Python リクエスト ライブラリを使用してダウンロードしようとすると、推定時間は驚異の 15 年であることがわかります。これは、古典的なネットワーク I/O ブロック問題です。さらに悪いことに、予備サンプリングにより、次のことがわかります。ダウンロードされた画像の 30% は電子商取引広告 (ノイズだらけ)、15% には深刻な NSFW コンテンツが含まれており、このデータを直接使用すると、数百万ドルのコンピューティングを無駄にするだけでなく、トレーニングされたモデルが禁止されたコンテンツを生成することによって法的リスクに直面する可能性があります。この課題に対処するには、産業グレードの高スループットのインテリジェントなデータ エンジニアリング ソリューションが必要です。」

### 6.1 データパラダイム: 画像とテキストのペア (LAION-5B) とインターリーブされたドキュメント (OBELICS/MMC4)

データ パイプラインを設計する前に、私たちの最初の責任は、データの組織形式を明確にすることです。これはストレージ構造に関するだけでなく、トレーニングの目的と下流モデルの創発的な機能も直接決定します。異なるデータ形式は、本質的に、「知識が世界にどのように存在するか」の異なる抽象化です。

#### 6.1.1 中心となる概念と原則

**画像とテキストのペア**
CLIP、ALIGN、LAION-5B に代表されるマルチモーダル学習の基礎です。
* **理論的分析**: このパラダイムは、画像 $I$ とテキスト $T$ の間に強い意味的相関関係があり、この相関関係は独立していて原子的であると仮定しています。通常、トレーニングの目的は、共有Embedding空間で $I$ と $T$ のコサイン類似度を最大化することです (対照学習)。その利点は、非常に高い「信号対ノイズ比」の改良の可能性です。対照学習を通じて、モデルはオブジェクトと語彙の間の直接マッピングを学習できます。
* **エンジニアリングの観点**: データ構造は単純で、通常は `(url, caption, metadata)` のフラット化されたレコードとして表されます。このデータは、シャーディングやランダム シャッフルが非常に簡単です。トレーニング中に、サンプルの独立性により、グローバル バッチ シャッフリングを簡単に実装して、対比学習の効果を向上させることができます。

**インターリーブされた画像とテキストのドキュメント**
は、OBELICS や MMC4 に代表される、新世代のマルチモーダル大型モデル (Flamingo、GPT-4V、MM1 など) の重要な燃料です。
* **理論分析**: このパラダイムは、`<text>, <image>, <text>...` のシーケンスとして表示されるデータとともに、Web ページの元の DOM 構造順序を保存します。これにより、モデルは「マルチモーダル コンテキスト依存関係」(マルチモーダル インコンテキスト学習) を学習するようになります。たとえば、「ケーキの作り方」Web ページでは、最初の画像 (材料) と 5 番目の画像 (最終製品) の関係、および周囲のテキストとの論理的接続は、画像とテキストのペアでは提供できません。画像とテキストが混在した文書を人間が読む認知プロセスをシミュレートします。
* **エンジニアリングの観点**: データ パイプラインは非常に複雑です。単一のサンプル (ドキュメント) は可変長であり、複数の画像が含まれる場合があるため、バッチの組み立てが困難になります。従来の Collat​​or では、複雑なパディング戦略が必要です。さらに、クリーニングの際には、ドキュメントの整合性を維持するように注意する必要があります。低品質の画像を恣意的に削除すると、コンテキスト ロジックが壊れ、モデルが誤った参照関係を学習する可能性があります。

#### 6.1.2 アーキテクチャ上の決定: パラダイム比較表

リソースが限られている中で、これら 2 つのデータ パラダイムのバランスをどのようにとればよいのでしょうか?これは単純な二者択一ではなく、モデル アーキテクチャ、トレーニング コスト、最終的なアプリケーション シナリオの間の深いトレードオフが関係します。

初期のマルチモーダル研究 (2021 年以前) では、モデルがすべてを学習するには十分なデータ量 (CLIP の 4 億ペアなど) があれば十分であると業界は広く信じていました。しかし、GPT-4V の出現により、画像とテキストのペアのみでトレーニングされたモデルは、「これは猫である」を正確に識別できる一方で、論理的推論のためのコンテキストが欠如しているため、「画像内のこの猫が何をするのか」には答えることができないことがわかりました。逆に、インターリーブされたドキュメントはロジックが豊富ですが、データがまばらで、処理コストが非常に高くなります。

以下の表は、エンジニアリング実装レベルでの 2 つのパラダイム間の主な違いを比較しており、アーキテクトが実際の要件に基づいて技術的な選択を行うのに役立ちます。

|寸法 |画像とテキストのペア (LAION スタイル) |インターリーブドドキュメント (OBELICS スタイル) |徹底した分析と推奨事項 |
| :--- | :--- | :--- | :--- |
| **トレーニングの目的** |対照学習 (CLIP)、テキストから画像へ (安定拡散) |次のトークンの予測、マルチモーダルダイアログ (GPT-4V) | **ハイブリッド戦略が最善策です**。研究によると、インターリーブされたドキュメントのみを使用してビジュアル エンコーダをトレーニングするのは非効率的 (画像の密度が十分でない) が、画像とテキストのペアのみを使用すると推論能力が不足することがわかっています。カリキュラム学習戦略をお勧めします。 |
| **データ ソースの解析** |シンプル: `<img>` タグと代替テキストを抽出するだけです。複雑: DOM ツリーを解析し、広告/サイドバーをフィルタリングし、メイン コンテンツ ロジックを保持する必要があります。 **エンジニアリングの複雑さに関する警告**。インターリーブされたドキュメントを構築するには、非常に複雑な HTML レンダリング ロジックを処理する必要があります。インターネット全体を最初から再クリーンしようとするのではなく、最初は Common Crawl の WET ファイルを構築に使用するか、OBELICS オープンソース データセットを拡張に直接使用することをお勧めします。 |
| **保管コスト** |中: メタデータは CSV/Parquet のみ、画像は個別に保存 |高: ドキュメント トポロジを保存する必要があるため、WebDataset または TFRecord を推奨します。 **I/O パフォーマンスのボトルネック**。インターリーブされたドキュメントの場合、小さなファイルの断片化を避けるために、シャード ストレージを使用する必要があります。読み取りには文書全体を事前に読み取る必要があり、メモリ帯域幅の要求が高くなります。 |
| **清掃の課題** |ポイントごと: 各画像は独立して判断され、並列化が容易 |コンテキスト: クリーニング ロジックを組み合わせて、テキストの一貫性と画質を同時に考慮する必要があります。 **戦略の選択**。インターリーブされたドキュメントを処理するときに、画像が NSFW であるとみなされる場合は、位置Embeddingの精度を維持するために、画像を直接削除するのではなく、特別な `<BLOCKED_IMAGE>` トークンに置き換えることをお勧めします。 |
| **モデルのメリット** |非常に強力な視覚と意味の整合性、強力なゼロショット分類 |強力な少数ショット学習、マルチターン対話と論理的推論をサポート | **ビジネス指向**。シナリオが「画像検索」の場合は、画像とテキストのペアで十分です。ビジネスに複雑な文書の理解 (調査レポートの分析、長編ストーリーの生成など) が含まれる場合は、インターリーブ文書を導入する必要があります。 |

> **ヒント:**
> MM1 や Idefics2 のような最先端の研究では、どちらかを選択するのではなく、混合することがベスト プラクティスです。通常、**20% のインターリーブされたドキュメント**を混合しながら、強固な視覚言語マッピング基盤を確立するために、初期の事前トレーニング段階で **80% の画像とテキストのペア**を使用することが推奨されます。後期の事前トレーニング段階 (アニーリング段階) では、インターリーブされたドキュメントの割合を大幅に増やして、モデルのロングコンテキスト推論能力を刺激します。この「基盤が先、ロジックは後」戦略により、コンピューティング使用率が最大化されます。

### 6.2 画像の取得と前処理

データ マニフェストが決定したら、次のステップは高スループットのダウンロードおよび前処理パイプラインを構築することです。これは典型的な I/O 集中型のタスクであり、ネットワーク帯域幅、DNS 解決の遅延、および大量の小さなファイルのディスク書き込みに主なボトルネックがあります。

#### 6.2.1 img2dataset の高同時ダウンロードの実際

`img2dataset` は現在、コミュニティで認められたベスト プラクティス ツールです。これは単なるダウンロード スクリプトではなく、MapReduce の原則に基づいた分散データ処理フレームワークです。

単純な `requests.get` ループを作成するのではなく、専用のツールを使用する理由は何でしょうか?インターネット環境が非常に厳しいため。リンクの有効期限が切れ（リンク ロット）、サーバーのレート制限が発生し、DNS がタイムアウトになります。数十億の URL を処理する場合、わずかなロングテール遅延が数週間の時間コストに増幅されます。

**中心原則**:
1. **シャーディング**: 10 億の URL を数万の小さなタスク (シャード) に分割します。これは分散コンピューティングの基礎です。
2. **非同期 I/O**: Python の aiohttp または Go のゴルーチンを使用して、コアごとに数百のネットワーク リクエストを同時に開始し、ネットワーク遅延をマスクします。
3. **ストリーミング アーカイブ**: ダウンロードされたイメージはディスクに書き込まれません。これらはメモリ内で tar パッケージ (WebDataset 形式) に直接アセンブルされ、オブジェクト ストレージ (S3/HDFS) にストリーミングされます。これにより、1 つのディレクトリに何百万もの小さなファイルを作成するときにファイルシステムの i ノードが使い果たされることがなくなります。これは、初心者がよく遭遇する落とし穴です。

**エンジニアリング実装: PySpark 分散ダウンロード スクリプト**

PB スケールのデータを処理する場合、単一マシンのマルチ処理モードでは不十分です。 Spark クラスターを使用する必要があります。

```python
# Recommended environment: PySpark 3_2+, img2dataset 1_41+
# Run command: spark-submit --master yarn --deploy-mode cluster...

from img2dataset import download
import shutil
import os

def run_distributed_download():
    """
    Configuration tuning is key to throughput.
    process_count: Number of processes per Spark Executor.
    thread_count: Number of async threads per process.
    For nodes with 10Gbps NIC, typically recommend total_concurrency around 1000.
    """
    
    # Define output path (S3 or HDFS)
    output_dir = "s3a://multimodal-lake/raw-images/laion-5b-subset"
    
    # Clean old data (use with caution, production recommends versioning)
    if os.path.exists(output_dir): 
        # shutil.rmtree(output_dir) # Dangerous operation, commented out
        pass

    download(
        processes_count=4,          # 4 CPU cores per node
        thread_count=64,            # 64 download threads per core
        url_list="s3a://multimodal-lake/meta/laion-urls.parquet",
        image_size=256,             # 256x256 sufficient for pre-training, saves bandwidth
        resize_only_if_bigger=True, # Avoid blur from upscaling small images
        resize_mode="keep_ratio",   # Maintain aspect ratio, pad or center crop
        skip_reencode=True,         # If original is JPG and size suitable, store directly, saves CPU
        output_folder=output_dir,
        output_format="webdataset", # Force WebDataset format
        input_format="parquet",
        url_col="url",
        caption_col="caption",
        enable_wandb=True,          # Strongly recommended for monitoring download rate and error rate
        number_sample_per_shard=10000, # 10k images per tar, ~200-300MB, easy to transfer
        distributor="pyspark",      # Use Spark for task distribution
        save_additional_columns=["similarity", "hash"], # Preserve original metadata
        timeout=10                  # Short timeout, fail fast, long-tail requests not worth waiting
    )

if __name__ == "__main__":
    # Initialize Spark Session (usually handled by spark-submit, but declare explicitly for IDE debugging)
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("Img2Dataset-Production") \
        .config("spark.executor.memory", "8g") \
        .config("spark.task.maxFailures", "10") \
        .getOrCreate()
    
    run_distributed_download()
```

**プロのヒント**:
* **DNS キャッシュ**: 同時実行性が高い場合、DNS 解決がボトルネックになったり、プロバイダーによってブロックされたりする可能性があります。ローカル DNS キャッシュ (例: dnsmasq) をワーカー ノードにデプロイするか、コードでドメインから IP へのマッピング テーブルを維持します。
* **ユーザー エージェントのローテーション**: 「公然の」秘密ではありますが、ユーザー エージェントをローテーションすると 403 Forbidden の割合を減らすことができます。
* **エラー処理**: WandB ダッシュボードで success_rate を監視します。 80% を下回る場合は、通常、URL リストが非常に古いか、IP プールが汚染されていることを意味します。

#### 6.2.2 視覚的な前処理の落とし穴: トリミングとセマンティック アラインメント

大量のデータの取得 (バイトの取得) という課題を解決した後、すぐに 2 番目の課題、データの使いやすさに直面します。生のインターネット画像のアスペクト比は大きく異なりますが、モデルでは通常、固定解像度の入力 (例: 224x224 または 512x512) が必要です。

初心者向けのエンジニアリング ソリューションの多くは、次元を統一するために総当たり的なランダム前処理を習慣的に使用していますが、これがモデルの「目に見えないパフォーマンスの上限」の根本であることがよくあります。 「イメージを取り込む」だけではなく、「何が入れられているか」にも焦点を当てなければなりません。



![図 6-1: 画像前処理におけるトリミングとセマンティック調整](../../images/part3/图6_1_图片预处理中裁剪与语义对齐问题.png)
*図 6-1: 画像の前処理におけるトリミングとセマンティック アライメント*

* **悪いケース (左 - 機械的トリミングのコスト)**:
    従来の `RandomCrop` または `CenterCrop` は構成を認識しません。縦方向の構図でポートレート写真を処理する場合、中央をトリミングすると重要な部分 (頭など) が簡単に切り取られ、胴体だけが残ることがあります。この時点で、テキスト ラベルがまだ「笑顔の人」である場合、モデルは誤ったマッピング (胴体の特徴を「笑顔の人」と誤認) を確立することを強制され、トレーニング済みモデルに重度の幻視が発生します。

* **良いケース (右 - セマンティック整合性)**:
    「画像とテキストの一貫性」を追求した高品質なデータエンジニアリング。
    1. **スマート サイズ変更**: 完全な視覚的主題を維持するには、`Resize with Padding` (アスペクト比を維持し、黒/白の境界線でパディング) を優先します。これにより無効なピクセルが発生しますが、セマンティックな整合性は確保されます。
    2. **アスペクト比バケット**: これは、SDXL や Midjourney などの世代モデルで一般的に使用される高度なテクニックです。同様のアスペクト比を持つ画像をトレーニング用に同じバッチにグループ化し、パディングの無駄を減らしながらトリミングを回避します。
    3. **要約**: 第 7 章で詳しく説明されているように、VLM を使用して高密度の説明を生成すると、テキストが画像内の詳細 (標識テキスト、背景オブジェクトなど) に正確に対応し、データのトレーニング値が最大化されます。

#### 6.2.3 GPU アクセラレーションによるデコードと変換 (NVIDIA DALI)

ディープ ラーニング モデルのトレーニング フェーズでは、ほとんどの研究者や開発者がモデル アーキテクチャの設計、ハイパーパラメータ調整、損失関数の改善、およびモデルの精度に直接影響するその他のモジュールに注目しますが、データ ロード (DataLoader) フェーズは見落とされがちですが、実際にはそれがトレーニング効率を制限する「目に見えないパフォーマンス キラー」になることが多く、さらにはハイエンド GPU コンピューティングの完全な利用を妨げ、重大なハードウェアの浪費を引き起こすことさえあります。

この問題点を理解するには、まず深層学習トレーニング フローの完全なロジックを明確にする必要があります。モデル トレーニングのコア コンピューティングは、GPU の大規模な並列コンピューティング機能に依存しています。 GPU は大規模なテンソル演算を効率的に処理し、バックプロパゲーションとパラメータの更新を完了できます。ただし、データが GPU に到達する前に、一連の前処理操作を通過する必要があり、その中で最も基本的で時間がかかるのは画像のデコードとサイズ変更です。従来の PyTorch トレーニング フローでは、これらの重要な前処理操作はすべて CPU によって実行され、「CPU 前処理のボトルネック」と「GPU コンピューティングの冗長性」の間に矛盾が生じます。

具体的には、従来の PyTorch データセットのワークフローは次のとおりです。まず、ディスクに保存されている画像ファイル (主に JPEG 形式) を CPU 経由で読み取り、次に CPU が JPEG デコードを実行します。このプロセスには、圧縮画像バイナリ データに対するハフマン デコードや逆離散コサイン変換 (IDCT) などの複雑な計算が必要であり、一般的な CPU 負荷の高いタスクです。デコード後、CPU はサイズ変更、正規化、色空間変換、その他の前処理を実行し、最後にデータ コピーを介してモデル トレーニングのために処理された画像テンソルを GPU に転送します。

さらに重要なのは、CPU アーキテクチャはシリアル計算とロジック制御用に設計されており、並列計算能力は GPU よりもはるかに劣っています。ただし、画像の前処理におけるデコードとサイズ変更は高度に並列化可能であり、マルチスレッドまたはマルチコア処理を通じて効率を向上させることができます。しかし、DataLoader の num_workers パラメータを使用して CPU 並列処理を増やしても、従来の PyTorch データセットは CPU の計算上限を突破するのに苦労します。特に、トレーニング データセットが大きく (例: 数百万の画像)、単一画像の解像度が高い (例: 1080P 以上) 場合、CPU の前処理速度が GPU トレーニング速度より大幅に遅れ、データ待ちで GPU が頻繁にアイドル状態になり、GPU 使用率が大幅に低下します。最終的には全体的なトレーニング効率が低下します。これが、データの読み込みが「無視されたパフォーマンスキラー」と呼ばれる理由です。

この核心的な問題点に対処するために、NVIDIA は、ディープ ラーニング トレーニング用に最適化された GPU 高速データ前処理ライブラリである DALI (Data Loading Library) を導入しました。その中心的な目標は、もともと CPU に依存していた画像のデコード、サイズ変更、その他の集中的な前処理操作を並列実行のために GPU に移行し、データ読み込みパフォーマンスのボトルネックを解消し、GPU コンピューティングを解放することです。


![図 6-2: DALI を使用した場合と使用しない場合のデータのデコードと変換](../../images/part3/图6_2_使用DALI与不使用DALI下数据解码与变换的区别.png)
*図 6-2: DALI を使用した場合と使用しない場合のデータのデコードと変換*

**コードのチュートリアル: DALI に基づく高性能パイプライン**

```python
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def


@pipeline_def(batch_size=256, num_threads=8, device_id=0)
def webdataset_gpu_pipeline(shard_id, num_shards):
    """
    Define end-to-end GPU data loading pipeline
    Input: WebDataset (Tar) -> Output: GPU Tensor
    """
    
    # Step 1: Read WebDataset (CPU stage)
    # Using index_paths is required; otherwise init phase needs to traverse entire tar, extremely slow [5]
    jpegs, captions = fn.readers.webdataset(
        paths=["/data/shards/shard-{:05d}.tar".format(i) for i in range(100)],
        index_paths=["/data/indices/shard-{:05d}.idx".format(i) for i in range(100)],
        ext=["jpg", "txt"],
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=True,
        initial_fill=10000,      # Shuffle buffer size, larger = more random but slower startup
        pad_last_batch=True,     # Ensure all batches have consistent size
        name="Reader",
        read_ahead=True          # Enable prefetch
    )

    # Step 2: GPU Decoding (core acceleration point)
    # device="mixed" means input in Host memory, output in Device memory
    # output_type=types.RGB handles color space conversion automatically
    images = fn.decoders.image(
        jpegs,
        device="mixed",
        output_type=types.RGB,
        # Fault tolerance for corrupted images
        # In production, never let one bad image crash training
    )

    # Step 3: GPU transformation pipeline
    # resize: scale while maintaining aspect ratio
    images = fn.resize(
        images,
        resize_x=224,
        resize_y=224,
        interp_type=types.INTERP_LINEAR
    )
    
    # crop_mirror_normalize: random crop + flip + normalize (fused operator)
    # This step converts uint8 to float and subtracts mean, divides by std
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(224, 224),
        mean=[0_485 * 255, 0_456 * 255, 0_406 * 255],
        std=[0_229 * 255, 0_224 * 255, 0_225 * 255],
        mirror=fn.random.coin_flip(probability=0_5)
    )

    # Text data typically processed directly on CPU or passed to Tokenizer
    # Here we only return raw bytes for subsequent PyTorch processing
    return images, captions

# Use DALIGenericIterator integrated with PyTorch
from nvidia.dali.plugin.pytorch import DALIGenericIterator

pipe = webdataset_gpu_pipeline(shard_id=0, num_shards=1)
pipe.build()
dataloader = DALIGenericIterator(pipe, ["images", "captions"], reader_name="Reader")

# Benchmark: On A100, this pipeline typically achieves 3000-5000 FPS, 5-10x CPU Loader
```

### 6.3 マルチモーダルな洗浄パイプライン

大量のデータには大量のノイズが伴います。 LAION-5B の生データでは、真に高品質のサンプルは 10% 未満である可能性があります。データの多様性をできるだけ少なくしながらデータ密度を向上させるには、多段階のクリーニング ファネルを構築する必要があります。いわゆる「データ クリーニング」は本質的に **データ ダイエット** であり、モデルへのフィードの量を減らし、より良くすることです。

#### 6.3.1 アーキテクチャ設計: レイ データ分散クリーニング

クリーニングフェーズにスパークではなくレイを選択する理由は何ですか?なぜなら、クリーニングにはもはや単純な ETL ではなく、実質的な **深層学習推論 (モデル推論)** が含まれるからです。 Spark の MapReduce パラダイムと比較して、Ray はより柔軟な Actor メカニズムを提供し、GPU モデル (CLIP、Safety Checker など) を常駐させ、小さなバッチごとに複数 GB のモデルをリロードするという膨大なオーバーヘッドを回避できます。

Ray Data は、CPU 集中型 (解凍、ハッシュ、正規表現) タスクと GPU 集中型 (CLIP Embedding推論) タスクの両方の混合ワークロードを処理するのに適しています。以下は、一般的な 3 段階のパイプライン設計です。
* **ステージ 1 (CPU)**: クイック フィルタリング。解像度が不十分 (<256px)、テキストが短すぎる、英語以外 (英語のみのモデルをトレーニングしている場合)、または異常なアスペクト比のサンプルを直接削除します。
* **ステージ 2 (GPU)**: 詳細な特徴抽出。 CLIP モデルを使用してエンベディングを生成し、エンベディングに基づいて画像とテキストの類似性と美的スコアを計算します。
* **ステージ 3 (CPU/混合)**: 論理的な判断と重複排除。安全性 (NSFW)、美的スコア、画像とテキストの関連性、さらにセマンティックな重複排除に基づいた包括的なしきい値処理。

**データ フロー図**

![図 6-3: Ray Data 分散型洗浄データの流れ](../../images/part3/图6_3_Ray_Data分布式清洗数据流向图.png)
*図 6-3: Ray データ分散クリーニング データ フロー*

#### 6.3.2 コアアルゴリズムの実装

クリーニングとは、単に削除するだけではなく、データの価値を数値化することです。画像とそれに対応するテキストの「ゴールド コンテンツ」を測定するには、多次元のメトリクスが必要です。

1. **美的スコア**
    * **原則**: データセットは請求書、スクリーンショット、不鮮明な監視映像でいっぱいです。これらは美しい画像を生成するのには役に立ちません。通常は LAION-Aesthetics Predictor が使用されます。
    * **技術的な詳細**: これは単純な MLP (多層パーセプトロン) です。入力は CLIP 画像Embedding、出力は 1 ～ 10 のスコアです。トレーニング データは AVA データセット (プロの写真家からのスコアを含む) から取得されます。
    * **推奨されるしきい値**: 基本的な事前トレーニングの場合は、スコア > 4_5 のデータを保持します。高品質の生成モデル (SFT フェーズ) を微調整するには、スコア > 6_0、または 6_5 を推奨します。

2. **画像とテキストの配置フィルタリング**
    * **原則**: 代替テキストの多くは、画像コンテンツとは関係のない、SEO のゴミ単語の積み重ねやファイル名 (「DSC_001.jpg」) です。
    * **技術詳細**: CLIP 画像EmbeddingとテキストEmbeddingの間のコサイン類似度 (ドット積) を計算します。
    * **注意点**: CLIP のバージョンが異なると (例: OpenAI ViT-L/14 と OpenCLIP ViT-G/14)、エンベディング スペースの分布も異なります。スコアは直接比較できません。特定のモデルに基づいてしきい値を再調整する必要があります。一般的な方法は、データセット全体にわたる類似度の分布を計算し、上位 50% または上位 70% を保持することです。

3. **安全検知**
    * **原則**: ポルノ、暴力、明らかなブランドの透かしのある画像は削除する必要があります。
    * **戦略**: 特別に訓練された分類子ヘッド (これも CLIP Embeddingに基づく) を使用して、NSFW とウォーターマークを検出します。透かし検出の場合: 生成モデル (SDXL など) をトレーニングすることが目的の場合、生成モデルは透かし特徴を容易にオーバーフィットするため、非常に厳密 (リコール優先) にする必要があります。目標が理解モデル (GPT-4V など) のトレーニングである場合、理解モデルは「画像に透かしがある」ことを認識する必要があるため、緩和することができます。

**コードの実装: レイ データ クリーニング オペレーター**

```python
import ray
import torch
import open_clip
import numpy as np
from PIL import Image
import io

# Define Ray Actor class to ensure model is loaded only once
class QualityScorer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load CLIP model (ViT-B-32 fast, suitable for cleaning)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device
        )
        # Load aesthetic scoring head (Linear Layer)
        self.aesthetic_head = torch.nn.Linear(512, 1).to(self.device)
        self.aesthetic_head.load_state_dict(torch.load("sac+logos+ava1-l14-linearMSE.pth"))
        self.aesthetic_head.eval()

    def __call__(self, batch: dict) -> dict:
        """
        Process one batch of data. Ray automatically shards and transfers data to Actor.
        """
        images = []
        valid_indices = []
        
        # Preprocess images (CPU operation)
        for idx, img_bytes in enumerate(batch["jpg"]):
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img_tensor = self.preprocess(img)
                images.append(img_tensor)
                valid_indices.append(idx)
            except Exception:
                # Log bad image but don't interrupt
                continue
        
        if not images:
            return {"aesthetic_score": [], "clip_score": []}

        image_input = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            # 1. Extract features
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # 2. Compute aesthetic score
            aesthetic_scores = self.aesthetic_head(image_features).squeeze().cpu().numpy()
            
            # 3. Compute image-text match (assuming batch has text field)
            # text_tokens = self.tokenizer(batch["txt"]).to(self.device)
            # text_features = self.model.encode_text(text_tokens)
            #... compute cosine similarity
            
        # Return results (note alignment with original batch indices)
        return {"aesthetic_score": aesthetic_scores}

# Orchestrate Ray pipeline
ray.init()
ds = ray.data.read_webdataset("s3://raw-bucket/{00000..00099}.tar")

# map_batches automatically schedules GPU resources
# num_gpus=0_25 means one GPU can run 4 Actors concurrently, improving throughput
scored_ds = ds.map_batches(
    QualityScorer, 
    compute=ray.data.ActorPoolStrategy(size=8), 
    num_gpus=0_25, 
    batch_size=128
)

# Final filtering
filtered_ds = scored_ds.filter(lambda row: row["aesthetic_score"] > 4_5)
filtered_ds.write_webdataset("s3://clean-bucket/")
```

### 6.4 落とし穴とトラブルシューティング

数十億規模のマルチモーダル データセットを構築する場合、エンジニアリング チームは詳細でつまずくことがよくあります。つらい経験から学んだ教訓は次のとおりです。

* **寄木細工のメタデータの爆発的増加**:
    * **エラー**: パンダで 20 億行を含む Parquet ファイルを直接読み取る習慣があります。
    * **結果**: メモリ不足 (OOM)。パンダは 1 列だけを読み取る場合でもインデックス全体をメモリにロードしようとするためです。
    * **修正**: Polars または PySpark の遅延評価モードを使用します。または、単一の巨大なメタデータ ファイルの処理を避けるために、行数 (100 万行など) ごとに Parquet ファイルをより小さいファイルに厳密に分割します。

* **WebDataset シャッフルが不十分です**:
    * **エラー**: ダウンロード中に、データはドメイン順に書き込まれます。トレーニング中は、DataLoader のバッファ シャッフル (通常は 10k のバッファ) のみに依存します。
    * **結果**: モデルは 100,000 個の e-コマース画像を連続して表示し、次に 100,000 個の風景画像を表示する可能性があります。バッファーが小さいと、この「時間的相関」を打ち破ることができず、トレーニング カーブの激しい振動や発散さえも引き起こします。
    * **修正**: WebDataset を書き込む前に、URL リストに対して**グローバル シャッフル**を実行する必要があります。 Spark の `orderBy(rand())` を使用できます。

* **ロングテール データを誤って削除してしまう**:
    * **エラー**: 極端な美的スコアを追求し、スコア < 4_5 の画像をすべて削除します。
    * **結果**: モデルは「狭く」なり、医療画像、街路風景、手書きのメモなどの現実世界の (おそらく醜い) 写真ではなく、アート写真と壁紙のみを認識します。モデルの一般化を大幅に軽減します。
    * **修正**: 層別サンプリングを使用します。 5% ～ 10% の低スコア データを「正規化」として保持するか、美的フィルターをバイパスする特定のドメイン (OCR、グラフなど) のホワイトリストを確立します。

* **重複データの隠れた危険 (重複排除)**:
    * **エラー**: インターネット上の大量の重複画像 (ミーム、バイラル ニュース画像など) を無視します。
    * **結果**: モデルは特定のサンプルをオーバーフィットし、生成中にトレーニング セットの画像を「記憶」することもあり、深刻な著作権問題につながります。
    * **修正**: クリーニング パイプラインに**セマンティック重複排除**を追加する必要があります。すべての画像のEmbeddingを計算し、クラスタリングに Faiss または MinHashLSH を使用し、類似性の高いグループごとに 1 つの画像のみを保持します。
