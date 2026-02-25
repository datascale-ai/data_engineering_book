## 第 6 章: 画像とテキストのペアのデータ処理

### 章の概要

次世代の基礎モデルを構築する過程において、データ エンジニアリングの焦点は、単なるテキストのクリーニングから、物理世界からの多次元信号のキャプチャ、位置合わせ、再構築へと移りました。言語モデル データ エンジニアリングが「ノイズ除去」に関するものであれば、マルチモーダル データ エンジニアリングは「関連付け」と「調整」に関するものです。 GPT-4V、Gemini、Sora の出現により、単一モダリティ データではもはや世界を理解したいというモデルの欲求を満たすことができないことがわかりました。

この章では、10 億規模のマルチモーダル データセットを構築するための完全なエンジニアリング パイプラインの詳細な分析を提供します。これは、画像をダウンロードするためのいくつかのスクリプトを記述するだけではなく、ネットワーク プロトコル、分散ストレージ、異種コンピューティング、美的評価を含む包括的なキャンペーンです。データ パラダイムの基礎となるロジックを調査し、分散コンピューティング フレームワークを活用して大規模な画像の同時実行性の高い取得の課題を解決する方法を分析し、GPU ハードウェア アクセラレーションを使用して画像前処理の I/O ボトルネックを突破します。さらに、セマンティクスと美学に基づいて自動クリーニング パイプラインを構築し、モデルに入力されるデータの関連性と安全性を確保します。

**学習目標**:
* LAION-5B (画像とテキストのペア) および OBELICS (インターリーブ ドキュメント) パラダイムのトレーニングの利点とエンジニアリングの課題を理解し、ハイブリッド データ戦略の設計を習得します。
* PySpark と Ray Data に基づいて分散ダウンローダーを作成し、DNS ボトルネックとロングテール レイテンシーを処理し、10,000 画像/秒以上のスループットを達成できるようになります。
* NVIDIA DALI パイプライン設計をマスターし、CPU デコードのボトルネックを解決し、GPU ダイレクト原理を使用してデータ読み込みを最適化します。
* CLIP セマンティック フィルタリング、審美的スコアリング、安全性検出を含む多段階のクリーニング ファネルを構築し、さまざまなビジネス シナリオに合わせたマスターしきい値調整戦略を構築します。

**シナリオの紹介**:
> 「次のシナリオを想像してください。私たちのクローラ チームは、Common Crawl から 20 億の生の URL を抽出し、数千の Parquet ファイルに保存しました。あなたの仕事は、このデータを 2 週間以内に GPT-4V の事前トレーニングに適した高品質のデータセットに変換することです。単一マシンで従来の Python リクエスト ライブラリを使用してダウンロードしようとすると、推定時間が 15 年もかかることがわかります。これは典型的なネットワーク I/O ブロック問題です。さらに悪いことに、予備サンプリングでは、ダウンロードされた画像は電子商取引広告 (ノイズだらけ)、15% には重大なウォーターマークが含まれており、重大な NSFW コンテンツさえあります。このデータを直接使用すると、計算に数百万ドルを無駄にするだけでなく、トレーニングされたモデルが不快なコンテンツを生成するために法的リスクに直面する可能性があります。この課題に対処するには、産業グレードの高スループットのインテリジェントなデータ エンジニアリング ソリューションが必要です。」

### 6.1 データ パラダイム: 画像とテキストのペア (LAION-5B) とインターリーブされたドキュメント (OBELICS/MMC4)

データ パイプラインを設計する前に、私たちの最初の責任はデータの編成形式を明確にすることです。これはストレージ構造だけでなく、トレーニングの目的と下流モデルの創発的な機能も直接決定します。異なるデータ形式は、本質的に、「知識が世界にどのように存在するか」の異なる抽象化です。

#### 6.1.1 中心となる概念と原則

**画像とテキストのペア**
CLIP、ALIGN、LAION-5B に代表されるマルチモーダル学習の基礎です。
* **理論的分析**: このパラダイムは、画像 $I$ とテキスト $T$ の間に強い意味的関連性があり、この関連性が独立かつ原子的であることを前提としています。通常、トレーニングの目的は、共有Embedding空間で $I$ と $T$ のコサイン類似度を最大化することです (対照学習)。その利点は、非常に高い「信号対ノイズ比」の洗練の可能性にあります。モデルは、対照学習を通じて、オブジェクトと語彙の間の直接マッピングを学習します。
* **エンジニアリングの観点**: データ構造は単純で、通常は `(url, caption, metadata)` のフラット化されたレコードとして表されます。このデータは、シャーディングやランダム シャッフルが非常に簡単です。トレーニング中、サンプルは独立しているため、グローバル バッチ シャッフリングを簡単に実装して、対比学習の効果を向上させることができます。

**インターリーブされた画像とテキストのドキュメント**
OBELICSやMMC4に代表される次世代マルチモーダル大型モデル（Flamingo、GPT-4V、MM1など）の鍵となる燃料です。
* **理論分析**: このパラダイムは、`<text>, <image>, <text>...` のシーケンスとして表示されるデータとともに、Web ページの元の DOM 構造順序を保存します。これにより、モデルは「マルチモーダル コンテキスト依存関係」(マルチモーダル インコンテキスト学習) を学習するようになります。たとえば、「ケーキの作り方」Web ページでは、画像 1 (材料) と画像 5 (完成品) の関係、および周囲のテキストとの論理的接続は、画像とテキストのペアでは提供できません。人間が絵本を読んでいるときの認知プロセスをシミュレートします。
* **エンジニアリングの観点**: データ パイプラインは非常に複雑です。個々のサンプル (ドキュメント) は可変長であり、複数の画像が含まれる場合があるため、バッチの組み立てが困難になります。従来の Collat​​or では、複雑なパディング戦略が必要です。さらに、クリーニング中はドキュメントの整合性を注意深く維持する必要があります。低品質の画像を恣意的に削除すると、コンテキスト ロジックが破損し、モデルが誤った参照関係を学習する可能性があります。

#### 6.1.2 アーキテクチャの決定: 比較表

リソースが限られている中で、これら 2 つのデータ パラダイムをどのように比較検討すればよいでしょうか?これは単純な二者択一ではなく、モデル アーキテクチャ、トレーニング コスト、最終的なアプリケーション シナリオ間の深いトレードオフが関係します。

初期のマルチモーダル研究 (2021 年以前) では、データ量が十分である限り (CLIP の場合は 400M ペアなど)、モデルはすべてを学習できると業界は広く信じていました。しかし、GPT-4V の出現により、画像とテキストのペアのみでトレーニングされたモデルは、「これは猫である」を正確に識別できるものの、論理的な推論コンテキストが欠如しているため、「画像内のこの猫が何をするか」には答えることができないことがわかりました。逆に、インターリーブされたドキュメントはロジックが豊富ですが、データがまばらで、処理コストが非常に高くなります。

以下の表は、アーキテクトが実際のニーズに基づいて技術的な選択を行えるようにすることを目的として、エンジニアリング実装レベルでの 2 つのパラダイム間の主な違いを比較しています。

|寸法 |画像とテキストのペア (LAION スタイル) |インターリーブドドキュメント (OBELICS スタイル) |徹底した分析と推奨事項 |
| :--- | :--- | :--- | :--- |
| **トレーニングの目的** |対照学習 (CLIP)、テキストから画像へ (安定拡散) |次のトークンの予測、マルチモーダルダイアログ (GPT-4V) | **ハイブリッド戦略は王様です**。研究によると、インターリーブされたドキュメントのみを使用してビジュアル エンコーダをトレーニングするのは非効率的 (画像の密度が十分でない) が、画像とテキストのペアのみを使用すると推論機能が不足することがわかっています。カリキュラム学習戦略を推奨します。 |
| **データ ソースの解析** |シンプル: `<img>` タグと代替テキストを抽出するだけです。複雑: DOM ツリーを解析し、広告とサイ​​ドバーをフィルタリングし、メイン コンテンツ ロジックを保持する必要があります。 **エンジニアリングの複雑さに関する警告**。インターリーブされたドキュメントを構築するには、非常に複雑な HTML レンダリング ロジックを処理する必要があります。最初は Common Crawl WET ファイルを使用するか、拡張のために OBELICS オープンソース セットを直接使用することをお勧めします。インターネット全体を最初からクリーンアップしようとしないでください。 |
| **保管コスト** |中: メタデータは CSV/Parquet のみで、画像は個別に保存されます。高: ドキュメント トポロジを保存する必要があるため、WebDataset または TFRecord のカプセル化を推奨します。 **I/O パフォーマンスのボトルネック**。インターリーブされたドキュメントの場合は、小さなファイルの断片化を避けるためにシャード ストレージを使用する必要があります。読み取りにはドキュメント全体を事前にロードする必要があり、メモリ帯域幅の要求が高くなります。 |
| **清掃の課題** |シングルポイントクリーニング: 各画像は独立して判断され、並列化が容易 |コンテキストのクリーニング: クリーニング ロジックを組み合わせて、テキストの一貫性と画像の品質を同時に考慮する必要があります。 **戦略の選択**。インターリーブされたドキュメントを処理するときに、画像が NSFW と判断された場合は、位置Embeddingの精度を維持するために、削除するのではなく特別な `<BLOCKED_IMAGE>` トークンに置き換えることをお勧めします。 |
| **モデルのメリット** |強力な視覚的意味論的整合、強力なゼロショット分類 |強力な少数ショット学習、複数ターンの対話と論理的推論をサポート | **ビジネス指向**。シナリオが「画像検索」の場合は、画像とテキストのペアで十分です。複雑な文書の理解 (研究レポートの分析、長編ストーリーの生成など) が含まれる場合は、インターリーブ文書を導入する必要があります。 |

> **ヒント:**
> MM1 や Idefics2 のような最先端の研究では、ベスト プラクティスは二者択一ではなく、比例することです。通常は、**20% インターリーブされたドキュメント**を混合しながら、強固な視覚言語マッピング基盤を確立するために、初期の事前トレーニング段階で **80% の画像とテキストのペア**を使用することをお勧めします。後期の事前トレーニング フェーズ (アニーリング フェーズ) では、インターリーブされたドキュメントの割合を大幅に増やして、モデルのロングコンテキスト推論能力を刺激します。この「基盤が先、ロジックは後」戦略により、コンピューティング使用率が最大化されます。

### 6.2 画像の取得と前処理

データ マニフェストが決定したら、次のステップは高スループットのダウンロードおよび前処理パイプラインを構築することです。これは典型的な I/O 集中型のタスクであり、ネットワーク帯域幅、DNS 解決の遅延、大量の小さなファイルのディスク書き込みが主なボトルネックとなります。

#### 6.2.1 img2dataset の高同時ダウンロードの実際

`img2dataset` は現在、コミュニティで認められたベスト プラクティス ツールです。これは単なるダウンロード スクリプトではなく、MapReduce の原則に基づいた分散データ処理フレームワークです。

単純な `requests.get` ループを作成する代わりに、特殊なツールが必要なのはなぜですか?インターネット環境が非常に厳しいため。リンクの期限切れ (リンクの破損)、サーバーのレート制限、DNS のタイムアウト。数十億の URL を処理する場合、わずかなロングテールの遅延が数週間の時間コストに増幅されます。

**中心原則**:
1. **シャーディング**: 10 億の URL を数万の小さなタスク (シャード) に分割します。これは分散コンピューティングの基礎です。
2. **非同期 I/O**: Python の aiohttp または Go コルーチンを使用して、単一コア上で数百のネットワーク リクエストを同時に開始し、ネットワーク遅延をマスクします。
3. **ストリーミング アーカイブ**: ダウンロードされた画像はディスクに記録されません。これらはメモリ内で tar パッケージ (WebDataset 形式) に直接アセンブルされ、オブジェクト ストレージ (S3/HDFS) にストリーミングされます。これにより、1 つのディレクトリに何百万もの小さなファイルが作成されることによるファイル システムの i ノードの枯渇の問題 (初心者がよく遭遇する落とし穴) が回避されます。

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
    For 10Gbps NIC nodes, typically recommend total_concurrency around 1000.
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
        resize_mode="keep_ratio",   # Maintain aspect ratio, black padding or center crop
        skip_reencode=True,         # If original is JPG and size acceptable, store directly, saves CPU
        output_folder=output_dir,
        output_format="webdataset", # Force WebDataset format
        input_format="parquet",
        url_col="url",
        caption_col="caption",
        enable_wandb=True,          # Strongly recommended for monitoring download rate and error rate
        number_sample_per_shard=10000, # 10k images per tar, ~200-300MB, for easy transfer
        distributor="pyspark",      # Use Spark for task distribution
        save_additional_columns=["similarity", "hash"], # Preserve original metadata
        timeout=10                  # Shorter timeout for fast failure, long-tail requests not worth waiting
    )

if __name__ == "__main__":
    # Initialize Spark Session (usually handled by spark-submit, but explicit for IDE debugging)
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("Img2Dataset-Production") \
        .config("spark.executor.memory", "8g") \
        .config("spark.task.maxFailures", "10") \
        .getOrCreate()
    
    run_distributed_download()
```

**プロのヒント**:
* **DNS キャッシュ**: 同時実行性が高い場合、DNS 解決がボトルネックになったり、プロバイダーによってブロックされたりする可能性があります。ローカル DNS キャッシュ (例: dnsmasq) をワーカー ノードにデプロイするか、コードでドメインから IP へのマッピング テーブルを維持することをお勧めします。
* **ユーザー エージェントのローテーション**: 「公然の」秘密ではありますが、ユーザー エージェントをローテーションすると 403 Forbidden の割合を減らすことができます。
* **エラー処理**: WandB ダッシュボードで success_rate を監視します。 80% を下回る場合は、通常、URL リストが著しく古いか、IP プールが汚染されていることを意味します。

#### 6.2.2 視覚的な前処理の落とし穴: トリミングとセマンティック アライメント

大量のデータの取得 (バイトの取得) という課題を解決した後、すぐに 2 番目の課題、データの使いやすさに直面します。生のインターネット画像のアスペクト比は大きく異なりますが、モデルでは通常、固定解像度の入力 (例: 224x224 または 512x512) が必要です。

初心者向けのエンジニアリング ソリューションの多くは、ディメンションを統一するために単純な総当たりランダム前処理を習慣的に使用していますが、これがモデルの「目に見えないパフォーマンスの上限」の根本であることがよくあります。 「イメージをはめ込む」だけではなく、「何が込められているのか」にも注力しなければなりません。



![図 6-1: 画像前処理におけるトリミングとセマンティック調整](../../images/part3/图6_1_图片预处理中裁剪与语义对齐问题.png)
*図 6-1: 画像の前処理におけるトリミングとセマンティック アライメント*

* **悪いケース (左の画像 - 単純なトリミングのコスト)**:
    従来の `RandomCrop` または `CenterCrop` は構成を意識しません。縦方向の構図でポートレート写真を処理する場合、中央のトリミングにより重要な部分 (頭など) が簡単に切り取られ、胴体だけが残ります。この時点で、テキスト ラベルがまだ「笑顔の人」である場合、モデルは間違ったマッピング (胴体の特徴を「笑顔の人」と誤認) を確立することを強制され、トレーニングされたモデルが重度の幻覚を引き起こす原因になります。

* **良いケース (右の画像 - セマンティックな完全性)**:
    「画像とテキストの一貫性」を追求した高品質なデータエンジニアリング。
    1. **スマート サイズ変更**: 完全な視覚的主題を維持するには、`Resize with Padding` (アスペクト比、黒/白のパディングを維持) を優先します。これにより無効なピクセルが発生しますが、セマンティックな完全性が保証されます。
    2. **アスペクト比バケット**: SDXL と Midjourney で一般的に使用される高度な技術。同様のアスペクト比を持つ画像をトレーニング用に同じバッチにグループ化し、パディングの無駄を減らしながらトリミングを回避します。
    3. **要約**: 以下の第 7 章で詳しく説明されているように、VLM を使用して高密度の説明を生成すると、テキストが画面上の詳細 (標識テキスト、背景オブジェクトなど) に正確に対応し、データ トレーニングの価値が最大化されます。

#### 6.2.3 GPU アクセラレーションによるデコードと変換 (NVIDIA DALI)

ディープ ラーニング モデルのトレーニング フェーズでは、ほとんどの研究者や開発者がモデル アーキテクチャの設計、ハイパーパラメータの調整、損失関数の改善 (モデルの精度に直接影響するモジュール) に注目しますが、データ ロード (DataLoader) の基礎を見落としがちです。実際には、これがトレーニング効率を制限する「目に見えないパフォーマンスキラー」となることが多く、ハイエンド GPU コンピューティングの完全な利用を妨げ、深刻なハードウェアの無駄を引き起こすことさえあります。

この問題点を理解するには、まず深層学習トレーニング フローの完全なロジックを明確にする必要があります。モデル トレーニングのコア コンピューティングは、GPU の大規模な並列コンピューティング機能に依存しています。 GPU は大規模なテンソル演算を効率的に処理し、バックプロパゲーションとパラメーターの更新を完了できます。ただし、データが GPU に入力される前に、一連の前処理操作を通過する必要があります。その中で最も基本的で時間のかかる操作は、画像のデコードとサイズ変更です。従来の PyTorch トレーニング フローでは、これらの重要な前処理操作はすべて CPU 上で実行され、「CPU 前処理のボトルネック」と「GPU 計算の冗長性」の間に矛盾が生じます。

具体的には、従来の PyTorch Dataset ワークフローは次のとおりです。まず、CPU を介してディスクから画像ファイル (主に JPEG) を読み取り、次に CPU が JPEG デコードを完了します。このプロセスには、圧縮画像バイナリ データに対するハフマン デコード、逆離散コサイン変換 (IDCT) およびその他の複雑な計算が必要であり、一般的な CPU 負荷の高いタスクです。デコード後、CPU はサイズ変更、正規化、色空間変換などの前処理を実行し、最終的に処理された画像テンソルをモデルのトレーニングのために GPU にコピーします。

さらに重要なのは、CPU アーキテクチャがシリアル計算とロジック制御により適していることです。並列計算能力は GPU よりもはるかに劣ります。それでも、デコードやサイズ変更などの画像前処理操作は本質的に高度に並列化可能であり、マルチスレッドまたはマルチコア並列処理によって効率を向上させることができます。しかし、従来の PyTorch データセットは、CPU 並列処理を向上させる DataLoader の num_workers を使用したとしても、CPU 自体のコンピューティングの上限を突破することはほとんどできません。特に、トレーニング データセットが大規模 (数百万の画像) で、単一画像の解像度が高い (1080p+) 場合、CPU の前処理速度が GPU トレーニング速度に大幅に遅れ、GPU が頻繁に「データ待ち」でアイドル状態になり、GPU 使用率が大幅に低下し、最終的に全体のトレーニング効率が低下します。これが、データの読み込みが「無視されたパフォーマンスキラー」と呼ばれる理由です。

この中心的な問題点に対処するために、NVIDIA は、ディープ ラーニング トレーニング用に最適化された GPU アクセラレーションのデータ前処理ライブラリである DALI (Data Loading Library) を導入しました。その中心的な目標は、画像のデコードやサイズ変更などの CPU を集中的に使用する操作を GPU に移行して並列実行し、データ読み込みパフォーマンスのボトルネックを解消し、GPU を最大限に活用できるようにすることです。



![図 6-2: DALI を使用した場合と使用しない場合のデータのデコードと変換](../../images/part3/图6_2_使用DALI与不使用DALI下数据解码与变换的区别.png)
*図 6-2: DALI を使用した場合と使用しない場合のデータのデコードと変換*

**コードのチュートリアル: 高性能 DALI パイプライン**

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
    # Using index_paths is necessary, otherwise initialization requires traversing entire tar, extremely slow [5]
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
        read_ahead=True          # Enable read-ahead
    )

    # Step 2: GPU Decoding (core acceleration point)
    # device="mixed" means input in Host memory, output in Device memory
    # output_type=types.RGB handles color space conversion automatically
    images = fn.decoders.image(
        jpegs,
        device="mixed",
        output_type=types.RGB,
        # Error handling for corrupted images
        # In production, never let a single bad image crash training
    )

    # Step 3: GPU transformation pipeline
    # resize: maintain aspect ratio scaling
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

# Benchmark test
# On A100, this pipeline typically achieves 3000-5000 FPS, 5-10x CPU Loader
```

### 6.3 マルチモーダルな洗浄パイプライン

大量のデータには大量のノイズが伴います。 LAION-5B の生データでは、真に高品質のサンプルは 10% 未満である可能性があります。データの多様性の損失を最小限に抑えながらデータ密度を向上させるには、多段階のクリーニング ファネルを確立する必要があります。いわゆる「データ クリーニング」は本質的に **データ ダイエット** であり、モデルへのフィードの量を減らし、より良くすることです。

#### 6.3.1 アーキテクチャ設計: レイ データ分散クリーニング

クリーニングフェーズにスパークではなくレイを選択する理由は何ですか?なぜなら、クリーニングにはもはや単純な ETL ではなく、大量の **深層学習推論 (モデル推論)** が含まれているからです。 Spark の MapReduce パラダイムと比較して、Ray はより柔軟な Actor メカニズムを提供し、GPU モデル (CLIP、Safety Checker など) を常駐させ、小さなバッチごとに複数 GB のモデルをリロードするという膨大なオーバーヘッドを回避できます。

Ray Data は、CPU 集中型 (解凍、ハッシュ、正規表現) タスクと GPU 集中型 (CLIP Embedding推論) タスクの両方を伴うこの混合ワークロードに適しています。以下は、一般的な 3 段階のパイプライン設計です。
* **ステージ 1 (CPU)**: 高速フィルタリング。解像度が不十分 (<256px)、テキストが短すぎる、英語以外 (英語モデルのみをトレーニングしている場合)、または異常なアスペクト比を持つサンプルを直接削除します。
* **ステージ 2 (GPU)**: 詳細な特徴抽出。 CLIP モデルを使用してエンベディングを生成し、エンベディングに基づいて画像とテキストの類似性と美的スコアを計算します。
* **ステージ 3 (CPU/混合)**: ロジックの評価と重複排除。安全性 (NSFW)、美的スコア、画像とテキストの関連性に基づいて最終しきい値カットオフを適用し、セマンティックな重複排除を実行します。



**データ フロー図**

![図 6-3: Ray Data 分散型洗浄データの流れ](../../images/part3/图6_3_Ray_Data分布式清洗数据流向图.png)
*図 6-3: Ray データ分散クリーニング データ フロー*

#### 6.3.2 コアアルゴリズムの実装

クリーニングは単なる削除ではなく、データ値の量子化でもあります。画像とそれに対応するテキストの「ゴールド コンテンツ」を測定するには、多次元のメトリクスが必要です。

1. **美的スコア**
    * **原則**: データセットは請求書、スクリーンショット、不鮮明な監視映像でいっぱいです。これらは美しい画像を生成するのには役に立ちません。通常は LAION-Aesthetics Predictor を使用します。
    * **技術詳細**: CLIP 画像Embeddingを入力として使用し、1 ～ 10 のスコアを出力する単純な MLP (多層パーセプトロン)。 AVA データセットからのトレーニング データ (プロの写真家の人間による評価)。
    * **推奨しきい値**: 基本的な事前トレーニングでは、スコア > 4_5 を維持します。高品質の生成モデル (SFT ステージ) を微調整するには、スコア > 6_0、または 6_5 を推奨します。

2. **画像とテキストの配置フィルタリング**
    * **原則**: 代替テキストの多くは、画像コンテンツとは関係のない、SEO のゴミ単語の積み重ねやファイル名 (「DSC_001.jpg」) です。
    * **技術詳細**: CLIP 画像EmbeddingとテキストEmbeddingのコサイン類似度 (ドット積) を計算します。
    * **落とし穴**: CLIP のバージョンが異なると (例: OpenAI ViT-L/14 と OpenCLIP ViT-G/14)、エンベディング スペースの分布が異なります。スコアは直接比較できません。特定のモデルのしきい値を再調整する必要があります。一般的なアプローチ: データセット全体の類似度分布を計算し、上位 50% または上位 70% を維持します。

3. **安全性の検出 (安全性とウォーターマーク)**
    * **原則**: ポルノ、暴力、目立つ透かし入りの画像は削除する必要があります。
    * **戦略**: NSFW とウォーターマークの検出には、特別にトレーニングされた分類子ヘッド (CLIP Embeddingにも基づく) を使用します。ウォーターマーク検出の場合: ターゲットが生成モデル (SDXL など) をトレーニングしている場合、生成モデルはウォーターマーク特徴に簡単にオーバーフィットするため、非常に厳密 (リコール優先) である必要があります。ターゲットが理解モデル (GPT-4V など) をトレーニングしている場合、理解モデルは「画像に透かしがある」ことを認識する必要があるため、多少緩和できます。

**コード実装: Ray Data Cleaning Operator**

```python
import ray
import torch
import open_clip
import numpy as np
from PIL import Image
import io

# Define Ray Actor class to ensure model loaded only once
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
        Process a batch of data. Ray will automatically partition and transfer data to Actor.
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
            
        # Return results (must align with original batch indices)
        return {"aesthetic_score": aesthetic_scores}

# Orchestrate Ray pipeline
ray.init()
ds = ray.data.read_webdataset("s3://raw-bucket/{00000..00099}.tar")

# map_batches will automatically schedule GPU resources
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

数十億規模のマルチモーダル データセットを構築する際、エンジニアリング チームは詳細でつまずくことがよくあります。ここでは、いくつかの教訓を学びました。

* **寄木細工のメタデータの爆発的増加**:
    * **エラー**: パンダで 20 億行の Parquet ファイルを直接読み取る習慣があります。
    * **結果**: メモリ オーバーフロー (OOM)。パンダは 1 列だけを読み取る場合でもインデックス全体をメモリに読み込もうとするためです。
    * **修正**: Polars または PySpark 遅延評価モードを使用します。または、単一の巨大なメタデータ ファイルの処理を避けるために、Parquet ファイルを行数 (例: 100 万行) ごとに小さなファイルに厳密に分割します。

* **WebDataset のシャッフルが不十分です**:
    * **エラー**: データはダウンロード中にドメイン順序で書き込まれます。トレーニングは DataLoader バッファー シャッフルのみに依存します (通常、バッファーは 10k のみ)。
    * **結果**: モデルは 100,000 個の e-コマース画像を連続して表示し、次に 100,000 個の風景画像を連続して表示する可能性があります。バッファーが小さいと、この「時間的相関」を打ち破ることができず、トレーニング カーブの激しい振動や発散さえも引き起こします。
    * **修正**: WebDataset を書き込む前に、URL リストに対して **グローバル シャッフル** を実行する必要があります。 Spark の `orderBy(rand())` を使用できます。

* **ロングテール データを誤って削除してしまう**:
    * **エラー**: 究極の美的スコアを追求するため、スコア < 4_5 の画像をすべて削除します。
    * **結果**: モデルは「特殊化」され、アート写真と壁紙のみを認識し、医療画像、ストリート ビュー、手書きのメモなどの現実世界の (おそらく醜い) 写真は認識しません。モデルの一般化を大幅に軽減します。
    * **修正**: 層別サンプリング戦略を使用します。 5% ～ 10% の低スコア データを「正規化」として保持するか、美的フィルターをバイパスする特定のドメイン (OCR、グラフなど) に特別なホワイトリストを設定します。

* **重複データの危険性 (重複排除)**:
    * **エラー**: インターネット上の大量の重複画像 (ミーム、バイラル ニュース画像など) を無視します。
    * **結果**: モデルは特定のサンプルをオーバーフィットし、生成中にトレーニング セットの画像を「記憶」することもあり、重大な著作権リスクを引き起こします。
    * **修正**: クリーニング パイプラインに **セマンティック重複排除** を追加する必要があります。すべての画像のEmbeddingを計算し、クラスタリングに Faiss または MinHashLSH を使用し、類似性の高い画像グループごとに 1 つだけ保持します。
