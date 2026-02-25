## 第 8 章: ビデオおよびオーディオ データの処理

### 章の概要

ビデオ データは、マルチモーダル エンジニアリングの「深層水」と呼ばれる、マルチモーダル ラージ モデル (LMM) トレーニングにおいて、最大の量、最も処理の難易度が高く、最も複雑な情報密度を持つモダリティです。静止画像とは異なり、ビデオでは **時間次元** が導入されます。これは、データが単なるピクセルの積み重ねではなく、因果関係の論理、物理法則、および運動パターンを運ぶことを意味します。

この章では、連続した非構造化ビデオ ストリームをモデルが理解できる離散トークンに変換する方法を系統的に説明します。基礎となる **ショット境界検出** から始めて、コンテンツベースのセグメンテーション アルゴリズムを詳しく分析します。次に、ビデオ生成の「核心」である **Video Tokenizer** を分析し、VQ-VAE と Google DeepMind の最新の MagViT-v2 の基礎となる原理を比較します。最後に、**WhisperX** を使用して、音声とビデオの単語レベル、さらには音素レベルの正確な位置合わせを実現し、モデルの時空間的に同期された監視信号を構築する方法を示します。

**学習目標**:
* **エンジニアリング機能**: PySceneDetect と ffmpeg キーフレーム メタデータを組み合わせたマスターを使用して、効率的な 2 段階のシーン セグメンテーション (粗いから細かい) 戦略を実現します。
* **理論的な深さ**: ビデオ トークン化における「コードブックの崩壊」問題と、MagViT-v2 がルックアップフリー量子化 (LFQ) を通じてこのボトルネックをどのように完全に解決するかを深く理解します。
* **データ パイプライン**: WhisperX ベースの強制位置合わせフローを実装して、マルチスピーカーやバックグラウンド ノイズの音響環境での正確な字幕位置合わせを解決します。
* **ストレージの最適化**: ストレージのシャーディングと大量のビデオ データの効率的な読み込みについて理解します。

**シナリオの紹介**:
> 「ソラのような世界モデルをトレーニングしていると想像してください。トレーニング データとして 2 時間の映画「タイタニック」をダウンロードしました。
>
> 単純に 10 秒ごとに分割すると、重大な「意味的不連続性」が発生します。セグメントの最初の 5 秒はデッキ上の穏やかな海風かもしれませんが、次の 5 秒は突然騒々しいレストランにジャンプします。このクロスシーンの「ハードカット」はモデルを混乱させます。「人はどのようにして屋外から屋内に 0.1 秒でテレポートしたのでしょうか?」これは計算を無駄にするだけでなく、モデルに間違った物理学を教えます。
>
> さらに、オーディオの時間精度が命です。ローズの口が画面上で動いているとき、字幕が画像より 2 秒遅れている場合、対応するトークンはジャックのセリフです。モデルは、「ジャックの声の特徴」を「ローズの顔の特徴」と誤って関連付けます。兆トークンのトレーニングでは、このような微妙なずれが増幅して重度の幻覚を引き起こす可能性があります。」

---

### 8.1 ビデオ処理パイプライン: シーン検出

ビデオは基本的に連続したストリームではなく、一連の独立した「ショット」が連結されたものです。各ショットは、1 つのカメラのオンとオフ (または連続的なカメラの動き) を表します。ビデオ生成モデル (ビデオ生成モデル) をトレーニングするには、**時空間連続性** を確保するために、各トレーニング サンプル (トレーニング クリップ) が同じショット内にある必要があります。

#### 8.1.1 ビデオ構造のミクロビュー: GOP と I フレーム

セグメンテーション アルゴリズムに入る前に、ビデオ エンコーディングの基本を理解する必要があります。


* **I フレーム (イントラコーディングされたピクチャ)**: キーフレーム。他のフレームに依存せずにデコードできる完全な画像です。通常、シーン遷移の開始点でもあります。
* **P フレーム (予測ピクチャ)**: 前方予測フレーム。前のフレームとの差分のみを保存します。
* **B フレーム (双方向予測ピクチャ)**: 双方向予測フレーム。過去と将来のフレームの両方を参照して圧縮し、最も高い圧縮率を実現します。

**GOP (Group of Pictures)**: 2 つの I フレーム間のシーケンス。ビデオ プレーヤーがシークすると、デコードはそこから開始する必要があるため、通常、最も近い I フレームに「スナップ」します。私たちのセグメンテーション戦略はこれを活用して加速する必要があります。

#### 8.1.2 アルゴリズムの選択と戦略



![図 8-1: 2 つのビデオ シーン セグメンテーション戦略と HSV ヒストグラムの違い](../../images/part3/图8_1_视频场景切分的两种策略与HSV直方图差异.png)
*図 8-1: 2 つのビデオ シーン セグメンテーション戦略と HSV ヒストグラムの違い*

**PySceneDetect** は、業界標準のオープンソース ツールです。フレーム間差分分析に基づいたコア ロジックを備えた複数の検出器を提供します。

* **戦略 1: 閾値検出器 (ハードカット)**
    * **原理**: HSV 色空間または RGB 輝度で隣接するフレーム間の平均差 (デルタ) を計算します。デルタ > `threshold` (例: 30_0) の場合、カットポイントとしてマークされます。
    * **該当**: ほとんどの映画とユーザー生成コンテンツ (UGC)。
    * **制限事項**: 段階的な遷移を検出できません。

* **戦略 2: 適応検出器 (段階的な遷移 / 高速カット)**
    * **原則**: 固定しきい値は使用しなくなりました。スライディングウィンドウを維持します。 「現在のフレーム」と「ウィンドウ内の平均フレーム差」の比率を比較します。
    * **適用可能**: フェードイン/アウト、ディゾルブ、またはカメラの動きが激しいアクション シーン。

**高度な戦略: 2 段階のカスケード分割**
フルデコードされた TB スケールのビデオで PySceneDetect を実行すると、非常に時間がかかります。工業用の「粗くしてから細かい」アプローチをお勧めします。

1. **レベル 1 (メタデータ スキャン)**: `ffprobe` を使用して、ビデオ ストリームのメタデータを迅速にスキャンし、すべての **I フレーム** タイムスタンプを抽出します。 I フレームは多くの場合、シーンの遷移時に表示されます (エンコーダーは、突然の変化時に I フレームを挿入する傾向があります)。このステップではフレームのデコードは必要ありません。再生速度は 100 倍以上です。
2. **レベル 2 (コンテンツ分析)**: レベル 1 で識別された潜在的なカット ポイントから ±2 秒以内の正確なフレームレベルの位置特定には、PySceneDetect の `ContentDetector` のみを実行します。

#### 8.1.3 コア コード: シーン検出とロスレス セグメンテーション

以下のコードは、運用環境における標準的なセグメンテーション フローを示しています。 「ストリーム コピー」テクニックに注目してください。これは、大規模なビデオを処理する際のストレージの爆発を回避するための鍵です。

```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_video_scenes(video_path, output_dir, threshold=27_0):
    """
    Detect scenes and cut video with ffmpeg losslessly
    Args:
        video_path: Input video path
        output_dir: Output directory
        threshold: Segmentation threshold (empirical: 27_0 works for most 1080p video)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info(f"Starting scene detection for: {video_path}")

    # 1. Scene detection
    # threshold=27_0: HSV space histogram difference threshold
    # min_scene_len=15: Ignore segments shorter than 0.5s (30fps).
    # Very short segments are usually flash, glitch, or segmentation noise—unsuitable for training data.
    scene_list = detect(
        video_path, 
        ContentDetector(threshold=threshold, min_scene_len=15)
    )
    
    # 2. Statistics and filtering
    # Add logic here: e.g., merge overly short adjacent scenes, or discard scenes under 3 seconds
    valid_scenes = []
    for scene in scene_list:
        start, end = scene
        duration = (end.get_frames() - start.get_frames()) / start.get_framerate()
        if duration >= 3_0: # Keep only segments >3s for training
            valid_scenes.append(scene)

    logging.info(f"Detected {len(scene_list)} scenes, kept {len(valid_scenes)} valid scenes.")
    
    # 3. Split video (Stream Copy)
    # Key: arg_override='-c:v copy -c:a copy'
    # This instructs ffmpeg to directly copy the binary stream without [decode -> pixels -> encode].
    # Benefit 1: Extremely fast (limited by disk I/O, not CPU).
    # Benefit 2: 100% lossless quality, no re-encoding artifacts.
    split_video_ffmpeg(
        video_path, 
        valid_scenes, 
        output_dir=output_dir, 
        show_progress=True,
        arg_override='-c:v copy -c:a copy' 
    )

# Pitfall: Data storage explosion disaster
# NEVER decode segmented video into image sequences (png/jpg) or numpy arrays for long-term storage!
# Do the math:
# 1 hour 1080p H.264 video ≈ 2GB
# Decoded: 3600s * 30fps * 1920 * 1080 * 3 bytes ≈ 670 GB
# Expansion factor > 300x.
# Always store in compressed format (mp4/mkv), only use GPU (NVDEC) for real-time decoding in training DataLoader __getitem__.
```

---

### 8.2 ビデオのトークン化: ピクセル オーシャンから離散島まで

Sora、Gen-2、およびその他の Transformer ベースの拡散モデル (DiT) の場合、ピクセル空間で直接モデリングすることは不可能です。 4 秒の 1080p ビデオには、約 $3 \times 10^8$ ピクセルが含まれます。注意行列を計算すると、即時 OOM が発生します。

したがって、ビデオはまず潜在空間内の個別のトークンに「圧縮」される必要があります。このプロセスは **Video Tokenizer** によって実行されます。

#### 8.2.1 従来のアプローチの問題: VQ-VAE と「デッドコード」

**VQ-VAE (ベクトル量子化変分オートエンコーダー)** は、初期のビデオ生成モデル (VideoGPT など) の基礎です。

* **フロー**:
    1. **エンコーダー**: ビデオを 3D パッチ (例: $16 \times 16 \times 16$ の時空間ブロック) に分割し、低次元ベクトル $z_e(x)$ に圧縮します。
    2. **量子化**: $K$ プロトタイプ ベクトル (Embedding) を使用してコードブックを維持します。各 $z_e(x)$ について、ユークリッド距離でコードブック内の最も近いベクトル $e_k$ を見つけて置き換えます。
    3. **デコーダー**: $e_k$ を使用してビデオを再構築します。

* **致命的な欠陥: コードブックの崩壊**
    トレーニングの初期段階では、少数のコード (コード #5 や #100 など) だけが誤って選択されます。選択されたコードのみが勾配更新を受け取るため、コードは「より良く」なり、再度選択されやすくなります。これは「金持ちはさらに金持ちになる」マシュー効果を形成します。
    * **結果**: コードブック ベクトルの 90% は、決して使用されず「デッド コード」になります。これにより、有効な語彙が非常に少なくなり、不鮮明で詳細が欠如したビデオが生成されます。
    * **修復**: 従来の方法では複雑なリセット戦略 (K 平均法リセットなど) が必要であり、トレーニングは非常に不安定です。

#### 8.2.2 SOTA アプローチ: MagViT-v2 および LFQ

Google DeepMind は、MagViT-v2 に **LFQ (ルックアップフリー量子化)** を導入し、ゲームを根本的に変えました。



* **中心的なアイデア: ルックアップなし、直接計算。**
    LFQ は「最近傍検索」アプローチを放棄し、潜在変数の **符号**からトークンを直接生成します。

* **数学的原理**:
    エンコーダーの出力潜在ベクトル $z \in \mathbb{R}^D$ (例: $D=18$) を仮定します。
    LFQ は各次元を 2 値化します。
    $$q_i = \begin{cases} 1 & \text{if } z_i > 0 \\ 0 & \text{if } z_i \le 0 \end{cases}$$
    
次に、$D$ バイナリ ビットを整数インデックスに結合します。
    $$\text{トークン ID} = \sum_{i=0}^{D-1} q_i \cdot 2^i$$

* **LFQ はなぜ革新的ですか?**
    1. **無限の有効コードブック**: $D=18$ の場合、自然なコードブックのサイズは $2^{18} = 262,144$ です。すべてのコードは $D$ の独立した次元の組み合わせです。各次元は常にグラデーションの更新に参加します。 **コードブックの使用率は 100% で一定です。**
    2. **計算コストゼロ**: 高価な「完全なコードブック距離計算」は不要で、単純なビット演算のみです。
    3. **時空間圧縮**: MagViT-v2 は **3D 因果関係 CNN** を組み合わせ、空間を圧縮しながら時間的因果関係を維持します (現在のトークンが将来の情報を漏らすことはありません)。これは生成モデルにとって重要です。

#### 8.2.3 アーキテクチャ比較表

|特集 | VQ-VAE (TATS/VideoGPT) | MagViT-v2 (LFQ) |
| :--- | :--- | :--- |
| **量子化メカニズム** |最近傍検索 (ルックアップ) |符号関数（符号投影） |
| **語彙サイズ (語彙)** |通常は 1024 ～ 8192 (VRAM と折りたたみによって制限されます) | $2^{18}$ (262k) 以上、簡単に拡張可能 |
| **コードブックの使用率** |低い（崩壊しやすい、EMAが必要など） | **100% (デザインの崩壊を回避)** |
| **グラデーション バックプロップ** | Straight-Through Estimator (STE) が必要 |改善されたエントロピー ペナルティ + STE |
| **世代の品質** |ぼやけやすく、ディテールの質感が失われます |非常にクリアで、オリジナルよりさらに優れています（ノイズ除去効果） |
| **推論速度** |遅い (特にコードブックが大きい場合) |非常に速い |
---

VQ-VAE から MagViT-v2 への進化は、単純なパラメーターの最適化ではなく、「検索ベースの近似」から「計算ベースの構築」へのビデオ離散化テクノロジーのパラダイム シフトです。

まず、計算の複雑さとスケーラビリティの点で、従来の VQ-VAE には根本的なボトルネックがあります。その量子化は最近傍検索に依存しており、特徴ベクトルとコードブック内のすべての $K$ プロトタイプ間のユークリッド距離 (時間計算量 $O(K)$) を計算する必要があります。これは、表現を改善するために語彙を拡張すると、推論レイテンシが直線的に増加することを意味します。対照的に、MagViT-v2 の LFQ (ルックアップフリー量子化) はルックアップを放棄し、符号関数を使用して潜在変数をバイナリ文字列に射影します。このプロセスにより、計算の複雑さが定数 $O(1)$ に軽減され、推論速度を犠牲にすることなくモデルが $2^{18}$ 以上の語彙をサポートできるようになり、大きな語彙と低遅延の間の矛盾が解決されます。

第二に、コードブックの利用とトレーニングの安定性において、この 2 つは著しく異なります。 VQ-VAE は長い間、「コードブック崩壊」に悩まされてきました。初期化や不均一な勾配割り当てにより一部のエンコード ベクトルがアクティブにならず、有効なボキャブラリが設計値を大幅に下回ります (多くの場合、わずか 1024 ～ 8192)。このため、研究者は EMA (指数移動平均) や K-means リセット、その他の複雑なエンジニアリング手法を導入する必要があります。 MagViT-v2 の LFQ は、独立した次元の 2 値化の組み合わせに基づいており、コードブック空間が「個別に検索される」のではなく「組み合わせ的に生成される」ことを数学的に保証します。潜在空間次元がアクティブである限り、結合されたコードは自然にコードブック空間全体をカバーし、理論上は 100% の使用率が達成されます。

要約すると、MagViT-v2 の LFQ は、高圧縮、高忠実度、低計算コストの統合を実現し、細部のテクスチャ損失と時空間一貫性の低下という従来の VQ-VAE の欠点を完全に解決します。 Sora スケールの大規模ビデオ生成モデルを構築するには、MagViT-v2 と派生 Tokenizer アーキテクチャが業界で好まれる選択肢となっています。

### 8.3 オーディオ調整: WhisperX と強制調整

ビデオは単なる視覚データではありません。音声 (オーディオ) は、自然で時間的に密度の高いテキストの説明を提供します。音声を使用すると、「爆発音は爆発光に対応する」「泣き声は涙に対応する」など、複数のモーダルな関連付けをモデルに学習させることができます。

ただし、通常の ASR (生の Whisper など) は「文レベル」のタイムスタンプのみを提供し、通常は 1 ～ 2 秒のエラーが発生します。これでは、細かいビデオ トレーニング (リップシンクなど) にはまったく不十分です。 **WhisperX** が必要です。


![図 8-2: 通常の ASR (セグメント レベル) と WhisperX (単語/音素レベル) の精度の比較](../../images/part3/图8_2_ASR与WhisperX的精度对比.png)
*図 8-2: 通常の ASR (セグメント レベル) と WhisperX (単語/音素レベル) の精度の比較*

#### 8.3.1 なぜ強制的に位置合わせを行うのか?
* **ASR (OpenAI Whisper)**:
    * 出力: `"Hello world"` -> `Timestamp: [0_0s -> 2_0s]`
    * 問題: モデルは、文がこの 2 秒以内に含まれることのみを認識しており、「world」がいつ始まるかは正確には認識していません。
* **強制アライメント (WhisperX)**:
    * 原則: まずテキストに書き起こしてから、事前にトレーニングされた音響モデル (Wav2Vec2 など) を使用して、テキスト内の **音素** を音声波形と強制的に一致させます。
    * 出力:
        * `"Hello"`: `[0_12s -> 0_58s]`
        * `"world"`: `[0_85s -> 1_45s]`
    * **値**: 次のようなトレーニング ペアを構築できます: ビデオ フレームが 0_85 秒の場合、モデルに「世界」のテキストEmbeddingに焦点を当てるように強制します。これは、優れたマルチモーダル アライメントの基礎です。

#### 8.3.2 エンジニアリング実装: WhisperX フル パイプライン
WhisperX は、VAD (音声アクティビティ検出)、Whisper (文字起こし)、Wav2Vec2 (位置合わせ)、および Pyannote (話者ダイアリゼーション) を組み合わせた複雑なパイプラインです。

```python
import whisperx
import gc
import torch

def align_audio_transcript(audio_file, device="cuda", batch_size=16):
    """
    Use WhisperX for transcription and word-level forced alignment
    """
    # Step 1: Transcription
    # Use Large-v2 model for transcript accuracy
    # compute_type="float16" significantly speeds up, but requires Ampere+ GPU (A100/A10/3090/4090)
    print("1. Loading Whisper model...")
    model = whisperx.load_model(
        "large-v2", 
        device, 
        compute_type="float16" 
    )
    
    print("2. Transcribing...")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    
    # Critical: VRAM management
    # Whisper model is huge, and the next Alignment model is also VRAM-heavy.
    # Must explicitly delete model and trigger garbage collection, otherwise easily OOM (Out of Memory).
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Forced Alignment
    # Auto-loads corresponding language Wav2Vec2 model (e.g., wav2vec2-large-960h for English)
    print("3. Aligning...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], 
        device=device
    )
    
    # align() executes a Dynamic Programming-like algorithm
    # finding best matching path between text phoneme sequence and audio waveform features
    aligned_result = whisperx.align(
        result["segments"], 
        model_a, 
        metadata, 
        audio, 
        device, 
        return_char_alignments=False # Set True for character-level alignment (e.g., for karaoke subtitles)
    )

    # Result contains word_segments with precise start/end for each word
    # e.g.: [{'word': 'Hello', 'start': 0_1, 'end': 0_5, 'score': 0_98}, ...]
    return aligned_result

# Advanced tip:
# For speaker diarization (who said what), further call:
# diarize_model = whisperx.DiarizationPipeline(use_auth_token="YOUR_HF_TOKEN", device=device)
# diarize_segments = diarize_model(audio)
# whisperx.assign_word_speakers(diarize_segments, aligned_result)
```

#### 8.3.3 本番環境の落とし穴

1. **VAD の誤った判断と BGM の妨害**:
    * **問題**: WhisperX は、サイレント セグメントのセグメント化に VAD に大きく依存しています。ビデオの BGM が大きい場合、VAD はセグメント全体を音声として処理するか、その逆の場合があり、音声がかき消されます。
    * **解決策**: ソース分離のために **Demucs** または **Spleeter** を導入します。
    * **フロー**: `Raw Audio` -> `Demucs (Extract Vocal Track)` -> `WhisperX`。抽出された純粋なボーカル トラックのみを認識にフィードして、精度を大幅に高めます。

2. **マルチスピーカーオーバーラップ (オーバーラップスピーチ)**:
    * **問題**: Whisper は複数の人が同時に話している場合 (カクテル パーティーの問題) に弱く、通常は最も声の大きい人だけを文字に起こすか、混乱したテキストを生成します。
    * **解決策**: `diarization=True` を有効にします。これにより推論時間が 30% ～ 50% 増加しますが、テレビ ドラマやインタビュー クラスのビデオ データの場合、これが「誰が何を言ったか」を区別する唯一の方法であり、キャラクターのアイデンティティに関するモデルの混乱を避けることができます。

3. **幻覚タイムスタンプ**:
    * **問題**: ささやき声は、長い沈黙または純粋な音楽セグメント中に「幻覚」を引き起こす可能性があり、間違ったタイムスタンプで前の歌詞を繰り返します。
    * **チェック**: 後処理で、`word['score']` (信頼性) をチェックします。連続した単語の文字列の信頼度が 0_4 未満の場合は、そのセグメントのアラインメントを破棄することをお勧めします。

---
