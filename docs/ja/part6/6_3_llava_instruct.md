# プロジェクト 3: LLaVA マルチモーダル命令データセットの構築

> **範囲**: マルチモーダル LLM (LMM) 開発、データ エンジニアリング、ビジュアル命令チューニング (ビジュアル命令チューニング)

#### 1. プロジェクトの背景 (プロジェクトの概要)

- **タスク定義:**
  LLaVA や Qwen-VL などのマルチモーダル モデルをトレーニングするために、単一画像 QA (Visual QA)、オブジェクト位置特定 (Grounding)、および複数画像コンテキスト推論 (Interleaved Image-Text) をサポートする高品質の視覚的SFTデータセットを構築します。

- **入力と出力:**
  - **入力:** 
    - RAW画像ライブラリ(`.jpg` / `.png`)
    - 構造化されたアノテーション データ (例: Bbox 座標を含む COCO 形式 `instances.json`)
  - **出力:** 
    - LLaVA トレーニング標準に準拠した JSON ファイル (`image`、`conversations` フィールドあり)。
    - 座標の正規化と形式の調整によるデータのグラウンディング。

- **課題分析:**
  1. **座標位置合わせ (座標位置合わせ):** 生の検出データの座標は通常、ピクセル絶対値 (x、y、w、h) ですが、LLaVA では、`[ymin, xmin, ymax, xmax]` の順序で `[0-1000]` 範囲に正規化する必要があります。一度間違えると、モデルは深刻な「幻覚」に見舞われます。
  2. **複数画像ロジックの構築:** 従来の画像キャプション データは 1 つの画像と 1 つのテキストです。 「複数の画像がインターリーブされた」ダイアログを構築するには、モデルに画像間の関係を理解させるための合理的な比較プロンプトを構築する必要があります。

#### 2. アーキテクチャ設計 (アーキテクチャ設計)

- **データ パイプライン図:**
![図 3: LLaVA マルチモーダル命令データセットの構築](../../images/part6/图3_构建LLaVA多模态指令集数据流水线图.png)



- **テクノロジースタック:**
  - **OpenAI 互換 API (SiliconFlow/Qwen):** 高品質の画像テキスト説明と複数画像比較ロジックを生成します。対話構築に LLM 推論を活用します。
  - **Python & OpenCV:** コアグルー言語。 OpenCV は、座標正規化と「ドローボックス検証」視覚化のために画像寸法 (H、W) を読み取るために不可欠です。
  - **JSON:** LLaVA 標準データ交換形式。

#### 3. 段階的な実装

##### フェーズ 1: マルチ画像インターリーブ データの生成

2 つの画像を「比較」するようにモデルに教えるには、API を使用して複数の画像を動的に入力し、比較をリクエストします。

**キー ロジック:** VLM API を使用してマルチイメージ入力プロンプトを構築します。

```python
# From interleaved.py
def generate_comparison(img1_path, img2_path):
    # Construct Prompt: Require multi-image comparison
    prompt = "Here are two images. Please briefly compare them..."
    
    # Build multi-image payload
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"...{img1_path}..."}}, # Image 1
                {"type": "image_url", "image_url": {"url": f"...{img2_path}..."}}  # Image 2
            ]
        }
    ]
    # ... send request and parse result ...
```

##### フェーズ 2: コア処理 - 境界ボックスの位置合わせ

これがプロジェクトの中核となる計算です。 COCO は `[x_topleft, y_topleft, width, height]` を使用しますが、LLaVA は 0 ～ 1000 の整数に正規化された値を持つ `[ymin, xmin, ymax, xmax]` を必要とします。

**キー機能:** 座標正規化変換

```python
# From alignment.py
def convert_bbox(bbox, width, height):
    # COCO raw input: x, y, w, h
    x, y, w, h = bbox
    
    # Convert to LLaVA format: [ymin, xmin, ymax, xmax] normalized to 0-1000
    # Must use max/min for clipping to prevent float error overflow
    xmin = int((x / width) * 1000)
    ymin = int((y / height) * 1000)
    xmax = int((x + w) / width * 1000)
    ymax = int((y + h) / height * 1000)
    
    return [
        max(0, min(1000, ymin)),
        max(0, min(1000, xmin)),
        max(0, min(1000, ymax)),
        max(0, min(1000, xmax))
    ]
```

##### フェーズ 3: フォーマットと検証

データ生成を直接トレーニングに使用してはなりません。 **視覚化リバース検証**に合格する必要があります。描画したボックスが間違っていると、トレーニングされたモデルは役に立たなくなります。

**検証ロジック:** 生成された JSON を解析し、`[0-1000]` 座標をピクセル座標に復元して描画します。

```python
# From visualize_bbox.py
def draw_bbox(image, bbox, label, color):
    h, w, _ = image.shape
    ymin, xmin, ymax, xmax = bbox # Read LLaVA format
    
    # Restore to pixel coordinates for drawing
    x1 = int(xmin / 1000 * w)
    y1 = int(ymin / 1000 * h)
    x2 = int(xmax / 1000 * w)
    y2 = int(ymax / 1000 * h)
    
    # OpenCV draw box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    # ...
```

#### 4. 結果ショーケース (ショーケース)

**1.データ構造の例:**
最終的に生成された `llava_instruct.json` は次の標準構造を持ち、トレーニング パイプラインで直接読み取ることができます。

```json
{
  "id": "1296_laptop",
  "image": "000000001296.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "Where is the laptop in the image? <image>"
    },
    {
      "from": "qwen",
      "value": "The laptop is located at [350, 201, 680, 505]."
    }
  ]
}
```

**2.可視化検証レポート:**
`visualize_bbox.py` を実行すると、検証イメージが `viz_debug` ディレクトリに生成されます。ボックスがオブジェクトを正確にフレーム化している場合 (以下に示すように)、データ パイプライン ロジックは正しいです。

**エフェクト画像の生成:**

![図 4: エフェクト画像](../../images/part6/图4_viz_000000001490.jpg)


#### 5. コストと最適化 (コストと最適化)

- **リソース消費量:**
  - **API コスト:** `interleaved.py` は外部 LLM API に依存します。 0.5 ドル/100 万トークンで 10,000 個のマルチ画像比較サンプルを生成するには、約 20 ～ 30 ドルの費用がかかります。
  - **計算時間:** `alignment.py` は純粋な CPU です。 COCO 検証セット (5k 画像) の処理には数秒かかります。

- **スケーリングに関する考慮事項:**
  - **同時処理:** 数百万の画像 (Objects365 など) を処理する場合、`(h, w)` のシングルスレッドの画像読み取りがボトルネックになります。 16 プロセスの並列読み取りと変換のために `multiprocessing` を導入できます。
  - **ネガティブ サンプル マイニング:** 現在のコードは、「オブジェクトがどこにあるか」ポジティブ サンプルのみを生成します。モデルの堅牢性を高めるために、「画像に象はありますか? -> いいえ」のような否定的なサンプルを生成するようにコードを拡張します。
