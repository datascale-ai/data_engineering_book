# プロジェクト 4: 総合数学/コードの教科書

> **シナリオ**: 小規模モデルの論理推論能力を強化します。
>
> **コア テクノロジー**: Evol-Instruct 進化戦略、Python コード実行サンドボックス (Sandbox) 検証、PoT (Program of Thought) データ フォーマット。
>
> **出力**: 検証後の高品質な合成推論データセット。

### 1. プロジェクトの背景 (プロジェクトの概要)

* **タスク定義:** 高品質の「思考プログラム」(PoT) データセットを構築します。 LLM (DeepSeek-V3) を使用して、単純な数学の問題を複雑な文章問題に「進化」させ、対応する Python コード解を生成し、コード実行サンドボックスを通じて解答の正しさを検証します。
* **入力と出力:**
    * **入力:** 基本数学データセット (GSM8K、MBPP など) 生の JSONL ファイル。
    * **出力:** `question` (発展した問題)、`thought_process` (コード解決推論)、`execution_output` (実行結果) を含む、クリーン化された JSONL データセット。
* **課題の分析:** このプロジェクトの最大の困難は **「幻覚の除去」です。** LLM で生成されたコードは、正しく見えても実行できない (構文エラーまたはロジック バグ) ことがよくあります。自動化された「サンドボックス」を構築して、実行不可能なサンプルを除外し、「教科書」の厳密さを確保する必要があります。

### 2. アーキテクチャ設計 (アーキテクチャ設計)

### データパイプライン図
![図 5: 合成数学/コード教科書](../../images/part6/图5_合成数学代码教科书数据流水线图.png)

### テクノロジースタック

* **データ ソース:** `HuggingFace Datasets` (GSM8K/MBPP)。
* **生成エンジン:** `DeepSeek-V3` (SiliconFlow API 経由) —— コスト効率の高いコード生成モデル。
* **オーケストレーション ロジック:** Python スクリプト (Evol-Instruct 戦略)。
* **検証環境:** Python `subprocess` (ローカル サンドボックス) —— *運用環境では Docker または MicroVM を推奨します。*

### 3. 段階的な実装

### フェーズ 1: シード データの取得 (シードの準備)

すべては高品質の種子から始まります。大量のデータは必要ありません。代表的なロジック コアだけが必要です。

**主なアクション:**
1. GSM8K (数学) および MBPP (コード) データをダウンロードします。
2. 「進化」基盤としてのランダムサンプル。

**グルー コード (データ サンプラー):**
*`download_data.py` および `sampler.py` のコード*

```python
# Core logic: Extract seeds from massive data, keep only Question field
# Original Answer discarded because we let model regenerate code-based solution
sampled = random.sample(data, SAMPLE_SIZE)
for entry in sampled:
    seed_entry = {
        "id": random.randint(1000, 9999), 
        "seed_question": entry['question'], # Keep only question
        "original_answer": entry['answer']  # For reference only
    }
```

### フェーズ 2: Evol-Instruct および PoT 生成 (進化と生成)

これがプロジェクトの核心です。単純な「Q&A ペア」を行うだけでは不十分です。人間の専門家のように考えるモデルが必要です。

**フローロジック:**
1. **Evol (進化):** 単純な問題 (例: 「1+1=?」) を複雑なシナリオ (例: 「Xiaoming にはリンゴが 1 個あり、インフレの影響を受けています...」) に書き直し、制約を追加します。
2. **PoT (コード ソリューション):** テキストの回答を直接出力するのではなく、モデルに解決する Python コードを記述するように強制します。

**コア プロンプト (プロンプト エンジニアリング):**
*`evol.py` のコード*

```python
def get_evol_prompt(seed_question):
    return f"""
    You are a professional math competition problem composer. Please rewrite the following basic math problem into a more complex, logically rigorous one.
    【Original】: {seed_question}
    【Rewrite Requirements】:
    1. Add constraints: Introduce more variables or limitations.
    2. Add reasoning depth: Don't give numbers directly; have logical relationships between numbers.
    3. Scenario-ize: Put abstract numbers into concrete physical or business scenarios.
    ...
    """

def get_pot_prompt(evolved_question):
    return f"""
    Please write Python code to solve the following math problem.
    ...
    1. Write a function named `solve()`.
    2. Clearly write reasoning steps in code comments.
    3. `solve()` must return the final numerical answer.
    ...
    """
```

### フェーズ 3: サンドボックスの検証

生成されたデータには、大量の「無効」サンプル (構文エラー、タイムアウト、ループ) が含まれています。実行を通じて検証する必要があります。

**サンドボックス ロジック:**
1. 正規表現を使用して、Markdown からコード ブロックを抽出します。
2. サブプロセス (`subprocess`) を開始してコードを実行します。
3. **重要:** `timeout` を設定して、無限ループによるパイプラインのブロックを防ぎます。

**検証スクリプト:**
*`sandbox.py` のコード*

```python
def execute_code(code, timeout=5):
    """
    Execute Python code and get output.
    WARNING: This function should only be called in strongly isolated sandbox (minimal privilege container/micro-VM, no network, restricted filesystem).
    To prevent accidentally executing arbitrary code in host environment, will raise exception if sandbox not explicitly declared.
    Can explicitly allow by setting environment variable EXECUTE_CODE_SANDBOXED=1 inside sandbox container.
    """
    # Basic protection: Prohibit executing arbitrary code in undeclared sandbox
    if os.environ.get("EXECUTE_CODE_SANDBOXED") != "1":
        raise RuntimeError(
            "execute_code can only be used in controlled sandbox environment; "
            "please set environment variable EXECUTE_CODE_SANDBOXED=1 in isolated container/micro-VM before calling."
        )
    try:
        # Use subprocess to start independent process
        result = subprocess.run(
            ['python3', '-c', code],
            capture_output=True,  # Capture stdout
            text=True,
            timeout=timeout,      # Must set timeout!
            check=False,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, f"Error: {result.stderr.strip()}"
```

### 4. 結果ショーケース (ショーケース)

サンドボックスのクリーニング後、`verified_textbook.jsonl` が得られます。これは教科書レベルの合成データです。

**データサンプルの比較:**

|フェーズ |コンテンツ例 |
| :--- | :--- |
| **オリジナルシード** |ジェニーはリンゴを 5 つ持っていますが、2 つ食べました。あと何個残っていますか? |
| **進化の進化** |ジェニーは果物屋を経営しており、リンゴが 5 箱 (1 箱あたり 12 個) 入っています。月曜日、彼女は在庫の 40% を売却し、2 つの単品商品が不適切な保管により破損してしまいました。残りの販​​売可能なリンゴの正確な数を計算してください。 |
| **PoT ソリューション** | `def solve(): total = 5 * 12; sold = total * 0.4; ... return remaining` |
| **実行結果** | `34` (検証済み、データセットに保存) |

**検証統計:**
通常、Evol 後のコードのワンショット合格率 (Pass@1) は **60% ～ 80%** です。サンドボックスによってフィルタリングされた 20% のエラー データは、まさにモデル トレーニングを汚染するデータです。**これらを削除すると、SFT モデルのロジックの一貫性が大幅に向上します。**

### 5. コストと最適化 (コストと最適化)

* **リソース消費量:**
    * **API コスト:** 有効なサンプルごとに、最大 2 回の LLM 呼び出し (進化 + ソリューション) が消費されます。 DeepSeek-V3 のようなコスト効率の高いモデルを使用すると、1,000 個の高品質教科書サンプルの生成を 5 ドル未満で制御できます。
    * **時間コスト:** ローカル Python シングルスレッドは遅いです。 1,000 個のコード サンプルの検証には約 5 ～ 10 分かかります。

* **セキュリティ警告 (重大):**
    * このプロジェクトはローカル実行に `subprocess` を使用します。未知または信頼できないモデル生成コードを処理する場合、**非常に高いリスク**が存在します (例: `os.system('rm -rf /')`)。
    * **本番環境の変革計画:** ネットワーク アクセスを無効にして、`sandbox.py` 実行環境を **Docker コンテナ** または **AWS Firecracker** マイクロ VM に移行する必要があります。

* **スケーリングに関する考慮事項:**
    * データが数百万にスケールする場合、単一マシンのスクリプトではサポートできません。タスク分散のために `RabbitMQ` または `Kafka` を導入し、分散「生成-検証」クラスターを構築する必要があります。
