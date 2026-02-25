# 第 9 章: SFTデータ (SFT データ) —— モデルの「行動規範」の構築

## 章の概要

この章では、大規模言語モデル (LLM) のライフサイクルにおける重要な段階、つまり「一般的な事前トレーニング」から「特定の命令に従う」への重要な移行について詳しく説明します。私たちは単純なデータ収集を超えて、高度に設計された Prompt システムと自動パイプライン (Self-Instruct、Evol-Instruct) を通じて高品質の SFT (教師あり微調整) データセットを構築する方法を探ります。技術的な実装だけでなく、理論的基礎も分析します。なぜ少量の高品質なデータから膨大なモデルの機能を解き放つことができるのでしょうか?私たちは、データの多様性の不足、命令の複雑さの不十分さ、推論能力の不足に対処することに重点を置き、最終的には知識とルールの両方を理解するインテリジェントなエージェントを構築します。

### 学習目標
* **反復的なシステム プロンプト エンジニアリングをマスター**: モデルの出力形式、スタイル、深さを制御するシステム プロンプトを作成します。ロールの定義がデータ分散にどのような影響を与えるかを理解します。
* **自動データ生成パイプラインの深い理解**: Self-Instruct および Evol-Instruct アルゴリズムを再現および改善し、ドメイン命令セットを最初から構築し、その背後にある「教師と生徒」の蒸留ロジックを理解します。
* **思考連鎖 (CoT) データの構築方法を学習します**: 明示的な推論ステップを通じてモデル ロジックを強化し、トランスフォーマーの「ブラック ボックス」マッピングを打ち破ります。
* **データ フィルタリングと重複排除メカニズムの設計**: ROUGE 重複排除からセマンティック ベクトル クラスタリングまでの高度なクリーニング戦略をマスターし、合成データの品質と多様性を確保します。

> シナリオの紹介:
「あなたのチームが 70B パラメーターの基本モデルの事前トレーニングを終えたところだと想像してください。計算に数百万ドルを費やし、インターネット上のほぼすべてのテキストを読んでいます。この時点では、知識は豊富だが内向的な図書館員のようなもので、頭の中はシェイクスピアの戯曲、Python コード、量子力学の公式でいっぱいです。それでも、デモで興奮して「減量計画の作成を手伝ってください」と入力すると、モデルは機械的に次のように続行します。 「...は良い目標です。通常は食事と運動が含まれます。」あるいは「減量計画の定義」を書き始めて、ウィキペディア風のナンセンスの山を生成します。

なぜこのようなことが起こるのでしょうか?基本モデルのトレーニング目標は「次のトークンを予測する」ことであるため、「命令」と「応答」の相互作用パターンを理解していません。この「学者」をあなたの意図を理解する個人アシスタントに変えるには、何千もの質の高い Q&A ペアを与えて、話し方や問題の解決方法を段階的に教える必要があります。しかし、100,000 個の命令を手動で記述するのは費用がかかり、時間がかかり、人間の想像力は特定のパターンに限定されることがよくあります。大規模な人によるアノテーションに頼らずに、複雑かつ多様な高品質のトレーニング データを自動的に生成するにはどうすればよいでしょうか?これが、この章で取り上げるエンジニアリング上の主要な課題です。」

---

##9.1。中核となる概念と原則 (概念と原則)

SFT フェーズでは、**データの量よりも質がはるかに重要である**という点で、業界はコンセンサスに達しました。私たちが必要とするデータは、単なる「入出力」のペアではなく、さまざまなタスクの種類、複雑さのレベル、推論パターンをカバーするサンプルです。

### 9.1.1 なぜ量よりも品質が重要なのでしょうか? — 表面形状仮説

多くの初心者は、SFT が「新しい知識を学ぶためのもの」であると誤解しています。ただし、LIMA (Less Is More for Alignment) などの古典的な研究の結果に基づくと、SFT の中心的な役割は、**知識を注入することではなく**、**フォーマットを調整すること**です。

**曲面形状仮説**では、モデルは事前トレーニング中に世界の知識と論理的能力の大部分をすでに獲得していると仮定します。 SFT は、事前トレーニングから潜在的な機能を抽出するための特定の「対話形式」または「スタイル」をモデルに教えるだけです。言い換えれば、事前トレーニングがモデルに図書館全体を読ませるようなものだとすると、SFT は本の内容ではなく、人間が好む口調で答える方法だけをモデルに教えます。

これは、データにエラー、ノイズ、または論理ギャップが含まれるとモデルのパフォーマンスが急激に低下する理由を説明します。これは、モデルが「知識にインデックスを付ける方法」を学習しているためです。インデックスが間違っていると、どれだけの知識を正しく取得することもできません。したがって、数千の高品質で多様性の高いサンプルは、数百万の低品質で均質なサンプルよりも優れたモデルをトレーニングすることがよくあります。

### 9.1.2 迅速なエンジニアリングに関するエンジニアリングの観点

データ合成において、プロンプトは単なるダイアログ入力ではなく、データを生成するためのソース コードになります。プロンプトをプログラム可能なモジュールとして扱い、反復的な最適化を通じて合成データの分布を制御します。優れたプロンプト システムには通常、次のものが含まれます。

* **システム プロンプト**: データ ジェネレーターの「ペルソナ」と「境界」を定義します。これは単にアイデンティティを割り当てるだけではなく、ロールプレイングを通じてモデルの潜在的な領域固有の語彙分布を活性化します。たとえば、「厳格な弁護士」と「熱心な営業マン」を演じると、明らかに異なる文構造が生成されます。
* **少数ショットの例**: コンテキスト学習を通じて出力形式とスタイルを固定します。これらの例は「ノイズ除去」として機能し、モデルに「これが私が望む標準的な答えです」と伝えます。
* **負の制約**: モデルが特定のデータ パターンを生成することを明示的に禁止します。 LLM 生成では、モデルが遅延したり、一般的な決まり文句を使用したりする傾向があります。この統計的慣性を打ち破るには、否定的な制約が鍵となります（例：「物語の冒頭に『むかしむかし、山がありました』を使用しないでください）。」

### 9.1.3 自動化された建設手法

人間データのボトルネックを打破するために、業界は 2 つの中心的な戦略を進化させてきました。バランスの取れたデータセットを構築するには、それらの違いを理解することが重要です。

* **自己指導**: **幅広い**に焦点を当てます。強力なモデル (GPT-4 など) を使用して、少数のシード タスクから多くの新しいタスクを生成します。その中心的な前提: モデルは十分なタスク タイプを確認しています。プロンプトを通じてそれらを誘導するだけで済みます。
* **Evol-Instruct**: **深さ**に焦点を当てます。特定の進化演算子 (「制約を追加する」、「推論を深める」など) を使用して、単純な命令を複雑な命令に書き換えます。これは、単純で短い命令を生成する Self-Instruct の傾向に直接対処し、モデルの論理的複雑さと制約を満たすことを強制します。

![図 9-1: Self-Instruct と Evol-Instruct の比較](../../images/part4/图9_1_自我指令和进化指令对比.png)
*図 9-1: Self-Instruct と Evol-Instruct の比較*

**表 9-1: 主流の命令データ構築戦略の比較**

|特集 |手動注釈 |自己指導 | Evol-Instruct |
| :--- | :--- | :--- | :--- |
| **中心的な目標** |非常に精度の高い、ドメイン固有の知識 |タスクの多様性を高める |タスクの複雑さの増加 |
| **コスト** |非常に高い (1 アイテムあたり $1 ～ $10) |低価格 ($0.01/アイテム) |中 ($0.03/アイテム、複数回の通話が必要) |
| **入力ソース** |ドメインの専門家 |シードタスクプール |既存の簡単な指示 |
| **動作ロジック** |専門家の執筆とレビュー | 「既存のタスクとは異なる新しいタスクを生成する」 | 「このタスクをより難しいものに書き直してください。たとえば、制約を追加します。」 |
| **典型的な演算子/メソッド** |掃除、レビュー、クラウドソーシング | ROUGE 重複排除、名詞/動詞フィルタリング |深化する進化、広がる進化 |
| **使用例** |コア ビジネス ロジック、RLHF ゴールデン データセット |一般的なタスクの対象範囲の拡張、コールド スタート |コード、数学、論理的推論の強化 |
| **潜在的なリスク** |スケールが難しい。疲労による品質の変動 |均質で単純な指示になりがち |過度に複雑または解決不可能な「幻覚指示」が生成される可能性があります。

### 9.1.4 思考連鎖 (CoT) データ: 推論のブラック ボックスを打ち破る

CoT の核心は、「入力→出力」のブラックボックス マッピングを破壊し、モデルに暗黙的な推論を明示的にさせることにあります。

認知科学の観点から見ると、人間は複雑な問題 (数学など) を解決するときに、頭の中で一連の中間計算を実行します。 Transformer モデルは強力ですが、CoT トレーニングがないと、生徒に自分の作業を見せずに答えを書くように求めるなど、答えを直接推測する傾向があり、間違いが非常に発生しやすくなります。 CoT データは、Transformer の中間計算層をアクティブにし、生成されたトークン シーケンスを拡張することで、より多くの計算を困難な問題に割り当てることができるようにします (より多くの計算時間 = より多くのトークンが生成されます)。

### 9.1.5 データフォーマットの標準化（データフォーマット規格）

エンジニアリング実装の前に、データがどのようにモデルに「供給」されるかを理解する必要があります。これは単なる JSON 解析の問題ではなく、モデルが会話履歴をどのように理解するかに関係しています。業界では主に **ChatML** (チャット マークアップ言語) 形式を採用しています。これにより、システム、ユーザー、アシスタントの境界が明確に区別され、プロンプト インジェクション攻撃が防止されます。

トレーニング中は、通常、`assistant` 応答内のトークンの損失のみを計算し、`system` および `user` 部分をマスクすることに注意してください。これは、モデルに「質問の仕方」ではなく「答え方」を学習してもらいたいからです。

```json
// ChatML format example
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum entanglement."},
    {"role": "assistant", "content": "Quantum entanglement is a phenomenon..."}
  ]
}
```

---

##9.2。エンジニアリング実装 (エンジニアリング実装)

このセクションでは、ツールチェーンの選択とパイプラインの安定性の設計を含む、完全なデータ合成パイプラインの構築について説明します。

### 環境/依存関係
* `langchain` / `langsmith`: プロンプト テンプレート、LLM 呼び出しチェーン、およびデバッグの追跡を管理します。
* `rouge_score`: テキストの類似性と重複排除を計算します。
* `numpy / scikit-learn`: ベクトル化された重複排除 (高度な) の場合、Embeddingを介して意味論的な距離を計算します。
* `openai` または `vllm`: 教師モデルを呼び出すため。 vLLM は、ローカルに展開された高スループットのオープンソース教師モデル (Mixtral-8x7B など) に適しています。
* `chromadb` / `faiss`: 大規模な重複排除と検索用のベクトル データベース。

### 9.2.1 データ生成のための迅速なエンジニアリング

合成データ エンジニアリングでは、プロンプトは非常に堅牢である必要があります。ソフトウェアのバージョンを開発するのと同じように、反復的な思考を採用してシステム プロンプトを改良します。

#### タスクの目的: 「財務分析アシスタント」をトレーニングするための指導データのバッチを構築します。

**ステップ 1: システム プロンプトを繰り返し作成する**
* **V1: シンプルすぎるため、データの均質化につながる**
    * **欠陥分析**: モデルで生成された指示は非常に短く、基本的な概念の説明 (例: 「株式とは何ですか?」) に集中する傾向があります。これは、「高確率で低複雑性」のテキストを生成する LLM のデフォルトの傾向を反映しています。
    ```python
    # V1 Prompt - Poor performance
    system_prompt_v1 = """
    You are a financial expert. Please generate 5 questions and answers about finance.
    """
    ```

* **V2: 構造化された要件を追加**
    * **改善**: 解析を容易にするために JSON 形式の要件を導入しました。ガイドスタイルに「役割」設定を追加しました。
    * **欠陥分析**: 形式は正しいものの、内容に深みがまだなく、推論プロセスがありません。モデルは教科書的な定義のみを生成し、実際的なシナリオは生成しません。
    ```python
    # V2 Prompt - Structural improvement
    system_prompt_v2 = """
    You are a Senior Financial Analyst with 20 years of experience.
    Generate 5 pairs of instruction-response data focused on corporate finance.
    Format the output as a JSON list.
    Each item should have: 'instruction', 'input' (optional), 'output'.
    """
    ```

* **V3: 最終運用対応バージョン**
    * **改善**: 少数ショット、負の制約、および複雑さの要件が導入されました。これは業界で使用されている標準的なパラダイムです。
    * **詳細な分析**: 明示的な「アンチパターン」を通じて、モデルが「ナンセンス」を生成する経路を遮断します。見本は単なる形式の参照ではなく、思考の深さのアンカーでもあります。

    ```python
    # V3 Prompt - High robustness production version
    system_prompt_v3 = """
    ### ROLE
    You are a Chief Market Strategist at a top-tier investment bank. Your goal is to train a junior analyst model. You demand precision, depth, and actionable insights.

    ### OBJECTIVE
    Generate 5 high-quality, complex instruction-following examples related to market analysis, risk management, or quantitative trading.

    ### CONSTRAINTS
    1. **Complexity**: Do NOT ask simple definitional questions (e.g., "What is a bond?"). Instead, ask for scenario analysis, portfolio adjustments, or impact assessments.
    2. **Format**: Strictly output a valid JSON list.
    3. **Reasoning**: The 'output' must demonstrate step-by-step analytical reasoning before giving the conclusion.
    4. **Anti-Patterns**:
       - Avoid generic advice like "Consult a financial advisor."
       - Avoid short, one-sentence responses.
       - Avoid vague statements; use numbers and specific financial instruments where possible.

    ### OUTPUT FORMAT
    [
      {
        "instruction": "...",
        "input_context": "..." (can be null),
        "output": "..."
      }
    ]

    ### EXEMPLAR (One-Shot)
    [
      {
        "instruction": "Given a portfolio heavily weighted in tech stocks (60%), analyze the impact of a sudden 50bps rate hike by the Fed.",
        "input_context": null,
        "output": "First, we identify the correlation... Tech stocks are long-duration assets... Discounted Cash Flow (DCF) models would show... Therefore, the portfolio would likely suffer significant drawdown. I recommend hedging via..."
      }
    ]
    """
    ```
**プロのヒント:** V3 では、「アンチパターン」を明示しました。これはモデルの「たるみ」を防ぐ鍵となります。 LLM は安全で平凡な回答 (例: 「専門家に相談してください」) を生成する傾向がありますが、これはトレーニング データ内の低値のノイズであり、明示的に禁止する必要があります。

### 9.2.2 自動化された構築方法: Self-Instruct および Evol-Instruct

Evol-Instruct に基づいて簡素化されたパイプラインを実装します。中心となるのは、プロセスに検証メカニズムが導入され、プロンプトを通じて単純な命令を複雑な命令に「進化」させる方法です。

#### コア コードの内訳: Evol-Instruct パイプライン

**ステップ 1: 進化オペレーターを定義する (進化プロンプト)**

Evol-Instruct の本質は、このプロンプト テンプレートのセットにあります。深さ (制約を追加し、推論を深めます) と幅 (突然変異) という、さまざまな進化の方向を定義する必要があります。次のコードは、「制約の追加」と「推論の深化」のためのプロンプトを構築する方法を示しています。これらのプロンプトの設計は、データ品質の上限を直接決定します。

```python
class EvolutionPrompts:
    @staticmethod
    def get_deepening_prompt(instruction):
        """
        Depth evolution: Increase logical reasoning depth.
        By requiring 'explicitly ask for multiple-step reasoning', force the model from intuitive to analytical answers.
        """
        return f"""
        I want you to act as a Prompt Rewriter.
        Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle.
        But the rewritten prompt must be reasonable and must be understood and responded by humans.
        
        # Given Prompt #:
        {instruction}
        
        # Method #:
        If #Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly ask for multiple-step reasoning.
        
        # Rewritten Prompt #:
        """

    @staticmethod
    def get_constraints_prompt(instruction):
        """
        Depth evolution: Add specific constraints.
        Limit word increase (10-20 words) to prevent instructions from becoming verbose without substance.
        """
        return f"""
        I want you to act as a Prompt Rewriter.
        ... [header omitted for brevity]...
        
        # Given Prompt #:
        {instruction}
        
        # Method #:
        Please add one more constraint/requirement into #Given Prompt#.
        You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
        
        # Rewritten Prompt #:
        """

    @staticmethod
    def get_breadth_prompt(instruction):
        """
        Breadth evolution: Generate entirely new instructions on different topics based on existing ones.
        Prevents data distribution collapse into narrow domains.
        """
        return f"""
        I want you to act as a Prompt Creator.
        Please generate a brand new prompt that has the same difficulty level as #Given Prompt# but covers a completely different topic or domain.
        
        # Given Prompt #:
        {instruction}
        
        # New Prompt #:
        """
```

**ステップ 2: 進化ループと例外処理を実行する**

```python
import random

# Assume we have an LLM call interface
def call_llm(prompt):
    # Call GPT-4 or other strong model
    # In production, add retry mechanisms for network jitter
    pass

def evolve_instruction(base_instruction, depth=1):
    current_instruction = base_instruction
    
    for i in range(depth):
        # Randomly select an evolution strategy
        # Strategy probability can be adjusted; e.g., more Breadth early, more Deepening later
        strategy = random.choice(['deepening', 'constraints', 'breadth'])
        
        if strategy == 'deepening':
            prompt = EvolutionPrompts.get_deepening_prompt(current_instruction)
        elif strategy == 'constraints':
            prompt = EvolutionPrompts.get_constraints_prompt(current_instruction)
        else:
            prompt = EvolutionPrompts.get_breadth_prompt(current_instruction)
            
        # Get evolved instruction
        evolved_candidate = call_llm(prompt)
        
        # Quality check (simple): Prevent evolution failure
        # Often models output "Sorry, I can't do that" or simply repeat the original
        if "sorry" in evolved_candidate.lower() or len(evolved_candidate) < 10:
            print(f"Evolution failed at step {i}, keeping previous instruction.")
            break
            
        # Advanced check: Simple heuristic to detect simple repetition
        if evolved_candidate.strip() == current_instruction.strip():
             print(f"Evolution stagnant at step {i}.")
             break

        current_instruction = evolved_candidate
        
    return current_instruction

# Example run
seed = "Write a Python script to calculate Fibonacci numbers."
complex_instruction = evolve_instruction(seed, depth=3)
# Expected result: "Write a Python script to calculate the nth Fibonacci number using dynamic programming, optimize for memory usage, and handle negative input values."
```

**ステップ 3: パフォーマンス最適化のヒント**
* **バッチ処理:** API を 1 つずつ呼び出さないでください。 20 個の命令を含むプロンプト リストを構築し、モデルが一度に 20 個の進化した結果を返すようにします。これにより、トークン コストとネットワーク遅延が大幅に削減されます (高スループット)。
* **失敗フィルター:** 進化は頻繁に失敗します (モデルが繰り返しを開始するなど)。フィルターを実装します。進化した命令の長さが短くなったり、典型的な拒否フレーズ (「AI として…」) が含まれている場合は、サンプルを破棄します。
* **多様性制御:** バッチ生成では、1 つのバッチ内のすべての命令が「Python プログラミング」に関するものになるのを避けるために、システム プロンプトで「多様なトピックの生成」を明示的に要求します。

### 9.2.3 思考連鎖 (CoT) データ: 段階的な推論サンプルの構築

SFT データの中核となる価値は、モデルに「考え方」を教えることにあります。通常の Q&A ペア (Q: 1+1? A: 2) では、結果のみを教えます。 CoT はプロセスを教えます。

#### CoT データ構築プロンプト テンプレート

「ステップごとに考えてみましょう」を追加するだけではなく、モデルが特定の形式 (`<thinking>` タグなど) で出力する必要があります。このフォーマットされたデータは、その後のトレーニングで非常に役立ちます。モデルに思考プロセスを出力させるか、推論中に非表示にするかを選択できます。

```python
# System Prompt for CoT Generation
cot_system_prompt = """
You are a reasoning expert. When answering the user's question, you MUST follow this format:

1. Enclose your thought process in <thinking> tags.
2. Break down the problem into logical steps.
3. Check for edge cases or potential errors in your logic (Self-Correction).
4. Output the final answer in <answer> tags.

Do NOT output the answer directly without the thinking process.
"""

# V3 Prompt Iteration for Math CoT
user_prompt_cot = """
Question: A farmer has chickens and cows. There are 20 heads and 50 legs. How many of each are there?

Please provide a Step-by-Step solution.
"""
```

**出力例 (トレーニング データとして):**
```xml
<thinking>
Step 1: Define variables. Let C be chickens, K be cows.
Step 2: Set up equations based on the problem statement.
   Equation 1 (Heads): C + K = 20
   Equation 2 (Legs): 2C + 4K = 50
Step 3: Solve for C from Equation 1. 
   C = 20 - K.
Step 4: Substitute C into Equation 2.
   2(20 - K) + 4K = 50
   40 - 2K + 4K = 50
   2K = 10 
   K = 5.
Step 5: Solve for C using the value of K. 
   C = 20 - 5 = 15.
Step 6: Verify the solution. 
   Heads: 15 + 5 = 20. Correct.
   Legs: 15*2 + 5*4 = 30 + 20 = 50. Correct.
</thinking>
<answer>
Chickens: 15, Cows: 5
</answer>
```
**プロのヒント:** `<thinking>` タグを使用したこのデータ戦略は、「内部独白」トレーニングと呼ばれます。 SFT 中に、モデルが思考プロセスの出力を学習できるように、これらのタグを保持します。運用環境では、解析コードを使用して `<thinking>` コンテンツをキャプチャし、`<answer>` のみをユーザーに表示するか、説明しやすくするために部分的な推論手順を示す「思考...」UI アニメーションを実装します。

---

## 9.2.4。実績と評価 (実績と評価)

データ生成は最初のステップにすぎません。生成されたデータの品質を評価する方法も同様に重要です。モデルのトレーニングが完了するまで (場合によっては数日、数千ドルかかる)、データ品質の低下を発見するのを待つことはできません。

### 評価指標
* **指示フォロー率:** 通常は自動テストです。 GPT-4 を判断材料として使用し、モデルで生成された応答が入力内のすべての制約 (例: 「単語数制限」、「特定のキーワードを含む」、「JSON 形式」) を厳密に満たしているかどうかを判断します。
* **複雑さの分布:** NLP ツール (SpaCy など) を使用して、動詞の多様性、構文ツリーの深さ、生成された命令の平均長を分析します。分布ヒストグラムをプロットして、Evol-Instruct が単に冗長であるだけでなく、実際に難易度が上がっていることを確認します。
* **多様性:** ROUGE-L を計算するか、Embeddingコサイン類似度を使用します。データセット内のサンプル間の平均類似性が高すぎる場合、「モード崩壊」が発生しており、データに多様性が欠けています。

### ベンチマーク

学界と産業界では、SFT 後のモデルの機能をテストするためのベンチマークが認められています。
* **WizardLM 論文データ:** 4 ラウンドの Evol-Instruct を通じて進化したデータでトレーニングされたモデルは、通常、生データのみを使用したモデルと比較して、GSM8K (数学) と HumanEval (コード) で 10% ～ 20% 以上の改善を示します。
* **MT-Bench:** マルチターン対話評価セットは、特に指示への従う能力、推論能力、およびマルチターン対話能力をテストするもので、通常 GPT-4 によってスコア化されます。
* **コストの参考:** `gpt-3.5-turbo` を使用して 52K の自己指示データを生成するには、約 500 ドルから 1000 ドルの費用がかかります (プロンプトの長さとラウンドによって異なります)。これは、手動による注釈の数十万ドルに比べて、非常に費用対効果が高くなります。

---

## 9.2.5。落とし穴とトラブルシューティング

実際には、データ合成には落とし穴がたくさんあります。ここでは、一般的な障害モードと解決策を示します。

* **落とし穴 1: モードの崩壊**
    * **症状:** モデル生成の命令は単調です。たとえば、1000 個の生成されたサンプルはすべて「X に関する記事を書いてください」または「Python 関数を書いてください」です。
    * **原因:** シード タスクが同種すぎるか、システム プロンプト温度の設定が低すぎるため、モデルが局所最適化に陥ります。
    * **修正:** シード タスクの多様性を高めます (100 以上の領域をカバー: 料理、法律、プログラミング、文学)。温度を上げる (0.7 → 1.0);システム プロンプトで「前の例とは異なるドメインからタスクを生成する」ことを明示的に要求します。

* **落とし穴 2: 幻覚のような制約**
    * **症状:** モデルはトレーニング データから「JSON を出力する必要がある」ことを学習し、カジュアル チャット (「こんにちは」) であっても JSON 出力を強制したり、要求されていないときに `<thinking>` タグを出力したりします。
    * **原因:** トレーニング データの分布が著しく歪んでいます。100% 複雑な指示があり、単純な一般的な対話データが不足しています。
    * **修正:** データの混合。特定の形式への過剰適合を防ぐために、10% ～ 20% の一般的な対話データ (例: ShareGPT または雑談) を Evol-Instruct データに混ぜます。これを「一般能力リプレイ」と呼びます。

* **落とし穴 3: 進化の失敗 (劣化)**
    * **症状:** 進化した指示は不条理になったり、論理的に矛盾したり (「母音なしで 1000 語の記事を書きなさい」)、または非常に冗長になります。
    * **修正:** 「長さペナルティ」または「複雑さの切り捨て」を実装します。進化した命令が複雑であるにもかかわらず GPT-4 が応答できない (または応答の品質が低い) 場合、サンプルは無効になります (悪いケース)。進化した指導の実現可能性を評価するために、GPT-4 の「教師モデル スコアリング」を導入します。

* **落とし穴 4: 壊滅的な忘却**
    * **症状:** SFT 後、モデルは指示に従うことを学習しますが、「愚か」に見え、事前トレーニングで得た世界の知識の一部を忘れています。
    * **原因:** SFT データセットはモデルの重み配分を変更し、特定のタスク フォームに過度に焦点を当て、一般的な知識のストレージを圧迫します。
    * **修正:** 学習率が低くなり、エポックが減少します (SFT は通常、2 ～ 3 エポックのみを必要とします)。パラメータ分布の安定性を維持するために、少量の事前トレーニング データ (事前トレーニング リプレイ) を SFT データに追加します。

---

## 9.2.6。各章の概要と詳細情報

私たちはプロンプトをデータのソース コードとして扱います。プロンプトは、ソフトウェア エンジニアリング コードと同様に、厳密なバージョン管理と反復テストによって管理する必要があります。このフレームワークでは、Self-Instruct は「ゼロから 1 へ」のコールド スタートの課題を解決し、一方、Evol-Instruct は「簡単から困難へ」の複雑さの上昇を克服します。それらの有機的な組み合わせは、高性能データセットを構築するための黄金のパラダイムを構成します。一方、思考連鎖 (CoT) データは、単純な問題解決のトリックとは程遠いものです。推論を明示することで、計算リソースを重要な推論ステップに効果的に割り当て、複雑なロジックを処理するモデルの能力を根本的に強化します。結局のところ、データ合成における中心的な障壁は生成能力ではなく、フィルタリングとクリーニングの技術です。大量生成の容易さと正確なスクリーニングの難しさの間では、砂から金を抽出する能力だけが真の競争力の中心を構成します。

### 参考文献と詳細情報
* *Wang, Y. 他（2022年）。 Self-Instruct: 言語モデルと自己生成命令の調整* (自動命令生成に関する基礎作業)
* *Xu、C.、他。 （2023年）。 WizardLM: 大規模言語モデルが複雑な命令に従うことができるようにする* (Evol-Instruct 進化演算子の詳細な紹介)
* *Wei、J.、他。 （2022年）。 Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* (CoT の基礎研究)
* *Zhou、C.、他。 （2023年）。 LIMA: Less Is More for Alignment.* (SFT は知識ではなく形式を主に学習するという理論的根拠を確立します。「質 > 量」)
