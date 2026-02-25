# プロジェクト 2: 垂直領域エキスパート SFT (法務)

> **シナリオ**: 業界の専門家が非構造化 PDF ドキュメントから微調整したデータを構築します。
> **コア テクノロジー**: 命令の構築、CoT 推論の強化、データの多様性のバランスのための Self-Instruct。
> **出力**: `domain_expert.jsonl` SFTデータセット。

### 1. プロジェクトの背景 (プロジェクトの概要)

- **タスク定義:** 非構造化法規制 PDF 文書から知識を抽出し、LLM Self-Instruct を使用して、「思考連鎖 (CoT)」機能を備えた垂直ドメインSFTデータセットを構築します。
- **入力と出力:**
  - **入力:** 生の PDF ドキュメント (例: ヘッダー、フッター、透かしの干渉のある民法、刑法)。
  - **出力:** `domain_expert.jsonl`、指示 (ユーザー指示) と出力 (思考プロセスによる専門家の応答) が含まれます。
- **課題分析:**
  1. **PDF ノイズ クリーニング**: 法律文書には引用マーカー (例: `[1]`)、改行で区切られた中国語 (例: `法 律`)、およびテキストに埋め込まれたページ番号 (例: `- 195 -`) が頻繁に含まれており、これらを除去するのは非常に困難です。
  2. **データの均一性**: 単純な「法規定の説明」だけでは、エキスパート モデルのトレーニングには不十分です。事例分析や文書作成など、さまざまなタスクを構築する必要があります。
  3. **推論能力の欠如**: 通常の QA ペアには論理的導出がありません。モデルに CoT (思考の連鎖) を生成させる必要があります。

### 2. アーキテクチャ設計 (アーキテクチャ設計)

**データ パイプライン図:**

![図 2: 垂直ドメイン エキスパート SFT の構築](../../images/part6/图2_构建垂直领域专家SFT数据流水线图.png)


- **テクノロジースタック:**
  - **PDF 解析 (pdfplumber)**: PyPDF2 と比較して、pdfplumber はより正確な境界ボックス コントロールを提供し、ヘッダー/フッターの削除に便利です (コード セットの上部/下部カットオフは 5%)。
  - **クリーニング エンジン (正規表現)**: 中国語の単語区切りおよび引用マーカー用の「グルー コード」 - データ品質を向上させる鍵。
  - **生成モデル (DeepSeek-V3)**: Self-Instruct データ合成に強力な論理的推論と低コストの API を活用します。
  - **オーケストレーション ロジック (Python)**: タスク タイプの多様性バランスに加重ルーレット ホイール アルゴリズムを使用します。

### 3. 段階的な実装

#### フェーズ 1: データの取得とインテリジェントなクリーニング (汚い仕事)

PDF 抽出の最大の問題は、形式が混乱していることです。 `data_processing.py` の `clean_text_smart` 関数がこのための中核です。私たちは「中国語の誤ったスペース」と「埋め込まれたページ番号」の問題の解決に重点を置いています。

**キーコードロジック:**

```python
def clean_text_smart(text):
    """
    Cleaning core logic: Fix format damage from PDF parsing
    """
    # 1. Remove citation references (e.g., [1], [1-3])
    text = re.sub(r'\[\s*\d+(?:[-–,]\d+)*\s*\]', '', text)

    # 2. Remove page numbers embedded in text (e.g., "- 195 -")
    # Use lookahead assertion to prevent accidental deletion of body numbering
    text = re.sub(r'(?:^|\s|\n)[-—–－]\s*\d+\s*[-—–－](?=\s|\n|$)', ' ', text)

    # 3. Fix Chinese word-break (core fix)
    # Scenario: PDF "法 律 规 定" gets recognized with spaces, need to merge
    pattern_broken_zh = r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])'
    # Execute twice for consecutive breaks
    text = re.sub(pattern_broken_zh, r'\1\2', text)
    text = re.sub(pattern_broken_zh, r'\1\2', text) 
    
    return text.strip()
```

#### フェーズ 2: 多様な命令合成 (多様性と CoT)

モデルが法則を暗記する本の虫になるのを避けるために、`generate_instructions.py` で **タスク プール** と **確率サンプリング** を設計し、モデルに 3 つの異なるタスク タイプを生成させます。

**ダイバーシティバランス戦略:**

```python
# Task weight configuration (data distribution control)
TASK_POOL = [
    # Task A: Complex case analysis (reasoning focus) - weight 60%
    ("case_analysis", PROMPT_CASE_ANALYSIS, 0.6),
    # Task B: Legal document drafting (generation focus) - weight 20%
    ("doc_drafting", PROMPT_DOCUMENT_DRAFTING, 0.2),
    # Task C: Legal concept discrimination (knowledge focus) - weight 20%
    ("concept_explain", PROMPT_CONCEPT_EXPLAIN, 0.2)
]

# Roulette wheel selection logic
rand = random.random()
cumulative_prob = 0
for name, tpl, prob in TASK_POOL:
    cumulative_prob += prob
    if rand <= cumulative_prob:
        # Hit task type, use corresponding Prompt
        selected_prompt_tpl = tpl
        break
```

#### フェーズ 3: フォーマットと CoT の強化

プロンプトでは、モデルが「思考プロセス」を含む JSON を返すことを明示的に要求します。後処理では、暗黙的な思考連鎖を明示化し、最終的なトレーニング ターゲット形式に連結します。

**CoT フォーマット ロジック:**

```python
# Parse model JSON response, force chain-of-thought format construction
if isinstance(raw_output, dict):
    thought = raw_output.get("思考过程") or raw_output.get("analysis")
    answer = raw_output.get("法律建议") or raw_output.get("conclusion")
    
    # Explicitly write thinking process into Output, training model to "think before speak"
    formatted_output = f"#### 🧠 Thinking Process\n{thought}\n\n#### 📝 Expert Advice\n{answer}"
```

### 4. 結果ショーケース (ショーケース)

処理後、生の鈍い法律条項は、深い推論を備えた Q&A ペアに変わります。以下は、`domain_expert.jsonl` からの単一のデータ サンプルです。

**出典 (生の法規定の断片):**
> ...第 150 条: 一方の当事者が詐欺を利用して相手方の真意に反して民事法律行為を実行させた場合、詐欺を受けた当事者は人民法院または仲裁機関に取消しを請求する権利を有する...

**生成された SFT データ (JSONL エントリ):**

|フィールド |コンテンツ例 |
| :--- | :--- |
| **指示** |契約時に騙されてしまいました。相手は骨董品は本物だと言いましたが、偽物でした。今何をすべきでしょうか？それを尊重することを拒否することはできますか？ |
| **出力** | **#### 🧠思考プロセス**<br>1. **法的関係の分析**: ユーザーが相手方の詐欺に基づいて契約を締結した (民事法律行為を行った)。<br>2. **適用規定**: 民法第 150 条により、これは「無効となる」民事法律行為であり、直接「無効」または「名誉の拒否」ではありません。<br>3. **対応策**: 詐欺を受けた当事者は取り消しの権利を行使する必要があり、裁判所または仲裁機関を通じて行う必要があります。<br><br>**#### 📝 専門家のアドバイス**<br>説明によると、あなたは真の意図に反して契約を締結したことになります。これは **無効可能な民事法律行為**です。直接「履行を拒否」したり、契約を一方的に拒否したりすることはできません。以下をお勧めします。<br>1. **取消権の行使**: 人民法院に訴訟を起こすか、仲裁機関に契約の取消しを申請することができます。<br>2. ... |

**データ分布分析:**
- **ケース分析 (60%)**: モデルの複雑なロジック処理を強化します。
- **ドキュメント ドラフト (20%)**: モデルのプロフェッショナルなライティングを強化します。
- **概念の説明 (20%)**: 基礎的なドメイン知識を強化します。

### 5. コストと最適化 (コストと最適化)

- **リソース消費量:**
  - **API コスト**: DeepSeek-V3 を使用して 1,000 個の高品質 CoT サンプルを生成するには、~0.5 ドルから 1.0 ドルかかります (入力/出力トークンが長くなります)。
  - **時間コスト**: シングルスレッドでサンプルあたり約 2 秒。
- **スケーリングに関する考慮事項:**
  - **同時実行アクセラレーション**: 現在のコードはシングルスレッドです (`time.sleep`)。本番環境では、同時リクエストに `asyncio` + `Semaphore` を使用する必要があります。 10～20倍の効率向上が可能。
  - **品質管理**: 現在はプロンプト制約のみに依存しています。 「報酬モデルのスコアリング」または「ルール フィルター」ステップを追加して、短すぎるサンプルや JSON 解析に失敗したサンプルを削除することをお勧めします。
