# 大規模モデルのデータ エンジニアリング: アーキテクチャ、アルゴリズム、プロジェクト

[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://datascale-ai.github.io/data_engineering_book/ja/)
[![ライセンス](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**[English](README_en.md) | [中文](README.md) | 日本語**

## 導入

> *「データは新しい石油ですが、それを精製する方法を知っている場合に限ります。」*

大規模モデルの時代では、**データ品質がモデルのパフォーマンスの上限を決定します**。しかし、LLM データ エンジニアリングに関する体系的なリソースは依然として非常に不足しており、ほとんどのチームはまだ試行錯誤によって学習しています。

本書はそのギャップを埋めるために企画されました。 **トレーニング前データ クリーニング**から**マルチモーダル アライメント**、**RAG 取得拡張**から**合成データ生成**に至るまで、次のような完全な技術スタックを体系的にカバーしています。

- 🧹 **事前トレーニング データ エンジニアリング**: Common Crawl などの大規模なノイズの多いデータ ソースから高品質のコーパスを抽出する
- 🖼️ **マルチモーダル データ処理**: 画像とテキストのペア、ビデオ、オーディオ データの収集、クリーニング、配置
- 🎯 **アライメント データ構築**: SFT 命令データ、RLHF 優先データ、および CoT 推論データの自動生成
- 🔍 **RAG データ パイプライン**: エンタープライズ グレードのドキュメント解析、セマンティック Chunking、マルチモーダル検索

この本には、詳細な理論的説明に加えて、実践的な学習のための実行可能なコードと詳細なアーキテクチャ設計を備えた **5 つのエンドツーエンドのキャップストーン プロジェクト**が含まれています。

**オンラインで読む**: [https://datascale-ai.github.io/data_engineering_book/ja/](https://datascale-ai.github.io/data_engineering_book/ja/)

## 本のアーキテクチャ

![本のアーキテクチャ](images/structure_en.png)

*生データからエンドツーエンドのアプリケーションまでの完全なデータ エンジニアリング パイプライン*

## 目次

```
📖 6 Parts, 13 Chapters + 5 Capstone Projects
│
├── Part 1: Infrastructure & Core Concepts
│   ├── Chapter 1: Data Revolution in the LLM Era
│   └── Chapter 2: Data Infrastructure Selection
│
├── Part 2: Text Pre-training Data Engineering
│   ├── Chapter 3: Data Acquisition
│   ├── Chapter 4: Cleaning & Deduplication
│   └── Chapter 5: Tokenization & Serialization
│
├── Part 3: Multimodal Data Engineering
│   ├── Chapter 6: Image-Text Pair Processing
│   ├── Chapter 7: Recaptioning
│   └── Chapter 8: Video & Audio Data
│
├── Part 4: Alignment & Synthetic Data Engineering
│   ├── Chapter 9: Instruction Fine-tuning Data
│   ├── Chapter 10: Synthetic Data
│   └── Chapter 11: Human Preference Data
│
├── Part 5: Application-level Data Engineering
│   ├── Chapter 12: RAG Data Pipeline
│   └── Chapter 13: Multimodal RAG
│
└── Part 6: Capstone Projects
    ├── Project 1: Building Mini-C4 Pre-training Set
    ├── Project 2: Domain Expert SFT (Legal)
    ├── Project 3: Building LLaVA Multimodal Instruction Set
    ├── Project 4: Synthetic Math/Code Textbook
    └── Project 5: Multimodal RAG Financial Report Assistant
```

## 主要なハイライト

### 総合理論
- **データ中心の AI** 哲学全体
- LLM データのライフサイクル全体をカバーします: 事前トレーニング → 微調整 → RLHF → RAG
- スケーリングの法則、データ品質評価、マルチモーダル調整などを詳しくカバー

### 最新の技術スタック
|ドメイン |テクノロジー |
|--------|-------------|
|分散コンピューティング |レイデータ、スパーク |
|データストレージ |寄木細工、WebDataset、ベクトル データベース |
|テキスト処理 |トラフィラトゥーラ、KenLM、ミンハッシュ LSH |
|マルチモーダル | CLIP、ColPali、img2dataset |
|データのバージョン管理 | DVC、LakeFS |

### 豊富なキャップストーン プロジェクト

|プロジェクト |コア技術 |出力 |
|---------|-------------------|--------|
| Mini-C4 プレトレーニングセット |トラフィラトゥーラ + レイ + ミンハッシュ |高品質のテキストコーパス |
|法律専門家 SFT |自己指導 + CoT |ドメイン命令データセット |
| LLaVA マルチモーダル | Bbox アライメント + マルチ画像インターリーブ |視覚的な指示データセット |
|数学の教科書 | Evol-Instruct + サンドボックス検証 | PoT 推論データセット |
|財務報告書 RAG |コルパリ + クウェン VL |マルチモーダル QA システム |

## 地域開発

### 要件

- Python 3.8以降
- MkDocs マテリアル
- mkdocs-static-i18n (i18n サポート)

### インストールとプレビュー

```bash
# Clone the repository
git clone https://github.com/datascale-ai/data_engineering_book.git
cd data_engineering_book

# Install dependencies
pip install mkdocs-material mkdocs-glightbox pymdown-extensions "mkdocs-static-i18n[material]"

# Local preview
mkdocs serve
```

この書籍をプレビューするには、http://127.0.0.1:8000 にアクセスしてください (中国語/英語の言語スイッチャーを使用)。

### 静的サイトを構築する

```bash
mkdocs build
```

生成された静的ファイルは `site/` ディレクトリにあります。

## プロジェクトの構造

```
data_engineering_book/
├── docs/
│   ├── zh/                  # Chinese content
│   │   ├── index.md         # Chinese homepage
│   │   └── part1/ ~ part6/  # All chapters
│   ├── en/                  # English content
│   │   ├── index.md         # English homepage
│   │   └── part1/ ~ part6/  # All chapters
│   ├── images/              # Image assets (shared)
│   ├── stylesheets/         # Custom styles
│   └── javascripts/         # JavaScript (MathJax etc.)
├── .github/workflows/       # GitHub Actions CI/CD
├── images/                  # Project image assets
│   ├── structure_cn.png     # Book architecture diagram (Chinese)
│   └── structure_en.png     # Book architecture diagram (English)
├── mkdocs.yml               # MkDocs configuration
├── LICENSE                  # License
├── README.md                # 中文说明
└── README_en.md             # English README (this file)
```

## 対象読者

- LLM 研究開発エンジニア
- データ エンジニア / MLOps エンジニア
- AI プロダクト マネージャー (技術)
- LLM データ パイプラインに興味のある研究者

## 貢献する

貢献は大歓迎です！問題やプルリクエストを遠慮なく送信してください。

1. このリポジトリをフォークする
2. 機能ブランチを作成します (`git checkout -b feature/AmazingFeature`)
3. 変更をコミットします (`git commit -m 'Add some AmazingFeature'`)
4. ブランチにプッシュします (`git push origin feature/AmazingFeature`)
5. プルリクエストを開く

## ライセンス

このプロジェクトは MIT ライセンスに基づいてライセンスされています。詳細については、[LICENSE](LICENSE) ファイルを参照してください。

## 接触

- GitHub の問題: [問題を送信する](https://github.com/datascale-ai/data_engineering_book/issues)
- オンラインで読む: [https://datascale-ai.github.io/data_engineering_book/ja/](https://datascale-ai.github.io/data_engineering_book/ja/)

---

**この本が役に立ったと思われる場合は、スターを付けてください!** ⭐
