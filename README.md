# 地図の中の哲学者 🗾

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

> 地図から始まる、新たな地域発見の物語  
> AIが紡ぐ哲学的な視点で、あなたの知らない日本に出会う

![デモ画像](https://github.com/user-attachments/assets/ae7edb5b-8a2d-4c90-a724-8c701749ea9f)

**第2回 AI Agent Hackathon with Google Cloud エントリー作品**

---

## 🌟 プロジェクトの背景

日本の多くの地域が直面する「地方の過疎化」という深刻な社会課題。人口減少と高齢化が進む中、地域の経済やインフラの縮小だけでなく、祭りや伝統技術といった貴重な文化的遺産も失われつつあります。

しかし、私たちは確信しています。**どの地域にも、まだ光の当たっていない魅力が眠っている**と。

「地図の中の哲学者」は、AIの力でその隠れた輝きを発見し、人と地域の新たな関係性を築くための知的探求ツールです。

## 🎯 ビジョン

AIが地域の代弁者となり、その土地ならではの「物語」を紡ぎ出すことで、移住検討者には暮らしのイメージを、旅人には訪問の動機を提供し、**人と地域の新たな出会いを創造する**ことを目指しています。

## ✨ 主な特徴

### 🗺️ 直感的な地域選択
- **ワンクリック探索**: 日本地図から気になる都道府県を選ぶだけ
- **現在地検出**: GPS連携で今いる場所の新たな魅力を発見
- **偶発的な出会い**: 知らなかった地域との運命的な出会いを演出

### 🤖 AIによる自律的な記事生成
- **完全自動化**: 情報収集から記事作成まで、AIが自律的に実行
- **多角的分析**: Web上の散在する情報を文脈的に読み解き
- **物語的構成**: 単なる情報羅列ではない、示唆に富んだ記事を生成

### 🎨 魅力的なコンテンツ表現
- **4コマイラスト**: 地域の特性を反映した起承転結のあるビジュアル
- **セクション画像**: 各テーマに最適化された個別画像
- **哲学的名言**: その土地ゆかりの深い洞察を込めた一言
- **洗練されたデザイン**: 読みやすさと美しさを両立した記事レイアウト

### ⏱️ 体験としてのローディング
- **リアルタイム進捗**: AIの思考プロセスを可視化
- **期待感の演出**: 「情報収集中」「記事構成検討中」など詳細な状況表示
- **没入感の維持**: 待つ時間も体験の一部として設計

## 🎯 ターゲットユーザー

### 🏠 未来の暮らしを探す移住検討者
制度や統計では見えない、その土地の**文化や雰囲気といった「暮らしの質感」**を伝え、納得感のある意思決定を支援

### 🎒 まだ見ぬ物語を求める旅人
ガイドブックには載っていない地域の歴史や文化を**「旅のテーマ」**として提案し、より深い体験を提供

## 🏗️ システムアーキテクチャ

<img width="891" alt="システムアーキテクチャ図" src="https://github.com/user-attachments/assets/2ee1abc0-fa9c-4245-be59-59e07d0820d4" />

## 🛠️ 技術スタック

| カテゴリ | 技術 | 役割 |
|:---|:---|:---|
| **フロントエンド** | Streamlit | WebアプリケーションUI構築 |
| **インフラ** | Cloud Run | コンテナベースアプリケーション実行 |
| **CI/CD** | Cloud Build | 自動ビルド・デプロイ |
| **バージョン管理** | GitHub | ソースコード管理 |
| **AIオーケストレーション** | LangChain | 複数AIモデルの連携制御 |
| **テキスト生成** | Gemini 2.0 Flash Lite | 記事・タイトル・名言生成 |
| **画像生成** | Imagen 3.0 | 4コマイラスト・挿絵生成 |
| **情報収集** | Custom Search API | Web情報検索・収集 |

## 🚀 クイックスタート

### 前提条件

- [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)
- Google Cloud Platform アカウント
- Google Custom Search API アクセス

### セットアップ

1. **リポジトリクローン**
   ```bash
   git clone <repository-url>
   cd ai-agent-hackason-app
   ```

2. **環境変数設定**
   
   `.env.dev` ファイルを作成：
   ```bash
   # Google Cloud Platform
   GCP_PROJECT_ID=your-gcp-project-id
   GCP_LOCATION=asia-northeast1
   
   # Vertex AI Models
   VERTEX_AI_MODEL_NAME=gemini-2.0-flash-lite-001
   IMAGE_MODEL_NAME=imagen-3.0-generate-002
   
   # Google Search API
   GOOGLE_API_KEY=your-google-api-key
   GOOGLE_CSE_ID=your-custom-search-engine-id
   ```

3. **GCP認証**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project your-gcp-project-id
   ```

4. **アプリケーション起動**
   ```bash
   # 開発環境で起動
   make dev-up
   
   # ブラウザで http://localhost:8501 にアクセス
   ```

## 🎮 使用方法

### 基本的な体験フロー

1. **🗺️ 地域選択**
   - 日本地図で都道府県をクリック
   - または位置情報ボタンで現在地を自動選択

2. **🤖 記事生成**
   - 「タイトルと記事を生成する」ボタンをクリック
   - AIの思考プロセスをリアルタイムで確認

3. **📖 物語体験**
   - 生成された哲学的記事を閲覧
   - 4コマイラストや画像で視覚的に楽しむ

### 生成されるコンテンツ

- **📝 メインタイトル**: 20-30文字の哲学的なタイトル
- **📋 サブタイトル**: 5つのテーマ別セクション
- **🎨 4コマイラスト**: 地域の魅力を表現した起承転結の物語
- **🖼️ セクション画像**: 各テーマに最適化された個別画像
- **📚 記事本文**: 800-1000文字の示唆に富んだ文章
- **💭 哲学的名言**: その土地にちなんだオリジナル格言

## 🔧 開発者向け情報

### プロジェクト構成

```
ai-agent-hackason-app/
├── app/
│   ├── components/              # UIコンポーネント
│   │   ├── article_html_section.py     # 記事生成・表示
│   │   ├── map_section.py              # 地図セクション
│   │   └── sidebar_controls.py         # サイドバー制御
│   ├── config/                  # 設定管理
│   ├── prompts/                 # AIプロンプト定義
│   ├── utils/                   # 核心ロジック
│   │   ├── agent_generate_article.py   # 記事生成ワークフロー
│   │   ├── generate_titles.py          # タイトル生成
│   │   ├── generate_four_images.py     # 4コマ画像生成
│   │   └── workflow_steps.py           # ワークフローステップ
│   └── main.py                  # メインアプリケーション
├── docker-compose.dev.yml       # 開発環境Docker設定
├── docker-compose.prod.yml      # 本番環境Docker設定
└── Makefile                     # 開発タスク定義
```

### 開発コマンド

```bash
# 開発環境の起動・停止
make dev-up
make dev-down

# 本番環境の起動・停止  
make prod-up
make prod-down

# ローカル開発
poetry install
poetry run streamlit run app/main.py
```

### 技術的なこだわり

#### 🤝 チーム開発基盤
- **Docker統一環境**: 開発環境の完全統一
- **Git-flow戦略**: 効率的なブランチ管理
- **タスク分割**: 詳細なスケジュール計画

#### 🧠 自律的AIエージェント設計
- **LangChain活用**: ReAct思考プロセスによる自律実行
- **モジュール化**: 検索→分析→執筆→統合の独立したステップ
- **柔軟性確保**: 状況に応じた動的な処理フロー

#### 🎯 プロンプトエンジニアリング
- **ペルソナ設定**: AIに「親しみやすい賢人」の役割を付与
- **動的プロンプト**: 地域特性×記事テーマ×アートスタイルの組み合わせ
- **品質管理**: 一貫した高品質なアウトプットの実現

## 🚨 トラブルシューティング

### よくある問題と解決法

**🔐 認証エラー**
```bash
# GCP再認証
gcloud auth application-default login
```

**🖼️ 画像生成失敗**
- Vertex AI APIの有効化確認
- プロジェクトでImagen APIの利用可否確認

**🗺️ 地図表示問題**
- ネットワーク接続確認
- ブラウザJavaScript有効化確認

**🐳 Docker関連**
```bash
# キャッシュクリア再ビルド
docker-compose down
docker-compose build --no-cache
docker-compose up
```

## 🔮 今後の展望

### 近期アップデート
- **💬 対話機能**: 生成記事を基にした深掘り質問対応
- **🗣️ 地域住民連携**: 住民からの知識獲得システム
- **🌍 多言語対応**: インバウンド観光客向け記事生成

### 長期ビジョン
- **🤝 地域コミュニティ統合**: 住民と訪問者を繋ぐプラットフォーム
- **📊 データ分析**: 地域魅力度の定量化と改善提案
- **🎯 パーソナライゼーション**: 個人の関心に最適化された記事生成

## 🏆 ハッカソン成果

**第2回 AI Agent Hackathon with Google Cloud**にて、地方過疎化という社会課題に対するAIソリューションとして開発されました。

### 解決したい課題
- 地方の魅力が十分に伝わっていない
- 移住・観光の意思決定に必要な「暮らしの質感」情報の不足
- 画一的な地域紹介による差別化の困難

### 提供する価値
- **発見**: まだ知らない地域の魅力との出会い
- **理解**: 統計では見えない文化的・感情的な地域理解
- **動機**: 実際に訪問・移住したくなる具体的なきっかけ

## 🤝 コントリビューション

このプロジェクトへの貢献を歓迎します！

1. リポジトリをフォーク
2. フィーチャーブランチ作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエスト作成

## 📄 ライセンス

このプロジェクトは [MIT License](LICENSE) の下で公開されています。

## 📞 お問い合わせ

プロジェクトに関するご質問や提案は、[Issues](../../issues) または下記までお気軽にどうぞ。

---

**開発チーム**: AI Agent Hackathon Team  
**イベント**: [第2回 AI Agent Hackathon with Google Cloud](https://zenn.dev/hackathons/google-cloud-japan-ai-hackathon-vol2)  
**バージョン**: 1.0.0

---

> 地図を開き、あなただけの賢者との対話を始めましょう。  
> さあ、新たな日本の物語の扉を開く時です。 🌸
