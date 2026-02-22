# LLM Ops Platform

本番運用可能な LLM アプリケーション基盤。RAG・Agent・評価・監視・セキュリティを統合したエンタープライズ向け LLM プラットフォーム。

## 特徴

- **LLM マルチプロバイダ**: Gemini (デフォルト)、OpenAI、Anthropic、OpenRouter を切替可能
- **RAG パイプライン**: ドキュメント登録 → チャンキング → Embedding → pgvector 検索
- **Agent (ReAct)**: 思考 → ツール選択 → 実行 → 観察のループで自律的にタスク解決
- **評価エンジン**: Faithfulness、Answer Relevancy、Context Precision 等のメトリクス
- **Web UI**: htmx + DaisyUI による Web フロントエンド (SSE ストリーミング対応)
- **セキュリティ**: JWT 認証、RBAC 認可、CSRF 保護、PII マスキング、プロンプトインジェクション検出
- **監視**: structlog 構造化ログ、Prometheus メトリクス、コスト追跡
- **セマンティックキャッシュ**: Redis による類似クエリのキャッシュ (cosine similarity >= 0.95)

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway (FastAPI)                  │
│                  認証 / レート制限 / PIIマスキング            │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│   RAG    │  Agent   │  評価    │   監視   │  セキュリティ    │
│ Pipeline │ Runtime  │ Engine   │ Metrics  │  Guardian       │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│              LLM Router (プロバイダ切替・フォールバック)       │
│    Gemini API (デフォルト) / OpenRouter / OpenAI / Anthropic  │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL 16 + pgvector  │  Redis  │  Gemini Embedding   │
└─────────────────────────────────────────────────────────────┘
```

## 技術スタック

| カテゴリ | 技術 |
|---------|------|
| 言語 | Python 3.12 |
| フレームワーク | FastAPI, Pydantic v2 |
| ORM | SQLModel (SQLAlchemy + Pydantic 統合) |
| LLM API | Gemini API (デフォルト) / OpenRouter / OpenAI / Anthropic |
| Embedding | Gemini API `gemini-embedding-001` (デフォルト) / vLLM ローカル |
| Vector DB | pgvector (PostgreSQL 16 拡張) |
| キャッシュ | Redis 7 |
| Web UI | htmx + DaisyUI (CDN 配信、ビルドステップ不要) |
| パッケージ管理 | uv |
| テスト | pytest, pytest-asyncio |
| Linter/Formatter | ruff, mypy |

## クイックスタート

### 前提条件

- Docker / Docker Compose
- [uv](https://docs.astral.sh/uv/) (Python パッケージマネージャ)
- Gemini API キー ([Google AI Studio](https://aistudio.google.com/) で取得)

### セットアップ

```bash
# リポジトリのクローン
git clone https://github.com/dakanat/llm-ops-platform.git
cd llm-ops-platform

# 環境変数の設定
cp .env.example .env
# .env を編集: GEMINI_API_KEY, JWT_SECRET_KEY, CSRF_SECRET_KEY を設定

# コンテナの起動
make up

# DB マイグレーション
make migrate

# 初期データ投入 (admin/user/viewer アカウント)
make seed

# ヘルスチェック
curl http://localhost:8000/health

# Web UI へアクセス
# http://localhost:8000/web/login
```

### ローカル開発 (コンテナなし)

```bash
# 依存関係のインストール
uv sync

# DB と Redis は Docker で起動
docker compose up -d db redis

# 環境変数を設定
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/llm_platform
export REDIS_URL=redis://localhost:6379/0

# アプリケーション起動
uv run uvicorn src.main:app --reload --port 8000
```

## 設定

主要な環境変数 (`.env.example` 参照):

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `LLM_PROVIDER` | `gemini` | LLM プロバイダ (gemini / openrouter / openai / anthropic) |
| `LLM_MODEL` | `gemini-2.5-flash-lite` | 使用モデル名 |
| `GEMINI_API_KEY` | — | Gemini API キー |
| `EMBEDDING_PROVIDER` | `gemini` | Embedding プロバイダ (gemini / local) |
| `DATABASE_URL` | `postgresql+asyncpg://...` | DB 接続 URL |
| `REDIS_URL` | `redis://redis:6379/0` | Redis 接続 URL |
| `JWT_SECRET_KEY` | — | JWT 署名シークレット |
| `CSRF_SECRET_KEY` | — | CSRF トークン署名シークレット |

## API エンドポイント

### REST API

| メソッド | パス | 説明 |
|---------|------|------|
| POST | `/auth/token` | JWT トークン取得 |
| POST | `/chat` | チャット (ストリーミング対応) |
| POST | `/rag/query` | RAG 検索 |
| POST | `/rag/index` | ドキュメント登録 |
| POST | `/agent/run` | Agent 実行 |
| POST | `/eval/run` | 評価実行 |
| GET | `/eval/datasets` | データセット一覧 |
| GET | `/admin/metrics` | メトリクス |
| GET | `/health` | ヘルスチェック |

### Web UI

| パス | 説明 |
|------|------|
| `/web/login` | ログイン |
| `/web/chat` | チャット (SSE ストリーミング) |
| `/web/rag` | RAG 検索・ドキュメント管理 |
| `/web/eval` | 評価ダッシュボード |
| `/web/eval/datasets` | データセット管理 |
| `/web/admin` | 管理ダッシュボード |

## 開発

### コマンド一覧

```bash
make up              # コンテナ起動
make down            # コンテナ停止
make logs            # ログ表示
make lint            # ruff check
make format          # ruff format
make typecheck       # mypy
make test            # ユニットテスト
make test-integration  # 統合テスト
make test-all        # 全テスト
make test-no-llm     # LLM API 不要なテストのみ
make coverage        # カバレッジレポート
make migrate         # DB マイグレーション
make seed            # 初期データ投入
```

### TDD

本プロジェクトは TDD (テスト駆動開発) を基本方針とする。Red → Green → Refactor サイクルに従い、テストファーストで実装を進める。詳細は [CLAUDE.md](CLAUDE.md) を参照。

## プロジェクト構成

```
llm-ops-platform/
├── src/
│   ├── main.py              # FastAPI エントリポイント
│   ├── api/                 # REST API (routes, middleware, dependencies)
│   ├── web/                 # Web フロントエンド (htmx + DaisyUI)
│   ├── rag/                 # RAG パイプライン
│   ├── agent/               # Agent Runtime (ReAct)
│   ├── llm/                 # LLM プロバイダ・キャッシュ・PII保護
│   ├── eval/                # 評価エンジン
│   ├── monitoring/          # ログ・メトリクス・コスト追跡
│   ├── security/            # PII検出・インジェクション対策・RBAC
│   └── db/                  # SQLModel モデル・セッション管理
├── tests/
│   ├── unit/                # ユニットテスト
│   └── integration/         # 統合テスト
├── docs/                    # 設計ドキュメント
├── scripts/                 # 運用スクリプト
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## ドキュメント

| ファイル | 内容 |
|---------|------|
| [docs/architecture.md](docs/architecture.md) | アーキテクチャ、技術スタック、ディレクトリ構成、設計方針 |
| [docs/rag-design.md](docs/rag-design.md) | RAG パイプライン設計、各コンポーネント仕様 |
| [docs/agent-design.md](docs/agent-design.md) | Agent 設計、ReAct ループ、ツール一覧 |
| [docs/eval-strategy.md](docs/eval-strategy.md) | 評価メトリクス、回帰テスト、合成データ生成 |
| [docs/security.md](docs/security.md) | 認証認可、CSRF 保護、PII 保護、インジェクション対策 |
| [docs/runbook.md](docs/runbook.md) | 環境構築、日常オペレーション、トラブルシューティング |
| [CLAUDE.md](CLAUDE.md) | 開発ルール、コーディング規約、コマンド一覧 |

## ライセンス

[MIT License](LICENSE)
