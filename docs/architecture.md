# アーキテクチャ

## システム概要

LLM Ops Platform は RAG・Agent・評価・監視・セキュリティを統合したエンタープライズ向け LLM プラットフォーム。
Embedding はデフォルトで Gemini API (`gemini-embedding-001`) を使用（GPU 不要）。ローカル vLLM (`cl-nagoya/ruri-v3-310m`) も代替として利用可能。LLM 推論は Gemini API 経由（デフォルト）。

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

## コンテナ構成 (docker-compose)

| サービス | イメージ | 役割 | ポート |
|---------|---------|------|--------|
| app | Dockerfile (FastAPI) | メイン API | 8000 |
| db | pgvector/pgvector:pg16 | PostgreSQL 16 + pgvector | 5432 |
| redis | redis:7-alpine | キャッシュ・セッション・レート制限 | 6379 |
| embedding | vllm/vllm-openai:latest | vLLM Embedding サーバー (cl-nagoya/ruri-v3-310m) — `local-embedding` profile | 8001 |

**ボリューム**: `pgdata` (PostgreSQL), `redisdata` (Redis), `hf_cache` (Hugging Face モデル)

**起動順序**: db + redis が並列起動 → healthcheck 完了後に app が起動。embedding サービスは `--profile local-embedding` 指定時のみ起動

## 技術スタック

| カテゴリ | 技術 |
|---------|------|
| 言語 | Python 3.12 |
| フレームワーク | FastAPI, Pydantic v2 |
| ORM | SQLModel (SQLAlchemy + Pydantic 統合) |
| LLM API | Gemini API (デフォルト: gemini-2.5-flash-lite 無料枠) — OpenRouter, OpenAI, Anthropic に切替可能 |
| Embedding | Gemini API `gemini-embedding-001` (デフォルト, 1024 次元) / cl-nagoya/ruri-v3-310m (vLLM ローカル代替) |
| Vector DB | pgvector (PostgreSQL 16 拡張, cosine distance) |
| キャッシュ | Redis 7 (セマンティックキャッシュ, レート制限) |
| PII 検出 | 正規表現ベース (日本語対応: メール, 電話, マイナンバー, クレカ, 住所) |
| 監視 | Prometheus client, structlog (JSON) |
| テスト | pytest, pytest-asyncio |
| Linter/Formatter | ruff, mypy |

## ディレクトリ構成

```
llm-ops-platform/
├── src/
│   ├── main.py                       # FastAPI エントリポイント (App Factory)
│   ├── config.py                     # pydantic-settings による設定管理
│   ├── api/
│   │   ├── routes/                   # chat, rag, agent, eval, admin
│   │   ├── middleware/               # auth (JWT), rate_limit, request_logger
│   │   └── dependencies.py          # FastAPI DI
│   ├── rag/                          # RAG パイプライン (→ docs/rag-design.md)
│   ├── agent/                        # Agent Runtime (→ docs/agent-design.md)
│   ├── llm/
│   │   ├── providers/                # base (Protocol), openrouter
│   │   ├── router.py                 # LLM プロバイダルーティング
│   │   ├── cache.py                  # セマンティックキャッシュ (Redis)
│   │   ├── prompt_manager.py         # プロンプト管理・バージョニング
│   │   └── pii_sanitizing_provider.py # PII マスキング Provider ラッパー
│   ├── eval/                         # 評価エンジン (→ docs/eval-strategy.md)
│   ├── monitoring/                   # structlog, Prometheus, コスト追跡
│   ├── security/                     # PII, インジェクション対策, RBAC (→ docs/security.md)
│   └── db/                           # SQLModel モデル, pgvector, セッション管理
├── tests/
│   ├── unit/                         # ユニットテスト (外部依存はモック)
│   └── integration/                  # 統合テスト (Docker DB 使用)
├── docs/                             # 設計ドキュメント
└── scripts/                          # 運用スクリプト
```

## 設計方針

### LLM プロバイダ抽象化

`LLMProvider` Protocol (`src/llm/providers/base.py`) を全プロバイダが実装する。

```
LLMProvider Protocol
  ├── complete(messages, model) → LLMResponse
  └── stream(messages, model)  → AsyncGenerator[LLMChunk]

LLMRouter (Factory)
  └── get_provider() → settings.llm_provider に応じたインスタンス生成
```

- 現在の実装: `GeminiProvider` (デフォルト), `OpenRouterProvider`, `OpenAIProvider`, `AnthropicProvider`
- 新規プロバイダ追加 = Protocol 実装 + Router 登録のみ

### PII 保護アーキテクチャ (信頼境界アプローチ)

ASGI ミドルウェアではなく、信頼境界ごとに PII をマスキングする設計。

```
ユーザー入力
  ↓
PIISanitizingProvider (LLM API 送信前にマスキング)
  ↓ [マスク済みテキスト]
LLM API (外部サービス — PII を受け取らない)
  ↓
pii_log_processor (ログ出力前にマスキング)
  ↓ [マスク済みログ]
structlog JSON 出力
```

### DB スキーマ

```
users (id, email, name, hashed_password, role, is_active)
  ├── 1:N documents (id, title, content, metadata, user_id)
  │     └── 1:N chunks (id, content, chunk_index, embedding[Vector(1024)])
  ├── 1:N conversations (id, title, user_id)
  │     └── 1:N messages (id, role, content, token_count)
  ├── 1:N audit_logs (id, action, resource_type, resource_id, details)
  └── 1:N eval_datasets (id, name, description, created_by)
        └── 1:N eval_examples (id, dataset_id, query, expected_answer)
```

### Embedding

`EmbedderProtocol` (`src/rag/embedder.py`) で統一インターフェースを提供し、2 つの実装を切替可能。

| プロバイダ | 実装 | モデル | 次元数 | 特徴 |
|-----------|------|--------|--------|------|
| `gemini` (デフォルト) | `GeminiEmbedder` | `gemini-embedding-001` | 1024 (`outputDimensionality`) | GPU 不要、無料枠 (100 RPM / 1,000 RPD) |
| `local` | `Embedder` | `cl-nagoya/ruri-v3-310m` | 1024 | 日本語特化、vLLM ローカルサーバー (VRAM ~1GB) |

- `EMBEDDING_PROVIDER` 環境変数で切替
- `_create_embedder()` ファクトリ関数 (`src/api/dependencies.py`) でインスタンス生成

### セマンティックキャッシュ

- Redis に query embedding + response を保存
- cosine similarity >= 0.95 のクエリはキャッシュヒット
- TTL 3600 秒 (設定可能)
- キャッシュ障害時はスルー (graceful degradation)

## トレードオフ方針

### 品質 vs コスト
- デフォルトは Gemini 無料枠 (gemini-2.5-flash-lite)。品質要件に応じてプロバイダ/モデルを切替
- セマンティックキャッシュで同一・類似クエリの API 呼び出しを削減
- Embedding はデフォルトで Gemini 無料枠 (gemini-embedding-001)。ローカル vLLM も選択可能

### 品質 vs レイテンシ
- ストリーミングレスポンスで TTFT (Time to First Token) を最適化
- 再ランキングは未実装 (将来の拡張ポイントとして設計上確保)

### 安全性 vs ユーザビリティ
- ガードレールは段階的: allow → warn → block。過剰なブロックを避ける
- PII マスキングはログ・LLM API には必須、ユーザー表示は設定で切替可能
