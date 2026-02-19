# アーキテクチャ

## システム概要

LLM Ops Platform は RAG・Agent・評価・監視・セキュリティを統合したエンタープライズ向け LLM プラットフォーム。
ローカル VRAM 8GB 環境で動作し、Embedding はローカル vLLM サーバー、LLM 推論は Gemini API 経由（デフォルト）。

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
│  PostgreSQL 16 + pgvector  │  Redis  │  vLLM Embedding     │
└─────────────────────────────────────────────────────────────┘
```

## コンテナ構成 (docker-compose)

| サービス | イメージ | 役割 | ポート |
|---------|---------|------|--------|
| app | Dockerfile (FastAPI) | メイン API | 8000 |
| db | pgvector/pgvector:pg16 | PostgreSQL 16 + pgvector | 5432 |
| redis | redis:7-alpine | キャッシュ・セッション・レート制限 | 6379 |
| embedding | vllm/vllm-openai:latest | vLLM Embedding サーバー (cl-nagoya/ruri-v3-310m) | 8001 |

**ボリューム**: `pgdata` (PostgreSQL), `redisdata` (Redis), `hf_cache` (Hugging Face モデル)

**起動順序**: db + redis + embedding が並列起動 → db/redis の healthcheck 完了後に app が起動

## 技術スタック

| カテゴリ | 技術 |
|---------|------|
| 言語 | Python 3.12 |
| フレームワーク | FastAPI, Pydantic v2 |
| ORM | SQLModel (SQLAlchemy + Pydantic 統合) |
| LLM API | Gemini API (デフォルト: gemini-2.5-flash-lite 無料枠) — OpenRouter, OpenAI, Anthropic に切替可能 |
| Embedding | cl-nagoya/ruri-v3-310m (vLLM ローカルサーバー, 1024 次元) |
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
  └── 1:N audit_logs (id, action, resource_type, resource_id, details)
```

### Embedding

- vLLM サーバー (`embedding` コンテナ, ポート 8001)
- モデル: `cl-nagoya/ruri-v3-310m` (日本語特化, 310M パラメータ, VRAM ~1GB)
- 出力次元: 1024
- OpenAI 互換 API (`/v1/embeddings`) で httpx AsyncClient からアクセス

### セマンティックキャッシュ

- Redis に query embedding + response を保存
- cosine similarity >= 0.95 のクエリはキャッシュヒット
- TTL 3600 秒 (設定可能)
- キャッシュ障害時はスルー (graceful degradation)

## トレードオフ方針

### 品質 vs コスト
- デフォルトは Gemini 無料枠 (gemini-2.5-flash-lite)。品質要件に応じてプロバイダ/モデルを切替
- セマンティックキャッシュで同一・類似クエリの API 呼び出しを削減
- Embedding はローカル vLLM 実行で API 費用ゼロ

### 品質 vs レイテンシ
- ストリーミングレスポンスで TTFT (Time to First Token) を最適化
- 再ランキングは未実装 (将来の拡張ポイントとして設計上確保)

### 安全性 vs ユーザビリティ
- ガードレールは段階的: allow → warn → block。過剰なブロックを避ける
- PII マスキングはログ・LLM API には必須、ユーザー表示は設定で切替可能
