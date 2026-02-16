# CLAUDE.md

## プロジェクト概要

**LLM Ops Platform** — 本番運用可能なLLMアプリケーション基盤

RAG・Agent・評価・監視・セキュリティを統合した、エンタープライズ向けLLMプラットフォーム。
PoCではなく本番運用を前提とし、品質・コスト・安全性のトレードオフを考慮した設計。

**ローカル実行前提**: VRAM 8GB環境で動作。Embeddingはローカルvllmサーバー(別コンテナ)、LLM推論はOpenRouter API経由。

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
│         OpenRouter API (デフォルト) / OpenAI / Anthropic     │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL 16 + pgvector  │  Redis  │  vLLM Embedding     │
└─────────────────────────────────────────────────────────────┘
```

### コンテナ構成 (docker-compose)

| サービス | 役割 | ポート |
|---------|------|--------|
| app | FastAPI メインAPI | 8000 |
| db | PostgreSQL 16 + pgvector | 5432 |
| redis | キャッシュ・セッション | 6379 |
| embedding | vLLM Embeddingサーバー (cl-nagoya/ruri-v3-310m) | 8001 |

## ディレクトリ構成

```
llm-ops-platform/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── alembic.ini
├── .env.example
├── .github/
│   └── workflows/
│       ├── ci.yml                    # lint, test, type check
│       └── eval-regression.yml       # LLM評価回帰テスト
├── src/
│   ├── __init__.py
│   ├── main.py                       # FastAPI エントリポイント
│   ├── config.py                     # 設定管理 (pydantic-settings)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── chat.py               # チャットエンドポイント (streaming対応)
│   │   │   ├── rag.py                # RAG検索・回答エンドポイント
│   │   │   ├── agent.py              # Agent実行エンドポイント
│   │   │   ├── eval.py               # 評価実行エンドポイント
│   │   │   └── admin.py              # 管理・監視エンドポイント
│   │   ├── middleware/
│   │   │   ├── auth.py               # JWT認証・権限管理
│   │   │   ├── rate_limit.py         # レート制限 (Redis Token Bucket)
│   │   │   ├── pii_filter.py         # PIIマスキング (入出力)
│   │   │   └── request_logger.py     # 構造化リクエストログ
│   │   └── dependencies.py           # FastAPI DI
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── pipeline.py               # RAGパイプライン統合 (検索→生成)
│   │   ├── chunker.py                # チャンク戦略 (fixed / recursive)
│   │   ├── embedder.py               # Embedding生成 (vLLMローカルサーバー呼出)
│   │   ├── retriever.py              # pgvector ベクトル検索
│   │   ├── generator.py              # 回答生成 + ソース引用
│   │   ├── preprocessor.py           # ドキュメント前処理 (テキスト正規化)
│   │   └── index_manager.py          # インデックス管理・ドキュメント登録
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── runtime.py                # Agent実行エンジン (ReAct loop)
│   │   ├── state.py                  # 状態管理 (会話履歴・タスク状態)
│   │   ├── tools/
│   │   │   ├── base.py               # Tool基底クラス (Protocol)
│   │   │   ├── search.py             # RAG検索ツール
│   │   │   ├── database.py           # DBクエリツール
│   │   │   ├── calculator.py         # 計算ツール
│   │   │   └── registry.py           # ツールレジストリ (動的登録)
│   │   ├── guardrails.py             # 入出力ガードレール
│   │   ├── planner.py                # タスク計画・分解
│   │   └── fallback.py               # フォールバック戦略
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── router.py                 # LLMルーター (プロバイダ選択・フォールバック)
│   │   ├── providers/
│   │   │   ├── base.py               # LLMProvider Protocol 定義
│   │   │   ├── openrouter.py         # OpenRouter API (デフォルト)
│   │   │   ├── openai_provider.py    # OpenAI / Azure OpenAI
│   │   │   └── anthropic_provider.py # Anthropic Claude
│   │   ├── prompt_manager.py         # プロンプト管理・バージョニング
│   │   ├── token_counter.py          # トークン計算・コスト推定
│   │   └── cache.py                  # セマンティックキャッシュ (Redis)
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── runner.py                 # 評価実行エンジン
│   │   ├── metrics/
│   │   │   ├── faithfulness.py       # 忠実性 (Hallucination検出)
│   │   │   ├── relevance.py          # 関連性スコア
│   │   │   └── latency.py            # レイテンシ計測
│   │   ├── datasets.py               # 評価データセット管理
│   │   ├── synthetic_data.py         # 合成データ生成 (テスト用QA自動生成)
│   │   └── regression.py             # 回帰テスト (ベースライン比較)
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── logger.py                 # 構造化ログ (structlog, JSON)
│   │   ├── metrics.py                # Prometheus メトリクス
│   │   ├── cost_tracker.py           # LLMコスト追跡・アラート
│   │   └── error_analyzer.py         # エラー分類・分析
│   ├── security/
│   │   ├── __init__.py
│   │   ├── pii_detector.py           # PII検出・マスキング (正規表現, 日本語対応)
│   │   ├── prompt_injection.py       # プロンプトインジェクション検出
│   │   ├── permission.py             # RBAC権限管理
│   │   └── audit_log.py              # 監査ログ
│   └── db/
│       ├── __init__.py
│       ├── models.py                 # SQLModel モデル定義
│       ├── vector_store.py           # pgvector 検索操作
│       ├── session.py                # async DB接続管理
│       └── migrations/               # Alembic マイグレーション
│           ├── env.py
│           └── versions/
├── tests/
│   ├── conftest.py                   # 共通fixture (mock LLM, test DB)
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_retriever.py
│   │   ├── test_guardrails.py
│   │   ├── test_pii_detector.py
│   │   └── test_prompt_injection.py
│   ├── integration/
│   │   ├── test_rag_pipeline.py
│   │   └── test_agent_runtime.py
│   └── eval/
│       ├── test_sets/                # 評価用ゴールドデータ (JSON)
│       └── regression_baseline.json  # 回帰テスト基準値
├── docs/
│   ├── architecture.md
│   ├── rag-design.md
│   ├── agent-design.md
│   ├── eval-strategy.md
│   ├── security.md
│   └── runbook.md
└── scripts/
    ├── seed_data.py                  # 初期データ投入
    ├── run_eval.py                   # 評価一括実行
    └── cost_report.py                # コストレポート生成
```

## 技術スタック

| カテゴリ | 技術 |
|---------|------|
| 言語 | Python 3.12 |
| フレームワーク | FastAPI, Pydantic v2 |
| ORM | SQLModel (SQLAlchemy + Pydantic 統合) |
| LLM API | OpenRouter API (デフォルト: gpt-oss-120b free) — OpenAI, Anthropic に切替可能 |
| Embedding | cl-nagoya/ruri-v3-310m (vLLMローカルサーバー, 別コンテナ, ポート8001) |
| Vector DB | pgvector (PostgreSQL 16 拡張) |
| キャッシュ | Redis 7 |
| DB | PostgreSQL 16, Alembic (migration) |
| PII検出 | 正規表現ベース (日本語対応) |
| 監視 | Prometheus client, structlog (JSON) |
| CI/CD | GitHub Actions |
| コンテナ | Docker, docker-compose |
| テスト | pytest, pytest-asyncio |
| Linter/Formatter | ruff, mypy |

## 設計方針

### LLMプロバイダ切替設計
- `src/llm/providers/base.py` に `LLMProvider` Protocol を定義
- 各プロバイダ (OpenRouter, OpenAI, Anthropic) が Protocol を実装
- すべてのプロバイダは OpenAI互換の chat completions インターフェースに統一
- `src/llm/router.py` が環境変数 `LLM_PROVIDER` に基づきプロバイダをルーティング
- 選択肢: `openrouter` (default) / `openai` / `anthropic`
- 新規プロバイダ追加 = Protocol 実装 + router登録のみ。既存コードの変更不要

### Embedding設計
- vLLMサーバーが `embedding` コンテナで起動 (ポート8001)
- モデル: `cl-nagoya/ruri-v3-310m` (日本語特化, 310Mパラメータ, VRAM ~1GB)
- 出力次元: 1024
- OpenAI互換API (`/v1/embeddings`) でアクセス → httpx AsyncClient で非同期リクエスト
- pgvector の Vector(1024) カラムに格納

### DB設計 (SQLModel)
- SQLModel を使用 (SQLAlchemy + Pydantic の統合)
- テーブルモデルは `table=True` で定義
- API入出力スキーマは `table=False` の SQLModel or 素の Pydantic BaseModel
- async セッション管理: `asyncpg` ドライバ + `async_sessionmaker`
- マイグレーション: Alembic

### RAGパイプライン
1. **前処理**: テキスト正規化 (Unicode正規化, 空白除去)
2. **チャンキング**: recursive character splitter (chunk_size=512, overlap=64)
3. **Embedding**: vLLMローカルサーバー経由で ruri-v3-310m
4. **検索**: pgvector cosine similarity search (top_k=5)
5. **生成**: LLMプロバイダ経由で回答生成 + ソースチャンク引用
- 再ランキングは現時点では不要。将来追加する場合は retriever と generator の間に挿入する設計

### Agent設計
- ReActパターン (Reasoning + Acting) のループ実行
- Tool は Protocol ベースで定義、RegistryパターンでRuntime に動的登録
- 状態管理: 会話履歴 + 中間思考ステップを保持
- ガードレール: 入力バリデーション (トークン上限, 禁止パターン) + 出力チェック
- フォールバック: ツール実行失敗時のリトライ → 直接回答へ degradation
- 最大ステップ数制限によるループ防止

## 開発ルール

### コーディング規約
- **型ヒント必須**: すべての関数に型アノテーションを付与。mypyでチェック
- **SQLModel**: テーブル定義は `SQLModel, table=True`。APIスキーマは `SQLModel` or `BaseModel`
- **async/await**: I/O処理はすべて非同期。`httpx.AsyncClient`, `asyncpg`
- **依存性注入**: FastAPI `Depends` によるDI。テスタビリティを確保
- **エラーハンドリング**: カスタム例外クラス → FastAPI exception_handler でHTTPレスポンスに変換
- **ログ**: `structlog` で構造化JSON出力。リクエストID をコンテキストに含める
- **docstring**: 各モジュール・クラス・publicメソッドに記載

### コミット・ブランチ
- **ブランチ戦略**: `main` → `develop` → `feature/xxx`, `fix/xxx`
- **コミットメッセージ**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)

### テスト駆動開発 (TDD)

本プロジェクトはTDDを基本方針とする。すべての機能実装は以下のサイクルに従う。

#### Red → Green → Refactor サイクル
1. **Red**: 実装したい振る舞いを記述したテストを先に書く。テストが失敗することを確認する
2. **Green**: テストを通す最小限の実装コードを書く。過剰な設計・最適化はしない
3. **Refactor**: テストが通った状態を維持しながらコードを整理する (重複排除, 命名改善, 構造整理)

#### TDDの実践ルール
- **テストファースト厳守**: 実装コードよりも先にテストを書く。テストが存在しないコードはマージしない
- **1テスト1アサーション**: 各テストケースは1つの振る舞いを検証する。複数の関心事を混ぜない
- **テスト名は仕様**: `test_chunker_splits_text_at_512_chars` のように、何を検証しているかをテスト名で明示する
- **小さなステップ**: 一度に大きな機能を作らない。テスト1つ → 実装 → テスト1つ → 実装のサイクルを細かく回す
- **外部依存のモック**: LLM API、Embedding API、DB等の外部依存はモック/スタブで分離。ユニットテストは高速に実行できること

#### テストの分類
- **ユニットテスト** (`tests/unit/`): 各モジュールの純粋なロジック。外部依存はすべてモック。高速実行 (数秒以内)
- **統合テスト** (`tests/integration/`): Docker Compose でDB等を起動し、コンポーネント間の結合を検証
- **評価テスト** (`tests/eval/`): `@pytest.mark.llm` マーカー。LLM API呼び出しを含む品質評価。`pytest -m "not llm"` でスキップ可能

#### カバレッジ
- 80%以上を目標
- カバレッジが低い場合はテストの追加を優先する (機能追加よりテスト追加が先)

### セキュリティ
- **シークレット管理**: 環境変数のみ。コードにハードコードしない
- **PII**: 入出力の両方でPII検出・マスキング。ログにもPIIを含めない
- **プロンプトインジェクション**: 入力サニタイズ + 出力バリデーションの二重チェック
- **権限**: RBAC (admin / user / viewer)。JWT + スコープベース

## 実装の優先順位 (TDDサイクル)

各ステップは「テストを先に書く → 実装 → リファクタ」の順で進める。

### Phase 1: コア基盤 (MVP)

**Step 1-1: プロジェクト骨格 + config**
```
テスト: test_config.py — Settings がデフォルト値で生成できること、環境変数で上書きできること
実装:  config.py, main.py (FastAPI app + /health エンドポイント)
```

**Step 1-2: SQLModel モデル + DB接続**
```
テスト: test_models.py — 各モデルがインスタンス化できること、フィールドのバリデーション
実装:  db/models.py, db/session.py, Alembic セットアップ
```

**Step 1-3: LLMプロバイダ抽象化**
```
テスト: test_llm_provider.py — Protocol の型チェック、OpenRouter がレスポンスを返すこと (モック)
       test_llm_router.py — 設定に応じて正しいプロバイダが選択されること
実装:  llm/providers/base.py (Protocol), llm/providers/openrouter.py, llm/router.py
```

**Step 1-4: Embedding クライアント**
```
テスト: test_embedder.py — テキストを渡すとベクトル (list[float]) が返ること (モック)
       次元数が1024であること、空文字でエラーになること
実装:  rag/embedder.py (vLLM ローカルサーバー呼出)
```

**Step 1-5: チャンキング**
```
テスト: test_chunker.py — 指定サイズで分割されること、オーバーラップが正しいこと
       空文字、chunk_size以下の短文、日本語テキストの境界処理
実装:  rag/chunker.py
```

**Step 1-6: ベクトル検索 (pgvector)**
```
テスト: test_vector_store.py — embedding保存→類似検索で正しい結果が返ること (統合テスト)
       top_k パラメータが反映されること
実装:  db/vector_store.py
```

**Step 1-7: RAGパイプライン統合**
```
テスト: test_rag_pipeline.py — ドキュメント登録→クエリ→回答生成の一連フロー (モック LLM + モック Embedding)
       ソース引用が含まれること
実装:  rag/pipeline.py, rag/retriever.py, rag/generator.py, rag/preprocessor.py, rag/index_manager.py
```

**Step 1-8: チャットエンドポイント**
```
テスト: test_chat_route.py — POST /chat が200を返すこと、streaming レスポンスが正しい形式であること
       不正なリクエストで422が返ること
実装:  api/routes/chat.py, api/routes/rag.py
```

**Step 1-9: Docker Compose + CI**
```
テスト: CI上で make test が通ること
実装:  Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

**Step 1-10: 構造化ログ**
```
テスト: test_logger.py — ログ出力がJSON形式であること、リクエストIDが含まれること
実装:  monitoring/logger.py, api/middleware/request_logger.py
```

### Phase 2: 品質・評価・監視

**Step 2-1: 評価メトリクス**
```
テスト: test_faithfulness.py — 忠実性スコアが0-1範囲で返ること、明らかなHallucinationを検出すること
       test_relevance.py — 関連性スコアの正常系・異常系
       test_latency.py — レイテンシ計測が正の値を返すこと
実装:  eval/metrics/faithfulness.py, eval/metrics/relevance.py, eval/metrics/latency.py
```

**Step 2-2: 評価実行エンジン + 回帰テスト**
```
テスト: test_eval_runner.py — データセットを渡して評価結果が返ること
       test_regression.py — ベースライン比較でpass/failが正しく判定されること
実装:  eval/runner.py, eval/regression.py, eval/datasets.py
```

**Step 2-3: プロンプト管理**
```
テスト: test_prompt_manager.py — テンプレート取得、変数埋め込み、バージョン管理
実装:  llm/prompt_manager.py
```

**Step 2-4: セマンティックキャッシュ**
```
テスト: test_cache.py — キャッシュヒット/ミスの判定、TTLによる期限切れ
実装:  llm/cache.py
```

**Step 2-5: 監視・コスト追跡**
```
テスト: test_metrics.py — Prometheusメトリクスが正しく記録されること
       test_cost_tracker.py — トークン数からコスト計算、アラート閾値判定
実装:  monitoring/metrics.py, monitoring/cost_tracker.py, monitoring/error_analyzer.py
```

### Phase 3: Agent・セキュリティ

**Step 3-1: Agentツール基盤**
```
テスト: test_tool_base.py — Tool Protocol の型チェック
       test_registry.py — ツール登録、名前による取得、重複登録エラー
       test_calculator.py — 基本的な計算が正しい結果を返すこと
実装:  agent/tools/base.py, agent/tools/registry.py, agent/tools/calculator.py
```

**Step 3-2: Agent Runtime**
```
テスト: test_agent_runtime.py — ReActループが正しくツールを呼び出すこと (モック LLM + モック Tool)
       最大ステップ数で停止すること、ツール結果が次のステップに渡ること
実装:  agent/runtime.py, agent/state.py, agent/planner.py
```

**Step 3-3: ガードレール + フォールバック**
```
テスト: test_guardrails.py — 禁止パターン検出、トークン上限チェック、出力バリデーション
       test_fallback.py — ツール失敗時にリトライ→直接回答へ degradation
実装:  agent/guardrails.py, agent/fallback.py
```

**Step 3-4: PII検出**
```
テスト: test_pii_detector.py — 電話番号、メールアドレス、マイナンバー等の検出・マスキング
       日本語の住所、氏名パターンの検出
実装:  security/pii_detector.py, api/middleware/pii_filter.py
```

**Step 3-5: プロンプトインジェクション対策**
```
テスト: test_prompt_injection.py — 典型的なインジェクションパターンの検出
       正常な入力を誤検出しないこと
実装:  security/prompt_injection.py
```

**Step 3-6: 認証・権限**
```
テスト: test_auth.py — JWT生成・検証、期限切れトークンの拒否
       test_permission.py — RBAC (admin/user/viewer) の権限チェック
実装:  api/middleware/auth.py, security/permission.py, security/audit_log.py
```

### Phase 4: 応用

**Step 4-1: 合成データ生成**
```
テスト: test_synthetic_data.py — ドキュメントからQAペアが生成されること (モック LLM)
実装:  eval/synthetic_data.py
```

**Step 4-2: Agentツール拡充**
```
テスト: test_search_tool.py — RAG検索ツールが結果を返すこと
       test_database_tool.py — DBクエリツールが安全にクエリを実行すること (SQLインジェクション防止)
実装:  agent/tools/search.py, agent/tools/database.py
```

**Step 4-3: APIエンドポイント拡充**
```
テスト: test_agent_route.py — POST /agent が正しいレスポンスを返すこと
       test_eval_route.py — POST /eval/run が評価を開始すること
       test_admin_route.py — GET /admin/metrics がメトリクスを返すこと
実装:  api/routes/agent.py, api/routes/eval.py, api/routes/admin.py
```

**Step 4-4: レート制限**
```
テスト: test_rate_limit.py — 制限以下のリクエストは通過、超過で429が返ること
実装:  api/middleware/rate_limit.py
```

**Step 4-5: ドキュメント整備**
```
docs/architecture.md, docs/rag-design.md, docs/agent-design.md,
docs/eval-strategy.md, docs/security.md, docs/runbook.md
```

## よく使うコマンド

```bash
# 開発環境
make up                    # docker compose up -d --build
make down                  # docker compose down
make logs                  # docker compose logs -f app

# コード品質
make lint                  # ruff check src/ tests/
make format                # ruff format src/ tests/
make typecheck             # mypy src/
make test                  # pytest tests/unit/ -v
make test-integration      # pytest tests/integration/ -v
make test-all              # pytest tests/ -v
make test-no-llm           # pytest -m "not llm"
make coverage              # pytest --cov=src --cov-report=html

# DB
make migrate               # alembic upgrade head
make seed                  # python scripts/seed_data.py

# 評価
make eval                  # python scripts/run_eval.py
```

## 環境変数 (.env.example)

```env
# LLM Provider: openrouter | openai | anthropic
LLM_PROVIDER=openrouter
LLM_MODEL=gpt-oss-120b
OPENROUTER_API_KEY=your-key-here

# Optional: switch providers
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-xxx
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-xxx

# Embedding (local vLLM server)
EMBEDDING_BASE_URL=http://embedding:8001/v1
EMBEDDING_MODEL=cl-nagoya/ruri-v3-310m

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/llm_platform

# Redis
REDIS_URL=redis://redis:6379/0

# Security
JWT_SECRET_KEY=change-me-in-production
PII_DETECTION_ENABLED=true
PROMPT_INJECTION_DETECTION_ENABLED=true

# Monitoring
LOG_LEVEL=INFO
COST_ALERT_THRESHOLD_DAILY_USD=10
```

## トレードオフ方針

### 品質 vs コスト
- デフォルトはOpenRouter無料モデル (gpt-oss-120b)。品質要件に応じてプロバイダ/モデルを切替
- セマンティックキャッシュで同一・類似クエリのAPI呼び出しを削減
- Embedding はローカルvLLM実行でAPI費用ゼロ

### 品質 vs レイテンシ
- ストリーミングレスポンスでTTFT (Time to First Token) を最適化
- 再ランキングは未実装 (将来のパイプライン拡張ポイントとして設計上は確保)

### 安全性 vs ユーザビリティ
- ガードレールは段階的: Warning → Block。過剰なブロックを避ける
- PIIマスキングはログ・監視には必須、ユーザー表示は設定で切替可能

## 注意事項

- **TDD厳守**: 実装コードを書く前に必ずテストを書く。テストが失敗する (Red) ことを確認してから実装に入る
- **コミット粒度**: 「テスト追加 (Red)」「実装 (Green)」「リファクタ」を別コミットにすることを推奨。少なくとも Red→Green は1コミットにまとめてよいが、テストなしの実装コミットは禁止
- VRAM 8GB環境前提。ruri-v3-310m (~620MB VRAM) + vLLMオーバーヘッドで計約1-2GB使用
- LLM推論はOpenRouter API経由のためネットワーク接続が必要
- vLLM embeddingコンテナの初回起動時にHugging Faceからモデルをダウンロード (約600MB)
- `hf_cache` volumeによりモデルは永続化される (2回目以降は高速起動)
- テスト時、LLM API呼び出しを含むテストは `@pytest.mark.llm` でマーク。CIではスキップ可能
- プロンプトテンプレート変更時は回帰テスト (`make eval`) を実行してからマージ
