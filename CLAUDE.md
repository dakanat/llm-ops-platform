# CLAUDE.md

## プロジェクト概要

**LLM Ops Platform** — 本番運用可能なLLMアプリケーション基盤

RAG・Agent・評価・監視・セキュリティを統合した、エンタープライズ向けLLMプラットフォーム。
PoCではなく本番運用を前提とし、品質・コスト・安全性のトレードオフを考慮した設計。

**ローカル実行前提**: Embedding はデフォルトで Gemini API (`gemini-embedding-001`) を使用（GPU 不要）。ローカル vLLM サーバー (`cl-nagoya/ruri-v3-310m`) も代替として利用可能（VRAM 8GB 環境向け）。LLM 推論は Gemini API 経由（デフォルト）。

## 開発ルール

### コーディング規約
- **uv必須**: `python`, `pip`, `pip3` コマンドを直接使用しない。すべて `uv run` 経由で実行する（例: `uv run python`, `uv run pytest`）。パッケージ追加は `uv add`、削除は `uv remove` を使用
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
3. **Refactor**: テストが通った状態を維持しながらコードを整理する

#### TDDの実践ルール
- **テストファースト厳守**: 実装コードよりも先にテストを書く。テストが存在しないコードはマージしない
- **1テスト1アサーション**: 各テストケースは1つの振る舞いを検証する
- **テスト名は仕様**: `test_chunker_splits_text_at_512_chars` のように明示する
- **小さなステップ**: テスト1つ → 実装 → テスト1つ → 実装のサイクルを細かく回す
- **外部依存のモック**: LLM API、Embedding API、DB等はモック/スタブで分離

#### テストの分類
- **ユニットテスト** (`tests/unit/`): 外部依存はすべてモック。高速実行
- **統合テスト** (`tests/integration/`): Docker Compose でDB等を起動して結合検証
- **評価テスト**: `POST /eval/run` API 経由で実行。`@pytest.mark.llm` マーカー付きテストは `pytest -m "not llm"` でスキップ可能

#### カバレッジ
- 80%以上を目標。カバレッジが低い場合はテストの追加を優先する

### セキュリティ
- **シークレット管理**: 環境変数のみ。コードにハードコードしない
- **PII**: 信頼境界でマスキング (PIISanitizingProvider + pii_log_processor)。ログにPIIを含めない
- **プロンプトインジェクション**: パターンベース検出 (4カテゴリ: instruction_override, system_prompt_leak, role_manipulation, delimiter_injection)
- **権限**: RBAC (admin / user / viewer)。JWT + パーミッションベース

## よく使うコマンド

```bash
# 開発環境
make up                    # docker compose up -d --build
make down                  # docker compose down
make logs                  # docker compose logs -f app

# コード品質
make lint                  # ruff check src/ tests/
make format                # ruff format src/ tests/
make typecheck             # mypy src/ tests/
make test                  # pytest tests/unit/ -v
make test-integration      # pytest tests/integration/ -v
make test-all              # pytest tests/ -v
make test-no-llm           # pytest -m "not llm"
make coverage              # pytest --cov=src --cov-report=html

# DB
make migrate               # alembic upgrade head
make seed                  # python scripts/seed_data.py

```

## 環境変数 (.env.example)

```env
# LLM Provider: gemini | openrouter | openai | anthropic
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash-lite
GEMINI_API_KEY=your-key-here

# OpenRouter (alternative)
OPENROUTER_API_KEY=your-key-here

# Embedding: gemini | local
EMBEDDING_PROVIDER=gemini
EMBEDDING_GEMINI_MODEL=gemini-embedding-001

# Embedding (local vLLM server — used when EMBEDDING_PROVIDER=local)
EMBEDDING_BASE_URL=http://embedding:8001/v1
EMBEDDING_MODEL=cl-nagoya/ruri-v3-310m

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/llm_platform

# Redis
REDIS_URL=redis://redis:6379/0

# Security
JWT_SECRET_KEY=change-me-in-production
CSRF_SECRET_KEY=change-me-csrf-secret
PII_DETECTION_ENABLED=true
PROMPT_INJECTION_DETECTION_ENABLED=true

# Web Frontend
SESSION_COOKIE_SECURE=false  # Set true in production (HTTPS)

# Monitoring
LOG_LEVEL=INFO
COST_ALERT_THRESHOLD_DAILY_USD=10
```

## 注意事項

- **TDD厳守**: 実装コードを書く前に必ずテストを書く。テストなしの実装コミットは禁止
- **コミット粒度**: Red→Green は1コミットにまとめてよいが、テストなしの実装コミットは禁止
- Embedding はデフォルトで Gemini API (`gemini-embedding-001`, 無料枠 100 RPM / 1,000 RPD) を使用。GPU 不要
- ローカル vLLM Embedding を使う場合は `EMBEDDING_PROVIDER=local` に変更し `docker compose --profile local-embedding up` で起動。VRAM 8GB 環境前提 (~1-2GB 使用)
- LLM推論はGemini API経由（デフォルト）のためネットワーク接続が必要
- テスト時、LLM API呼び出しを含むテストは `@pytest.mark.llm` でマーク。CIではスキップ可能
- プロンプトテンプレート変更時は回帰テスト (`POST /eval/run` API) を実行してからマージ

## 詳細ドキュメント

- [docs/architecture.md](docs/architecture.md) — アーキテクチャ、技術スタック、ディレクトリ構成、設計方針
- [docs/rag-design.md](docs/rag-design.md) — RAGパイプライン設計、各コンポーネント仕様
- [docs/agent-design.md](docs/agent-design.md) — Agent設計、ReActループ、ツール一覧
- [docs/eval-strategy.md](docs/eval-strategy.md) — 評価メトリクス、回帰テスト、合成データ生成
- [docs/security.md](docs/security.md) — 認証認可、PII保護、インジェクション対策、監査ログ
- [docs/runbook.md](docs/runbook.md) — 環境構築、日常オペレーション、トラブルシューティング

## 残タスク

### Phase 5: 完成に向けて

**5-1: LLMマルチプロバイダ対応** ✅
- `src/llm/providers/gemini_provider.py` — Gemini OpenAI互換実装（デフォルト）
- `src/llm/providers/openai_provider.py` — OpenAI互換実装
- `src/llm/providers/anthropic_provider.py` — Anthropic Messages API実装
- `src/llm/router.py` — gemini / openrouter / openai / anthropic ルーティング

**5-2: トークンカウンター**
- `src/llm/token_counter.py` — tiktoken使用、コスト推定

**5-3: Alembic初期マイグレーション**
- `uv run alembic revision --autogenerate -m "initial schema"`

**5-4: 運用スクリプト**
- `scripts/seed_data.py` — 初期データ投入 (admin/user/viewer)
- `scripts/cost_report.py` — コストレポート生成

**5-5: 統合テスト**
- `tests/integration/test_rag_pipeline.py` — RAGフロー (Docker DB使用)
- `tests/integration/test_agent_runtime.py` — ReActループ
- `tests/integration/test_api_endpoints.py` — APIエンドポイント

**5-6: Webフロントエンド** ✅
- `src/web/` — htmx + DaisyUI によるWebフロントエンド（CDN配信、ビルドステップ不要）
- `src/web/routes/auth.py` — ログイン/ログアウト（HttpOnly Cookie JWT）
- `src/web/routes/chat.py` — チャットUI（SSEストリーミング対応）
- `src/web/routes/rag.py` — RAGクエリUI
- `src/web/routes/agent.py` — Agent実行UI（ステップ表示）
- `src/web/routes/eval.py` — 評価実行ダッシュボード
- `src/web/routes/eval_datasets.py` — データセット一覧/詳細/作成フォーム
- `src/web/routes/admin.py` — 管理ダッシュボード（コストレポート自動更新）
- `src/web/csrf.py` — CSRF double-submit cookie保護
- `src/web/dependencies.py` — Cookie認証DI（WebAuthRedirect例外）
- `src/web/templates.py` — Jinja2テンプレート設定
- テスト: `tests/unit/web/` (42テスト)

