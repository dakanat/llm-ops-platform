# 運用ランブック

## 環境構築

### 前提条件

- Docker / Docker Compose
- NVIDIA GPU (VRAM 8GB 以上) + nvidia-container-toolkit
- uv (Python パッケージマネージャ)

### 初期セットアップ

```bash
# 1. リポジトリのクローン
git clone <repo-url>
cd llm-ops-platform

# 2. 環境変数の設定
cp .env.example .env
# .env を編集: GEMINI_API_KEY, JWT_SECRET_KEY を設定

# 3. コンテナの起動
make up    # docker compose up -d --build

# 4. DB マイグレーション
make migrate    # uv run alembic upgrade head

# 5. 初期データ投入 (任意)
make seed       # uv run python scripts/seed_data.py

# 6. ヘルスチェック
curl http://localhost:8000/health
```

### ローカル開発 (コンテナなし)

```bash
# uv で依存関係をインストール
uv sync

# DB と Redis は Docker で起動
docker compose up -d db redis

# 環境変数を設定 (.env を読み込む)
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/llm_platform
export REDIS_URL=redis://localhost:6379/0

# アプリケーション起動
uv run uvicorn src.main:app --reload --port 8000
```

## 日常オペレーション

### 起動・停止

```bash
make up       # 全サービス起動
make down     # 全サービス停止
make logs     # app コンテナのログをフォロー
```

### コード品質チェック

```bash
make lint         # ruff check
make format       # ruff format
make typecheck    # mypy
make test         # ユニットテスト
make test-all     # 全テスト
make test-no-llm  # LLM API 不要なテストのみ
make coverage     # カバレッジレポート生成
```

### DB 操作

```bash
# マイグレーション実行
make migrate

# マイグレーション作成
uv run alembic revision --autogenerate -m "description"

# マイグレーション履歴
uv run alembic history

# ロールバック
uv run alembic downgrade -1
```

### 評価

評価は `POST /eval/run` API 経由で実行する。ユーザーが自身のデータセットを送信し、サーバー側でメトリクスを計算する。

## 環境変数

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `LLM_PROVIDER` | `gemini` | LLM プロバイダ (gemini / openrouter / openai / anthropic) |
| `LLM_MODEL` | `gemini-2.5-flash-lite` | 使用モデル名 |
| `GEMINI_API_KEY` | — | Gemini API キー |
| `OPENROUTER_API_KEY` | — | OpenRouter API キー |
| `EMBEDDING_BASE_URL` | `http://embedding:8001/v1` | Embedding サーバー URL |
| `EMBEDDING_MODEL` | `cl-nagoya/ruri-v3-310m` | Embedding モデル名 |
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/llm_platform` | DB 接続 URL |
| `REDIS_URL` | `redis://redis:6379/0` | Redis 接続 URL |
| `JWT_SECRET_KEY` | — | JWT 署名シークレット (**本番では必ず変更**) |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | JWT 有効期限 (分) |
| `PII_DETECTION_ENABLED` | `true` | PII 検出の有効/無効 |
| `PROMPT_INJECTION_DETECTION_ENABLED` | `true` | インジェクション検出の有効/無効 |
| `RATE_LIMIT_ENABLED` | `true` | レート制限の有効/無効 |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | `60` | レート制限 (リクエスト/分) |
| `RATE_LIMIT_BURST_SIZE` | `10` | バースト許容量 |
| `LOG_LEVEL` | `INFO` | ログレベル |
| `COST_ALERT_THRESHOLD_DAILY_USD` | `10` | 日次コストアラート閾値 (USD) |

## トラブルシューティング

### Embedding サーバーが起動しない

```bash
# ログ確認
docker compose logs embedding

# GPU が認識されているか確認
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# VRAM 不足の場合
# docker-compose.yml の gpu-memory-utilization を下げる (0.3 → 0.2)
```

**初回起動時**: Hugging Face からモデルをダウンロード (~600MB)。`hf_cache` ボリュームにキャッシュされるため 2 回目以降は高速。

### DB 接続エラー

```bash
# PostgreSQL の状態確認
docker compose exec db pg_isready -U postgres

# pgvector 拡張が有効か確認
docker compose exec db psql -U postgres -d llm_platform -c "SELECT * FROM pg_extension WHERE extname = 'vector';"

# DB を再作成
docker compose down -v    # ボリュームも削除
make up
make migrate
```

### Redis 接続エラー

```bash
# Redis の状態確認
docker compose exec redis redis-cli ping

# メモリ使用量確認
docker compose exec redis redis-cli info memory
```

**レート制限の Redis 障害**: Redis が利用不可でもリクエストは通過する (graceful degradation)。

### アプリケーションの 500 エラー

```bash
# 構造化ログを確認 (JSON)
make logs | jq '.event'

# リクエスト ID でフィルタ
make logs | jq 'select(.request_id == "TARGET_ID")'

# エラーのみ
make logs | jq 'select(.log_level == "error")'
```

### テストが通らない

```bash
# 特定テストを実行
uv run pytest tests/unit/test_chunker.py -v

# デバッグ出力付き
uv run pytest tests/unit/ -v -s

# LLM API テストをスキップ
make test-no-llm
```

## コスト管理

- デフォルトモデル (`gemini-2.5-flash-lite`) は Gemini 無料枠 (15 RPM / 250K TPM / 1000 RPD)
- セマンティックキャッシュ (Redis) で同一・類似クエリの API 呼び出しを削減
- Embedding はローカル vLLM 実行で API 費用ゼロ
- コスト追跡: `src/monitoring/cost_tracker.py` で日次コストを監視
- アラート閾値: `COST_ALERT_THRESHOLD_DAILY_USD` (デフォルト: $10/日)

## モニタリング

### 構造化ログ

全ログは JSON 形式で stdout に出力 (structlog)。

```json
{
  "event": "request_completed",
  "request_id": "550e8400-...",
  "method": "POST",
  "path": "/chat",
  "status_code": 200,
  "duration_ms": 245.67,
  "timestamp": "2025-01-01T00:00:00Z",
  "log_level": "info"
}
```

- リクエスト ID は全ログに自動付与 (`X-Request-ID` ヘッダまたは自動生成)
- PII はログ出力前に自動マスキング (`pii_log_processor`)

### Prometheus メトリクス

`src/monitoring/metrics.py` で Prometheus メトリクスを公開。

- リクエスト数・レイテンシ
- LLM API 呼び出し数・トークン使用量
- キャッシュヒット率
- エラー率
