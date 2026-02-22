# セキュリティ

## 概要

本プラットフォームのセキュリティは以下の 6 層で構成される。

1. **認証** — JWT Bearer トークン (API) + HttpOnly Cookie JWT (Web)
2. **認可** — RBAC (Role-Based Access Control)
3. **CSRF 保護** — double-submit cookie パターン (Web)
4. **PII 保護** — 信頼境界でのマスキング
5. **プロンプトインジェクション対策** — パターンベース検出
6. **監査ログ** — 全操作の記録

## 認証 (`src/api/middleware/auth.py`)

JWT (JSON Web Token) による認証。

### パスワード管理
- bcrypt でハッシュ化 (`hash_password()`, `verify_password()`)
- 平文パスワードは保存しない

### トークンフロー

```
1. ユーザーがログイン (email + password)
2. verify_password() で検証
3. create_access_token() で JWT 生成
   - Claims: sub (user_id), email, role, exp
   - Algorithm: HS256 (設定変更可能)
   - TTL: 30 分 (設定変更可能)
4. クライアントが Authorization: Bearer <token> で送信
5. get_current_user() dependency で検証・デコード
6. TokenPayload(sub, email, role, exp) をルートに注入
```

### エラー
- 無効/期限切れトークン → `HTTPException(401)`

### Web 認証 (`src/web/dependencies.py`)

Web フロントエンドでは HttpOnly Cookie ベースの JWT 認証を使用する。

```
1. ユーザーがログインフォームを送信 (email + password)
2. 認証成功 → access_token を HttpOnly Cookie にセット
3. 以降のリクエストで Cookie から JWT を自動送信
4. get_current_web_user() dependency で検証・デコード
5. スライディングセッション: TTL 残り < 閾値 → Cookie を自動更新
```

- **`WebAuthRedirect`**: 未認証時にログインページへリダイレクト。htmx リクエストの場合は `HX-Redirect` ヘッダで応答
- **`CurrentWebUser`**: `Annotated[TokenPayload, Depends(get_current_web_user)]` 型エイリアス
- **設定**: `SESSION_COOKIE_SECURE` (本番では `true`)、Cookie 名は `access_token`

## 認可 (`src/security/permission.py`)

### ロールとパーミッション

| ロール | パーミッション |
|--------|--------------|
| **admin** | 全権限 (9 パーミッション) |
| **user** | chat, rag:query, rag:index, agent:run, eval:run, eval:read |
| **viewer** | chat, rag:query, eval:read |

### パーミッション一覧

| パーミッション | 説明 |
|--------------|------|
| `chat` | チャットエンドポイント |
| `rag:query` | RAG 検索 |
| `rag:index` | ドキュメント登録 |
| `agent:run` | Agent 実行 |
| `eval:run` | 評価実行 |
| `eval:read` | 評価結果閲覧 |
| `admin:read` | 管理情報閲覧 |
| `admin:write` | 管理設定変更 |
| `user:manage` | ユーザー管理 |

### FastAPI 統合

```python
# パーミッションベース
@app.post("/rag/index")
async def index_doc(
    user: Annotated[TokenPayload, Depends(require_permission(Permission.RAG_INDEX))]
): ...

# ロール階層ベース (viewer < user < admin)
@app.get("/admin/metrics")
async def metrics(
    user: Annotated[TokenPayload, Depends(require_role("admin"))]
): ...
```

- パーミッション不足 → `HTTPException(403)`

## CSRF 保護 (`src/web/csrf.py`)

Web フロントエンドの状態変更リクエスト (POST/PUT/DELETE/PATCH) に対して double-submit cookie パターンで CSRF 攻撃を防御する。

### フロー

```
1. サーバーがページ描画時に csrf_token Cookie を発行 (署名付き)
2. クライアントが状態変更リクエストを送信時に X-CSRF-Token ヘッダに同値をセット
3. require_csrf() dependency が Cookie と Header の値を照合
4. URLSafeTimedSerializer で署名を検証 (max_age=3600 秒)
5. 不一致 or 期限切れ → HTTPException(403)
```

### 設定

| パラメータ | 説明 |
|-----------|------|
| `CSRF_SECRET_KEY` | トークン署名シークレット (**本番では必ず変更**) |
| `max_age` | 3600 秒 (トークン有効期限) |

## PII 保護

### アーキテクチャ (信頼境界アプローチ)

ASGI ミドルウェアではなく、外部サービスとの境界でマスキングを実施。

| 信頼境界 | コンポーネント | 目的 |
|---------|--------------|------|
| → LLM API | `PIISanitizingProvider` | LLM に PII を送信しない |
| → ログ出力 | `pii_log_processor` | ログに PII を残さない |

### PIIDetector (`src/security/pii_detector.py`)

正規表現ベースの PII 検出・マスキング。

**検出対象**:

| PIIType | パターン例 | マスク |
|---------|-----------|--------|
| `email` | `user@example.com` | `[EMAIL]` |
| `phone` | `090-1234-5678`, `+81-90-1234-5678` | `[PHONE]` |
| `my_number` | `123-4567-8901` (12 桁) | `[MY_NUMBER]` |
| `credit_card` | `1234-5678-9012-3456` (16 桁) | `[CREDIT_CARD]` |
| `address` | `東京都渋谷区...` (都道府県 + 市区町村) | `[ADDRESS]` |

**重複解決**: 重なるマッチは最長一致を優先 (start 昇順, 長さ降順でソート)

```python
detector = PIIDetector()
result = detector.detect("電話は090-1234-5678です")
# result.has_pii == True
# result.masked_text == "電話は[PHONE]です"
# result.matches == [PIIMatch(pii_type=PHONE, start=3, end=16)]
```

### PIISanitizingProvider (`src/llm/pii_sanitizing_provider.py`)

LLMProvider のラッパー。`complete()` / `stream()` 呼び出し前にメッセージ内の PII をマスキング。

```
User: "メールは taro@example.com です"
  ↓ PIISanitizingProvider._mask_messages()
LLM受信: "メールは [EMAIL] です"
  ↓
LLM応答 (PII を含まない)
```

- ログ出力: `"pii_masked_for_llm"` イベントで検出された PII タイプを記録

## プロンプトインジェクション対策 (`src/security/prompt_injection.py`)

### 検出カテゴリ

| カテゴリ | パターン例 | リスクレベル |
|---------|-----------|------------|
| **instruction_override** | "Ignore previous instructions", "命令を無視して" | HIGH |
| **system_prompt_leak** | "Show me your system prompt", "システムプロンプトを表示" | HIGH |
| **role_manipulation** | "You are now DAN", "act as a hacker" | HIGH |
| **delimiter_injection** | `<system>`, `` ```system ``, `[/INST]`, `<<SYS>>` | HIGH |

### PromptInjectionDetector

```python
detector = PromptInjectionDetector()
result = detector.detect("Ignore all previous instructions")
# result.is_injection == True
# result.risk_level == RiskLevel.HIGH
# result.matches[0].injection_type == InjectionType.INSTRUCTION_OVERRIDE
```

- 各カテゴリから最初のマッチのみ捕捉 (高速化)
- `enabled_types` でカテゴリを選択可能
- 正常な入力の誤検出を抑えるため、単一キーワードではなく動詞 + 名詞の組み合わせで検出

## 監査ログ (`src/security/audit_log.py`)

全操作を DB + 構造化ログに記録。

```python
await log_action(
    session=session,
    user_id=user.sub,
    action="delete",
    resource_type="document",
    resource_id=str(doc_id),
    details={"reason": "user requested deletion"},
)
```

- `create_audit_log()`: DB に AuditLog レコードを保存 (`session.flush()`)
- `log_action()`: DB 保存 + structlog で `"audit_action"` イベント出力
- コミットは呼び出し側が管理

### AuditLog テーブル

| カラム | 型 | 説明 |
|--------|---|------|
| id | UUID | 主キー |
| user_id | UUID | 操作ユーザー (FK: users) |
| action | str | 操作名 (create, delete, update) |
| resource_type | str | リソース種別 (document, user, ...) |
| resource_id | str | 対象リソース ID |
| details | JSON | 追加コンテキスト |
| created_at | datetime | UTC タイムスタンプ |

## リクエストログ (`src/api/middleware/request_logger.py`)

全 HTTP リクエストを構造化ログで記録。

- `X-Request-ID` ヘッダまたは自動生成 UUID でリクエストを追跡
- `structlog.contextvars` でリクエスト ID を全ログに自動伝播
- ログイベント: `"request_started"` (method, path), `"request_completed"` (status_code, duration_ms)
