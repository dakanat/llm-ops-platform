# Agent 設計

## 概要

Agent は ReAct (Reasoning + Acting) パターンで動作するタスク実行エンジン。
LLM が思考 → ツール選択 → 実行 → 観察のループを繰り返し、最終回答を生成する。

## ReAct ループ

```
User Query
  ↓
AgentRuntime.run()
  ↓
┌─────────────────────────────────────────┐
│  1. Guardrails.check_input()            │ ← 入力バリデーション
│  2. ReActPlanner.plan()                 │ ← LLM に思考を要求
│     → Thought + Action + Action Input   │
│  3. ToolRegistry.get(action)            │ ← ツール取得
│  4. FallbackStrategy.execute_with_fallback() │ ← リトライ付き実行
│  5. Observation を AgentState に追加      │
│  6. max_steps チェック                   │
│  7. Final Answer なら終了、なければ 2 へ   │
└─────────────────────────────────────────┘
  ↓
AgentResult (answer + steps + tool_calls)
```

## コンポーネント

### AgentRuntime (`src/agent/runtime.py`)

Agent の実行エンジン。ReAct ループ全体を管理する。

```python
class AgentRuntime:
    def __init__(
        self,
        planner: ReActPlanner,
        tool_registry: ToolRegistry,
        guardrails: Guardrails,
        fallback_strategy: FallbackStrategy,
        max_steps: int = 10,
    ) -> None: ...

    async def run(self, query: str) -> AgentResult: ...
```

**AgentResult**:
- `answer: str` — 最終回答
- `steps: list[AgentStep]` — 各ステップの思考・行動・観察
- `tool_calls: int` — ツール呼び出し回数
- `stopped_reason: str` — 終了理由 ("completed" | "max_steps_reached")

### AgentState (`src/agent/state.py`)

会話履歴と中間ステップを保持する状態管理。

```python
class AgentState:
    def __init__(self, max_history: int = 50) -> None: ...
    def add_message(self, role: str, content: str) -> None: ...
    def add_step(self, step: AgentStep) -> None: ...
    def get_messages(self) -> list[dict[str, str]]: ...
    def get_steps(self) -> list[AgentStep]: ...
    def clear(self) -> None: ...
```

**AgentStep**: `thought`, `action`, `action_input`, `observation` を保持

### ReActPlanner (`src/agent/planner.py`)

LLM に ReAct 形式で思考・行動計画を生成させる。

```python
class ReActPlanner:
    def __init__(self, llm_provider: LLMProvider, model: str) -> None: ...
    async def plan(
        self, query: str, state: AgentState, available_tools: list[str]
    ) -> PlanResult: ...
```

**PlanResult**:
- `thought: str` — LLM の思考
- `action: str | None` — 実行するツール名 (None = Final Answer)
- `action_input: str | None` — ツールへの入力
- `final_answer: str | None` — 最終回答 (action が None の場合)

### Guardrails (`src/agent/guardrails.py`)

入出力バリデーション。3 段階のアクション: `allow`, `warn`, `block`。

```python
class GuardrailAction(StrEnum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"

class Guardrails:
    def __init__(
        self,
        max_input_tokens: int = 4096,
        max_output_tokens: int = 4096,
        blocked_patterns: list[str] | None = None,
    ) -> None: ...

    def check_input(self, text: str) -> GuardrailResult: ...
    def check_output(self, text: str) -> GuardrailResult: ...
```

**チェック項目**:
- **入力**: トークン上限 (文字数ベース推定), 禁止パターン (正規表現)
- **出力**: トークン上限, 禁止パターン
- `GuardrailResult(action, reason, details)`

### FallbackStrategy (`src/agent/fallback.py`)

ツール実行失敗時のリトライと degradation。

```python
class FallbackStrategy:
    def __init__(self, max_retries: int = 2) -> None: ...
    async def execute_with_fallback(
        self, tool: Tool, input_text: str
    ) -> FallbackResult: ...
```

**動作**:
1. ツール実行を試行
2. `ToolResult.is_error` が True または例外発生時にリトライ
3. 全試行失敗 → `FallbackResult(degraded=True)` を返す
4. Runtime は degraded 時に LLM 直接回答へフォールバック

## ツール

### Tool Protocol (`src/agent/tools/base.py`)

全ツールが満たすべきインターフェース。

```python
@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    async def execute(self, input_text: str) -> ToolResult: ...
```

`ToolResult(output: str, error: str | None)` — `is_error` プロパティでエラー判定。

### ToolRegistry (`src/agent/tools/registry.py`)

ツールの登録・取得を管理。

```python
class ToolRegistry:
    def register(self, tool: Tool) -> None: ...    # 重複時 DuplicateToolError
    def get(self, name: str) -> Tool: ...           # 未登録時 ToolNotFoundError
    def has(self, name: str) -> bool: ...
    def list_tools(self) -> list[str]: ...          # ソート済み名前リスト
```

### 実装済みツール

| ツール | クラス | 説明 |
|--------|--------|------|
| `calculator` | `CalculatorTool` | AST ベースの安全な数式評価。`+`, `-`, `*`, `/`, `//`, `%`, `**` に対応。関数呼び出し・属性アクセスは AST レベルで拒否 |
| `search` | `SearchTool` | RAGPipeline を使用したドキュメント検索。結果にソースチャンク引用を含む |
| `database` | `DatabaseTool` | 読み取り専用 SQL クエリ。SELECT のみ許可、書き込み系キーワード・複数ステートメント・コメントを拒否。結果は JSON 形式、最大 100 行 |

## エラーハンドリング

```
ToolResult.is_error = True
  ↓
FallbackStrategy: リトライ (最大 2 回)
  ↓ (全失敗)
FallbackResult.degraded = True
  ↓
AgentRuntime: LLM 直接回答へ切り替え
  ↓
AgentResult.stopped_reason = "tool_failure_degraded"
```

## 設定パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `max_steps` | 10 | ReAct ループの最大ステップ数 |
| `max_retries` | 2 | ツール失敗時のリトライ回数 |
| `max_input_tokens` | 4096 | 入力トークン上限 |
| `max_output_tokens` | 4096 | 出力トークン上限 |
| `search.top_k` | 5 | 検索ツールの返却チャンク数 |
| `database.max_rows` | 100 | DB ツールの最大返却行数 |
