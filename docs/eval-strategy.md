# 評価戦略

## 概要

LLM 出力の品質を定量的に評価し、回帰テストで品質低下を検出する。
メトリクスは LLM-as-judge パターン (LLM が別の LLM 出力を評価) と直接計測の 2 種類。

## メトリクス

### Faithfulness (忠実性) — `src/eval/metrics/faithfulness.py`

コンテキストに対して回答がどの程度忠実かを評価。Hallucination 検出に使用。

```python
class FaithfulnessMetric:
    def __init__(self, llm_provider: LLMProvider, model: str) -> None: ...
    async def evaluate(self, context: str, answer: str) -> FaithfulnessResult: ...
```

- **入力**: コンテキスト (検索チャンク), 回答 (LLM 生成テキスト)
- **出力**: `FaithfulnessResult(score: float, reason: str)`
- **スコア**: 0.0 (完全にハルシネーション) 〜 1.0 (完全に忠実)
- **方法**: LLM にシステムプロンプトで評価基準を与え、`Score: X.XX / Reason: ...` 形式で回答させる
- `parse_evaluation_response()` でスコアと理由をパース (0.0-1.0 にクランプ)

### Relevance (関連性) — `src/eval/metrics/relevance.py`

クエリに対して回答がどの程度関連しているかを評価。

```python
class RelevanceMetric:
    def __init__(self, llm_provider: LLMProvider, model: str) -> None: ...
    async def evaluate(self, query: str, answer: str) -> RelevanceResult: ...
```

- **入力**: ユーザー質問, 回答
- **出力**: `RelevanceResult(score: float, reason: str)`
- **スコア**: 0.0 (無関係) 〜 1.0 (完全に関連)
- **方法**: Faithfulness と同パターン (LLM-as-judge)

### Latency (レイテンシ) — `src/eval/metrics/latency.py`

任意の callable の実行時間を計測。

```python
async def measure_latency(func, *args, **kwargs) -> LatencyResult: ...
```

- **出力**: `LatencyResult(duration_seconds: float)`
- async/sync 両対応 (`asyncio.iscoroutine` で判定)
- callable の例外は `MetricError` に変換

### 共通ユーティリティ (`src/eval/metrics/__init__.py`)

```python
def parse_evaluation_response(content: str) -> tuple[float, str]:
    """LLM レスポンスから Score と Reason をパース。
    Score は 0.0-1.0 にクランプ。"""
```

## データセット

### インメモリモデル (`src/eval/datasets.py`)

評価実行時に使用するデータ構造。

```python
class EvalExample(BaseModel):
    query: str
    expected_answer: str | None = None

class EvalDataset(BaseModel):
    name: str
    examples: list[EvalExample]
```

- `load_dataset(path)` / `save_dataset(dataset, path)` で JSON ファイルの読み書きが可能

### DB モデル (`src/db/models.py`)

永続化されたデータセット。Web UI・API からの CRUD に使用。

```python
class EvalDatasetRecord(SQLModel, table=True):
    __tablename__ = "eval_datasets"
    id: UUID
    name: str          # unique, indexed
    description: str | None
    created_by: UUID   # FK → users.id
    created_at: datetime
    updated_at: datetime

class EvalExampleRecord(SQLModel, table=True):
    __tablename__ = "eval_examples"
    id: UUID
    dataset_id: UUID   # FK → eval_datasets.id
    query: str
    expected_answer: str | None
    created_at: datetime
```

### 合成データ生成 (`src/eval/synthetic_data.py`)

ドキュメントテキストから LLM で QA ペアを自動生成。

```python
class SyntheticDataGenerator:
    def __init__(
        self, llm_provider: LLMProvider, model: str, num_pairs: int = 3
    ) -> None: ...

    async def generate(self, text: str, num_pairs: int | None = None) -> EvalDataset: ...
    async def generate_from_chunks(
        self, chunks: list[str], num_pairs_per_chunk: int | None = None
    ) -> EvalDataset: ...
```

- LLM に JSON 配列形式で QA ペアを生成させる
- マークダウンコードフェンスの除去、不正アイテムのスキップに対応
- `generate_from_chunks()` で複数チャンクから統合データセットを生成

## 評価実行

### EvalRunner (`src/eval/runner.py`)

データセットに対してメトリクスを一括実行する。各クエリを RAGPipeline に通して実際の回答・コンテキストを取得し、メトリクスで採点する。

```python
class EvalRunner:
    def __init__(
        self,
        pipeline: RAGPipeline,
        faithfulness_metric: FaithfulnessMetric | None = None,
        relevance_metric: RelevanceMetric | None = None,
    ) -> None: ...

    async def run(self, dataset: EvalDataset) -> EvalRunResult: ...
```

**実行フロー** (1件のサンプルあたり):
1. `pipeline.query(example.query)` で RAG パイプラインを実行し、回答とコンテキストを取得
2. `time.perf_counter()` でレイテンシを計測
3. `faithfulness_metric.evaluate(context, answer)` で忠実性を採点
4. `relevance_metric.evaluate(query, answer)` で関連性を採点
5. 例外が発生した場合は `ExampleResult.error` に記録して続行 (全体を止めない)

**ExampleResult**: 1件の評価結果

```python
class ExampleResult(BaseModel):
    query: str
    expected_answer: str | None = None
    rag_answer: str | None = None
    rag_context: str | None = None
    faithfulness_score: float | None = None
    relevance_score: float | None = None
    latency_seconds: float | None = None
    error: str | None = None
```

**EvalRunResult**: 各サンプルの結果 + 統計サマリ

```python
class EvalRunResult(BaseModel):
    dataset_name: str
    results: list[ExampleResult]
    faithfulness_summary: MetricSummary | None
    relevance_summary: MetricSummary | None
    latency_summary: MetricSummary | None

class MetricSummary(BaseModel):
    mean: float
    count: int
```

- メトリクスが None の場合はスキップ
- 個別サンプルの例外はエラー記録してスキップ (全体を止めない)

## 回帰テスト (`src/eval/regression.py`)

ベースラインとの比較で品質低下を検出する。

```python
class RegressionThresholds(BaseModel):
    faithfulness_drop: float = 0.05   # 許容する低下幅
    relevance_drop: float = 0.05

class RegressionResult(BaseModel):
    passed: bool
    details: list[str]
    current_faithfulness: float | None
    baseline_faithfulness: float | None
    current_relevance: float | None
    baseline_relevance: float | None

def compare(
    current: EvalRunResult,
    baseline: EvalRunResult,
    thresholds: RegressionThresholds | None = None,
) -> RegressionResult: ...
```

- baseline の `mean` と current の `mean` の差が閾値を超えたら FAIL
- faithfulness/relevance それぞれ独立に判定
- サマリが無い場合は SKIP
- `load_baseline()` / `save_baseline()` で JSON ファイルの読み書きが可能

## API による評価実行

### REST API — `POST /eval/run`

```python
class EvalRunRequest(BaseModel):
    dataset_name: str
    dataset_id: UUID | None = None       # DB に保存済みのデータセット
    examples: list[EvalExampleInput] | None = None  # インライン指定

    # dataset_id と examples は排他 (どちらか一方のみ)
```

- `dataset_id` 指定: DB から `EvalDatasetRecord` + `EvalExampleRecord` をロード
- `examples` 指定: リクエストボディ内のインライン例を使用
- レスポンス: `EvalRunResponse` (各サンプルの結果 + メトリクスサマリ)
- 権限: `EVAL_RUN` パーミッション (admin / user のみ、viewer は不可)

### Web UI — `/web/eval`

Eval Dashboard (htmx ベース) から DB 保存済みデータセットを選択して評価を実行する。

**データセット選択**:
- `GET /web/eval` で DB のデータセット一覧を取得し、`<select>` ドロップダウンで表示
- 各選択肢にデータセット名とサンプル数を表示 (例: `my-dataset (10 examples)`)
- データセットが存在しない場合は作成ページへのリンクを表示

**評価実行**:
- `POST /web/eval/run` にフォームで `dataset_id` を送信
- サーバー側で DB からデータセットをロードし、`EvalRunner` で評価を実行
- 結果は htmx で `eval/run_result.html` テンプレートにインラインレンダリング

**結果表示**:
- サマリ統計 (Faithfulness / Relevance / Latency の平均値とサンプル数)
- 詳細テーブル (Query, RAG Answer, Expected, スコア, Error)
- 長いテキストはクリックで展開/折りたたみ可能 (`toggleExpand()`)

### データセット管理 — `/web/eval/datasets`

| エンドポイント | 説明 |
|---|---|
| `GET /web/eval/datasets` | データセット一覧 (名前, 説明, サンプル数, 作成日) |
| `GET /web/eval/datasets/create` | 作成フォーム (Manual / Generate タブ切替) |
| `POST /api/eval/datasets` | Manual: JSON 配列で examples を直接指定して作成 |
| `POST /web/eval/datasets/generate` | Generate: ソーステキストから LLM で QA ペアを自動生成 |
| `GET /web/eval/datasets/{id}` | データセット詳細 (全 examples をテーブル表示) |

**テストマーカー**: `@pytest.mark.llm` — LLM API 呼び出しを含むテスト。`pytest -m "not llm"` でスキップ可能。
