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

## 評価実行

### EvalRunner (`src/eval/runner.py`)

データセットに対してメトリクスを一括実行する。

```python
class EvalRunner:
    def __init__(
        self,
        faithfulness: FaithfulnessMetric | None = None,
        relevance: RelevanceMetric | None = None,
    ) -> None: ...

    async def run(self, dataset: EvalDataset) -> EvalRunResult: ...
```

**EvalDataset**: 評価用データセット

```python
class EvalExample(BaseModel):
    query: str
    context: str
    answer: str
    expected_answer: str | None = None

class EvalDataset(BaseModel):
    name: str
    examples: list[EvalExample]
```

**EvalRunResult**: 各サンプルの結果 + 統計サマリ

```python
class EvalRunResult(BaseModel):
    dataset_name: str
    results: list[EvalSampleResult]
    faithfulness_summary: MetricSummary | None
    relevance_summary: MetricSummary | None

class MetricSummary(BaseModel):
    mean: float
    median: float
    min: float
    max: float
    std: float
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

def compare_with_baseline(
    current: EvalRunResult,
    baseline: EvalRunResult,
    thresholds: RegressionThresholds | None = None,
) -> RegressionResult: ...
```

- baseline の `mean` と current の `mean` の差が閾値を超えたら FAIL
- faithfulness/relevance それぞれ独立に判定
- サマリが無い場合は SKIP

## 合成データ生成 (`src/eval/synthetic_data.py`)

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

## CI 統合

```yaml
# .github/workflows/eval-regression.yml (将来実装)
# 1. 評価データセット (tests/eval/test_sets/) でメトリクスを計算
# 2. ベースライン (tests/eval/regression_baseline.json) と比較
# 3. 閾値を超える品質低下があれば CI を失敗させる
```

**テストマーカー**: `@pytest.mark.llm` — LLM API 呼び出しを含むテスト。`pytest -m "not llm"` でスキップ可能。
