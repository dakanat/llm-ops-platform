"""Tests for eval datasets and runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from src.eval import DatasetError, EvalError
from src.eval.datasets import EvalDataset, EvalExample, load_dataset, save_dataset
from src.eval.metrics.faithfulness import FaithfulnessMetric
from src.eval.metrics.relevance import RelevanceMetric
from src.eval.runner import EvalRunner, EvalRunResult, ExampleResult, MetricSummary
from src.llm.providers.base import LLMResponse

# =============================================================================
# DatasetError 例外階層
# =============================================================================


class TestDatasetError:
    """DatasetError の例外階層テスト。"""

    def test_dataset_error_is_eval_error(self) -> None:
        """DatasetError が EvalError を継承していること。"""
        assert issubclass(DatasetError, EvalError)

    def test_dataset_error_can_be_raised_and_caught_as_eval_error(self) -> None:
        """DatasetError を EvalError として捕捉できること。"""
        with pytest.raises(EvalError):
            raise DatasetError("test")


# =============================================================================
# EvalExample モデル
# =============================================================================


class TestEvalExample:
    """EvalExample モデルのテスト。"""

    def test_creates_with_required_fields(self) -> None:
        """query, context, answer の必須フィールドで生成できること。"""
        example = EvalExample(query="質問", context="文脈", answer="回答")

        assert example.query == "質問"
        assert example.context == "文脈"
        assert example.answer == "回答"

    def test_expected_answer_is_optional(self) -> None:
        """expected_answer がオプションで None がデフォルトであること。"""
        example = EvalExample(query="q", context="c", answer="a")

        assert example.expected_answer is None

    def test_creates_with_expected_answer(self) -> None:
        """expected_answer を指定して生成できること。"""
        example = EvalExample(query="q", context="c", answer="a", expected_answer="expected")

        assert example.expected_answer == "expected"


# =============================================================================
# EvalDataset モデル
# =============================================================================


class TestEvalDataset:
    """EvalDataset モデルのテスト。"""

    def test_creates_with_name_and_examples(self) -> None:
        """name と examples で生成できること。"""
        examples = [EvalExample(query="q", context="c", answer="a")]
        dataset = EvalDataset(name="test-dataset", examples=examples)

        assert dataset.name == "test-dataset"
        assert len(dataset.examples) == 1

    def test_creates_with_empty_examples(self) -> None:
        """空の examples リストで生成できること。"""
        dataset = EvalDataset(name="empty", examples=[])

        assert dataset.examples == []


# =============================================================================
# load_dataset
# =============================================================================


class TestLoadDataset:
    """load_dataset() のテスト。"""

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        """正常な JSON ファイルを読み込めること。"""
        data = {
            "name": "test",
            "examples": [{"query": "q1", "context": "c1", "answer": "a1"}],
        }
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        dataset = load_dataset(path)

        assert dataset.name == "test"
        assert len(dataset.examples) == 1
        assert dataset.examples[0].query == "q1"

    def test_raises_dataset_error_on_file_not_found(self, tmp_path: Path) -> None:
        """存在しないファイルで DatasetError が発生すること。"""
        path = tmp_path / "nonexistent.json"

        with pytest.raises(DatasetError):
            load_dataset(path)

    def test_raises_dataset_error_on_invalid_json(self, tmp_path: Path) -> None:
        """不正な JSON で DatasetError が発生すること。"""
        path = tmp_path / "bad.json"
        path.write_text("{invalid json", encoding="utf-8")

        with pytest.raises(DatasetError):
            load_dataset(path)

    def test_raises_dataset_error_on_validation_error(self, tmp_path: Path) -> None:
        """バリデーションエラーで DatasetError が発生すること。"""
        data = {"name": "test"}  # examples フィールドが欠落
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(DatasetError):
            load_dataset(path)

    def test_raises_dataset_error_on_missing_required_example_fields(self, tmp_path: Path) -> None:
        """example に必須フィールドが欠けている場合に DatasetError が発生すること。"""
        data = {
            "name": "test",
            "examples": [{"query": "q1"}],  # context, answer が欠落
        }
        path = tmp_path / "missing_fields.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(DatasetError):
            load_dataset(path)


# =============================================================================
# save_dataset
# =============================================================================


class TestSaveDataset:
    """save_dataset() のテスト。"""

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """ファイルに書き込めること。"""
        dataset = EvalDataset(
            name="test",
            examples=[EvalExample(query="q", context="c", answer="a")],
        )
        path = tmp_path / "output.json"

        save_dataset(dataset, path)

        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["name"] == "test"

    def test_roundtrip(self, tmp_path: Path) -> None:
        """save → load のラウンドトリップが成功すること。"""
        original = EvalDataset(
            name="roundtrip",
            examples=[
                EvalExample(query="q1", context="c1", answer="a1", expected_answer="e1"),
                EvalExample(query="q2", context="c2", answer="a2"),
            ],
        )
        path = tmp_path / "roundtrip.json"

        save_dataset(original, path)
        loaded = load_dataset(path)

        assert loaded.name == original.name
        assert len(loaded.examples) == len(original.examples)
        assert loaded.examples[0].expected_answer == "e1"
        assert loaded.examples[1].expected_answer is None


# =============================================================================
# ExampleResult / MetricSummary / EvalRunResult モデル
# =============================================================================


class TestExampleResult:
    """ExampleResult モデルのテスト。"""

    def test_creates_with_example_only(self) -> None:
        """example のみで生成でき、スコアは None であること。"""
        example = EvalExample(query="q", context="c", answer="a")
        result = ExampleResult(example=example)

        assert result.example == example
        assert result.faithfulness_score is None
        assert result.relevance_score is None
        assert result.latency_seconds is None
        assert result.error is None

    def test_creates_with_all_fields(self) -> None:
        """全フィールドを指定して生成できること。"""
        example = EvalExample(query="q", context="c", answer="a")
        result = ExampleResult(
            example=example,
            faithfulness_score=0.9,
            relevance_score=0.8,
            latency_seconds=1.5,
            error="some error",
        )

        assert result.faithfulness_score == 0.9
        assert result.relevance_score == 0.8
        assert result.latency_seconds == 1.5
        assert result.error == "some error"


class TestMetricSummary:
    """MetricSummary モデルのテスト。"""

    def test_creates_with_mean_and_count(self) -> None:
        """mean と count で生成できること。"""
        summary = MetricSummary(mean=0.85, count=10)

        assert summary.mean == 0.85
        assert summary.count == 10


class TestEvalRunResult:
    """EvalRunResult モデルのテスト。"""

    def test_creates_with_dataset_name_and_results(self) -> None:
        """dataset_name と results で生成でき、summary は None であること。"""
        result = EvalRunResult(dataset_name="test", results=[])

        assert result.dataset_name == "test"
        assert result.results == []
        assert result.faithfulness_summary is None
        assert result.relevance_summary is None
        assert result.latency_summary is None

    def test_creates_with_summaries(self) -> None:
        """summary を指定して生成できること。"""
        summary = MetricSummary(mean=0.9, count=5)
        result = EvalRunResult(
            dataset_name="test",
            results=[],
            faithfulness_summary=summary,
            relevance_summary=summary,
            latency_summary=summary,
        )

        assert result.faithfulness_summary is not None
        assert result.faithfulness_summary.mean == 0.9


# =============================================================================
# EvalRunner
# =============================================================================


def _make_llm_response(content: str) -> LLMResponse:
    """テスト用の LLMResponse を生成。"""
    return LLMResponse(content=content, model="test-model")


class TestEvalRunnerInit:
    """EvalRunner.__init__() のテスト。"""

    def test_creates_with_no_metrics(self) -> None:
        """メトリクスなしで生成できること。"""
        runner = EvalRunner()

        assert runner._faithfulness_metric is None
        assert runner._relevance_metric is None

    def test_creates_with_metrics(self) -> None:
        """メトリクスを注入して生成できること。"""
        provider = AsyncMock()
        faith = FaithfulnessMetric(llm_provider=provider, model="m")
        rel = RelevanceMetric(llm_provider=provider, model="m")

        runner = EvalRunner(faithfulness_metric=faith, relevance_metric=rel)

        assert runner._faithfulness_metric is faith
        assert runner._relevance_metric is rel


class TestComputeSummary:
    """EvalRunner._compute_summary() のテスト。"""

    def test_computes_mean_and_count(self) -> None:
        """平均とカウントを正しく計算すること。"""
        runner = EvalRunner()
        summary = runner._compute_summary([0.8, 0.9, 1.0])

        assert summary is not None
        assert summary.count == 3
        assert summary.mean == pytest.approx(0.9)

    def test_returns_summary_for_single_score(self) -> None:
        """単一スコアで正しく計算すること。"""
        runner = EvalRunner()
        summary = runner._compute_summary([0.5])

        assert summary is not None
        assert summary.count == 1
        assert summary.mean == 0.5

    def test_returns_none_for_empty_scores(self) -> None:
        """空リストで None を返すこと。"""
        runner = EvalRunner()
        result = runner._compute_summary([])

        assert result is None


class TestEvalRunnerRun:
    """EvalRunner.run() のテスト。"""

    @pytest.fixture
    def provider(self) -> AsyncMock:
        mock = AsyncMock()
        mock.complete.return_value = _make_llm_response("Score: 0.8\nReason: Good answer.")
        return mock

    @pytest.fixture
    def dataset(self) -> EvalDataset:
        return EvalDataset(
            name="test-dataset",
            examples=[
                EvalExample(query="q1", context="c1", answer="a1"),
                EvalExample(query="q2", context="c2", answer="a2"),
            ],
        )

    async def test_returns_eval_run_result(self, provider: AsyncMock, dataset: EvalDataset) -> None:
        """EvalRunResult が返ること。"""
        faith = FaithfulnessMetric(llm_provider=provider, model="m")
        runner = EvalRunner(faithfulness_metric=faith)

        result = await runner.run(dataset)

        assert isinstance(result, EvalRunResult)

    async def test_dataset_name_is_set(self, provider: AsyncMock, dataset: EvalDataset) -> None:
        """dataset_name が設定されること。"""
        runner = EvalRunner()
        result = await runner.run(dataset)

        assert result.dataset_name == "test-dataset"

    async def test_evaluates_all_examples(self, provider: AsyncMock, dataset: EvalDataset) -> None:
        """全 example が評価されること。"""
        runner = EvalRunner()
        result = await runner.run(dataset)

        assert len(result.results) == 2

    async def test_faithfulness_scores_are_set(
        self, provider: AsyncMock, dataset: EvalDataset
    ) -> None:
        """faithfulness メトリクスのスコアが設定されること。"""
        faith = FaithfulnessMetric(llm_provider=provider, model="m")
        runner = EvalRunner(faithfulness_metric=faith)

        result = await runner.run(dataset)

        assert result.results[0].faithfulness_score == 0.8
        assert result.results[1].faithfulness_score == 0.8

    async def test_relevance_scores_are_set(
        self, provider: AsyncMock, dataset: EvalDataset
    ) -> None:
        """relevance メトリクスのスコアが設定されること。"""
        rel = RelevanceMetric(llm_provider=provider, model="m")
        runner = EvalRunner(relevance_metric=rel)

        result = await runner.run(dataset)

        assert result.results[0].relevance_score == 0.8
        assert result.results[1].relevance_score == 0.8

    async def test_skips_faithfulness_when_none(self, dataset: EvalDataset) -> None:
        """faithfulness_metric が None の場合スキップされること。"""
        runner = EvalRunner(faithfulness_metric=None)
        result = await runner.run(dataset)

        for r in result.results:
            assert r.faithfulness_score is None

    async def test_skips_relevance_when_none(self, dataset: EvalDataset) -> None:
        """relevance_metric が None の場合スキップされること。"""
        runner = EvalRunner(relevance_metric=None)
        result = await runner.run(dataset)

        for r in result.results:
            assert r.relevance_score is None

    async def test_error_does_not_stop_evaluation(
        self, provider: AsyncMock, dataset: EvalDataset
    ) -> None:
        """1件のエラーで全体が止まらないこと。"""
        call_count = 0

        async def side_effect(**kwargs: object) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("LLM failed")
            return _make_llm_response("Score: 0.9\nReason: OK.")

        provider.complete.side_effect = side_effect
        faith = FaithfulnessMetric(llm_provider=provider, model="m")
        runner = EvalRunner(faithfulness_metric=faith)

        result = await runner.run(dataset)

        assert len(result.results) == 2
        assert result.results[0].error is not None
        assert result.results[1].faithfulness_score == 0.9

    async def test_error_field_contains_message(
        self, provider: AsyncMock, dataset: EvalDataset
    ) -> None:
        """エラー時に error フィールドにメッセージが格納されること。"""
        provider.complete.side_effect = RuntimeError("LLM failed")
        faith = FaithfulnessMetric(llm_provider=provider, model="m")
        runner = EvalRunner(faithfulness_metric=faith)

        result = await runner.run(dataset)

        assert result.results[0].error is not None
        assert "LLM failed" in result.results[0].error

    async def test_faithfulness_summary_is_computed(
        self, provider: AsyncMock, dataset: EvalDataset
    ) -> None:
        """faithfulness_summary が計算されること。"""
        faith = FaithfulnessMetric(llm_provider=provider, model="m")
        runner = EvalRunner(faithfulness_metric=faith)

        result = await runner.run(dataset)

        assert result.faithfulness_summary is not None
        assert result.faithfulness_summary.mean == 0.8
        assert result.faithfulness_summary.count == 2

    async def test_relevance_summary_is_computed(
        self, provider: AsyncMock, dataset: EvalDataset
    ) -> None:
        """relevance_summary が計算されること。"""
        rel = RelevanceMetric(llm_provider=provider, model="m")
        runner = EvalRunner(relevance_metric=rel)

        result = await runner.run(dataset)

        assert result.relevance_summary is not None
        assert result.relevance_summary.mean == 0.8
        assert result.relevance_summary.count == 2

    async def test_summary_is_none_when_metric_not_provided(self, dataset: EvalDataset) -> None:
        """メトリクスが None の場合 summary も None であること。"""
        runner = EvalRunner()
        result = await runner.run(dataset)

        assert result.faithfulness_summary is None
        assert result.relevance_summary is None

    async def test_summary_excludes_errored_examples(
        self, provider: AsyncMock, dataset: EvalDataset
    ) -> None:
        """エラーの example は summary の計算から除外されること。"""
        call_count = 0

        async def side_effect(**kwargs: object) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("fail")
            return _make_llm_response("Score: 0.9\nReason: OK.")

        provider.complete.side_effect = side_effect
        faith = FaithfulnessMetric(llm_provider=provider, model="m")
        runner = EvalRunner(faithfulness_metric=faith)

        result = await runner.run(dataset)

        assert result.faithfulness_summary is not None
        assert result.faithfulness_summary.count == 1
        assert result.faithfulness_summary.mean == 0.9

    async def test_empty_dataset_returns_empty_results(self) -> None:
        """空データセットで空の results が返ること。"""
        dataset = EvalDataset(name="empty", examples=[])
        runner = EvalRunner()

        result = await runner.run(dataset)

        assert result.results == []
        assert result.faithfulness_summary is None
        assert result.relevance_summary is None
