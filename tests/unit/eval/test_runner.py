"""Tests for eval datasets and runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from src.eval import DatasetError, EvalError
from src.eval.datasets import EvalDataset, EvalExample, load_dataset, save_dataset
from src.eval.metrics.faithfulness import FaithfulnessMetric
from src.eval.metrics.relevance import RelevanceMetric
from src.eval.runner import EvalRunner, EvalRunResult, ExampleResult, MetricSummary
from src.llm.providers.base import LLMResponse
from src.rag.generator import GenerationResult
from src.rag.retriever import RetrievedChunk

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

    def test_creates_with_query_only(self) -> None:
        """query のみで生成できること。"""
        example = EvalExample(query="What is RAG?")

        assert example.query == "What is RAG?"
        assert example.expected_answer is None

    def test_creates_with_expected_answer(self) -> None:
        """expected_answer を指定して生成できること。"""
        example = EvalExample(query="q", expected_answer="expected")

        assert example.expected_answer == "expected"


# =============================================================================
# EvalDataset モデル
# =============================================================================


class TestEvalDataset:
    """EvalDataset モデルのテスト。"""

    def test_creates_with_name_and_examples(self) -> None:
        """name と examples で生成できること。"""
        examples = [EvalExample(query="q")]
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
            "examples": [{"query": "q1"}],
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
            "examples": [{"expected_answer": "a"}],  # query が欠落
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
            examples=[EvalExample(query="q")],
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
                EvalExample(query="q1", expected_answer="e1"),
                EvalExample(query="q2"),
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

    def test_creates_with_query_only(self) -> None:
        """query のみで生成でき、スコアは None であること。"""
        result = ExampleResult(query="q")

        assert result.query == "q"
        assert result.expected_answer is None
        assert result.rag_answer is None
        assert result.rag_context is None
        assert result.faithfulness_score is None
        assert result.relevance_score is None
        assert result.latency_seconds is None
        assert result.error is None

    def test_creates_with_all_fields(self) -> None:
        """全フィールドを指定して生成できること。"""
        result = ExampleResult(
            query="q",
            expected_answer="expected",
            rag_answer="rag answer",
            rag_context="rag context",
            faithfulness_score=0.9,
            relevance_score=0.8,
            latency_seconds=1.5,
            error="some error",
        )

        assert result.rag_answer == "rag answer"
        assert result.rag_context == "rag context"
        assert result.expected_answer == "expected"
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

    def test_creates_with_pipeline_only(self) -> None:
        """pipeline のみで生成できること。"""
        pipeline = AsyncMock()
        runner = EvalRunner(pipeline=pipeline)

        assert runner._pipeline is pipeline
        assert runner._faithfulness_metric is None
        assert runner._relevance_metric is None

    def test_creates_with_metrics(self) -> None:
        """メトリクスを注入して生成できること。"""
        pipeline = AsyncMock()
        provider = AsyncMock()
        faith = FaithfulnessMetric(llm_provider=provider, model="m")
        rel = RelevanceMetric(llm_provider=provider, model="m")

        runner = EvalRunner(pipeline=pipeline, faithfulness_metric=faith, relevance_metric=rel)

        assert runner._faithfulness_metric is faith
        assert runner._relevance_metric is rel


class TestComputeSummary:
    """EvalRunner._compute_summary() のテスト。"""

    def test_computes_mean_and_count(self) -> None:
        """平均とカウントを正しく計算すること。"""
        runner = EvalRunner(pipeline=AsyncMock())
        summary = runner._compute_summary([0.8, 0.9, 1.0])

        assert summary is not None
        assert summary.count == 3
        assert summary.mean == pytest.approx(0.9)

    def test_returns_summary_for_single_score(self) -> None:
        """単一スコアで正しく計算すること。"""
        runner = EvalRunner(pipeline=AsyncMock())
        summary = runner._compute_summary([0.5])

        assert summary is not None
        assert summary.count == 1
        assert summary.mean == 0.5

    def test_returns_none_for_empty_scores(self) -> None:
        """空リストで None を返すこと。"""
        runner = EvalRunner(pipeline=AsyncMock())
        result = runner._compute_summary([])

        assert result is None


def _make_pipeline(
    answer: str = "RAG answer",
    sources: list[RetrievedChunk] | None = None,
) -> AsyncMock:
    """テスト用のパイプラインモックを生成。"""
    mock = AsyncMock()
    mock.query.return_value = GenerationResult(
        answer=answer,
        sources=sources or [RetrievedChunk(content="chunk1", chunk_index=0, document_id=uuid4())],
        model="test-model",
    )
    return mock


class TestEvalRunnerRun:
    """EvalRunner.run() のテスト。"""

    @pytest.fixture
    def provider(self) -> AsyncMock:
        mock = AsyncMock()
        mock.complete.return_value = _make_llm_response("Score: 0.8\nReason: Good answer.")
        return mock

    @pytest.fixture
    def pipeline(self) -> AsyncMock:
        return _make_pipeline()

    @pytest.fixture
    def dataset(self) -> EvalDataset:
        return EvalDataset(
            name="test-dataset",
            examples=[
                EvalExample(query="q1"),
                EvalExample(query="q2"),
            ],
        )

    async def test_returns_eval_run_result(
        self, pipeline: AsyncMock, provider: AsyncMock, dataset: EvalDataset
    ) -> None:
        """EvalRunResult が返ること。"""
        faith = FaithfulnessMetric(llm_provider=provider, model="m")
        runner = EvalRunner(pipeline=pipeline, faithfulness_metric=faith)

        result = await runner.run(dataset)

        assert isinstance(result, EvalRunResult)

    async def test_dataset_name_is_set(self, pipeline: AsyncMock, dataset: EvalDataset) -> None:
        """dataset_name が設定されること。"""
        runner = EvalRunner(pipeline=pipeline)
        result = await runner.run(dataset)

        assert result.dataset_name == "test-dataset"

    async def test_evaluates_all_examples(self, pipeline: AsyncMock, dataset: EvalDataset) -> None:
        """全 example が評価されること。"""
        runner = EvalRunner(pipeline=pipeline)
        result = await runner.run(dataset)

        assert len(result.results) == 2

    async def test_calls_pipeline_query_for_each_example(
        self, pipeline: AsyncMock, dataset: EvalDataset
    ) -> None:
        """各 example に対して pipeline.query() が呼ばれること。"""
        runner = EvalRunner(pipeline=pipeline)
        await runner.run(dataset)

        assert pipeline.query.call_count == 2
        pipeline.query.assert_any_call("q1")
        pipeline.query.assert_any_call("q2")

    async def test_rag_answer_is_set(self, pipeline: AsyncMock, dataset: EvalDataset) -> None:
        """rag_answer が pipeline の回答に設定されること。"""
        runner = EvalRunner(pipeline=pipeline)
        result = await runner.run(dataset)

        assert result.results[0].rag_answer == "RAG answer"

    async def test_rag_context_is_set(self, pipeline: AsyncMock, dataset: EvalDataset) -> None:
        """rag_context が pipeline のソースから結合されること。"""
        runner = EvalRunner(pipeline=pipeline)
        result = await runner.run(dataset)

        assert result.results[0].rag_context == "chunk1"

    async def test_rag_context_joins_multiple_sources(self, dataset: EvalDataset) -> None:
        """複数ソースが改行で結合されること。"""
        pipeline = _make_pipeline(
            sources=[
                RetrievedChunk(content="chunk1", chunk_index=0, document_id=uuid4()),
                RetrievedChunk(content="chunk2", chunk_index=1, document_id=uuid4()),
            ],
        )
        runner = EvalRunner(pipeline=pipeline)
        result = await runner.run(dataset)

        assert result.results[0].rag_context == "chunk1\n\nchunk2"

    async def test_latency_is_measured(self, pipeline: AsyncMock, dataset: EvalDataset) -> None:
        """latency_seconds が計測されること。"""
        runner = EvalRunner(pipeline=pipeline)
        result = await runner.run(dataset)

        assert result.results[0].latency_seconds is not None
        assert result.results[0].latency_seconds >= 0

    async def test_faithfulness_scores_are_set(
        self, pipeline: AsyncMock, provider: AsyncMock, dataset: EvalDataset
    ) -> None:
        """faithfulness メトリクスのスコアが設定されること。"""
        faith = FaithfulnessMetric(llm_provider=provider, model="m")
        runner = EvalRunner(pipeline=pipeline, faithfulness_metric=faith)

        result = await runner.run(dataset)

        assert result.results[0].faithfulness_score == 0.8
        assert result.results[1].faithfulness_score == 0.8

    async def test_relevance_scores_are_set(
        self, pipeline: AsyncMock, provider: AsyncMock, dataset: EvalDataset
    ) -> None:
        """relevance メトリクスのスコアが設定されること。"""
        rel = RelevanceMetric(llm_provider=provider, model="m")
        runner = EvalRunner(pipeline=pipeline, relevance_metric=rel)

        result = await runner.run(dataset)

        assert result.results[0].relevance_score == 0.8
        assert result.results[1].relevance_score == 0.8

    async def test_skips_faithfulness_when_none(
        self, pipeline: AsyncMock, dataset: EvalDataset
    ) -> None:
        """faithfulness_metric が None の場合スキップされること。"""
        runner = EvalRunner(pipeline=pipeline, faithfulness_metric=None)
        result = await runner.run(dataset)

        for r in result.results:
            assert r.faithfulness_score is None

    async def test_skips_relevance_when_none(
        self, pipeline: AsyncMock, dataset: EvalDataset
    ) -> None:
        """relevance_metric が None の場合スキップされること。"""
        runner = EvalRunner(pipeline=pipeline, relevance_metric=None)
        result = await runner.run(dataset)

        for r in result.results:
            assert r.relevance_score is None

    async def test_pipeline_error_does_not_stop_evaluation(
        self, provider: AsyncMock, dataset: EvalDataset
    ) -> None:
        """1件の pipeline エラーで全体が止まらないこと。"""
        pipeline = AsyncMock()
        call_count = 0

        async def side_effect(query: str) -> GenerationResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Pipeline failed")
            return GenerationResult(
                answer="RAG answer",
                sources=[RetrievedChunk(content="chunk1", chunk_index=0, document_id=uuid4())],
                model="test-model",
            )

        pipeline.query.side_effect = side_effect
        faith = FaithfulnessMetric(llm_provider=provider, model="m")
        runner = EvalRunner(pipeline=pipeline, faithfulness_metric=faith)

        result = await runner.run(dataset)

        assert len(result.results) == 2
        assert result.results[0].error is not None
        assert result.results[1].faithfulness_score == 0.8

    async def test_metric_error_does_not_stop_evaluation(
        self, pipeline: AsyncMock, provider: AsyncMock, dataset: EvalDataset
    ) -> None:
        """メトリクスエラーで全体が止まらないこと。"""
        call_count = 0

        async def side_effect(**kwargs: object) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("LLM failed")
            return _make_llm_response("Score: 0.9\nReason: OK.")

        provider.complete.side_effect = side_effect
        faith = FaithfulnessMetric(llm_provider=provider, model="m")
        runner = EvalRunner(pipeline=pipeline, faithfulness_metric=faith)

        result = await runner.run(dataset)

        assert len(result.results) == 2
        assert result.results[0].error is not None
        assert result.results[0].rag_answer == "RAG answer"
        assert result.results[1].faithfulness_score == 0.9

    async def test_error_field_contains_message(
        self, provider: AsyncMock, dataset: EvalDataset
    ) -> None:
        """エラー時に error フィールドにメッセージが格納されること。"""
        pipeline = AsyncMock()
        pipeline.query.side_effect = RuntimeError("Pipeline failed")
        runner = EvalRunner(pipeline=pipeline)

        result = await runner.run(dataset)

        assert result.results[0].error is not None
        assert "Pipeline failed" in result.results[0].error

    async def test_faithfulness_summary_is_computed(
        self, pipeline: AsyncMock, provider: AsyncMock, dataset: EvalDataset
    ) -> None:
        """faithfulness_summary が計算されること。"""
        faith = FaithfulnessMetric(llm_provider=provider, model="m")
        runner = EvalRunner(pipeline=pipeline, faithfulness_metric=faith)

        result = await runner.run(dataset)

        assert result.faithfulness_summary is not None
        assert result.faithfulness_summary.mean == 0.8
        assert result.faithfulness_summary.count == 2

    async def test_relevance_summary_is_computed(
        self, pipeline: AsyncMock, provider: AsyncMock, dataset: EvalDataset
    ) -> None:
        """relevance_summary が計算されること。"""
        rel = RelevanceMetric(llm_provider=provider, model="m")
        runner = EvalRunner(pipeline=pipeline, relevance_metric=rel)

        result = await runner.run(dataset)

        assert result.relevance_summary is not None
        assert result.relevance_summary.mean == 0.8
        assert result.relevance_summary.count == 2

    async def test_latency_summary_is_computed(
        self, pipeline: AsyncMock, dataset: EvalDataset
    ) -> None:
        """latency_summary が計算されること。"""
        runner = EvalRunner(pipeline=pipeline)
        result = await runner.run(dataset)

        assert result.latency_summary is not None
        assert result.latency_summary.count == 2

    async def test_summary_is_none_when_metric_not_provided(
        self, pipeline: AsyncMock, dataset: EvalDataset
    ) -> None:
        """メトリクスが None の場合 summary も None であること。"""
        runner = EvalRunner(pipeline=pipeline)
        result = await runner.run(dataset)

        assert result.faithfulness_summary is None
        assert result.relevance_summary is None

    async def test_summary_excludes_errored_examples(
        self, provider: AsyncMock, dataset: EvalDataset
    ) -> None:
        """エラーの example は summary の計算から除外されること。"""
        pipeline = AsyncMock()
        call_count = 0

        async def side_effect(query: str) -> GenerationResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("fail")
            return GenerationResult(
                answer="RAG answer",
                sources=[RetrievedChunk(content="chunk1", chunk_index=0, document_id=uuid4())],
                model="test-model",
            )

        pipeline.query.side_effect = side_effect
        provider.complete.return_value = _make_llm_response("Score: 0.9\nReason: OK.")
        faith = FaithfulnessMetric(llm_provider=provider, model="m")
        runner = EvalRunner(pipeline=pipeline, faithfulness_metric=faith)

        result = await runner.run(dataset)

        assert result.faithfulness_summary is not None
        assert result.faithfulness_summary.count == 1
        assert result.faithfulness_summary.mean == 0.9

    async def test_empty_dataset_returns_empty_results(self) -> None:
        """空データセットで空の results が返ること。"""
        dataset = EvalDataset(name="empty", examples=[])
        runner = EvalRunner(pipeline=AsyncMock())

        result = await runner.run(dataset)

        assert result.results == []
        assert result.faithfulness_summary is None
        assert result.relevance_summary is None

    async def test_expected_answer_is_preserved(self, pipeline: AsyncMock) -> None:
        """expected_answer が結果に含まれること。"""
        dataset = EvalDataset(
            name="test",
            examples=[EvalExample(query="q", expected_answer="expected")],
        )
        runner = EvalRunner(pipeline=pipeline)
        result = await runner.run(dataset)

        assert result.results[0].expected_answer == "expected"
