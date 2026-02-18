"""Tests for latency metric."""

from __future__ import annotations

import asyncio

import pytest
from src.eval import MetricError
from src.eval.metrics.latency import LatencyResult, measure_latency

# =============================================================================
# LatencyResult
# =============================================================================


class TestLatencyResult:
    """LatencyResult モデルのテスト。"""

    def test_creates_with_duration(self) -> None:
        """duration_seconds で生成できること。"""
        result = LatencyResult(duration_seconds=0.5)

        assert result.duration_seconds == 0.5

    def test_duration_is_float(self) -> None:
        """duration_seconds が float であること。"""
        result = LatencyResult(duration_seconds=1)

        assert isinstance(result.duration_seconds, float)


# =============================================================================
# measure_latency
# =============================================================================


class TestMeasureLatency:
    """measure_latency() のテスト。"""

    async def test_returns_latency_result(self) -> None:
        """LatencyResult が返ること。"""

        async def dummy() -> str:
            return "ok"

        result = await measure_latency(dummy)

        assert isinstance(result, LatencyResult)

    async def test_duration_is_positive(self) -> None:
        """duration_seconds が正の値であること。"""

        async def dummy() -> str:
            return "ok"

        result = await measure_latency(dummy)

        assert result.duration_seconds > 0.0

    async def test_measures_actual_elapsed_time(self) -> None:
        """実際の経過時間を計測すること。"""

        async def slow() -> str:
            await asyncio.sleep(0.1)
            return "done"

        result = await measure_latency(slow)

        assert result.duration_seconds >= 0.08  # Allow some tolerance

    async def test_passes_args_to_callable(self) -> None:
        """引数が callable に渡されること。"""
        received: list[tuple[int, str]] = []

        async def func(a: int, b: str) -> None:
            received.append((a, b))

        await measure_latency(func, 42, "hello")

        assert received == [(42, "hello")]

    async def test_passes_kwargs_to_callable(self) -> None:
        """キーワード引数が callable に渡されること。"""
        received: dict[str, int] = {}

        async def func(x: int = 0, y: int = 0) -> None:
            received["x"] = x
            received["y"] = y

        await measure_latency(func, x=10, y=20)

        assert received == {"x": 10, "y": 20}

    async def test_wraps_exception_in_metric_error(self) -> None:
        """callable の例外が MetricError でラップされること。"""
        original_error = ValueError("something went wrong")

        async def failing() -> None:
            raise original_error

        with pytest.raises(MetricError) as exc_info:
            await measure_latency(failing)

        assert exc_info.value.__cause__ is original_error

    async def test_works_with_sync_callable(self) -> None:
        """同期関数も計測できること。"""

        def sync_func() -> str:
            return "sync"

        result = await measure_latency(sync_func)

        assert isinstance(result, LatencyResult)
        assert result.duration_seconds > 0.0
