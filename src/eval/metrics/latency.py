"""レイテンシ計測メトリクス。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from src.eval import MetricError


class LatencyResult(BaseModel):
    """レイテンシ計測結果。

    Attributes:
        duration_seconds: 経過時間 (秒)。
    """

    duration_seconds: float


async def measure_latency(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> LatencyResult:
    """callable の実行時間を計測する。

    async / sync 両方の callable に対応する。

    Args:
        func: 計測対象の callable。
        *args: callable に渡す位置引数。
        **kwargs: callable に渡すキーワード引数。

    Returns:
        経過時間を含む LatencyResult。

    Raises:
        MetricError: callable が例外を送出した場合。
    """
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            await result
    except Exception as e:
        raise MetricError(str(e)) from e
    elapsed = time.perf_counter() - start

    return LatencyResult(duration_seconds=elapsed)
