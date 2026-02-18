"""Evaluation execution endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_eval_runner
from src.eval import EvalError
from src.eval.datasets import EvalDataset, EvalExample
from src.eval.runner import EvalRunner, EvalRunResult
from src.security.permission import Permission, require_permission

router = APIRouter(prefix="/eval")


class EvalExampleInput(BaseModel):
    """評価サンプル入力。"""

    query: str
    context: str
    answer: str
    expected_answer: str | None = None


class ExampleResultResponse(BaseModel):
    """1件の評価結果レスポンス。"""

    query: str
    context: str
    answer: str
    expected_answer: str | None = None
    faithfulness_score: float | None = None
    relevance_score: float | None = None
    latency_seconds: float | None = None
    error: str | None = None


class MetricSummaryResponse(BaseModel):
    """メトリクスサマリレスポンス。"""

    mean: float
    count: int


class EvalRunRequest(BaseModel):
    """評価実行リクエスト。"""

    dataset_name: str = Field(min_length=1)
    examples: list[EvalExampleInput] = Field(min_length=1)


class EvalRunResponse(BaseModel):
    """評価実行レスポンス。"""

    dataset_name: str
    results: list[ExampleResultResponse]
    faithfulness_summary: MetricSummaryResponse | None = None
    relevance_summary: MetricSummaryResponse | None = None
    latency_summary: MetricSummaryResponse | None = None


def _map_eval_result(result: EvalRunResult) -> EvalRunResponse:
    """EvalRunResult を EvalRunResponse にマッピングする。"""
    return EvalRunResponse(
        dataset_name=result.dataset_name,
        results=[
            ExampleResultResponse(
                query=r.example.query,
                context=r.example.context,
                answer=r.example.answer,
                expected_answer=r.example.expected_answer,
                faithfulness_score=r.faithfulness_score,
                relevance_score=r.relevance_score,
                latency_seconds=r.latency_seconds,
                error=r.error,
            )
            for r in result.results
        ],
        faithfulness_summary=(
            MetricSummaryResponse(
                mean=result.faithfulness_summary.mean,
                count=result.faithfulness_summary.count,
            )
            if result.faithfulness_summary
            else None
        ),
        relevance_summary=(
            MetricSummaryResponse(
                mean=result.relevance_summary.mean,
                count=result.relevance_summary.count,
            )
            if result.relevance_summary
            else None
        ),
        latency_summary=(
            MetricSummaryResponse(
                mean=result.latency_summary.mean,
                count=result.latency_summary.count,
            )
            if result.latency_summary
            else None
        ),
    )


@router.post("/run")
async def eval_run(
    request: EvalRunRequest,
    runner: Annotated[EvalRunner, Depends(get_eval_runner)],
    _user: Annotated[None, Depends(require_permission(Permission.EVAL_RUN))],
) -> EvalRunResponse:
    """Run evaluation on the provided dataset."""
    dataset = EvalDataset(
        name=request.dataset_name,
        examples=[
            EvalExample(
                query=ex.query,
                context=ex.context,
                answer=ex.answer,
                expected_answer=ex.expected_answer,
            )
            for ex in request.examples
        ],
    )

    try:
        result = await runner.run(dataset)
    except EvalError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return _map_eval_result(result)
