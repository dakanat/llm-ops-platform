"""Evaluation execution endpoint."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.api.dependencies import get_eval_runner
from src.db.models import EvalDatasetRecord, EvalExampleRecord
from src.db.session import get_session
from src.eval import EvalError
from src.eval.datasets import EvalDataset, EvalExample
from src.eval.runner import EvalRunner, EvalRunResult
from src.security.permission import Permission, require_permission

router = APIRouter(prefix="/eval")


class EvalExampleInput(BaseModel):
    """評価サンプル入力。"""

    query: str
    expected_answer: str | None = None


class ExampleResultResponse(BaseModel):
    """1件の評価結果レスポンス。"""

    query: str
    expected_answer: str | None = None
    rag_answer: str | None = None
    rag_context: str | None = None
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
    dataset_id: uuid.UUID | None = None
    examples: list[EvalExampleInput] | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> EvalRunRequest:
        if self.dataset_id and self.examples:
            raise ValueError("dataset_id and examples are mutually exclusive")
        if not self.dataset_id and not self.examples:
            raise ValueError("Either dataset_id or examples must be provided")
        return self


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
                query=r.query,
                expected_answer=r.expected_answer,
                rag_answer=r.rag_answer,
                rag_context=r.rag_context,
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
    session: Annotated[AsyncSession, Depends(get_session)],
) -> EvalRunResponse:
    """Run evaluation on the provided dataset."""
    if request.dataset_id:
        # Load examples from DB
        ds = await session.get(EvalDatasetRecord, request.dataset_id)
        if ds is None:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset {request.dataset_id} not found",
            )

        stmt = select(EvalExampleRecord).where(EvalExampleRecord.dataset_id == request.dataset_id)
        db_result = await session.exec(stmt)
        records = db_result.all()

        examples = [
            EvalExample(
                query=r.query,
                expected_answer=r.expected_answer,
            )
            for r in records
        ]
    else:
        examples = [
            EvalExample(
                query=ex.query,
                expected_answer=ex.expected_answer,
            )
            for ex in request.examples  # type: ignore[union-attr]
        ]

    dataset = EvalDataset(
        name=request.dataset_name,
        examples=examples,
    )

    try:
        run_result = await runner.run(dataset)
    except EvalError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return _map_eval_result(run_result)
