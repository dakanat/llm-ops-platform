"""Web Eval dashboard routes."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.api.dependencies import get_eval_runner
from src.db.models import EvalDatasetRecord, EvalExampleRecord
from src.eval import EvalError
from src.eval.datasets import EvalDataset, EvalExample
from src.eval.runner import EvalRunner, EvalRunResult
from src.web.dependencies import CurrentWebUser
from src.web.templates import templates

router = APIRouter(prefix="/web")


async def _get_session() -> AsyncSession:
    """Get an async database session. Separated for testability."""
    from src.db.session import engine

    return AsyncSession(engine)


async def _run_eval(
    runner: EvalRunner,
    session: AsyncSession,
    dataset_id: UUID,
) -> EvalRunResult:
    """Load dataset from DB and execute evaluation. Extracted for testability."""
    dataset_record = await session.get(EvalDatasetRecord, dataset_id)
    if dataset_record is None:
        raise DatasetNotFoundError(dataset_id)

    stmt = select(EvalExampleRecord).where(EvalExampleRecord.dataset_id == dataset_id)
    result = await session.exec(stmt)
    records = result.all()

    examples = [EvalExample(query=r.query, expected_answer=r.expected_answer) for r in records]
    dataset = EvalDataset(name=dataset_record.name, examples=examples)
    return await runner.run(dataset)


class DatasetNotFoundError(Exception):
    """Raised when a dataset is not found in the database."""

    def __init__(self, dataset_id: UUID) -> None:
        super().__init__(f"Dataset {dataset_id} not found")


@router.get("/eval", response_class=HTMLResponse)
async def eval_page(request: Request, user: CurrentWebUser) -> Response:
    """Display the Eval dashboard."""
    session = await _get_session()
    try:
        stmt = (
            select(
                EvalDatasetRecord,
                func.count(EvalExampleRecord.id),  # type: ignore[arg-type]
            )
            .outerjoin(
                EvalExampleRecord,
                EvalDatasetRecord.id == EvalExampleRecord.dataset_id,  # type: ignore[arg-type]
            )
            .group_by(EvalDatasetRecord.id)  # type: ignore[arg-type]
        )
        result = await session.exec(stmt)
        rows = list(result.all())
    finally:
        await session.close()

    datasets = [{"dataset": ds, "example_count": count} for ds, count in rows]

    return templates.TemplateResponse(
        request,
        "eval/page.html",
        {"user": user, "active_page": "eval", "datasets": datasets},
    )


@router.post("/eval/run", response_class=HTMLResponse)
async def eval_run(
    request: Request,
    user: CurrentWebUser,
    runner: Annotated[EvalRunner, Depends(get_eval_runner)],
) -> Response:
    """Run evaluation and return results as HTML."""
    form = await request.form()
    dataset_id_raw = str(form.get("dataset_id", "")).strip()

    if not dataset_id_raw:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": "Please select a dataset"},
        )

    try:
        dataset_id = UUID(dataset_id_raw)
    except ValueError:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": "Invalid dataset ID"},
        )

    session = await _get_session()
    try:
        result = await _run_eval(runner, session, dataset_id)
    except DatasetNotFoundError:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": "Dataset not found"},
        )
    except EvalError as e:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": f"Eval error: {e}"},
        )
    finally:
        await session.close()

    return templates.TemplateResponse(
        request,
        "eval/run_result.html",
        {"result": result},
    )
