"""Web Eval datasets CRUD routes."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.models import EvalDatasetRecord, EvalExampleRecord
from src.web.dependencies import CurrentWebUser
from src.web.templates import templates

router = APIRouter(prefix="/web")


async def _get_session() -> AsyncSession:
    """Get an async database session. Separated for testability."""
    from sqlmodel.ext.asyncio.session import AsyncSession as _AsyncSession

    from src.db.session import engine

    return _AsyncSession(engine)


@router.get("/eval/datasets", response_class=HTMLResponse)
async def eval_datasets_list(request: Request, user: CurrentWebUser) -> Response:
    """Display the datasets list page."""
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
        "eval/dataset_list.html",
        {"user": user, "active_page": "eval", "datasets": datasets},
    )


@router.get("/eval/datasets/create", response_class=HTMLResponse)
async def eval_datasets_create_form(request: Request, user: CurrentWebUser) -> Response:
    """Display the dataset creation form."""
    return templates.TemplateResponse(
        request,
        "eval/dataset_form.html",
        {"user": user, "active_page": "eval"},
    )


@router.get("/eval/datasets/{dataset_id}", response_class=HTMLResponse)
async def eval_dataset_detail(dataset_id: str, request: Request, user: CurrentWebUser) -> Response:
    """Display a single dataset's details."""
    session = await _get_session()
    try:
        from uuid import UUID

        ds_uuid = UUID(dataset_id)
        dataset = await session.get(EvalDatasetRecord, ds_uuid)
        if dataset is None:
            return templates.TemplateResponse(
                request,
                "components/error_toast.html",
                {"error_message": "Dataset not found"},
                status_code=404,
            )

        stmt = select(EvalExampleRecord).where(EvalExampleRecord.dataset_id == ds_uuid)
        result = await session.exec(stmt)
        examples = result.all()
    finally:
        await session.close()

    return templates.TemplateResponse(
        request,
        "eval/dataset_detail.html",
        {
            "user": user,
            "active_page": "eval",
            "dataset": dataset,
            "examples": examples,
        },
    )
