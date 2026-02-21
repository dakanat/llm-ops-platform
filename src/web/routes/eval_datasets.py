"""Web Eval datasets CRUD routes."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse
from sqlalchemy.exc import IntegrityError
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.api.dependencies import get_synthetic_data_generator
from src.db.models import EvalDatasetRecord, EvalExampleRecord
from src.eval import SyntheticDataError
from src.eval.synthetic_data import SyntheticDataGenerator
from src.web.dependencies import CurrentWebUser
from src.web.templates import templates

router = APIRouter(prefix="/web")


async def _get_session() -> AsyncSession:
    """Get an async database session. Separated for testability."""
    from sqlmodel.ext.asyncio.session import AsyncSession as _AsyncSession

    from src.db.session import engine

    return _AsyncSession(engine)


async def _generate_and_save(
    generator: SyntheticDataGenerator,
    session: AsyncSession,
    name: str,
    description: str,
    text: str,
    num_pairs: int,
    created_by: UUID,
) -> EvalDatasetRecord:
    """Generate QA pairs and save as a dataset. Extracted for testability."""
    dataset = await generator.generate(text, num_pairs=num_pairs)

    record = EvalDatasetRecord(
        name=name,
        description=description or None,
        created_by=created_by,
    )
    session.add(record)
    await session.flush()

    for example in dataset.examples:
        example_record = EvalExampleRecord(
            dataset_id=record.id,
            query=example.query,
            expected_answer=example.expected_answer,
        )
        session.add(example_record)

    await session.commit()
    await session.refresh(record)
    return record


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


@router.post("/eval/datasets/generate", response_class=HTMLResponse)
async def eval_datasets_generate(
    request: Request,
    user: CurrentWebUser,
    generator: Annotated[SyntheticDataGenerator, Depends(get_synthetic_data_generator)],
) -> Response:
    """Generate a synthetic dataset from source text."""
    form = await request.form()
    name = str(form.get("name", "")).strip()
    description = str(form.get("description", "")).strip()
    text = str(form.get("text", "")).strip()
    num_pairs_raw = str(form.get("num_pairs", "5")).strip()

    if not name:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": "Dataset name is required"},
        )

    if not text:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": "Source text is required"},
        )

    try:
        num_pairs = int(num_pairs_raw)
        num_pairs = max(1, min(50, num_pairs))
    except ValueError:
        num_pairs = 5

    session = await _get_session()
    try:
        dataset = await _generate_and_save(
            generator=generator,
            session=session,
            name=name,
            description=description,
            text=text,
            num_pairs=num_pairs,
            created_by=UUID(user.sub),
        )
    except SyntheticDataError as e:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": f"Generation error: {e}"},
        )
    except IntegrityError:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": "A dataset with this name already exists"},
        )
    finally:
        await session.close()

    return Response(
        status_code=200,
        headers={"HX-Redirect": f"/web/eval/datasets/{dataset.id}"},
    )


@router.get("/eval/datasets/{dataset_id}", response_class=HTMLResponse)
async def eval_dataset_detail(dataset_id: str, request: Request, user: CurrentWebUser) -> Response:
    """Display a single dataset's details."""
    session = await _get_session()
    try:
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
