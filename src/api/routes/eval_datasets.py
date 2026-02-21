"""評価データセット CRUD + 合成データ生成エンドポイント。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.api.dependencies import get_synthetic_data_generator
from src.api.routes.eval import EvalExampleInput
from src.db.models import EvalDatasetRecord, EvalExampleRecord
from src.db.session import get_session
from src.eval import SyntheticDataError
from src.eval.synthetic_data import SyntheticDataGenerator
from src.security.permission import Permission, require_permission

router = APIRouter(prefix="/eval/datasets")


# --- Request schemas ---


class CreateEvalDatasetRequest(BaseModel):
    """データセット作成リクエスト。"""

    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    examples: list[EvalExampleInput] = Field(min_length=1)


class GenerateSyntheticDatasetRequest(BaseModel):
    """合成データセット生成リクエスト。"""

    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    text: str = Field(min_length=1)
    num_pairs: int | None = Field(default=None, ge=1, le=50)


# --- Response schemas ---


class EvalExampleResponse(BaseModel):
    """評価サンプルレスポンス。"""

    id: uuid.UUID
    query: str
    expected_answer: str | None
    created_at: datetime


class EvalDatasetResponse(BaseModel):
    """データセット一覧レスポンス (examples なし)。"""

    id: uuid.UUID
    name: str
    description: str | None
    example_count: int
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime


class EvalDatasetDetailResponse(BaseModel):
    """データセット詳細レスポンス (examples 付き)。"""

    id: uuid.UUID
    name: str
    description: str | None
    examples: list[EvalExampleResponse]
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime


# --- Helpers ---


def _build_detail_response(
    dataset: EvalDatasetRecord,
    examples: list[EvalExampleRecord],
) -> EvalDatasetDetailResponse:
    """EvalDatasetRecord + examples から詳細レスポンスを構築する。"""
    return EvalDatasetDetailResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        examples=[
            EvalExampleResponse(
                id=ex.id,
                query=ex.query,
                expected_answer=ex.expected_answer,
                created_at=ex.created_at,
            )
            for ex in examples
        ],
        created_by=dataset.created_by,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
    )


# --- Endpoints ---


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_eval_dataset(
    request: CreateEvalDatasetRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[None, Depends(require_permission(Permission.EVAL_RUN))],
) -> EvalDatasetDetailResponse:
    """データセットを作成する。"""
    # require_permission returns TokenPayload when successful
    from src.api.middleware.auth import TokenPayload

    current_user: TokenPayload = user  # type: ignore[assignment]

    dataset = EvalDatasetRecord(
        name=request.name,
        description=request.description,
        created_by=uuid.UUID(current_user.sub),
    )
    session.add(dataset)

    examples: list[EvalExampleRecord] = []
    for ex in request.examples:
        record = EvalExampleRecord(
            dataset_id=dataset.id,
            query=ex.query,
            expected_answer=ex.expected_answer,
        )
        session.add(record)
        examples.append(record)

    try:
        await session.commit()
    except IntegrityError as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Dataset name '{request.name}' already exists",
        ) from e

    return _build_detail_response(dataset, examples)


@router.get("")
async def list_eval_datasets(
    session: Annotated[AsyncSession, Depends(get_session)],
    _user: Annotated[None, Depends(require_permission(Permission.EVAL_READ))],
) -> list[EvalDatasetResponse]:
    """データセット一覧を取得する。"""
    stmt = (
        select(
            EvalDatasetRecord,
            func.count(EvalExampleRecord.id),  # type: ignore[arg-type]  # SQLModel stub mismatch
        )
        .outerjoin(
            EvalExampleRecord,
            EvalDatasetRecord.id == EvalExampleRecord.dataset_id,  # type: ignore[arg-type]
        )
        .group_by(EvalDatasetRecord.id)  # type: ignore[arg-type]
    )
    result = await session.exec(stmt)
    rows = result.all()

    return [
        EvalDatasetResponse(
            id=ds.id,
            name=ds.name,
            description=ds.description,
            example_count=count,
            created_by=ds.created_by,
            created_at=ds.created_at,
            updated_at=ds.updated_at,
        )
        for ds, count in rows
    ]


@router.get("/{dataset_id}")
async def get_eval_dataset(
    dataset_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    _user: Annotated[None, Depends(require_permission(Permission.EVAL_READ))],
) -> EvalDatasetDetailResponse:
    """データセット詳細を取得する。"""
    dataset = await session.get(EvalDatasetRecord, dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found",
        )

    stmt = select(EvalExampleRecord).where(EvalExampleRecord.dataset_id == dataset_id)
    result = await session.exec(stmt)
    examples = result.all()

    return _build_detail_response(dataset, list(examples))


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_eval_dataset(
    dataset_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    _user: Annotated[None, Depends(require_permission(Permission.EVAL_RUN))],
) -> Response:
    """データセットを削除する (CASCADE で examples も削除)。"""
    dataset = await session.get(EvalDatasetRecord, dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found",
        )

    await session.delete(dataset)
    await session.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/generate", status_code=status.HTTP_201_CREATED)
async def generate_synthetic_dataset(
    request: GenerateSyntheticDatasetRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    generator: Annotated[SyntheticDataGenerator, Depends(get_synthetic_data_generator)],
    user: Annotated[None, Depends(require_permission(Permission.EVAL_RUN))],
) -> EvalDatasetDetailResponse:
    """LLM で合成データを生成し、データセットとして保存する。"""
    from src.api.middleware.auth import TokenPayload

    current_user: TokenPayload = user  # type: ignore[assignment]

    try:
        eval_dataset = await generator.generate(
            text=request.text,
            num_pairs=request.num_pairs,
        )
    except SyntheticDataError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

    dataset = EvalDatasetRecord(
        name=request.name,
        description=request.description,
        created_by=uuid.UUID(current_user.sub),
    )
    session.add(dataset)

    examples: list[EvalExampleRecord] = []
    for ex in eval_dataset.examples:
        record = EvalExampleRecord(
            dataset_id=dataset.id,
            query=ex.query,
            expected_answer=ex.expected_answer,
        )
        session.add(record)
        examples.append(record)

    try:
        await session.commit()
    except IntegrityError as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Dataset name '{request.name}' already exists",
        ) from e

    return _build_detail_response(dataset, examples)
