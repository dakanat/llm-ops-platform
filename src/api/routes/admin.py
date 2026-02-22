"""Admin monitoring endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.api.dependencies import get_cost_tracker
from src.db.models import AuditLog
from src.db.session import get_session
from src.monitoring.cost_tracker import CostTracker
from src.security.permission import Permission, require_permission

router = APIRouter(prefix="/admin")


class ModelCostSummaryResponse(BaseModel):
    """モデル別コストサマリレスポンス。"""

    cost: float
    requests: int


class CostReportResponse(BaseModel):
    """コストレポートレスポンス。"""

    total_cost: float
    model_costs: dict[str, ModelCostSummaryResponse]
    alert_triggered: bool


class AuditLogResponse(BaseModel):
    """監査ログレスポンス。"""

    id: uuid.UUID
    user_id: uuid.UUID
    action: str
    resource_type: str
    resource_id: str
    details: dict[str, Any]
    created_at: datetime


class AuditLogListResponse(BaseModel):
    """監査ログ一覧レスポンス。"""

    items: list[AuditLogResponse]
    total: int
    page: int
    page_size: int


@router.get("/metrics")
async def admin_metrics(
    _user: Annotated[None, Depends(require_permission(Permission.ADMIN_READ))],
) -> Response:
    """Return Prometheus metrics in exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/costs")
async def admin_costs(
    tracker: Annotated[CostTracker, Depends(get_cost_tracker)],
    _user: Annotated[None, Depends(require_permission(Permission.ADMIN_READ))],
) -> CostReportResponse:
    """Return cost report as JSON."""
    report = tracker.get_cost_report()
    return CostReportResponse(
        total_cost=report["total_cost"],
        model_costs={
            model: ModelCostSummaryResponse(
                cost=summary["cost"],
                requests=summary["requests"],
            )
            for model, summary in report["model_costs"].items()
        },
        alert_triggered=tracker.is_alert_triggered(),
    )


@router.get("/audit-logs")
async def list_audit_logs(
    session: Annotated[AsyncSession, Depends(get_session)],
    _user: Annotated[None, Depends(require_permission(Permission.ADMIN_READ))],
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
) -> AuditLogListResponse:
    """監査ログ一覧を返す (降順ソート、ページネーション付き)。"""
    count_stmt = select(func.count(AuditLog.id))  # type: ignore[arg-type]
    count_result = await session.exec(count_stmt)
    total = count_result.one()

    offset = (page - 1) * page_size
    items_stmt = (
        select(AuditLog)
        .order_by(AuditLog.created_at.desc())  # type: ignore[attr-defined]
        .offset(offset)
        .limit(page_size)
    )
    items_result = await session.exec(items_stmt)
    logs = items_result.all()

    return AuditLogListResponse(
        items=[
            AuditLogResponse(
                id=log.id,
                user_id=log.user_id,
                action=log.action,
                resource_type=log.resource_type,
                resource_id=log.resource_id,
                details=log.details,
                created_at=log.created_at,
            )
            for log in logs
        ],
        total=total,
        page=page,
        page_size=page_size,
    )
