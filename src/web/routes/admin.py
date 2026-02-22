"""Web Admin dashboard routes."""

from __future__ import annotations

import math
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.api.dependencies import get_cost_tracker
from src.db.models import AuditLog
from src.db.session import get_session
from src.monitoring.cost_tracker import CostTracker
from src.security.permission import Permission, has_permission
from src.web.dependencies import CurrentWebUser
from src.web.templates import templates

router = APIRouter(prefix="/web")


def _check_admin(user: CurrentWebUser) -> None:
    """Raise 403 if user lacks admin:read permission."""
    if not has_permission(user.role, Permission.ADMIN_READ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permission required",
        )


@router.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, user: CurrentWebUser) -> Response:
    """Display the admin dashboard."""
    _check_admin(user)
    return templates.TemplateResponse(
        request, "admin/page.html", {"user": user, "active_page": "admin"}
    )


@router.get("/admin/costs", response_class=HTMLResponse)
async def admin_costs(
    request: Request,
    user: CurrentWebUser,
    tracker: Annotated[CostTracker, Depends(get_cost_tracker)],
) -> Response:
    """Return cost report as an HTML fragment."""
    _check_admin(user)
    report = tracker.get_cost_report()
    return templates.TemplateResponse(
        request,
        "admin/cost_report.html",
        {
            "total_cost": report["total_cost"],
            "model_costs": report["model_costs"],
            "alert_triggered": tracker.is_alert_triggered(),
        },
    )


@router.get("/admin/audit-logs", response_class=HTMLResponse)
async def admin_audit_logs(
    request: Request,
    user: CurrentWebUser,
    session: Annotated[AsyncSession, Depends(get_session)],
    page: int = 1,
) -> Response:
    """Return audit logs as an HTML fragment."""
    _check_admin(user)
    page_size = 20

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
    audit_logs = items_result.all()

    total_pages = math.ceil(total / page_size) if total > 0 else 1

    return templates.TemplateResponse(
        request,
        "admin/audit_logs.html",
        {
            "audit_logs": audit_logs,
            "page": page,
            "total_pages": total_pages,
        },
    )
