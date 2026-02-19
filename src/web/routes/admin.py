"""Web Admin dashboard routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse

from src.api.dependencies import get_cost_tracker
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
