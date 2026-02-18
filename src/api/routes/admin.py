"""Admin monitoring endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from src.api.dependencies import get_cost_tracker
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
