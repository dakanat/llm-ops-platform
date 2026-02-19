"""Web Eval dashboard routes."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse

from src.api.dependencies import get_eval_runner
from src.eval import EvalError
from src.eval.datasets import EvalDataset, EvalExample
from src.eval.runner import EvalRunner, EvalRunResult
from src.web.dependencies import CurrentWebUser
from src.web.templates import templates

router = APIRouter(prefix="/web")


async def _run_eval(
    runner: EvalRunner,
    dataset_name: str,
    examples_data: list[dict[str, str]],
) -> EvalRunResult:
    """Execute evaluation. Extracted for testability."""
    examples = [EvalExample(**ex) for ex in examples_data]
    dataset = EvalDataset(name=dataset_name, examples=examples)
    return await runner.run(dataset)


@router.get("/eval", response_class=HTMLResponse)
async def eval_page(request: Request, user: CurrentWebUser) -> Response:
    """Display the Eval dashboard."""
    return templates.TemplateResponse(
        request, "eval/page.html", {"user": user, "active_page": "eval"}
    )


@router.post("/eval/run", response_class=HTMLResponse)
async def eval_run(
    request: Request,
    user: CurrentWebUser,
    runner: Annotated[EvalRunner, Depends(get_eval_runner)],
) -> Response:
    """Run evaluation and return results as HTML."""
    form = await request.form()
    dataset_name = str(form.get("dataset_name", "")).strip()
    examples_raw = str(form.get("examples", "[]"))

    if not dataset_name:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": "Dataset name is required"},
        )

    try:
        examples_data: list[dict[str, str]] = json.loads(examples_raw)
    except json.JSONDecodeError:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": "Invalid examples JSON"},
        )

    if not examples_data:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": "At least one example is required"},
        )

    try:
        result = await _run_eval(runner, dataset_name, examples_data)
    except EvalError as e:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": f"Eval error: {e}"},
        )

    return templates.TemplateResponse(
        request,
        "eval/run_result.html",
        {
            "result": result,
        },
    )
