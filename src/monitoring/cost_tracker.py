"""LLM cost tracking and daily alert management.

Provides:
- ``ModelPricing`` — per-model pricing configuration.
- ``CostTracker`` — tracks cumulative cost per day, supports alert thresholds
  and per-model cost reports.
"""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel

from src.llm.providers.base import TokenUsage


class ModelCostSummary(TypedDict):
    """Per-model cost breakdown."""

    cost: float
    requests: int


class CostReport(TypedDict):
    """Daily cost report."""

    total_cost: float
    model_costs: dict[str, ModelCostSummary]


class ModelPricing(BaseModel):
    """Pricing for a single LLM model (USD per 1 million tokens).

    Attributes:
        input_cost_per_million: Cost per 1 M input (prompt) tokens.
        output_cost_per_million: Cost per 1 M output (completion) tokens.
    """

    input_cost_per_million: float
    output_cost_per_million: float


class _ModelCostRecord:
    """Internal accumulator for a single model."""

    __slots__ = ("cost", "requests")

    def __init__(self) -> None:
        self.cost: float = 0.0
        self.requests: int = 0

    def add(self, cost: float) -> None:
        self.cost += cost
        self.requests += 1


class CostTracker:
    """Tracks LLM API costs and checks daily alert thresholds.

    Args:
        alert_threshold_daily_usd: Daily cost in USD that triggers an alert.
    """

    def __init__(self, alert_threshold_daily_usd: float = 10.0) -> None:
        self.alert_threshold_daily_usd = alert_threshold_daily_usd
        self._pricing: dict[str, ModelPricing] = {}
        self._daily_total: float = 0.0
        self._model_costs: dict[str, _ModelCostRecord] = {}

    # -- pricing management --------------------------------------------------

    def register_pricing(self, model: str, pricing: ModelPricing) -> None:
        """Register pricing for a model.

        Args:
            model: Model identifier (e.g. ``"gpt-4"``).
            pricing: Pricing configuration for the model.
        """
        self._pricing[model] = pricing

    def get_pricing(self, model: str) -> ModelPricing | None:
        """Return pricing for *model*, or ``None`` if not registered."""
        return self._pricing.get(model)

    # -- cost calculation ----------------------------------------------------

    def calculate_cost(self, model: str, usage: TokenUsage) -> float:
        """Calculate the cost in USD for a given token usage.

        Returns ``0.0`` when the model has no registered pricing.

        Args:
            model: Model identifier.
            usage: Token usage from the LLM response.
        """
        pricing = self._pricing.get(model)
        if pricing is None:
            return 0.0

        input_cost = (usage.prompt_tokens / 1_000_000) * pricing.input_cost_per_million
        output_cost = (usage.completion_tokens / 1_000_000) * pricing.output_cost_per_million
        return input_cost + output_cost

    # -- daily tracking ------------------------------------------------------

    def record_cost(self, model: str, usage: TokenUsage) -> float:
        """Record a request's cost and return the computed cost value.

        Args:
            model: Model identifier.
            usage: Token usage from the LLM response.

        Returns:
            The cost in USD for this single request.
        """
        cost = self.calculate_cost(model, usage)
        self._daily_total += cost

        if model not in self._model_costs:
            self._model_costs[model] = _ModelCostRecord()
        self._model_costs[model].add(cost)

        return cost

    def get_daily_cost(self) -> float:
        """Return the accumulated daily cost in USD."""
        return self._daily_total

    def reset_daily(self) -> None:
        """Reset all daily cost accumulators to zero."""
        self._daily_total = 0.0
        self._model_costs.clear()

    # -- alerts --------------------------------------------------------------

    def is_alert_triggered(self) -> bool:
        """Return ``True`` when the daily cost meets or exceeds the threshold."""
        return self._daily_total >= self.alert_threshold_daily_usd

    # -- reporting -----------------------------------------------------------

    def get_cost_report(self) -> CostReport:
        """Return a cost report broken down by model.

        Returns:
            A ``CostReport`` with ``total_cost`` (float) and ``model_costs``
            (dict mapping model names to ``ModelCostSummary``).
        """
        model_costs: dict[str, ModelCostSummary] = {}
        for model, record in self._model_costs.items():
            model_costs[model] = ModelCostSummary(
                cost=record.cost,
                requests=record.requests,
            )

        return CostReport(
            total_cost=self._daily_total,
            model_costs=model_costs,
        )
