"""Prometheus metrics collection for LLM operations.

Provides:
- ``LLMMetrics`` — container for all Prometheus metrics with convenience
  recording helpers and an ``in_progress`` context manager.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class LLMMetrics:
    """Prometheus metrics for LLM request tracking.

    Each instance owns its own Prometheus collectors, registered against the
    supplied *registry* (or the global default registry when omitted).

    Args:
        registry: Optional ``CollectorRegistry``.  Pass a fresh registry in
            tests to avoid cross-test contamination.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        kwargs: dict[str, Any] = {}
        if registry is not None:
            kwargs["registry"] = registry

        self._request_counter: Counter = Counter(
            "llm_requests_total",
            "Total number of LLM requests",
            labelnames=["provider", "model", "status"],
            **kwargs,
        )

        self._latency_histogram: Histogram = Histogram(
            "llm_request_duration_seconds",
            "LLM request latency in seconds",
            labelnames=["provider", "model"],
            **kwargs,
        )

        self._token_counter: Counter = Counter(
            "llm_tokens_total",
            "Total number of tokens consumed",
            labelnames=["provider", "model", "token_type"],
            **kwargs,
        )

        self._error_counter: Counter = Counter(
            "llm_errors_total",
            "Total number of LLM errors",
            labelnames=["provider", "error_type"],
            **kwargs,
        )

        self._in_progress_gauge: Gauge = Gauge(
            "llm_requests_in_progress",
            "Number of LLM requests currently in progress",
            labelnames=["provider", "model"],
            **kwargs,
        )

    # -- request counter -----------------------------------------------------

    def record_request(self, *, provider: str, model: str, status: str) -> None:
        """Increment the request counter.

        Args:
            provider: LLM provider name (e.g. ``"openrouter"``).
            model: Model identifier.
            status: Outcome — ``"success"`` or ``"error"``.
        """
        self._request_counter.labels(provider=provider, model=model, status=status).inc()

    def get_request_count(self, *, provider: str, model: str, status: str) -> float:
        """Return the current value of the request counter for the given labels."""
        value: float = self._request_counter.labels(
            provider=provider, model=model, status=status
        )._value.get()
        return value

    # -- latency histogram ---------------------------------------------------

    def record_latency(self, *, provider: str, model: str, duration_seconds: float) -> None:
        """Observe a request latency value.

        Args:
            provider: LLM provider name.
            model: Model identifier.
            duration_seconds: Elapsed time in seconds.
        """
        self._latency_histogram.labels(provider=provider, model=model).observe(duration_seconds)

    def get_latency_count(self, *, provider: str, model: str) -> int:
        """Return the number of observations for the given labels."""
        for metric in self._latency_histogram.collect():
            for sample in metric.samples:
                if (
                    sample.name == "llm_request_duration_seconds_count"
                    and sample.labels.get("provider") == provider
                    and sample.labels.get("model") == model
                ):
                    return int(sample.value)
        return 0

    def get_latency_sum(self, *, provider: str, model: str) -> float:
        """Return the sum of all observed latencies for the given labels."""
        for metric in self._latency_histogram.collect():
            for sample in metric.samples:
                if (
                    sample.name == "llm_request_duration_seconds_sum"
                    and sample.labels.get("provider") == provider
                    and sample.labels.get("model") == model
                ):
                    return float(sample.value)
        return 0.0

    # -- token counter -------------------------------------------------------

    def record_tokens(
        self,
        *,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Record token usage for a single request.

        Args:
            provider: LLM provider name.
            model: Model identifier.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
        """
        self._token_counter.labels(provider=provider, model=model, token_type="prompt").inc(
            prompt_tokens
        )
        self._token_counter.labels(provider=provider, model=model, token_type="completion").inc(
            completion_tokens
        )

    def get_token_count(self, *, provider: str, model: str, token_type: str) -> float:
        """Return the current token count for the given labels."""
        value: float = self._token_counter.labels(
            provider=provider, model=model, token_type=token_type
        )._value.get()
        return value

    # -- error counter -------------------------------------------------------

    def record_error(self, *, provider: str, error_type: str) -> None:
        """Increment the error counter.

        Args:
            provider: LLM provider name.
            error_type: Error category (e.g. ``"timeout"``, ``"rate_limit"``).
        """
        self._error_counter.labels(provider=provider, error_type=error_type).inc()

    def get_error_count(self, *, provider: str, error_type: str) -> float:
        """Return the current error count for the given labels."""
        value: float = self._error_counter.labels(
            provider=provider, error_type=error_type
        )._value.get()
        return value

    # -- in-progress gauge ---------------------------------------------------

    def track_in_progress(self, *, provider: str, model: str, delta: int) -> None:
        """Adjust the in-progress gauge by *delta* (``+1`` or ``-1``).

        Args:
            provider: LLM provider name.
            model: Model identifier.
            delta: Value to add (positive) or subtract (negative).
        """
        self._in_progress_gauge.labels(provider=provider, model=model).inc(delta)

    def get_in_progress(self, *, provider: str, model: str) -> float:
        """Return the current in-progress count for the given labels."""
        value: float = self._in_progress_gauge.labels(provider=provider, model=model)._value.get()
        return value

    @contextmanager
    def in_progress(self, *, provider: str, model: str) -> Generator[None, None, None]:
        """Context manager that increments the gauge on entry and decrements on exit.

        The gauge is decremented even when the block raises an exception.

        Args:
            provider: LLM provider name.
            model: Model identifier.
        """
        self.track_in_progress(provider=provider, model=model, delta=1)
        try:
            yield
        finally:
            self.track_in_progress(provider=provider, model=model, delta=-1)
