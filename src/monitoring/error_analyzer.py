"""Error classification, recording, and analysis for LLM operations.

Provides:
- ``ErrorCategory`` — enumeration of error categories.
- ``ErrorRecord`` — immutable snapshot of a classified error.
- ``ErrorAnalyzer`` — stateful collector that classifies, records, and
  aggregates errors.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypedDict

import httpx
from pydantic import BaseModel, Field


class ErrorCategory(StrEnum):
    """High-level error categories for LLM operations."""

    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    UNKNOWN = "unknown"


class ErrorSummary(TypedDict):
    """Aggregated error summary."""

    total_errors: int
    by_category: dict[ErrorCategory, int]
    by_provider: dict[str, int]


class ErrorRecord(BaseModel):
    """Immutable snapshot of a classified error.

    Attributes:
        category: The classified error category.
        error_type: The Python exception class name.
        message: Human-readable error message.
        provider: LLM provider name (optional).
        timestamp: UTC timestamp of when the error was recorded.
    """

    category: ErrorCategory
    error_type: str
    message: str
    provider: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Classification mapping
# ---------------------------------------------------------------------------

_CATEGORY_MAP: list[tuple[type[BaseException], ErrorCategory]] = [
    (httpx.TimeoutException, ErrorCategory.TIMEOUT),
    (TimeoutError, ErrorCategory.TIMEOUT),
    (httpx.ConnectError, ErrorCategory.PROVIDER_ERROR),
    (ConnectionError, ErrorCategory.PROVIDER_ERROR),
    (PermissionError, ErrorCategory.AUTHENTICATION_ERROR),
    (ValueError, ErrorCategory.VALIDATION_ERROR),
    (TypeError, ErrorCategory.VALIDATION_ERROR),
]


class ErrorAnalyzer:
    """Classifies, records, and aggregates LLM-related errors."""

    def __init__(self) -> None:
        self._records: list[ErrorRecord] = []

    # -- classification ------------------------------------------------------

    @staticmethod
    def classify(error: BaseException) -> ErrorCategory:
        """Return the ``ErrorCategory`` for *error* based on its type.

        The mapping is checked from most-specific to least-specific; if no
        match is found the category defaults to ``UNKNOWN``.

        Args:
            error: The exception to classify.
        """
        for exc_type, category in _CATEGORY_MAP:
            if isinstance(error, exc_type):
                return category
        return ErrorCategory.UNKNOWN

    # -- recording -----------------------------------------------------------

    def record(self, error: BaseException, provider: str | None = None) -> ErrorRecord:
        """Classify and record an error.

        Args:
            error: The exception to record.
            provider: Optional LLM provider name.

        Returns:
            The created ``ErrorRecord``.
        """
        record = ErrorRecord(
            category=self.classify(error),
            error_type=type(error).__name__,
            message=str(error),
            provider=provider,
        )
        self._records.append(record)
        return record

    # -- queries -------------------------------------------------------------

    def total_error_count(self) -> int:
        """Return the total number of recorded errors."""
        return len(self._records)

    def get_recent(self, n: int) -> list[ErrorRecord]:
        """Return the *n* most recent errors, newest first.

        Args:
            n: Maximum number of records to return.
        """
        return list(reversed(self._records[-n:])) if self._records else []

    def get_summary(self) -> ErrorSummary:
        """Return an aggregated summary of all recorded errors.

        Returns:
            An ``ErrorSummary`` with ``total_errors``, ``by_category``, and
            ``by_provider``.
        """
        by_category: dict[ErrorCategory, int] = defaultdict(int)
        by_provider: dict[str, int] = defaultdict(int)

        for record in self._records:
            by_category[record.category] += 1
            if record.provider is not None:
                by_provider[record.provider] += 1

        return ErrorSummary(
            total_errors=len(self._records),
            by_category=dict(by_category),
            by_provider=dict(by_provider),
        )

    # -- management ----------------------------------------------------------

    def clear(self) -> None:
        """Remove all recorded errors."""
        self._records.clear()
