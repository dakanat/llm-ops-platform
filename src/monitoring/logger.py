"""Structured logging setup using structlog with JSON output.

Provides:
- ``setup_logging(log_level)`` — configure structlog + stdlib logging
- ``get_logger(**initial_bindings)`` — obtain a typed BoundLogger
- ``request_id_ctx_var`` — ContextVar for propagating request IDs downstream
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import cast

import structlog

request_id_ctx_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog and stdlib logging for JSON structured output.

    Args:
        log_level: Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, log_level.upper()))


def get_logger(**initial_bindings: object) -> structlog.stdlib.BoundLogger:
    """Return a structlog BoundLogger with optional initial bindings.

    Args:
        **initial_bindings: Key-value pairs to bind to the logger.

    Returns:
        A configured BoundLogger instance.
    """
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(**initial_bindings))
