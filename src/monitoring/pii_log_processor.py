"""structlog プロセッサ: ログ出力前にPIIをマスクする。

``event_dict`` 中の全文字列値をスキャンし、PIIが含まれていればマスクする。
構造的キー (timestamp, log_level 等) はスキップする。
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import structlog

from src.security.pii_detector import PIIDetector

# 構造的キー: PIIスキャンをスキップする
_EXEMPT_KEYS = frozenset(
    {
        "timestamp",
        "log_level",
        "event",
        "request_id",
        "logger_name",
        "level",
    }
)


def create_pii_masking_processor(
    enabled: bool = True,
) -> structlog.types.Processor:
    """PII マスキング用 structlog プロセッサを生成する。

    Args:
        enabled: False の場合はパススルーするプロセッサを返す。

    Returns:
        structlog プロセッサ関数。
    """
    detector = PIIDetector()

    def _pii_masking_processor(
        logger: Any,
        method_name: str,
        event_dict: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        if not enabled:
            return event_dict

        for key, value in event_dict.items():
            if key in _EXEMPT_KEYS:
                continue
            if isinstance(value, str) and value:
                event_dict[key] = detector.mask(value)

        return event_dict

    return _pii_masking_processor
