"""LLMプロバイダラッパー: 外部LLM APIへの送信前にPIIをマスクする。

任意の ``LLMProvider`` をラップし、``complete()`` / ``stream()`` に渡される
メッセージ中のPIIを検出・マスクしてから内部プロバイダに委譲する。
レスポンスはマスクせずそのまま返す (LLMはPIIを受け取っていないため生成もしない)。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from src.llm.providers.base import ChatMessage, LLMChunk, LLMProvider, LLMResponse
from src.monitoring.logger import get_logger
from src.security.pii_detector import PIIDetector

logger = get_logger()


class PIISanitizingProvider:
    """PII をマスクしてから内部 LLM プロバイダに委譲するラッパー。

    Args:
        inner: ラップ対象の LLM プロバイダ。
        enabled: False の場合はマスクせずパススルーする。
    """

    def __init__(self, inner: LLMProvider, enabled: bool = True) -> None:
        self._inner = inner
        self._enabled = enabled
        self._detector = PIIDetector()

    def _mask_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """メッセージ中のPIIをマスクする。PIIが検出された場合はログに種別を記録する。"""
        if not self._enabled:
            return messages

        masked: list[ChatMessage] = []
        all_pii_types: list[str] = []

        for msg in messages:
            result = self._detector.detect(msg.content)
            if result.has_pii:
                all_pii_types.extend(m.pii_type.value for m in result.matches)
                masked.append(ChatMessage(role=msg.role, content=result.masked_text))
            else:
                masked.append(msg)

        if all_pii_types:
            logger.info("pii_masked_for_llm", pii_types=all_pii_types)

        return masked

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """メッセージのPIIをマスクしてから内部プロバイダの complete() を呼び出す。"""
        masked_messages = self._mask_messages(messages)
        return await self._inner.complete(masked_messages, model, **kwargs)

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMChunk, None]:
        """メッセージのPIIをマスクしてから内部プロバイダの stream() を呼び出す。"""
        masked_messages = self._mask_messages(messages)
        async for chunk in self._inner.stream(masked_messages, model, **kwargs):
            yield chunk
