"""get_tool_registry 依存のテスト。"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from src.api.dependencies import get_tool_registry
from src.rag.pipeline import RAGPipeline


class TestGetToolRegistry:
    """get_tool_registry が全ツールを登録することを検証する。"""

    @pytest.mark.asyncio
    async def test_registers_calculator(self) -> None:
        pipeline = AsyncMock(spec=RAGPipeline)
        gen = get_tool_registry(pipeline=pipeline)
        registry = await gen.__anext__()
        assert registry.has("calculator")

    @pytest.mark.asyncio
    async def test_registers_search(self) -> None:
        pipeline = AsyncMock(spec=RAGPipeline)
        gen = get_tool_registry(pipeline=pipeline)
        registry = await gen.__anext__()
        assert registry.has("search")

    @pytest.mark.asyncio
    async def test_registers_exactly_two_tools(self) -> None:
        pipeline = AsyncMock(spec=RAGPipeline)
        gen = get_tool_registry(pipeline=pipeline)
        registry = await gen.__anext__()
        assert len(registry.list_tools()) == 2
