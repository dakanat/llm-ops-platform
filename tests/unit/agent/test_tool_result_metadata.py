"""ToolResult.metadata と AgentStep.metadata のテスト。"""

from __future__ import annotations

from typing import Any

from src.agent.state import AgentStep
from src.agent.tools.base import ToolResult


class TestToolResultMetadata:
    """ToolResult の metadata フィールドを検証する。"""

    def test_metadata_defaults_to_none(self) -> None:
        result = ToolResult(output="ok")
        assert result.metadata is None

    def test_metadata_accepts_dict(self) -> None:
        meta: dict[str, Any] = {"sources": [{"id": "1"}]}
        result = ToolResult(output="ok", metadata=meta)
        assert result.metadata == meta

    def test_metadata_preserved_in_model_dump(self) -> None:
        meta: dict[str, Any] = {"key": "value"}
        result = ToolResult(output="ok", metadata=meta)
        dumped = result.model_dump()
        assert dumped["metadata"] == meta


class TestAgentStepMetadata:
    """AgentStep の metadata フィールドを検証する。"""

    def test_metadata_defaults_to_none(self) -> None:
        step = AgentStep(thought="thinking")
        assert step.metadata is None

    def test_metadata_accepts_dict(self) -> None:
        meta: dict[str, Any] = {"sources": [{"id": "1"}]}
        step = AgentStep(thought="thinking", metadata=meta)
        assert step.metadata == meta
