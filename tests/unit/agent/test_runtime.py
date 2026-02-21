"""AgentRuntime のテスト。"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest
from src.agent import AgentError
from src.agent.planner import ReActPlanner
from src.agent.runtime import AgentResult, AgentRuntime
from src.agent.state import AgentStep
from src.agent.tools.base import ToolResult
from src.agent.tools.registry import ToolRegistry
from src.llm.providers.base import ChatMessage, LLMResponse, Role

# ── Stub Tool ────────────────────────────────────────────────


class _StubTool:
    """テスト用スタブツール。"""

    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = "Evaluate math."

    def __init__(
        self,
        result: str = "4",
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self._metadata = metadata

    async def execute(self, input_text: str) -> ToolResult:
        return ToolResult(output=self._result, error=self._error, metadata=self._metadata)


# ── Helper ───────────────────────────────────────────────────


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="test-model")


def _make_mock_provider(responses: list[str]) -> AsyncMock:
    """応答テキストリストから LLMProvider モックを作成する。"""
    provider = AsyncMock()
    provider.complete = AsyncMock(side_effect=[_make_llm_response(r) for r in responses])
    return provider


def _make_registry(*tools: Any) -> ToolRegistry:
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    return registry


# ── AgentResult model ────────────────────────────────────────


class TestAgentResult:
    """AgentResult モデルのテスト。"""

    def test_creation(self) -> None:
        result = AgentResult(
            answer="42",
            steps=[AgentStep(thought="done")],
            total_steps=1,
        )
        assert result.answer == "42"
        assert len(result.steps) == 1
        assert result.total_steps == 1

    def test_default_stopped_by_max_steps(self) -> None:
        result = AgentResult(answer="x", steps=[], total_steps=0)
        assert result.stopped_by_max_steps is False


# ── AgentRuntime init ────────────────────────────────────────


class TestAgentRuntimeInit:
    """AgentRuntime 初期化のテスト。"""

    def test_creates_with_required_args(self) -> None:
        provider = AsyncMock()
        registry = ToolRegistry()
        runtime = AgentRuntime(
            llm_provider=provider,
            model="test-model",
            tool_registry=registry,
        )
        assert runtime is not None

    def test_default_max_steps_is_10(self) -> None:
        runtime = AgentRuntime(
            llm_provider=AsyncMock(),
            model="m",
            tool_registry=ToolRegistry(),
        )
        assert runtime._max_steps == 10

    def test_custom_max_steps(self) -> None:
        runtime = AgentRuntime(
            llm_provider=AsyncMock(),
            model="m",
            tool_registry=ToolRegistry(),
            max_steps=5,
        )
        assert runtime._max_steps == 5

    def test_default_planner_created_when_none(self) -> None:
        runtime = AgentRuntime(
            llm_provider=AsyncMock(),
            model="m",
            tool_registry=ToolRegistry(),
        )
        assert isinstance(runtime._planner, ReActPlanner)


# ── Direct final answer ──────────────────────────────────────


class TestDirectFinalAnswer:
    """LLM が即座に Final Answer を返すケースのテスト。"""

    @pytest.mark.asyncio
    async def test_returns_agent_result(self) -> None:
        provider = _make_mock_provider(["Thought: I know this\nFinal Answer: Paris"])
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        result = await runtime.run("What is the capital of France?")
        assert isinstance(result, AgentResult)

    @pytest.mark.asyncio
    async def test_answer_matches(self) -> None:
        provider = _make_mock_provider(["Thought: I know\nFinal Answer: Paris"])
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        result = await runtime.run("Capital of France?")
        assert result.answer == "Paris"

    @pytest.mark.asyncio
    async def test_single_step(self) -> None:
        provider = _make_mock_provider(["Thought: Easy\nFinal Answer: 42"])
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        result = await runtime.run("q")
        assert result.total_steps == 1
        assert len(result.steps) == 1

    @pytest.mark.asyncio
    async def test_stopped_by_max_steps_false(self) -> None:
        provider = _make_mock_provider(["Thought: Done\nFinal Answer: ok"])
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        result = await runtime.run("q")
        assert result.stopped_by_max_steps is False


# ── Tool call flow ───────────────────────────────────────────


class TestToolCallFlow:
    """ツール呼び出しフローのテスト。"""

    @pytest.mark.asyncio
    async def test_calls_tool_from_registry(self) -> None:
        tool = _StubTool(result="4")
        provider = _make_mock_provider(
            [
                "Thought: Calculate\nAction: calculator\nAction Input: 2+2",
                "Thought: Got it\nFinal Answer: 4",
            ]
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
        )
        result = await runtime.run("What is 2+2?")
        assert result.answer == "4"

    @pytest.mark.asyncio
    async def test_tool_result_passed_as_observation(self) -> None:
        tool = _StubTool(result="computed: 4")
        provider = _make_mock_provider(
            [
                "Thought: Calc\nAction: calculator\nAction Input: 2+2",
                "Thought: Done\nFinal Answer: 4",
            ]
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
        )
        result = await runtime.run("q")
        assert result.steps[0].observation == "computed: 4"

    @pytest.mark.asyncio
    async def test_two_step_execution(self) -> None:
        tool = _StubTool(result="10")
        provider = _make_mock_provider(
            [
                "Thought: step1\nAction: calculator\nAction Input: 5+5",
                "Thought: step2\nAction: calculator\nAction Input: 10*2",
                "Thought: done\nFinal Answer: 20",
            ]
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
        )
        result = await runtime.run("q")
        assert result.total_steps == 3
        assert len(result.steps) == 3

    @pytest.mark.asyncio
    async def test_tool_execution_error_becomes_observation(self) -> None:
        tool = _StubTool(result="", error="division by zero")
        provider = _make_mock_provider(
            [
                "Thought: Calc\nAction: calculator\nAction Input: 1/0",
                "Thought: Error occurred\nFinal Answer: Cannot divide by zero",
            ]
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
        )
        result = await runtime.run("q")
        assert result.steps[0].is_error is True
        assert "division by zero" in result.steps[0].observation  # type: ignore[operator]


# ── Tool not found ───────────────────────────────────────────


class TestToolNotFound:
    """存在しないツール呼び出しのテスト。"""

    @pytest.mark.asyncio
    async def test_unknown_tool_produces_error_observation(self) -> None:
        provider = _make_mock_provider(
            [
                "Thought: Search\nAction: nonexistent\nAction Input: query",
                "Thought: Let me answer directly\nFinal Answer: I don't know",
            ]
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        result = await runtime.run("q")
        assert result.steps[0].is_error is True
        assert "nonexistent" in result.steps[0].observation  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_loop_continues_after_unknown_tool(self) -> None:
        provider = _make_mock_provider(
            [
                "Thought: Try\nAction: unknown_tool\nAction Input: x",
                "Thought: OK\nFinal Answer: fallback",
            ]
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        result = await runtime.run("q")
        assert result.answer == "fallback"
        assert result.total_steps == 2


# ── Max steps ────────────────────────────────────────────────


class TestMaxSteps:
    """最大ステップ数制限のテスト。"""

    @pytest.mark.asyncio
    async def test_stops_at_limit(self) -> None:
        tool = _StubTool(result="ok")
        # LLM always requests tool — never gives final answer
        responses = [f"Thought: step{i}\nAction: calculator\nAction Input: {i}" for i in range(3)]
        provider = _make_mock_provider(responses)
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
            max_steps=3,
        )
        result = await runtime.run("q")
        assert result.stopped_by_max_steps is True

    @pytest.mark.asyncio
    async def test_total_steps_equals_max(self) -> None:
        tool = _StubTool(result="ok")
        responses = [f"Thought: s{i}\nAction: calculator\nAction Input: {i}" for i in range(2)]
        provider = _make_mock_provider(responses)
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
            max_steps=2,
        )
        result = await runtime.run("q")
        assert result.total_steps == 2

    @pytest.mark.asyncio
    async def test_all_steps_recorded(self) -> None:
        tool = _StubTool(result="ok")
        responses = [f"Thought: s{i}\nAction: calculator\nAction Input: {i}" for i in range(2)]
        provider = _make_mock_provider(responses)
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
            max_steps=2,
        )
        result = await runtime.run("q")
        assert len(result.steps) == 2


# ── LLM error ────────────────────────────────────────────────


class TestLLMError:
    """LLM 呼び出し失敗のテスト。"""

    @pytest.mark.asyncio
    async def test_wraps_in_agent_error_with_cause(self) -> None:
        original = RuntimeError("API down")
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=original)
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        with pytest.raises(AgentError) as exc_info:
            await runtime.run("q")
        assert exc_info.value.__cause__ is original


# ── LLM call sequence ───────────────────────────────────────


class TestLLMCallSequence:
    """LLM 呼び出しメッセージの検証テスト。"""

    @pytest.mark.asyncio
    async def test_first_call_has_system_and_user(self) -> None:
        provider = _make_mock_provider(["Thought: Done\nFinal Answer: ok"])
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        await runtime.run("Hello")
        call_args = provider.complete.call_args_list[0]
        messages: list[ChatMessage] = call_args[0][0]
        assert messages[0].role == Role.system
        assert messages[1].role == Role.user
        assert "Hello" in messages[1].content

    @pytest.mark.asyncio
    async def test_second_call_includes_observation(self) -> None:
        tool = _StubTool(result="42")
        provider = _make_mock_provider(
            [
                "Thought: Calc\nAction: calculator\nAction Input: 6*7",
                "Thought: Got it\nFinal Answer: 42",
            ]
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
        )
        await runtime.run("q")
        call_args = provider.complete.call_args_list[1]
        messages: list[ChatMessage] = call_args[0][0]
        # Should have: system, user, assistant (step 1), user (observation)
        assert len(messages) == 4
        assert "Observation:" in messages[3].content
        assert "42" in messages[3].content


# ── Metadata propagation ─────────────────────────────────────


class TestMetadataPropagation:
    """ツール実行結果の metadata が AgentStep に伝播されることを検証する。"""

    @pytest.mark.asyncio
    async def test_tool_metadata_propagated_to_step(self) -> None:
        meta = {"sources": [{"id": "doc-1"}]}
        tool = _StubTool(result="found it", metadata=meta)
        provider = _make_mock_provider(
            [
                "Thought: Search\nAction: calculator\nAction Input: query",
                "Thought: Done\nFinal Answer: result",
            ]
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
        )
        result = await runtime.run("q")
        assert result.steps[0].metadata == meta

    @pytest.mark.asyncio
    async def test_no_metadata_when_tool_returns_none(self) -> None:
        tool = _StubTool(result="4")
        provider = _make_mock_provider(
            [
                "Thought: Calc\nAction: calculator\nAction Input: 2+2",
                "Thought: Done\nFinal Answer: 4",
            ]
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
        )
        result = await runtime.run("q")
        assert result.steps[0].metadata is None

    @pytest.mark.asyncio
    async def test_error_tool_has_no_metadata(self) -> None:
        tool = _StubTool(result="", error="fail", metadata={"should": "ignore"})
        provider = _make_mock_provider(
            [
                "Thought: Try\nAction: calculator\nAction Input: bad",
                "Thought: Done\nFinal Answer: error",
            ]
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
        )
        result = await runtime.run("q")
        assert result.steps[0].metadata is None
