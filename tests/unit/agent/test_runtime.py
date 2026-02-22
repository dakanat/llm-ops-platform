"""AgentRuntime のテスト。"""

from __future__ import annotations

from collections.abc import AsyncGenerator
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


# ── Conversation history ─────────────────────────────────────


class TestConversationHistory:
    """会話履歴パススルーのテスト。"""

    @pytest.mark.asyncio
    async def test_conversation_history_passed_to_llm(self) -> None:
        provider = _make_mock_provider(["Thought: Done\nFinal Answer: yes"])
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        history = [
            ChatMessage(role=Role.user, content="What is Python?"),
            ChatMessage(role=Role.assistant, content="A language."),
        ]
        await runtime.run("Tell me more", conversation_history=history)
        call_args = provider.complete.call_args_list[0]
        messages: list[ChatMessage] = call_args[0][0]
        # system + 2 history + user query
        assert len(messages) == 4
        assert messages[1].content == "What is Python?"
        assert messages[2].content == "A language."
        assert messages[3].content == "Tell me more"

    @pytest.mark.asyncio
    async def test_run_without_conversation_history(self) -> None:
        provider = _make_mock_provider(["Thought: Done\nFinal Answer: ok"])
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        result = await runtime.run("Hello")
        assert result.answer == "ok"
        call_args = provider.complete.call_args_list[0]
        messages: list[ChatMessage] = call_args[0][0]
        # system + user query only
        assert len(messages) == 2


# ── Streaming ────────────────────────────────────────────────


class TestRunStreaming:
    """run_streaming のテスト (provider.stream 使用)。"""

    @pytest.mark.asyncio
    async def test_streaming_yields_step_and_answer_events(self) -> None:
        from src.agent.runtime import AgentAnswerEndEvent, AgentStepEvent

        tool = _StubTool(result="4")
        provider = _make_stream_provider(
            [
                ["Thought: Calculate\nAction: calculator\nAction Input: 2+2"],
                ["Thought: Got it\nFinal Answer: 4"],
            ],
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
        )
        events = []
        async for event in runtime.run_streaming("What is 2+2?"):
            events.append(event)

        step_events = [e for e in events if isinstance(e, AgentStepEvent)]
        assert len(step_events) >= 1
        assert step_events[0].step.action == "calculator"

        end_events = [e for e in events if isinstance(e, AgentAnswerEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].full_answer == "4"

    @pytest.mark.asyncio
    async def test_streaming_direct_answer(self) -> None:
        from src.agent.runtime import AgentAnswerEndEvent

        provider = _make_stream_provider(
            [["Thought: I know\nFinal Answer: Paris"]],
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        events = []
        async for event in runtime.run_streaming("Capital of France?"):
            events.append(event)

        end_events = [e for e in events if isinstance(e, AgentAnswerEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].full_answer == "Paris"
        assert end_events[0].stopped_by_max_steps is False

    @pytest.mark.asyncio
    async def test_streaming_max_steps(self) -> None:
        from src.agent.runtime import AgentAnswerEvent

        tool = _StubTool(result="ok")
        provider = _make_stream_provider(
            [
                ["Thought: step0\nAction: calculator\nAction Input: 0"],
                ["Thought: step1\nAction: calculator\nAction Input: 1"],
            ],
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
            max_steps=2,
        )
        events = []
        async for event in runtime.run_streaming("q"):
            events.append(event)

        answer_event = events[-1]
        assert isinstance(answer_event, AgentAnswerEvent)
        assert answer_event.stopped_by_max_steps is True

    @pytest.mark.asyncio
    async def test_streaming_with_conversation_history(self) -> None:
        from src.agent.runtime import AgentAnswerEndEvent

        provider = _make_stream_provider(
            [["Thought: Done\nFinal Answer: yes"]],
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        history = [
            ChatMessage(role=Role.user, content="prev"),
            ChatMessage(role=Role.assistant, content="prev answer"),
        ]
        events = []
        async for event in runtime.run_streaming("new q", conversation_history=history):
            events.append(event)

        end_events = [e for e in events if isinstance(e, AgentAnswerEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].full_answer == "yes"

    @pytest.mark.asyncio
    async def test_streaming_collects_sources(self) -> None:
        from src.agent.runtime import AgentAnswerEndEvent

        meta = {"sources": [{"id": "doc-1"}]}
        tool = _StubTool(result="found", metadata=meta)
        provider = _make_stream_provider(
            [
                ["Thought: Search\nAction: calculator\nAction Input: q"],
                ["Thought: Done\nFinal Answer: result"],
            ],
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
        )
        events = []
        async for event in runtime.run_streaming("q"):
            events.append(event)

        end_events = [e for e in events if isinstance(e, AgentAnswerEndEvent)]
        assert len(end_events) == 1
        assert len(end_events[0].sources) == 1
        assert end_events[0].sources[0]["id"] == "doc-1"


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


# ── New streaming event models ────────────────────────────────


class TestStreamingEventModels:
    """新しいストリーミングイベントモデルのテスト。"""

    def test_answer_start_event_creation(self) -> None:
        from src.agent.runtime import AgentAnswerStartEvent

        event = AgentAnswerStartEvent(thought="I know the answer")
        assert event.thought == "I know the answer"

    def test_answer_chunk_event_creation(self) -> None:
        from src.agent.runtime import AgentAnswerChunkEvent

        event = AgentAnswerChunkEvent(chunk="Hello")
        assert event.chunk == "Hello"

    def test_answer_end_event_creation(self) -> None:
        from src.agent.runtime import AgentAnswerEndEvent

        event = AgentAnswerEndEvent(
            total_steps=2,
            stopped_by_max_steps=False,
            sources=[{"id": "doc-1"}],
            full_answer="Paris is the capital",
        )
        assert event.total_steps == 2
        assert event.stopped_by_max_steps is False
        assert len(event.sources) == 1
        assert event.full_answer == "Paris is the capital"

    def test_answer_end_event_default_sources(self) -> None:
        from src.agent.runtime import AgentAnswerEndEvent

        event = AgentAnswerEndEvent(
            total_steps=1,
            stopped_by_max_steps=False,
            full_answer="answer",
        )
        assert event.sources == []


# ── Token-level streaming ─────────────────────────────────────


def _make_stream_provider(
    stream_responses: list[list[str]],
    complete_responses: list[str] | None = None,
) -> AsyncMock:
    """Create a mock LLMProvider that returns chunks from stream().

    Args:
        stream_responses: List of chunk lists, one per stream() call.
        complete_responses: Optional list of complete() responses for fallback.
    """
    from src.llm.providers.base import LLMChunk

    provider = AsyncMock()

    async def _make_stream(chunks: list[str]) -> AsyncGenerator[Any, None]:
        for c in chunks:
            yield LLMChunk(content=c)

    # Each call to provider.stream() returns the next set of chunks
    call_count = 0

    def _stream_side_effect(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        nonlocal call_count
        idx = call_count
        call_count += 1
        return _make_stream(stream_responses[idx])

    provider.stream = _stream_side_effect

    if complete_responses:
        provider.complete = AsyncMock(
            side_effect=[_make_llm_response(r) for r in complete_responses]
        )

    return provider


class TestTokenLevelStreaming:
    """トークンレベルストリーミングのテスト。"""

    @pytest.mark.asyncio
    async def test_direct_final_answer_yields_start_chunks_end(self) -> None:
        """直接 Final Answer が返される場合、Start + Chunk + End イベントシーケンス。"""
        from src.agent.runtime import (
            AgentAnswerEndEvent,
        )

        provider = _make_stream_provider(
            [["Thought: I know\nFinal Answer: Paris"]],
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        events = []
        async for event in runtime.run_streaming("Capital of France?"):
            events.append(event)

        # Should have: StartEvent, ChunkEvent(s), EndEvent
        event_types = [type(e).__name__ for e in events]
        assert "AgentAnswerStartEvent" in event_types
        assert "AgentAnswerChunkEvent" in event_types
        assert "AgentAnswerEndEvent" in event_types

        # Start should come before chunks, chunks before end
        start_idx = event_types.index("AgentAnswerStartEvent")
        end_idx = event_types.index("AgentAnswerEndEvent")
        assert start_idx < end_idx

        # End event should have the full answer
        end_event = events[end_idx]
        assert isinstance(end_event, AgentAnswerEndEvent)
        assert end_event.full_answer == "Paris"
        assert end_event.stopped_by_max_steps is False

    @pytest.mark.asyncio
    async def test_final_answer_marker_split_across_chunks(self) -> None:
        """Final Answer: マーカーがチャンク境界をまたぐケース。"""
        from src.agent.runtime import AgentAnswerEndEvent

        provider = _make_stream_provider(
            [["Thought: I know\nFinal", " Answer: Paris"]],
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        events = []
        async for event in runtime.run_streaming("q"):
            events.append(event)

        end_events = [e for e in events if isinstance(e, AgentAnswerEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].full_answer == "Paris"

    @pytest.mark.asyncio
    async def test_action_step_then_final_answer(self) -> None:
        """Action ステップ後に Final Answer が来る複合フロー。"""
        from src.agent.runtime import (
            AgentAnswerEndEvent,
            AgentAnswerStartEvent,
            AgentStepEvent,
        )

        tool = _StubTool(result="4")
        provider = _make_stream_provider(
            [
                ["Thought: Calculate\nAction: calculator\nAction Input: 2+2"],
                ["Thought: Got it\nFinal Answer: 4"],
            ],
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
        )
        events = []
        async for event in runtime.run_streaming("What is 2+2?"):
            events.append(event)

        # Should have step events for action, then streaming answer
        step_events = [e for e in events if isinstance(e, AgentStepEvent)]
        assert len(step_events) >= 1
        assert step_events[0].step.action == "calculator"

        start_events = [e for e in events if isinstance(e, AgentAnswerStartEvent)]
        assert len(start_events) == 1

        end_events = [e for e in events if isinstance(e, AgentAnswerEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].full_answer == "4"

    @pytest.mark.asyncio
    async def test_max_steps_fallback_uses_answer_event(self) -> None:
        """最大ステップ到達時は従来の AgentAnswerEvent を使用。"""
        from src.agent.runtime import AgentAnswerEvent

        tool = _StubTool(result="ok")
        provider = _make_stream_provider(
            [
                ["Thought: step0\nAction: calculator\nAction Input: 0"],
                ["Thought: step1\nAction: calculator\nAction Input: 1"],
            ],
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
            max_steps=2,
        )
        events = []
        async for event in runtime.run_streaming("q"):
            events.append(event)

        answer_event = events[-1]
        assert isinstance(answer_event, AgentAnswerEvent)
        assert answer_event.stopped_by_max_steps is True

    @pytest.mark.asyncio
    async def test_streaming_answer_chunks_are_incremental(self) -> None:
        """回答テキストがチャンクごとに増分で送信されること。"""
        from src.agent.runtime import AgentAnswerChunkEvent

        provider = _make_stream_provider(
            [["Thought: I know\nFinal Answer: ", "Hello", " world"]],
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        events = []
        async for event in runtime.run_streaming("q"):
            events.append(event)

        chunks = [e for e in events if isinstance(e, AgentAnswerChunkEvent)]
        # Should have received incremental chunks
        assert len(chunks) >= 1
        combined = "".join(c.chunk for c in chunks)
        assert combined == "Hello world"

    @pytest.mark.asyncio
    async def test_streaming_collects_sources_from_tool_metadata(self) -> None:
        """ツールメタデータのソースがEndEventに含まれること。"""
        from src.agent.runtime import AgentAnswerEndEvent

        meta = {"sources": [{"id": "doc-1"}]}
        tool = _StubTool(result="found", metadata=meta)
        provider = _make_stream_provider(
            [
                ["Thought: Search\nAction: calculator\nAction Input: q"],
                ["Thought: Done\nFinal Answer: result"],
            ],
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=_make_registry(tool),
        )
        events = []
        async for event in runtime.run_streaming("q"):
            events.append(event)

        end_events = [e for e in events if isinstance(e, AgentAnswerEndEvent)]
        assert len(end_events) == 1
        assert len(end_events[0].sources) == 1
        assert end_events[0].sources[0]["id"] == "doc-1"

    @pytest.mark.asyncio
    async def test_streaming_with_conversation_history(self) -> None:
        """会話履歴がストリーミングでも正しく渡されること。"""
        from src.agent.runtime import AgentAnswerEndEvent

        provider = _make_stream_provider(
            [["Thought: Done\nFinal Answer: yes"]],
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        history = [
            ChatMessage(role=Role.user, content="prev"),
            ChatMessage(role=Role.assistant, content="prev answer"),
        ]
        events = []
        async for event in runtime.run_streaming("new q", conversation_history=history):
            events.append(event)

        end_events = [e for e in events if isinstance(e, AgentAnswerEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].full_answer == "yes"

    @pytest.mark.asyncio
    async def test_empty_answer_after_marker(self) -> None:
        """Final Answer: の後にトークンが無い場合、空の回答として処理。"""
        from src.agent.runtime import AgentAnswerEndEvent

        provider = _make_stream_provider(
            [["Thought: I know\nFinal Answer: "]],
        )
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        events = []
        async for event in runtime.run_streaming("q"):
            events.append(event)

        end_events = [e for e in events if isinstance(e, AgentAnswerEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].full_answer == ""

    @pytest.mark.asyncio
    async def test_llm_error_during_stream_raises_agent_error(self) -> None:
        """ストリーム中のエラーがAgentErrorとして伝播すること。"""
        provider = AsyncMock()

        async def _error_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            raise RuntimeError("API down")
            yield  # make it a generator  # noqa: E501

        provider.stream = _error_stream
        runtime = AgentRuntime(
            llm_provider=provider,
            model="m",
            tool_registry=ToolRegistry(),
        )
        with pytest.raises(AgentError):
            async for _ in runtime.run_streaming("q"):
                pass
