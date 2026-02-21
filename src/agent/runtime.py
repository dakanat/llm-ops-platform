"""Agent 実行エンジン。

ReAct (Reasoning + Acting) ループで LLM とツールを連携させ、
ユーザークエリに対する回答を生成する。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from pydantic import BaseModel

from src.agent import AgentError, ToolNotFoundError
from src.agent.planner import ParsedAction, ParsedFinalAnswer, ReActPlanner
from src.agent.state import AgentState, AgentStep
from src.agent.tools.base import Tool
from src.agent.tools.registry import ToolRegistry
from src.llm.providers.base import ChatMessage, LLMProvider


class AgentResult(BaseModel):
    """Agent 実行結果。

    Attributes:
        answer: 最終回答テキスト。
        steps: 実行された全ステップ。
        total_steps: 総ステップ数。
        stopped_by_max_steps: 最大ステップ数で停止したかどうか。
    """

    answer: str
    steps: list[AgentStep] = []
    total_steps: int = 0
    stopped_by_max_steps: bool = False


class AgentStepEvent(BaseModel):
    """ストリーミング中のステップイベント。"""

    step: AgentStep
    step_number: int


class AgentAnswerEvent(BaseModel):
    """ストリーミング中の最終回答イベント。"""

    answer: str
    total_steps: int
    stopped_by_max_steps: bool
    sources: list[dict[str, object]] = []


class AgentRuntime:
    """ReAct ループの実行エンジン。

    LLM とツールレジストリを組み合わせて、
    ステップバイステップでクエリに回答する。
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        model: str,
        tool_registry: ToolRegistry,
        planner: ReActPlanner | None = None,
        max_steps: int = 10,
    ) -> None:
        self._llm_provider = llm_provider
        self._model = model
        self._tool_registry = tool_registry
        self._planner = planner if planner is not None else ReActPlanner()
        self._max_steps = max_steps

    def _get_tools(self) -> list[Tool]:
        """レジストリから全ツールを取得する。"""
        return [self._tool_registry.get(name) for name in self._tool_registry.list_tools()]

    async def run(
        self,
        query: str,
        conversation_history: list[ChatMessage] | None = None,
    ) -> AgentResult:
        """ReAct ループを実行してクエリに回答する。

        Args:
            query: ユーザークエリ。
            conversation_history: 過去の会話履歴。

        Returns:
            AgentResult。

        Raises:
            AgentError: LLM 呼び出しに失敗した場合。
        """
        state = AgentState(
            query=query,
            max_steps=self._max_steps,
            conversation_history=conversation_history,
        )
        tools = self._get_tools()

        for _ in range(self._max_steps):
            # LLM 呼び出し
            messages = self._planner.build_messages(state, tools)
            try:
                response = await self._llm_provider.complete(messages, self._model)
            except Exception as e:
                raise AgentError(f"LLM call failed: {e}") from e

            # レスポンスパース
            parsed = self._planner.parse_response(response.content)

            if isinstance(parsed, ParsedFinalAnswer):
                step = AgentStep(thought=parsed.thought)
                state.add_step(step)
                state.set_final_answer(parsed.answer)
                return AgentResult(
                    answer=parsed.answer,
                    steps=state.steps,
                    total_steps=state.step_count,
                    stopped_by_max_steps=False,
                )

            # ParsedAction: ツール実行
            assert isinstance(parsed, ParsedAction)
            observation: str
            is_error = False

            metadata: dict[str, object] | None = None
            try:
                tool = self._tool_registry.get(parsed.tool_name)
            except ToolNotFoundError:
                observation = f"Tool not found: {parsed.tool_name}"
                is_error = True
            else:
                result = await tool.execute(parsed.tool_input)
                if result.is_error:
                    observation = result.error or "Unknown tool error"
                    is_error = True
                else:
                    observation = result.output
                    metadata = result.metadata

            step = AgentStep(
                thought=parsed.thought,
                action=parsed.tool_name,
                action_input=parsed.tool_input,
                observation=observation,
                is_error=is_error,
                metadata=metadata,
            )
            state.add_step(step)

        # 最大ステップ数に達した
        last_step = state.steps[-1] if state.steps else None
        answer = last_step.observation or "" if last_step else ""
        return AgentResult(
            answer=answer,
            steps=state.steps,
            total_steps=state.step_count,
            stopped_by_max_steps=True,
        )

    async def run_streaming(
        self,
        query: str,
        conversation_history: list[ChatMessage] | None = None,
    ) -> AsyncGenerator[AgentStepEvent | AgentAnswerEvent, None]:
        """ReAct ループをストリーミング実行する。

        各ステップ完了時に AgentStepEvent を、
        最終回答時に AgentAnswerEvent を yield する。

        Args:
            query: ユーザークエリ。
            conversation_history: 過去の会話履歴。

        Yields:
            AgentStepEvent または AgentAnswerEvent。

        Raises:
            AgentError: LLM 呼び出しに失敗した場合。
        """
        state = AgentState(
            query=query,
            max_steps=self._max_steps,
            conversation_history=conversation_history,
        )
        tools = self._get_tools()
        sources: list[dict[str, object]] = []

        for step_num in range(self._max_steps):
            messages = self._planner.build_messages(state, tools)
            try:
                response = await self._llm_provider.complete(messages, self._model)
            except Exception as e:
                raise AgentError(f"LLM call failed: {e}") from e

            parsed = self._planner.parse_response(response.content)

            if isinstance(parsed, ParsedFinalAnswer):
                step = AgentStep(thought=parsed.thought)
                state.add_step(step)
                yield AgentStepEvent(step=step, step_number=step_num + 1)
                yield AgentAnswerEvent(
                    answer=parsed.answer,
                    total_steps=state.step_count,
                    stopped_by_max_steps=False,
                    sources=sources,
                )
                return

            assert isinstance(parsed, ParsedAction)
            observation: str
            is_error = False
            metadata: dict[str, object] | None = None

            try:
                tool = self._tool_registry.get(parsed.tool_name)
            except ToolNotFoundError:
                observation = f"Tool not found: {parsed.tool_name}"
                is_error = True
            else:
                result = await tool.execute(parsed.tool_input)
                if result.is_error:
                    observation = result.error or "Unknown tool error"
                    is_error = True
                else:
                    observation = result.output
                    metadata = result.metadata
                    if metadata and "sources" in metadata:
                        sources.extend(metadata["sources"])  # type: ignore[arg-type]

            step = AgentStep(
                thought=parsed.thought,
                action=parsed.tool_name,
                action_input=parsed.tool_input,
                observation=observation,
                is_error=is_error,
                metadata=metadata,
            )
            state.add_step(step)
            yield AgentStepEvent(step=step, step_number=step_num + 1)

        # 最大ステップ数に達した
        last_step = state.steps[-1] if state.steps else None
        answer = last_step.observation or "" if last_step else ""
        yield AgentAnswerEvent(
            answer=answer,
            total_steps=state.step_count,
            stopped_by_max_steps=True,
            sources=sources,
        )
