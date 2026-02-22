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
    """ストリーミング中の最終回答イベント（max-steps フォールバック用）。"""

    answer: str
    total_steps: int
    stopped_by_max_steps: bool
    sources: list[dict[str, object]] = []


class AgentAnswerStartEvent(BaseModel):
    """ストリーミング回答の開始を通知。"""

    thought: str


class AgentAnswerChunkEvent(BaseModel):
    """回答テキストの個別チャンク。"""

    chunk: str


class AgentAnswerEndEvent(BaseModel):
    """ストリーミング回答の完了を通知。"""

    total_steps: int
    stopped_by_max_steps: bool
    sources: list[dict[str, object]] = []
    full_answer: str


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

    _FINAL_ANSWER_MARKER = "Final Answer:"

    async def run_streaming(
        self,
        query: str,
        conversation_history: list[ChatMessage] | None = None,
    ) -> AsyncGenerator[
        AgentStepEvent
        | AgentAnswerEvent
        | AgentAnswerStartEvent
        | AgentAnswerChunkEvent
        | AgentAnswerEndEvent,
        None,
    ]:
        """ReAct ループをストリーミング実行する。

        provider.stream() を使用してトークン単位でレスポンスを受信する。
        Final Answer: マーカー検出後は後続トークンを AgentAnswerChunkEvent として
        リアルタイムに yield する。

        Args:
            query: ユーザークエリ。
            conversation_history: 過去の会話履歴。

        Yields:
            AgentStepEvent, AgentAnswerStartEvent, AgentAnswerChunkEvent,
            AgentAnswerEndEvent, または AgentAnswerEvent (max-steps フォールバック)。

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

            # stream() でトークンをバッファリング
            buffer = ""
            found_marker = False
            answer_buffer = ""

            try:
                async for chunk in self._llm_provider.stream(messages, self._model):
                    buffer += chunk.content

                    if not found_marker:
                        marker_pos = buffer.find(self._FINAL_ANSWER_MARKER)
                        if marker_pos >= 0:
                            found_marker = True
                            # マーカー後のテキストを回答バッファに移動
                            after_marker = buffer[marker_pos + len(self._FINAL_ANSWER_MARKER) :]
                            answer_buffer = after_marker.lstrip()

                            # thought を抽出して StartEvent を送信
                            pre_marker = buffer[:marker_pos]
                            thought = self._extract_thought(pre_marker)
                            step = AgentStep(thought=thought)
                            state.add_step(step)

                            yield AgentAnswerStartEvent(thought=thought)

                            # バッファ内の回答テキストがあればチャンクとして送信
                            if answer_buffer:
                                yield AgentAnswerChunkEvent(chunk=answer_buffer)
                    else:
                        # マーカー検出済み: 後続チャンクを即座に転送
                        if chunk.content:
                            answer_buffer += chunk.content
                            yield AgentAnswerChunkEvent(chunk=chunk.content)
            except Exception as e:
                raise AgentError(f"LLM call failed: {e}") from e

            if found_marker:
                # ストリーミング回答完了
                yield AgentAnswerEndEvent(
                    total_steps=state.step_count,
                    stopped_by_max_steps=False,
                    sources=sources,
                    full_answer=answer_buffer.strip(),
                )
                return

            # Final Answer なし → Action パース
            parsed = self._planner.parse_response(buffer)

            if isinstance(parsed, ParsedFinalAnswer):
                # parse_response がFinal Answerを検出（バッファリング段階で見逃した場合）
                step = AgentStep(thought=parsed.thought)
                state.add_step(step)
                yield AgentAnswerStartEvent(thought=parsed.thought)
                if parsed.answer:
                    yield AgentAnswerChunkEvent(chunk=parsed.answer)
                yield AgentAnswerEndEvent(
                    total_steps=state.step_count,
                    stopped_by_max_steps=False,
                    sources=sources,
                    full_answer=parsed.answer,
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

    @staticmethod
    def _extract_thought(text: str) -> str:
        """テキストから Thought: 以降の内容を抽出する。"""
        marker = "Thought:"
        pos = text.find(marker)
        if pos >= 0:
            return text[pos + len(marker) :].strip()
        return text.strip()
