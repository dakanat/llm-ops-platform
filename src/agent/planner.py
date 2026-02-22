"""ReAct プランナー。

ReAct (Reasoning + Acting) 形式のプロンプト構築と LLM 出力パースを行う。
"""

from __future__ import annotations

import re

import structlog
from pydantic import BaseModel

from src.agent import AgentParseError
from src.agent.state import AgentState
from src.agent.tools.base import Tool
from src.llm.providers.base import ChatMessage, Role

logger = structlog.get_logger()

# パースに使用する正規表現パターン
_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\n\s*(?:Action:|Final Answer:))", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(.+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.*)", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


class ParsedAction(BaseModel):
    """パース結果: ツール呼び出し。

    Attributes:
        thought: LLM の推論テキスト。
        tool_name: 呼び出すツール名。
        tool_input: ツールへの入力。
    """

    thought: str
    tool_name: str
    tool_input: str


class ParsedFinalAnswer(BaseModel):
    """パース結果: 最終回答。

    Attributes:
        thought: LLM の推論テキスト。
        answer: 最終回答テキスト。
    """

    thought: str
    answer: str


ParsedResponse = ParsedAction | ParsedFinalAnswer


class ReActPlanner:
    """ReAct 形式のプロンプト構築・パースを行うステートレスクラス。"""

    def build_system_prompt(self, tools: list[Tool]) -> str:
        """ツール情報を含む ReAct システムプロンプトを構築する。

        Args:
            tools: 利用可能なツールのリスト。

        Returns:
            システムプロンプト文字列。
        """
        tool_section = ""
        if tools:
            tool_lines = [f"- {t.name}: {t.description}" for t in tools]
            tool_section = (
                "You have access to the following tools:\n" + "\n".join(tool_lines) + "\n\n"
            )

        return (
            "You are a helpful assistant that answers questions by reasoning step-by-step "
            "and using tools when needed.\n\n"
            f"{tool_section}"
            "To use a tool, respond in EXACTLY this format:\n"
            "Thought: <your reasoning>\n"
            "Action: <tool_name>\n"
            "Action Input: <input to the tool>\n\n"
            "When you have the final answer, respond in EXACTLY this format:\n"
            "Thought: <your reasoning>\n"
            "Final Answer: <your answer>\n\n"
            "IMPORTANT: You MUST always respond using one of the two formats above.\n"
            "Even for simple greetings or questions, use the Thought/Final Answer format."
        )

    def build_messages(self, state: AgentState, tools: list[Tool]) -> list[ChatMessage]:
        """AgentState からメッセージリストを構築する。

        システムプロンプト + ユーザークエリ + ステップ履歴を
        assistant/user ペアとして交互に配置する。

        Args:
            state: 現在の Agent 状態。
            tools: 利用可能なツールのリスト。

        Returns:
            ChatMessage のリスト。
        """
        messages: list[ChatMessage] = [
            ChatMessage(role=Role.system, content=self.build_system_prompt(tools)),
        ]

        # 会話履歴を挿入（システムプロンプトと現在クエリの間）
        for msg in state.conversation_history:
            messages.append(msg)

        # 現在のクエリ
        messages.append(ChatMessage(role=Role.user, content=state.query))

        for step in state.steps:
            # Assistant メッセージ: Thought + Action + Action Input
            assistant_content = f"Thought: {step.thought}"
            if step.action is not None:
                assistant_content += f"\nAction: {step.action}"
                if step.action_input is not None:
                    assistant_content += f"\nAction Input: {step.action_input}"
            messages.append(ChatMessage(role=Role.assistant, content=assistant_content))

            # User メッセージ: Observation
            if step.observation is not None:
                messages.append(
                    ChatMessage(role=Role.user, content=f"Observation: {step.observation}")
                )

        return messages

    def parse_response(self, text: str) -> ParsedResponse:
        """LLM 出力を ReAct 形式としてパースする。

        ReAct 形式 (Action: / Final Answer:) に従わないレスポンスは、
        フォールバックとして全文を ParsedFinalAnswer(thought="", answer=text) として返す。

        Args:
            text: LLM の生テキスト出力。

        Returns:
            ParsedAction または ParsedFinalAnswer。

        Raises:
            AgentParseError: 空のレスポンス、または Action 形式で
                Thought:/Action Input: が欠落している場合。
        """
        text = text.strip()
        if not text:
            raise AgentParseError("Empty response from LLM")

        # Final Answer パターンのチェック
        final_match = _FINAL_ANSWER_RE.search(text)
        if final_match:
            thought_match = _THOUGHT_RE.search(text)
            thought = thought_match.group(1).strip() if thought_match else ""
            answer = final_match.group(1).strip()
            return ParsedFinalAnswer(thought=thought, answer=answer)

        # Action パターンのチェック
        action_match = _ACTION_RE.search(text)
        if action_match:
            thought_match = _THOUGHT_RE.search(text)
            if thought_match is None:
                raise AgentParseError("Missing 'Thought:' in response")

            # Action Input を探す
            action_input_match = _ACTION_INPUT_RE.search(text)
            if action_input_match is None:
                raise AgentParseError("Missing 'Action Input:' in response")

            thought = thought_match.group(1).strip()
            tool_name = action_match.group(1).strip()
            tool_input = action_input_match.group(1).strip()
            return ParsedAction(thought=thought, tool_name=tool_name, tool_input=tool_input)

        logger.warning("llm_response_missing_react_format", response_preview=text[:200])
        return ParsedFinalAnswer(thought="", answer=text)
