"""ReActPlanner のテスト。"""

from __future__ import annotations

import pytest
from src.agent import AgentParseError
from src.agent.planner import ParsedAction, ParsedFinalAnswer, ReActPlanner
from src.agent.state import AgentState, AgentStep
from src.agent.tools.base import ToolResult
from src.llm.providers.base import Role

# ── Stub Tool ────────────────────────────────────────────────


class _StubTool:
    """テスト用スタブツール。"""

    def __init__(self) -> None:
        self.name = "search"
        self.description = "Search for documents."

    async def execute(self, input_text: str) -> ToolResult:
        return ToolResult(output="stub")


class _StubTool2:
    """テスト用スタブツール 2。"""

    def __init__(self) -> None:
        self.name = "calculator"
        self.description = "Evaluate math expressions."

    async def execute(self, input_text: str) -> ToolResult:
        return ToolResult(output="stub")


# ── ParsedAction / ParsedFinalAnswer models ──────────────────


class TestParsedAction:
    """ParsedAction モデルのテスト。"""

    def test_creation(self) -> None:
        pa = ParsedAction(thought="考える", tool_name="calc", tool_input="1+1")
        assert pa.thought == "考える"
        assert pa.tool_name == "calc"
        assert pa.tool_input == "1+1"


class TestParsedFinalAnswer:
    """ParsedFinalAnswer モデルのテスト。"""

    def test_creation(self) -> None:
        pf = ParsedFinalAnswer(thought="完了", answer="42")
        assert pf.thought == "完了"
        assert pf.answer == "42"


# ── build_system_prompt ──────────────────────────────────────


class TestBuildSystemPrompt:
    """ReActPlanner.build_system_prompt のテスト。"""

    def test_includes_tool_names(self) -> None:
        planner = ReActPlanner()
        prompt = planner.build_system_prompt([_StubTool(), _StubTool2()])
        assert "search" in prompt
        assert "calculator" in prompt

    def test_includes_tool_descriptions(self) -> None:
        planner = ReActPlanner()
        prompt = planner.build_system_prompt([_StubTool()])
        assert "Search for documents." in prompt

    def test_includes_format_instructions(self) -> None:
        planner = ReActPlanner()
        prompt = planner.build_system_prompt([_StubTool()])
        assert "Thought:" in prompt
        assert "Action:" in prompt
        assert "Action Input:" in prompt
        assert "Final Answer:" in prompt

    def test_empty_tool_list(self) -> None:
        planner = ReActPlanner()
        prompt = planner.build_system_prompt([])
        assert "Thought:" in prompt
        assert "Final Answer:" in prompt


# ── build_messages ───────────────────────────────────────────


class TestBuildMessages:
    """ReActPlanner.build_messages のテスト。"""

    def test_first_turn_has_system_and_user(self) -> None:
        planner = ReActPlanner()
        state = AgentState(query="Hello")
        msgs = planner.build_messages(state, [_StubTool()])
        assert msgs[0].role == Role.system
        assert msgs[1].role == Role.user

    def test_user_message_contains_query(self) -> None:
        planner = ReActPlanner()
        state = AgentState(query="What is 2+2?")
        msgs = planner.build_messages(state, [])
        assert "What is 2+2?" in msgs[1].content

    def test_includes_step_history_as_assistant_user_pairs(self) -> None:
        planner = ReActPlanner()
        state = AgentState(query="q")
        state.add_step(
            AgentStep(
                thought="Let me search",
                action="search",
                action_input="query",
                observation="result1",
            )
        )
        msgs = planner.build_messages(state, [_StubTool()])
        roles = [m.role for m in msgs]
        # system, user, assistant (step), user (observation)
        assert roles == [Role.system, Role.user, Role.assistant, Role.user]

    def test_message_count_increases_with_steps(self) -> None:
        planner = ReActPlanner()
        state = AgentState(query="q")
        msgs_0 = planner.build_messages(state, [])
        assert len(msgs_0) == 2  # system + user

        state.add_step(AgentStep(thought="t1", action="a", action_input="i", observation="o1"))
        msgs_1 = planner.build_messages(state, [])
        assert len(msgs_1) == 4  # system + user + assistant + user

        state.add_step(AgentStep(thought="t2", action="b", action_input="j", observation="o2"))
        msgs_2 = planner.build_messages(state, [])
        assert len(msgs_2) == 6

    def test_assistant_message_contains_thought_and_action(self) -> None:
        planner = ReActPlanner()
        state = AgentState(query="q")
        state.add_step(
            AgentStep(
                thought="reasoning here",
                action="search",
                action_input="some query",
                observation="found it",
            )
        )
        msgs = planner.build_messages(state, [])
        assistant_msg = msgs[2]
        assert "Thought: reasoning here" in assistant_msg.content
        assert "Action: search" in assistant_msg.content
        assert "Action Input: some query" in assistant_msg.content

    def test_observation_message_contains_observation(self) -> None:
        planner = ReActPlanner()
        state = AgentState(query="q")
        state.add_step(
            AgentStep(
                thought="t",
                action="a",
                action_input="i",
                observation="the result",
            )
        )
        msgs = planner.build_messages(state, [])
        observation_msg = msgs[3]
        assert "Observation: the result" in observation_msg.content


# ── build_messages with conversation history ─────────────────


class TestBuildMessagesWithConversationHistory:
    """会話履歴付き build_messages のテスト。"""

    def test_conversation_history_inserted_between_system_and_query(self) -> None:
        from src.llm.providers.base import ChatMessage, Role

        planner = ReActPlanner()
        history = [
            ChatMessage(role=Role.user, content="What is Python?"),
            ChatMessage(role=Role.assistant, content="A programming language."),
        ]
        state = AgentState(query="Tell me more", conversation_history=history)
        msgs = planner.build_messages(state, [])
        assert msgs[0].role == Role.system
        assert msgs[1].role == Role.user
        assert msgs[1].content == "What is Python?"
        assert msgs[2].role == Role.assistant
        assert msgs[2].content == "A programming language."
        assert msgs[3].role == Role.user
        assert msgs[3].content == "Tell me more"

    def test_empty_conversation_history_same_as_no_history(self) -> None:
        planner = ReActPlanner()
        state_no_history = AgentState(query="Hello")
        state_empty = AgentState(query="Hello", conversation_history=[])
        msgs_no = planner.build_messages(state_no_history, [])
        msgs_empty = planner.build_messages(state_empty, [])
        assert len(msgs_no) == len(msgs_empty)

    def test_conversation_history_with_steps(self) -> None:
        from src.llm.providers.base import ChatMessage, Role

        planner = ReActPlanner()
        history = [
            ChatMessage(role=Role.user, content="prev question"),
            ChatMessage(role=Role.assistant, content="prev answer"),
        ]
        state = AgentState(query="new question", conversation_history=history)
        state.add_step(
            AgentStep(
                thought="search",
                action="search",
                action_input="query",
                observation="result",
            )
        )
        msgs = planner.build_messages(state, [_StubTool()])
        # system + 2 history + user query + assistant step + observation
        assert len(msgs) == 6
        assert msgs[1].content == "prev question"
        assert msgs[2].content == "prev answer"
        assert msgs[3].content == "new question"


# ── parse_response ───────────────────────────────────────────


class TestParseResponse:
    """ReActPlanner.parse_response のテスト。"""

    def test_parses_action_format(self) -> None:
        planner = ReActPlanner()
        text = "Thought: I need to search\nAction: search\nAction Input: python docs"
        result = planner.parse_response(text)
        assert isinstance(result, ParsedAction)
        assert result.thought == "I need to search"
        assert result.tool_name == "search"
        assert result.tool_input == "python docs"

    def test_parses_final_answer_format(self) -> None:
        planner = ReActPlanner()
        text = "Thought: I know the answer\nFinal Answer: 42"
        result = planner.parse_response(text)
        assert isinstance(result, ParsedFinalAnswer)
        assert result.thought == "I know the answer"
        assert result.answer == "42"

    def test_strips_whitespace(self) -> None:
        planner = ReActPlanner()
        text = "  Thought:  spaced out  \n  Action:  calc  \n  Action Input:  1+1  "
        result = planner.parse_response(text)
        assert isinstance(result, ParsedAction)
        assert result.thought == "spaced out"
        assert result.tool_name == "calc"
        assert result.tool_input == "1+1"

    def test_multiline_thought(self) -> None:
        planner = ReActPlanner()
        text = "Thought: first line\nsecond line\nthird line\nFinal Answer: done"
        result = planner.parse_response(text)
        assert isinstance(result, ParsedFinalAnswer)
        assert "first line" in result.thought
        assert "second line" in result.thought

    def test_multiline_final_answer(self) -> None:
        planner = ReActPlanner()
        text = "Thought: done\nFinal Answer: line1\nline2\nline3"
        result = planner.parse_response(text)
        assert isinstance(result, ParsedFinalAnswer)
        assert "line1" in result.answer
        assert "line2" in result.answer

    def test_raises_on_missing_thought(self) -> None:
        planner = ReActPlanner()
        with pytest.raises(AgentParseError):
            planner.parse_response("Action: search\nAction Input: q")

    def test_raises_on_action_without_input(self) -> None:
        planner = ReActPlanner()
        with pytest.raises(AgentParseError):
            planner.parse_response("Thought: hmm\nAction: search")

    def test_raises_on_empty_string(self) -> None:
        planner = ReActPlanner()
        with pytest.raises(AgentParseError):
            planner.parse_response("")
