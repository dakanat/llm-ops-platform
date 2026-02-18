"""AgentStep, AgentState, AgentError, AgentParseError のテスト。"""

from __future__ import annotations

import pytest
from src.agent import AgentError, AgentParseError
from src.agent.state import AgentState, AgentStep

# ── AgentStep model ──────────────────────────────────────────


class TestAgentStep:
    """AgentStep モデルのテスト。"""

    def test_creation_with_all_fields(self) -> None:
        step = AgentStep(
            thought="考える",
            action="calculator",
            action_input="1+1",
            observation="2",
            is_error=False,
        )
        assert step.thought == "考える"
        assert step.action == "calculator"
        assert step.action_input == "1+1"
        assert step.observation == "2"
        assert step.is_error is False

    def test_defaults_action_none(self) -> None:
        step = AgentStep(thought="結論")
        assert step.action is None

    def test_defaults_action_input_none(self) -> None:
        step = AgentStep(thought="結論")
        assert step.action_input is None

    def test_defaults_observation_none(self) -> None:
        step = AgentStep(thought="結論")
        assert step.observation is None

    def test_defaults_is_error_false(self) -> None:
        step = AgentStep(thought="結論")
        assert step.is_error is False


# ── AgentState init ──────────────────────────────────────────


class TestAgentStateInit:
    """AgentState 初期化のテスト。"""

    def test_creates_with_query(self) -> None:
        state = AgentState(query="テストクエリ")
        assert state.query == "テストクエリ"

    def test_default_max_steps_is_10(self) -> None:
        state = AgentState(query="q")
        assert state.max_steps == 10

    def test_custom_max_steps(self) -> None:
        state = AgentState(query="q", max_steps=5)
        assert state.max_steps == 5

    def test_initial_steps_empty(self) -> None:
        state = AgentState(query="q")
        assert state.steps == []

    def test_initial_is_complete_false(self) -> None:
        state = AgentState(query="q")
        assert state.is_complete is False

    def test_initial_final_answer_none(self) -> None:
        state = AgentState(query="q")
        assert state.final_answer is None

    def test_initial_step_count_zero(self) -> None:
        state = AgentState(query="q")
        assert state.step_count == 0


# ── AgentState.add_step ──────────────────────────────────────


class TestAgentStateAddStep:
    """AgentState.add_step のテスト。"""

    def test_appends_step(self) -> None:
        state = AgentState(query="q")
        step = AgentStep(thought="考える", action="calc", action_input="1+1")
        state.add_step(step)
        assert len(state.steps) == 1
        assert state.steps[0].thought == "考える"

    def test_multiple_steps(self) -> None:
        state = AgentState(query="q")
        state.add_step(AgentStep(thought="1st"))
        state.add_step(AgentStep(thought="2nd"))
        assert len(state.steps) == 2

    def test_step_count_increments(self) -> None:
        state = AgentState(query="q")
        assert state.step_count == 0
        state.add_step(AgentStep(thought="a"))
        assert state.step_count == 1
        state.add_step(AgentStep(thought="b"))
        assert state.step_count == 2


# ── AgentState.set_final_answer ──────────────────────────────


class TestAgentStateSetFinalAnswer:
    """AgentState.set_final_answer のテスト。"""

    def test_sets_final_answer(self) -> None:
        state = AgentState(query="q")
        state.set_final_answer("答え")
        assert state.final_answer == "答え"

    def test_sets_is_complete_true(self) -> None:
        state = AgentState(query="q")
        state.set_final_answer("答え")
        assert state.is_complete is True


# ── AgentState.max_steps_reached ─────────────────────────────


class TestAgentStateMaxStepsReached:
    """AgentState.max_steps_reached のテスト。"""

    def test_false_when_under_limit(self) -> None:
        state = AgentState(query="q", max_steps=3)
        state.add_step(AgentStep(thought="a"))
        assert state.max_steps_reached is False

    def test_true_when_at_limit(self) -> None:
        state = AgentState(query="q", max_steps=2)
        state.add_step(AgentStep(thought="a"))
        state.add_step(AgentStep(thought="b"))
        assert state.max_steps_reached is True


# ── AgentState.steps returns copy ────────────────────────────


class TestAgentStateStepsCopy:
    """AgentState.steps がコピーを返すことのテスト。"""

    def test_returns_copy(self) -> None:
        state = AgentState(query="q")
        state.add_step(AgentStep(thought="a"))
        steps = state.steps
        steps.append(AgentStep(thought="外部追加"))
        assert state.step_count == 1


# ── Exceptions ───────────────────────────────────────────────


class TestAgentExceptions:
    """AgentError / AgentParseError のテスト。"""

    def test_agent_error_is_exception(self) -> None:
        assert issubclass(AgentError, Exception)

    def test_agent_parse_error_is_agent_error(self) -> None:
        assert issubclass(AgentParseError, AgentError)

    def test_agent_error_message(self) -> None:
        err = AgentError("something went wrong")
        assert str(err) == "something went wrong"

    def test_agent_parse_error_message(self) -> None:
        err = AgentParseError("parse failed")
        assert str(err) == "parse failed"

    def test_agent_error_can_be_raised_and_caught(self) -> None:
        with pytest.raises(AgentError):
            raise AgentError("boom")

    def test_agent_parse_error_caught_as_agent_error(self) -> None:
        with pytest.raises(AgentError):
            raise AgentParseError("bad format")
