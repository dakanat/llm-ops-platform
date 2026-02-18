"""Agent 状態管理。

会話履歴と中間思考ステップを保持する。
"""

from __future__ import annotations

from pydantic import BaseModel


class AgentStep(BaseModel):
    """ReAct ループの1ステップ。

    Attributes:
        thought: LLM の推論テキスト。
        action: 呼び出すツール名。最終回答ステップでは None。
        action_input: ツールへの入力。
        observation: ツール実行結果。
        is_error: observation がエラーかどうか。
    """

    thought: str
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None
    is_error: bool = False


class AgentState:
    """Agent 実行の状態を管理する。

    クエリ、ステップ履歴、最終回答を保持し、
    ループ制御に必要なプロパティを提供する。
    """

    def __init__(self, query: str, max_steps: int = 10) -> None:
        self._query = query
        self._max_steps = max_steps
        self._steps: list[AgentStep] = []
        self._final_answer: str | None = None
        self._is_complete: bool = False

    @property
    def query(self) -> str:
        """ユーザークエリ。"""
        return self._query

    @property
    def max_steps(self) -> int:
        """最大ステップ数。"""
        return self._max_steps

    @property
    def steps(self) -> list[AgentStep]:
        """ステップ履歴のコピーを返す。"""
        return list(self._steps)

    @property
    def step_count(self) -> int:
        """現在のステップ数。"""
        return len(self._steps)

    @property
    def max_steps_reached(self) -> bool:
        """最大ステップ数に達したかどうか。"""
        return self.step_count >= self._max_steps

    @property
    def final_answer(self) -> str | None:
        """最終回答。未設定時は None。"""
        return self._final_answer

    @property
    def is_complete(self) -> bool:
        """実行が完了したかどうか。"""
        return self._is_complete

    def add_step(self, step: AgentStep) -> None:
        """ステップを追加する。

        Args:
            step: 追加する AgentStep。
        """
        self._steps.append(step)

    def set_final_answer(self, answer: str) -> None:
        """最終回答を設定し、実行を完了にする。

        Args:
            answer: 最終回答テキスト。
        """
        self._final_answer = answer
        self._is_complete = True
