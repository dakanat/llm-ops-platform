"""関連性 (Relevance) 評価メトリクス。

クエリに対して回答がどの程度関連しているかを LLM-as-judge で評価する。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.eval import MetricError
from src.eval.metrics import parse_evaluation_response
from src.llm.providers.base import ChatMessage, LLMProvider, Role


class RelevanceResult(BaseModel):
    """関連性評価の結果。

    Attributes:
        score: 関連性スコア (0.0-1.0)。1.0 が完全に関連。
        reason: 評価理由。
    """

    score: float
    reason: str


class RelevanceMetric:
    """クエリと回答の関連性を LLM で評価するメトリクス。

    FaithfulnessMetric と対称的な構造。評価対象が query + answer に変わる。
    """

    DEFAULT_SYSTEM_PROMPT: str = (
        "あなたは回答の関連性を評価する専門家です。\n"
        "ユーザーの質問に対して、回答がどの程度関連しているかを評価してください。\n"
        "質問に直接答えていない場合、関連性は低くなります。\n\n"
        "以下の形式で回答してください:\n"
        "Score: <0.0から1.0の数値>\n"
        "Reason: <評価理由>"
    )

    def __init__(
        self,
        llm_provider: LLMProvider,
        model: str,
        system_prompt: str | None = None,
    ) -> None:
        self._provider = llm_provider
        self._model = model
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    async def evaluate(self, query: str, answer: str, **kwargs: Any) -> RelevanceResult:
        """クエリに対する回答の関連性を評価する。

        Args:
            query: ユーザーの質問。
            answer: 評価対象の回答。
            **kwargs: LLM プロバイダに渡す追加パラメータ。

        Returns:
            スコアと評価理由を含む RelevanceResult。

        Raises:
            MetricError: LLM 呼び出しまたはレスポンスのパースに失敗した場合。
        """
        messages = self._build_messages(query, answer)

        try:
            response = await self._provider.complete(
                messages=messages,
                model=self._model,
                **kwargs,
            )
        except Exception as e:
            raise MetricError(str(e)) from e

        score, reason = parse_evaluation_response(response.content)
        return RelevanceResult(score=score, reason=reason)

    def _build_messages(self, query: str, answer: str) -> list[ChatMessage]:
        """LLM に送信するメッセージリストを構築する。"""
        return [
            ChatMessage(role=Role.system, content=self._system_prompt),
            ChatMessage(
                role=Role.user,
                content=f"質問:\n{query}\n\n回答:\n{answer}",
            ),
        ]
