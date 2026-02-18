"""忠実性 (Faithfulness) 評価メトリクス。

コンテキストに対して回答がどの程度忠実かを LLM-as-judge で評価する。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.eval import MetricError
from src.eval.metrics import parse_evaluation_response
from src.llm.providers.base import ChatMessage, LLMProvider, Role


class FaithfulnessResult(BaseModel):
    """忠実性評価の結果。

    Attributes:
        score: 忠実性スコア (0.0-1.0)。1.0 が完全に忠実。
        reason: 評価理由。
    """

    score: float
    reason: str


class FaithfulnessMetric:
    """コンテキストと回答の忠実性を LLM で評価するメトリクス。

    Generator と同じパターンで LLMProvider をコンストラクタ注入する。
    """

    DEFAULT_SYSTEM_PROMPT: str = (
        "あなたは回答の忠実性を評価する専門家です。\n"
        "提供されたコンテキストに対して、回答がどの程度忠実かを評価してください。\n"
        "コンテキストに含まれない情報が回答に含まれている場合、忠実性は低くなります。\n\n"
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

    async def evaluate(self, context: str, answer: str, **kwargs: Any) -> FaithfulnessResult:
        """コンテキストに対する回答の忠実性を評価する。

        Args:
            context: 参照コンテキスト。
            answer: 評価対象の回答。
            **kwargs: LLM プロバイダに渡す追加パラメータ。

        Returns:
            スコアと評価理由を含む FaithfulnessResult。

        Raises:
            MetricError: LLM 呼び出しまたはレスポンスのパースに失敗した場合。
        """
        messages = self._build_messages(context, answer)

        try:
            response = await self._provider.complete(
                messages=messages,
                model=self._model,
                **kwargs,
            )
        except Exception as e:
            raise MetricError(str(e)) from e

        score, reason = parse_evaluation_response(response.content)
        return FaithfulnessResult(score=score, reason=reason)

    def _build_messages(self, context: str, answer: str) -> list[ChatMessage]:
        """LLM に送信するメッセージリストを構築する。"""
        return [
            ChatMessage(role=Role.system, content=self._system_prompt),
            ChatMessage(
                role=Role.user,
                content=f"コンテキスト:\n{context}\n\n回答:\n{answer}",
            ),
        ]
