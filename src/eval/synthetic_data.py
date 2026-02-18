"""合成データ生成。

ドキュメントテキストから LLM を用いて QA ペアを自動生成し、
評価データセット (EvalDataset) を構築する。
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from src.eval import SyntheticDataError
from src.eval.datasets import EvalDataset, EvalExample
from src.llm.providers.base import ChatMessage, LLMProvider, Role


class QAPair(BaseModel):
    """LLM が生成する QA ペアの中間モデル。

    Attributes:
        question: 生成された質問。
        answer: 生成された回答。
    """

    question: str
    answer: str


class SyntheticDataGenerator:
    """ドキュメントから QA ペアを自動生成するジェネレータ。

    FaithfulnessMetric と同じパターンで LLMProvider をコンストラクタ注入する。
    """

    DEFAULT_SYSTEM_PROMPT: str = (
        "あなたはドキュメントから質問と回答のペアを生成する専門家です。\n"
        "与えられたテキストに基づいて、正確で多様な質問と回答のペアを生成してください。\n"
        "質問はテキストの内容を理解しているか確認するものにしてください。\n"
        "回答はテキストの内容に基づいた正確なものにしてください。\n\n"
        "以下の JSON 配列形式で回答してください:\n"
        '[{"question": "質問文", "answer": "回答文"}, ...]'
    )

    def __init__(
        self,
        llm_provider: LLMProvider,
        model: str,
        system_prompt: str | None = None,
        num_pairs: int | None = None,
    ) -> None:
        """SyntheticDataGenerator を初期化する。

        Args:
            llm_provider: LLM プロバイダ。
            model: 使用するモデル名。
            system_prompt: カスタムシステムプロンプト。None の場合デフォルトを使用。
            num_pairs: 生成する QA ペア数のデフォルト値。None の場合 3。
        """
        self._provider = llm_provider
        self._model = model
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._num_pairs = num_pairs if num_pairs is not None else 3

    async def generate(
        self,
        text: str,
        num_pairs: int | None = None,
        **kwargs: Any,
    ) -> EvalDataset:
        """テキストから QA ペアを生成し EvalDataset を返す。

        Args:
            text: QA ペア生成元のテキスト。
            num_pairs: 生成する QA ペア数。None の場合コンストラクタのデフォルト値を使用。
            **kwargs: LLM プロバイダに渡す追加パラメータ。

        Returns:
            生成された QA ペアを含む EvalDataset。

        Raises:
            SyntheticDataError: テキストが空、LLM 呼び出し失敗、またはパース失敗の場合。
        """
        if not text or not text.strip():
            raise SyntheticDataError("テキストが空です")

        effective_num_pairs = num_pairs if num_pairs is not None else self._num_pairs
        messages = self._build_messages(text, effective_num_pairs)

        try:
            response = await self._provider.complete(
                messages=messages,
                model=self._model,
                **kwargs,
            )
        except Exception as e:
            raise SyntheticDataError(f"LLM 呼び出しに失敗しました: {e}") from e

        examples = self._parse_response(response.content, text)
        return EvalDataset(name="synthetic", examples=examples)

    async def generate_from_chunks(
        self,
        chunks: list[str],
        num_pairs_per_chunk: int | None = None,
        **kwargs: Any,
    ) -> EvalDataset:
        """複数チャンクからそれぞれ QA ペアを生成し、統合した EvalDataset を返す。

        Args:
            chunks: QA ペア生成元のテキストチャンクのリスト。
            num_pairs_per_chunk: チャンクごとに生成する QA ペア数。
            **kwargs: LLM プロバイダに渡す追加パラメータ。

        Returns:
            全チャンクの QA ペアを統合した EvalDataset。

        Raises:
            SyntheticDataError: チャンクリストが空の場合。
        """
        if not chunks:
            raise SyntheticDataError("チャンクリストが空です")

        all_examples: list[EvalExample] = []
        for chunk in chunks:
            dataset = await self.generate(chunk, num_pairs=num_pairs_per_chunk, **kwargs)
            all_examples.extend(dataset.examples)

        return EvalDataset(name="synthetic", examples=all_examples)

    def _build_messages(self, text: str, num_pairs: int) -> list[ChatMessage]:
        """LLM に送信するメッセージリストを構築する。

        Args:
            text: QA ペア生成元のテキスト。
            num_pairs: 生成する QA ペア数。

        Returns:
            system メッセージと user メッセージのリスト。
        """
        return [
            ChatMessage(role=Role.system, content=self._system_prompt),
            ChatMessage(
                role=Role.user,
                content=(
                    f"以下のテキストから {num_pairs} 個の質問と回答のペアを生成してください。\n\n"
                    f"テキスト:\n{text}"
                ),
            ),
        ]

    def _parse_response(self, content: str, source_text: str) -> list[EvalExample]:
        """LLM レスポンスから EvalExample のリストをパースする。

        マークダウンコードフェンスを除去し、JSON 配列をパースする。
        不正なアイテムはスキップし、有効なもののみ返す。

        Args:
            content: LLM レスポンスの文字列。
            source_text: 生成元テキスト (context に設定)。

        Returns:
            パースされた EvalExample のリスト。

        Raises:
            SyntheticDataError: JSON パース失敗、配列でない、または有効なアイテムがない場合。
        """
        # マークダウンコードフェンスを除去
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", content.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise SyntheticDataError(f"LLM レスポンスの JSON パースに失敗しました: {e}") from e

        if not isinstance(parsed, list):
            raise SyntheticDataError("LLM レスポンスが JSON 配列ではありません")

        examples: list[EvalExample] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            question = item.get("question", "")
            answer = item.get("answer", "")
            if not question or not answer:
                continue
            examples.append(
                EvalExample(
                    query=question,
                    context=source_text,
                    answer=answer,
                    expected_answer=answer,
                )
            )

        if not examples:
            raise SyntheticDataError("有効な QA ペアが生成されませんでした")

        return examples
