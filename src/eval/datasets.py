"""評価データセット管理。

評価用のクエリ・コンテキスト・回答セットを JSON ファイルとして読み書きする。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from src.eval import DatasetError


class EvalExample(BaseModel):
    """評価用の1件のサンプル。

    Attributes:
        query: ユーザーの質問。
        context: 検索で取得されたコンテキスト。
        answer: LLM が生成した回答。
        expected_answer: 期待される回答 (オプション)。
    """

    query: str
    context: str
    answer: str
    expected_answer: str | None = None


class EvalDataset(BaseModel):
    """評価データセット。

    Attributes:
        name: データセット名。
        examples: 評価サンプルのリスト。
    """

    name: str
    examples: list[EvalExample]


def load_dataset(path: Path) -> EvalDataset:
    """JSON ファイルからデータセットを読み込む。

    Args:
        path: JSON ファイルのパス。

    Returns:
        読み込んだ EvalDataset。

    Raises:
        DatasetError: ファイルが存在しない、JSON が不正、またはバリデーションに失敗した場合。
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise DatasetError(f"データセットファイルが見つかりません: {path}") from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise DatasetError(f"JSON のパースに失敗しました: {path}") from e

    try:
        return EvalDataset.model_validate(data)
    except ValidationError as e:
        raise DatasetError(f"データセットのバリデーションに失敗しました: {e}") from e


def save_dataset(dataset: EvalDataset, path: Path) -> None:
    """データセットを JSON ファイルに保存する。

    Args:
        dataset: 保存するデータセット。
        path: 出力先の JSON ファイルパス。
    """
    data = dataset.model_dump(mode="json")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
