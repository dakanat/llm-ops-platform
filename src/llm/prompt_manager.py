"""プロンプトテンプレート管理。

テンプレートの登録、バージョン管理、変数埋め込みを提供する。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from src.llm import PromptNotFoundError, PromptRenderError


class PromptTemplate(BaseModel):
    """プロンプトテンプレート。

    Attributes:
        name: テンプレートの一意識別名。
        version: バージョン文字列。
        template: ``{variable}`` 形式の変数を含むテンプレート文字列。
        description: テンプレートの説明。
        metadata: 任意のメタデータ。
        created_at: 作成日時 (UTC)。
    """

    name: str
    version: str = "1.0.0"
    template: str
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PromptManager:
    """プロンプトテンプレートのレジストリ。

    テンプレートを名前とバージョンで管理し、変数埋め込みを行う。

    内部ストレージは ``dict[name, dict[version, PromptTemplate]]`` 形式。
    """

    def __init__(self) -> None:
        self._templates: dict[str, dict[str, PromptTemplate]] = {}

    def register(self, template: PromptTemplate) -> None:
        """テンプレートを登録する。

        同一 name + version が既に存在する場合は上書きする。

        Args:
            template: 登録するテンプレート。
        """
        if template.name not in self._templates:
            self._templates[template.name] = {}
        self._templates[template.name][template.version] = template

    def get(self, name: str, *, version: str | None = None) -> PromptTemplate:
        """テンプレートを取得する。

        Args:
            name: テンプレート名。
            version: バージョン文字列。None の場合は created_at が最大のものを返す。

        Returns:
            対応する PromptTemplate。

        Raises:
            PromptNotFoundError: テンプレートが見つからない場合。
        """
        versions = self._templates.get(name)
        if versions is None:
            raise PromptNotFoundError(f"Template not found: {name}")

        if version is not None:
            tpl = versions.get(version)
            if tpl is None:
                raise PromptNotFoundError(f"Template '{name}' version {version} not found")
            return tpl

        # version=None: created_at が最大のものを返す
        return max(versions.values(), key=lambda t: t.created_at)

    def render(
        self,
        name: str,
        variables: dict[str, Any],
        *,
        version: str | None = None,
    ) -> str:
        """テンプレートを取得し、変数を埋め込んで文字列を返す。

        Args:
            name: テンプレート名。
            variables: 埋め込み変数の辞書。
            version: バージョン文字列。None の場合は最新バージョン。

        Returns:
            変数埋め込み済みの文字列。

        Raises:
            PromptNotFoundError: テンプレートが見つからない場合。
            PromptRenderError: 変数埋め込みに失敗した場合。
        """
        tpl = self.get(name, version=version)
        return self.render_template(tpl, variables)

    def render_template(self, template: PromptTemplate, variables: dict[str, Any]) -> str:
        """レジストリを経由せず、テンプレートに直接変数を埋め込む。

        Args:
            template: テンプレート。
            variables: 埋め込み変数の辞書。

        Returns:
            変数埋め込み済みの文字列。

        Raises:
            PromptRenderError: 変数埋め込みに失敗した場合。
        """
        try:
            return template.template.format_map(variables)
        except KeyError as e:
            raise PromptRenderError(f"Missing variable {e} in template '{template.name}'") from e

    def list_templates(self) -> list[str]:
        """登録済みテンプレート名の一覧を返す。

        Returns:
            ソート済みのテンプレート名リスト。
        """
        return sorted(self._templates.keys())

    def list_versions(self, name: str) -> list[PromptTemplate]:
        """指定テンプレートの全バージョンを返す。

        Args:
            name: テンプレート名。

        Returns:
            created_at 昇順でソートされた PromptTemplate リスト。

        Raises:
            PromptNotFoundError: テンプレートが見つからない場合。
        """
        versions = self._templates.get(name)
        if versions is None:
            raise PromptNotFoundError(f"Template not found: {name}")
        return sorted(versions.values(), key=lambda t: t.created_at)
