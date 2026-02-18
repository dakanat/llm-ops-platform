"""PromptManager のユニットテスト。

テンプレート登録、取得、変数埋め込み、バージョン管理を検証する。
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest
from src.llm import PromptError, PromptNotFoundError, PromptRenderError
from src.llm.prompt_manager import PromptManager, PromptTemplate


# ---------------------------------------------------------------------------
# TestPromptErrorHierarchy
# ---------------------------------------------------------------------------
class TestPromptErrorHierarchy:
    """例外クラスの継承関係を検証する。"""

    def test_prompt_error_is_exception(self) -> None:
        assert issubclass(PromptError, Exception)

    def test_prompt_not_found_error_is_prompt_error(self) -> None:
        assert issubclass(PromptNotFoundError, PromptError)

    def test_prompt_render_error_is_prompt_error(self) -> None:
        assert issubclass(PromptRenderError, PromptError)

    def test_prompt_error_can_be_raised_with_message(self) -> None:
        with pytest.raises(PromptError, match="test message"):
            raise PromptError("test message")


# ---------------------------------------------------------------------------
# TestPromptTemplate
# ---------------------------------------------------------------------------
class TestPromptTemplate:
    """PromptTemplate モデルのフィールドとデフォルト値を検証する。"""

    def test_create_with_required_fields(self) -> None:
        tpl = PromptTemplate(name="test", template="Hello {name}")
        assert tpl.name == "test"
        assert tpl.template == "Hello {name}"

    def test_default_version(self) -> None:
        tpl = PromptTemplate(name="test", template="Hello")
        assert tpl.version == "1.0.0"

    def test_default_description_is_empty(self) -> None:
        tpl = PromptTemplate(name="test", template="Hello")
        assert tpl.description == ""

    def test_default_metadata_is_empty_dict(self) -> None:
        tpl = PromptTemplate(name="test", template="Hello")
        assert tpl.metadata == {}

    def test_created_at_is_utc_datetime(self) -> None:
        before = datetime.now(UTC)
        tpl = PromptTemplate(name="test", template="Hello")
        after = datetime.now(UTC)
        assert before <= tpl.created_at <= after

    def test_custom_fields(self) -> None:
        tpl = PromptTemplate(
            name="qa",
            version="2.0.0",
            template="Q: {question}\nA:",
            description="QA prompt",
            metadata={"author": "taro"},
        )
        assert tpl.name == "qa"
        assert tpl.version == "2.0.0"
        assert tpl.description == "QA prompt"
        assert tpl.metadata == {"author": "taro"}

    def test_metadata_default_not_shared_between_instances(self) -> None:
        tpl1 = PromptTemplate(name="a", template="a")
        tpl2 = PromptTemplate(name="b", template="b")
        tpl1.metadata["key"] = "value"
        assert "key" not in tpl2.metadata


# ---------------------------------------------------------------------------
# TestPromptManagerRegister
# ---------------------------------------------------------------------------
class TestPromptManagerRegister:
    """テンプレートの登録を検証する。"""

    def test_register_and_get(self) -> None:
        manager = PromptManager()
        tpl = PromptTemplate(name="greet", template="Hello {name}")
        manager.register(tpl)
        result = manager.get("greet")
        assert result.template == "Hello {name}"

    def test_register_multiple_versions(self) -> None:
        manager = PromptManager()
        tpl_v1 = PromptTemplate(name="greet", version="1.0.0", template="Hi {name}")
        tpl_v2 = PromptTemplate(name="greet", version="2.0.0", template="Hello {name}")
        manager.register(tpl_v1)
        manager.register(tpl_v2)
        assert manager.get("greet", version="1.0.0").template == "Hi {name}"
        assert manager.get("greet", version="2.0.0").template == "Hello {name}"

    def test_register_same_name_and_version_overwrites(self) -> None:
        manager = PromptManager()
        tpl1 = PromptTemplate(name="greet", version="1.0.0", template="Old")
        tpl2 = PromptTemplate(name="greet", version="1.0.0", template="New")
        manager.register(tpl1)
        manager.register(tpl2)
        assert manager.get("greet", version="1.0.0").template == "New"


# ---------------------------------------------------------------------------
# TestPromptManagerGet
# ---------------------------------------------------------------------------
class TestPromptManagerGet:
    """テンプレートの取得を検証する。"""

    def test_get_specific_version(self) -> None:
        manager = PromptManager()
        tpl = PromptTemplate(name="greet", version="1.0.0", template="Hi {name}")
        manager.register(tpl)
        result = manager.get("greet", version="1.0.0")
        assert result.template == "Hi {name}"

    def test_get_latest_version_when_no_version_specified(self) -> None:
        manager = PromptManager()
        tpl_v1 = PromptTemplate(name="greet", version="1.0.0", template="Hi {name}")
        manager.register(tpl_v1)
        # Small delay to ensure different created_at timestamps
        time.sleep(0.01)
        tpl_v2 = PromptTemplate(name="greet", version="2.0.0", template="Hello {name}")
        manager.register(tpl_v2)
        result = manager.get("greet")
        assert result.template == "Hello {name}"

    def test_get_nonexistent_name_raises_not_found(self) -> None:
        manager = PromptManager()
        with pytest.raises(PromptNotFoundError, match="nonexistent"):
            manager.get("nonexistent")

    def test_get_nonexistent_version_raises_not_found(self) -> None:
        manager = PromptManager()
        tpl = PromptTemplate(name="greet", version="1.0.0", template="Hi")
        manager.register(tpl)
        with pytest.raises(PromptNotFoundError, match="9.9.9"):
            manager.get("greet", version="9.9.9")

    def test_get_returns_prompt_template_instance(self) -> None:
        manager = PromptManager()
        tpl = PromptTemplate(name="greet", template="Hi")
        manager.register(tpl)
        result = manager.get("greet")
        assert isinstance(result, PromptTemplate)

    def test_get_latest_after_overwrite(self) -> None:
        manager = PromptManager()
        tpl_old = PromptTemplate(name="greet", version="1.0.0", template="Old")
        manager.register(tpl_old)
        time.sleep(0.01)
        tpl_new = PromptTemplate(name="greet", version="1.0.0", template="New")
        manager.register(tpl_new)
        result = manager.get("greet")
        assert result.template == "New"


# ---------------------------------------------------------------------------
# TestPromptManagerRender
# ---------------------------------------------------------------------------
class TestPromptManagerRender:
    """変数埋め込み (render) を検証する。"""

    def test_render_simple_variable(self) -> None:
        manager = PromptManager()
        tpl = PromptTemplate(name="greet", template="Hello {name}")
        manager.register(tpl)
        result = manager.render("greet", {"name": "World"})
        assert result == "Hello World"

    def test_render_multiple_variables(self) -> None:
        manager = PromptManager()
        tpl = PromptTemplate(name="qa", template="Q: {question}\nA: {answer}")
        manager.register(tpl)
        result = manager.render("qa", {"question": "What?", "answer": "This."})
        assert result == "Q: What?\nA: This."

    def test_render_missing_variable_raises_render_error(self) -> None:
        manager = PromptManager()
        tpl = PromptTemplate(name="greet", template="Hello {name}")
        manager.register(tpl)
        with pytest.raises(PromptRenderError, match="name"):
            manager.render("greet", {})

    def test_render_nonexistent_template_raises_not_found(self) -> None:
        manager = PromptManager()
        with pytest.raises(PromptNotFoundError):
            manager.render("nonexistent", {"key": "value"})

    def test_render_specific_version(self) -> None:
        manager = PromptManager()
        tpl_v1 = PromptTemplate(name="greet", version="1.0.0", template="Hi {name}")
        tpl_v2 = PromptTemplate(name="greet", version="2.0.0", template="Hello {name}")
        manager.register(tpl_v1)
        manager.register(tpl_v2)
        result = manager.render("greet", {"name": "World"}, version="1.0.0")
        assert result == "Hi World"

    def test_render_japanese_variables(self) -> None:
        manager = PromptManager()
        tpl = PromptTemplate(
            name="qa_ja",
            template="質問: {question}\n回答:",
        )
        manager.register(tpl)
        result = manager.render("qa_ja", {"question": "日本語のテスト"})
        assert result == "質問: 日本語のテスト\n回答:"

    def test_render_extra_variables_are_ignored(self) -> None:
        manager = PromptManager()
        tpl = PromptTemplate(name="greet", template="Hello {name}")
        manager.register(tpl)
        result = manager.render("greet", {"name": "World", "extra": "ignored"})
        assert result == "Hello World"

    def test_render_literal_braces_escaped(self) -> None:
        manager = PromptManager()
        tpl = PromptTemplate(name="code", template="Use {{curly}} with {name}")
        manager.register(tpl)
        result = manager.render("code", {"name": "test"})
        assert result == "Use {curly} with test"


# ---------------------------------------------------------------------------
# TestPromptManagerRenderTemplate
# ---------------------------------------------------------------------------
class TestPromptManagerRenderTemplate:
    """レジストリ経由せず直接レンダリングを検証する。"""

    def test_render_template_directly(self) -> None:
        manager = PromptManager()
        tpl = PromptTemplate(name="direct", template="Hello {name}")
        result = manager.render_template(tpl, {"name": "Direct"})
        assert result == "Hello Direct"

    def test_render_template_missing_variable_raises_render_error(self) -> None:
        manager = PromptManager()
        tpl = PromptTemplate(name="direct", template="Hello {name}")
        with pytest.raises(PromptRenderError, match="name"):
            manager.render_template(tpl, {})


# ---------------------------------------------------------------------------
# TestPromptManagerListTemplates
# ---------------------------------------------------------------------------
class TestPromptManagerListTemplates:
    """テンプレート一覧の取得を検証する。"""

    def test_list_templates_empty(self) -> None:
        manager = PromptManager()
        assert manager.list_templates() == []

    def test_list_templates_returns_sorted_names(self) -> None:
        manager = PromptManager()
        manager.register(PromptTemplate(name="charlie", template="c"))
        manager.register(PromptTemplate(name="alpha", template="a"))
        manager.register(PromptTemplate(name="bravo", template="b"))
        assert manager.list_templates() == ["alpha", "bravo", "charlie"]

    def test_list_templates_no_duplicates_for_multiple_versions(self) -> None:
        manager = PromptManager()
        manager.register(PromptTemplate(name="greet", version="1.0.0", template="a"))
        manager.register(PromptTemplate(name="greet", version="2.0.0", template="b"))
        assert manager.list_templates() == ["greet"]

    def test_list_templates_returns_new_list_each_time(self) -> None:
        manager = PromptManager()
        manager.register(PromptTemplate(name="test", template="t"))
        result1 = manager.list_templates()
        result2 = manager.list_templates()
        assert result1 == result2
        assert result1 is not result2


# ---------------------------------------------------------------------------
# TestPromptManagerListVersions
# ---------------------------------------------------------------------------
class TestPromptManagerListVersions:
    """バージョン一覧の取得を検証する。"""

    def test_list_versions_returns_all_versions_sorted_by_created_at(self) -> None:
        manager = PromptManager()
        tpl_v1 = PromptTemplate(name="greet", version="1.0.0", template="Hi")
        manager.register(tpl_v1)
        time.sleep(0.01)
        tpl_v2 = PromptTemplate(name="greet", version="2.0.0", template="Hello")
        manager.register(tpl_v2)
        versions = manager.list_versions("greet")
        assert len(versions) == 2
        assert versions[0].version == "1.0.0"
        assert versions[1].version == "2.0.0"

    def test_list_versions_nonexistent_name_raises_not_found(self) -> None:
        manager = PromptManager()
        with pytest.raises(PromptNotFoundError, match="nonexistent"):
            manager.list_versions("nonexistent")

    def test_list_versions_returns_prompt_template_instances(self) -> None:
        manager = PromptManager()
        manager.register(PromptTemplate(name="greet", version="1.0.0", template="Hi"))
        versions = manager.list_versions("greet")
        assert all(isinstance(v, PromptTemplate) for v in versions)
