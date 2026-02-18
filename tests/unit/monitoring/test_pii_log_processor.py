"""Tests for structlog PII masking processor.

ログ出力前にevent_dict中の文字列値のPIIをマスクすること、
構造的キーをスキップすること、無効時はパススルーすることを検証する。
"""

from __future__ import annotations

from typing import Any

import pytest
from src.monitoring.pii_log_processor import create_pii_masking_processor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_processor(
    event_dict: dict[str, Any],
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    """プロセッサを実行してevent_dictを返す。"""
    processor = create_pii_masking_processor(enabled=enabled)
    # structlog processors are called as (logger, method_name, event_dict)
    from typing import cast

    result = processor(None, "info", event_dict)
    return cast(dict[str, Any], result)


# ---------------------------------------------------------------------------
# PII masking in string values
# ---------------------------------------------------------------------------


class TestMasksPIIInStringValues:
    """文字列値中のPIIがマスクされること。"""

    def test_masks_email_in_string_value(self) -> None:
        """メールアドレスがマスクされること。"""
        result = _run_processor({"user_input": "連絡先: user@example.com"})
        assert "user@example.com" not in result["user_input"]
        assert "[EMAIL]" in result["user_input"]

    def test_masks_phone_in_string_value(self) -> None:
        """電話番号がマスクされること。"""
        result = _run_processor({"query": "電話: 090-1234-5678"})
        assert "090-1234-5678" not in result["query"]
        assert "[PHONE]" in result["query"]

    def test_masks_multiple_pii_types(self) -> None:
        """複数種類のPIIが同時にマスクされること。"""
        result = _run_processor({"data": "メール: user@example.com 電話: 090-1234-5678"})
        assert "user@example.com" not in result["data"]
        assert "090-1234-5678" not in result["data"]


# ---------------------------------------------------------------------------
# Exempt keys
# ---------------------------------------------------------------------------


class TestPreservesExemptKeys:
    """構造的キーの値はマスクされないこと。"""

    @pytest.mark.parametrize(
        "key",
        ["timestamp", "log_level", "event", "request_id", "logger_name", "level"],
    )
    def test_preserves_exempt_key(self, key: str) -> None:
        """免除対象のキーはスキャンされないこと。"""
        result = _run_processor({key: "user@example.com"})
        assert result[key] == "user@example.com"


# ---------------------------------------------------------------------------
# Non-string values
# ---------------------------------------------------------------------------


class TestPreservesNonStringValues:
    """非文字列の値はそのまま保持されること。"""

    def test_preserves_int_value(self) -> None:
        result = _run_processor({"count": 42})
        assert result["count"] == 42

    def test_preserves_dict_value(self) -> None:
        result = _run_processor({"nested": {"key": "value"}})
        assert result["nested"] == {"key": "value"}

    def test_preserves_list_value(self) -> None:
        result = _run_processor({"items": [1, 2, 3]})
        assert result["items"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Disabled mode
# ---------------------------------------------------------------------------


class TestDisabledMode:
    """無効時は値をマスクしないこと。"""

    def test_disabled_mode_passes_through(self) -> None:
        result = _run_processor(
            {"user_input": "連絡先: user@example.com"},
            enabled=False,
        )
        assert result["user_input"] == "連絡先: user@example.com"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """エッジケース。"""

    def test_handles_empty_string_values(self) -> None:
        """空文字列を処理してもエラーにならないこと。"""
        result = _run_processor({"data": ""})
        assert result["data"] == ""
