"""Tests for error classification and analysis (monitoring/error_analyzer.py).

エラー分類・記録・集計を検証する。
"""

from __future__ import annotations

import pytest
from src.monitoring.error_analyzer import ErrorAnalyzer, ErrorCategory, ErrorRecord

# =============================================================================
# ErrorCategory
# =============================================================================


class TestErrorCategory:
    """ErrorCategory 列挙型を検証する。"""

    def test_provider_error_exists(self) -> None:
        """provider_error カテゴリが存在すること。"""
        assert ErrorCategory.PROVIDER_ERROR is not None

    def test_timeout_exists(self) -> None:
        """timeout カテゴリが存在すること。"""
        assert ErrorCategory.TIMEOUT is not None

    def test_rate_limit_exists(self) -> None:
        """rate_limit カテゴリが存在すること。"""
        assert ErrorCategory.RATE_LIMIT is not None

    def test_validation_error_exists(self) -> None:
        """validation_error カテゴリが存在すること。"""
        assert ErrorCategory.VALIDATION_ERROR is not None

    def test_authentication_error_exists(self) -> None:
        """authentication_error カテゴリが存在すること。"""
        assert ErrorCategory.AUTHENTICATION_ERROR is not None

    def test_unknown_exists(self) -> None:
        """unknown カテゴリが存在すること。"""
        assert ErrorCategory.UNKNOWN is not None


# =============================================================================
# ErrorRecord
# =============================================================================


class TestErrorRecord:
    """ErrorRecord データモデルを検証する。"""

    def test_create_error_record(self) -> None:
        """ErrorRecord をインスタンス化できること。"""
        record = ErrorRecord(
            category=ErrorCategory.TIMEOUT,
            error_type="TimeoutError",
            message="Connection timed out",
            provider="openrouter",
        )
        assert record.category == ErrorCategory.TIMEOUT
        assert record.error_type == "TimeoutError"
        assert record.message == "Connection timed out"
        assert record.provider == "openrouter"

    def test_error_record_has_timestamp(self) -> None:
        """ErrorRecord に timestamp が含まれること。"""
        record = ErrorRecord(
            category=ErrorCategory.TIMEOUT,
            error_type="TimeoutError",
            message="timeout",
        )
        assert record.timestamp is not None


# =============================================================================
# ErrorAnalyzer 初期化
# =============================================================================


class TestErrorAnalyzerInit:
    """ErrorAnalyzer の初期化を検証する。"""

    def test_creates_instance(self) -> None:
        """ErrorAnalyzer をインスタンス化できること。"""
        analyzer = ErrorAnalyzer()
        assert analyzer is not None

    def test_initial_error_count_is_zero(self) -> None:
        """初期のエラー件数が 0 であること。"""
        analyzer = ErrorAnalyzer()
        assert analyzer.total_error_count() == 0


# =============================================================================
# エラー分類
# =============================================================================


class TestErrorClassification:
    """例外からのエラー分類を検証する。"""

    @pytest.fixture
    def analyzer(self) -> ErrorAnalyzer:
        return ErrorAnalyzer()

    def test_classify_timeout_error(self, analyzer: ErrorAnalyzer) -> None:
        """TimeoutError が TIMEOUT に分類されること。"""
        category = analyzer.classify(TimeoutError("connection timed out"))
        assert category == ErrorCategory.TIMEOUT

    def test_classify_connection_error(self, analyzer: ErrorAnalyzer) -> None:
        """ConnectionError が PROVIDER_ERROR に分類されること。"""
        category = analyzer.classify(ConnectionError("refused"))
        assert category == ErrorCategory.PROVIDER_ERROR

    def test_classify_permission_error(self, analyzer: ErrorAnalyzer) -> None:
        """PermissionError が AUTHENTICATION_ERROR に分類されること。"""
        category = analyzer.classify(PermissionError("forbidden"))
        assert category == ErrorCategory.AUTHENTICATION_ERROR

    def test_classify_value_error(self, analyzer: ErrorAnalyzer) -> None:
        """ValueError が VALIDATION_ERROR に分類されること。"""
        category = analyzer.classify(ValueError("invalid input"))
        assert category == ErrorCategory.VALIDATION_ERROR

    def test_classify_unknown_exception(self, analyzer: ErrorAnalyzer) -> None:
        """未知の例外が UNKNOWN に分類されること。"""
        category = analyzer.classify(RuntimeError("something went wrong"))
        assert category == ErrorCategory.UNKNOWN

    def test_classify_httpx_timeout(self, analyzer: ErrorAnalyzer) -> None:
        """httpx.TimeoutException が TIMEOUT に分類されること。"""
        import httpx

        category = analyzer.classify(httpx.TimeoutException("read timeout"))
        assert category == ErrorCategory.TIMEOUT

    def test_classify_httpx_connect_error(self, analyzer: ErrorAnalyzer) -> None:
        """httpx.ConnectError が PROVIDER_ERROR に分類されること。"""
        import httpx

        category = analyzer.classify(httpx.ConnectError("connection refused"))
        assert category == ErrorCategory.PROVIDER_ERROR


# =============================================================================
# エラー記録
# =============================================================================


class TestErrorRecording:
    """エラーの記録を検証する。"""

    @pytest.fixture
    def analyzer(self) -> ErrorAnalyzer:
        return ErrorAnalyzer()

    def test_record_error_increments_count(self, analyzer: ErrorAnalyzer) -> None:
        """record() でエラー件数が増加すること。"""
        analyzer.record(TimeoutError("timeout"), provider="openrouter")

        assert analyzer.total_error_count() == 1

    def test_record_multiple_errors(self, analyzer: ErrorAnalyzer) -> None:
        """複数のエラーを記録できること。"""
        analyzer.record(TimeoutError("timeout"), provider="openrouter")
        analyzer.record(ConnectionError("refused"), provider="openai")
        analyzer.record(ValueError("invalid"), provider="openrouter")

        assert analyzer.total_error_count() == 3

    def test_record_returns_error_record(self, analyzer: ErrorAnalyzer) -> None:
        """record() が ErrorRecord を返すこと。"""
        record = analyzer.record(TimeoutError("timeout"), provider="openrouter")

        assert isinstance(record, ErrorRecord)
        assert record.category == ErrorCategory.TIMEOUT
        assert record.provider == "openrouter"

    def test_record_without_provider(self, analyzer: ErrorAnalyzer) -> None:
        """provider を省略してもエラーを記録できること。"""
        record = analyzer.record(ValueError("bad input"))

        assert record.provider is None
        assert analyzer.total_error_count() == 1


# =============================================================================
# エラー集計
# =============================================================================


class TestErrorSummary:
    """エラー集計 (get_summary) を検証する。"""

    @pytest.fixture
    def analyzer(self) -> ErrorAnalyzer:
        a = ErrorAnalyzer()
        a.record(TimeoutError("t1"), provider="openrouter")
        a.record(TimeoutError("t2"), provider="openrouter")
        a.record(ConnectionError("c1"), provider="openai")
        a.record(ValueError("v1"), provider="openrouter")
        return a

    def test_summary_total_count(self, analyzer: ErrorAnalyzer) -> None:
        """集計の合計件数が正しいこと。"""
        summary = analyzer.get_summary()
        assert summary["total_errors"] == 4

    def test_summary_by_category(self, analyzer: ErrorAnalyzer) -> None:
        """カテゴリ別の集計が正しいこと。"""
        summary = analyzer.get_summary()
        by_category = summary["by_category"]

        assert by_category[ErrorCategory.TIMEOUT] == 2
        assert by_category[ErrorCategory.PROVIDER_ERROR] == 1
        assert by_category[ErrorCategory.VALIDATION_ERROR] == 1

    def test_summary_by_provider(self, analyzer: ErrorAnalyzer) -> None:
        """プロバイダ別の集計が正しいこと。"""
        summary = analyzer.get_summary()
        by_provider = summary["by_provider"]

        assert by_provider["openrouter"] == 3
        assert by_provider["openai"] == 1

    def test_empty_summary(self) -> None:
        """エラーがない場合の集計が正しいこと。"""
        analyzer = ErrorAnalyzer()
        summary = analyzer.get_summary()

        assert summary["total_errors"] == 0
        assert summary["by_category"] == {}
        assert summary["by_provider"] == {}


# =============================================================================
# 最近のエラー取得
# =============================================================================


class TestRecentErrors:
    """最近のエラー取得 (get_recent) を検証する。"""

    @pytest.fixture
    def analyzer(self) -> ErrorAnalyzer:
        a = ErrorAnalyzer()
        a.record(TimeoutError("t1"), provider="openrouter")
        a.record(ConnectionError("c1"), provider="openai")
        a.record(ValueError("v1"), provider="openrouter")
        return a

    def test_get_recent_returns_list(self, analyzer: ErrorAnalyzer) -> None:
        """get_recent() がリストを返すこと。"""
        recent = analyzer.get_recent(n=2)
        assert isinstance(recent, list)

    def test_get_recent_respects_limit(self, analyzer: ErrorAnalyzer) -> None:
        """get_recent() が指定件数を返すこと。"""
        recent = analyzer.get_recent(n=2)
        assert len(recent) == 2

    def test_get_recent_returns_newest_first(self, analyzer: ErrorAnalyzer) -> None:
        """get_recent() が新しいエラーから順に返すこと。"""
        recent = analyzer.get_recent(n=3)
        assert recent[0].message == "v1"
        assert recent[1].message == "c1"
        assert recent[2].message == "t1"

    def test_get_recent_with_n_larger_than_total(self, analyzer: ErrorAnalyzer) -> None:
        """n が総数より大きい場合に全件を返すこと。"""
        recent = analyzer.get_recent(n=100)
        assert len(recent) == 3

    def test_get_recent_empty(self) -> None:
        """エラーがない場合に空リストを返すこと。"""
        analyzer = ErrorAnalyzer()
        recent = analyzer.get_recent(n=5)
        assert recent == []


# =============================================================================
# エラー記録のクリア
# =============================================================================


class TestClearErrors:
    """エラー記録のクリアを検証する。"""

    def test_clear_removes_all_errors(self) -> None:
        """clear() で全エラーが削除されること。"""
        analyzer = ErrorAnalyzer()
        analyzer.record(TimeoutError("t1"))
        analyzer.record(ConnectionError("c1"))

        analyzer.clear()

        assert analyzer.total_error_count() == 0
        assert analyzer.get_recent(n=10) == []
