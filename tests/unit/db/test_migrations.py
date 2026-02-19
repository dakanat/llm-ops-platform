"""Tests for Alembic initial migration script."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

VERSIONS_DIR = Path("src/db/migrations/versions")
EXPECTED_TABLES = {"users", "documents", "chunks", "conversations", "messages", "audit_logs"}


def _load_migration_module() -> object:
    """Load the single migration file from versions/ as a Python module."""
    py_files = sorted(VERSIONS_DIR.glob("*.py"))
    assert len(py_files) == 1, f"Expected 1 migration file, found {len(py_files)}"
    spec = importlib.util.spec_from_file_location("migration", py_files[0])
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestInitialMigration:
    """初期マイグレーションスクリプトの構造・内容を検証するテスト。"""

    def test_initial_migration_file_exists(self) -> None:
        """versions/ ディレクトリにマイグレーションファイルが1つ存在すること。"""
        py_files = sorted(VERSIONS_DIR.glob("*.py"))
        assert len(py_files) == 1

    def test_initial_migration_creates_all_tables(self) -> None:
        """upgrade 関数が全6テーブルを作成すること。"""
        module = _load_migration_module()
        source = inspect.getsource(module.upgrade)  # type: ignore[attr-defined]
        for table in EXPECTED_TABLES:
            assert f"'{table}'" in source or f'"{table}"' in source, (
                f"upgrade() should create table '{table}'"
            )
        assert source.count("create_table") == len(EXPECTED_TABLES)

    def test_initial_migration_drops_all_tables(self) -> None:
        """downgrade 関数が全6テーブルを削除すること。"""
        module = _load_migration_module()
        source = inspect.getsource(module.downgrade)  # type: ignore[attr-defined]
        for table in EXPECTED_TABLES:
            assert f"'{table}'" in source or f'"{table}"' in source, (
                f"downgrade() should drop table '{table}'"
            )
        assert source.count("drop_table") == len(EXPECTED_TABLES)

    def test_initial_migration_creates_pgvector_extension(self) -> None:
        """upgrade 関数が vector 拡張を有効化すること。"""
        module = _load_migration_module()
        source = inspect.getsource(module.upgrade)  # type: ignore[attr-defined]
        assert "CREATE EXTENSION IF NOT EXISTS vector" in source

    def test_initial_migration_has_no_down_revision(self) -> None:
        """最初のマイグレーションなので down_revision が None であること。"""
        module = _load_migration_module()
        assert module.down_revision is None  # type: ignore[attr-defined]
