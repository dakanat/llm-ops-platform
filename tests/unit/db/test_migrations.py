"""Tests for Alembic migration scripts."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

VERSIONS_DIR = Path("src/db/migrations/versions")
INITIAL_TABLES = {"users", "documents", "chunks", "conversations", "messages", "audit_logs"}
EVAL_TABLES = {"eval_datasets", "eval_examples"}


def _load_migration_module(filename_pattern: str) -> object:
    """Load a migration file matching the given pattern from versions/."""
    matches = [f for f in VERSIONS_DIR.glob("*.py") if filename_pattern in f.name]
    assert len(matches) == 1, f"Expected 1 file matching '{filename_pattern}', found {len(matches)}"
    spec = importlib.util.spec_from_file_location("migration", matches[0])
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestInitialMigration:
    """初期マイグレーションスクリプトの構造・内容を検証するテスト。"""

    def test_initial_migration_file_exists(self) -> None:
        """versions/ ディレクトリに初期マイグレーションファイルが存在すること。"""
        matches = [f for f in VERSIONS_DIR.glob("*.py") if "initial_schema" in f.name]
        assert len(matches) == 1

    def test_initial_migration_creates_all_tables(self) -> None:
        """upgrade 関数が全6テーブルを作成すること。"""
        module = _load_migration_module("initial_schema")
        source = inspect.getsource(module.upgrade)  # type: ignore[attr-defined]
        for table in INITIAL_TABLES:
            assert f"'{table}'" in source or f'"{table}"' in source, (
                f"upgrade() should create table '{table}'"
            )
        assert source.count("create_table") == len(INITIAL_TABLES)

    def test_initial_migration_drops_all_tables(self) -> None:
        """downgrade 関数が全6テーブルを削除すること。"""
        module = _load_migration_module("initial_schema")
        source = inspect.getsource(module.downgrade)  # type: ignore[attr-defined]
        for table in INITIAL_TABLES:
            assert f"'{table}'" in source or f'"{table}"' in source, (
                f"downgrade() should drop table '{table}'"
            )
        assert source.count("drop_table") == len(INITIAL_TABLES)

    def test_initial_migration_creates_pgvector_extension(self) -> None:
        """upgrade 関数が vector 拡張を有効化すること。"""
        module = _load_migration_module("initial_schema")
        source = inspect.getsource(module.upgrade)  # type: ignore[attr-defined]
        assert "CREATE EXTENSION IF NOT EXISTS vector" in source

    def test_initial_migration_has_no_down_revision(self) -> None:
        """最初のマイグレーションなので down_revision が None であること。"""
        module = _load_migration_module("initial_schema")
        assert module.down_revision is None  # type: ignore[attr-defined]


class TestEvalDatasetsMigration:
    """eval_datasets マイグレーションスクリプトの構造・内容を検証するテスト。"""

    def test_migration_file_exists(self) -> None:
        """eval_datasets マイグレーションファイルが存在すること。"""
        matches = [f for f in VERSIONS_DIR.glob("*.py") if "eval_datasets" in f.name]
        assert len(matches) == 1

    def test_creates_eval_tables(self) -> None:
        """upgrade 関数が eval_datasets と eval_examples テーブルを作成すること。"""
        module = _load_migration_module("eval_datasets")
        source = inspect.getsource(module.upgrade)  # type: ignore[attr-defined]
        for table in EVAL_TABLES:
            assert f"'{table}'" in source or f'"{table}"' in source, (
                f"upgrade() should create table '{table}'"
            )
        assert source.count("create_table") == len(EVAL_TABLES)

    def test_drops_eval_tables(self) -> None:
        """downgrade 関数が eval テーブルを削除すること。"""
        module = _load_migration_module("eval_datasets")
        source = inspect.getsource(module.downgrade)  # type: ignore[attr-defined]
        for table in EVAL_TABLES:
            assert f"'{table}'" in source or f'"{table}"' in source, (
                f"downgrade() should drop table '{table}'"
            )
        assert source.count("drop_table") == len(EVAL_TABLES)

    def test_references_initial_migration(self) -> None:
        """down_revision が初期マイグレーションを参照すること。"""
        module = _load_migration_module("eval_datasets")
        assert module.down_revision == "1016d6b8e887"  # type: ignore[attr-defined]

    def test_cascade_delete_on_examples(self) -> None:
        """eval_examples の FK に CASCADE 削除が設定されていること。"""
        module = _load_migration_module("eval_datasets")
        source = inspect.getsource(module.upgrade)  # type: ignore[attr-defined]
        assert "CASCADE" in source
