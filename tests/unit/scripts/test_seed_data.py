"""Tests for scripts/seed_data.py — initial user seeding."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from scripts.seed_data import SEED_USERS, SeedUserData, seed_users


class TestSeedUserData:
    """SeedUserData モデルと SEED_USERS 定数の検証。"""

    def test_seed_user_data_creation(self) -> None:
        """SeedUserData を正しく生成できる。"""
        user = SeedUserData(
            email="test@example.com",
            name="Test User",
            role="admin",
            password="test-password",
        )
        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.role == "admin"
        assert user.password == "test-password"

    def test_seed_users_contains_three_users(self) -> None:
        """SEED_USERS には3名分のデータが含まれる。"""
        assert len(SEED_USERS) == 3

    def test_seed_users_has_admin(self) -> None:
        """SEED_USERS に admin ユーザーが含まれる。"""
        admin = next(u for u in SEED_USERS if u.role == "admin")
        assert admin.email == "admin@example.com"
        assert admin.name == "Admin User"

    def test_seed_users_has_regular_user(self) -> None:
        """SEED_USERS に user ロールが含まれる。"""
        user = next(u for u in SEED_USERS if u.role == "user")
        assert user.email == "user@example.com"
        assert user.name == "Regular User"

    def test_seed_users_has_viewer(self) -> None:
        """SEED_USERS に viewer ロールが含まれる。"""
        viewer = next(u for u in SEED_USERS if u.role == "viewer")
        assert viewer.email == "viewer@example.com"
        assert viewer.name == "Viewer User"


class TestSeedUsersCreatesNewUsers:
    """全ユーザーが未存在の場合、全員を作成する。"""

    @pytest.mark.asyncio
    async def test_creates_all_users_when_none_exist(self) -> None:
        """全ユーザーが未存在の場合、全員を created として返す。"""
        mock_result = MagicMock()
        mock_result.first.return_value = None

        session = AsyncMock()
        session.exec.return_value = mock_result

        with patch("scripts.seed_data.hash_password", return_value="hashed"):
            results = await seed_users(session)

        created = [r for r in results if r.startswith("created:")]
        assert len(created) == 3

    @pytest.mark.asyncio
    async def test_session_add_called_for_each_new_user(self) -> None:
        """未存在ユーザーごとに session.add が呼ばれる。"""
        mock_result = MagicMock()
        mock_result.first.return_value = None

        session = AsyncMock()
        session.exec.return_value = mock_result

        with patch("scripts.seed_data.hash_password", return_value="hashed"):
            await seed_users(session)

        assert session.add.call_count == 3

    @pytest.mark.asyncio
    async def test_session_commit_called(self) -> None:
        """seed_users は最後に session.commit を呼ぶ。"""
        mock_result = MagicMock()
        mock_result.first.return_value = None

        session = AsyncMock()
        session.exec.return_value = mock_result

        with patch("scripts.seed_data.hash_password", return_value="hashed"):
            await seed_users(session)

        session.commit.assert_awaited_once()


class TestSeedUsersSkipsExisting:
    """全ユーザーが既存の場合、全員をスキップする。"""

    @pytest.mark.asyncio
    async def test_skips_all_existing_users(self) -> None:
        """全ユーザーが既存の場合、全員を skipped として返す。"""
        existing_user = MagicMock()
        mock_result = MagicMock()
        mock_result.first.return_value = existing_user

        session = AsyncMock()
        session.exec.return_value = mock_result

        results = await seed_users(session)

        skipped = [r for r in results if r.startswith("skipped:")]
        assert len(skipped) == 3

    @pytest.mark.asyncio
    async def test_session_add_not_called_for_existing(self) -> None:
        """既存ユーザーに対して session.add は呼ばれない。"""
        existing_user = MagicMock()
        mock_result = MagicMock()
        mock_result.first.return_value = existing_user

        session = AsyncMock()
        session.exec.return_value = mock_result

        results = await seed_users(session)

        session.add.assert_not_called()
        assert all(r.startswith("skipped:") for r in results)


class TestSeedUsersIdempotency:
    """作成とスキップが混在するケース。"""

    @pytest.mark.asyncio
    async def test_mixed_create_and_skip(self) -> None:
        """一部が既存、一部が新規の場合に正しく処理される。"""
        existing_user = MagicMock()

        # 1番目: 存在する, 2番目: 存在しない, 3番目: 存在する
        result_exists = MagicMock()
        result_exists.first.return_value = existing_user
        result_not_exists = MagicMock()
        result_not_exists.first.return_value = None

        session = AsyncMock()
        session.exec.side_effect = [result_exists, result_not_exists, result_exists]

        with patch("scripts.seed_data.hash_password", return_value="hashed"):
            results = await seed_users(session)

        created = [r for r in results if r.startswith("created:")]
        skipped = [r for r in results if r.startswith("skipped:")]
        assert len(created) == 1
        assert len(skipped) == 2
        assert session.add.call_count == 1


class TestSeedUsersPasswordHashing:
    """パスワードが hash_password 経由でハッシュ化される。"""

    @pytest.mark.asyncio
    async def test_hash_password_called_for_each_new_user(self) -> None:
        """新規ユーザーのパスワードが hash_password でハッシュ化される。"""
        mock_result = MagicMock()
        mock_result.first.return_value = None

        session = AsyncMock()
        session.exec.return_value = mock_result

        with patch("scripts.seed_data.hash_password", return_value="hashed") as mock_hash:
            await seed_users(session)

        assert mock_hash.call_count == 3
        # 各ユーザーのパスワードで呼ばれている
        called_passwords = [c.args[0] for c in mock_hash.call_args_list]
        for seed_user in SEED_USERS:
            assert seed_user.password in called_passwords

    @pytest.mark.asyncio
    async def test_hashed_password_stored_in_user(self) -> None:
        """User モデルに hashed_password が設定される。"""
        mock_result = MagicMock()
        mock_result.first.return_value = None

        session = AsyncMock()
        session.exec.return_value = mock_result

        with patch("scripts.seed_data.hash_password", return_value="bcrypt-hashed-value"):
            await seed_users(session)

        # session.add に渡された User オブジェクトの hashed_password を検証
        for add_call in session.add.call_args_list:
            user_obj = add_call.args[0]
            assert user_obj.hashed_password == "bcrypt-hashed-value"
