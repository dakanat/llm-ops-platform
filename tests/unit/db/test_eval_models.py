"""Tests for EvalDatasetRecord and EvalExampleRecord DB models."""

import uuid
from datetime import UTC, datetime


class TestEvalDatasetRecordModel:
    """EvalDatasetRecord モデルのインスタンス化・バリデーションテスト。"""

    def test_creates_with_required_fields(self) -> None:
        """必須フィールドで EvalDatasetRecord が生成できること。"""
        from src.db.models import EvalDatasetRecord

        user_id = uuid.uuid4()
        record = EvalDatasetRecord(
            name="test-dataset",
            created_by=user_id,
        )

        assert record.name == "test-dataset"
        assert record.created_by == user_id

    def test_id_defaults_to_uuid(self) -> None:
        """id がデフォルトで UUID を生成すること。"""
        from src.db.models import EvalDatasetRecord

        record = EvalDatasetRecord(
            name="test-dataset",
            created_by=uuid.uuid4(),
        )

        assert record.id is not None
        assert isinstance(record.id, uuid.UUID)

    def test_description_defaults_to_none(self) -> None:
        """description のデフォルト値が None であること。"""
        from src.db.models import EvalDatasetRecord

        record = EvalDatasetRecord(
            name="test-dataset",
            created_by=uuid.uuid4(),
        )

        assert record.description is None

    def test_description_can_be_set(self) -> None:
        """description に値を設定できること。"""
        from src.db.models import EvalDatasetRecord

        record = EvalDatasetRecord(
            name="test-dataset",
            description="A test dataset for RAG evaluation",
            created_by=uuid.uuid4(),
        )

        assert record.description == "A test dataset for RAG evaluation"

    def test_created_at_defaults(self) -> None:
        """created_at がデフォルトで現在時刻付近を返すこと。"""
        from src.db.models import EvalDatasetRecord

        before = datetime.now(UTC)
        record = EvalDatasetRecord(
            name="test-dataset",
            created_by=uuid.uuid4(),
        )
        after = datetime.now(UTC)

        assert record.created_at is not None
        assert before <= record.created_at <= after

    def test_updated_at_defaults(self) -> None:
        """updated_at がデフォルトで現在時刻付近を返すこと。"""
        from src.db.models import EvalDatasetRecord

        before = datetime.now(UTC)
        record = EvalDatasetRecord(
            name="test-dataset",
            created_by=uuid.uuid4(),
        )
        after = datetime.now(UTC)

        assert record.updated_at is not None
        assert before <= record.updated_at <= after

    def test_is_table_model(self) -> None:
        """EvalDatasetRecord が SQLModel テーブルモデルであること。"""
        from src.db.models import EvalDatasetRecord

        assert hasattr(EvalDatasetRecord, "__tablename__")
        assert EvalDatasetRecord.__tablename__ == "eval_datasets"


class TestEvalExampleRecordModel:
    """EvalExampleRecord モデルのインスタンス化・バリデーションテスト。"""

    def test_creates_with_required_fields(self) -> None:
        """必須フィールドで EvalExampleRecord が生成できること。"""
        from src.db.models import EvalExampleRecord

        dataset_id = uuid.uuid4()
        record = EvalExampleRecord(
            dataset_id=dataset_id,
            query="What is RAG?",
            context="RAG stands for Retrieval-Augmented Generation.",
            answer="RAG is Retrieval-Augmented Generation.",
        )

        assert record.dataset_id == dataset_id
        assert record.query == "What is RAG?"
        assert record.context == "RAG stands for Retrieval-Augmented Generation."
        assert record.answer == "RAG is Retrieval-Augmented Generation."

    def test_id_defaults_to_uuid(self) -> None:
        """id がデフォルトで UUID を生成すること。"""
        from src.db.models import EvalExampleRecord

        record = EvalExampleRecord(
            dataset_id=uuid.uuid4(),
            query="q",
            context="c",
            answer="a",
        )

        assert record.id is not None
        assert isinstance(record.id, uuid.UUID)

    def test_expected_answer_defaults_to_none(self) -> None:
        """expected_answer のデフォルト値が None であること。"""
        from src.db.models import EvalExampleRecord

        record = EvalExampleRecord(
            dataset_id=uuid.uuid4(),
            query="q",
            context="c",
            answer="a",
        )

        assert record.expected_answer is None

    def test_expected_answer_can_be_set(self) -> None:
        """expected_answer に値を設定できること。"""
        from src.db.models import EvalExampleRecord

        record = EvalExampleRecord(
            dataset_id=uuid.uuid4(),
            query="q",
            context="c",
            answer="a",
            expected_answer="expected",
        )

        assert record.expected_answer == "expected"

    def test_created_at_defaults(self) -> None:
        """created_at がデフォルトで現在時刻付近を返すこと。"""
        from src.db.models import EvalExampleRecord

        before = datetime.now(UTC)
        record = EvalExampleRecord(
            dataset_id=uuid.uuid4(),
            query="q",
            context="c",
            answer="a",
        )
        after = datetime.now(UTC)

        assert record.created_at is not None
        assert before <= record.created_at <= after

    def test_is_table_model(self) -> None:
        """EvalExampleRecord が SQLModel テーブルモデルであること。"""
        from src.db.models import EvalExampleRecord

        assert hasattr(EvalExampleRecord, "__tablename__")
        assert EvalExampleRecord.__tablename__ == "eval_examples"
