"""Tests for RAG pipeline modules: preprocessor, retriever, generator, index_manager, pipeline."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.db.models import Chunk, Document
from src.db.vector_store import VectorStoreError
from src.llm.providers.base import LLMResponse, Role, TokenUsage
from src.rag.chunker import RecursiveCharacterSplitter, TextChunk
from src.rag.embedder import EmbeddingError
from src.rag.generator import GenerationError, GenerationResult, Generator
from src.rag.index_manager import IndexingError, IndexManager
from src.rag.pipeline import RAGPipeline, RAGPipelineError
from src.rag.preprocessor import PreprocessingError, Preprocessor
from src.rag.retriever import RetrievalError, RetrievedChunk, Retriever

EMBEDDING_DIM = 1024


def _make_document(
    content: str = "テスト文書の内容です。",
    title: str = "テスト文書",
) -> Document:
    """テスト用の Document インスタンスを生成。"""
    return Document(
        id=uuid.uuid4(),
        title=title,
        content=content,
        user_id=uuid.uuid4(),
    )


def _make_chunk(
    document_id: uuid.UUID | None = None,
    chunk_index: int = 0,
    content: str = "テストチャンク",
) -> Chunk:
    """テスト用の Chunk インスタンスを生成。"""
    return Chunk(
        id=uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        content=content,
        chunk_index=chunk_index,
        embedding=[0.1] * EMBEDDING_DIM,
    )


def _make_retrieved_chunk(
    content: str = "テストチャンク",
    chunk_index: int = 0,
    document_id: uuid.UUID | None = None,
) -> RetrievedChunk:
    """テスト用の RetrievedChunk インスタンスを生成。"""
    return RetrievedChunk(
        content=content,
        chunk_index=chunk_index,
        document_id=document_id or uuid.uuid4(),
    )


def _make_text_chunk(
    content: str = "テストチャンク",
    index: int = 0,
    start: int = 0,
    end: int | None = None,
) -> TextChunk:
    """テスト用の TextChunk インスタンスを生成。"""
    return TextChunk(
        content=content,
        index=index,
        start=start,
        end=end if end is not None else start + len(content),
    )


def _make_llm_response(
    content: str = "回答テキスト",
    model: str = "test-model",
    usage: TokenUsage | None = None,
) -> LLMResponse:
    """テスト用の LLMResponse を生成。"""
    return LLMResponse(
        content=content,
        model=model,
        usage=usage,
    )


# =============================================================================
# Preprocessor
# =============================================================================


class TestPreprocessorNormalize:
    """Preprocessor.normalize() のテスト。"""

    def test_nfkc_normalizes_fullwidth_to_halfwidth(self) -> None:
        """全角英数字が半角に正規化されること。"""
        preprocessor = Preprocessor()

        result = preprocessor.normalize("Ｈｅｌｌｏ　Ｗｏｒｌｄ　１２３")

        assert result == "Hello World 123"

    def test_collapses_multiple_spaces_to_single(self) -> None:
        """連続スペースが単一スペースに正規化されること。"""
        preprocessor = Preprocessor()

        result = preprocessor.normalize("hello   world")

        assert result == "hello world"

    def test_preserves_single_newlines(self) -> None:
        """単一改行が保持されること。"""
        preprocessor = Preprocessor()

        result = preprocessor.normalize("hello\nworld")

        assert result == "hello\nworld"

    def test_preserves_double_newlines(self) -> None:
        """二重改行 (段落区切り) が保持されること。"""
        preprocessor = Preprocessor()

        result = preprocessor.normalize("hello\n\nworld")

        assert result == "hello\n\nworld"

    def test_collapses_triple_newlines_to_double(self) -> None:
        """3 つ以上の改行が 2 つに正規化されること。"""
        preprocessor = Preprocessor()

        result = preprocessor.normalize("hello\n\n\n\nworld")

        assert result == "hello\n\nworld"

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        """先頭・末尾の空白が除去されること。"""
        preprocessor = Preprocessor()

        result = preprocessor.normalize("  hello world  ")

        assert result == "hello world"

    def test_empty_string_raises_error(self) -> None:
        """空文字で PreprocessingError が発生すること。"""
        preprocessor = Preprocessor()

        with pytest.raises(PreprocessingError, match="empty"):
            preprocessor.normalize("")

    def test_whitespace_only_raises_error(self) -> None:
        """空白のみの文字列で PreprocessingError が発生すること。"""
        preprocessor = Preprocessor()

        with pytest.raises(PreprocessingError, match="empty"):
            preprocessor.normalize("   \n\n  ")


# =============================================================================
# RetrievedChunk model
# =============================================================================


class TestRetrievedChunk:
    """RetrievedChunk モデルのテスト。"""

    def test_creates_with_required_fields(self) -> None:
        """必須フィールドで生成できること。"""
        doc_id = uuid.uuid4()
        chunk = RetrievedChunk(
            content="テスト内容",
            chunk_index=0,
            document_id=doc_id,
        )

        assert chunk.content == "テスト内容"
        assert chunk.chunk_index == 0
        assert chunk.document_id == doc_id


# =============================================================================
# Retriever
# =============================================================================


class TestRetrieverSearch:
    """Retriever.search() のテスト。"""

    @pytest.fixture
    def embedder(self) -> AsyncMock:
        mock = AsyncMock()
        mock.embed.return_value = [0.1] * EMBEDDING_DIM
        return mock

    @pytest.fixture
    def vector_store(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def retriever(self, embedder: AsyncMock, vector_store: AsyncMock) -> Retriever:
        return Retriever(embedder=embedder, vector_store=vector_store)

    async def test_returns_retrieved_chunks(
        self, retriever: Retriever, embedder: AsyncMock, vector_store: AsyncMock
    ) -> None:
        """Chunk が RetrievedChunk に変換されて返ること。"""
        doc_id = uuid.uuid4()
        chunk = _make_chunk(document_id=doc_id, chunk_index=0, content="結果テキスト")
        vector_store.search.return_value = [chunk]

        result = await retriever.search("クエリ")

        assert len(result) == 1
        assert isinstance(result[0], RetrievedChunk)
        assert result[0].content == "結果テキスト"
        assert result[0].chunk_index == 0
        assert result[0].document_id == doc_id

    async def test_calls_embedder_with_query(
        self, retriever: Retriever, embedder: AsyncMock, vector_store: AsyncMock
    ) -> None:
        """Embedder.embed がクエリ文字列で呼ばれること。"""
        vector_store.search.return_value = []

        await retriever.search("テストクエリ")

        embedder.embed.assert_awaited_once_with("テストクエリ")

    async def test_calls_vector_store_with_embedding_and_top_k(
        self, retriever: Retriever, embedder: AsyncMock, vector_store: AsyncMock
    ) -> None:
        """VectorStore.search が embedding と top_k で呼ばれること。"""
        vector_store.search.return_value = []

        await retriever.search("クエリ", top_k=3)

        vector_store.search.assert_awaited_once_with(
            query_embedding=[0.1] * EMBEDDING_DIM,
            top_k=3,
            document_id=None,
        )

    async def test_passes_document_id_to_vector_store(
        self, retriever: Retriever, embedder: AsyncMock, vector_store: AsyncMock
    ) -> None:
        """document_id が VectorStore.search に渡されること。"""
        doc_id = uuid.uuid4()
        vector_store.search.return_value = []

        await retriever.search("クエリ", document_id=doc_id)

        vector_store.search.assert_awaited_once_with(
            query_embedding=[0.1] * EMBEDDING_DIM,
            top_k=5,
            document_id=doc_id,
        )

    async def test_returns_empty_list_when_no_results(
        self, retriever: Retriever, vector_store: AsyncMock
    ) -> None:
        """検索結果がない場合に空リストが返ること。"""
        vector_store.search.return_value = []

        result = await retriever.search("クエリ")

        assert result == []

    async def test_wraps_embedding_error(self, retriever: Retriever, embedder: AsyncMock) -> None:
        """EmbeddingError が RetrievalError でラップされること。"""
        embedding_error = EmbeddingError("connection failed")
        embedder.embed.side_effect = embedding_error

        with pytest.raises(RetrievalError) as exc_info:
            await retriever.search("クエリ")

        assert exc_info.value.__cause__ is embedding_error

    async def test_wraps_vector_store_error(
        self, retriever: Retriever, vector_store: AsyncMock
    ) -> None:
        """VectorStoreError が RetrievalError でラップされること。"""
        vs_error = VectorStoreError("query failed")
        vector_store.search.side_effect = vs_error

        with pytest.raises(RetrievalError) as exc_info:
            await retriever.search("クエリ")

        assert exc_info.value.__cause__ is vs_error


# =============================================================================
# Generator — _build_context
# =============================================================================


class TestGeneratorBuildContext:
    """Generator._build_context() のテスト。"""

    @pytest.fixture
    def generator(self) -> Generator:
        provider = AsyncMock()
        return Generator(llm_provider=provider, model="test-model")

    def test_builds_numbered_context(self, generator: Generator) -> None:
        """番号付きコンテキストが生成されること。"""
        chunks = [
            _make_retrieved_chunk(content="チャンク1"),
            _make_retrieved_chunk(content="チャンク2", chunk_index=1),
        ]

        result = generator._build_context(chunks)

        assert "[1] チャンク1" in result
        assert "[2] チャンク2" in result

    def test_separates_chunks_with_double_newline(self, generator: Generator) -> None:
        """チャンク間が二重改行で区切られること。"""
        chunks = [
            _make_retrieved_chunk(content="A"),
            _make_retrieved_chunk(content="B", chunk_index=1),
        ]

        result = generator._build_context(chunks)

        assert result == "[1] A\n\n[2] B"

    def test_empty_chunks_returns_empty_string(self, generator: Generator) -> None:
        """空リストで空文字列が返ること。"""
        result = generator._build_context([])

        assert result == ""


# =============================================================================
# Generator — _build_messages
# =============================================================================


class TestGeneratorBuildMessages:
    """Generator._build_messages() のテスト。"""

    @pytest.fixture
    def generator(self) -> Generator:
        provider = AsyncMock()
        return Generator(llm_provider=provider, model="test-model")

    def test_builds_system_and_user_messages(self, generator: Generator) -> None:
        """system メッセージと user メッセージが構築されること。"""
        messages = generator._build_messages("質問テキスト", "コンテキストテキスト")

        assert len(messages) == 2
        assert messages[0].role == Role.system
        assert messages[1].role == Role.user
        assert "質問テキスト" in messages[1].content
        assert "コンテキストテキスト" in messages[1].content

    def test_uses_custom_system_prompt(self) -> None:
        """カスタムシステムプロンプトが使用されること。"""
        provider = AsyncMock()
        custom_prompt = "カスタムプロンプト"
        generator = Generator(
            llm_provider=provider, model="test-model", system_prompt=custom_prompt
        )

        messages = generator._build_messages("質問", "コンテキスト")

        assert messages[0].content == custom_prompt


# =============================================================================
# Generator — generate
# =============================================================================


class TestGeneratorGenerate:
    """Generator.generate() のテスト。"""

    @pytest.fixture
    def provider(self) -> AsyncMock:
        mock = AsyncMock()
        mock.complete.return_value = _make_llm_response()
        return mock

    @pytest.fixture
    def generator(self, provider: AsyncMock) -> Generator:
        return Generator(llm_provider=provider, model="test-model")

    async def test_returns_generation_result(self, generator: Generator) -> None:
        """GenerationResult が返ること。"""
        chunks = [_make_retrieved_chunk()]

        result = await generator.generate("質問", chunks)

        assert isinstance(result, GenerationResult)

    async def test_result_contains_answer(self, generator: Generator, provider: AsyncMock) -> None:
        """回答テキストが含まれること。"""
        provider.complete.return_value = _make_llm_response(content="これは回答です")
        chunks = [_make_retrieved_chunk()]

        result = await generator.generate("質問", chunks)

        assert result.answer == "これは回答です"

    async def test_result_contains_sources(self, generator: Generator) -> None:
        """ソースチャンクが含まれること。"""
        chunks = [
            _make_retrieved_chunk(content="ソース1"),
            _make_retrieved_chunk(content="ソース2", chunk_index=1),
        ]

        result = await generator.generate("質問", chunks)

        assert len(result.sources) == 2
        assert result.sources[0].content == "ソース1"

    async def test_result_contains_model(self, generator: Generator) -> None:
        """モデル名が含まれること。"""
        chunks = [_make_retrieved_chunk()]

        result = await generator.generate("質問", chunks)

        assert result.model == "test-model"

    async def test_result_contains_usage(self, generator: Generator, provider: AsyncMock) -> None:
        """トークン使用量が含まれること。"""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        provider.complete.return_value = _make_llm_response(usage=usage)
        chunks = [_make_retrieved_chunk()]

        result = await generator.generate("質問", chunks)

        assert result.usage is not None
        assert result.usage.total_tokens == 30

    async def test_wraps_llm_error(self, generator: Generator, provider: AsyncMock) -> None:
        """LLM エラーが GenerationError でラップされること。"""
        llm_error = RuntimeError("LLM failed")
        provider.complete.side_effect = llm_error
        chunks = [_make_retrieved_chunk()]

        with pytest.raises(GenerationError) as exc_info:
            await generator.generate("質問", chunks)

        assert exc_info.value.__cause__ is llm_error


# =============================================================================
# GenerationResult model
# =============================================================================


class TestGenerationResult:
    """GenerationResult モデルのテスト。"""

    def test_creates_with_required_fields(self) -> None:
        """必須フィールドで生成できること。"""
        chunk = _make_retrieved_chunk()
        result = GenerationResult(
            answer="回答テキスト",
            sources=[chunk],
            model="test-model",
        )

        assert result.answer == "回答テキスト"
        assert len(result.sources) == 1
        assert result.model == "test-model"
        assert result.usage is None


# =============================================================================
# IndexManager
# =============================================================================


class TestIndexManagerIndexDocument:
    """IndexManager.index_document() のテスト。"""

    @pytest.fixture
    def preprocessor(self) -> Preprocessor:
        return Preprocessor()

    @pytest.fixture
    def chunker(self) -> MagicMock:
        mock = MagicMock(spec=RecursiveCharacterSplitter)
        return mock

    @pytest.fixture
    def embedder(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def vector_store(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def index_manager(
        self,
        preprocessor: Preprocessor,
        chunker: MagicMock,
        embedder: AsyncMock,
        vector_store: AsyncMock,
    ) -> IndexManager:
        return IndexManager(
            preprocessor=preprocessor,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )

    async def test_returns_list_of_chunks(
        self,
        index_manager: IndexManager,
        chunker: MagicMock,
        embedder: AsyncMock,
        vector_store: AsyncMock,
    ) -> None:
        """Chunk のリストが返ること。"""
        doc = _make_document(content="テスト文書")
        chunker.split.return_value = [_make_text_chunk(content="テスト文書")]
        embedder.embed_batch.return_value = [[0.1] * EMBEDDING_DIM]

        result = await index_manager.index_document(doc)

        assert len(result) == 1
        assert isinstance(result[0], Chunk)

    async def test_preprocesses_document_content(
        self,
        index_manager: IndexManager,
        chunker: MagicMock,
        embedder: AsyncMock,
    ) -> None:
        """前処理が適用されること (全角→半角)。"""
        doc = _make_document(content="Ｈｅｌｌｏ")
        chunker.split.return_value = [_make_text_chunk(content="Hello")]
        embedder.embed_batch.return_value = [[0.1] * EMBEDDING_DIM]

        await index_manager.index_document(doc)

        chunker.split.assert_called_once_with("Hello")

    async def test_chunks_preprocessed_text(
        self,
        index_manager: IndexManager,
        chunker: MagicMock,
        embedder: AsyncMock,
    ) -> None:
        """前処理済みテキストがチャンカーに渡されること。"""
        doc = _make_document(content="テスト文書")
        chunker.split.return_value = [_make_text_chunk(content="テスト文書")]
        embedder.embed_batch.return_value = [[0.1] * EMBEDDING_DIM]

        await index_manager.index_document(doc)

        chunker.split.assert_called_once_with("テスト文書")

    async def test_embeds_chunk_contents(
        self,
        index_manager: IndexManager,
        chunker: MagicMock,
        embedder: AsyncMock,
    ) -> None:
        """チャンク内容が embed_batch に渡されること。"""
        doc = _make_document()
        chunker.split.return_value = [
            _make_text_chunk(content="チャンク1"),
            _make_text_chunk(content="チャンク2", index=1, start=5),
        ]
        embedder.embed_batch.return_value = [
            [0.1] * EMBEDDING_DIM,
            [0.2] * EMBEDDING_DIM,
        ]

        await index_manager.index_document(doc)

        embedder.embed_batch.assert_awaited_once_with(["チャンク1", "チャンク2"])

    async def test_saves_chunks_to_vector_store(
        self,
        index_manager: IndexManager,
        chunker: MagicMock,
        embedder: AsyncMock,
        vector_store: AsyncMock,
    ) -> None:
        """VectorStore.save_chunks が呼ばれること。"""
        doc = _make_document()
        chunker.split.return_value = [_make_text_chunk()]
        embedder.embed_batch.return_value = [[0.1] * EMBEDDING_DIM]

        await index_manager.index_document(doc)

        vector_store.save_chunks.assert_awaited_once()
        saved_chunks = vector_store.save_chunks.call_args[0][0]
        assert len(saved_chunks) == 1
        assert saved_chunks[0].document_id == doc.id

    async def test_wraps_embedding_error(
        self,
        index_manager: IndexManager,
        chunker: MagicMock,
        embedder: AsyncMock,
    ) -> None:
        """EmbeddingError が IndexingError でラップされること。"""
        doc = _make_document()
        chunker.split.return_value = [_make_text_chunk()]
        embedding_error = EmbeddingError("embed failed")
        embedder.embed_batch.side_effect = embedding_error

        with pytest.raises(IndexingError) as exc_info:
            await index_manager.index_document(doc)

        assert exc_info.value.__cause__ is embedding_error

    async def test_wraps_vector_store_error(
        self,
        index_manager: IndexManager,
        chunker: MagicMock,
        embedder: AsyncMock,
        vector_store: AsyncMock,
    ) -> None:
        """VectorStoreError が IndexingError でラップされること。"""
        doc = _make_document()
        chunker.split.return_value = [_make_text_chunk()]
        embedder.embed_batch.return_value = [[0.1] * EMBEDDING_DIM]
        vs_error = VectorStoreError("save failed")
        vector_store.save_chunks.side_effect = vs_error

        with pytest.raises(IndexingError) as exc_info:
            await index_manager.index_document(doc)

        assert exc_info.value.__cause__ is vs_error

    async def test_empty_content_raises_error(
        self,
        index_manager: IndexManager,
    ) -> None:
        """空コンテンツの Document で IndexingError が発生すること。"""
        doc = _make_document(content="")

        with pytest.raises(IndexingError):
            await index_manager.index_document(doc)


# =============================================================================
# RAGPipeline — index_document
# =============================================================================


class TestRAGPipelineIndexDocument:
    """RAGPipeline.index_document() のテスト。"""

    @pytest.fixture
    def index_manager(self) -> AsyncMock:
        return AsyncMock(spec=IndexManager)

    @pytest.fixture
    def retriever(self) -> AsyncMock:
        return AsyncMock(spec=Retriever)

    @pytest.fixture
    def generator(self) -> AsyncMock:
        return AsyncMock(spec=Generator)

    @pytest.fixture
    def pipeline(
        self,
        index_manager: AsyncMock,
        retriever: AsyncMock,
        generator: AsyncMock,
    ) -> RAGPipeline:
        return RAGPipeline(
            index_manager=index_manager,
            retriever=retriever,
            generator=generator,
        )

    async def test_delegates_to_index_manager(
        self, pipeline: RAGPipeline, index_manager: AsyncMock
    ) -> None:
        """IndexManager.index_document に委譲されること。"""
        doc = _make_document()
        expected_chunks = [_make_chunk()]
        index_manager.index_document.return_value = expected_chunks

        result = await pipeline.index_document(doc)

        index_manager.index_document.assert_awaited_once_with(doc)
        assert result == expected_chunks

    async def test_wraps_indexing_error(
        self, pipeline: RAGPipeline, index_manager: AsyncMock
    ) -> None:
        """IndexingError が RAGPipelineError でラップされること。"""
        doc = _make_document()
        indexing_error = IndexingError("index failed")
        index_manager.index_document.side_effect = indexing_error

        with pytest.raises(RAGPipelineError) as exc_info:
            await pipeline.index_document(doc)

        assert exc_info.value.__cause__ is indexing_error

    async def test_returns_chunks_from_index_manager(
        self, pipeline: RAGPipeline, index_manager: AsyncMock
    ) -> None:
        """IndexManager から返されたチャンクがそのまま返ること。"""
        doc = _make_document()
        chunks = [_make_chunk(), _make_chunk(chunk_index=1)]
        index_manager.index_document.return_value = chunks

        result = await pipeline.index_document(doc)

        assert len(result) == 2


# =============================================================================
# RAGPipeline — query
# =============================================================================


class TestRAGPipelineQuery:
    """RAGPipeline.query() のテスト。"""

    @pytest.fixture
    def index_manager(self) -> AsyncMock:
        return AsyncMock(spec=IndexManager)

    @pytest.fixture
    def retriever(self) -> AsyncMock:
        mock = AsyncMock(spec=Retriever)
        mock.search.return_value = [_make_retrieved_chunk()]
        return mock

    @pytest.fixture
    def mock_generator(self) -> AsyncMock:
        mock = AsyncMock(spec=Generator)
        mock.generate.return_value = GenerationResult(
            answer="回答",
            sources=[_make_retrieved_chunk()],
            model="test-model",
        )
        return mock

    @pytest.fixture
    def pipeline(
        self,
        index_manager: AsyncMock,
        retriever: AsyncMock,
        mock_generator: AsyncMock,
    ) -> RAGPipeline:
        return RAGPipeline(
            index_manager=index_manager,
            retriever=retriever,
            generator=mock_generator,
        )

    async def test_returns_generation_result(self, pipeline: RAGPipeline) -> None:
        """GenerationResult が返ること。"""
        result = await pipeline.query("質問")

        assert isinstance(result, GenerationResult)

    async def test_calls_retriever_with_query_and_top_k(
        self, pipeline: RAGPipeline, retriever: AsyncMock
    ) -> None:
        """Retriever.search がクエリと top_k で呼ばれること。"""
        await pipeline.query("テスト質問", top_k=3)

        retriever.search.assert_awaited_once_with("テスト質問", top_k=3, document_id=None)

    async def test_calls_generator_with_query_and_chunks(
        self, pipeline: RAGPipeline, retriever: AsyncMock, mock_generator: AsyncMock
    ) -> None:
        """Generator.generate がクエリとチャンクで呼ばれること。"""
        retrieved = [_make_retrieved_chunk(content="コンテキスト")]
        retriever.search.return_value = retrieved

        await pipeline.query("質問")

        mock_generator.generate.assert_awaited_once_with("質問", retrieved)

    async def test_wraps_retrieval_error(self, pipeline: RAGPipeline, retriever: AsyncMock) -> None:
        """RetrievalError が RAGPipelineError でラップされること。"""
        retrieval_error = RetrievalError("search failed")
        retriever.search.side_effect = retrieval_error

        with pytest.raises(RAGPipelineError) as exc_info:
            await pipeline.query("質問")

        assert exc_info.value.__cause__ is retrieval_error

    async def test_wraps_generation_error(
        self, pipeline: RAGPipeline, mock_generator: AsyncMock
    ) -> None:
        """GenerationError が RAGPipelineError でラップされること。"""
        gen_error = GenerationError("generate failed")
        mock_generator.generate.side_effect = gen_error

        with pytest.raises(RAGPipelineError) as exc_info:
            await pipeline.query("質問")

        assert exc_info.value.__cause__ is gen_error
