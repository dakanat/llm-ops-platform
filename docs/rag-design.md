# RAG パイプライン設計

## 概要

RAG (Retrieval-Augmented Generation) パイプラインは、ドキュメントのインデックス登録とクエリ応答を統合する。
`RAGPipeline` クラスがオーケストレーターとして各コンポーネントを連携させる。

## パイプラインフロー

### ドキュメント登録 (Indexing)

```
Document
  ↓
Preprocessor.normalize()      — NFKC 正規化, 空白正規化, 改行正規化
  ↓
RecursiveCharacterSplitter.split() — チャンク分割 (512 文字, overlap 64)
  ↓
Embedder.embed_batch()        — vLLM サーバーでベクトル化 (1024 次元)
  ↓
VectorStore.save_chunks()     — pgvector に保存
```

### クエリ応答 (Query)

```
User Query
  ↓
Retriever.search()            — Embedding 生成 → cosine similarity 検索
  ↓
Top-K Chunks (default: 5)
  ↓
Generator.generate()          — LLM に番号付きコンテキスト + 質問を送信
  ↓
GenerationResult (answer + sources + usage)
```

## コンポーネント仕様

### Preprocessor (`src/rag/preprocessor.py`)

テキスト正規化を行う。

| 処理 | 詳細 |
|------|------|
| NFKC 正規化 | 全角英数→半角、互換文字の統一 |
| 空白正規化 | 連続スペース/タブ → 単一スペース (改行は保持) |
| 改行正規化 | 3 つ以上の連続改行 → 2 つ |
| 先頭末尾除去 | `.strip()` |

- 空文字・空白のみのテキストは `PreprocessingError` を送出

### RecursiveCharacterSplitter (`src/rag/chunker.py`)

再帰的にテキストを分割する。

**パラメータ**:
- `chunk_size`: 512 (デフォルト)
- `chunk_overlap`: 64 (デフォルト)
- `separators`: `["\n\n", "\n", "。", ". ", " ", ""]`

**アルゴリズム**:
1. セパレータリストを優先度順に試行
2. セパレータで分割 → 小さな断片を `chunk_size` 以下にマージ
3. `chunk_size` 超のチャンクは次のセパレータで再帰分割
4. 句読点系セパレータ (`。`, `. `) は直前のチャンクに付与して保持
5. overlap: 前チャンク末尾から `chunk_overlap` 文字分を次チャンクに引き継ぐ

**出力**: `TextChunk(content, index, start, end)` のリスト

### Embedder (`src/rag/embedder.py`)

vLLM ローカルサーバーに OpenAI 互換 API でリクエストを送る。

```python
class Embedder:
    def __init__(self, base_url: str, model: str) -> None: ...
    async def embed(self, text: str) -> list[float]: ...        # 単一テキスト
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...  # バッチ
    async def close(self) -> None: ...
```

- エンドポイント: `POST {base_url}/embeddings`
- 出力次元: 1024 (ruri-v3-310m)
- 空テキスト・空リストは `EmbeddingError` を送出

### VectorStore (`src/db/vector_store.py`)

pgvector を使用したベクトル保存・検索。

```python
class VectorStore:
    def __init__(self, session: AsyncSession) -> None: ...
    async def save_chunks(self, chunks: list[Chunk]) -> None: ...
    async def search(
        self,
        query_embedding: list[float],  # 1024 次元必須
        top_k: int = 5,
        document_id: UUID | None = None,
    ) -> list[Chunk]: ...
```

- cosine distance (`<->` 演算子) で類似度順にソート
- `document_id` 指定でドキュメント内検索に限定可能
- 次元数バリデーション: 1024 以外は `VectorStoreError`

### Retriever (`src/rag/retriever.py`)

Embedder + VectorStore のラッパー。

```python
class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None: ...
    async def search(
        self, query: str, top_k: int = 5, document_id: UUID | None = None
    ) -> list[RetrievedChunk]: ...
```

- `RetrievedChunk(content, chunk_index, document_id)` を返す
- Embedder/VectorStore のエラーは `RetrievalError` に変換

### Generator (`src/rag/generator.py`)

LLM にコンテキストと質問を送って回答を生成する。

```python
class Generator:
    def __init__(
        self, llm_provider: LLMProvider, model: str, system_prompt: str | None = None
    ) -> None: ...
    async def generate(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> GenerationResult: ...
```

- チャンクを `[1] 内容\n\n[2] 内容...` 形式でフォーマット
- システムプロンプト: 「コンテキストに基づいて回答し、`[1]` 形式で引用せよ」
- `GenerationResult(answer, sources, model, usage)` を返す

### IndexManager (`src/rag/index_manager.py`)

ドキュメント登録の一連のフローを統合する。

```python
class IndexManager:
    def __init__(
        self, preprocessor, chunker, embedder, vector_store
    ) -> None: ...
    async def index_document(self, document: Document) -> list[Chunk]: ...
```

フロー: 前処理 → チャンク分割 → バッチ Embedding → Chunk モデル構築 → VectorStore 保存

### RAGPipeline (`src/rag/pipeline.py`)

オーケストレーター。IndexManager, Retriever, Generator を統合。

```python
class RAGPipeline:
    def __init__(self, index_manager, retriever, generator) -> None: ...
    async def index_document(self, document: Document) -> list[Chunk]: ...
    async def query(
        self, query: str, top_k: int = 5, document_id: UUID | None = None
    ) -> GenerationResult: ...
```

- 各コンポーネントのエラーは `RAGPipelineError` に変換

## 拡張ポイント

### 再ランキング
- Retriever と Generator の間に挿入する設計
- `Retriever.search()` の結果を受け取り、スコアを再計算して並べ替える
- 現時点では未実装 (top_k=5 の cosine similarity で十分な品質)

### ハイブリッド検索
- ベクトル検索 + キーワード検索 (BM25) の組み合わせ
- pgvector + PostgreSQL 全文検索で実現可能
- 拡張時は Retriever 内に検索戦略を追加
