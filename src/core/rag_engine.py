"""
RAG Engine with swappable vector store backends.
Supports ChromaDB and FAISS for semantic search.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document for RAG indexing."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """Result from vector search."""
    document: Document
    score: float
    rank: int


class VectorStoreBase(ABC):
    """Abstract base for vector store backends."""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to the store. Returns count added."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete(self, doc_ids: List[str]) -> int:
        """Delete documents by ID. Returns count deleted."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total document count."""
        pass


class ChromaVectorStore(VectorStoreBase):
    """ChromaDB vector store implementation."""

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings

            if self.persist_directory:
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory
                )
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name
            )
            logger.info(f"ChromaDB initialized: {self.collection_name}")
        except ImportError:
            logger.error("chromadb not installed. Run: pip install chromadb")
            raise

    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to ChromaDB."""
        if not documents:
            return 0

        self._collection.add(
            ids=[doc.id for doc in documents],
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )
        return len(documents)

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Search ChromaDB for similar documents."""
        results = self._collection.query(
            query_texts=[query],
            n_results=limit,
            where=filters,
        )

        search_results = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = Document(
                    id=doc_id,
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                )
                distance = results["distances"][0][i] if results["distances"] else 0.0
                search_results.append(SearchResult(
                    document=doc,
                    score=1.0 - distance,  # Convert distance to similarity
                    rank=i + 1,
                ))

        return search_results

    def delete(self, doc_ids: List[str]) -> int:
        """Delete documents from ChromaDB."""
        self._collection.delete(ids=doc_ids)
        return len(doc_ids)

    def count(self) -> int:
        """Return document count."""
        return self._collection.count()


class FAISSVectorStore(VectorStoreBase):
    """FAISS vector store implementation."""

    def __init__(
        self,
        embedding_dim: int = 384,
        index_path: Optional[str] = None,
    ):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self._index = None
        self._documents: Dict[int, Document] = {}
        self._embedder = None
        self._id_counter = 0
        self._initialize()

    def _initialize(self) -> None:
        """Initialize FAISS index."""
        try:
            import faiss
            self._index = faiss.IndexFlatL2(self.embedding_dim)

            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

            logger.info(f"FAISS initialized with dim={self.embedding_dim}")
        except ImportError as e:
            logger.error(f"FAISS dependencies missing: {e}")
            raise

    def _embed(self, texts: List[str]) -> Any:
        """Generate embeddings for texts."""
        return self._embedder.encode(texts, convert_to_numpy=True)

    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to FAISS index."""
        if not documents:
            return 0

        texts = [doc.content for doc in documents]
        embeddings = self._embed(texts)

        for i, doc in enumerate(documents):
            self._documents[self._id_counter] = doc
            self._id_counter += 1

        self._index.add(embeddings)
        return len(documents)

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Search FAISS index."""
        query_embedding = self._embed([query])
        distances, indices = self._index.search(query_embedding, limit)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx in self._documents:
                doc = self._documents[idx]
                results.append(SearchResult(
                    document=doc,
                    score=1.0 / (1.0 + distances[0][i]),
                    rank=i + 1,
                ))

        return results

    def delete(self, doc_ids: List[str]) -> int:
        """Delete not fully supported in FAISS flat index."""
        logger.warning("FAISS flat index does not support deletion")
        return 0

    def count(self) -> int:
        """Return document count."""
        return self._index.ntotal


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.

    Orchestrates document retrieval and context augmentation
    for LLM prompts.

    Example:
        >>> rag = RAGEngine(store=ChromaVectorStore())
        >>> rag.add_documents([Document(id="1", content="...")])
        >>> context = rag.retrieve("query", limit=5)
    """

    def __init__(
        self,
        store: VectorStoreBase,
        context_template: Optional[str] = None,
    ):
        self.store = store
        self.context_template = context_template or (
            "Context:\n{context}\n\nQuestion: {query}"
        )
        logger.info(f"RAGEngine initialized with {store.__class__.__name__}")

    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to the vector store."""
        return self.store.add_documents(documents)

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Retrieve relevant documents for a query."""
        return self.store.search(query, limit=limit, filters=filters)

    def build_context(
        self,
        query: str,
        limit: int = 5,
    ) -> str:
        """Build augmented context string for LLM."""
        results = self.retrieve(query, limit=limit)

        context_parts = []
        for result in results:
            context_parts.append(
                f"[{result.rank}] {result.document.content}"
            )

        context = "\n\n".join(context_parts)
        return self.context_template.format(
            context=context,
            query=query,
        )

    def document_count(self) -> int:
        """Return total indexed document count."""
        return self.store.count()
