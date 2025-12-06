"""
Enhanced ChromaDB vector database for multi-modal indexing.
Supports images, text, and metadata with LangGraph integration.
"""
import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content that can be indexed."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    MIXED = "mixed"


@dataclass
class MultiModalDocument:
    """
    Document supporting multiple content types.

    Can contain text, image paths, or embeddings.
    """
    id: str
    content_type: ContentType
    text_content: Optional[str] = None
    image_path: Optional[str] = None
    image_embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_content_hash(self) -> str:
        """Generate content hash for deduplication."""
        if self.text_content:
            return hashlib.md5(self.text_content.encode()).hexdigest()
        elif self.image_path:
            return hashlib.md5(self.image_path.encode()).hexdigest()
        return hashlib.md5(self.id.encode()).hexdigest()


@dataclass
class SearchQuery:
    """Query for multi-modal search."""
    text: Optional[str] = None
    image_path: Optional[str] = None
    content_types: List[ContentType] = field(default_factory=lambda: [ContentType.TEXT])
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 10


class EnhancedChromaStore:
    """
    Enhanced ChromaDB store with multi-modal support.

    Features:
    - Text and image indexing
    - CLIP embeddings for images
    - Metadata filtering
    - Collection management
    - LangGraph state integration

    Example:
        >>> store = EnhancedChromaStore()
        >>> store.add_image("path/to/image.png", metadata={"style": "anime"})
        >>> results = store.search(SearchQuery(text="anime style"))
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_prefix: str = "assetfarm",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = persist_directory
        self.collection_prefix = collection_prefix
        self.embedding_model = embedding_model
        self._client = None
        self._text_collection = None
        self._image_collection = None
        self._text_embedder = None
        self._image_embedder = None
        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB and embedding models."""
        try:
            import chromadb
            from chromadb.config import Settings

            if self.persist_directory:
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory
                )
            else:
                self._client = chromadb.Client()

            # Create separate collections for text and images
            self._text_collection = self._client.get_or_create_collection(
                name=f"{self.collection_prefix}_text",
                metadata={"hnsw:space": "cosine"}
            )
            self._image_collection = self._client.get_or_create_collection(
                name=f"{self.collection_prefix}_images",
                metadata={"hnsw:space": "cosine"}
            )

            logger.info("ChromaDB initialized with text and image collections")

            # Initialize text embedder
            try:
                from sentence_transformers import SentenceTransformer
                self._text_embedder = SentenceTransformer(self.embedding_model)
                logger.info(f"Text embedder loaded: {self.embedding_model}")
            except ImportError:
                logger.warning("sentence-transformers not installed")

            # Initialize CLIP for images
            try:
                from transformers import CLIPModel, CLIPProcessor
                self._image_embedder = {
                    "model": CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
                    "processor": CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
                }
                logger.info("CLIP image embedder loaded")
            except ImportError:
                logger.warning("transformers not installed for CLIP")
            except Exception as e:
                logger.warning(f"CLIP initialization failed: {e}")

        except ImportError:
            logger.error("chromadb not installed")
            raise

    def _embed_text(self, texts: List[str]) -> List[List[float]]:
        """Generate text embeddings."""
        if self._text_embedder is None:
            raise RuntimeError("Text embedder not available")
        embeddings = self._text_embedder.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def _embed_image(self, image_path: str) -> List[float]:
        """Generate image embedding using CLIP."""
        if self._image_embedder is None:
            raise RuntimeError("Image embedder not available")

        from PIL import Image
        image = Image.open(image_path)

        inputs = self._image_embedder["processor"](
            images=image,
            return_tensors="pt"
        )
        image_features = self._image_embedder["model"].get_image_features(**inputs)
        return image_features.detach().numpy().flatten().tolist()

    def _embed_text_for_image_search(self, text: str) -> List[float]:
        """Generate text embedding for searching images using CLIP."""
        if self._image_embedder is None:
            raise RuntimeError("Image embedder not available")

        inputs = self._image_embedder["processor"](
            text=[text],
            return_tensors="pt",
            padding=True
        )
        text_features = self._image_embedder["model"].get_text_features(**inputs)
        return text_features.detach().numpy().flatten().tolist()

    # =========================================================================
    # TEXT OPERATIONS
    # =========================================================================

    def add_text(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Add text content to the database."""
        doc_id = doc_id or hashlib.md5(text.encode()).hexdigest()
        metadata = metadata or {}
        metadata["content_type"] = ContentType.TEXT.value

        embedding = self._embed_text([text])[0]

        self._text_collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

        logger.debug(f"Added text document: {doc_id}")
        return doc_id

    def add_texts(
        self,
        texts: List[str],
        doc_ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None,
    ) -> List[str]:
        """Batch add text content."""
        if doc_ids is None:
            doc_ids = [hashlib.md5(t.encode()).hexdigest() for t in texts]

        if metadatas is None:
            metadatas = [{"content_type": ContentType.TEXT.value} for _ in texts]
        else:
            for m in metadatas:
                m["content_type"] = ContentType.TEXT.value

        embeddings = self._embed_text(texts)

        self._text_collection.add(
            ids=doc_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(texts)} text documents")
        return doc_ids

    def search_text(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Search text documents."""
        embedding = self._embed_text([query])[0]

        results = self._text_collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where=filters,
        )

        return self._format_results(results, ContentType.TEXT)

    # =========================================================================
    # IMAGE OPERATIONS
    # =========================================================================

    def add_image(
        self,
        image_path: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Add image to the database with CLIP embedding."""
        doc_id = doc_id or hashlib.md5(image_path.encode()).hexdigest()
        metadata = metadata or {}
        metadata["content_type"] = ContentType.IMAGE.value
        metadata["image_path"] = image_path

        embedding = self._embed_image(image_path)

        self._image_collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[image_path],  # Store path as document
            metadatas=[metadata],
        )

        logger.debug(f"Added image document: {doc_id}")
        return doc_id

    def add_images(
        self,
        image_paths: List[str],
        doc_ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None,
    ) -> List[str]:
        """Batch add images."""
        if doc_ids is None:
            doc_ids = [hashlib.md5(p.encode()).hexdigest() for p in image_paths]

        if metadatas is None:
            metadatas = []

        ids_added = []
        for i, path in enumerate(image_paths):
            try:
                meta = metadatas[i] if i < len(metadatas) else {}
                doc_id = self.add_image(path, doc_ids[i], meta)
                ids_added.append(doc_id)
            except Exception as e:
                logger.error(f"Failed to add image {path}: {e}")

        logger.info(f"Added {len(ids_added)} images")
        return ids_added

    def search_images_by_text(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Search images using text query (CLIP)."""
        embedding = self._embed_text_for_image_search(query)

        results = self._image_collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where=filters,
        )

        return self._format_results(results, ContentType.IMAGE)

    def search_images_by_image(
        self,
        image_path: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Search similar images."""
        embedding = self._embed_image(image_path)

        results = self._image_collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where=filters,
        )

        return self._format_results(results, ContentType.IMAGE)

    # =========================================================================
    # UNIFIED SEARCH
    # =========================================================================

    def search(self, query: SearchQuery) -> List[Dict]:
        """
        Unified multi-modal search.

        Searches across text and image collections based on query.
        """
        all_results = []

        if ContentType.TEXT in query.content_types and query.text:
            text_results = self.search_text(
                query.text,
                limit=query.limit,
                filters=query.filters,
            )
            all_results.extend(text_results)

        if ContentType.IMAGE in query.content_types:
            if query.text:
                image_results = self.search_images_by_text(
                    query.text,
                    limit=query.limit,
                    filters=query.filters,
                )
                all_results.extend(image_results)
            elif query.image_path:
                image_results = self.search_images_by_image(
                    query.image_path,
                    limit=query.limit,
                    filters=query.filters,
                )
                all_results.extend(image_results)

        # Sort by score and limit
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:query.limit]

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _format_results(
        self,
        results: Dict,
        content_type: ContentType,
    ) -> List[Dict]:
        """Format ChromaDB results into standard structure."""
        formatted = []
        if results and results.get("ids"):
            for i, doc_id in enumerate(results["ids"][0]):
                formatted.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results.get("documents") else None,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "score": 1.0 - results["distances"][0][i] if results.get("distances") else 0.0,
                    "content_type": content_type.value,
                })
        return formatted

    def get_stats(self) -> Dict[str, int]:
        """Get collection statistics."""
        return {
            "text_documents": self._text_collection.count(),
            "image_documents": self._image_collection.count(),
            "total": self._text_collection.count() + self._image_collection.count(),
        }

    def delete_by_id(self, doc_id: str, content_type: ContentType) -> bool:
        """Delete document by ID."""
        try:
            if content_type == ContentType.TEXT:
                self._text_collection.delete(ids=[doc_id])
            else:
                self._image_collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def clear_collection(self, content_type: Optional[ContentType] = None):
        """Clear specified or all collections."""
        if content_type == ContentType.TEXT or content_type is None:
            self._client.delete_collection(f"{self.collection_prefix}_text")
            self._text_collection = self._client.get_or_create_collection(
                name=f"{self.collection_prefix}_text"
            )

        if content_type == ContentType.IMAGE or content_type is None:
            self._client.delete_collection(f"{self.collection_prefix}_images")
            self._image_collection = self._client.get_or_create_collection(
                name=f"{self.collection_prefix}_images"
            )

        logger.info(f"Cleared collections: {content_type or 'all'}")


# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def create_vector_store_tools(store: EnhancedChromaStore) -> List[Dict]:
    """
    Create LangGraph-compatible tools for vector store operations.

    Returns tool definitions for use with LLM agents.
    """
    from .llm_provider import ToolDefinition

    return [
        ToolDefinition(
            name="search_knowledge",
            description="Search the knowledge base for relevant information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            handler=lambda query, limit=5: store.search_text(query, limit)
        ),
        ToolDefinition(
            name="search_images",
            description="Search for images matching a text description",
            parameters={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Text description of desired image"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 5
                    }
                },
                "required": ["description"]
            },
            handler=lambda description, limit=5: store.search_images_by_text(description, limit)
        ),
        ToolDefinition(
            name="add_to_knowledge",
            description="Add new information to the knowledge base",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Text content to add"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for the content"
                    }
                },
                "required": ["content"]
            },
            handler=lambda content, tags=None: store.add_text(
                content,
                metadata={"tags": tags or []}
            )
        ),
    ]
