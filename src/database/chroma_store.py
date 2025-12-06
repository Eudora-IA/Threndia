"""
ChromaDB Store for Market Analysis.
Specialized wrapper for storing and querying market signals.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

logger = logging.getLogger(__name__)

class ChromaMarketStore:
    """
    ChromaDB wrapper for market analysis data.

    Collections:
    - signals: Market signals with embeddings
    - trends: Aggregated trend data
    """

    def __init__(self, persist_directory: str = "./data/market_chroma"):
        """
        Initialize ChromaDB client.

        Args:
            persist_directory: Path for persistent storage.
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is not installed. Run: pip install chromadb")

        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Initialize collections
        self.signals_collection = self.client.get_or_create_collection(
            name="market_signals",
            metadata={"description": "Market analysis signals"}
        )
        self.trends_collection = self.client.get_or_create_collection(
            name="market_trends",
            metadata={"description": "Aggregated trend data"}
        )

        logger.info(f"ChromaMarketStore initialized at {persist_directory}")

    def add_signal(self, signal_id: str, content: str, metadata: Dict[str, Any]) -> str:
        """
        Add a market signal to the database.

        Args:
            signal_id: Unique identifier.
            content: Text content to embed.
            metadata: Signal metadata.

        Returns:
            The signal ID.
        """
        # Ensure metadata values are JSON serializable
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, datetime):
                clean_metadata[k] = v.isoformat()
            elif isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            else:
                clean_metadata[k] = str(v)

        self.signals_collection.add(
            ids=[signal_id],
            documents=[content],
            metadatas=[clean_metadata]
        )
        return signal_id

    def search_similar_signals(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar signals using semantic search.

        Args:
            query: Search query text.
            limit: Maximum results to return.

        Returns:
            List of matching signal documents.
        """
        results = self.signals_collection.query(
            query_texts=[query],
            n_results=limit
        )

        # Format results
        formatted = []
        if results and results.get("ids"):
            for i, doc_id in enumerate(results["ids"][0]):
                formatted.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results.get("documents") else None,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else None
                })

        return formatted

    def get_signals_by_source(self, source: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get signals filtered by source."""
        results = self.signals_collection.get(
            where={"source": source},
            limit=limit
        )

        formatted = []
        if results and results.get("ids"):
            for i, doc_id in enumerate(results["ids"]):
                formatted.append({
                    "id": doc_id,
                    "content": results["documents"][i] if results.get("documents") else None,
                    "metadata": results["metadatas"][i] if results.get("metadatas") else {}
                })

        return formatted

    def count_signals(self) -> int:
        """Get total number of stored signals."""
        return self.signals_collection.count()
