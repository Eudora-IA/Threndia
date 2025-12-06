"""
Memory layer using Mem0 for agent long-term memory.
Provides persistent memory for AI agents across sessions.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    metadata: Dict[str, Any]
    category: str  # fact, preference, rule, identity, relationship


class AgentMemory:
    """
    Mem0-based memory system for agents.

    Provides:
    - Long-term memory persistence
    - Semantic search over memories
    - User/agent-specific memory isolation

    Example:
        >>> memory = AgentMemory(user_id="user_123")
        >>> memory.add("User prefers dark mode UI")
        >>> results = memory.search("UI preferences")
    """

    def __init__(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        self.user_id = user_id
        self.agent_id = agent_id or "default"
        self.config = config or {}
        self._client = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize Mem0 client."""
        try:
            from mem0 import Memory
            self._client = Memory()
            logger.info(f"Mem0 initialized for user={self.user_id}")
        except ImportError:
            logger.warning("mem0ai not installed. Run: pip install mem0ai")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to initialize Mem0: {e}")
            self._client = None

    @property
    def is_available(self) -> bool:
        """Check if memory system is available."""
        return self._client is not None

    def add(
        self,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Add a memory entry.

        Args:
            content: Memory content text
            metadata: Optional metadata dict

        Returns:
            Memory ID if successful, None otherwise
        """
        if not self.is_available:
            logger.warning("Memory system not available")
            return None

        try:
            result = self._client.add(
                content,
                user_id=self.user_id,
                agent_id=self.agent_id,
                metadata=metadata or {},
            )
            logger.debug(f"Added memory: {result}")
            return result.get("id") if isinstance(result, dict) else str(result)
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return None

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search memories by semantic similarity.

        Args:
            query: Search query text
            limit: Maximum results to return

        Returns:
            List of matching memory entries
        """
        if not self.is_available:
            return []

        try:
            results = self._client.search(
                query,
                user_id=self.user_id,
                limit=limit,
            )
            return results if isinstance(results, list) else []
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all memories for the user."""
        if not self.is_available:
            return []

        try:
            return self._client.get_all(user_id=self.user_id)
        except Exception as e:
            logger.error(f"Failed to get memories: {e}")
            return []

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        if not self.is_available:
            return False

        try:
            self._client.delete(memory_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False

    def clear(self) -> bool:
        """Clear all memories for the user."""
        if not self.is_available:
            return False

        try:
            self._client.delete_all(user_id=self.user_id)
            logger.info(f"Cleared all memories for user={self.user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return False
