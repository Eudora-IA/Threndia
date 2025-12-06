"""
Core exceptions for Asset Generator Farm.
"""


class AssetFarmError(Exception):
    """Base exception for Asset Generator Farm."""
    pass


class NodeExecutionError(AssetFarmError):
    """Raised when a node fails to execute."""
    def __init__(self, node_name: str, message: str):
        self.node_name = node_name
        super().__init__(f"[{node_name}] {message}")


class IntegrationError(AssetFarmError):
    """Raised when external integration fails."""
    pass


class MarketplaceError(AssetFarmError):
    """Raised for marketplace API errors."""
    pass


class ValidationError(AssetFarmError):
    """Raised for input validation failures."""
    pass


class MemoryError(AssetFarmError):
    """Raised for Mem0 memory operations."""
    pass


class RAGError(AssetFarmError):
    """Raised for RAG pipeline failures."""
    pass
