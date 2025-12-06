"""
Base classes for integration nodes.
All external tool integrations inherit from BaseNode.
"""
import functools
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class NodeResult:
    """Standardized node execution result."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


def log_execution(func):
    """Log function entry, exit, and duration."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.info(f"START: {func.__name__}")
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            logger.info(f"END: {func.__name__} ({duration:.2f}s)")
            return result
        except Exception as e:
            logger.exception(f"FAIL: {func.__name__} - {e}")
            raise
    return wrapper


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


class BaseNode(ABC):
    """
    Abstract base class for integration nodes.

    All external tool integrations (ComfyUI, Anime4K, FireCrawl, etc.)
    MUST inherit from this class.

    Attributes:
        name: Human-readable node identifier
        logger: Configured logger instance
        config: Node-specific configuration
    """

    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"node.{name}")

    @abstractmethod
    def execute(self, input_data: Any) -> NodeResult:
        """
        Execute the node's primary function.

        Args:
            input_data: Input data for processing

        Returns:
            NodeResult with success status and data/error
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate configuration and dependencies.

        Returns:
            True if node is ready to execute
        """
        pass

    def health_check(self) -> Dict[str, Any]:
        """Check node health and return status."""
        try:
            is_valid = self.validate()
            return {
                "name": self.name,
                "status": "healthy" if is_valid else "unhealthy",
                "valid": is_valid,
            }
        except Exception as e:
            return {
                "name": self.name,
                "status": "error",
                "error": str(e),
            }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"
