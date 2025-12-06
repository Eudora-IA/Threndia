# AGENTS.MD - Python Code Standards & Templates

> State-of-the-art Python development guidelines for the Asset Generator Farm project.

---

## Table of Contents

1. [Code Style](#code-style)
2. [Class Templates](#class-templates)
3. [Function Patterns](#function-patterns)
4. [Decorators](#decorators)
5. [Error Handling](#error-handling)
6. [Logging](#logging)
7. [Testing](#testing)
8. [LangGraph Nodes](#langgraph-nodes)

---

## Code Style

### General Rules

```python
# ✅ Use type hints everywhere
def process_image(path: str, scale: float = 2.0) -> np.ndarray:
    ...

# ✅ Use Pathlib for paths
from pathlib import Path
config_path = Path(__file__).parent / "config.yaml"

# ✅ Use dataclasses for configs
from dataclasses import dataclass, field

@dataclass
class NodeConfig:
    name: str
    timeout: int = 30
    retries: int = 3
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `ImageProcessor` |
| Functions | snake_case | `process_batch` |
| Constants | UPPER_SNAKE | `MAX_RETRIES` |
| Private | _prefix | `_internal_cache` |
| Protected | __prefix | `__secret_key` |

---

## Class Templates

### @BaseNode - Integration Node Template

```python
"""
Base class for all integration nodes.
All external tool integrations MUST inherit from this.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging

@dataclass
class NodeResult:
    """Standardized node execution result."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseNode(ABC):
    """
    Abstract base class for integration nodes.

    Attributes:
        name: Human-readable node identifier
        logger: Configured logger instance
        config: Node-specific configuration

    Example:
        >>> class ComfyUINode(BaseNode):
        ...     def execute(self, input_data):
        ...         return NodeResult(success=True, data=generated_image)
    """

    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"node.{name}")

    @abstractmethod
    def execute(self, input_data: Any) -> NodeResult:
        """Execute the node's primary function."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate configuration and dependencies."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"
```

### @BaseClient - Marketplace Client Template

```python
"""
Base class for marketplace API clients.
All marketplace integrations MUST inherit from this.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class AssetUploadResult:
    """Result from marketplace upload operation."""
    success: bool
    asset_id: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None


class BaseMarketplaceClient(ABC):
    """
    Abstract base for marketplace API clients.

    Attributes:
        name: Marketplace identifier
        api_key: Authentication key
        base_url: API endpoint base URL
    """

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logging.getLogger(f"marketplace.{self.__class__.__name__}")

    @abstractmethod
    def authenticate(self) -> bool:
        """Verify API authentication."""
        pass

    @abstractmethod
    def upload(self, file_path: Path, metadata: Dict) -> AssetUploadResult:
        """Upload asset to marketplace."""
        pass

    @abstractmethod
    def list_assets(self, limit: int = 100) -> List[Dict]:
        """List uploaded assets."""
        pass
```

---

## Function Patterns

### @def - Standard Function Template

```python
def process_batch(
    items: List[str],
    processor: Callable[[str], Any],
    *,
    batch_size: int = 10,
    timeout: float = 30.0,
) -> List[Any]:
    """
    Process items in batches with timeout.

    Args:
        items: List of items to process
        processor: Callable that processes each item
        batch_size: Number of items per batch (default: 10)
        timeout: Max seconds per batch (default: 30.0)

    Returns:
        List of processed results

    Raises:
        TimeoutError: If batch exceeds timeout
        ProcessingError: If processor fails

    Example:
        >>> results = process_batch(
        ...     items=["a", "b", "c"],
        ...     processor=lambda x: x.upper(),
        ...     batch_size=2
        ... )
        >>> results
        ['A', 'B', 'C']
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        # Process batch...
        results.extend([processor(item) for item in batch])
    return results
```

---

## Decorators

### @log_execution

```python
import functools
import time
import logging

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
```

### @retry

```python
import functools
import time
from typing import Type, Tuple

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
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
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator
```

### @cache_result

```python
import functools
from typing import Callable, Any

def cache_result(ttl_seconds: int = 300):
    """Cache function results with TTL."""
    def decorator(func: Callable) -> Callable:
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()

            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    return result

            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        return wrapper
    return decorator
```

---

## Error Handling

### Exception Hierarchy

```python
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
```

---

## Logging

### Structured Logging Setup

```python
import logging
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def setup_logging(level: str = "INFO") -> None:
    """Configure application logging."""
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())

    root = logging.getLogger()
    root.setLevel(getattr(logging, level))
    root.addHandler(handler)
```

---

## Testing

### Test Template

```python
import pytest
from unittest.mock import Mock, patch

class TestNodeExecution:
    """Test suite for node execution."""

    @pytest.fixture
    def node(self):
        """Create test node instance."""
        return ComfyUINode(name="test", config={"timeout": 10})

    @pytest.fixture
    def mock_api(self):
        """Mock external API."""
        with patch("src.integrations.comfyui.ComfyAPI") as mock:
            yield mock

    def test_execute_success(self, node, mock_api):
        """Test successful node execution."""
        mock_api.return_value.generate.return_value = {"image": "data"}

        result = node.execute({"prompt": "test"})

        assert result.success is True
        assert result.data is not None

    def test_execute_failure_returns_error(self, node, mock_api):
        """Test node execution handles errors gracefully."""
        mock_api.return_value.generate.side_effect = Exception("API down")

        result = node.execute({"prompt": "test"})

        assert result.success is False
        assert "API down" in result.error
```

---

## LangGraph Nodes

### LangGraph State & Node Pattern

```python
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

class AssetState(TypedDict):
    """State schema for asset generation workflow."""
    prompt: str
    style: str
    images: Sequence[str]
    metadata: dict
    errors: Sequence[str]


def generate_node(state: AssetState) -> AssetState:
    """Generate images from prompt."""
    # Node implementation
    return {
        **state,
        "images": [...],
    }


def upscale_node(state: AssetState) -> AssetState:
    """Upscale generated images with Anime4K."""
    return {
        **state,
        "images": [upscale(img) for img in state["images"]],
    }


def build_workflow() -> StateGraph:
    """Build the asset generation workflow graph."""
    workflow = StateGraph(AssetState)

    workflow.add_node("generate", generate_node)
    workflow.add_node("upscale", upscale_node)
    workflow.add_node("denoise", denoise_node)
    workflow.add_node("export", export_node)

    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "upscale")
    workflow.add_edge("upscale", "denoise")
    workflow.add_edge("denoise", "export")
    workflow.add_edge("export", END)

    return workflow.compile()
```

---

## Quick Reference

| Pattern | Usage |
|---------|-------|
| `@log_execution` | Add to any function needing audit trail |
| `@retry(3)` | Wrap external API calls |
| `BaseNode` | Inherit for all integration nodes |
| `NodeResult` | Return from all node executions |
| `AssetFarmError` | Base for custom exceptions |

---

*Last updated: 2025-12-05*
