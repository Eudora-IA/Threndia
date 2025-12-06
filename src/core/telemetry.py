"""
Structured logging and telemetry system.
Provides JSON logging, Prometheus metrics, and Celery telemetry.
"""
import functools
import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional

# Prometheus (conditional import)
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# ============================================================================
# STRUCTURED LOGGING
# ============================================================================

class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter for observability."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add trace ID if present
        if hasattr(record, "trace_id"):
            log_data["trace_id"] = record.trace_id

        return json.dumps(log_data)


class ContextLogger(logging.LoggerAdapter):
    """Logger with context propagation."""

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON structured format
        log_file: Optional file path for logs
    """
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    if json_format:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
    handlers.append(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        handlers.append(file_handler)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    for handler in handlers:
        root.addHandler(handler)


def get_logger(name: str, **context) -> ContextLogger:
    """Get a logger with optional context."""
    logger = logging.getLogger(name)
    return ContextLogger(logger, context)


# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

@dataclass
class MetricsRegistry:
    """Central registry for Prometheus metrics."""

    # Counters
    tasks_total: Any = None
    tasks_failed: Any = None
    images_generated: Any = None
    api_requests: Any = None

    # Histograms
    task_duration: Any = None
    generation_time: Any = None
    upscale_time: Any = None

    # Gauges
    active_tasks: Any = None
    queue_size: Any = None
    gpu_memory: Any = None

    def __post_init__(self):
        if PROMETHEUS_AVAILABLE:
            self._init_metrics()

    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        # Counters
        self.tasks_total = Counter(
            "assetfarm_tasks_total",
            "Total number of tasks",
            ["task_type", "status"]
        )
        self.tasks_failed = Counter(
            "assetfarm_tasks_failed_total",
            "Total number of failed tasks",
            ["task_type", "error_type"]
        )
        self.images_generated = Counter(
            "assetfarm_images_generated_total",
            "Total images generated",
            ["model", "style"]
        )
        self.api_requests = Counter(
            "assetfarm_api_requests_total",
            "Total API requests",
            ["endpoint", "method", "status"]
        )

        # Histograms
        self.task_duration = Histogram(
            "assetfarm_task_duration_seconds",
            "Task execution duration",
            ["task_type"],
            buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120]
        )
        self.generation_time = Histogram(
            "assetfarm_generation_seconds",
            "Image generation time",
            ["model"]
        )
        self.upscale_time = Histogram(
            "assetfarm_upscale_seconds",
            "Image upscale time",
            ["scale"]
        )

        # Gauges
        self.active_tasks = Gauge(
            "assetfarm_active_tasks",
            "Currently running tasks",
            ["task_type"]
        )
        self.queue_size = Gauge(
            "assetfarm_queue_size",
            "Tasks in queue",
            ["queue_name"]
        )
        self.gpu_memory = Gauge(
            "assetfarm_gpu_memory_bytes",
            "GPU memory usage",
            ["gpu_id"]
        )


# Global metrics instance
metrics = MetricsRegistry()


def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics HTTP server."""
    if PROMETHEUS_AVAILABLE:
        start_http_server(port)
        logging.info(f"Prometheus metrics server started on port {port}")
    else:
        logging.warning("Prometheus not available, metrics server not started")


# ============================================================================
# DECORATORS
# ============================================================================

def track_time(metric_name: str, labels: Optional[Dict] = None):
    """Decorator to track function execution time."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start

                if PROMETHEUS_AVAILABLE and hasattr(metrics, metric_name):
                    histogram = getattr(metrics, metric_name)
                    if labels:
                        histogram.labels(**labels).observe(duration)
                    else:
                        histogram.observe(duration)

                return result
            except Exception as e:
                if PROMETHEUS_AVAILABLE:
                    metrics.tasks_failed.labels(
                        task_type=func.__name__,
                        error_type=type(e).__name__
                    ).inc()
                raise
        return wrapper
    return decorator


def count_calls(counter_name: str, labels: Optional[Dict] = None):
    """Decorator to count function calls."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if PROMETHEUS_AVAILABLE and hasattr(metrics, counter_name):
                counter = getattr(metrics, counter_name)
                if labels:
                    counter.labels(**labels).inc()
                else:
                    counter.inc()
            return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def track_active(gauge_name: str, labels: Optional[Dict] = None):
    """Context manager to track active operations."""
    if PROMETHEUS_AVAILABLE and hasattr(metrics, gauge_name):
        gauge = getattr(metrics, gauge_name)
        if labels:
            gauge.labels(**labels).inc()
        else:
            gauge.inc()
    try:
        yield
    finally:
        if PROMETHEUS_AVAILABLE and hasattr(metrics, gauge_name):
            if labels:
                gauge.labels(**labels).dec()
            else:
                gauge.dec()


# ============================================================================
# ERROR TRACKING
# ============================================================================

@dataclass
class ErrorEvent:
    """Structured error event for tracking."""
    error_type: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ErrorTracker:
    """Centralized error tracking and aggregation."""

    def __init__(self, max_errors: int = 1000):
        self.max_errors = max_errors
        self._errors: list = []
        self._lock = threading.Lock()
        self.logger = get_logger("error_tracker")

    def track(
        self,
        error: Exception,
        context: Optional[Dict] = None,
        severity: str = "error",
    ) -> ErrorEvent:
        """Track an error with context."""
        import traceback

        event = ErrorEvent(
            error_type=type(error).__name__,
            message=str(error),
            stack_trace=traceback.format_exc(),
            context=context or {},
            severity=severity,
        )

        with self._lock:
            self._errors.append(event)
            if len(self._errors) > self.max_errors:
                self._errors = self._errors[-self.max_errors:]

        self.logger.error(
            f"{event.error_type}: {event.message}",
            extra={"error_event": event.to_dict()}
        )

        return event

    def get_recent(self, count: int = 10) -> list:
        """Get recent errors."""
        with self._lock:
            return list(reversed(self._errors[-count:]))

    def get_by_type(self, error_type: str) -> list:
        """Get errors by type."""
        with self._lock:
            return [e for e in self._errors if e.error_type == error_type]

    def clear(self):
        """Clear all tracked errors."""
        with self._lock:
            self._errors.clear()


# Global error tracker
error_tracker = ErrorTracker()


# ============================================================================
# CELERY TELEMETRY
# ============================================================================

def setup_celery_telemetry(app):
    """Configure Celery task telemetry."""
    if not PROMETHEUS_AVAILABLE:
        return

    from celery.signals import task_failure, task_postrun, task_prerun, task_retry

    @task_prerun.connect
    def task_prerun_handler(task_id, task, *args, **kwargs):
        metrics.active_tasks.labels(task_type=task.name).inc()

    @task_postrun.connect
    def task_postrun_handler(task_id, task, *args, **kwargs):
        metrics.active_tasks.labels(task_type=task.name).dec()
        metrics.tasks_total.labels(task_type=task.name, status="success").inc()

    @task_failure.connect
    def task_failure_handler(task_id, exception, *args, **kwargs):
        metrics.tasks_failed.labels(
            task_type=kwargs.get("sender", {}).name or "unknown",
            error_type=type(exception).__name__
        ).inc()

    @task_retry.connect
    def task_retry_handler(request, reason, *args, **kwargs):
        metrics.tasks_total.labels(task_type=request.name, status="retry").inc()
