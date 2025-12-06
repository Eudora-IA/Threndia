"""
TrendRadar - Market Signal Processor.
Analyzes and aggregates signals from multiple sources.
Designed for Threndia cooperation.
"""
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .signals import MarketSignal, SignalBatch, SignalType

logger = logging.getLogger(__name__)

class TrendRadar:
    """
    Main class for processing market trends and signals.

    Integrates with:
    - Web scrapers (social media, marketplaces)
    - ChromaDB for signal storage and retrieval
    - External APIs (Google Trends, Twitter, etc.)
    """

    def __init__(self, db_client=None):
        """
        Initialize TrendRadar.

        Args:
            db_client: ChromaDB client for signal persistence (optional).
        """
        self.db_client = db_client
        self._signal_cache: List[MarketSignal] = []

    def process_raw_data(self, source: str, data: Dict[str, Any]) -> Optional[MarketSignal]:
        """
        Convert raw scraped data into a MarketSignal.

        Args:
            source: Name of the data source.
            data: Raw data dictionary.

        Returns:
            MarketSignal or None if data is invalid.
        """
        try:
            # Extract common fields
            keyword = data.get("keyword") or data.get("topic") or data.get("term", "unknown")
            confidence = float(data.get("confidence", data.get("score", 0.5)))

            # Determine signal type based on data
            signal_type = self._infer_signal_type(data)

            signal = MarketSignal(
                id=str(uuid.uuid4()),
                signal_type=signal_type,
                source=source,
                keyword=keyword,
                confidence=min(max(confidence, 0.0), 1.0),  # Clamp to 0-1
                metadata=data
            )

            self._signal_cache.append(signal)
            logger.info(f"Processed signal: {signal.keyword} ({signal.signal_type})")
            return signal

        except Exception as e:
            logger.error(f"Failed to process data from {source}: {e}")
            return None

    def _infer_signal_type(self, data: Dict[str, Any]) -> SignalType:
        """Infer signal type from raw data."""
        # Check for explicit type
        if "type" in data:
            try:
                return SignalType(data["type"])
            except ValueError:
                pass

        # Infer from sentiment
        sentiment = data.get("sentiment", 0)
        if sentiment > 0.3:
            return SignalType.SENTIMENT_POSITIVE
        elif sentiment < -0.3:
            return SignalType.SENTIMENT_NEGATIVE

        # Default
        return SignalType.TREND_RISING

    def get_recent_signals(self, hours: int = 24) -> List[MarketSignal]:
        """Get signals from the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [s for s in self._signal_cache if s.timestamp > cutoff]

    def create_batch(self) -> SignalBatch:
        """Create a batch from cached signals."""
        batch = SignalBatch(
            signals=self._signal_cache.copy(),
            batch_id=str(uuid.uuid4())
        )
        self._signal_cache.clear()
        return batch

    def store_to_db(self, signal: MarketSignal) -> bool:
        """Store a signal to ChromaDB (if client is configured)."""
        if not self.db_client:
            logger.warning("No DB client configured, signal not persisted")
            return False

        try:
            # Implementation depends on ChromaDB wrapper
            self.db_client.add_document(
                content=f"{signal.keyword}: {signal.signal_type.value}",
                metadata=signal.model_dump()
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store signal: {e}")
            return False
