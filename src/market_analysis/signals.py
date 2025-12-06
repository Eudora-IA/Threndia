"""
Market Signal Definitions.
Pydantic models for standardizing market analysis data.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    """Types of market signals."""
    TREND_RISING = "trend_rising"
    TREND_FALLING = "trend_falling"
    VOLUME_SPIKE = "volume_spike"
    SENTIMENT_POSITIVE = "sentiment_positive"
    SENTIMENT_NEGATIVE = "sentiment_negative"
    SOCIAL_VIRAL = "social_viral"
    NEWS_MENTION = "news_mention"

class MarketSignal(BaseModel):
    """
    Represents a market analysis signal.
    Used for communication between Fazenda and Threndia.
    """
    id: str = Field(..., description="Unique signal ID")
    signal_type: SignalType
    source: str = Field(..., description="Data source (e.g., twitter, opensea, google_trends)")
    keyword: str = Field(..., description="Keyword or topic detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Optional fields for enrichment
    related_keywords: List[str] = Field(default_factory=list)
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)

class SignalBatch(BaseModel):
    """Batch of signals for bulk processing."""
    signals: List[MarketSignal]
    batch_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
