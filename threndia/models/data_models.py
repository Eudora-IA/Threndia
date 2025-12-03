"""
Data models for Thread Analysis API
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class MetricData(BaseModel):
    """Model for market metric data"""
    metric_name: str
    value: float
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """Model for analysis results"""
    thread_id: str
    agent_id: str
    metrics_analyzed: List[str]
    insights: Dict[str, Any]
    confidence_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)


class ThreadConfig(BaseModel):
    """Configuration for analysis threads"""
    thread_id: str
    max_concurrent_analyses: int = 5
    timeout_seconds: int = 300
    retry_attempts: int = 3
    metrics_to_analyze: List[str] = Field(default_factory=list)


class LearningResult(BaseModel):
    """Model for agent learning results"""
    agent_id: str
    learning_type: str
    learned_patterns: Dict[str, Any]
    accuracy_improvement: float
    training_iterations: int
    timestamp: datetime = Field(default_factory=datetime.now)
