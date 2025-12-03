"""
Threndia - Thread Analysis API for Market Analysis and Agent Self-Learning
"""

__version__ = "0.1.0"
__author__ = "Threndia Team"
__description__ = "Trends for Market Analysis and Agents Creator"

from .api.thread_analysis import ThreadAnalysisAPI
from .core.agent import Agent, AgentLearningResult
from .core.metrics import MetricAnalyzer, MetricResult
from .core.thread_manager import ThreadManager

__all__ = [
    "ThreadAnalysisAPI",
    "Agent",
    "AgentLearningResult",
    "MetricAnalyzer",
    "MetricResult",
    "ThreadManager",
]
