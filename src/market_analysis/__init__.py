"""
Market Analysis Module.
Core logic for trend detection and signal processing.
Designed to cooperate with the Threndia repository.
"""
from .signals import MarketSignal, SignalType
from .trendradar import TrendRadar

__all__ = ["TrendRadar", "MarketSignal", "SignalType"]
