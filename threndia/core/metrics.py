"""
Metric Analyzer for market data analysis
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from ..models.data_models import MetricData


class MetricResult:
    """Result of a metric analysis"""
    
    def __init__(
        self,
        metric_name: str,
        value: float,
        trend: str,
        volatility: float,
        analysis_details: Optional[Dict[str, Any]] = None
    ):
        self.metric_name = metric_name
        self.value = value
        self.trend = trend
        self.volatility = volatility
        self.analysis_details = analysis_details or {}
        self.timestamp = datetime.now()


class MetricAnalyzer:
    """Analyzes market metrics for patterns and trends"""
    
    def __init__(self, window_size: int = 20):
        """
        Initialize the MetricAnalyzer
        
        Args:
            window_size: Number of data points to consider for analysis
        """
        self.window_size = window_size
        self.metric_history: Dict[str, List[MetricData]] = {}
    
    def add_metric_data(self, metric_data: MetricData) -> None:
        """
        Add new metric data point
        
        Args:
            metric_data: MetricData instance to add
        """
        metric_name = metric_data.metric_name
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
        
        self.metric_history[metric_name].append(metric_data)
        
        # Keep only the latest window_size points
        if len(self.metric_history[metric_name]) > self.window_size:
            self.metric_history[metric_name] = self.metric_history[metric_name][-self.window_size:]
    
    def analyze_metric(self, metric_name: str) -> Optional[MetricResult]:
        """
        Analyze a specific metric
        
        Args:
            metric_name: Name of the metric to analyze
            
        Returns:
            MetricResult with analysis or None if insufficient data
        """
        if metric_name not in self.metric_history:
            return None
        
        data_points = self.metric_history[metric_name]
        if len(data_points) < 2:
            return None
        
        values = np.array([point.value for point in data_points])
        
        # Calculate trend
        if len(values) >= 2:
            trend_slope = (values[-1] - values[0]) / len(values)
            if trend_slope > 0.01:
                trend = "upward"
            elif trend_slope < -0.01:
                trend = "downward"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        # Calculate volatility
        volatility = float(np.std(values))
        
        # Calculate additional metrics
        mean_value = float(np.mean(values))
        max_value = float(np.max(values))
        min_value = float(np.min(values))
        
        analysis_details = {
            "mean": mean_value,
            "max": max_value,
            "min": min_value,
            "data_points": len(values),
            "trend_slope": float(trend_slope) if len(values) >= 2 else 0.0,
        }
        
        return MetricResult(
            metric_name=metric_name,
            value=float(values[-1]),
            trend=trend,
            volatility=volatility,
            analysis_details=analysis_details
        )
    
    def analyze_all_metrics(self) -> List[MetricResult]:
        """
        Analyze all available metrics
        
        Returns:
            List of MetricResult objects
        """
        results = []
        for metric_name in self.metric_history.keys():
            result = self.analyze_metric(metric_name)
            if result:
                results.append(result)
        return results
    
    def get_correlation(self, metric1: str, metric2: str) -> Optional[float]:
        """
        Calculate correlation between two metrics
        
        Args:
            metric1: First metric name
            metric2: Second metric name
            
        Returns:
            Correlation coefficient or None
        """
        if metric1 not in self.metric_history or metric2 not in self.metric_history:
            return None
        
        data1 = self.metric_history[metric1]
        data2 = self.metric_history[metric2]
        
        # Align data by timestamp
        min_length = min(len(data1), len(data2))
        if min_length < 2:
            return None
        
        values1 = np.array([point.value for point in data1[-min_length:]])
        values2 = np.array([point.value for point in data2[-min_length:]])
        
        correlation = float(np.corrcoef(values1, values2)[0, 1])
        return correlation
