"""
Agent with self-learning capabilities
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from .metrics import MetricResult


class AgentLearningResult:
    """Result of agent learning process"""
    
    def __init__(
        self,
        agent_id: str,
        learning_type: str,
        learned_patterns: Dict[str, Any],
        accuracy_improvement: float,
        training_iterations: int
    ):
        self.agent_id = agent_id
        self.learning_type = learning_type
        self.learned_patterns = learned_patterns
        self.accuracy_improvement = accuracy_improvement
        self.training_iterations = training_iterations
        self.timestamp = datetime.now()


class Agent:
    """Agent capable of self-learning from metric analysis"""
    
    def __init__(self, agent_id: str, learning_rate: float = 0.01):
        """
        Initialize an Agent
        
        Args:
            agent_id: Unique identifier for the agent
            learning_rate: Learning rate for self-learning (0.0 to 1.0)
        """
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.knowledge_base: Dict[str, Any] = {
            "patterns": {},
            "metric_weights": {},
            "predictions": [],
        }
        self.learning_history: List[AgentLearningResult] = []
    
    def analyze_metrics(self, metric_results: List[MetricResult]) -> Dict[str, Any]:
        """
        Analyze metrics and generate insights
        
        Args:
            metric_results: List of MetricResult objects
            
        Returns:
            Dictionary with analysis insights
        """
        insights = {
            "total_metrics": len(metric_results),
            "trends": {},
            "volatility_summary": {},
            "recommendations": [],
        }
        
        for result in metric_results:
            insights["trends"][result.metric_name] = result.trend
            insights["volatility_summary"][result.metric_name] = {
                "volatility": result.volatility,
                "current_value": result.value,
            }
        
        # Generate recommendations based on patterns
        recommendations = self._generate_recommendations(metric_results)
        insights["recommendations"] = recommendations
        
        return insights
    
    def _generate_recommendations(self, metric_results: List[MetricResult]) -> List[str]:
        """Generate recommendations based on metric analysis"""
        recommendations = []
        
        for result in metric_results:
            if result.volatility > 10.0:
                recommendations.append(
                    f"High volatility detected in {result.metric_name}: Consider risk mitigation"
                )
            
            if result.trend == "upward":
                recommendations.append(
                    f"{result.metric_name} showing upward trend: Potential growth opportunity"
                )
            elif result.trend == "downward":
                recommendations.append(
                    f"{result.metric_name} showing downward trend: Monitor closely"
                )
        
        return recommendations
    
    def learn_from_analysis(
        self,
        metric_results: List[MetricResult],
        actual_outcomes: Optional[Dict[str, float]] = None
    ) -> AgentLearningResult:
        """
        Perform self-learning based on metric analysis
        
        Args:
            metric_results: List of analyzed metrics
            actual_outcomes: Optional actual outcomes for supervised learning
            
        Returns:
            AgentLearningResult with learning details
        """
        iterations = 0
        learned_patterns = {}
        
        # Learn patterns from metrics
        for result in metric_results:
            pattern_key = f"{result.metric_name}_trend"
            if pattern_key not in self.knowledge_base["patterns"]:
                self.knowledge_base["patterns"][pattern_key] = []
            
            self.knowledge_base["patterns"][pattern_key].append({
                "trend": result.trend,
                "volatility": result.volatility,
                "value": result.value,
                "timestamp": result.timestamp,
            })
            
            learned_patterns[pattern_key] = result.trend
            iterations += 1
        
        # Update metric weights based on learning
        if actual_outcomes:
            for metric_name, actual_value in actual_outcomes.items():
                if metric_name not in self.knowledge_base["metric_weights"]:
                    self.knowledge_base["metric_weights"][metric_name] = 1.0
                
                # Adjust weight based on prediction accuracy
                predicted = next(
                    (r.value for r in metric_results if r.metric_name == metric_name),
                    None
                )
                if predicted is not None:
                    error = abs(predicted - actual_value)
                    accuracy = 1.0 / (1.0 + error)
                    
                    # Update weight with learning rate
                    current_weight = self.knowledge_base["metric_weights"][metric_name]
                    self.knowledge_base["metric_weights"][metric_name] = (
                        current_weight * (1 - self.learning_rate) +
                        accuracy * self.learning_rate
                    )
        
        # Calculate accuracy improvement
        accuracy_improvement = self._calculate_accuracy_improvement()
        
        learning_result = AgentLearningResult(
            agent_id=self.agent_id,
            learning_type="pattern_recognition",
            learned_patterns=learned_patterns,
            accuracy_improvement=accuracy_improvement,
            training_iterations=iterations
        )
        
        self.learning_history.append(learning_result)
        return learning_result
    
    def _calculate_accuracy_improvement(self) -> float:
        """Calculate improvement in accuracy over time"""
        if len(self.learning_history) < 2:
            return 0.0
        
        # Simple improvement calculation based on learning history
        recent_iterations = sum(
            lr.training_iterations for lr in self.learning_history[-5:]
        )
        baseline = 10.0
        improvement = min((recent_iterations / baseline) * 100, 100.0)
        return improvement
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's knowledge base"""
        return {
            "agent_id": self.agent_id,
            "total_patterns": len(self.knowledge_base["patterns"]),
            "metric_weights": self.knowledge_base["metric_weights"],
            "learning_sessions": len(self.learning_history),
            "total_iterations": sum(
                lr.training_iterations for lr in self.learning_history
            ),
        }
