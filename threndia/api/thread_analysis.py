"""
Thread Analysis API - Main API for agent self-learning and market analysis
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from ..core.agent import Agent, AgentLearningResult
from ..core.metrics import MetricAnalyzer, MetricResult
from ..core.thread_manager import ThreadManager
from ..models.data_models import (
    MetricData,
    AnalysisResult,
    ThreadConfig,
    LearningResult,
)


class ThreadAnalysisAPI:
    """
    Main API for Thread Analysis with agent self-learning capabilities
    
    This API enables agents to perform concurrent market analysis across
    multiple metrics with self-learning capabilities.
    """
    
    # Confidence calculation constant
    MIN_DATA_POINTS_FOR_FULL_CONFIDENCE = 20.0
    
    def __init__(self, max_concurrent_threads: int = 10):
        """
        Initialize the Thread Analysis API
        
        Args:
            max_concurrent_threads: Maximum number of concurrent analysis threads
        """
        self.thread_manager = ThreadManager(max_workers=max_concurrent_threads)
        self.metric_analyzer = MetricAnalyzer()
        self.agents: Dict[str, Agent] = {}
        self.analysis_history: List[AnalysisResult] = []
    
    def create_agent(self, agent_id: str, learning_rate: float = 0.01) -> Agent:
        """
        Create a new agent for analysis
        
        Args:
            agent_id: Unique identifier for the agent
            learning_rate: Learning rate for self-learning
            
        Returns:
            Created Agent instance
        """
        agent = Agent(agent_id=agent_id, learning_rate=learning_rate)
        self.agents[agent_id] = agent
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get an existing agent
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(agent_id)
    
    def add_metric_data(self, metric_data: MetricData) -> None:
        """
        Add market metric data for analysis
        
        Args:
            metric_data: MetricData instance
        """
        self.metric_analyzer.add_metric_data(metric_data)
    
    def add_multiple_metrics(self, metrics: List[MetricData]) -> None:
        """
        Add multiple metric data points
        
        Args:
            metrics: List of MetricData instances
        """
        for metric in metrics:
            self.add_metric_data(metric)
    
    def analyze_metrics(
        self,
        metric_names: Optional[List[str]] = None
    ) -> List[MetricResult]:
        """
        Analyze market metrics
        
        Args:
            metric_names: Optional list of specific metrics to analyze.
                         If None, analyzes all available metrics.
        
        Returns:
            List of MetricResult objects
        """
        if metric_names:
            results = []
            for name in metric_names:
                result = self.metric_analyzer.analyze_metric(name)
                if result:
                    results.append(result)
            return results
        else:
            return self.metric_analyzer.analyze_all_metrics()
    
    async def threaded_analysis(
        self,
        agent_id: str,
        metric_names: List[str],
        thread_config: Optional[ThreadConfig] = None
    ) -> AnalysisResult:
        """
        Perform threaded analysis with an agent
        
        Args:
            agent_id: Agent to perform analysis
            metric_names: List of metrics to analyze
            thread_config: Optional thread configuration
            
        Returns:
            AnalysisResult with insights
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Analyze metrics
        metric_results = self.analyze_metrics(metric_names)
        
        # Agent analyzes the results
        insights = agent.analyze_metrics(metric_results)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(metric_results)
        
        # Create analysis result
        thread_id = thread_config.thread_id if thread_config else f"thread_{datetime.now().timestamp()}"
        
        analysis_result = AnalysisResult(
            thread_id=thread_id,
            agent_id=agent_id,
            metrics_analyzed=metric_names,
            insights=insights,
            confidence_score=confidence_score,
        )
        
        self.analysis_history.append(analysis_result)
        return analysis_result
    
    async def multi_agent_analysis(
        self,
        agent_ids: List[str],
        metric_names: List[str],
        thread_config: Optional[ThreadConfig] = None
    ) -> List[AnalysisResult]:
        """
        Perform concurrent analysis with multiple agents
        
        Args:
            agent_ids: List of agent IDs to use
            metric_names: Metrics to analyze
            thread_config: Optional thread configuration
            
        Returns:
            List of AnalysisResult from each agent
        """
        tasks = []
        for agent_id in agent_ids:
            config = ThreadConfig(
                thread_id=f"{agent_id}_thread",
                metrics_to_analyze=metric_names,
            ) if not thread_config else thread_config
            
            tasks.append((
                self.threaded_analysis,
                (agent_id, metric_names, config),
                {}
            ))
        
        results = await self.thread_manager.run_concurrent_analyses(tasks, thread_config)
        return [r for r in results if isinstance(r, AnalysisResult)]
    
    def agent_self_learning(
        self,
        agent_id: str,
        metric_names: Optional[List[str]] = None,
        actual_outcomes: Optional[Dict[str, float]] = None
    ) -> LearningResult:
        """
        Trigger agent self-learning from analysis
        
        Args:
            agent_id: Agent to perform learning
            metric_names: Optional specific metrics to learn from
            actual_outcomes: Optional actual outcomes for supervised learning
            
        Returns:
            LearningResult with learning details
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Analyze metrics
        metric_results = self.analyze_metrics(metric_names)
        
        # Perform learning
        learning_result = agent.learn_from_analysis(metric_results, actual_outcomes)
        
        # Convert to LearningResult model
        return LearningResult(
            agent_id=learning_result.agent_id,
            learning_type=learning_result.learning_type,
            learned_patterns=learning_result.learned_patterns,
            accuracy_improvement=learning_result.accuracy_improvement,
            training_iterations=learning_result.training_iterations,
        )
    
    async def continuous_learning_cycle(
        self,
        agent_id: str,
        metric_names: List[str],
        cycles: int = 5,
        interval_seconds: float = 1.0
    ) -> List[LearningResult]:
        """
        Run continuous learning cycles for an agent
        
        Args:
            agent_id: Agent to train
            metric_names: Metrics to analyze and learn from
            cycles: Number of learning cycles
            interval_seconds: Time between cycles
            
        Returns:
            List of LearningResult from each cycle
        """
        results = []
        
        for i in range(cycles):
            learning_result = self.agent_self_learning(agent_id, metric_names)
            results.append(learning_result)
            
            if i < cycles - 1:
                await asyncio.sleep(interval_seconds)
        
        return results
    
    def get_metric_correlation(self, metric1: str, metric2: str) -> Optional[float]:
        """
        Get correlation between two metrics
        
        Args:
            metric1: First metric name
            metric2: Second metric name
            
        Returns:
            Correlation coefficient or None
        """
        return self.metric_analyzer.get_correlation(metric1, metric2)
    
    def get_analysis_history(
        self,
        agent_id: Optional[str] = None,
        limit: int = 10
    ) -> List[AnalysisResult]:
        """
        Get analysis history
        
        Args:
            agent_id: Optional filter by agent
            limit: Maximum number of results
            
        Returns:
            List of AnalysisResult
        """
        history = self.analysis_history
        
        if agent_id:
            history = [h for h in history if h.agent_id == agent_id]
        
        return history[-limit:]
    
    def get_agent_knowledge(self, agent_id: str) -> Dict[str, Any]:
        """
        Get knowledge summary for an agent
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Knowledge summary dictionary
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        return agent.get_knowledge_summary()
    
    def _calculate_confidence(self, metric_results: List[MetricResult]) -> float:
        """
        Calculate confidence score based on metric results
        
        Args:
            metric_results: List of metric analysis results
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not metric_results:
            return 0.0
        
        # Base confidence on number of metrics and their data quality
        total_confidence = 0.0
        for result in metric_results:
            data_points = result.analysis_details.get("data_points", 0)
            metric_confidence = min(
                data_points / self.MIN_DATA_POINTS_FOR_FULL_CONFIDENCE, 1.0
            )
            total_confidence += metric_confidence
        
        return min(total_confidence / len(metric_results), 1.0)
    
    def shutdown(self) -> None:
        """Shutdown the API and cleanup resources"""
        self.thread_manager.shutdown(wait=True)
