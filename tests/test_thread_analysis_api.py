"""
Tests for the Thread Analysis API
"""
import pytest
import asyncio
from datetime import datetime
from threndia import ThreadAnalysisAPI, Agent, MetricAnalyzer
from threndia.models.data_models import MetricData, ThreadConfig


class TestMetricAnalyzer:
    """Test MetricAnalyzer functionality"""
    
    def test_add_metric_data(self):
        """Test adding metric data"""
        analyzer = MetricAnalyzer()
        metric = MetricData(metric_name="price", value=100.0)
        analyzer.add_metric_data(metric)
        
        assert "price" in analyzer.metric_history
        assert len(analyzer.metric_history["price"]) == 1
    
    def test_analyze_metric(self):
        """Test metric analysis"""
        analyzer = MetricAnalyzer()
        
        # Add multiple data points
        for i in range(10):
            metric = MetricData(metric_name="price", value=100.0 + i)
            analyzer.add_metric_data(metric)
        
        result = analyzer.analyze_metric("price")
        
        assert result is not None
        assert result.metric_name == "price"
        assert result.trend == "upward"
        assert result.value == 109.0
    
    def test_analyze_all_metrics(self):
        """Test analyzing all metrics"""
        analyzer = MetricAnalyzer()
        
        # Add data for multiple metrics
        for i in range(5):
            analyzer.add_metric_data(MetricData(metric_name="price", value=100.0 + i))
            analyzer.add_metric_data(MetricData(metric_name="volume", value=1000.0 - i * 10))
        
        results = analyzer.analyze_all_metrics()
        
        assert len(results) == 2
        metric_names = [r.metric_name for r in results]
        assert "price" in metric_names
        assert "volume" in metric_names
    
    def test_correlation(self):
        """Test correlation calculation"""
        analyzer = MetricAnalyzer()
        
        # Add correlated data
        for i in range(10):
            analyzer.add_metric_data(MetricData(metric_name="metric1", value=float(i)))
            analyzer.add_metric_data(MetricData(metric_name="metric2", value=float(i * 2)))
        
        correlation = analyzer.get_correlation("metric1", "metric2")
        
        assert correlation is not None
        assert correlation > 0.9  # Should be highly correlated


class TestAgent:
    """Test Agent functionality"""
    
    def test_agent_creation(self):
        """Test agent creation"""
        agent = Agent(agent_id="agent_1", learning_rate=0.01)
        
        assert agent.agent_id == "agent_1"
        assert agent.learning_rate == 0.01
        assert len(agent.learning_history) == 0
    
    def test_analyze_metrics(self):
        """Test agent metric analysis"""
        agent = Agent(agent_id="agent_1")
        analyzer = MetricAnalyzer()
        
        # Add metric data
        for i in range(10):
            analyzer.add_metric_data(MetricData(metric_name="price", value=100.0 + i))
        
        metric_results = analyzer.analyze_all_metrics()
        insights = agent.analyze_metrics(metric_results)
        
        assert "total_metrics" in insights
        assert insights["total_metrics"] == 1
        assert "trends" in insights
        assert "recommendations" in insights
    
    def test_self_learning(self):
        """Test agent self-learning"""
        agent = Agent(agent_id="agent_1", learning_rate=0.1)
        analyzer = MetricAnalyzer()
        
        # Add metric data
        for i in range(10):
            analyzer.add_metric_data(MetricData(metric_name="price", value=100.0 + i))
        
        metric_results = analyzer.analyze_all_metrics()
        learning_result = agent.learn_from_analysis(metric_results)
        
        assert learning_result.agent_id == "agent_1"
        assert learning_result.training_iterations > 0
        assert len(agent.learning_history) == 1
    
    def test_knowledge_summary(self):
        """Test getting knowledge summary"""
        agent = Agent(agent_id="agent_1")
        
        summary = agent.get_knowledge_summary()
        
        assert summary["agent_id"] == "agent_1"
        assert "total_patterns" in summary
        assert "learning_sessions" in summary


class TestThreadAnalysisAPI:
    """Test ThreadAnalysisAPI functionality"""
    
    def test_api_initialization(self):
        """Test API initialization"""
        api = ThreadAnalysisAPI(max_concurrent_threads=5)
        
        assert api.thread_manager.max_workers == 5
        assert len(api.agents) == 0
        assert len(api.analysis_history) == 0
    
    def test_create_agent(self):
        """Test creating agents through API"""
        api = ThreadAnalysisAPI()
        
        agent = api.create_agent("agent_1", learning_rate=0.05)
        
        assert agent.agent_id == "agent_1"
        assert api.get_agent("agent_1") == agent
    
    def test_add_metric_data(self):
        """Test adding metric data through API"""
        api = ThreadAnalysisAPI()
        
        metric = MetricData(metric_name="price", value=100.0)
        api.add_metric_data(metric)
        
        assert "price" in api.metric_analyzer.metric_history
    
    def test_analyze_metrics(self):
        """Test analyzing metrics through API"""
        api = ThreadAnalysisAPI()
        
        # Add multiple metrics
        for i in range(10):
            api.add_metric_data(MetricData(metric_name="price", value=100.0 + i))
        
        results = api.analyze_metrics(["price"])
        
        assert len(results) == 1
        assert results[0].metric_name == "price"
    
    @pytest.mark.asyncio
    async def test_threaded_analysis(self):
        """Test threaded analysis"""
        api = ThreadAnalysisAPI()
        
        # Create agent and add data
        api.create_agent("agent_1")
        for i in range(10):
            api.add_metric_data(MetricData(metric_name="price", value=100.0 + i))
        
        # Perform threaded analysis
        result = await api.threaded_analysis("agent_1", ["price"])
        
        assert result.agent_id == "agent_1"
        assert "price" in result.metrics_analyzed
        assert result.confidence_score >= 0.0
        assert result.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_multi_agent_analysis(self):
        """Test multi-agent concurrent analysis"""
        api = ThreadAnalysisAPI()
        
        # Create multiple agents
        api.create_agent("agent_1")
        api.create_agent("agent_2")
        api.create_agent("agent_3")
        
        # Add metric data
        for i in range(10):
            api.add_metric_data(MetricData(metric_name="price", value=100.0 + i))
            api.add_metric_data(MetricData(metric_name="volume", value=1000.0 + i * 10))
        
        # Perform multi-agent analysis
        results = await api.multi_agent_analysis(
            ["agent_1", "agent_2", "agent_3"],
            ["price", "volume"]
        )
        
        assert len(results) == 3
        agent_ids = [r.agent_id for r in results]
        assert "agent_1" in agent_ids
        assert "agent_2" in agent_ids
        assert "agent_3" in agent_ids
    
    def test_agent_self_learning_api(self):
        """Test agent self-learning through API"""
        api = ThreadAnalysisAPI()
        
        # Create agent and add data
        api.create_agent("agent_1")
        for i in range(10):
            api.add_metric_data(MetricData(metric_name="price", value=100.0 + i))
        
        # Trigger learning
        learning_result = api.agent_self_learning("agent_1", ["price"])
        
        assert learning_result.agent_id == "agent_1"
        assert learning_result.training_iterations > 0
    
    @pytest.mark.asyncio
    async def test_continuous_learning(self):
        """Test continuous learning cycles"""
        api = ThreadAnalysisAPI()
        
        # Create agent and add data
        api.create_agent("agent_1")
        for i in range(10):
            api.add_metric_data(MetricData(metric_name="price", value=100.0 + i))
        
        # Run continuous learning
        results = await api.continuous_learning_cycle(
            "agent_1",
            ["price"],
            cycles=3,
            interval_seconds=0.1
        )
        
        assert len(results) == 3
        for result in results:
            assert result.agent_id == "agent_1"
    
    def test_get_metric_correlation(self):
        """Test getting metric correlation through API"""
        api = ThreadAnalysisAPI()
        
        # Add correlated data
        for i in range(10):
            api.add_metric_data(MetricData(metric_name="metric1", value=float(i)))
            api.add_metric_data(MetricData(metric_name="metric2", value=float(i * 2)))
        
        correlation = api.get_metric_correlation("metric1", "metric2")
        
        assert correlation is not None
        assert correlation > 0.9
    
    def test_get_analysis_history(self):
        """Test getting analysis history"""
        api = ThreadAnalysisAPI()
        api.create_agent("agent_1")
        
        # Add data and perform analysis
        for i in range(10):
            api.add_metric_data(MetricData(metric_name="price", value=100.0 + i))
        
        # Note: This requires running async analysis first
        # For now, just test the method exists and returns empty list
        history = api.get_analysis_history()
        
        assert isinstance(history, list)
    
    def test_get_agent_knowledge(self):
        """Test getting agent knowledge"""
        api = ThreadAnalysisAPI()
        api.create_agent("agent_1")
        
        knowledge = api.get_agent_knowledge("agent_1")
        
        assert knowledge["agent_id"] == "agent_1"
        assert "total_patterns" in knowledge
    
    def test_shutdown(self):
        """Test API shutdown"""
        api = ThreadAnalysisAPI()
        
        # Should not raise any exceptions
        api.shutdown()
