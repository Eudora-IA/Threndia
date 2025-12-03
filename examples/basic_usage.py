"""
Example usage of Thread Analysis API
"""
import asyncio
from datetime import datetime
from threndia import ThreadAnalysisAPI
from threndia.models.data_models import MetricData


async def main():
    """Example demonstrating the Thread Analysis API capabilities"""
    
    # Initialize the API
    print("=== Thread Analysis API Example ===\n")
    api = ThreadAnalysisAPI(max_concurrent_threads=10)
    
    # Step 1: Create agents for market analysis
    print("Step 1: Creating agents...")
    agent1 = api.create_agent("market_analyst_1", learning_rate=0.05)
    agent2 = api.create_agent("market_analyst_2", learning_rate=0.03)
    agent3 = api.create_agent("market_analyst_3", learning_rate=0.07)
    print(f"Created {len(api.agents)} agents\n")
    
    # Step 2: Add market metric data
    print("Step 2: Adding market metric data...")
    
    # Simulate stock price data
    for i in range(30):
        api.add_metric_data(MetricData(
            metric_name="stock_price",
            value=100.0 + i * 0.5 + (i % 3) * 0.2,
            metadata={"source": "market_feed"}
        ))
    
    # Simulate trading volume data
    for i in range(30):
        api.add_metric_data(MetricData(
            metric_name="trading_volume",
            value=10000.0 + i * 100 - (i % 5) * 50,
            metadata={"source": "market_feed"}
        ))
    
    # Simulate volatility index
    for i in range(30):
        api.add_metric_data(MetricData(
            metric_name="volatility_index",
            value=20.0 + (i % 10) * 0.5,
            metadata={"source": "market_feed"}
        ))
    
    print(f"Added data for 3 metrics with 30 data points each\n")
    
    # Step 3: Analyze metrics
    print("Step 3: Analyzing metrics...")
    metric_results = api.analyze_metrics(["stock_price", "trading_volume", "volatility_index"])
    
    for result in metric_results:
        print(f"\nMetric: {result.metric_name}")
        print(f"  Current Value: {result.value:.2f}")
        print(f"  Trend: {result.trend}")
        print(f"  Volatility: {result.volatility:.2f}")
        print(f"  Analysis Details: {result.analysis_details}")
    
    # Step 4: Single agent threaded analysis
    print("\n\nStep 4: Single agent threaded analysis...")
    analysis_result = await api.threaded_analysis(
        agent_id="market_analyst_1",
        metric_names=["stock_price", "trading_volume", "volatility_index"]
    )
    
    print(f"\nAgent: {analysis_result.agent_id}")
    print(f"Thread ID: {analysis_result.thread_id}")
    print(f"Confidence Score: {analysis_result.confidence_score:.2f}")
    print(f"Insights:")
    for key, value in analysis_result.insights.items():
        print(f"  {key}: {value}")
    
    # Step 5: Multi-agent concurrent analysis
    print("\n\nStep 5: Multi-agent concurrent analysis...")
    multi_results = await api.multi_agent_analysis(
        agent_ids=["market_analyst_1", "market_analyst_2", "market_analyst_3"],
        metric_names=["stock_price", "trading_volume"]
    )
    
    print(f"\nAnalyzed with {len(multi_results)} agents concurrently:")
    for result in multi_results:
        print(f"\n  Agent {result.agent_id}:")
        print(f"    Confidence: {result.confidence_score:.2f}")
        print(f"    Recommendations: {len(result.insights.get('recommendations', []))}")
    
    # Step 6: Agent self-learning
    print("\n\nStep 6: Agent self-learning...")
    
    # Simulate learning with actual outcomes
    actual_outcomes = {
        "stock_price": 114.5,
        "trading_volume": 12500.0,
    }
    
    learning_result = api.agent_self_learning(
        agent_id="market_analyst_1",
        metric_names=["stock_price", "trading_volume"],
        actual_outcomes=actual_outcomes
    )
    
    print(f"\nLearning Result for {learning_result.agent_id}:")
    print(f"  Learning Type: {learning_result.learning_type}")
    print(f"  Training Iterations: {learning_result.training_iterations}")
    print(f"  Accuracy Improvement: {learning_result.accuracy_improvement:.2f}%")
    print(f"  Learned Patterns: {learning_result.learned_patterns}")
    
    # Step 7: Continuous learning cycles
    print("\n\nStep 7: Running continuous learning cycles...")
    continuous_results = await api.continuous_learning_cycle(
        agent_id="market_analyst_2",
        metric_names=["stock_price", "volatility_index"],
        cycles=5,
        interval_seconds=0.5
    )
    
    print(f"\nCompleted {len(continuous_results)} learning cycles:")
    for i, result in enumerate(continuous_results, 1):
        print(f"  Cycle {i}: {result.training_iterations} iterations, "
              f"{result.accuracy_improvement:.2f}% improvement")
    
    # Step 8: Metric correlation analysis
    print("\n\nStep 8: Metric correlation analysis...")
    correlation = api.get_metric_correlation("stock_price", "trading_volume")
    print(f"\nCorrelation between stock_price and trading_volume: {correlation:.3f}")
    
    # Step 9: Get agent knowledge
    print("\n\nStep 9: Agent knowledge summary...")
    for agent_id in ["market_analyst_1", "market_analyst_2", "market_analyst_3"]:
        knowledge = api.get_agent_knowledge(agent_id)
        print(f"\nAgent {agent_id}:")
        print(f"  Total Patterns: {knowledge['total_patterns']}")
        print(f"  Learning Sessions: {knowledge['learning_sessions']}")
        print(f"  Total Iterations: {knowledge['total_iterations']}")
    
    # Step 10: Get analysis history
    print("\n\nStep 10: Analysis history...")
    history = api.get_analysis_history(limit=5)
    print(f"\nRecent analysis history ({len(history)} entries):")
    for entry in history:
        print(f"  {entry.timestamp.strftime('%H:%M:%S')} - "
              f"Agent: {entry.agent_id}, "
              f"Metrics: {len(entry.metrics_analyzed)}, "
              f"Confidence: {entry.confidence_score:.2f}")
    
    # Cleanup
    print("\n\nShutting down API...")
    api.shutdown()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
