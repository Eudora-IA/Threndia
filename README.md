# Threndia

**Thread Analysis API for Market Analysis and Agent Self-Learning**

Threndia is a powerful Python framework for market analysis and agent creation, featuring concurrent thread-based analysis and self-learning capabilities.

## Features

- 🧵 **Thread-Based Analysis**: Concurrent execution of multiple analysis tasks
- 🤖 **Agent Self-Learning**: Agents that learn from market patterns and improve over time
- 📊 **Multi-Metric Analysis**: Analyze multiple market metrics simultaneously
- 🔄 **Continuous Learning**: Automated learning cycles for agent improvement
- 📈 **Pattern Recognition**: Identify trends, volatility, and correlations
- ⚡ **Asynchronous Operations**: High-performance async/await support

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import asyncio
from threndia import ThreadAnalysisAPI
from threndia.models.data_models import MetricData

async def main():
    # Initialize API
    api = ThreadAnalysisAPI(max_concurrent_threads=10)
    
    # Create agents
    agent = api.create_agent("analyst_1", learning_rate=0.05)
    
    # Add market data
    api.add_metric_data(MetricData(
        metric_name="stock_price",
        value=100.0
    ))
    
    # Perform analysis
    results = await api.threaded_analysis(
        agent_id="analyst_1",
        metric_names=["stock_price"]
    )
    
    print(f"Confidence: {results.confidence_score}")
    
    # Agent self-learning
    learning = api.agent_self_learning(
        agent_id="analyst_1",
        metric_names=["stock_price"]
    )
    
    api.shutdown()

asyncio.run(main())
```

## API Components

### ThreadAnalysisAPI

Main API for managing agents and analysis:
- `create_agent(agent_id, learning_rate)` - Create a new agent
- `add_metric_data(metric_data)` - Add market metric data
- `analyze_metrics(metric_names)` - Analyze specific metrics
- `threaded_analysis(agent_id, metric_names)` - Perform threaded analysis
- `multi_agent_analysis(agent_ids, metric_names)` - Concurrent multi-agent analysis
- `agent_self_learning(agent_id, metric_names, actual_outcomes)` - Trigger agent learning
- `continuous_learning_cycle(agent_id, metric_names, cycles)` - Run learning cycles
- `get_metric_correlation(metric1, metric2)` - Calculate metric correlation

### Agent

Self-learning agent for market analysis:
- `analyze_metrics(metric_results)` - Analyze metrics and generate insights
- `learn_from_analysis(metric_results, actual_outcomes)` - Self-learning from data
- `get_knowledge_summary()` - Get agent's knowledge base summary

### MetricAnalyzer

Analyzes market metrics:
- `add_metric_data(metric_data)` - Add data points
- `analyze_metric(metric_name)` - Analyze specific metric
- `analyze_all_metrics()` - Analyze all available metrics
- `get_correlation(metric1, metric2)` - Calculate correlation

### ThreadManager

Manages concurrent thread execution:
- `submit_task(thread_id, task)` - Submit task for execution
- `get_thread_status(thread_id)` - Get thread status
- `wait_for_all()` - Wait for all threads to complete
- `run_concurrent_analyses(tasks, config)` - Run tasks concurrently

## Example Usage

See `examples/basic_usage.py` for a comprehensive example demonstrating:
1. Agent creation
2. Metric data ingestion
3. Single and multi-agent analysis
4. Self-learning with feedback
5. Continuous learning cycles
6. Correlation analysis
7. Knowledge tracking

Run the example:
```bash
python examples/basic_usage.py
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Use Cases

- **Market Analysis**: Analyze stock prices, trading volumes, and market indicators
- **Agent Training**: Train multiple agents to recognize market patterns
- **Concurrent Processing**: Perform parallel analysis across different metrics
- **Self-Improving Systems**: Build agents that learn and adapt over time
- **Pattern Recognition**: Identify trends, correlations, and anomalies

## Architecture

```
threndia/
├── api/
│   └── thread_analysis.py    # Main API
├── core/
│   ├── agent.py               # Agent with self-learning
│   ├── metrics.py             # Metric analysis
│   └── thread_manager.py      # Thread management
└── models/
    └── data_models.py         # Data models
```

## License

See LICENSE file for details.
