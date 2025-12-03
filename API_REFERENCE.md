# Thread Analysis API Reference

## Overview

The Thread Analysis API provides a comprehensive framework for market analysis with agent self-learning capabilities, supporting concurrent analysis across multiple metrics.

## Classes

### ThreadAnalysisAPI

Main entry point for the API.

#### Constructor
```python
ThreadAnalysisAPI(max_concurrent_threads: int = 10)
```

#### Methods

##### Agent Management
- `create_agent(agent_id: str, learning_rate: float = 0.01) -> Agent`
  - Creates a new agent for analysis
  - **Parameters:**
    - `agent_id`: Unique identifier for the agent
    - `learning_rate`: Learning rate for self-learning (0.0 to 1.0)
  - **Returns:** Agent instance

- `get_agent(agent_id: str) -> Optional[Agent]`
  - Retrieves an existing agent
  - **Returns:** Agent instance or None

##### Metric Operations
- `add_metric_data(metric_data: MetricData) -> None`
  - Adds a single metric data point

- `add_multiple_metrics(metrics: List[MetricData]) -> None`
  - Adds multiple metric data points

- `analyze_metrics(metric_names: Optional[List[str]] = None) -> List[MetricResult]`
  - Analyzes market metrics
  - **Parameters:**
    - `metric_names`: Optional list of specific metrics to analyze
  - **Returns:** List of MetricResult objects

##### Analysis Operations
- `async threaded_analysis(agent_id: str, metric_names: List[str], thread_config: Optional[ThreadConfig] = None) -> AnalysisResult`
  - Performs threaded analysis with an agent
  - **Parameters:**
    - `agent_id`: Agent to perform analysis
    - `metric_names`: List of metrics to analyze
    - `thread_config`: Optional thread configuration
  - **Returns:** AnalysisResult with insights

- `async multi_agent_analysis(agent_ids: List[str], metric_names: List[str], thread_config: Optional[ThreadConfig] = None) -> List[AnalysisResult]`
  - Performs concurrent analysis with multiple agents
  - **Returns:** List of AnalysisResult from each agent

##### Learning Operations
- `agent_self_learning(agent_id: str, metric_names: Optional[List[str]] = None, actual_outcomes: Optional[Dict[str, float]] = None) -> LearningResult`
  - Triggers agent self-learning from analysis
  - **Parameters:**
    - `agent_id`: Agent to perform learning
    - `metric_names`: Optional specific metrics to learn from
    - `actual_outcomes`: Optional actual outcomes for supervised learning
  - **Returns:** LearningResult with learning details

- `async continuous_learning_cycle(agent_id: str, metric_names: List[str], cycles: int = 5, interval_seconds: float = 1.0) -> List[LearningResult]`
  - Runs continuous learning cycles for an agent
  - **Parameters:**
    - `cycles`: Number of learning cycles
    - `interval_seconds`: Time between cycles
  - **Returns:** List of LearningResult from each cycle

##### Utility Operations
- `get_metric_correlation(metric1: str, metric2: str) -> Optional[float]`
  - Calculates correlation between two metrics
  - **Returns:** Correlation coefficient or None

- `get_analysis_history(agent_id: Optional[str] = None, limit: int = 10) -> List[AnalysisResult]`
  - Retrieves analysis history
  - **Parameters:**
    - `agent_id`: Optional filter by agent
    - `limit`: Maximum number of results

- `get_agent_knowledge(agent_id: str) -> Dict[str, Any]`
  - Gets knowledge summary for an agent

- `shutdown() -> None`
  - Shuts down the API and cleanup resources

### Agent

Self-learning agent for market analysis.

#### Constructor
```python
Agent(agent_id: str, learning_rate: float = 0.01)
```

#### Methods
- `analyze_metrics(metric_results: List[MetricResult]) -> Dict[str, Any]`
  - Analyzes metrics and generates insights

- `learn_from_analysis(metric_results: List[MetricResult], actual_outcomes: Optional[Dict[str, float]] = None) -> AgentLearningResult`
  - Performs self-learning based on metric analysis

- `get_knowledge_summary() -> Dict[str, Any]`
  - Returns summary of agent's knowledge base

### MetricAnalyzer

Analyzes market metrics for patterns and trends.

#### Constructor
```python
MetricAnalyzer(window_size: int = 20)
```

#### Class Attributes
- `UPWARD_TREND_THRESHOLD = 0.01`: Threshold for upward trend detection
- `DOWNWARD_TREND_THRESHOLD = -0.01`: Threshold for downward trend detection

#### Methods
- `add_metric_data(metric_data: MetricData) -> None`
  - Adds new metric data point

- `analyze_metric(metric_name: str) -> Optional[MetricResult]`
  - Analyzes a specific metric

- `analyze_all_metrics() -> List[MetricResult]`
  - Analyzes all available metrics

- `get_correlation(metric1: str, metric2: str) -> Optional[float]`
  - Calculates correlation between two metrics

### ThreadManager

Manages concurrent thread execution for analysis tasks.

#### Constructor
```python
ThreadManager(max_workers: int = 5)
```

#### Methods
- `submit_task(thread_id: str, task: Callable, *args, **kwargs) -> None`
  - Submits a task for execution

- `get_thread_status(thread_id: str) -> Optional[str]`
  - Gets the status of a thread

- `get_result(thread_id: str, timeout: Optional[float] = None) -> Any`
  - Gets the result of a completed thread

- `wait_for_all(timeout: Optional[float] = None) -> Dict[str, Any]`
  - Waits for all active threads to complete

- `async submit_async_task(thread_id: str, task: Callable, *args, **kwargs) -> Any`
  - Submits and awaits an async task

- `async run_concurrent_analyses(tasks: List[tuple], config: Optional[ThreadConfig] = None) -> List[Any]`
  - Runs multiple analysis tasks concurrently

- `shutdown(wait: bool = True) -> None`
  - Shuts down the thread executor

## Data Models

### MetricData
```python
class MetricData(BaseModel):
    metric_name: str
    value: float
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### AnalysisResult
```python
class AnalysisResult(BaseModel):
    thread_id: str
    agent_id: str
    metrics_analyzed: List[str]
    insights: Dict[str, Any]
    confidence_score: float  # 0.0 to 1.0
    timestamp: datetime = Field(default_factory=datetime.now)
```

### ThreadConfig
```python
class ThreadConfig(BaseModel):
    thread_id: str
    max_concurrent_analyses: int = 5
    timeout_seconds: int = 300
    retry_attempts: int = 3
    metrics_to_analyze: List[str] = Field(default_factory=list)
```

### LearningResult
```python
class LearningResult(BaseModel):
    agent_id: str
    learning_type: str
    learned_patterns: Dict[str, Any]
    accuracy_improvement: float
    training_iterations: int
    timestamp: datetime = Field(default_factory=datetime.now)
```

## Example Workflow

```python
import asyncio
from threndia import ThreadAnalysisAPI
from threndia.models.data_models import MetricData

async def analyze_market():
    # 1. Initialize API
    api = ThreadAnalysisAPI(max_concurrent_threads=10)
    
    # 2. Create agents
    api.create_agent("analyst_1", learning_rate=0.05)
    api.create_agent("analyst_2", learning_rate=0.03)
    
    # 3. Add market data
    for i in range(30):
        api.add_metric_data(MetricData(
            metric_name="stock_price",
            value=100.0 + i * 0.5
        ))
    
    # 4. Multi-agent analysis
    results = await api.multi_agent_analysis(
        agent_ids=["analyst_1", "analyst_2"],
        metric_names=["stock_price"]
    )
    
    # 5. Agent learning
    for agent_id in ["analyst_1", "analyst_2"]:
        learning_result = api.agent_self_learning(
            agent_id=agent_id,
            metric_names=["stock_price"]
        )
        print(f"{agent_id}: {learning_result.accuracy_improvement}% improvement")
    
    # 6. Cleanup
    api.shutdown()

asyncio.run(analyze_market())
```

## Configuration

### Thread Analysis API Constants
- `MIN_DATA_POINTS_FOR_FULL_CONFIDENCE = 20.0`: Minimum data points for full confidence score

### Metric Analyzer Constants
- `UPWARD_TREND_THRESHOLD = 0.01`: Slope threshold for upward trend
- `DOWNWARD_TREND_THRESHOLD = -0.01`: Slope threshold for downward trend

## Best Practices

1. **Data Quality**: Ensure sufficient data points (20+) for reliable analysis
2. **Learning Rate**: Use lower learning rates (0.01-0.05) for stable learning
3. **Concurrent Operations**: Limit concurrent threads based on system resources
4. **Error Handling**: Always handle potential None returns from analysis methods
5. **Resource Cleanup**: Call `shutdown()` when done to properly cleanup resources
6. **Supervised Learning**: Provide `actual_outcomes` when available for better learning
7. **Continuous Monitoring**: Use continuous learning cycles for ongoing improvement
