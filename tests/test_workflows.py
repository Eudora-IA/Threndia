from unittest.mock import MagicMock, patch

import pytest
from src.pipelines.trend_research_workflow import (
    analyze_trends,
    fetch_trends,
    store_insights,
)


@pytest.fixture
def mock_trend_client():
    with patch('src.pipelines.trend_research_workflow.TrendRadarClient') as mock:
        client = Mock()
        trend = Mock()
        trend.title = "A"
        trend.platform = "B"
        trend.rank = 1
        client.get_hot_trends.return_value = [trend]
        mock.return_value = client
        yield mock

def test_fetch_trends_node():
    with patch('src.pipelines.trend_research_workflow.TrendRadarClient') as MockClient:
        instance = MockClient.return_value
        instance.get_hot_trends.return_value = [
            MagicMock(title="T1", platform="P1", rank=1)
        ]

        state = {"keywords": [], "platforms": []}
        result = fetch_trends(state)

        assert "raw_trends" in result
        assert len(result["raw_trends"]) == 1
        assert result["raw_trends"][0]["title"] == "T1"

@patch('src.pipelines.trend_research_workflow.LLMManager')
def test_analyze_trends_node(MockLLM):
    llm = MockLLM.return_value
    llm.chat.return_value.content = "Analysis Report"

    state = {
        "raw_trends": [{"title": "T1", "platform": "P1", "rank": 1}],
        "keywords": [],
        "platforms": []
    }

    result = analyze_trends(state)

    assert result["analysis_report"] == "Analysis Report"
    assert "messages" in result

@patch('src.core.vector_database.EnhancedChromaStore')
def test_store_insights_node(MockStore):
    store = MockStore.return_value
    store.add_text.return_value = "doc_id_123"

    state = {
        "analysis_report": "Report",
        "platforms": ["P1"],
        "raw_trends": [{"title": "T1", "platform": "P1", "rank": 1}]
    }

    result = store_insights(state)

    assert store.add_text.call_count >= 1
