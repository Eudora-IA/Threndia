import logging
from typing import Dict, List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from src.core.llm_provider import LLMManager
from src.integrations.trendradar import TrendRadarClient

logger = logging.getLogger(__name__)

# Define State
class ResearchState(TypedDict):
    """State for the trend research workflow."""
    keywords: List[str]
    platforms: List[str]
    raw_trends: List[Dict]
    analysis_report: str
    messages: List[BaseMessage]

# Node: Fetch Trends
def fetch_trends(state: ResearchState) -> Dict:
    """Fetches hot trends from TrendRadar."""
    logger.info("Fetching trends from TrendRadar...")
    client = TrendRadarClient()

    # Use keywords/platforms from state or defaults
    keywords = state.get("keywords", [])
    platforms = state.get("platforms", ["zhihu", "weibo", "bilibili"])

    trends = client.get_hot_trends(platforms=platforms, keywords=keywords, limit=10)

    # Convert dataclass to dict for state
    trends_data = [
        {"title": t.title, "platform": t.platform, "rank": t.rank}
        for t in trends
    ]

    return {"raw_trends": trends_data}

# Node: Analyze Trends
def analyze_trends(state: ResearchState) -> Dict:
    """Analyzes trends using the LLM to identify opportunities."""
    logger.info("Analyzing trends with LLM...")
    llm = LLMManager()
    trends = state["raw_trends"]

    prompt = f"""
    Analyze the following social media trends and identify top 3 opportunities for creating visual stock assets (images/videos).

    Trends:
    {trends}

    Output Format:
    1. Opportunity Title
    2. Visual Description (Prompt concept)
    3. Target Audience
    """

    response = llm.chat([
        SystemMessage(content="You are a Market Research Agent specialized in visual asset trends."),
        HumanMessage(content=prompt)
    ])

    return {
        "analysis_report": response.content,
        "messages": [HumanMessage(content=prompt), response]
    }

# Node: Store Insights
def store_insights(state: ResearchState) -> Dict:
    """Stores the analysis report and raw trends into ChromaDB."""
    logger.info("Storing insights to ChromaDB...")
    from src.core.vector_database import EnhancedChromaStore

    store = EnhancedChromaStore(persist_directory="./data/chroma_db")

    # Store the analysis report
    report_id = store.add_text(
        text=state["analysis_report"],
        metadata={
            "type": "trend_analysis_report",
            "platforms": str(state["platforms"]),
            "timestamp": "now" # In real app, use actual timestamp
        }
    )

    # Store individual high-ranking trends
    raw_trends = state["raw_trends"]
    stored_count = 0
    for trend in raw_trends:
        if trend.get('rank', 100) <= 3: # Only store top 3 trends per platform
            store.add_text(
                text=f"Trend: {trend['title']} (Platform: {trend['platform']})",
                metadata={
                    "type": "raw_trend",
                    "platform": trend['platform'],
                    "rank": trend['rank'],
                    "report_id": report_id
                }
            )
            stored_count += 1

    logger.info(f"Stored report {report_id} and {stored_count} top trends.")
    return {}

# Build Graph
def build_trend_research_workflow():
    workflow = StateGraph(ResearchState)

    workflow.add_node("fetch_trends", fetch_trends)
    workflow.add_node("analyze_trends", analyze_trends)
    workflow.add_node("store_insights", store_insights)

    workflow.set_entry_point("fetch_trends")
    workflow.add_edge("fetch_trends", "analyze_trends")
    workflow.add_edge("analyze_trends", "store_insights")
    workflow.add_edge("store_insights", END)

    return workflow.compile()

if __name__ == "__main__":
    # Test Run
    import asyncio

    async def run_test():
        print("Starting Trend Research Workflow...")
        app = build_trend_research_workflow()

        # Determine if we can run (needs LLM/TrendRadar configured)
        print("Workflow compiled successfully.")
        print(app.get_graph().draw_ascii())

    asyncio.run(run_test())
