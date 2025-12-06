"""
TrendRadar integration for multi-platform trend monitoring.
Connects to sansan0/TrendRadar for Chinese social media and news aggregation.

GitHub: https://github.com/sansan0/TrendRadar
"""
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class TrendRadarPlatform(Enum):
    """Supported monitoring platforms."""
    ZHIHU = "zhihu"           # 知乎
    DOUYIN = "douyin"         # 抖音
    BILIBILI = "bilibili"     # B站热搜
    WEIBO = "weibo"           # 微博
    BAIDU = "baidu"           # 百度热搜
    TIEBA = "tieba"           # 贴吧
    TOUTIAO = "toutiao"       # 今日头条
    WALLSTREETCN = "wallstreetcn"  # 华尔街见闻
    CAILIANSHE = "cailianshe"     # 财联社
    PENGPAI = "pengpai"       # 澎湃新闻
    IFENG = "ifeng"           # 凤凰网


class PushMode(Enum):
    """Push notification modes."""
    INCREMENTAL = "incremental"  # Only new items
    CURRENT = "current"          # Current full list
    DAILY = "daily"              # Daily digest


@dataclass
class TrendItem:
    """A single trending item."""
    title: str
    platform: str
    rank: int
    url: Optional[str] = None
    heat_score: float = 0.0
    is_new: bool = False
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Analysis result from TrendRadar MCP."""
    query: str
    results: List[Dict[str, Any]]
    analysis_type: str
    timestamp: str
    summary: Optional[str] = None


class TrendRadarClient:
    """
    Client for TrendRadar trend monitoring and MCP analysis.

    Features:
    - 35+ platform monitoring (Chinese social media/news)
    - MCP-based AI analysis (13 tools)
    - Trend tracking and sentiment analysis
    - Cross-platform comparison

    Example:
        >>> client = TrendRadarClient(mcp_url="http://localhost:8080")
        >>> trends = client.get_hot_trends(platforms=["zhihu", "weibo"])
        >>> analysis = client.analyze("分析比特币最近的热度趋势")

    Setup:
        Docker deployment recommended:
        ```bash
        docker pull sansan0/trend-radar
        docker pull sansan0/trend-radar-mcp
        docker-compose up -d
        ```
    """

    def __init__(
        self,
        mcp_url: str = "http://localhost:8080",
        data_dir: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        self.mcp_url = mcp_url.rstrip("/")
        self.data_dir = Path(data_dir) if data_dir else Path("./data/trendradar")
        self.config = config or {}
        self._http_client = None
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_client(self):
        """Get HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=60.0)
        return self._http_client

    # =========================================================================
    # TREND FETCHING
    # =========================================================================

    def get_hot_trends(
        self,
        platforms: Optional[List[str]] = None,
        limit: int = 20,
        keywords: Optional[List[str]] = None,
    ) -> List[TrendItem]:
        """
        Get hot trends from specified platforms.

        Args:
            platforms: List of platforms to monitor (default: all)
            limit: Maximum items per platform
            keywords: Filter by keywords

        Returns:
            List of trending items
        """
        client = self._get_client()
        if not client:
            return self._load_cached_trends()

        if platforms is None:
            platforms = [p.value for p in TrendRadarPlatform]

        try:
            # Try MCP endpoint first
            response = client.post(
                f"{self.mcp_url}/api/trends",
                json={
                    "platforms": platforms,
                    "limit": limit,
                    "keywords": keywords or [],
                }
            )

            if response.status_code == 200:
                data = response.json()
                trends = []
                for item in data.get("items", []):
                    trends.append(TrendItem(
                        title=item.get("title", ""),
                        platform=item.get("platform", "unknown"),
                        rank=item.get("rank", 0),
                        url=item.get("url"),
                        heat_score=item.get("heat_score", 0.0),
                        is_new=item.get("is_new", False),
                        first_seen=item.get("first_seen"),
                        last_seen=item.get("last_seen"),
                        metadata=item,
                    ))
                self._cache_trends(trends)
                return trends

        except Exception as e:
            logger.warning(f"MCP fetch failed: {e}, using cached data")

        return self._load_cached_trends()

    def get_platform_trends(
        self,
        platform: str,
        limit: int = 20,
    ) -> List[TrendItem]:
        """Get trends from a specific platform."""
        return self.get_hot_trends(platforms=[platform], limit=limit)

    # =========================================================================
    # MCP AI ANALYSIS (13 Tools)
    # =========================================================================

    def analyze(self, query: str) -> TrendAnalysis:
        """
        Natural language analysis using MCP tools.

        Supported queries:
        - "查询昨天知乎的热点" (Query yesterday's Zhihu trends)
        - "分析比特币最近的热度趋势" (Analyze Bitcoin's recent trend)
        - "对比各平台对AI的关注度" (Compare AI attention across platforms)

        Args:
            query: Natural language query

        Returns:
            TrendAnalysis with results
        """
        client = self._get_client()
        if not client:
            return TrendAnalysis(
                query=query,
                results=[],
                analysis_type="error",
                timestamp="",
                summary="MCP client not available"
            )

        try:
            response = client.post(
                f"{self.mcp_url}/api/analyze",
                json={"query": query}
            )

            if response.status_code == 200:
                data = response.json()
                return TrendAnalysis(
                    query=query,
                    results=data.get("results", []),
                    analysis_type=data.get("analysis_type", "general"),
                    timestamp=data.get("timestamp", ""),
                    summary=data.get("summary"),
                )

        except Exception as e:
            logger.error(f"Analysis failed: {e}")

        return TrendAnalysis(
            query=query,
            results=[],
            analysis_type="error",
            timestamp="",
            summary="Analysis failed"
        )

    def track_topic_trend(
        self,
        topic: str,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Track a topic's trend over time.

        Analyzes:
        - Heat changes
        - Lifecycle
        - Viral detection
        - Trend prediction
        """
        return self.analyze(f"追踪话题'{topic}'最近{days}天的热度变化")

    def compare_platforms(
        self,
        topic: str,
    ) -> Dict[str, Any]:
        """Compare topic coverage across platforms."""
        return self.analyze(f"对比各平台对'{topic}'的关注度")

    def sentiment_analysis(
        self,
        topic: str,
    ) -> Dict[str, Any]:
        """Analyze sentiment for a topic."""
        return self.analyze(f"分析'{topic}'相关新闻的情感倾向")

    def find_similar_news(
        self,
        topic: str,
        limit: int = 10,
    ) -> List[TrendItem]:
        """Find similar news items."""
        result = self.analyze(f"查找与'{topic}'相似的新闻，限制{limit}条")
        return [
            TrendItem(
                title=item.get("title", ""),
                platform=item.get("platform", "unknown"),
                rank=0,
                metadata=item,
            )
            for item in result.results
        ]

    def get_topic_summary(
        self,
        topic: str,
    ) -> str:
        """Generate AI summary for a topic."""
        result = self.analyze(f"生成'{topic}'相关热点的智能摘要")
        return result.summary or ""

    # =========================================================================
    # KEYWORD CONFIGURATION
    # =========================================================================

    def set_keywords(
        self,
        keywords: List[str],
        must_include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Configure monitoring keywords.

        Syntax:
        - Normal word: basic matching
        - +word: must include
        - !word: exclude
        - @N: limit count

        Example:
            >>> client.set_keywords(
            ...     keywords=["AI", "比特币"],
            ...     must_include=["中国"],
            ...     exclude=["广告"]
            ... )
        """
        keyword_config = []

        for kw in keywords:
            keyword_config.append(kw)

        if must_include:
            for kw in must_include:
                keyword_config.append(f"+{kw}")

        if exclude:
            for kw in exclude:
                keyword_config.append(f"!{kw}")

        config_path = self.data_dir / "frequency_words.txt"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("\n".join(keyword_config))

        logger.info(f"Keywords saved: {keyword_config}")
        return {"keywords": keyword_config, "path": str(config_path)}

    # =========================================================================
    # CACHING
    # =========================================================================

    def _cache_trends(self, trends: List[TrendItem]):
        """Cache trends locally."""
        cache_file = self.data_dir / "trends_cache.json"
        data = [
            {
                "title": t.title,
                "platform": t.platform,
                "rank": t.rank,
                "url": t.url,
                "heat_score": t.heat_score,
                "is_new": t.is_new,
            }
            for t in trends
        ]
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_cached_trends(self) -> List[TrendItem]:
        """Load cached trends."""
        cache_file = self.data_dir / "trends_cache.json"
        if cache_file.exists():
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
            return [
                TrendItem(
                    title=item["title"],
                    platform=item["platform"],
                    rank=item["rank"],
                    url=item.get("url"),
                    heat_score=item.get("heat_score", 0),
                    is_new=item.get("is_new", False),
                )
                for item in data
            ]
        return []

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """Check TrendRadar service health."""
        client = self._get_client()
        if not client:
            return {"status": "error", "message": "HTTP client not available"}

        try:
            response = client.get(f"{self.mcp_url}/health")
            if response.status_code == 200:
                return {"status": "healthy", "data": response.json()}
            return {"status": "unhealthy", "code": response.status_code}
        except Exception as e:
            return {"status": "error", "message": str(e)}


# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def create_trendradar_tools(client: TrendRadarClient) -> List[Dict]:
    """Create LangGraph-compatible tools for TrendRadar."""
    from ..core.llm_provider import ToolDefinition

    return [
        ToolDefinition(
            name="get_hot_trends",
            description="Get current hot/trending topics from Chinese social media and news platforms (Zhihu, Weibo, Douyin, Bilibili, etc.)",
            parameters={
                "type": "object",
                "properties": {
                    "platforms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Platforms to monitor: zhihu, weibo, douyin, bilibili, baidu, etc."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum items",
                        "default": 20
                    }
                }
            },
            handler=lambda platforms=None, limit=20: [
                {"title": t.title, "platform": t.platform, "rank": t.rank}
                for t in client.get_hot_trends(platforms, limit)
            ]
        ),
        ToolDefinition(
            name="analyze_trend",
            description="Analyze trends using natural language query (supports Chinese)",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query about trends"
                    }
                },
                "required": ["query"]
            },
            handler=lambda query: client.analyze(query).__dict__
        ),
        ToolDefinition(
            name="track_topic",
            description="Track a specific topic's trend over time",
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to track"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days",
                        "default": 7
                    }
                },
                "required": ["topic"]
            },
            handler=lambda topic, days=7: client.track_topic_trend(topic, days)
        ),
        ToolDefinition(
            name="sentiment_analysis",
            description="Analyze sentiment for news/trends about a topic",
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic for sentiment analysis"
                    }
                },
                "required": ["topic"]
            },
            handler=lambda topic: client.sentiment_analysis(topic)
        ),
    ]
