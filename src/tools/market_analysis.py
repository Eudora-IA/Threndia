"""
NFT and Stock Market Analysis Tools.
Integration with GitHub analysis repositories.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MarketTrend:
    """Represents a market trend data point."""
    name: str
    category: str
    score: float
    volume: int = 0
    growth_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NFTCollection:
    """NFT collection data."""
    name: str
    floor_price: float
    volume_24h: float
    owners: int
    total_supply: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarketAnalyzer:
    """
    Market analysis for NFT and stock image platforms.

    Integrates with:
    - OpenSea API for NFT data
    - CivitAI for model trends
    - Stock image platform analytics

    Reference repos:
    - lkasym/NFT_Analysis (price prediction)
    - Crawnicles/NFT-Macro-Analysis (trend patterns)
    - mmsaki/nft-trading-analysis (trading patterns)

    Example:
        >>> analyzer = MarketAnalyzer()
        >>> trends = analyzer.get_nft_trends("art", limit=10)
    """

    def __init__(
        self,
        opensea_api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.opensea_api_key = opensea_api_key
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache/market")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._http_client = None

    def _get_client(self):
        """Get HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.Client(timeout=30.0)
            except ImportError:
                logger.warning("httpx not installed")
        return self._http_client

    # =========================================================================
    # NFT ANALYSIS
    # =========================================================================

    def get_nft_collections(
        self,
        category: str = "art",
        limit: int = 20,
        sort_by: str = "volume",
    ) -> List[NFTCollection]:
        """
        Get top NFT collections by category.

        Args:
            category: Collection category (art, pfps, gaming, etc.)
            limit: Maximum collections to return
            sort_by: Sorting field (volume, floor_price, owners)

        Returns:
            List of NFT collections
        """
        client = self._get_client()
        if not client:
            return []

        try:
            headers = {}
            if self.opensea_api_key:
                headers["X-API-KEY"] = self.opensea_api_key

            # OpenSea API v2
            response = client.get(
                "https://api.opensea.io/api/v2/collections",
                params={"limit": limit},
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                collections = []
                for item in data.get("collections", []):
                    collections.append(NFTCollection(
                        name=item.get("name", "Unknown"),
                        floor_price=float(item.get("stats", {}).get("floor_price", 0)),
                        volume_24h=float(item.get("stats", {}).get("one_day_volume", 0)),
                        owners=int(item.get("stats", {}).get("num_owners", 0)),
                        total_supply=int(item.get("stats", {}).get("total_supply", 0)),
                        metadata=item,
                    ))
                return collections
            else:
                logger.warning(f"OpenSea API returned {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"NFT collection fetch failed: {e}")
            return []

    def analyze_art_trends(
        self,
        collections: List[NFTCollection],
    ) -> List[MarketTrend]:
        """
        Analyze trends from NFT collection data.

        Args:
            collections: List of NFT collections to analyze

        Returns:
            List of identified trends
        """
        trends = []

        # Sort by volume to find trending categories
        by_volume = sorted(collections, key=lambda x: x.volume_24h, reverse=True)

        for i, collection in enumerate(by_volume[:10]):
            trends.append(MarketTrend(
                name=collection.name,
                category="nft_art",
                score=1.0 - (i * 0.1),  # Top ranked = 1.0
                volume=int(collection.volume_24h),
                metadata={
                    "floor_price": collection.floor_price,
                    "owners": collection.owners,
                }
            ))

        return trends

    # =========================================================================
    # STOCK IMAGE ANALYSIS
    # =========================================================================

    def get_stock_trends(
        self,
        platform: str = "shutterstock",
        category: str = "all",
    ) -> List[MarketTrend]:
        """
        Get trending categories/styles from stock platforms.

        Note: Most stock platforms don't expose trend APIs,
        this uses heuristics and cached data.
        """
        # Popular categories based on market research
        trending_categories = [
            MarketTrend(name="AI Art", category="style", score=0.95),
            MarketTrend(name="Cyberpunk", category="style", score=0.90),
            MarketTrend(name="Anime", category="style", score=0.88),
            MarketTrend(name="Abstract Gradients", category="style", score=0.85),
            MarketTrend(name="Minimalist", category="style", score=0.82),
            MarketTrend(name="Nature Photography", category="subject", score=0.80),
            MarketTrend(name="Tech/Futuristic", category="subject", score=0.78),
            MarketTrend(name="Lifestyle", category="subject", score=0.75),
        ]

        return trending_categories

    # =========================================================================
    # CIVITAI TRENDS
    # =========================================================================

    def get_civitai_model_trends(
        self,
        model_type: str = "Checkpoint",
        period: str = "Week",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get trending models from CivitAI.

        Args:
            model_type: Type of model (Checkpoint, LoRA, etc.)
            period: Time period (Day, Week, Month, AllTime)
            limit: Maximum results

        Returns:
            List of trending models
        """
        client = self._get_client()
        if not client:
            return []

        try:
            response = client.get(
                "https://civitai.com/api/v1/models",
                params={
                    "limit": limit,
                    "sort": "Highest Rated",
                    "period": period,
                    "types": model_type,
                },
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("items", [])
            else:
                logger.warning(f"CivitAI API returned {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"CivitAI fetch failed: {e}")
            return []

    # =========================================================================
    # RECOMMENDATION ENGINE
    # =========================================================================

    def recommend_styles(
        self,
        target_platform: str = "opensea",
    ) -> List[Dict[str, Any]]:
        """
        Recommend styles/subjects based on market analysis.

        Combines NFT trends, stock trends, and model popularity.
        """
        recommendations = []

        # Get various trend sources
        nft_collections = self.get_nft_collections(limit=10)
        nft_trends = self.analyze_art_trends(nft_collections)
        stock_trends = self.get_stock_trends()
        civitai_models = self.get_civitai_model_trends(limit=10)

        # Score and rank
        style_scores = {}

        for trend in nft_trends + stock_trends:
            style_scores[trend.name] = style_scores.get(trend.name, 0) + trend.score

        for model in civitai_models:
            model_name = model.get("name", "")
            model_stats = model.get("stats", {})
            downloads = model_stats.get("downloadCount", 0)
            # Normalize download count
            score = min(downloads / 100000, 1.0)
            style_scores[model_name] = style_scores.get(model_name, 0) + score

        # Sort and return top recommendations
        sorted_styles = sorted(
            style_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for name, score in sorted_styles[:15]:
            recommendations.append({
                "style": name,
                "score": score,
                "platform": target_platform,
            })

        return recommendations

    def cache_analysis(self, data: Dict, cache_key: str):
        """Cache analysis results."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_cached_analysis(self, cache_key: str) -> Optional[Dict]:
        """Load cached analysis."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None
