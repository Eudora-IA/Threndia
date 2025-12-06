"""
Base Scraper - Abstract class for web scrapers.
All scrapers should inherit from this class.
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class BaseScraper(ABC):
    """
    Abstract base class for web scrapers.

    Subclasses should implement:
    - scrape(): Main scraping logic
    - parse(): Data parsing logic
    """

    def __init__(self, name: str, rate_limit: float = 1.0):
        """
        Initialize the scraper.

        Args:
            name: Scraper name for logging.
            rate_limit: Minimum seconds between requests.
        """
        self.name = name
        self.rate_limit = rate_limit
        self._last_request: Optional[datetime] = None

    @abstractmethod
    def scrape(self, target: str, **kwargs) -> Dict[str, Any]:
        """
        Perform the scraping operation.

        Args:
            target: URL or search term.
            **kwargs: Additional parameters.

        Returns:
            Raw scraped data.
        """
        pass

    @abstractmethod
    def parse(self, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse raw data into structured format.

        Args:
            raw_data: Data from scrape().

        Returns:
            List of parsed items.
        """
        pass

    def run(self, target: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute full scrape + parse pipeline.

        Args:
            target: Scraping target.
            **kwargs: Additional parameters.

        Returns:
            Parsed results.
        """
        logger.info(f"[{self.name}] Starting scrape for: {target}")

        try:
            raw_data = self.scrape(target, **kwargs)
            results = self.parse(raw_data)
            logger.info(f"[{self.name}] Scraped {len(results)} items")
            return results
        except Exception as e:
            logger.error(f"[{self.name}] Scrape failed: {e}")
            return []
