# Contributing to Threndia Pairing

## Overview
Thank you for contributing! This document outlines the process for contributing to the Market Analysis tools.

## Branch Strategy
- **Main Branch**: `threndia-pairing` (synced with Threndia)
- **Feature Branches**: `feature/your-feature-name`

## Code Standards

### Scrapers
All scrapers must:
1. Inherit from `BaseScraper`
2. Implement `scrape()` and `parse()` methods
3. Respect rate limits

```python
from src.scrapers import BaseScraper

class MyScraper(BaseScraper):
    def scrape(self, target: str, **kwargs):
        # Your scraping logic
        pass

    def parse(self, raw_data):
        # Your parsing logic
        return []
```

### Signals
Use `MarketSignal` for all signal data:

```python
from src.market_analysis import MarketSignal, SignalType

signal = MarketSignal(
    id="unique-id",
    signal_type=SignalType.TREND_RISING,
    source="my-scraper",
    keyword="trending-topic",
    confidence=0.85
)
```

## Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure linting passes (`ruff check`)
5. Submit PR with clear description

## Questions?
Open an issue or contact the maintainers.
