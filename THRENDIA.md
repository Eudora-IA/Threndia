# Threndia Cooperation Guide

## Overview

This branch (`threndia-pairing`) is dedicated to **Market Analysis** and **Web Scraping** tools that cooperate with the [Threndia](https://github.com/Eudora-IA/Threndia) repository.

## Architecture

```
src/
├── market_analysis/     # Signal processing and trend detection
│   ├── signals.py       # MarketSignal schema (Pydantic)
│   └── trendradar.py    # Main processor class
├── database/            # Data persistence
│   └── chroma_store.py  # ChromaDB wrapper for signals
└── scrapers/            # Web scraping tools
    └── base_scraper.py  # Abstract scraper class
```

## Key Components

### MarketSignal Schema
Standardized format for market signals:
- `signal_type`: Type of signal (trend, sentiment, volume)
- `source`: Data source name
- `keyword`: Detected keyword/topic
- `confidence`: Score 0-1

### ChromaMarketStore
ChromaDB wrapper for:
- Storing signals with embeddings
- Semantic search for similar signals
- Filtering by source/type

### TrendRadar
Main processor that:
- Converts raw scraped data into signals
- Caches signals for batch processing
- Integrates with ChromaDB

## Synchronization

This branch is automatically synced with Threndia via:
1. **GitHub Actions**: `.github/workflows/threndia_sync.yml`
2. **Manual Script**: `python scripts/sync_threndia.py --push`

## Getting Started

```bash
# Install dependencies
pip install pydantic chromadb

# Run signal collection
python -m src.market_analysis.trendradar

# Search signals
python -c "from src.database import ChromaMarketStore; s = ChromaMarketStore(); print(s.count_signals())"
```

## Contributing

1. Create a feature branch from `threndia-pairing`
2. Follow the `BaseScraper` pattern for new scrapers
3. Use `MarketSignal` schema for all signal data
4. Submit PR to this branch
