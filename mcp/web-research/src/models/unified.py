"""Unified data models for search and scrape results"""

from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum


class ScrapingMethod(str, Enum):
    HTTPX = "httpx"
    CRAWL4AI = "crawl4ai"
    SELENIUM = "selenium"
    REDDIT_API = "reddit_api"
    PDF = "pdf"
    BLACKLISTED = "blacklisted"


class SearchResult(BaseModel):
    """Unified search result format"""
    title: str
    url: str
    snippet: str
    domain: str
    score: float = 0.0  # SearXNG native engine consensus score


class CombinedSearchResponse(BaseModel):
    """Combined search from multiple engines"""
    query: str
    total_results: int
    pages_scraped: int
    results: List[SearchResult]
    engines: dict[str, int]  # {"searxng": N} where N is results per engine
    search_time_ms: float
    cached: bool = False  # True if results came from cache


class ScrapeRequest(BaseModel):
    url: HttpUrl  # Validates URL format
    force_method: Optional[ScrapingMethod] = None
    css_selector: Optional[str] = Field(None, description="CSS selector for targeted content extraction")
    text_only: bool = Field(False, description="Disable images for faster loading")

    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        """Convert HttpUrl to string for compatibility"""
        return str(v)


class ScrapeResponse(BaseModel):
    """Unified scrape response"""
    success: bool
    url: str
    domain: str
    method_used: ScrapingMethod
    title: Optional[str] = None
    content: Optional[str] = None  # Unified markdown format
    summary: Optional[str] = None  # AI-generated summary (currently unused)
    metadata: dict = {}
    error: Optional[str] = None
    cached: bool = False  # True if result came from cache


class DomainRecord(BaseModel):
    """Database record for domain tracking"""
    domain: str
    preferred_method: ScrapingMethod
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    failure_count: int = 0
    is_blacklisted: bool = False
