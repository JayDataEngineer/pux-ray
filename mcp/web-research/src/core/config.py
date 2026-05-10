"""Configuration for MCP Server using Pydantic Settings"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import (
    BLACKLIST_FAILURE_THRESHOLD,
    DEFAULT_SEARCH_ENGINES,
    CELERY_WORKER_CONCURRENCY,
)


class Settings(BaseSettings):
    """Application settings with automatic environment variable loading"""

    # API
    host: str = "0.0.0.0"
    port: int = 8000

    # External Services
    searxng_url: str = "http://searxng:8080"

    # Database
    db_path: str = "/app/data/mcp_server.db"

    # Domain tracking
    known_waf_domains: List[str] = [
        "stackoverflow.com",
        "reddit.com",
        "linkedin.com",
        "twitter.com",
        "facebook.com",
        "instagram.com",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
