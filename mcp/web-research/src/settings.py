"""Application Settings using Pydantic

Centralized configuration management using pydantic_settings.
All environment variables are prefixed with MCP_ for clarity.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables

    All settings can be overridden via environment variables with the MCP_ prefix.
    For example: MCP_REDIS_HOST, MCP_SCRAPE_CACHE_TTL, etc.
    """

    model_config = SettingsConfigDict(
        env_prefix="MCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS Configuration
    allowed_origins: str = "*"  # Comma-separated list

    # Redis Configuration
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: Optional[str] = None

    # Cache TTL (seconds)
    search_cache_ttl: int = 300  # 5 minutes for search results
    scrape_cache_ttl: int = 3600  # 1 hour for scraped content
    docs_cache_ttl: int = 3600  # 1 hour for documentation

    # PostgreSQL Configuration (for domain tracking - will be removed later)
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "mcp_server"
    postgres_user: str = "postgres"
    postgres_password: Optional[str] = None

    # Rate Limiting
    rate_limit_max_concurrent: int = 3
    rate_limit_acquire_timeout: float = 30.0
    rate_limit_ttl: int = 300

    # Documentation Configuration
    docs_config_path: str = "/app/docs_config.yaml"
    docs_local_dir: str = "/app/docs_local"

    # SearXNG Configuration
    searxng_url: str = "http://searxng:8080"

    # HTTP Configuration
    http_request_timeout: float = 30.0

    # Scraping Configuration
    crawl4ai_word_count_threshold: int = 10
    selenium_page_load_wait_seconds: int = 3
    min_content_length: int = 100

    # Retry Configuration
    crawl4ai_retry_count: int = 3
    selenium_retry_count: int = 3

    # Blacklist Threshold
    blacklist_failure_threshold: int = 3

    # Proxy Configuration
    proxy_url: Optional[str] = None  # Single proxy, e.g. socks5://127.0.0.1:1080
    proxy_urls: Optional[str] = None  # Comma-separated list for rotation
    proxy_rotation: str = "round-robin"  # "round-robin" or "random"
    proxy_exclude: str = "searxng,postgres,redis,localhost,127.0.0.1"  # Hostnames to bypass

    # Vision Model Configuration (Florence-2)
    vision_enabled: bool = True
    vision_model: str = "microsoft/Florence-2-base"
    vision_device: str = "cpu"
    vision_max_new_tokens: int = 1024
    vision_download_timeout: int = 600  # 10 minutes for first download
    vision_inference_timeout: float = 120.0  # 2 minutes per inference


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance (singleton)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings():
    """Reset settings (useful for testing)"""
    global _settings
    _settings = None
