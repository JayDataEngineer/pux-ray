"""Application constants - single source of truth for magic numbers"""

# Scraping timeouts and thresholds
CRAWL4AI_WORD_COUNT_THRESHOLD = 10
SELENIUM_PAGE_LOAD_WAIT_SECONDS = 3
HTTP_REQUEST_TIMEOUT = 30.0

# Content limits
REDDIT_MAX_COMMENTS = 20
REDDIT_MAX_POSTS = 20
MIN_CONTENT_LENGTH = 100
MAX_CONTENT_LENGTH = 50000  # Cap scrape output to ~50K chars

# Celery configuration
CELERY_TASK_TIMEOUT_SECONDS = 300
CELERY_TASK_SOFT_TIMEOUT_SECONDS = 270
CELERY_WORKER_CONCURRENCY = 10
CELERY_RESULT_EXPIRE_SECONDS = 86400

# Rate limiting
RATE_LIMIT_MAX_CONCURRENT = 3
RATE_LIMIT_ACQUIRE_TIMEOUT = 30
RATE_LIMIT_TTL = 300

# Blacklist threshold
BLACKLIST_FAILURE_THRESHOLD = 3

# Retry counts before switching methods
CRAWL4AI_RETRY_COUNT = 2
SELENIUM_RETRY_COUNT = 2

# Crawl4AI browser concurrency limit (async MCP server)
# Reduced due to "Target crashed" errors under load - Playwright browsers are memory heavy
CRAWL4AI_MAX_CONCURRENT = 5

# Search configuration
DEFAULT_SEARCH_ENGINES = ["duckduckgo", "brave", "mojeek", "qwant", "google", "bing"]
MAX_SEARCH_PAGES = 5

# Cache TTL (seconds)
CACHE_TTL_SCRAPE = 86400
CACHE_TTL_SEARCH = 3600

# User agents
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

# Headers for HTTP requests
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
