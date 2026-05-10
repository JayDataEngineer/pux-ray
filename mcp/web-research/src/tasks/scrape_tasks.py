"""Celery task for scraping with concurrency control and caching"""

from ..celery_app import app
from ..tasks.base import BaseTask
from ..scrapers.base import scrape_with_fallback


@app.task(bind=True, base=BaseTask, name="scrape_task")
def scrape_task(
    self,
    url: str,
    force_method: str | None = None,
    css_selector: str | None = None
) -> dict:
    """
    Scrape a URL with automatic method routing and caching

    This runs in the Celery worker with controlled concurrency.
    Returns dict that can be serialized to JSON for Redis.

    Args:
        url: URL to scrape
        force_method: Force specific scraping method
        css_selector: Optional CSS selector for targeted content extraction
    """
    # Check cache first
    if self.cache:
        cached = self.run_async(self.cache.get_scrape(url))
        if cached:
            cached["cached"] = True
            return cached

    # Run the async scrape
    result = self.run_async(
        scrape_with_fallback(
            url=url,
            cleaner=self.cleaner,
            db=self.db,
            force_method=force_method,
            css_selector=css_selector
        )
    )

    # Cache successful results
    if result.get("success") and self.cache:
        self.run_async(self.cache.set_scrape(url, result))

    result["cached"] = False
    return result
