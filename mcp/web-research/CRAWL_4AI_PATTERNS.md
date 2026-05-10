# Crawl4AI Patterns for MCP Research Server

This document contains detailed patterns and techniques from Crawl4AI that can enhance the MCP Research Server's scraping, crawling, and content extraction capabilities.

---

## Table of Contents

1. [Non-LLM Extraction Strategies](#1-non-llm-extraction-strategies)
2. [Schema-Based Extraction](#2-schema-based-extraction)
3. [Regex Pattern Extraction](#3-regex-pattern-extraction)
4. [Deep Crawling Enhancements](#4-deep-crawling-enhancements)
5. [URL Seeding Patterns](#5-url-seeding-patterns)
6. [Multi-URL Crawling](#6-multi-url-crawling)
7. [Configuration Patterns](#7-configuration-patterns)
8. [Error Handling and Retries](#8-error-handling-and-retries)
9. [Performance Optimization](#9-performance-optimization)
10. [Integration with Existing Architecture](#10-integration-with-existing-architecture)

---

## 1. Non-LLM Extraction Strategies

### 1.1 The Decision Hierarchy

Before reaching for LLM extraction, follow this decision tree:

```
Does the page have consistent HTML structure?
    ↓ YES
    Use JsonCssExtractionStrategy (manual or generate_schema())

Is it simple patterns (emails, dates, prices)?
    ↓ YES
    Use RegexExtractionStrategy

Do you need semantic understanding?
    ↓ MAYBE
    Try generate_schema() first, then consider LLM

Is the content truly unstructured text?
    ↓ ONLY THEN
    Consider LLMExtractionStrategy
```

**Cost Analysis**:
- Non-LLM: ~$0.000001 per page
- LLM: ~$0.01-$0.10 per page (10,000x more expensive)

### 1.2 Basic JsonCssExtractionStrategy

The most powerful non-LLM strategy for structured content:

```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai import JsonCssExtractionStrategy
import json

# Define a schema for extraction
schema = {
    "name": "Blog Posts",
    "baseSelector": "article.post",
    "fields": [
        {
            "name": "title",
            "selector": "h1.post-title, h2.title",
            "type": "text"
        },
        {
            "name": "author",
            "selector": ".author-name, [data-author]",
            "type": "text"
        },
        {
            "name": "publish_date",
            "selector": "time[datetime], .date",
            "type": "attribute",
            "attribute": "datetime"
        },
        {
            "name": "content",
            "selector": ".post-content, .entry-content",
            "type": "html"
        },
        {
            "name": "tags",
            "selector": ".tags a",
            "type": "list"
        }
    ]
}

async def extract_with_schema():
    strategy = JsonCssExtractionStrategy(schema, verbose=True)
    config = CrawlerRunConfig(
        extraction_strategy=strategy,
        bypass_cache=True
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://blog.example.com/posts",
            config=config
        )

        if result.success and result.extracted_content:
            data = json.loads(result.extracted_content)
            print(f"Extracted {len(data)} blog posts")
            for post in data[:3]:
                print(f"- {post.get('title', 'No title')}")
                print(f"  by {post.get('author', 'Unknown')}")
```

### 1.3 Complex Nested Schema

For pages with hierarchical structure:

```python
complex_schema = {
    "name": "E-commerce Product Catalog",
    "baseSelector": "div.category-section",
    "baseFields": [
        {
            "name": "category_name",
            "selector": "h2.category-title",
            "type": "text"
        },
        {
            "name": "category_id",
            "type": "attribute",
            "attribute": "data-category-id"
        }
    ],
    "fields": [
        {
            "name": "products",
            "selector": "div.product-item",
            "type": "nested_list",
            "fields": [
                {
                    "name": "name",
                    "selector": "h3.product-name",
                    "type": "text"
                },
                {
                    "name": "price",
                    "selector": ".price",
                    "type": "text"
                },
                {
                    "name": "details",
                    "selector": ".product-specs",
                    "type": "nested",
                    "fields": [
                        {"name": "brand", "selector": ".brand", "type": "text"},
                        {"name": "model", "selector": ".model", "type": "text"}
                    ]
                },
                {
                    "name": "features",
                    "selector": "ul.features li",
                    "type": "list"
                },
                {
                    "name": "reviews",
                    "selector": ".review",
                    "type": "nested_list",
                    "fields": [
                        {"name": "reviewer", "selector": ".reviewer", "type": "text"},
                        {"name": "rating", "selector": ".rating", "type": "attribute", "attribute": "data-rating"},
                        {"name": "comment", "selector": ".comment-text", "type": "text"}
                    ]
                }
            ]
        }
    ]
}
```

### 1.4 XPath Alternative

When CSS selectors aren't enough:

```python
from crawl4ai import JsonXPathExtractionStrategy

xpath_schema = {
    "name": "News Articles with XPath",
    "baseSelector": "//article[@class='news-item']",
    "fields": [
        {
            "name": "headline",
            "selector": ".//h2[contains(@class, 'headline')]",
            "type": "text"
        },
        {
            "name": "author",
            "selector": ".//span[@class='author']/text()",
            "type": "text"
        },
        {
            "name": "publish_date",
            "selector": ".//time/@datetime",
            "type": "text"
        },
        {
            "name": "content",
            "selector": ".//div[@class='article-body']//text()",
            "type": "text"
        }
    ]
}

strategy = JsonXPathExtractionStrategy(xpath_schema, verbose=True)
```

---

## 2. Schema-Based Extraction

### 2.1 Auto-Generate Schemas with LLM (One-Time Cost)

The `generate_schema()` method uses LLM ONCE to create a reusable extraction pattern:

```python
import json
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig
from crawl4ai import JsonCssExtractionStrategy

async def smart_extraction_workflow():
    """
    Step 1: Generate schema once using LLM
    Step 2: Cache schema for unlimited reuse
    Step 3: Extract from thousands of pages with ZERO LLM calls
    """

    cache_dir = Path("./schema_cache")
    cache_dir.mkdir(exist_ok=True)
    schema_file = cache_dir / "product_schema.json"

    # Check for cached schema first
    if schema_file.exists():
        # Load cached schema - NO LLM CALLS
        schema = json.load(schema_file.open())
        print("✅ Using cached schema (FREE)")
    else:
        # Generate schema ONCE
        print("🔄 Generating schema (ONE-TIME LLM COST)...")

        llm_config = LLMConfig(
            provider="openai/gpt-4o-mini",
            api_token="env:OPENAI_API_KEY"
        )

        # Get sample HTML from target site
        async with AsyncWebCrawler() as crawler:
            sample_result = await crawler.arun(
                url="https://example.com/products",
                config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
            )
            sample_html = sample_result.cleaned_html[:8000]

        # AUTO-GENERATE SCHEMA (ONE LLM CALL)
        schema = JsonCssExtractionStrategy.generate_schema(
            html=sample_html,
            schema_type="CSS",
            query="Extract product information including name, price, description, features",
            llm_config=llm_config
        )

        # Cache for unlimited future use
        json.dump(schema, schema_file.open("w"), indent=2)
        print("✅ Schema generated and cached")

    # Use schema for fast extraction (NO MORE LLM CALLS)
    strategy = JsonCssExtractionStrategy(schema, verbose=True)

    config = CrawlerRunConfig(
        extraction_strategy=strategy,
        cache_mode=CacheMode.BYPASS
    )

    # Extract from multiple pages - ALL FREE
    urls = [
        "https://example.com/products",
        "https://example.com/electronics",
        "https://example.com/books"
    ]

    async with AsyncWebCrawler() as crawler:
        for url in urls:
            result = await crawler.arun(url=url, config=config)
            if result.success:
                data = json.loads(result.extracted_content)
                print(f"✅ {url}: Extracted {len(data)} items (FREE)")
```

### 2.2 Generate Schema from Target JSON Example

When you know exactly what JSON structure you want:

```python
target_json_example = """
{
    "name": "Product Name",
    "price": "$99.99",
    "rating": 4.5,
    "features": ["feature1", "feature2"],
    "description": "Product description"
}
"""

schema = JsonCssExtractionStrategy.generate_schema(
    html=sample_html,
    target_json_example=target_json_example,
    llm_config=llm_config
)
```

### 2.3 Common Extraction Schemas

Pre-built schemas for common page types:

```python
# E-commerce Products
ecommerce_schema = {
    "name": "E-commerce Products",
    "baseSelector": ".product, [data-product], .item",
    "fields": [
        {"name": "title", "selector": "h1, h2, h3, .title, .name", "type": "text"},
        {"name": "price", "selector": ".price, .cost, [data-price]", "type": "text"},
        {"name": "image", "selector": "img", "type": "attribute", "attribute": "src"},
        {"name": "url", "selector": "a", "type": "attribute", "attribute": "href"},
        {"name": "rating", "selector": ".rating, .stars", "type": "text"},
        {"name": "availability", "selector": ".stock, .availability", "type": "text"}
    ]
}

# News Articles
news_schema = {
    "name": "News Articles",
    "baseSelector": "article, .article, .post",
    "fields": [
        {"name": "headline", "selector": "h1, h2, .headline, .title", "type": "text"},
        {"name": "author", "selector": ".author, .byline, [rel='author']", "type": "text"},
        {"name": "date", "selector": "time, .date, .published", "type": "text"},
        {"name": "content", "selector": ".content, .body, .text", "type": "text"},
        {"name": "category", "selector": ".category, .section", "type": "text"}
    ]
}

# Job Listings
job_schema = {
    "name": "Job Listings",
    "baseSelector": ".job, .listing, [data-job]",
    "fields": [
        {"name": "title", "selector": ".job-title, h2, h3", "type": "text"},
        {"name": "company", "selector": ".company, .employer", "type": "text"},
        {"name": "location", "selector": ".location, .place", "type": "text"},
        {"name": "salary", "selector": ".salary, .pay, .compensation", "type": "text"},
        {"name": "description", "selector": ".description, .summary", "type": "text"},
        {"name": "url", "selector": "a", "type": "attribute", "attribute": "href"}
    ]
}

# Social Media Posts
social_schema = {
    "name": "Social Media Posts",
    "baseSelector": ".post, .tweet, .update",
    "fields": [
        {"name": "username", "selector": ".username, .handle, .author", "type": "text"},
        {"name": "content", "selector": ".content, .text, .message", "type": "text"},
        {"name": "timestamp", "selector": ".time, .date, time", "type": "text"},
        {"name": "likes", "selector": ".likes, .hearts", "type": "text"},
        {"name": "shares", "selector": ".shares, .retweets", "type": "text"}
    ]
}
```

---

## 3. Regex Pattern Extraction

### 3.1 Built-in Pattern Matching

For simple data types like emails, phones, URLs, prices, dates:

```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai import RegexExtractionStrategy
import json

async def extract_common_patterns():
    # Use built-in patterns
    strategy = RegexExtractionStrategy(
        pattern=(
            RegexExtractionStrategy.Email |
            RegexExtractionStrategy.PhoneUS |
            RegexExtractionStrategy.Url |
            RegexExtractionStrategy.Currency |
            RegexExtractionStrategy.DateIso
        )
    )

    config = CrawlerRunConfig(extraction_strategy=strategy)

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://example.com/contact",
            config=config
        )

        if result.success:
            matches = json.loads(result.extracted_content)

            # Group by pattern type
            by_type = {}
            for match in matches:
                label = match['label']
                if label not in by_type:
                    by_type[label] = []
                by_type[label].append(match['value'])

            for pattern_type, values in by_type.items():
                print(f"{pattern_type}: {len(values)} matches")
                for value in values[:3]:
                    print(f"  {value}")
```

### 3.2 Available Built-in Patterns

```python
# Individual patterns
RegexExtractionStrategy.Email          # Email addresses
RegexExtractionStrategy.PhoneUS        # US phone numbers
RegexExtractionStrategy.PhoneIntl      # International phones
RegexExtractionStrategy.Url            # HTTP/HTTPS URLs
RegexExtractionStrategy.Currency       # Currency values ($99.99)
RegexExtractionStrategy.Percentage     # Percentage values (25%)
RegexExtractionStrategy.DateIso        # ISO dates (2024-01-01)
RegexExtractionStrategy.DateUS         # US dates (01/01/2024)
RegexExtractionStrategy.IPv4           # IP addresses
RegexExtractionStrategy.CreditCard     # Credit card numbers
RegexExtractionStrategy.TwitterHandle  # @username
RegexExtractionStrategy.Hashtag        # #hashtag

# Use all patterns
RegexExtractionStrategy.All
```

### 3.3 Custom Regex Patterns

```python
async def extract_custom_patterns():
    custom_patterns = {
        "product_sku": r"SKU[-:]?\s*([A-Z0-9]{4,12})",
        "discount": r"(\d{1,2})%\s*off",
        "model_number": r"Model\s*#?\s*([A-Z0-9-]+)",
        "isbn": r"ISBN[-:]?\s*(\d{10}|\d{13})",
        "stock_ticker": r"\$([A-Z]{2,5})",
        "version": r"v(\d+\.\d+(?:\.\d+)?)"
    }

    strategy = RegexExtractionStrategy(custom=custom_patterns)
    config = CrawlerRunConfig(extraction_strategy=strategy)

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://example.com/products",
            config=config
        )

        if result.success:
            data = json.loads(result.extracted_content)
            for item in data:
                print(f"{item['label']}: {item['value']}")
```

### 3.4 LLM-Generated Regex Patterns

```python
async def generate_optimized_regex():
    """Use LLM ONCE to generate optimized regex patterns"""
    cache_file = Path("./patterns/price_patterns.json")

    if cache_file.exists():
        patterns = json.load(cache_file.open())
        print("✅ Using cached regex patterns (FREE)")
    else:
        print("🔄 Generating regex patterns (ONE-TIME LLM COST)...")

        llm_config = LLMConfig(
            provider="openai/gpt-4o-mini",
            api_token="env:OPENAI_API_KEY"
        )

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun("https://example.com/pricing")
            sample_html = result.cleaned_html

        patterns = RegexExtractionStrategy.generate_pattern(
            label="pricing_info",
            html=sample_html,
            query="Extract all pricing information including discounts and special offers",
            llm_config=llm_config
        )

        cache_file.parent.mkdir(exist_ok=True)
        json.dump(patterns, cache_file.open("w"), indent=2)
        print("✅ Patterns generated and cached")

    # Use cached patterns (NO MORE LLM CALLS)
    strategy = RegexExtractionStrategy(custom=patterns)
    return strategy
```

---

## 4. Deep Crawling Enhancements

### 4.1 Filter Chains for URL Discovery

Your current `crawl_site` implementation can be enhanced with sophisticated filtering:

```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    URLPatternFilter,
    DomainFilter,
    ContentTypeFilter,
    SEOFilter,
    ContentRelevanceFilter
)

async def advanced_filtered_crawl():
    # Build sophisticated filter chain
    filter_chain = FilterChain([
        # Domain filtering
        DomainFilter(
            allowed_domains=["docs.example.com"],
            blocked_domains=["old.docs.example.com", "staging.example.com"]
        ),

        # URL pattern matching
        URLPatternFilter(patterns=["*tutorial*", "*guide*", "*api*"]),

        # Content type filtering
        ContentTypeFilter(allowed_types=["text/html"]),

        # SEO quality filter
        SEOFilter(
            threshold=0.5,
            keywords=["tutorial", "guide", "documentation"]
        ),

        # Content relevance filter (LLM-based, use sparingly)
        ContentRelevanceFilter(
            query="API documentation and usage examples",
            threshold=0.7
        )
    ])

    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=2,
            filter_chain=filter_chain,
            max_pages=20
        ),
        word_count_threshold=100,
        verbose=True
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://docs.example.com",
            config=config
        )

        print(f"Crawled {len(result.deep_crawl_results)} pages with filters")
```

### 4.2 Best-First Strategy with Relevance Scoring

For intelligent crawling that prioritizes relevant pages:

```python
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

async def scored_deep_crawl():
    # Create scorer for prioritization
    keyword_scorer = KeywordRelevanceScorer(
        keywords=["api", "reference", "endpoint", "method"],
        weight=1.0
    )

    config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=2,
            include_external=False,
            url_scorer=keyword_scorer,
            max_pages=25
        ),
        stream=True,  # Recommended with BestFirst
        verbose=True
    )

    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun(
            "https://docs.example.com",
            config=config
        ):
            score = result.metadata.get("score", 0)
            depth = result.metadata.get("depth", 0)
            print(f"Depth: {depth} | Score: {score:.2f} | {result.url}")
```

### 4.3 Streaming vs Batch Deep Crawling

```python
# Batch mode - wait for all results
async def batch_deep_crawl():
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=2,
            max_pages=50
        ),
        stream=False  # Default
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://example.com", config=config)
        # Process all results at once
        for page_result in result.deep_crawl_results:
            print(f"Batch: {page_result.url}")

# Streaming mode - process results as they arrive
async def streaming_deep_crawl():
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=2,
            max_pages=50
        ),
        stream=True  # Process immediately
    )

    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun(
            "https://example.com",
            config=config
        ):
            depth = result.metadata.get("depth", 0)
            print(f"Stream: Depth {depth} | {result.url}")
```

### 4.4 Error-Aware Deep Crawling

```python
async def robust_deep_crawl():
    config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=2,
            max_pages=15,
            url_scorer=KeywordRelevanceScorer(
                keywords=["guide", "tutorial"]
            )
        ),
        stream=True,
        page_timeout=30000  # 30 second timeout
    )

    successful_pages = []
    failed_pages = []

    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun(
            "https://docs.example.com",
            config=config
        ):
            if result.success:
                successful_pages.append(result)
                depth = result.metadata.get("depth", 0)
                score = result.metadata.get("score", 0)
                print(f"✅ Depth {depth} | Score: {score:.2f} | {result.url}")
            else:
                failed_pages.append({
                    'url': result.url,
                    'error': result.error_message,
                    'depth': result.metadata.get("depth", 0)
                })
                print(f"❌ Failed: {result.url} - {result.error_message}")

    # Analyze failures by depth
    if failed_pages:
        failure_by_depth = {}
        for failure in failed_pages:
            depth = failure['depth']
            failure_by_depth[depth] = failure_by_depth.get(depth, 0) + 1

        print("❌ Failures by depth:")
        for depth, count in sorted(failure_by_depth.items()):
            print(f"   Depth {depth}: {count} failures")
```

---

## 5. URL Seeding Patterns

### 5.1 URL Discovery for Large-Scale Crawling

The `AsyncUrlSeeder` provides instant URL discovery without crawling:

```python
from crawl4ai import AsyncUrlSeeder, SeedingConfig

async def smart_research_pipeline():
    """Complete pipeline: discover URLs, filter by relevance, crawl top results"""

    async with AsyncUrlSeeder() as seeder:
        # Step 1: Discover relevant URLs
        print("🔍 Discovering URLs...")
        config = SeedingConfig(
            source="sitemap+cc",
            extract_head=True,
            query="machine learning deep learning tutorial",
            scoring_method="bm25",
            score_threshold=0.4,
            max_urls=100
        )

        urls = await seeder.urls("example.com", config)
        print(f"   Found {len(urls)} relevant URLs")

        # Step 2: Select top articles
        top_articles = sorted(
            urls,
            key=lambda x: x.get('relevance_score', 0),
            reverse=True
        )[:10]

        print(f"   Selected top {len(top_articles)} for crawling")

        # Step 3: Show what we're about to crawl
        print("\n📋 Articles to crawl:")
        for i, article in enumerate(top_articles, 1):
            score = article.get('relevance_score', 0)
            title = article.get('head_data', {}).get('title', 'No title')[:60]
            print(f"  {i}. [{score:.2f}] {title}")

    # Step 4: Crawl selected articles
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

    print(f"\n🕷️ Crawling {len(top_articles)} articles...")

    async with AsyncWebCrawler() as crawler:
        config = CrawlerRunConfig(
            only_text=True,
            word_count_threshold=200
        )

        article_urls = [a['url'] for a in top_articles]
        results = await crawler.arun_many(article_urls, config=config)

        for result in results:
            if result.success:
                word_count = len(result.markdown.raw_markdown.split())
                print(f"   ✅ {word_count} words from {result.url[:50]}...")
```

### 5.2 BM25 Relevance Scoring

```python
async def relevance_scoring():
    async with AsyncUrlSeeder() as seeder:
        # Find pages about Python async programming
        config = SeedingConfig(
            source="sitemap",
            extract_head=True,
            query="python async await concurrency",
            scoring_method="bm25",
            score_threshold=0.3,
            max_urls=20
        )

        urls = await seeder.urls("docs.python.org", config)

        # Results are automatically sorted by relevance
        print("Most relevant Python async content:")
        for url in urls[:5]:
            score = url.get('relevance_score', 0)
            title = url.get('head_data', {}).get('title', 'No title')
            print(f"[{score:.2f}] {title}")
            print(f"        {url['url']}")
```

### 5.3 Multi-Domain Discovery

```python
async def multi_domain_research():
    async with AsyncUrlSeeder() as seeder:
        domains = [
            "docs.python.org",
            "realpython.com",
            "python-course.eu"
        ]

        config = SeedingConfig(
            source="sitemap",
            extract_head=True,
            query="python beginner tutorial basics",
            scoring_method="bm25",
            score_threshold=0.3,
            max_urls=15
        )

        # Discover across all domains in parallel
        results = await seeder.many_urls(domains, config)

        # Collect and rank all tutorials
        all_tutorials = []
        for domain, urls in results.items():
            for url in urls:
                url['domain'] = domain
                all_tutorials.append(url)

        all_tutorials.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        print(f"Top 10 Python tutorials across {len(domains)} sites:")
        for i, tutorial in enumerate(all_tutorials[:10], 1):
            score = tutorial.get('relevance_score', 0)
            title = tutorial.get('head_data', {}).get('title', 'No title')[:60]
            domain = tutorial['domain']
            print(f"{i:2d}. [{score:.2f}] {title}")
            print(f"     {domain}")
```

---

## 6. Multi-URL Crawling

### 6.1 Memory-Adaptive Dispatching

For large-scale crawling without OOM errors:

```python
from crawl4ai import MemoryAdaptiveDispatcher, CrawlerMonitor, DisplayMode

async def memory_adaptive_crawl():
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=80.0,
        check_interval=1.0,
        max_session_permit=15,
        memory_wait_timeout=300.0
    )

    config = CrawlerRunConfig(
        bypass_cache=True,
        word_count_threshold=50
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun_many(
            urls=large_url_list,
            config=config,
            dispatcher=dispatcher
        )

        for result in results:
            if result.dispatch_result:
                dr = result.dispatch_result
                print(f"Memory used: {dr.memory_usage:.1f}MB")
                print(f"Duration: {dr.end_time - dr.start_time}")
```

### 6.2 Rate-Limited Crawling

```python
from crawl4ai import RateLimiter, SemaphoreDispatcher

async def rate_limited_crawl():
    rate_limiter = RateLimiter(
        base_delay=(1.0, 3.0),
        max_delay=60.0,
        max_retries=3,
        rate_limit_codes=[429, 503]
    )

    dispatcher = SemaphoreDispatcher(
        max_session_permit=5,
        rate_limiter=rate_limiter
    )

    config = CrawlerRunConfig(
        user_agent_mode="random",
        simulate_user=True
    )

    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun_many(
            urls=urls,
            config=config,
            dispatcher=dispatcher
        ):
            print(f"Processed: {result.url}")
```

---

## 7. Configuration Patterns

### 7.1 Browser Configuration

```python
from crawl4ai import BrowserConfig, AsyncWebCrawler

# Lightweight browser config for performance
browser_config = BrowserConfig(
    headless=True,
    text_mode=True,  # Disable images for speed
    light_mode=True,
    viewport_width=1280,
    viewport_height=720,
    verbose=True
)

# Stealth configuration
stealth_browser_config = BrowserConfig(
    headless=True,
    proxy="http://user:pass@proxy:8080",
    use_persistent_context=True,
    user_data_dir="./browser_data",
    cookies=[{"name": "session", "value": "abc123", "domain": "example.com"}],
    user_agent="Mozilla/5.0 (X11; Linux x86_64) Chrome/116.0.0.0 Safari/537.36",
    extra_args=["--disable-extensions", "--no-sandbox"]
)
```

### 7.2 Advanced CrawlerRunConfig

```python
from crawl4ai import CrawlerRunConfig, CacheMode

# Comprehensive configuration
config = CrawlerRunConfig(
    # Content processing
    css_selector="main.content",
    excluded_tags=["nav", "footer", "script"],
    excluded_selector="#ads, .tracker",
    word_count_threshold=50,

    # Page interaction
    js_code=[
        "window.scrollTo(0, document.body.scrollHeight);",
        "document.querySelector('.load-more')?.click();"
    ],
    wait_for="css:.content-loaded",
    wait_for_timeout=10000,
    scan_full_page=True,

    # Session management
    session_id="persistent_session",

    # Media handling
    screenshot=True,
    pdf=True,
    image_score_threshold=5,

    # Anti-detection
    simulate_user=True,
    override_navigator=True,
    magic=True,

    # Caching
    cache_mode=CacheMode.BYPASS
)
```

### 7.3 Advanced wait_for Conditions

```python
# CSS selector waiting
config = CrawlerRunConfig(
    wait_for="css:.content-loaded",
    wait_for_timeout=15000
)

# JavaScript boolean expression waiting
config = CrawlerRunConfig(
    wait_for="js:() => window.dataLoaded === true",
    wait_for_timeout=20000
)

# Complex JavaScript conditions
config = CrawlerRunConfig(
    wait_for="js:() => document.querySelectorAll('.item').length >= 10",
    js_code=[
        "document.querySelector('.load-more')?.click();",
        "window.scrollTo(0, document.body.scrollHeight);"
    ]
)
```

---

## 8. Error Handling and Retries

### 8.1 Multi-Strategy Extraction with Fallback

```python
async def multi_strategy_extraction():
    """Try multiple strategies until one works"""

    strategies = [
        # Try fast regex first
        RegexExtractionStrategy(pattern=RegexExtractionStrategy.Currency),

        # Fallback to CSS schema
        JsonCssExtractionStrategy({
            "name": "Prices",
            "baseSelector": ".price",
            "fields": [{"name": "amount", "selector": "span", "type": "text"}]
        }),

        # Last resort: different selector
        JsonCssExtractionStrategy({
            "name": "Fallback Prices",
            "baseSelector": "[data-price]",
            "fields": [{"name": "amount", "type": "attribute", "attribute": "data-price"}]
        })
    ]

    async with AsyncWebCrawler() as crawler:
        for i, strategy in enumerate(strategies):
            try:
                config = CrawlerRunConfig(extraction_strategy=strategy)
                result = await crawler.arun(url="https://example.com", config=config)

                if result.success and result.extracted_content:
                    data = json.loads(result.extracted_content)
                    if data:  # Validate non-empty results
                        print(f"✅ Success with strategy {i+1}")
                        return data

            except Exception as e:
                print(f"❌ Strategy {i+1} failed: {e}")
                continue

    print("❌ All strategies failed")
    return None
```

### 8.2 Validation and Retry Logic

```python
def validate_extraction_quality(data):
    """Validate that extraction meets quality standards"""
    if not data or not isinstance(data, (list, dict)):
        return False

    if isinstance(data, list):
        if len(data) == 0:
            return False
        for item in data:
            if not isinstance(item, dict) or len(item) < 2:
                return False

    return True

async def robust_llm_extraction():
    max_retries = 3
    strategies = [
        LLMExtractionStrategy(
            llm_config=LLMConfig(provider="openai/gpt-4o-mini"),
            schema=YourModel.model_json_schema(),
            extraction_type="schema",
            instruction="Extract data accurately..."
        ),
        LLMExtractionStrategy(
            llm_config=LLMConfig(provider="openai/gpt-4o"),
            schema=YourModel.model_json_schema(),
            extraction_type="schema",
            instruction="Extract data with high accuracy..."
        )
    ]

    for strategy_idx, strategy in enumerate(strategies):
        for attempt in range(max_retries):
            try:
                config = CrawlerRunConfig(extraction_strategy=strategy)
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url="https://example.com", config=config)

                    if result.success and result.extracted_content:
                        data = json.loads(result.extracted_content)

                        if validate_extraction_quality(data):
                            print(f"✅ Success with strategy {strategy_idx+1}, attempt {attempt+1}")
                            return data

            except Exception as e:
                print(f"❌ Attempt {attempt+1} failed: {e}")

    print("❌ All strategies and retries failed")
    return None
```

---

## 9. Performance Optimization

### 9.1 Fast Text-Only Crawling

```python
config = CrawlerRunConfig(
    cache_mode=CacheMode.ENABLED,
    text_mode=True,
    exclude_external_links=True,
    exclude_external_images=True,
    word_count_threshold=50,
    excluded_tags=["script", "style", "nav", "footer", "header"]
)
```

### 9.2 Configuration Cloning

```python
# Clone configurations for variations
base_config = CrawlerRunConfig(
    cache_mode=CacheMode.ENABLED,
    word_count_threshold=200,
    verbose=True
)

# Create streaming version
stream_config = base_config.clone(
    stream=True,
    cache_mode=CacheMode.BYPASS
)

# Create debug version
debug_config = base_config.clone(
    headless=False,
    page_timeout=120000,
    verbose=True
)
```

### 9.3 Selector Optimization

```python
# Optimized selectors for speed
fast_schema = {
    "name": "Optimized Extraction",
    "baseSelector": "#products > .product",  # Direct child, faster
    "fields": [
        {
            "name": "title",
            "selector": "> h3",  # Direct child of product
            "type": "text"
        },
        {
            "name": "price",
            "selector": ".price:first-child",  # More specific
            "type": "text"
        }
    ]
}

# Avoid slow selectors
slow_schema = {
    "baseSelector": "div div div .product",  # Too many levels
    "fields": [
        {
            "selector": "* h3",  # Universal selector is slow
            "type": "text"
        }
    ]
}
```

---

## 10. Integration with Existing Architecture

### 10.1 Extending MapCrawlService

Enhance your existing `MapCrawlService` with new features:

```python
# src/services/crawl_service.py enhancements

from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter

@dataclass
class CrawlConfig:
    """Enhanced configuration for deep crawling"""
    max_depth: int = 2
    max_pages: int = 50
    include_external: bool = False
    pattern: str | None = None
    only_text: bool = True
    word_count_threshold: int = 100

    # New: Scoring options
    score_keywords: list[str] | None = None  # For Best-First strategy
    score_threshold: float = 0.3  # Minimum relevance score

    # New: Strategy selection
    strategy: Literal["bfs", "dfs", "best_first"] = "bfs"

    # New: Filter chain
    url_patterns: list[str] | None = None  # URL patterns to include

class MapCrawlService:
    """Enhanced service with new crawl strategies"""

    async def crawl_site(self, url: str, config: CrawlConfig) -> CrawlResult:
        """Enhanced deep crawl with multiple strategies"""
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
        from crawl4ai.deep_crawling import (
            BFSDeepCrawlStrategy,
            DFSDeepCrawlStrategy,
            BestFirstCrawlingStrategy
        )

        logger.info(f"Deep crawling: {url} (strategy={config.strategy})")

        # Build strategy based on config
        if config.strategy == "best_first" and config.score_keywords:
            strategy = BestFirstCrawlingStrategy(
                max_depth=config.max_depth,
                include_external=config.include_external,
                max_pages=config.max_pages,
                pattern=config.pattern,
                url_scorer=KeywordRelevanceScorer(
                    keywords=config.score_keywords,
                    weight=1.0
                ),
                score_threshold=config.score_threshold
            )
        elif config.strategy == "dfs":
            strategy = DFSDeepCrawlStrategy(
                max_depth=config.max_depth,
                include_external=config.include_external,
                max_pages=config.max_pages,
                pattern=config.pattern
            )
        else:
            # Default BFS with optional filter chain
            filter_chain = None
            if config.url_patterns:
                from crawl4ai.deep_crawling.filters import URLPatternFilter
                filter_chain = FilterChain([
                    URLPatternFilter(patterns=config.url_patterns)
                ])

            strategy = BFSDeepCrawlStrategy(
                max_depth=config.max_depth,
                include_external=config.include_external,
                max_pages=config.max_pages,
                pattern=config.pattern,
                filter_chain=filter_chain
            )

        crawl_config = CrawlerRunConfig(
            deep_crawl_strategy=strategy,
            only_text=config.only_text,
            word_count_threshold=config.word_count_threshold,
            verbose=True,
            stream=True  # Enable streaming for better performance
        )

        # ... rest of implementation
```

### 10.2 Adding Extraction Strategy to Scrapers

```python
# src/scrapers/base.py enhancements

async def scrape_crawl4ai_with_extraction(
    url: str,
    cleaner,
    css_selector: str = None,
    extraction_schema: dict = None
) -> dict:
    """
    Enhanced Crawl4AI scraping with optional extraction schema

    Args:
        url: URL to scrape
        cleaner: ContentCleaner instance
        css_selector: Optional CSS selector for targeted extraction
        extraction_schema: Optional JsonCssExtractionStrategy schema

    Returns dict response
    """
    try:
        from crawl4ai import AsyncWebCrawler, JsonCssExtractionStrategy, CrawlerRunConfig

        # Build config with optional extraction strategy
        extraction_strategy = None
        if extraction_schema:
            extraction_strategy = JsonCssExtractionStrategy(
                extraction_schema,
                verbose=True
            )

        crawl_config = CrawlerRunConfig(
            word_count_threshold=CRAWL4AI_WORD_COUNT_THRESHOLD,
            bypass_cache=True,
            process_iframes=False,
            css_selector=css_selector,
            extraction_strategy=extraction_strategy
        )

        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url, config=crawl_config)

            if result.success:
                domain = extract_domain(url)

                # If extraction strategy was used, return structured data
                if result.extracted_content:
                    import json
                    structured_data = json.loads(result.extracted_content)

                    return build_scrape_response(
                        success=True,
                        url=url,
                        method="crawl4ai_structured",
                        title=result.metadata.get("title", ""),
                        content=json.dumps(structured_data),
                        metadata={
                            "extracted_items": len(structured_data) if isinstance(structured_data, list) else 1,
                            "extraction_type": "structured"
                        }
                    )

                # Fallback to regular content cleaning
                html_content = result.html
                clean_markdown = cleaner.clean(html_content, url, css_selector)

                if len(clean_markdown) < MIN_CONTENT_LENGTH:
                    crawl4ai_md = result.markdown
                    if hasattr(crawl4ai_md, 'raw_markdown'):
                        crawl4ai_md = crawl4ai_md.raw_markdown

                    if len(crawl4ai_md) >= MIN_CONTENT_LENGTH:
                        clean_markdown = crawl4ai_md
                    else:
                        return build_content_too_short_response(url, "crawl4ai", len(clean_markdown))

                title = result.metadata.get("title", "") if hasattr(result, "metadata") else ""

                return build_scrape_response(
                    success=True,
                    url=url,
                    method="crawl4ai",
                    title=title,
                    content=clean_markdown,
                    metadata=_build_metadata(len(clean_markdown.split()))
                )

        return build_error_response(url, "crawl4ai", "Scraping failed")

    except Exception as e:
        logger.warning(f"Crawl4AI error for {url}: {e}")
        return build_error_response(url, "crawl4ai", e)
```

### 10.3 MCP Tool Extensions

```python
# src/mcp_sse.py additions

@mcp.tool()
async def scrape_structured(
    url: Annotated[str, Field(description="URL to scrape with structured extraction")],
    schema_type: Annotated[Literal["ecommerce", "news", "jobs", "blog"], Field(
        description="Pre-built schema type to use"
    )] = "ecommerce",
    custom_selector: Annotated[str | None, Field(
        description="Custom CSS selector for targeted extraction"
    )] = None,
    ctx: Context | None = None
) -> dict:
    """Scrape a URL and extract structured data using schemas

    Uses pre-built extraction schemas for common page types.
    Returns structured JSON data instead of markdown.

    SCHEMA TYPES:
    - ecommerce: Products with name, price, rating, availability
    - news: Articles with headline, author, date, content
    - jobs: Listings with title, company, location, salary
    - blog: Posts with title, author, date, content, tags
    """
    # Implementation using enhanced scrape_crawl4ai_with_extraction
    pass


@mcp.tool()
async def discover_and_crawl(
    domain: Annotated[str, Field(description="Domain to explore")],
    query: Annotated[str, Field(description="Search query for relevance scoring")],
    max_discover: Annotated[int, Field(description="Max URLs to discover")] = 100,
    max_crawl: Annotated[int, Field(description="Max URLs to crawl")] = 20,
    ctx: Context | None = None
) -> dict:
    """Two-phase crawling: discover relevant URLs, then crawl top results

    PHASE 1 - Discovery: Uses URL seeding with BM25 relevance scoring
    PHASE 2 - Crawling: Deep crawls only the most relevant URLs

    This is more efficient than blind deep crawling for large sites.
    """
    # Implementation combining map_domain and crawl_site
    pass
```

---

## Summary

The Crawl4AI patterns presented here focus on:

1. **Cost Efficiency**: Using non-LLM extraction 99% of the time
2. **Performance**: Memory-adaptive dispatching, selector optimization, caching
3. **Reliability**: Multi-strategy fallbacks, validation, error handling
4. **Intelligence**: BM25 relevance scoring, filter chains, smart URL discovery
5. **Maintainability**: Schema caching, configuration cloning, reusable patterns

The most impactful additions for your MCP Research Server would be:

1. **`generate_schema()` workflow** - Learn extraction patterns once, reuse forever
2. **Filter chains** - More sophisticated URL filtering in deep crawls
3. **Best-First strategy** - Prioritize relevant pages during crawling
4. **Memory-adaptive dispatching** - Handle large URL batches safely
5. **Regex extraction** - Fast pattern matching without LLM

These patterns complement your existing architecture (domain learning, Waterfall content cleaning, Selenium fallback) while adding new capabilities for structured data extraction and intelligent crawling.
