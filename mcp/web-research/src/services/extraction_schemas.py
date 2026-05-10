"""Pre-built extraction schemas for common page types

These schemas use JsonCssExtractionStrategy from Crawl4AI
to extract structured data without LLM costs.
"""

from typing import Literal

# Schema type names
SCHEMA_TYPES = Literal["ecommerce", "news", "jobs", "blog", "social", "products"]

# Pre-built schemas for common page types
EXTRACTION_SCHEMAS: dict[str, dict] = {
    "ecommerce": {
        "name": "E-commerce Products",
        "baseSelector": ".product, [data-product], .item, .product-item",
        "fields": [
            {"name": "title", "selector": "h1, h2, h3, .title, .name, .product-name", "type": "text"},
            {"name": "price", "selector": ".price, .cost, [data-price], .current-price", "type": "text"},
            {"name": "original_price", "selector": ".original-price, .was-price, .msrp", "type": "text"},
            {"name": "image", "selector": "img.product-image, .product-img img, img:first-child", "type": "attribute", "attribute": "src"},
            {"name": "url", "selector": "a.product-link, .product-url a, a:first-child", "type": "attribute", "attribute": "href"},
            {"name": "rating", "selector": ".rating, .stars, .review-score, [data-rating]", "type": "text"},
            {"name": "review_count", "selector": ".review-count, .reviews-count", "type": "text"},
            {"name": "availability", "selector": ".stock, .availability, .in-stock, [data-stock]", "type": "text"},
            {"name": "brand", "selector": ".brand, .manufacturer, [data-brand]", "type": "text"},
            {"name": "sku", "selector": ".sku, [data-sku], .product-id", "type": "text"},
            {"name": "discount", "selector": ".discount, .savings, .you-save", "type": "text"},
        ]
    },

    "news": {
        "name": "News Articles",
        "baseSelector": "article, .article, .post, .news-item, .story",
        "fields": [
            {"name": "headline", "selector": "h1, h2, .headline, .title, .article-title", "type": "text"},
            {"name": "author", "selector": ".author, .byline, .writer, [rel='author'], .author-name", "type": "text"},
            {"name": "date", "selector": "time, .date, .published, .publish-date, time[datetime]", "type": "text"},
            {"name": "datetime", "selector": "time[datetime], .date[data-datetime]", "type": "attribute", "attribute": "datetime"},
            {"name": "content", "selector": ".content, .body, .text, .article-body, .story-content", "type": "html"},
            {"name": "category", "selector": ".category, .section, .tag, .topic", "type": "text"},
            {"name": "image", "selector": "img.article-image, .featured-image img, img:first-child", "type": "attribute", "attribute": "src"},
            {"name": "summary", "selector": ".summary, .excerpt, .lead, .teaser, .description", "type": "text"},
            {"name": "tags", "selector": ".tags a, .tag, .labels a", "type": "list"},
        ]
    },

    "jobs": {
        "name": "Job Listings",
        "baseSelector": ".job, .listing, [data-job], .job-card, .posting",
        "fields": [
            {"name": "title", "selector": ".job-title, h2, h3, .position, .role", "type": "text"},
            {"name": "company", "selector": ".company, .employer, .organization, [data-company]", "type": "text"},
            {"name": "location", "selector": ".location, .place, .city, [data-location]", "type": "text"},
            {"name": "salary", "selector": ".salary, .pay, .compensation, .wage, [data-salary]", "type": "text"},
            {"name": "description", "selector": ".description, .summary, .job-description", "type": "text"},
            {"name": "url", "selector": "a.job-link, .apply-link a, a:first-child", "type": "attribute", "attribute": "href"},
            {"name": "employment_type", "selector": ".employment-type, .job-type, [data-employment-type]", "type": "text"},
            {"name": "posted_date", "selector": ".posted, .date-posted, time, .posted-date", "type": "text"},
            {"name": "department", "selector": ".department, .team, [data-department]", "type": "text"},
            {"name": "remote", "selector": ".remote, [data-remote], .location-remote", "type": "text"},
        ]
    },

    "blog": {
        "name": "Blog Posts",
        "baseSelector": "article, .post, .blog-post, .entry",
        "fields": [
            {"name": "title", "selector": "h1, h2, .title, .entry-title, .post-title", "type": "text"},
            {"name": "author", "selector": ".author, .byline, .post-author, [rel='author']", "type": "text"},
            {"name": "date", "selector": "time, .date, .published, .post-date", "type": "text"},
            {"name": "content", "selector": ".content, .entry-content, .post-content, .article-body", "type": "html"},
            {"name": "excerpt", "selector": ".excerpt, .summary, .teaser, .lead", "type": "text"},
            {"name": "categories", "selector": ".categories a, .category a", "type": "list"},
            {"name": "tags", "selector": ".tags a, .post-tags a", "type": "list"},
            {"name": "featured_image", "selector": ".featured-image img, .post-thumbnail img", "type": "attribute", "attribute": "src"},
            {"name": "comment_count", "selector": ".comment-count, .comments-count", "type": "text"},
        ]
    },

    "social": {
        "name": "Social Media Posts",
        "baseSelector": ".post, .tweet, .update, .status",
        "fields": [
            {"name": "username", "selector": ".username, .handle, .author, .author-name", "type": "text"},
            {"name": "display_name", "selector": ".display-name, .author-name, .name", "type": "text"},
            {"name": "content", "selector": ".content, .text, .message, .post-content", "type": "text"},
            {"name": "timestamp", "selector": ".time, .date, time, .timestamp", "type": "text"},
            {"name": "likes", "selector": ".likes, .hearts, .like-count, [data-likes]", "type": "text"},
            {"name": "shares", "selector": ".shares, .retweets, .share-count", "type": "text"},
            {"name": "comments", "selector": ".comments, .replies, .comment-count", "type": "text"},
            {"name": "url", "selector": "a.permalink, .post-link, a:first-child", "type": "attribute", "attribute": "href"},
            {"name": "attachments", "selector": ".attachment img, .media img", "type": "list", "output_type": "attribute", "attribute": "src"},
        ]
    },

    "products": {
        "name": "Product Catalog (Multi-item)",
        "baseSelector": ".product-grid .product, .catalog .item, [data-product]",
        "fields": [
            {"name": "name", "selector": "h3, h4, .product-name, .title", "type": "text"},
            {"name": "price", "selector": ".price", "type": "text"},
            {"name": "image", "selector": "img", "type": "attribute", "attribute": "src"},
            {"name": "url", "selector": "a", "type": "attribute", "attribute": "href"},
            {"name": "badge", "selector": ".badge, .tag, .label", "type": "text"},
        ]
    },
}


def get_schema(schema_type: str) -> dict | None:
    """Get a pre-built schema by type.

    Args:
        schema_type: Type of schema (ecommerce, news, jobs, blog, social, products)

    Returns:
        Schema dict or None if not found
    """
    return EXTRACTION_SCHEMAS.get(schema_type)


def list_schemas() -> list[dict]:
    """List all available schema types with descriptions.

    Returns:
        List of dicts with name and description
    """
    return [
        {"type": key, "name": value.get("name", key), "fields": len(value.get("fields", []))}
        for key, value in EXTRACTION_SCHEMAS.items()
    ]
