"""Profile scrape_url performance - measures each stage of the scraping pipeline.

Usage:
    docker exec mcp-server python /app/profiling/profile_scrape.py
    docker exec mcp-server python /app/profiling/profile_scrape.py --url https://example.com
"""

import asyncio
import time
import argparse
import json
import statistics


# Default URLs to profile (diverse set)
DEFAULT_URLS = [
    "https://fastapi.tiangolo.com/tutorial/",
    "https://news.ycombinator.com/",
    "https://docs.python.org/3/library/asyncio.html",
]


async def profile_single(url: str) -> dict:
    """Profile a single scrape request, timing each stage."""
    from src.scrapers.base import scrape_crawl4ai, scrape_selenium, scrape_with_fallback
    from src.services.content_cleaner import ContentCleaner
    from src.db.database import Database

    cleaner = ContentCleaner()
    db = Database()
    await db.init()

    stages = {}

    # Stage 1: Crawl4AI (full pipeline)
    t0 = time.time()
    result = await scrape_crawl4ai(url, cleaner, text_only=True)
    stages["crawl4ai_total"] = time.time() - t0
    stages["crawl4ai_success"] = result["success"]
    if result["success"]:
        stages["content_length"] = len(result.get("content", ""))
        stages["title"] = result.get("title", "")

    # Stage 2: Selenium (if Crawl4AI failed, or for comparison)
    t0 = time.time()
    result_sel = await scrape_selenium(url, cleaner)
    stages["selenium_total"] = time.time() - t0
    stages["selenium_success"] = result_sel["success"]
    if result_sel["success"] and "content_length" not in stages:
        stages["content_length"] = len(result_sel.get("content", ""))

    # Stage 3: Full fallback pipeline
    t0 = time.time()
    result_full = await scrape_with_fallback(url, cleaner, db)
    stages["fallback_total"] = time.time() - t0
    stages["fallback_success"] = result_full["success"]
    stages["fallback_method"] = result_full.get("method_used", "unknown")

    await db.close()
    return stages


async def profile_url(url: str, runs: int = 3) -> dict:
    """Profile a URL multiple times and return stats."""
    all_stages = []

    for i in range(runs):
        print(f"  Run {i+1}/{runs}...", end="", flush=True)
        stages = await profile_single(url)
        all_stages.append(stages)
        c4ai = stages["crawl4ai_total"]
        sel = stages["selenium_total"]
        fb = stages["fallback_total"]
        print(f" Crawl4AI={c4ai:.2f}s  Selenium={sel:.2f}s  Fallback={fb:.2f}s")

    # Aggregate
    keys = ["crawl4ai_total", "selenium_total", "fallback_total"]
    stats = {}
    for key in keys:
        values = [s[key] for s in all_stages]
        stats[key] = {
            "mean": round(statistics.mean(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }
    stats["url"] = url
    stats["runs"] = runs
    stats["content_length"] = all_stages[0].get("content_length", 0)
    stats["crawl4ai_success_rate"] = sum(1 for s in all_stages if s["crawl4ai_success"]) / runs

    return stats


async def main():
    parser = argparse.ArgumentParser(description="Profile scrape_url performance")
    parser.add_argument("--url", help="Single URL to profile")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per URL")
    args = parser.parse_args()

    urls = [args.url] if args.url else DEFAULT_URLS

    print("=" * 60)
    print("SCRAPING PERFORMANCE PROFILE")
    print("=" * 60)

    all_stats = []
    for url in urls:
        print(f"\nURL: {url}")
        print("-" * 40)
        stats = await profile_url(url, runs=args.runs)
        all_stats.append(stats)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'URL':<45} {'C4AI':>8} {'Selen':>8} {'Fback':>8} {'Len':>8}")
    print("-" * 80)
    for s in all_stats:
        short_url = s["url"][:43] + ".." if len(s["url"]) > 45 else s["url"]
        print(f"{short_url:<45} {s['crawl4ai_total']['mean']:>7.2f}s {s['selenium_total']['mean']:>7.2f}s {s['fallback_total']['mean']:>7.2f}s {s.get('content_length',0):>8}")


if __name__ == "__main__":
    asyncio.run(main())
