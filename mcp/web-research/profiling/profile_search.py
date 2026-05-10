"""Profile search_web performance - measures search latency and result counts.

Usage:
    docker exec mcp-server python /app/profiling/profile_search.py
    docker exec mcp-server python /app/profiling/profile_search.py --pages 5
"""

import asyncio
import time
import argparse
import statistics


QUERIES = [
    "FastAPI tutorial",
    "Python web scraping",
    "machine learning frameworks 2025",
    "docker compose best practices",
    "react server components guide",
]


async def profile_search(query: str, pages: int = 3, top_k: int = 20, runs: int = 3) -> dict:
    """Profile a search query multiple times."""
    from src.services.search_service import get_search_service

    svc = get_search_service()
    timings = []
    result_counts = []

    for i in range(runs):
        print(f"  Run {i+1}/{runs}...", end="", flush=True)
        t0 = time.time()
        result = await svc.search(query, pages=pages, top_k=top_k, rerank=True)
        elapsed = time.time() - t0
        timings.append(elapsed)
        result_counts.append(result.total_results)
        print(f" {elapsed:.2f}s  ({result.total_results} results, {result.search_time_ms:.0f}ms)")

    return {
        "query": query,
        "pages": pages,
        "top_k": top_k,
        "runs": runs,
        "time_mean": round(statistics.mean(timings), 3),
        "time_min": round(min(timings), 3),
        "time_max": round(max(timings), 3),
        "results_mean": round(statistics.mean(result_counts), 1),
        "results_min": min(result_counts),
        "results_max": max(result_counts),
    }


async def main():
    parser = argparse.ArgumentParser(description="Profile search_web performance")
    parser.add_argument("--pages", type=int, default=3, help="Pages per search")
    parser.add_argument("--top-k", type=int, default=20, help="Max results")
    parser.add_argument("--runs", type=int, default=3, help="Runs per query")
    args = parser.parse_args()

    print("=" * 60)
    print("SEARCH PERFORMANCE PROFILE")
    print(f"  pages={args.pages}  top_k={args.top_k}  runs={args.runs}")
    print("=" * 60)

    all_stats = []
    for query in QUERIES:
        print(f"\nQuery: \"{query}\"")
        print("-" * 40)
        stats = await profile_search(query, pages=args.pages, top_k=args.top_k, runs=args.runs)
        all_stats.append(stats)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Query':<35} {'Mean':>8} {'Min':>8} {'Max':>8} {'Results':>8}")
    print("-" * 70)
    for s in all_stats:
        q = s["query"][:33] + ".." if len(s["query"]) > 35 else s["query"]
        print(f"{q:<35} {s['time_mean']:>7.2f}s {s['time_min']:>7.2f}s {s['time_max']:>7.2f}s {s['results_mean']:>8.1f}")

    # Overall
    all_means = [s["time_mean"] for s in all_stats]
    print(f"\nOverall average: {statistics.mean(all_means):.2f}s across {len(all_stats)} queries")


if __name__ == "__main__":
    asyncio.run(main())
