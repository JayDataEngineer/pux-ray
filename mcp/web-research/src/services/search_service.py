"""Unified search service - SearXNG multi-engine"""

import httpx
import re
from loguru import logger
from urllib.parse import urlparse

from ..models.unified import SearchResult, CombinedSearchResponse
from ..core.constants import DEFAULT_SEARCH_ENGINES, HTTP_REQUEST_TIMEOUT
from ..utils.proxy import create_proxied_client


class UnifiedSearchService:
    """SearXNG multi-engine search with unified output format"""

    def __init__(self, searxng_url: str = "http://searxng:8080"):
        self.searxng_url = searxng_url
        # SearXNG is an internal service — pass target_url so exclusion kicks in
        self.client = create_proxied_client(
            timeout=HTTP_REQUEST_TIMEOUT,
            target_url=searxng_url,
        )
        self._db = None

    async def _get_db(self):
        """Lazy database initialization — returns None if unavailable."""
        if self._db is None:
            from ..db.database import get_db
            try:
                self._db = await get_db()
            except Exception:
                from loguru import logger
                logger.warning("Search service: database unavailable, skipping blacklist filtering")
                return None
        return self._db

    async def search(
        self,
        query: str,
        pages: int = 1,
        exclude_blacklist: bool = True,
        top_k: int | None = None,
        rerank: bool = False,
        time_filter: str | None = None
    ) -> CombinedSearchResponse:
        """Search SearXNG with pagination, blacklist filtering, and re-ranking

        Args:
            query: Search query
            pages: Number of pages to fetch
            exclude_blacklist: Filter out blacklisted domains
            top_k: Maximum number of results to return (None = all results)
            rerank: Apply flash re-ranking based on query relevance
            time_filter: Filter results by time (day, week, month, year)
        """
        import asyncio

        start_time = asyncio.get_event_loop().time()

        # Get blacklist if needed
        blacklisted = set()
        if exclude_blacklist:
            db = await self._get_db()
            blacklisted = await db.get_blacklisted_domains()
            logger.info(f"Blacklisted domains: {blacklisted}")

        results = await self._fetch_results(query, pages, blacklisted, time_filter)
        unique_results = self._deduplicate(results)

        # Apply flash re-ranking if requested
        if rerank and unique_results:
            unique_results = self._flash_rerank(query, unique_results)
            logger.info(f"Re-ranked {len(unique_results)} results")

        # Apply top_k limit
        if top_k is not None and top_k > 0:
            unique_results = unique_results[:top_k]

        search_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        return CombinedSearchResponse(
            query=query,
            total_results=len(unique_results),
            pages_scraped=pages,
            results=unique_results,
            engines={"searxng": len(results)},
            search_time_ms=round(search_time_ms, 2)
        )

    # Shared sets for flash_rerank to prevent repeated allocation
    _STOP_WORDS = frozenset({"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                 "of", "with", "by", "from", "as", "is", "was", "are", "be", "been",
                 "this", "that", "these", "those", "it", "its", "what", "which", "who"})

    # +40: Official documentation / high-trust structural patterns
    _AUTHORITATIVE_PATTERNS = (
        "docs.", "developer.", ".edu", ".gov",
        "developer.mozilla.org", "python.org", "nodejs.org",
        "pkg.go.dev", "docs.rs", "crates.io", "npmjs.com",
        "pypi.org", "docs.docker.com", "kubernetes.io",
        "cloud.google.com", "aws.amazon.com", "learn.microsoft.com",
        "react.dev", "nextjs.org", "vuejs.org", "angular.io",
        "tailwindcss.com", "fastapi.tiangolo.com", "flask.palletsprojects.com",
        "django.readthedocs.io", "readthedocs.io",
        "postgresql.org", "sqlite.org", "mongodb.com/docs",
    )

    # +20: Community references (trusted but not official docs)
    _COMMUNITY_PATTERNS = (
        "stackoverflow.com", "github.com", "wikipedia.org",
        "realpython.com", "digitalocean.com/community",
        "freecodecamp.org",
    )

    # -80: Known SEO slop farms (heavy penalty to overcome high SearXNG ranks)
    _SLOP_DOMAINS = frozenset({
        "geeksforgeeks.org", "w3schools.com", "tutorialspoint.com",
        "javatpoint.com", "programiz.com", "educba.com",
        "studytonight.com", "beginnersbook.com", "journaldev.com",
        "baeldung.com", "howtodoinjava.com",
        "quora.com", "answers.com", "stackshare.io",
        "ask.com", "chegg.com", "consumersearch.com",
        "questionsanswered.net",
        "zhihu.com",  # Chinese-language Q&A, useless for English queries
    })

    # -25: Mixed-quality content mills
    _CONTENT_MILL_DOMAINS = frozenset({
        "medium.com", "dev.to", "hackernoon.com",
        "towardsdatascience.com",  # Cookie walls, low content yield
    })

    # -30: SEO slug patterns in URL paths
    _SEO_URL_PATTERNS = re.compile(
        r"/(?:best|top|ultimate|complete|step.by.step|tutorial|guide|explained|vs|comparison|review|updated|2024|2025|2026)",
        re.I,
    )

    # -20: Excessive dashes in domain (keyword stuffing)
    _SEO_DOMAIN_PATTERN = re.compile(r"(?:[a-z]-){4,}", re.I)

    def _quality_score(self, url: str, domain: str) -> float:
        """Score a URL/domain for content quality based on patterns.

        Returns a modifier to add to the base score:
        +40 for authoritative sources (official docs, .edu, .gov)
        +20 for community references (SO, GitHub, Wikipedia)
        -80 for known SEO slop farms
        -25 for mixed-quality content mills
        -30 for SEO-optimized URL slugs
        -20 for keyword-stuffed domains
        """
        score = 0.0
        domain_lower = domain.lower()

        # +40: Authoritative patterns (official docs)
        for pattern in self._AUTHORITATIVE_PATTERNS:
            if pattern in domain_lower:
                score += 40
                break

        # +20: Community references (only if not already matched authoritative)
        if score == 0.0:
            for pattern in self._COMMUNITY_PATTERNS:
                if pattern in domain_lower:
                    score += 20
                    break

        # -80: Known slop farms
        for slop in self._SLOP_DOMAINS:
            if slop in domain_lower or domain_lower.endswith(slop):
                score -= 80
                break

        # -25: Content mills (mixed quality, often SEO-heavy)
        for mill in self._CONTENT_MILL_DOMAINS:
            if mill in domain_lower:
                score -= 25
                break

        # -30: SEO-optimized URL patterns
        if self._SEO_URL_PATTERNS.search(url):
            score -= 30

        # -20: Excessive dashes in domain
        if self._SEO_DOMAIN_PATTERN.search(domain_lower):
            score -= 20

        return score

    def _flash_rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Re-rank search results using position + quality + relevance scoring.

        Scoring:
        1. Base: SearXNG rank position (rank 1=100, rank 2=95, ...)
        2. Plus: SearXNG native engine consensus score (scaled x10)
        3. Plus: Quality modifier (+40 official docs, +20 community, -80 slop, etc.)
        4. Tiebreaker: Keyword relevance (title match x2, snippet match x1)

        Example:
          docs.docker.com (rank 5): base 80 + native*10 + 40 (authoritative) = ~130
          astconsulting.in (rank 1): base 100 + native*10 - 30 (SEO URL) = ~70
          → docs.docker.com jumps to #1
        """
        # Extract query terms for keyword relevance tiebreaker
        query_lower = query.lower()
        query_terms = [w for w in re.findall(r"\b\w+\b", query_lower)
                       if w not in self._STOP_WORDS and len(w) > 1]

        scored_results = []
        for rank, result in enumerate(results):
            # 1. Position base score: rank 1=100, rank 2=95, etc.
            position_score = max(0, 100 - rank * 5)

            # 2. SearXNG native engine consensus (scaled to comparable range)
            native_score = result.score * 10

            # 3. Quality modifier from URL/domain patterns
            quality = self._quality_score(result.url, result.domain)

            # 4. Keyword relevance tiebreaker
            relevance = 0.0
            if query_terms:
                title_lower = result.title.lower()
                snippet_lower = result.snippet.lower()
                title_len = len(title_lower)
                snippet_len = len(snippet_lower)

                for term in query_terms:
                    if title_len > 0:
                        pos = title_lower.find(term)
                        if pos != -1:
                            relevance += 2.0 * (1.0 - (pos / title_len) * 0.5)
                    if snippet_len > 0:
                        pos = snippet_lower.find(term)
                        if pos != -1:
                            relevance += 1.0 * (1.0 - (pos / snippet_len) * 0.3)

            final_score = position_score + native_score + quality + relevance
            scored_results.append((result, final_score))

        # Sort by final score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)

        if len(scored_results) > 1:
            top = scored_results[0]
            logger.info(
                f"Re-rank: top={top[0].domain} ({top[1]:.1f}), "
                f"results={len(scored_results)}"
            )

        return [r for r, s in scored_results]

    async def _fetch_results(self, query: str, pages: int, blacklisted: set[str], time_filter: str | None = None) -> list[SearchResult]:
        """Fetch results from SearXNG with parallel pagination

        Args:
            query: Search query
            pages: Number of pages to fetch
            blacklisted: Set of blacklisted domains to filter out
            time_filter: Optional time range filter (day, week, month, year)
        """
        import asyncio

        async def _fetch_page(page: int) -> list[SearchResult]:
            params = {
                "q": query,
                "format": "json",
                "pageno": page,
                "engines": ",".join(DEFAULT_SEARCH_ENGINES)
            }

            if time_filter:
                params["time_range"] = time_filter

            try:
                response = await self.client.get(f"{self.searxng_url}/search", params=params)
                response.raise_for_status()
                data = response.json()

                page_results = []
                for item in data.get("results", []):
                    url = item.get("url", "")
                    domain = urlparse(url).netloc or urlparse(url).path

                    if domain in blacklisted:
                        continue

                    page_results.append(SearchResult(
                        title=self._clean_text(item.get("title", "")),
                        url=url,
                        snippet=self._clean_text(item.get("content", "")),
                        domain=domain,
                        score=float(item.get("score", 0)),
                    ))
                return page_results

            except httpx.HTTPStatusError as e:
                logger.warning(f"SearXNG HTTP error on page {page}: {e.response.status_code}")
            except Exception as e:
                logger.warning(f"SearXNG error on page {page}: {e}")
            return []

        page_results = await asyncio.gather(*[_fetch_page(p) for p in range(1, pages + 1)])
        results = []
        for batch in page_results:
            results.extend(batch)
        return results

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate results by URL"""
        # Python 3.7+ dictionaries maintain insertion order
        seen = {}
        for r in results:
            if r.url not in seen:
                seen[r.url] = r
        return list(seen.values())

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Remove common artifacts and extra whitespace
        return " ".join(text.replace("\u2026", " ").replace("\u00a0", " ").replace("\u200b", " ").split())

    async def close(self):
        await self.client.aclose()


# Singleton factory
_search_service: UnifiedSearchService | None = None


def get_search_service() -> UnifiedSearchService:
    global _search_service
    if _search_service is None:
        _search_service = UnifiedSearchService()
    return _search_service
