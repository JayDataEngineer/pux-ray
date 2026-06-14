"""Tests for scraper utilities - checkpoint detection, block detection, etc."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.scrapers.base import (
    is_security_checkpoint,
    is_low_quality_response,
    detect_blocking,
    normalize_reddit_url,
    scrape_httpx,
)


class TestSecurityCheckpointDetection:
    """Test security checkpoint detection"""

    def test_checkpoint_in_title_english(self):
        """Detect checkpoint in English title"""
        assert is_security_checkpoint("Security Checkpoint", "Some content") is True
        assert is_security_checkpoint("Verifying Your Browser", "Content") is True
        assert is_security_checkpoint("Browser Verification", "Content") is True
        assert is_security_checkpoint("Access Verification", "Content") is True

    def test_checkpoint_in_title_german(self):
        """Detect German checkpoint title (exact pattern)"""
        # Only the exact German pattern from the code
        assert is_security_checkpoint("Wir überprüfen Ihren Browser", "Content") is True
        # "Sicherheitsprüfung" alone is not in the pattern list, so it returns None
        result = is_security_checkpoint("Sicherheitsprüfung", "Content")
        assert result is None or result is False

    def test_checkpoint_in_content(self):
        """Detect checkpoint indicators in content"""
        content = "Please wait while we verify your browser"
        assert is_security_checkpoint("Normal Title", content) is True

        content_with_vercel = "See https://vercel.link/security-checkpoint"
        assert is_security_checkpoint("Title", content_with_vercel) is True

        content_with_cloudflare = "Cloudflare challenge detected"
        assert is_security_checkpoint("Title", content_with_cloudflare) is True

    def test_checkpoint_on_docs_url_with_short_content(self):
        """Detect checkpoint on documentation URL with suspiciously short content"""
        url = "https://example.com/docs/api"
        short_content = "Please verify you are human"
        assert is_security_checkpoint("Title", short_content, url) is True

    def test_no_checkpoint_on_valid_content(self):
        """Don't flag valid content as checkpoint"""
        assert is_security_checkpoint("API Documentation", """
        This is the API documentation.
        It has plenty of content here.
        Methods: GET, POST, PUT, DELETE
        """) is None

    def test_checkpoint_returns_true_not_false(self):
        """Checkpoint detection returns True, not False"""
        result = is_security_checkpoint("Security Checkpoint", "content")
        assert result is True


class TestLowQualityResponseDetection:
    """Test low quality response detection"""

    def test_empty_content(self):
        """Detect empty content"""
        assert is_low_quality_response("", "https://example.com") == "Blocked: Empty or near-empty response"
        assert is_low_quality_response("   ", "https://example.com") == "Blocked: Empty or near-empty response"

    def test_very_short_content(self):
        """Detect suspiciously short content"""
        assert is_low_quality_response("hi", "https://example.com") == "Blocked: Empty or near-empty response"

    def test_short_content_on_docs_url(self):
        """Short content on docs URL is suspicious"""
        # 200 chars on docs URL should be flagged
        short = "x" * 200
        result = is_low_quality_response(short, "https://example.com/docs/api")
        assert result is not None

    def test_valid_content_passes(self):
        """Valid content passes check"""
        long_content = "x" * 500
        assert is_low_quality_response(long_content, "https://example.com") is None


class TestBlockDetection:
    """Test blocking pattern detection"""

    def test_detect_captcha(self):
        """Detect CAPTCHA blocks"""
        result = detect_blocking("Please complete the CAPTCHA challenge", 200)
        assert result == "Blocked: CAPTCHA challenge detected"

    def test_detect_rate_limit(self):
        """Detect rate limiting via status code"""
        result = detect_blocking("content", 429)
        assert result == "Rate limited: Too many requests"

        # Also detect via content pattern
        result = detect_blocking("Too many requests", 200)
        assert "Rate limited" in result

    def test_detect_blocked(self):
        """Detect access blocked"""
        result = detect_blocking("Access denied", 200)
        assert result == "Blocked: Access denied"

    def test_detect_403_forbidden(self):
        """Detect HTTP 403"""
        result = detect_blocking("content", 403)
        assert result == "Blocked: HTTP 403 Forbidden"

    def test_detect_checkpoint(self):
        """Detect security checkpoint"""
        result = detect_blocking("Security checkpoint verification required", 200)
        assert "checkpoint" in result.lower()

    def test_no_block_on_valid_content(self):
        """Valid content is not flagged"""
        result = detect_blocking("Welcome to our website", 200)
        assert result is None

    def test_http_status_errors(self):
        """Detect HTTP error status codes (404 is not handled, returns None)"""
        # 404 is not explicitly handled, so it returns None
        result = detect_blocking("Not found", 404)
        assert result is None

        # 500 is handled
        result = detect_blocking("Server error", 500)
        assert "500" in result


class TestRedditNormalization:
    """Test Reddit URL normalization"""

    def test_normalize_reddit_adds_json(self):
        """Add .json to Reddit URLs"""
        result = normalize_reddit_url("https://www.reddit.com/r/python")
        # Normalizes to www and adds .json
        assert ".json" in result
        assert "www.reddit.com" in result

    def test_normalize_old_reddit(self):
        """Normalize old.reddit.com URLs"""
        result = normalize_reddit_url("https://old.reddit.com/r/python")
        assert result == "https://www.reddit.com/r/python.json"

    def test_normalize_new_reddit(self):
        """Normalize new.reddit.com URLs"""
        result = normalize_reddit_url("https://new.reddit.com/r/python")
        assert result == "https://www.reddit.com/r/python.json"

    def test_already_has_json(self):
        """Don't double-add .json"""
        url = "https://www.reddit.com/r/python.json"
        assert normalize_reddit_url(url) == url

    def test_remove_trailing_slash(self):
        """Remove trailing slash before adding .json"""
        result = normalize_reddit_url("https://www.reddit.com/r/python/")
        assert result == "https://www.reddit.com/r/python.json"


SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head><title>Hello World</title></head>
<body>
<article>
<h1>Hello World</h1>
<p>This is a test article with enough content to pass the minimum length threshold.
It has multiple sentences covering various topics. The quick brown fox jumps over
the lazy dog. Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
</article>
</body>
</html>"""


def _make_mock_response(html: str, status: int = 200, content_type: str = "text/html"):
    resp = MagicMock()
    resp.status_code = status
    resp.text = html
    resp.headers = {"content-type": content_type}
    resp.raise_for_status = MagicMock()
    return resp


def _make_mock_cleaner(output: str = "cleaned content that is long enough to pass"):
    cleaner = MagicMock()
    cleaner.clean = MagicMock(return_value=output)
    return cleaner


class TestScrapeHttpx:
    """Tests for the httpx-based scraper (no browser required)."""

    @pytest.mark.asyncio
    async def test_success_via_trafilatura(self):
        """httpx scraper returns content using trafilatura extraction."""
        mock_resp = _make_mock_response(SAMPLE_HTML)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("src.utils.proxy.create_proxied_client", return_value=mock_client):
            result = await scrape_httpx("https://example.com", _make_mock_cleaner())

        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert len(result.get("content", "")) > 0

    @pytest.mark.asyncio
    async def test_falls_back_to_content_cleaner(self):
        """Falls back to ContentCleaner when trafilatura extracts nothing."""
        minimal_html = "<html><body><p>x</p></body></html>"
        mock_resp = _make_mock_response(minimal_html)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        long_output = "cleaner output " * 20
        cleaner = _make_mock_cleaner(long_output)

        with patch("src.utils.proxy.create_proxied_client", return_value=mock_client):
            with patch("trafilatura.extract", return_value=None):
                result = await scrape_httpx("https://example.com", cleaner)

        assert result["success"] is True
        assert result["content"] == long_output.strip()

    @pytest.mark.asyncio
    async def test_non_html_content_type_fails(self):
        """Non-HTML content type returns error without crashing."""
        mock_resp = _make_mock_response("<data/>", content_type="application/json")
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("src.utils.proxy.create_proxied_client", return_value=mock_client):
            result = await scrape_httpx("https://example.com/api", _make_mock_cleaner())

        assert result["success"] is False
        assert "Not HTML" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_http_error_returns_failure(self):
        """HTTP 403 returns error result without raising."""
        import httpx as _httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        error_resp = MagicMock()
        error_resp.status_code = 403
        http_err = _httpx.HTTPStatusError("403", request=MagicMock(), response=error_resp)
        mock_client.get = AsyncMock(side_effect=http_err)

        with patch("src.utils.proxy.create_proxied_client", return_value=mock_client):
            result = await scrape_httpx("https://blocked.example.com", _make_mock_cleaner())

        assert result["success"] is False
        assert "403" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_timeout_returns_failure(self):
        """Network timeout returns error result without raising."""
        import httpx as _httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=_httpx.TimeoutException("timeout"))

        with patch("src.utils.proxy.create_proxied_client", return_value=mock_client):
            result = await scrape_httpx("https://slow.example.com", _make_mock_cleaner())

        assert result["success"] is False
        assert "timed out" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_security_checkpoint_detected(self):
        """Security checkpoint pages return failure, not content."""
        # Must have enough text to pass MIN_CONTENT_LENGTH so the checkpoint
        # check runs (rather than early-exiting with "content too short").
        checkpoint_html = """<html><head><title>Security Checkpoint</title></head>
<body>
<h1>Security Checkpoint</h1>
<p>Please wait while we verify your browser before granting access to this site.
This security checkpoint is required to protect the website from automated bots.
Wir überprüfen Ihren Browser bevor wir Ihnen Zugang gewähren. Bitte warten Sie.</p>
</body></html>"""
        mock_resp = _make_mock_response(checkpoint_html)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        long_cleaner_out = "security checkpoint " * 10
        with patch("src.utils.proxy.create_proxied_client", return_value=mock_client):
            result = await scrape_httpx("https://example.com", _make_mock_cleaner(long_cleaner_out))

        assert result["success"] is False
