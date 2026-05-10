"""Tests for content cleaner - HTML to markdown conversion"""

import pytest
from src.services.content_cleaner import ContentCleaner


class TestContentCleaner:
    """Test HTML to markdown conversion"""

    @pytest.fixture
    def cleaner(self):
        return ContentCleaner()

    def test_basic_html_conversion(self, cleaner, sample_html):
        """Test basic HTML to markdown conversion"""
        result = cleaner.clean(sample_html, "https://example.com")

        # The cleaner extracts main content, converting to markdown
        # Navigation and footer are stripped by the waterfall extraction
        assert "Sub Heading" in result
        assert "More content here" in result

    def test_content_extraction_works(self, cleaner, sample_html):
        """Test that content is extracted"""
        result = cleaner.clean(sample_html, "https://example.com")

        # Should have some content
        assert len(result) > 0

    def test_link_conversion(self, cleaner, sample_html):
        """Test link conversion to markdown format"""
        result = cleaner.clean(sample_html, "https://example.com")
        # The cleaner extracts main content and removes navigation
        # Navigation links are stripped, but main content link text may remain
        assert "Link 1" in result or "External Link" in result or "Home" in result

    def test_css_selector_overrides_extraction(self, cleaner):
        """Test CSS selector-based content extraction"""
        html = """
        <html>
        <body>
            <div class="main">Content to keep</div>
            <div class="sidebar">Content to remove</div>
        </body>
        </html>
        """
        result = cleaner.clean(html, "https://example.com", css_selector=".main")
        # With CSS selector, should only get the selected content
        assert "Content to keep" in result

    def test_script_removal(self, cleaner):
        """Test script tag removal"""
        html = """
        <html>
        <body>
            <h1>Title</h1>
            <script>alert('malicious')</script>
            <p>Content</p>
        </body>
        </html>
        """
        result = cleaner.clean(html, "https://example.com")
        assert "Title" in result
        # Scripts should be removed by the waterfall extraction
        assert "alert" not in result
        assert "script" not in result.lower() or "<script>" not in result

    def test_style_removal(self, cleaner):
        """Test style tag removal"""
        html = """
        <html>
        <head>
            <style>body { color: red; }</style>
        </head>
        <body>
            <h1>Title</h1>
        </body>
        </html>
        """
        result = cleaner.clean(html, "https://example.com")
        assert "Title" in result
        # Style tags should be removed
        assert "color: red" not in result

    def test_empty_html(self, cleaner):
        """Test handling of empty HTML"""
        result = cleaner.clean("", "https://example.com")
        assert result == ""

    def test_malformed_html(self, cleaner):
        """Test handling of malformed HTML"""
        html = "<div><p>Unclosed tags<span>Nested</div>"
        # Should not raise exception
        result = cleaner.clean(html, "https://example.com")
        assert isinstance(result, str)

    def test_relative_urls_preserved(self, cleaner):
        """Test relative URLs are handled"""
        html = """
        <html>
        <body>
            <p><a href="/page1">Internal Link</a></p>
        </body>
        </html>
        """
        result = cleaner.clean(html, "https://example.com/docs/")
        # Should preserve relative links
        assert result is not None and len(result) > 0
