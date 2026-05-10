"""Tests for search service - result processing, deduplication, cleaning"""

import pytest
from src.services.search_service import UnifiedSearchService
from src.models.unified import SearchResult


class TestSearchResultDeduplication:
    """Test result deduplication"""

    @pytest.fixture
    def search_service(self):
        return UnifiedSearchService()

    def test_deduplicate_removes_duplicates(self, search_service):
        """Remove duplicate URLs"""
        results = [
            SearchResult(
                title="Page 1",
                url="https://example.com/page",
                snippet="Content 1",
                domain="example.com"
            ),
            SearchResult(
                title="Page 2",
                url="https://example.com/page",  # Duplicate URL
                snippet="Content 2",
                domain="example.com"
            ),
        ]

        unique = search_service._deduplicate(results)
        assert len(unique) == 1
        assert unique[0].url == "https://example.com/page"

    def test_deduplicate_keeps_first_occurrence(self, search_service):
        """Keep first occurrence of duplicate URLs"""
        results = [
            SearchResult(
                title="First",
                url="https://example.com/page",
                snippet="First content",
                domain="example.com"
            ),
            SearchResult(
                title="Second",
                url="https://example.com/page",
                snippet="Second content",
                domain="example.com"
            ),
        ]

        unique = search_service._deduplicate(results)
        assert unique[0].title == "First"

    def test_deduplicate_handles_empty_list(self, search_service):
        """Handle empty list"""
        assert search_service._deduplicate([]) == []

    def test_deduplicate_handles_no_duplicates(self, search_service):
        """Handle list with no duplicates"""
        results = [
            SearchResult(
                title="Page 1",
                url="https://example.com/page1",
                snippet="Content 1",
                domain="example.com"
            ),
            SearchResult(
                title="Page 2",
                url="https://example.com/page2",
                snippet="Content 2",
                domain="example.com"
            ),
        ]

        unique = search_service._deduplicate(results)
        assert len(unique) == 2


class TestTextCleaning:
    """Test text cleaning utilities"""

    @pytest.fixture
    def search_service(self):
        return UnifiedSearchService()

    def test_clean_removes_extra_whitespace(self, search_service):
        """Remove extra whitespace"""
        assert search_service._clean_text("hello     world") == "hello world"
        assert search_service._clean_text("test   \n\n  text") == "test text"

    def test_clean_removes_unicode_artifacts(self, search_service):
        """Remove common unicode artifacts"""
        text_with_artifacts = "hello\u2026world\u00a0test"
        cleaned = search_service._clean_text(text_with_artifacts)
        assert "\u2026" not in cleaned
        assert "\u00a0" not in cleaned

    def test_clean_strips_leading_trailing(self, search_service):
        """Strip leading and trailing whitespace"""
        assert search_service._clean_text("  hello world  ") == "hello world"

    def test_clean_empty_string(self, search_service):
        """Handle empty string"""
        assert search_service._clean_text("") == ""
        assert search_service._clean_text("   ") == ""
