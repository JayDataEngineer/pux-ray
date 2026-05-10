"""Pytest configuration and fixtures"""

import asyncio
import os
import sys
import pytest
from pathlib import Path

# Add src to path (conftest is in tests/pytest/, so we need to go up to project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_html():
    """Sample HTML content for testing"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page Title</title>
        <meta name="description" content="Test description">
    </head>
    <body>
        <h1>Main Heading</h1>
        <p>This is a paragraph with some text.</p>
        <div class="content">
            <h2>Sub Heading</h2>
            <p>More content here.</p>
            <a href="/page1">Link 1</a>
            <a href="https://example.com/page2">External Link</a>
        </div>
        <nav>
            <ul>
                <li><a href="/home">Home</a></li>
                <li><a href="/about">About</a></li>
            </ul>
        </nav>
        <footer>
            <p>Copyright 2024</p>
        </footer>
    </body>
    </html>
    """


@pytest.fixture
def checkpoint_html():
    """HTML simulating a security checkpoint page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Security Checkpoint - Verifying Your Browser</title>
    </head>
    <body>
        <h1>Security Checkpoint</h1>
        <p>Please wait while we verify your browser...</p>
        <p>Wir überprüfen Ihren Browser</p>
        <script src="https://vercel.link/security-checkpoint"></script>
    </body>
    </html>
    """


@pytest.fixture
def block_responses():
    """Various blocking/error responses"""
    return {
        "captcha": "<title>CAPTCHA Verification</title><p>Prove you are human</p>",
        "rate_limit": "<title>429 Too Many Requests</title><p>Rate limit exceeded</p>",
        "blocked": "<title>Access Denied</title><p>You have been blocked</p>",
        "cloudflare": "<title>Just a moment...</title><p>Checking your browser...</p>",
    }
