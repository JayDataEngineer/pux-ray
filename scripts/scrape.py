#!/usr/bin/env python3
"""
SOTA waterfall scraper — never-fails content extraction.

Tiers cascade independently; each has its own timeout and failure mode.
HTTP-first (cheap, robust) → browser-last (expensive, fragile).
Picks the BEST content by quality score, not just the first success.

Usage:
  scrape.py <url> [--json] [--selector CSS] [--no-browser] [--timeout 30]
  scrape.py https://docs.sglang.io/docs/sglang-diffusion
  cat urls.txt | xargs -P8 -I{} scrape.py {} --json

Exit codes: 0=success (any tier), 1=all tiers exhausted, 2=bad args.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys, time
from dataclasses import dataclass, field, asdict
from urllib.parse import urlsplit, urljoin, urlparse

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class Result:
    url: str
    final_url: str = ""
    title: str = ""
    content: str = ""
    method: str = ""
    tier: int = -1
    success: bool = False
    elapsed_ms: int = 0
    error: str = ""
    attempts: list = field(default_factory=list)

    def score(self) -> int:
        """Content quality score — longer meaningful text wins."""
        if not self.content:
            return 0
        text = re.sub(r"\s+", " ", self.content).strip()
        # reject browser/connection error pages masquerading as content
        low = text.lower()[:400]
        if any(s in low for s in ("can't be reached", "err_name_not_resolved",
                "err_connection_refused", "err_internet_disconnected",
                "this site can't be reached", "unable to connect",
                "502 bad gateway", "503 service temporarily", "404 not found",
                "err_timed_out", "err_name", "dns_probe_finished_nxdomain")):
            return 0
        # reward length, penalize tag soup / css leakage
        tag_ratio = (len(re.findall(r"[<>]", self.content)) / max(len(self.content), 1))
        return int(len(text) * (1.0 - min(tag_ratio, 0.9)))

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _curl(url: str, timeout: int, extra=None) -> tuple[int, str, str]:
    """Fetch URL via curl. Returns (http_code, body, final_url)."""
    args = ["curl", "-sSL", "--compressed", "--max-time", str(timeout),
            "-A", UA, "-w", "\n__HTTP_CODE__%{http_code}\n__FINAL_URL__%{url_effective}",
            "-H", "Accept-Language: en-US,en;q=0.9"]
    for k, v in HEADERS.items():
        if k != "User-Agent":
            args += ["-H", f"{k}: {v}"]
    if extra:
        args += extra
    args.append(url)
    out = subprocess.run(args, capture_output=True, text=True, timeout=timeout + 5)
    body = out.stdout
    code = 0; final = url
    m = re.search(r"__HTTP_CODE__(\d+)", body)
    if m:
        code = int(m.group(1)); body = body[:m.start()]
    m = re.search(r"__FINAL_URL__(.+)", body)
    if m:
        final = m.group(1).strip(); body = body[:m.start()]
    return code, body, final

def _strip_html(html: str) -> tuple[str, str]:
    """Crude last-resort HTML→text. Returns (title, text)."""
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
    for pat in (r"<script.*?</script>", r"<style.*?</style>", r"<nav.*?</nav>",
                r"<footer.*?</footer>", r"<header.*?</header>"):
        html = re.sub(pat, "", html, flags=re.I | re.S)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    import html as ihtml
    return title, re.sub(r"[ \t]+", " ", ihtml.unescape(html)).strip()

# ---------------------------------------------------------------------------
# Tier 0: /llms.txt + .md shortcut (docs domains — cleanest possible source)
# ---------------------------------------------------------------------------
def tier_llmstxt(url: str, timeout: int) -> Result:
    r = Result(url=url, method="llms.txt", tier=0)
    t0 = time.time()
    try:
        # Many docs platforms (Mintlify, docs.rs, Fumadocs) serve raw MD at URL + ".md"
        for candidate in (url.rstrip("/") + ".md", url.rstrip("/") + "/index.md"):
            code, body, final = _curl(candidate, timeout)
            if code == 200 and body.strip().startswith(("#", "---", "1", "-")):
                r.content = body.strip()
                r.title = re.split(r"\n+", body)[0].lstrip("# ").strip()
                r.final_url = final; r.success = True; break
    except Exception as e:
        r.error = str(e)
    r.elapsed_ms = int((time.time() - t0) * 1000)
    return r

# ---------------------------------------------------------------------------
# Tier 1: trafilatura (SOTA boilerplate-stripping content extractor, HTTP-only)
# ---------------------------------------------------------------------------
def tier_trafilatura(url: str, timeout: int) -> Result:
    import trafilatura
    r = Result(url=url, method="trafilatura", tier=1); t0 = time.time()
    try:
        code, html, final = _curl(url, timeout)
        if code != 200 or not html:
            r.error = f"http {code}"; return r
        r.final_url = final
        extracted = trafilatura.extract(
            html, include_links=True, include_tables=True,
            include_images=True, output_format="markdown", favor_recall=True)
        if extracted and len(extracted.strip()) > 80:
            r.content = extracted.strip()
            meta = trafilatura.extract_metadata(html)
            r.title = (meta.title if meta else "") or ""
            r.success = True
        else:
            r.error = "extracted too short"
    except Exception as e:
        r.error = str(e)
    r.elapsed_ms = int((time.time() - t0) * 1000)
    return r

# ---------------------------------------------------------------------------
# Tier 2: readability-lxml + html2text (alternative extraction engine)
# ---------------------------------------------------------------------------
def tier_readability(url: str, timeout: int) -> Result:
    import html2text, readability
    r = Result(url=url, method="readability", tier=2); t0 = time.time()
    try:
        code, html, final = _curl(url, timeout)
        if code != 200 or not html:
            r.error = f"http {code}"; return r
        r.final_url = final
        doc = readability.Document(html)
        r.title = doc.short_title() or ""
        article = doc.summary()
        h = html2text.HTML2Text()
        h.body_width = 0; h.ignore_images = False; h.ignore_links = False
        r.content = h.handle(article).strip()
        if len(r.content) > 80:
            r.success = True
        else:
            r.error = "readability too short"
    except Exception as e:
        r.error = str(e)
    r.elapsed_ms = int((time.time() - t0) * 1000)
    return r

# ---------------------------------------------------------------------------
# Tier 3: host Chromium headless --dump-dom (JS-rendered pages) + trafilatura
# ---------------------------------------------------------------------------
def tier_browser(url: str, timeout: int) -> Result:
    import trafilatura
    r = Result(url=url, method="chromium", tier=3); t0 = time.time()
    chrome = next((p for p in
                   ("/usr/bin/chromium-browser", "/snap/bin/chromium",
                    "/usr/bin/chromium", "/usr/bin/google-chrome")
                   if os.path.exists(p)), None)
    if not chrome:
        r.error = "no browser binary"; return r
    try:
        # Preflight: reject non-2xx (browser --dump-dom hides HTTP status, so check first).
        # Cheap HEAD-equivalent via curl; only launch the browser for real pages.
        code, _, _ = _curl(url, min(timeout, 10), extra=["-I", "-o", "/dev/null"])
        if code and not (200 <= code < 300):
            r.error = f"http {code}"; return r
        proc = subprocess.run(
            [chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
             "--disable-dev-shm-usage", "--disable-crash-reporter",
             "--virtual-time-budget=8000", "--timeout=%d" % (timeout * 1000),
             "--dump-dom", url],
            capture_output=True, text=True, timeout=timeout + 10)
        html = proc.stdout
        if not html or len(html) < 200:
            r.error = f"browser empty ({proc.returncode})"; return r
        extracted = trafilatura.extract(
            html, include_links=True, include_tables=True,
            output_format="markdown", favor_recall=True)
        if extracted and len(extracted.strip()) > 80:
            r.content = extracted.strip()
            meta = trafilatura.extract_metadata(html)
            r.title = (meta.title if meta else "") or ""
            r.success = True
        else:
            _, txt = _strip_html(html)
            if len(txt) > 80:
                r.content = txt; r.success = True
            else:
                r.error = "browser content too short"
    except subprocess.TimeoutExpired:
        r.error = "browser timeout"
    except Exception as e:
        r.error = str(e)
    r.elapsed_ms = int((time.time() - t0) * 1000)
    return r

# ---------------------------------------------------------------------------
# Tier 4: raw curl + regex strip (NEVER fails if the server responds at all)
# ---------------------------------------------------------------------------
def tier_raw(url: str, timeout: int) -> Result:
    r = Result(url=url, method="curl-raw", tier=4); t0 = time.time()
    try:
        code, html, final = _curl(url, timeout)
        if code != 200 or not html:
            r.error = f"http {code}"; return r
        r.final_url = final
        title, text = _strip_html(html)
        r.title = title; r.content = text; r.success = len(text) > 50
        if not r.success:
            r.error = "no text extracted"
    except Exception as e:
        r.error = str(e)
    r.elapsed_ms = int((time.time() - t0) * 1000)
    return r

# ---------------------------------------------------------------------------
# Orchestrator — waterfall with quality-scored selection
# ---------------------------------------------------------------------------
TIERS = [tier_llmstxt, tier_trafilatura, tier_readability, tier_browser, tier_raw]

def scrape(url: str, timeout: int = 25, no_browser: bool = False,
           selector: str | None = None) -> Result:
    tiers = TIERS[:-1] if no_browser else TIERS  # always keep raw last-resort
    attempts = []
    best = Result(url=url)
    for fn in tiers:
        if no_browser and fn is tier_browser:
            continue
        res = fn(url, timeout)
        attempts.append({"tier": res.tier, "method": res.method,
                         "success": res.success, "elapsed_ms": res.elapsed_ms,
                         "error": res.error,
                         "score": res.score() if res.success else -1})
        if res.success and res.score() > best.score():
            best = res
            best.attempts = attempts
            # Only early-return when a high-tier method produced substantial content.
            # Thin wins (SPAs that barely extracted) let the browser tier compete.
            if res.tier <= 1 and res.score() >= 1500:
                best.attempts = attempts
                return best
        # cap total work: stop once we have good content past the browser tier
        if best.success and best.score() >= 1500 and res.tier >= 3:
            break
    best.attempts = attempts
    return best

def main():
    ap = argparse.ArgumentParser(description="SOTA waterfall scraper")
    ap.add_argument("url"); ap.add_argument("--json", action="store_true")
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--selector", default=None)
    a = ap.parse_args()

    res = scrape(a.url, timeout=a.timeout, no_browser=a.no_browser, selector=a.selector)
    if a.json:
        print(json.dumps(asdict(res), indent=2))
    else:
        if res.success:
            if res.title:
                print(f"# {res.title}\n", file=sys.stderr)
            print(f"[via {res.method} · tier {res.tier} · {res.elapsed_ms}ms]\n",
                  file=sys.stderr)
            print(res.content)
        else:
            print(f"FAILED all tiers for {a.url}", file=sys.stderr)
            for att in res.attempts:
                print(f"  tier {att['tier']} {att['method']}: {att['error']}",
                      file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
