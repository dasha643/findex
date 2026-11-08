import pytest
import httpx
from findex.crawler import canonicalize, fetch_with_retry, RobotsCache

def test_canonicalize():
    assert canonicalize("HTTP://Example.com:80/path/?b=2&a=1#frag") == "http://example.com:80/path?a=1&b=2"
    assert canonicalize("https://test.com/") == "https://test.com/"

@pytest.mark.asyncio
async def test_retry_on_429():
    async def handler(request):
        if request.url.path == "/fail":
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, text="ok")
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp, size, elapsed = await fetch_with_retry(client, "http://test/fail", max_retries=2)
        assert resp.status_code == 429
        assert size == 0 # Тіло не повертається, бо статус 429

@pytest.mark.asyncio
async def test_robots_cache():
    async def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
        return httpx.Response(200, text="ok")
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        robots = RobotsCache(client)
        assert await robots.allowed("http://test/public") == True
        assert await robots.allowed("http://test/private") == False