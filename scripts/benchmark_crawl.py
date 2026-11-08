import asyncio
import time
import httpx
import psutil
import os
from findex.crawler import crawl, CrawlStats


async def measure(concurrency: int):
    process = psutil.Process(os.getpid())
    headers = {"User-Agent": "FindexCrawler/1.0 (+https://github.com/dasha643/findex)"}

    limits = httpx.Limits(max_connections=100, max_keepalive_connections=100)

    start_wall = time.perf_counter()
    mem_before = process.memory_info().rss

    stats = CrawlStats()
    async with httpx.AsyncClient(headers=headers, limits=limits) as client:
        async for _ in crawl(
                start_url="http://127.0.0.1:8001/page0.html",
                client=client,
                max_pages=200,
                concurrency=concurrency,
                max_depth=100,
                stats=stats,
        ):
            pass

    wall_time = time.perf_counter() - start_wall
    mem_after = process.memory_info().rss
    peak_rss = max(0, mem_after - mem_before) / 1024 / 1024

    return {
        "concurrency": concurrency,
        "wall_time": wall_time,
        "pages_per_sec": stats.pages_fetched / wall_time if wall_time > 0 else 0,
        "errors": stats.errors,
        "rss_mb": peak_rss,
        "pages_fetched": stats.pages_fetched
    }


async def main():
    print(f"{'Concurrency':>12} | {'Wall (s)':>8} | {'Pages/s':>8} | {'Errors':>6} | {'RSS (MB)':>8} | {'Fetched':>7}")
    print("-" * 70)

    for c in [1, 5, 20]:
        r = await measure(c)
        print(
            f"{r['concurrency']:>12} | {r['wall_time']:>8.2f} | {r['pages_per_sec']:>8.2f} | {r['errors']:>6} | {r['rss_mb']:>8.1f} | {r['pages_fetched']:>7}")


if __name__ == "__main__":
    asyncio.run(main())