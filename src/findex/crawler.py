import asyncio
import random
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode, urljoin
from urllib.robotparser import RobotFileParser

import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser


def canonicalize(url: str) -> str:
    """Приводить URL до єдиного вигляду."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    path = parsed.path
    if path.endswith('/'):
        path = path.rstrip('/')
    if not path:
        path = '/'

    query_params = parse_qs(parsed.query, keep_blank_values=True)
    sorted_query = urlencode(sorted(query_params.items()), doseq=True)

    return urlunparse((scheme, netloc, path, parsed.params, sorted_query, ''))


async def fetch_with_retry(
        client: httpx.AsyncClient,
        url: str,
        max_retries: int = 3,
        timeout_sec: float = 10.0,
) -> tuple[httpx.Response | None, int, float]:
    """Завантажує сторінку з логікою повторів."""
    for attempt in range(max_retries):
        try:
            async with asyncio.timeout(timeout_sec):
                start = asyncio.get_event_loop().time()
                resp = await client.get(url)
                elapsed = asyncio.get_event_loop().time() - start
                size = len(resp.content) if resp.content else 0

                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < max_retries - 1:
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            wait_time = float(retry_after)
                        else:
                            wait_time = (2 ** attempt) + random.uniform(0, 1)
                        await asyncio.sleep(wait_time)
                        continue

                return resp, size, elapsed

        except (asyncio.TimeoutError, httpx.RequestError):
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait_time)

    return None, 0, 0.0


class RobotsCache:
    """Кешує правила robots.txt для кожного хоста."""

    def __init__(self, client: httpx.AsyncClient, user_agent: str = "*"):
        self._client = client
        self._user_agent = user_agent
        self._cache: dict[str, RobotFileParser] = {}
        self._lock = asyncio.Lock()

    async def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"

        async with self._lock:
            if host not in self._cache:
                robots_url = urljoin(host, "/robots.txt")
                try:
                    async with asyncio.timeout(5.0):
                        resp = await self._client.get(robots_url)
                        if resp.status_code == 200:
                            rp = RobotFileParser()
                            rp.parse(resp.text.splitlines())
                            self._cache[host] = rp
                        else:
                            rp = RobotFileParser()
                            rp.parse([])
                            self._cache[host] = rp
                except Exception:
                    rp = RobotFileParser()
                    rp.parse([])
                    self._cache[host] = rp

            return self._cache[host].can_fetch(self._user_agent, url)


@dataclass
class CrawlStats:
    """Статистика для відображення прогресу."""
    pages_fetched: int = 0
    errors: int = 0
    queue_depth: int = 0
    active_requests: int = 0
    start_time: float = field(default_factory=lambda: asyncio.get_event_loop().time())

    @property
    def pages_per_sec(self) -> float:
        elapsed = asyncio.get_event_loop().time() - self.start_time
        return self.pages_fetched / elapsed if elapsed > 0 else 0.0


async def crawl(
        start_url: str,
        client: httpx.AsyncClient,
        max_pages: int = 200,
        concurrency: int = 10,
        max_depth: int = 3,
        stats: CrawlStats | None = None,
        user_agent: str = "*",
):
    """
    Асинхронний генератор. Віддає сторінки в міру їх завантаження (yield).
    """
    if stats is None:
        stats = CrawlStats()

    start_url = canonicalize(start_url)
    seen: set[str] = {start_url}

    to_visit: asyncio.Queue[tuple[str, int] | None] = asyncio.Queue()
    await to_visit.put((start_url, 0))

    global_semaphore = asyncio.Semaphore(concurrency)
    host_semaphores: dict[str, asyncio.Semaphore] = {}
    host_last_request: dict[str, float] = {}
    lock = asyncio.Lock()

    robots = RobotsCache(client, user_agent=user_agent)

    result_queue: asyncio.Queue[dict | None] = asyncio.Queue()
    pages_fetched = 0
    stop_event = asyncio.Event()

    async def process_url(url: str, depth: int):
        nonlocal pages_fetched

        if depth > max_depth or stop_event.is_set():
            return

        if not await robots.allowed(url):
            return

        host = urlparse(url).netloc.lower()
        if host not in host_semaphores:
            host_semaphores[host] = asyncio.Semaphore(concurrency)

        wait_time = 0
        async with lock:
            now = asyncio.get_event_loop().time()
            last_time = host_last_request.get(host, 0)
            wait_time = max(0, 0.01 - (now - last_time))
            host_last_request[host] = now + wait_time

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        async with global_semaphore, host_semaphores[host]:
            stats.active_requests += 1
            resp, size, elapsed = await fetch_with_retry(client, url, timeout_sec=10.0)
            stats.active_requests -= 1

            if resp is None or resp.status_code != 200:
                stats.errors += 1
                return

            tree = HTMLParser(resp.text)
            text = tree.body.text(separator=' ') if tree.body else ""

            links = []
            for a in tree.css('a[href]'):
                href = a.attributes.get('href')
                if not href:
                    continue
                if href.startswith(('#', 'mailto:', 'javascript:', 'tel:')):
                    continue

                full_url = urljoin(url, href)
                if not full_url.startswith('http'):
                    continue

                canonical_link = canonicalize(full_url)
                if canonical_link not in seen:
                    seen.add(canonical_link)
                    links.append(canonical_link)

            async with lock:
                pages_fetched += 1
                stats.pages_fetched = pages_fetched
                if pages_fetched >= max_pages:
                    stop_event.set()

            await result_queue.put({
                "url": url,
                "text": text,
                "links": links,
                "status": resp.status_code,
                "size": size,
                "elapsed": elapsed,
                "depth": depth,
            })

            if not stop_event.is_set():
                for link in links:
                    await to_visit.put((link, depth + 1))

    async def worker():
        while not stop_event.is_set() and pages_fetched < max_pages:
            try:
                item = await asyncio.wait_for(to_visit.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if to_visit.empty():
                    break
                continue

            if item is None:
                break

            url, depth = item
            stats.queue_depth = to_visit.qsize()

            await process_url(url, depth)
            to_visit.task_done()

    num_workers = max(1, concurrency * 2)

    workers = [asyncio.create_task(worker()) for _ in range(num_workers)]

    try:
        while pages_fetched < max_pages:
            try:
                page = await asyncio.wait_for(result_queue.get(), timeout=0.2)
                yield page
            except asyncio.TimeoutError:
                if all(w.done() for w in workers):
                    break
                continue
    finally:
        for w in workers:
            if not w.done():
                w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)