import asyncio
import time
import httpx


async def make_request(client, url):
    start = time.perf_counter()
    resp = await client.get(url)
    return time.perf_counter() - start


async def measure(concurrency=20):
    url = "http://localhost:8000/search?q=hello"

    async with httpx.AsyncClient() as client:
        tasks = [make_request(client, url) for _ in range(concurrency)]

        start_wall = time.perf_counter()
        times = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - start_wall

    avg_time = sum(times) / len(times)
    print(f"Всього запитів: {concurrency}")
    print(f"Загальний час (wall time): {wall_time:.3f} сек")
    print(f"Середній час одного запиту: {avg_time * 1000:.2f} мс")
    print(f"Запитів за секунду: {concurrency / wall_time:.2f}")


if __name__ == "__main__":
    asyncio.run(measure())