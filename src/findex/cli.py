import logging
import sys
import time
import tracemalloc
import json
import asyncio
from pathlib import Path
from typing import Literal

import typer
import httpx
import uvicorn
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from findex.inverted import InvertedIndex
from findex.parallel import run_build
from findex.search import parse_query
from findex.crawler import crawl, CrawlStats

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("findex")

app = typer.Typer(help="findex: inverted index & boolean search CLI")
console = Console(stderr=True)


@app.command()
def index(
        data_dir: str = typer.Argument(..., help="Path to corpus directory"),
        out: str = typer.Option(..., "--out", help="Path to output index file"),
        workers: int = typer.Option(1, "--workers", help="Number of workers"),
        executor: Literal["serial", "threads", "processes"] = typer.Option(
            "serial", "--executor", help="Execution mode"
        ),
        verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity"),
) -> None:
    """Build and save inverted index."""
    if verbose == 1:
        logger.setLevel(logging.INFO)
    elif verbose >= 2:
        logger.setLevel(logging.DEBUG)

    try:
        corpus_path = Path(data_dir)
        if not corpus_path.exists():
            logger.error(f"Directory {data_dir} does not exist.")
            raise typer.Exit(code=1)

        logger.info(f"Building index from {data_dir} with {executor}, workers={workers}")
        tracemalloc.start()
        start_time = time.perf_counter()

        index_obj, _ = run_build(corpus_path, executor, workers)
        index_obj.save_pickle(out)

        elapsed = time.perf_counter() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        console.print(f"[green]Index successfully saved to {out}[/green]")
        logger.info(f"Build time: {elapsed:.4f} sec, Peak memory: {peak / 1024 / 1024:.2f} MB")

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        raise typer.Exit(code=1)


@app.command()
def search(
        index_file: str = typer.Argument(..., help="Path to saved index file"),
        query: str = typer.Argument(..., help="Search query"),
        engine: Literal["merge", "set"] = typer.Option("merge", "--engine", help="Search engine method"),
        json_output: bool = typer.Option(False, "--json", help="Output results as JSON to stdout"),
        verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity"),
) -> None:
    """Search in saved index."""
    if verbose == 1:
        logger.setLevel(logging.INFO)
    elif verbose >= 2:
        logger.setLevel(logging.DEBUG)

    try:
        index_path = Path(index_file)
        if not index_path.exists():
            logger.error(f"Index file {index_file} does not exist.")
            raise typer.Exit(code=1)

        tracemalloc.start()
        start_time = time.perf_counter()

        index_obj = InvertedIndex()
        index_obj.load_pickle(index_file)

        query_node = parse_query(query)
        results: set[int] = query_node.evaluate(index_obj)
        sorted_results: list[int] = sorted(results)

        elapsed = time.perf_counter() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if json_output:
            print(json.dumps({"results": sorted_results, "count": len(sorted_results)}))
        else:
            table = Table(title="Search Results")
            table.add_column("Doc ID", justify="right", style="cyan")
            table.add_column("Document Path", style="magenta")

            for doc_id in sorted_results:
                doc_meta = index_obj.docs.get(doc_id) if hasattr(index_obj, "docs") else None
                doc_title = doc_meta.title if doc_meta and hasattr(doc_meta, "title") else "N/A"
                table.add_row(str(doc_id), str(doc_title))

            console.print(table)
            logger.info(
                f"Found documents: {len(sorted_results)}, Search time: {elapsed * 1000:.4f} ms, Peak memory: {peak / 1024 / 1024:.2f} MB")

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise typer.Exit(code=1)


@app.command()
def stats(
        index_file: str = typer.Argument(..., help="Path to saved index file"),
) -> None:
    """Show index statistics."""
    try:
        index_path = Path(index_file)
        if not index_path.exists():
            logger.error(f"Index file {index_file} does not exist.")
            raise typer.Exit(code=1)

        index_obj = InvertedIndex()
        index_obj.load_pickle(index_file)

        console.print(f"[bold green]Index Statistics for {index_file}:[/bold green]")
        console.print(f"Total documents: {len(getattr(index_obj, 'docs', {}))}")
        console.print(f"Total unique terms: {len(getattr(index_obj, 'index', {}))}")

    except Exception as e:
        logger.error(f"Error loading stats: {e}")
        raise typer.Exit(code=1)


@app.command()
def crawl_cmd(
        start_url: str = typer.Argument(..., help="Початковий URL для обходу"),
        out: str = typer.Option("crawl.jsonl", "--out", help="Файл для збереження JSONL"),
        log_file: str = typer.Option("crawl.log", "--log", help="Файл логу"),
        max_pages: int = typer.Option(200, "--max-pages", help="Максимальна кількість сторінок"),
        concurrency: int = typer.Option(10, "--concurrency", help="Кількість одночасних запитів"),
        max_depth: int = typer.Option(3, "--depth", help="Максимальна глибина обходу"),
) -> None:
    """Асинхронний кролер для збору сторінок."""

    async def run_crawl():
        stats = CrawlStats()
        headers = {"User-Agent": "FindexCrawler/1.0 (+https://github.com/dasha643/findex)"}

        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            with open(out, "w", encoding="utf-8") as f_out, open(log_file, "w", encoding="utf-8") as f_log:
                with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TextColumn("{task.completed}/{task.total} сторінок"),
                        TextColumn("| {task.fields[stats]}"),
                        TimeElapsedColumn(),
                        console=console,
                ) as progress:
                    task = progress.add_task("[cyan]Кроулінг...", total=max_pages, stats="")

                    async for page in crawl(
                            start_url=start_url,
                            client=client,
                            max_pages=max_pages,
                            concurrency=concurrency,
                            max_depth=max_depth,
                            stats=stats,
                    ):
                        f_out.write(json.dumps(page, ensure_ascii=False) + "\n")
                        f_out.flush()

                        f_log.write(f"{page['status']} {page['url']} ({page['size']} bytes, {page['elapsed']:.2f}s)\n")
                        f_log.flush()

                        stats_str = f"{stats.pages_per_sec:.1f} p/s | Wait: {stats.active_requests} | Err: {stats.errors} | Q: {stats.queue_depth}"
                        progress.update(task, advance=1, stats=stats_str)

    try:
        asyncio.run(run_crawl())
        console.print(f"[green]Кроулінг завершено! Дані збережено у {out}[/green]")
    except Exception as e:
        console.print(f"[red]Помилка під час кроулінгу: {e}[/red]")
        raise typer.Exit(code=1)



@app.command()
def serve(
    port: int = typer.Option(8000, "--port", help="Port to run the server on"),
    workers: int = typer.Option(1, "--workers", help="Number of uvicorn workers"),
):
    """Запуск веб-сервера FastAPI."""
    console.print(f"[green]Starting server on port {port} with {workers} workers...[/green]")
    # Вказуємо uvicorn шлях до нашого додатку
    uvicorn.run("findex.web.app:app", host="0.0.0.0", port=port, workers=workers)


if __name__ == "__main__":
    app()