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

# Імпорти для Лабораторної 8
from findex.numpy_index import NumpyInvertedIndex
from findex.embedder import SemanticIndex
from findex.search import (
    search_numpy_bm25,
    reciprocal_rank_fusion,
    parse_query,
)
from findex.indexer import tokenize
from findex.parallel import collect_paths, build_partial

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
    """Build and save inverted index (NumPy format)."""
    if verbose == 1:
        logger.setLevel(logging.INFO)
    elif verbose >= 2:
        logger.setLevel(logging.DEBUG)

    try:
        corpus_path = Path(data_dir)
        if not corpus_path.exists():
            logger.error(f"Directory {data_dir} does not exist.")
            raise typer.Exit(code=1)

        logger.info(f"Building NumPy index from {data_dir}...")
        tracemalloc.start()
        start_time = time.perf_counter()

        # Отримуємо всі шляхи до файлів
        paths = collect_paths(corpus_path)

        # Розбиваємо на шматки для паралельної обробки (якщо треба)
        chunk_size = (len(paths) + workers - 1) // workers if workers > 0 else len(paths)
        chunks = []
        start_id = 1
        for i in range(0, len(paths), chunk_size):
            chunk = paths[i:i + chunk_size]
            chunks.append((chunk, start_id))
            start_id += len(chunk)

        # Збираємо токенізовані документи
        all_docs = []
        if executor == "serial" or workers == 1:
            for chunk, sid in chunks:
                all_docs.extend(build_partial(chunk, sid))
        elif executor == "threads":
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(build_partial, chunk, sid) for chunk, sid in chunks]
                for f in futures:
                    all_docs.extend(f.result())
        elif executor == "processes":
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(build_partial, chunk, sid) for chunk, sid in chunks]
                for f in futures:
                    all_docs.extend(f.result())

        # Будуємо NumPy індекс
        np_index = NumpyInvertedIndex()
        np_index.build(all_docs)
        np_index.save_pickle(out)

        elapsed = time.perf_counter() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        console.print(f"[green]NumPy Index successfully saved to {out}[/green]")
        logger.info(f"Build time: {elapsed:.4f} sec, Peak memory: {peak / 1024 / 1024:.2f} MB")

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        raise typer.Exit(code=1)


@app.command()
def embed(
        data_dir: str = typer.Argument(..., help="Path to corpus directory"),
        out: str = typer.Option("embeddings.npz", "--out", help="Path to output embeddings file"),
):
    """Створює ембединги для семантичного пошуку."""
    corpus_path = Path(data_dir)
    if not corpus_path.exists():
        console.print(f"[red]Directory {data_dir} does not exist.[/red]")
        raise typer.Exit(code=1)

    paths = collect_paths(corpus_path)
    console.print(f"[cyan]Створюємо ембединги для {len(paths)} файлів...[/cyan]")

    semantic_idx = SemanticIndex()
    semantic_idx.embed_corpus(paths)
    semantic_idx.save(out)

    console.print(f"[green]Ембединги збережено у {out}[/green]")


@app.command()
def search(
        index_file: str = typer.Argument(..., help="Path to saved index file"),
        query: str = typer.Argument(..., help="Search query"),
        mode: str = typer.Option("bm25", "--mode", help="bm25, semantic, or hybrid"),
        embeddings_file: str = typer.Option("embeddings.npz", "--embeddings", help="Path to embeddings"),
        k: int = typer.Option(10, "--k", help="Number of results"),
        json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
        verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity"),
) -> None:
    """Search in saved index (BM25, Semantic, or Hybrid)."""
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

        # 1. ЗАВЖДИ завантажуємо індекс для відображення назв файлів
        index_obj = NumpyInvertedIndex()
        index_obj.load_pickle(index_file)

        bm25_res = []
        semantic_res = []

        # 2. BM25 через NumPy
        if mode in ("bm25", "hybrid"):
            query_terms = list(tokenize(query))
            bm25_res = search_numpy_bm25(index_obj, query_terms, top_k=k)

        # 3. Семантичний пошук
        if mode in ("semantic", "hybrid"):
            if not Path(embeddings_file).exists():
                logger.error(f"Embeddings file {embeddings_file} not found. Run 'findex embed' first.")
                raise typer.Exit(code=1)
            semantic_idx = SemanticIndex()
            semantic_idx.load(embeddings_file)
            semantic_res = semantic_idx.search(query, k=k)

        # 4. Об'єднання (RRF)
        if mode == "bm25":
            final_res = bm25_res
        elif mode == "semantic":
            final_res = semantic_res
        else:
            final_res = reciprocal_rank_fusion(bm25_res, semantic_res)[:k]

        elapsed = time.perf_counter() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if json_output:
            print(json.dumps([{"doc_id": d, "score": s} for d, s in final_res]))
        else:
            table = Table(title=f"Search Results (Mode: {mode})")
            table.add_column("Rank", justify="right", style="dim")
            table.add_column("Doc ID", justify="right", style="cyan")
            table.add_column("Score", justify="right", style="green")
            table.add_column("Document Path", style="magenta")

            for i, (doc_id, score) in enumerate(final_res):
                title = "N/A"
                # Тепер index_obj завжди доступний
                if doc_id in index_obj.docs:
                    title = index_obj.docs[doc_id].title

                table.add_row(str(i + 1), str(doc_id), f"{score:.4f}", title)

            console.print(table)
            logger.info(f"Search time: {elapsed * 1000:.2f} ms, Peak memory: {peak / 1024 / 1024:.2f} MB")

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

        index_obj = NumpyInvertedIndex()
        index_obj.load_pickle(index_file)

        console.print(f"[bold green]Index Statistics for {index_file}:[/bold green]")
        console.print(f"Total documents: {len(index_obj.docs)}")
        console.print(f"Total unique terms: {len(index_obj.index)}")

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
    from findex.crawler import crawl, CrawlStats

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
    uvicorn.run("findex.web.app:app", host="0.0.0.0", port=port, workers=workers)


if __name__ == "__main__":
    app()