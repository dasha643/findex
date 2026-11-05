import logging
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Literal

import typer
from findex.inverted import InvertedIndex
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from findex.search import parse_query

from findex.indexer import iter_documents, tokenize

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
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity (-v for INFO, -vv for DEBUG)"),
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

        logger.info(f"Building index from directory: {data_dir}")
        tracemalloc.start()
        start_time = time.perf_counter()

        tokenized_docs: list[tuple[int, str, list[str]]] = []
        doc_id = 1

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Processing documents...", total=None)
            for file_path, content in iter_documents(corpus_path):
                tokens = list(tokenize(content))
                title = str(file_path)
                tokenized_docs.append((doc_id, title, tokens))
                doc_id += 1
            progress.update(task, completed=True)

        index_obj = InvertedIndex()
        index_obj.build(tokenized_docs)
        index_obj.save_pickle(out)

        elapsed = time.perf_counter() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        console.print(f"[green]Index successfully saved to {out}[/green]")
        logger.info(f"Build time: {elapsed:.4f} sec, Peak memory: {peak / 1024 / 1024:.2f} MB")

    except Exception as e:  # noqa: BLE001
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

        # Парсимо та виконуємо булевий запит через синтаксичне дерево
        query_node = parse_query(query)
        results: set[int] = query_node.evaluate(index_obj)
        sorted_results: list[int] = sorted(results)

        elapsed = time.perf_counter() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if json_output:
            import json
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
            logger.info(f"Found documents: {len(sorted_results)}, Search time: {elapsed * 1000:.4f} ms, Peak memory: {peak / 1024 / 1024:.2f} MB")

    except Exception as e:  # noqa: BLE001
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

    except Exception as e:  # noqa: BLE001
        logger.error(f"Error loading stats: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()