from typing import Literal
from findex.parallel import run_build

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

        index_obj = run_build(corpus_path, executor, workers)

        index_obj.save_pickle(out)

        elapsed = time.perf_counter() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        console.print(f"[green]Index successfully saved to {out}[/green]")
        logger.info(f"Build time: {elapsed:.4f} sec, Peak memory: {peak / 1024 / 1024:.2f} MB")

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        raise typer.Exit(code=1)