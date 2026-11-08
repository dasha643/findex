import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import os
import statistics
import time

import matplotlib.pyplot as plt
from findex.parallel import run_build


def measure_once(corpus: Path, executor: str, workers: int) -> float:
    times = []
    for _ in range(3):
        start = time.perf_counter()
        run_build(corpus, executor, workers)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


if __name__ == "__main__":
    corpus = Path(__file__).resolve().parent.parent / "data"
    if not corpus.exists():
        print(f"Error: directory {corpus} does not exist!")
        sys.exit(1)

    cpu_count = os.cpu_count() or 1
    workers_list = [1, 2, 4, 8, cpu_count]
    workers_list = list(dict.fromkeys(workers_list))

    print("Measuring serial baseline...")
    base_time = measure_once(corpus, "serial", 1)

    speedups_threads = []
    speedups_processes = []

    for w in workers_list:
        print(f"Measuring threads with {w} workers...")
        t_threads = measure_once(corpus, "threads", w)
        speedups_threads.append(base_time / t_threads)

        print(f"Measuring processes with {w} workers...")
        t_processes = measure_once(corpus, "processes", w)
        speedups_processes.append(base_time / t_processes)

    plt.figure(figsize=(8, 5))
    plt.plot(workers_list, speedups_threads, marker="o", label="Threads")
    plt.plot(workers_list, speedups_processes, marker="s", label="Processes")
    plt.plot(workers_list, workers_list, linestyle="--", color="gray", label="Ideal linear")

    plt.xlabel("Number of workers")
    plt.ylabel("Speedup (x)")
    plt.title("Speedup vs Workers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)

    out_path = docs_dir / "speedup.png"
    plt.savefig(out_path)
    print(f"Saved {out_path}")