import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import os
import statistics
import time
import psutil
from findex.parallel import run_build


def measure(corpus: Path, executor: str, workers: int) -> dict:
    wall_times = []
    cpu_times = []
    peak_rss = []
    merge_times = []

    process = psutil.Process(os.getpid())

    for _ in range(3):
        mem_before = process.memory_info().rss
        start_wall = time.perf_counter()
        start_cpu = time.process_time()

        _, m_time = run_build(corpus, executor, workers)

        wall_times.append(time.perf_counter() - start_wall)
        cpu_times.append(time.process_time() - start_cpu)
        merge_times.append(m_time)

        mem_after = process.memory_info().rss
        peak_rss.append(max(0, mem_after - mem_before))

    return {
        "wall": statistics.median(wall_times),
        "cpu": statistics.median(cpu_times),
        "rss_mb": statistics.median(peak_rss) / 1024 / 1024,
        "merge": statistics.median(merge_times),
    }


if __name__ == "__main__":
    corpus = Path(__file__).resolve().parent.parent / "data"
    if not corpus.exists():
        print(f"Error: directory {corpus} does not exist!")
        sys.exit(1)

    cpu_count = os.cpu_count() or 1
    configs = [("serial", 1)]
    for ex in ("threads", "processes"):
        for w in (1, 2, 4, 8, cpu_count):
            if w not in [c[1] for c in configs if c[0] == ex]:
                configs.append((ex, w))

    print("Measuring baseline (serial)...")
    baseline = measure(corpus, "serial", 1)
    base_wall = baseline["wall"]

    print(
        f"\n{'Executor':>10} | {'Workers':>7} | {'Wall (s)':>8} | {'CPU (s)':>8} | {'RSS (MB)':>8} | {'Merge (s)':>9} | {'Speedup':>7}")
    print("-" * 75)

    print(
        f"{'serial':>10} | {1:>7} | {base_wall:>8.2f} | {baseline['cpu']:>8.2f} | {baseline['rss_mb']:>8.1f} | {baseline['merge']:>9.3f} | {1.0:>7.2f}x")

    for ex, w in configs:
        if ex == "serial":
            continue
        try:
            r = measure(corpus, ex, w)
            speedup = base_wall / r["wall"]
            print(
                f"{ex:>10} | {w:>7} | {r['wall']:>8.2f} | {r['cpu']:>8.2f} | {r['rss_mb']:>8.1f} | {r['merge']:>9.3f} | {speedup:>7.2f}x")
        except Exception as e:
            print(f"{ex:>10} | {w:>7} | ERROR: {e}")