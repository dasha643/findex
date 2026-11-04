import time
import tracemalloc
from pathlib import Path
from src.indexer import collect_corpus_statistics
from src.greedy import run_greedy_approach

corpus_path = Path("data")

tracemalloc.start()
start_time = time.perf_counter()

stats = collect_corpus_statistics(corpus_path)

end_time = time.perf_counter()
current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

print("генератори")
print(f"час виконання: {end_time - start_time:.4f} сек")
print(f"пікове споживання пам'яті: {peak_mem / 1024 / 1024:.2f} МБ")
print(f"опрацьовано документів: {stats['doc_count']}, токенів: {stats['total_tokens']}")

tracemalloc.start()
start_time = time.perf_counter()

doc_c, token_c, _ = run_greedy_approach(corpus_path)

end_time = time.perf_counter()
current_mem_g, peak_mem_g = tracemalloc.get_traced_memory()
tracemalloc.stop()

print("списки")
print(f"час виконання: {end_time - start_time:.4f} сек")
print(f"пікове споживання пам'яті: {peak_mem_g / 1024 / 1024:.2f} МБ")