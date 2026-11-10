import time
import numpy as np
from pathlib import Path

from findex.inverted import InvertedIndex
from findex.numpy_index import NumpyInvertedIndex
from findex.indexer import tokenize
from findex.parallel import collect_paths


def build_old_index(paths: list[str]) -> InvertedIndex:
    """Будує старий індекс (як у лабі 3)."""
    index = InvertedIndex()
    tokenized_docs = []
    for i, path in enumerate(paths):
        try:
            text = Path(path).read_text(encoding="utf-8")
            tokenized_docs.append((i + 1, path, list(tokenize(text))))
        except Exception:
            pass
    index.build(tokenized_docs)
    return index


def build_numpy_index(paths: list[str]) -> NumpyInvertedIndex:
    """Будує новий індекс на NumPy."""
    index = NumpyInvertedIndex()
    tokenized_docs = []
    for i, path in enumerate(paths):
        try:
            text = Path(path).read_text(encoding="utf-8")
            tokenized_docs.append((i + 1, path, list(tokenize(text))))
        except Exception:
            pass
    index.build(tokenized_docs)
    return index


def old_bm25_score(index: InvertedIndex, term: str, k1=1.5, b=0.75) -> dict[int, float]:
    """Симуляція старого скорера на циклах Python."""
    if term not in index.index:
        return {}

    postings = index.index[term]
    scores = {}
    df = len(postings)
    num_docs = index.num_docs
    avg_dl = sum(d.doc_len for d in index.docs.values()) / num_docs if num_docs else 0
    idf = np.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)

    for p in postings:
        doc_len = index.docs[p.doc_id].doc_len
        numerator = p.term_freq * (k1 + 1)
        denominator = p.term_freq + k1 * (1 - b + b * (doc_len / avg_dl))
        scores[p.doc_id] = idf * (numerator / denominator)
    return scores


def measure(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return (time.perf_counter() - start) * 1000, result


if __name__ == "__main__":
    data_dir = Path("data")
    paths = collect_paths(data_dir)

    print("Будуємо старий індекс...")
    old_idx = build_old_index(paths)

    print("Будуємо NumPy індекс...")
    np_idx = build_numpy_index(paths)

    freq_term = "hello"
    rare_term = "algorithm"  # або будь-яке інше
    query_terms = ["hello", "world", "python"]

    print("\n--- Результати бенчмарку (мілісекунди) ---")
    print(f"{'Запит':<20} | {'Старий (Python)':<15} | {'NumPy':<15} | {'Прискорення':<10}")
    print("-" * 65)

    t_old, _ = measure(old_bm25_score, old_idx, freq_term)
    t_np, _ = measure(np_idx.bm25_score, freq_term)
    print(
        f"{'Частий терм (' + freq_term + ')':<20} | {t_old:<15.2f} | {t_np:<15.2f} | {t_old / t_np if t_np else 0:<10.1f}x")

    t_old, _ = measure(old_bm25_score, old_idx, rare_term)
    t_np, _ = measure(np_idx.bm25_score, rare_term)
    print(
        f"{'Рідкісний (' + rare_term + ')':<20} | {t_old:<15.2f} | {t_np:<15.2f} | {t_old / t_np if t_np else 0:<10.1f}x")

    t_old_total = 0
    t_np_total = 0
    for term in query_terms:
        t1, _ = measure(old_bm25_score, old_idx, term)
        t2, _ = measure(np_idx.bm25_score, term)
        t_old_total += t1
        t_np_total += t2

    print(
        f"{'Запит з 3 термів':<20} | {t_old_total:<15.2f} | {t_np_total:<15.2f} | {t_old_total / t_np_total if t_np_total else 0:<10.1f}x")

    scores_np = np_idx.bm25_score(freq_term)
    t_topk, _ = measure(np_idx.get_top_k, scores_np, 10)
    print(f"\nЧас top-k через np.argpartition: {t_topk:.3f} ms")