import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from .indexer import tokenize
from .inverted import InvertedIndex

PartialResult = list[tuple[int, str, list[str]]]


def build_partial(paths: list[str], start_doc_id: int) -> PartialResult:
    """
    Функція рівня модуля. Приймає шляхи до файлів (не вміст!)
    та стартовий doc_id. Повертає список (doc_id, title, tokens).
    """
    result: PartialResult = []
    current_id = start_doc_id
    for path in paths:
        try:
            content = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        tokens = list(tokenize(content))
        result.append((current_id, path, tokens))
        current_id += 1
    return result


def merge(partials: list[PartialResult]) -> InvertedIndex:
    """Зливає часткові результати у фінальний InvertedIndex."""
    flat: PartialResult = []
    for part in partials:
        flat.extend(part)

    index_obj = InvertedIndex()
    index_obj.build(flat)
    return index_obj


def collect_paths(corpus_dir: Path) -> list[str]:
    """Збирає всі шляхи до .txt файлів корпусу."""
    return sorted(str(p) for p in corpus_dir.rglob("*.txt") if p.is_file())


def split_paths(paths: list[str], workers: int) -> list[tuple[list[str], int]]:
    """Розбиває шляхи на шматки і призначає кожному start_doc_id."""
    if workers <= 1:
        return [(paths, 1)]
    chunk_size = (len(paths) + workers - 1) // workers
    chunks = []
    start_id = 1
    for i in range(0, len(paths), chunk_size):
        chunk = paths[i:i + chunk_size]
        chunks.append((chunk, start_id))
        start_id += len(chunk)
    return chunks


def run_build(
    corpus_dir: Path,
    executor: Literal["serial", "threads", "processes"],
    workers: int,
) -> tuple[InvertedIndex, float]:
    """Головна функція запуску побудови. Повертає (індекс, час_витрачений_на_merge)."""
    paths = collect_paths(corpus_dir)
    chunks = split_paths(paths, workers)

    if executor == "serial" or workers == 1:
        partials = [build_partial(chunk, start) for chunk, start in chunks]
    elif executor == "threads":
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(build_partial, chunk, start) for chunk, start in chunks]
            partials = [f.result() for f in futures]
    elif executor == "processes":
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(build_partial, chunk, start) for chunk, start in chunks]
            partials = [f.result() for f in futures]
    else:
        raise ValueError(f"Unknown executor: {executor}")

    start_merge = time.perf_counter()
    final_index = merge(partials)
    merge_time = time.perf_counter() - start_merge

    return final_index, merge_time