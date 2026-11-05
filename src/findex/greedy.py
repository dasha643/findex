from pathlib import Path

from findex.indexer import tokenize


def run_greedy_approach(corpus_dir: Path) -> tuple[int, int, list[str]]:
    """жадібна версія. завантажує всі документи та всі токени у великі списки в пам'яті"""
    documents: list[tuple[str, str]] = []
    for file_path in corpus_dir.rglob("*.txt"):
        if file_path.is_file():
            documents.append((str(file_path), file_path.read_text(encoding="utf-8")))

    all_tokens: list[str] = []
    for _, content in documents:
        all_tokens.extend(list(tokenize(content)))

    return len(documents), len(all_tokens), all_tokens