from pathlib import Path
from src.indexer import tokenize


def run_greedy_approach(corpus_dir: Path):
    """жадібна версія. завантажує всі документи та всі токени у великі списки в пам'яті"""
    documents = []
    for file_path in corpus_dir.rglob("*.txt"):
        if file_path.is_file():
            documents.append((str(file_path), file_path.read_text(encoding="utf-8")))

    all_tokens = []
    for _, content in documents:
        all_tokens.extend(list(tokenize(content)))

    return len(documents), len(all_tokens), all_tokens