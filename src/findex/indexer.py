import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_documents(corpus_dir: Path) -> Iterator[tuple[str, str]]:
    """Генератор, що читає документи (txt або jsonl)."""
    # Спочатку шукаємо .txt файли
    for file_path in corpus_dir.rglob("*.txt"):
        if file_path.is_file():
            try:
                content = file_path.read_text(encoding="utf-8")
                yield str(file_path), content
            except (OSError, UnicodeDecodeError):
                continue

    for file_path in corpus_dir.rglob("*.jsonl"):
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        url = data.get("url", str(file_path))
                        text = data.get("text", "")
                        yield url, text
            except (OSError, json.JSONDecodeError):
                continue


def tokenize(text: str) -> Iterator[str]:
    """Потоковий токенізатор із нормалізацією Unicode та правилами."""
    normalized_text = unicodedata.normalize("NFC", text.lower())
    pattern = re.compile(r"[a-zа-їґєії'0-9]+(?:[-'][a-zа-їґєії'0-9]+)*")

    for match in pattern.finditer(normalized_text):
        token = match.group(0).strip("'-")
        if token:
            yield token


def collect_corpus_statistics(corpus_dir: Path) -> dict[str, Any]:
    """Збирає статистику корпусу за один прохід."""
    doc_count = 0
    total_tokens = 0
    term_freq: Counter[str] = Counter()

    for _, content in iter_documents(corpus_dir):
        doc_count += 1
        for token in tokenize(content):
            total_tokens += 1
            term_freq[token] += 1

    top_50 = term_freq.most_common(50)

    return {
        "doc_count": doc_count,
        "total_tokens": total_tokens,
        "unique_terms": len(term_freq),
        "top_50": top_50,
    }