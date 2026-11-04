from collections import Counter
from pathlib import Path
import re
import unicodedata
from typing import Generator, Tuple, Dict, Any


def iter_documents(corpus_dir: Path) -> Generator[Tuple[str, str], None, None]:
    """генератор, що читає документи по одному, не завантажуючи весь корпус у пам'ять."""
    for file_path in corpus_dir.rglob("*.txt"):
        if file_path.is_file():
            try:
                content = file_path.read_text(encoding="utf-8")
                yield str(file_path), content
            except Exception as e:
                print(f"помилка читання файлу {file_path}: {e}")


def tokenize(text: str) -> Generator[str, None, None]:
    """потоковий токенізатор із нормалізацією Unicode та правилами"""
    normalized_text = unicodedata.normalize("NFC", text.lower())
    pattern = re.compile(r"[a-zа-їґєії'0-9]+(?:[-'][a-zа-їґєії'0-9]+)*")

    for match in pattern.finditer(normalized_text):
        token = match.group(0).strip("'-")
        if token:
            yield token


def collect_corpus_statistics(corpus_dir: Path) -> Dict[str, Any]:
    """збирає статистику корпусу за один прохід"""
    doc_count = 0
    total_tokens = 0
    term_freq = Counter()

    for file_path, content in iter_documents(corpus_dir):
        doc_count += 1
        for token in tokenize(content):
            total_tokens += 1
            term_freq[token] += 1

    top_50 = term_freq.most_common(50)

    return {
        "doc_count": doc_count,
        "total_tokens": total_tokens,
        "unique_terms": len(term_freq),
        "top_50": top_50
    }