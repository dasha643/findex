import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
sys.path.insert(0, str(src_dir))

# 2. ТІЛЬКИ ПІСЛЯ ЦЬОГО виконуємо імпорти з пакету findex
import tempfile
from collections.abc import Generator

import pytest

from findex.inverted import InvertedIndex


@pytest.fixture
def sample_corpus() -> list[tuple[int, str, list[str]]]:
    """Невеликий тестовий корпус для перевірки збірки індексу."""
    return [
        (1, "doc1.txt", ["python", "search", "engine", "index"]),
        (2, "doc2.txt", ["python", "programming", "language"]),
        (3, "doc3.txt", ["inverted", "index", "retrieval", "engine"]),
    ]


@pytest.fixture
def temp_index_file(sample_corpus: list[tuple[int, str, list[str]]]) -> Generator[str, None, None]:
    """Створює тимчасовий файл з індексом для тестування збереження та завантаження."""
    index = InvertedIndex()
    index.build(sample_corpus)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp_path = tmp.name

    index.save_pickle(tmp_path)
    yield tmp_path

    Path(tmp_path).unlink(missing_ok=True)