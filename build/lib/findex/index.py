import pickle
import re
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from functools import cached_property
from typing import Any, cast  # Видалили неможливий імпорт dict

from .models import DocMeta, Posting


def tokenize(text: str) -> list[str]:
    """Токенізує текст: переводить у нижній регістр та витягує слова і числа з підтримкою Unicode."""
    if not text:
        return []
    return re.findall(r'[^\W\d_]+|\d+', text.lower())


class Index(Mapping[str, list[Posting]]):
    """Pythonic-обгортка для інвертованого індексу"""

    def __init__(self, index_data: dict[str, Any]) -> None:
        self._terms: dict[str, list[Posting]] = index_data.get("terms", {})
        self._doc_metas: dict[int, DocMeta] = index_data.get("doc_metas", {})

        total_len = 0
        for meta in self._doc_metas.values():
            # Підтримуємо обидва варіанти назви поля довжини (length або doc_len)
            length_val = getattr(meta, "length", getattr(meta, "doc_len", 0))
            total_len += int(length_val)
        self._total_length: int = total_len

    def __getitem__(self, key: str) -> list[Posting]:
        """Доступ до терміна через індекс: index['python']"""
        return self._terms[key]

    def __iter__(self) -> Iterator[str]:
        """Ітерація по всіх термінах в індексі"""
        return iter(self._terms)

    def __len__(self) -> int:
        """Кількість унікальних термінів: len(index)"""
        return len(self._terms)

    def __repr__(self) -> str:
        """Красиве відображення об'єкта в консолі"""
        return f"<Index terms={len(self._terms)}, docs={self.num_docs}>"

    @cached_property
    def num_docs(self) -> int:
        """Кількість документів у корпусі"""
        return len(self._doc_metas)

    @cached_property
    def avg_doc_length(self) -> float:
        """Середня довжина документа в корпусі"""
        if self.num_docs == 0:
            return 0.0
        return float(self._total_length / self.num_docs)

    def df(self, term: str) -> int:
        """кількість документів, що містять термін"""
        postings = self._terms.get(term, [])
        return len(postings)


@contextmanager
def open_index(path: str) -> Generator[Index, None, None]:
    """Контекстний менеджер для безпечного відкриття індексу з файлу"""
    with open(path, "rb") as file_obj:
        raw_data: Any = pickle.load(file_obj)
        index_dict: dict[str, Any] = {}  # Використовуємо вбудований dict замість Dict

        if isinstance(raw_data, tuple):
            raw_tuple = cast(tuple[Any, ...], raw_data)
            if len(raw_tuple) == 2:
                index_dict = {"terms": raw_tuple[0], "doc_metas": raw_tuple[1]}
        elif isinstance(raw_data, dict):
            index_dict = cast(dict[str, Any], raw_data)

        yield Index(index_dict)