from collections.abc import Mapping
from functools import cached_property
from contextlib import contextmanager
import pickle


class Index(Mapping):
    """Pythonic-обгортка для інвертованого індексу"""

    def __init__(self, index_data: dict):
        self._terms = index_data.get('terms', {})
        self._doc_metas = index_data.get('doc_metas', {})
        self._total_length = sum(meta.length for meta in self._doc_metas.values())

    def __getitem__(self, key):
        """Доступ до терміна через індекс: index['python']"""
        return self._terms[key]

    def __iter__(self):
        """Ітерація по всіх термінах в індексі"""
        return iter(self._terms)

    def __len__(self):
        """Кількість унікальних термінів: len(index)"""
        return len(self._terms)

    def __repr__(self):
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
        return self._total_length / self.num_docs

    def df(self, term: str) -> int:
        """кількість документів, що містять термін"""
        postings = self._terms.get(term, [])
        return len(postings)


@contextmanager
def open_index(path: str):
    """Контекстний менеджер для безпечного відкриття індексу з файлу"""
    file_obj = open(path, 'rb')
    try:
        raw_data = pickle.load(file_obj)
        yield Index(raw_data)
    finally:
        file_obj.close()