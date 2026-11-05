import pickle
from collections import defaultdict

from .models import DocMeta, Posting


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, list[Posting]] = defaultdict(list)
        self.docs: dict[int, DocMeta] = {}

    @property
    def num_docs(self) -> int:
        """Повертає загальну кількість проіндексованих документів."""
        return len(self.docs)

    def build(self, tokenized_docs: list[tuple[int, str, list[str]]]) -> None:
        """побудова індексу одним проходом по токенізованих документах. tokenized_docs: список кортежів (doc_id, title, tokens)"""
        for doc_id, title, tokens in tokenized_docs:
            doc_len = len(tokens)
            self.docs[doc_id] = DocMeta(doc_id=doc_id, doc_len=doc_len, title=title)

            term_freqs: dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1

            for term, freq in term_freqs.items():
                self.index[term].append(Posting(doc_id=doc_id, term_freq=freq))

        for term in self.index:
            self.index[term].sort(key=lambda p: p.doc_id)

    def save_pickle(self, filepath: str) -> None:
        """збереження індексу у бінарному форматі через pickle"""
        with open(filepath, "wb") as f:
            pickle.dump((self.index, self.docs), f)

    def load_pickle(self, filepath: str) -> None:
        """завантаження індексу з pickle файлу"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.index, self.docs = data