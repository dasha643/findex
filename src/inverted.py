from collections import defaultdict
import pickle
from typing import Dict, List, Tuple
from .models import Posting, DocMeta


class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, List[Posting]] = defaultdict(list)
        self.docs: Dict[int, DocMeta] = {}

    def build(self, tokenized_docs: List[Tuple[int, str, List[str]]]):
        """побудова індексу одним проходом по токенізованих документах. tokenized_docs: список кортежів (doc_id, title, tokens)"""
        for doc_id, title, tokens in tokenized_docs:
            doc_len = len(tokens)
            self.docs[doc_id] = DocMeta(doc_id=doc_id, doc_len=doc_len, title=title)

            term_freqs = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1

            for term, freq in term_freqs.items():
                self.index[term].append(Posting(doc_id=doc_id, term_freq=freq))

        for term in self.index:
            self.index[term].sort(key=lambda p: p.doc_id)

    def save_pickle(self, filepath: str):
        """збереження індексу у бінарному форматі через pickle"""
        with open(filepath, "wb") as f:
            pickle.dump((self.index, self.docs), f)

    def load_pickle(self, filepath: str):
        """завантаження індексу з pickle файлу"""
        with open(filepath, "rb") as f:
            self.index, self.docs = pickle.load(f)