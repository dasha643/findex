import pickle
from collections import defaultdict
import numpy as np
from .models import DocMeta


class NumpyInvertedIndex:
    def __init__(self) -> None:

        self.index: dict[str, np.ndarray] = {}
        self.doc_lens: np.ndarray = np.array([], dtype=np.int32)
        self.docs: dict[int, DocMeta] = {}
        self.num_docs: int = 0
        self.avg_doc_len: float = 0.0

    def build(self, tokenized_docs: list[tuple[int, str, list[str]]]) -> None:
        """Побудова індексу з NumPy масивами (int32)."""
        if not tokenized_docs:
            return

        self.num_docs = len(tokenized_docs)
        self.doc_lens = np.zeros(self.num_docs, dtype=np.int32)

        temp_index: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for idx, (doc_id, title, tokens) in enumerate(tokenized_docs):
            self.docs[doc_id] = DocMeta(doc_id=doc_id, doc_len=len(tokens), title=title)
            self.doc_lens[idx] = len(tokens)

            term_freqs: dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1

            for term, freq in term_freqs.items():
                temp_index[term].append((doc_id, freq))

        self.avg_doc_len = float(np.mean(self.doc_lens)) if self.num_docs > 0 else 0.0

        for term, postings in temp_index.items():
            arr = np.array(postings, dtype=np.int32)

            self.index[term] = arr[arr[:, 0].argsort()]

    def save_pickle(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump((self.index, self.docs, self.doc_lens, self.num_docs, self.avg_doc_len), f)

    def load_pickle(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            self.index, self.docs, self.doc_lens, self.num_docs, self.avg_doc_len = pickle.load(f)

    def bm25_score(self, term: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        """Обчислення BM25 для одного терміну (векторизовано)."""
        if term not in self.index:
            return np.array([], dtype=np.float32)

        postings = self.index[term]
        doc_ids = postings[:, 0]
        freqs = postings[:, 1].astype(np.float32)

        df = len(doc_ids)
        idf = np.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

        doc_ids_zero_based = doc_ids - 1
        doc_len_for_postings = self.doc_lens[doc_ids_zero_based]

        numerator = freqs * (k1 + 1)
        denominator = freqs + k1 * (1 - b + b * (doc_len_for_postings / self.avg_doc_len))
        scores = idf * (numerator / denominator)

        return np.column_stack((doc_ids, scores))

    def get_top_k(self, scores: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Повертає топ-k результатів за допомогою np.argpartition."""
        if len(scores) == 0:
            return []

        score_values = scores[:, 1]

        if len(score_values) <= k:
            top_indices = np.argsort(score_values)[::-1]
        else:
            top_indices = np.argpartition(score_values, -k)[-k:]
            top_indices = top_indices[np.argsort(score_values[top_indices])[::-1]]

        return [(int(scores[idx, 0]), float(scores[idx, 1])) for idx in top_indices]