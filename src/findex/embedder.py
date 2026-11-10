import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


class SemanticIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.doc_ids: list[int] = []
        self.fragment_texts: list[str] = []

    def embed_corpus(self, paths: list[str], chunk_size: int = 200):
        """Розбиває документи на фрагменти і рахує ембединги."""
        fragments = []
        doc_ids = []

        for i, path in enumerate(paths):
            try:
                text = Path(path).read_text(encoding="utf-8")
                words = text.split()
                for j in range(0, len(words), chunk_size):
                    fragment = " ".join(words[j:j + chunk_size])
                    fragments.append(fragment)
                    # doc_id починається з 1, як у нашому індексаторі
                    doc_ids.append(i + 1)
            except Exception:
                pass

        print(f"Обчислюємо ембединги для {len(fragments)} фрагментів...")
        embeddings = self.model.encode(fragments, show_progress_bar=True, normalize_embeddings=True)

        self.embeddings = embeddings
        self.doc_ids = doc_ids
        self.fragment_texts = fragments
        print("Готово!")

    def save(self, filepath: str):
        if self.embeddings is None:
            raise ValueError("Немає ембедингів для збереження")
        np.savez(filepath, embeddings=self.embeddings, doc_ids=self.doc_ids, fragment_texts=self.fragment_texts)

    def load(self, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        self.embeddings = data['embeddings']
        self.doc_ids = list(data['doc_ids'])
        self.fragment_texts = list(data['fragment_texts'])

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Пошук за косинусною близькістю."""
        if self.embeddings is None:
            return []

        query_vec = self.model.encode([query], normalize_embeddings=True)[0]

        scores = np.dot(self.embeddings, query_vec)

        top_indices = np.argsort(scores)[::-1][:k * 3]

        results = {}
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = float(scores[idx])
            if doc_id not in results or score > results[doc_id]:
                results[doc_id] = score

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:k]
        return sorted_results