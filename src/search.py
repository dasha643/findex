from typing import List, Set
from .inverted import InvertedIndex


class BooleanSearch:
    def __init__(self, index: InvertedIndex, engine: str = "merge"):
        """engine: 'merge' (двовказівниковий merge по списках) або 'set' (через Python set)"""
        self.index = index
        self.engine = engine

    def search_and(self, terms: List[str]) -> Set[int]:
        """виконання операції AND для списку термінів"""
        if not terms:
            return set()

        if self.engine == "set":
            result = None
            for term in terms:
                postings = self.index.index.get(term, [])
                doc_ids = {p.doc_id for p in postings}
                if result is None:
                    result = doc_ids
                else:
                    result.intersection_update(doc_ids)
                if not result:
                    break
            return result or set()
        else:
            postings_lists = [self.index.index.get(term, []) for term in terms]
            if not all(postings_lists):
                return set()

            postings_lists.sort(key=len)

            pointers = [0] * len(postings_lists)
            result = set()

            while True:
                current_docs = [pl[ptr].doc_id for pl, ptr in zip(postings_lists, pointers)]

                if len(set(current_docs)) == 1:
                    result.add(current_docs[0])
                    pointers = [ptr + 1 for ptr in pointers]
                else:
                    max_doc = max(current_docs)
                    for i, (pl, ptr) in enumerate(zip(postings_lists, pointers)):
                        while ptr < len(pl) and pl[ptr].doc_id < max_doc:
                            ptr += 1
                        pointers[i] = ptr

                if any(ptr >= len(pl) for pl, ptr in zip(postings_lists, pointers)):
                    break

            return result

    def search_or(self, terms: List[str]) -> Set[int]:
        """виконання операції OR для списку термінів"""
        if not terms:
            return set()

        result = set()
        for term in terms:
            postings = self.index.index.get(term, [])
            for p in postings:
                result.add(p.doc_id)
        return result

    def search_not(self, base_docs: Set[int], exclude_terms: List[str]) -> Set[int]:
        """виконання операції NOT"""
        exclude_docs = self.search_or(exclude_terms)
        return base_docs - exclude_docs