from dataclasses import dataclass
import math
import heapq
import time
from functools import wraps, lru_cache
from typing import Set, List, Union
import re


@dataclass(order=True)
class SearchResult:
    """Результат пошуку з підтримкою порівняння за релевантністю"""
    score: float
    doc_id: int


class TfIdf:
    """Реалізація метрики TF-IDF для ранжування"""

    def score(self, term: str, freq: int, doc_len: int, index) -> float:
        df = index.df(term)
        if df == 0:
            return 0.0
        idf = math.log(index.num_docs / df)
        return freq * idf


class BM25:
    """Реалізація алгоритму BM25 для ранжування"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score(self, term: str, freq: int, doc_len: int, index) -> float:
        df = index.df(term)
        if df == 0:
            return 0.0

        idf = math.log((index.num_docs - df + 0.5) / (df + 0.5) + 1.0)

        avg_dl = index.avg_doc_length
        if avg_dl == 0:
            avg_dl = 1.0

        numerator = freq * (self.k1 + 1)
        denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_dl))

        return idf * (numerator / denominator)


def timed(func):
    """Декоратор для вимірювання часу виконання функцій з functools.wraps"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        print(f"[TIMED] Функція '{func.__name__}' виконалась за {duration_ms:.2f} мс")
        return result

    return wrapper


def generate_snippet(text: str, term: str, window: int = 80) -> str:
    """Генерує контекстний сніпет: вікно ±80 символів навколо терміна"""
    lower_text = text.lower()
    lower_term = term.lower()

    pos = lower_text.find(lower_term)
    if pos == -1:
        return text[:window * 2] + "..." if len(text) > window * 2 else text

    start = max(0, pos - window)
    end = min(len(text), pos + len(term) + window)

    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    return snippet


@timed
def rank_documents(query_terms: List[str], index, scorer, top_k: int = 5) -> List[SearchResult]:
    """Ранжує документи за заданими термінами, використовуючи скрорер і heapq.nlargest"""
    scores = {}

    for term in query_terms:
        if term in index:
            for posting in index[term]:
                doc_meta = index._doc_metas.get(posting.doc_id)
                doc_len = doc_meta.length if doc_meta else 100

                term_score = scorer.score(term, posting.freq, doc_len, index)
                scores[posting.doc_id] = scores.get(posting.doc_id, 0.0) + term_score

    results = [SearchResult(score=s, doc_id=doc_id) for doc_id, s in scores.items()]
    return heapq.nlargest(top_k, results)



class QueryNode:
    """Базовий клас для вузлів синтаксичного дерева запиту"""

    def evaluate(self, index) -> Set[int]:
        raise NotImplementedError

    def __and__(self, other):
        return AndNode(self, other)

    def __or__(self, other):
        return OrNode(self, other)

    def __invert__(self):
        return NotNode(self)


class TermNode(QueryNode):
    def __init__(self, term: str):
        self.term = term

    def evaluate(self, index) -> Set[int]:
        postings = index.get(self.term, [])
        return {p.doc_id for p in postings}


class AndNode(QueryNode):
    def __init__(self, left: QueryNode, right: QueryNode):
        self.left = left
        self.right = right

    def evaluate(self, index) -> Set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)


class OrNode(QueryNode):
    def __init__(self, left: QueryNode, right: QueryNode):
        self.left = left
        self.right = right

    def evaluate(self, index) -> Set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)


class NotNode(QueryNode):
    def __init__(self, node: QueryNode):
        self.node = node

    def evaluate(self, index) -> Set[int]:
        all_docs = set(index._doc_metas.keys())
        return all_docs - self.node.evaluate(index)



class QueryParser:
    """Простий лексер і парсер на основі рекурсивного спуску для булевих запитів."""

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Union[str, None]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> Union[str, None]:
        token = self.peek()
        self.pos += 1
        return token

    def parse(self) -> QueryNode:
        return self.parse_or()

    def parse_or(self) -> QueryNode:
        node = self.parse_and()
        while self.peek() == "OR":
            self.consume()
            right = self.parse_and()
            node = OrNode(node, right)
        return node

    def parse_and(self) -> QueryNode:
        node = self.parse_unary()
        while self.peek() == "AND":
            self.consume()
            right = self.parse_unary()
            node = AndNode(node, right)
        return node

    def parse_unary(self) -> QueryNode:
        if self.peek() == "NOT":
            self.consume()
            node = self.parse_unary()
            return NotNode(node)
        return self.parse_primary()

    def parse_primary(self) -> QueryNode:
        token = self.consume()
        if token == "(":
            node = self.parse_or()
            if self.consume() != ")":
                raise SyntaxError("Очікувалась закривча дужка ')'")
            return node
        elif token and token.isalnum():
            return TermNode(token.lower())
        else:
            raise SyntaxError(f"Неочікуваний токен у запиті: {token}")


def parse_query(query_str: str) -> QueryNode:
    """Допоміжна функція для токенізації та парсингу рядка запиту."""
    tokens = re.findall(r'\(|\)|AND|OR|NOT|[a-zA-Zа-яА-Я0-9_]+', query_str)
    parser = QueryParser(tokens)
    return parser.parse()