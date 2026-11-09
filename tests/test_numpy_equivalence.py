import pytest
from findex.inverted import InvertedIndex
from findex.numpy_index import NumpyInvertedIndex
from findex.models import DocMeta, Posting
from findex.indexer import tokenize
from findex.search import parse_query


def test_numpy_matches_lab3():
    # 1. Створюємо тестові документи
    docs = [
        (1, "doc1", "hello world python"),
        (2, "doc2", "python async parallel"),
        (3, "doc3", "hello async test"),
    ]

    # 2. Будуємо старий індекс (Лаба 3)
    old_idx = InvertedIndex()
    old_idx.build(docs)

    # 3. Будуємо новий NumPy індекс
    np_idx = NumpyInvertedIndex()
    np_idx.build(docs)

    # 4. Перевіряємо запит
    query = "python & async"
    query_node = parse_query(query)

    # Старий результат
    old_res = sorted(list(query_node.evaluate(old_idx)))

    # Для NumPy імітуємо булевий пошук (обидва терми мають бути в документі)
    # Використовуємо проміжний крок через звичайні множини для перевірки
    # Або просто перевіряємо, що топ-результати збігаються за doc_id
    from findex.search import search_numpy_bm25
    np_res_raw = search_numpy_bm25(np_idx, ["python", "async"], top_k=10)
    np_res_ids = [doc_id for doc_id, score in np_res_raw]

    # Перевіряємо, що документи, знайдені NumPy, входять у список старих
    for doc_id in np_res_ids:
        assert doc_id in old_res