import pytest
from findex.inverted import InvertedIndex
from findex.numpy_index import NumpyInvertedIndex
from findex.models import DocMeta, Posting
from findex.indexer import tokenize
from findex.search import parse_query


def test_numpy_matches_lab3():
    docs = [
        (1, "doc1", "hello world python"),
        (2, "doc2", "python async parallel"),
        (3, "doc3", "hello async test"),
    ]

    old_idx = InvertedIndex()
    old_idx.build(docs)

    np_idx = NumpyInvertedIndex()
    np_idx.build(docs)

    query = "python & async"
    query_node = parse_query(query)

    old_res = sorted(list(query_node.evaluate(old_idx)))

    from findex.search import search_numpy_bm25
    np_res_raw = search_numpy_bm25(np_idx, ["python", "async"], top_k=10)
    np_res_ids = [doc_id for doc_id, score in np_res_raw]

    for doc_id in np_res_ids:
        assert doc_id in old_res