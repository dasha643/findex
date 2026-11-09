import pytest
from pathlib import Path
from findex.numpy_index import NumpyInvertedIndex
from findex.parallel import collect_paths, build_partial
from findex.search import search_numpy_bm25
from findex.indexer import tokenize

@pytest.fixture
def sample_index():
    # Беремо невелику частину корпусу для швидкого тесту
    paths = collect_paths(Path("data"))[:100]
    docs = build_partial(paths, 1)
    idx = NumpyInvertedIndex()
    idx.build(docs)
    return idx

def test_benchmark_build_index(benchmark):
    paths = collect_paths(Path("data"))[:100]
    # benchmark запускає функцію багато разів і рахує середній час
    benchmark(build_partial, paths, 1)

def test_benchmark_search(benchmark, sample_index):
    query_terms = list(tokenize("python async parallel"))
    benchmark(search_numpy_bm25, sample_index, query_terms, 10)