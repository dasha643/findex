from findex.inverted import InvertedIndex


def test_index_build_and_stats(sample_corpus: list[tuple[int, str, list[str]]]) -> None:
    """Перевірка правильності побудови індексу та кількості документів."""
    index = InvertedIndex()
    index.build(sample_corpus)

    assert index.num_docs == 3
    assert "python" in index.index
    assert len(index.index["python"]) == 2  # зустрічається в doc1 і doc2