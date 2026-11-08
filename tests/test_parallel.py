from findex.indexer import iter_documents, tokenize
from findex.inverted import InvertedIndex
from findex.parallel import build_partial, collect_paths, merge


def test_serial_equivalent_to_lab4(tmp_path):
    (tmp_path / "a.txt").write_text("hello world hello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("world python", encoding="utf-8")

    tokenized_docs = []
    doc_id = 1
    for file_path, content in iter_documents(tmp_path):
        tokens = list(tokenize(content))
        tokenized_docs.append((doc_id, str(file_path), tokens))
        doc_id += 1
    old = InvertedIndex()
    old.build(tokenized_docs)

    paths = collect_paths(tmp_path)
    new = merge([build_partial(paths, 1)])

    assert old.index == new.index
    assert old.docs == new.docs