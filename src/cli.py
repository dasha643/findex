import argparse
from pathlib import Path
import time
import tracemalloc
from .indexer import iter_documents, tokenize
from .inverted import InvertedIndex
from .search import BooleanSearch


def main():
    parser = argparse.ArgumentParser(description="findex: inverted index & boolean search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="build and save inverted index")
    index_parser.add_argument("data_dir", type=str, help="path to corpus directory")
    index_parser.add_argument("--out", type=str, required=True, help="path to output index file")

    search_parser = subparsers.add_parser("search", help="search in saved index")
    search_parser.add_argument("index_file", type=str, help="path to saved index file")
    search_parser.add_argument("query", type=str, help="search query")
    search_parser.add_argument("--engine", choices=["merge", "set"], default="merge", help="search engine")

    args = parser.parse_args()

    if args.command == "index":
        print(f"побудова індексу з директорії: {args.data_dir}")
        tracemalloc.start()
        start_time = time.perf_counter()

        tokenized_docs = []
        doc_id = 1
        for file_path, content in iter_documents(Path(args.data_dir)):
            tokens = list(tokenize(content))
            title = str(file_path)
            tokenized_docs.append((doc_id, title, tokens))
            doc_id += 1

        index = InvertedIndex()
        index.build(tokenized_docs)
        index.save_pickle(args.out)

        elapsed = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"індекс збережено у {args.out}")
        print(f"час побудови: {elapsed:.4f} сек")
        print(f"пік пам'яті: {peak / 1024 / 1024:.2f} MB")

    elif args.command == "search":
        tracemalloc.start()
        start_time = time.perf_counter()

        index = InvertedIndex()
        index.load_pickle(args.index_file)

        searcher = BooleanSearch(index, engine=args.engine)
        terms = args.query.lower().split()

        results = searcher.search_and(terms)

        elapsed = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"знайдено документів: {len(results)}")
        print(f"результати (doc_id): {sorted(list(results))}")
        print(f"час пошуку: {elapsed * 1000:.4f} мс")
        print(f"пік пам'яті: {peak / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()