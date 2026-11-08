import random
from pathlib import Path


def generate_corpus(num_files=3000, words_per_file=5000):
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    words = ["hello", "world", "python", "index", "search", "engine", "fast", "data", "test", "banana", "computer",
             "science", "university", "student", "laboratory", "algorithm", "structure", "database", "network",
             "system"]

    print(f"Generating {num_files} files ({words_per_file} words each) in {data_dir}...")
    for i in range(num_files):
        content = " ".join(random.choices(words, k=words_per_file))
        (data_dir / f"doc_{i}.txt").write_text(content, encoding="utf-8")
    print("Done!")


if __name__ == "__main__":
    generate_corpus()