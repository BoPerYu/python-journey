"""python -m textstats 的入口。"""

from . import count_tokens, read_tokens, top_k


def main():
    tokens = read_tokens("data/sample.txt")
    counts = count_tokens(tokens)
    for word, count in top_k(counts, 10):
        print(f"{word} {count}")


if __name__ == "__main__":
    main()
