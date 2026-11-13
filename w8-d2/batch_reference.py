from __future__ import annotations

import argparse
import csv
from pathlib import Path

from reference_search import (
    build_postings,
    build_tfidf_norms,
    load_documents,
    search_tfidf_from_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write Python TF-IDF Top-K results as TSV."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    documents = load_documents(args.corpus)
    postings = build_postings(documents)
    norms = build_tfidf_norms(postings, len(documents))

    queries = [
        line.strip()
        for line in args.queries.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not queries:
        raise ValueError("query file is empty")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(["query", "rank", "doc_id", "score"])

        for query in queries:
            results = search_tfidf_from_index(
                doc_postings=postings,
                doc_norms=norms,
                document_count=len(documents),
                query=query,
                top_k=args.top_k,
            )
            for rank, (doc_id, score) in enumerate(results, start=1):
                writer.writerow(
                    [query, rank, doc_id, format(score, ".17g")]
                )

    print("documents:", len(documents))
    print("queries:", len(queries))
    print("output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
