"""给 C++ BM25 对拍生成参考 TSV。以逐文档扫描独立核对倒排检索。"""

import argparse
from collections import Counter
import csv
import math
from pathlib import Path

from reference_search import load_documents


def rank_documents(documents, query_tf, document_df, average_length, top_k):
    if not average_length:
        return []

    document_count = len(documents)
    term_weights = {
        term: frequency * math.log1p((document_count - df + 0.5) / (df + 0.5))
        for term, frequency in query_tf.items()
        if (df := document_df[term]) > 0
    }
    if not term_weights:
        return []

    results = []
    k1, b = 1.2, 0.75
    for doc_id, (length, counts) in enumerate(documents):
        length_penalty = k1 * (1.0 - b + b * (length / average_length))
        score = 0.0
        for term, weight in term_weights.items():
            tf = counts[term]
            if tf:
                score += weight * tf * (k1 + 1.0) / (tf + length_penalty)
        if score > 0.0:
            results.append((doc_id, score))

    results.sort(key=lambda row: (-row[1], row[0]))
    return results[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Write Python BM25 Top-K results as TSV.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    with args.queries.open(encoding="utf-8") as handle:
        queries = [line.rstrip("\n") for line in handle if line.strip()]
    # 当前批量 TSV 协议按查询原文识别结果，要求查询唯一且不含制表符。
    if not queries or len(set(queries)) != len(queries) or any("\t" in query for query in queries):
        parser.error("queries must be nonempty, unique and tab-free")

    query_terms = {term for query in queries for term in query.split()}
    documents = []
    document_df = Counter()
    for tokens in load_documents(args.corpus):
        counts = Counter(token for token in tokens if token in query_terms)
        document_df.update(counts.keys())
        documents.append((len(tokens), counts))
    average_length = sum(length for length, _ in documents) / len(documents) if documents else 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("query", "rank", "doc_id", "score"))
        for query in queries:
            results = rank_documents(documents, Counter(query.split()), document_df, average_length, args.top_k)
            for rank, (doc_id, score) in enumerate(results, start=1):
                writer.writerow((query, rank, doc_id, format(score, ".17g")))
            row_count += len(results)
    print(f"BM25_REFERENCE_OK documents={len(documents)} queries={len(queries)} rows={row_count}")


if __name__ == "__main__":
    main()
