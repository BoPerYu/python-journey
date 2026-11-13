from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import reference_search as search


class ReferenceSearchTests(unittest.TestCase):
    def make_sample_file(self, directory: str) -> Path:
        path = Path(directory) / "sample.tokens.txt"
        path.write_text(
            "apple apple banana\n"
            "\n"
            "apple carrot\n",
            encoding="utf-8",
        )
        return path

    def test_load_documents_preserves_blank_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_sample_file(directory)
            documents = search.load_documents(path)

        self.assertEqual(
            documents,
            [["apple", "apple", "banana"], [], ["apple", "carrot"]],
        )

    def test_postings_and_document_frequency(self) -> None:
        documents = [["apple", "apple", "banana"], [], ["apple", "carrot"]]
        postings = search.build_postings(documents)

        self.assertEqual(postings["apple"], [(0, 2), (2, 1)])
        self.assertEqual(postings["banana"], [(0, 1)])
        self.assertEqual(postings["carrot"], [(2, 1)])
        self.assertEqual(len(postings["apple"]), 2)

    def test_document_norms(self) -> None:
        documents = [["apple", "apple", "banana"], [], ["apple", "carrot"]]
        postings = search.build_postings(documents)
        norms = search.build_tfidf_norms(postings, len(documents))

        self.assertEqual(len(norms), 3)
        self.assertGreater(norms[0], 0.0)
        self.assertEqual(norms[1], 0.0)
        self.assertGreater(norms[2], 0.0)

    def test_apple_ranking_and_scores(self) -> None:
        documents = [["apple", "apple", "banana"], [], ["apple", "carrot"]]
        results = search.search_tfidf(documents, "apple", 10)

        self.assertEqual([doc_id for doc_id, _ in results], [0, 2])
        self.assertAlmostEqual(results[0][1], 0.8355915419449176, places=12)
        self.assertAlmostEqual(results[1][1], 0.6053485081062916, places=12)

    def test_banana_and_carrot(self) -> None:
        documents = [["apple", "apple", "banana"], [], ["apple", "carrot"]]

        banana = search.search_tfidf(documents, "banana", 10)
        carrot = search.search_tfidf(documents, "carrot", 10)

        self.assertEqual([doc_id for doc_id, _ in banana], [0])
        self.assertAlmostEqual(banana[0][1], 0.5493512310263032, places=12)
        self.assertEqual([doc_id for doc_id, _ in carrot], [2])
        self.assertAlmostEqual(carrot[0][1], 0.7959605415681652, places=12)

    def test_multi_term_ranking(self) -> None:
        documents = [["apple", "apple", "banana"], [], ["apple", "carrot"]]
        results = search.search_tfidf(documents, "apple carrot", 10)

        self.assertEqual([doc_id for doc_id, _ in results], [2, 0])
        self.assertAlmostEqual(results[0][1], 1.0, places=12)
        self.assertAlmostEqual(results[1][1], 0.5058240933025916, places=12)

    def test_empty_unknown_and_zero_limit(self) -> None:
        documents = [["apple", "apple", "banana"], [], ["apple", "carrot"]]

        self.assertEqual(search.search_tfidf(documents, "", 10), [])
        self.assertEqual(search.search_tfidf(documents, "missing", 10), [])
        self.assertEqual(search.search_tfidf(documents, "apple", 0), [])
        self.assertEqual(search.search_tfidf(documents, "apple", -1), [])

    def test_top_k(self) -> None:
        documents = [["apple", "apple", "banana"], [], ["apple", "carrot"]]
        results = search.search_tfidf(documents, "apple", 1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 0)


if __name__ == "__main__":
    unittest.main()
