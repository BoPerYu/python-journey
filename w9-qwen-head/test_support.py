"""只测试工程外围；不实现、不调用学生特征或拟合核心，不加载模型。"""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import pipeline


class ManifestChecks(unittest.TestCase):
    """使用临时合成语料，核对 ID、来源与固定子集边界。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.train_path = self.data / "train.txt"
        self.dev_path = self.data / "dev.txt"
        self.train_path.write_text(
            "".join(f"train_a_{i}\ttrain_b_{i}\t{i % 2}\n" for i in range(1, 22041)),
            encoding="utf-8",
        )
        self.dev_path.write_text(
            "".join(f"dev_a_{i}\tdev_b_{i}\t{i % 2}\n" for i in range(1, 501)),
            encoding="utf-8",
        )
        self.pilot = self.root / "fixed500.json"
        self.pilot_obj = {
            "seed": 42,
            "data_hashes": {"train.txt": pipeline.sha(self.train_path),
                            "dev.txt": pipeline.sha(self.dev_path)},
            "sample": [self.row("dev", i) for i in range(1, 501)],
        }
        pipeline.dump(self.pilot, self.pilot_obj)
        data_patch = patch.object(pipeline, "DATA", self.data)
        pilot_patch = patch.object(pipeline, "PILOT", self.pilot)
        data_patch.start()
        pilot_patch.start()
        self.addCleanup(data_patch.stop)
        self.addCleanup(pilot_patch.stop)
        self.manifest = self.root / "smoke.json"
        self.obj = pipeline.prepare_manifest(self.manifest)

    @staticmethod
    def row(split, number):
        return {"pair_id": f"{split}:{number:06d}", "split": split,
                "line_number": number, "s1": f"{split}_a_{number}",
                "s2": f"{split}_b_{number}", "label": number % 2}

    def save(self, obj):
        pipeline.dump(self.manifest, obj)

    def test_roundtrip_and_fixed_selection(self):
        actual = pipeline.read_manifest(self.manifest)
        self.assertEqual(actual, self.obj)
        self.assertEqual(len(actual["train"]), 20)
        self.assertEqual([r["label"] for r in actual["train"]].count(0), 10)
        self.assertEqual([r["label"] for r in actual["train"]].count(1), 10)
        self.assertEqual([r["line_number"] for r in actual["dev_diagnostic"]], list(range(1, 21)))
        self.assertEqual(pipeline.prepare_manifest(self.manifest), actual)

    def test_source_change_rejected(self):
        with self.train_path.open("a", encoding="utf-8") as handle:
            handle.write("extra_a\textra_b\t0\n")
        with self.assertRaises(ValueError):
            pipeline.read_manifest(self.manifest)

    def test_one_based_pair_id_required(self):
        obj = copy.deepcopy(self.obj)
        row = obj["train"][0]
        row["pair_id"] = f"train:{row['line_number'] - 1:06d}"
        self.save(obj)
        with self.assertRaises(ValueError):
            pipeline.read_manifest(self.manifest)

    def test_label_content_and_split_tampering_rejected(self):
        for field, value in (("label", 2), ("s1", "tampered"), ("split", "dev")):
            with self.subTest(field=field):
                obj = copy.deepcopy(self.obj)
                obj["train"][0][field] = value
                self.save(obj)
                with self.assertRaises(ValueError):
                    pipeline.read_manifest(self.manifest)

    def test_subset_size_must_remain_twenty(self):
        for key in ("train", "dev_diagnostic"):
            with self.subTest(key=key):
                obj = copy.deepcopy(self.obj)
                obj[key] = obj[key][:-1]
                self.save(obj)
                with self.assertRaises(ValueError):
                    pipeline.read_manifest(self.manifest)

    def test_duplicate_pair_rejected(self):
        obj = copy.deepcopy(self.obj)
        obj["train"][1] = copy.deepcopy(obj["train"][0])
        self.save(obj)
        with self.assertRaises(ValueError):
            pipeline.read_manifest(self.manifest)

    def test_other_valid_dev_rows_rejected(self):
        obj = copy.deepcopy(self.obj)
        obj["dev_diagnostic"] = [self.row("dev", i) for i in range(21, 41)]
        self.save(obj)
        with self.assertRaises(ValueError):
            pipeline.read_manifest(self.manifest)

    def test_other_valid_train_rows_cannot_claim_original_selection(self):
        obj = copy.deepcopy(self.obj)
        existing = {row["line_number"] for row in obj["train"]}
        label = obj["train"][0]["label"]
        candidate = next(i for i in range(1, 22041) if i not in existing and i % 2 == label)
        obj["train"][0] = self.row("train", candidate)
        self.save(obj)
        with self.assertRaises(ValueError):
            pipeline.read_manifest(self.manifest)

    def test_pilot_must_agree_with_original_source_hashes(self):
        changed = copy.deepcopy(self.pilot_obj)
        changed["data_hashes"]["train.txt"] = "0" * 64
        pipeline.dump(self.pilot, changed)
        with self.assertRaises(ValueError):
            pipeline.prepare_manifest(self.root / "new_smoke.json")

    def test_changed_existing_manifest_is_not_silently_overwritten(self):
        obj = copy.deepcopy(self.obj)
        obj["scope"] = "edited"
        self.save(obj)
        before = self.manifest.read_bytes()
        with self.assertRaises(ValueError):
            pipeline.prepare_manifest(self.manifest)
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_cross_split_swapped_pair_rejected(self):
        self.train_path.write_text("a\tb\t1\nc\td\t0\n", encoding="utf-8")
        self.dev_path.write_text("b\ta\t1\n", encoding="utf-8")
        train = [
            {"pair_id": "train:000001", "split": "train", "line_number": 1,
             "s1": "a", "s2": "b", "label": 1},
            {"pair_id": "train:000002", "split": "train", "line_number": 2,
             "s1": "c", "s2": "d", "label": 0},
        ]
        dev = [{"pair_id": "dev:000001", "split": "dev", "line_number": 1,
                "s1": "b", "s2": "a", "label": 1}]
        with self.assertRaises(ValueError):
            pipeline.verify_records(train, dev)


class EmbeddingChecks(unittest.TestCase):
    @staticmethod
    def good_vectors():
        vectors = np.zeros((2, 1024), dtype=np.float32)
        vectors[0, 0] = 1.0
        vectors[1, 1] = 1.0
        return vectors

    def test_unit_float32_vectors_accepted(self):
        pipeline.check_embeddings(self.good_vectors(), 2)

    def test_shape_and_dtype_rejected(self):
        for vectors in (np.zeros((1, 1024), dtype=np.float32),
                        np.zeros((2, 1023), dtype=np.float32),
                        self.good_vectors().astype(np.float64)):
            with self.subTest(shape=vectors.shape, dtype=str(vectors.dtype)):
                with self.assertRaises(ValueError):
                    pipeline.check_embeddings(vectors, 2)

    def test_nonfinite_vectors_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                vectors = self.good_vectors()
                vectors[0, 0] = value
                with self.assertRaises(ValueError):
                    pipeline.check_embeddings(vectors, 2)

    def test_zero_and_nonunit_vectors_rejected(self):
        for scale in (0.0, 2.0):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    pipeline.check_embeddings(self.good_vectors() * scale, 2)


class CacheAlignmentChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.obj = {
            "train": [{"s1": "a", "s2": "b", "label": 1},
                      {"s1": "b", "s2": "c", "label": 0}],
            "dev_diagnostic": [{"s1": "c", "s2": "a", "label": 1}],
        }
        self.manifest = self.root / "manifest.json"
        pipeline.dump(self.manifest, self.obj)
        self.contract = {"test_contract": "no_real_model"}
        self.contract_patch = patch.object(pipeline, "encoding_contract", return_value=self.contract)
        self.contract_patch.start()
        self.addCleanup(self.contract_patch.stop)
        versions_patch = patch.object(pipeline, "check_cache_versions", return_value=None)
        versions_patch.start()
        self.addCleanup(versions_patch.stop)
        texts, _ = pipeline.text_layout(self.obj)
        self.vectors = np.zeros((3, 1024), dtype=np.float32)
        self.vectors[0, 0] = 1.0
        self.vectors[1, 1] = 1.0
        self.vectors[2, 2] = 1.0
        np.save(self.root / "embeddings.npy", self.vectors, allow_pickle=False)
        self.meta = {
            "identity": {"manifest_sha256": pipeline.sha(self.manifest), "texts": texts,
                         "encoding": self.contract},
            "embeddings_sha256": pipeline.sha(self.root / "embeddings.npy"),
            "frozen_weights_before": "same", "frozen_weights_after": "same",
            "all_requires_grad_false": True, "all_grad_none": True,
        }
        pipeline.dump(self.root / "cache.json", self.meta)

    def load(self):
        return pipeline.load_pair_arrays(self.obj, self.manifest, self.root / "unused_model", self.root)

    def test_text_deduplication_and_pair_label_alignment(self):
        texts, index = pipeline.text_layout(self.obj)
        self.assertEqual(texts, ["a", "b", "c"])
        self.assertEqual(index, {"a": 0, "b": 1, "c": 2})
        train, dev = self.load()
        np.testing.assert_array_equal(train[0], self.vectors[[0, 1]])
        np.testing.assert_array_equal(train[1], self.vectors[[1, 2]])
        np.testing.assert_array_equal(train[2], np.array([1, 0], dtype=np.int64))
        np.testing.assert_array_equal(dev[0], self.vectors[[2]])
        np.testing.assert_array_equal(dev[1], self.vectors[[0]])
        self.assertEqual(train[2].dtype, np.int64)

    def test_changed_text_order_rejected(self):
        self.meta["identity"]["texts"] = ["b", "a", "c"]
        pipeline.dump(self.root / "cache.json", self.meta)
        with self.assertRaises(ValueError):
            self.load()

    def test_changed_cache_bytes_rejected(self):
        np.save(self.root / "embeddings.npy", self.vectors[[1, 0, 2]], allow_pickle=False)
        with self.assertRaises(ValueError):
            self.load()

    def test_changed_encoding_contract_rejected(self):
        with patch.object(pipeline, "encoding_contract", return_value={"test_contract": "changed"}):
            with self.assertRaises(ValueError):
                self.load()


class CacheVersionChecks(unittest.TestCase):
    """冻结记录必须提供有效且一致的参数指纹，不加载模型。"""

    def setUp(self):
        self.versions = {"torch": "test-torch", "numpy": "test-numpy",
                         "sentence_transformers": "test-sentence-transformers"}
        installed = {"torch": self.versions["torch"], "numpy": self.versions["numpy"],
                     "sentence-transformers": self.versions["sentence_transformers"]}
        version_patch = patch.object(pipeline, "version", side_effect=installed.__getitem__)
        version_patch.start()
        self.addCleanup(version_patch.stop)
        self.metadata = {"versions": self.versions.copy(), "all_requires_grad_false": True,
                         "all_grad_none": True, "frozen_weights_before": "a" * 64,
                         "frozen_weights_after": "a" * 64}

    def test_valid_frozen_record_accepted(self):
        pipeline.check_cache_versions(self.metadata)
        uppercase = copy.deepcopy(self.metadata)
        uppercase["frozen_weights_before"] = "A" * 64
        uppercase["frozen_weights_after"] = "A" * 64
        pipeline.check_cache_versions(uppercase)

    def test_missing_fingerprint_rejected(self):
        for missing in (("frozen_weights_before",), ("frozen_weights_after",),
                        ("frozen_weights_before", "frozen_weights_after")):
            with self.subTest(missing=missing):
                metadata = copy.deepcopy(self.metadata)
                for key in missing:
                    del metadata[key]
                with self.assertRaises(ValueError):
                    pipeline.check_cache_versions(metadata)

    def test_malformed_fingerprint_rejected(self):
        for bad in (None, "", "a" * 63, "a" * 65, "g" * 64, " " * 64, 123, b"a" * 64):
            for key in ("frozen_weights_before", "frozen_weights_after"):
                with self.subTest(key=key, bad=repr(bad)):
                    metadata = copy.deepcopy(self.metadata)
                    metadata[key] = bad
                    with self.assertRaises(ValueError):
                        pipeline.check_cache_versions(metadata)

    def test_different_valid_fingerprints_rejected(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["frozen_weights_after"] = "b" * 64
        with self.assertRaises(ValueError):
            pipeline.check_cache_versions(metadata)


class FeatureValidationChecks(unittest.TestCase):
    """固定小数值答案只验证外围，不提供通用句对特征算法。"""

    def setUp(self):
        self.left = np.array([[1.0, -2.0], [0.0, 3.0]], dtype=np.float32)
        self.right = np.array([[-1.0, 2.0], [4.0, 1.0]], dtype=np.float32)
        self.expected = np.array([[2.0, 4.0, -1.0, -4.0],
                                  [4.0, 2.0, 0.0, 3.0]], dtype=np.float32)

    def fixed_fixture(self, left, right):
        # 只适用于本测试的固定输入与其交换顺序，没有通用特征实现。
        return self.expected.copy()

    def test_known_good_fixture_accepted(self):
        before_a, before_b = self.left.copy(), self.right.copy()
        actual = pipeline.validate_features(self.fixed_fixture, self.left, self.right)
        np.testing.assert_array_equal(actual, self.expected)
        np.testing.assert_array_equal(self.left, before_a)
        np.testing.assert_array_equal(self.right, before_b)

    def test_shape_dtype_and_finite_output_rejected(self):
        wrong_values = [np.zeros((2, 3), dtype=np.float32),
                        np.zeros((4, 2), dtype=np.float32),
                        self.expected.astype(np.float64),
                        np.full((2, 4), np.nan, dtype=np.float32)]
        for value in wrong_values:
            with self.subTest(shape=value.shape, dtype=str(value.dtype)):
                with self.assertRaises(AssertionError):
                    pipeline.validate_features(lambda left, right: value.copy(), self.left, self.right)

    def test_asymmetric_output_rejected(self):
        def asymmetric_fixture(left, right):
            value = self.expected.copy()
            value[0, 0] = left[0, 0]
            return value
        with self.assertRaises(AssertionError):
            pipeline.validate_features(asymmetric_fixture, self.left, self.right)

    def test_input_mutation_rejected(self):
        for which in (0, 1):
            with self.subTest(which=which):
                def mutating_fixture(left, right):
                    (left, right)[which][0, 0] = 99
                    return self.expected.copy()
                with self.assertRaises(AssertionError):
                    pipeline.validate_features(mutating_fixture, self.left.copy(), self.right.copy())

    def test_empty_mismatched_or_nonfinite_input_rejected(self):
        cases = [(self.left[:0], self.right[:0]),
                 (self.left, self.right[:1]),
                 (self.left[:, 0], self.right[:, 0]),
                 (np.full((2, 2), np.inf, dtype=np.float32), self.right)]
        for left, right in cases:
            with self.subTest(shape=left.shape, right_shape=right.shape):
                with self.assertRaises(ValueError):
                    pipeline.validate_features(self.fixed_fixture, left, right)

    def test_symmetric_zero_output_cannot_prove_feature_definition(self):
        # 形状、对称性、有限数本身不足以证明按题意构造了特征。
        with self.assertRaises(AssertionError):
            pipeline.validate_features(
                lambda left, right: np.zeros((2, 4), dtype=np.float32), self.left, self.right)


if __name__ == "__main__":
    unittest.main(verbosity=2)
