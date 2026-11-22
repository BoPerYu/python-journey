"""本人填写后运行。仅使用固定小例子，不含学生函数的参考实现。"""
import unittest
import numpy as np
from student_head import make_pair_features


class FeatureExamples(unittest.TestCase):
    def test_two_pairs_known_answer(self):
        left = np.array([[1, -2], [0, 3]], dtype=np.float32)
        right = np.array([[-1, 2], [4, 1]], dtype=np.float32)
        expected = np.array([[2, 4, -1, -4], [4, 2, 0, 3]], dtype=np.float32)
        np.testing.assert_array_equal(make_pair_features(left, right), expected)

    def test_swapping_preserves_result(self):
        a = np.array([[1, -2]], dtype=np.float32)
        b = np.array([[-1, 2]], dtype=np.float32)
        np.testing.assert_array_equal(make_pair_features(a, b), make_pair_features(b, a))

    def test_identical_vectors(self):
        vector = np.array([[1, -2]], dtype=np.float32)
        np.testing.assert_array_equal(make_pair_features(vector, vector),
                                      np.array([[0, 0, 1, 4]], dtype=np.float32))

    def test_inputs_not_modified_and_output_dtype(self):
        a = np.array([[1, -2]], dtype=np.float32)
        b = np.array([[-1, 2]], dtype=np.float32)
        a_before, b_before = a.copy(), b.copy()
        features = make_pair_features(a, b)
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_array_equal(a, a_before)
        np.testing.assert_array_equal(b, b_before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
