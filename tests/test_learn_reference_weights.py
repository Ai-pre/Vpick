from __future__ import annotations

import unittest

import numpy as np

from evaluation.learn_reference_weights import _column_spearman, weight_grid


class ReferenceWeightLearningTest(unittest.TestCase):
    def test_weight_grid_is_complete_and_normalized(self) -> None:
        checklist = weight_grid(7)
        extended = weight_grid(8)
        self.assertEqual(checklist.shape, (8008, 7))
        self.assertEqual(extended.shape, (19448, 8))
        np.testing.assert_allclose(checklist.sum(axis=1), 1.0)
        np.testing.assert_allclose(extended.sum(axis=1), 1.0)
        self.assertTrue(np.all(checklist >= 0))
        self.assertTrue(np.all(extended >= 0))

    def test_column_spearman_handles_ties(self) -> None:
        predictions = np.asarray(
            [
                [1.0, 3.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [4.0, 1.0],
            ]
        )
        target = np.asarray([1.0, 2.0, 2.0, 4.0])
        correlations = _column_spearman(predictions, target)
        self.assertAlmostEqual(float(correlations[0]), 1.0)
        self.assertAlmostEqual(float(correlations[1]), -1.0)


if __name__ == "__main__":
    unittest.main()
