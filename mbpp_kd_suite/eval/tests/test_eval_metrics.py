from __future__ import annotations

import unittest

import numpy as np

from mbpp_kd_suite.metrics import paired_ranking_metrics


class EvalMetricsTest(unittest.TestCase):
    def test_perfect_score_matrix(self) -> None:
        score_matrix = np.eye(4, dtype=np.float32) * 10.0
        metrics = paired_ranking_metrics(score_matrix, ks=(1, 5, 10))
        self.assertAlmostEqual(metrics["MRR"], 1.0)
        self.assertAlmostEqual(metrics["Recall@1"], 1.0)
        self.assertAlmostEqual(metrics["Recall@5"], 1.0)
        self.assertAlmostEqual(metrics["nDCG@10"], 1.0)
        self.assertAlmostEqual(metrics["MAP@10"], 1.0)

    def test_nonperfect_score_matrix_stays_bounded(self) -> None:
        score_matrix = np.array(
            [
                [0.1, 0.9, 0.0],
                [0.8, 0.2, 0.1],
                [0.1, 0.3, 0.4],
            ],
            dtype=np.float32,
        )
        metrics = paired_ranking_metrics(score_matrix, ks=(1, 2, 3))
        self.assertLess(metrics["MRR"], 1.0)
        for value in metrics.values():
            if isinstance(value, float) and value != metrics["MedianRank"]:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
