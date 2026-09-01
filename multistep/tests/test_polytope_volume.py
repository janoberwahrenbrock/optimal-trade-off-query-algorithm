from __future__ import annotations

import unittest

from multistep.src.models import Query
from multistep.src.polytope_volume import (
    clear_polytope_volume_cache,
    compute_exact_query_answer_probabilities,
    compute_polytope_intrinsic_volume,
    polytope_volume_cache_info,
)
from multistep.src.weight_space import build_weight_space


class PolytopeVolumeTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_polytope_volume_cache()

    def test_free_simplex_query_matches_analytic_probability(self) -> None:
        weight_space = build_weight_space(goal_count=7, answered_queries=[])

        for ratio in (0.1, 0.5, 1.0, 2.0, 10.0):
            with self.subTest(ratio=ratio):
                probabilities = compute_exact_query_answer_probabilities(
                    weight_space=weight_space,
                    query=Query(ziel_index_a=0, ziel_index_b=1, value=ratio),
                )
                self.assertAlmostEqual(
                    probabilities["<"],
                    ratio / (1.0 + ratio),
                    places=8,
                )
                self.assertAlmostEqual(
                    probabilities[">"],
                    1.0 / (1.0 + ratio),
                    places=8,
                )
                self.assertEqual(probabilities["="], 0.0)
                self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_volume_results_are_cached_by_constraints(self) -> None:
        weight_space = build_weight_space(goal_count=3, answered_queries=[])

        first = compute_polytope_intrinsic_volume(weight_space)
        second = compute_polytope_intrinsic_volume(weight_space)

        self.assertGreater(first, 0.0)
        self.assertEqual(first, second)
        self.assertEqual(polytope_volume_cache_info().hits, 1)


if __name__ == "__main__":
    unittest.main()
