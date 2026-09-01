from __future__ import annotations

import unittest

from multistep.src.models import AnsweredQuery
import numpy as np

from multistep.src.sampling import (
    find_relative_interior_point,
    sample_points_from_constraint_system,
    sample_points_with_diagnostics,
)
from multistep.src.weight_space import build_weight_space


class SamplingTests(unittest.TestCase):
    def test_relative_interior_start_is_centered_in_simplex(self) -> None:
        system = build_weight_space(goal_count=3, answered_queries=[])

        point, radius = find_relative_interior_point(system)

        np.testing.assert_allclose(point, [1.0 / 3.0] * 3, atol=1e-9)
        self.assertGreater(radius, 0.0)

    def test_multiple_chains_explore_dimension_seven_simplex(self) -> None:
        system = build_weight_space(goal_count=7, answered_queries=[])

        samples, diagnostics = sample_points_with_diagnostics(
            system=system,
            num_samples=600,
            burn_in=200,
            thinning=3,
            seed=4,
            chain_count=4,
        )

        sample_matrix = np.asarray(samples)
        self.assertEqual(diagnostics.chain_count, 4)
        self.assertGreater(diagnostics.unique_sample_count, 590)
        self.assertGreater(diagnostics.minimum_effective_sample_size, 1.0)
        np.testing.assert_allclose(
            np.mean(sample_matrix, axis=0),
            np.full(7, 1.0 / 7.0),
            atol=0.06,
        )

    def test_sampling_points_from_simplex(self) -> None:
        system = build_weight_space(goal_count=3, answered_queries=[])

        samples = sample_points_from_constraint_system(
            system=system,
            num_samples=5,
            burn_in=0,
            thinning=1,
            seed=1,
        )

        self.assertEqual(len(samples), 5)
        for sample in samples:
            self.assertEqual(len(sample), 3)
            self.assertAlmostEqual(sum(sample), 1.0)
            for value in sample:
                self.assertGreaterEqual(value, -1e-9)

    def test_sampling_returns_same_point_for_zero_dimensional_region(self) -> None:
        system = build_weight_space(
            goal_count=2,
            answered_queries=[
                AnsweredQuery(
                    ziel_index_a=0,
                    ziel_index_b=1,
                    value=1.0,
                    operator="=",
                )
            ],
        )

        samples = sample_points_from_constraint_system(
            system=system,
            num_samples=3,
            burn_in=0,
            thinning=1,
            seed=1,
        )

        self.assertEqual(samples, [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

    def test_sampling_accepts_linprog_feasible_start_point_at_solver_tolerance(self) -> None:
        answered_queries = [
            AnsweredQuery(
                ziel_index_a=0,
                ziel_index_b=4,
                value=5.980121291776192,
                operator="<",
            ),
            AnsweredQuery(
                ziel_index_a=4,
                ziel_index_b=6,
                value=1.1016599577570163,
                operator="<",
            ),
            AnsweredQuery(
                ziel_index_a=1,
                ziel_index_b=6,
                value=10.904275609051547,
                operator="<",
            ),
            AnsweredQuery(
                ziel_index_a=5,
                ziel_index_b=6,
                value=1.3095453788006652,
                operator=">",
            ),
            AnsweredQuery(
                ziel_index_a=3,
                ziel_index_b=5,
                value=1.139659900048831,
                operator="<",
            ),
            AnsweredQuery(
                ziel_index_a=5,
                ziel_index_b=6,
                value=113.96424155807941,
                operator="<",
            ),
            AnsweredQuery(
                ziel_index_a=5,
                ziel_index_b=6,
                value=16.65563878152816,
                operator="<",
            ),
            AnsweredQuery(
                ziel_index_a=2,
                ziel_index_b=6,
                value=1.7769803532090718,
                operator=">",
            ),
            AnsweredQuery(
                ziel_index_a=2,
                ziel_index_b=5,
                value=172.05108394967047,
                operator="<",
            ),
            AnsweredQuery(
                ziel_index_a=2,
                ziel_index_b=6,
                value=225.30839235855754,
                operator="<",
            ),
            AnsweredQuery(
                ziel_index_a=3,
                ziel_index_b=6,
                value=7.236647336301882,
                operator="<",
            ),
        ]
        system = build_weight_space(goal_count=7, answered_queries=answered_queries)

        samples = sample_points_from_constraint_system(
            system=system,
            num_samples=1,
            burn_in=0,
            thinning=1,
            seed=4106120257,
        )

        self.assertEqual(len(samples), 1)

    def test_sampling_rejects_invalid_parameters(self) -> None:
        system = build_weight_space(goal_count=2, answered_queries=[])

        with self.assertRaisesRegex(ValueError, "num_samples must be positive"):
            sample_points_from_constraint_system(system=system, num_samples=0)

        with self.assertRaisesRegex(ValueError, "burn_in must not be negative"):
            sample_points_from_constraint_system(
                system=system,
                num_samples=1,
                burn_in=-1,
            )

        with self.assertRaisesRegex(ValueError, "thinning must be positive"):
            sample_points_from_constraint_system(
                system=system,
                num_samples=1,
                thinning=0,
            )


if __name__ == "__main__":
    unittest.main()
