from __future__ import annotations

import unittest

from multistep.src.candidates import (
    compute_candidate_set,
    estimate_candidate_set_from_samples,
)
from multistep.src.models import AlternativenMatrix
from multistep.src.weight_space import build_weight_space


class CandidateTests(unittest.TestCase):
    def test_compute_candidate_set_returns_exact_candidates(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
                [0.4, 0.4],
            ],
        )
        weight_space = build_weight_space(goal_count=2, answered_queries=[])

        candidates = compute_candidate_set(
            alternatives=alternatives,
            weight_space=weight_space,
        )

        self.assertEqual(candidates, [0, 1])

    def test_compute_candidate_set_can_return_single_candidate_in_restricted_weight_space(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        weight_space = build_weight_space(goal_count=2, answered_queries=[])
        weight_space.add_inequality([1.0, -1.0], -0.1)

        candidates = compute_candidate_set(
            alternatives=alternatives,
            weight_space=weight_space,
        )

        self.assertEqual(candidates, [1])

    def test_estimate_candidate_set_from_samples_returns_sample_optima(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
                [0.4, 0.4],
            ],
        )

        candidates = estimate_candidate_set_from_samples(
            alternatives=alternatives,
            samples=[
                [0.9, 0.1],
                [0.1, 0.9],
            ],
        )

        self.assertEqual(candidates, [0, 1])

    def test_estimate_candidate_set_from_samples_includes_ties_with_tolerance(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        candidates = estimate_candidate_set_from_samples(
            alternatives=alternatives,
            samples=[[0.5, 0.5]],
        )

        self.assertEqual(candidates, [0, 1])

    def test_estimate_candidate_set_from_samples_rejects_empty_samples(self) -> None:
        alternatives = AlternativenMatrix(entries=[[1.0, 0.0]])

        with self.assertRaisesRegex(ValueError, "samples must not be empty"):
            estimate_candidate_set_from_samples(
                alternatives=alternatives,
                samples=[],
            )

    def test_estimate_candidate_set_from_samples_rejects_wrong_sample_dimension(self) -> None:
        alternatives = AlternativenMatrix(entries=[[1.0, 0.0]])

        with self.assertRaisesRegex(ValueError, "samples must match"):
            estimate_candidate_set_from_samples(
                alternatives=alternatives,
                samples=[[1.0, 0.0, 0.0]],
            )


if __name__ == "__main__":
    unittest.main()
