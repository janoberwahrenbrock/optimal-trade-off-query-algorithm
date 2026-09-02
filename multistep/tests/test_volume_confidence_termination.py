from __future__ import annotations

import unittest

from multistep.scripts.benchmark_exact_end_to_end import solve_problem
from multistep.src.models import AlternativenMatrix


class VolumeConfidenceTerminationTest(unittest.TestCase):
    def test_stops_with_multiple_exact_candidates_above_threshold(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0, 0.0],
                [0.99, 1.0, 1.0],
            ]
        )

        result = solve_problem(
            alternatives=alternatives,
            target_weights=[0.2, 0.4, 0.4],
            max_questions=3,
            depth=2,
            volume_confidence_threshold=0.99,
        )

        self.assertTrue(result["solved"])
        self.assertFalse(result["exactly_solved"])
        self.assertEqual(result["termination_reason"], "volume_confidence")
        self.assertEqual(result["question_count"], 0)
        self.assertEqual(result["final_candidates"], [0, 1])
        self.assertIsNone(result["final_candidate"])
        self.assertEqual(result["selected_candidate"], 1)
        self.assertGreaterEqual(result["selected_candidate_volume_share"], 0.99)
        self.assertLessEqual(result["residual_volume_share"], 0.01)
        self.assertTrue(result["selection_is_correct"])

    def test_exact_single_candidate_keeps_exact_termination_reason(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )

        result = solve_problem(
            alternatives=alternatives,
            target_weights=[0.2, 0.4, 0.4],
            max_questions=3,
            depth=2,
            volume_confidence_threshold=0.99,
        )

        self.assertTrue(result["solved"])
        self.assertTrue(result["exactly_solved"])
        self.assertEqual(result["termination_reason"], "exact_single_candidate")
        self.assertEqual(result["selected_candidate"], 1)
        self.assertEqual(result["selected_candidate_volume_share"], 1.0)
        self.assertEqual(result["residual_volume_share"], 0.0)


if __name__ == "__main__":
    unittest.main()
