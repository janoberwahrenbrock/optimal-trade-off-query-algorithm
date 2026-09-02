from __future__ import annotations

import unittest

from multistep.src.models import AlternativenMatrix, Query
from multistep.src.query_value_function import (
    build_linear_query_values,
    compute_sample_ratio_range,
    estimate_depth_two_value_curve_for_goal_pair_from_samples,
    estimate_query_value_from_samples,
    evaluate_query_value,
    evaluate_query_value_curve_for_goal_pair,
    filter_samples_for_query_answer,
    find_best_estimated_one_step_query_for_samples,
    get_ordered_goal_pairs,
)


class QueryValueFunctionTests(unittest.TestCase):
    def test_compute_sample_ratio_range(self) -> None:
        samples = [
            [0.2, 0.8],
            [0.5, 0.5],
            [0.8, 0.2],
        ]

        self.assertEqual(
            compute_sample_ratio_range(
                samples=samples,
                numerator_index=0,
                denominator_index=1,
            ),
            (0.25, 4.0),
        )

    def test_build_linear_query_values(self) -> None:
        self.assertEqual(build_linear_query_values(0.0, 1.0, 3), [0.0, 0.5, 1.0])

    def test_get_ordered_goal_pairs(self) -> None:
        self.assertEqual(
            get_ordered_goal_pairs(3),
            [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)],
        )

    def test_evaluate_query_value(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        query = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)
        samples = [
            [0.2, 0.8],
            [0.7, 0.3],
        ]

        evaluation = evaluate_query_value(
            alternatives=alternatives,
            answered_queries=[],
            query=query,
            samples=samples,
        )

        self.assertEqual(evaluation.expected_candidate_count, 2.0)
        self.assertEqual(
            [answer.probability for answer in evaluation.answer_evaluations],
            [0.5, 0.0, 0.5],
        )
        self.assertEqual(
            [answer.candidate_count for answer in evaluation.answer_evaluations],
            [2, 2, 2],
        )

    def test_evaluate_query_value_curve_for_goal_pair(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        evaluations = evaluate_query_value_curve_for_goal_pair(
            alternatives=alternatives,
            answered_queries=[],
            samples=[
                [0.2, 0.8],
                [0.7, 0.3],
            ],
            goal_pair=(0, 1),
            query_values=[0.5, 1.0],
        )

        self.assertEqual(len(evaluations), 2)
        self.assertEqual([evaluation.query.value for evaluation in evaluations], [0.5, 1.0])

    def test_filter_samples_for_query_answer(self) -> None:
        query = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)

        self.assertEqual(
            filter_samples_for_query_answer(
                samples=[
                    [0.2, 0.8],
                    [0.5, 0.5],
                    [0.8, 0.2],
                ],
                query=query,
                answer=">",
            ),
            [[0.8, 0.2]],
        )

    def test_estimate_query_value_from_samples(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        query = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)

        evaluation = estimate_query_value_from_samples(
            alternatives=alternatives,
            query=query,
            samples=[
                [0.2, 0.8],
                [0.8, 0.2],
            ],
        )

        self.assertEqual(evaluation.expected_candidate_count, 1.0)

    def test_find_best_estimated_one_step_query_for_samples(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        evaluation = find_best_estimated_one_step_query_for_samples(
            alternatives=alternatives,
            samples=[
                [0.2, 0.8],
                [0.8, 0.2],
            ],
            query_value_steps=3,
            max_query_value=2.0,
        )

        self.assertGreaterEqual(evaluation.expected_candidate_count, 1.0)
        self.assertLessEqual(evaluation.expected_candidate_count, 2.0)

    def test_estimate_depth_two_value_curve_for_goal_pair_from_samples(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        evaluations = estimate_depth_two_value_curve_for_goal_pair_from_samples(
            alternatives=alternatives,
            samples=[
                [0.2, 0.8],
                [0.8, 0.2],
            ],
            goal_pair=(0, 1),
            query_values=[0.5, 1.0],
            inner_query_value_steps=3,
            max_query_value=2.0,
        )

        self.assertEqual(len(evaluations), 2)
        self.assertEqual([evaluation.query.value for evaluation in evaluations], [0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
