from __future__ import annotations

import unittest

from multistep.src.models import Query
from multistep.src.query_probability import (
    ANSWER_OPTIONS,
    classify_query_answer,
    estimate_query_answer_probabilities,
    estimate_query_answer_probability,
)


class QueryProbabilityTests(unittest.TestCase):
    def test_answer_options_are_ordered_like_formula(self) -> None:
        self.assertEqual(ANSWER_OPTIONS, ("<", "=", ">"))

    def test_classify_query_answer_without_equality_tolerance(self) -> None:
        query = Query(ziel_index_a=0, ziel_index_b=1, value=2.0)

        self.assertEqual(classify_query_answer([0.3, 0.2], query), "<")
        self.assertEqual(classify_query_answer([0.4, 0.2], query), "=")
        self.assertEqual(classify_query_answer([0.5, 0.2], query), ">")

    def test_classify_query_answer_with_equality_tolerance(self) -> None:
        query = Query(ziel_index_a=0, ziel_index_b=1, value=2.0)

        self.assertEqual(
            classify_query_answer(
                weights=[0.399, 0.2],
                query=query,
                equality_tol=0.002,
            ),
            "=",
        )
        self.assertEqual(
            classify_query_answer(
                weights=[0.397, 0.2],
                query=query,
                equality_tol=0.002,
            ),
            "<",
        )

    def test_estimate_query_answer_probabilities(self) -> None:
        query = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)

        probabilities = estimate_query_answer_probabilities(
            query=query,
            samples=[
                [0.2, 0.8],
                [0.5, 0.5],
                [0.7, 0.3],
                [0.8, 0.2],
            ],
        )

        self.assertEqual(
            probabilities,
            {
                "<": 0.25,
                "=": 0.25,
                ">": 0.5,
            },
        )

    def test_estimate_query_answer_probability_returns_single_answer_probability(self) -> None:
        query = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)

        probability = estimate_query_answer_probability(
            query=query,
            answer=">",
            samples=[
                [0.2, 0.8],
                [0.7, 0.3],
            ],
        )

        self.assertEqual(probability, 0.5)

    def test_estimate_query_answer_probabilities_rejects_empty_samples(self) -> None:
        query = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)

        with self.assertRaisesRegex(ValueError, "samples must not be empty"):
            estimate_query_answer_probabilities(query=query, samples=[])

    def test_negative_equality_tolerance_is_rejected(self) -> None:
        query = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)

        with self.assertRaisesRegex(ValueError, "equality_tol must not be negative"):
            classify_query_answer(
                weights=[0.5, 0.5],
                query=query,
                equality_tol=-1e-6,
            )

    def test_weight_dimension_must_match_query_indices(self) -> None:
        query = Query(ziel_index_a=0, ziel_index_b=2, value=1.0)

        with self.assertRaisesRegex(IndexError, "ziel_index_b is out of range"):
            classify_query_answer(weights=[0.5, 0.5], query=query)


if __name__ == "__main__":
    unittest.main()
